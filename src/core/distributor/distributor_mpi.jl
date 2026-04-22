# MPI array movement helpers, cache limits, and mesh utilities.

# ============================================================================
# MPI communication helpers
"""Gather array from all processes (PencilArrays-aware)"""
function gather_array(dist::Distributor, local_array::PencilArrays.PencilArray)

    start_time = time()

    if dist.size == 1
        result = Array(local_array)
    else
        result = PencilArrays.gather(local_array, 0)
        if dist.rank != 0
            result = Array{eltype(local_array)}(undef, PencilArrays.size_global(local_array)...)
        end
        MPI.Bcast!(result, dist.comm; root=0)
        dist.performance_stats.mpi_operations += 1
    end

    dist.performance_stats.total_time += time() - start_time
    return result
end

"""
    Gather array from all processes (fallback for non-PencilArray types).
    Note: MPI.Allgather flattens arrays - use the PencilArray version for
    shape-preserving gather of multi-dimensional distributed arrays.

    GPU-aware: Automatically transfers GPU arrays to CPU before MPI operations,
    since MPI requires CPU memory.
    """
function gather_array(dist::Distributor, local_array::AbstractArray)

    start_time = time()

    # CRITICAL: Check if this is actually a PencilArray wrapper (view, reshape, etc.)
    # These wrappers dispatch to AbstractArray but should use PencilArray gather
    underlying = _get_underlying_pencil_array(local_array)
    if underlying !== nothing
        @warn "gather_array received a PencilArray wrapper ($(typeof(local_array))). " *
              "This may cause incorrect MPI gather behavior. Using PencilArray gather instead." maxlog=1
        dist.performance_stats.total_time += time() - start_time
        return gather_array(dist, underlying)
    end

    # For GPU arrays, transfer to CPU first (MPI requires CPU memory)
    cpu_array = on_architecture(CPU(), local_array)

    if dist.size == 1
        result = cpu_array
    else
        result = MPI.Allgather(cpu_array, dist.comm)
        dist.performance_stats.mpi_operations += 1
    end

    # Update performance stats
    dist.performance_stats.total_time += time() - start_time

    return result
end

"""
    _get_underlying_pencil_array(array)

Check if an array is a wrapper around a PencilArray and return the underlying
PencilArray if so. Returns `nothing` if not a PencilArray wrapper.

This handles cases like:
- SubArray{..., PencilArray} (views)
- ReshapedArray{..., PencilArray} (reshaped)
- Other array wrappers that contain PencilArrays
"""
function _get_underlying_pencil_array(array::AbstractArray)
    # Direct PencilArray - handled by specific method, shouldn't reach here
    if isa(array, PencilArrays.PencilArray)
        return array
    end

    # Check for SubArray wrapping PencilArray
    if array isa SubArray
        parent_arr = parent(array)
        if isa(parent_arr, PencilArrays.PencilArray)
            @warn "View of PencilArray detected in gather_array. " *
                  "Consider using PencilArray directly or copying to a new PencilArray. " *
                  "View slicing may not preserve the correct distributed structure." maxlog=1
            # Return the parent PencilArray - note: this may not be exactly what was intended
            # since the view might select a subset
            return parent_arr
        end
        # Recursively check parent
        return _get_underlying_pencil_array(parent_arr)
    end

    # Check for ReshapedArray wrapping PencilArray
    if array isa Base.ReshapedArray
        parent_arr = parent(array)
        if isa(parent_arr, PencilArrays.PencilArray)
            @warn "Reshaped PencilArray detected in gather_array. " *
                  "Reshaping may break the distributed array structure. " *
                  "Consider gathering first, then reshaping the result." maxlog=1
            return parent_arr
        end
        return _get_underlying_pencil_array(parent_arr)
    end

    # Check for other wrappers that have a parent array
    if applicable(parent, array)
        parent_arr = try
            parent(array)
        catch
            nothing
        end
        if parent_arr !== nothing && parent_arr !== array
            return _get_underlying_pencil_array(parent_arr)
        end
    end

    return nothing
end

"""
    Scatter array to all processes.

    IMPORTANT: Uses different decomposition conventions based on dist.use_pencil_arrays:
    - use_pencil_arrays=true (CPU+MPI): PencilArrays convention, decompose LAST dims
    - use_pencil_arrays=false (GPU+MPI): TransposableField ZLocal convention, decompose FIRST dims

    Note: For GPU architectures, the input global_array should be a CPU array.
    The function will return the local portion on the target architecture (GPU if applicable).

    Communication pattern: Rank 0 distributes data using blocking Send operations to each
    destination rank sequentially, while other ranks post blocking Recv from rank 0.
    This pattern is safe because:
    - Each Recv has exactly one matching Send
    - MPI buffers small/medium messages automatically
    - Receives are posted before rank 0 completes all sends

    For very large arrays that exceed MPI buffer limits, consider using non-blocking
    Isend/Irecv with Waitall for better overlap.
    """
function scatter_array(dist::Distributor, global_array::AbstractArray)

    start_time = time()

    if dist.size == 1 || dist.mesh === nothing
        dist.performance_stats.total_time += time() - start_time
        return _maybe_to_architecture(dist.architecture, global_array)
    end

    # CRITICAL: For MPI scatter, global_array must be on CPU for rank 0
    # Other ranks don't need global_array, but MPI.Recv! needs CPU buffers
    # unless using CUDA-aware MPI
    cpu_global_array = global_array
    if is_gpu_array(global_array)
        if !check_cuda_aware_mpi()
            # Stage through CPU for MPI operations
            cpu_global_array = Array(global_array)
        end
    end

    global_shape = size(global_array)
    ndims_global = length(global_shape)
    ndims_mesh = length(dist.mesh)

    if dist.use_pencil_arrays
        # PencilArrays convention: decompose LAST dims
        decomp_dims = if ndims_global >= ndims_mesh
            ntuple(i -> ndims_global - ndims_mesh + i, ndims_mesh)
        else
            ntuple(identity, ndims_global)
        end

        pencil = nothing
        cache_key = (global_shape, decomp_dims, eltype(global_array))
        if haskey(dist.pencil_cache, cache_key)
            pencil = dist.pencil_cache[cache_key]
        else
            try
                # CRITICAL: Initialize mpi_topology if not done, then reuse.
                # Creating temporary MPITopology each call leaks MPI communicators.
                if dist.mpi_topology === nothing
                    dist.mpi_topology = PencilArrays.MPITopology(dist.comm, dist.mesh)
                    if dist.rank == 0
                        @debug "Initialized MPI topology in scatter_array: $(dist.mesh)"
                    end
                end
                pencil = PencilArrays.Pencil(dist.mpi_topology, global_shape, decomp_dims)
                dist.pencil_cache[cache_key] = pencil
            catch e
                @error "PencilArrays scatter failed in MPI mode" exception=e global_shape=global_shape decomp_dims=decomp_dims
                error("PencilArrays scatter failed with $(dist.size) MPI processes. " *
                      "Cannot fall back to flat scatter as this would produce incorrect data distribution. " *
                      "Please check your PencilArrays installation or use serial execution.")
            end
        end

        n_spatial = length(pencil.size_global)
        extra_dims = ndims_global > n_spatial ? global_shape[(n_spatial + 1):end] : ()
        local_spatial = length.(pencil.axes_local)
        local_shape = (local_spatial..., extra_dims...)
        local_array = zeros(eltype(global_array), local_shape...)

        if dist.rank == 0
            nprocs = length(pencil.topology)
            colons_extra = ntuple(_ -> Colon(), length(extra_dims))

            for n in 1:nprocs
                rrange = pencil.axes_all[n]
                dest_rank = pencil.topology.ranks[n]
                if dest_rank == 0
                    local_array .= cpu_global_array[rrange..., colons_extra...]
                else
                    send_buf = cpu_global_array[rrange..., colons_extra...]
                    MPI.Send(send_buf, dist.comm; dest=dest_rank, tag=0)
                end
            end
        else
            # Use MPI.Probe to check incoming message size before Recv
            # This catches mismatches between rank 0's PencilArray ranges and our local shape
            status = MPI.Probe(dist.comm, MPI.Status; source=0, tag=0)
            msg_count = MPI.Get_count(status, eltype(local_array))
            expected_count = length(local_array)
            if msg_count != expected_count
                error("Scatter recv shape mismatch on rank $(dist.rank): " *
                      "expected $expected_count elements (shape=$(size(local_array))), " *
                      "but rank 0 is sending $msg_count elements. " *
                      "This may indicate a PencilArrays configuration issue.")
            end
            MPI.Recv!(local_array, dist.comm; source=0, tag=0)
        end
    else
        # GPU+MPI / TransposableField ZLocal convention: decompose FIRST dims
        # Use get_local_array_size which respects use_pencil_arrays convention
        local_shape = get_local_array_size(dist, global_shape)
        local_array = zeros(eltype(global_array), local_shape...)

        # Compute local ranges for FIRST dims decomposition
        # This matches TransposableField's ZLocal convention
        mesh = dist.mesh
        P1 = mesh[1]
        P2 = ndims_mesh >= 2 ? mesh[2] : 1
        coord1 = dist.rank % P1
        coord2 = dist.rank ÷ P1

        # Compute ranges for first decomposed dimensions
        function compute_range(global_size, n_procs, coord)
            base_size = div(global_size, n_procs)
            remainder = global_size % n_procs
            if coord < remainder
                start = coord * (base_size + 1) + 1
                stop = start + base_size
            else
                start = coord * base_size + remainder + 1
                stop = start + base_size - 1
            end
            return start:stop
        end

        # Build ranges for each dimension
        ranges = Vector{UnitRange{Int}}(undef, ndims_global)
        for d in 1:ndims_global
            if d == 1 && ndims_mesh >= 1
                ranges[d] = compute_range(global_shape[1], P1, coord1)
            elseif d == 2 && ndims_mesh >= 2
                ranges[d] = compute_range(global_shape[2], P2, coord2)
            else
                ranges[d] = 1:global_shape[d]  # Not decomposed
            end
        end

        if dist.rank == 0
            # Rank 0 distributes data
            for dest_rank in 0:(dist.size-1)
                dest_coord1 = dest_rank % P1
                dest_coord2 = dest_rank ÷ P1

                dest_ranges = Vector{UnitRange{Int}}(undef, ndims_global)
                for d in 1:ndims_global
                    if d == 1 && ndims_mesh >= 1
                        dest_ranges[d] = compute_range(global_shape[1], P1, dest_coord1)
                    elseif d == 2 && ndims_mesh >= 2
                        dest_ranges[d] = compute_range(global_shape[2], P2, dest_coord2)
                    else
                        dest_ranges[d] = 1:global_shape[d]
                    end
                end

                if dest_rank == 0
                    local_array .= cpu_global_array[dest_ranges...]
                else
                    send_buf = cpu_global_array[dest_ranges...]
                    # CRITICAL: Validate send size matches expected recv size for dest_rank
                    # The dest_rank computed its local_shape independently; if they disagree,
                    # MPI.Recv! could either fail (too large) or leave uninitialized memory (too small)
                    MPI.Send(send_buf, dist.comm; dest=dest_rank, tag=0)
                end
            end
        else
            # Use MPI.Probe to check incoming message size before Recv
            # This catches mismatches between rank 0's send and our expected recv
            status = MPI.Probe(dist.comm, MPI.Status; source=0, tag=0)
            msg_count = MPI.Get_count(status, eltype(local_array))
            expected_count = length(local_array)
            if msg_count != expected_count
                error("Scatter recv shape mismatch on rank $(dist.rank): " *
                      "expected $expected_count elements (shape=$(size(local_array))), " *
                      "but rank 0 is sending $msg_count elements. " *
                      "This indicates a decomposition computation divergence between ranks.")
            end
            MPI.Recv!(local_array, dist.comm; source=0, tag=0)
        end
    end

    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time

    return _maybe_to_architecture(dist.architecture, local_array)
end

"""All-reduce operation on array"""
function allreduce_array(dist::Distributor, local_array::AbstractArray, op=MPI.SUM)

    start_time = time()

    arch = dist.architecture
    if is_gpu_array(local_array) || is_gpu(arch)
        local_cpu = _ensure_cpu_array(local_array)
        result_cpu = similar(local_cpu)
        MPI.Allreduce!(local_cpu, result_cpu, op, dist.comm)
        result = _maybe_to_architecture(arch, result_cpu)
    else
        result = similar(local_array)
        MPI.Allreduce!(local_array, result, op, dist.comm)
    end

    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time

    return result
end

# Maximum cache sizes to prevent unbounded memory growth
const MAX_LAYOUT_CACHE_SIZE = 100
const MAX_PENCIL_CACHE_SIZE = 50

"""Clear caches for distributor"""
function clear_distributor_cache!(dist::Distributor)

    # Clear layout cache
    empty!(dist.layouts)

    # Clear pencil cache
    empty!(dist.pencil_cache)

    # Reset PencilArrays configuration if needed
    dist.pencil_config = nothing

    @info "Cleared distributor caches"

    return dist
end

"""
    enforce_cache_limits!(dist::Distributor)

Enforce cache size limits by evicting oldest entries when limits are exceeded.
This prevents unbounded memory growth in long-running simulations.
"""
function enforce_cache_limits!(dist::Distributor)
    # Check layout cache
    if length(dist.layouts) > MAX_LAYOUT_CACHE_SIZE
        # Remove half of the entries (LRU would be better but more complex)
        n_to_remove = length(dist.layouts) ÷ 2
        keys_to_remove = collect(keys(dist.layouts))[1:n_to_remove]
        for k in keys_to_remove
            delete!(dist.layouts, k)
        end
        dist.performance_stats.cache_misses += n_to_remove  # Track evictions
        @debug "Evicted $n_to_remove layout cache entries"
    end

    # Check pencil cache
    if length(dist.pencil_cache) > MAX_PENCIL_CACHE_SIZE
        n_to_remove = length(dist.pencil_cache) ÷ 2
        keys_to_remove = collect(keys(dist.pencil_cache))[1:n_to_remove]
        for k in keys_to_remove
            delete!(dist.pencil_cache, k)
        end
        @debug "Evicted $n_to_remove pencil cache entries"
    end
end

"""
    maybe_cleanup_caches!(dist::Distributor)

Periodically check and enforce cache limits.
Called automatically during pencil/layout creation.
"""
function maybe_cleanup_caches!(dist::Distributor)
    # Only check every 10 operations to amortize overhead
    total_ops = dist.performance_stats.pencil_creations + dist.performance_stats.layout_creations
    if total_ops > 0 && total_ops % 10 == 0
        enforce_cache_limits!(dist)
    end
end

"""Get memory usage information for distributor"""
function get_distributor_memory_info(dist::Distributor)

    return (
        cached_layouts = length(dist.layouts),
        cached_pencils = length(dist.pencil_cache)
    )
end

"""Log distributor performance statistics"""
function log_distributor_performance(dist::Distributor)

    stats = dist.performance_stats

    @info "Distributor performance:"
    @info "  Pencil creations: $(stats.pencil_creations)"
    @info "  Layout creations: $(stats.layout_creations)"
    @info "  MPI operations: $(stats.mpi_operations)"
    @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    @info "  Transpose time: $(round(stats.transpose_time, digits=3)) seconds"
    @info "  Communication time: $(round(stats.communication_time, digits=3)) seconds"
    @info "  Cache performance: $(stats.cache_hits) hits / $(stats.cache_misses) misses"

    # Memory usage
    mem_info = get_distributor_memory_info(dist)
    @info "  Cached layouts: $(mem_info.cached_layouts)"
    @info "  Cached pencils: $(mem_info.cached_pencils)"
end

# MPI communication functions
"""All-to-all communication"""
function mpi_alltoall(dist::Distributor, send_data::AbstractArray, recv_data::AbstractArray)

    start_time = time()

    nprocs = MPI.Comm_size(dist.comm)

    # Validate that buffer sizes are evenly divisible by nprocs
    # MPI.Alltoall requires uniform message sizes
    send_len = length(send_data)
    recv_len = length(recv_data)

    if send_len % nprocs != 0
        error("mpi_alltoall: send_data length ($send_len) must be divisible by nprocs ($nprocs). " *
              "Use mpi_alltoallv for non-uniform message sizes.")
    end
    if recv_len % nprocs != 0
        error("mpi_alltoall: recv_data length ($recv_len) must be divisible by nprocs ($nprocs). " *
              "Use mpi_alltoallv for non-uniform message sizes.")
    end

    send_count = send_len ÷ nprocs
    recv_count = recv_len ÷ nprocs

    if is_gpu_array(send_data) || is_gpu_array(recv_data) || is_gpu(dist.architecture)
        send_cpu = _ensure_cpu_array(send_data)
        recv_cpu = Array{eltype(recv_data)}(undef, size(recv_data)...)
        MPI.Alltoall!(MPI.UBuffer(send_cpu, send_count), MPI.UBuffer(recv_cpu, recv_count), dist.comm)
        if is_gpu_array(recv_data)
            copyto!(recv_data, on_architecture(dist.architecture, recv_cpu))
        else
            copyto!(recv_data, recv_cpu)
        end
    else
        MPI.Alltoall!(MPI.UBuffer(send_data, send_count), MPI.UBuffer(recv_data, recv_count), dist.comm)
    end

    # Update performance stats
    dist.performance_stats.mpi_operations += 1
    dist.performance_stats.total_time += time() - start_time

    return recv_data
end

"""Create optimal 2D process mesh for given number of processes"""
function create_2d_process_mesh(nproc::Int)
    # Find factors closest to square
    factors = []
    for i in 1:floor(Int, sqrt(nproc))
        if nproc % i == 0
            push!(factors, (i, nproc ÷ i))
        end
    end
    
    if isempty(factors)
        return (1, nproc)
    end
    
    # Choose factors closest to square
    best_factor = factors[end]
    return best_factor
end

"""Create optimal 3D process mesh for given number of processes"""
function create_3d_process_mesh(nproc::Int)
    # Simple heuristic for 3D decomposition
    if nproc <= 8
        # Small cases
        if nproc == 1
            return (1, 1, 1)
        elseif nproc == 2
            return (2, 1, 1)
        elseif nproc == 4
            return (2, 2, 1)
        elseif nproc == 8
            return (2, 2, 2)
        else
            return (nproc, 1, 1)
        end
    else
        # For larger cases, try to find good 3D factorization
        cube_root = round(Int, nproc^(1/3))

        # Search around cube root
        for i in max(1, cube_root-2):cube_root+2
            if nproc % i == 0
                remaining = nproc ÷ i
                sqrt_remaining = round(Int, sqrt(remaining))

                for j in max(1, sqrt_remaining-1):sqrt_remaining+1
                    if remaining % j == 0
                        k = remaining ÷ j
                        if i * j * k == nproc
                            return (i, j, k)
                        end
                    end
                end
            end
        end

        # Fallback to 2D decomposition in z-plane
        mesh_2d = create_2d_process_mesh(nproc)
        return (mesh_2d[1], mesh_2d[2], 1)
    end
end
