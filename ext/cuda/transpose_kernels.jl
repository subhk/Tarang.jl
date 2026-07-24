# ============================================================================
# GPU Kernels for Distributed Transpose Operations
# ============================================================================

"""
GPU kernels for efficient pack/unpack operations during distributed transposes.
These kernels enable high-performance MPI communication by preparing data
in contiguous buffers suitable for MPI.Alltoallv.

The pack operation reorganizes data from a multidimensional array into a
flat buffer with chunks destined for different MPI ranks.

The unpack operation reverses this, taking received data and placing it
into the appropriate positions in the output array.
"""

# Binary search helpers (_gpu_find_rank, _gpu_find_rank_1based) are defined in
# nccl_transpose.jl which is loaded before this file.

# ============================================================================
# Pack Kernels for Transpose
# ============================================================================

"""
    pack_for_transpose_kernel_3d!(buffer, data, Nx, Ny, Nz, nranks, dim, chunk_sizes, displs)

Pack 3D array data into contiguous buffer for MPI.Alltoallv.
Uses binary search via _gpu_find_rank for O(log P) rank lookup per element.

Arguments:
- buffer: Output flat buffer
- data: Input 3D array
- Nx, Ny, Nz: Array dimensions
- nranks: Number of MPI ranks in the transpose communicator
- dim: Dimension being redistributed (1=x, 2=y, 3=z)
- chunk_sizes: Array of chunk sizes for each rank (for local index computation)
- displs: Buffer displacements (prefix sums of element counts)
- prefix_sums: Cumulative sum of chunk_sizes (for binary search rank lookup)
"""
@kernel function pack_for_transpose_kernel_3d!(buffer, @Const(data),
                                               Nx::Int, Ny::Int, Nz::Int,
                                               nranks::Int, dim::Int,
                                               @Const(chunk_sizes), @Const(displs),
                                               @Const(prefix_sums))
    i = @index(Global)

    # Convert linear index to 3D indices (column-major order)
    ix = ((i - 1) % Nx) + 1
    iy = (((i - 1) ÷ Nx) % Ny) + 1
    iz = ((i - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank this element goes to using binary search
    if dim == 3  # Z→Y transpose: z-dimension being redistributed
        rank, z_offset = _gpu_find_rank(iz, prefix_sums, nranks)
        local_iz = iz - z_offset
        local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx

    elseif dim == 2  # Y→X transpose: y-dimension being redistributed
        rank, y_offset = _gpu_find_rank(iy, prefix_sums, nranks)
        local_iy = iy - y_offset
        local_idx = (iz - 1) * Nx * chunk_sizes[rank + 1] + (local_iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx

    else  # dim == 1: X being redistributed
        rank, x_offset = _gpu_find_rank(ix, prefix_sums, nranks)
        local_ix = ix - x_offset
        local_idx = (iz - 1) * chunk_sizes[rank + 1] * Ny + (iy - 1) * chunk_sizes[rank + 1] + local_ix
        buf_idx = displs[rank + 1] + local_idx
    end

    @inbounds buffer[buf_idx] = data[ix, iy, iz]
end

"""
    pack_for_transpose_kernel_2d!(buffer, data, Nx, Ny, nranks, dim, chunk_sizes, displs, prefix_sums)

Pack 2D array data into contiguous buffer for MPI.Alltoallv.
Uses binary search for O(log P) rank lookup.
"""
@kernel function pack_for_transpose_kernel_2d!(buffer, @Const(data),
                                               Nx::Int, Ny::Int,
                                               nranks::Int, dim::Int,
                                               @Const(chunk_sizes), @Const(displs),
                                               @Const(prefix_sums))
    i = @index(Global)

    ix = ((i - 1) % Nx) + 1
    iy = ((i - 1) ÷ Nx) + 1

    if dim == 2  # Y being redistributed
        rank, y_offset = _gpu_find_rank(iy, prefix_sums, nranks)
        local_iy = iy - y_offset
        local_idx = (local_iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx
    else  # dim == 1: X being redistributed
        rank, x_offset = _gpu_find_rank(ix, prefix_sums, nranks)
        local_ix = ix - x_offset
        local_idx = (iy - 1) * chunk_sizes[rank + 1] + local_ix
        buf_idx = displs[rank + 1] + local_idx
    end

    @inbounds buffer[buf_idx] = data[ix, iy]
end

# ============================================================================
# Unpack Kernels for Transpose
# ============================================================================

"""
    unpack_from_transpose_kernel_3d!(data, buffer, Nx, Ny, Nz, nranks, dim, chunk_sizes, displs, prefix_sums)

Unpack data from flat buffer after MPI.Alltoallv into 3D array.
Uses binary search for O(log P) rank lookup.
"""
@kernel function unpack_from_transpose_kernel_3d!(data, @Const(buffer),
                                                  Nx::Int, Ny::Int, Nz::Int,
                                                  nranks::Int, dim::Int,
                                                  @Const(chunk_sizes), @Const(displs),
                                                  @Const(prefix_sums))
    i = @index(Global)

    ix = ((i - 1) % Nx) + 1
    iy = (((i - 1) ÷ Nx) % Ny) + 1
    iz = ((i - 1) ÷ (Nx * Ny)) + 1

    if dim == 2  # After Z→Y transpose: receiving y-chunks
        rank, y_offset = _gpu_find_rank(iy, prefix_sums, nranks)
        local_iy = iy - y_offset
        local_idx = (iz - 1) * Nx * chunk_sizes[rank + 1] + (local_iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx
    elseif dim == 1  # After Y→X transpose: receiving x-chunks
        rank, x_offset = _gpu_find_rank(ix, prefix_sums, nranks)
        local_ix = ix - x_offset
        local_idx = (iz - 1) * chunk_sizes[rank + 1] * Ny + (iy - 1) * chunk_sizes[rank + 1] + local_ix
        buf_idx = displs[rank + 1] + local_idx
    else  # dim == 3: receiving z-chunks
        rank, z_offset = _gpu_find_rank(iz, prefix_sums, nranks)
        local_iz = iz - z_offset
        local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx
    end

    @inbounds data[ix, iy, iz] = buffer[buf_idx]
end

"""
    unpack_from_transpose_kernel_2d!(data, buffer, Nx, Ny, nranks, dim, chunk_sizes, displs, prefix_sums)

Unpack data from flat buffer after MPI.Alltoallv into 2D array.
Uses binary search for O(log P) rank lookup.
"""
@kernel function unpack_from_transpose_kernel_2d!(data, @Const(buffer),
                                                  Nx::Int, Ny::Int,
                                                  nranks::Int, dim::Int,
                                                  @Const(chunk_sizes), @Const(displs),
                                                  @Const(prefix_sums))
    i = @index(Global)

    ix = ((i - 1) % Nx) + 1
    iy = ((i - 1) ÷ Nx) + 1

    if dim == 1  # Receiving x-chunks
        rank, x_offset = _gpu_find_rank(ix, prefix_sums, nranks)
        local_ix = ix - x_offset
        local_idx = (iy - 1) * chunk_sizes[rank + 1] + local_ix
        buf_idx = displs[rank + 1] + local_idx
    else  # dim == 2: receiving y-chunks
        rank, y_offset = _gpu_find_rank(iy, prefix_sums, nranks)
        local_iy = iy - y_offset
        local_idx = (local_iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx
    end

    @inbounds data[ix, iy] = buffer[buf_idx]
end

# ============================================================================
# Simple Copy Kernel (for serial or single-rank cases)
# ============================================================================

"""
    copy_to_buffer_kernel!(buffer, data)

Simple kernel to copy data to flat buffer.
"""
@kernel function copy_to_buffer_kernel!(buffer, @Const(data))
    i = @index(Global)
    if i <= length(data)
        @inbounds buffer[i] = data[i]
    end
end

"""
    copy_from_buffer_kernel!(data, buffer)

Simple kernel to copy from flat buffer to data array.
"""
@kernel function copy_from_buffer_kernel!(data, @Const(buffer))
    i = @index(Global)
    if i <= length(data)
        @inbounds data[i] = buffer[i]
    end
end

# ============================================================================
# GPU FFT in Dimension Kernel Helpers
# ============================================================================

"""
    apply_fft_normalization_kernel!(data, scale)

Apply FFT normalization factor.
"""
@kernel function apply_fft_normalization_kernel!(data, scale)
    i = @index(Global)
    if i <= length(data)
        @inbounds data[i] = data[i] * scale
    end
end

# ============================================================================
# Launcher Functions
# ============================================================================

# Constants for kernel operations
const GPU_PACK_3D_OP = KernelOperation(pack_for_transpose_kernel_3d!) do buffer, data, Nx, Ny, Nz, _, _, _, _, _
    Nx * Ny * Nz
end

const GPU_UNPACK_3D_OP = KernelOperation(unpack_from_transpose_kernel_3d!) do data, buffer, Nx, Ny, Nz, _, _, _, _, _
    Nx * Ny * Nz
end

const GPU_PACK_2D_OP = KernelOperation(pack_for_transpose_kernel_2d!) do buffer, data, Nx, Ny, _, _, _, _, _
    Nx * Ny
end

const GPU_UNPACK_2D_OP = KernelOperation(unpack_from_transpose_kernel_2d!) do data, buffer, Nx, Ny, _, _, _, _, _
    Nx * Ny
end

"""
    _validate_chunk_divisibility(count, divisor, dim, rank, operation)

Validate that count is evenly divisible by divisor for chunk size computation.
Throws ArgumentError if not divisible.
"""
function _validate_chunk_divisibility(count::Int, divisor::Int, dim::Int, rank::Int, operation::String)
    if divisor == 0
        throw(ArgumentError("GPU $operation: divisor is zero for dim=$dim, rank=$rank. " *
                           "This indicates zero array dimensions, which is invalid."))
    end
    if count % divisor != 0
        throw(ArgumentError("GPU $operation: count=$count is not evenly divisible by divisor=$divisor " *
                           "for dim=$dim, rank=$rank. This indicates misaligned MPI counts/displs. " *
                           "Check that array dimensions are compatible with the MPI decomposition."))
    end
end

# Convert small CPU int arrays to GPU. These are tiny (2-8 elements for MPI
# process counts), so allocation cost is negligible — no caching needed.
_to_gpu(cpu_array::Vector{Int}) = CuArray(cpu_array)

"""
    gpu_pack_for_transpose!(buffer, data, counts, displs, dim, nranks)

Launch GPU kernel to pack data for transpose operation.
"""
function gpu_pack_for_transpose!(buffer::CuArray, data::CuArray,
                                 counts::Vector{Int}, displs::Vector{Int},
                                 dim::Int, nranks::Int;
                                 synchronize::Bool=true)
    ndims_data = ndims(data)

    # Ensure we're on the device where data lives for allocations, kernel launch, and sync
    data_device = CUDA.device(data)
    prev_device = CUDA.device()
    CUDA.device!(data_device)

    if ndims_data == 3
        Nx, Ny, Nz = size(data)
        n_elements = Nx * Ny * Nz

        # Compute chunk_sizes from counts (kernel expects chunk size, not total count)
        # CRITICAL: Validate divisibility to catch MPI count/displ mismatches early
        chunk_sizes = zeros(Int, nranks)
        if dim == 3
            divisor = Nx * Ny
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], divisor, dim, r, "pack")
                chunk_sizes[r] = counts[r] ÷ divisor
            end
        elseif dim == 2
            divisor = Nx * Nz
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], divisor, dim, r, "pack")
                chunk_sizes[r] = counts[r] ÷ divisor
            end
        else  # dim == 1
            divisor = Ny * Nz
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], divisor, dim, r, "pack")
                chunk_sizes[r] = counts[r] ÷ divisor
            end
        end

        chunk_sizes_gpu = _to_gpu(chunk_sizes)
        displs_gpu = _to_gpu(displs)
        prefix_sums_gpu = _to_gpu(cumsum(chunk_sizes))

        kernel = pack_for_transpose_kernel_3d!(CUDABackend())
        kernel(buffer, data, Nx, Ny, Nz, nranks, dim, chunk_sizes_gpu, displs_gpu, prefix_sums_gpu;
               ndrange=n_elements)

    elseif ndims_data == 2
        Nx, Ny = size(data)
        n_elements = Nx * Ny

        # Compute chunk_sizes from counts
        # CRITICAL: Validate divisibility to catch MPI count/displ mismatches early
        chunk_sizes = zeros(Int, nranks)
        if dim == 2
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], Nx, dim, r, "pack")
                chunk_sizes[r] = counts[r] ÷ Nx
            end
        else  # dim == 1
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], Ny, dim, r, "pack")
                chunk_sizes[r] = counts[r] ÷ Ny
            end
        end

        chunk_sizes_gpu = _to_gpu(chunk_sizes)
        displs_gpu = _to_gpu(displs)
        prefix_sums_gpu = _to_gpu(cumsum(chunk_sizes))

        kernel = pack_for_transpose_kernel_2d!(CUDABackend())
        kernel(buffer, data, Nx, Ny, nranks, dim, chunk_sizes_gpu, displs_gpu, prefix_sums_gpu;
               ndrange=n_elements)
    else
        # Fallback: simple copy
        copyto!(view(buffer, 1:length(data)), vec(data))
    end

    synchronize && CUDA.synchronize()
    CUDA.device!(prev_device)
    return buffer
end

"""
    gpu_unpack_from_transpose!(data, buffer, counts, displs, dim, nranks; synchronize=true)

Launch GPU kernel to unpack data after transpose operation.
"""
function gpu_unpack_from_transpose!(data::CuArray, buffer::CuArray,
                                    counts::Vector{Int}, displs::Vector{Int},
                                    dim::Int, nranks::Int;
                                    synchronize::Bool=true)
    ndims_data = ndims(data)

    # Ensure we're on the device where data lives for allocations, kernel launch, and sync
    data_device = CUDA.device(data)
    prev_device = CUDA.device()
    CUDA.device!(data_device)

    if ndims_data == 3
        Nx, Ny, Nz = size(data)
        n_elements = Nx * Ny * Nz

        # Compute chunk_sizes from counts (kernel expects chunk size, not total count)
        # CRITICAL: Validate divisibility to catch MPI count/displ mismatches early
        chunk_sizes = zeros(Int, nranks)
        if dim == 2  # After Z→Y: receiving y-chunks
            divisor = Nx * Nz
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], divisor, dim, r, "unpack")
                chunk_sizes[r] = counts[r] ÷ divisor
            end
        elseif dim == 1  # After Y→X: receiving x-chunks
            divisor = Ny * Nz
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], divisor, dim, r, "unpack")
                chunk_sizes[r] = counts[r] ÷ divisor
            end
        else  # dim == 3: receiving z-chunks
            divisor = Nx * Ny
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], divisor, dim, r, "unpack")
                chunk_sizes[r] = counts[r] ÷ divisor
            end
        end

        chunk_sizes_gpu = _to_gpu(chunk_sizes)
        displs_gpu = _to_gpu(displs)
        prefix_sums_gpu = _to_gpu(cumsum(chunk_sizes))

        kernel = unpack_from_transpose_kernel_3d!(CUDABackend())
        kernel(data, buffer, Nx, Ny, Nz, nranks, dim, chunk_sizes_gpu, displs_gpu, prefix_sums_gpu;
               ndrange=n_elements)

    elseif ndims_data == 2
        Nx, Ny = size(data)
        n_elements = Nx * Ny

        # Compute chunk_sizes from counts
        # CRITICAL: Validate divisibility to catch MPI count/displ mismatches early
        chunk_sizes = zeros(Int, nranks)
        if dim == 2  # Receiving y-chunks
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], Nx, dim, r, "unpack")
                chunk_sizes[r] = counts[r] ÷ Nx
            end
        else  # dim == 1: Receiving x-chunks
            for r in 1:nranks
                _validate_chunk_divisibility(counts[r], Ny, dim, r, "unpack")
                chunk_sizes[r] = counts[r] ÷ Ny
            end
        end

        chunk_sizes_gpu = _to_gpu(chunk_sizes)
        displs_gpu = _to_gpu(displs)
        prefix_sums_gpu = _to_gpu(cumsum(chunk_sizes))

        kernel = unpack_from_transpose_kernel_2d!(CUDABackend())
        kernel(data, buffer, Nx, Ny, nranks, dim, chunk_sizes_gpu, displs_gpu, prefix_sums_gpu;
               ndrange=n_elements)
    else
        copyto!(vec(data), view(buffer, 1:length(data)))
    end

    synchronize && CUDA.synchronize()
    CUDA.device!(prev_device)
    return data
end

# ============================================================================
# Override pack/unpack for GPU architecture
# ============================================================================

"""
Override pack_for_transpose! for GPU architecture.
"""
function Tarang.pack_for_transpose!(buffer::CuArray, data::CuArray,
                                    counts::Vector{Int}, displs::Vector{Int},
                                    dim::Int, nranks::Int, arch::Tarang.GPU)
    return gpu_pack_for_transpose!(buffer, data, counts, displs, dim, nranks)
end

"""
Override unpack_from_transpose! for GPU architecture.
"""
function Tarang.unpack_from_transpose!(data::CuArray, buffer::CuArray,
                                       counts::Vector{Int}, displs::Vector{Int},
                                       dim::Int, nranks::Int, arch::Tarang.GPU)
    return gpu_unpack_from_transpose!(data, buffer, counts, displs, dim, nranks)
end

# ============================================================================
# Override _gpu_pack_for_transpose! for CUDA-aware MPI path
# ============================================================================

"""
    Tarang._gpu_pack_for_transpose!(send_buf::CuArray, data::CuArray, dim::Int, config::Tarang.DistributedGPUConfig)

Override for CuArray arguments in the CUDA-aware MPI distributed FFT path.

The default implementation in gpu_distributed.jl does a flat `copyto!` which is
incorrect when the dimension being redistributed is not the last (column-major)
dimension. This override uses GPU pack kernels to correctly rearrange data so
that each rank's portion is contiguous in the send buffer.

Handles both forward and reverse transpose directions:
- Forward: `dims[dim] < global_n` → dim is distributed, pack splits along other_dim
- Reverse: `dims[dim] == global_n` → dim is local (full), pack splits along dim
"""
function Tarang._gpu_pack_for_transpose!(send_buf::CuArray, data::CuArray,
                                          dim::Int, config::Tarang.DistributedGPUConfig)
    nprocs = config.size
    ndims_data = ndims(data)
    dims = size(data)

    if nprocs == 1
        copyto!(send_buf, vec(data))
        return send_buf
    end

    global_n = config.global_shape[dim]
    other_dim = dim == ndims_data ? 1 : ndims_data

    # Determine pack direction from data shape:
    # Forward: dim is distributed (small), split other_dim among ranks
    # Reverse: dim is local (global_n), split dim among ranks
    # Degenerate global_n == 1: the shapes cannot distinguish the directions
    # (dims[dim] == global_n either way). Classify as FORWARD so the pack
    # matches the forward `Tarang._compute_alltoall_counts` (split other_dim),
    # which is what `mpi_alltoall_transpose` uses for the alltoallv itself.
    if dims[dim] < global_n || global_n == 1
        # Forward: split along other_dim
        split_dim = other_dim
        split_n = dims[split_dim]
    else
        # Reverse: split along dim (now fully local)
        split_dim = dim
        split_n = dims[split_dim]
    end

    # Compute per-rank chunk sizes along split_dim
    remaining = div(prod(dims), split_n)
    counts = Vector{Int}(undef, nprocs)
    displs = Vector{Int}(undef, nprocs)
    offset = 0
    for r in 1:nprocs
        chunk_r = div(split_n, nprocs) + (r - 1 < mod(split_n, nprocs) ? 1 : 0)
        counts[r] = chunk_r * remaining
        displs[r] = offset
        offset += counts[r]
    end

    gpu_pack_for_transpose!(send_buf, data, counts, displs, split_dim, nprocs)
    return send_buf
end

"""
    Tarang._gpu_unpack_from_transpose!(output::CuArray, recv_buf::CuArray, dim::Int, config::Tarang.DistributedGPUConfig)

Override for CuArray arguments. Unpacks the received MPI buffer into the
correctly-shaped output array using GPU kernels.

Handles both directions:
- Forward: output has `dim` of size global_n → assemble along dim
- Reverse: output has `dim` of size local_n → assemble along other_dim
"""
function Tarang._gpu_unpack_from_transpose!(output::CuArray, recv_buf::CuArray,
                                             dim::Int, config::Tarang.DistributedGPUConfig)
    nprocs = config.size
    ndims_data = ndims(output)
    out_dims = size(output)

    if nprocs == 1
        copyto!(vec(output), recv_buf)
        return output
    end

    global_n = config.global_shape[dim]
    other_dim = dim == ndims_data ? 1 : ndims_data

    # Determine unpack direction from output shape:
    # Forward: output[dim] == global_n → assemble along dim from per-rank chunks
    # Reverse: output[dim] < global_n → assemble along other_dim
    # Degenerate global_n == 1: ambiguous by shape (out_dims[dim] is 1 == global_n
    # on the owning rank, 0 elsewhere). Classify as FORWARD on every rank, matching
    # the pack side and the forward `Tarang._compute_alltoall_counts` recv counts.
    if out_dims[dim] == global_n || global_n == 1
        # Forward unpack: assemble along dim
        assemble_dim = dim
        assemble_n = global_n
    else
        # Reverse unpack: assemble along other_dim (restoring original layout)
        assemble_dim = other_dim
        assemble_n = out_dims[other_dim]
    end

    # Compute per-rank chunk sizes along assemble_dim
    remaining = div(prod(out_dims), assemble_n)
    counts = Vector{Int}(undef, nprocs)
    displs = Vector{Int}(undef, nprocs)
    offset = 0
    for r in 1:nprocs
        chunk_r = div(assemble_n, nprocs) + (r - 1 < mod(assemble_n, nprocs) ? 1 : 0)
        counts[r] = chunk_r * remaining
        displs[r] = offset
        offset += counts[r]
    end

    gpu_unpack_from_transpose!(output, recv_buf, counts, displs, assemble_dim, nprocs)
    return output
end

# ============================================================================
# GPU FFT in dimension
# ============================================================================

"""
Override fft_in_dim! for GPU arrays using CUFFT.

Accepts the `plan` keyword its callers always pass (see `transform_in_dim!` in
src/core/transpose/transpose_transforms.jl) but ignores it: the plan handed down
may be a CPU FFTW plan or a placeholder, so the cached CUFFT plan is used instead.
"""
function Tarang.fft_in_dim!(data::CuArray, dim::Int, direction::Symbol, arch::Tarang.GPU;
                            plan=nothing)
    # Use cached CUFFT plan to avoid expensive plan creation per call
    cufft_plan = get_fft_1d_plan(size(data), dim, eltype(data); inverse=(direction != :forward))
    data .= cufft_plan * data
    CUDA.synchronize()
    return data
end

# ============================================================================
# GPU DCT in dimension (for Chebyshev transforms)
# ============================================================================

"""Undo the odd-degree sign flip along `dim` of a 3D array in place:
`a[k, …] *= (-1)^(k-1)` (1-based `k` along `dim`). Since
`REDFT00(reverse(x))[k] = (-1)^k REDFT00(x)[k]`, this converts between the
reversed-grid DCT-I convention of `gpu_dct1_along_dim!` and the plain (unflipped)
REDFT00 convention of the CPU distributed reference."""
function _dct1_undo_odd_flip!(a::CuArray{T,3}, dim::Int) where {T}
    n = size(a, dim)
    sgn = CuArray(T[iseven(k) ? -one(T) : one(T) for k in 1:n])
    a .*= reshape(sgn, ntuple(i -> i == dim ? n : 1, 3))
    return a
end

"""
Override dct_in_dim! for GPU arrays.

Tarang's Chebyshev transform is DCT-I (REDFT00) on the Gauss–Lobatto grid, so
this must match the CPU distributed reference `Tarang.dct_in_dim!(…, ::CPU)` in
src/core/transpose/transpose_transforms.jl EXACTLY:
  forward:  REDFT00, 1/(N-1) normalization, half-weight at both endpoints,
            NO odd-degree sign flip
  backward: double both endpoint coefficients, REDFT00, divide by 2

The verified GPU DCT-I building block `gpu_dct1_along_dim!` (ext/cuda/cheb_deriv.jl)
implements the same transform but in the reversed-grid convention of
transform_chebyshev.jl, whose output is `(-1)^k ×` the CPU reference here
(a grid reversal ≡ an odd-degree coefficient sign flip). We therefore undo that
flip explicitly: AFTER the forward transform, and on the coefficients BEFORE the
backward transform.
"""
function Tarang.dct_in_dim!(data::CuArray{T,N}, dim::Int, direction::Symbol, arch::Tarang.GPU) where {T,N}
    # Handle complex data by transforming real and imaginary parts separately
    if T <: Complex
        # Use GPU-native real/imag extraction (no host round-trip)
        real_part = real.(data)
        imag_part = imag.(data)

        Tarang.dct_in_dim!(real_part, dim, direction, arch)
        Tarang.dct_in_dim!(imag_part, dim, direction, arch)

        data .= complex.(real_part, imag_part)
        CUDA.synchronize()
        return data
    end

    # Real data path — DCT-I via the verified gpu_dct1_along_dim! (3D kernels).
    n = size(data, dim)
    n <= 1 && return data

    if N > 3 || !(T <: AbstractFloat)
        error("GPU DCT-I supports real floating-point arrays with at most three " *
              "dimensions; got $(typeof(data)). CPU fallback is disabled.")
    end

    # Reshape 1D/2D input to 3D (CuArray reshape shares storage, so writing
    # through data3 writes data). `dim` stays valid: trailing dims are appended.
    data3 = N == 3 ? data : reshape(data, size(data)..., ntuple(_ -> 1, 3 - N)...)
    out3 = similar(data3)

    if direction == :forward
        gpu_dct1_along_dim!(out3, data3, dim, :forward)
        _dct1_undo_odd_flip!(out3, dim)
        data3 .= out3
    else
        _dct1_undo_odd_flip!(data3, dim)
        gpu_dct1_along_dim!(out3, data3, dim, :backward)
        data3 .= out3
    end

    CUDA.synchronize()
    return data
end

# ============================================================================
# Exports
# ============================================================================

export pack_for_transpose_kernel_3d!, pack_for_transpose_kernel_2d!
export unpack_from_transpose_kernel_3d!, unpack_from_transpose_kernel_2d!
export copy_to_buffer_kernel!, copy_from_buffer_kernel!
export apply_fft_normalization_kernel!
export gpu_pack_for_transpose!, gpu_unpack_from_transpose!
export GPU_PACK_3D_OP, GPU_UNPACK_3D_OP, GPU_PACK_2D_OP, GPU_UNPACK_2D_OP
