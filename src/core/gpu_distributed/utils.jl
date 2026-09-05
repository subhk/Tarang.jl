# ============================================================================
# Utility Functions
# ============================================================================

"""
    check_cuda_aware_mpi()

Check if MPI implementation is CUDA-aware.

Detection priority:
1. Explicit user override via `TARANG_CUDA_AWARE_MPI` env var ("1"=enabled, "0"=disabled)
2. MPI.jl's implementation-aware `MPI.has_cuda()` capability probe (after MPI
   initialization, or when its explicit override is present)
3. OpenMPI CUDA support indicator
4. MVAPICH2 CUDA indicator
5. MPICH/Cray GPU indicators

Returns false by default if no positive indicator is found.
"""
function check_cuda_aware_mpi()
    # Priority 1: Explicit user override
    if haskey(ENV, "TARANG_CUDA_AWARE_MPI")
        val = uppercase(strip(ENV["TARANG_CUDA_AWARE_MPI"]))
        if val in ("1", "TRUE", "YES")
            @info "CUDA-aware MPI enabled via TARANG_CUDA_AWARE_MPI environment variable"
            return true
        elseif val in ("0", "FALSE", "NO")
            return false
        end
    end

    # Priority 2: Prefer MPI.jl's native capability query. For Open MPI this
    # uses MPIX_Query_cuda_support(), and JULIA_MPI_HAS_CUDA provides MPI.jl's
    # supported override for implementations that cannot be queried directly.
    try
        probe_is_safe = MPI.Initialized() || haskey(ENV, "JULIA_MPI_HAS_CUDA")
        if isdefined(MPI, :has_cuda) && probe_is_safe && MPI.has_cuda()
            return true
        end
    catch err
        # Some Open MPI builds do not expose the optional
        # MPIX_Query_cuda_support symbol that MPI.jl probes. That compatibility
        # miss is safe to fall back from; a malformed JULIA_MPI_HAS_CUDA override
        # and every other probe failure must reach the caller, or a distributed
        # GPU run silently downgrades to "no CUDA-aware MPI".
        missing_query_symbol = err isa ErrorException &&
                               occursin("could not load symbol", err.msg) &&
                               occursin("MPIX_Query_cuda_support", err.msg)
        missing_query_symbol || rethrow()
        @debug "MPI.jl CUDA-awareness probe failed" exception = err
    end

    # Priority 3-5: Library-specific indicators. Guarded `ENV` lookups do not
    # need exception-driven fallback.
    # OpenMPI with CUDA support
    if get(ENV, "OMPI_MCA_opal_cuda_support", "") == "true"
        return true
    end
    # MVAPICH2 with CUDA
    if get(ENV, "MV2_USE_CUDA", "") == "1"
        return true
    end
    # MPICH with GPU support
    if get(ENV, "MPIR_CVAR_ENABLE_GPU", "") == "1"
        return true
    end
    # Cray MPI with GPU support
    if get(ENV, "MPICH_GPU_SUPPORT_ENABLED", "") == "1"
        return true
    end

    return false
end

"""
    setup_distributed_gpu!(dist::Distributor)

Setup distributed GPU computing for a Distributor.
"""
function setup_distributed_gpu!(dist)
    if !is_gpu(dist.architecture)
        return nothing
    end

    if dist.size == 1
        # Single GPU, no distribution needed
        @info "Single GPU mode - no distributed setup needed"
        return nothing
    end

    # Check for CUDA-aware MPI
    cuda_aware = check_cuda_aware_mpi()

    @info "Setting up distributed GPU computing"
    @info "  MPI processes: $(dist.size)"
    @info "  CUDA-aware MPI: $(cuda_aware)"

    # Create distributed GPU config
    # Will be populated when domain is created
    return cuda_aware
end

