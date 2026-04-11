# ============================================================================
# Pencil Linear Operator for Chebyshev-Fourier IMEX Methods (MPI-compatible)
# ============================================================================
#
# For Chebyshev-Fourier domains, the linear operator is NOT diagonal in spectral
# space because Chebyshev differentiation couples modes. However, we can exploit
# the pencil decomposition:
#
# For a 3D domain with (Fourier_x, Fourier_y, Chebyshev_z):
#   - Each (kx, ky) Fourier mode has a 1D Chebyshev problem in z
#   - The linear operator L = -ν∇² becomes: L(kx,ky) = ν(kx² + ky²)*I - ν*D²_z
#   - Each (kx, ky) mode can be solved independently
#
# This enables MPI parallelization:
#   - Data distributed across Fourier modes (kx, ky)
#   - Each rank solves its local 1D Chebyshev problems
#   - No global communication needed for the linear solve
#
# ============================================================================

# LinearAlgebra, SparseArrays already in Tarang.jl

"""
    PencilLinearOperator{T}

Linear operator for pencil-based IMEX timestepping with Chebyshev-Fourier domains.

Stores per-wavenumber 1D matrices that allow independent solves for each Fourier mode.
This enables implicit treatment of linear terms (like diffusion) in MPI-distributed
simulations.

# Fields
- `Nz::Int`: Number of Chebyshev modes in the non-Fourier direction
- `local_kx_range::UnitRange{Int}`: Local range of kx indices on this rank
- `local_ky_range::UnitRange{Int}`: Local range of ky indices on this rank
- `k2_values::Matrix{T}`: k² = kx² + ky² values for local modes
- `chebyshev_D2::SparseMatrixCSC{T,Int}`: 1D Chebyshev second derivative matrix
- `operator_type::Symbol`: Type of operator (:laplacian, :hyperviscosity)
- `parameters::Dict{Symbol, Any}`: Parameters (ν, order, etc.)
- `chebyshev_basis_idx::Int`: Index of Chebyshev basis (typically last)
- `_rhs_buf::Vector{Complex{T}}`: Pre-allocated RHS workspace buffer (length Nz)
- `_sol_buf::Vector{Complex{T}}`: Pre-allocated solution workspace buffer (length Nz)

# Example
```julia
# Create for 3D Chebyshev-Fourier domain
L = PencilLinearOperator(dist, bases, :laplacian; ν=1e-3)

# Apply implicit step for each pencil
pencil_implicit_solve!(u_new, u_old, L, dt, γ)
```
"""
struct PencilLinearOperator{T<:AbstractFloat}
    Nz::Int                                      # Chebyshev dimension size
    local_kx_range::UnitRange{Int}               # Local kx indices
    local_ky_range::UnitRange{Int}               # Local ky indices
    k2_values::Matrix{T}                         # k² for each local (kx, ky)
    chebyshev_D2::SparseMatrixCSC{T,Int}         # 1D Chebyshev D² matrix
    operator_type::Symbol
    parameters::Dict{Symbol, Any}
    chebyshev_basis_idx::Int
    fourier_basis_indices::Vector{Int}           # Indices of Fourier bases
    # Pre-allocated workspace buffers for zero-alloc wavenumber loop
    _rhs_buf::Vector{Complex{T}}
    _sol_buf::Vector{Complex{T}}
end

# ============================================================================
# Solve Layout Utilities
# ============================================================================
# The pencil IMEX solve requires Chebyshev-local data: all Nz physical points
# for each Fourier mode. PencilFFTs output may have Chebyshev distributed
# (the last FFT dimension ends up local, pushing Chebyshev into the decomposed
# set). These utilities transpose between PencilFFT output and the "solve
# layout" where Fourier modes are distributed and Chebyshev is local.
# ============================================================================

"""Compute local range for a Fourier axis distributed across `mesh_dim` in solve layout."""
function _solve_layout_range(dist::Distributor, global_size::Int, mesh_dim::Int)
    if dist.size <= 1 || dist.mesh === nothing || mesh_dim > length(dist.mesh)
        return (1, global_size)
    end
    n_procs = dist.mesh[mesh_dim]
    n_procs <= 1 && return (1, global_size)
    coord = get_process_coordinate(dist, mesh_dim)
    base = div(global_size, n_procs)
    rem = mod(global_size, n_procs)
    start = coord * base + min(coord, rem) + 1
    count = base + (coord < rem ? 1 : 0)
    return (start, start + count - 1)
end

"""Block partition: returns (counts, offsets) vectors for 0-indexed ranks."""
function _block_counts(global_size::Int, nprocs::Int)
    base = div(global_size, nprocs)
    rem = mod(global_size, nprocs)
    counts = [base + (r < rem ? 1 : 0) for r in 0:nprocs-1]
    offsets = [sum(counts[1:r]) for r in 0:nprocs]
    return counts, offsets
end

"""Check if MPI transpose is needed between PencilFFT output and solve layout."""
_needs_solve_transpose(dist::Distributor) = dist.use_pencil_arrays && dist.size > 1

"""
Check if MPI transpose is needed, accounting for actual data layout.
Returns false if Chebyshev is already local (nz_local == Nz).
"""
function _needs_solve_transpose(dist::Distributor, data, L::PencilLinearOperator)
    dist.use_pencil_arrays && dist.size > 1 || return false
    nz_local = size(data, L.chebyshev_basis_idx)
    return nz_local != L.Nz
end

"""
Get or create cached partition info and MPI buffers for solve-layout transposes.
Stored in `cache_dict` (typically `state.timestepper_data`) to avoid repeated
MPI.Allgather calls and buffer allocations across timesteps.
"""
function _get_transpose_info!(cache_dict::Dict, dist::Distributor, L::PencilLinearOperator,
                              nkx_global::Int, nz_local::Int, ::Type{T}) where T
    key = :solve_transpose_info
    if haskey(cache_dict, key)
        info = cache_dict[key]
        if info.nkx_global == nkx_global && info.nz_local == nz_local && eltype(info.send_buf) === T
            return info
        end
    end

    nprocs = dist.size
    rank = dist.rank
    nz_global = L.Nz

    kx_counts, kx_offs = _block_counts(nkx_global, nprocs)
    z_counts = MPI.Allgather(nz_local, dist.comm)
    z_offs = [sum(z_counts[1:r]) for r in 0:nprocs]
    nkx_local = kx_counts[rank + 1]

    fwd_s = [kx_counts[j+1] * nz_local for j in 0:nprocs-1]
    fwd_r = [nkx_local * z_counts[j+1] for j in 0:nprocs-1]
    max_buf = max(sum(fwd_s), sum(fwd_r))

    info = (
        nkx_global = nkx_global, nz_local = nz_local, nz_global = nz_global,
        nkx_local = nkx_local,
        kx_counts = kx_counts, kx_offs = kx_offs,
        z_counts = z_counts, z_offs = z_offs,
        fwd_s_counts = fwd_s, fwd_r_counts = fwd_r,
        send_buf = Vector{T}(undef, max_buf),
        recv_buf = Vector{T}(undef, max_buf),
    )
    cache_dict[key] = info
    return info
end

"""
Transpose coefficient data from PencilFFT output layout to Chebyshev-local solve layout.
For serial, returns the underlying array directly (no copy).
Pass `cache` (a Dict, e.g. `state.timestepper_data`) to reuse partition info and MPI buffers.
"""
function _to_solve_layout(data, dist::Distributor, L::PencilLinearOperator{T};
                          cache::Union{Nothing,Dict}=nothing) where T
    if !_needs_solve_transpose(dist, data, L)
        # Serial or Chebyshev already local: return plain array in logical order
        return data isa PencilArrays.PencilArray ? collect(data) : data
    end
    # Use collect() to get logically-ordered array (parent() may have permuted dims)
    src = data isa PencilArrays.PencilArray ? collect(data) : data
    if length(L.fourier_basis_indices) == 1
        return _transpose_fft_to_solve_2d(src, dist, L, cache)
    elseif length(L.fourier_basis_indices) == 2
        return _transpose_fft_to_solve_3d(src, dist, L, cache)
    else
        error("Solve transpose not supported for $(length(L.fourier_basis_indices)) Fourier dimensions.")
    end
end

"""Transpose solve-layout data back into PencilFFT output layout."""
function _from_solve_layout!(dest, src::AbstractArray, dist::Distributor, L::PencilLinearOperator{T};
                             cache::Union{Nothing,Dict}=nothing) where T
    if !_needs_solve_transpose(dist, dest, L)
        dest .= src
        return
    end
    if length(L.fourier_basis_indices) == 1
        _transpose_solve_to_fft_2d!(dest, src, dist, L, cache)
    elseif length(L.fourier_basis_indices) == 2
        _transpose_solve_to_fft_3d!(dest, src, dist, L, cache)
    else
        error("Solve transpose not supported for $(length(L.fourier_basis_indices)) Fourier dimensions.")
    end
end

"""2D FFT-output → solve-layout transpose via MPI.Alltoallv."""
function _transpose_fft_to_solve_2d(
    src::AbstractArray{T,2},
    dist::Distributor,
    L::PencilLinearOperator,
    cache::Union{Nothing,Dict}
) where T
    nkx_global = size(src, 1)
    nz_local = size(src, 2)
    comm = dist.comm
    nprocs = dist.size

    # Get cached partition info and buffers (or create on first call)
    if cache !== nothing
        info = _get_transpose_info!(cache, dist, L, nkx_global, nz_local, T)
    else
        kx_counts, kx_offs = _block_counts(nkx_global, nprocs)
        z_counts = MPI.Allgather(nz_local, comm)
        z_offs = [sum(z_counts[1:r]) for r in 0:nprocs]
        nkx_local = kx_counts[dist.rank + 1]
        fwd_s = [kx_counts[j+1] * nz_local for j in 0:nprocs-1]
        fwd_r = [nkx_local * z_counts[j+1] for j in 0:nprocs-1]
        max_buf = max(sum(fwd_s), sum(fwd_r))
        info = (nkx_global=nkx_global, nz_local=nz_local, nz_global=L.Nz,
                nkx_local=nkx_local, kx_counts=kx_counts, kx_offs=kx_offs,
                z_counts=z_counts, z_offs=z_offs,
                fwd_s_counts=fwd_s, fwd_r_counts=fwd_r,
                send_buf=Vector{T}(undef, max_buf), recv_buf=Vector{T}(undef, max_buf))
    end

    (; kx_counts, kx_offs, z_counts, z_offs, nkx_local, nz_global,
       fwd_s_counts, fwd_r_counts, send_buf, recv_buf) = info

    # Pack: for dest rank j, send rows for their kx range
    pos = 0
    for j in 0:nprocs-1
        kx_start = kx_offs[j+1] + 1
        nkx_j = kx_counts[j+1]
        for iz in 1:nz_local
            for ikx in 1:nkx_j
                pos += 1
                send_buf[pos] = src[kx_start + ikx - 1, iz]
            end
        end
    end

    MPI.Alltoallv!(MPI.VBuffer(send_buf, fwd_s_counts), MPI.VBuffer(recv_buf, fwd_r_counts), comm)

    # Unpack into freshly allocated (nkx_local, nz_global) — caller keeps this
    dst = Array{T}(undef, nkx_local, nz_global)
    pos = 0
    for j in 0:nprocs-1
        z_start = z_offs[j+1] + 1
        nz_j = z_counts[j+1]
        for iz in 1:nz_j
            for ikx in 1:nkx_local
                pos += 1
                dst[ikx, z_start + iz - 1] = recv_buf[pos]
            end
        end
    end
    return dst
end

"""2D solve-layout → FFT-output transpose via MPI.Alltoallv."""
function _transpose_solve_to_fft_2d!(
    dst::AbstractArray{T,2},
    src::AbstractArray{T,2},
    dist::Distributor,
    L::PencilLinearOperator,
    cache::Union{Nothing,Dict}
) where T
    nkx_global = size(dst, 1)
    nz_local = size(dst, 2)
    nkx_local = size(src, 1)
    comm = dist.comm
    nprocs = dist.size

    if cache !== nothing
        info = _get_transpose_info!(cache, dist, L, nkx_global, nz_local, T)
    else
        kx_counts, kx_offs = _block_counts(nkx_global, nprocs)
        z_counts = MPI.Allgather(nz_local, comm)
        z_offs = [sum(z_counts[1:r]) for r in 0:nprocs]
        fwd_s = [kx_counts[j+1] * nz_local for j in 0:nprocs-1]
        fwd_r = [nkx_local * z_counts[j+1] for j in 0:nprocs-1]
        max_buf = max(sum(fwd_s), sum(fwd_r))
        info = (nkx_global=nkx_global, nz_local=nz_local, nz_global=L.Nz,
                nkx_local=nkx_local, kx_counts=kx_counts, kx_offs=kx_offs,
                z_counts=z_counts, z_offs=z_offs,
                fwd_s_counts=fwd_s, fwd_r_counts=fwd_r,
                send_buf=Vector{T}(undef, max_buf), recv_buf=Vector{T}(undef, max_buf))
    end

    (; kx_counts, kx_offs, z_counts, z_offs, nkx_local,
       fwd_s_counts, fwd_r_counts, send_buf, recv_buf) = info

    # Reverse direction: s/r counts are swapped relative to forward
    rev_s_counts = fwd_r_counts
    rev_r_counts = fwd_s_counts

    # Pack: for dest rank j, send our kx rows for their z range
    pos = 0
    for j in 0:nprocs-1
        z_start = z_offs[j+1] + 1
        nz_j = z_counts[j+1]
        for iz in 1:nz_j
            for ikx in 1:nkx_local
                pos += 1
                send_buf[pos] = src[ikx, z_start + iz - 1]
            end
        end
    end

    MPI.Alltoallv!(MPI.VBuffer(send_buf, rev_s_counts), MPI.VBuffer(recv_buf, rev_r_counts), comm)

    # Unpack into (nkx_global, nz_local)
    pos = 0
    for j in 0:nprocs-1
        kx_start = kx_offs[j+1] + 1
        nkx_j = kx_counts[j+1]
        for iz in 1:nz_local
            for ikx in 1:nkx_j
                pos += 1
                dst[kx_start + ikx - 1, iz] = recv_buf[pos]
            end
        end
    end
end

"""
Get or create cached transpose info for 3D mixed domains.
Uses a column sub-communicator (ranks sharing same kx block) to swap ky↔z.

PencilFFTs output for 3D (RFFT, FFT, NoTransform) with input decomp_dims=(1,2):
  output pencil: decomp_dims=(1,3), dim 2 (ky) local
  → dim 1 (kx) distributed along mesh dim 1, dim 3 (z) along mesh dim 2

Solve layout needs: decomp_dims=(1,2), dim 3 (z) local
  → swap ky (local) ↔ z (distributed) within column sub-communicator
"""
function _get_transpose_info_3d!(cache_dict::Dict, dist::Distributor, L::PencilLinearOperator,
                                  nkx_local::Int, nky_global::Int, nz_local::Int, ::Type{T}) where T
    key = :solve_transpose_info_3d
    if haskey(cache_dict, key)
        info = cache_dict[key]
        if info.nkx_local == nkx_local && info.nky_global == nky_global &&
           info.nz_local == nz_local && eltype(info.send_buf) === T
            return info
        end
    end

    coord1 = get_process_coordinate(dist, 1)
    coord2 = get_process_coordinate(dist, 2)
    col_comm = MPI.Comm_split(dist.comm, coord1, coord2)
    col_nprocs = MPI.Comm_size(col_comm)
    col_rank = MPI.Comm_rank(col_comm)

    ky_counts, ky_offs = _block_counts(nky_global, col_nprocs)
    nky_local = ky_counts[col_rank + 1]
    z_counts = MPI.Allgather(nz_local, col_comm)
    z_offs = [sum(z_counts[1:r]) for r in 0:col_nprocs]

    fwd_s = [nkx_local * ky_counts[j+1] * nz_local for j in 0:col_nprocs-1]
    fwd_r = [nkx_local * nky_local * z_counts[j+1] for j in 0:col_nprocs-1]
    max_buf = max(sum(fwd_s), sum(fwd_r))

    info = (
        nkx_local=nkx_local, nky_global=nky_global, nky_local=nky_local,
        nz_local=nz_local, nz_global=L.Nz,
        col_comm=col_comm, col_nprocs=col_nprocs, col_rank=col_rank,
        ky_counts=ky_counts, ky_offs=ky_offs,
        z_counts=z_counts, z_offs=z_offs,
        fwd_s_counts=fwd_s, fwd_r_counts=fwd_r,
        send_buf=Vector{T}(undef, max_buf),
        recv_buf=Vector{T}(undef, max_buf),
    )
    cache_dict[key] = info
    return info
end

"""3D FFT-output → solve-layout transpose via MPI.Alltoallv on column sub-communicator."""
function _transpose_fft_to_solve_3d(
    src::AbstractArray{T,3},  # (Nkx_local, Nky, Nz_local) from PencilFFT output
    dist::Distributor,
    L::PencilLinearOperator,
    cache::Union{Nothing,Dict}
) where T
    nkx_local = size(src, 1)
    nky_global = size(src, 2)
    nz_local = size(src, 3)

    if cache !== nothing
        info = _get_transpose_info_3d!(cache, dist, L, nkx_local, nky_global, nz_local, T)
    else
        coord1 = get_process_coordinate(dist, 1)
        coord2 = get_process_coordinate(dist, 2)
        col_comm = MPI.Comm_split(dist.comm, coord1, coord2)
        col_nprocs = MPI.Comm_size(col_comm)
        col_rank = MPI.Comm_rank(col_comm)
        ky_counts, ky_offs = _block_counts(nky_global, col_nprocs)
        nky_local = ky_counts[col_rank + 1]
        z_counts = MPI.Allgather(nz_local, col_comm)
        z_offs = [sum(z_counts[1:r]) for r in 0:col_nprocs]
        fwd_s = [nkx_local * ky_counts[j+1] * nz_local for j in 0:col_nprocs-1]
        fwd_r = [nkx_local * nky_local * z_counts[j+1] for j in 0:col_nprocs-1]
        max_buf = max(sum(fwd_s), sum(fwd_r))
        info = (nkx_local=nkx_local, nky_global=nky_global, nky_local=nky_local,
                nz_local=nz_local, nz_global=L.Nz,
                col_comm=col_comm, col_nprocs=col_nprocs, col_rank=col_rank,
                ky_counts=ky_counts, ky_offs=ky_offs,
                z_counts=z_counts, z_offs=z_offs,
                fwd_s_counts=fwd_s, fwd_r_counts=fwd_r,
                send_buf=Vector{T}(undef, max_buf), recv_buf=Vector{T}(undef, max_buf))
    end

    (; nky_local, nz_global, col_comm, col_nprocs,
       ky_counts, ky_offs, z_counts, z_offs,
       fwd_s_counts, fwd_r_counts, send_buf, recv_buf) = info

    # Pack: for dest rank j in col_comm, send their ky slice for all our kx and z
    pos = 0
    for j in 0:col_nprocs-1
        ky_start = ky_offs[j+1] + 1
        nky_j = ky_counts[j+1]
        for iz in 1:nz_local
            for iky in 1:nky_j
                for ikx in 1:nkx_local
                    pos += 1
                    send_buf[pos] = src[ikx, ky_start + iky - 1, iz]
                end
            end
        end
    end

    MPI.Alltoallv!(MPI.VBuffer(send_buf, fwd_s_counts), MPI.VBuffer(recv_buf, fwd_r_counts), col_comm)

    # Unpack into (nkx_local, nky_local, nz_global) — Chebyshev fully local
    dst = Array{T}(undef, nkx_local, nky_local, nz_global)
    pos = 0
    for j in 0:col_nprocs-1
        z_start = z_offs[j+1] + 1
        nz_j = z_counts[j+1]
        for iz in 1:nz_j
            for iky in 1:nky_local
                for ikx in 1:nkx_local
                    pos += 1
                    dst[ikx, iky, z_start + iz - 1] = recv_buf[pos]
                end
            end
        end
    end
    return dst
end

"""3D solve-layout → FFT-output transpose via MPI.Alltoallv on column sub-communicator."""
function _transpose_solve_to_fft_3d!(
    dst::AbstractArray{T,3},  # (Nkx_local, Nky, Nz_local) FFT output layout
    src::AbstractArray{T,3},  # (Nkx_local, Nky_local, Nz) solve layout
    dist::Distributor,
    L::PencilLinearOperator,
    cache::Union{Nothing,Dict}
) where T
    nkx_local = size(dst, 1)
    nky_global = size(dst, 2)
    nz_local = size(dst, 3)
    nky_local = size(src, 2)

    if cache !== nothing
        info = _get_transpose_info_3d!(cache, dist, L, nkx_local, nky_global, nz_local, T)
    else
        coord1 = get_process_coordinate(dist, 1)
        coord2 = get_process_coordinate(dist, 2)
        col_comm = MPI.Comm_split(dist.comm, coord1, coord2)
        col_nprocs = MPI.Comm_size(col_comm)
        col_rank = MPI.Comm_rank(col_comm)
        ky_counts, ky_offs = _block_counts(nky_global, col_nprocs)
        z_counts = MPI.Allgather(nz_local, col_comm)
        z_offs = [sum(z_counts[1:r]) for r in 0:col_nprocs]
        fwd_s = [nkx_local * ky_counts[j+1] * nz_local for j in 0:col_nprocs-1]
        fwd_r = [nkx_local * nky_local * z_counts[j+1] for j in 0:col_nprocs-1]
        max_buf = max(sum(fwd_s), sum(fwd_r))
        info = (nkx_local=nkx_local, nky_global=nky_global, nky_local=nky_local,
                nz_local=nz_local, nz_global=L.Nz,
                col_comm=col_comm, col_nprocs=col_nprocs, col_rank=col_rank,
                ky_counts=ky_counts, ky_offs=ky_offs,
                z_counts=z_counts, z_offs=z_offs,
                fwd_s_counts=fwd_s, fwd_r_counts=fwd_r,
                send_buf=Vector{T}(undef, max_buf), recv_buf=Vector{T}(undef, max_buf))
    end

    (; nz_global, col_comm, col_nprocs,
       ky_counts, ky_offs, z_counts, z_offs,
       fwd_s_counts, fwd_r_counts, send_buf, recv_buf) = info

    # Reverse: s/r counts swapped
    rev_s_counts = fwd_r_counts
    rev_r_counts = fwd_s_counts

    # Pack: for dest rank j, send their z range for all our kx and ky
    pos = 0
    for j in 0:col_nprocs-1
        z_start = z_offs[j+1] + 1
        nz_j = z_counts[j+1]
        for iz in 1:nz_j
            for iky in 1:nky_local
                for ikx in 1:nkx_local
                    pos += 1
                    send_buf[pos] = src[ikx, iky, z_start + iz - 1]
                end
            end
        end
    end

    MPI.Alltoallv!(MPI.VBuffer(send_buf, rev_s_counts), MPI.VBuffer(recv_buf, rev_r_counts), col_comm)

    # Unpack into (nkx_local, nky_global, nz_local)
    pos = 0
    for j in 0:col_nprocs-1
        ky_start = ky_offs[j+1] + 1
        nky_j = ky_counts[j+1]
        for iz in 1:nz_local
            for iky in 1:nky_j
                for ikx in 1:nkx_local
                    pos += 1
                    dst[ikx, ky_start + iky - 1, iz] = recv_buf[pos]
                end
            end
        end
    end
end

"""
    PencilLinearOperator(dist::Distributor, bases::Tuple, operator_type::Symbol; kwargs...)

Create a pencil linear operator for Chebyshev-Fourier IMEX methods.

# Arguments
- `dist`: Distributor (determines local data distribution)
- `bases`: Tuple of basis objects (must have exactly one Chebyshev and rest Fourier)
- `operator_type`: Type of operator
  - `:laplacian` - `-ν∇²` with coefficient `ν`
  - `:hyperviscosity` - `-ν∇^(2p)` with coefficient `ν` and `order=p`

# Keyword Arguments
- `ν::Real=1.0`: Viscosity/diffusion coefficient
- `order::Int=1`: Power of Laplacian (1 for ∇², 2 for ∇⁴)
- `dtype::Type=Float64`: Element type

# Example
```julia
# 2D channel: Fourier in x, Chebyshev in z
bases = (fourier_x, chebyshev_z)
L = PencilLinearOperator(dist, bases, :laplacian; ν=1e-3)

# 3D channel: Fourier in x,y, Chebyshev in z
bases = (fourier_x, fourier_y, chebyshev_z)
L = PencilLinearOperator(dist, bases, :laplacian; ν=1e-3)
```
"""
function PencilLinearOperator(
    dist::Distributor,
    bases::Tuple,
    operator_type::Symbol;
    ν::Real=1.0,
    order::Int=1,
    dtype::Type{T}=Float64
) where {T<:AbstractFloat}

    # Identify Chebyshev and Fourier bases
    chebyshev_idx = 0
    fourier_indices = Int[]

    for (i, basis) in enumerate(bases)
        if isa(basis, JacobiBasis)
            if chebyshev_idx != 0
                error("PencilLinearOperator: only one Chebyshev basis supported")
            end
            chebyshev_idx = i
        elseif isa(basis, FourierBasis)
            push!(fourier_indices, i)
        else
            error("PencilLinearOperator: unsupported basis type $(typeof(basis))")
        end
    end

    if chebyshev_idx == 0
        error("PencilLinearOperator: requires exactly one Chebyshev basis")
    end

    if isempty(fourier_indices)
        error("PencilLinearOperator: requires at least one Fourier basis")
    end

    # Get Chebyshev size and D² matrix
    cheb_basis = bases[chebyshev_idx]
    Nz = cheb_basis.meta.size
    D2_cheb = T.(differentiation_matrix(cheb_basis, 2))

    # Get Fourier wavenumbers and local ranges
    # Use get_local_range to determine which global wavenumber indices this rank owns.
    # In serial (dist.size == 1), this returns the full range; in MPI, the local slice.
    fourier_bases = [bases[i] for i in fourier_indices]

    # Compute k² values for each Fourier mode combination
    if length(fourier_bases) == 1
        # 2D: single Fourier direction
        k_vals_global = T.(wavenumbers(fourier_bases[1]))
        kx_start, kx_end = _solve_layout_range(dist, length(k_vals_global), 1)
        local_kx_range = kx_start:kx_end
        k_vals = k_vals_global[local_kx_range]
        Nk_local = length(k_vals)
        k2_values = zeros(T, Nk_local, 1)
        for i in 1:Nk_local
            k2_values[i, 1] = k_vals[i]^2
        end
        local_ky_range = 1:1
    elseif length(fourier_bases) == 2
        # 3D: two Fourier directions
        kx_vals_global = T.(wavenumbers(fourier_bases[1]))
        ky_vals_global = T.(wavenumbers(fourier_bases[2]))
        kx_start, kx_end = _solve_layout_range(dist, length(kx_vals_global), 1)
        ky_start, ky_end = _solve_layout_range(dist, length(ky_vals_global), 2)
        local_kx_range = kx_start:kx_end
        local_ky_range = ky_start:ky_end
        kx_vals = kx_vals_global[local_kx_range]
        ky_vals = ky_vals_global[local_ky_range]
        Nkx_local = length(kx_vals)
        Nky_local = length(ky_vals)
        k2_values = zeros(T, Nkx_local, Nky_local)
        for j in 1:Nky_local
            for i in 1:Nkx_local
                k2_values[i, j] = kx_vals[i]^2 + ky_vals[j]^2
            end
        end
    else
        error("PencilLinearOperator: maximum 2 Fourier dimensions supported")
    end

    params = Dict{Symbol, Any}(:ν => ν, :order => order)

    PencilLinearOperator{T}(
        Nz, local_kx_range, local_ky_range, k2_values, D2_cheb,
        operator_type, params, chebyshev_idx, fourier_indices,
        Vector{Complex{T}}(undef, Nz),
        Vector{Complex{T}}(undef, Nz)
    )
end

"""
    PencilLinearOperator(field::ScalarField, operator_type::Symbol; kwargs...)

Convenience constructor from a ScalarField.
"""
function PencilLinearOperator(
    field::ScalarField,
    operator_type::Symbol;
    kwargs...
)
    PencilLinearOperator(field.dist, field.bases, operator_type; kwargs...)
end

"""
    build_pencil_lhs_matrix(L::PencilLinearOperator, k2::Real, dt::Real, γ::Real)

Build the LHS matrix (I + dt*γ*L_k) for a single Fourier mode with wavenumber k².

For Laplacian: L_k = ν*k²*I - ν*D²_z
So: LHS = I + dt*γ*ν*k²*I - dt*γ*ν*D²_z
        = (1 + dt*γ*ν*k²)*I - dt*γ*ν*D²_z

Returns a sparse matrix that can be factorized for the implicit solve.
"""
function build_pencil_lhs_matrix(
    L::PencilLinearOperator{T},
    k2::Real,
    dt::Real,
    γ::Real
) where T

    ν = L.parameters[:ν]
    order = L.parameters[:order]
    Nz = L.Nz

    if L.operator_type == :laplacian
        # L = ν*(kx² + ky²)*I - ν*D²_z
        # For Laplacian: ∇² = -(kx² + ky²) + D²_z in mixed spectral space
        # So -ν∇² = ν*(kx² + ky²) - ν*D²_z
        diag_term = 1 + dt * γ * ν * k2
        off_diag = -dt * γ * ν
        LHS = diag_term * sparse(I, Nz, Nz) + off_diag * L.chebyshev_D2

    elseif L.operator_type == :hyperviscosity
        # Hyperviscosity: -ν∇⁴ where ∇⁴ = (∂²_x + ∂²_z)² = ∂⁴_x + 2∂²_x∂²_z + ∂⁴_z
        # In mixed spectral space: ∇⁴ → k⁴ - 2k²·D²_z + D⁴_z
        # So -ν∇⁴ = -ν·k⁴ + 2ν·k²·D²_z - ν·D⁴_z
        # LHS = I + dt·γ·(-ν∇⁴) = I - dt·γ·ν·k⁴ + 2·dt·γ·ν·k²·D²_z - dt·γ·ν·D⁴_z
        # Wait: the operator convention here is L = -(-ν∇⁴) = ν∇⁴, treated implicitly.
        # L_k = ν·k⁴ - 2ν·k²·D²_z + ν·D⁴_z
        # LHS = I + dt·γ·L_k
        if order == 2
            D4 = L.chebyshev_D2 * L.chebyshev_D2
            c = dt * γ * ν
            LHS = (1 + c * k2^2) * sparse(I, Nz, Nz) - 2 * c * k2 * L.chebyshev_D2 + c * D4
        else
            error("Hyperviscosity order $order not supported in pencil operator (only order=2)")
        end
    else
        error("build_pencil_lhs_matrix: unsupported operator type :$(L.operator_type)")
    end

    return LHS
end

"""
    PencilLHSCache{T}

Cache for factorized LHS matrices at different (dt, γ) combinations.
"""
mutable struct PencilLHSCache{T}
    L::PencilLinearOperator{T}
    dt::Float64
    γ::Float64
    factors::Matrix{Union{Nothing, Factorization}}  # Factorized LHS for each (kx, ky) mode

    function PencilLHSCache{T}(L::PencilLinearOperator{T}) where T
        nkx = length(L.local_kx_range)
        nky = length(L.local_ky_range)
        new{T}(L, 0.0, 0.0, Matrix{Union{Nothing, Factorization}}(nothing, nkx, nky))
    end
end

"""
    get_pencil_lhs_factor!(cache::PencilLHSCache, ikx::Int, iky::Int, dt::Real, γ::Real)

Get or compute the factorized LHS matrix for mode (ikx, iky).
"""
function get_pencil_lhs_factor!(
    cache::PencilLHSCache{T},
    ikx::Int, iky::Int,
    dt::Real, γ::Real
) where T

    # Check if cache is valid for this (dt, γ)
    if cache.dt != dt || cache.γ != γ
        # Invalidate cache
        cache.dt = dt
        cache.γ = γ
        fill!(cache.factors, nothing)
    end

    # Get or compute factorization
    if cache.factors[ikx, iky] === nothing
        k2 = cache.L.k2_values[ikx, iky]
        LHS = build_pencil_lhs_matrix(cache.L, k2, dt, γ)
        cache.factors[ikx, iky] = lu(LHS)
    end

    return cache.factors[ikx, iky]::Factorization
end

"""
    pencil_implicit_solve!(u_new::ScalarField, u_old::ScalarField,
                           L::PencilLinearOperator, dt::Real, γ::Real;
                           cache::Union{Nothing, PencilLHSCache}=nothing)

Apply implicit step using pencil-based solve for Chebyshev-Fourier field.

For each Fourier mode (kx, ky):
    û_new(kx, ky, :) = (I + dt*γ*L_k)⁻¹ * û_old(kx, ky, :)

This enables implicit treatment with MPI-distributed data because each rank
solves only its local Fourier modes independently.
"""
function pencil_implicit_solve!(
    u_new::ScalarField,
    u_old::ScalarField,
    L::PencilLinearOperator{T},
    dt::Real,
    γ::Real;
    cache::Union{Nothing, PencilLHSCache{T}}=nothing
) where T

    ensure_layout!(u_old, :c)
    ensure_layout!(u_new, :c)

    # Get coefficient data
    data_old = get_coeff_data(u_old)
    data_new = get_coeff_data(u_new)

    # Determine data layout based on basis ordering
    # Assume data is stored as (Fourier_x, Fourier_y, Chebyshev_z) or (Fourier_x, Chebyshev_z)
    ndims = length(L.fourier_basis_indices)
    Nz = L.Nz

    # Pre-allocate a reusable buffer to avoid per-pencil heap allocation
    rhs_buf = Vector{T}(undef, Nz)

    if ndims == 1
        # 2D case: (Nkx, Nz)
        nkx = size(data_old, 1)
        @assert size(data_old, 2) == Nz "Data shape mismatch: expected Nz=$Nz, got $(size(data_old, 2))"

        for ikx in 1:nkx
            if ikx <= length(L.local_kx_range)
                k2 = L.k2_values[ikx, 1]

                pencil_old = @view data_old[ikx, :]
                pencil_new = @view data_new[ikx, :]
                copyto!(rhs_buf, pencil_old)

                if cache !== nothing
                    factor = get_pencil_lhs_factor!(cache, ikx, 1, dt, γ)
                    pencil_new .= factor \ rhs_buf
                else
                    LHS = build_pencil_lhs_matrix(L, k2, dt, γ)
                    pencil_new .= LHS \ rhs_buf
                end
            end
        end

    elseif ndims == 2
        # 3D case: (Nkx, Nky, Nz)
        nkx = size(data_old, 1)
        nky = size(data_old, 2)
        @assert size(data_old, 3) == Nz "Data shape mismatch: expected Nz=$Nz, got $(size(data_old, 3))"

        for iky in 1:nky
            for ikx in 1:nkx
                if ikx <= length(L.local_kx_range) && iky <= length(L.local_ky_range)
                    k2 = L.k2_values[ikx, iky]

                    pencil_old = @view data_old[ikx, iky, :]
                    pencil_new = @view data_new[ikx, iky, :]
                    copyto!(rhs_buf, pencil_old)

                    if cache !== nothing
                        factor = get_pencil_lhs_factor!(cache, ikx, iky, dt, γ)
                        pencil_new .= factor \ rhs_buf
                    else
                        LHS = build_pencil_lhs_matrix(L, k2, dt, γ)
                        pencil_new .= LHS \ rhs_buf
                    end
                end
            end
        end
    else
        error("pencil_implicit_solve!: unsupported dimensionality $ndims")
    end

    return u_new
end

"""
    pencil_implicit_solve_inplace!(u::ScalarField, L::PencilLinearOperator, dt::Real, γ::Real;
                                    cache::Union{Nothing, PencilLHSCache}=nothing)

In-place implicit step using pencil-based solve.
"""
function pencil_implicit_solve_inplace!(
    u::ScalarField,
    L::PencilLinearOperator{T},
    dt::Real,
    γ::Real;
    cache::Union{Nothing, PencilLHSCache{T}}=nothing
) where T

    ensure_layout!(u, :c)
    data = get_coeff_data(u)
    ndims = length(L.fourier_basis_indices)
    Nz = L.Nz

    # Pre-allocate a reusable buffer to avoid per-pencil heap allocation
    rhs_buf = Vector{T}(undef, Nz)

    if ndims == 1
        nkx = size(data, 1)
        for ikx in 1:nkx
            if ikx <= length(L.local_kx_range)
                k2 = L.k2_values[ikx, 1]
                pencil = @view data[ikx, :]
                copyto!(rhs_buf, pencil)

                if cache !== nothing
                    factor = get_pencil_lhs_factor!(cache, ikx, 1, dt, γ)
                    pencil .= factor \ rhs_buf
                else
                    LHS = build_pencil_lhs_matrix(L, k2, dt, γ)
                    pencil .= LHS \ rhs_buf
                end
            end
        end

    elseif ndims == 2
        nkx = size(data, 1)
        nky = size(data, 2)

        for iky in 1:nky
            for ikx in 1:nkx
                if ikx <= length(L.local_kx_range) && iky <= length(L.local_ky_range)
                    k2 = L.k2_values[ikx, iky]
                    pencil = @view data[ikx, iky, :]
                    copyto!(rhs_buf, pencil)

                    if cache !== nothing
                        factor = get_pencil_lhs_factor!(cache, ikx, iky, dt, γ)
                        pencil .= factor \ rhs_buf
                    else
                        LHS = build_pencil_lhs_matrix(L, k2, dt, γ)
                        pencil .= LHS \ rhs_buf
                    end
                end
            end
        end
    end

    return u
end

"""
    _get_pencil_linear_operator(solver::InitialValueSolver)

Get the pencil linear operator from the solver or problem.
Returns nothing if not configured.
"""
function _get_pencil_linear_operator(solver::InitialValueSolver)
    # Check solver's timestepper_state first
    if solver.timestepper_state !== nothing &&
       haskey(solver.timestepper_state.timestepper_data, :pencil_linear_operator)
        return solver.timestepper_state.timestepper_data[:pencil_linear_operator]
    end

    # Check problem parameters
    if haskey(solver.problem.parameters, "pencil_linear_operator")
        return solver.problem.parameters["pencil_linear_operator"]
    end

    return nothing
end

"""
    _get_pencil_lhs_cache(solver::InitialValueSolver)

Get or create the pencil LHS cache from the solver.
"""
function _get_pencil_lhs_cache(solver::InitialValueSolver)
    if solver.timestepper_state === nothing
        return nothing
    end

    data = solver.timestepper_state.timestepper_data

    if !haskey(data, :pencil_lhs_cache)
        L = _get_pencil_linear_operator(solver)
        if L !== nothing
            T = eltype(L.k2_values)
            data[:pencil_lhs_cache] = PencilLHSCache{T}(L)
        else
            return nothing
        end
    end

    return data[:pencil_lhs_cache]
end

"""
    set_pencil_linear_operator!(solver::InitialValueSolver, L::PencilLinearOperator)

Set the pencil linear operator for Chebyshev-Fourier IMEX methods.

# Example
```julia
L = PencilLinearOperator(dist, bases, :laplacian; ν=1e-3)
set_pencil_linear_operator!(solver, L)
```
"""
function set_pencil_linear_operator!(solver::InitialValueSolver, L::PencilLinearOperator)
    if solver.timestepper_state !== nothing
        solver.timestepper_state.timestepper_data[:pencil_linear_operator] = L
        # Clear any existing cache
        delete!(solver.timestepper_state.timestepper_data, :pencil_lhs_cache)
    else
        solver.problem.parameters["pencil_linear_operator"] = L
    end
    return solver
end

"""
    has_chebyshev_basis(bases::Tuple)

Check if the domain has at least one Chebyshev (Jacobi) basis.
"""
function has_chebyshev_basis(bases::Tuple)
    return any(isa(b, JacobiBasis) for b in bases)
end

"""
    is_pencil_imex_compatible(bases::Tuple)

Check if the bases are compatible with pencil-based IMEX.
Requires exactly one Chebyshev basis and at least one Fourier basis.
"""
function is_pencil_imex_compatible(bases::Tuple)
    n_cheb = count(isa(b, JacobiBasis) for b in bases)
    n_fourier = count(isa(b, FourierBasis) for b in bases)
    return n_cheb == 1 && n_fourier >= 1
end
