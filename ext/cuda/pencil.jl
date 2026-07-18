# ============================================================================
# Pencil Decomposition for Distributed GPU Transforms
# ============================================================================

"""
    PencilDecomposition

2D pencil decomposition for distributed 3D transforms.

For a domain (Nx, Ny, Nz) on a P1 x P2 process grid:
- X-pencil: (Nx, Ny/P1, Nz/P2) - full X for local transform
- Y-pencil: (Nx/P1, Ny, Nz/P2) - full Y for local transform
- Z-pencil: (Nx/P1, Ny/P2, Nz) - full Z for local transform

This struct is used for multi-GPU DCT (Discrete Cosine Transform) with NCCL
communication for all-to-all transpose operations between pencil orientations.
"""
mutable struct PencilDecomposition
    # Global domain shape
    global_shape::NTuple{3, Int}

    # Process grid (P1 x P2)
    proc_grid::NTuple{2, Int}

    # This rank's position in grid
    rank::Int
    grid_coords::NTuple{2, Int}  # (row, col) in process grid

    # MPI communicators (Union with Nothing for safe cleanup)
    world_comm::MPI.Comm
    row_comm::Union{Nothing, MPI.Comm}   # Ranks in same row (for Y<->Z transpose)
    col_comm::Union{Nothing, MPI.Comm}   # Ranks in same column (for X<->Y transpose)

    # Local shapes for each pencil orientation
    x_pencil_shape::NTuple{3, Int}
    y_pencil_shape::NTuple{3, Int}
    z_pencil_shape::NTuple{3, Int}

    # Current orientation
    current_orientation::Ref{Symbol}  # :x_pencil, :y_pencil, :z_pencil
end

"""
    rank_to_grid(rank::Int, proc_grid::NTuple{2, Int})

Convert linear rank to (row, col) grid coordinates.
Row-major ordering: rank = row * P2 + col
"""
function rank_to_grid(rank::Int, proc_grid::NTuple{2, Int})
    P1, P2 = proc_grid
    row = div(rank, P2)
    col = mod(rank, P2)
    return (row, col)
end

"""
    grid_to_rank(row::Int, col::Int, proc_grid::NTuple{2, Int})

Convert (row, col) grid coordinates to linear rank.
"""
function grid_to_rank(row::Int, col::Int, proc_grid::NTuple{2, Int})
    P1, P2 = proc_grid
    return row * P2 + col
end

"""
    compute_pencil_shapes(global_shape, proc_grid, grid_coords)

Compute local shapes for each pencil orientation.

Handles uneven division when N % P != 0 by distributing the remainder
across the first (N % P) processes.
"""
function compute_pencil_shapes(global_shape::NTuple{3, Int},
                                proc_grid::NTuple{2, Int},
                                grid_coords::NTuple{2, Int})
    Nx, Ny, Nz = global_shape
    P1, P2 = proc_grid
    row, col = grid_coords

    # Compute local sizes (handle uneven division)
    # For dimension split by P, rank r gets: floor(N/P) + (r < N%P ? 1 : 0)
    local_Nx = div(Nx, P1) + (row < mod(Nx, P1) ? 1 : 0)
    local_Ny_by_P1 = div(Ny, P1) + (row < mod(Ny, P1) ? 1 : 0)
    local_Ny_by_P2 = div(Ny, P2) + (col < mod(Ny, P2) ? 1 : 0)
    local_Nz = div(Nz, P2) + (col < mod(Nz, P2) ? 1 : 0)

    # X-pencil: full X, split Y by P1, split Z by P2
    x_pencil = (Nx, local_Ny_by_P1, local_Nz)

    # Y-pencil: split X by P1, full Y, split Z by P2
    y_pencil = (local_Nx, Ny, local_Nz)

    # Z-pencil: split X by P1, split Y by P2, full Z
    z_pencil = (local_Nx, local_Ny_by_P2, Nz)

    return x_pencil, y_pencil, z_pencil
end

"""
    column_major_grid_coords(rank, proc_grid) -> (row, col)

Grid coordinates for `rank` under the **distributor's column-major** block
ownership convention, where the X-block index varies fastest:

    X-block (row) = rank % P1          # the P1 (col_comm / X<->Y) dimension
    Y-block (col) = (rank ÷ P1) % P2   # the P2 (row_comm / Y<->Z) dimension

This is the alignment a `ScalarField`'s distributed coeff/grid buffer actually
uses. It differs from the default row-major `rank_to_grid` (`row = rank ÷ P2`,
`col = rank % P2`). Use it (via the `grid_coords` keyword of the constructor) so
the pencil owns the SAME global block the field buffer holds.

NOTE (np≥4): this only matters when P1>1 AND P2>1; for a 1×P or P×1 grid the two
conventions coincide. Needs GPU validation at np≥4.
"""
column_major_grid_coords(rank::Int, proc_grid::NTuple{2, Int}) =
    (mod(rank, proc_grid[1]), div(rank, proc_grid[1]) % proc_grid[2])

"""
    PencilDecomposition(global_shape, proc_grid, rank, comm; grid_coords=nothing)

Create a pencil decomposition for the given domain and process grid.

# Arguments
- `global_shape::NTuple{3, Int}`: Global domain dimensions (Nx, Ny, Nz)
- `proc_grid::NTuple{2, Int}`: Process grid dimensions (P1, P2) where P1*P2 = nprocs
- `rank::Int`: This process's MPI rank
- `comm::MPI.Comm`: MPI world communicator

# Keyword
- `grid_coords::Union{Nothing,NTuple{2,Int}}`: explicit `(row, col)` grid
  position. When `nothing` (default) the row-major `rank_to_grid(rank, proc_grid)`
  is used (legacy behaviour). Pass `column_major_grid_coords(rank, proc_grid)` to
  align the pencil with the distributor's column-major field-buffer ownership
  (design decision #5 — needs np≥4 GPU validation). The row/col sub-communicators
  are built consistently with whichever `grid_coords` is supplied: `row_comm`
  groups equal-`row` ranks (Y<->Z, size P2), `col_comm` groups equal-`col` ranks
  (X<->Y, size P1).

# Returns
A PencilDecomposition struct with pre-computed local shapes for all orientations
and MPI sub-communicators for row and column communication.
"""
function PencilDecomposition(global_shape::NTuple{3, Int},
                              proc_grid::NTuple{2, Int},
                              rank::Int,
                              comm::MPI.Comm;
                              grid_coords::Union{Nothing, NTuple{2, Int}}=nothing)
    gc = grid_coords === nothing ? rank_to_grid(rank, proc_grid) : grid_coords
    row, col = gc

    # Create row and column sub-communicators
    # Row comm: all ranks with same row coordinate (for Y<->Z transpose)
    # Col comm: all ranks with same col coordinate (for X<->Y transpose)
    row_comm = MPI.Comm_split(comm, row, col)
    col_comm = MPI.Comm_split(comm, col, row)

    # Compute local shapes
    x_shape, y_shape, z_shape = compute_pencil_shapes(global_shape, proc_grid, gc)

    pd = PencilDecomposition(
        global_shape,
        proc_grid,
        rank,
        gc,
        comm,
        row_comm,
        col_comm,
        x_shape,
        y_shape,
        z_shape,
        Ref(:z_pencil)  # Start in Z-pencil orientation
    )
    finalizer(free_pencil_decomposition!, pd)
    return pd
end

"""
    build_coeff_pencil(main::PencilDecomposition, coeff_global_shape) -> PencilDecomposition

Build a **coeff-sized** pencil that SHARES `main`'s row/col communicators, rank,
proc grid and grid_coords, but uses a different global shape — specifically a
dim-1 length of `div(Nx,2)+1` for a RealFourier dim-1 axis (half-spectrum).

WHY this exists (the truncation-vs-transpose-back crux): the forward transform
truncates dim 1 to the half-spectrum while dim 1 is LOCAL in the X-pencil, then
transposes X→Y→Z so the coeffs land Z-local (matching the field coeff buffer).
The verified fixed-shape NCCL transposes derive their pack/unpack counts from the
pencil's `global_shape`/`*_pencil_shape`, so they MUST be driven by a pencil whose
dim-1 length is the truncated `div(Nx,2)+1`, not `Nx`. This builder produces that
pencil. In the Z-local coeff layout dim 1 is therefore decomposed by P1 with the
half-spectrum length, exactly matching the framework's coeff convention.

CRITICAL: the returned pencil aliases `main`'s `row_comm`/`col_comm` (so its NCCL
sub-communicators match the shared `NCCLTransposeBuffer`). It is built WITHOUT a
finalizer — do NOT free it / its comms independently; the owning `main` pencil's
finalizer frees them.
"""
function build_coeff_pencil(main::PencilDecomposition, coeff_global_shape::NTuple{3, Int})
    x_shape, y_shape, z_shape =
        compute_pencil_shapes(coeff_global_shape, main.proc_grid, main.grid_coords)
    # Raw (finalizer-free) construction via the default field constructor — shares
    # main's communicators; see CRITICAL note above.
    return PencilDecomposition(
        coeff_global_shape,
        main.proc_grid,
        main.rank,
        main.grid_coords,
        main.world_comm,
        main.row_comm,
        main.col_comm,
        x_shape,
        y_shape,
        z_shape,
        Ref(:z_pencil)
    )
end

"""
    free_pencil_decomposition!(pd::PencilDecomposition)

Free MPI sub-communicators to prevent communicator leaks.
Safe to call multiple times.
"""
function free_pencil_decomposition!(pd::PencilDecomposition)
    # Guard against GC running after MPI.Finalize() (e.g., during Julia shutdown)
    if !MPI.Initialized() || MPI.Finalized()
        return
    end
    if pd.row_comm !== nothing && pd.row_comm != MPI.COMM_NULL
        try
            MPI.free(pd.row_comm)
        catch e
            @warn "Failed to free PencilDecomposition row communicator: $e" maxlog=1
        end
        pd.row_comm = nothing
    end
    if pd.col_comm !== nothing && pd.col_comm != MPI.COMM_NULL
        try
            MPI.free(pd.col_comm)
        catch e
            @warn "Failed to free PencilDecomposition col communicator: $e" maxlog=1
        end
        pd.col_comm = nothing
    end
end

# ============================================================================
# Accessor Functions
# ============================================================================

"""
    current_orientation(p::PencilDecomposition)

Get the current pencil orientation (:x_pencil, :y_pencil, or :z_pencil).
"""
current_orientation(p::PencilDecomposition) = p.current_orientation[]

"""
    set_orientation!(p::PencilDecomposition, orient::Symbol)

Set the current pencil orientation.

# Arguments
- `orient::Symbol`: One of :x_pencil, :y_pencil, or :z_pencil
"""
function set_orientation!(p::PencilDecomposition, orient::Symbol)
    orient in (:x_pencil, :y_pencil, :z_pencil) || throw(ArgumentError("Invalid orientation: $orient. Must be :x_pencil, :y_pencil, or :z_pencil"))
    p.current_orientation[] = orient
end

"""
    current_local_shape(p::PencilDecomposition)

Get the local array shape for the current pencil orientation.
"""
function current_local_shape(p::PencilDecomposition)
    orient = current_orientation(p)
    if orient == :x_pencil
        return p.x_pencil_shape
    elseif orient == :y_pencil
        return p.y_pencil_shape
    else  # :z_pencil
        return p.z_pencil_shape
    end
end

# ============================================================================
# GPU Synchronization
# ============================================================================

"""
    Tarang.synchronize_device!(::GPU{CuDevice})

CUDA override of the no-op `synchronize_device!` fallback declared in
`src/core/gpu_distributed.jl`. `Tarang.architecture(::CuArray)` returns a
`GPU{CuDevice}`, so the "CRITICAL: Synchronize GPU before MPI" call sites in
`distributed_fft_cuda_aware!` dispatch here and actually block until pending
GPU work (e.g. pack kernels) has completed before handing buffers to MPI.
"""
Tarang.synchronize_device!(::GPU{CuDevice}) = CUDA.synchronize()
