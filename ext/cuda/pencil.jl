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
struct PencilDecomposition
    # Global domain shape
    global_shape::NTuple{3, Int}

    # Process grid (P1 x P2)
    proc_grid::NTuple{2, Int}

    # This rank's position in grid
    rank::Int
    grid_coords::NTuple{2, Int}  # (row, col) in process grid

    # MPI communicators
    world_comm::MPI.Comm
    row_comm::MPI.Comm   # Ranks in same row (for Y<->Z transpose)
    col_comm::MPI.Comm   # Ranks in same column (for X<->Y transpose)

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
    PencilDecomposition(global_shape, proc_grid, rank, comm)

Create a pencil decomposition for the given domain and process grid.

# Arguments
- `global_shape::NTuple{3, Int}`: Global domain dimensions (Nx, Ny, Nz)
- `proc_grid::NTuple{2, Int}`: Process grid dimensions (P1, P2) where P1*P2 = nprocs
- `rank::Int`: This process's MPI rank
- `comm::MPI.Comm`: MPI world communicator

# Returns
A PencilDecomposition struct with pre-computed local shapes for all orientations
and MPI sub-communicators for row and column communication.
"""
function PencilDecomposition(global_shape::NTuple{3, Int},
                              proc_grid::NTuple{2, Int},
                              rank::Int,
                              comm::MPI.Comm)
    P1, P2 = proc_grid
    grid_coords = rank_to_grid(rank, proc_grid)
    row, col = grid_coords

    # Create row and column sub-communicators
    # Row comm: all ranks with same row coordinate (for Y<->Z transpose)
    # Col comm: all ranks with same col coordinate (for X<->Y transpose)
    row_comm = MPI.Comm_split(comm, row, col)
    col_comm = MPI.Comm_split(comm, col, row)

    # Compute local shapes
    x_shape, y_shape, z_shape = compute_pencil_shapes(global_shape, proc_grid, grid_coords)

    return PencilDecomposition(
        global_shape,
        proc_grid,
        rank,
        grid_coords,
        comm,
        row_comm,
        col_comm,
        x_shape,
        y_shape,
        z_shape,
        Ref(:z_pencil)  # Start in Z-pencil orientation
    )
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
    @assert orient in (:x_pencil, :y_pencil, :z_pencil) "Invalid orientation: $orient"
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
