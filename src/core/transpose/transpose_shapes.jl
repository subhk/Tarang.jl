"""
    Transpose Shapes - Local shape computation for TransposableField

This file contains functions for computing local array shapes for each
transpose layout based on the 2D process topology.
"""

# ============================================================================
# Local Shape Computation (2D Topology)
# ============================================================================

"""
    compute_local_shapes_2d(global_shape::NTuple{N,Int}, topo::Topology2D)

Compute local array shapes for each transpose layout with 2D topology.

For a 3D domain (Nx, Ny, Nz) with Rx × Ry topology:
- ZLocal: [Nx/Rx, Ny/Ry, Nz]     - z fully local (starting pencil)
- YLocal: [Nx/Rx, Ny, Nz/Ry]     - y fully local (after Z→Y transpose)
- XLocal: [Nx, Ny/Rx, Nz/Ry]     - x fully local (after Y→X transpose)
"""
function compute_local_shapes_2d(global_shape::NTuple{N,Int}, topo::Topology2D) where N
    shapes = Dict{TransposeLayout, NTuple{N,Int}}()

    Rx, Ry = topo.Rx, topo.Ry
    rx, ry = topo.rx, topo.ry

    if N == 3
        Nx, Ny, Nz = global_shape

        # ZLocal: z is local, x decomposed by Rx, y decomposed by Ry
        shapes[ZLocal] = (
            divide_evenly(Nx, Rx, rx),
            divide_evenly(Ny, Ry, ry),
            Nz
        )

        # YLocal: y is local, x decomposed by Rx, z decomposed by Ry
        shapes[YLocal] = (
            divide_evenly(Nx, Rx, rx),
            Ny,
            divide_evenly(Nz, Ry, ry)
        )

        # XLocal: x is local, y decomposed by Rx, z decomposed by Ry
        shapes[XLocal] = (
            Nx,
            divide_evenly(Ny, Rx, rx),
            divide_evenly(Nz, Ry, ry)
        )

    elseif N == 2
        Nx, Ny = global_shape

        # Check if we have true 2D mesh (both Rx > 1 and Ry > 1)
        if Rx > 1 && Ry > 1
            # True 2D mesh decomposition for 2D domain
            # row_comm has Ry processes (same rx), col_comm has Rx processes (same ry)
            #
            # Transform sequence:
            # ZLocal: (Nx/Rx, Ny/Ry) - both dims decomposed
            # Z→Y via row_comm Allgatherv: gather y portions from Ry processes
            # YLocal: (Nx/Rx, Ny) - y local, can do FFT in y
            # Y→X via col_comm Alltoallv: exchange (give away y portions, gather x portions)
            # XLocal: (Nx, Ny/Rx) - x local, can do FFT in x
            #         Note: y is now decomposed by Rx (col_comm members), not Ry
            shapes[ZLocal] = (
                divide_evenly(Nx, Rx, rx),
                divide_evenly(Ny, Ry, ry)
            )
            shapes[YLocal] = (
                divide_evenly(Nx, Rx, rx),
                Ny
            )
            shapes[XLocal] = (
                Nx,
                divide_evenly(Ny, Rx, rx)  # y decomposed by col_comm size (Rx)
            )
        else
            # 1D decomposition (either Rx > 1 OR Ry > 1, but not both)
            # CRITICAL: Must match get_local_array_size convention:
            # - mesh[1] (Rx) decomposes dimension 1 (x)
            # - mesh[2] (Ry) decomposes dimension 2 (y)

            if Ry > 1
                # Decompose y (dimension 2), keep x local
                # ZLocal: (Nx, Ny/Ry) - y distributed, x local
                # YLocal: (Nx/Ry, Ny) - after transpose, y local, x distributed
                # XLocal: same as YLocal for 2D with 1D decomp
                shapes[ZLocal] = (Nx, divide_evenly(Ny, Ry, ry))
                shapes[YLocal] = (divide_evenly(Nx, Ry, ry), Ny)
                shapes[XLocal] = (divide_evenly(Nx, Ry, ry), Ny)
            else
                # Rx > 1: Decompose x (dimension 1), keep y local
                # ZLocal: (Nx/Rx, Ny) - x distributed, y local
                # YLocal: (Nx, Ny/Rx) - after transpose, x local, y distributed
                # XLocal: same as YLocal for 2D with 1D decomp
                shapes[ZLocal] = (divide_evenly(Nx, Rx, rx), Ny)
                shapes[YLocal] = (Nx, divide_evenly(Ny, Rx, rx))
                shapes[XLocal] = (Nx, divide_evenly(Ny, Rx, rx))
            end
        end

    else
        # 1D or unsupported
        shapes[ZLocal] = global_shape
        shapes[YLocal] = global_shape
        shapes[XLocal] = global_shape
    end

    return shapes
end

# Keep old function for backward compatibility
function compute_local_shapes(global_shape::NTuple{N,Int}, dist::Distributor) where N
    if dist.size == 1
        shapes = Dict{TransposeLayout, NTuple{N,Int}}()
        shapes[ZLocal] = global_shape
        shapes[YLocal] = global_shape
        shapes[XLocal] = global_shape
        return shapes
    end

    topo = if dist.mesh !== nothing && length(dist.mesh) >= 2
        create_topology_2d(dist.comm, dist.mesh[1], dist.mesh[2])
    else
        Rx, Ry = auto_topology(dist.size, N)
        create_topology_2d(dist.comm, Rx, Ry)
    end

    # Use try/finally to ensure MPI communicators are freed even on exception
    local shapes
    try
        shapes = compute_local_shapes_2d(global_shape, topo)
    finally
        # CRITICAL: Free topology to avoid MPI communicator leak
        # create_topology_2d allocates row_comm and col_comm which must be freed
        free_topology_2d!(topo)
    end

    return shapes
end

"""
    divide_evenly(n::Int, nprocs::Int, rank::Int)

Compute local size for even division with remainder distributed to first processes.
Rank must be in range [0, nprocs-1].
"""
function divide_evenly(n::Int, nprocs::Int, rank::Int)
    if nprocs <= 0
        return n
    end
    # Validate rank is in valid range
    if rank < 0 || rank >= nprocs
        error("divide_evenly: rank=$rank is out of valid range [0, $(nprocs-1)] for nprocs=$nprocs. " *
              "This indicates incorrect MPI rank or topology configuration.")
    end
    base = div(n, nprocs)
    remainder = mod(n, nprocs)
    return base + (rank < remainder ? 1 : 0)
end

"""
    local_range(n::Int, nprocs::Int, rank::Int)

Get the global index range for a given rank.
Rank must be in range [0, nprocs-1].
"""
function local_range(n::Int, nprocs::Int, rank::Int)
    # Validate rank is in valid range
    if rank < 0 || rank >= nprocs
        error("local_range: rank=$rank is out of valid range [0, $(nprocs-1)] for nprocs=$nprocs. " *
              "This indicates incorrect MPI rank or topology configuration.")
    end
    if nprocs <= 0
        error("local_range: nprocs=$nprocs must be positive.")
    end

    base = div(n, nprocs)
    remainder = mod(n, nprocs)

    start = 1
    for r in 0:(rank-1)
        start += base + (r < remainder ? 1 : 0)
    end

    local_n = base + (rank < remainder ? 1 : 0)
    return start:(start + local_n - 1)
end
