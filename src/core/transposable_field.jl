"""
    TransposableField - Efficient GPU+MPI distributed spectral transforms

This module implements the TransposableField pattern inspired by Oceananigans.jl
to enable efficient 2D pencil decomposition for distributed GPU computing.

## Design

TransposableField wraps a ScalarField and provides multi-layout transpose
capabilities for distributed FFTs. Each layout corresponds to a different
dimension being "local" (not decomposed):

- XLocal: x-dimension is local, y and z are distributed
- YLocal: y-dimension is local, x and z are distributed
- ZLocal: z-dimension is local, x and y are distributed

## 2D Decomposition (Oceananigans-style)

For a domain of size (Nx, Ny, Nz) with topology Rx × Ry:
- Each rank is identified by (rx, ry) where 0 ≤ rx < Rx and 0 ≤ ry < Ry
- Pencil transposes happen along rows (Ry processes) or columns (Rx processes)

## Async Communication Overlap

The module supports overlapping communication with computation:
1. Start async transpose (non-blocking MPI)
2. Compute interior FFT (data not involved in transpose)
3. Wait for transpose completion
4. Compute boundary FFT (data just received)

## Algorithm

Forward transform (physical → spectral):
1. Start in ZLocal layout [Nx/Rx, Ny/Ry, Nz]
2. FFT in z (local)
3. Transpose Z→Y: Pack → MPI.Ialltoallv! → Unpack (async possible)
4. Now in YLocal layout [Nx/Rx, Ny, Nz/Ry]
5. FFT in y (local)
6. Transpose Y→X: Pack → MPI.Ialltoallv! → Unpack (async possible)
7. Now in XLocal layout [Nx, Ny/Rx, Nz/Ry]
8. FFT in x (local)

Backward transform reverses this process.

## References
- Oceananigans.jl distributed GPU implementation
- 2DECOMP&FFT library concepts

## File Organization

This module is split into multiple files for maintainability:
- transpose_types.jl: Core type definitions
- transpose_shapes.jl: Local shape computation
- transpose_buffers.jl: Buffer allocation
- transpose_counts.jl: MPI count computation
- transpose_pack_unpack.jl: Data packing operations
- transpose_mpi.jl: MPI communication helpers
- transpose_sync.jl: Synchronous transpose operations
- transpose_async.jl: Asynchronous transpose operations
- transpose_transforms.jl: Distributed FFT operations
- transpose_grouped.jl: Grouped transpose operations
"""

using MPI
using FFTW

# Include all the split files
include("transpose/transpose_types.jl")
include("transpose/transpose_shapes.jl")
include("transpose/transpose_buffers.jl")
include("transpose/transpose_counts.jl")
include("transpose/transpose_pack_unpack.jl")
include("transpose/transpose_mpi.jl")
include("transpose/transpose_sync.jl")
include("transpose/transpose_async.jl")
include("transpose/transpose_transforms.jl")
include("transpose/transpose_grouped.jl")

# ============================================================================
# TransposableFieldStorage (deferred from field.jl due to type dependencies)
# ============================================================================

"""
    TransposableFieldStorage{CT, N} <: AbstractFieldStorage

Storage for distributed GPU+MPI fields with 2D pencil decomposition.
Absorbs the functionality previously in TransposableField wrapper.

CT is the complex element type (Complex{T}), N is the number of dimensions.
"""
mutable struct TransposableFieldStorage{CT, N} <: AbstractFieldStorage
    base::SerialFieldStorage
    transpose_buffers::TransposeBuffers{CT, N}
    counts::TransposeCounts
    comms::TransposeComms
    topology::Topology2D
    global_shape::NTuple{N, Int}
    local_shapes::Dict{TransposeLayout, NTuple{N, Int}}
    async_state::AsyncTransposeState
    fft_plans::Dict{TransposeLayout, Any}
    total_transpose_time::Float64
    total_fft_time::Float64
end

# Deferred storage_mode dispatch (TransposableFieldStorage is now defined)
storage_mode(::ScalarField{T, <:TransposableFieldStorage}) where T = TransposableStorage()

# ============================================================================
# Constructor
# ============================================================================

"""
    TransposableField(field::ScalarField; topology=nothing)

Create a TransposableField from an existing ScalarField.
Sets up buffers, counts, and communicators for distributed transpose operations.

# Arguments
- `field`: The ScalarField to wrap
- `topology`: Optional (Rx, Ry) tuple for 2D topology. If not provided, uses distributor mesh.
"""
function TransposableField(field::ScalarField; topology=nothing)
    dist = field.dist
    arch = dist.architecture

    # Validate: TransposableField only supports ComplexFourier for MPI
    # (RealFourier's half-spectrum layout is incompatible with custom transposes)
    if dist.size > 1 && field.domain !== nothing
        bases = field.domain.bases
        validate_mpi_fourier_only(bases, dist.size; use_pencil_arrays=false)
    end

    # Get global shape from field's domain
    gshape = if field.domain !== nothing
        global_shape(field.domain)
    else
        size(field["g"])
    end

    N = length(gshape)

    # Validate: 1D domains with MPI don't benefit from TransposableField
    # TransposableField is designed for multi-dimensional distributed transposes
    # For 1D, either the single dimension is local (no distribution needed) or
    # it's distributed (regular 1D FFT handles this, no transpose needed)
    if N == 1 && dist.size > 1
        error("TransposableField is not supported for 1D domains with MPI (nprocs=$(dist.size)). " *
              "1D domains have only one dimension which cannot be transposed. " *
              "Use regular Field with PencilFFTs for 1D distributed FFTs, or use a single process.")
    end
    # Spectral transforms use complex; if dtype is already complex, use it directly
    T = dist.dtype <: Complex ? dist.dtype : Complex{dist.dtype}

    # Create 2D topology
    topo = if topology !== nothing
        Rx, Ry = topology
        create_topology_2d(dist.comm, Rx, Ry)
    elseif dist.mesh !== nothing && length(dist.mesh) >= 2
        create_topology_2d(dist.comm, dist.mesh[1], dist.mesh[2])
    elseif dist.size > 1
        # Auto-determine topology
        Rx, Ry = auto_topology(dist.size, N)
        create_topology_2d(dist.comm, Rx, Ry)
    else
        Topology2D()
    end

    # Create buffers
    buffers = TransposeBuffers{T,N}(arch)

    # Create counts with correct sizes for each transpose operation
    # For 3D: Z↔Y uses row_comm, Y↔X uses col_comm
    # For 2D with 1D decomposition: Z↔Y uses whichever comm has multiple processes
    if N >= 3
        zy_nprocs = max(topo.row_size, 1)  # row_comm size for Z↔Y transpose
        yx_nprocs = max(topo.col_size, 1)  # col_comm size for Y↔X transpose
    else
        # 2D case: Z↔Y uses row_comm for true 2D mesh, but col_comm for 1D decomposition
        if topo.Rx > 1 && topo.Ry > 1
            # True 2D mesh: use row_comm (Ry processes)
            zy_nprocs = max(topo.row_size, 1)
        else
            # 1D decomposition: use whichever comm has multiple processes
            zy_nprocs = topo.row_size > 1 ? topo.row_size : max(topo.col_size, 1)
        end
        yx_nprocs = max(topo.col_size, 1)  # Y↔X always uses col_comm
    end
    counts = TransposeCounts(zy_nprocs, yx_nprocs)

    # Create comms wrapper
    comms = TransposeComms(topo)

    # Compute local shapes for each layout
    local_shapes = compute_local_shapes_2d(gshape, topo)

    # Validate: field storage must match ZLocal shape for GPU+MPI
    if dist.size > 1
        field_shape = size(field["g"])
        expected_shape = local_shapes[ZLocal]
        if field_shape != expected_shape
            error("TransposableField layout mismatch: field storage shape $field_shape " *
                  "does not match expected ZLocal shape $expected_shape for topology " *
                  "(Rx=$(topo.Rx), Ry=$(topo.Ry)). " *
                  "Ensure field allocation uses ZLocal decomposition: " *
                  "x decomposed by Rx, y decomposed by Ry, z local (for 3D); " *
                  "or use serial execution (nprocs=1).")
        end

        # Warn about async transpose limitations for 2D domain with 2D true mesh
        # In this configuration, Z→Y uses Allgatherv (not Alltoallv), so async is not supported
        if N == 2 && topo.Rx > 1 && topo.Ry > 1
            @warn "TransposableField for 2D domain with 2D true mesh (Rx=$(topo.Rx), Ry=$(topo.Ry)): " *
                  "async transposes are NOT supported. Use blocking transposes " *
                  "(transpose_z_to_y!, transpose_y_to_x!) or distributed_forward_transform! " *
                  "with overlap=false. Async functions will error at runtime." maxlog=1
        end
    end

    # Async state
    async_state = AsyncTransposeState()

    # FFT plans dictionary
    fft_plans = Dict{TransposeLayout, Any}()

    tf = TransposableField{typeof(field), T, N}(
        field, buffers, counts, comms, topo, gshape, local_shapes, async_state, fft_plans,
        0.0, 0.0, 0.0, 0.0, 0
    )

    # Register finalizer to free MPI sub-communicators when TransposableField is garbage collected
    # CRITICAL: Without this, row_comm and col_comm would leak, eventually exhausting MPI resources
    if dist.size > 1 && topo.row_comm !== nothing
        finalizer(tf) do x
            free_topology_2d!(x.topology)
        end
    end

    # Allocate buffers
    allocate_transpose_buffers!(tf)

    # Compute transpose counts
    compute_transpose_counts!(tf)

    return tf
end

"""
    make_transposable(field::ScalarField; kwargs...)

Helper function to create a TransposableField from a ScalarField.
"""
make_transposable(field::ScalarField; kwargs...) = TransposableField(field; kwargs...)

# ============================================================================
# MPI Communicator Creation (Legacy wrapper)
# ============================================================================

"""
    create_transpose_comms(dist::Distributor)

Create MPI sub-communicators for transpose operations.
This is a wrapper that creates a 2D topology internally.

WARNING: This function is deprecated. Prefer using TransposableField which handles
MPI communicator cleanup automatically via finalizers. If you must use this function,
call `free_comms!(comms)` when done to avoid MPI resource leaks.
"""
function create_transpose_comms(dist::Distributor)
    if dist.size == 1
        return TransposeComms()
    end

    if dist.mesh !== nothing && length(dist.mesh) >= 2
        topo = create_topology_2d(dist.comm, dist.mesh[1], dist.mesh[2])
    else
        Rx, Ry = auto_topology(dist.size, 3)
        topo = create_topology_2d(dist.comm, Rx, Ry)
    end

    return TransposeComms(topo)
end

# ============================================================================
# Accessor Functions
# ============================================================================

"""Get the current active layout"""
active_layout(tf::TransposableField) = tf.buffers.active_layout[]

"""Get data array for current layout"""
function current_data(tf::TransposableField)
    layout = active_layout(tf)
    if layout == ZLocal
        return tf.buffers.z_local_data
    elseif layout == YLocal
        return tf.buffers.y_local_data
    else
        return tf.buffers.x_local_data
    end
end

"""Get local shape for specified layout"""
local_shape(tf::TransposableField, layout::TransposeLayout) = tf.local_shapes[layout]

"""Get global shape"""
global_shape(tf::TransposableField) = tf.global_shape

"""Get performance statistics"""
function get_transpose_stats(tf::TransposableField)
    return (
        total_transpose_time = tf.total_transpose_time,
        total_fft_time = tf.total_fft_time,
        total_pack_time = tf.total_pack_time,
        total_unpack_time = tf.total_unpack_time,
        num_transposes = tf.num_transposes,
        avg_transpose_time = tf.num_transposes > 0 ? tf.total_transpose_time / tf.num_transposes : 0.0
    )
end

"""Reset performance statistics"""
function reset_transpose_stats!(tf::TransposableField)
    tf.total_transpose_time = 0.0
    tf.total_fft_time = 0.0
    tf.total_pack_time = 0.0
    tf.total_unpack_time = 0.0
    tf.num_transposes = 0
end

# ============================================================================
# Exports
# ============================================================================

export TransposeLayout, XLocal, YLocal, ZLocal
export Topology2D, create_topology_2d, auto_topology, free_topology_2d!
export AsyncTransposeState
export TransposeBuffers, TransposeCounts, TransposeComms
export TransposableField
export make_transposable
export transpose_z_to_y!, transpose_y_to_z!, transpose_y_to_x!, transpose_x_to_y!
export async_transpose_z_to_y!, async_transpose_y_to_x!, wait_transpose!, is_transpose_complete
export distributed_forward_transform!, distributed_backward_transform!
export active_layout, current_data, local_shape
export pack_for_transpose!, unpack_from_transpose!
export compute_local_shapes, compute_local_shapes_2d, divide_evenly, local_range
export create_transpose_comms, free_comms!
export get_transpose_stats, reset_transpose_stats!
export get_active_buffers, swap_buffers!
# Basis-aware transform helpers
export transform_in_dim!, fft_in_dim!, dct_in_dim!, get_basis_for_dim

# Grouped transposes (Dedalus GROUP_TRANSPOSES equivalent)
export GroupedTransposeConfig, set_group_transposes!
export group_transpose_z_to_y!, group_transpose_y_to_z!
export group_transpose_y_to_x!, group_transpose_x_to_y!
export group_distributed_forward_transform!, group_distributed_backward_transform!
