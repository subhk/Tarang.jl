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
"""

# MPI, FFTW already in Tarang.jl

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

# ============================================================================
# TransposableFieldStorage (deferred from field.jl due to type dependencies)
# ============================================================================

"""
    TransposableFieldStorage{CT, N} <: AbstractFieldStorage

Storage for distributed GPU+MPI fields with 2D pencil decomposition.
Absorbs the functionality previously in TransposableField wrapper.

CT is the complex element type (Complex{T}), N is the number of dimensions.
"""
mutable struct TransposableFieldStorage{CT, N, B<:SerialFieldStorage} <: AbstractFieldStorage
    base::B
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

    # TransposableField is designed for multi-dimensional distributed
    # transposes. Tarang's spectral Domain contract requires one process in 1D.
    if N == 1 && dist.size > 1
        error("TransposableField is not supported for 1D domains with MPI (nprocs=$(dist.size)). " *
              "1D domains have only one dimension which cannot be transposed. " *
              "Use a single process for 1D spectral transforms.")
    end
    # Spectral transforms use complex storage at the wrapped field's precision.
    # A Distributor may own fields with a dtype different from its default dtype.
    T = field.dtype <: Complex ? field.dtype : Complex{field.dtype}

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

    # Every operation after the communicator split belongs to this constructor.
    # Capture local failures, agree over the parent communicator, then have every
    # rank release its sub-communicators in the same order before throwing.
    tf = nothing
    construction_error = nothing
    try
        buffers = TransposeBuffers{T,N}(arch)

        # Create counts with correct sizes for each transpose operation.
        if N >= 3
            zy_nprocs = max(topo.row_size, 1)
            yx_nprocs = max(topo.col_size, 1)
        else
            if topo.Rx > 1 && topo.Ry > 1
                zy_nprocs = max(topo.row_size, 1)
            else
                zy_nprocs = topo.row_size > 1 ? topo.row_size : max(topo.col_size, 1)
            end
            yx_nprocs = max(topo.col_size, 1)
        end
        counts = TransposeCounts(zy_nprocs, yx_nprocs)
        comms = TransposeComms(topo)
        local_shapes = compute_local_shapes_2d(gshape, topo)

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

            if N == 2 && topo.Rx > 1 && topo.Ry > 1
                @warn "TransposableField for 2D domain with 2D true mesh (Rx=$(topo.Rx), Ry=$(topo.Ry)): " *
                      "async transposes are NOT supported. Use blocking transposes " *
                      "(transpose_z_to_y!, transpose_y_to_x!) or distributed_forward_transform! " *
                      "with overlap=false. Async functions will error at runtime." maxlog=1
            end
        end

        async_state = AsyncTransposeState()
        fft_plans = Dict{TransposeLayout, Any}()
        tf = TransposableField{typeof(field), T, N}(
            field, buffers, counts, comms, topo, gshape, local_shapes, async_state, fft_plans,
            0.0, 0.0, 0.0, 0.0, 0, false
        )
        allocate_transpose_buffers!(tf)
        compute_transpose_counts!(tf)
    catch err
        construction_error = err
    end

    construction_failed = construction_error !== nothing
    if dist.size > 1 && MPI.Initialized() && !MPI.Finalized()
        construction_failed = MPI.Allreduce(construction_failed ? 1 : 0, MPI.MAX, dist.comm) != 0
    end

    if construction_failed
        free_topology_2d!(topo)
        construction_error === nothing && error(
            "TransposableField construction failed on another MPI rank; " *
            "all ranks released their transpose communicators.")
        throw(construction_error)
    end

    return tf
end

"""
    make_transposable(field::ScalarField; kwargs...)

Helper function to create a TransposableField from a ScalarField.
"""
make_transposable(field::ScalarField; kwargs...) = TransposableField(field; kwargs...)

"""
    close(tf::TransposableField)

Collectively release the MPI sub-communicators owned by `tf`. Every rank in the
field's distributor must call `close` in the same order before `MPI.Finalize`.
The operation is idempotent. Garbage collection never performs this collective.
"""
function Base.close(tf::TransposableField)
    tf.closed && return nothing

    if tf.async_state.in_progress && MPI.Initialized() && !MPI.Finalized()
        wait_transpose!(tf)
    end

    free_topology_2d!(tf.topology)
    tf.comms.zy_comm = nothing
    tf.comms.yx_comm = nothing
    tf.closed = true
    return nothing
end

Base.isopen(tf::TransposableField) = !tf.closed

@inline function _require_open(tf::TransposableField, operation::AbstractString)
    tf.closed && throw(ArgumentError(
        "$operation cannot use a closed TransposableField; construct a new wrapper."))
    return nothing
end

# ============================================================================
# MPI Communicator Creation (Legacy wrapper)
# ============================================================================

"""
    create_transpose_comms(dist::Distributor)

Create MPI sub-communicators for transpose operations.
This is a wrapper that creates a 2D topology internally.

WARNING: This function is deprecated. Prefer using `TransposableField` and call
`close(tf)` collectively when it is no longer needed. If you must use this function,
call `free_comms!(comms)` collectively when done to avoid MPI resource leaks.
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

"""Get the authoritative data array for the wrapper's current layout."""
function current_data(tf::TransposableField)
    # A one-rank distributed transform delegates to ScalarField's ordinary
    # basis-aware transform because grid and coefficient shapes can differ
    # (for example, a RealFourier half spectrum). No transpose buffer can
    # represent both shapes, so the wrapped field remains authoritative.
    if tf.field.dist.size == 1
        if tf.field.current_layout === :g
            return get_grid_data(tf.field)
        elseif tf.field.current_layout === :c
            return get_coeff_data(tf.field)
        end
        error("TransposableField has unsupported serial field layout " *
              "$(repr(tf.field.current_layout))")
    end

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
