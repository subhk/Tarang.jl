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
"""

using MPI
using FFTW

# ============================================================================
# Transpose Layout Enum
# ============================================================================

"""
    TransposeLayout

Enum representing which dimension is local (not distributed).
"""
@enum TransposeLayout begin
    XLocal  # x-dimension fully local, y,z distributed
    YLocal  # y-dimension fully local, x,z distributed
    ZLocal  # z-dimension fully local, x,y distributed (typical starting layout)
end

# ============================================================================
# 2D Topology (Oceananigans-style)
# ============================================================================

"""
    Topology2D

2D process topology for pencil decomposition, following Oceananigans.jl conventions.

For a Rx × Ry topology:
- Rx: number of processes in x-direction (column communicator size)
- Ry: number of processes in y-direction (row communicator size)
- rank_x (rx): this process's x-coordinate (0 ≤ rx < Rx)
- rank_y (ry): this process's y-coordinate (0 ≤ ry < Ry)
"""
struct Topology2D
    Rx::Int  # Processes in x-direction
    Ry::Int  # Processes in y-direction
    rx::Int  # This process's x-coordinate
    ry::Int  # This process's y-coordinate

    # Row communicator: processes with same rx (for Z↔Y transposes)
    row_comm::Union{Nothing, MPI.Comm}
    row_rank::Int
    row_size::Int

    # Column communicator: processes with same ry (for Y↔X transposes)
    col_comm::Union{Nothing, MPI.Comm}
    col_rank::Int
    col_size::Int
end

function Topology2D()
    return Topology2D(1, 1, 0, 0, nothing, 0, 1, nothing, 0, 1)
end

"""
    create_topology_2d(comm::MPI.Comm, Rx::Int, Ry::Int)

Create a 2D process topology with row and column communicators.
Following Oceananigans.jl's DistributedFFTs approach.
"""
function create_topology_2d(comm::MPI.Comm, Rx::Int, Ry::Int)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    @assert Rx * Ry == size "Topology Rx×Ry ($Rx×$Ry) must equal number of processes ($size)"

    # Compute (rx, ry) coordinates from rank (row-major ordering)
    ry = rank ÷ Rx
    rx = rank % Rx

    # Row communicator: all processes with same rx (used for Z↔Y transpose)
    # These processes share a column and communicate along y
    row_color = rx
    row_comm = MPI.Comm_split(comm, row_color, ry)
    row_rank = MPI.Comm_rank(row_comm)
    row_size = MPI.Comm_size(row_comm)

    # Column communicator: all processes with same ry (used for Y↔X transpose)
    # These processes share a row and communicate along x
    col_color = ry
    col_comm = MPI.Comm_split(comm, col_color, rx)
    col_rank = MPI.Comm_rank(col_comm)
    col_size = MPI.Comm_size(col_comm)

    return Topology2D(Rx, Ry, rx, ry, row_comm, row_rank, row_size, col_comm, col_rank, col_size)
end

# ============================================================================
# Async Communication State
# ============================================================================

"""
    AsyncTransposeState

State for asynchronous transpose operations, enabling communication/computation overlap.
"""
mutable struct AsyncTransposeState
    # MPI request for non-blocking operation
    request::Union{Nothing, MPI.Request}

    # Whether an async operation is in progress
    in_progress::Bool

    # Source and destination layouts
    from_layout::TransposeLayout
    to_layout::TransposeLayout

    # Timing statistics
    pack_time::Float64
    comm_time::Float64
    unpack_time::Float64
    wait_time::Float64
end

function AsyncTransposeState()
    return AsyncTransposeState(nothing, false, ZLocal, ZLocal, 0.0, 0.0, 0.0, 0.0)
end

# ============================================================================
# Transpose Buffers (Enhanced)
# ============================================================================

"""
    TransposeBuffers{T,N}

Pre-allocated buffers for transpose operations.
Holds data arrays for each layout, communication buffers, and staging buffers.
"""
struct TransposeBuffers{T,N}
    # Data arrays for each layout
    x_local_data::Union{Nothing, AbstractArray{T,N}}
    y_local_data::Union{Nothing, AbstractArray{T,N}}
    z_local_data::Union{Nothing, AbstractArray{T,N}}

    # Double-buffered communication buffers for async operations
    send_buffer::Union{Nothing, AbstractArray{T,1}}
    recv_buffer::Union{Nothing, AbstractArray{T,1}}
    send_buffer_2::Union{Nothing, AbstractArray{T,1}}  # For double buffering
    recv_buffer_2::Union{Nothing, AbstractArray{T,1}}  # For double buffering

    # CPU staging buffers (for non-CUDA-aware MPI)
    send_staging::Union{Nothing, Vector{T}}
    recv_staging::Union{Nothing, Vector{T}}

    # Current active layout
    active_layout::Base.RefValue{TransposeLayout}

    # Which buffer set is active (for double buffering)
    active_buffer::Base.RefValue{Int}

    # Architecture (CPU or GPU)
    architecture::AbstractArchitecture
end

function TransposeBuffers{T,N}(arch::AbstractArchitecture) where {T,N}
    active = Ref(ZLocal)
    active_buf = Ref(1)
    return TransposeBuffers{T,N}(
        nothing, nothing, nothing,
        nothing, nothing, nothing, nothing,
        nothing, nothing,
        active, active_buf, arch
    )
end

# ============================================================================
# Transpose Counts (for MPI.Alltoallv)
# ============================================================================

"""
    TransposeCounts

Send/receive counts and displacements for MPI.Alltoallv operations.
Pre-computed for Z↔Y and Y↔X transposes.
"""
struct TransposeCounts
    # Z ↔ Y transpose (along row communicator)
    zy_send_counts::Vector{Int}
    zy_recv_counts::Vector{Int}
    zy_send_displs::Vector{Int}
    zy_recv_displs::Vector{Int}

    # Y ↔ X transpose (along column communicator)
    yx_send_counts::Vector{Int}
    yx_recv_counts::Vector{Int}
    yx_send_displs::Vector{Int}
    yx_recv_displs::Vector{Int}
end

function TransposeCounts(nprocs::Int)
    return TransposeCounts(
        zeros(Int, nprocs), zeros(Int, nprocs), zeros(Int, nprocs), zeros(Int, nprocs),
        zeros(Int, nprocs), zeros(Int, nprocs), zeros(Int, nprocs), zeros(Int, nprocs)
    )
end

# ============================================================================
# Transpose Communicators (Legacy - kept for compatibility)
# ============================================================================

"""
    TransposeComms

MPI sub-communicators for transpose operations.
Wrapper around Topology2D for backward compatibility.
"""
struct TransposeComms
    # Z↔Y transpose communicator (row communicator)
    zy_comm::Union{Nothing, MPI.Comm}
    zy_rank::Int
    zy_size::Int

    # Y↔X transpose communicator (column communicator)
    yx_comm::Union{Nothing, MPI.Comm}
    yx_rank::Int
    yx_size::Int
end

function TransposeComms()
    return TransposeComms(nothing, 0, 1, nothing, 0, 1)
end

function TransposeComms(topo::Topology2D)
    return TransposeComms(
        topo.row_comm, topo.row_rank, topo.row_size,
        topo.col_comm, topo.col_rank, topo.col_size
    )
end

# ============================================================================
# TransposableField Type
# ============================================================================

"""
    TransposableField{F<:ScalarField,T,N} <: Operand

A wrapper around ScalarField that provides efficient multi-layout transpose
capabilities for distributed GPU+MPI spectral transforms.

## Fields
- `field`: The underlying ScalarField
- `buffers`: Pre-allocated buffers for each layout
- `counts`: MPI send/recv counts for Alltoallv
- `comms`: MPI sub-communicators for transpose operations
- `topology`: 2D process topology (Oceananigans-style)
- `global_shape`: Global array dimensions
- `local_shapes`: Local shapes for each layout
- `async_state`: State for async communication overlap
- `fft_plans`: FFT plans for each layout

## Usage
```julia
# Create from existing field
tf = TransposableField(field)
# or use the helper function
tf = make_transposable(field)

# Synchronous distributed transform
distributed_forward_transform!(tf)
distributed_backward_transform!(tf)

# Async with overlap (advanced)
async_transpose_z_to_y!(tf)
# ... do other work ...
wait_transpose!(tf)
```
"""
mutable struct TransposableField{F<:ScalarField,T,N} <: Operand
    field::F
    buffers::TransposeBuffers{T,N}
    counts::TransposeCounts
    comms::TransposeComms
    topology::Topology2D
    global_shape::NTuple{N,Int}
    local_shapes::Dict{TransposeLayout, NTuple{N,Int}}
    async_state::AsyncTransposeState

    # FFT plans for each layout (lazily initialized)
    fft_plans::Dict{TransposeLayout, Any}

    # Performance statistics
    total_transpose_time::Float64
    total_fft_time::Float64
    total_pack_time::Float64
    total_unpack_time::Float64
    num_transposes::Int
end

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

    # Get global shape from field's domain
    gshape = if field.domain !== nothing
        global_shape(field.domain)
    else
        size(field["g"])
    end

    N = length(gshape)
    T = Complex{dist.dtype}  # Spectral transforms use complex

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

    # Create counts
    counts = TransposeCounts(max(topo.row_size, topo.col_size))

    # Create comms wrapper
    comms = TransposeComms(topo)

    # Compute local shapes for each layout
    local_shapes = compute_local_shapes_2d(gshape, topo)

    # Async state
    async_state = AsyncTransposeState()

    # FFT plans dictionary
    fft_plans = Dict{TransposeLayout, Any}()

    tf = TransposableField{typeof(field), T, N}(
        field, buffers, counts, comms, topo, gshape, local_shapes, async_state, fft_plans,
        0.0, 0.0, 0.0, 0.0, 0
    )

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

"""
    auto_topology(nprocs::Int, ndims::Int)

Automatically determine a good 2D topology for the given number of processes.
"""
function auto_topology(nprocs::Int, ndims::Int)
    if ndims <= 2
        return (nprocs, 1)
    end

    # Find factors closest to square root
    sqrt_n = isqrt(nprocs)
    for i in sqrt_n:-1:1
        if nprocs % i == 0
            return (i, nprocs ÷ i)
        end
    end
    return (1, nprocs)
end

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
        total_procs = Rx * Ry
        my_rank = topo.rx + topo.ry * Rx

        # For 2D with 1D decomposition:
        # ZLocal (starting): x local, y distributed → (Nx, Ny/P)
        # YLocal (after transpose): x distributed, y local → (Nx/P, Ny)
        # XLocal: same as YLocal for 2D
        shapes[ZLocal] = (Nx, divide_evenly(Ny, total_procs, my_rank))
        shapes[YLocal] = (divide_evenly(Nx, total_procs, my_rank), Ny)
        shapes[XLocal] = (divide_evenly(Nx, total_procs, my_rank), Ny)

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

    return compute_local_shapes_2d(global_shape, topo)
end

"""
    divide_evenly(n::Int, nprocs::Int, rank::Int)

Compute local size for even division with remainder distributed to first processes.
"""
function divide_evenly(n::Int, nprocs::Int, rank::Int)
    if nprocs <= 0
        return n
    end
    base = div(n, nprocs)
    remainder = mod(n, nprocs)
    return base + (rank < remainder ? 1 : 0)
end

"""
    local_range(n::Int, nprocs::Int, rank::Int)

Get the global index range for a given rank.
"""
function local_range(n::Int, nprocs::Int, rank::Int)
    base = div(n, nprocs)
    remainder = mod(n, nprocs)

    start = 1
    for r in 0:(rank-1)
        start += base + (r < remainder ? 1 : 0)
    end

    local_n = base + (rank < remainder ? 1 : 0)
    return start:(start + local_n - 1)
end

# ============================================================================
# MPI Communicator Creation (Legacy wrapper)
# ============================================================================

"""
    create_transpose_comms(dist::Distributor)

Create MPI sub-communicators for transpose operations.
This is a wrapper that creates a 2D topology internally.
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
# Buffer Allocation
# ============================================================================

"""
    allocate_transpose_buffers!(tf::TransposableField)

Allocate data and communication buffers for transpose operations.
Includes double-buffered communication buffers for async operations.
"""
function allocate_transpose_buffers!(tf::TransposableField{F,T,N}) where {F,T,N}
    arch = tf.buffers.architecture

    # Allocate data arrays for each layout
    z_shape = tf.local_shapes[ZLocal]
    y_shape = tf.local_shapes[YLocal]
    x_shape = tf.local_shapes[XLocal]

    # Create arrays on appropriate architecture
    z_data = zeros(arch, T, z_shape...)
    y_data = zeros(arch, T, y_shape...)
    x_data = zeros(arch, T, x_shape...)

    # Compute max buffer size needed for communication
    max_size = max(prod(z_shape), prod(y_shape), prod(x_shape))

    # Double-buffered communication buffers
    send_buf_1 = zeros(arch, T, max_size)
    recv_buf_1 = zeros(arch, T, max_size)
    send_buf_2 = zeros(arch, T, max_size)
    recv_buf_2 = zeros(arch, T, max_size)

    # CPU staging buffers for non-CUDA-aware MPI
    send_staging = is_gpu(arch) ? zeros(T, max_size) : nothing
    recv_staging = is_gpu(arch) ? zeros(T, max_size) : nothing

    # Update buffers struct
    tf.buffers = TransposeBuffers{T,N}(
        x_data, y_data, z_data,
        send_buf_1, recv_buf_1, send_buf_2, recv_buf_2,
        send_staging, recv_staging,
        tf.buffers.active_layout,
        tf.buffers.active_buffer,
        arch
    )

    return tf
end

# ============================================================================
# Transpose Count Computation
# ============================================================================

"""
    compute_transpose_counts!(tf::TransposableField)

Compute send/receive counts and displacements for MPI.Alltoallv.
"""
function compute_transpose_counts!(tf::TransposableField{F,T,N}) where {F,T,N}
    topo = tf.topology

    if topo.Rx * topo.Ry == 1
        return
    end

    # Z↔Y transpose counts (along row communicator)
    compute_zy_counts_2d!(tf)

    # Y↔X transpose counts (along column communicator)
    compute_yx_counts_2d!(tf)
end

function compute_zy_counts_2d!(tf::TransposableField{F,T,N}) where {F,T,N}
    topo = tf.topology
    gshape = tf.global_shape

    if N == 3
        # For 3D, Z→Y uses row_comm
        if topo.row_size == 1
            return
        end
        nprocs = topo.row_size
        Nx, Ny, Nz = gshape

        z_shape = tf.local_shapes[ZLocal]
        y_shape = tf.local_shapes[YLocal]

        # Z→Y: redistribute z-dimension among row communicator
        # Local x-size stays the same (Nx/Rx)
        local_nx = z_shape[1]

        send_offset = 0
        recv_offset = 0

        for p in 0:(nprocs-1)
            # Chunks of z we send to process p
            Nz_p = divide_evenly(Nz, nprocs, p)
            local_ny_send = z_shape[2]
            send_count = local_nx * local_ny_send * Nz_p

            # Chunks of y we receive from process p
            Ny_p = divide_evenly(Ny, nprocs, p)
            local_nz_recv = y_shape[3]
            recv_count = local_nx * Ny_p * local_nz_recv

            tf.counts.zy_send_counts[p+1] = send_count
            tf.counts.zy_recv_counts[p+1] = recv_count
            tf.counts.zy_send_displs[p+1] = send_offset
            tf.counts.zy_recv_displs[p+1] = recv_offset

            send_offset += send_count
            recv_offset += recv_count
        end

    elseif N == 2
        Nx, Ny = gshape

        z_shape = tf.local_shapes[ZLocal]  # (Nx, Ny/P)
        y_shape = tf.local_shapes[YLocal]  # (Nx/P, Ny)

        # For 2D, use whichever communicator has multiple processes
        # (same logic as in transpose_z_to_y!)
        actual_nprocs = topo.row_size > 1 ? topo.row_size : topo.col_size

        if actual_nprocs <= 1
            return  # No transpose needed
        end

        # 2D Z→Y: redistribute x-dimension
        # ZLocal = (Nx, Ny_local) → YLocal = (Nx_local, Ny)
        local_ny_send = z_shape[2]  # Ny/P

        send_offset = 0
        recv_offset = 0

        for p in 0:(actual_nprocs-1)
            # Chunks of x we send to process p
            Nx_p = divide_evenly(Nx, actual_nprocs, p)
            send_count = Nx_p * local_ny_send

            # Chunks of y we receive from process p
            Ny_p = divide_evenly(Ny, actual_nprocs, p)
            local_nx_recv = y_shape[1]  # Nx/P
            recv_count = local_nx_recv * Ny_p

            tf.counts.zy_send_counts[p+1] = send_count
            tf.counts.zy_recv_counts[p+1] = recv_count
            tf.counts.zy_send_displs[p+1] = send_offset
            tf.counts.zy_recv_displs[p+1] = recv_offset

            send_offset += send_count
            recv_offset += recv_count
        end
    end
end

function compute_yx_counts_2d!(tf::TransposableField{F,T,N}) where {F,T,N}
    topo = tf.topology

    if topo.col_size == 1
        return
    end

    nprocs = topo.col_size
    gshape = tf.global_shape

    if N == 3
        Nx, Ny, Nz = gshape

        y_shape = tf.local_shapes[YLocal]
        x_shape = tf.local_shapes[XLocal]

        # Y→X: redistribute x-dimension among column communicator
        local_nz = y_shape[3]

        send_offset = 0
        recv_offset = 0

        for p in 0:(nprocs-1)
            # Chunks of x we send to process p
            Nx_p = divide_evenly(Nx, nprocs, p)
            local_ny_send = y_shape[2]
            send_count = Nx_p * local_ny_send * local_nz

            # Chunks of y we receive from process p
            Ny_p = divide_evenly(Ny, nprocs, p)
            recv_count = x_shape[1] * Ny_p * local_nz

            tf.counts.yx_send_counts[p+1] = send_count
            tf.counts.yx_recv_counts[p+1] = recv_count
            tf.counts.yx_send_displs[p+1] = send_offset
            tf.counts.yx_recv_displs[p+1] = recv_offset

            send_offset += send_count
            recv_offset += recv_count
        end
    end
end

# ============================================================================
# Synchronous Transpose Operations
# ============================================================================

"""
    transpose_z_to_y!(tf::TransposableField)

Transpose from ZLocal layout to YLocal layout (synchronous).
Uses MPI.Alltoallv for communication.
"""
function transpose_z_to_y!(tf::TransposableField{F,T,N}) where {F,T,N}
    @assert tf.buffers.active_layout[] == ZLocal "Must be in ZLocal layout"

    topo = tf.topology

    # For 2D with 1D decomposition, use col_comm if row_comm has only 1 process
    # For 3D, always use row_comm for Z→Y
    if N >= 3
        comm = topo.row_comm
        comm_size = topo.row_size
    else
        # 2D case: use whichever communicator has multiple processes
        if topo.row_size > 1
            comm = topo.row_comm
            comm_size = topo.row_size
        elseif topo.col_size > 1
            comm = topo.col_comm
            comm_size = topo.col_size
        else
            comm = nothing
            comm_size = 1
        end
    end

    if comm_size == 1
        # No transpose needed - just copy data
        copyto!(vec(tf.buffers.y_local_data), vec(tf.buffers.z_local_data))
        tf.buffers.active_layout[] = YLocal
        return tf
    end

    arch = tf.buffers.architecture
    start_time = time()

    # Get active buffers
    send_buf, recv_buf = get_active_buffers(tf)

    # Determine pack/unpack dimensions based on ndims
    # 3D: pack dim=3 (Z redistributed), unpack dim=2 (receiving Y chunks)
    # 2D: pack dim=1 (X redistributed), unpack dim=2 (receiving Y chunks)
    pack_dim = N >= 3 ? 3 : 1
    unpack_dim = 2

    # Pack data into send buffer
    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.z_local_data,
                        tf.counts.zy_send_counts, tf.counts.zy_send_displs,
                        pack_dim, comm_size, arch)
    tf.total_pack_time += time() - pack_start

    # Perform MPI communication
    _do_alltoallv!(send_buf, recv_buf,
                   tf.counts.zy_send_counts, tf.counts.zy_recv_counts,
                   comm, arch, tf.buffers)

    # Unpack received data
    unpack_start = time()
    unpack_from_transpose!(tf.buffers.y_local_data, recv_buf,
                          tf.counts.zy_recv_counts, tf.counts.zy_recv_displs,
                          unpack_dim, comm_size, arch)
    tf.total_unpack_time += time() - unpack_start

    tf.buffers.active_layout[] = YLocal
    tf.total_transpose_time += time() - start_time
    tf.num_transposes += 1

    return tf
end

"""
    transpose_y_to_z!(tf::TransposableField)

Transpose from YLocal layout to ZLocal layout (reverse of Z→Y).
"""
function transpose_y_to_z!(tf::TransposableField{F,T,N}) where {F,T,N}
    @assert tf.buffers.active_layout[] == YLocal "Must be in YLocal layout"

    topo = tf.topology

    # Use same communicator logic as transpose_z_to_y! (reverse direction)
    if N >= 3
        comm = topo.row_comm
        comm_size = topo.row_size
    else
        if topo.row_size > 1
            comm = topo.row_comm
            comm_size = topo.row_size
        elseif topo.col_size > 1
            comm = topo.col_comm
            comm_size = topo.col_size
        else
            comm = nothing
            comm_size = 1
        end
    end

    if comm_size == 1
        copyto!(vec(tf.buffers.z_local_data), vec(tf.buffers.y_local_data))
        tf.buffers.active_layout[] = ZLocal
        return tf
    end

    arch = tf.buffers.architecture
    start_time = time()

    send_buf, recv_buf = get_active_buffers(tf)

    # Determine pack/unpack dimensions based on ndims (reverse of Z→Y)
    # 3D: pack dim=2 (Y redistributed), unpack dim=3 (receiving Z chunks)
    # 2D: pack dim=2 (Y redistributed), unpack dim=1 (receiving X chunks)
    pack_dim = 2
    unpack_dim = N >= 3 ? 3 : 1

    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.y_local_data,
                        tf.counts.zy_recv_counts, tf.counts.zy_recv_displs,
                        pack_dim, comm_size, arch)
    tf.total_pack_time += time() - pack_start

    # Note: swap send/recv counts for reverse direction
    _do_alltoallv!(send_buf, recv_buf,
                   tf.counts.zy_recv_counts, tf.counts.zy_send_counts,
                   comm, arch, tf.buffers)

    unpack_start = time()
    unpack_from_transpose!(tf.buffers.z_local_data, recv_buf,
                          tf.counts.zy_send_counts, tf.counts.zy_send_displs,
                          unpack_dim, comm_size, arch)
    tf.total_unpack_time += time() - unpack_start

    tf.buffers.active_layout[] = ZLocal
    tf.total_transpose_time += time() - start_time
    tf.num_transposes += 1

    return tf
end

"""
    transpose_y_to_x!(tf::TransposableField)

Transpose from YLocal layout to XLocal layout.
"""
function transpose_y_to_x!(tf::TransposableField{F,T,N}) where {F,T,N}
    @assert tf.buffers.active_layout[] == YLocal "Must be in YLocal layout"

    topo = tf.topology

    if topo.col_size == 1
        copyto!(vec(tf.buffers.x_local_data), vec(tf.buffers.y_local_data))
        tf.buffers.active_layout[] = XLocal
        return tf
    end

    arch = tf.buffers.architecture
    start_time = time()

    send_buf, recv_buf = get_active_buffers(tf)

    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.y_local_data,
                        tf.counts.yx_send_counts, tf.counts.yx_send_displs,
                        1, topo.col_size, arch)
    tf.total_pack_time += time() - pack_start

    _do_alltoallv!(send_buf, recv_buf,
                   tf.counts.yx_send_counts, tf.counts.yx_recv_counts,
                   topo.col_comm, arch, tf.buffers)

    unpack_start = time()
    unpack_from_transpose!(tf.buffers.x_local_data, recv_buf,
                          tf.counts.yx_recv_counts, tf.counts.yx_recv_displs,
                          1, topo.col_size, arch)
    tf.total_unpack_time += time() - unpack_start

    tf.buffers.active_layout[] = XLocal
    tf.total_transpose_time += time() - start_time
    tf.num_transposes += 1

    return tf
end

"""
    transpose_x_to_y!(tf::TransposableField)

Transpose from XLocal layout to YLocal layout (reverse of Y→X).
"""
function transpose_x_to_y!(tf::TransposableField{F,T,N}) where {F,T,N}
    @assert tf.buffers.active_layout[] == XLocal "Must be in XLocal layout"

    topo = tf.topology

    if topo.col_size == 1
        copyto!(vec(tf.buffers.y_local_data), vec(tf.buffers.x_local_data))
        tf.buffers.active_layout[] = YLocal
        return tf
    end

    arch = tf.buffers.architecture
    start_time = time()

    send_buf, recv_buf = get_active_buffers(tf)

    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.x_local_data,
                        tf.counts.yx_recv_counts, tf.counts.yx_recv_displs,
                        1, topo.col_size, arch)
    tf.total_pack_time += time() - pack_start

    _do_alltoallv!(send_buf, recv_buf,
                   tf.counts.yx_recv_counts, tf.counts.yx_send_counts,
                   topo.col_comm, arch, tf.buffers)

    unpack_start = time()
    unpack_from_transpose!(tf.buffers.y_local_data, recv_buf,
                          tf.counts.yx_send_counts, tf.counts.yx_send_displs,
                          1, topo.col_size, arch)
    tf.total_unpack_time += time() - unpack_start

    tf.buffers.active_layout[] = YLocal
    tf.total_transpose_time += time() - start_time
    tf.num_transposes += 1

    return tf
end

# ============================================================================
# Async Transpose Operations
# ============================================================================

"""
    async_transpose_z_to_y!(tf::TransposableField)

Start asynchronous transpose from ZLocal to YLocal.
Returns immediately after initiating communication.
Use `wait_transpose!(tf)` to complete the operation.
"""
function async_transpose_z_to_y!(tf::TransposableField{F,T,N}) where {F,T,N}
    @assert tf.buffers.active_layout[] == ZLocal "Must be in ZLocal layout"
    @assert !tf.async_state.in_progress "Another async operation is in progress"

    topo = tf.topology

    # Use same communicator selection logic as transpose_z_to_y!
    if N >= 3
        comm = topo.row_comm
        comm_size = topo.row_size
    else
        if topo.row_size > 1
            comm = topo.row_comm
            comm_size = topo.row_size
        elseif topo.col_size > 1
            comm = topo.col_comm
            comm_size = topo.col_size
        else
            comm = nothing
            comm_size = 1
        end
    end

    if comm_size == 1
        copyto!(vec(tf.buffers.y_local_data), vec(tf.buffers.z_local_data))
        tf.buffers.active_layout[] = YLocal
        return tf
    end

    arch = tf.buffers.architecture

    # Get buffers (use second set for async to avoid conflicts)
    send_buf = tf.buffers.send_buffer_2
    recv_buf = tf.buffers.recv_buffer_2

    # Determine pack dimension (same as transpose_z_to_y!)
    pack_dim = N >= 3 ? 3 : 1

    # Pack data
    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.z_local_data,
                        tf.counts.zy_send_counts, tf.counts.zy_send_displs,
                        pack_dim, comm_size, arch)
    tf.async_state.pack_time = time() - pack_start

    # Start non-blocking alltoallv
    request = _do_ialltoallv!(send_buf, recv_buf,
                              tf.counts.zy_send_counts, tf.counts.zy_recv_counts,
                              comm, arch, tf.buffers)

    tf.async_state.request = request
    tf.async_state.in_progress = true
    tf.async_state.from_layout = ZLocal
    tf.async_state.to_layout = YLocal

    return tf
end

"""
    async_transpose_y_to_x!(tf::TransposableField)

Start asynchronous transpose from YLocal to XLocal.
"""
function async_transpose_y_to_x!(tf::TransposableField{F,T,N}) where {F,T,N}
    @assert tf.buffers.active_layout[] == YLocal "Must be in YLocal layout"
    @assert !tf.async_state.in_progress "Another async operation is in progress"

    topo = tf.topology

    if topo.col_size == 1
        copyto!(vec(tf.buffers.x_local_data), vec(tf.buffers.y_local_data))
        tf.buffers.active_layout[] = XLocal
        return tf
    end

    arch = tf.buffers.architecture

    send_buf = tf.buffers.send_buffer_2
    recv_buf = tf.buffers.recv_buffer_2

    pack_start = time()
    pack_for_transpose!(send_buf, tf.buffers.y_local_data,
                        tf.counts.yx_send_counts, tf.counts.yx_send_displs,
                        1, topo.col_size, arch)
    tf.async_state.pack_time = time() - pack_start

    request = _do_ialltoallv!(send_buf, recv_buf,
                              tf.counts.yx_send_counts, tf.counts.yx_recv_counts,
                              topo.col_comm, arch, tf.buffers)

    tf.async_state.request = request
    tf.async_state.in_progress = true
    tf.async_state.from_layout = YLocal
    tf.async_state.to_layout = XLocal

    return tf
end

"""
    wait_transpose!(tf::TransposableField)

Wait for asynchronous transpose to complete and finalize the operation.
"""
function wait_transpose!(tf::TransposableField{F,T,N}) where {F,T,N}
    if !tf.async_state.in_progress
        return tf
    end

    arch = tf.buffers.architecture
    topo = tf.topology

    # Wait for MPI communication to complete
    wait_start = time()
    if tf.async_state.request !== nothing
        MPI.Wait(tf.async_state.request)
    end
    tf.async_state.wait_time = time() - wait_start

    # Get the receive buffer - for non-CUDA-aware MPI on GPU, data is in staging buffer
    recv_buf = tf.buffers.recv_buffer_2

    # If we used staging buffers (non-CUDA-aware MPI with GPU), copy back to GPU
    if is_gpu(arch) && !check_cuda_aware_mpi() && tf.buffers.recv_staging !== nothing
        copyto!(recv_buf, on_architecture(arch, tf.buffers.recv_staging))
    end

    # Unpack based on destination layout
    unpack_start = time()
    if tf.async_state.to_layout == YLocal
        # Determine correct comm_size for 2D vs 3D (same logic as transpose_z_to_y!)
        if N >= 3
            comm_size = topo.row_size
        else
            comm_size = topo.row_size > 1 ? topo.row_size : topo.col_size
        end
        unpack_dim = 2
        unpack_from_transpose!(tf.buffers.y_local_data, recv_buf,
                              tf.counts.zy_recv_counts, tf.counts.zy_recv_displs,
                              unpack_dim, comm_size, arch)
    elseif tf.async_state.to_layout == XLocal
        unpack_from_transpose!(tf.buffers.x_local_data, recv_buf,
                              tf.counts.yx_recv_counts, tf.counts.yx_recv_displs,
                              1, topo.col_size, arch)
    end
    tf.async_state.unpack_time = time() - unpack_start

    # Update state
    tf.buffers.active_layout[] = tf.async_state.to_layout
    tf.async_state.in_progress = false
    tf.async_state.request = nothing

    # Update statistics
    tf.total_pack_time += tf.async_state.pack_time
    tf.total_unpack_time += tf.async_state.unpack_time
    tf.num_transposes += 1

    return tf
end

"""
    is_transpose_complete(tf::TransposableField)

Check if an async transpose has completed without blocking.
"""
function is_transpose_complete(tf::TransposableField)
    if !tf.async_state.in_progress
        return true
    end

    if tf.async_state.request === nothing
        return true
    end

    flag, _ = MPI.Test(tf.async_state.request)
    return flag
end

# ============================================================================
# Buffer Helpers
# ============================================================================

"""Get active send/receive buffers"""
function get_active_buffers(tf::TransposableField)
    if tf.buffers.active_buffer[] == 1
        return tf.buffers.send_buffer, tf.buffers.recv_buffer
    else
        return tf.buffers.send_buffer_2, tf.buffers.recv_buffer_2
    end
end

"""Swap active buffer set"""
function swap_buffers!(tf::TransposableField)
    tf.buffers.active_buffer[] = tf.buffers.active_buffer[] == 1 ? 2 : 1
end

# ============================================================================
# Pack/Unpack Operations (CPU implementations)
# ============================================================================

"""
    pack_for_transpose!(buffer, data, counts, displs, dim, nranks, arch::CPU)

Pack data into contiguous buffer for MPI.Alltoallv (CPU version).
Reorders data so each destination rank receives a contiguous chunk.

For dim=3 (Z→Y transpose): packs z-slices for each rank
For dim=2 (Y→X transpose): packs y-slices for each rank
For dim=1 (X redistribution): packs x-slices for each rank
"""
function pack_for_transpose!(buffer, data, counts, displs, dim::Int,
                             nranks::Int, arch::CPU)
    if nranks == 1
        # Single process - simple copy
        copyto!(view(buffer, 1:length(data)), vec(data))
        return buffer
    end

    ndims_data = ndims(data)

    if ndims_data == 3
        Nx, Ny, Nz = size(data)
        _pack_3d_cpu!(buffer, data, counts, displs, dim, nranks, Nx, Ny, Nz)
    elseif ndims_data == 2
        Nx, Ny = size(data)
        _pack_2d_cpu!(buffer, data, counts, displs, dim, nranks, Nx, Ny)
    else
        # 1D - simple copy
        copyto!(view(buffer, 1:length(data)), vec(data))
    end

    return buffer
end

function _pack_3d_cpu!(buffer, data, counts, displs, dim::Int, nranks::Int,
                       Nx::Int, Ny::Int, Nz::Int)
    # Compute chunk sizes from counts
    chunk_sizes = zeros(Int, nranks)
    if dim == 3
        # Z being redistributed - chunk_size[r] = Nz_r (z-size for rank r)
        # count[r] = Nx * Ny * Nz_r, so Nz_r = count[r] / (Nx * Ny)
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Nx * Ny)
        end
    elseif dim == 2
        # Y being redistributed
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Nx * Nz)
        end
    else  # dim == 1
        # X being redistributed
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Ny * Nz)
        end
    end

    # Pack data
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        if dim == 3  # Z→Y: z-dimension redistributed
            # Find which rank owns this z-index
            rank, z_offset = 0, 0
            for r in 1:nranks
                if iz <= z_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                z_offset += chunk_sizes[r]
            end
            local_iz = iz - z_offset
            local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx

        elseif dim == 2  # Y→X: y-dimension redistributed
            rank, y_offset = 0, 0
            for r in 1:nranks
                if iy <= y_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                y_offset += chunk_sizes[r]
            end
            local_iy = iy - y_offset
            local_idx = (iz - 1) * Nx * chunk_sizes[rank] + (local_iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx

        else  # dim == 1: X redistributed
            rank, x_offset = 0, 0
            for r in 1:nranks
                if ix <= x_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                x_offset += chunk_sizes[r]
            end
            local_ix = ix - x_offset
            local_idx = (iz - 1) * chunk_sizes[rank] * Ny + (iy - 1) * chunk_sizes[rank] + local_ix
            buf_idx = displs[rank] + local_idx
        end

        @inbounds buffer[buf_idx] = data[ix, iy, iz]
    end
end

function _pack_2d_cpu!(buffer, data, counts, displs, dim::Int, nranks::Int,
                       Nx::Int, Ny::Int)
    # Compute chunk sizes
    chunk_sizes = zeros(Int, nranks)
    if dim == 2
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ Nx
        end
    else
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ Ny
        end
    end

    for iy in 1:Ny, ix in 1:Nx
        if dim == 2  # Y redistributed
            rank, y_offset = 0, 0
            for r in 1:nranks
                if iy <= y_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                y_offset += chunk_sizes[r]
            end
            local_iy = iy - y_offset
            local_idx = (local_iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx
        else  # dim == 1: X redistributed
            rank, x_offset = 0, 0
            for r in 1:nranks
                if ix <= x_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                x_offset += chunk_sizes[r]
            end
            local_ix = ix - x_offset
            local_idx = (iy - 1) * chunk_sizes[rank] + local_ix
            buf_idx = displs[rank] + local_idx
        end

        @inbounds buffer[buf_idx] = data[ix, iy]
    end
end

function pack_for_transpose!(buffer, data, counts, displs, dim::Int,
                             nranks::Int, arch::AbstractArchitecture)
    # Default implementation for GPU - use CPU version via staging
    # GPU version overrides this in TarangCUDAExt
    data_cpu = on_architecture(CPU(), data)
    buffer_cpu = on_architecture(CPU(), buffer)

    pack_for_transpose!(buffer_cpu, data_cpu, counts, displs, dim, nranks, CPU())
    copyto!(buffer, on_architecture(arch, buffer_cpu))

    return buffer
end

"""
    unpack_from_transpose!(data, buffer, counts, displs, dim, nranks, arch::CPU)

Unpack data from buffer after MPI.Alltoallv (CPU version).
Reconstructs the array from chunks received from different ranks.
"""
function unpack_from_transpose!(data, buffer, counts, displs, dim::Int,
                                nranks::Int, arch::CPU)
    if nranks == 1
        # Single process - simple copy
        copyto!(vec(data), view(buffer, 1:length(data)))
        return data
    end

    ndims_data = ndims(data)

    if ndims_data == 3
        Nx, Ny, Nz = size(data)
        _unpack_3d_cpu!(data, buffer, counts, displs, dim, nranks, Nx, Ny, Nz)
    elseif ndims_data == 2
        Nx, Ny = size(data)
        _unpack_2d_cpu!(data, buffer, counts, displs, dim, nranks, Nx, Ny)
    else
        copyto!(vec(data), view(buffer, 1:length(data)))
    end

    return data
end

function _unpack_3d_cpu!(data, buffer, counts, displs, dim::Int, nranks::Int,
                         Nx::Int, Ny::Int, Nz::Int)
    # Compute chunk sizes
    chunk_sizes = zeros(Int, nranks)
    if dim == 2  # After Z→Y: receiving y-chunks
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Nx * Nz)
        end
    elseif dim == 1  # After Y→X: receiving x-chunks
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Ny * Nz)
        end
    else  # dim == 3: receiving z-chunks
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ (Nx * Ny)
        end
    end

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        if dim == 2  # After Z→Y: receiving y-chunks
            rank, y_offset = 0, 0
            for r in 1:nranks
                if iy <= y_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                y_offset += chunk_sizes[r]
            end
            local_iy = iy - y_offset
            local_idx = (iz - 1) * Nx * chunk_sizes[rank] + (local_iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx

        elseif dim == 1  # After Y→X: receiving x-chunks
            rank, x_offset = 0, 0
            for r in 1:nranks
                if ix <= x_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                x_offset += chunk_sizes[r]
            end
            local_ix = ix - x_offset
            local_idx = (iz - 1) * chunk_sizes[rank] * Ny + (iy - 1) * chunk_sizes[rank] + local_ix
            buf_idx = displs[rank] + local_idx

        else  # dim == 3: receiving z-chunks
            rank, z_offset = 0, 0
            for r in 1:nranks
                if iz <= z_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                z_offset += chunk_sizes[r]
            end
            local_iz = iz - z_offset
            local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx
        end

        @inbounds data[ix, iy, iz] = buffer[buf_idx]
    end
end

function _unpack_2d_cpu!(data, buffer, counts, displs, dim::Int, nranks::Int,
                         Nx::Int, Ny::Int)
    chunk_sizes = zeros(Int, nranks)
    if dim == 2
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ Nx
        end
    else
        for r in 1:nranks
            chunk_sizes[r] = counts[r] ÷ Ny
        end
    end

    for iy in 1:Ny, ix in 1:Nx
        if dim == 2  # Receiving y-chunks
            rank, y_offset = 0, 0
            for r in 1:nranks
                if iy <= y_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                y_offset += chunk_sizes[r]
            end
            local_iy = iy - y_offset
            local_idx = (local_iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx
        else  # Receiving x-chunks
            rank, x_offset = 0, 0
            for r in 1:nranks
                if ix <= x_offset + chunk_sizes[r]
                    rank = r
                    break
                end
                x_offset += chunk_sizes[r]
            end
            local_ix = ix - x_offset
            local_idx = (iy - 1) * chunk_sizes[rank] + local_ix
            buf_idx = displs[rank] + local_idx
        end

        @inbounds data[ix, iy] = buffer[buf_idx]
    end
end

function unpack_from_transpose!(data, buffer, counts, displs, dim::Int,
                                nranks::Int, arch::AbstractArchitecture)
    # Default implementation
    data_cpu = on_architecture(CPU(), data)
    buffer_cpu = on_architecture(CPU(), buffer)

    unpack_from_transpose!(data_cpu, buffer_cpu, counts, displs, dim, nranks, CPU())
    copyto!(data, on_architecture(arch, data_cpu))

    return data
end

# ============================================================================
# MPI Communication Helpers
# ============================================================================

"""
    _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm, arch, buffers)

Perform blocking MPI.Alltoallv with appropriate handling for CPU and GPU arrays.
"""
function _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts,
                        comm::MPI.Comm, arch::CPU, buffers=nothing)
    MPI.Alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm)
    return recv_buf
end

function _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts,
                        comm::MPI.Comm, arch::AbstractArchitecture, buffers=nothing)
    # Check for CUDA-aware MPI
    if is_gpu(arch) && check_cuda_aware_mpi()
        # Direct GPU buffer transfer
        MPI.Alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm)
    else
        # Stage through CPU using pre-allocated staging buffers
        if buffers !== nothing && buffers.send_staging !== nothing
            send_cpu = buffers.send_staging
            recv_cpu = buffers.recv_staging
            copyto!(send_cpu, on_architecture(CPU(), send_buf))
        else
            send_cpu = on_architecture(CPU(), send_buf)
            recv_cpu = similar(send_cpu)
        end

        MPI.Alltoallv!(send_cpu, recv_cpu, send_counts, recv_counts, comm)

        if buffers !== nothing && buffers.recv_staging !== nothing
            copyto!(recv_buf, on_architecture(arch, recv_cpu))
        else
            copyto!(recv_buf, on_architecture(arch, recv_cpu))
        end
    end
    return recv_buf
end

"""
    _do_ialltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm, arch, buffers)

Perform non-blocking MPI.Ialltoallv (async).
Returns MPI.Request for later waiting.
"""
function _do_ialltoallv!(send_buf, recv_buf, send_counts, recv_counts,
                         comm::MPI.Comm, arch::CPU, buffers=nothing)
    request = MPI.Ialltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm)
    return request
end

function _do_ialltoallv!(send_buf, recv_buf, send_counts, recv_counts,
                         comm::MPI.Comm, arch::AbstractArchitecture, buffers=nothing)
    if is_gpu(arch) && check_cuda_aware_mpi()
        # Direct GPU buffer transfer (non-blocking)
        request = MPI.Ialltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm)
        return request
    else
        # For non-CUDA-aware MPI, we need to use staging buffers
        # Copy to CPU first, then do async MPI
        if buffers !== nothing && buffers.send_staging !== nothing
            send_cpu = buffers.send_staging
            recv_cpu = buffers.recv_staging
            copyto!(send_cpu, on_architecture(CPU(), send_buf))

            request = MPI.Ialltoallv!(send_cpu, recv_cpu, send_counts, recv_counts, comm)

            # Note: We'll need to copy recv_cpu back to GPU in wait_transpose!
            return request
        else
            # Fallback: use blocking version
            _do_alltoallv!(send_buf, recv_buf, send_counts, recv_counts, comm, arch, buffers)
            return nothing
        end
    end
end

# ============================================================================
# Distributed Transform Operations (with overlap support)
# ============================================================================

"""
    distributed_forward_transform!(tf::TransposableField; overlap=false)

Perform complete forward transform with transposes for distributed GPU+MPI.

Automatically selects FFT for Fourier bases and DCT for Chebyshev/Jacobi bases.

# Arguments
- `tf`: The TransposableField
- `overlap`: If true, use async transposes with computation overlap

Algorithm (3D Fourier-Fourier-Fourier):
1. Start in ZLocal layout
2. FFT/DCT in z (local)
3. Transpose Z→Y (with optional overlap)
4. FFT/DCT in y (local)
5. Transpose Y→X (with optional overlap)
6. FFT/DCT in x (local)

Supports mixed basis types (e.g., Chebyshev-Fourier).
"""
function distributed_forward_transform!(tf::TransposableField{F,T,N};
                                        overlap::Bool=false,
                                        plans=nothing) where {F,T,N}
    arch = tf.buffers.architecture

    # Get bases for each dimension
    xbasis = get_basis_for_dim(tf, 1)
    ybasis = get_basis_for_dim(tf, 2)
    zbasis = N >= 3 ? get_basis_for_dim(tf, 3) : nothing

    # Ensure we start in ZLocal layout with data from field
    tf.buffers.active_layout[] = ZLocal
    copyto!(vec(tf.buffers.z_local_data), vec(tf.field["g"]))

    if N >= 3
        # 3D case with full transpose sequence
        fft_start = time()

        # Step 1: Transform in z (local in ZLocal layout)
        transform_in_dim!(tf.buffers.z_local_data, 3, :forward, zbasis, arch)

        if overlap && tf.topology.row_size > 1
            # Async transpose with overlap pattern
            async_transpose_z_to_y!(tf)
            # Could do other work here while transpose is in progress
            wait_transpose!(tf)
        else
            transpose_z_to_y!(tf)
        end

        # Step 3: Transform in y (local in YLocal layout)
        transform_in_dim!(tf.buffers.y_local_data, 2, :forward, ybasis, arch)

        if overlap && tf.topology.col_size > 1
            async_transpose_y_to_x!(tf)
            wait_transpose!(tf)
        else
            transpose_y_to_x!(tf)
        end

        # Step 5: Transform in x (local in XLocal layout)
        transform_in_dim!(tf.buffers.x_local_data, 1, :forward, xbasis, arch)

        tf.total_fft_time += time() - fft_start

        # Copy result to field
        copyto!(vec(tf.field["c"]), vec(tf.buffers.x_local_data))

    elseif N == 2
        # 2D case: ZLocal=(Nx, Ny/P) → YLocal=(Nx/P, Ny)
        # Start with x local, y distributed
        fft_start = time()

        # Step 1: Transform in x (dim 1, which is local in ZLocal)
        transform_in_dim!(tf.buffers.z_local_data, 1, :forward, xbasis, arch)

        # Step 2: Transpose to get y local
        transpose_z_to_y!(tf)

        # Step 3: Transform in y (dim 2, which is now local in YLocal)
        transform_in_dim!(tf.buffers.y_local_data, 2, :forward, ybasis, arch)

        tf.total_fft_time += time() - fft_start

        copyto!(vec(tf.field["c"]), vec(tf.buffers.y_local_data))

    else
        # 1D case - no transpose needed
        fft_start = time()
        transform_in_dim!(tf.buffers.z_local_data, 1, :forward, xbasis, arch)
        tf.total_fft_time += time() - fft_start
        copyto!(vec(tf.field["c"]), vec(tf.buffers.z_local_data))
    end

    return tf
end

"""
    distributed_backward_transform!(tf::TransposableField; overlap=false)

Perform complete backward transform with transposes for distributed GPU+MPI.

Automatically selects IFFT for Fourier bases and inverse DCT for Chebyshev/Jacobi bases.
"""
function distributed_backward_transform!(tf::TransposableField{F,T,N};
                                         overlap::Bool=false,
                                         plans=nothing) where {F,T,N}
    arch = tf.buffers.architecture

    # Get bases for each dimension
    xbasis = get_basis_for_dim(tf, 1)
    ybasis = get_basis_for_dim(tf, 2)
    zbasis = N >= 3 ? get_basis_for_dim(tf, 3) : nothing

    if N >= 3
        # Start in XLocal layout with spectral data
        tf.buffers.active_layout[] = XLocal
        copyto!(vec(tf.buffers.x_local_data), vec(tf.field["c"]))

        fft_start = time()

        # Step 1: Inverse transform in x (local in XLocal layout)
        transform_in_dim!(tf.buffers.x_local_data, 1, :backward, xbasis, arch)

        # Step 2: Transpose X→Y
        transpose_x_to_y!(tf)

        # Step 3: Inverse transform in y (local in YLocal layout)
        transform_in_dim!(tf.buffers.y_local_data, 2, :backward, ybasis, arch)

        # Step 4: Transpose Y→Z
        transpose_y_to_z!(tf)

        # Step 5: Inverse transform in z (local in ZLocal layout)
        transform_in_dim!(tf.buffers.z_local_data, 3, :backward, zbasis, arch)

        tf.total_fft_time += time() - fft_start

        # Copy result to field
        copyto!(vec(tf.field["g"]), real.(vec(tf.buffers.z_local_data)))

    elseif N == 2
        # 2D case: YLocal=(Nx/P, Ny) → ZLocal=(Nx, Ny/P)
        # Start with y local (spectral data), x distributed
        tf.buffers.active_layout[] = YLocal
        copyto!(vec(tf.buffers.y_local_data), vec(tf.field["c"]))

        fft_start = time()

        # Step 1: Inverse transform in y (dim 2, which is local in YLocal)
        transform_in_dim!(tf.buffers.y_local_data, 2, :backward, ybasis, arch)

        # Step 2: Transpose to get x local
        transpose_y_to_z!(tf)

        # Step 3: Inverse transform in x (dim 1, which is now local in ZLocal)
        transform_in_dim!(tf.buffers.z_local_data, 1, :backward, xbasis, arch)

        tf.total_fft_time += time() - fft_start

        copyto!(vec(tf.field["g"]), real.(vec(tf.buffers.z_local_data)))

    else
        tf.buffers.active_layout[] = ZLocal
        copyto!(vec(tf.buffers.z_local_data), vec(tf.field["c"]))

        fft_start = time()
        transform_in_dim!(tf.buffers.z_local_data, 1, :backward, xbasis, arch)
        tf.total_fft_time += time() - fft_start

        copyto!(vec(tf.field["g"]), real.(vec(tf.buffers.z_local_data)))
    end

    return tf
end

# ============================================================================
# Basis-Aware Transform Helpers
# ============================================================================

"""
    transform_in_dim!(data, dim, direction, basis, arch)

Perform spectral transform along specified dimension, choosing FFT or DCT
based on the basis type.

# Arguments
- `data`: Array to transform (modified in-place)
- `dim`: Dimension along which to transform (1, 2, or 3)
- `direction`: `:forward` (physical→spectral) or `:backward` (spectral→physical)
- `basis`: The Basis object for this dimension (Fourier, ChebyshevT, etc.)
- `arch`: Architecture (CPU() or GPU())
"""
function transform_in_dim!(data, dim::Int, direction::Symbol, basis::Basis, arch::AbstractArchitecture)
    if basis isa FourierBasis
        fft_in_dim!(data, dim, direction, arch)
    elseif basis isa JacobiBasis  # Includes ChebyshevT, ChebyshevU, Legendre, etc.
        dct_in_dim!(data, dim, direction, arch)
    else
        # Fallback to FFT for unknown basis types
        @warn "Unknown basis type $(typeof(basis)), falling back to FFT"
        fft_in_dim!(data, dim, direction, arch)
    end
    return data
end

# Convenience method when no basis is specified (defaults to FFT)
function transform_in_dim!(data, dim::Int, direction::Symbol, ::Nothing, arch::AbstractArchitecture)
    fft_in_dim!(data, dim, direction, arch)
end

"""
    fft_in_dim!(data, dim, direction, arch::CPU)

Perform FFT along specified dimension (CPU version using FFTW).
"""
function fft_in_dim!(data, dim::Int, direction::Symbol, arch::CPU)
    if direction == :forward
        data .= FFTW.fft(data, dim)
    else
        data .= FFTW.ifft(data, dim)
    end
    return data
end

function fft_in_dim!(data, dim::Int, direction::Symbol, arch::AbstractArchitecture)
    # Default: fall back to CPU
    data_cpu = on_architecture(CPU(), data)
    fft_in_dim!(data_cpu, dim, direction, CPU())
    copyto!(data, on_architecture(arch, data_cpu))
    return data
end

"""
    dct_in_dim!(data, dim, direction, arch::CPU)

Perform DCT along specified dimension for Chebyshev transforms (CPU version).

Uses the same algorithm and scaling as ChebyshevTransform in transforms.jl:
- Forward (DCT-II): FFTW.REDFT10 - grid values to Chebyshev coefficients
- Backward (DCT-III): FFTW.REDFT01 - Chebyshev coefficients to grid values

Tarang normalization (matches ChebyshevTransform):
- Forward: k=0 scaled by 1/(2N), k>0 scaled by 1/N
- Backward: k=0 scaled by 1, k>0 scaled by 0.5
"""
function dct_in_dim!(data, dim::Int, direction::Symbol, arch::CPU)
    n = size(data, dim)

    # Handle complex data by transforming real and imaginary parts separately
    # (same approach as _chebyshev_forward in transforms.jl)
    if eltype(data) <: Complex
        real_part = real.(data)
        imag_part = imag.(data)

        dct_in_dim!(real_part, dim, direction, arch)
        dct_in_dim!(imag_part, dim, direction, arch)

        data .= complex.(real_part, imag_part)
        return data
    end

    # Real data path - matches ChebyshevTransform scaling
    if direction == :forward
        # DCT-II: grid → coefficients (same as _chebyshev_forward)
        result = FFTW.r2r(data, FFTW.REDFT10, (dim,))

        # Apply Tarang normalization (matches transform.forward_rescale_*)
        scale_zero = 1.0 / n / 2.0   # Same as forward_rescale_zero
        scale_pos = 1.0 / n          # Same as forward_rescale_pos
        _apply_dct_scaling!(result, dim, scale_zero, scale_pos)
        data .= result
    else
        # DCT-III: coefficients → grid (same as _chebyshev_backward)
        # Pre-scale: k=0 by 1.0, k>0 by 0.5 (matches backward_rescale_*)
        scaled_data = copy(data)
        _apply_dct_scaling!(scaled_data, dim, 1.0, 0.5)

        result = FFTW.r2r(scaled_data, FFTW.REDFT01, (dim,))
        data .= result
    end

    return data
end

"""
Apply scaling along a dimension (used for DCT normalization).
Matches the _scale_along_axis! pattern in transforms.jl.
"""
function _apply_dct_scaling!(data, dim::Int, scale_zero::Float64, scale_pos::Float64)
    n = size(data, dim)
    # Build scale vector: [scale_zero, scale_pos, scale_pos, ...]
    scale_vec = fill(scale_pos, n)
    scale_vec[1] = scale_zero

    # Reshape for broadcasting along the specified dimension
    shape = ntuple(i -> i == dim ? n : 1, ndims(data))
    data .*= reshape(scale_vec, shape...)
end

function dct_in_dim!(data, dim::Int, direction::Symbol, arch::AbstractArchitecture)
    # Default: fall back to CPU
    data_cpu = on_architecture(CPU(), data)
    dct_in_dim!(data_cpu, dim, direction, CPU())
    copyto!(data, on_architecture(arch, data_cpu))
    return data
end

"""
    get_basis_for_dim(tf::TransposableField, dim::Int)

Get the basis for a specific dimension from the field.
Returns nothing if bases are not available.
"""
function get_basis_for_dim(tf::TransposableField, dim::Int)
    if isempty(tf.field.bases)
        return nothing
    end
    if dim > length(tf.field.bases)
        return nothing
    end
    return tf.field.bases[dim]
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
export Topology2D, create_topology_2d, auto_topology
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
export create_transpose_comms
export get_transpose_stats, reset_transpose_stats!
export get_active_buffers, swap_buffers!
# Basis-aware transform helpers
export transform_in_dim!, fft_in_dim!, dct_in_dim!, get_basis_for_dim
