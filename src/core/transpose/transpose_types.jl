"""
    Transpose Types - Core type definitions for TransposableField

This file contains all the type definitions needed for the TransposableField
distributed transpose system.
"""

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

IMPORTANT: This is a mutable struct to allow nullifying communicator references after
free_topology_2d!() is called. This prevents use-after-free bugs where freed MPI
communicators are accidentally used (they don't become COMM_NULL automatically).
"""
mutable struct Topology2D
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
    free_topology_2d!(topo::Topology2D)

Free MPI communicators associated with the topology.
CRITICAL: Must be called before the topology is garbage collected to prevent
MPI communicator leaks. In long-running applications with many TransposableField
instances, failing to free communicators will exhaust MPI resources.

Note: This is safe to call multiple times or on a default Topology2D (nothing comms).
"""
function free_topology_2d!(topo::Topology2D)
    # Free row communicator if it exists and is not MPI.COMM_NULL
    if topo.row_comm !== nothing && topo.row_comm != MPI.COMM_NULL
        try
            MPI.free(topo.row_comm)
        catch e
            @warn "Failed to free row communicator: $e" maxlog=1
        end
        # CRITICAL: Nullify reference to prevent use-after-free
        # Freed MPI communicators don't become COMM_NULL automatically
        topo.row_comm = nothing
    end

    # Free column communicator if it exists and is not MPI.COMM_NULL
    if topo.col_comm !== nothing && topo.col_comm != MPI.COMM_NULL
        try
            MPI.free(topo.col_comm)
        catch e
            @warn "Failed to free column communicator: $e" maxlog=1
        end
        # CRITICAL: Nullify reference to prevent use-after-free
        topo.col_comm = nothing
    end
end

"""
    create_topology_2d(comm::MPI.Comm, Rx::Int, Ry::Int)

Create a 2D process topology with row and column communicators.
Following Oceananigans.jl's DistributedFFTs approach.

IMPORTANT: The returned Topology2D contains MPI communicators that must be freed
when no longer needed. Call `free_topology_2d!(topo)` to release resources.
"""
function create_topology_2d(comm::MPI.Comm, Rx::Int, Ry::Int)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # Critical invariant check - use throw instead of @assert for production safety
    if Rx * Ry != size
        throw(ArgumentError(
            "Topology Rx×Ry ($Rx×$Ry = $(Rx*Ry)) must equal number of processes ($size). " *
            "Check your mesh configuration or MPI process count."
        ))
    end

    # Validate Rx and Ry are positive
    if Rx < 1 || Ry < 1
        throw(ArgumentError("Topology dimensions must be positive: Rx=$Rx, Ry=$Ry"))
    end

    # Compute (rx, ry) coordinates from rank (row-major ordering)
    ry = rank ÷ Rx
    rx = rank % Rx

    # Row communicator: all processes with same rx (used for Z↔Y transpose)
    # These processes share a column and communicate along y
    row_color = rx
    row_comm = MPI.Comm_split(comm, row_color, ry)

    # Validate communicator split succeeded
    if row_comm == MPI.COMM_NULL
        error("Row communicator split failed for rank $rank (rx=$rx, ry=$ry). " *
              "This should not happen with valid topology configuration.")
    end

    row_rank = MPI.Comm_rank(row_comm)
    row_size = MPI.Comm_size(row_comm)

    # Column communicator: all processes with same ry (used for Y↔X transpose)
    # These processes share a row and communicate along x
    col_color = ry
    col_comm = try
        MPI.Comm_split(comm, col_color, rx)
    catch e
        # CRITICAL: Free row_comm if col_comm creation fails to prevent resource leak
        MPI.free(row_comm)
        rethrow(e)
    end

    if col_comm == MPI.COMM_NULL
        # CRITICAL: Free row_comm before erroring to prevent resource leak
        MPI.free(row_comm)
        error("Column communicator split failed for rank $rank (rx=$rx, ry=$ry).")
    end

    col_rank = MPI.Comm_rank(col_comm)
    col_size = MPI.Comm_size(col_comm)

    # Validate expected communicator sizes
    if row_size != Ry
        # Free both communicators before erroring
        MPI.free(row_comm)
        MPI.free(col_comm)
        error("Row communicator size ($row_size) does not match expected Ry ($Ry).")
    end
    if col_size != Rx
        # Free both communicators before erroring
        MPI.free(row_comm)
        MPI.free(col_comm)
        error("Column communicator size ($col_size) does not match expected Rx ($Rx).")
    end

    return Topology2D(Rx, Ry, rx, ry, row_comm, row_rank, row_size, col_comm, col_rank, col_size)
end

"""
    auto_topology(nprocs::Int, ndims::Int)

Automatically determine optimal 2D topology for given number of processes.
"""
function auto_topology(nprocs::Int, ndims::Int)
    if ndims == 1 || nprocs == 1
        return (nprocs, 1)
    end

    # Find factors closest to square root
    Rx = isqrt(nprocs)
    while nprocs % Rx != 0
        Rx -= 1
    end
    Ry = nprocs ÷ Rx

    return (Rx, Ry)
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

    # CRITICAL: Track the recv_size for correct staging buffer copy in wait_transpose!
    # When using non-CUDA-aware MPI, only recv_size elements of recv_staging are valid
    recv_size::Int
end

function AsyncTransposeState()
    return AsyncTransposeState(nothing, false, ZLocal, ZLocal, 0.0, 0.0, 0.0, 0.0, 0)
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

    # True when staging buffers are in use by an in-flight async MPI operation.
    # Sync operations must NOT reuse staging when this is set.
    staging_locked::Base.RefValue{Bool}

    # Architecture (CPU or GPU)
    architecture::AbstractArchitecture
end

function TransposeBuffers{T,N}(arch::AbstractArchitecture) where {T,N}
    active = Ref(ZLocal)
    active_buf = Ref(1)
    staging_lock = Ref(false)
    return TransposeBuffers{T,N}(
        nothing, nothing, nothing,
        nothing, nothing, nothing, nothing,
        nothing, nothing,
        active, active_buf, staging_lock, arch
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
struct TransposeCounts <: AbstractTransposeCounts
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

"""
    TransposeCounts(zy_nprocs::Int, yx_nprocs::Int)

Create TransposeCounts with separate sizes for Z↔Y (row_comm) and Y↔X (col_comm) transposes.
This ensures MPI.Alltoallv receives correctly sized counts/displs vectors.
"""
function TransposeCounts(zy_nprocs::Int, yx_nprocs::Int)
    return TransposeCounts(
        zeros(Int, zy_nprocs), zeros(Int, zy_nprocs), zeros(Int, zy_nprocs), zeros(Int, zy_nprocs),
        zeros(Int, yx_nprocs), zeros(Int, yx_nprocs), zeros(Int, yx_nprocs), zeros(Int, yx_nprocs)
    )
end

# Convenience constructor when using same size for both (for 2D decomposition with 1D mesh)
function TransposeCounts(nprocs::Int)
    return TransposeCounts(nprocs, nprocs)
end

# ============================================================================
# Transpose Communicators (Legacy - kept for compatibility)
# ============================================================================

"""
    TransposeComms

MPI sub-communicators for transpose operations.
Wrapper around Topology2D for backward compatibility.
"""
struct TransposeComms <: AbstractTransposeComms
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

"""
    free_comms!(comms::TransposeComms)

Free MPI communicators associated with TransposeComms.
CRITICAL: Must be called before the TransposeComms is discarded to prevent
MPI communicator leaks. Safe to call multiple times or on default (nothing) comms.

Note: For new code, prefer using TransposableField which handles cleanup automatically.
"""
function free_comms!(comms::TransposeComms)
    # Free Z↔Y communicator if it exists
    if comms.zy_comm !== nothing && comms.zy_comm != MPI.COMM_NULL
        try
            MPI.free(comms.zy_comm)
        catch e
            @warn "Failed to free zy_comm: $e" maxlog=1
        end
    end

    # Free Y↔X communicator if it exists
    if comms.yx_comm !== nothing && comms.yx_comm != MPI.COMM_NULL
        try
            MPI.free(comms.yx_comm)
        catch e
            @warn "Failed to free yx_comm: $e" maxlog=1
        end
    end
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
