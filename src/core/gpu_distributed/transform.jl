# ============================================================================
# Enhanced Distributed GPU Transform (with TransposableField support)
# ============================================================================

"""
    DistributedGPUTransform

Enhanced distributed GPU transform that integrates with TransposableField
for efficient 2D pencil decomposition.

This type manages FFT plans for each transpose layout and coordinates
the full forward/backward transform sequence.
"""
mutable struct DistributedGPUTransform
    # Basic configuration
    config::DistributedGPUConfig

    # FFT plans for each layout (ZLocal, YLocal, XLocal)
    # Keys are TransposeLayout enum values, values are FFT plans
    plans::Dict{Any, Any}

    # Working TransposableField (created lazily)
    workspace::Any  # Union{Nothing, TransposableField}

    # Basis information for transform planning
    bases::Tuple{Vararg{Basis}}

    # Transform execution order
    transform_order::Vector{Int}

    # Performance statistics
    total_transpose_time::Float64
    total_fft_time::Float64
    num_transforms::Int

    function DistributedGPUTransform(config::DistributedGPUConfig, bases::Tuple{Vararg{Basis}})
        ndims = length(bases)
        transform_order = collect(1:ndims)  # Default order: 1, 2, 3, ...

        new(config, Dict{Any, Any}(), nothing, bases, transform_order, 0.0, 0.0, 0)
    end
end

"""
    create_distributed_gpu_transform(dist::Distributor, domain::Domain)

Create a DistributedGPUTransform for the given distributor and domain.
"""
function create_distributed_gpu_transform(dist::Distributor, domain::Domain)
    if dist.distributed_gpu_config === nothing
        # Create config if not exists
        gshape = global_shape(domain)
        config = DistributedGPUConfig(dist.comm, gshape;
                                       cuda_aware_mpi=check_cuda_aware_mpi())
        dist.distributed_gpu_config = config
    else
        config = dist.distributed_gpu_config
    end

    return DistributedGPUTransform(config, domain.bases)
end

"""
    setup_transposable_workspace!(transform::DistributedGPUTransform, field::ScalarField)

Setup or retrieve a TransposableField workspace for distributed transforms.
"""
function setup_transposable_workspace!(transform::DistributedGPUTransform, field)
    # The Distributor owns one TransposableField per (global shape, eltype) and
    # releases its communicators in close(dist); borrowing it here means this
    # transform never creates, and therefore never has to close, a wrapper.
    workspace = transpose_workspace!(field.dist, field)
    transform.workspace = workspace
    return workspace
end

"""
    close(transform::DistributedGPUTransform)

Drop the transform's reference to the Distributor-owned workspace. The
communicators belong to the Distributor and are released by `close(dist)`.
"""
function Base.close(transform::DistributedGPUTransform)
    transform.workspace = nothing
    return nothing
end

"""
    distributed_transform_forward!(transform::DistributedGPUTransform, field)

Perform forward distributed transform using TransposableField infrastructure.
"""
function distributed_transform_forward!(transform::DistributedGPUTransform, field)
    workspace = setup_transposable_workspace!(transform, field)

    start_time = time()

    # Use TransposableField's distributed transform
    distributed_forward_transform!(workspace; plans=transform.plans)

    transform.total_fft_time += time() - start_time
    transform.num_transforms += 1

    return field
end

"""
    distributed_transform_backward!(transform::DistributedGPUTransform, field)

Perform backward distributed transform using TransposableField infrastructure.
"""
function distributed_transform_backward!(transform::DistributedGPUTransform, field)
    workspace = setup_transposable_workspace!(transform, field)

    start_time = time()

    distributed_backward_transform!(workspace; plans=transform.plans)

    transform.total_fft_time += time() - start_time
    transform.num_transforms += 1

    return field
end

"""
    get_distributed_transform_stats(transform::DistributedGPUTransform)

Get performance statistics for the distributed transform.
"""
function get_distributed_transform_stats(transform::DistributedGPUTransform)
    return (
        total_transpose_time = transform.total_transpose_time,
        total_fft_time = transform.total_fft_time,
        num_transforms = transform.num_transforms,
        avg_time_per_transform = transform.num_transforms > 0 ?
            transform.total_fft_time / transform.num_transforms : 0.0
    )
end

