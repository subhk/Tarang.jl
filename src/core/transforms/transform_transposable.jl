"""
    Transform Transposable - TransposableField transform planning

This file contains functions for planning FFT transforms for
TransposableField layouts.
"""

# ============================================================================
# TransposableField Transform Planning
# ============================================================================

"""
    _plan_local_axis_fft(arch, CT, shape, dim) -> plan

Plan the in-layout FFT along `dim` for a `TransposableField` buffer of element
type `CT` and local `shape`.

Two methods rather than five copies of `if is_gpu(arch) … else … end`: the host
method builds the FFTW plan, and the device method defers to `TarangCUDAExt`,
which creates the plan lazily from the resident array type.
"""
_plan_local_axis_fft(::GPU, ::Type, ::Tuple, ::Int) = :gpu_pending

function _plan_local_axis_fft(::CPU, ::Type{CT}, shape::Tuple, dim::Int) where {CT}
    return FFTW.plan_fft(zeros(CT, shape...), dim; flags=FFTW.MEASURE)
end

"""
    plan_transposable_transforms!(tf)

Create FFT plans for each layout in a TransposableField.

This function sets up FFTW/CUFFT plans for the dimension that is local
in each layout:
- ZLocal: plan FFT for z-dimension
- YLocal: plan FFT for y-dimension
- XLocal: plan FFT for x-dimension

The plans are cached in tf.fft_plans for efficient reuse.
"""
function plan_transposable_transforms!(tf)
    # Access the TransposableField's internal shapes
    arch = tf.buffers.architecture
    # Use the field's actual element type for FFTW plans (not always ComplexF64).
    # Extract T from the buffers rather than a type parameter to avoid load-order issues.
    T = eltype(tf.buffers.z_local_data !== nothing ? tf.buffers.z_local_data :
               tf.buffers.y_local_data !== nothing ? tf.buffers.y_local_data :
               tf.buffers.x_local_data)
    CT = T <: Complex ? T : Complex{T}

    # Plan for each layout based on which dimension is local
    ndim = length(tf.global_shape)

    # Each layout leaves exactly one axis local; plan the FFT along that axis.
    layout_axes = ndim >= 3 ? ((ZLocal, 3), (YLocal, 2), (XLocal, 1)) :
                  ndim == 2 ? ((YLocal, 2), (XLocal, 1)) :
                  ()

    for (layout, dim) in layout_axes
        haskey(tf.fft_plans, layout) && continue
        tf.fft_plans[layout] =
            _plan_local_axis_fft(arch, CT, tf.local_shapes[layout], dim)
    end

    return tf
end

"""
    setup_distributed_transforms!(dist::Distributor, domain::Domain)

Setup distributed transforms using TransposableField for GPU+MPI parallelism.
This is called when using GPU architecture with MPI.
"""
function setup_distributed_transforms!(dist::Distributor, domain::Domain)
    if !is_gpu(dist.architecture) || dist.size == 1
        return  # Only needed for distributed GPU
    end

    # Create DistributedGPUConfig if needed
    if dist.distributed_gpu_config === nothing
        gshape = global_shape(domain)
        config = DistributedGPUConfig(dist.comm, gshape;
                                       cuda_aware_mpi=check_cuda_aware_mpi())
        dist.distributed_gpu_config = config
    end

    @info "Distributed transforms setup for GPU+MPI" rank=dist.rank size=dist.size
end

