"""
    Transform Transposable - TransposableField transform planning

This file contains functions for planning FFT transforms for
TransposableField layouts.
"""

# ============================================================================
# TransposableField Transform Planning
# ============================================================================

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

    if ndim >= 3
        # ZLocal layout: z-dimension is local, plan FFT in z
        z_shape = tf.local_shapes[ZLocal]
        if !haskey(tf.fft_plans, ZLocal)
            if is_gpu(arch)
                # GPU plans will be created by TarangCUDAExt
                tf.fft_plans[ZLocal] = :gpu_pending
            else
                # CPU FFTW plan for z-dimension (dim 3)
                dummy = zeros(CT, z_shape...)
                tf.fft_plans[ZLocal] = FFTW.plan_fft(dummy, 3; flags=FFTW.MEASURE)
            end
        end

        # YLocal layout: y-dimension is local, plan FFT in y
        y_shape = tf.local_shapes[YLocal]
        if !haskey(tf.fft_plans, YLocal)
            if is_gpu(arch)
                tf.fft_plans[YLocal] = :gpu_pending
            else
                dummy = zeros(CT, y_shape...)
                tf.fft_plans[YLocal] = FFTW.plan_fft(dummy, 2; flags=FFTW.MEASURE)
            end
        end

        # XLocal layout: x-dimension is local, plan FFT in x
        x_shape = tf.local_shapes[XLocal]
        if !haskey(tf.fft_plans, XLocal)
            if is_gpu(arch)
                tf.fft_plans[XLocal] = :gpu_pending
            else
                dummy = zeros(CT, x_shape...)
                tf.fft_plans[XLocal] = FFTW.plan_fft(dummy, 1; flags=FFTW.MEASURE)
            end
        end

    elseif ndim == 2
        # 2D case: plan FFT for x and y dimensions
        y_shape = tf.local_shapes[YLocal]
        x_shape = tf.local_shapes[XLocal]

        if !haskey(tf.fft_plans, YLocal)
            if is_gpu(arch)
                tf.fft_plans[YLocal] = :gpu_pending
            else
                dummy = zeros(CT, y_shape...)
                tf.fft_plans[YLocal] = FFTW.plan_fft(dummy, 2; flags=FFTW.MEASURE)
            end
        end

        if !haskey(tf.fft_plans, XLocal)
            if is_gpu(arch)
                tf.fft_plans[XLocal] = :gpu_pending
            else
                dummy = zeros(CT, x_shape...)
                tf.fft_plans[XLocal] = FFTW.plan_fft(dummy, 1; flags=FFTW.MEASURE)
            end
        end
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

