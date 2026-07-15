"""Transform setup and lookup helpers for nonlinear evaluation."""

"""Setup PencilFFT transforms for nonlinear term evaluation.

    Transforms are created lazily on first use and cached for reuse.
    This avoids pre-computing transforms for shapes that may never be used
    and ensures the correct shape is always available (including dealiased sizes
    like 3/2-rule padding which depend on the actual domain).

    The lazy creation path uses local FFTW plans (no MPI collectives), so it is
    safe to call from within the evaluation loop.
    """
function setup_nonlinear_transforms!(evaluator::NonlinearEvaluator)

    dist = evaluator.dist
    # All transform families are created on demand by `get_nonlinear_transform`
    # or by the distributed padded-dealiasing workspace builders.  Eagerly
    # populating the legacy 1-D-mesh catalogue retained >200 MiB of scratch
    # arrays per evaluator/rank, including unrelated 3-D sizes.
    @info "Nonlinear evaluator configured for lazy transform creation"
    @info "  Process mesh: $(dist.mesh)"
    @info "  Dealiasing factor: $(evaluator.dealiasing_factor)"
end

"""Setup PencilFFT transforms for specific 2D shape"""
function setup_pencil_transforms_for_shape!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int})

    dist = evaluator.dist
    shape_key = "$(shape[1])x$(shape[2])"

    # Create pencil configuration for this shape
    if haskey(evaluator.pencil_transforms, shape_key)
        return  # Already configured
    end

    try
        # Create pencil configuration for this shape
        config = PencilConfig(shape, dist.mesh, comm=dist.comm)

        # For serial execution, use simple arrays and FFTW plans
        if MPI.Comm_size(dist.comm) == 1
            # Serial execution - use regular arrays
            forward_data_1 = zeros(ComplexF64, shape...)
            forward_data_2 = zeros(ComplexF64, shape...)

            # Create FFTW plans
            fft_plan_1 = FFTW.plan_fft(forward_data_1)
            fft_plan_2 = FFTW.plan_fft(forward_data_2)

            _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
                PencilTransformConfig(
                    config,
                    forward_data_1,
                    forward_data_2,
                    fft_plan_1,
                    fft_plan_2,
                    shape,
                    true,
                    nothing,
                    nothing,
                ),
            )
        else
            # Parallel execution - use PencilArrays/PencilFFTs with PROPER TOPOLOGY
            # CRITICAL: Must use same decomposition convention as Distributor
            if !dist.use_pencil_arrays
                # GPU+MPI mode uses TransposableField with FIRST dimensions decomposed
                # NonlinearEvaluator with PencilArrays would create layout mismatch
                error("NonlinearEvaluator requires PencilArrays but Distributor has use_pencil_arrays=false (GPU+MPI mode). " *
                      "For GPU+MPI, use TransposableField-based nonlinear evaluation instead, or " *
                      "set use_pencil_arrays=true for CPU+MPI execution.")
            end

            try
                # CRITICAL FIX: Use Distributor's MPI topology and decomposition
                # to ensure consistency with field data layout
                ndim = length(shape)
                ndims_mesh = length(dist.mesh)

                # Decompose LAST dimensions (PencilArrays convention, matches dist.use_pencil_arrays=true)
                decomp_dims = if ndim >= ndims_mesh
                    ntuple(i -> ndim - ndims_mesh + i, ndims_mesh)
                else
                    ntuple(identity, ndim)
                end

                # Use existing MPI topology from Distributor if available
                if dist.mpi_topology !== nothing
                    pencil = PencilArrays.Pencil(dist.mpi_topology, shape, decomp_dims)
                else
                    # Keep the communicator owner on Distributor so `close(dist)`
                    # can release it; a temporary topology would leak its MPI
                    # Cartesian communicator and subcommunicators.
                    initialize_mpi_topology!(dist)
                    pencil = PencilArrays.Pencil(dist.mpi_topology, shape, decomp_dims)
                end

                # Create PencilArrays with proper pencil configuration
                forward_data_1 = PencilArrays.PencilArray{ComplexF64}(undef, pencil)
                forward_data_2 = PencilArrays.PencilArray{ComplexF64}(undef, pencil)

                # Create FFT plans for pencil arrays
                # PencilFFTPlan expects a tuple of transforms, one per dimension
                ndims_shape = length(shape)
                transforms = ntuple(_ -> PencilFFTs.Transforms.FFT(), ndims_shape)
                fft_plan_1 = PencilFFTs.PencilFFTPlan(pencil, transforms)
                fft_plan_2 = PencilFFTs.PencilFFTPlan(pencil, transforms)

                _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
                    PencilTransformConfig(
                        config,
                        forward_data_1,
                        forward_data_2,
                        fft_plan_1,
                        fft_plan_2,
                        shape,
                        false,
                        pencil,
                        decomp_dims,
                    ),
                )
            catch pe
                # CRITICAL: In MPI mode, falling back to serial FFTW produces incorrect results
                if MPI.Comm_size(dist.comm) > 1
                    @error "PencilArrays setup failed in MPI mode - local FFTW will produce incorrect results" exception=pe
                    error("Nonlinear evaluator requires PencilArrays for MPI execution. " *
                          "Check your PencilArrays/PencilFFTs installation.")
                end

                # Serial fallback is only safe for single process
                @warn "PencilArrays setup failed, falling back to serial FFTW" exception=pe
                forward_data_1 = zeros(ComplexF64, shape...)
                forward_data_2 = zeros(ComplexF64, shape...)
                fft_plan_1 = FFTW.plan_fft(forward_data_1)
                fft_plan_2 = FFTW.plan_fft(forward_data_2)

                _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
                    PencilTransformConfig(
                        config,
                        forward_data_1,
                        forward_data_2,
                        fft_plan_1,
                        fft_plan_2,
                        shape,
                        true,
                        nothing,
                        nothing,
                    ),
                )
            end
        end

        @debug "Created FFT transforms for shape $shape"

    catch e
        @error "Failed to create FFT transforms for shape $shape: $e\n" *
               "Dealiasing will be disabled for this shape — results may contain aliasing errors."
    end
end

"""
    Setup 1D FFT transforms for nonlinear term evaluation.

    This is the fallback for when only 1D domain decomposition is used
    (single process or 1D process mesh). Uses FFTW directly instead of
    PencilFFTs since there's no need for pencil transposes.

    For 1D parallelization:
    - The domain is split along one dimension only
    - FFTs along the local (non-decomposed) dimensions use FFTW
    - FFTs along the decomposed dimension require MPI communication

    This setup creates:
    - Local FFTW plans for each common array size
    - Scratch arrays for in-place transforms
    - Dealiased array configurations
    """
function setup_1d_nonlinear_transforms!(evaluator::NonlinearEvaluator)

    dist = evaluator.dist
    @info "Setting up 1D nonlinear transforms"

    # Common 1D sizes for spectral methods
    common_1d_sizes = [32, 64, 128, 256, 512, 1024]

    # Common 2D shapes (for 2D problems with 1D decomposition)
    common_2d_shapes = [(64, 64), (128, 64), (128, 128), (256, 128), (256, 256), (512, 256)]

    # Common 3D shapes (for 3D problems with 1D decomposition)
    common_3d_shapes = [(64, 64, 64), (128, 64, 64), (128, 128, 64), (128, 128, 128)]

    # Setup 1D transforms
    for n in common_1d_sizes
        setup_1d_fftw_plans!(evaluator, n)
    end

    # Setup 2D transforms (1D decomposition means one axis is fully local)
    for shape in common_2d_shapes
        setup_2d_fftw_plans!(evaluator, shape)
    end

    # Setup 3D transforms
    for shape in common_3d_shapes
        setup_3d_fftw_plans!(evaluator, shape)
    end

    @info "1D nonlinear transform setup complete"
    @info "  MPI size: $(dist.size)"
    @info "  Dealiasing factor: $(evaluator.dealiasing_factor)"
end

"""Setup FFTW plans for 1D transforms of size n."""
function setup_1d_fftw_plans!(evaluator::NonlinearEvaluator, n::Int)

    shape_key = "1d_$n"

    if haskey(evaluator.pencil_transforms, shape_key)
        return  # Already configured
    end

    try
        # Dealiased size using 3/2 rule
        n_dealias = ceil(Int, n * evaluator.dealiasing_factor)

        # Create scratch arrays
        scratch_real = zeros(Float64, n_dealias)
        scratch_complex = zeros(ComplexF64, div(n_dealias, 2) + 1)

        # Create FFTW plans
        # Real-to-complex forward transform
        forward_plan = FFTW.plan_rfft(scratch_real; flags=FFTW.MEASURE)

        # Complex-to-real backward transform
        backward_plan = FFTW.plan_brfft(scratch_complex, n_dealias; flags=FFTW.MEASURE)

        _cache_shape_transform!(evaluator.pencil_transforms, (n,), shape_key,
            FFTWTransformConfig(
                :fftw_1d,
                (n,),
                (n_dealias,),
                forward_plan,
                backward_plan,
                scratch_real,
                scratch_complex,
            ),
        )

        @debug "Created 1D FFTW plans for size $n (dealiased: $n_dealias)"

    catch e
        @warn "Failed to create 1D FFTW plans for size $n: $e"
    end
end

"""Setup FFTW plans for 2D transforms."""
function setup_2d_fftw_plans!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int})

    shape_key = "2d_$(shape[1])x$(shape[2])"

    if haskey(evaluator.pencil_transforms, shape_key)
        return  # Already configured
    end

    try
        # Dealiased sizes
        nx_dealias = ceil(Int, shape[1] * evaluator.dealiasing_factor)
        ny_dealias = ceil(Int, shape[2] * evaluator.dealiasing_factor)
        dealias_shape = (nx_dealias, ny_dealias)

        # Create scratch arrays
        scratch_real = zeros(Float64, dealias_shape)
        scratch_complex = zeros(ComplexF64, div(nx_dealias, 2) + 1, ny_dealias)

        # Create FFTW plans for 2D real-to-complex transforms
        forward_plan = FFTW.plan_rfft(scratch_real; flags=FFTW.MEASURE)
        backward_plan = FFTW.plan_brfft(scratch_complex, nx_dealias; flags=FFTW.MEASURE)

        _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
            FFTWTransformConfig(
                :fftw_2d,
                shape,
                dealias_shape,
                forward_plan,
                backward_plan,
                scratch_real,
                scratch_complex,
            ),
        )

        @debug "Created 2D FFTW plans for shape $shape (dealiased: $dealias_shape)"

    catch e
        @warn "Failed to create 2D FFTW plans for shape $shape: $e"
    end
end

"""Setup FFTW plans for 3D transforms."""
function setup_3d_fftw_plans!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int, Int})

    shape_key = "3d_$(shape[1])x$(shape[2])x$(shape[3])"

    if haskey(evaluator.pencil_transforms, shape_key)
        return  # Already configured
    end

    try
        # Dealiased sizes
        nx_dealias = ceil(Int, shape[1] * evaluator.dealiasing_factor)
        ny_dealias = ceil(Int, shape[2] * evaluator.dealiasing_factor)
        nz_dealias = ceil(Int, shape[3] * evaluator.dealiasing_factor)
        dealias_shape = (nx_dealias, ny_dealias, nz_dealias)

        # Create scratch arrays
        scratch_real = zeros(Float64, dealias_shape)
        scratch_complex = zeros(ComplexF64, div(nx_dealias, 2) + 1, ny_dealias, nz_dealias)

        # Create FFTW plans for 3D real-to-complex transforms
        forward_plan = FFTW.plan_rfft(scratch_real; flags=FFTW.MEASURE)
        backward_plan = FFTW.plan_brfft(scratch_complex, nx_dealias; flags=FFTW.MEASURE)

        _cache_shape_transform!(evaluator.pencil_transforms, shape, shape_key,
            FFTWTransformConfig(
                :fftw_3d,
                shape,
                dealias_shape,
                forward_plan,
                backward_plan,
                scratch_real,
                scratch_complex,
            ),
        )

        @debug "Created 3D FFTW plans for shape $shape (dealiased: $dealias_shape)"

    catch e
        @warn "Failed to create 3D FFTW plans for shape $shape: $e"
    end
end

"""
    Get the appropriate transform configuration for a given shape.

    Automatically selects between PencilFFT (for multi-D parallelization)
    and FFTW (for 1D parallelization or serial) based on what's available.
    """
function get_nonlinear_transform(evaluator::NonlinearEvaluator, shape::Tuple)
    ndims_shape = length(shape)

    # Try to find exact match first
    if ndims_shape == 1
        shape_key = "1d_$(shape[1])"
    elseif ndims_shape == 2
        shape_key = "2d_$(shape[1])x$(shape[2])"
        # Also try PencilFFT format
        pencil_key = "$(shape[1])x$(shape[2])"
    elseif ndims_shape == 3
        shape_key = "3d_$(shape[1])x$(shape[2])x$(shape[3])"
        pencil_key = "$(shape[1])x$(shape[2])x$(shape[3])"
    else
        @warn "Unsupported shape dimension: $ndims_shape"
        return nothing
    end

    # Check for FFTW-based transform
    if haskey(evaluator.pencil_transforms, shape_key)
        return evaluator.pencil_transforms[shape_key]
    end

    # Check for PencilFFT-based transform (2D/3D only)
    if ndims_shape >= 2 && haskey(evaluator.pencil_transforms, pencil_key)
        return evaluator.pencil_transforms[pencil_key]
    end

    # No exact match - try to create one on the fly
    @debug "Creating transform on-the-fly for shape $shape"

    if ndims_shape == 1
        setup_1d_fftw_plans!(evaluator, shape[1])
    elseif ndims_shape == 2
        setup_2d_fftw_plans!(evaluator, shape)
    elseif ndims_shape == 3
        setup_3d_fftw_plans!(evaluator, shape)
    end

    # Return the newly created transform
    return get(evaluator.pencil_transforms, shape_key, nothing)
end
