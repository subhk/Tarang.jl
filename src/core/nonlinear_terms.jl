"""
Nonlinear term evaluation using PencilArrays and PencilFFTs

This module implements efficient evaluation of nonlinear terms in spectral methods,
designed for Julia with PencilArrays/PencilFFTs.
Supports both 2D and 3D parallelization with proper dealiasing.

Key features:
- Transform-based multiplication for nonlinear terms (u·∇u, etc.)
- Automatic dealiasing using 3/2 rule
- MPI parallelization through PencilArrays
- Efficient memory management and reuse
- Support for various nonlinear operators
"""

using PencilArrays
using PencilFFTs
using MPI
using LinearAlgebra

# CPU-only (GPU support removed)

# Performance monitoring (defined first as it's used by NonlinearEvaluator)
mutable struct NonlinearPerformanceStats
    total_evaluations::Int
    total_time::Float64
    dealiasing_time::Float64
    transform_time::Float64

    function NonlinearPerformanceStats()
        new(0, 0.0, 0.0, 0.0)
    end
end

# Nonlinear operator types
abstract type NonlinearOperator <: Operator end

struct AdvectionOperator <: NonlinearOperator
    velocity::VectorField
    scalar::ScalarField
    name::String
    
    function AdvectionOperator(velocity::VectorField, scalar::ScalarField, name::String="advection")
        new(velocity, scalar, name)
    end
end

struct NonlinearAdvectionOperator <: NonlinearOperator
    velocity::VectorField
    name::String
    
    function NonlinearAdvectionOperator(velocity::VectorField, name::String="nonlinear_advection")
        new(velocity, name)
    end
end

struct ConvectiveOperator <: NonlinearOperator
    field1::Union{ScalarField, VectorField}
    field2::Union{ScalarField, VectorField}
    operation::Symbol  # :multiply, :dot_product, :cross_product
    name::String
    
    function ConvectiveOperator(field1, field2, operation::Symbol, name::String="convective")
        new(field1, field2, operation, name)
    end
end

# Nonlinear evaluation engine
mutable struct NonlinearEvaluator
    dist::Distributor
    pencil_transforms::Dict{String, Any}
    dealiasing_factor::Float64
    temp_fields::Dict{String, Any}
    memory_pool::Vector{PencilArray}
    scratch_arrays::Vector{AbstractArray}
    performance_stats::NonlinearPerformanceStats

    function NonlinearEvaluator(dist::Distributor; dealiasing_factor::Float64=3.0/2.0)
        evaluator = new(dist, Dict{String, Any}(), dealiasing_factor, Dict{String, Any}(), PencilArray[],
                       AbstractArray[], NonlinearPerformanceStats())
        setup_nonlinear_transforms!(evaluator)
        return evaluator
    end
end

function setup_nonlinear_transforms!(evaluator::NonlinearEvaluator)
    """Setup PencilFFT transforms for nonlinear term evaluation"""
    
    dist = evaluator.dist
    
    # For 2D problems with both horizontal and vertical parallelization
    if length(dist.mesh) >= 2
        @info "Setting up nonlinear transforms for 2D+ parallelization"
        
        # Create transform configurations for different domain sizes
        # This enables flexible handling of different problem sizes
        common_shapes = [(64, 64), (128, 128), (256, 128), (512, 256), (256, 64), (1024, 256)]
        
        for shape in common_shapes
            setup_pencil_transforms_for_shape!(evaluator, shape)
        end
        
        @info "Nonlinear evaluator configured for 2D+ parallelization"
        @info "  Process mesh: $(dist.mesh)"
        @info "  Dealiasing factor: $(evaluator.dealiasing_factor)"
        @info "  Precomputed shapes: $common_shapes"
    else
        @info "Setting up nonlinear transforms for 1D parallelization (fallback)"
        # 1D parallelization fallback
        setup_1d_nonlinear_transforms!(evaluator)
    end
end

function setup_pencil_transforms_for_shape!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int})
    """Setup PencilFFT transforms for specific 2D shape"""
    
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

            evaluator.pencil_transforms[shape_key] = Dict(
                "config" => config,
                "forward_pencil_1" => forward_data_1,
                "forward_pencil_2" => forward_data_2,
                "fft_plan_1" => fft_plan_1,
                "fft_plan_2" => fft_plan_2,
                "shape" => shape,
                "serial" => true
            )
        else
            # Parallel execution - use PencilArrays/PencilFFTs
            try
                # Create PencilArray-based transforms
                forward_data_1 = PencilArrays.PencilArray{ComplexF64}(undef, shape, config.comm)
                forward_data_2 = PencilArrays.PencilArray{ComplexF64}(undef, shape, config.comm)

                # Create FFT plans for pencil arrays
                fft_plan_1 = PencilFFTs.PencilFFTPlan(forward_data_1, PencilFFTs.Transforms.FFT())
                fft_plan_2 = PencilFFTs.PencilFFTPlan(forward_data_2, PencilFFTs.Transforms.FFT())

                evaluator.pencil_transforms[shape_key] = Dict(
                    "config" => config,
                    "forward_pencil_1" => forward_data_1,
                    "forward_pencil_2" => forward_data_2,
                    "fft_plan_1" => fft_plan_1,
                    "fft_plan_2" => fft_plan_2,
                    "shape" => shape,
                    "serial" => false
                )
            catch pe
                # Fallback to serial FFTW if PencilArrays fails
                @warn "PencilArrays setup failed, falling back to serial FFTW" exception=pe
                forward_data_1 = zeros(ComplexF64, shape...)
                forward_data_2 = zeros(ComplexF64, shape...)
                fft_plan_1 = FFTW.plan_fft(forward_data_1)
                fft_plan_2 = FFTW.plan_fft(forward_data_2)

                evaluator.pencil_transforms[shape_key] = Dict(
                    "config" => config,
                    "forward_pencil_1" => forward_data_1,
                    "forward_pencil_2" => forward_data_2,
                    "fft_plan_1" => fft_plan_1,
                    "fft_plan_2" => fft_plan_2,
                    "shape" => shape,
                    "serial" => true
                )
            end
        end

        @debug "Created FFT transforms for shape $shape"

    catch e
        @warn "Failed to create FFT transforms for shape $shape: $e"
    end
end

function setup_1d_nonlinear_transforms!(evaluator::NonlinearEvaluator)
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

function setup_1d_fftw_plans!(evaluator::NonlinearEvaluator, n::Int)
    """Setup FFTW plans for 1D transforms of size n."""

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

        evaluator.pencil_transforms[shape_key] = Dict(
            "type" => :fftw_1d,
            "size" => n,
            "dealiased_size" => n_dealias,
            "forward_plan" => forward_plan,
            "backward_plan" => backward_plan,
            "scratch_real" => scratch_real,
            "scratch_complex" => scratch_complex
        )

        @debug "Created 1D FFTW plans for size $n (dealiased: $n_dealias)"

    catch e
        @warn "Failed to create 1D FFTW plans for size $n: $e"
    end
end

function setup_2d_fftw_plans!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int})
    """Setup FFTW plans for 2D transforms."""

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

        evaluator.pencil_transforms[shape_key] = Dict(
            "type" => :fftw_2d,
            "shape" => shape,
            "dealiased_shape" => dealias_shape,
            "forward_plan" => forward_plan,
            "backward_plan" => backward_plan,
            "scratch_real" => scratch_real,
            "scratch_complex" => scratch_complex
        )

        @debug "Created 2D FFTW plans for shape $shape (dealiased: $dealias_shape)"

    catch e
        @warn "Failed to create 2D FFTW plans for shape $shape: $e"
    end
end

function setup_3d_fftw_plans!(evaluator::NonlinearEvaluator, shape::Tuple{Int, Int, Int})
    """Setup FFTW plans for 3D transforms."""

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

        evaluator.pencil_transforms[shape_key] = Dict(
            "type" => :fftw_3d,
            "shape" => shape,
            "dealiased_shape" => dealias_shape,
            "forward_plan" => forward_plan,
            "backward_plan" => backward_plan,
            "scratch_real" => scratch_real,
            "scratch_complex" => scratch_complex
        )

        @debug "Created 3D FFTW plans for shape $shape (dealiased: $dealias_shape)"

    catch e
        @warn "Failed to create 3D FFTW plans for shape $shape: $e"
    end
end

function get_nonlinear_transform(evaluator::NonlinearEvaluator, shape::Tuple)
    """
    Get the appropriate transform configuration for a given shape.

    Automatically selects between PencilFFT (for multi-D parallelization)
    and FFTW (for 1D parallelization or serial) based on what's available.
    """
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

# Main nonlinear evaluation functions
function evaluate_nonlinear_term(op::AdvectionOperator, layout::Symbol=:g)
    """Evaluate u·∇φ nonlinear term using transform method"""
    
    velocity = op.velocity
    scalar = op.scalar
    
    # Get distributor for transform operations
    dist = velocity.dist
    
    # Create nonlinear evaluator if not exists
    if !hasfield(typeof(dist), :nonlinear_evaluator)
        evaluator = NonlinearEvaluator(dist)
        dist.nonlinear_evaluator = evaluator
    else
        evaluator = dist.nonlinear_evaluator
    end
    
    # Compute gradient of scalar field: ∇φ
    grad_scalar = evaluate_gradient(Gradient(scalar, dist.coordsys), :g)
    
    # Evaluate u·∇φ = u_x ∂φ/∂x + u_y ∂φ/∂y (+ u_z ∂φ/∂z in 3D)
    result = ScalarField(dist, "$(op.name)_$(scalar.name)", scalar.bases, scalar.dtype)
    ensure_layout!(result, :g)
    fill!(result["g"], 0.0)
    
    # Sum velocity components times gradient components
    for i in 1:length(velocity.components)
        # Multiply u_i * (∂φ/∂x_i) using transform-based multiplication
        product = evaluate_transform_multiply(velocity.components[i], grad_scalar.components[i], evaluator)
        result = result + product
    end
    
    return result
end

function evaluate_nonlinear_term(op::NonlinearAdvectionOperator, layout::Symbol=:g)
    """Evaluate (u·∇)u nonlinear momentum term"""
    
    velocity = op.velocity
    dist = velocity.dist
    
    # Create nonlinear evaluator if needed
    if !hasfield(typeof(dist), :nonlinear_evaluator)
        evaluator = NonlinearEvaluator(dist)
        dist.nonlinear_evaluator = evaluator
    else
        evaluator = dist.nonlinear_evaluator
    end
    
    # Result is a vector field
    result = VectorField(dist, dist.coordsys, "$(op.name)_$(velocity.name)", velocity.bases, velocity.dtype)
    
    # For each component: (u·∇)u_i = u_j ∂u_i/∂x_j (summed over j)
    for i in 1:length(velocity.components)
        component_result = ScalarField(dist, "nl_$(velocity.name)_$i", velocity.bases, velocity.dtype)
        ensure_layout!(component_result, :g)
        fill!(component_result["g"], 0.0)
        
        # Sum over all spatial directions
        for j in 1:length(velocity.components)
            # Compute ∂u_i/∂x_j
            coord = dist.coordsys[j]
            du_i_dx_j = evaluate_differentiate(Differentiate(velocity.components[i], coord, 1), :g)
            
            # Multiply u_j * (∂u_i/∂x_j) using efficient transform method
            product = evaluate_transform_multiply(velocity.components[j], du_i_dx_j, evaluator)
            component_result = component_result + product
        end
        
        result.components[i] = component_result
    end
    
    return result
end

function evaluate_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator)
    """Efficiently multiply two fields using transform method and proper dealiasing

    This follows the standard spectral method approach:
    1. Transform both fields to grid space (if not already)
    2. Pointwise multiply in grid space
    3. Transform product back to spectral space
    4. Apply dealiasing to remove aliasing errors

    Uses efficient field multiplication operations.
    """

    start_time = time()

    # Ensure both fields are in grid space for multiplication
    ensure_layout!(field1, :g)
    ensure_layout!(field2, :g)

    # Check compatibility
    if field1.bases != field2.bases
        throw(ArgumentError("Cannot multiply fields with different bases"))
    end

    # Use the field multiplication directly
    result = field1 * field2

    # Apply dealiasing if requested
    if evaluator.dealiasing_factor > 1.0
        apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
    end

    # Update performance statistics
    evaluator.performance_stats.total_evaluations += 1
    evaluator.performance_stats.total_time += (time() - start_time)

    return result
end

function evaluate_2d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)
    """2D transform-based multiplication using PencilFFTs"""
    
    # Create result field
    result = ScalarField(field1.dist, "product_$(field1.name)_$(field2.name)", field1.bases, field1.dtype)
    ensure_layout!(result, :g)
    
    # Try to find matching transform configuration
    shape_key = "$(shape[1])x$(shape[2])"
    
    if haskey(evaluator.pencil_transforms, shape_key)
        # Use precomputed PencilFFT transforms
        transform_info = evaluator.pencil_transforms[shape_key]
        
        # Get data in appropriate format for PencilArrays
        data1 = get_pencil_compatible_data(field1, transform_info["config"])
        data2 = get_pencil_compatible_data(field2, transform_info["config"])
        
        # Pointwise multiplication in grid space
        # This is where the actual nonlinear interaction happens
        result_data = data1 .* data2
        
        # Apply dealiasing by transforming to spectral space and back
        if evaluator.dealiasing_factor > 1.0
            result_data = apply_2d_dealiasing(result_data, transform_info, evaluator.dealiasing_factor)
        end
        
        # Set result data
        set_pencil_compatible_data!(result, result_data, transform_info["config"])
        
    else
        # Fallback to direct multiplication without PencilFFT optimization
        @debug "No precomputed transforms for shape $shape, using fallback"
        result["g"] .= field1["g"] .* field2["g"]
        
        # Apply basic dealiasing if possible
        if evaluator.dealiasing_factor > 1.0
            apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
        end
    end
    
    return result
end

function evaluate_3d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)
    """3D transform-based multiplication using 3D PencilFFTs"""
    
    result = ScalarField(field1.dist, "product_$(field1.name)_$(field2.name)", field1.bases, field1.dtype)
    ensure_layout!(result, :g)
    
    # For 3D, we need more sophisticated pencil management
    # This would involve 3D pencil decomposition across all three spatial dimensions
    
    if length(evaluator.dist.mesh) >= 3
        # Use 3D PencilFFT approach
        @debug "Using 3D PencilFFT multiplication for shape $shape"
        
        # Direct multiplication for now - would implement full 3D PencilFFT logic
        result["g"] .= field1["g"] .* field2["g"]
        
        # Apply 3D dealiasing
        if evaluator.dealiasing_factor > 1.0
            apply_3d_dealiasing!(result, evaluator.dealiasing_factor)
        end
        
    else
        # Fallback for insufficient parallelization
        result["g"] .= field1["g"] .* field2["g"]
        
        if evaluator.dealiasing_factor > 1.0
            apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
        end
    end
    
    return result
end

function evaluate_fallback_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator)
    """Fallback multiplication for unsupported dimensions"""
    
    result = ScalarField(field1.dist, "product_$(field1.name)_$(field2.name)", field1.bases, field1.dtype)
    ensure_layout!(result, :g)
    
    # Simple pointwise multiplication
    result["g"] .= field1["g"] .* field2["g"]
    
    # Apply dealiasing if requested
    if evaluator.dealiasing_factor > 1.0
        apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
    end
    
    return result
end

# Dealiasing functions
function apply_2d_dealiasing(data::PencilArray, transform_info::Dict, dealiasing_factor::Float64)
    """Apply 2D dealiasing using PencilFFT transforms"""
    
    # Transform to spectral space
    fft_plan = transform_info["fft_plan_1"]
    spectral_data = fft_plan * data
    
    # Zero out high-frequency modes (3/2 rule)
    shape = transform_info["shape"]
    cutoff_x = Int(floor(shape[1] / dealiasing_factor))
    cutoff_y = Int(floor(shape[2] / dealiasing_factor))
    
    # Apply dealiasing cutoff
    apply_spectral_cutoff!(spectral_data, (cutoff_x, cutoff_y))
    
    # Transform back to grid space
    dealiased_data = inv(fft_plan) * spectral_data
    
    return dealiased_data
end

function apply_3d_dealiasing!(field::ScalarField, dealiasing_factor::Float64)
    """Apply 3D dealiasing"""
    
    # Transform to coefficient space
    ensure_layout!(field, :c)
    
    # Apply cutoff in each direction based on dealiasing factor
    for (i, basis) in enumerate(field.bases)
        if isa(basis, Union{RealFourier, ComplexFourier})
            cutoff = Int(floor(basis.meta.size / dealiasing_factor))
            # Zero high-frequency modes along axis i
            apply_1d_spectral_cutoff!(field.data_c, i, cutoff)
        end
    end
    
    # Transform back to grid space
    backward_transform!(field)
end

function apply_basic_dealiasing!(field::ScalarField, dealiasing_factor::Float64)
    """Basic dealiasing for fields without PencilFFT support"""
    
    # Transform to spectral space
    forward_transform!(field)
    
    # Apply cutoff
    for (i, basis) in enumerate(field.bases)
        if isa(basis, Union{RealFourier, ComplexFourier})
            cutoff = Int(floor(basis.meta.size / dealiasing_factor))
            apply_1d_spectral_cutoff!(field.data_c, i, cutoff)
        end
    end
    
    # Transform back to grid space
    backward_transform!(field)
end


function apply_spectral_cutoff!(data::AbstractArray, cutoffs::Tuple)
    """
    Apply spectral cutoff to remove high-frequency modes (dealiasing).

    For spectral data stored in standard FFT layout:
    - Positive frequencies: indices 1 to N/2+1
    - Negative frequencies: indices N/2+2 to N (for complex FFT)

    This function zeros out modes beyond the cutoff wavenumber in each dimension.
    Used for dealiasing in nonlinear term evaluation.

    Arguments:
    - data: Complex spectral coefficient array
    - cutoffs: Tuple of cutoff wavenumbers for each dimension

    The cutoff is applied symmetrically: modes with |k| > cutoff are zeroed.
    """
    ndims_data = ndims(data)
    shape = size(data)

    if ndims_data == 1
        apply_1d_spectral_cutoff!(data, 1, cutoffs[1])
    elseif ndims_data == 2
        apply_2d_spectral_cutoff!(data, cutoffs)
    elseif ndims_data == 3
        apply_3d_spectral_cutoff!(data, cutoffs)
    else
        # General N-dimensional case
        apply_nd_spectral_cutoff!(data, cutoffs)
    end
end

function apply_1d_spectral_cutoff!(data::AbstractVector, axis::Int, cutoff::Int)
    """
    Apply 1D spectral cutoff along a vector.

    For FFT layout with N points:
    - Index 1: k=0 (DC component)
    - Indices 2 to N/2+1: positive frequencies k=1 to N/2
    - Indices N/2+2 to N: negative frequencies k=-(N/2-1) to -1

    Modes with |k| > cutoff are set to zero.
    """
    n = length(data)
    if cutoff >= div(n, 2)
        return  # No cutoff needed
    end

    # Zero positive high frequencies: indices cutoff+2 to N/2+1
    # (index 1 is k=0, index 2 is k=1, ..., index cutoff+1 is k=cutoff)
    half_n = div(n, 2)
    for i in (cutoff + 2):(half_n + 1)
        if i <= n
            data[i] = zero(eltype(data))
        end
    end

    # Zero negative high frequencies: indices N/2+2 to N-cutoff
    # Negative frequencies are stored in reverse order at the end
    for i in (half_n + 2):(n - cutoff)
        if i <= n
            data[i] = zero(eltype(data))
        end
    end
end

function apply_1d_spectral_cutoff!(data::AbstractArray, axis::Int, cutoff::Int)
    """
    Apply 1D spectral cutoff along specified axis of multi-dimensional array.
    """
    shape = size(data)
    n = shape[axis]

    if cutoff >= div(n, 2)
        return  # No cutoff needed
    end

    half_n = div(n, 2)

    # Create index ranges for slicing
    # Zero out positive high frequencies
    for k in (cutoff + 2):(half_n + 1)
        if k <= n
            indices = ntuple(ndims(data)) do d
                d == axis ? k : Colon()
            end
            data[indices...] .= zero(eltype(data))
        end
    end

    # Zero out negative high frequencies
    for k in (half_n + 2):(n - cutoff)
        if k <= n
            indices = ntuple(ndims(data)) do d
                d == axis ? k : Colon()
            end
            data[indices...] .= zero(eltype(data))
        end
    end
end

function apply_2d_spectral_cutoff!(data::AbstractMatrix, cutoffs::Tuple)
    """
    Apply 2D spectral cutoff for dealiasing.

    Zeros out modes where |kx| > cutoffs[1] or |ky| > cutoffs[2].
    """
    nx, ny = size(data)
    kx_cut = min(cutoffs[1], div(nx, 2))
    ky_cut = length(cutoffs) >= 2 ? min(cutoffs[2], div(ny, 2)) : div(ny, 2)

    half_nx = div(nx, 2)
    half_ny = div(ny, 2)

    for j in 1:ny
        # Determine if this y-frequency is within cutoff
        ky = j <= half_ny + 1 ? j - 1 : j - ny - 1
        y_in_range = abs(ky) <= ky_cut

        for i in 1:nx
            # Determine if this x-frequency is within cutoff
            kx = i <= half_nx + 1 ? i - 1 : i - nx - 1
            x_in_range = abs(kx) <= kx_cut

            # Zero out if either frequency is outside cutoff
            if !x_in_range || !y_in_range
                data[i, j] = zero(eltype(data))
            end
        end
    end
end

function apply_3d_spectral_cutoff!(data::AbstractArray{T, 3}, cutoffs::Tuple) where T
    """
    Apply 3D spectral cutoff for dealiasing.

    Zeros out modes where |kx| > cutoffs[1], |ky| > cutoffs[2], or |kz| > cutoffs[3].
    """
    nx, ny, nz = size(data)
    kx_cut = min(cutoffs[1], div(nx, 2))
    ky_cut = length(cutoffs) >= 2 ? min(cutoffs[2], div(ny, 2)) : div(ny, 2)
    kz_cut = length(cutoffs) >= 3 ? min(cutoffs[3], div(nz, 2)) : div(nz, 2)

    half_nx = div(nx, 2)
    half_ny = div(ny, 2)
    half_nz = div(nz, 2)

    for k in 1:nz
        kz = k <= half_nz + 1 ? k - 1 : k - nz - 1
        z_in_range = abs(kz) <= kz_cut

        for j in 1:ny
            ky = j <= half_ny + 1 ? j - 1 : j - ny - 1
            y_in_range = abs(ky) <= ky_cut

            for i in 1:nx
                kx = i <= half_nx + 1 ? i - 1 : i - nx - 1
                x_in_range = abs(kx) <= kx_cut

                if !x_in_range || !y_in_range || !z_in_range
                    data[i, j, k] = zero(T)
                end
            end
        end
    end
end

function apply_nd_spectral_cutoff!(data::AbstractArray, cutoffs::Tuple)
    """
    Apply N-dimensional spectral cutoff (general case).
    """
    shape = size(data)
    ndims_data = ndims(data)

    # Extend cutoffs to match dimensions
    actual_cutoffs = ntuple(ndims_data) do d
        d <= length(cutoffs) ? min(cutoffs[d], div(shape[d], 2)) : div(shape[d], 2)
    end

    half_shape = div.(shape, 2)

    for I in CartesianIndices(data)
        # Check if any frequency is outside cutoff
        outside_cutoff = false

        for d in 1:ndims_data
            idx = I[d]
            n = shape[d]
            half_n = half_shape[d]

            # Convert index to wavenumber
            k = idx <= half_n + 1 ? idx - 1 : idx - n - 1

            if abs(k) > actual_cutoffs[d]
                outside_cutoff = true
                break
            end
        end

        if outside_cutoff
            data[I] = zero(eltype(data))
        end
    end
end

function apply_spherical_spectral_cutoff!(data::AbstractArray, k_max::Int)
    """
    Apply spherical spectral cutoff: zero modes with |k| > k_max.

    This is useful for isotropic dealiasing where the cutoff is based
    on the magnitude of the wavevector rather than individual components.

    |k|² = kx² + ky² + kz² (for 3D)
    """
    shape = size(data)
    ndims_data = ndims(data)
    half_shape = div.(shape, 2)
    k_max_sq = k_max^2

    for I in CartesianIndices(data)
        # Compute |k|²
        k_sq = 0
        for d in 1:ndims_data
            idx = I[d]
            n = shape[d]
            half_n = half_shape[d]
            k = idx <= half_n + 1 ? idx - 1 : idx - n - 1
            k_sq += k^2
        end

        if k_sq > k_max_sq
            data[I] = zero(eltype(data))
        end
    end
end

function get_dealiasing_cutoffs(shape::Tuple, dealiasing_factor::Float64=1.5)
    """
    Compute spectral cutoffs for dealiasing.

    For the 3/2 rule (dealiasing_factor=1.5):
    cutoff = N / dealiasing_factor = 2N/3

    This ensures that when two fields with max wavenumber k_max are multiplied,
    the product (with max wavenumber 2*k_max) doesn't alias back into the
    resolved modes.
    """
    return tuple([floor(Int, n / dealiasing_factor) for n in shape]...)
end

# Utility functions for PencilArray compatibility
function get_pencil_compatible_data(field::ScalarField, config::PencilConfig)
    """
    Convert field data to PencilArray format compatible with the given PencilConfig.

    This function ensures that the field's data is:
    1. In grid space layout (for nonlinear operations)
    2. Compatible with the PencilConfig's global shape
    3. Uses the correct data type
    4. Properly distributed according to the mesh configuration

    Returns the field's grid space data as a PencilArray or compatible array.
    """

    # Ensure field is in grid space layout for nonlinear operations
    ensure_layout!(field, :g)

    # Verify field has allocated data
    if field.data_g === nothing
        throw(ArgumentError("Field $(field.name) has no grid space data allocated"))
    end

    field_pencil = field.data_g
    field_shape = size(field_pencil)

    # Verify shape compatibility
    # The field's local shape should be consistent with the global shape and mesh decomposition
    if !is_shape_compatible(field_shape, config.global_shape, config.mesh, config.comm)
        @warn "Shape mismatch: field local shape $(field_shape) may not be compatible with " *
              "global shape $(config.global_shape) and mesh $(config.mesh)"
    end

    # Verify data type compatibility and convert if needed
    if eltype(field_pencil) != config.dtype
        @debug "Converting field data type from $(eltype(field_pencil)) to $(config.dtype)"
        # Create converted copy if types don't match
        converted_data = convert.(config.dtype, field_pencil)
        return converted_data
    end

    # Verify MPI communicator compatibility
    if field.dist.use_pencil_arrays && field.dist.pencil_config !== nothing
        if field.dist.pencil_config.comm != config.comm
            @warn "MPI communicator mismatch between field and config"
        end
    end

    @debug "Retrieved pencil compatible data for field $(field.name)" size=field_shape eltype=eltype(field_pencil)

    return field_pencil
end

function is_shape_compatible(local_shape::Tuple, global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)
    """
    Check if the local shape is compatible with the global shape given the mesh decomposition.

    For a valid pencil decomposition:
    - The product of local shapes across all ranks should equal the global shape
    - The local shape should be approximately global_shape / mesh for distributed dimensions
    """

    if length(local_shape) != length(global_shape)
        return false
    end

    # For serial execution (single rank), local shape should match global shape
    if MPI.Comm_size(comm) == 1
        return local_shape == global_shape
    end

    # For parallel execution, check that local shape is reasonable
    # (within expected range given the mesh decomposition)
    ndims = length(global_shape)
    mesh_dims = length(mesh)

    for i in 1:ndims
        # Determine if this dimension is decomposed
        if i <= mesh_dims && mesh[i] > 1
            # This dimension is distributed
            expected_local = ceil(Int, global_shape[i] / mesh[i])
            min_local = floor(Int, global_shape[i] / mesh[i])

            if local_shape[i] < min_local || local_shape[i] > expected_local
                return false
            end
        else
            # This dimension is not distributed, should match global
            if local_shape[i] != global_shape[i]
                return false
            end
        end
    end

    return true
end

function get_pencil_config_from_field(field::ScalarField)
    """
    Extract a PencilConfig from a ScalarField's distributor configuration.
    """
    dist = field.dist

    if dist.pencil_config !== nothing
        return dist.pencil_config
    end

    # Build a config from field properties
    gshape = global_shape(field.domain)
    mesh_config = dist.mesh !== nothing ? dist.mesh : (1,)

    return PencilConfig(
        gshape,
        mesh_config;
        comm=dist.comm,
        dtype=field.dtype
    )
end

function ensure_pencil_compatibility!(field::ScalarField, config::PencilConfig)
    """
    Ensure field is compatible with the given PencilConfig, reallocating if necessary.

    This function modifies the field in-place to ensure compatibility with the config.
    Returns true if the field was modified, false otherwise.
    """

    # Ensure field is in grid space
    ensure_layout!(field, :g)

    if field.data_g === nothing
        # Allocate new data with the correct configuration
        allocate_field_data!(field, config)
        return true
    end

    current_shape = size(field.data_g)

    # Check if reallocation is needed
    needs_realloc = false

    # Check shape compatibility
    if !is_shape_compatible(current_shape, config.global_shape, config.mesh, config.comm)
        needs_realloc = true
    end

    # Check dtype compatibility
    if eltype(field.data_g) != config.dtype
        needs_realloc = true
    end

    if needs_realloc
        # Store old data for potential interpolation
        old_data = copy(field.data_g)

        # Allocate new data
        allocate_field_data!(field, config)

        # Attempt to interpolate/copy data if shapes are compatible enough
        try
            interpolate_field_data!(field.data_g, old_data)
        catch e
            @warn "Could not preserve field data during reallocation: $e"
            fill!(field.data_g, zero(config.dtype))
        end

        return true
    end

    return false
end

function allocate_field_data!(field::ScalarField, config::PencilConfig)
    """
    Allocate field data according to the PencilConfig.
    """
    dist = field.dist

    if dist.use_pencil_arrays && MPI.Comm_size(config.comm) > 1
        # For parallel execution, compute local shape based on pencil decomposition
        local_shape = compute_local_shape(config.global_shape, config.mesh, config.comm)
        field.data_g = zeros(config.dtype, local_shape...)
    else
        # For serial execution, use global shape directly
        field.data_g = zeros(config.dtype, config.global_shape...)
    end

    # Also allocate coefficient space data with same shape
    field.data_c = zeros(Complex{real(config.dtype)}, size(field.data_g)...)

    field.current_layout = :g
end

function compute_local_shape(global_shape::Tuple, mesh::Tuple, comm::MPI.Comm)
    """
    Compute the local shape for this rank given the global shape and mesh decomposition.
    """
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    ndims = length(global_shape)
    mesh_dims = length(mesh)

    local_shape = collect(global_shape)

    # Compute local sizes for each decomposed dimension
    for i in 1:min(ndims, mesh_dims)
        if mesh[i] > 1
            # Compute which portion of this dimension belongs to this rank
            n = global_shape[i]
            p = mesh[i]

            # Determine rank position in this dimension of the mesh
            # (simplified: assumes row-major ordering)
            mesh_rank = (rank % prod(mesh[1:i])) ÷ prod(mesh[1:i-1])

            # Compute local size (even distribution with remainder going to last ranks)
            base_size = n ÷ p
            remainder = n % p

            if mesh_rank < p - remainder
                local_shape[i] = base_size
            else
                local_shape[i] = base_size + 1
            end
        end
    end

    return tuple(local_shape...)
end

function interpolate_field_data!(dest::AbstractArray, src::AbstractArray)
    """
    Interpolate source data into destination array.
    Uses nearest-neighbor or linear interpolation depending on relative sizes.
    """
    src_shape = size(src)
    dest_shape = size(dest)

    if src_shape == dest_shape
        copyto!(dest, src)
        return
    end

    # Use nearest-neighbor interpolation for simplicity
    ndims = length(dest_shape)

    for I in CartesianIndices(dest)
        src_indices = ntuple(ndims) do d
            # Map destination index to source index
            src_idx = round(Int, (I[d] - 1) * (src_shape[d] - 1) / max(dest_shape[d] - 1, 1)) + 1
            clamp(src_idx, 1, src_shape[d])
        end
        dest[I] = src[src_indices...]
    end
end

function set_pencil_compatible_data!(field::ScalarField, data, config::PencilConfig)
    """
    Set field data from PencilArray format.
    Since ScalarField stores data as PencilArrays, this mainly ensures
    proper layout and copies the data.
    """
    
    # Ensure field is in grid space layout
    ensure_layout!(field, :g)
    
    # Verify that field has allocated data
    if field.data_g === nothing
        throw(ArgumentError("Field $(field.name) has no grid space data allocated"))
    end
    
    # Verify data compatibility
    if size(data) != size(field.data_g)
        throw(DimensionMismatch("Data size $(size(data)) does not match field size $(size(field.data_g))"))
    end
    
    if eltype(data) != eltype(field.data_g)
        @warn "Data type mismatch during set: incoming $(eltype(data)), field $(eltype(field.data_g))"
    end
    
    # Copy data into the field's PencilArray
    copyto!(field.data_g, data)
    
    # Mark field as having valid grid space data
    field.current_layout = :g
    
    @debug "Set pencil compatible data for field $(field.name)" size(data) eltype(data)
end

# Memory management
function get_temp_field(evaluator::NonlinearEvaluator, template::ScalarField, name::String)
    """Get temporary field for intermediate calculations """
    
    key = "$(name)_$(hash(template.bases))"
    
    if !haskey(evaluator.temp_fields, key)
        temp_field = ScalarField(template.dist, name, template.bases, template.dtype)
        
        # Ensure temporary field is on the same device as evaluator
        temp_field.data_g = temp_field.data_g
        temp_field.data_c = temp_field.data_c
        
        evaluator.temp_fields[key] = temp_field
    end
    
    return evaluator.temp_fields[key]
end

function clear_temp_fields!(evaluator::NonlinearEvaluator)
    """Clear temporary fields to free memory"""
    empty!(evaluator.temp_fields)
    GC.gc()
end

function get_temp_array(evaluator::NonlinearEvaluator, shape::Tuple, dtype::Type)
    """Get temporary array for intermediate calculations"""
    return zeros(dtype, shape)
end

function return_temp_array!(evaluator::NonlinearEvaluator, arr::AbstractArray)
    """Return temporary array (no-op for CPU, let GC reclaim)"""
    return nothing
end

# Integration with existing operator evaluation
function evaluate_operator(op::NonlinearOperator)
    """Evaluate nonlinear operator"""
    
    if isa(op, AdvectionOperator)
        return evaluate_nonlinear_term(op)
    elseif isa(op, NonlinearAdvectionOperator)
        return evaluate_nonlinear_term(op)
    elseif isa(op, ConvectiveOperator)
        return evaluate_convective_operator(op)
    else
        throw(ArgumentError("Nonlinear operator evaluation not implemented for $(typeof(op))"))
    end
end

function evaluate_convective_operator(op::ConvectiveOperator)
    """Evaluate general convective operator"""
    
    field1, field2 = op.field1, op.field2
    
    if op.operation == :multiply
        # Simple multiplication
        if isa(field1, ScalarField) && isa(field2, ScalarField)
            evaluator = NonlinearEvaluator(field1.dist)
            return evaluate_transform_multiply(field1, field2, evaluator)
        else
            throw(ArgumentError("Multiplication not implemented for field types $(typeof(field1)), $(typeof(field2))"))
        end
        
    elseif op.operation == :dot_product
        # Dot product between vectors
        if isa(field1, VectorField) && isa(field2, VectorField)
            return evaluate_vector_dot_product(field1, field2)
        else
            throw(ArgumentError("Dot product requires two vector fields"))
        end
        
    elseif op.operation == :cross_product
        # Cross product between vectors
        if isa(field1, VectorField) && isa(field2, VectorField)
            return evaluate_vector_cross_product(field1, field2)
        else
            throw(ArgumentError("Cross product requires two vector fields"))
        end
        
    else
        throw(ArgumentError("Unknown convective operation: $(op.operation)"))
    end
end

function evaluate_vector_dot_product(v1::VectorField, v2::VectorField)
    """Evaluate v1·v2 dot product"""
    
    if length(v1.components) != length(v2.components)
        throw(ArgumentError("Vector fields must have same number of components"))
    end
    
    # Create nonlinear evaluator
    evaluator = NonlinearEvaluator(v1.dist)
    
    # Sum products of components
    result = evaluate_transform_multiply(v1.components[1], v2.components[1], evaluator)
    
    for i in 2:length(v1.components)
        product = evaluate_transform_multiply(v1.components[i], v2.components[i], evaluator)
        result = result + product
    end
    
    return result
end

function evaluate_vector_cross_product(v1::VectorField, v2::VectorField)
    """Evaluate v1×v2 cross product (3D only)"""
    
    if length(v1.components) != 3 || length(v2.components) != 3
        throw(ArgumentError("Cross product requires 3D vector fields"))
    end
    
    evaluator = NonlinearEvaluator(v1.dist)
    
    # Cross product: (a×b)_x = a_y*b_z - a_z*b_y
    #                (a×b)_y = a_z*b_x - a_x*b_z  
    #                (a×b)_z = a_x*b_y - a_y*b_x
    
    result = VectorField(v1.dist, v1.coordsys, "cross_$(v1.name)_$(v2.name)", v1.bases, v1.dtype)
    
    # x-component
    term1 = evaluate_transform_multiply(v1.components[2], v2.components[3], evaluator)
    term2 = evaluate_transform_multiply(v1.components[3], v2.components[2], evaluator)
    result.components[1] = term1 - term2
    
    # y-component  
    term1 = evaluate_transform_multiply(v1.components[3], v2.components[1], evaluator)
    term2 = evaluate_transform_multiply(v1.components[1], v2.components[3], evaluator)
    result.components[2] = term1 - term2
    
    # z-component
    term1 = evaluate_transform_multiply(v1.components[1], v2.components[2], evaluator)
    term2 = evaluate_transform_multiply(v1.components[2], v2.components[1], evaluator)
    result.components[3] = term1 - term2
    
    return result
end

# Convenience constructors
function advection(u::VectorField, φ::ScalarField)
    """Create advection operator u·∇φ"""
    return AdvectionOperator(u, φ)
end

function nonlinear_momentum(u::VectorField)
    """Create nonlinear momentum operator (u·∇)u"""
    return NonlinearAdvectionOperator(u)
end

function convection(f1, f2, op::Symbol)
    """Create convective operator"""
    return ConvectiveOperator(f1, f2, op)
end

function log_nonlinear_performance(stats::NonlinearPerformanceStats)
    """Log nonlinear evaluation performance statistics"""

    if MPI.Initialized()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if rank == 0
            @info "Nonlinear evaluation performance:"
            @info "  Total evaluations: $(stats.total_evaluations)"
            @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
            @info "  Average time per evaluation: $(round(stats.total_time/max(stats.total_evaluations, 1), digits=6)) seconds"
            @info "  Dealiasing time: $(round(stats.dealiasing_time, digits=3)) seconds ($(round(100*stats.dealiasing_time/max(stats.total_time, 1e-10), digits=1))%)"
            @info "  Transform time: $(round(stats.transform_time, digits=3)) seconds ($(round(100*stats.transform_time/max(stats.total_time, 1e-10), digits=1))%)"
        end
    else
        @info "Nonlinear evaluation performance:"
        @info "  Total evaluations: $(stats.total_evaluations)"
        @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
        @info "  Average time per evaluation: $(round(stats.total_time/max(stats.total_evaluations, 1), digits=6)) seconds"
    end
end
