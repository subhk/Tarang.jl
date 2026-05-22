# Set true to collect wall-clock timing stats; false = zero overhead (dead-code eliminated).
const _TRACK_NL_TIMING = false

# Main nonlinear evaluation functions
"""Evaluate u·∇φ nonlinear term using transform method"""
function evaluate_nonlinear_term(op::AdvectionOperator, layout::Symbol=:g)

    velocity = op.velocity
    scalar = op.scalar

    # Get distributor for transform operations
    dist = velocity.dist

    # Create nonlinear evaluator if not exists
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    evaluator = dist.nonlinear_evaluator

    # Compute gradient of scalar field: ∇φ
    grad_scalar = evaluate_gradient(Gradient(scalar, dist.coordsys), :g)

    # Evaluate u·∇φ = u_x ∂φ/∂x + u_y ∂φ/∂y (+ u_z ∂φ/∂z in 3D)
    # Use in-place accumulation to avoid lazy Add trees and per-iteration allocation
    result = ScalarField(dist, "$(op.name)_$(scalar.name)", scalar.bases, scalar.dtype)
    ensure_layout!(result, :g)
    result_data = get_grid_data(result)
    fill!(result_data, zero(scalar.dtype))

    # Sum velocity components times gradient components (in-place)
    for i in 1:length(velocity.components)
        product = evaluate_transform_multiply(velocity.components[i], grad_scalar.components[i], evaluator)
        ensure_layout!(product, :g)
        product_data = get_grid_data(product)
        if isa(result_data, PencilArrays.PencilArray) && isa(product_data, PencilArrays.PencilArray)
            parent(result_data) .+= parent(product_data)
        else
            result_data .+= product_data
        end
    end

    return result
end

"""Evaluate (u·∇)u nonlinear momentum term"""
function evaluate_nonlinear_term(op::NonlinearAdvectionOperator, layout::Symbol=:g)

    velocity = op.velocity
    dist = velocity.dist

    # Create nonlinear evaluator if needed
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    evaluator = dist.nonlinear_evaluator

    # Result is a vector field
    result = VectorField(dist, dist.coordsys, "$(op.name)_$(velocity.name)", velocity.bases, velocity.dtype)

    # For each component: (u·∇)u_i = u_j ∂u_i/∂x_j (summed over j)
    # Use in-place accumulation to avoid lazy Add trees and allocation per iteration
    for i in 1:length(velocity.components)
        ensure_layout!(result.components[i], :g)
        comp_data = get_grid_data(result.components[i])
        fill!(comp_data, zero(velocity.dtype))

        # Sum over all spatial directions (in-place)
        for j in 1:length(velocity.components)
            coord = dist.coordsys[j]
            du_i_dx_j = evaluate_differentiate(Differentiate(velocity.components[i], coord, 1), :g)

            product = evaluate_transform_multiply(velocity.components[j], du_i_dx_j, evaluator)
            ensure_layout!(product, :g)
            product_data = get_grid_data(product)
            if isa(comp_data, PencilArrays.PencilArray) && isa(product_data, PencilArrays.PencilArray)
                parent(comp_data) .+= parent(product_data)
            else
                comp_data .+= product_data
            end
        end
    end

    return result
end

"""Efficiently multiply two fields using transform method and proper dealiasing.

    Uses proper 3/2-rule padded dealiasing (Orszag 1971) when possible:
    - CPU serial: full padded dealiasing on all Fourier dimensions
    - GPU serial: full padded dealiasing using GPU FFTs (CUFFT)
    - MPI distributed: padded dealiasing on LOCAL (non-decomposed) Fourier
      dimensions; truncation-after-multiply on distributed dimensions

    Falls back to truncation-after-multiply only when no Fourier bases exist.
    """
function evaluate_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator)

    _TRACK_NL_TIMING && (start_time = time())

    if field1.bases != field2.bases
        throw(ArgumentError(
            "Cannot multiply fields with different bases: " *
            "'$(field1.name)' has bases=$(field1.bases) but " *
            "'$(field2.name)' has bases=$(field2.bases). " *
            "Both fields must be defined on the same domain."))
    end

    # Try proper 3/2-rule padded dealiasing. Gate on the per-axis basis `dealias`
    # settings (factor > 1 on any Fourier axis) rather than the evaluator's global
    # default, so bases set to dealias ≤ 1 are computed without dealiasing.
    if _any_axis_dealias(field1.bases, evaluator.dealiasing_factor)
        T = field1.dtype <: Complex ? real(field1.dtype) : field1.dtype
        # Detect the MPI pencil path WITHOUT forcing a layout first, so the
        # distributed routine can truncate inputs in coefficient space (avoids a
        # c→g→c round trip when operands arrive in coeff space — the usual case).
        if field1.dist.use_pencil_arrays && evaluator.dist.size > 1
            # MPI path: 2/3-rule truncation dealiasing using correct per-rank global
            # wavenumbers. Exact 3/2 zero-padding is avoided here because, under MPI
            # decomposition, embedding the N-mode spectrum into the 3N/2 padded spectrum
            # would require cross-rank redistribution (the original and padded
            # PencilArrays decompose differently). Truncation is purely local per rank
            # and dealiases quadratic nonlinear terms within the retained |k| ≤ N/3 band.
            result = evaluate_truncated_multiply_distributed(field1, field2, evaluator)
            evaluator.performance_stats.total_evaluations += 1
            if _TRACK_NL_TIMING
                elapsed = time() - start_time
                evaluator.performance_stats.total_time += elapsed
                evaluator.performance_stats.dealiasing_time += elapsed
            end
            return result
        else
            # Serial path (CPU or GPU): pad all Fourier dimensions (needs grid inputs)
            ensure_layout!(field1, :g)
            ensure_layout!(field2, :g)
            ws = _get_padded_workspace!(evaluator, field1.bases, T)
            if ws !== nothing
                result = evaluate_padded_multiply(field1, field2, evaluator, ws)
                evaluator.performance_stats.total_evaluations += 1
                if _TRACK_NL_TIMING
                    elapsed = time() - start_time
                    evaluator.performance_stats.total_time += elapsed
                    evaluator.performance_stats.dealiasing_time += elapsed
                end
                return result
            end
        end
    end

    # Fallback: direct multiplication + truncation-after-multiply dealiasing
    # (only reached for fields with no Fourier bases, or dealiasing_factor <= 1)
    ensure_layout!(field1, :g)
    ensure_layout!(field2, :g)
    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    result_data = get_grid_data(result)
    field1_data = get_grid_data(field1)
    field2_data = get_grid_data(field2)

    if isa(result_data, PencilArrays.PencilArray)
        parent(result_data) .= parent(field1_data) .* parent(field2_data)
    else
        gpu_multiply_fields!(result["g"], field1["g"], field2["g"])
    end

    # Only apply truncation-after-multiply dealiasing if a Fourier axis requests it
    # (dealias > 1). For pure Chebyshev/Legendre fields, or bases with dealias ≤ 1,
    # the forward/backward roundtrip in apply_basic_dealiasing! is skipped.
    if _any_axis_dealias(field1.bases, evaluator.dealiasing_factor)
        apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
    end

    evaluator.performance_stats.total_evaluations += 1
    _TRACK_NL_TIMING && (evaluator.performance_stats.total_time += time() - start_time)

    return result
end

"""
    gpu_multiply_fields!(result_data, data1, data2)

GPU-accelerated pointwise multiplication.
This function is overridden by the CUDA extension to use GPU kernels.
"""
function gpu_multiply_fields!(result_data::AbstractArray, data1::AbstractArray, data2::AbstractArray)
    # Default CPU implementation using broadcasting
    result_data .= data1 .* data2
    return result_data
end

"""2D transform-based multiplication using PencilFFTs"""
function evaluate_2d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)

    # Create result field (static name avoids string allocation per call)
    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    # Try to find matching transform configuration
    shape_key = shape  # Use tuple directly as key (no string allocation)

    if haskey(evaluator.pencil_transforms, shape_key)
        # Use precomputed transform config. PencilFFT configs keep the old
        # optimized path; FFTW configs fall back to direct multiply plus the
        # standard spectral truncation pass.
        transform_info = evaluator.pencil_transforms[shape_key]
        if transform_info isa PencilTransformConfig
            data1 = get_pencil_compatible_data(field1, transform_info.config)
            data2 = get_pencil_compatible_data(field2, transform_info.config)
            result_data = data1 .* data2

            if evaluator.dealiasing_factor > 1.0
                result_data = apply_2d_dealiasing(result_data, transform_info, evaluator.dealiasing_factor)
            end

            set_pencil_compatible_data!(result, result_data, transform_info.config)
        else
            result["g"] .= field1["g"] .* field2["g"]

            if evaluator.dealiasing_factor > 1.0
                apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
            end
        end

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

"""3D transform-based multiplication using 3D PencilFFTs"""
function evaluate_3d_transform_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator, shape::Tuple)

    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    # For 3D, we need more sophisticated pencil management
    # This would involve 3D pencil decomposition across all three spatial dimensions

    if length(evaluator.dist.mesh) >= 3
        # Use 3D PencilFFT approach
        @warn "3D dealiasing via padding is not yet implemented; using undealiased multiplication" maxlog=1

        # Direct multiplication for now - would implement full 3D PencilFFT logic
        result["g"] .= field1["g"] .* field2["g"]

        # Apply 3D dealiasing
        if evaluator.dealiasing_factor > 1.0
            apply_3d_dealiasing!(result, evaluator.dealiasing_factor)
        end

    else
        # Fallback for insufficient parallelization
        @warn "3D nonlinear multiplication falling back to undealiased pointwise multiply" maxlog=1
        result["g"] .= field1["g"] .* field2["g"]

        if evaluator.dealiasing_factor > 1.0
            apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
        end
    end

    return result
end

"""Fallback multiplication for unsupported dimensions"""
function evaluate_fallback_multiply(field1::ScalarField, field2::ScalarField, evaluator::NonlinearEvaluator)

    result = ScalarField(field1.dist, "_nl_product", field1.bases, field1.dtype)
    ensure_layout!(result, :g)

    # Simple pointwise multiplication
    result["g"] .= field1["g"] .* field2["g"]

    # Apply dealiasing if requested
    if evaluator.dealiasing_factor > 1.0
        apply_basic_dealiasing!(result, evaluator.dealiasing_factor)
    end

    return result
end

# Integration with existing operator evaluation
evaluate_operator(op::AdvectionOperator) = evaluate_nonlinear_term(op)
evaluate_operator(op::NonlinearAdvectionOperator) = evaluate_nonlinear_term(op)
evaluate_operator(op::ConvectiveOperator) = evaluate_convective_operator(op)
function evaluate_operator(op::NonlinearOperator)
    throw(ArgumentError("Nonlinear operator evaluation not implemented for $(typeof(op))"))
end

"""Get the cached NonlinearEvaluator for a distributor, creating one if needed."""
function _get_evaluator(dist::Distributor)
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    return dist.nonlinear_evaluator
end

"""Evaluate general convective operator"""
function evaluate_convective_operator(op::ConvectiveOperator)

    field1, field2 = op.field1, op.field2

    if op.operation == :multiply
        # Simple multiplication
        if isa(field1, ScalarField) && isa(field2, ScalarField)
            evaluator = _get_evaluator(field1.dist)
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

"""Evaluate v1·v2 dot product"""
function evaluate_vector_dot_product(v1::VectorField, v2::VectorField)

    if length(v1.components) != length(v2.components)
        throw(ArgumentError("Vector fields must have same number of components"))
    end

    evaluator = _get_evaluator(v1.dist)

    # Sum products of components
    result = evaluate_transform_multiply(v1.components[1], v2.components[1], evaluator)

    for i in 2:length(v1.components)
        product = evaluate_transform_multiply(v1.components[i], v2.components[i], evaluator)
        result = result + product
    end

    return result
end

"""Evaluate v1×v2 cross product (3D only)"""
function evaluate_vector_cross_product(v1::VectorField, v2::VectorField)

    if length(v1.components) != 3 || length(v2.components) != 3
        throw(ArgumentError("Cross product requires 3D vector fields"))
    end
    if v1.coordsys !== v2.coordsys
        throw(ArgumentError("Cross product requires identical coordinate systems"))
    end
    if v1.bases != v2.bases
        throw(ArgumentError("Cannot compute cross product of VectorFields with different bases"))
    end

    evaluator = _get_evaluator(v1.dist)

    handedness = (hasproperty(v1.coordsys, :right_handed) && v1.coordsys.right_handed === false) ? -1 : 1

    # Cross product: (a×b)_x = a_y*b_z - a_z*b_y
    #                (a×b)_y = a_z*b_x - a_x*b_z
    #                (a×b)_z = a_x*b_y - a_y*b_x

    result = VectorField(v1.dist, v1.coordsys, "cross_$(v1.name)_$(v2.name)", v1.bases, v1.dtype)

    # x-component
    term1 = evaluate_transform_multiply(v1.components[2], v2.components[3], evaluator)
    term2 = evaluate_transform_multiply(v1.components[3], v2.components[2], evaluator)
    result.components[1] = (term1 - term2) * handedness

    # y-component
    term1 = evaluate_transform_multiply(v1.components[3], v2.components[1], evaluator)
    term2 = evaluate_transform_multiply(v1.components[1], v2.components[3], evaluator)
    result.components[2] = (term1 - term2) * handedness

    # z-component
    term1 = evaluate_transform_multiply(v1.components[1], v2.components[2], evaluator)
    term2 = evaluate_transform_multiply(v1.components[2], v2.components[1], evaluator)
    result.components[3] = (term1 - term2) * handedness

    return result
end

# Helper to evaluate any operand (Future, Operator, or Field)
function _evaluate_any_operand(arg, layout::Symbol)
    if isa(arg, Future)
        return evaluate(arg; force=true)
    elseif isa(arg, Operator)
        return evaluate(arg, layout)
    else
        return arg
    end
end

# Evaluate methods for DotProduct and CrossProduct from arithmetic.jl
"""Evaluate DotProduct of two VectorFields"""
function evaluate(op::DotProduct, layout::Symbol=:g)
    args = future_args(op)
    if length(args) != 2
        throw(ArgumentError("DotProduct expects exactly two operands"))
    end

    # Evaluate operands first - they might be operators that return VectorFields
    v1 = _evaluate_any_operand(args[1], layout)
    v2 = _evaluate_any_operand(args[2], layout)

    if !isa(v1, VectorField) || !isa(v2, VectorField)
        throw(ArgumentError("DotProduct requires two VectorField operands, got $(typeof(v1)) and $(typeof(v2))"))
    end

    result = evaluate_vector_dot_product(v1, v2)
    ensure_layout!(result, layout)
    return result
end

"""Evaluate CrossProduct of two VectorFields"""
function evaluate(op::CrossProduct, layout::Symbol=:g)
    args = future_args(op)
    if length(args) != 2
        throw(ArgumentError("CrossProduct expects exactly two operands"))
    end

    # Evaluate operands first - they might be operators that return VectorFields
    v1 = _evaluate_any_operand(args[1], layout)
    v2 = _evaluate_any_operand(args[2], layout)

    if !isa(v1, VectorField) || !isa(v2, VectorField)
        throw(ArgumentError("CrossProduct requires two VectorField operands, got $(typeof(v1)) and $(typeof(v2))"))
    end

    result = evaluate_vector_cross_product(v1, v2)
    for comp in result.components
        ensure_layout!(comp, layout)
    end
    return result
end

# Convenience constructors
"""Create advection operator u·∇φ"""
function advection(u::VectorField, φ::ScalarField)
    return AdvectionOperator(u, φ)
end

"""Create nonlinear momentum operator (u·∇)u"""
function nonlinear_momentum(u::VectorField)
    return NonlinearAdvectionOperator(u)
end

"""Create convective operator"""
function convection(f1, f2, op::Symbol)
    return ConvectiveOperator(f1, f2, op)
end

"""Log nonlinear evaluation performance statistics"""
function log_nonlinear_performance(stats::NonlinearPerformanceStats)

    if MPI.Initialized()
        mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if mpi_rank == 0
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
