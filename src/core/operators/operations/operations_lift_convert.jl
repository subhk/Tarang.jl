# Lift and basis-conversion evaluation helpers.

# ============================================================================
# Lift Evaluation
# ============================================================================

"""
    evaluate_lift(lift_op::Lift, layout::Symbol=:g)

Evaluate lifting operator for tau method boundary conditions.
Following the LiftJacobi implementation (basis.py:790-814).

The Lift operator creates a polynomial field P on the output basis with coefficient
at mode n set to 1, then returns P * operand. This "lifts" the operand (typically
a tau variable) into spectral space at the specified mode.

Convention:
- n < 0: wraps around (n = -1 means last mode, n = -2 means second-to-last, etc.)
- n >= 0: sets mode n directly (0-indexed convention, 1-indexed in Julia)
"""
function evaluate_lift(lift_op::Lift, layout::Symbol=:g)
    operand = lift_op.operand
    output_basis = lift_op.basis  # The output basis
    n = lift_op.n

    if !isa(operand, ScalarField)
        throw(ArgumentError("Lift currently only supports scalar fields"))
    end

    # Get basis size
    N = output_basis.meta.size

    # Handle negative index wrap-around (spectral convention)
    lift_mode = n
    if lift_mode < 0
        lift_mode = N + lift_mode
    end
    lift_mode += 1  # Convert from 0-indexed to 1-indexed Julia convention

    # Validate mode index
    if lift_mode < 1 || lift_mode > N
        throw(ArgumentError("Lift mode index $n (resolved to $lift_mode) out of bounds for basis size $N"))
    end

    # Find or create the output bases tuple
    output_bases = _get_lift_output_bases(operand, output_basis)

    # Step 1: Build polynomial P with coefficient 1 at mode n
    P = ScalarField(operand.dist, "lift_poly", output_bases, operand.dtype)
    ensure_layout!(P, :c)

    # Find which axis corresponds to the output basis
    basis_axis = _find_basis_axis(output_bases, output_basis)

    # Build P coefficients on CPU, then transfer to GPU if needed
    p_data = get_coeff_data(P)
    arch = operand.dist.architecture
    if is_gpu_array(p_data)
        # Build on CPU first, then copy to GPU
        cpu_p = zeros(eltype(p_data), size(p_data))
        if ndims(cpu_p) == 1
            cpu_p[lift_mode] = one(eltype(cpu_p))
        else
            selectdim(cpu_p, basis_axis, lift_mode) .= one(eltype(cpu_p))
        end
        copyto!(p_data, on_architecture(arch, cpu_p))
    else
        fill!(p_data, zero(eltype(p_data)))
        if ndims(p_data) == 1
            p_data[lift_mode] = one(eltype(p_data))
        else
            selectdim(p_data, basis_axis, lift_mode) .= one(eltype(p_data))
        end
    end

    # Step 2: Compute result = P * operand
    ensure_layout!(operand, :c)

    # Create result field
    result = ScalarField(operand.dist, "lift_$(operand.name)", output_bases, operand.dtype)
    ensure_layout!(result, :c)

    # Multiply P * operand: place operand's data at mode lift_mode
    _multiply_lift_polynomial!(get_coeff_data(result), get_coeff_data(P),
                               get_coeff_data(operand), basis_axis, lift_mode, arch)

    if layout == :g
        backward_transform!(result)
    end

    return result
end

"""
    _get_lift_output_bases(operand, output_basis)

Get output bases for lift operation, substituting input basis with output basis.
"""
function _get_lift_output_bases(operand::ScalarField, output_basis::Basis)
    output_coord = output_basis.meta.element_label

    new_bases = Vector{Union{Nothing, Basis}}(undef, length(operand.bases))
    found = false

    for (i, b) in enumerate(operand.bases)
        if b === nothing
            new_bases[i] = nothing
        elseif b.meta.element_label == output_coord
            new_bases[i] = output_basis
            found = true
        else
            new_bases[i] = b
        end
    end

    # If no matching basis found, this is a lift from no-basis to output_basis
    if !found
        push!(new_bases, output_basis)
    end

    return tuple(new_bases...)
end

"""
    _find_basis_axis(bases, target_basis)

Find which axis (1-indexed) corresponds to the target basis.
"""
function _find_basis_axis(bases::Tuple, target_basis::Basis)
    for (i, b) in enumerate(bases)
        if b === target_basis ||
           (b !== nothing && b.meta.element_label == target_basis.meta.element_label)
            return i
        end
    end
    return 1  # Default to first axis
end

"""
    _set_lift_coefficient!(data, axis, mode, value)

Set coefficient at specified mode along axis to given value.
"""
function _set_lift_coefficient!(data::AbstractArray, axis::Int, mode::Int, value::Real)
    view = selectdim(data, axis, mode)
    fill!(view, value)
end

"""
    _multiply_lift_polynomial!(result, P_data, operand_data, basis_axis, lift_mode, arch)

Multiply lift polynomial P by operand.
P has a single non-zero coefficient at lift_mode.
Result = P * operand places operand's values at mode lift_mode.

GPU-compatible: avoids scalar indexing by building on CPU and copying,
or using broadcasting operations that work on GPU arrays.
"""
function _multiply_lift_polynomial!(result::AbstractArray, P_data::AbstractArray,
                                    operand_data::AbstractArray, basis_axis::Int,
                                    lift_mode::Int, arch=nothing)
    if is_gpu_array(result)
        # GPU path: build result on CPU, then copy to GPU
        cpu_result = zeros(eltype(result), size(result))
        cpu_operand = is_gpu_array(operand_data) ? Array(operand_data) : operand_data

        if ndims(cpu_result) == 1
            if length(cpu_operand) >= 1
                cpu_result[lift_mode] = cpu_operand[1]
            end
        else
            result_slice = selectdim(cpu_result, basis_axis, lift_mode)
            if ndims(cpu_operand) == ndims(cpu_result)
                operand_slice = selectdim(cpu_operand, basis_axis, 1)
                result_slice .= operand_slice
            elseif ndims(cpu_operand) < ndims(cpu_result)
                result_slice .= cpu_operand
            else
                result_slice .= selectdim(cpu_operand, basis_axis, 1)
            end
        end

        # Transfer to GPU
        if arch !== nothing
            copyto!(result, on_architecture(arch, cpu_result))
        else
            copyto!(result, cpu_result)
        end
    else
        # CPU path: direct operations
        fill!(result, zero(eltype(result)))

        if ndims(result) == 1
            if length(operand_data) >= 1
                result[lift_mode] = operand_data[1]
            end
        else
            result_slice = selectdim(result, basis_axis, lift_mode)
            if ndims(operand_data) == ndims(result)
                operand_slice = selectdim(operand_data, basis_axis, 1)
                result_slice .= operand_slice
            elseif ndims(operand_data) < ndims(result)
                result_slice .= operand_data
            else
                result_slice .= selectdim(operand_data, basis_axis, 1)
            end
        end
    end
end

"""
    apply_lift_nd!(result, operand, axis, lift_mode)

Apply lift operation along specified axis for multi-dimensional arrays.
(Legacy helper - kept for compatibility)
"""
function apply_lift_nd!(result::AbstractArray, operand::AbstractArray, axis::Int, lift_mode::Int)
    selectdim(result, axis, lift_mode) .= selectdim(operand, axis, 1)
end

# ============================================================================
# Convert Evaluation
# ============================================================================

"""
    evaluate_convert(conv_op::Convert, layout::Symbol=:g)

Evaluate basis conversion operator.
Following operators Convert implementation.

Converts field from one basis representation to another using
spectral conversion matrices.
"""
function evaluate_convert(conv_op::Convert, layout::Symbol=:g)
    operand = conv_op.operand
    out_basis = conv_op.basis

    if !isa(operand, ScalarField)
        throw(ArgumentError("Convert currently only supports scalar fields"))
    end

    # Find the input basis to convert
    in_basis_index = nothing
    in_basis = nothing

    for (i, b) in enumerate(operand.bases)
        if b !== nothing && isa(b, JacobiBasis) && isa(out_basis, JacobiBasis)
            # Check if bases are on same coordinate
            if b.meta.element_label == out_basis.meta.element_label
                in_basis_index = i
                in_basis = b
                break
            end
        end
    end

    if in_basis === nothing
        # No conversion needed or not applicable
        return copy(operand)
    end

    # Work in coefficient space
    ensure_layout!(operand, :c)

    # Build or retrieve conversion matrix
    conv_mat = conversion_matrix(in_basis, out_basis)

    # Create result field
    new_bases = collect(operand.bases)
    new_bases[in_basis_index] = out_basis
    result = ScalarField(operand.dist, "conv_$(operand.name)", tuple(new_bases...), operand.dtype)
    ensure_layout!(result, :c)

    # Apply conversion matrix
    if ndims(get_coeff_data(operand)) == 1
        get_coeff_data(result) .= conv_mat * get_coeff_data(operand)
    else
        get_coeff_data(result) .= apply_matrix_along_axis(conv_mat, get_coeff_data(operand), in_basis_index)
    end

    if layout == :g
        backward_transform!(result)
    end

    return result
end
