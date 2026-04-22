# General-function, component, copy, and Hilbert-transform evaluation helpers.

# ============================================================================
# General Function Evaluation
# ============================================================================

"""
    evaluate_general_function(gf_op::GeneralFunction, layout::Symbol=:g)

Evaluate general function operator in grid space.
"""
function evaluate_general_function(gf_op::GeneralFunction, layout::Symbol=:g)
    operand = gf_op.operand
    f = gf_op.func
    name = gf_op.name

    if !isa(operand, ScalarField)
        throw(ArgumentError("GeneralFunction currently only supports scalar fields"))
    end

    # Work in grid space
    ensure_layout!(operand, :g)

    # Create result field
    result = ScalarField(operand.dist, "$(name)_$(operand.name)", operand.bases, operand.dtype)
    ensure_layout!(result, :g)

    # Apply function element-wise
    get_grid_data(result) .= f.(get_grid_data(operand))

    if layout == :c
        forward_transform!(result)
    end

    return result
end

"""
    evaluate_unary_grid_function(ugf_op::UnaryGridFunction, layout::Symbol=:g)

Evaluate unary grid function operator.
"""
function evaluate_unary_grid_function(ugf_op::UnaryGridFunction, layout::Symbol=:g)
    return evaluate_general_function(
        GeneralFunction(ugf_op.operand, ugf_op.func, ugf_op.name),
        layout
    )
end

# ============================================================================
# Grid and Coeff Conversion Evaluation
# ============================================================================

"""
    evaluate_grid(grid_op::Grid)

Convert operand to grid space.
"""
function evaluate_grid(grid_op::Grid)
    operand = grid_op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :g)
        return operand
    else
        throw(ArgumentError("Grid conversion not implemented for $(typeof(operand))"))
    end
end

"""
    evaluate_coeff(coeff_op::Coeff)

Convert operand to coefficient space.
"""
function evaluate_coeff(coeff_op::Coeff)
    operand = coeff_op.operand

    if isa(operand, ScalarField)
        ensure_layout!(operand, :c)
        return operand
    else
        throw(ArgumentError("Coeff conversion not implemented for $(typeof(operand))"))
    end
end

# ============================================================================
# Component Extraction Evaluation
# ============================================================================

"""
    evaluate_component(comp_op::Component)

Extract specific component from vector/tensor field.
"""
function evaluate_component(comp_op::Component)
    operand = comp_op.operand
    index = comp_op.index

    if isa(operand, VectorField)
        if index < 1 || index > length(operand.components)
            throw(BoundsError("Component index $index out of bounds"))
        end
        return operand.components[index]

    elseif isa(operand, TensorField)
        # For tensors, index could be linear or we need (i,j)
        if index < 1 || index > length(operand.components)
            throw(BoundsError("Component index $index out of bounds"))
        end
        return operand.components[index]

    else
        throw(ArgumentError("Component extraction requires VectorField or TensorField"))
    end
end

"""
    evaluate_radial_component(rc_op::RadialComponent)

Extract radial component from vector field.
For Cartesian coordinates, this is the x-component.
"""
function evaluate_radial_component(rc_op::RadialComponent)
    operand = rc_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("RadialComponent requires a VectorField"))
    end

    # In Cartesian, "radial" is typically the first component
    return operand.components[1]
end

"""
    evaluate_angular_component(ac_op::AngularComponent)

Extract angular component from vector field.
For Cartesian 2D, this is the y-component.
"""
function evaluate_angular_component(ac_op::AngularComponent)
    operand = ac_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("AngularComponent requires a VectorField"))
    end

    if length(operand.components) < 2
        throw(ArgumentError("VectorField must have at least 2 components"))
    end

    return operand.components[2]
end

"""
    evaluate_azimuthal_component(az_op::AzimuthalComponent)

Extract azimuthal component from vector field.
For Cartesian 3D, this is the z-component.
"""
function evaluate_azimuthal_component(az_op::AzimuthalComponent)
    operand = az_op.operand

    if !isa(operand, VectorField)
        throw(ArgumentError("AzimuthalComponent requires a VectorField"))
    end

    if length(operand.components) < 3
        throw(ArgumentError("VectorField must have at least 3 components"))
    end

    return operand.components[3]
end

# ============================================================================
# Copy Evaluation
# ============================================================================

"""
    evaluate_copy(op::Copy, layout::Symbol=:g)

Evaluate copy operator: produces an independent deep copy of the operand.

GPU-compatible: Uses ScalarField's custom deepcopy which properly handles
CuArray data (deepcopy creates an independent copy on the same device).
"""
function evaluate_copy(op::Copy, layout::Symbol=:g)
    operand = op.operand
    if isa(operand, Operator)
        operand = evaluate(operand, layout)
    end

    result = deepcopy(operand)
    if isa(result, ScalarField)
        ensure_layout!(result, layout)
    end
    return result
end

# ============================================================================
# Hilbert Transform Evaluation
# ============================================================================

"""
    evaluate_hilbert_transform(op::HilbertTransform, layout::Symbol=:g)

Evaluate Hilbert transform in spectral space.

For ComplexFourier: multiply mode k by -i*sign(k), k=0 → 0.
For RealFourier (interleaved [a0, a1, b1, a2, b2, ...]): swap cos↔sin
with sign change: H[cos(nx)] = sin(nx), H[sin(nx)] = -cos(nx).

GPU-compatible: Coefficient manipulation uses scalar indexing, so GPU arrays
are transferred to CPU, transformed, and copied back (same pattern as
interpolation and lift operations).
"""
function evaluate_hilbert_transform(op::HilbertTransform, layout::Symbol=:g)
    operand = op.operand
    if isa(operand, Operator)
        operand = evaluate(operand, :c)
    end

    if !isa(operand, ScalarField)
        throw(ArgumentError("HilbertTransform currently only supports scalar fields"))
    end

    result = deepcopy(operand)
    ensure_layout!(result, :c)
    coeff = get_coeff_data(result)

    # GPU path: transfer to CPU, apply, copy back (scalar indexing required)
    if is_gpu_array(coeff)
        cpu_coeff = Array(coeff)
        _apply_hilbert_spectral!(cpu_coeff, result.bases)
        arch = result.dist.architecture
        copyto!(coeff, on_architecture(arch, cpu_coeff))
    else
        _apply_hilbert_spectral!(coeff, result.bases)
    end

    if layout == :g
        backward_transform!(result)
    end
    return result
end

"""
    _apply_hilbert_spectral!(coeff, bases)

Apply -i*sign(k) multiplier in spectral space for each Fourier basis.
Non-Fourier bases are left unchanged (Hilbert transform only acts on
periodic dimensions).

Note: This function assumes `coeff` is a CPU array. GPU arrays must be
transferred to CPU before calling this (handled by evaluate_hilbert_transform).
"""
function _apply_hilbert_spectral!(coeff::AbstractArray, bases::Tuple)
    # For 1D fields, apply directly
    if ndims(coeff) == 1 && length(bases) >= 1
        basis = bases[1]
        _apply_hilbert_1d!(coeff, basis)
        return
    end

    # For multi-D, apply along each Fourier axis
    for (axis, basis) in enumerate(bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            _apply_hilbert_along_axis!(coeff, basis, axis)
        end
    end
end

"""
Apply Hilbert transform to 1D coefficient array for a single basis.
Operates on CPU arrays only (scalar indexing).
"""
function _apply_hilbert_1d!(coeff::AbstractVector, basis::Basis)
    if isa(basis, ComplexFourier)
        N = length(coeff)
        for i in 1:N
            if i <= N ÷ 2 + 1
                k = i - 1           # DC and positive frequencies (including Nyquist)
            else
                k = i - N - 1       # negative frequencies
            end
            if k == 0
                coeff[i] = zero(eltype(coeff))
            else
                # Multiply by -i*sign(k)
                coeff[i] = -im * sign(k) * coeff[i]
            end
        end
    elseif isa(basis, RealFourier)
        # RealFourier coefficients from rfft: complex vector [c0, c1, ..., c_{N/2}]
        # where c_k is the complex coefficient for wavenumber k (all k >= 0).
        # Hilbert transform: multiply by -i*sign(k).
        # Since rfft only stores k >= 0, sign(k) = +1 for k > 0.
        N = length(coeff)
        coeff[1] = zero(eltype(coeff))  # DC (k=0) → 0
        for i in 2:N
            coeff[i] *= -im  # -i*sign(k) = -i for k > 0
        end
        # Nyquist mode (k=N/2): Hilbert of cos(N/2*x) is sin(N/2*x)
        # but sin at Nyquist is unrepresentable in rfft → zero it out
        grid_size = basis.meta.size
        if grid_size % 2 == 0
            coeff[N] = zero(eltype(coeff))
        end
    end
    # Non-Fourier bases: no-op
end

"""
Apply Hilbert transform along a specific axis of a multi-dimensional array.
Operates on CPU arrays only (scalar indexing via CartesianIndices and view).
"""
function _apply_hilbert_along_axis!(coeff::AbstractArray, basis::Basis, axis::Int)
    for idx in CartesianIndices(ntuple(d -> d == axis ? (1:1) : (1:size(coeff, d)), ndims(coeff)))
        ranges = ntuple(d -> d == axis ? (1:size(coeff, d)) : (idx[d]:idx[d]), ndims(coeff))
        slice = view(coeff, ranges...)
        _apply_hilbert_1d!(vec(slice), basis)
    end
end
