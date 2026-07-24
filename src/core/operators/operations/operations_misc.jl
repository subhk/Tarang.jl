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

GPU-compatible: The spectral multiplier is assembled as small configuration
data and uploaded once per Fourier axis; coefficient data remains on-device.
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

    # The spectral Hilbert multiplier −i·sign(k) needs each mode's GLOBAL wavenumber.
    # Under MPI a decomposed Fourier axis hands each rank only a LOCAL coefficient
    # slice, so the per-axis kernel would use local indices as wavenumbers and silently
    # produce a wrong result. Fail loudly instead. (A Fourier axis that stays LOCAL —
    # e.g. with a Chebyshev axis decomposed — is fine and not blocked here.)
    if result.dist.size > 1 && isa(coeff, PencilArrays.PencilArray)
        decomp = PencilArrays.decomposition(PencilArrays.pencil(coeff))
        if any(d -> isa(result.bases[d], Union{RealFourier, ComplexFourier}), decomp)
            throw(ErrorException(
                "HilbertTransform is not supported under MPI decomposition of a Fourier " *
                "axis (the spectral −i·sign(k) multiplier needs global wavenumbers; the " *
                "decomposed axis only exposes per-rank-local indices). Run serially or " *
                "gather the field onto one rank first."))
        end
    end

    _apply_hilbert_spectral!(coeff, result.bases)

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

For GPU arrays, only the one-dimensional multiplier is transferred to the
device. The coefficient array is modified in place by a device broadcast.
"""
function _apply_hilbert_spectral!(coeff::AbstractArray, bases::Tuple)
    local_coeff = coeff isa PencilArrays.PencilArray ? parent(coeff) : coeff

    # PencilArray parents may store logical axes in a permuted physical order.
    perm_tuple = if coeff isa PencilArrays.PencilArray
        raw = Tuple(PencilArrays.permutation(coeff))
        raw === nothing ? ntuple(identity, ndims(local_coeff)) : raw
    else
        ntuple(identity, ndims(local_coeff))
    end

    for (logical_axis, basis) in enumerate(bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            physical_axis = findfirst(==(logical_axis), perm_tuple)
            physical_axis === nothing && (physical_axis = logical_axis)
            _apply_hilbert_along_axis!(local_coeff, basis, physical_axis)
        end
    end
    return coeff
end

"""
Apply Hilbert transform to 1D coefficient array for a single basis.
Uses a broadcast and is valid on CPU and GPU arrays.
"""
function _apply_hilbert_1d!(coeff::AbstractVector, basis::Basis)
    multiplier = _hilbert_multiplier(basis, length(coeff), eltype(coeff))
    if is_gpu_array(coeff)
        multiplier = copy_to_device(multiplier, coeff)
    end
    coeff .*= multiplier
    return coeff
end

"""Build the one-dimensional `-im * sign(k)` Hilbert multiplier."""
function _hilbert_multiplier(basis::Basis, N::Int, ::Type{T}) where T
    multiplier = ones(T, N)
    if isa(basis, ComplexFourier)
        for i in 1:N
            if i <= N ÷ 2 + 1
                k = i - 1           # DC and positive frequencies (including Nyquist)
            else
                k = i - N - 1       # negative frequencies
            end
            if k == 0
                multiplier[i] = zero(T)
            else
                multiplier[i] = convert(T, -im * sign(k))
            end
        end
    elseif isa(basis, RealFourier)
        # A RealFourier axis is stored as an rfft HALF-spectrum ([c0..c_{N/2}], all
        # k>=0) only when it is the FIRST RealFourier axis; any other RealFourier axis
        # is stored as a FULL FFT (length N, FFT-ordered wavenumbers including
        # negatives). The half-spectrum-only multiplier (-i for every stored mode) is
        # WRONG for a full-FFT axis — its negative-frequency modes need +i. Detect the
        # layout by the stored length and apply -i*sign(k) accordingly. Hilbert: multiply
        # by -i*sign(k); the DC and (even-N) Nyquist modes map to 0 for a real field.
        grid_size = basis.meta.size
        half = grid_size ÷ 2 + 1
        if N == half && half != grid_size
            # rfft half-spectrum (first RealFourier axis): all stored k >= 0.
            multiplier[1] = zero(T)                    # DC (k=0)
            for i in 2:N
                multiplier[i] = convert(T, -im)        # -i*sign(k), k > 0
            end
            if grid_size % 2 == 0                      # Nyquist (k=N/2)
                multiplier[end] = zero(T)
            end
        else
            # full FFT (RealFourier axis other than the first): FFT-ordered k.
            for i in 1:N
                k = i <= N ÷ 2 + 1 ? i - 1 : i - N - 1
                if k == 0 || (iseven(N) && i == N ÷ 2 + 1)   # DC and Nyquist → 0
                    multiplier[i] = zero(T)
                else
                    multiplier[i] = convert(T, -im * sign(k))
                end
            end
        end
    end
    return multiplier
end

"""
Apply Hilbert transform along a specific axis of a multi-dimensional array.
Uses a broadcast and is valid on CPU and GPU arrays.
"""
function _apply_hilbert_along_axis!(coeff::AbstractArray, basis::Basis, axis::Int)
    multiplier = _hilbert_multiplier(basis, size(coeff, axis), eltype(coeff))
    if is_gpu_array(coeff)
        multiplier = copy_to_device(multiplier, coeff)
    end
    shape = ntuple(d -> d == axis ? length(multiplier) : 1, ndims(coeff))
    coeff .*= reshape(multiplier, shape)
    return coeff
end
