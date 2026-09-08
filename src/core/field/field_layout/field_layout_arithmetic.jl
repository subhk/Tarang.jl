"""
    Field arithmetic

Scalar-field arithmetic helpers. The NetCDF save/load half of the original
`field_layout_arithmetic_io.jl` now lives in `src/tools/field_netcdf_io.jl`,
which is the layer that owns NetCDF.
"""

const _FIELD_ARITH_TMP_NAME = "_arith_tmp"

# Helper: get local data for broadcasting (handles PencilArray vs plain array)
@inline _local_data(data::PencilArrays.PencilArray) = parent(data)
@inline _local_data(data::AbstractArray) = data

"""Validate metadata that must agree for pointwise binary field arithmetic."""
function _check_field_arithmetic_compatibility(a::ScalarField, b::ScalarField,
                                               operation::AbstractString)
    a.bases == b.bases || throw(ArgumentError(
        "Cannot $operation fields with different bases"))
    a.dist === b.dist || throw(ArgumentError(
        "Cannot $operation fields owned by different distributors"))
    a.scales == b.scales || throw(ArgumentError(
        "Cannot $operation fields with different scales ($(a.scales) and $(b.scales))"))
    return nothing
end

"""Allocate an arithmetic result with the source field's current grid shape."""
function _allocate_field_arithmetic_result(field::ScalarField, dtype::DataType)
    result = ScalarField(field.dist, _FIELD_ARITH_TMP_NAME, field.bases, dtype)
    preset_scales!(result, field.scales)
    return result
end

"""Return `field` at `dtype`, copying only when arithmetic promotion requires it."""
function _promote_field_arithmetic_operand(field::ScalarField, dtype::DataType)
    field.dtype === dtype && return field

    promoted = _allocate_field_arithmetic_result(field, dtype)
    if field.current_layout == :c
        _local_data(coeff_data!(promoted)) .= _local_data(get_coeff_data(field))
    else
        _local_data(grid_data!(promoted)) .= _local_data(get_grid_data(field))
    end
    return promoted
end

@inline _has_nonunit_arithmetic_scale(field::ScalarField) =
    field.scales !== nothing && any(!isone, field.scales)

# Field arithmetic
# NOTE: Fresh ScalarField allocation via constructor (not copy()) avoids copying
# data that is immediately overwritten. allocate_data!() inside the constructor
# correctly creates PencilArray storage for MPI mode.
function Base.:+(a::ScalarField, b::ScalarField)
    _check_field_arithmetic_compatibility(a, b, "add")

    result = _allocate_field_arithmetic_result(a, promote_type(a.dtype, b.dtype))
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)

    _local_data(grid_data!(result)) .= _local_data(get_grid_data(a)) .+ _local_data(get_grid_data(b))

    return result
end

function Base.:-(a::ScalarField, b::ScalarField)
    _check_field_arithmetic_compatibility(a, b, "subtract")

    result = _allocate_field_arithmetic_result(a, promote_type(a.dtype, b.dtype))
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)

    _local_data(grid_data!(result)) .= _local_data(get_grid_data(a)) .- _local_data(get_grid_data(b))

    return result
end

function Base.:*(a::ScalarField, b::Number)
    # Accept any Number (was ::Real). A complex scalar must actually scale the field —
    # the old ::Real method let `field * (2+3im)` fall through to the lazy `Multiply`
    # Operand path and return an UNEVALUATED future (the scaling was silently never
    # applied). Promote the dtype so a real field times a complex scalar yields a
    # complex field.
    T = promote_type(a.dtype, typeof(b))
    result = _allocate_field_arithmetic_result(a, T)
    ensure_layout!(a, :g)

    _local_data(grid_data!(result)) .= b .* _local_data(get_grid_data(a))

    return result
end

function Base.:*(a::ScalarField, b::ScalarField)
    _check_field_arithmetic_compatibility(a, b, "multiply")

    # Dealiased product for spectral fields: delegate to the SAME nonlinear-product
    # machinery the solver RHS uses — `evaluate_transform_multiply` (3/2-padded on
    # serial, 2/3 input-truncation under MPI). The previous implementation multiplied
    # on the un-padded grid and then applied a 2/3 OUTPUT cutoff, which left input
    # modes in (N/3, N/2] aliased BELOW the cutoff (contaminating `a*b` and `dot(u,u)`).
    # The gate uses the GLOBAL element count (prod of basis grid sizes) rather than the
    # local-slab `length`, so under MPI every rank makes the same (collective) decision.
    # The result is handed straight to user code, which may hold any number of
    # products at once, so it must NOT be a borrowed pool buffer — keep the default
    # `own=true`. See `_own_borrowed_field` in `src/core/field_pool.jl`.
    if has_spectral_bases(a) && prod(basis.meta.size for basis in a.bases) > 64 &&
       !_has_nonunit_arithmetic_scale(a)
        dtype = promote_type(a.dtype, b.dtype)
        promoted_a = _promote_field_arithmetic_operand(a, dtype)
        promoted_b = _promote_field_arithmetic_operand(b, dtype)
        return evaluate_transform_multiply(promoted_a, promoted_b, _get_evaluator(a.dist))
    end

    result = _allocate_field_arithmetic_result(a, promote_type(a.dtype, b.dtype))
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)

    _local_data(grid_data!(result)) .= _local_data(get_grid_data(a)) .* _local_data(get_grid_data(b))

    return result
end

# Commutative scalar multiplication
Base.:*(b::Number, a::ScalarField) = a * b

# NetCDF save/load for fields used to live here. It moved to
# `src/tools/field_netcdf_io.jl`: it sits above the NetCDF slab layer, which
# loads after the entire core stack, so keeping it under `core/` inverted the
# dependency direction. See that file's header and `test/test_layering.jl`.
