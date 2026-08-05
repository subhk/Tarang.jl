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

# Field arithmetic
# NOTE: Fresh ScalarField allocation via constructor (not copy()) avoids copying
# data that is immediately overwritten. allocate_data!() inside the constructor
# correctly creates PencilArray storage for MPI mode.
function Base.:+(a::ScalarField, b::ScalarField)
    if a.bases != b.bases
        throw(ArgumentError("Cannot add fields with different bases"))
    end

    result = ScalarField(a.dist, _FIELD_ARITH_TMP_NAME, a.bases, a.dtype)
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)
    ensure_layout!(result, :g)

    _local_data(get_grid_data(result)) .= _local_data(get_grid_data(a)) .+ _local_data(get_grid_data(b))

    return result
end

function Base.:-(a::ScalarField, b::ScalarField)
    if a.bases != b.bases
        throw(ArgumentError("Cannot subtract fields with different bases"))
    end

    result = ScalarField(a.dist, _FIELD_ARITH_TMP_NAME, a.bases, a.dtype)
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)
    ensure_layout!(result, :g)

    _local_data(get_grid_data(result)) .= _local_data(get_grid_data(a)) .- _local_data(get_grid_data(b))

    return result
end

function Base.:*(a::ScalarField, b::Number)
    # Accept any Number (was ::Real). A complex scalar must actually scale the field —
    # the old ::Real method let `field * (2+3im)` fall through to the lazy `Multiply`
    # Operand path and return an UNEVALUATED future (the scaling was silently never
    # applied). Promote the dtype so a real field times a complex scalar yields a
    # complex field.
    T = promote_type(a.dtype, typeof(b))
    result = ScalarField(a.dist, _FIELD_ARITH_TMP_NAME, a.bases, T)
    ensure_layout!(a, :g)
    ensure_layout!(result, :g)

    _local_data(get_grid_data(result)) .= b .* _local_data(get_grid_data(a))

    return result
end

function Base.:*(a::ScalarField, b::ScalarField)
    if a.bases != b.bases
        throw(ArgumentError("Cannot multiply fields with different bases"))
    end

    # Dealiased product for spectral fields: delegate to the SAME nonlinear-product
    # machinery the solver RHS uses — `evaluate_transform_multiply` (3/2-padded on
    # serial, 2/3 input-truncation under MPI). The previous implementation multiplied
    # on the un-padded grid and then applied a 2/3 OUTPUT cutoff, which left input
    # modes in (N/3, N/2] aliased BELOW the cutoff (contaminating `a*b` and `dot(u,u)`).
    # The gate uses the GLOBAL element count (prod of basis grid sizes) rather than the
    # local-slab `length`, so under MPI every rank makes the same (collective) decision.
    if has_spectral_bases(a) && prod(basis.meta.size for basis in a.bases) > 64
        return evaluate_transform_multiply(a, b, _get_evaluator(a.dist))
    end

    result = ScalarField(a.dist, _FIELD_ARITH_TMP_NAME, a.bases, a.dtype)
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)
    ensure_layout!(result, :g)

    _local_data(get_grid_data(result)) .= _local_data(get_grid_data(a)) .* _local_data(get_grid_data(b))

    return result
end

# Commutative scalar multiplication
Base.:*(b::Number, a::ScalarField) = a * b

# NetCDF save/load for fields used to live here. It moved to
# `src/tools/field_netcdf_io.jl`: it sits above the NetCDF slab layer, which
# loads after the entire core stack, so keeping it under `core/` inverted the
# dependency direction. See that file's header and `test/test_layering.jl`.
