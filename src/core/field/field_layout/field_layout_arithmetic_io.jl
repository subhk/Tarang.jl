"""
    Field arithmetic and I/O

This file contains scalar-field arithmetic helpers plus basic NetCDF-backed
save/load operations.
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

function Base.:*(a::ScalarField, b::Real)
    result = ScalarField(a.dist, _FIELD_ARITH_TMP_NAME, a.bases, a.dtype)
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
Base.:*(b::Real, a::ScalarField) = a * b

# I/O operations
"""Save field to NetCDF file"""
function save_field(field::ScalarField, filename::String, dataset_name::String="field")
    ensure_layout!(field, :g)

    # Gather data to root process for writing
    global_data = gather_array(field.dist, get_grid_data(field))

    if field.dist.rank == 0
        if !endswith(filename, ".nc")
            filename = replace(filename, r"\.(h5|hdf5)$" => "") * ".nc"
        end
        ncwrite(global_data, filename, dataset_name)
    end
end

"""Load field from NetCDF file"""
function load_field!(field::ScalarField, filename::String, dataset_name::String="field")
    # Broadcast success/failure from rank 0 to all ranks before scatter_array
    # to prevent deadlock if ncread throws on rank 0.
    load_ok = Ref(true)
    global_data = nothing
    load_error = nothing

    if field.dist.rank == 0
        try
            global_data = ncread(filename, dataset_name)
        catch e
            load_ok[] = false
            load_error = e
        end
    end

    # Single collective broadcast — all ranks participate regardless of success/failure
    if field.dist.size > 1 && MPI.Initialized() && !MPI.Finalized()
        MPI.Bcast!(load_ok, field.dist.comm; root=0)
    end

    # After the collective, check the result and abort coherently
    if !load_ok[]
        if field.dist.rank == 0
            rethrow(load_error)
        else
            error("Rank 0 failed to load field from '$filename' dataset '$dataset_name'")
        end
    end

    if field.dist.rank != 0
        global_shape = get_global_grid_shape(field.dist, field.domain; scales=field.scales)
        global_data = zeros(eltype(get_local_data(get_grid_data(field))), global_shape...)
    end

    # Scatter data to all processes
    local_data = scatter_array(field.dist, global_data)

    ensure_layout!(field, :g)

    # CRITICAL: Validate that scattered data shape matches field storage shape
    # scatter_array uses default decomposition (LAST dims for PencilArrays, FIRST dims for GPU+MPI),
    # but the field may have been created with a different decomp_index. If shapes don't match,
    # the data would be incorrectly distributed.
    field_shape = size(get_local_data(get_grid_data(field)))
    scatter_shape = size(local_data)
    if field_shape != scatter_shape
        error("load_field! decomposition mismatch: field storage has shape $field_shape but " *
              "scatter_array produced shape $scatter_shape. This can happen when the field was " *
              "created with a non-default decomp_index (e.g., decomp_index=1 for pencil decomposition). " *
              "For fields with custom pencil decomposition, use PencilArrays.scatter! directly with " *
              "the field's underlying Pencil configuration.")
    end

    set_local_data!(get_grid_data(field), local_data)
end
