# ---------------------------------------------------------------------------
# Subsystem methods
# ---------------------------------------------------------------------------

function coeff_slices(subsystem::Subsystem, domain)
    if domain === nothing
        return ntuple(_ -> Colon(), length(subsystem.group))
    end

    coeff_shape = coefficient_shape(domain)
    return ntuple(_ -> Colon(), length(coeff_shape))
end

function coeff_shape(subsystem::Subsystem, domain)
    if domain === nothing
        return ()
    end

    return coefficient_shape(domain)
end

function coeff_size(subsystem::Subsystem, domain)
    return prod(coeff_shape(subsystem, domain))
end

function field_slices(subsystem::Subsystem, field::ScalarField)
    # Component slices (none for scalar) + coefficient slices
    return coeff_slices(subsystem, field_domain(field))
end

function field_shape(subsystem::Subsystem, field::ScalarField)
    return coeff_shape(subsystem, field_domain(field))
end

function field_size(subsystem::Subsystem, field::ScalarField)
    range = get(subsystem.scalar_ranges, field, 1:0)
    if isempty(range)
        return scalar_field_dofs(field)
    else
        return length(range)
    end
end

field_size(subsystem::Subsystem, field::VectorField) = sum(field_size(subsystem, comp) for comp in field.components)
field_size(subsystem::Subsystem, field::TensorField) = sum(field_size(subsystem, comp) for comp in vec(field.components))

function field_domain(field::ScalarField)
    if hasfield(typeof(field), :domain) && field.domain !== nothing
        return field.domain
    end
    return nothing
end

function field_domain(field::VectorField)
    # Get domain from first component
    if !isempty(field.components)
        return field_domain(field.components[1])
    end
    return nothing
end

function field_domain(field::TensorField)
    # Get domain from first component
    if !isempty(field.components)
        return field_domain(field.components[1])
    end
    return nothing
end

# Fallback for other Operand types
field_domain(::Operand) = nothing

function _coeff_data(field)
    ensure_layout!(field, :c)
    if get_coeff_data(field) === nothing
        throw(ArgumentError("Field $(field.name) has no coefficient data available."))
    end
    return get_coeff_data(field)
end

function _field_coeff_vector(field::ScalarField)
    data = _coeff_data(field)
    cpu_data = is_gpu_array(data) ? get_cpu_data(data) : data
    return vec(cpu_data)
end

function _assign_coefficients_from_slice!(field::ScalarField, coeffs::AbstractArray, slice::AbstractVector{<:Number})
    target_shape = size(coeffs)
    expected = prod(target_shape)

    if length(slice) != expected
        throw(ArgumentError("Slice length $(length(slice)) does not match coefficient size $expected for field $(field.name)"))
    end

    target_eltype = eltype(coeffs)
    data_array = if target_eltype <: Real
        if any(!iszero(imag(val)) for val in slice)
            @warn "Discarding imaginary part when assigning real coefficients for field $(field.name)"
        end
        reshape(real.(slice), target_shape)
    elseif target_eltype == eltype(slice)
        reshape(slice, target_shape)
    else
        reshape(convert.(target_eltype, slice), target_shape)
    end

    arch = field.dist.architecture
    if is_gpu(arch)
        copyto!(coeffs, on_architecture(arch, data_array))
    else
        copyto!(coeffs, data_array)
    end

    field.current_layout = :c
end

"""
    gather(subsystem, fields)

Gather coefficient data from fields into a single vector.
Following subsystems:213-220.
"""
function gather(subsystem::Subsystem, fields::Vector{<:ScalarField})
    buffers = ComplexF64[]
    for field in fields
        append!(buffers, _field_coeff_vector(field))
    end
    return buffers
end

"""
    scatter(subsystem, data, fields)

Scatter vector entries back into field coefficient arrays.
Following subsystems:222-231.
"""
function scatter(subsystem::Subsystem, data::AbstractVector, fields::Vector{<:ScalarField})
    offset = 0
    for field in fields
        coeffs = _coeff_data(field)
        n = length(coeffs)
        if offset + n > length(data)
            throw(ArgumentError("Insufficient data provided for scatter."))
        end
        _assign_coefficients_from_slice!(field, coeffs, view(data, offset+1:offset+n))
        offset += n
    end
    return nothing
end

