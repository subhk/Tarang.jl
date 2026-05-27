"""
    Field vectorized helpers

This file contains small vectorized array kernels, `fast_axpy!`, and
`unit_vector_fields`.
"""

# LoopVectorization functions
@inline """Vectorized addition: result = a + b"""
function vectorized_add!(result::AbstractArray, a::AbstractArray, b::AbstractArray)
    if is_gpu_array(result) || is_gpu_array(a) || is_gpu_array(b)
        result .= a .+ b
    elseif length(result) > 100
        @turbo for i in eachindex(result, a, b)
            result[i] = a[i] + b[i]
        end
    else
        result .= a .+ b  # Use broadcasting for very small arrays
    end
end

@inline """Vectorized subtraction: result = a - b"""
function vectorized_sub!(result::AbstractArray, a::AbstractArray, b::AbstractArray)
    if is_gpu_array(result) || is_gpu_array(a) || is_gpu_array(b)
        result .= a .- b
    elseif length(result) > 100
        @turbo for i in eachindex(result, a, b)
            result[i] = a[i] - b[i]
        end
    else
        result .= a .- b
    end
end

@inline """Vectorized multiplication: result = a * b (element-wise)"""
function vectorized_mul!(result::AbstractArray, a::AbstractArray, b::AbstractArray)
    if is_gpu_array(result) || is_gpu_array(a) || is_gpu_array(b)
        result .= a .* b
    elseif length(result) > 100
        @turbo for i in eachindex(result, a, b)
            result[i] = a[i] * b[i]
        end
    else
        result .= a .* b
    end
end

@inline """Vectorized scaling: result = α * a"""
function vectorized_scale!(result::AbstractArray, a::AbstractArray, α::Real)
    if is_gpu_array(result) || is_gpu_array(a)
        result .= α .* a
    elseif length(result) > 100
        @turbo for i in eachindex(result, a)
            result[i] = α * a[i]
        end
    else
        result .= α .* a
    end
end

@inline """Vectorized AXPY: result = α*x + y"""
function vectorized_axpy!(result::AbstractArray, α::Real, x::AbstractArray, y::AbstractArray)
    if is_gpu_array(result) || is_gpu_array(x) || is_gpu_array(y)
        result .= α .* x .+ y
    elseif length(result) > 100
        @turbo for i in eachindex(result, x, y)
            result[i] = α * x[i] + y[i]
        end
    else
        result .= α .* x .+ y
    end
end

@inline """Vectorized linear combination: result = α*a + β*b"""
function vectorized_linear_combination!(result::AbstractArray, α::Real, a::AbstractArray, β::Real, b::AbstractArray)
    if is_gpu_array(result) || is_gpu_array(a) || is_gpu_array(b)
        result .= α .* a .+ β .* b
    elseif length(result) > 100
        @turbo for i in eachindex(result, a, b)
            result[i] = α * a[i] + β * b[i]
        end
    else
        result .= α .* a .+ β .* b
    end
end

# Fast field arithmetic with multi-tier implementation
"""Fast y ← α*x + y using best available method"""
function fast_axpy!(α::Real, x::ScalarField, y::ScalarField)
    ensure_layout!(x, :g)
    ensure_layout!(y, :g)

    x_data = get_grid_data(x)
    y_data = get_grid_data(y)
    n = length(x_data)
    if is_gpu_array(x_data) || is_gpu_array(y_data)
        y_data .+= α .* x_data
    elseif n > 2000  # Use BLAS for very large arrays
        BLAS.axpy!(α, x_data, y_data)
    elseif n > 100  # Use LoopVectorization for medium arrays
        @turbo for i in eachindex(y_data, x_data)
            y_data[i] = y_data[i] + α * x_data[i]
        end
    else  # Use broadcasting for small arrays
        y_data .+= α .* x_data
    end
end

# Coordinate system utilities (moved from coords.jl to avoid circular dependency)
"""
    Return unit vector fields for each coordinate direction.
    Following implementation in coords:183

    Note: This function was moved from coords.jl to field.jl to avoid circular dependency,
    as it needs VectorField which is defined in field.jl.
    """
function unit_vector_fields(coordsys::CoordinateSystem, dist)
    fields = VectorField[]
    for (i, coord) in enumerate(coords(coordsys))
        # Create vector field for each coordinate direction
        ec = VectorField(dist, coordsys, "e$(coord.name)")

        # Set the i-th component to 1 (unit vector in that direction)
        # Implementation: ec['g'][i] = 1
        # This means the i-th component of the vector field is set to 1
        for j in 1:length(ec.components)
            comp = ec.components[j]

            # Ensure data exists even when no bases are provided (0D fields).
            # For 0D fields (constant unit vectors), use a single scalar value.
            # Guard on isempty(comp.bases) rather than data === nothing because
            # 0-D fields now carry a typed length-0 sentinel instead of nothing.
            if isempty(comp.bases)
                set_grid_data!(comp, zeros(dist.architecture, comp.dtype, 1))
                coeff_dtype = coefficient_eltype(comp.dtype)
                set_coeff_data!(comp, zeros(dist.architecture, coeff_dtype, 1))
            end

            data = get_grid_data(comp)
            if j == i
                # Set the i-th component to 1 (unit vector in that direction)
                fill!(data, one(eltype(data)))
            else
                # Set all other components to 0
                fill!(data, zero(eltype(data)))
            end
        end

        push!(fields, ec)
    end
    return tuple(fields...)
end
