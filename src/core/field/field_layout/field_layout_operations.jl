"""
    Field layout operations

This file contains random-fill helpers, integration, and the VectorField /
TensorField convenience accessors that sit on top of layout-aware storage.
"""

# Field operations
"""
    fill_random!(field, layout="g"; seed=nothing, distribution="normal", scale=1.0, reproducible=true)

Fill field with random data in the specified layout.
Follows fill_random API for familiar usage.

# Arguments
- `field`: ScalarField or VectorField to fill
- `layout`: Layout to fill ("g" for grid, "c" for coefficient)
- `seed`: Random seed for reproducibility (optional)
- `distribution`: Distribution type - "normal", "uniform", or "standard_normal"
- `scale`: Scale factor to multiply random values
- `reproducible`: If true (default), generates identical global random field regardless
  of MPI decomposition. Each grid point gets a deterministic random value based on its
  global index. If false, uses rank-local seeding (faster but MPI-dependent).

# Example
```julia
# Fill with reproducible random noise (same result with 1 or 4 MPI ranks)
fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3)

# Fill with rank-local random noise (faster, but result varies with MPI configuration)
fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3, reproducible=false)
```
"""
function fill_random!(field::ScalarField, layout::String="g";
                      seed::Union{Int, Nothing}=nothing,
                      distribution::String="normal",
                      scale::Real=1.0,
                      reproducible::Bool=true)
    ensure_layout!(field, Symbol(layout))
    data = layout == "g" ? get_grid_data(field) : get_coeff_data(field)

    if seed !== nothing && reproducible
        # Reproducible mode: global index-based seeding — produces identical results
        # regardless of MPI decomposition, INCLUDING np=1 (must use the SAME algorithm
        # as np>1, else the serial result diverges from the distributed one).
        _fill_random_reproducible!(data, field, layout, seed, distribution, scale)
    else
        # Standard mode: rank-local seeding (original behavior)
        if seed !== nothing
            Random.seed!(seed)
        end

        if distribution == "normal" || distribution == "standard_normal"
            randn!(data)
        elseif distribution == "uniform"
            rand!(data)
            data .-= 0.5  # Center around zero
            data .*= 2.0  # Scale to [-1, 1]
        else
            throw(ArgumentError("Unknown distribution: $distribution. Use 'normal' or 'uniform'."))
        end

        data .*= scale
    end

    return field
end

"""
    _fill_random_reproducible!(data, field, layout, seed, distribution, scale)

Fill data array with reproducible random values using global index-based seeding.
Produces identical results regardless of MPI decomposition.

Uses a simple but effective approach: generates random values point-by-point
using a hash of (seed, global_indices) as the per-point seed.
"""
function _fill_random_reproducible!(data::AbstractArray, field::ScalarField,
                                    layout::String, seed::Int,
                                    distribution::String, scale::Real)
    dist = field.dist

    # Get global shape from domain
    if field.domain === nothing
        # No domain info - fall back to standard random fill
        Random.seed!(seed)
        if distribution == "normal" || distribution == "standard_normal"
            randn!(data)
        elseif distribution == "uniform"
            rand!(data)
            data .-= 0.5
            data .*= 2.0
        end
        data .*= scale
        return
    end

    gshape = global_shape(field.domain)
    local_size = size(data)
    ndims_data = ndims(data)

    # Compute global index offsets for this rank
    # For each dimension, find the starting global index
    global_offsets = zeros(Int, ndims_data)
    for dim in 1:min(ndims_data, length(gshape))
        start_idx, _ = get_local_range(dist, gshape[dim], dim)
        global_offsets[dim] = start_idx - 1  # Convert to 0-based offset
    end

    # Transfer data to CPU for random generation if on GPU
    arch = dist.architecture
    cpu_data = is_gpu(arch) ? Array(data) : data

    # Fill each point using deterministic RNG based on global index
    # Use a simple hash: seed + linear_global_index
    for I in CartesianIndices(cpu_data)
        # Compute global linear index using column-major ordering
        global_idx = 0
        stride = 1
        for dim in 1:ndims_data
            global_coord = I[dim] + global_offsets[dim] - 1  # 0-based global coordinate
            global_idx += global_coord * stride
            stride *= dim <= length(gshape) ? gshape[dim] : local_size[dim]
        end

        # Use deterministic seed for this point
        point_seed = seed + global_idx
        Random.seed!(point_seed)

        if distribution == "normal" || distribution == "standard_normal"
            cpu_data[I] = randn() * scale
        elseif distribution == "uniform"
            cpu_data[I] = (rand() - 0.5) * 2.0 * scale
        else
            throw(ArgumentError("Unknown distribution: $distribution. Use 'normal' or 'uniform'."))
        end
    end

    # Copy back to GPU if needed
    if is_gpu(arch)
        copyto!(data, on_architecture(arch, cpu_data))
    end
end

function fill_random!(field::VectorField, layout::String="g";
                      seed::Union{Int, Nothing}=nothing,
                      distribution::String="normal",
                      scale::Real=1.0,
                      reproducible::Bool=true)
    for (i, component) in enumerate(field.components)
        # Use different seed for each component to get uncorrelated noise
        # Multiply by large prime to ensure non-overlapping seed ranges
        comp_seed = seed !== nothing ? seed + (i - 1) * 1000003 : nothing
        fill_random!(component, layout; seed=comp_seed, distribution=distribution,
                     scale=scale, reproducible=reproducible)
    end
    return field
end

"""
Integrate a field over the specified axes (default: all axes → a scalar).

Delegates to the operator-path `integrate`/`evaluate_integrate`, which applies the
basis quadrature weights to each rank's LOCAL slab and reduces collectively across
MPI ranks. The previous direct implementation multiplied a per-rank local slab by
GLOBAL-length weight vectors (DimensionMismatch on decomposed axes) and never
reduced across ranks (each rank returned only its partial sum), and returned a
1×1×… array instead of a scalar for full integration.
"""
# `axes` is constrained to axis specifiers (Colon / integer indices) ONLY. Without
# this, the untyped `axes` also matched a `Coordinate`/`CartesianCoordinates`, making
# `integrate(field, coord)` ambiguous with the operator-path `integrate(::Operand, ::Coordinate)`
# (constructors.jl) — neither method is more specific (ScalarField wins arg 1, Coordinate
# wins arg 2). Constraining the type routes Coordinate/coordsys calls to the operator path.
function integrate(field::ScalarField,
                   axes::Union{Colon,Integer,AbstractVector{<:Integer},Tuple{Vararg{Integer}}}=:)
    if field.domain === nothing
        return 0.0
    end

    coords_all = field.dist.coords
    sel = if axes === Colon()
        coords_all
    else
        tuple((coords_all[i] for i in axes)...)
    end
    # Build the lazy Integrate operator directly (constructing it via the 2-arg
    # `integrate` would be ambiguous with this very method); evaluate_integrate
    # performs the MPI-correct weighted reduction over each rank's local slab.
    # Returns a scalar for full integration, or a ScalarField over the remaining
    # axes for partial integration.
    result = evaluate_integrate(Integrate(field, sel), :g)
    ensure_layout!(field, :g)   # restore the input field to grid layout (post-condition)
    return result
end

# Vector field operations
"""Get component field"""
function Base.getindex(field::VectorField, i::Int)
    return field.components[i]
end

"""Set component field"""
function Base.setindex!(field::VectorField, value, i::Int)
    field.components[i] = value
end

"""Get all components in specified layout"""
function Base.getindex(field::VectorField, layout::String)
    return [comp[layout] for comp in field.components]
end

"""
    Base.getproperty(field::VectorField, name::Symbol)

Access vector field components by coordinate name.

# Examples
```julia
u = VectorField(domain, "u")   # 2D field with coordinates (x, z)
u.x                             # first component (same as u[1] or u.components[1])
u.z                             # second component
```
"""
function Base.getproperty(field::VectorField, name::Symbol)
    if hasfield(typeof(field), name)
        return getfield(field, name)
    end
    # Look up coordinate name in coordsys
    cs = getfield(field, :coordsys)
    for (i, cname) in enumerate(cs.names)
        if Symbol(cname) == name
            return getfield(field, :components)[i]
        end
    end
    throw(ArgumentError(
        "VectorField '$(getfield(field, :name))' has no component '$name'. " *
        "Available components: $(join(cs.names, ", "))"))
end

function Base.propertynames(field::VectorField, private::Bool=false)
    coord_syms = Symbol.(getfield(field, :coordsys).names)
    if private
        return (fieldnames(typeof(field))..., coord_syms...)
    else
        return (fieldnames(typeof(field))..., coord_syms...)
    end
end

# Tensor field operations  
"""Get tensor component"""
function Base.getindex(field::TensorField, i::Int, j::Int)
    return field.components[i, j]
end

"""Set tensor component"""
function Base.setindex!(field::TensorField, value, i::Int, j::Int)
    field.components[i, j] = value
end

# Static name for temporary arithmetic fields — avoids string allocation per operation
