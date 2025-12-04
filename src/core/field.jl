"""
Field classes for data fields

Translated from dedalus/core/field.py with PencilArrays integration
"""

using PencilArrays
using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays
using HDF5
using LoopVectorization  # For SIMD loops

abstract type Operand end

mutable struct ScalarField <: Operand
    dist::Distributor
    name::String
    bases::Tuple{Vararg{Basis}}
    domain::Union{Nothing, Domain}
    dtype::Type

    # Data storage - both grid and coefficient layouts
    # For MPI: Store Pencil objects to maintain distribution information
    # For serial: Store regular arrays
    data_g::Union{Nothing, AbstractArray, PencilArrays.Pencil}  # Grid space data
    data_c::Union{Nothing, AbstractArray, PencilArrays.Pencil}  # Coefficient space data

    # Layout information
    layout::Union{Nothing, Layout}
    current_layout::Symbol  # :g for grid, :c for coefficient

    # Scale information (following Dedalus pattern)
    scales::Union{Nothing, Tuple{Vararg{Float64}}}  # Current scales for each dimension

    function ScalarField(dist::Distributor, name::String="field", bases::Tuple{Vararg{Basis}}=(),
                         dtype::Type=dist.dtype)
        domain = length(bases) > 0 ? Domain(dist, bases) : nothing
        layout = length(bases) > 0 ? get_layout(dist, bases, dtype) : nothing

        # Initialize with grid layout
        data_g = nothing
        data_c = nothing
        current_layout = :g

        # Initialize scales (following Dedalus pattern: (1,) * dist.dim)
        initial_scales = length(bases) > 0 ? tuple(ones(Float64, dist.dim)...) : nothing

        field = new(dist, name, bases, domain, dtype, data_g, data_c, layout, current_layout, initial_scales)
        
        # Allocate data if we have a domain
        if domain !== nothing
            allocate_data!(field)
        end
        
        return field
    end
end

mutable struct VectorField <: Operand
    dist::Distributor
    coordsys::CoordinateSystem
    name::String
    bases::Tuple{Vararg{Basis}}
    domain::Union{Nothing, Domain}
    dtype::Type
    
    # Component fields
    components::Vector{ScalarField}

    function VectorField(dist::Distributor, coordsys::CoordinateSystem, name::String="vector",
                         bases::Tuple{Vararg{Basis}}=(), dtype::Type=dist.dtype)
        domain = length(bases) > 0 ? Domain(dist, bases) : nothing

        # Create component fields
        components = ScalarField[]
        for (i, coord_name) in enumerate(coordsys.names)
            component_name = "$(name)_$coord_name"
            component = ScalarField(dist, component_name, bases, dtype)
            push!(components, component)
        end

        new(dist, coordsys, name, bases, domain, dtype, components)
    end
end

mutable struct TensorField <: Operand
    dist::Distributor
    coordsys::CoordinateSystem
    name::String
    bases::Tuple{Vararg{Basis}}
    domain::Union{Nothing, Domain}
    dtype::Type
    
    # Component fields as matrix
    components::Matrix{ScalarField}

    function TensorField(dist::Distributor, coordsys::CoordinateSystem, name::String="tensor",
                         bases::Tuple{Vararg{Basis}}=(), dtype::Type=dist.dtype)
        domain = length(bases) > 0 ? Domain(dist, bases) : nothing

        # Create component fields
        dim = coordsys.dim
        components = Matrix{ScalarField}(undef, dim, dim)
        for i in 1:dim, j in 1:dim
            component_name = "$(name)_$(coordsys.names[i])$(coordsys.names[j])"
            components[i,j] = ScalarField(dist, component_name, bases, dtype)
        end

        new(dist, coordsys, name, bases, domain, dtype, components)
    end
end

struct LockedField <: Operand
    field::ScalarField
    layout::Symbol
    
    function LockedField(field::ScalarField, layout::Symbol)
        new(field, layout)
    end
end

# Data allocation and management
function allocate_data!(field::ScalarField)
    """
    Allocate data for field following proper PencilArrays pattern.

    Key principles:
    1. For MPI (use_pencil_arrays=true): Store Pencil objects to maintain distribution
    2. For serial: Use regular arrays
    3. NEVER convert Pencil to Array - work with pencil.data for local access
    """
    if field.domain === nothing
        return
    end

    global_shape = global_shape(field.domain)

    if field.dist.use_pencil_arrays
        # CORRECT: Store Pencil objects directly for MPI parallelization
        # The Pencil object maintains decomposition information needed for:
        # - Transpose operations between decompositions
        # - PencilFFT transforms
        # - MPI communication patterns

        field.data_g = create_pencil(field.dist, global_shape, 1, dtype=field.dtype)
        field.data_c = create_pencil(field.dist, global_shape, 1, dtype=field.dtype)

        # For local computations on pencil.data
    else
        # Serial computation
        local_size = get_local_array_size(field.dist, global_shape)
        field.data_g = zeros(field.dtype, local_size...)
        field.data_c = zeros(field.dtype, local_size...)
    end
end

function get_local_array_size(dist::Distributor, global_shape::Tuple)
    """Get local array size for this process"""
    # Simplified - in practice would depend on MPI decomposition
    return global_shape
end

function preset_scales!(field::ScalarField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    """
    Set new transform scales without data transformation.
    Following Dedalus implementation in field.py:498-515
    """
    new_scales = remedy_scales(field.dist, scales)
    old_scales = field.scales
    
    # Return if scales are unchanged
    if new_scales == old_scales
        return field
    end
    
    # Update scales
    field.scales = new_scales
    
    # Note: In full implementation, this would:
    # 1. Get required buffer size: buffer_size = dist.buffer_size(domain, new_scales, dtype=dtype)
    # 2. Allocate new buffer if needed
    # 3. Reset layout to build new data view: preset_layout(layout)
    
    @debug "Updated field scales" old_scales new_scales
    return field
end

function set_scales!(field::ScalarField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    """
    Change data to specified scales.
    Following Dedalus implementation in field.py:631-649
    """
    # Remedy scales
    new_scales = remedy_scales(field.dist, scales)
    old_scales = field.scales
    
    # Quit if new scales aren't new
    if new_scales == old_scales
        return field
    end
    
    # In full implementation, this would:
    # 1. Forward transform until remaining scales match
    # 2. Handle coefficient space transforms as needed
    # 3. Copy data with scale change
    
    # For now, use preset_scales approach
    old_data_g = field.data_g !== nothing ? copy(field.data_g) : nothing
    old_data_c = field.data_c !== nothing ? copy(field.data_c) : nothing
    
    preset_scales!(field, scales)
    
    # Copy over data (in full implementation this would handle transforms)
    if old_data_g !== nothing && field.data_g !== nothing
        copyto!(field.data_g, old_data_g)
    end
    if old_data_c !== nothing && field.data_c !== nothing
        copyto!(field.data_c, old_data_c)
    end
    
    @debug "Changed field scales with data" old_scales new_scales
    return field
end

# Alias for compatibility  
change_scales!(field::ScalarField, scales) = set_scales!(field, scales)

# VectorField scaling methods
function preset_scales!(field::VectorField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    """Set scales for all vector field components."""
    for component in field.components
        preset_scales!(component, scales)
    end
    return field
end

function set_scales!(field::VectorField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    """Change scales for all vector field components."""
    for component in field.components
        set_scales!(component, scales)
    end
    return field
end

change_scales!(field::VectorField, scales) = set_scales!(field, scales)

# Helper functions for safe data access from Pencil or Array
"""
    get_local_data(field_data)

Get local data array from either a Pencil object or regular Array.
For Pencil: returns pencil.data (local portion on this MPI rank)
For Array: returns the array itself
"""
function get_local_data(field_data::PencilArrays.Pencil)
    return parent(field_data)  # Get underlying local array
end

function get_local_data(field_data::AbstractArray)
    return field_data
end

function get_local_data(field_data::Nothing)
    return nothing
end

"""
    set_local_data!(field_data, values)

Set local data in either a Pencil object or regular Array.
"""
function set_local_data!(field_data::PencilArrays.Pencil, values)
    parent(field_data) .= values
    return field_data
end

function set_local_data!(field_data::AbstractArray, values)
    field_data .= values
    return field_data
end

# Data access and manipulation
function Base.getindex(field::ScalarField, layout::String)
    """
    Get data in specified layout.

    Returns local data if using PencilArrays (MPI), otherwise returns full array.
    For user code operating on local data, this is the correct access pattern.
    """
    if layout == "g"
        ensure_layout!(field, :g)
        return get_local_data(field.data_g)
    elseif layout == "c"
        ensure_layout!(field, :c)
        return get_local_data(field.data_c)
    else
        throw(ArgumentError("Unknown layout: $layout"))
    end
end

function Base.setindex!(field::ScalarField, values, layout::String)
    """
    Set data in specified layout.

    Properly handles both Pencil objects (MPI) and regular arrays.
    """
    if layout == "g"
        ensure_layout!(field, :g)
        set_local_data!(field.data_g, values)
        field.current_layout = :g
    elseif layout == "c"
        ensure_layout!(field, :c)
        set_local_data!(field.data_c, values)
        field.current_layout = :c
    else
        throw(ArgumentError("Unknown layout: $layout"))
    end
end

function ensure_layout!(field::ScalarField, target_layout::Symbol)
    """Ensure field is in the target layout, transforming if necessary"""
    if field.current_layout == target_layout
        return
    end
    
    if target_layout == :g && field.current_layout == :c
        # Transform from coefficient to grid space
        backward_transform!(field)
    elseif target_layout == :c && field.current_layout == :g
        # Transform from grid to coefficient space
        forward_transform!(field)
    end
    
    field.current_layout = target_layout
end

function require_grid_space!(field::ScalarField, axis::Union{Int, Nothing}=nothing)
    """
    Require one axis (default: all axes) to be in grid space.
    Following Dedalus implementation in field.py:674-681
    """
    if field.domain === nothing
        return
    end
    
    if axis === nothing
        # Require all axes to be in grid space
        while field.current_layout != :g
            towards_grid_space!(field)
        end
    else
        # For specific axis (simplified - would need layout tracking per axis)
        towards_grid_space!(field)
    end
end

function require_coeff_space!(field::ScalarField, axis::Union{Int, Nothing}=nothing)
    """
    Require one axis (default: all axes) to be in coefficient space.
    Following Dedalus implementation in field.py:683-690
    """
    if field.domain === nothing
        return
    end
    
    if axis === nothing
        # Require all axes to be in coefficient space  
        while field.current_layout != :c
            towards_coeff_space!(field)
        end
    else
        # For specific axis (simplified - would need layout tracking per axis)
        towards_coeff_space!(field)
    end
end

function towards_grid_space!(field::ScalarField)
    """
    Change to next layout towards grid space.
    Following Dedalus implementation in field.py:664-667
    """
    if field.current_layout == :c
        # Transform from coefficient to grid space
        backward_transform_axis!(field)
        field.current_layout = :g
    end
end

function towards_coeff_space!(field::ScalarField)
    """
    Change to next layout towards coefficient space.
    Following Dedalus implementation in field.py:669-672
    """
    if field.current_layout == :g
        # Transform from grid to coefficient space
        forward_transform_axis!(field)
        field.current_layout = :c
    end
end

function forward_transform_axis!(field::ScalarField)
    """
    Forward transform field using PencilFFTs for parallel transforms.

    CORRECT PencilFFTs usage pattern:
    1. Input/output are Pencil objects (NOT arrays)
    2. PencilFFT automatically handles:
       - Required transpose operations between decompositions
       - Multi-dimensional FFT across decomposed axes
       - Scaling and normalization
    3. For 2D: Enables BOTH vertical and horizontal parallelization

    Following Dedalus distributor pattern in distributor.py:636-649
    """
    if field.domain === nothing || field.bases === ()
        return
    end

    # Use PencilFFTs-based transforms from the distributor's transform plans
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # CORRECT: Apply PencilFFT to Pencil objects
            # PencilFFT handles transposes internally

            if field.dist.use_pencil_arrays && isa(field.data_g, PencilArrays.Pencil)
                # Apply forward transform: grid space (physical) → coefficient space (spectral)
                # Note: mul! is the in-place version
                # Result goes into data_c pencil
                if field.data_c === nothing || !isa(field.data_c, PencilArrays.Pencil)
                    # Allocate output pencil if needed
                    field.data_c = create_pencil(field.dist, size(field.data_g), 1, dtype=field.dtype)
                end

                # Apply PencilFFT: transforms AND transposes as needed
                mul!(field.data_c, transform, field.data_g)

                @debug "Applied PencilFFT forward transform" typeof(transform) size(field.data_g)
            else
                @warn "Cannot apply PencilFFT: field.data_g is not a Pencil object"
            end
            field.current_layout = :c
            return  # Found and applied transform
        end
    end
    
    # Fallback: copy data if no PencilFFT transforms available
    if field.data_c !== nothing && field.data_g !== nothing
        copyto!(field.data_c, field.data_g)
    end
end

function backward_transform_axis!(field::ScalarField)
    """
    Backward transform field using PencilFFTs for parallel transforms.

    CORRECT PencilFFTs usage pattern:
    1. Inverse FFT: coefficient space (spectral) → grid space (physical)
    2. Uses ldiv! (\\) for backward transform
    3. Maintains Pencil objects throughout

    Following Dedalus distributor pattern in distributor.py:621-634
    """
    if field.domain === nothing || field.bases === ()
        return
    end

    # Use PencilFFTs-based transforms from the distributor's transform plans
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # CORRECT: Apply inverse PencilFFT to Pencil objects

            if field.dist.use_pencil_arrays && isa(field.data_c, PencilArrays.Pencil)
                # Apply backward transform: coefficient space → grid space
                if field.data_g === nothing || !isa(field.data_g, PencilArrays.Pencil)
                    # Allocate output pencil if needed
                    field.data_g = create_pencil(field.dist, size(field.data_c), 1, dtype=field.dtype)
                end

                # Apply inverse PencilFFT: transforms AND transposes as needed
                # ldiv! is in-place inverse (like \ but in-place)
                ldiv!(field.data_g, transform, field.data_c)

                @debug "Applied PencilFFT backward transform" typeof(transform) size(field.data_c)
            else
                @warn "Cannot apply PencilFFT: field.data_c is not a Pencil object"
            end
            field.current_layout = :g
            return  # Found and applied transform
        end
    end

    # Fallback: copy data if no PencilFFT transforms available
    if field.data_g !== nothing && field.data_c !== nothing
        copyto!(field.data_g, field.data_c)
    end
end

# VectorField transform methods
function require_grid_space!(field::VectorField, axis::Union{Int, Nothing}=nothing)
    """Require vector field components to be in grid space."""
    for component in field.components
        require_grid_space!(component, axis)
    end
end

function require_coeff_space!(field::VectorField, axis::Union{Int, Nothing}=nothing)
    """Require vector field components to be in coefficient space."""
    for component in field.components
        require_coeff_space!(component, axis)
    end
end

function forward_transform!(field::VectorField)
    """Transform vector field from grid to coefficient space."""
    for component in field.components
        forward_transform!(component)
    end
end

function backward_transform!(field::VectorField)
    """Transform vector field from coefficient to grid space."""
    for component in field.components
        backward_transform!(component)
    end
end

# Field operations
function fill_random!(field::ScalarField, layout::String="g"; seed=nothing, distribution="normal", scale=1.0)
    """Fill field with random data"""
    if seed !== nothing
        Random.seed!(seed)
    end
    
    data = field[layout]
    if distribution == "normal"
        randn!(data)
    elseif distribution == "uniform"
        rand!(data)
    end
    data .*= scale
end

function integrate(field::ScalarField, axes=:)
    """Integrate field over specified axes"""
    if field.domain === nothing
        return 0.0
    end
    
    ensure_layout!(field, :g)
    weights = integration_weights(field.domain)
    
    result = field.data_g
    for (i, w) in enumerate(weights)
        if axes === Colon() || i in axes
            # Apply weights and sum along dimension i
            result = sum(result .* reshape(w, ntuple(j -> j==i ? length(w) : 1, ndims(result))), dims=i)
        end
    end
    
    return result
end

# Vector field operations
function Base.getindex(field::VectorField, i::Int)
    """Get component field"""
    return field.components[i]
end

function Base.setindex!(field::VectorField, value, i::Int)
    """Set component field"""
    field.components[i] = value
end

function Base.getindex(field::VectorField, layout::String)
    """Get all components in specified layout"""
    return [comp[layout] for comp in field.components]
end

# Tensor field operations  
function Base.getindex(field::TensorField, i::Int, j::Int)
    """Get tensor component"""
    return field.components[i, j]
end

function Base.setindex!(field::TensorField, value, i::Int, j::Int)
    """Set tensor component"""
    field.components[i, j] = value
end

# Field arithmetic
function Base.:+(a::ScalarField, b::ScalarField)
    """Add two scalar fields"""
    if a.bases != b.bases
        throw(ArgumentError("Cannot add fields with different bases"))
    end

    result = ScalarField(a.dist, "$(a.name)_plus_$(b.name)", a.bases, a.dtype)
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)
    ensure_layout!(result, :g)

    result.data_g .= a.data_g .+ b.data_g

    return result
end

function Base.:-(a::ScalarField, b::ScalarField)
    """Subtract two scalar fields"""
    if a.bases != b.bases
        throw(ArgumentError("Cannot subtract fields with different bases"))
    end

    result = ScalarField(a.dist, "$(a.name)_minus_$(b.name)", a.bases, a.dtype)
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)
    ensure_layout!(result, :g)

    result.data_g .= a.data_g .- b.data_g

    return result
end

function Base.:*(a::ScalarField, b::Union{Real, ScalarField})
    """Multiply scalar field by scalar or another field"""
    if isa(b, Real)
        result = ScalarField(a.dist, "$(a.name)_times_$(b)", a.bases, a.dtype)
        ensure_layout!(a, :g)
        ensure_layout!(result, :g)

        result.data_g .= b .* a.data_g

        return result
    else
        if a.bases != b.bases
            throw(ArgumentError("Cannot multiply fields with different bases"))
        end
        result = ScalarField(a.dist, "$(a.name)_times_$(b.name)", a.bases, a.dtype)
        ensure_layout!(a, :g)
        ensure_layout!(b, :g)
        ensure_layout!(result, :g)

        result.data_g .= a.data_g .* b.data_g

        # Apply basic dealiasing for spectral methods (3/2 rule)
        if has_spectral_bases(a) && length(a.data_g) > 64
            apply_dealiasing_to_product!(result)
        end

        return result
    end
end

# I/O operations
function save_field(field::ScalarField, filename::String, dataset_name::String="field")
    """Save field to HDF5 file"""
    ensure_layout!(field, :g)
    
    # Gather data to root process for writing
    global_data = gather_array(field.dist, Array(field.data_g))
    
    if field.dist.rank == 0
        h5open(filename, "w") do file
            write(file, dataset_name, global_data)
        end
    end
end

function load_field!(field::ScalarField, filename::String, dataset_name::String="field")
    """Load field from HDF5 file"""
    if field.dist.rank == 0
        global_data = h5open(filename, "r") do file
            read(file, dataset_name)
        end
    else
        global_data = nothing
    end
    
    # Scatter data to all processes
    local_data = scatter_array(field.dist, global_data)
    
    ensure_layout!(field, :g)
    field.data_g .= local_data
end

# Optimization support functions
function has_spectral_bases(field::ScalarField)
    """Check if field uses spectral bases that benefit from dealiasing"""
    for basis in field.bases
        if isa(basis, Union{RealFourier, ComplexFourier, ChebyshevT})
            return true
        end
    end
    return false
end

function apply_dealiasing_to_product!(field::ScalarField)
    """Apply 3/2 rule dealiasing to nonlinear product"""
    # Apply 2/3 rule cutoff for dealiasing
    # This removes the highest 1/3 of modes in each direction
    cutoff_scale = 2.0/3.0
    apply_spectral_cutoff!(field, cutoff_scale)
end

function apply_spectral_cutoff!(field::ScalarField, cutoff_scales::Union{Float64, Tuple{Vararg{Float64}}})
    """
    Apply spectral cutoff by zeroing modes above specified relative scales.
    Following Dedalus low_pass_filter implementation.
    """
    # Store original scales
    original_scales = field.scales
    
    # Normalize cutoff_scales to tuple
    if isa(cutoff_scales, Float64)
        scales = tuple(fill(cutoff_scales, length(field.bases))...)
    else
        scales = cutoff_scales
    end
    
    # Apply low-pass filter by changing scales
    set_scales!(field, scales)
    require_grid_space!(field)
    set_scales!(field, original_scales)
end

function low_pass_filter!(field::ScalarField; shape=nothing, scales=nothing)
    """
    Apply a spectral low-pass filter by zeroing modes above specified relative scales.
    The scales can be specified directly or deduced from a specified global grid shape.
    Following Dedalus field.py:945-968 implementation.
    """
    original_scales = field.scales
    
    # Determine scales from shape
    if shape !== nothing
        if scales !== nothing
            error("Specify either shape or scales.")
        end
        # Get global grid shape
        global_shape = get_global_grid_shape(field.dist, field.domain, scales=ones(Float64, length(field.bases)))
        scales = tuple((shape ./ global_shape)...)
    end
    
    # Apply low-pass filter by changing scales
    set_scales!(field, scales)
    require_grid_space!(field)
    set_scales!(field, original_scales)
end

function high_pass_filter!(field::ScalarField; shape=nothing, scales=nothing)
    """
    Apply a spectral high-pass filter by zeroing modes below specified relative scales.
    Following Dedalus field.py:969-984 implementation.
    """
    # Store original data in coefficient space
    require_coeff_space!(field)
    data_orig = copy(get_data(field, :c))
    
    # Apply low-pass filter
    low_pass_filter!(field; shape=shape, scales=scales)
    
    # Get filtered data in coefficient space
    require_coeff_space!(field)
    data_filt = copy(get_data(field, :c))
    
    # High-pass = original - low-pass
    field_data = get_data(field, :c)
    field_data .= data_orig .- data_filt
end

function get_data(field::ScalarField, layout::Symbol)
    """Get field data in specified layout"""
    if layout == :g
        ensure_layout!(field, :g)
        return field.data_g
    elseif layout == :c
        ensure_layout!(field, :c)
        return field.data_c
    else
        throw(ArgumentError("Unknown layout: $layout"))
    end
end

function get_global_grid_shape(dist::Distributor, domain::Domain; scales=ones(Float64, length(domain.bases)))
    """Get global grid shape for a domain with given scales."""
    # This is a simplified implementation - should get from layout
    return tuple([Int(round(basis.meta.size * scales[i])) for (i, basis) in enumerate(domain.bases)]...)
end

# LoopVectorization functions
@inline function vectorized_add!(result::AbstractArray, a::AbstractArray, b::AbstractArray)
    """Vectorized addition: result = a + b"""
    if length(result) > 100
        @turbo for i in eachindex(result, a, b)
            result[i] = a[i] + b[i]
        end
    else
        result .= a .+ b  # Use broadcasting for very small arrays
    end
end

@inline function vectorized_sub!(result::AbstractArray, a::AbstractArray, b::AbstractArray)
    """Vectorized subtraction: result = a - b"""
    if length(result) > 100
        @turbo for i in eachindex(result, a, b)
            result[i] = a[i] - b[i]
        end
    else
        result .= a .- b
    end
end

@inline function vectorized_mul!(result::AbstractArray, a::AbstractArray, b::AbstractArray)
    """Vectorized multiplication: result = a * b (element-wise)"""
    if length(result) > 100
        @turbo for i in eachindex(result, a, b)
            result[i] = a[i] * b[i]
        end
    else
        result .= a .* b
    end
end

@inline function vectorized_scale!(result::AbstractArray, a::AbstractArray, α::Real)
    """Vectorized scaling: result = α * a"""
    if length(result) > 100
        @turbo for i in eachindex(result, a)
            result[i] = α * a[i]
        end
    else
        result .= α .* a
    end
end

@inline function vectorized_axpy!(result::AbstractArray, α::Real, x::AbstractArray, y::AbstractArray)
    """Vectorized AXPY: result = α*x + y"""
    if length(result) > 100
        @turbo for i in eachindex(result, x, y)
            result[i] = α * x[i] + y[i]
        end
    else
        result .= α .* x .+ y
    end
end

@inline function vectorized_linear_combination!(result::AbstractArray, α::Real, a::AbstractArray, β::Real, b::AbstractArray)
    """Vectorized linear combination: result = α*a + β*b"""
    if length(result) > 100
        @turbo for i in eachindex(result, a, b)
            result[i] = α * a[i] + β * b[i]
        end
    else
        result .= α .* a .+ β .* b
    end
end

# Fast field arithmetic with multi-tier implementation
function fast_axpy!(α::Real, x::ScalarField, y::ScalarField)
    """Fast y ← α*x + y using best available method"""
    ensure_layout!(x, :g)
    ensure_layout!(y, :g)

    n = length(x.data_g)
    if n > 2000  # Use BLAS for very large arrays
        BLAS.axpy!(α, x.data_g, y.data_g)
    elseif n > 100  # Use LoopVectorization for medium arrays
        @turbo for i in eachindex(y.data_g, x.data_g)
            y.data_g[i] = y.data_g[i] + α * x.data_g[i]
        end
    else  # Use broadcasting for small arrays
        y.data_g .+= α .* x.data_g
    end
end

# Coordinate system utilities (moved from coords.jl to avoid circular dependency)
function unit_vector_fields(coordsys::CoordinateSystem, dist)
    """
    Return unit vector fields for each coordinate direction.
    Following Dedalus implementation in coords.py:183

    Note: This function was moved from coords.jl to field.jl to avoid circular dependency,
    as it needs VectorField which is defined in field.jl.
    """
    fields = VectorField[]
    for (i, coord) in enumerate(coords(coordsys))
        # Create vector field for each coordinate direction
        ec = VectorField(dist, coordsys, name="e$(coord.name)")

        # Set the i-th component to 1 (unit vector in that direction)
        # In Dedalus: ec['g'][i] = 1
        # This means the i-th component of the vector field is set to 1
        for j in 1:length(ec.components)
            if j == i
                # Set the i-th component to 1 (unit vector in that direction)
                fill!(ec.components[j]["g"], 1.0)
            else
                # Set all other components to 0
                fill!(ec.components[j]["g"], 0.0)
            end
        end

        push!(fields, ec)
    end
    return tuple(fields...)
end
