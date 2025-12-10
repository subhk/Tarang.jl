"""
Field classes for data fields
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

    # Scale information
    scales::Union{Nothing, Tuple{Vararg{Float64}}}  # Current scales for each dimension

    function ScalarField(dist::Distributor, name::String="field", bases::Tuple{Vararg{Basis}}=(),
                         dtype::Type=dist.dtype)
        domain = length(bases) > 0 ? Domain(dist, bases) : nothing
        layout = length(bases) > 0 ? get_layout(dist, bases, dtype) : nothing

        # Initialize with grid layout
        data_g = nothing
        data_c = nothing
        current_layout = :g

        # Initialize scales: (1,) * dist.dim
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

"""
    LockedField <: Operand

A field wrapper that restricts layout and scale changes to specific allowed values.
Following Dedalus field:LockedField pattern.

This is useful for:
- Output handlers that need fields in specific layouts
- Preventing accidental layout changes during evaluation
- Enforcing coefficient-space or grid-space operations

# Example
```julia
locked = LockedField(u, :g)          # Lock to grid space
locked = LockedField(u, :c)          # Lock to coefficient space
locked = LockedField(u, :g, (1,))    # Lock layout and scales
```
"""
struct LockedField <: Operand
    field::ScalarField
    layout::Symbol
    scales::Union{Nothing, Tuple}

    function LockedField(field::ScalarField, layout::Symbol, scales::Union{Nothing, Tuple}=nothing)
        # Ensure field is in the locked layout
        ensure_layout!(field, layout)
        new(field, layout, scales)
    end
end

# LockedField convenience constructors
LockedField(field::ScalarField) = LockedField(field, field.current_layout, field.scales)

# Forward field access methods to underlying field
Base.getindex(lf::LockedField, key) = getindex(lf.field, key)
Base.size(lf::LockedField) = size(lf.field.data_g !== nothing ? lf.field.data_g : lf.field.data_c)

# Property forwarding
function Base.getproperty(lf::LockedField, s::Symbol)
    if s in (:field, :layout, :scales)
        return getfield(lf, s)
    elseif s == :name
        return lf.field.name
    elseif s == :dist
        return lf.field.dist
    elseif s == :bases
        return lf.field.bases
    elseif s == :dtype
        return lf.field.dtype
    elseif s == :domain
        return lf.field.domain
    elseif s == :data_g
        return lf.field.data_g
    elseif s == :data_c
        return lf.field.data_c
    elseif s == :current_layout
        return lf.layout  # Return locked layout, not field's current
    else
        return getfield(lf, s)
    end
end

"""
    change_scales!(lf::LockedField, new_scales)

Attempt to change scales on a locked field.
Only succeeds if new_scales matches locked scales or locked scales is nothing.
"""
function change_scales!(lf::LockedField, new_scales)
    if lf.scales !== nothing && new_scales != lf.scales
        throw(ArgumentError("Cannot change scales on LockedField from $(lf.scales) to $new_scales"))
    end
    change_scales!(lf.field, new_scales)
    return lf
end

"""
    ensure_layout!(lf::LockedField, layout::Symbol)

Attempt to change layout on a locked field.
Only succeeds if layout matches the locked layout.
"""
function ensure_layout!(lf::LockedField, layout::Symbol)
    if layout != lf.layout
        throw(ArgumentError("Cannot change layout on LockedField from $(lf.layout) to $layout"))
    end
    return lf
end

"""
    unlock(lf::LockedField)

Get the underlying unlocked field.
"""
unlock(lf::LockedField) = lf.field

# Copy methods for ScalarField
function Base.copy(field::ScalarField)
    """Create a shallow copy of ScalarField with copied data arrays"""
    new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
    if field.data_g !== nothing
        new_field.data_g = copy(field.data_g)
    end
    if field.data_c !== nothing
        new_field.data_c = copy(field.data_c)
    end
    new_field.current_layout = field.current_layout
    new_field.scales = field.scales
    return new_field
end

function Base.deepcopy(field::ScalarField)
    """Create a deep copy of ScalarField"""
    new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
    if field.data_g !== nothing
        new_field.data_g = deepcopy(field.data_g)
    end
    if field.data_c !== nothing
        new_field.data_c = deepcopy(field.data_c)
    end
    new_field.current_layout = field.current_layout
    new_field.scales = field.scales
    return new_field
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

    gshape = global_shape(field.domain)

    if field.dist.use_pencil_arrays
        # CORRECT: Store Pencil objects directly for MPI parallelization
        # The Pencil object maintains decomposition information needed for:
        # - Transpose operations between decompositions
        # - PencilFFT transforms
        # - MPI communication patterns

        field.data_g = create_pencil(field.dist, gshape, 1, dtype=field.dtype)
        field.data_c = create_pencil(field.dist, gshape, 1, dtype=field.dtype)

        # For local computations on pencil.data
    else
        # Serial computation
        local_size = get_local_array_size(field.dist, gshape)
        field.data_g = zeros(field.dtype, local_size...)
        field.data_c = zeros(field.dtype, local_size...)
    end
end

function get_local_array_size(dist::Distributor, global_shape::Tuple)
    """
    Get local array size for this process based on MPI decomposition.

    For a pencil decomposition with process mesh (P₁, P₂, ..., Pₖ),
    the global array is divided along the last k dimensions.

    For example, with a 3D array of shape (Nx, Ny, Nz) and mesh (Py, Pz):
    - x dimension is not decomposed: local_nx = Nx
    - y dimension is split among Py processes: local_ny = Ny / Py
    - z dimension is split among Pz processes: local_nz = Nz / Pz

    Arguments:
    - dist: Distributor with MPI decomposition info
    - global_shape: Tuple of global array dimensions

    Returns:
    - Tuple of local array dimensions for this process
    """
    # Serial case: local = global
    if dist.size == 1 || dist.mesh === nothing
        return global_shape
    end

    mesh = dist.mesh
    ndims_global = length(global_shape)
    ndims_mesh = length(mesh)

    # Compute local shape
    local_shape = collect(global_shape)

    # Pencil decomposition: last ndims_mesh dimensions are distributed
    # The decomposition starts from the last dimension
    for i in 1:min(ndims_mesh, ndims_global)
        # Map mesh dimension to global dimension
        # mesh[1] corresponds to global_shape[end - ndims_mesh + 1], etc.
        global_dim_idx = ndims_global - ndims_mesh + i
        mesh_dim_idx = i

        if global_dim_idx >= 1 && global_dim_idx <= ndims_global
            n_global = global_shape[global_dim_idx]
            n_procs = mesh[mesh_dim_idx]

            # Compute local size for this dimension
            # Use ceiling division for load balancing
            base_size = div(n_global, n_procs)
            remainder = n_global % n_procs

            # Get process coordinate in this mesh dimension
            proc_coord = get_process_coordinate(dist, mesh_dim_idx)

            # Processes with coord < remainder get one extra element
            if proc_coord < remainder
                local_shape[global_dim_idx] = base_size + 1
            else
                local_shape[global_dim_idx] = base_size
            end
        end
    end

    return tuple(local_shape...)
end

function get_process_coordinate(dist::Distributor, dim::Int)
    """
    Get the coordinate of this process in the specified mesh dimension.

    For a mesh (P₁, P₂, ..., Pₖ), the process with rank r has coordinates:
    (r % P₁, (r ÷ P₁) % P₂, ..., (r ÷ (P₁×P₂×...×Pₖ₋₁)) % Pₖ)
    """
    if dist.mesh === nothing || dim < 1 || dim > length(dist.mesh)
        return 0
    end

    mesh = dist.mesh
    rank = dist.rank

    # Compute coordinate using row-major ordering
    stride = 1
    for i in 1:(dim-1)
        stride *= mesh[i]
    end

    coord = div(rank, stride) % mesh[dim]
    return coord
end

function get_local_range(dist::Distributor, global_size::Int, dim::Int)
    """
    Get the local range [start, end] for this process in a given dimension.

    Arguments:
    - dist: Distributor with MPI decomposition info
    - global_size: Size of the global array in this dimension
    - dim: Mesh dimension index (1-based)

    Returns:
    - (start_idx, end_idx) tuple with 1-based indices
    """
    if dist.size == 1 || dist.mesh === nothing || dim > length(dist.mesh)
        return (1, global_size)
    end

    n_procs = dist.mesh[dim]
    proc_coord = get_process_coordinate(dist, dim)

    base_size = div(global_size, n_procs)
    remainder = global_size % n_procs

    if proc_coord < remainder
        local_size = base_size + 1
        start_idx = proc_coord * (base_size + 1) + 1
    else
        local_size = base_size
        start_idx = remainder * (base_size + 1) + (proc_coord - remainder) * base_size + 1
    end

    end_idx = start_idx + local_size - 1
    return (start_idx, end_idx)
end

function global_to_local_index(dist::Distributor, global_idx::Int, dim::Int)
    """
    Convert a global index to a local index for this process.

    Returns nothing if the global index is not owned by this process.
    """
    start_idx, end_idx = get_local_range(dist, get_global_size(dist, dim), dim)

    if global_idx >= start_idx && global_idx <= end_idx
        return global_idx - start_idx + 1
    else
        return nothing
    end
end

function local_to_global_index(dist::Distributor, local_idx::Int, global_size::Int, dim::Int)
    """
    Convert a local index to a global index.
    """
    start_idx, _ = get_local_range(dist, global_size, dim)
    return start_idx + local_idx - 1
end

"""
    get_global_size(dist::Distributor, dim::Int)

Get the global size in a dimension. This method requires domain/basis information
to determine actual sizes. Without that context, it returns a conservative default.

For accurate global sizes, use one of the following methods instead:
- `get_global_size(dist, basis, dim)` - for a specific basis
- `get_global_size(dist, domain, dim)` - for a specific domain
- `get_global_grid_shape(dist, domain; scales=...)` - for full grid shape

# Arguments
- `dist`: The Distributor
- `dim`: Dimension index (1-based)

# Returns
- The global size in the specified dimension, or a default value if unknown
"""
function get_global_size(dist::Distributor, dim::Int)
    # Without domain/basis context, we cannot determine the actual global size.
    # Check if distributor has cached layout information that might help.
    if !isempty(dist.layouts)
        # Try to get size from cached layouts
        for (key, layout) in dist.layouts
            if hasfield(typeof(layout), :global_shape) && layout.global_shape !== nothing
                if dim <= length(layout.global_shape)
                    return layout.global_shape[dim]
                end
            end
        end
    end

    # Fallback: Check if pencil_cache has any entries with shape info
    if !isempty(dist.pencil_cache)
        for (shape_key, pencil) in dist.pencil_cache
            if isa(shape_key, Tuple) && dim <= length(shape_key)
                return shape_key[dim]
            end
        end
    end

    @warn "get_global_size called without domain context; returning default. " *
          "Use get_global_size(dist, basis, dim) or get_global_size(dist, domain, dim) for accurate sizes." maxlog=1
    return 64
end

"""
    get_global_size(dist::Distributor, basis::Basis, dim::Int=1)

Get the global size for a specific basis dimension.

# Arguments
- `dist`: The Distributor (unused but kept for API consistency)
- `basis`: The Basis to query
- `dim`: Dimension within the basis (default 1, as most bases are 1D)

# Returns
- The global size (number of grid/coefficient points) for this basis
"""
function get_global_size(dist::Distributor, basis::Basis, dim::Int=1)
    if dim != 1
        @warn "Most bases are 1D; dim=$dim requested but using basis size"
    end
    return basis.meta.size
end

"""
    get_global_size(dist::Distributor, domain::Domain, dim::Int)

Get the global size in a specific dimension of the domain.

# Arguments
- `dist`: The Distributor (unused but kept for API consistency)
- `domain`: The Domain containing the bases
- `dim`: Dimension index (1-based)

# Returns
- The global size in the specified dimension
"""
function get_global_size(dist::Distributor, domain::Domain, dim::Int)
    if dim < 1 || dim > length(domain.bases)
        throw(BoundsError("Dimension $dim out of range for domain with $(length(domain.bases)) dimensions"))
    end
    return domain.bases[dim].meta.size
end

"""
    get_global_sizes(dist::Distributor, domain::Domain)

Get all global sizes for a domain as a tuple.

# Arguments
- `dist`: The Distributor
- `domain`: The Domain containing the bases

# Returns
- Tuple of global sizes for each dimension
"""
function get_global_sizes(dist::Distributor, domain::Domain)
    return tuple([basis.meta.size for basis in domain.bases]...)
end

function preset_scales!(field::ScalarField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    """
    Set new transform scales without data transformation.

    Scales control the grid resolution relative to the coefficient resolution.
    A scale of 1.0 means grid size equals coefficient size.
    A scale of 1.5 (3/2 rule) is used for dealiasing in nonlinear computations.

    This function:
    1. Updates the field's scale parameters
    2. Reallocates data arrays if the new scaled size differs
    3. Does NOT preserve or transform existing data (use set_scales! for that)

    Arguments:
    - field: ScalarField to modify
    - scales: New scale values (scalar applied to all dims, or per-dimension)

    Returns:
    - The modified field
    """
    new_scales = remedy_scales(field.dist, scales)
    old_scales = field.scales

    # Return if scales are unchanged
    if new_scales == old_scales
        return field
    end

    # Compute old and new grid sizes
    old_grid_shape = get_scaled_shape(field, old_scales)
    new_grid_shape = get_scaled_shape(field, new_scales)

    # Update scales
    field.scales = new_scales

    # Reallocate grid data if shape changed
    if old_grid_shape != new_grid_shape && field.domain !== nothing
        @debug "Reallocating field data for new scales" old_grid_shape new_grid_shape

        # Reallocate grid-space data with new size
        if field.dist.use_pencil_arrays
            # For MPI: create new pencil with scaled shape
            field.data_g = create_pencil(field.dist, new_grid_shape, 1, dtype=field.dtype)
        else
            # For serial: create new array
            local_shape = get_local_array_size(field.dist, new_grid_shape)
            field.data_g = zeros(field.dtype, local_shape...)
        end

        # Coefficient data size typically doesn't change with scales
        # (scales affect grid resolution, not spectral resolution)
    end

    @debug "Updated field scales" old_scales=old_scales new_scales=new_scales
    return field
end

function get_scaled_shape(field::ScalarField, scales::Union{Tuple, Nothing})
    """
    Compute the grid shape with given scales applied.

    For each basis, the scaled size is: ceil(Int, basis_size * scale)
    """
    if field.domain === nothing || isempty(field.bases)
        return ()
    end

    # Handle nothing scales (use 1.0 for all dimensions)
    if scales === nothing
        scales = tuple(ones(Float64, length(field.bases))...)
    end

    scaled_shape = Int[]
    for (i, basis) in enumerate(field.bases)
        base_size = basis.meta.size
        scale = i <= length(scales) ? scales[i] : 1.0
        scaled_size = ceil(Int, base_size * scale)
        push!(scaled_shape, scaled_size)
    end

    return tuple(scaled_shape...)
end

function get_scaled_shape(field::ScalarField)
    """Get the current scaled grid shape."""
    if field.scales === nothing
        scales = tuple(ones(Float64, length(field.bases))...)
    else
        scales = field.scales
    end
    return get_scaled_shape(field, scales)
end

function get_coefficient_shape(field::ScalarField)
    """Get the coefficient (unscaled) shape."""
    if field.domain === nothing || isempty(field.bases)
        return ()
    end

    return tuple([basis.meta.size for basis in field.bases]...)
end

function require_scales!(field::ScalarField, scales::Union{Real, Tuple, Nothing})
    """
    Ensure field has the specified scales, reallocating if necessary.
    Similar to require_scales pattern.
    """
    new_scales = remedy_scales(field.dist, scales)

    if field.scales != new_scales
        preset_scales!(field, new_scales)
    end

    return field
end

function dealias_scales(field::ScalarField)
    """
    Get the standard 3/2 dealiasing scales for this field.
    Used for computing nonlinear terms without aliasing errors.
    """
    ndims = length(field.bases)
    return tuple(fill(1.5, ndims)...)
end

function apply_dealiasing_scales!(field::ScalarField)
    """Apply 3/2 dealiasing scales to the field."""
    return preset_scales!(field, dealias_scales(field))
end

function set_scales!(field::ScalarField, scales::Union{Real, Vector{Real}, Tuple{Vararg{Real}}, Nothing})
    """
    Change data to specified scales, properly handling data transformation.
    Following implementation in field:631-649

    When changing scales:
    - If in grid space: interpolate/resample data to new grid size
    - If in coefficient space: pad/truncate spectral coefficients

    For upscaling (scale increases):
    - Grid space: interpolate to finer grid
    - Coefficient space: zero-pad high frequencies

    For downscaling (scale decreases):
    - Grid space: subsample or average to coarser grid
    - Coefficient space: truncate high frequencies

    Arguments:
    - field: ScalarField to modify
    - scales: New scale values

    Returns:
    - The modified field with transformed data
    """
    # Remedy scales
    new_scales = remedy_scales(field.dist, scales)
    old_scales = field.scales

    # Quit if new scales aren't new
    if new_scales == old_scales
        return field
    end

    # Get old and new shapes
    old_grid_shape = get_scaled_shape(field, old_scales)
    new_grid_shape = get_scaled_shape(field, new_scales)
    coeff_shape = get_coefficient_shape(field)

    # Determine current layout
    is_grid_space = (field.current_layout == :g)

    if is_grid_space && field.data_g !== nothing
        # Transform grid-space data to new resolution
        old_data = get_local_data(field.data_g)

        if old_data !== nothing && !isempty(old_data)
            # Store coefficient data if it exists
            old_coeff_data = field.data_c !== nothing ? copy(get_local_data(field.data_c)) : nothing

            # Reallocate with new scales
            preset_scales!(field, new_scales)

            # Resample grid data to new resolution
            new_data = get_local_data(field.data_g)
            if new_data !== nothing
                resample_grid_data!(new_data, old_data, old_grid_shape, new_grid_shape)
            end

            # Restore coefficient data (unchanged by grid scaling)
            if old_coeff_data !== nothing && field.data_c !== nothing
                coeff_data = get_local_data(field.data_c)
                if coeff_data !== nothing && size(coeff_data) == size(old_coeff_data)
                    copyto!(coeff_data, old_coeff_data)
                end
            end
        else
            preset_scales!(field, new_scales)
        end

    elseif !is_grid_space && field.data_c !== nothing
        # In coefficient space: scales don't affect coefficient storage
        # but we still need to update the field's scale parameter
        # The grid data will be recomputed on next backward transform
        preset_scales!(field, new_scales)

    else
        # No data to transform
        preset_scales!(field, new_scales)
    end

    @debug "Changed field scales with data transformation" old_scales=old_scales new_scales=new_scales
    return field
end

function resample_grid_data!(new_data::AbstractArray, old_data::AbstractArray,
                             old_shape::Tuple, new_shape::Tuple)
    """
    Resample grid data from old resolution to new resolution.

    Uses spectral interpolation for accurate resampling:
    1. FFT old data to spectral space
    2. Pad/truncate spectral coefficients
    3. IFFT back to new grid

    For simple cases, uses linear interpolation as fallback.
    """
    old_size = size(old_data)
    new_size = size(new_data)

    # Handle dimension mismatch
    if length(old_size) != length(new_size)
        @warn "Dimension mismatch in resample_grid_data!"
        fill!(new_data, 0)
        return
    end

    # If sizes match, just copy
    if old_size == new_size
        copyto!(new_data, old_data)
        return
    end

    ndims_data = length(old_size)

    if ndims_data == 1
        resample_1d!(new_data, old_data)
    elseif ndims_data == 2
        resample_2d!(new_data, old_data)
    elseif ndims_data == 3
        resample_3d!(new_data, old_data)
    else
        # Fallback: simple nearest-neighbor for higher dimensions
        resample_nearest!(new_data, old_data)
    end
end

function resample_1d!(new_data::AbstractVector, old_data::AbstractVector)
    """Resample 1D data using spectral interpolation."""
    n_old = length(old_data)
    n_new = length(new_data)

    if n_old == n_new
        copyto!(new_data, old_data)
        return
    end

    # Use FFT-based resampling for spectral accuracy
    try
        # Forward FFT
        old_fft = FFTW.fft(complex(old_data))

        # Create padded/truncated spectrum
        new_fft = zeros(ComplexF64, n_new)

        if n_new > n_old
            # Upsampling: zero-pad high frequencies
            # Copy positive frequencies
            n_pos = div(n_old, 2)
            new_fft[1:n_pos+1] = old_fft[1:n_pos+1]
            # Copy negative frequencies
            new_fft[end-n_pos+1:end] = old_fft[end-n_pos+1:end]
            # Scale for energy conservation
            new_fft .*= n_new / n_old
        else
            # Downsampling: truncate high frequencies
            n_pos = div(n_new, 2)
            new_fft[1:n_pos+1] = old_fft[1:n_pos+1]
            new_fft[end-n_pos+1:end] = old_fft[end-n_pos+1:end]
            new_fft .*= n_new / n_old
        end

        # Inverse FFT
        result = real(FFTW.ifft(new_fft))
        copyto!(new_data, result)

    catch e
        @warn "FFT resampling failed, using linear interpolation: $e"
        resample_linear_1d!(new_data, old_data)
    end
end

function resample_linear_1d!(new_data::AbstractVector, old_data::AbstractVector)
    """Fallback linear interpolation for 1D resampling."""
    n_old = length(old_data)
    n_new = length(new_data)

    for i in 1:n_new
        # Map new index to old index (0-based for interpolation)
        t = (i - 1) / (n_new - 1) * (n_old - 1)
        i_old = floor(Int, t) + 1
        frac = t - (i_old - 1)

        if i_old >= n_old
            new_data[i] = old_data[n_old]
        elseif i_old < 1
            new_data[i] = old_data[1]
        else
            new_data[i] = (1 - frac) * old_data[i_old] + frac * old_data[i_old + 1]
        end
    end
end

function resample_2d!(new_data::AbstractMatrix, old_data::AbstractMatrix)
    """Resample 2D data using spectral interpolation."""
    n_old = size(old_data)
    n_new = size(new_data)

    if n_old == n_new
        copyto!(new_data, old_data)
        return
    end

    try
        # Forward 2D FFT
        old_fft = FFTW.fft(complex(old_data))

        # Create padded/truncated spectrum
        new_fft = zeros(ComplexF64, n_new)

        # Copy/pad each dimension
        for dim in 1:2
            n_o = n_old[dim]
            n_n = n_new[dim]
            n_copy = min(div(n_o, 2), div(n_n, 2))

            # This is simplified; proper 2D padding requires careful index handling
        end

        # For now, use separable 1D resampling as simpler approach
        temp = zeros(eltype(new_data), n_new[1], n_old[2])

        # Resample along first dimension
        for j in 1:n_old[2]
            resample_1d!(view(temp, :, j), view(old_data, :, j))
        end

        # Resample along second dimension
        for i in 1:n_new[1]
            resample_1d!(view(new_data, i, :), view(temp, i, :))
        end

    catch e
        @warn "2D spectral resampling failed: $e"
        resample_nearest!(new_data, old_data)
    end
end

function resample_3d!(new_data::AbstractArray{T,3}, old_data::AbstractArray{T,3}) where T
    """Resample 3D data using separable 1D spectral interpolation."""
    n_old = size(old_data)
    n_new = size(new_data)

    if n_old == n_new
        copyto!(new_data, old_data)
        return
    end

    try
        # Separable resampling: resample along each dimension sequentially
        temp1 = zeros(T, n_new[1], n_old[2], n_old[3])
        temp2 = zeros(T, n_new[1], n_new[2], n_old[3])

        # Resample along first dimension
        for k in 1:n_old[3], j in 1:n_old[2]
            resample_1d!(view(temp1, :, j, k), view(old_data, :, j, k))
        end

        # Resample along second dimension
        for k in 1:n_old[3], i in 1:n_new[1]
            resample_1d!(view(temp2, i, :, k), view(temp1, i, :, k))
        end

        # Resample along third dimension
        for j in 1:n_new[2], i in 1:n_new[1]
            resample_1d!(view(new_data, i, j, :), view(temp2, i, j, :))
        end

    catch e
        @warn "3D spectral resampling failed: $e"
        resample_nearest!(new_data, old_data)
    end
end

function resample_nearest!(new_data::AbstractArray, old_data::AbstractArray)
    """Nearest-neighbor resampling for arbitrary dimensions."""
    old_size = size(old_data)
    new_size = size(new_data)
    ndims_data = length(old_size)

    for I in CartesianIndices(new_data)
        # Map new indices to old indices
        old_indices = ntuple(ndims_data) do d
            # Scale index from new to old grid
            new_idx = I[d]
            old_idx = round(Int, (new_idx - 1) / (new_size[d] - 1) * (old_size[d] - 1)) + 1
            clamp(old_idx, 1, old_size[d])
        end

        new_data[I] = old_data[CartesianIndex(old_indices)]
    end
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

function ensure_layout!(field::VectorField, target_layout::Symbol)
    """Ensure all components of VectorField are in the target layout"""
    for comp in field.components
        ensure_layout!(comp, target_layout)
    end
end

function ensure_layout!(field::TensorField, target_layout::Symbol)
    """Ensure all components of TensorField are in the target layout"""
    for row in field.components
        for comp in row
            ensure_layout!(comp, target_layout)
        end
    end
end

function require_grid_space!(field::ScalarField, axis::Union{Int, Nothing}=nothing)
    """
    Require one axis (default: all axes) to be in grid space.
    Following implementation in field:674-681
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
    Following implementation in field:683-690
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
    Following implementation in field:664-667
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
    Following implementation in field:669-672
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

    Following distributor pattern in distributor:636-649
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

    Following distributor pattern in distributor:621-634
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
    Following low_pass_filter implementation.
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
    Following field:945-968 implementation.
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
    Following field:969-984 implementation.
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

function get_global_grid_shape(dist::Distributor, domain::Domain; scales=nothing)
    """
    Get global grid shape for a domain with given scales.

    The global grid shape is the full size of the grid across all MPI processes.
    Each dimension's size is determined by:
    - The basis size (number of modes/coefficients)
    - The scale factor (for dealiasing, typically 1.0 or 1.5)

    Arguments:
    - dist: Distributor with domain decomposition info
    - domain: Domain containing basis information
    - scales: Scale factors per dimension (default: 1.0 for all)
              Can be a scalar (applied to all), vector, or tuple

    Returns:
    - Tuple of global grid dimensions

    Example:
    - For a 2D domain with bases of size (64, 32) and scales (1.5, 1.5):
      Returns (96, 48)
    """
    if isempty(domain.bases)
        return ()
    end

    n_bases = length(domain.bases)

    # Handle scales argument
    if scales === nothing
        scales = ones(Float64, n_bases)
    elseif isa(scales, Number)
        scales = fill(Float64(scales), n_bases)
    elseif isa(scales, Tuple)
        scales = collect(Float64, scales)
    end

    # Ensure scales vector has correct length
    if length(scales) < n_bases
        scales = vcat(scales, ones(Float64, n_bases - length(scales)))
    end

    # Compute scaled grid shape
    grid_shape = Int[]
    for (i, basis) in enumerate(domain.bases)
        base_size = get_basis_grid_size(basis)
        scale = scales[i]
        scaled_size = ceil(Int, base_size * scale)
        push!(grid_shape, scaled_size)
    end

    return tuple(grid_shape...)
end

function get_basis_grid_size(basis::Basis)
    """
    Get the natural grid size for a basis.

    For most bases, this is the number of modes/coefficients.
    Some bases may have different grid vs coefficient sizes.
    """
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :size)
        return basis.meta.size
    else
        # Fallback
        return 64
    end
end

function get_global_coeff_shape(dist::Distributor, domain::Domain)
    """
    Get global coefficient shape for a domain.

    The coefficient shape is the unscaled size (number of spectral modes).
    This is independent of the grid scale factor.

    Returns:
    - Tuple of global coefficient dimensions
    """
    if isempty(domain.bases)
        return ()
    end

    coeff_shape = Int[]
    for basis in domain.bases
        push!(coeff_shape, get_basis_coeff_size(basis))
    end

    return tuple(coeff_shape...)
end

function get_basis_coeff_size(basis::Basis)
    """
    Get the coefficient size for a basis.

    For Fourier bases: same as grid size
    For Chebyshev/Legendre: may differ due to boundary conditions
    """
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :size)
        return basis.meta.size
    else
        return 64
    end
end

function get_local_grid_shape(dist::Distributor, domain::Domain; scales=nothing)
    """
    Get local grid shape for this MPI process.

    Arguments:
    - dist: Distributor with domain decomposition info
    - domain: Domain containing basis information
    - scales: Scale factors per dimension

    Returns:
    - Tuple of local grid dimensions for this process
    """
    global_shape = get_global_grid_shape(dist, domain; scales=scales)
    return get_local_array_size(dist, global_shape)
end

function get_local_coeff_shape(dist::Distributor, domain::Domain)
    """
    Get local coefficient shape for this MPI process.

    Returns:
    - Tuple of local coefficient dimensions for this process
    """
    global_shape = get_global_coeff_shape(dist, domain)
    return get_local_array_size(dist, global_shape)
end

function get_grid_layout_info(dist::Distributor, domain::Domain; scales=nothing)
    """
    Get comprehensive grid layout information.

    Returns a NamedTuple with:
    - global_shape: Full grid size across all processes
    - local_shape: Grid size on this process
    - local_start: Starting global index for this process (1-based)
    - local_end: Ending global index for this process (1-based)
    - scales: Applied scale factors
    """
    n_bases = length(domain.bases)

    # Handle scales
    if scales === nothing
        scales = tuple(ones(Float64, n_bases)...)
    elseif isa(scales, Number)
        scales = tuple(fill(Float64(scales), n_bases)...)
    elseif isa(scales, Vector)
        scales = tuple(scales...)
    end

    global_shape = get_global_grid_shape(dist, domain; scales=scales)
    local_shape = get_local_array_size(dist, global_shape)

    # Compute local start/end indices
    ndims_mesh = dist.mesh !== nothing ? length(dist.mesh) : 0
    ndims_global = length(global_shape)

    local_start = ones(Int, ndims_global)
    local_end = collect(global_shape)

    if dist.mesh !== nothing && dist.size > 1
        for i in 1:min(ndims_mesh, ndims_global)
            global_dim_idx = ndims_global - ndims_mesh + i
            if global_dim_idx >= 1
                start_idx, end_idx = get_local_range(dist, global_shape[global_dim_idx], i)
                local_start[global_dim_idx] = start_idx
                local_end[global_dim_idx] = end_idx
            end
        end
    end

    return (
        global_shape = global_shape,
        local_shape = local_shape,
        local_start = tuple(local_start...),
        local_end = tuple(local_end...),
        scales = scales
    )
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
    Following implementation in coords:183

    Note: This function was moved from coords.jl to field.jl to avoid circular dependency,
    as it needs VectorField which is defined in field.jl.
    """
    fields = VectorField[]
    for (i, coord) in enumerate(coords(coordsys))
        # Create vector field for each coordinate direction
        ec = VectorField(dist, coordsys, name="e$(coord.name)")

        # Set the i-th component to 1 (unit vector in that direction)
        # Implementation: ec['g'][i] = 1
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
