# ============================================================================
# GPU-Aware Data Access (for File I/O)
# ============================================================================

"""
    get_cpu_data(field_data)

Get data as a CPU Array, automatically transferring from GPU if needed.
This is essential for file I/O since the NetCDF library requires CPU arrays.

For GPU arrays (CuArray): Transfers data to CPU
For CPU arrays: Returns as-is
For PencilArray: Gets local data then transfers if on GPU
"""
function get_cpu_data(field_data::AbstractArray)
    # Check if data is on GPU and transfer to CPU
    # The on_architecture function handles the conversion
    return on_architecture(CPU(), field_data)
end

function get_cpu_data(field_data::PencilArrays.PencilArray)
    # Get local data first, then transfer to CPU if needed
    local_data = parent(field_data)
    return on_architecture(CPU(), local_data)
end

function get_cpu_data(field_data::Nothing)
    return nothing
end

"""
    get_cpu_local_data(field::ScalarField, layout::Symbol)

Get field data in the specified layout as a CPU Array.
Automatically handles GPU to CPU transfer for file I/O operations.

# Arguments
- `field`: The ScalarField to extract data from
- `layout`: :g for grid space, :c for coefficient space

# Returns
CPU Array containing the field data
"""
function get_cpu_local_data(field::ScalarField, layout::Symbol)
    ensure_layout!(field, layout)
    if layout == :g
        return get_cpu_data(get_grid_data(field))
    else
        return get_cpu_data(get_coeff_data(field))
    end
end

"""
    get_cpu_local_data(field::VectorField, layout::Symbol)

Get vector field data as CPU Arrays.
"""
function get_cpu_local_data(field::VectorField, layout::Symbol)
    return [get_cpu_local_data(comp, layout) for comp in field.components]
end

"""
    get_cpu_local_data(field::TensorField, layout::Symbol)

Get tensor field data as CPU Arrays.
"""
function get_cpu_local_data(field::TensorField, layout::Symbol)
    return [get_cpu_local_data(field.components[i,j], layout)
            for i in 1:size(field.components, 1), j in 1:size(field.components, 2)]
end

"""
    is_gpu_field(field::ScalarField)

Check if a field's data is on GPU.
"""
function is_gpu_field(field::ScalarField)
    # Check actual array storage rather than architecture metadata
    gd = field.buffers.grid
    if gd isa AbstractArray
        return is_gpu_array(gd)
    end
    cd = field.buffers.coeff
    if cd isa AbstractArray
        return is_gpu_array(cd)
    end
    # Fallback to architecture if no data allocated yet (or Pencil arrays = CPU)
    return is_gpu(field.buffers.architecture)
end

is_gpu_field(field::VectorField) = is_gpu_field(field.components[1])
is_gpu_field(field::TensorField) = is_gpu_field(field.components[1,1])

"""
    set_local_data!(field_data, values)

Set local data in either a PencilArray or regular Array.
"""
function set_local_data!(field_data::PencilArrays.PencilArray, values)
    parent(field_data) .= values
    return field_data
end

function set_local_data!(field_data::AbstractArray, values)
    field_data .= values
    return field_data
end

# Data access and manipulation
"""
    Get data in specified layout.

    Returns local data if using PencilArrays (MPI), otherwise returns full array.
    For user code operating on local data, this is the correct access pattern.
    """
function Base.getindex(field::ScalarField, layout::String)
    if layout == "g"
        ensure_layout!(field, :g)
        return get_local_data(get_grid_data(field))
    elseif layout == "c"
        ensure_layout!(field, :c)
        return get_local_data(get_coeff_data(field))
    else
        throw(ArgumentError("Unknown layout: $layout"))
    end
end

"""
    Set data in specified layout.

    Properly handles both PencilArray data (MPI) and regular arrays.
    """
function Base.setindex!(field::ScalarField, values, layout::String)
    if layout == "g"
        ensure_layout!(field, :g)
        set_local_data!(get_grid_data(field), values)
        field.current_layout = :g
    elseif layout == "c"
        ensure_layout!(field, :c)
        set_local_data!(get_coeff_data(field), values)
        field.current_layout = :c
    else
        throw(ArgumentError("Unknown layout: $layout"))
    end
end

"""Ensure field is in the target layout, transforming if necessary"""
function ensure_layout!(field::ScalarField, target_layout::Symbol)
    # Skip 0D fields (tau variables) which have no spatial data
    if isempty(field.bases)
        return
    end

    # Check if field has any data allocated - if not, try to allocate
    if get_grid_data(field) === nothing && get_coeff_data(field) === nothing
        if field.domain !== nothing
            allocate_data!(field)
        else
            # No domain, no data - nothing to transform
            return
        end
    end

    if field.current_layout == target_layout
        return
    end

    if target_layout == :g && field.current_layout == :c
        # Transform from coefficient to grid space
        backward_transform!(field)
        # Note: backward_transform! sets current_layout = :g when successful
    elseif target_layout == :c && field.current_layout == :g
        # Transform from grid to coefficient space
        forward_transform!(field)
        # Note: forward_transform! sets current_layout = :c when successful
    end
    # Don't set current_layout here - the transform functions handle it
    # Setting it unconditionally would be incorrect if the transform failed or returned early
end

"""Ensure all components of VectorField are in the target layout"""
function ensure_layout!(field::VectorField, target_layout::Symbol)
    for comp in field.components
        ensure_layout!(comp, target_layout)
    end
end

"""Ensure all components of TensorField are in the target layout"""
function ensure_layout!(field::TensorField, target_layout::Symbol)
    for comp in field.components  # Matrix iteration goes element-by-element
        ensure_layout!(comp, target_layout)
    end
end

# Note: forward_transform! and backward_transform! for ScalarField are defined in transforms.jl
# to avoid duplicate method definitions. The transforms.jl versions have more complete
# implementations with optional target_layout parameters.

"""
    Require one axis (default: all axes) to be in grid space.
    Following implementation in field:674-681
    """
function require_grid_space!(field::ScalarField, axis::Union{Int, Nothing}=nothing)
    if field.domain === nothing
        return
    end
    
    if axis === nothing
        # Require all axes to be in grid space
        while field.current_layout != :g
            towards_grid_space!(field)
        end
    else
        # For specific axis: ensure field is in grid space
        # Tarang uses a two-state layout model (:c for coefficient, :g for grid)
        # rather than per-axis tracking. For single-axis requirements,
        # we transform the entire field to grid space, which ensures the
        # requested axis is in grid space along with all others.
        towards_grid_space!(field)
    end
end

"""
    Require one axis (default: all axes) to be in coefficient space.
    Following implementation in field:683-690
    """
function require_coeff_space!(field::ScalarField, axis::Union{Int, Nothing}=nothing)
    if field.domain === nothing
        return
    end

    if axis === nothing
        # Require all axes to be in coefficient space
        while field.current_layout != :c
            towards_coeff_space!(field)
        end
    else
        # For specific axis: ensure field is in coefficient space
        # Tarang uses a two-state layout model (:c for coefficient, :g for grid)
        # rather than per-axis tracking. For single-axis requirements,
        # we transform the entire field to coefficient space, which ensures
        # the requested axis is in coefficient space along with all others.
        towards_coeff_space!(field)
    end
end

"""
    Change to next layout towards grid space.
    Following implementation in field:664-667
    """
function towards_grid_space!(field::ScalarField)
    if field.current_layout == :c
        # Transform from coefficient to grid space
        # Note: backward_transform_axis! sets current_layout = :g when successful
        # It may return early without transforming if field.domain is nothing
        backward_transform_axis!(field)
    end
end

"""
    Change to next layout towards coefficient space.
    Following implementation in field:669-672
    """
function towards_coeff_space!(field::ScalarField)
    if field.current_layout == :g
        # Transform from grid to coefficient space
        # Note: forward_transform_axis! sets current_layout = :c when successful
        # It may return early without transforming if field.domain is nothing
        forward_transform_axis!(field)
    end
end

"""
    Forward transform field using PencilFFTs for parallel transforms.

    CORRECT PencilFFTs usage pattern:
    1. Input/output are PencilArray objects (NOT plain arrays)
    2. PencilFFT automatically handles:
       - Required transpose operations between decompositions
       - Multi-dimensional FFT across decomposed axes
       - Scaling and normalization
    3. For 2D: Enables BOTH vertical and horizontal parallelization

    Following distributor pattern in distributor:636-649
    """
function forward_transform_axis!(field::ScalarField)
    if field.domain === nothing || field.bases === ()
        return
    end

    # Use PencilFFTs-based transforms from the distributor's transform plans
    pencil_plan = _find_pencil_plan(field.dist)
    if pencil_plan !== nothing
        # CORRECT: Apply PencilFFT to PencilArray objects
        # PencilFFT handles transposes internally

        if field.dist.use_pencil_arrays && isa(get_grid_data(field), PencilArrays.PencilArray)
            # Apply forward transform: grid space (physical) → coefficient space (spectral)
            # Note: mul! is the in-place version
            # Result goes into data_c pencil
            if get_coeff_data(field) === nothing || !isa(get_coeff_data(field), PencilArrays.PencilArray)
                # CRITICAL: Use PencilFFTs.allocate_output for compatible coeff-space array
                set_coeff_data!(field, PencilFFTs.allocate_output(pencil_plan))
            end

            # Apply PencilFFT: transforms AND transposes as needed
            mul!(get_coeff_data(field), pencil_plan, get_grid_data(field))

            @debug "Applied PencilFFT forward transform" typeof(pencil_plan) size(get_grid_data(field))
            field.current_layout = :c
            return  # Successfully applied transform
        else
            # CRITICAL: PencilFFT found but cannot be applied - this indicates a configuration error
            if field.dist.size > 1
                error("PencilFFT transform found but field data is not a PencilArray. " *
                      "In MPI mode, field data must be stored as PencilArrays for correct parallel transforms. " *
                      "use_pencil_arrays=$(field.dist.use_pencil_arrays), " *
                      "grid_data type=$(typeof(get_grid_data(field)))")
            else
                # Serial mode: fall through to standard transform
                @debug "PencilFFT found but using serial transform (serial execution)"
            end
        end
    end

    # No PencilFFT or serial mode: use standard transform
    forward_transform!(field)
end

"""
    Backward transform field using PencilFFTs for parallel transforms.

    CORRECT PencilFFTs usage pattern:
    1. Inverse FFT: coefficient space (spectral) → grid space (physical)
    2. Uses ldiv! (\\) for backward transform
    3. Maintains PencilArray objects throughout

    Following distributor pattern in distributor:621-634
    """
function backward_transform_axis!(field::ScalarField)
    if field.domain === nothing || field.bases === ()
        return
    end

    # Use PencilFFTs-based transforms from the distributor's transform plans
    pencil_plan = _find_pencil_plan(field.dist)
    if pencil_plan !== nothing
        # CORRECT: Apply inverse PencilFFT to PencilArray objects

        if field.dist.use_pencil_arrays && isa(get_coeff_data(field), PencilArrays.PencilArray)
            # Apply backward transform: coefficient space → grid space
            if get_grid_data(field) === nothing || !isa(get_grid_data(field), PencilArrays.PencilArray)
                # CRITICAL: Use PencilFFTs.allocate_input for compatible grid-space array
                # allocate_input creates the input array for forward transform,
                # which is the output of backward transform
                set_grid_data!(field, PencilFFTs.allocate_input(pencil_plan))
            end

            # Apply inverse PencilFFT: transforms AND transposes as needed
            # ldiv! is in-place inverse (like \ but in-place)
            ldiv!(get_grid_data(field), pencil_plan, get_coeff_data(field))

            @debug "Applied PencilFFT backward transform" typeof(pencil_plan) size(get_coeff_data(field))
            field.current_layout = :g
            return  # Successfully applied transform
        else
            # CRITICAL: PencilFFT found but cannot be applied - this indicates a configuration error
            if field.dist.size > 1
                error("PencilFFT transform found but field data is not a PencilArray. " *
                      "In MPI mode, field data must be stored as PencilArrays for correct parallel transforms. " *
                      "use_pencil_arrays=$(field.dist.use_pencil_arrays), " *
                      "coeff_data type=$(typeof(get_coeff_data(field)))")
            else
                # Serial mode: fall through to standard transform
                @debug "PencilFFT found but using serial transform (serial execution)"
            end
        end
    end

    # No PencilFFT or serial mode: use standard transform
    backward_transform!(field)
end

# VectorField transform methods
"""Require vector field components to be in grid space."""
function require_grid_space!(field::VectorField, axis::Union{Int, Nothing}=nothing)
    for component in field.components
        require_grid_space!(component, axis)
    end
end

"""Require vector field components to be in coefficient space."""
function require_coeff_space!(field::VectorField, axis::Union{Int, Nothing}=nothing)
    for component in field.components
        require_coeff_space!(component, axis)
    end
end

"""Transform vector field from grid to coefficient space."""
function forward_transform!(field::VectorField)
    for component in field.components
        forward_transform!(component)
    end
end

"""Transform vector field from coefficient to grid space."""
function backward_transform!(field::VectorField)
    for component in field.components
        backward_transform!(component)
    end
end

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

    if seed !== nothing && reproducible && field.dist.size > 1
        # MPI-reproducible mode: use global index-based seeding
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

"""Integrate field over specified axes"""
function integrate(field::ScalarField, axes=:)
    if field.domain === nothing
        return 0.0
    end
    
    ensure_layout!(field, :g)
    weights = integration_weights(field.domain)
    
    result = get_grid_data(field)
    for (i, w) in enumerate(weights)
        if axes === Colon() || i in axes
            # Apply weights and sum along dimension i
            result = sum(result .* reshape(w, ntuple(j -> j==i ? length(w) : 1, ndims(result))), dims=i)
        end
    end
    
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

    result = ScalarField(a.dist, _FIELD_ARITH_TMP_NAME, a.bases, a.dtype)
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)
    ensure_layout!(result, :g)

    _local_data(get_grid_data(result)) .= _local_data(get_grid_data(a)) .* _local_data(get_grid_data(b))

    # Apply basic dealiasing for spectral methods (3/2 rule)
    if has_spectral_bases(a) && length(get_grid_data(a)) > 64
        apply_dealiasing_to_product!(result)
    end

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

# Optimization support functions
"""Check if field uses spectral bases that benefit from dealiasing"""
function has_spectral_bases(field::ScalarField)
    for basis in field.bases
        if isa(basis, Union{RealFourier, ComplexFourier, ChebyshevT})
            return true
        end
    end
    return false
end

"""Apply 3/2 rule dealiasing to nonlinear product"""
function apply_dealiasing_to_product!(field::ScalarField)
    # Apply 2/3 rule cutoff for dealiasing
    # This removes the highest 1/3 of modes in each direction
    cutoff_scale = 2.0/3.0
    apply_spectral_cutoff!(field, cutoff_scale)
end

"""
    Apply spectral cutoff by zeroing modes above specified relative scales.
    Following low_pass_filter implementation.
    """
function apply_spectral_cutoff!(field::ScalarField, cutoff_scales::Union{Float64, Tuple{Vararg{Float64}}})
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

"""
    Apply a spectral low-pass filter by zeroing modes above specified relative scales.
    The scales can be specified directly or deduced from a specified global grid shape.
    Following field:945-968 implementation.
    """
function low_pass_filter!(field::ScalarField; shape=nothing, scales=nothing)
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

"""
    Apply a spectral high-pass filter by zeroing modes below specified relative scales.
    Following field:969-984 implementation.
    """
function high_pass_filter!(field::ScalarField; shape=nothing, scales=nothing)
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

"""Get field data in specified layout"""
function get_data(field::ScalarField, layout::Symbol)
    if layout == :g
        ensure_layout!(field, :g)
        return get_grid_data(field)
    elseif layout == :c
        ensure_layout!(field, :c)
        return get_coeff_data(field)
    else
        throw(ArgumentError("Unknown layout: $layout"))
    end
end

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
function get_global_grid_shape(dist::Distributor, domain::Domain; scales=nothing)
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

"""
    Get the natural grid size for a basis.

    For most bases, this is the number of modes/coefficients.
    Some bases may have different grid vs coefficient sizes.
    """
function get_basis_grid_size(basis::Basis)
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :size)
        return basis.meta.size
    else
        # Fallback
        return 64
    end
end

"""
    Get global coefficient shape for a domain.

    The coefficient shape is the unscaled size (number of spectral modes).
    This is independent of the grid scale factor.

    Returns:
    - Tuple of global coefficient dimensions
    """
function get_global_coeff_shape(dist::Distributor, domain::Domain)
    if isempty(domain.bases)
        return ()
    end

    coeff_shape = Int[]
    for basis in domain.bases
        push!(coeff_shape, get_basis_coeff_size(basis))
    end

    return tuple(coeff_shape...)
end

"""
    Get the coefficient size for a basis.

    For Fourier bases: same as grid size
    For Chebyshev/Legendre: may differ due to boundary conditions
    """
function get_basis_coeff_size(basis::Basis)
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :size)
        return basis.meta.size
    else
        return 64
    end
end

"""
    Get local grid shape for this MPI process.

    Arguments:
    - dist: Distributor with domain decomposition info
    - domain: Domain containing basis information
    - scales: Scale factors per dimension

    Returns:
    - Tuple of local grid dimensions for this process
    """
function get_local_grid_shape(dist::Distributor, domain::Domain; scales=nothing)
    global_shape = get_global_grid_shape(dist, domain; scales=scales)
    return get_local_array_size(dist, global_shape)
end

"""
    Get local coefficient shape for this MPI process.

    Returns:
    - Tuple of local coefficient dimensions for this process
    """
function get_local_coeff_shape(dist::Distributor, domain::Domain)
    global_shape = get_global_coeff_shape(dist, domain)
    return get_local_array_size(dist, global_shape)
end

"""
    Get comprehensive grid layout information.

    Returns a NamedTuple with:
    - global_shape: Full grid size across all processes
    - local_shape: Grid size on this process
    - local_start: Starting global index for this process (1-based)
    - local_end: Ending global index for this process (1-based)
    - scales: Applied scale factors
    """
function get_grid_layout_info(dist::Distributor, domain::Domain; scales=nothing)
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
        if dist.use_pencil_arrays
            # PencilArrays convention: decompose LAST ndims_mesh dimensions
            for i in 1:min(ndims_mesh, ndims_global)
                global_dim_idx = ndims_global - ndims_mesh + i
                if global_dim_idx >= 1
                    # Pass global axis index to get_local_range (it handles convention internally)
                    start_idx, end_idx = get_local_range(dist, global_shape[global_dim_idx], global_dim_idx)
                    local_start[global_dim_idx] = start_idx
                    local_end[global_dim_idx] = end_idx
                end
            end
        else
            # TransposableField convention: decompose FIRST ndims_mesh dimensions
            for i in 1:min(ndims_mesh, ndims_global)
                # Pass global axis index to get_local_range (i is both mesh dim and axis here)
                start_idx, end_idx = get_local_range(dist, global_shape[i], i)
                local_start[i] = start_idx
                local_end[i] = end_idx
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

            # Ensure data exists even when no bases are provided (0D fields)
            # For 0D fields (constant unit vectors), use a single scalar value
            if get_grid_data(comp) === nothing
                set_grid_data!(comp, zeros(dist.architecture, comp.dtype, 1))
            end
            if get_coeff_data(comp) === nothing
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

