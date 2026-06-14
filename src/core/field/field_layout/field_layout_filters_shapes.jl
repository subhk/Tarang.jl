"""
    Field filters and shapes

This file contains spectral filter helpers and global/local grid or coefficient
shape queries.
"""

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
Zero Fourier modes above the given relative scale(s), operating DIRECTLY in
coefficient space. A scale `s` on a Fourier axis of grid size N keeps modes with
|k| ≤ s·N/2; non-Fourier (Chebyshev) axes are left untouched. The field is left
in grid layout (matching the previous behaviour).

The old implementation filtered via a `set_scales!`→`require_grid_space!`→
`set_scales!` resample round-trip. Under MPI that runs a per-rank LOCAL FFT (not a
global spectral operation) on an incompatible pencil, producing silently-wrong
results; this coefficient-space version uses each rank's GLOBAL wavenumbers.
"""
function apply_spectral_cutoff!(field::ScalarField, cutoff_scales::Union{Float64, Tuple{Vararg{Float64}}})
    require_coeff_space!(field)
    cd = get_coeff_data(field)
    cd === nothing && return field
    bases = field.bases
    nb = length(bases)
    scales = isa(cutoff_scales, Float64) ? ntuple(_ -> Float64(cutoff_scales), nb) : cutoff_scales

    if cd isa PencilArrays.PencilArray
        # MPI: zero modes above the cutoff using each rank's GLOBAL wavenumber
        # indices. A uniform relative scale s corresponds to a dealias factor 1/s.
        _apply_spectral_cutoff_distributed!(cd, bases, 1.0 / minimum(scales))
    else
        local_cd = get_local_data(cd)
        cutoffs = ntuple(nb) do i
            b = bases[i]
            if isa(b, RealFourier) || isa(b, ComplexFourier)
                max(0, floor(Int, scales[i] * b.meta.size / 2))
            else
                size(local_cd, i)        # non-Fourier (Chebyshev) axis: keep all modes
            end
        end
        rfft_dims = ntuple(i -> isa(bases[i], RealFourier) && _is_first_real_fourier_axis(bases, i), nb)
        apply_spectral_cutoff!(local_cd, cutoffs, rfft_dims)   # array method (nonlinear_dealiasing.jl)
    end

    require_grid_space!(field)   # preserve the previous output layout (:g)
    return field
end

"""
    Apply a spectral low-pass filter by zeroing modes above specified relative scales.
    The scales can be specified directly or deduced from a specified global grid shape.
    Following field:945-968 implementation.
    """
function low_pass_filter!(field::ScalarField; shape=nothing, scales=nothing)
    # Determine scales from a target grid shape if given.
    if shape !== nothing
        scales === nothing || error("Specify either shape or scales.")
        global_shape = get_global_grid_shape(field.dist, field.domain, scales=ones(Float64, length(field.bases)))
        scales = tuple((shape ./ global_shape)...)
    end
    scales === nothing && error("low_pass_filter!: provide `shape` or `scales`.")
    # Coefficient-space cutoff (MPI-correct); see apply_spectral_cutoff!.
    return apply_spectral_cutoff!(field, Tuple(Float64.(scales)))
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
