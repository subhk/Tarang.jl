"""
    Transform Legendre - Legendre transform execution

This file contains the forward and backward Legendre transform implementations
using Gauss-Legendre quadrature.
"""

function _legendre_forward(data::AbstractArray, transform::LegendreTransform)
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        if !haskey(transform.matrices, "forward")
            return host_data
        end

        mat = transform.matrices["forward"]
        coeff_size = size(mat, 1)
        out_shape = ntuple(i -> i == axis ? coeff_size : size(host_data, i), ndims(host_data))
        real_type = real(eltype(host_data))
        out_eltype = eltype(host_data) <: Complex ? eltype(host_data) : real_type
        out = zeros(out_eltype, out_shape)

        other_dims = Tuple(filter(i -> i != axis, 1:ndims(host_data)))

        wm = get_global_workspace()
        temp_real = get_workspace!(wm, real_type, (coeff_size,))
        temp_imag = eltype(host_data) <: Complex ? get_workspace!(wm, real_type, (coeff_size,)) : nothing

        if isempty(other_dims)
            mul!(temp_real, mat, real(host_data))
            if temp_imag === nothing
                copyto!(out, temp_real)
            else
                mul!(temp_imag, mat, imag(host_data))
                out .= complex.(temp_real, temp_imag)
            end
        else
            for (slice_in, slice_out) in zip(eachslice(host_data, dims=other_dims), eachslice(out, dims=other_dims))
                mul!(temp_real, mat, real(slice_in))
                if temp_imag === nothing
                    slice_out .= temp_real
                else
                    mul!(temp_imag, mat, imag(slice_in))
                    slice_out .= complex.(temp_real, temp_imag)
                end
            end
        end

        release_workspace!(wm, temp_real)
        if temp_imag !== nothing
            release_workspace!(wm, temp_imag)
        end

        return out
    end
end

function _legendre_backward(data::AbstractArray, transform::LegendreTransform)
    return _execute_on_cpu(data) do host_data
        axis = transform.axis
        if axis > ndims(host_data)
            return host_data
        end

        if !haskey(transform.matrices, "backward")
            return host_data
        end

        mat = transform.matrices["backward"]
        grid_size = size(mat, 1)
        out_shape = ntuple(i -> i == axis ? grid_size : size(host_data, i), ndims(host_data))
        real_type = real(eltype(host_data))
        out_eltype = eltype(host_data) <: Complex ? eltype(host_data) : real_type
        out = zeros(out_eltype, out_shape)

        other_dims = Tuple(filter(i -> i != axis, 1:ndims(host_data)))

        wm = get_global_workspace()
        temp_real = get_workspace!(wm, real_type, (grid_size,))
        temp_imag = eltype(host_data) <: Complex ? get_workspace!(wm, real_type, (grid_size,)) : nothing

        if isempty(other_dims)
            mul!(temp_real, mat, real(host_data))
            if temp_imag === nothing
                copyto!(out, temp_real)
            else
                mul!(temp_imag, mat, imag(host_data))
                out .= complex.(temp_real, temp_imag)
            end
        else
            for (slice_in, slice_out) in zip(eachslice(host_data, dims=other_dims), eachslice(out, dims=other_dims))
                mul!(temp_real, mat, real(slice_in))
                if temp_imag === nothing
                    slice_out .= temp_real
                else
                    mul!(temp_imag, mat, imag(slice_in))
                    slice_out .= complex.(temp_real, temp_imag)
                end
            end
        end

        release_workspace!(wm, temp_real)
        if temp_imag !== nothing
            release_workspace!(wm, temp_imag)
        end

        return out
    end
end

function apply_legendre_forward!(field::ScalarField, transform::LegendreTransform)
    """
    Apply forward Legendre transform (grid to coefficients) with in-place operations.

    Based on Tarang JacobiMMT.forward_matrix:
    - Uses Gauss-Legendre quadrature integration
    - Proper normalization for orthonormal Legendre expansion
    - OPTIMIZED: In-place matrix-vector multiplication
    """

    if haskey(transform.matrices, "forward")
        set_coeff_data!(field, _legendre_forward(get_grid_data(field), transform))
        @debug "Applied Legendre forward transform: grid_size=$(transform.grid_size), coeff_size=$(transform.coeff_size)"
        return
    end

    @warn "No forward matrix available for Legendre transform, using identity"
    set_coeff_data!(field, copy(get_grid_data(field)))
end

function apply_legendre_backward!(field::ScalarField, transform::LegendreTransform)
    """
    Apply backward Legendre transform (coefficients to grid) with in-place operations.

    Based on Tarang polynomial evaluation:
    - Evaluates f(x) = Σ c_n P_n(x) at Gauss-Legendre quadrature points
    - OPTIMIZED: Uses workspace buffer and in-place operations
    """

    if haskey(transform.matrices, "backward")
        set_grid_data!(field, _legendre_backward(get_coeff_data(field), transform))
        @debug "Applied Legendre backward transform: coeff_size=$(transform.coeff_size), grid_size=$(transform.grid_size)"
        return
    end

    @warn "No backward matrix available for Legendre transform, using identity"
    set_grid_data!(field, real.(get_coeff_data(field)))
end

# Dealiasing operations following Tarang patterns
function dealias!(field::ScalarField, scales::Union{Real, Vector{Real}})
    """
    Apply dealiasing to field following Tarang field.change_scales and low_pass_filter implementation.
    
    Based on Tarang field:
    - Ensures field is in coefficient space for mode truncation
    - Applies scale-based truncation for each basis type
    - Handles multi-dimensional tensor product bases properly
    - Uses basis-specific dealiasing patterns (Fourier, Chebyshev, Legendre)
    
    Parameters
    ----------
    scales : Real or Vector{Real}
        Scale factors for each basis dimension (0 < scale <= 1)
        Values < 1 remove high-frequency modes for dealiasing
    """
    
    # Ensure we're in coefficient space for dealiasing
    ensure_layout!(field, :c)
    
    if field.domain === nothing
        @debug "Field has no domain, skipping dealiasing"
        return
    end
    
    # Convert scales to vector format
    if isa(scales, Real)
        scale_vec = fill(scales, length(field.domain.bases))
    else
        scale_vec = scales
    end
    
    if length(scale_vec) != length(field.domain.bases)
        throw(ArgumentError("Number of scales must match number of bases"))
    end
    
    @debug "Applying dealiasing with scales: $scale_vec"
    
    # Apply dealiasing for each basis dimension following Tarang patterns
    for (axis, (basis, scale)) in enumerate(zip(field.domain.bases, scale_vec))
        if scale >= 1.0
            continue  # No dealiasing needed for this axis
        end
        
        apply_basis_dealiasing!(field, basis, axis, scale)
    end
    
    @debug "Dealiasing completed"
end

"""
    _dealiasing_axis_size(field, axis)

Get the GLOBAL axis size for dealiasing cutoff computation.
For PencilArrays (MPI mode), returns the global size, not the local slab size.
For regular arrays (serial mode), returns the array size directly.
"""
function _dealiasing_axis_size(field::ScalarField, axis::Int)
    coeff_data = get_coeff_data(field)
    if isa(coeff_data, PencilArrays.PencilArray)
        return PencilArrays.size_global(coeff_data)[axis]
    else
        return size(coeff_data, axis)
    end
end

"""
    _dealiasing_zero_range!(field, axis, global_range)

Zero out coefficient data along `axis` for the given global index range.
Handles PencilArrays by mapping global indices to local slab indices.
"""
function _dealiasing_zero_range!(field::ScalarField, axis::Int, global_start::Int, global_end::Int)
    coeff_data = get_coeff_data(field)

    if isa(coeff_data, PencilArrays.PencilArray)
        # Map global range to local slab indices
        local_ranges = PencilArrays.range_local(coeff_data)
        local_range_axis = local_ranges[axis]
        # Intersect global [global_start, global_end] with local slab range
        local_start = max(global_start, first(local_range_axis)) - first(local_range_axis) + 1
        local_end = min(global_end, last(local_range_axis)) - first(local_range_axis) + 1
        if local_start > local_end
            return  # No overlap with this rank's slab
        end
        local_data = parent(coeff_data)
        data_size = size(local_data)
        indices = [1:s for s in data_size]
        # Account for PencilArray permutation
        perm = PencilArrays.permutation(coeff_data)
        physical_axis = isa(perm, PencilArrays.NoPermutation) ? axis : findfirst(==(axis), Tuple(perm))
        if physical_axis === nothing
            physical_axis = axis
        end
        indices[physical_axis] = local_start:local_end
        local_data[indices...] .= 0
    else
        data_size = size(coeff_data)
        if global_end > data_size[axis] || global_start < 1
            return
        end
        indices = [1:s for s in data_size]
        indices[axis] = global_start:global_end
        coeff_data[indices...] .= 0
    end
end

function apply_basis_dealiasing!(field::ScalarField, basis::Union{RealFourier, ComplexFourier}, axis::Int, scale::Real)
    """Apply Fourier basis dealiasing following Tarang patterns"""

    # Use GLOBAL axis size for cutoff computation (critical for MPI mode)
    axis_size = _dealiasing_axis_size(field, axis)
    grid_size = basis.meta.size

    # Handle different Fourier representations
    if isa(basis, RealFourier)
        # Real Fourier (rfft): complex coefficients [DC, freq_1, freq_2, ..., freq_{N/2}]
        # axis_size = N/2 + 1, representing frequencies 0 to N/2
        # With scale (e.g., 2/3), keep frequencies 0 to floor(N * scale / 2)
        # This means keeping floor(N * scale / 2) + 1 modes (including DC)
        kept_modes = Int(floor(grid_size * scale / 2)) + 1

        @debug "RealFourier dealiasing: axis=$axis, grid_size=$grid_size, axis_size=$axis_size, kept_modes=$kept_modes, scale=$scale"

        if kept_modes < axis_size
            cutoff_index = kept_modes + 1
            _dealiasing_zero_range!(field, axis, cutoff_index, axis_size)
        end

    elseif isa(basis, ComplexFourier)
        # Complex Fourier (fft): FFTW layout is [k=0, 1, ..., N/2, -(N/2-1), ..., -1]
        # - Index 1: k=0 (DC)
        # - Indices 2 to N/2+1: positive frequencies k=1 to N/2
        # - Indices N/2+2 to N: negative frequencies k=-(N/2-1) to -1
        #
        # With scale (e.g., 2/3), keep |k| <= N*scale/2
        # This means keeping modes 0 to floor(N*scale/2) and -floor(N*scale/2) to -1
        half_kept = Int(floor(grid_size * scale / 2))

        @debug "ComplexFourier dealiasing: axis=$axis, grid_size=$grid_size, axis_size=$axis_size, half_kept=$half_kept, scale=$scale"

        # Positive frequency cutoff: keep indices 1 to half_kept+1, zero half_kept+2 to N/2+1
        pos_cutoff = half_kept + 2  # First index to zero (1-indexed)
        nyquist_idx = axis_size ÷ 2 + 1

        if pos_cutoff <= nyquist_idx
            _dealiasing_zero_range!(field, axis, pos_cutoff, nyquist_idx)
        end

        # Negative frequency cutoff: keep indices N-half_kept+1 to N, zero N/2+2 to N-half_kept
        neg_start = nyquist_idx + 1  # First negative frequency index
        neg_cutoff = axis_size - half_kept + 1  # First negative index to keep

        if neg_start < neg_cutoff
            _dealiasing_zero_range!(field, axis, neg_start, neg_cutoff - 1)
        end
    end
end

function apply_basis_dealiasing!(field::ScalarField, basis::ChebyshevT, axis::Int, scale::Real)
    """Apply Chebyshev basis dealiasing following Tarang patterns"""

    # Calculate cutoff mode for Chebyshev basis
    total_modes = basis.meta.size
    kept_modes = Int(floor(total_modes * scale))

    @debug "Chebyshev dealiasing: axis=$axis, total_modes=$total_modes, kept_modes=$kept_modes, scale=$scale"

    # Use GLOBAL axis size for cutoff (critical for MPI mode)
    axis_size = _dealiasing_axis_size(field, axis)

    # Chebyshev modes are ordered [T_0, T_1, T_2, ..., T_{N-1}]
    # Keep first kept_modes coefficients, zero out the rest
    if kept_modes < axis_size
        cutoff_index = kept_modes + 1
        _dealiasing_zero_range!(field, axis, cutoff_index, axis_size)
    end
end

function apply_basis_dealiasing!(field::ScalarField, basis::Legendre, axis::Int, scale::Real)
    """Apply Legendre basis dealiasing following Tarang patterns"""

    # Calculate cutoff mode for Legendre basis
    total_modes = basis.meta.size
    kept_modes = Int(floor(total_modes * scale))

    @debug "Legendre dealiasing: axis=$axis, total_modes=$total_modes, kept_modes=$kept_modes, scale=$scale"

    # Use GLOBAL axis size for cutoff (critical for MPI mode)
    axis_size = _dealiasing_axis_size(field, axis)

    # Legendre modes are ordered [P_0, P_1, P_2, ..., P_{N-1}]
    # Keep first kept_modes coefficients, zero out the rest
    if kept_modes < axis_size
        cutoff_index = kept_modes + 1
        _dealiasing_zero_range!(field, axis, cutoff_index, axis_size)
    end
end

# Generic fallback for unknown basis types
function apply_basis_dealiasing!(field::ScalarField, basis, axis::Int, scale::Real)
    """Generic dealiasing for unknown basis types"""
    
    @warn "Unknown basis type $(typeof(basis)) for axis $axis, using generic polynomial dealiasing"
    
    # Calculate cutoff mode generically
    total_modes = basis.meta.size
    kept_modes = Int(floor(total_modes * scale))
    
    # Get coefficient array dimensions
    data_size = size(get_coeff_data(field))
    
    if axis > length(data_size) || kept_modes >= data_size[axis]
        return
    end
    
    cutoff_index = kept_modes + 1
    
    # Zero out high modes generically
    indices = [1:i for i in data_size]
    indices[axis] = cutoff_index:data_size[axis]
    get_coeff_data(field)[indices...] .= 0
end

# Convenience function for domain-based dealiasing
function dealias!(field::ScalarField)
    """Apply default dealiasing using domain.dealias scales"""
    
    if field.domain !== nothing && hasfield(typeof(field.domain), :dealias)
        dealias!(field, field.domain.dealias)
    else
        @warn "No domain dealias information available, using default 2/3 rule"
        dealias!(field, 2/3)  # Standard 2/3 rule for dealiasing
    end
end

# Utility functions
function get_transform_for_basis(transforms::Vector, basis::Basis)
    """Find transform corresponding to a basis"""
    for transform in transforms
        if hasfield(typeof(transform), :basis) && transform.basis == basis
            return transform
        end
    end
    return nothing
end

function setup_pencil_fft_transforms_3d!(dist::Distributor, domain::Domain,
                                    global_shape::Tuple, fourier_axes::Vector{Int})

    """Setup PencilFFTs transforms for parallel 3D FFT"""

    # For serial execution, use regular FFTW transforms
    if dist.size == 1
        @info "Using FFTW transforms for 3D (serial)"
        for axis in fourier_axes
            basis = domain.bases[axis]
            if isa(basis, RealFourier) || isa(basis, ComplexFourier)
                setup_fftw_transform!(dist, basis, axis)
            end
        end
        return
    end

    # PencilFFT 3D requires at least 2D mesh for correct distributed transforms
    # 1D mesh with 3D domain would compute local FFTs on distributed data, yielding incorrect spectra
    if length(dist.mesh) < 2
        error("3D MPI transforms require a 2D process mesh (got 1D mesh with $(dist.size) processes). " *
              "Use a 2D mesh like (Rx, Ry) where Rx * Ry = $(dist.size), e.g., mesh=($(isqrt(dist.size)), $(cld(dist.size, isqrt(dist.size)))) " *
              "or reduce to 2D domain. Local FFTW on distributed data would produce incorrect results.")
    end

    if dist.pencil_config === nothing
        # Create 3D pencil configuration
        setup_pencil_arrays_3d(dist, global_shape)
    end

    # Determine FFT transforms - PencilFFTPlan expects a tuple of transforms, one per dimension
    # Use RFFT for RealFourier bases (real-to-complex), FFT for ComplexFourier

    # Build transforms tuple based on basis types
    # NOTE: RFFT can only be applied to the first transform dimension in PencilFFTs.
    # Case 1: First Fourier axis is RealFourier → RFFT, subsequent RealFourier → FFT (OK with warning)
    # Case 2: First Fourier axis is NOT RealFourier but later one is → ERROR (shape mismatch)
    transform_list = []
    uses_rfft = false
    first_fourier_is_real = length(fourier_axes) > 0 && isa(domain.bases[fourier_axes[1]], RealFourier)
    realfourier_warning_shown = false

    for (i, axis) in enumerate(fourier_axes)
        basis = domain.bases[axis]
        if isa(basis, RealFourier)
            if i == 1
                # First Fourier axis is RealFourier - can use RFFT (real-to-complex)
                push!(transform_list, PencilFFTs.Transforms.RFFT())
                uses_rfft = true
            elseif first_fourier_is_real
                # First axis is RealFourier (gets RFFT), subsequent RealFourier must use FFT
                # This produces full complex output for axis 2+, but axis 1 still gets half-spectrum
                if !realfourier_warning_shown
                    @warn "RealFourier basis on axis $axis is not the first Fourier axis. " *
                          "Using FFT instead of RFFT for this axis (output will be full complex size N, " *
                          "not half-spectrum N/2+1). The first RealFourier axis still uses RFFT correctly." maxlog=1
                    realfourier_warning_shown = true
                end
                push!(transform_list, PencilFFTs.Transforms.FFT())
            else
                # First Fourier axis is NOT RealFourier, but this one is - ERROR
                # RFFT can only be applied to dimension 1 in PencilFFTs
                error("RealFourier basis on axis $axis cannot use RFFT because the first Fourier axis " *
                      "(axis $(fourier_axes[1])) is not RealFourier. In MPI mode with PencilFFTs, " *
                      "RFFT can only be applied to dimension 1. Using FFT would produce full complex " *
                      "arrays (size N) where RealFourier expects half-spectrum (size N/2+1). " *
                      "Please reorder your domain bases to place RealFourier first, " *
                      "or use ComplexFourier for this axis.")
            end
        else
            # ComplexFourier uses FFT (complex-to-complex)
            push!(transform_list, PencilFFTs.Transforms.FFT())
        end
    end
    transforms = Tuple(transform_list)

    @info "Setting up $(length(fourier_axes))D FFT for axes: $fourier_axes"

    # Create the PencilFFT plan with 3D pencil decomposition (only for parallel execution)
    # RFFT expects real input, FFT expects complex input
    # If dtype is already complex, use it directly; otherwise wrap in Complex{}
    pencil_dtype = if uses_rfft
        dist.dtype
    else
        dist.dtype <: Complex ? dist.dtype : Complex{dist.dtype}
    end
    pencil = create_pencil(dist, global_shape, 1, dtype=pencil_dtype)

    # Try to create PencilFFT plan
    # CRITICAL: If this fails in MPI mode, we CANNOT safely fall back to local FFTW
    # because that would compute incorrect results on decomposed data
    try
        fft_plan = PencilFFTs.PencilFFTPlan(pencil, transforms)
        push!(dist.transforms, fft_plan)

        # CRITICAL: Store the plan's input/output pencils for field allocation
        # This provides a fallback if plan lookup fails in allocate_data!
        dist.pencil_fft_input = first(fft_plan.plans).pencil_in
        dist.pencil_fft_output = last(fft_plan.plans).pencil_out

        @info "Set up 3D PencilFFT transform for axes $fourier_axes with global shape $global_shape"
        @info "3D parallel decomposition: $(join(dist.mesh, " × ")) processes"
    catch e
        if dist.size > 1
            # In MPI mode, failing to create parallel FFT is a critical error
            @error "PencilFFT plan creation failed in MPI mode - cannot use local FFTW on distributed data" exception=e
            error("PencilFFT plan creation failed with $(dist.size) MPI processes. " *
                  "Local FFTW fallback would produce incorrect results. " *
                  "Please check your PencilFFTs installation or use serial execution.")
        else
            # Serial mode: local FFTW is correct
            @info "PencilFFT plan creation failed in serial mode, using FFTW transforms"
            for axis in fourier_axes
                basis = domain.bases[axis]
                if isa(basis, RealFourier) || isa(basis, ComplexFourier)
                    setup_fftw_transform!(dist, basis, axis)
                end
            end
        end
    end
end

function setup_fftw_transforms_nd!(dist::Distributor, domain::Domain, fourier_axes::Vector{Int})
    """Fallback FFTW transforms for high-dimensional problems"""
    
    @info "Using FFTW fallback for $(length(domain.bases))D problem"
    
    for (i, basis) in enumerate(domain.bases)
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            setup_fftw_transform!(dist, basis, i)
        end
    end
end

function setup_pencil_arrays_3d(dist::Distributor, global_shape::Tuple{Vararg{Int}})
    """Setup 3D PencilArrays configuration"""

    if length(global_shape) != 3
        throw(ArgumentError("3D setup requires 3D global shape, got $(length(global_shape))D"))
    end

    mesh_len = length(dist.mesh)
    if mesh_len < 2 || mesh_len > 3
        throw(ArgumentError("3D setup requires 2D or 3D process mesh, got $(mesh_len)D"))
    end

    # Validate mesh
    if prod(dist.mesh) != dist.size
        throw(ArgumentError("Process mesh $(dist.mesh) incompatible with $(dist.size) processes"))
    end

    # Create PencilArrays configuration
    # For 2D mesh with 3D data (pencil decomposition), only decompose last 2 dimensions
    # For 3D mesh with 3D data, decompose all 3 dimensions
    decomp_dims = mesh_len == 3 ? (true, true, true) : (false, true, true)

    dist.pencil_config = PencilConfig(
        global_shape,
        dist.mesh,
        comm=dist.comm,
        decomp_dims=decomp_dims
    )
    
    decomp_str = mesh_len == 3 ? "3D (all axes)" : "2D pencil (last 2 axes)"
    @info "Created 3D PencilArrays configuration:"
    @info "  Global shape: $global_shape"
    @info "  Process mesh: $(dist.mesh)"
    @info "  Decomposition: $decomp_str"
    
    return dist.pencil_config
end

function create_pencil_3d(dist::Distributor, global_shape::Tuple{Vararg{Int}},
                        decomp_index::Int=1; dtype::Type=dist.dtype)

    """Create a 3D pencil array with specified decomposition"""
    return create_pencil(dist, global_shape, decomp_index; dtype=dtype)
end

# Enhanced 3D transform execution
function forward_transform_3d!(field::ScalarField, target_layout::Symbol=:c)
    """Apply 3D forward transform to field"""
    
    if field.domain === nothing || length(field.domain.bases) != 3
        forward_transform!(field, target_layout)  # Fall back to general case
        return
    end
    
    ensure_layout!(field, :g)  # Start in grid space
    
    # Find appropriate 3D transform
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # Check if this is a 3D transform
            if hasfield(typeof(transform), :dims) && length(transform.dims) <= 3
                set_coeff_data!(field, transform * get_grid_data(field))
                field.current_layout = :c
                @debug "Applied 3D PencilFFT forward transform"
                return
            end
        end
    end
    
    # Fallback to regular transform
    forward_transform!(field, target_layout)
end

function backward_transform_3d!(field::ScalarField, target_layout::Symbol=:g)
    """Apply 3D backward transform to field"""
    
    if field.domain === nothing || length(field.domain.bases) != 3
        backward_transform!(field, target_layout)  # Fall back to general case
        return
    end
    
    ensure_layout!(field, :c)  # Start in coefficient space
    
    # Find appropriate 3D transform
    for transform in field.dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            # Check if this is a 3D transform (use \ for inverse transform)
            if hasfield(typeof(transform), :dims) && length(transform.dims) <= 3
                set_grid_data!(field, transform \ get_coeff_data(field))
                field.current_layout = :g
                @debug "Applied 3D PencilFFT backward transform"
                return
            end
        end
    end
    
    # Fallback to regular transform
    backward_transform!(field, target_layout)
end

"""
    dealias_3d!(field::ScalarField, scales::Union{Real, Vector{Real}})

Apply 3D dealiasing to a scalar field by zeroing out high-frequency modes.

Dealiasing is essential for nonlinear computations to avoid aliasing errors.
The 2/3 rule (scale = 2/3) is commonly used: modes with k > k_max * scale are zeroed.

For Fourier bases, this zeros modes beyond the cutoff frequency.
For Chebyshev bases, this zeros high-order polynomial coefficients.

# Arguments
- `field`: ScalarField to dealias (modified in-place)
- `scales`: Dealiasing scale(s). Can be:
  - A single Real value applied to all dimensions
  - A Vector of scales for each dimension

# Example
```julia
# Apply 2/3 dealiasing rule to all dimensions
dealias_3d!(field, 2/3)

# Apply different scales per dimension
dealias_3d!(field, [2/3, 2/3, 1.0])  # No dealiasing in z
```
"""
function dealias_3d!(field::ScalarField, scales::Union{Real, Vector{Real}})
    if field.domain === nothing
        @warn "Cannot dealias field without domain"
        return
    end

    ensure_layout!(field, :c)

    if get_coeff_data(field) === nothing
        @warn "No coefficient data to dealias"
        return
    end

    ndim = ndims(get_coeff_data(field))
    nbases = length(field.domain.bases)

    # Normalize scales to a vector matching the number of bases
    if isa(scales, Real)
        scale_vec = fill(Float64(scales), nbases)
    else
        if length(scales) >= nbases
            scale_vec = Float64.(scales[1:nbases])
        else
            # Extend with the last value
            scale_vec = vcat(Float64.(scales), fill(Float64(scales[end]), nbases - length(scales)))
        end
    end

    # Get array dimensions
    data_shape = size(get_coeff_data(field))

    # Apply dealiasing for each dimension using the correct basis-aware logic
    # This delegates to apply_basis_dealiasing! which handles each basis type correctly
    for (dim, (basis, scale)) in enumerate(zip(field.domain.bases, scale_vec))
        if dim > ndim
            break
        end

        if scale >= 1.0
            continue  # No dealiasing needed
        end

        # Use the correct basis-aware dealiasing function
        apply_basis_dealiasing!(field, basis, dim, scale)
    end
end

"""
    zero_fourier_modes!(data::AbstractArray, dim::Int, cutoff::Int, n::Int)

Zero out high-frequency Fourier modes beyond the cutoff in dimension `dim`.

For real FFTs, the data is typically stored as complex with size N÷2+1.
Modes to keep: 1:cutoff (DC and low frequencies)
Modes to zero: (cutoff+1):n (high frequencies)
"""
function zero_fourier_modes!(data::AbstractArray, dim::Int, cutoff::Int, n::Int)
    if cutoff >= n
        return
    end

    ndim = ndims(data)

    # Build index ranges for the high-frequency modes to zero
    # Use selectdim for dimension-agnostic indexing
    for i in (cutoff + 1):n
        indices = ntuple(d -> d == dim ? i : Colon(), ndim)
        view(data, indices...) .= zero(eltype(data))
    end
end

"""
    zero_polynomial_modes!(data::AbstractArray, dim::Int, cutoff::Int, n::Int)

Zero out high-order polynomial coefficients beyond the cutoff in dimension `dim`.

For Chebyshev/Jacobi bases, coefficients are stored in order of polynomial degree.
Modes to keep: 1:cutoff (low-order polynomials)
Modes to zero: (cutoff+1):n (high-order polynomials)
"""
function zero_polynomial_modes!(data::AbstractArray, dim::Int, cutoff::Int, n::Int)
    if cutoff >= n
        return
    end

    ndim = ndims(data)

    # Zero high-order coefficients
    for i in (cutoff + 1):n
        indices = ntuple(d -> d == dim ? i : Colon(), ndim)
        view(data, indices...) .= zero(eltype(data))
    end
end

"""
    dealias_field!(field::ScalarField)

Apply dealiasing using the field's basis dealias parameters.
"""
function dealias_field!(field::ScalarField)
    if field.domain === nothing
        return
    end

    # Get dealias scales from bases
    scales = Float64[]
    for basis in field.domain.bases
        if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :dealias)
            push!(scales, Float64(basis.meta.dealias))
        else
            push!(scales, 1.0)  # No dealiasing
        end
    end

    if all(s -> s >= 1.0, scales)
        # No dealiasing needed
        return
    end

    dealias_3d!(field, scales)
end

"""
    synchronize_transforms!(transforms::Vector)

Synchronize all transform operations to ensure completion before proceeding.

This is a no-op for CPU-only execution. When GPU support is added, this would
call the appropriate GPU synchronization (e.g., `CUDA.synchronize()`) to ensure
all asynchronous GPU operations have completed.

# Arguments
- `transforms`: Vector of transform objects (currently unused)
"""
function synchronize_transforms!(transforms::Vector)
    # No-op for CPU-only mode
    # For GPU: would call CUDA.synchronize() or equivalent
    return nothing
end

function is_pencil_compatible(bases::Tuple{Vararg{Basis}})
    """Check if bases are compatible with parallel transforms (PencilArrays)"""
    ndim = length(bases)

    if ndim < 2
        return false  # Parallel transforms are for multi-dimensional problems
    end

    # Count Fourier bases
    fourier_count = 0
    for basis in bases
        if isa(basis, RealFourier) || isa(basis, ComplexFourier)
            fourier_count += 1
        end
    end

    # Parallel transforms require at least one Fourier dimension (uses PencilFFTs)
    return fourier_count >= 1 && ndim >= 2
end

function is_3d_pencil_optimal(bases::Tuple{Vararg{Basis}})
    """Check if 3D PencilFFTs would be optimal for these bases"""
    if length(bases) != 3
        return false
    end

    fourier_count = count(b -> isa(b, Union{RealFourier, ComplexFourier}), bases)

    # 3D PencilFFTs is optimal when:
    # - All 3 dimensions are Fourier (best case)
    # - 2 out of 3 dimensions are Fourier (good case)
    # - Even 1 Fourier dimension can benefit from 3D decomposition
    return fourier_count >= 1
end

