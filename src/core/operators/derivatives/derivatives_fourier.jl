# Fourier derivative implementations for local and distributed layouts.

# ============================================================================
# Fourier Derivative Implementation
# ============================================================================

"""
    evaluate_fourier_derivative!(result, operand, axis, order, layout)

Evaluate Fourier derivative using FFT operations along a single axis.
Supports both CPU (FFTW) and GPU (CUFFT via extension) arrays.

This function computes spectral derivatives by:
1. Applying FFT along the specified axis only (not full N-D FFT)
2. Multiplying by (ik)^order for each wavenumber k along that axis
3. Applying inverse FFT along the same axis to get derivative in grid space

This correctly handles multi-dimensional derivatives where we want d/dx
to only apply FFT along the x-axis, not all axes.
"""
function evaluate_fourier_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    # Ensure operand is in grid space
    if operand.current_layout != :g
        @warn "evaluate_fourier_derivative!: operand not in grid space, results may be unexpected"
    end

    dist = operand.dist
    ndim = length(operand.bases)

    # CRITICAL: Check if axis is distributed (requires MPI transpose for correct derivative)
    if dist.size > 1 && ndim >= 2 && dist.mesh !== nothing
        ndims_mesh = length(dist.mesh)
        # Determine decomposed dimensions based on convention
        decomp_dims = if dist.use_pencil_arrays
            # PencilArrays convention: decompose LAST ndims_mesh dimensions
            if ndim >= ndims_mesh
                ntuple(i -> ndim - ndims_mesh + i, ndims_mesh)
            else
                ntuple(identity, ndim)
            end
        else
            # TransposableField convention: decompose FIRST ndims_mesh dimensions
            ntuple(identity, min(ndims_mesh, ndim))
        end

        if axis in decomp_dims
            # Axis is distributed - need MPI transpose to compute derivative correctly
            _evaluate_distributed_fourier_derivative!(result, operand, axis, order, layout, decomp_dims)
            return
        end
    end

    # Local case: axis is not distributed, can compute directly
    _evaluate_local_fourier_derivative!(result, operand, axis, order, layout)
end

"""
    _evaluate_distributed_fourier_derivative!(result, operand, axis, order, layout, decomp_dims)

Evaluate Fourier derivative on a distributed axis using coefficient-space operations.

The correct approach for MPI differentiation:
1. Transform to coefficient space using PencilFFTs (handles distributed FFT correctly)
2. Multiply coefficients by (ik)^order - works on local portion of distributed coefficients
3. Transform back using PencilFFTs if grid space result is needed

This is simpler and more correct than manual transposes because PencilFFTs handles
all MPI communication internally.
"""
function _evaluate_distributed_fourier_derivative!(result::ScalarField, operand::ScalarField,
                                                    axis::Int, order::Int, layout::Symbol,
                                                    decomp_dims::Tuple)
    dist = operand.dist
    basis = operand.bases[axis]

    # Step 1: Ensure operand is in coefficient space using PencilFFTs
    # Make a working copy to avoid modifying the original
    work_field = copy(operand)
    forward_transform!(work_field)

    # Get the coefficient data (distributed across processes)
    coeff_data = get_coeff_data(work_field)
    if coeff_data === nothing
        error("Forward transform did not produce coefficient data")
    end

    # Step 2: Apply spectral derivative in coefficient space
    # Each process works on its local portion of the distributed coefficient array
    # CRITICAL: Determine if this axis uses RFFT (only first RealFourier axis does)
    uses_rfft = _is_first_real_fourier_axis(operand.bases, axis)
    _apply_spectral_derivative_distributed!(coeff_data, basis, axis, order, dist; uses_rfft=uses_rfft)

    # Step 3: Allocate result coefficient data if needed
    if get_coeff_data(result) === nothing
        coeff_dtype = eltype(coeff_data)
        if dist.use_pencil_arrays && isa(coeff_data, PencilArrays.PencilArray)
            # CRITICAL: Use PencilFFTs.allocate_output for compatible coeff-space array
            # This ensures the array works with PencilFFTs' mul!/ldiv!
            pencil_plan = _find_pencil_plan(dist)
            if pencil_plan !== nothing
                set_coeff_data!(result, PencilFFTs.allocate_output(pencil_plan))
            else
                # Fallback: use global size for create_pencil
                global_coeff_shape = PencilArrays.size_global(coeff_data)
                set_coeff_data!(result, create_pencil(dist, global_coeff_shape, nothing, dtype=coeff_dtype))
            end
        else
            set_coeff_data!(result, similar(coeff_data))
        end
    end

    # Copy the derivative coefficients to result
    if isa(coeff_data, PencilArrays.PencilArray) && isa(get_coeff_data(result), PencilArrays.PencilArray)
        copyto!(parent(get_coeff_data(result)), parent(coeff_data))
    else
        get_coeff_data(result) .= coeff_data
    end
    result.current_layout = :c

    # Step 4: Transform back to grid space if requested
    if layout == :g
        backward_transform!(result)
    end
end

"""
    _is_first_real_fourier_axis(bases, axis)

Check if the given axis is the first RealFourier axis in the bases array.

RFFT (real-input FFT) halves storage by exploiting Hermitian symmetry: for N
real grid points, only N/2+1 complex coefficients are stored.  However, this
compaction can only be applied along **one** axis per multi-dimensional
transform.  FFTW and PencilFFTs choose the **first** RealFourier axis for
this; all subsequent RealFourier axes use a full complex-to-complex FFT and
retain N coefficients.

This distinction matters for spectral derivative operators because the
wavenumber array size differs:
  - First RealFourier axis: k = [0, 1, …, N/2]         (N/2+1 values)
  - Later RealFourier axes: k = [0, 1, …, N/2-1, -N/2, …, -1]  (N values)

Returns true if:
- The current axis is RealFourier, AND
- No earlier axis is RealFourier
"""
function _is_first_real_fourier_axis(bases, axis::Int)
    # If the current axis is not RealFourier, it definitely doesn't use RFFT
    if !isa(bases[axis], RealFourier)
        return false
    end

    # Check if there's any earlier RealFourier axis
    for i in 1:(axis-1)
        if isa(bases[i], RealFourier)
            return false  # There's an earlier RealFourier axis, so this one uses FFT not RFFT
        end
    end

    return true  # This is the first RealFourier axis, uses RFFT
end

"""
    _apply_spectral_derivative_distributed!(coeff_data, basis, axis, order, dist)

Apply spectral derivative operator to distributed coefficient data.
Each process multiplies its local coefficients by the appropriate (ik)^order factor.

For PencilArrays: uses axes_local to determine which wavenumber indices this process owns.

IMPORTANT: For RealFourier bases, the `uses_rfft` parameter must be set correctly:
- `uses_rfft=true`: First RealFourier axis, coefficient size is N/2+1
- `uses_rfft=false`: Subsequent RealFourier axes, coefficient size is N (full FFT)
"""
function _apply_spectral_derivative_distributed!(coeff_data::PencilArrays.PencilArray,
                                                   basis::Union{RealFourier, ComplexFourier},
                                                   axis::Int, order::Int, dist::Distributor;
                                                   uses_rfft::Bool=false)
    # Get the local data and the global index ranges for this process
    local_data = parent(coeff_data)
    pencil = PencilArrays.pencil(coeff_data)
    local_axes = pencil.axes_local  # Tuple of UnitRange for each dimension

    # CRITICAL: PencilArrays can use a permutation where parent array axes differ from logical axes
    # Get the permutation and map logical axis to physical axis in parent
    perm = PencilArrays.permutation(coeff_data)
    perm_tuple = Tuple(perm)  # Convert to tuple - identity perm gives (1, 2, ..., n)
    # perm_tuple[physical] = logical, so find physical where perm_tuple[physical] == logical_axis
    physical_axis = findfirst(==(axis), perm_tuple)
    if physical_axis === nothing
        # Fallback: if permutation doesn't contain axis (shouldn't happen), use axis directly
        physical_axis = axis
    end

    # The per-rank derivative multiplier (im*k)^order depends only on the basis,
    # order, uses_rfft, and this rank's owned index range — all fixed across a run.
    # Cache it so the wavenumber array isn't rebuilt and re-sliced every RHS eval.
    local_range = axis <= length(local_axes) ? local_axes[axis] : nothing
    deriv_mult = _get_cached_dist_deriv_mult!(coeff_data, basis, axis, order, uses_rfft, local_range)

    # Apply to local data along the PHYSICAL axis in parent array (accounting for permutation)
    mult_shape = ntuple(i -> i == physical_axis ? length(deriv_mult) : 1, ndims(local_data))
    local_data .*= reshape(deriv_mult, mult_shape...)
end

"""
Cached `(ik)^order` multiplier for this rank's owned slice of a distributed
Fourier axis. Keyed by `(order, uses_rfft, axis, local_range_lo, local_range_hi)`
in `basis.transforms`; the range bounds in the key guarantee a distinct entry if
the decomposition ever differs, so cached values can never be misapplied. All
size/bounds validation runs on the cache-miss path (config is fixed per run).
"""
function _get_cached_dist_deriv_mult!(coeff_data::PencilArrays.PencilArray,
                                      basis::Union{RealFourier, ComplexFourier},
                                      axis::Int, order::Int, uses_rfft::Bool, local_range)
    lr_lo = local_range === nothing ? 0 : Int(first(local_range))
    lr_hi = local_range === nothing ? 0 : Int(last(local_range))
    cache_key = (:dist_deriv_mult, order, uses_rfft, axis, lr_lo, lr_hi)
    cached = get(basis.transforms, cache_key, nothing)
    cached !== nothing && return cached::Vector{ComplexF64}

    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    N_global = basis.meta.size

    # CRITICAL: In PencilFFTs, RFFT is only used on the FIRST RealFourier axis.
    # Subsequent RealFourier axes use FFT with full size N (not N/2+1).
    if isa(basis, RealFourier) && uses_rfft
        N_coeff = div(N_global, 2) + 1
        k0 = 2π / L
        k_global = collect(0:(N_coeff-1)) .* k0  # rfft: [0, 1, …, N/2]
    elseif isa(basis, RealFourier)
        N_coeff = N_global
        k0 = 2π / L
        k_global = _fftfreq(N_global) .* N_global .* k0
    else
        N_coeff = N_global
        k0 = 2π / L
        k_global = _fftfreq(N_global) .* N_global .* k0
    end

    # CRITICAL: Validate coefficient global size matches expected N_coeff (catches
    # uses_rfft / data-size mismatch).
    global_size_axis = PencilArrays.size_global(coeff_data)[axis]
    if global_size_axis != N_coeff
        error("Spectral derivative coefficient size mismatch on axis $axis: " *
              "global coefficient size is $global_size_axis but expected $N_coeff " *
              "(basis=$(typeof(basis)), uses_rfft=$uses_rfft, N_global=$N_global). " *
              "Check that uses_rfft correctly identifies the first RealFourier axis.")
    end

    if local_range !== nothing
        if last(local_range) > length(k_global)
            error("Spectral derivative index out of bounds: local_range=$local_range but k_global has $(length(k_global)) elements. " *
                  "This may indicate a mismatch between coefficient sizing and wavenumber computation. " *
                  "axis=$axis, uses_rfft=$uses_rfft, basis=$(typeof(basis)), N_global=$N_global")
        end
        k_local = k_global[local_range]
    else
        k_local = k_global  # Axis not distributed: use full wavenumber array
    end

    deriv_mult = ComplexF64.((im .* k_local) .^ order)
    basis.transforms[cache_key] = deriv_mult
    return deriv_mult
end

function _apply_spectral_derivative_distributed!(coeff_data::AbstractArray,
                                                   basis::Union{RealFourier, ComplexFourier},
                                                   axis::Int, order::Int, dist::Distributor)
    # Fallback for non-PencilArray data (shouldn't happen in MPI mode, but handle gracefully)
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    N = size(coeff_data, axis)

    if isa(basis, RealFourier)
        k0 = 2π / L
        k_axis = collect(0:(N-1)) .* k0
    else
        k0 = 2π / L
        k_axis = _fftfreq(N) .* N .* k0
    end

    deriv_mult = (im .* k_axis) .^ order
    mult_shape = ntuple(i -> i == axis ? length(deriv_mult) : 1, ndims(coeff_data))
    coeff_data .*= reshape(deriv_mult, mult_shape...)
end

"""
    _get_cached_deriv_mult(basis::FourierBasis, N::Int, L::Float64, order::Int)

Get or compute cached derivative multiplier `(ik)^order` for Fourier derivatives.
The multiplier depends only on the basis parameters and derivative order, so it
is computed once and cached in the basis's transforms dict.
"""
function _get_cached_deriv_mult(basis::Union{RealFourier, ComplexFourier}, N::Int, L::Float64, order::Int)
    # Tuple key avoids string allocation on every call
    cache_key = (:deriv_mult, N, order)
    cached = get(basis.transforms, cache_key, nothing)
    if cached !== nothing
        return cached::Vector{ComplexF64}
    end
    k_axis = _fftfreq(N, L/N) .* 2π
    deriv_mult = (im .* k_axis) .^ order
    basis.transforms[cache_key] = deriv_mult
    return deriv_mult
end

"""
    _evaluate_local_fourier_derivative!(result, operand, axis, order, layout)

Evaluate Fourier derivative on a local axis (no MPI needed).
"""
function _evaluate_local_fourier_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    # Get the basis for the specified axis
    basis = operand.bases[axis]
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]

    # Use grid data for computation
    # For PencilArrays, extract the parent (local) array for FFT operations
    # Note: fft() is out-of-place (creates new output), so no copy needed
    operand_grid = get_grid_data(operand)
    data_g = isa(operand_grid, PencilArrays.PencilArray) ? parent(operand_grid) : operand_grid

    dims = ndims(data_g)
    data_shape = size(data_g)

    # Check if we're on GPU
    use_gpu = is_gpu_array(data_g)

    # Get cached derivative multiplier (avoids re-allocating wavenumber arrays every call)
    deriv_mult_cpu = _get_cached_deriv_mult(basis, data_shape[axis], Float64(L), order)

    if use_gpu
        # GPU path: use broadcasting for all operations
        evaluate_fourier_derivative_gpu!(result, data_g, deriv_mult_cpu, axis, dims, data_shape, layout)
    else
        # CPU path: optimized with explicit loops
        evaluate_fourier_derivative_cpu!(result, data_g, deriv_mult_cpu, axis, dims, data_shape, layout)
    end
end

"""
    _write_to_grid_data!(result, deriv_g)

Helper function to write derivative result to grid data, handling PencilArrays.
"""
function _write_to_grid_data!(result::ScalarField, deriv_g::AbstractArray)
    result_grid = get_grid_data(result)
    if isa(result_grid, PencilArrays.PencilArray)
        parent(result_grid) .= deriv_g
    else
        result_grid .= deriv_g
    end
end

"""
    evaluate_fourier_derivative_gpu!(result, data_g, deriv_mult, axis, dims, data_shape, layout)

GPU-specific implementation using broadcasting operations.
"""
function evaluate_fourier_derivative_gpu!(result::ScalarField, data_g::AbstractArray, deriv_mult_cpu::AbstractVector, axis::Int, dims::Int, data_shape::Tuple, layout::Symbol)
    # Move derivative multiplier to GPU
    deriv_mult = copy_to_device(deriv_mult_cpu, data_g)

    # GPU FFT and element-wise operations via broadcasting
    # Note: For GPU, fft/ifft dispatch to CUFFT when CUDA.jl is loaded
    if dims == 1
        f_hat = fft(data_g)
        f_hat .*= deriv_mult
        deriv_g = result.dtype <: Real ? real.(ifft(f_hat)) : ifft(f_hat)
        _write_to_grid_data!(result, deriv_g)
        result.current_layout = :g

    elseif dims == 2
        f_hat = fft(data_g, axis)

        # Reshape deriv_mult for broadcasting along the correct axis
        if axis == 1
            # Shape: (N, 1) to broadcast along first dimension
            mult_shaped = reshape(deriv_mult, :, 1)
        else  # axis == 2
            # Shape: (1, N) to broadcast along second dimension
            mult_shaped = reshape(deriv_mult, 1, :)
        end

        f_hat .*= mult_shaped
        deriv_g = result.dtype <: Real ? real.(ifft(f_hat, axis)) : ifft(f_hat, axis)
        _write_to_grid_data!(result, deriv_g)
        result.current_layout = :g

    elseif dims == 3
        f_hat = fft(data_g, axis)

        # Reshape deriv_mult for broadcasting along the correct axis
        if axis == 1
            mult_shaped = reshape(deriv_mult, :, 1, 1)
        elseif axis == 2
            mult_shaped = reshape(deriv_mult, 1, :, 1)
        else  # axis == 3
            mult_shaped = reshape(deriv_mult, 1, 1, :)
        end

        f_hat .*= mult_shaped
        deriv_g = result.dtype <: Real ? real.(ifft(f_hat, axis)) : ifft(f_hat, axis)
        _write_to_grid_data!(result, deriv_g)
        result.current_layout = :g
    else
        throw(ArgumentError("Fourier derivative only implemented for 1D, 2D, and 3D"))
    end

    # If coefficient space is requested, use full forward transform
    # (handles mixed bases correctly, not just Fourier axes)
    if layout == :c
        ensure_layout!(result, :c)
    end
end

# Cached FFT plans + complex/real buffers for the CPU Fourier derivative, keyed by
# (local shape, axis, dims, complex eltype). Reused across calls so the per-call
# fft/ifft outputs are not reallocated. Buffers are filled/consumed entirely within
# one derivative call (not held across calls), so a single cached set per key is safe.
const _DERIV_FFT_WS = Dict{Tuple, Any}()

function _get_deriv_workspace!(data_g::AbstractArray, axis::Int, dims::Int)
    CT = complex(float(real(eltype(data_g))))
    RT = real(CT)
    key = (size(data_g), axis, dims, CT)
    return get!(_DERIV_FFT_WS, key) do
        cin  = Array{CT}(undef, size(data_g))
        fhat = Array{CT}(undef, size(data_g))
        rout = Array{RT}(undef, size(data_g))
        pf  = dims == 1 ? plan_fft(cin)   : plan_fft(cin, axis)
        pin = dims == 1 ? plan_ifft(fhat) : plan_ifft(fhat, axis)
        (cin, fhat, rout, pf, pin)
    end
end

"""Multiply `fhat` by the derivative factor (ik)^order along `axis`, in place.
Concrete `fhat` type ⇒ no per-element boxing (the caller's `fhat` is `Any`-typed)."""
function _apply_deriv_mult!(fhat::AbstractArray, deriv_mult::AbstractVector, axis::Int, dims::Int, data_shape::Tuple)
    if dims == 1
        @inbounds for i in eachindex(fhat); fhat[i] *= deriv_mult[i]; end
    elseif dims == 2
        if axis == 1
            @inbounds for i in 1:data_shape[1]
                factor = deriv_mult[i]
                for j in 1:data_shape[2]; fhat[i, j] *= factor; end
            end
        else
            @inbounds for j in 1:data_shape[2]
                factor = deriv_mult[j]
                for i in 1:data_shape[1]; fhat[i, j] *= factor; end
            end
        end
    else  # dims == 3
        if axis == 1
            @inbounds for i in 1:data_shape[1]
                factor = deriv_mult[i]
                for j in 1:data_shape[2], k in 1:data_shape[3]; fhat[i, j, k] *= factor; end
            end
        elseif axis == 2
            @inbounds for j in 1:data_shape[2]
                factor = deriv_mult[j]
                for i in 1:data_shape[1], k in 1:data_shape[3]; fhat[i, j, k] *= factor; end
            end
        else
            @inbounds for k in 1:data_shape[3]
                factor = deriv_mult[k]
                for i in 1:data_shape[1], j in 1:data_shape[2]; fhat[i, j, k] *= factor; end
            end
        end
    end
    return fhat
end

"""
    evaluate_fourier_derivative_cpu!(result, data_g, deriv_mult, axis, dims, data_shape, layout)

CPU-specific implementation using optimized loops.
"""
function evaluate_fourier_derivative_cpu!(result::ScalarField, data_g::AbstractArray, deriv_mult::AbstractVector, axis::Int, dims::Int, data_shape::Tuple, layout::Symbol)
    if dims in (1, 2, 3)
        # Cached FFT plans + buffers (keyed by local shape/axis/eltype) reuse memory
        # across calls instead of allocating fft/ifft outputs each time. Same
        # semantics as fft(data_g, axis); for MPI the framework already orients the
        # derivative axis locally before calling this, so per-rank caching is valid.
        cin, fhat, rout, pf, pin = _get_deriv_workspace!(data_g, axis, dims)
        copyto!(cin, data_g)
        mul!(fhat, pf, cin)
        # The deriv-multiplier loop runs through a function barrier so it is not
        # boxed (cin/fhat come from an `Any`-typed cache here).
        _apply_deriv_mult!(fhat, deriv_mult, axis, dims, data_shape)
        mul!(cin, pin, fhat)   # cin = ifft(fhat) along axis (normalized)
        if result.dtype <: Real
            @inbounds @. rout = real(cin)
            _write_to_grid_data!(result, rout)
        else
            _write_to_grid_data!(result, cin)
        end
        result.current_layout = :g
    else
        throw(ArgumentError("Fourier derivative only implemented for 1D, 2D, and 3D"))
    end

    # If coefficient space is requested, use full forward transform
    # (handles mixed bases correctly, not just Fourier axes)
    if layout == :c
        ensure_layout!(result, :c)
    end
end
