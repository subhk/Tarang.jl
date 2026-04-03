"""
    Derivative evaluation functions

This file contains all differentiation implementations including:
- Gradient, divergence, and differentiate evaluators
- Fourier derivative functions (distributed and local, CPU and GPU)
- Chebyshev derivative functions
- Legendre derivative functions
- Matrix application helpers
"""

using LinearAlgebra
using SparseArrays
using FFTW

# ============================================================================
# Gradient and Divergence Evaluation
# ============================================================================

"""Evaluate gradient operator."""
function evaluate_gradient(grad_op::Gradient, layout::Symbol=:g)
    operand = grad_op.operand
    coordsys = grad_op.coordsys

    if isa(operand, ScalarField)
        # Create vector field for result
        result = VectorField(operand.dist, coordsys, "grad_$(operand.name)", operand.bases, operand.dtype)

        # Compute partial derivatives for each component
        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            # Apply differentiation operator
            result.components[i] = evaluate_differentiate(Differentiate(operand, coord, 1), layout)
        end

        return result
    else
        throw(ArgumentError("Gradient not implemented for operand type $(typeof(operand))"))
    end
end

"""Evaluate divergence operator."""
function evaluate_divergence(div_op::Divergence, layout::Symbol=:g)
    operand = div_op.operand

    if isa(operand, VectorField)
        # Sum partial derivatives of components
        coordsys = operand.coordsys

        # Create result using copy() to preserve PencilArray structure
        result = copy(operand.components[1])
        result.name = "div_$(operand.name)"

        # Initialize result to zero — ensure data is allocated even if copy didn't provide it
        ensure_layout!(result, layout)
        if layout == :g
            grid_data = get_grid_data(result)
            if grid_data === nothing
                # Allocate grid data if not present (copy may not have provided it)
                set_grid_data!(result, zeros(eltype(get_grid_data(operand.components[1])),
                                             size(get_grid_data(operand.components[1]))))
            elseif isa(grid_data, PencilArrays.PencilArray)
                fill!(parent(grid_data), zero(eltype(grid_data)))
            else
                fill!(grid_data, zero(eltype(grid_data)))
            end
        else
            coeff_data = get_coeff_data(result)
            if coeff_data === nothing
                set_coeff_data!(result, zeros(eltype(get_coeff_data(operand.components[1])),
                                              size(get_coeff_data(operand.components[1]))))
            elseif isa(coeff_data, PencilArrays.PencilArray)
                fill!(parent(coeff_data), zero(eltype(coeff_data)))
            else
                fill!(coeff_data, zero(eltype(coeff_data)))
            end
        end

        for (i, coord_name) in enumerate(coordsys.names)
            coord = coordsys[coord_name]
            # Add d(u_i)/d(x_i) — accumulate into field data directly, not via symbolic +
            component_deriv = evaluate_differentiate(Differentiate(operand.components[i], coord, 1), layout)
            if layout == :g
                get_grid_data(result) .+= get_grid_data(component_deriv)
            else
                get_coeff_data(result) .+= get_coeff_data(component_deriv)
            end
        end

        return result
    else
        throw(ArgumentError("Divergence not implemented for operand type $(typeof(operand))"))
    end
end

# ============================================================================
# Differentiate Evaluation
# ============================================================================

"""Evaluate differentiation operator."""
function evaluate_differentiate(diff_op::Differentiate, layout::Symbol=:g)
    operand = diff_op.operand
    coord = diff_op.coord
    order = diff_op.order

    if !isa(operand, ScalarField)
        throw(ArgumentError(
            "Differentiation currently only supports scalar fields, got $(typeof(operand)). " *
            "For VectorField, differentiate each component: " *
            "`evaluate_differentiate(Differentiate(v.components[i], coord, order))`"))
    end

    # Short-circuit for zero-order derivative (identity operation)
    if order == 0
        result = copy(operand)
        result.name = "d0_$(operand.name)"
        ensure_layout!(result, layout)
        return result
    end

    # Find which basis corresponds to this coordinate
    basis_index = nothing
    for (i, basis) in enumerate(operand.bases)
        if basis.meta.element_label == coord.name
            basis_index = i
            break
        end
    end

    if basis_index === nothing
        # Coordinate not present in bases (constant dimension): derivative is zero
        # Use copy() to preserve PencilArray structure in MPI mode
        result = copy(operand)
        result.name = "d$(order)_$(operand.name)_d$(coord.name)$(order)"
        ensure_layout!(result, layout)

        # Zero out the data
        if layout == :g
            grid_data = get_grid_data(result)
            if grid_data !== nothing
                if isa(grid_data, PencilArrays.PencilArray)
                    fill!(parent(grid_data), zero(eltype(grid_data)))
                else
                    fill!(grid_data, zero(eltype(grid_data)))
                end
            end
        else
            coeff_data = get_coeff_data(result)
            if coeff_data !== nothing
                if isa(coeff_data, PencilArrays.PencilArray)
                    fill!(parent(coeff_data), zero(eltype(coeff_data)))
                else
                    fill!(coeff_data, zero(eltype(coeff_data)))
                end
            end
        end

        return result
    end

    basis = operand.bases[basis_index]
    # Use copy() to preserve PencilArray structure in MPI mode
    result = copy(operand)
    result.name = "d$(order)_$(operand.name)_d$(coord.name)$(order)"

    # Apply differentiation based on basis type
    if isa(basis, RealFourier) || isa(basis, ComplexFourier)
        evaluate_fourier_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, ChebyshevT)
        evaluate_chebyshev_derivative!(result, operand, basis_index, order, layout)
    elseif isa(basis, Legendre)
        evaluate_legendre_derivative!(result, operand, basis_index, order, layout)
    else
        throw(ArgumentError(
            "Differentiation not implemented for basis type $(typeof(basis)). " *
            "Supported basis types: RealFourier, ComplexFourier, ChebyshevT, Legendre. " *
            "Check that the coordinate '$(coord.name)' has a valid basis assigned."))
    end

    return result
end

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
            pencil_plan = nothing
            for transform in dist.transforms
                if isa(transform, PencilFFTs.PencilFFTPlan)
                    pencil_plan = transform
                    break
                end
            end
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

    # Compute wavenumbers for the global axis
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    N_global = basis.meta.size

    # CRITICAL: In PencilFFTs, RFFT is only used on the FIRST RealFourier axis.
    # Subsequent RealFourier axes use FFT with full size N (not N/2+1).
    # The `uses_rfft` parameter indicates whether this specific axis used RFFT.
    if isa(basis, RealFourier) && uses_rfft
        # First RealFourier axis: coefficient size is div(N, 2) + 1
        N_coeff = div(N_global, 2) + 1
        k0 = 2π / L
        # Wavenumbers for rfft: [0, 1, 2, ..., N/2]
        k_global = collect(0:(N_coeff-1)) .* k0
    elseif isa(basis, RealFourier)
        # Subsequent RealFourier axis: uses FFT with full size N
        # Use fftfreq pattern for full complex spectrum
        N_coeff = N_global
        k0 = 2π / L
        k_global = fftfreq(N_global) .* N_global .* k0
    else
        # For ComplexFourier, use fftfreq pattern
        N_coeff = N_global
        k0 = 2π / L
        k_global = fftfreq(N_global) .* N_global .* k0
    end

    # CRITICAL: Validate that coefficient data global size matches expected N_coeff
    # This catches mismatches between uses_rfft setting and actual data size
    global_size_axis = PencilArrays.size_global(coeff_data)[axis]
    if global_size_axis != N_coeff
        error("Spectral derivative coefficient size mismatch on axis $axis: " *
              "global coefficient size is $global_size_axis but expected $N_coeff " *
              "(basis=$(typeof(basis)), uses_rfft=$uses_rfft, N_global=$N_global). " *
              "Check that uses_rfft correctly identifies the first RealFourier axis.")
    end

    # Get the range of global indices this process owns for the derivative axis
    if axis <= length(local_axes)
        local_range = local_axes[axis]
        # Validate that local_range is within k_global bounds
        if last(local_range) > length(k_global)
            error("Spectral derivative index out of bounds: local_range=$local_range but k_global has $(length(k_global)) elements. " *
                  "This may indicate a mismatch between coefficient sizing and wavenumber computation. " *
                  "axis=$axis, uses_rfft=$uses_rfft, basis=$(typeof(basis)), N_global=$N_global")
        end
        k_local = k_global[local_range]
    else
        # Axis is not distributed, use full wavenumber array
        k_local = k_global
    end

    # Compute derivative multiplier: (ik)^order
    deriv_mult = (im .* k_local) .^ order

    # Apply to local data along the PHYSICAL axis in parent array (accounting for permutation)
    mult_shape = ntuple(i -> i == physical_axis ? length(deriv_mult) : 1, ndims(local_data))
    local_data .*= reshape(deriv_mult, mult_shape...)
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
        k_axis = fftfreq(N) .* N .* k0
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
    cache_key = "deriv_mult_$(N)_$(order)"
    cached = get(basis.transforms, cache_key, nothing)
    if cached !== nothing
        return cached::Vector{ComplexF64}
    end
    k_axis = fftfreq(N, L/N) .* 2π
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
    operand_grid = get_grid_data(operand)
    if isa(operand_grid, PencilArrays.PencilArray)
        data_g = copy(parent(operand_grid))  # Copy local array for FFT
    else
        data_g = copy(operand_grid)  # Regular array copy
    end

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
        deriv_g = real.(ifft(f_hat))
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
        deriv_g = real.(ifft(f_hat, axis))
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
        deriv_g = real.(ifft(f_hat, axis))
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

"""
    evaluate_fourier_derivative_cpu!(result, data_g, deriv_mult, axis, dims, data_shape, layout)

CPU-specific implementation using optimized loops.
"""
function evaluate_fourier_derivative_cpu!(result::ScalarField, data_g::AbstractArray, deriv_mult::AbstractVector, axis::Int, dims::Int, data_shape::Tuple, layout::Symbol)
    if dims == 1
        # 1D case - FFT along only dimension
        f_hat = fft(data_g)

        # Apply derivative: multiply by (ik)^order
        @inbounds for i in 1:length(f_hat)
            f_hat[i] *= deriv_mult[i]
        end

        # Inverse transform
        deriv_g = real.(ifft(f_hat))
        _write_to_grid_data!(result, deriv_g)
        result.current_layout = :g

    elseif dims == 2
        # 2D case: apply FFT only along the specified axis
        f_hat = fft(data_g, axis)

        if axis == 1
            # Derivative along first axis - wavenumbers vary with first index
            @inbounds for i in 1:data_shape[1]
                factor = deriv_mult[i]
                for j in 1:data_shape[2]
                    f_hat[i, j] *= factor
                end
            end
        else  # axis == 2
            # Derivative along second axis - wavenumbers vary with second index
            @inbounds for j in 1:data_shape[2]
                factor = deriv_mult[j]
                for i in 1:data_shape[1]
                    f_hat[i, j] *= factor
                end
            end
        end

        # Inverse transform along same axis
        deriv_g = real.(ifft(f_hat, axis))
        _write_to_grid_data!(result, deriv_g)
        result.current_layout = :g

    elseif dims == 3
        # 3D case: apply FFT only along the specified axis
        f_hat = fft(data_g, axis)

        if axis == 1
            @inbounds for i in 1:data_shape[1]
                factor = deriv_mult[i]
                for j in 1:data_shape[2], k in 1:data_shape[3]
                    f_hat[i, j, k] *= factor
                end
            end
        elseif axis == 2
            @inbounds for j in 1:data_shape[2]
                factor = deriv_mult[j]
                for i in 1:data_shape[1], k in 1:data_shape[3]
                    f_hat[i, j, k] *= factor
                end
            end
        else  # axis == 3
            @inbounds for k in 1:data_shape[3]
                factor = deriv_mult[k]
                for i in 1:data_shape[1], j in 1:data_shape[2]
                    f_hat[i, j, k] *= factor
                end
            end
        end

        # Inverse transform along same axis
        deriv_g = real.(ifft(f_hat, axis))
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

# ============================================================================
# Chebyshev Derivative Implementation
# ============================================================================

"""
    evaluate_chebyshev_derivative!(result, operand, axis, order, layout)

Evaluate Chebyshev derivative using direct DCT operations.
Supports both CPU and GPU arrays (GPU arrays are processed on CPU for DCT).

This function computes Chebyshev spectral derivatives by:
1. Applying DCT-I to grid data to get Chebyshev coefficients
2. Applying the Chebyshev derivative recurrence on coefficients
3. Applying DCT-I (inverse) to get derivative in grid space

For Chebyshev polynomials on [-1, 1]:
d/dx T_n(x) = n * U_{n-1}(x)
where U_n are Chebyshev polynomials of the second kind.

Using the recurrence relation for derivatives in terms of T_n:
c'_{n-1} = 2*n*c_n + c'_{n+1}  (backward recurrence)
"""
function evaluate_chebyshev_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    if order < 0
        throw(ArgumentError("Chebyshev derivative order must be non-negative, got $order"))
    end
    # NOTE: MPI parallelization only supports pure Fourier domains.
    # Chebyshev derivatives are always local since MPI + Chebyshev is not supported.
    _evaluate_local_chebyshev_derivative!(result, operand, axis, order, layout)
end

"""
    _evaluate_local_chebyshev_derivative!(result, operand, axis, order, layout)

Evaluate Chebyshev derivative on a local axis (no MPI needed).
"""
function _evaluate_local_chebyshev_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    # Get the basis for the specified axis
    basis = operand.bases[axis]
    N = basis.meta.size
    a, b = basis.meta.bounds

    if b <= a
        throw(ArgumentError("Chebyshev basis bounds must satisfy a < b, got ($a, $b)"))
    end

    # Domain transformation scale factor (for mapping [a,b] to [-1,1])
    scale = 2.0 / (b - a)

    # Use grid data for computation
    data_g = get_grid_data(operand)
    dims = ndims(data_g)
    data_shape = size(data_g)

    # Check if we're on GPU - DCT requires CPU computation
    use_gpu = is_gpu_array(data_g)
    if use_gpu
        # Copy to CPU for DCT operations (CUFFT doesn't support DCT)
        data_g_cpu = Array(data_g)
    else
        data_g_cpu = data_g
    end

    # Helper: apply chebyshev_derivative_1d `order` times to support higher-order derivatives
    function _cheb_deriv_nth(vec, s, ord)
        d = vec
        for _ in 1:ord
            d = chebyshev_derivative_1d(d, s)
        end
        return d
    end

    if dims == 1
        # 1D case: use DCT directly
        deriv_g_cpu = _cheb_deriv_nth(data_g_cpu, scale, order)
        if use_gpu
            get_grid_data(result) .= copy_to_device(deriv_g_cpu, get_grid_data(result))
        else
            get_grid_data(result) .= deriv_g_cpu
        end
        result.current_layout = :g

    elseif dims == 2
        # 2D case: apply derivative along specified axis only
        deriv_g_cpu = zeros(eltype(data_g_cpu), data_shape)

        if axis == 1
            # Derivative along first axis: process each column
            for j in 1:data_shape[2]
                col = data_g_cpu[:, j]
                deriv_g_cpu[:, j] .= _cheb_deriv_nth(col, scale, order)
            end
        else  # axis == 2
            # Derivative along second axis: process each row
            # Get scale factor for axis 2
            basis2 = operand.bases[2]
            a2, b2 = basis2.meta.bounds
            scale2 = 2.0 / (b2 - a2)
            for i in 1:data_shape[1]
                row = data_g_cpu[i, :]
                deriv_g_cpu[i, :] .= _cheb_deriv_nth(row, scale2, order)
            end
        end

        if use_gpu
            get_grid_data(result) .= copy_to_device(deriv_g_cpu, get_grid_data(result))
        else
            get_grid_data(result) .= deriv_g_cpu
        end
        result.current_layout = :g

    elseif dims == 3
        # 3D case
        deriv_g_cpu = zeros(eltype(data_g_cpu), data_shape)

        if axis == 1
            for j in 1:data_shape[2], k in 1:data_shape[3]
                col = data_g_cpu[:, j, k]
                deriv_g_cpu[:, j, k] .= _cheb_deriv_nth(col, scale, order)
            end
        elseif axis == 2
            basis2 = operand.bases[2]
            a2, b2 = basis2.meta.bounds
            scale2 = 2.0 / (b2 - a2)
            for i in 1:data_shape[1], k in 1:data_shape[3]
                slice = data_g_cpu[i, :, k]
                deriv_g_cpu[i, :, k] .= _cheb_deriv_nth(slice, scale2, order)
            end
        else  # axis == 3
            basis3 = operand.bases[3]
            a3, b3 = basis3.meta.bounds
            scale3 = 2.0 / (b3 - a3)
            for i in 1:data_shape[1], j in 1:data_shape[2]
                slice = data_g_cpu[i, j, :]
                deriv_g_cpu[i, j, :] .= _cheb_deriv_nth(slice, scale3, order)
            end
        end

        if use_gpu
            get_grid_data(result) .= copy_to_device(deriv_g_cpu, get_grid_data(result))
        else
            get_grid_data(result) .= deriv_g_cpu
        end
        result.current_layout = :g
    else
        throw(ArgumentError("Chebyshev derivative only implemented for 1D, 2D, and 3D"))
    end

    # If coefficient space is requested, transform result
    if layout == :c
        # Apply forward DCT to transform grid values to Chebyshev coefficients
        if use_gpu
            result_data_cpu = Array(get_grid_data(result))
        else
            result_data_cpu = get_grid_data(result)
        end

        if dims == 1
            N_result = size(result_data_cpu, 1)
            coeffs = FFTW.r2r(result_data_cpu, FFTW.REDFT00)
            coeffs ./= (N_result - 1)
            coeffs[1] /= 2
            coeffs[end] /= 2
            if use_gpu
                get_coeff_data(result) .= copy_to_device(coeffs, get_coeff_data(result))
            else
                get_coeff_data(result) .= coeffs
            end
        elseif dims == 2
            coeffs = copy(result_data_cpu)
            data_shape_coeff = size(coeffs)

            if operand.bases[1] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N1 = data_shape_coeff[1]
                for j in 1:data_shape_coeff[2]
                    col = coeffs[:, j]
                    col_dct = FFTW.r2r(col, FFTW.REDFT00)
                    col_dct ./= (N1 - 1)
                    col_dct[1] /= 2
                    col_dct[end] /= 2
                    coeffs[:, j] .= col_dct
                end
            end

            if operand.bases[2] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N2 = data_shape_coeff[2]
                for i in 1:data_shape_coeff[1]
                    row = coeffs[i, :]
                    row_dct = FFTW.r2r(row, FFTW.REDFT00)
                    row_dct ./= (N2 - 1)
                    row_dct[1] /= 2
                    row_dct[end] /= 2
                    coeffs[i, :] .= row_dct
                end
            end

            if use_gpu
                get_coeff_data(result) .= copy_to_device(coeffs, get_coeff_data(result))
            else
                get_coeff_data(result) .= coeffs
            end
        elseif dims == 3
            coeffs = copy(result_data_cpu)
            data_shape_coeff = size(coeffs)

            if operand.bases[1] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N1 = data_shape_coeff[1]
                for j in 1:data_shape_coeff[2], k in 1:data_shape_coeff[3]
                    col = coeffs[:, j, k]
                    col_dct = FFTW.r2r(col, FFTW.REDFT00)
                    col_dct ./= (N1 - 1)
                    col_dct[1] /= 2
                    col_dct[end] /= 2
                    coeffs[:, j, k] .= col_dct
                end
            end

            if operand.bases[2] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N2 = data_shape_coeff[2]
                for i in 1:data_shape_coeff[1], k in 1:data_shape_coeff[3]
                    slice = coeffs[i, :, k]
                    slice_dct = FFTW.r2r(slice, FFTW.REDFT00)
                    slice_dct ./= (N2 - 1)
                    slice_dct[1] /= 2
                    slice_dct[end] /= 2
                    coeffs[i, :, k] .= slice_dct
                end
            end

            if operand.bases[3] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
                N3 = data_shape_coeff[3]
                for i in 1:data_shape_coeff[1], j in 1:data_shape_coeff[2]
                    slice = coeffs[i, j, :]
                    slice_dct = FFTW.r2r(slice, FFTW.REDFT00)
                    slice_dct ./= (N3 - 1)
                    slice_dct[1] /= 2
                    slice_dct[end] /= 2
                    coeffs[i, j, :] .= slice_dct
                end
            end

            if use_gpu
                get_coeff_data(result) .= copy_to_device(coeffs, get_coeff_data(result))
            else
                get_coeff_data(result) .= coeffs
            end
        end
        result.current_layout = :c
    end

    # NOTE: Higher-order derivatives are already handled by _cheb_deriv_nth
    # which loops `order` times internally. No additional recursion needed here.
end

"""
    chebyshev_derivative_1d(f, scale)

Compute the Chebyshev spectral derivative of a 1D array using DCT-I.

Arguments:
- f: Function values at Chebyshev-Gauss-Lobatto points
- scale: Domain scaling factor (2/(b-a) for domain [a,b])

Returns the derivative at the same grid points.

Note: The native grid for Chebyshev uses x_k = -cos(pi*k/(N-1)) which gives points
in ascending order (from -1 to +1). However, the standard DCT-I assumes points
cos(pi*k/(N-1)) in descending order (+1 to -1). To handle this, we reverse the
data before the DCT-I transform and reverse the result back.
"""
function chebyshev_derivative_1d(f::AbstractVector, scale::Float64)
    N = length(f)

    # A single point has zero derivative
    if N <= 1
        return zeros(eltype(f), N)
    end

    # Tarang uses ascending grid: x_k = -cos(pi*k/(N-1)) for k = 0, 1, ..., N-1
    # Standard DCT-I convention uses descending grid: x_k = cos(pi*k/(N-1))
    # To use DCT-I correctly with our ascending grid, we need to reverse f

    f_std = reverse(f)

    # Forward DCT-I to get Chebyshev coefficients
    coeffs_raw = FFTW.r2r(f_std, FFTW.REDFT00)

    # Normalize: DCT-I on N points needs (N-1) normalization
    coeffs = copy(coeffs_raw)
    coeffs ./= (N - 1)
    coeffs[1] /= 2
    coeffs[end] /= 2

    # Apply Chebyshev derivative recurrence: c'_{k-1} = 2k * c_k + c'_{k+1}
    deriv_coeffs = zeros(eltype(coeffs), N)
    deriv_coeffs[N] = 0.0

    for k in (N-1):-1:1
        if k + 2 <= N
            deriv_coeffs[k] = 2 * k * coeffs[k + 1] + deriv_coeffs[k + 2]
        else
            deriv_coeffs[k] = 2 * k * coeffs[k + 1]
        end
    end

    # First coefficient has factor of 1/2 due to Chebyshev series normalization
    deriv_coeffs[1] /= 2

    # Apply domain scaling
    deriv_coeffs .*= scale

    # Un-normalize for inverse DCT-I
    deriv_coeffs[1] *= 2
    deriv_coeffs[end] *= 2

    # Inverse DCT-I and normalize to get derivative at standard (descending) grid
    deriv_std = FFTW.r2r(deriv_coeffs, FFTW.REDFT00) ./ 2

    # Convert derivative back to our ascending grid
    return reverse(deriv_std)
end

# ============================================================================
# Legendre Derivative Implementation
# ============================================================================

"""Evaluate Legendre derivative using compatible Jacobi implementation."""
function evaluate_legendre_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    if order < 0
        throw(ArgumentError("Legendre derivative order must be non-negative, got $order"))
    end

    ensure_layout!(operand, :c)
    ensure_layout!(result, :c)

    basis = operand.bases[axis]
    N = basis.meta.size
    a, b = basis.meta.bounds

    if b <= a
        throw(ArgumentError("Legendre basis bounds must satisfy a < b, got ($a, $b)"))
    end

    # Domain transformation scale factor
    scale = 2.0 / (b - a)

    # Check if we're on GPU
    use_gpu = is_gpu_array(get_coeff_data(operand))

    if order == 1
        evaluate_legendre_single_derivative!(result, operand, N, scale, use_gpu)
    else
        temp_field = ScalarField(operand.dist, "temp_deriv", operand.bases, operand.dtype)
        current_operand = operand

        for i in 1:order
            if i == order
                evaluate_legendre_single_derivative!(result, current_operand, N, scale, use_gpu)
            else
                evaluate_legendre_single_derivative!(temp_field, current_operand, N, scale, use_gpu)
                current_operand = temp_field
            end
        end
    end

    if layout == :g
        backward_transform!(result)
    end
end

"""
    evaluate_legendre_single_derivative!(result, operand, N, scale, use_gpu)

Single Legendre derivative using Jacobi approach.
Supports both CPU and GPU arrays (GPU arrays are processed on CPU).

Legendre polynomials are Jacobi polynomials with a=0, b=0.
The standard Legendre derivative recurrence relation is:
P'_n = (2n-1)*P_{n-1} + (2n-5)*P_{n-3} + (2n-9)*P_{n-5} + ...
"""
function evaluate_legendre_single_derivative!(result::ScalarField, operand::ScalarField, N::Int, scale::Float64, use_gpu::Bool=false)
    if use_gpu
        operand_data_cpu = Array(get_coeff_data(operand))
        result_data_cpu = zeros(eltype(operand_data_cpu), size(get_coeff_data(result)))
    else
        # Defensive copy when operand and result alias (happens for order >= 3)
        operand_data_cpu = operand === result ? copy(get_coeff_data(operand)) : get_coeff_data(operand)
        result_data_cpu = get_coeff_data(result)
        fill!(result_data_cpu, 0.0)
    end

    # Legendre spectral derivative formula:
    # c'[k] = (2k-1) * sum_{j: j>k, j-k odd} c[j]

    @inbounds for k in 1:min(N, length(result_data_cpu))
        coeff_sum = 0.0
        for j in (k+1):min(N, length(operand_data_cpu))
            if (j - k) % 2 == 1
                coeff_sum += operand_data_cpu[j]
            end
        end
        result_data_cpu[k] = (2.0 * k - 1.0) * coeff_sum * scale
    end

    if use_gpu
        get_coeff_data(result) .= copy_to_device(result_data_cpu, get_coeff_data(result))
    end
end

# ============================================================================
# Matrix Application Helpers
# ============================================================================

"""
    apply_matrix_along_axis(matrix, array, axis; out=nothing)

Apply matrix along any axis of an array.
Following array:77-82 and apply_dense:104-126 implementation.
"""
function apply_matrix_along_axis(matrix::AbstractMatrix, array::AbstractArray, axis::Int; out=nothing)
    if issparse(matrix)
        return apply_sparse_along_axis(matrix, array, axis; out=out)
    else
        return apply_dense_along_axis(matrix, array, axis; out=out)
    end
end

"""
    apply_dense_along_axis(matrix, array, axis; out=nothing)

Apply dense matrix along any axis of an array.
Following apply_dense implementation in array:104-126.
"""
function apply_dense_along_axis(matrix::AbstractMatrix, array::AbstractArray, axis::Int; out=nothing)
    ndim = ndims(array)
    use_gpu = is_gpu_array(array)
    arch = architecture(array)
    array_cpu = use_gpu ? Array(array) : array

    # Resolve wraparound axis
    axis = mod1(axis, ndim)

    out_is_gpu = out !== nothing && is_gpu_array(out)

    if out === nothing
        out_shape = collect(size(array_cpu))
        out_shape[axis] = size(matrix, 1)
        out_cpu = zeros(eltype(array_cpu), out_shape...)
    elseif out === array
        throw(ArgumentError("Cannot apply in place"))
    else
        out_cpu = out_is_gpu ? Array(out) : out
    end

    # Move target axis to position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        array_cpu = permutedims(array_cpu, perm)
    end

    array_shape = size(array_cpu)

    # Flatten later axes for matrix multiplication
    if ndim > 2
        array_cpu = reshape(array_cpu, (array_shape[1], prod(array_shape[2:end])))
    end

    # Apply matrix multiplication
    temp = matrix * array_cpu

    # Unflatten later axes
    if ndim > 2
        new_shape = (size(temp, 1), array_shape[2:end]...)
        temp = reshape(temp, new_shape)
    end

    # Move axis back
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        temp = permutedims(temp, perm)
    end

    copyto!(out_cpu, temp)

    if use_gpu || out_is_gpu
        if out === nothing
            return on_architecture(arch, out_cpu)
        else
            if out_is_gpu
                out .= copy_to_device(out_cpu, out)
            else
                copyto!(out, out_cpu)
            end
            return out
        end
    else
        return out_cpu
    end
end

"""
    apply_sparse_along_axis(matrix, array, axis; out=nothing, check_shapes=false)

Apply sparse matrix along any axis of an array.
Supports both CPU and GPU arrays (GPU arrays are copied to CPU for sparse operations).
Following apply_sparse implementation in array:171-203.
Note: Uses SparseMatrixCSC (Julia's sparse format) instead of CSR.
"""
function apply_sparse_along_axis(matrix::SparseMatrixCSC, array::AbstractArray, axis::Int; out=nothing, check_shapes=false)
    ndim = ndims(array)

    use_gpu = is_gpu_array(array)
    if use_gpu
        array_cpu = Array(array)
    else
        array_cpu = array
    end

    axis = mod1(axis, ndim)

    if out === nothing
        out_shape = collect(size(array_cpu))
        out_shape[axis] = size(matrix, 1)
        out_cpu = zeros(eltype(array_cpu), out_shape...)
    elseif out === array
        throw(ArgumentError("Cannot apply in place"))
    else
        out_cpu = use_gpu ? Array(out) : out
    end

    if check_shapes
        if !(1 <= axis <= ndim)
            throw(BoundsError("Axis out of bounds"))
        end
        if size(matrix, 2) != size(array_cpu, axis) || size(matrix, 1) != size(out_cpu, axis)
            throw(DimensionMismatch("Matrix shape mismatch"))
        end
    end

    # Move target axis to position 1
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        array_cpu = permutedims(array_cpu, perm)
    end

    array_shape = size(array_cpu)

    # Flatten later axes
    if ndim > 2
        array_cpu = reshape(array_cpu, (array_shape[1], prod(array_shape[2:end])))
    end

    # Apply sparse matrix multiplication
    temp = matrix * array_cpu

    # Unflatten later axes
    if ndim > 2
        new_shape = (size(temp, 1), array_shape[2:end]...)
        temp = reshape(temp, new_shape)
    end

    # Move axis back
    if axis != 1
        perm = collect(1:ndim)
        perm[1] = axis
        perm[axis] = 1
        temp = permutedims(temp, perm)
    end

    copyto!(out_cpu, temp)

    if use_gpu
        if out === nothing
            return copy_to_device(out_cpu, array)
        else
            out .= copy_to_device(out_cpu, out)
            return out
        end
    else
        return out_cpu
    end
end
