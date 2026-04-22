# Polynomial-basis derivative implementations for Chebyshev and Legendre bases.

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
    if N <= 1
        return zeros(eltype(f), N)
    end
    result = similar(f)
    chebyshev_derivative_1d!(result, f, scale)
    return result
end

"""
    chebyshev_derivative_1d!(result, f, scale)

In-place Chebyshev derivative. Writes result into `result`.
"""
function chebyshev_derivative_1d!(result::AbstractVector, f::AbstractVector, scale::Float64)
    N = length(f)
    if N <= 1
        fill!(result, zero(eltype(f)))
        return result
    end

    # Reverse f into result as workspace (ascending → descending grid)
    @inbounds for i in 1:N
        result[i] = f[N - i + 1]
    end

    # Forward DCT-I to get Chebyshev coefficients (allocates, but only once per call)
    coeffs = FFTW.r2r(result, FFTW.REDFT00)

    # Normalize: DCT-I on N points needs (N-1) normalization
    inv_nm1 = 1.0 / (N - 1)
    @inbounds for i in 1:N
        coeffs[i] *= inv_nm1
    end
    coeffs[1] /= 2
    coeffs[end] /= 2

    # Apply Chebyshev derivative recurrence in-place into result:
    # c'_{k-1} = 2k * c_k + c'_{k+1}
    fill!(result, zero(eltype(f)))
    @inbounds for k in (N-1):-1:1
        result[k] = 2 * k * coeffs[k + 1]
        if k + 2 <= N
            result[k] += result[k + 2]
        end
    end

    result[1] /= 2
    @. result *= scale

    # Un-normalize for inverse DCT-I
    result[1] *= 2
    result[end] *= 2

    # Inverse DCT-I to get derivative at descending grid
    deriv_std = FFTW.r2r(result, FFTW.REDFT00)

    # Reverse back to ascending grid and normalize, writing into result
    @inbounds for i in 1:N
        result[i] = deriv_std[N - i + 1] / 2
    end

    return result
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
