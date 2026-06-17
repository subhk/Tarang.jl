# Polynomial-basis derivative implementations for Chebyshev and Legendre bases.

# ============================================================================
# DCT-I plan cache — zero-allocation Chebyshev transforms
#
# Keyed by (N::Int, T::DataType) → (plan, scratch::Vector{T}).
# `plan` is an FFTW in-place REDFT00 plan created once per (N,T) pair.
# `scratch` is the contiguous buffer the plan transforms in-place; callers
# copyto! their input into scratch, apply `plan * scratch`, then read scratch.
# Single-threaded assumption: scratch is reused across calls at the same (N,T).
# ============================================================================
const _CHEB_DERIV_PLANS = Dict{Tuple{Int, DataType}, Tuple{Any, Vector}}()

function _get_cheb_deriv_plan(N::Int, ::Type{T}) where {T<:AbstractFloat}
    key = (N, T)
    entry = get(_CHEB_DERIV_PLANS, key, nothing)
    entry !== nothing && return entry::Tuple{Any, Vector}
    v = Vector{T}(undef, N)
    plan = FFTW.plan_r2r!(v, FFTW.REDFT00)
    scratch = Vector{T}(undef, N)
    e = (plan, scratch)
    _CHEB_DERIV_PLANS[key] = e
    return e
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
# Stub overridden by TarangCUDAExt when CUDA is loaded.
# Returns true if GPU handled the derivative, false for CPU fallback.
function _gpu_chebyshev_deriv!(result::ScalarField, operand::ScalarField,
                                data_g::AbstractArray, axis::Int, order::Int, scale::Float64)
    return false
end

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
Uses cached FFTW plans and @view slices to avoid per-call allocations.
"""
function _evaluate_local_chebyshev_derivative!(result::ScalarField, operand::ScalarField, axis::Int, order::Int, layout::Symbol)
    basis = operand.bases[axis]
    N = basis.meta.size
    a, b = basis.meta.bounds

    if b <= a
        throw(ArgumentError("Chebyshev basis bounds must satisfy a < b, got ($a, $b)"))
    end

    scale = 2.0 / (b - a)

    data_g = get_grid_data(operand)
    dims = ndims(data_g)
    data_shape = size(data_g)

    use_gpu = is_gpu_array(data_g)

    if use_gpu
        # Try GPU-native DCT-I derivative (overridden by TarangCUDAExt for CuArray)
        if _gpu_chebyshev_deriv!(result, operand, data_g, axis, order, scale)
            result.current_layout = :g
            if layout == :c
                result_g_cpu = Array(get_grid_data(result))
                _cheb_coeff_convert!(result_g_cpu, operand, dims, eltype(result_g_cpu))
                get_coeff_data(result) .= copy_to_device(result_g_cpu, get_coeff_data(result))
                result.current_layout = :c
            end
            return
        end
    end

    data_g_cpu = use_gpu ? Array(data_g) : data_g

    eltype_data = eltype(data_g_cpu)

    # For CPU path, write derivative result directly into the result field's
    # pre-allocated grid buffer — avoids the intermediate zeros() allocation
    # and the final copy. For GPU, compute on CPU then transfer.
    if use_gpu
        deriv_g_cpu = zeros(eltype_data, data_shape)
    else
        deriv_g_cpu = get_grid_data(result)
        if deriv_g_cpu === nothing || size(deriv_g_cpu) != data_shape || eltype(deriv_g_cpu) != eltype_data
            deriv_g_cpu = zeros(eltype_data, data_shape)
            set_grid_data!(result, deriv_g_cpu)
        end
    end

    # Pre-allocate one temp vector for higher-order (≥2) derivatives.
    # For order=1 this is allocated but never used; that's one small alloc
    # per evaluate call, acceptable to keep the inner helpers uniform.
    tmp_buf = Vector{eltype_data}(undef, data_shape[axis])

    if dims == 1
        _cheb_deriv_nth_inplace!(deriv_g_cpu, data_g_cpu, scale, order, tmp_buf)

    elseif dims == 2
        if axis == 1
            for j in 1:data_shape[2]
                _cheb_deriv_nth_inplace!(@view(deriv_g_cpu[:, j]),
                                         @view(data_g_cpu[:, j]),
                                         scale, order, tmp_buf)
            end
        else  # axis == 2
            basis2 = operand.bases[2]
            a2, b2 = basis2.meta.bounds
            scale2 = 2.0 / (b2 - a2)
            for i in 1:data_shape[1]
                _cheb_deriv_nth_inplace!(@view(deriv_g_cpu[i, :]),
                                         @view(data_g_cpu[i, :]),
                                         scale2, order, tmp_buf)
            end
        end

    elseif dims == 3
        if axis == 1
            for j in 1:data_shape[2], k in 1:data_shape[3]
                _cheb_deriv_nth_inplace!(@view(deriv_g_cpu[:, j, k]),
                                         @view(data_g_cpu[:, j, k]),
                                         scale, order, tmp_buf)
            end
        elseif axis == 2
            basis2 = operand.bases[2]
            a2, b2 = basis2.meta.bounds
            scale2 = 2.0 / (b2 - a2)
            for i in 1:data_shape[1], k in 1:data_shape[3]
                _cheb_deriv_nth_inplace!(@view(deriv_g_cpu[i, :, k]),
                                         @view(data_g_cpu[i, :, k]),
                                         scale2, order, tmp_buf)
            end
        else  # axis == 3
            basis3 = operand.bases[3]
            a3, b3 = basis3.meta.bounds
            scale3 = 2.0 / (b3 - a3)
            for i in 1:data_shape[1], j in 1:data_shape[2]
                _cheb_deriv_nth_inplace!(@view(deriv_g_cpu[i, j, :]),
                                         @view(data_g_cpu[i, j, :]),
                                         scale3, order, tmp_buf)
            end
        end
    else
        throw(ArgumentError("Chebyshev derivative only implemented for 1D, 2D, and 3D"))
    end

    if use_gpu
        get_grid_data(result) .= copy_to_device(deriv_g_cpu, get_grid_data(result))
    end
    result.current_layout = :g

    # If coefficient space is requested, apply forward DCT in-place
    if layout == :c
        if use_gpu
            result_data_cpu = Array(get_grid_data(result))
        else
            result_data_cpu = get_grid_data(result)
        end

        _cheb_coeff_convert!(result_data_cpu, operand, dims, eltype_data)

        if use_gpu
            get_coeff_data(result) .= copy_to_device(result_data_cpu, get_coeff_data(result))
        else
            get_coeff_data(result) .= result_data_cpu
        end
        result.current_layout = :c
    end
end

"""
    _cheb_coeff_convert!(data, operand, dims, eltype_data)

Convert grid-space Chebyshev data to coefficient space in-place using cached
DCT-I plans. Operates on `data` directly — no copy() of the input.
"""
# DCT-I of the (descending) Chebyshev grid yields the coefficients of g(y)=f(-y).
# Undo the T_n(-x)=(-1)^n alternation (negate odd modes = even 1-based indices) so
# the result matches the basis' native forward_transform! sign convention, and
# apply the DCT-I normalization (1/(N-1), endpoints halved).
@inline function _cheb_dct_to_coeffs!(scratch::AbstractVector, N::Int)
    inv_n = 1.0 / (N - 1)
    @inbounds for m in 1:N
        scratch[m] *= inv_n * (isodd(m) ? 1.0 : -1.0)
    end
    scratch[1] /= 2
    scratch[N] /= 2
    return scratch
end

function _cheb_coeff_convert!(data::AbstractArray, operand::ScalarField, dims::Int, ::Type{T}) where {T<:AbstractFloat}
    data_shape = size(data)

    if dims == 1
        N1 = data_shape[1]
        plan, scratch = _get_cheb_deriv_plan(N1, T)
        copyto!(scratch, data)
        plan * scratch
        _cheb_dct_to_coeffs!(scratch, N1)
        data .= scratch

    elseif dims == 2
        if operand.bases[1] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
            N1 = data_shape[1]
            plan, scratch = _get_cheb_deriv_plan(N1, T)
            for j in 1:data_shape[2]
                col = @view data[:, j]
                copyto!(scratch, col)
                plan * scratch
                _cheb_dct_to_coeffs!(scratch, N1)
                col .= scratch
            end
        end
        if operand.bases[2] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
            N2 = data_shape[2]
            plan, scratch = _get_cheb_deriv_plan(N2, T)
            for i in 1:data_shape[1]
                row = @view data[i, :]
                copyto!(scratch, row)
                plan * scratch
                _cheb_dct_to_coeffs!(scratch, N2)
                row .= scratch
            end
        end

    elseif dims == 3
        if operand.bases[1] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
            N1 = data_shape[1]
            plan, scratch = _get_cheb_deriv_plan(N1, T)
            for j in 1:data_shape[2], k in 1:data_shape[3]
                col = @view data[:, j, k]
                copyto!(scratch, col)
                plan * scratch
                _cheb_dct_to_coeffs!(scratch, N1)
                col .= scratch
            end
        end
        if operand.bases[2] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
            N2 = data_shape[2]
            plan, scratch = _get_cheb_deriv_plan(N2, T)
            for i in 1:data_shape[1], k in 1:data_shape[3]
                sl = @view data[i, :, k]
                copyto!(scratch, sl)
                plan * scratch
                _cheb_dct_to_coeffs!(scratch, N2)
                sl .= scratch
            end
        end
        if operand.bases[3] isa Union{ChebyshevT, ChebyshevU, ChebyshevV}
            N3 = data_shape[3]
            plan, scratch = _get_cheb_deriv_plan(N3, T)
            for i in 1:data_shape[1], j in 1:data_shape[2]
                sl = @view data[i, j, :]
                copyto!(scratch, sl)
                plan * scratch
                _cheb_dct_to_coeffs!(scratch, N3)
                sl .= scratch
            end
        end
    end
end

"""
    _cheb_deriv_nth_inplace!(out, inp, scale, order, tmp)

Apply Chebyshev derivative `order` times, writing into `out`.
`tmp` is a pre-allocated buffer used as intermediate for order ≥ 2.
Zero-allocation on warm FFTW plan cache.
"""
function _cheb_deriv_nth_inplace!(out::AbstractVector, inp::AbstractVector,
                                   scale::Float64, order::Int, tmp::AbstractVector)
    chebyshev_derivative_1d!(out, inp, scale)
    for _ in 2:order
        copyto!(tmp, out)
        chebyshev_derivative_1d!(out, tmp, scale)
    end
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
Uses cached FFTW in-place DCT-I plans — zero-allocation after first call
for each (N, eltype) pair.
"""
function chebyshev_derivative_1d!(result::AbstractVector{T}, f::AbstractVector{T}, scale::Float64) where {T<:AbstractFloat}
    N = length(f)
    if N <= 1
        fill!(result, zero(T))
        return result
    end

    plan, scratch = _get_cheb_deriv_plan(N, T)

    # Reverse f into result (ascending → descending Chebyshev grid)
    @inbounds for i in 1:N
        result[i] = f[N - i + 1]
    end

    # Forward DCT-I via cached in-place plan (zero-allocation)
    copyto!(scratch, result)
    plan * scratch  # scratch ← DCT-I(reversed f) = Chebyshev coefficients

    # Normalize: DCT-I on N points needs 1/(N-1) factor
    inv_nm1 = 1.0 / (N - 1)
    @inbounds for i in 1:N
        scratch[i] *= inv_nm1
    end
    scratch[1] /= 2
    scratch[end] /= 2

    # Derivative recurrence: c'_{k-1} = 2k c_k + c'_{k+1}, write into result
    fill!(result, zero(T))
    @inbounds for k in (N-1):-1:1
        result[k] = 2 * k * scratch[k + 1]
        if k + 2 <= N
            result[k] += result[k + 2]
        end
    end
    result[1] /= 2
    @. result *= scale

    # Un-normalize endpoints for inverse DCT-I
    result[1] *= 2
    result[end] *= 2

    # Inverse DCT-I via same cached plan (zero-allocation)
    copyto!(scratch, result)
    plan * scratch  # scratch ← DCT-I(derivative coeffs) = derivative at descending grid

    # Reverse back to ascending grid and normalize
    @inbounds for i in 1:N
        result[i] = scratch[N - i + 1] / 2
    end

    return result
end

# AbstractVector fallback for non-Float inputs (e.g. views of mixed type)
function chebyshev_derivative_1d!(result::AbstractVector, f::AbstractVector, scale::Float64)
    T = promote_type(eltype(result), eltype(f), Float64)
    chebyshev_derivative_1d!(convert(Vector{T}, result), convert(Vector{T}, f), scale)
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
        evaluate_legendre_single_derivative!(result, operand, axis, N, scale, use_gpu)
    else
        temp_field = ScalarField(operand.dist, "temp_deriv", operand.bases, operand.dtype)
        current_operand = operand

        for i in 1:order
            if i == order
                evaluate_legendre_single_derivative!(result, current_operand, axis, N, scale, use_gpu)
            else
                evaluate_legendre_single_derivative!(temp_field, current_operand, axis, N, scale, use_gpu)
                current_operand = temp_field
            end
        end
    end

    if layout == :g
        backward_transform!(result)
    end
end

"""
    evaluate_legendre_single_derivative!(result, operand, axis, N, scale, use_gpu)

Single Legendre derivative using Jacobi approach, applied along `axis`.
Supports both CPU and GPU arrays (GPU arrays are processed on CPU).

Legendre polynomials are Jacobi polynomials with a=0, b=0.
The standard Legendre derivative recurrence relation is:
P'_n = (2n-1)*P_{n-1} + (2n-5)*P_{n-3} + (2n-9)*P_{n-5} + ...
"""
function evaluate_legendre_single_derivative!(result::ScalarField, operand::ScalarField, axis::Int, N::Int, scale::Float64, use_gpu::Bool=false)
    if use_gpu
        operand_data_cpu = Array(get_coeff_data(operand))
        result_data_cpu = zeros(eltype(operand_data_cpu), size(get_coeff_data(result)))
    else
        # Defensive copy when operand and result alias (happens for order >= 3)
        operand_data_cpu = operand === result ? copy(get_coeff_data(operand)) : get_coeff_data(operand)
        result_data_cpu = get_coeff_data(result)
        fill!(result_data_cpu, 0.0)
    end

    # Legendre spectral derivative. The classic recurrence
    #     c'[k] = (2k-1) * sum_{j>k, j-k odd} c[j]
    # is for UNNORMALIZED P_n. This basis (and its transforms) use ORTHONORMAL
    # P̃_n = γ_n P_n with γ_n = sqrt((2n+1)/2) (see setup_legendre_transform!),
    # so coefficients must be de-normalized into the P_n basis, differentiated,
    # then re-normalized: c̃'[k] = (1/γ_{k-1}) (2k-1) Σ γ_{j-1} c̃[j].
    # Here the 1-based index i corresponds to mode m=i-1, so γ(i)=sqrt((2i-1)/2).
    γ(i) = sqrt((2.0 * i - 1.0) / 2.0)

    # The recurrence is 1D in the Legendre coefficient axis. For a multi-dimensional
    # field the coefficient array is N-D, so apply it independently to every 1D fiber
    # along `axis`. Linear indexing (the old `[k]`) would walk column-major across the
    # flattened array, mixing unrelated modes from the other axes — wrong for any field
    # with more than one dimension.
    n_axis = min(N, size(operand_data_cpu, axis), size(result_data_cpu, axis))
    fiber_shape = ntuple(d -> d == axis ? 1 : size(operand_data_cpu, d), ndims(operand_data_cpu))
    @inbounds for base in CartesianIndices(fiber_shape)
        for k in 1:n_axis
            coeff_sum = zero(eltype(operand_data_cpu))
            for j in (k+1):n_axis
                if (j - k) % 2 == 1
                    coeff_sum += γ(j) * operand_data_cpu[_legendre_fiber_index(base, axis, j)]
                end
            end
            result_data_cpu[_legendre_fiber_index(base, axis, k)] =
                (2.0 * k - 1.0) * coeff_sum * scale / γ(k)
        end
    end

    if use_gpu
        get_coeff_data(result) .= copy_to_device(result_data_cpu, get_coeff_data(result))
    end
end

# Replace the `axis`-th component of a CartesianIndex with `val`, keeping the rest —
# lets the Legendre recurrence walk a single 1D fiber of an N-D coefficient array.
@inline function _legendre_fiber_index(base::CartesianIndex{D}, axis::Int, val::Int) where {D}
    return CartesianIndex(ntuple(d -> d == axis ? val : base[d], D))
end

"""
    _nodal_diff_matrix(x) -> Matrix

Barycentric nodal differentiation matrix `D` for distinct nodes `x`: `(D*f)[i]`
is the derivative at `x[i]` of the degree-`<N` polynomial interpolating `f` on
`x`. Exact for polynomials of degree `< length(x)`, independent of the basis —
so it differentiates a field on ANY Jacobi-family collocation grid (ChebyshevU,
ChebyshevV, Ultraspherical, generic Jacobi) whose `evaluate_basis` spans that
polynomial space. Built on PHYSICAL nodes, so the domain scaling is already
included (no extra 2/(b-a) factor).
"""
function _nodal_diff_matrix(x::AbstractVector{<:Real})
    N = length(x)
    w = ones(Float64, N)                      # barycentric weights
    @inbounds for i in 1:N, j in 1:N
        i != j && (w[i] /= (x[i] - x[j]))
    end
    D = zeros(Float64, N, N)
    @inbounds for i in 1:N
        for j in 1:N
            i != j && (D[i, j] = (w[j] / w[i]) / (x[i] - x[j]))
        end
        D[i, i] = -sum(@view D[i, :])         # negative-sum-trick for the diagonal
    end
    return D
end

"""
    evaluate_jacobi_collocation_derivative!(result, operand, axis, order, layout)

Differentiate `operand` along `axis` for Jacobi-family bases that have no
dedicated spectral derivative kernel (ChebyshevU, ChebyshevV, Ultraspherical,
generic Jacobi). Applies the nodal differentiation matrix in grid space
(`order`-th power for higher orders), which is exact for the degree-`<N`
polynomial these bases represent. Result is left in `:g`; a `:c` request uses the
basis' collocation forward transform.
"""
function evaluate_jacobi_collocation_derivative!(result::ScalarField, operand::ScalarField,
                                                 axis::Int, order::Int, layout::Symbol)
    ensure_layout!(operand, :g)
    basis = operand.bases[axis]
    nodes = vec(Array(local_grid(basis, operand.dist, 1)))
    Dbase = _nodal_diff_matrix(nodes)
    D = order == 1 ? Dbase : Dbase^order

    data_g = get_grid_data(operand)
    gpu = is_gpu_array(data_g)
    data_cpu = gpu ? Array(data_g) : data_g
    deriv = apply_matrix_along_axis(D, data_cpu, axis)

    ensure_layout!(result, :g)
    set_grid_data!(result, gpu ? copy_to_device(deriv, get_grid_data(result)) : deriv)
    result.current_layout = :g

    if layout == :c
        forward_transform!(result)
    end
    return result
end
