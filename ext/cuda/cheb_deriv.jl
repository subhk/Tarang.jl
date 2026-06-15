# ============================================================================
# GPU Chebyshev Derivative via DCT-I using cuFFT (no CPU transfer)
# ============================================================================
#
# DCT-I via rfft (symmetric extension trick):
#   For N-point input x, build extension z of length M = 2(N-1):
#     z[k] = x[k]       for k = 1..N
#     z[N+k] = x[N-k]   for k = 1..N-2  (interior mirror)
#   Then real(rfft(z))[k] == DCT-I(x)[k]  (imaginary parts cancel by symmetry)
#
# This replaces FFTW.plan_r2r!(v, FFTW.REDFT00) with cuFFT rfft, enabling
# the Chebyshev derivative recurrence entirely on GPU.
# ============================================================================

# Plan cache: keyed by (device, n, batch, T). The device MUST be part of the key —
# the plan's CuArray work buffers and CUFFT plan are allocated on whatever device is
# current at build time, so a plan built on device 0 must never be reused on device 1
# (illegal cross-device access). Guarded by a lock for concurrent first-touch inserts.
# NOTE: the cached plan's work_* buffers are shared across calls with the same key, so
# concurrent derivative calls on the SAME (device,n,batch,T) must not overlap — the
# transform layer is single-threaded per field (consistent with the rest of Tarang).
const _GPU_CHEB_DERIV_CACHE = Dict{Any, Any}()
const _GPU_CHEB_DERIV_LOCK = ReentrantLock()

struct GPUChebyshevDerivPlan{T}
    n::Int                          # points along transform dimension
    batch::Int                      # product of all other dimensions
    work_ext::CuMatrix{T}           # (2*(n-1), batch) extension buffer
    work_cx::CuMatrix{Complex{T}}   # (n, batch) rfft output
    work_real::CuMatrix{T}          # (n, batch) real scratch / DCT-I output
    work_deriv::CuMatrix{T}         # (n, batch) derivative coefficients
    rfft_plan::Any                  # CUFFT rfft along dim 1 of work_ext
end

function _get_gpu_cheb_deriv_plan(n::Int, batch::Int, ::Type{T}) where {T<:AbstractFloat}
    key = (CUDA.device(), n, batch, T)
    return lock(_GPU_CHEB_DERIV_LOCK) do
        p = get(_GPU_CHEB_DERIV_CACHE, key, nothing)
        if p === nothing
            M = 2 * (n - 1)
            work_ext   = CUDA.zeros(T,           M, batch)
            work_cx    = CUDA.zeros(Complex{T},  n, batch)
            work_real  = CUDA.zeros(T,           n, batch)
            work_deriv = CUDA.zeros(T,           n, batch)
            rfft_plan  = CUFFT.plan_rfft(work_ext, (1,))
            p = GPUChebyshevDerivPlan{T}(n, batch, work_ext, work_cx, work_real, work_deriv, rfft_plan)
            _GPU_CHEB_DERIV_CACHE[key] = p
        end
        p::GPUChebyshevDerivPlan{T}
    end
end

# ---------------------------------------------------------------------------
# KernelAbstractions kernels (all operate on (n, batch) matrices)
# ---------------------------------------------------------------------------

"""Build DCT-I symmetric extension: (n, batch) → (2*(n-1), batch)."""
@kernel function _dct1_ext_kernel!(work, @Const(inp), n, batch)
    j = @index(Global)
    if j <= batch
        @inbounds begin
            for i in 1:n
                work[i, j] = inp[i, j]
            end
            for k in 1:n-2
                work[n + k, j] = inp[n - k, j]
            end
        end
    end
end

"""Reverse along dim 1: out[i,j] = inp[n+1-i, j]."""
@kernel function _cheb_reverse_kernel!(out, @Const(inp), n, batch)
    j = @index(Global)
    if j <= batch
        @inbounds for i in 1:n
            out[i, j] = inp[n - i + 1, j]
        end
    end
end

"""Extract real part of complex matrix: out[k,j] = real(cx[k,j])."""
@kernel function _extract_real_kernel!(out, @Const(cx), n, batch)
    j = @index(Global)
    if j <= batch
        @inbounds for k in 1:n
            out[k, j] = real(cx[k, j])
        end
    end
end

"""
Chebyshev coefficient → derivative coefficient recurrence.

Reads raw DCT-I output (shape n×batch), writes derivative coefficients.
Mirrors the recurrence in `chebyshev_derivative_1d!` exactly:
  - Normalize by inv_nm1 = 1/(N-1), halve the last endpoint (c_{N-1})
  - Recurrence: c'_{k-1} = 2k*c_k + c'_{k+1}, k = N-1 down to 1
  - Halve c'_0, apply domain scale, un-normalize both endpoints
"""
@kernel function _cheb_coeff_to_deriv_kernel!(deriv, @Const(coeff), n, batch,
                                               inv_nm1::T, scale::T) where {T}
    j = @index(Global)
    if j <= batch
        @inbounds begin
            for k in 1:n
                deriv[k, j] = zero(T)
            end

            for k in n-1:-1:1
                s = coeff[k+1, j] * inv_nm1
                if k + 1 == n
                    s *= T(0.5)  # last endpoint c_{N-1} is halved
                end
                deriv[k, j] = 2 * T(k) * s
                if k + 2 <= n
                    deriv[k, j] += deriv[k+2, j]
                end
            end

            deriv[1, j] /= 2
            for k in 1:n
                deriv[k, j] *= scale
            end
            deriv[1, j] *= 2   # un-normalize for inverse DCT-I
            deriv[n, j] *= 2
        end
    end
end

"""Reverse dim 1 and scale by 1/2: out[i,j] = rfft_real[n+1-i, j] / 2."""
@kernel function _cheb_finalize_kernel!(out, @Const(rfft_real), n, batch)
    j = @index(Global)
    if j <= batch
        @inbounds for i in 1:n
            out[i, j] = rfft_real[n - i + 1, j] / 2
        end
    end
end

# ---------------------------------------------------------------------------
# 1-pass derivative (order = 1) on (n, batch) matrices
# ---------------------------------------------------------------------------

function _apply_gpu_cheb_deriv_1!(inp_mat::CuMatrix{T}, out_mat::CuMatrix{T},
                                   scale::Float64, plan::GPUChebyshevDerivPlan{T}) where {T}
    n     = plan.n
    batch = plan.batch
    arch  = Tarang.architecture(inp_mat)
    inv_nm1 = T(1.0 / (n - 1))
    sc_T    = T(scale)

    # Step 1: reverse input (ascending → descending CGL grid)
    launch!(arch, _cheb_reverse_kernel!, plan.work_real, inp_mat, n, batch; ndrange=batch)

    # Step 2: symmetric extension
    launch!(arch, _dct1_ext_kernel!, plan.work_ext, plan.work_real, n, batch; ndrange=batch)

    # Step 3: batched rfft along dim 1 → DCT-I output
    mul!(plan.work_cx, plan.rfft_plan, plan.work_ext)

    # Step 4: extract real part → Chebyshev coefficients (raw)
    launch!(arch, _extract_real_kernel!, plan.work_real, plan.work_cx, n, batch; ndrange=batch)

    # Step 5: recurrence → derivative coefficients in work_deriv
    launch!(arch, _cheb_coeff_to_deriv_kernel!, plan.work_deriv, plan.work_real,
            n, batch, inv_nm1, sc_T; ndrange=batch)

    # Step 6: symmetric extension of derivative coefficients
    launch!(arch, _dct1_ext_kernel!, plan.work_ext, plan.work_deriv, n, batch; ndrange=batch)

    # Step 7: batched rfft again → DCT-I of derivative coefficients
    mul!(plan.work_cx, plan.rfft_plan, plan.work_ext)

    # Step 8: extract real part
    launch!(arch, _extract_real_kernel!, plan.work_real, plan.work_cx, n, batch; ndrange=batch)

    # Step 9: reverse back and scale by 1/2
    launch!(arch, _cheb_finalize_kernel!, out_mat, plan.work_real, n, batch; ndrange=batch)

    return out_mat
end

function _apply_gpu_cheb_deriv_nth!(inp_mat::CuMatrix{T}, out_mat::CuMatrix{T},
                                     scale::Float64, order::Int,
                                     plan::GPUChebyshevDerivPlan{T}) where {T}
    order == 0 && (copyto!(out_mat, inp_mat); return out_mat)
    _apply_gpu_cheb_deriv_1!(inp_mat, out_mat, scale, plan)
    if order >= 2
        # Allocate a separate temp to avoid aliasing with plan.work_real
        tmp = similar(inp_mat)
        for _ in 2:order
            copyto!(tmp, out_mat)
            _apply_gpu_cheb_deriv_1!(tmp, out_mat, scale, plan)
        end
    end
    return out_mat
end

# ---------------------------------------------------------------------------
# Main dispatch: overrides Tarang._gpu_chebyshev_deriv! for CuArray
# ---------------------------------------------------------------------------

function Tarang._gpu_chebyshev_deriv!(result::Tarang.ScalarField,
                                       operand::Tarang.ScalarField,
                                       data_g::CuArray, axis::Int, order::Int,
                                       scale::Float64)
    n  = size(data_g, axis)
    nd = ndims(data_g)

    if n <= 1
        result_data = Tarang.get_grid_data(result)
        if result_data !== nothing
            fill!(result_data, zero(eltype(result_data)))
        end
        return true
    end

    T = eltype(data_g)

    # Pin the current CUDA device to the operand's device before allocating work buffers
    # / building the plan / launching kernels. The plan cache is keyed by CUDA.device()
    # at call time, so without this a multi-GPU run whose current device != data_g's device
    # would build the plan and scratch on the wrong device and then mix cross-device buffers.
    ensure_device!(Tarang.architecture(data_g))

    # Permute so transform axis is first, then reshape to (n, batch)
    other_dims = ntuple(i -> i < axis ? i : i + 1, nd - 1)
    perm  = (axis, other_dims...)
    iperm = invperm(perm)

    data_perm = permutedims(data_g, perm)
    batch     = prod(size(data_g)) ÷ n
    data_mat  = reshape(data_perm, n, batch)
    out_mat   = similar(data_mat)

    plan = _get_gpu_cheb_deriv_plan(n, batch, T)
    _apply_gpu_cheb_deriv_nth!(data_mat, out_mat, scale, order, plan)

    # Reshape and permute back to original layout
    out_perm = reshape(out_mat, size(data_perm))
    out_g    = permutedims(out_perm, iperm)

    result_data = Tarang.get_grid_data(result)
    if result_data !== nothing && size(result_data) == size(out_g)
        copyto!(result_data, out_g)
    else
        Tarang.set_grid_data!(result, copy(out_g))
    end

    return true
end
