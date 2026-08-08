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
#
# Kernel indexing: every kernel except the derivative recurrence is a pure
# element map, launched with a 2-D ndrange covering the whole matrix — one
# thread per ELEMENT. (They used to be one thread per COLUMN with a serial
# inner loop, which left the device ~99% idle at 2D sizes: a 128×128 mixed
# field has batch = 65.) Only `_cheb_coeff_to_deriv_kernel!` keeps the
# per-column layout, because its recurrence is inherently serial along the
# transform axis.
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
    work_perm::CuMatrix{T}          # (n, batch) permuted in/out staging (axis != 1)
    work_tmp::CuMatrix{T}           # (n, batch) ping buffer for order >= 2
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
            work_perm  = CUDA.zeros(T,           n, batch)
            work_tmp   = CUDA.zeros(T,           n, batch)
            rfft_plan  = CUFFT.plan_rfft(work_ext, (1,))
            p = GPUChebyshevDerivPlan{T}(n, batch, work_ext, work_cx, work_real,
                                         work_deriv, work_perm, work_tmp, rfft_plan)
            _GPU_CHEB_DERIV_CACHE[key] = p
        end
        p::GPUChebyshevDerivPlan{T}
    end
end

# ---------------------------------------------------------------------------
# KernelAbstractions kernels (all operate on (n, batch) matrices)
# ---------------------------------------------------------------------------

"""Build DCT-I symmetric extension: (n, batch) → (2*(n-1), batch).
Element map over the EXTENSION, ndrange = (2*(n-1), batch)."""
@kernel function _dct1_ext_kernel!(work, @Const(inp), n, batch)
    i, j = @index(Global, NTuple)
    # i in 1..n copies; i = n+k (k = 1..n-2) mirrors inp[n-k] = inp[2n-i].
    @inbounds work[i, j] = i <= n ? inp[i, j] : inp[2n - i, j]
end

"""Fused reverse + symmetric extension (forward DCT-I head), ndrange = (2*(n-1), batch):
work = sym-extension of reverse(inp). rev[i] = inp[n+1-i]; ext at i = n+k is
rev[n-k] = inp[k+1] = inp[i-n+1]."""
@kernel function _dct1_reverse_ext_kernel!(work, @Const(inp), n, batch)
    i, j = @index(Global, NTuple)
    @inbounds work[i, j] = i <= n ? inp[n - i + 1, j] : inp[i - n + 1, j]
end

"""Fused endpoint-double + symmetric extension (backward DCT-I head),
ndrange = (2*(n-1), batch). The mirrored region only touches interior rows
(src = 2..n-1), so the doubling condition never fires there."""
@kernel function _dct1_prescale_ext_kernel!(work, @Const(inp), n, batch)
    i, j = @index(Global, NTuple)
    @inbounds begin
        src = i <= n ? i : 2n - i
        v = inp[src, j]
        if src == 1 || src == n
            v *= 2
        end
        work[i, j] = v
    end
end

"""Extract real part of complex matrix, ndrange = (n, batch)."""
@kernel function _extract_real_kernel!(out, @Const(cx), n, batch)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = real(cx[i, j])
end

"""Fused extract-real + forward DCT-I normalization on raw REDFT00 output,
ndrange = (n, batch): multiply by 1/(N-1) and half-weight the two endpoints."""
@kernel function _dct1_extract_normalize_kernel!(out, @Const(cx), n, batch, inv_nm1::T) where {T}
    i, j = @index(Global, NTuple)
    @inbounds begin
        v = real(cx[i, j]) * inv_nm1
        if i == 1 || i == n
            v *= T(0.5)
        end
        out[i, j] = v
    end
end

"""Fused extract-real + reverse + ×½ (inverse DCT-I tail), ndrange = (n, batch):
out[i,j] = real(cx[n+1-i, j]) / 2."""
@kernel function _dct1_extract_finalize_kernel!(out, @Const(cx), n, batch)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = real(cx[n - i + 1, j]) / 2
end

"""Pack a complex (n, batch) matrix as a real (n, 2*batch) matrix — re parts in
columns 1..batch, im parts in batch+1..2*batch — so ONE batched DCT-I covers
both. ndrange = (n, 2*batch)."""
@kernel function _cheb_pack_reim_kernel!(out, @Const(cx), n, batch)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = j <= batch ? real(cx[i, j]) : imag(cx[i, j - batch])
end

"""Inverse of `_cheb_pack_reim_kernel!`: recombine the two column blocks into a
complex (n, batch) matrix. ndrange = (n, batch)."""
@kernel function _cheb_unpack_reim_kernel!(cx, @Const(re_im), n, batch)
    i, j = @index(Global, NTuple)
    @inbounds cx[i, j] = Complex(re_im[i, j], re_im[i, j + batch])
end

"""
Chebyshev coefficient → derivative coefficient recurrence.

Reads raw DCT-I output (shape n×batch), writes derivative coefficients.
Mirrors the recurrence in `chebyshev_derivative_1d!` exactly:
  - Normalize by inv_nm1 = 1/(N-1), halve the last endpoint (c_{N-1})
  - Recurrence: c'_{k-1} = 2k*c_k + c'_{k+1}, k = N-1 down to 1
  - Halve c'_0, apply domain scale, un-normalize both endpoints

One thread per COLUMN (ndrange = batch): the recurrence is serial in k.
"""
@kernel function _cheb_coeff_to_deriv_kernel!(deriv, @Const(coeff), n, batch,
                                               inv_nm1::T, scale::T) where {T}
    j = @index(Global)
    if j <= batch
        @inbounds begin
            # Every element of `deriv` is written EXACTLY ONCE and never read
            # back: the recurrence chains through two rolling registers instead
            # of the output array. The earlier formulation (write the raw
            # recurrence, then patch deriv[1]/deriv[n] with scalar statements
            # around a scale loop) was MISCOMPILED by the KernelAbstractions
            # CPU codegen — its no-alias/ivdep assumptions let the compiler
            # reorder the same-slot read-modify-writes, and deriv[1] came out
            # unscaled (caught by test_gpu_dct1_kernels_cpu.jl, which runs this
            # very kernel object against chebyshev_derivative_1d!). Single-
            # write form is immune on every backend, and faster.
            #
            # Net factors of the old dance: deriv[1] (/2 then ×2) → ×scale
            # exactly (halving/doubling by 2 is exact in binary FP);
            # deriv[n] → ×2·scale, but its raw value is always 0.
            deriv[n, j] = zero(T)
            raw2 = zero(T)   # raw c'_{k+2}
            raw1 = zero(T)   # raw c'_{k+1}
            for k in n-1:-1:1
                s = coeff[k+1, j] * inv_nm1
                if k + 1 == n
                    s *= T(0.5)  # last endpoint c_{N-1} is halved
                end
                raw0 = 2 * T(k) * s + raw2
                deriv[k, j] = raw0 * scale
                raw2 = raw1
                raw1 = raw0
            end
        end
    end
end

# ---------------------------------------------------------------------------
# 1-pass derivative (order = 1) on (n, batch) matrices
# ---------------------------------------------------------------------------

"""
`out_mat === inp_mat` is allowed: the input is fully consumed by the first
kernel (into `work_ext`) and `out_mat` is only written by the last one.
"""
function _apply_gpu_cheb_deriv_1!(inp_mat::CuMatrix{T}, out_mat::CuMatrix{T},
                                   scale::Float64, plan::GPUChebyshevDerivPlan{T}) where {T}
    n     = plan.n
    batch = plan.batch
    M     = 2 * (n - 1)
    arch  = Tarang.architecture(inp_mat)
    inv_nm1 = T(1.0 / (n - 1))
    sc_T    = T(scale)

    # Step 1: fused reverse (ascending → descending CGL grid) + symmetric extension
    launch!(arch, _dct1_reverse_ext_kernel!, plan.work_ext, inp_mat, n, batch;
            ndrange=(M, batch))

    # Step 2: batched rfft along dim 1 → DCT-I output
    mul!(plan.work_cx, plan.rfft_plan, plan.work_ext)

    # Step 3: extract real part → Chebyshev coefficients (raw)
    launch!(arch, _extract_real_kernel!, plan.work_real, plan.work_cx, n, batch;
            ndrange=(n, batch))

    # Step 4: recurrence → derivative coefficients in work_deriv
    launch!(arch, _cheb_coeff_to_deriv_kernel!, plan.work_deriv, plan.work_real,
            n, batch, inv_nm1, sc_T; ndrange=batch)

    # Step 5: symmetric extension of derivative coefficients (no reversal)
    launch!(arch, _dct1_ext_kernel!, plan.work_ext, plan.work_deriv, n, batch;
            ndrange=(M, batch))

    # Step 6: batched rfft again → DCT-I of derivative coefficients
    mul!(plan.work_cx, plan.rfft_plan, plan.work_ext)

    # Step 7: fused extract real + reverse back + scale by 1/2
    launch!(arch, _dct1_extract_finalize_kernel!, out_mat, plan.work_cx, n, batch;
            ndrange=(n, batch))

    return out_mat
end

function _apply_gpu_cheb_deriv_nth!(inp_mat::CuMatrix{T}, out_mat::CuMatrix{T},
                                     scale::Float64, order::Int,
                                     plan::GPUChebyshevDerivPlan{T}) where {T}
    if order == 0
        out_mat === inp_mat || copyto!(out_mat, inp_mat)
        return out_mat
    end
    _apply_gpu_cheb_deriv_1!(inp_mat, out_mat, scale, plan)
    for _ in 2:order
        # work_tmp is a dedicated plan buffer (never touched inside the 1-pass
        # body), so higher orders stay allocation-free.
        copyto!(plan.work_tmp, out_mat)
        _apply_gpu_cheb_deriv_1!(plan.work_tmp, out_mat, scale, plan)
    end
    return out_mat
end

# ---------------------------------------------------------------------------
# Main dispatch: overrides Tarang._gpu_chebyshev_deriv! for CuArray
# ---------------------------------------------------------------------------

"""
    _gpu_cheb_deriv_into!(dest, data_g::CuArray{<:AbstractFloat}, axis, order, scale)

Batched real Chebyshev derivative, written directly into `dest` (same shape and
eltype as `data_g`; `dest === data_g` is allowed). The permutation staging for
`axis != 1` runs through the cached plan's `work_perm` buffer — no per-call
device allocation anywhere on this path.
"""
function _gpu_cheb_deriv_into!(dest::CuArray{T}, data_g::CuArray{T}, axis::Int,
                               order::Int, scale::Float64) where {T<:AbstractFloat}
    nd = ndims(data_g)
    n  = size(data_g, axis)
    batch = prod(size(data_g)) ÷ n
    plan = _get_gpu_cheb_deriv_plan(n, batch, T)

    if axis == 1
        in_mat  = reshape(data_g, n, batch)
        out_mat = reshape(dest, n, batch)
        _apply_gpu_cheb_deriv_nth!(in_mat, out_mat, scale, order, plan)
    else
        other_dims = ntuple(i -> i < axis ? i : i + 1, nd - 1)
        perm  = (axis, other_dims...)
        iperm = invperm(perm)
        perm_shape = ntuple(i -> size(data_g, perm[i]), nd)

        in_perm = reshape(plan.work_perm, perm_shape)
        permutedims!(in_perm, data_g, perm)
        mat = reshape(plan.work_perm, n, batch)
        _apply_gpu_cheb_deriv_nth!(mat, mat, scale, order, plan)
        permutedims!(dest, reshape(mat, perm_shape), iperm)
    end
    return dest
end

"""
    _gpu_cheb_deriv_complex_into!(dest, data_g::CuArray{<:Complex}, axis, order, scale)

Complex Chebyshev derivative: the DCT-I workspace is real-only, so pack the real
and imaginary parts as EXTRA BATCH COLUMNS — one (n, 2*batch) matrix — and run a
single batched derivative over both, then recombine. (The transform is linear;
this replaces the old `real.(x)` / `imag.(x)` / `complex.(re, im)` version that
allocated ~10 full arrays and ran the whole DCT chain twice.)
"""
function _gpu_cheb_deriv_complex_into!(dest::CuArray{Complex{T}}, data_g::CuArray{Complex{T}},
                                       axis::Int, order::Int, scale::Float64) where {T<:AbstractFloat}
    nd = ndims(data_g)
    n  = size(data_g, axis)
    batch = prod(size(data_g)) ÷ n
    arch  = Tarang.architecture(data_g)
    plan2 = _get_gpu_cheb_deriv_plan(n, 2 * batch, T)
    packed = reshape(plan2.work_perm, n, 2 * batch)

    if axis == 1
        in_mat = reshape(data_g, n, batch)
        launch!(arch, _cheb_pack_reim_kernel!, packed, in_mat, n, batch;
                ndrange=(n, 2 * batch))
        _apply_gpu_cheb_deriv_nth!(packed, packed, scale, order, plan2)
        out_mat = reshape(dest, n, batch)
        launch!(arch, _cheb_unpack_reim_kernel!, out_mat, packed, n, batch;
                ndrange=(n, batch))
    else
        other_dims = ntuple(i -> i < axis ? i : i + 1, nd - 1)
        perm  = (axis, other_dims...)
        iperm = invperm(perm)
        perm_shape = ntuple(i -> size(data_g, perm[i]), nd)

        # Complex permutation staging comes from the shared scratch cache
        # (count=2 so this key never collides with the count=1 users in the
        # transform chain).
        cscratch = get_gpu_dct_scratch(arch, perm_shape, Complex{T}, 2)[1]
        permutedims!(cscratch, data_g, perm)
        cmat = reshape(cscratch, n, batch)
        launch!(arch, _cheb_pack_reim_kernel!, packed, cmat, n, batch;
                ndrange=(n, 2 * batch))
        _apply_gpu_cheb_deriv_nth!(packed, packed, scale, order, plan2)
        launch!(arch, _cheb_unpack_reim_kernel!, cmat, packed, n, batch;
                ndrange=(n, batch))
        permutedims!(dest, reshape(cmat, perm_shape), iperm)
    end
    return dest
end

function Tarang._gpu_chebyshev_deriv!(result::Tarang.ScalarField,
                                       operand::Tarang.ScalarField,
                                       data_g::CuArray, axis::Int, order::Int,
                                       scale::Float64)
    n  = size(data_g, axis)

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

    # The derivative writes STRAIGHT into the result's grid buffer (allocated
    # here only if missing or mis-shaped — steady-state calls reuse it).
    result_data = Tarang.get_grid_data(result)
    if !(result_data isa CuArray) || eltype(result_data) != T ||
       size(result_data) != size(data_g)
        Tarang.set_grid_data!(result, CUDA.zeros(T, size(data_g)...))
        result_data = Tarang.get_grid_data(result)
    end

    if T <: Complex
        # COMPLEX grid data (e.g. a ComplexFourier × Chebyshev field): packed
        # re/im batch — see _gpu_cheb_deriv_complex_into!.
        _gpu_cheb_deriv_complex_into!(result_data, data_g, axis, order, scale)
    else
        _gpu_cheb_deriv_into!(result_data, data_g, axis, order, scale)
    end

    return true
end

# ============================================================================
# Standalone DCT-I (REDFT00) along a dimension — extracted from the verified
# Chebyshev-derivative DCT-I building blocks above.
# ============================================================================
#
# The derivative path `_apply_gpu_cheb_deriv_1!` performs:
#     reverse+extension → rfft → extract-real   (forward DCT-I)
#     → recurrence (norm + endpoint half-weight + derivative + un-normalize)
#     → extension → rfft → extract-real+reverse+½  (inverse DCT-I)
#
# Here we lift JUST the transform (NOT the derivative recurrence) into a plain
# forward/backward DCT-I that matches the CPU `_chebyshev_forward` /
# `_chebyshev_backward` convention exactly (see transform_chebyshev.jl):
#
#   forward  = REDFT00 · (1/(N-1)) · (½ at endpoints) · (odd-index sign flip)
#   backward = (odd-index sign flip) · (×2 at endpoints) · REDFT00 · ½
#
# Convention judgment call (documented):
#   The verified GPU derivative code bridges Tarang's `-cos(πj/(N-1))` grid to
#   FFTW's `+cos(πj/(N-1))` grid by REVERSING the data in grid space, whereas
#   the CPU path applies `_flip_odd_indices_along_axis!` on the coefficients.
#   These are algebraically identical:  REDFT00(reverse(x))[k] = (-1)^k REDFT00(x)[k],
#   i.e. a grid reversal == an odd-degree coefficient sign flip. The reversal is
#   fused into the head/tail kernels (`_dct1_reverse_ext_kernel!` forward,
#   `_dct1_extract_finalize_kernel!` backward), and the (1/(N-1)) norm with the
#   endpoint half/double weights into `_dct1_extract_normalize_kernel!` /
#   `_dct1_prescale_ext_kernel!` — the SAME constants the recurrence kernel uses.

"""
    _dct1_batch_mat!(out_mat, in_mat, direction, plan, arch)

Shared 3-launch DCT-I core on (n, batch) matrices: head kernel (fused
reverse/prescale + symmetric extension) → batched rfft → tail kernel (fused
extract + normalize/finalize). `out_mat === in_mat` is allowed — the head
kernel fully consumes the input before the tail writes.
"""
function _dct1_batch_mat!(out_mat::CuMatrix{T}, in_mat::CuMatrix{T}, direction::Symbol,
                          plan::GPUChebyshevDerivPlan{T}, arch) where {T}
    n, batch = plan.n, plan.batch
    M = 2 * (n - 1)
    inv_nm1 = T(1.0 / (n - 1))

    if direction === :forward
        launch!(arch, _dct1_reverse_ext_kernel!, plan.work_ext, in_mat, n, batch;
                ndrange=(M, batch))
        mul!(plan.work_cx, plan.rfft_plan, plan.work_ext)
        launch!(arch, _dct1_extract_normalize_kernel!, out_mat, plan.work_cx, n, batch,
                inv_nm1; ndrange=(n, batch))
    elseif direction === :backward
        launch!(arch, _dct1_prescale_ext_kernel!, plan.work_ext, in_mat, n, batch;
                ndrange=(M, batch))
        mul!(plan.work_cx, plan.rfft_plan, plan.work_ext)
        launch!(arch, _dct1_extract_finalize_kernel!, out_mat, plan.work_cx, n, batch;
                ndrange=(n, batch))
    else
        error("direction must be :forward or :backward, got $direction")
    end
    return out_mat
end

"""
    gpu_dct1_along_dim!(output, input, dim, direction) -> output

Plain DCT-I (REDFT00, 1/(N-1) norm, endpoint half-weight, odd-index sign flip)
along `dim` of a CuArray, `:forward` or `:backward`. Real arrays run one batched
DCT-I; complex arrays pack re/im as extra batch columns and run ONE batched
DCT-I of width 2·batch (the transform is linear and real-coefficient). The
transform axis is viewed first as an (N, batch) matrix; the cached
`GPUChebyshevDerivPlan` supplies the symmetric-extension/rfft workspace and the
permutation buffer. Matches the CPU Chebyshev DCT-I convention in
`transform_chebyshev.jl` (see the comment block above for the grid-reversal ==
odd-flip equivalence).

NOTE: this is the plain transform only — it does NOT truncate/zero-pad the
coefficient axis. `size(output) == size(input)` is required.
"""
function gpu_dct1_along_dim!(output::CuArray{T,N}, input::CuArray{T,N},
                             dim::Int, direction::Symbol) where {T<:AbstractFloat,N}
    ensure_device!(Tarang.architecture(input))
    @assert size(output) == size(input) "gpu_dct1_along_dim!: output and input must match size"
    @assert 1 <= dim <= N "dim must be between 1 and $N"

    n = size(input, dim)

    if n <= 1
        output === input || copyto!(output, input)
        return output
    end

    batch = prod(size(input)) ÷ n
    arch  = Tarang.architecture(input)
    plan  = _get_gpu_cheb_deriv_plan(n, batch, T)

    if dim == 1
        _dct1_batch_mat!(reshape(output, n, batch), reshape(input, n, batch),
                         direction, plan, arch)
    else
        # Put the transform dimension first, staging through the plan's
        # dedicated permutation buffer (work_perm — NOT used by the DCT core).
        other_dims = ntuple(i -> i < dim ? i : i + 1, N - 1)
        perm   = (dim, other_dims...)
        iperm  = invperm(perm)
        perm_shape = ntuple(i -> size(input, perm[i]), N)

        in_perm = reshape(plan.work_perm, perm_shape)
        permutedims!(in_perm, input, perm)
        mat = reshape(plan.work_perm, n, batch)
        _dct1_batch_mat!(mat, mat, direction, plan, arch)
        permutedims!(output, reshape(mat, perm_shape), iperm)
    end
    return output
end

function gpu_dct1_along_dim!(output::CuArray{Complex{T},N}, input::CuArray{Complex{T},N},
                             dim::Int, direction::Symbol) where {T<:AbstractFloat,N}
    ensure_device!(Tarang.architecture(input))
    @assert size(output) == size(input) "gpu_dct1_along_dim!: output and input must match size"
    @assert 1 <= dim <= N "dim must be between 1 and $N"

    n = size(input, dim)

    if n <= 1
        output === input || copyto!(output, input)
        return output
    end

    batch = prod(size(input)) ÷ n
    arch  = Tarang.architecture(input)
    plan2 = _get_gpu_cheb_deriv_plan(n, 2 * batch, T)
    packed = reshape(plan2.work_perm, n, 2 * batch)

    if dim == 1
        in_mat = reshape(input, n, batch)
        launch!(arch, _cheb_pack_reim_kernel!, packed, in_mat, n, batch;
                ndrange=(n, 2 * batch))
        _dct1_batch_mat!(packed, packed, direction, plan2, arch)
        launch!(arch, _cheb_unpack_reim_kernel!, reshape(output, n, batch), packed,
                n, batch; ndrange=(n, batch))
    else
        other_dims = ntuple(i -> i < dim ? i : i + 1, N - 1)
        perm   = (dim, other_dims...)
        iperm  = invperm(perm)
        perm_shape = ntuple(i -> size(input, perm[i]), N)

        # count=2 keeps this scratch key disjoint from the count=1 users in the
        # transform chain (C2R input protection), which can share (shape, T).
        cscratch = get_gpu_dct_scratch(arch, perm_shape, Complex{T}, 2)[1]
        permutedims!(cscratch, input, perm)
        cmat = reshape(cscratch, n, batch)
        launch!(arch, _cheb_pack_reim_kernel!, packed, cmat, n, batch;
                ndrange=(n, 2 * batch))
        _dct1_batch_mat!(packed, packed, direction, plan2, arch)
        launch!(arch, _cheb_unpack_reim_kernel!, cmat, packed, n, batch;
                ndrange=(n, batch))
        permutedims!(output, reshape(cmat, perm_shape), iperm)
    end
    return output
end
