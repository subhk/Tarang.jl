# ── Batched dense LU over Fourier modes: CUDA specialization ────────────────
#
# `BatchedDenseLU.A` here is a genuine (n, n, nmodes) CuArray, exactly the
# layout CUBLAS's "strided batched" entry points expect. These methods call
# `getrf_strided_batched!`/`getrs_strided_batched!` directly on that 3-D
# array rather than building a `Vector{CuMatrix}` of per-mode `view`s.
#
# Both forms are valid and would call the identical `cublasZgetrfBatched`/
# `cublasZgetrsBatched` with the same pivot array and the same per-matrix
# `info` — `view(A, :, :, m)` on a CuArray does NOT produce a `SubArray`:
# GPUArrays classifies a trailing scalar index after full-dimension slices
# as `Contiguous` (`GPUArrays/*/src/host/base.jl`'s `GPUIndexStyle`) and
# routes it through `derive`, which for a `CuArray` returns a genuine
# `CuArray{T,2}` sharing the parent's buffer (`CUDA/*/src/array.jl`,
# `GPUArrays.derive(::Type{T}, a::CuArray, ...) = CuArray{T,N}(copy(a.data), ...)`
# — the same mechanism `reshape` uses, not the generic `SubArray` fallback).
# So `[view(A,:,:,m) for m in 1:nmodes]` really is `Vector{<:CuArray{T}}` and
# does satisfy `unsafe_batch`'s constraint; the Vector-of-views form would work.
#
# The strided entry point is used anyway because it is the better fit and the
# cheaper call, not because the alternative is broken: `BatchedDenseLU.A`
# already IS the (n, n, nmodes) layout `getrf_strided_batched!` wants, so
# there is no per-mode `view`/`derive` to build at all; and
# `unsafe_strided_batch` computes the batch of per-mode pointers with a
# single on-device kernel launch, whereas the `Vector`-of-matrices path's
# `unsafe_batch` computes them on the HOST (`pointer.(batch)`) and then
# uploads that host `Vector{CuPtr{T}}` to the device — an extra host→device
# transfer the strided form skips entirely. Confirmed present with identical
# signatures under both CUDA.jl v6.2.1 (latest) and v5.11.3 (this package's
# `[compat]` pin); see task-4-report.md for the full `methods(...)` output
# and the fix-round note correcting an earlier, wrong claim that the
# Vector-of-views form throws `MethodError` (it does not).
#
# No NVIDIA GPU is available on the machine this was written on; this file is
# UNEXECUTED. See task-4-report.md for exactly what was and was not verified.
#
# ── CPU/GPU factorization lifecycle: `s.A` survives on CPU, is DESTROYED on GPU ──
#
# `BatchedDenseLU`'s docstring (src/tools/batched_matsolvers.jl) says "`A` is
# mutated in place by the factorization" — true here, but the CPU reference
# path does NOT mutate `A`: `lu(view(A,:,:,m); check=false)` is the
# non-mutating `lu` (as opposed to `lu!`), which copies before factoring, so
# `s.A` still holds the original matrices after `batched_factor!` on CPU.
# `getrf_strided_batched!(A, true)` below has no such copy — cuBLAS's
# `getrfBatched` factors directly into the buffers `A`'s own pointers name,
# so on GPU `s.A` no longer holds the original matrices after
# `batched_factor!`; it holds the LU factors. Calling `batched_factor!` twice
# on the SAME `s.A` without reassembling it in between (as the CPU-only
# "refactoring after the matrix changes" testset in
# test/test_batched_dense_lu.jl does via `s.A .*= 2`) would, on GPU, scale
# and re-factor the LU FACTORS, not the original operator — a plausible,
# wrong answer with no error. The supported lifecycle is: fully overwrite
# `s.A` (e.g. via `batched_assemble_lhs!`, Task 6) before every
# `batched_factor!` call; never factor the same buffer twice. Task 6's brief
# has been amended to require assemble-then-factor and to test the
# invariant.

"""
    Tarang._batched_factor_impl!(s::BatchedDenseLU{<:CuArray}) [CUDA]

GPU specialization, selected by ordinary multiple dispatch on `s`'s type
parameter (see the parametrization note on `BatchedDenseLU`'s docstring in
`src/tools/batched_matsolvers.jl`) — no runtime `isa` check or `invoke`
fallback needed here, unlike a same-signature redefinition would require.
Factors every mode's matrix in one `cublas<T>getrfBatched` call.

`getrf_strided_batched!` reports per-matrix status in a DEVICE `info` array
and returns normally regardless of whether any mode is singular — the check
below is the only thing standing between a singular mode and a silently
wrong solve propagating through the timestep. No `try`/`catch` reroutes to
the CPU path: a GPU failure must raise (no-silent-CPU-fallback contract,
#74).

**Destroys `s.A`.** Unlike the CPU reference path (`lu`, not `lu!`, copies
before factoring), this factors directly into `A`'s own buffers — `s.A` no
longer holds the original matrices afterward, only the LU factors. Callers
must fully overwrite `s.A` before every call to this; never factor the same
buffer twice. See the file-level comment above for the full explanation.
"""
function Tarang._batched_factor_impl!(s::Tarang.BatchedDenseLU{<:CuArray})
    A = s.A
    n, _, nmodes = size(A)
    pivots, info = CUDA.CUBLAS.getrf_strided_batched!(A, true)
    host_info = Array(info)
    bad = findall(!iszero, host_info)
    if !isempty(bad)
        error("BatchedDenseLU (GPU): singular stage matrix at mode(s) " *
              "$(bad) of $nmodes; cuBLAS info = $(host_info[bad]). " *
              "No CPU fallback is attempted — see the no-silent-fallback " *
              "contract (#74).")
    end
    s.pivots = pivots
    s.info = info
    s.factored = true
    return s
end

"""
    Tarang._batched_solve_impl!(X, s::BatchedDenseLU{<:CuArray}, B) [CUDA]

GPU specialization: solves every mode in one `cublas<T>getrsBatched` call via
`getrs_strided_batched!`, reusing the pivots `_batched_factor_impl!` stored
in `s.pivots`. `X`/`B` are `(n, nmodes)` — one RHS column per mode — reshaped
to `(n, 1, nmodes)` to match the strided-batched 3-D convention. `reshape` on
a `CuArray` returns a new `CuArray` that shares the same device buffer
(`GPUArrays.derive`, not a lazy `Base.ReshapedArray` wrapper), so this
aliases `X`'s memory rather than copying it, and the in-place solve lands
back in `X`.
"""
function Tarang._batched_solve_impl!(X::AbstractMatrix{ComplexF64},
                                     s::Tarang.BatchedDenseLU{<:CuArray},
                                     B::AbstractMatrix{ComplexF64})
    A = s.A
    n, _, nmodes = size(A)
    X === B || copyto!(X, B)
    X3 = reshape(X, n, 1, nmodes)
    info, _ = CUDA.CUBLAS.getrs_strided_batched!('N', A, X3, s.pivots)
    # getrs does not re-check singularity (that is getrf's job, already
    # checked in _batched_factor_impl!); its `info` is a single host scalar
    # flagging invalid arguments, not a per-mode array. A nonzero value here
    # means a programming error (dimension/pivot mismatch), not a numerical
    # one, but it still must not be swallowed.
    info[] == 0 || error("BatchedDenseLU (GPU): getrs_strided_batched reported " *
                         "invalid arguments (info = $(info[])).")
    return X
end
