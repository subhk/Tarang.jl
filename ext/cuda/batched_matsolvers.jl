# ── Batched dense LU over Fourier modes: CUDA specialization ────────────────
#
# `BatchedDenseLU.A` here is a genuine (n, n, nmodes) CuArray, exactly the
# layout CUBLAS's "strided batched" entry points expect. These methods call
# `getrf_strided_batched!`/`getrs_strided_batched!` directly on that 3-D
# array rather than building a `Vector{CuMatrix}` of per-mode `view`s: the
# Vector-of-matrices overload (`CUDA.CUBLAS.getrf_batched!(ptrs, true)`, the
# form shown in most CUDA.jl examples) internally calls `unsafe_batch`, which
# is typed `Vector{<:CuArray{T}}`. `view(A, :, :, m)` produces a `SubArray`,
# and `SubArray` is NOT a subtype of `CuArray` (it wraps one) — a
# `Vector{<:SubArray}` of per-mode views does not dispatch to `unsafe_batch`
# at all and throws `MethodError`. Confirmed via `methods(CUDA.CUBLAS.unsafe_batch)`
# (single method, `Vector{<:CuArray{T}}`) in a throwaway environment, and
# cross-checked against both CUDA.jl v6.2.1 (latest) and v5.11.3 (this
# package's `[compat]` pin) — identical signatures, identical constraint, in
# both. `getrf_strided_batched!`/`getrs_strided_batched!` take the 3-D array
# directly (no `Vector` of views needed) and are present under both versions;
# see task-4-report.md for the full `methods(...)` output.
#
# No NVIDIA GPU is available on the machine this was written on; this file is
# UNEXECUTED. See task-4-report.md for exactly what was and was not verified.

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
