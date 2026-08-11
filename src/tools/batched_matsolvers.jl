# ── Batched dense LU over Fourier modes ──────────────────────────────────────
#
# Every mode's stage matrix `(M + dt*a_ii*L)` factored and solved in one call
# instead of one per mode. Dense rather than sparse because the per-mode
# matrices measure ~28% dense with full bandwidth (Chebyshev tau rows), so the
# sparse structure buys almost nothing while costing a per-mode launch.
#
# Sparse-with-shared-symbolic was considered and rejected: the sparsity pattern
# IS identical across modes, but partial pivoting diverges per mode, which
# breaks the shared-symbolic premise.

"""
    BatchedDenseLU(A)

Factor and solve `A[:, :, m] * x = b` for every `m` in one call. Callers are
expected to fully overwrite `A` (see `batched_assemble_lhs!`) and call
`batched_factor!` again when `dt` changes — **never factor the same buffer
twice**, on either backend (see below for why this matters more on one than
the other).

**`A`'s fate after `batched_factor!` differs by backend, and this is not
cosmetic.** On GPU, the factorization genuinely destroys `A`: cuBLAS's
batched `getrf` factors directly into the buffers `A`'s own pointers name, so
`s.A` holds the LU factors afterward, not the original matrices. On CPU, the
reference implementation below calls `lu` (not `lu!`), which copies before
factoring, so `s.A` happens to still hold the original matrices afterward —
an accident of using the non-mutating entry point, not a guarantee either
path makes to callers. Do not rely on `s.A` surviving `batched_factor!` on
either backend: the one CPU-only test that does (`test_batched_dense_lu.jl`'s
"refactoring after the matrix changes" testset, which does `s.A .*= 2` after
an initial factor) is marked CPU-specific for exactly this reason — the
identical sequence on GPU would scale and re-factor the LU factors, not the
operator, and return a plausible wrong answer with no error. The supported
lifecycle is assemble-then-factor: overwrite `A` completely, then
`batched_factor!`, on both backends alike.

Parametrized on the storage type of `A` (`AT`) rather than declared with a
bare `A::AbstractArray{ComplexF64,3}` field. This is not stylistic: it is
what lets the CUDA extension add a GPU method at all. `_batched_factor_impl!`
below dispatches on `s::BatchedDenseLU` — Julia sugar for
`s::BatchedDenseLU{AT} where AT` — so a monomorphic (non-parametric) struct
would give the extension no more-specific signature to hang a second method
on; it would have to redefine the IDENTICAL signature `Tuple{BatchedDenseLU}`,
which replaces the CPU method rather than adding to it (methods are keyed by
full type signature, not defining module). A same-signature replacement that
tries to fall back to "the CPU path" via
`A isa CuArray || return invoke(_batched_factor_impl!, Tuple{BatchedDenseLU}, s)`
recurses into itself — confirmed with a throwaway two-module reproduction
(`invoke` re-resolves the exact same signature, which by then IS the
replacement). Parametrizing `A` gives the extension a genuinely more specific
method, `_batched_factor_impl!(s::BatchedDenseLU{<:CuArray})`, which coexists
with the generic one under ordinary multiple dispatch — no `isa`, no
`invoke`, no collision. Mirrors the `_gpu_cusolver_module(::Val)` /
`::Val{:cuda}` precedent in `gpu_matsolvers.jl` for the same underlying
problem (a same-signature extension method overwriting the base one).
"""
mutable struct BatchedDenseLU{AT<:AbstractArray{ComplexF64, 3}}
    A::AT
    pivots::Any
    info::Any
    factored::Bool
end

BatchedDenseLU(A::AbstractArray{ComplexF64, 3}) =
    BatchedDenseLU(A, nothing, nothing, false)

"""
    batched_factor!(s::BatchedDenseLU) -> s

LU-factor every mode in place.

Raises if any mode is singular, naming it. This check is not optional: the
batched LAPACK/CUBLAS entry points report per-matrix status in an `info` array
and return normally regardless, so an unchecked singular mode yields buffer
contents that read as a plausible solution and propagate through the timestep
undetected.
"""
function batched_factor!(s::BatchedDenseLU)
    return _batched_factor_impl!(s)
end

# CPU reference path. The GPU method is added by the CUDA extension.
function _batched_factor_impl!(s::BatchedDenseLU)
    A = s.A
    n, _, nmodes = size(A)
    facts = Vector{Any}(undef, nmodes)
    for m in 1:nmodes
        F = lu(view(A, :, :, m); check=false)
        if !issuccess(F)
            error("BatchedDenseLU: mode $m of $nmodes is singular " *
                  "(order $n). A singular stage matrix usually means the " *
                  "problem is under-constrained at this Fourier mode — check " *
                  "the tau/BC rows for that mode.")
        end
        facts[m] = F
    end
    s.pivots = facts
    s.info = zeros(Int, nmodes)
    s.factored = true
    return s
end

"""
    batched_solve!(X, s::BatchedDenseLU, B) -> X

Solve every mode against the stored factorization. `X` and `B` are
`(n, nmodes)`, column `m` being mode `m`. `X` may alias `B`.
"""
function batched_solve!(X::AbstractMatrix{ComplexF64}, s::BatchedDenseLU,
                        B::AbstractMatrix{ComplexF64})
    s.factored || error("BatchedDenseLU: batched_solve! called before " *
                        "batched_factor!; the factorization is stale or absent")
    return _batched_solve_impl!(X, s, B)
end

function _batched_solve_impl!(X::AbstractMatrix{ComplexF64}, s::BatchedDenseLU,
                              B::AbstractMatrix{ComplexF64})
    X === B || copyto!(X, B)
    for m in axes(X, 2)
        ldiv!(s.pivots[m], view(X, :, m))
    end
    return X
end
