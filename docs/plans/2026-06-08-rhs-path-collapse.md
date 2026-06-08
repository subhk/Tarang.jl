# Plan: collapse the dual RHS-evaluation paths

Date: 2026-06-08
Status: proposed
Scope target: remove the *silent drift* between the two RHS implementations, not delete one outright.

## Problem

Equation-RHS evaluation has two implementations of the same operator semantics:

- **Lazy** (`src/core/solvers/lazy_rhs.jl`, 917 LOC) — `translate_to_lazy` builds a typed
  `LazyFuture` tree once (`build_lazy_rhs_plan!`), evaluated by `evaluate_lazy!` /
  `execute_lazy_rhs_buffered!`. Fast, type-specialized, low-alloc.
- **Interpreted** (`src/core/solvers/solver_compiled_rhs.jl`, 722 LOC) — `evaluate_solver_expression`
  walks the operator tree directly. ~100× more allocation.

`evaluate_rhs` (`state_utils.jl:15`) picks lazy when `is_compiled`, else silently falls back to
interpreted. The two grammars must be kept in lockstep by hand (B2, 2026-06-07, had to widen both),
they drift, and the fallback is **silent** — a single unsupported operator in one equation drops the
whole solver to the 100× path with no signal. The `dx` silent-drop bug lived in this seam.

## Key finding from recon (changes the goal)

**Interpreted is NOT just an RHS fallback — it is load-bearing elsewhere** and cannot be deleted:

- BVP steady solve — `solver_stepping.jl:238`
- Nonlinear BVP Newton iteration — `solver_stepping.jl:322`
- Algebraic constraint solving — `state_utils.jl:828, 933, 953` (`_try_solve_simple_constraint!`,
  `_evaluate_poisson_rhs`)

And: **full collapse buys ~zero performance** — lazy already compiles on first setup; interpreted is
fallback-only on the IVP path. So the win is **maintainability + correctness** (kill the drift and the
silent fallback), not speed.

Lazy is a strict *subset* of interpreted. To make lazy total for the IVP RHS, these must gain lazy
support (interpreted handles them, lazy returns `nothing` → fallback):

1. `TensorField` (`solver_compiled_rhs.jl:176`)
2. `ArrayOperator` (`:185`)
3. `IndexOperator` (`:283`)
4. `DotProduct` / `CrossProduct` Futures (`_translate_future_to_lazy` only does Add/Sub/Negate/Multiply)
5. Generic `Operator` catch-all (`:299` — interpreted calls `evaluate(expr,layout)`)

## Goal (revised, realistic)

Not "delete interpreted." Instead: **the IVP-RHS path has ONE implementation (lazy) and never silently
degrades.** Interpreted survives, explicitly scoped to constraints/BVP/Newton.

## Steps

1. **Make the fallback loud first (cheap, ship immediately).** In `build_lazy_rhs_plan!`, when
   `translate_to_lazy` returns `nothing`, record *which* sub-expression failed and `@warn` once with it
   (today it is a bare `@info`). Add a `solver` option `require_lazy_rhs=false`; when true, throw instead
   of falling back. This removes the silent-100× trap with ~20 LOC and no risk. **Do this regardless of
   the rest.**
2. **Close the lazy capability gap**, one operator per PR, each mirroring the interpreted semantics +
   a parity test (assert lazy result ≈ interpreted result on the same expression):
   - `LazyArray` (ArrayOperator) — trivial, embeds a constant array.
   - `LazyIndex` (IndexOperator).
   - Dot/Cross Futures in `_translate_future_to_lazy` (reuse existing vector machinery).
   - `LazyTensor` (TensorField) — largest; defer or leave to fallback if rarely used on RHS.
   - Generic catch-all: pre-lower unknown operators via `evaluate(...)` into a `LazyArray` leaf at
     translate time (turns "fallback whole solver" into "fallback one node").
3. **Demote interpreted to its real role.** Rename `evaluate_solver_expression` →
   `evaluate_expression_eager` (or similar), document it as the constraint/BVP/Newton evaluator, and make
   the IVP RHS default to `require_lazy_rhs=true` once the gap is closed. Interpreted is no longer reached
   on the IVP-RHS happy path → no drift.
4. **Single operator-semantics source (optional, later).** Factor the per-operator numeric kernels
   (add/mul/div/pow/ufunc/diff) so lazy `evaluate_lazy!` and eager `evaluate_solver_expression` call the
   *same* underlying field ops, so they cannot diverge.

## Risks

- Parity tests are essential — lazy is grid-pointwise for ufuncs/div/pow (no dealias) while a future
  reader might assume dealiasing; the parity test pins lazy == eager.
- `require_lazy_rhs=true` could break a user with an exotic operator → keep it opt-out, default-on only
  after the gap closes, and the loud `@warn` names the culprit.

## Validation

Parity test per operator (lazy ≈ eager on identical expr); the existing IVP integration tests
(diagonal_imex, lazy_rhs_fourier, nonlinear); alloc check that nothing regresses to the eager path.

## Effort / ROI

- Step 1 (loud fallback): ~½ day, **high ROI, no risk — do now.**
- Steps 2–3 (close gap + demote): ~3–5 days. Medium ROI (maintainability/correctness, no perf).
- Step 4 (unify kernels): ~2 days. Nice-to-have.

**Verdict:** worth it in the *scoped* form. Step 1 is a no-brainer. Do NOT pursue "delete the 722 LOC" —
interpreted is load-bearing for constraints/BVP and deletion buys no performance.
