# Plan: kill the `field.bases` type instability

Date: 2026-06-08
Status: proposed
**Verdict up front: do NOT do the full struct parametrization. Do the targeted iteration-barrier fix.**
Recon showed the big refactor is net-negative.

## Problem

`ScalarField` (and Vector/Tensor) store `bases::Tuple{Vararg{Basis}}` — abstract element type. Every
`for b in field.bases` boxes each element, causing widespread type instability (~900 JET reports; the
`_any_axis_dealias` per-step allocation found 2026-06-07; the per-field Domain rebuild was downstream).

The "obvious" fix: parametrize the field type on its bases, e.g.
`ScalarField{T, N, B<:NTuple{N,Basis}, S}`, so the bases tuple is concrete.

## Why the full parametrization is NOT worth it (recon findings)

Blast radius and a payoff-killing paradox:

- **160 constructor call sites**, **449 `.bases` references** (129 are iterations), **490 `::ScalarField`
  signatures** across **60–80 files**.
- **Container erasure eats ~60% of the win.** Parametrizing bases makes *each distinct bases tuple a
  distinct concrete type*. But fields live in heterogeneous containers:
  - `InitialValueSolver.state::Vector{<:ScalarField}` (`solver_types.jl:218`)
  - `FieldPool.available::Dict{PoolKey, Vector{ScalarField}}` (`field_pool.jl:56`)
  - RK stage buffers `Vector{Vector{ScalarField}}` (`step_diagonal_imex.jl`, `step_rk.jl`)
  - `_DERIV_RESULT_POOL::Dict{Tuple, Vector{ScalarField}}` (`derivatives_eval.jl:142`)
  A multi-field system (e.g. u, T on different bases) makes these vectors abstract-element *again* →
  iterating them still boxes.
- **It actively breaks an existing optimization:** `_concretize_state_fields` (`solver_types.jl:161`)
  narrows the state vector to a concrete type when all fields share a type. With per-bases types, a
  mixed-bases system fails the narrowing → state stays abstract. So the refactor can make the *solver
  state* less concrete, not more.

Net: ~70 files touched, weeks of work, and the dominant hot containers stay abstract. Bad ROI.

## Better plan: targeted iteration barriers (recommended)

The boxing is a property of the *iteration site*, not the struct. Fix the hot loops without changing
the type. Two complementary techniques:

1. **Function-barrier + splat for hot predicates.** Boxing in `for b in field.bases` disappears if the
   tuple is splatted into a vararg that Julia specializes per-arity:
   ```julia
   _any_axis_dealias(bases::Tuple, f) = _any_axis_dealias(f, bases...)   # barrier
   _any_axis_dealias(f) = false
   _any_axis_dealias(f, b, rest...) = _axis_dealias_gt1(b, f) || _any_axis_dealias(f, rest...)
   ```
   Recursion over `(b, rest...)` is type-stable (each element's concrete type is recovered). Apply to the
   handful of *per-step* loops (`_any_axis_dealias`, the dealias-cutoff loop, `_resolve_diff_axis`,
   `get_scalar_size`).
2. **Cache derived per-domain quantities** (the codebase already has `_domain_cached_get!`). Most
   `.bases` iterations recompute domain-level facts (axis maps, dealias factors, sizes) that are constant
   for a `(dist, bases)`. Route them through the domain attribute cache so the boxing loop runs *once*,
   not per step. Several of the 129 iteration sites are derivable this way.

This captures the actual *runtime* win (per-step boxing in hot loops) for a few dozen LOC across ~10
files, no type-signature churn, no container erasure, no `_concretize_state_fields` breakage.

## If deeper type-stability is still wanted later

Parametrize ONLY the dimensionality `N` (`ScalarField{T,N,S}`), not the full bases tuple. `N` is concrete
and useful (array dims, loop bounds) and causes far less type explosion than per-bases types — a
2-field system on different bases still shares `N`, so containers stay concrete. This is a middle path;
still ~40 files but avoids the worst container erasure. Evaluate only after the targeted fix, if JET
still shows material instability on a real solve.

## Steps (recommended path)

1. Identify the per-step `.bases` iteration sites (profile a real solve; `_any_axis_dealias` is known).
   Only the *hot* ones matter — the 320 non-iteration `.bases` accesses are fine.
2. Convert each hot loop to the splat/recursion barrier (technique 1). One PR, with an alloc + `@inferred`
   test per site.
3. Move constant-per-domain derivations into `_domain_cached_get!` (technique 2).
4. Re-run the JET report and the Profile.Allocs sweep; confirm the per-step boxing is gone.

## Validation

`@inferred` on each converted helper; Profile.Allocs on Burgers/multi-field solve showing the
`_any_axis_dealias`-class allocations gone; JET report count drops (gated at ≤1000 in `test_jet.jl`).

## Effort / ROI

- Targeted barriers (recommended): ~1–2 days, **good ROI** — kills the per-step boxing that actually
  shows up in profiles.
- `N`-only parametrization (optional middle path): ~1 week, medium ROI.
- Full `{T,N,B,S}` parametrization: weeks, **net-negative** (container erasure) — **not recommended.**
