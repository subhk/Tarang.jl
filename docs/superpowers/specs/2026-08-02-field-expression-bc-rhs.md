# Field-expression boundary-condition right-hand sides

**Date:** 2026-08-02
**Status:** specified, not implemented. Investigation complete — every piece below was
verified against the running code, not inferred.

## Problem

A boundary condition whose right-hand side is an expression over `ScalarField`s is
**silently enforced as zero**. Reported from a live 4-rank run:

```
Warning: [Rank 1/4] Boundary condition right-hand side of type
AddOperator{MultiplyOperator{ScalarField{...}, ScalarField{...}}, ScalarField{...}}
is not supported and is being enforced as ZERO.
  @ Tarang src/core/subsystems/subproblem_rhs.jl:55
```

It warns, so it is not silent in the strictest sense — but it still returns a
confident wrong answer, and the run continues. The solve satisfies `u = 0` on that
boundary instead of the requested condition.

## What works and what does not — measured

| BC form | Result |
|---|---|
| `"b(z=0) = 1 + 0.5*cos(x)"` — symbolic over coordinates | **works**, error 6.7e-16, no warnings |
| `"b(z=0) = 0.5"` — constant | works |
| `"b(z=0) = h*T_amb"` — compound constant | works (`_is_const_or_param` folds it) |
| `"b(z=0) = prof"` where `prof` is a precomputed `Array` via `add_parameters!` | **FAILS** — 5 warnings, enforced as 0 |
| `"b(z=0) = a*b + c"` where `a`, `b`, `c` are `ScalarField`s | **FAILS** — enforced as 0 |

The precomputed-array row is worth noting: it looks like it should work, because
`ArrayOperator` **is** a supported node at `_evaluate_alg_F`. It fails because a bare
array passed through `add_parameters!` never becomes an `ArrayOperator` — see the
root cause below.

## Root cause

`is_space_dependent` (`src/core/boundary_conditions/construction.jl:46`) recognises
only `String`, `SpaceDependentValue`, `TimeSpaceDependentValue` and `FieldReference`.
An operator expression tree over fields matches none of them, so:

1. the BC is never added to `manager.space_dependent_bcs`;
2. the spatial evaluation pipeline never runs for it, so nothing populates
   `cache.spatial_values`;
3. `_lift_bc_value_into_eq_data!` (`src/core/solvers/solver_types.jl:~805`) gets
   `value === nothing` from both the time and spatial caches and returns early,
   leaving the raw expression tree in `eq_data["F"]`;
4. that tree reaches `_evaluate_alg_F` (`src/core/subsystems/subproblem_rhs.jl:45`),
   which handles `Nothing`/`ZeroOperator`/`ConstantOperator`/`Number`/`ArrayOperator`
   and compound constants — and warns-and-zeroes everything else.

The warning at step 4 is the symptom. The gap is at step 1.

## Why the obvious fix is the wrong one

Extending `_evaluate_alg_F` looks natural and is not viable: it receives only
`(expr, sp)`. It has no boundary coordinate and no position, so it cannot know
*where* to evaluate the expression. Threading those through is possible but solves
the problem in the wrong layer — by then the per-step refresh machinery has already
been bypassed, and a BC over evolving fields must be re-evaluated every step.

## Design

Treat a field-expression BC as **space- and time-dependent**, and route it through
the pipeline that already serves symbolic space-dependent BCs — the one measured
above as exact to 6.7e-16. That pipeline already produces a global boundary-plane
array and already refreshes per step via `_drop_spatial_cache_entries!`.

Three pieces:

### 1. Recognise the value

`is_space_dependent(value)` returns `true` when `value` is an operator expression
whose tree references at least one `ScalarField`/`VectorField`. Such a BC must also
report `is_time_dependent == true` — the referenced fields evolve, so a value cached
against `(bc_index, time)` must not be reused across steps.

Reuse `_detect_equation_variables` (already used by `_references_variable` in
`problem_matrices_spectral.jl`) rather than writing a new tree walk.

### 2. Evaluate to a global boundary plane

Given the BC's `coordinate` and `position` (both are fields on `DirichletBC` /
`NeumannBC` / `RobinBC`) and the expression:

- evaluate the tree with `evaluate_solver_expression(expr, problem.variables;
  layout = :g)` — this yields a `ScalarField` on the full domain;
- take the slice at the boundary index along `coordinate`. For a Chebyshev axis the
  Gauss-Lobatto grid **includes both endpoints**, so a BC at either end is an exact
  grid slice and needs no interpolation;
- **refuse loudly** for a `position` that is not a grid endpoint. Interpolating to an
  interior point is a separate feature; guessing would reintroduce exactly the
  silent-wrong-value this work removes;
- gather to a global array over the Fourier axes. Under MPI the boundary row lives on
  whichever rank owns that index, and the consumer needs the whole line. Use the
  zero-fill + `MPI.Allreduce(+)` pattern from `_allgather_global_grid`
  (`operations_integrate.jl`) — a built-in reduction op, safe on every architecture,
  unlike `gather`.

The result must be a plain global `Array` over the Fourier axes, matching what
`_bc_array_projection` already consumes. `_bc_fourier_axis_sizes(sp)` returns those
sizes, and `_expand_bc_array_to_plane` handles singleton dims.

### 3. Store it

Write the array through `_store_spatial_bc_value!`, so `_lift_bc_value_into_eq_data!`
finds it and emits `ArrayOperator(value)` — the already-working path. No change is
needed in `_evaluate_alg_F` at all.

## Testing

This is an MPI boundary condition, which is the profile of several past bugs, so
value assertions against a serial reference are required — not smoke tests.

| Level | Assertion |
|---|---|
| serial | A BC `b(z=0) = a*b + c` over fields is enforced: the boundary row equals the expression evaluated there, to ~1e-14. Compare against the equivalent symbolic BC, which is known exact. |
| serial | The BC **refreshes**: after the fields change, the enforced boundary value tracks them. A cached first-step value would pass a single-step check. |
| serial | A `position` that is not a grid endpoint raises, naming the limitation. |
| **MPI** | The same problem at np=2 and np=4 matches the serial reference bit-for-bit. This is the assertion that catches a wrong boundary slice or a botched gather. |
| MPI | A decomposition where the boundary row is **not** on rank 0 — otherwise the gather is untested. |
| regression | The five rows of the table at the top of this document, so the working forms cannot regress and the failing ones cannot silently start returning zero again. |

Add the working cells to `test_configuration_matrix.jl`, whose contract is that every
cell either solves correctly or refuses.

## Interim guidance

Until this lands, express the BC symbolically in coordinates —
`"b(z=0) = 1 + 0.5*cos(x)"` — which is measured exact. That covers any RHS expressible
in the coordinates. It does **not** cover a BC that depends on evolving state; there
is no workaround for that case, and such a BC is currently enforced as zero with a
warning.
