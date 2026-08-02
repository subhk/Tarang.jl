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

**Corrected after a second pass — the first version of this document had this wrong,
and the correction changes where the fix belongs.**

`BCValueType` (`src/core/boundary_conditions/types.jl:146`) does not include operator
trees:

```julia
const BCValueType = Union{Real, String, Function, FieldReference,
                          TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue}
```

So `add_bc!(prob, "b(z=0) = h*w + q")` stores the **string** `"h*w + q"` on the BC. The
operator tree in the warning is built later, when the BC is lowered into an equation,
and lands in `eq_data["F"]`.

That rules out the fix this document originally proposed. `is_space_dependent`
(`construction.jl:46`) tests a string's free symbols against `_BC_SPACE_SYMBOLS`
(`x`, `y`, `z`, `r`, `theta`, `phi`). For `"h*w + q"` the free symbols are `h`, `w`, `q`
— none is a coordinate, so it is correctly not "space dependent" in that sense, and
the function has no namespace with which to discover that those names resolve to
field-valued parameters. Making it namespace-aware would push problem state into BC
construction, which runs before the namespace is necessarily complete.

The chain is therefore:

1. `add_bc!` stores the string; the BC is neither time- nor space-dependent by the
   existing tests, so it joins neither `time_dependent_bcs` nor `space_dependent_bcs`;
2. BC→equation lowering parses the string against the namespace, producing
   `AddOperator{MultiplyOperator{ScalarField, ScalarField}, ScalarField}` in
   `eq_data["F"]`;
3. `_lift_bc_value_into_eq_data!` finds nothing in either cache and returns early,
   leaving that tree in place;
4. `_evaluate_alg_F` (`subproblem_rhs.jl:45`) does not recognise it, warns, and
   returns zero.

**The fix belongs at step 2** — the BC→equation lowering, the one place where both the
BC object (carrying `coordinate` and `position`) and the resolved expression tree are
in scope at once. Neither `is_space_dependent` (string only, no namespace) nor
`_evaluate_alg_F` (tree only, no coordinate or position) can see both.

## Verified along the way

* An expression built directly in Julia — `a*b + c` over `ScalarField`s — **evaluates
  eagerly to a `ScalarField`**; it never becomes a tree. Only the string form produces
  one. Both must be handled, and they arrive by different routes.
* `evaluate_solver_expression(expr, Operand[]; layout = :g)` collapses such a tree to a
  `ScalarField` correctly with an EMPTY variables vector, because the leaves are
  already concrete field objects. Measured against the analytic value: exact.

That second point is what makes the remaining work tractable: evaluation is solved,
and what is left is the boundary slice plus the MPI gather.

## Design

Treat a field-expression BC as **space- and time-dependent**, and route it through
the pipeline that already serves symbolic space-dependent BCs — the one measured
above as exact to 6.7e-16. That pipeline already produces a global boundary-plane
array and already refreshes per step via `_drop_spatial_cache_entries!`.

Three pieces:

### 1. Recognise the value, at lowering time

At BC→equation lowering, after the string has been parsed against the namespace, test
whether the resulting tree references a `ScalarField`/`VectorField`. Reuse
`_detect_equation_variables` (already used by `_references_variable` in
`problem_matrices_spectral.jl`) rather than writing a new tree walk.

A BC that does must be registered as **both** space- and time-dependent, so its cached
value is keyed by time and re-evaluated every step: the referenced fields evolve, and a
value cached once would pin the boundary to its first-step value — a bug that a
single-step test would not catch.

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
