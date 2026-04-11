# Allocation Reduction Design — Tarang.jl

**Date:** 2026-04-04
**Goal:** Eliminate per-timestep memory allocations in all hot paths, achieving near-zero allocation after warmup across all timestepper methods (Diagonal IMEX, CNAB/SBDF, Explicit RK, Pencil IMEX, ETD).

---

## 1. FieldPool — Core Mechanism

### 1.1 Structure

```julia
struct PoolKey
    bases_hash::UInt64
    dtype::DataType
end

mutable struct FieldPool
    available::Dict{PoolKey, Vector{ScalarField}}
    dist::Distributor
    in_use::Int
    max_per_key::Int  # cap per key, default 32
end
```

The pool starts empty. Fields are allocated on first checkout and recycled thereafter. After 1-2 warmup timesteps the pool reaches steady-state and no further allocations occur.

### 1.2 API

```julia
checkout!(pool, bases, dtype) -> ScalarField   # pop from available or allocate
return!(pool, field)                            # push back, reset layout
with_pool_field(f, pool, bases, dtype)          # checkout + auto-return via do-block
```

On `return!`: reset `current_layout = :g`. Data arrays are NOT zeroed (caller will overwrite). Zeroing is the caller's responsibility when needed (e.g., `evaluate_rhs` zeros after checkout).

### 1.3 Global Access

```julia
const _FIELD_POOL = Ref{Union{Nothing, FieldPool}}(nothing)
get_field_pool() = _FIELD_POOL[]
set_field_pool!(pool) = (_FIELD_POOL[] = pool)
```

Matches the existing `get_global_workspace()` pattern. Set at solver init, cleared at teardown. When `nothing`, callers fall back to regular allocation (backward compatibility).

### 1.4 ScalarField Additions

Two new fields on `ScalarField`:
- `_from_pool::Bool` — distinguishes pool fields from user-owned fields
- `_pool_generation::Int` — incremented on checkout; `@assert`-guarded for use-after-return detection in debug mode

Only pool-tagged fields are eligible for `return!`. User-created fields and `state.history` fields are never returned.

---

## 2. Integration Points

### 2.1 Arithmetic Operations (`arithmetic.jl`)

`combine_add(a::ScalarField, b::ScalarField)`, `combine_multiply`, `scale_field`, `add_scalar_to_field`, `constant_field_like` — all currently allocate via `ScalarField(dist, name, bases, dtype)`.

**Change:** Replace with `checkout!(get_field_pool(), bases, dtype)` when pool is active. `constant_field_like` caches constant fields in the pool rather than creating new ones each time.

### 2.2 Operator Evaluation (`operators/evaluate.jl`)

`evaluate_multiply`, `evaluate_add`, `evaluate_subtract`, `evaluate_power` — each creates a result field.

**Change:** Checkout from pool. Return intermediate fields after consumption.

### 2.3 RHS Evaluation (`timesteppers/state_utils.jl`)

`evaluate_rhs` calls `create_rhs_zero_field()` for every state field on every RHS evaluation.

**Change:** Checkout from pool, then `fill!(get_grid_data(field), 0)` and `fill!(get_coeff_data(field), 0)` instead of allocating.

### 2.4 Derivative Evaluation (`operators/derivatives.jl`)

`evaluate_differentiate`, divergence, gradient create result fields via `copy(operand)`.

**Change:** Checkout from pool, then `copy_field_data!(result, operand)` in-place.

### 2.5 Future Evaluation (`future.jl`)

After `operate()` produces a final result, intermediate pool fields stored in `_eval_buffer` are returned. The `store_last` / `last_out` field is NOT returned until overwritten by the next evaluation.

---

## 3. Return Discipline

### 3.1 Natural Return Boundaries

1. **Arithmetic intermediates:** Returned after `operate()` in `Future.evaluate()`.
2. **RHS results:** Returned after accumulation into stage vectors or history (within step functions).
3. **Operator/derivative results:** Returned when consumed by the expression tree evaluator.

### 3.2 Fields That Are Never Pooled

- Fields in `state.history` — long-lived user-visible state
- Fields created by user via `ScalarField(dist, ...)` directly (`_from_pool = false`)
- `workspace_fields` in `TimestepperState` — already pre-allocated separately

### 3.3 Debug Safety

Generation counter on pool fields: `_pool_generation` incremented on each checkout. Use-after-return detected via `@assert` (zero cost when disabled).

---

## 4. Vector Caching (`fields_to_vector` / `vector_to_fields`)

### 4.1 Current State

`fields_to_vector()` in `solvers.jl` has a partial cache keyed by vector size. `vector_to_fields()` always allocates new `ScalarField` objects.

### 4.2 Changes

**`fields_to_vector`:** Extend the existing cache to be per-solver (stored in `TimestepperState.timestepper_data`) rather than a global dict. Pre-allocate at solver init based on initial state size.

**`vector_to_fields`:** Add an in-place variant `vector_to_fields!(output_state, vector, template)` that writes into pre-existing fields (from pool or workspace) instead of allocating new ones. Timestepper step functions call the in-place variant.

---

## 5. FFT Plan Caching in Dealiasing

### 5.1 Current State

`PaddedDealiasingWorkspace` pre-allocates padded arrays but `evaluate_padded_multiply` still creates temporaries for `fft()` / `ifft()` results.

### 5.2 Changes

Add pre-allocated spectral-space buffers to `PaddedDealiasingWorkspace`:
- `spec1::Array{Complex{T}, N}` — for forward FFT result of field 1
- `spec2::Array{Complex{T}, N}` — for forward FFT result of field 2
- `spec_result::Array{Complex{T}, N}` — for truncated product result

Use `mul!(spec1, plan_forward, padded1)` (in-place FFTW) instead of `fft(padded1)` (allocating). Requires FFTW plans that support `mul!`, which the existing `AbstractFFTPlan`-typed plans already do.

---

## 6. VectorField Pooling

`VectorField` is a thin wrapper over component `ScalarField`s. Pooling the components via `FieldPool` is sufficient — no separate VectorField pool needed.

For VectorField construction in arithmetic (`add_vector_fields`, `scale_vector_field`, etc.): checkout component fields from pool, wrap in a new `VectorField` shell. The shell is lightweight (~80 bytes, no data arrays).

---

## 7. GPU Memory Pool

CUDA.jl has its own caching memory allocator (`CUDA.pool`). Tarang's `FieldPool` works transparently for GPU fields because:
- `checkout!` returns fields with GPU-allocated data (created via `zeros(arch, ...)`)
- `return!` doesn't free GPU memory — it keeps the field for reuse
- CUDA.jl's allocator handles the underlying device memory caching

No custom GPU memory management needed.

---

## 8. Pool Initialization & Solver Integration

### 8.1 Setup

In `step!(solver, dt)` (solvers.jl), before the first timestep:
```julia
pool = FieldPool(solver.problem.dist)
set_field_pool!(pool)
```

### 8.2 Teardown

After solver completes (or on error): `set_field_pool!(nothing)`.

### 8.3 Pre-warming (Optional)

For predictable allocation patterns, pre-allocate N fields per key at init:
```julia
prewarm!(pool, bases, dtype, count=8)
```

---

## 9. Files Modified

| File | Change |
|------|--------|
| `src/core/field_pool.jl` (NEW) | FieldPool struct, checkout!/return!/with_pool_field |
| `src/core/field.jl` | Add `_from_pool`, `_pool_generation` fields to ScalarField |
| `src/core/arithmetic.jl` | Use pool in combine_add/multiply/scale/etc. |
| `src/core/operators/evaluate.jl` | Use pool in evaluate_multiply/add/subtract |
| `src/core/operators/derivatives.jl` | Use pool in evaluate_differentiate/divergence |
| `src/core/future.jl` | Return pool intermediates after operate() |
| `src/core/timesteppers/state_utils.jl` | Pool-based evaluate_rhs, in-place vector_to_fields! |
| `src/core/solvers.jl` | Pool init/teardown, vector cache per solver |
| `src/core/nonlinear.jl` | Add spec buffers to PaddedDealiasingWorkspace, use mul! |
| `src/Tarang.jl` | Include field_pool.jl |

---

## 10. Testing Strategy

1. **Unit test (`test_field_pool.jl`):** checkout → modify → return → checkout verifies same memory reused. Generation counter catches use-after-return.
2. **Allocation test:** Run solver for 10 steps, verify `@allocated` is zero after step 2 for the step function.
3. **Regression:** All existing tests (`test_solvers.jl`, `test_operators_basic.jl`, `test_transforms.jl`, `test_arithmetic.jl`) must pass unchanged.
4. **GPU test:** Run diagonal IMEX on GPU (if available), verify pool works with CuArrays.

---

## 11. Expected Outcomes

- After 1-2 warmup timesteps, per-step allocations drop to near zero for field operations
- Pool stabilizes at ~10-20 fields for typical 3D Navier-Stokes (3 state fields, ~6 intermediates per RHS)
- No user-facing API changes — existing code works identically
- FFT dealiasing uses in-place transforms, eliminating ~100-300 MB/multiply of temporaries
- `vector_to_fields!` eliminates ~48 MB/call of field recreation

**Total estimated reduction:** 80-95% of per-timestep allocations across all timestepper methods.
