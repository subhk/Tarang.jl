---
title: Memory Footprint & Allocation Reduction
date: 2026-04-11
status: draft
---

# Memory Footprint & Allocation Reduction

## Goal

Reduce per-timestep heap allocations and GC pressure across all timestepper paths (RK, IMEX pencil, multistep) without changing the public API. Target: zero allocations in the wavenumber-loop inner kernel, and ≤ O(N_fields) allocations per timestep elsewhere.

## Scope

Three tiers of changes, ordered by impact. All changes are internal — no public API modifications.

## Tier 1 — Eliminate per-wavenumber allocations (Critical)

### Problem

In `_pencil_sbdf1_field!`, `_pencil_sbdf2_field!`, `_pencil_cnab1_field!`, `_pencil_cnab2_field!` (step_pencil_imex.jl), the inner wavenumber loop does:

```julia
data_new[ikx, :] .= factor \ rhs_buf   # allocates a new Vector per wavenumber
```

For a 64×64×32 grid with 3 fields, this is ~6000 small allocations per timestep. Additionally, `rhs_buf = Vector{CT}(undef, Nz)` is allocated per function call (N per timestep).

### Fix

1. Add `sol_buf::Vector` and `rhs_buf::Vector` workspace fields to `PencilLinearOperator`:

```julia
struct PencilLinearOperator{T}
    # ... existing fields ...
    _rhs_workspace::Vector{T}   # length Nz, pre-allocated
    _sol_workspace::Vector{T}   # length Nz, pre-allocated
end
```

Since coefficient data is Complex, these should be `Vector{Complex{T}}`. Alternatively, allocate them lazily on first use based on `eltype(data)`.

2. Replace `factor \ rhs_buf` with `ldiv!(sol_buf, factor, rhs_buf)` + `copyto!`:

```julia
# Before (allocates):
data_new[ikx, :] .= factor \ rhs_buf

# After (zero-alloc):
ldiv!(sol_buf, factor, rhs_buf)
data_new[ikx, :] .= sol_buf
```

3. Remove per-call `rhs_buf = Vector{CT}(undef, Nz)` — use the workspace from the operator.

**Files**: `step_pencil_imex.jl`, `pencil_operators.jl`
**Affected functions**: `_pencil_sbdf1_field!`, `_pencil_sbdf2_field!`, `_pencil_cnab1_field!`, `_pencil_cnab2_field!`

## Tier 2 — Eliminate per-timestep array copies (Important)

### 2a. Layout-aware `copy(field)`

**Problem**: `copy(field::ScalarField)` copies both grid and coeff arrays unconditionally. When a field is in `:c` layout, the grid array is stale — copying it wastes memory and time. Called via `copy_state` every timestep.

**Fix**: Only copy the live array:

```julia
function Base.copy(field::ScalarField)
    new_field = ScalarField(...)  # shell only, no data
    if field.current_layout == :c && get_coeff_data(field) !== nothing
        set_coeff_data!(new_field, copy(get_coeff_data(field)))
    elseif get_grid_data(field) !== nothing
        set_grid_data!(new_field, copy(get_grid_data(field)))
    end
    new_field.current_layout = field.current_layout
    return new_field
end
```

**Files**: `field_data.jl`

### 2b. Pencil IMEX: allocate shell instead of `copy(field)`

**Problem**: In all pencil step functions, `new_field = copy(field)` copies data that is immediately overwritten by the solve.

**Fix**: Allocate a field shell with fresh arrays (no data copy):

```julia
new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
new_field.current_layout = :c
```

**Files**: `step_pencil_imex.jl` — 4 step functions

### 2c. Pre-allocated `fields_to_vector` buffer

**Problem**: `fields_to_vector` allocates a new `Vector{ComplexF64}` on every call. Called 2-6 times per timestep in RK paths.

**Fix**: Add a pre-allocated vector buffer to `TimestepperState`:

```julia
mutable struct TimestepperState
    # ... existing fields ...
    _vec_workspace::Vector{ComplexF64}  # pre-sized to total state DOF
end
```

Add `fields_to_vector!(buf, state)` in-place variant. Use it in `step_rk_imex!` and `_step_explicit_rk_cpu!`.

**Files**: `state.jl`, `state_utils.jl` or `solver_compiled_rhs.jl`, `step_rk.jl`

### 2d. Symbol constants for cache keys

**Problem**: `_get_pencil_lhs_cache!` converts String→Symbol on every call, allocating for the `"_inv"` concatenation.

**Fix**: Use module-level Symbol constants:

```julia
const _SBDF1_LHS = :sbdf1_lhs
const _SBDF1_LHS_INV = :sbdf1_lhs_inv
const _SBDF2_LHS = :sbdf2_lhs
# etc.
```

Change `_get_pencil_lhs_cache!` to accept `(key::Symbol, inv_key::Symbol)` directly.

**Files**: `step_pencil_imex.jl`

## Tier 3 — Type stability and struct layout (Important)

### 3a. Cache PencilFFTPlan reference directly on Distributor

**Problem**: `_find_pencil_plan` does a linear scan of `dist.transforms::Vector{Any}` with runtime `isa` checks on every transform call.

**Fix**: Add a direct reference:

```julia
mutable struct Distributor
    # ... existing fields ...
    pencil_fft_plan::Union{Nothing, PencilFFTs.PencilFFTPlan}  # cached reference
end
```

Set it during `plan_transforms!`. `_find_pencil_plan` becomes a field access.

**Files**: `distributor.jl`, `transform_planning.jl`, `transform_legendre.jl`, `transform_types.jl`

### 3b. Typed TimestepperState workspace

**Problem**: `timestepper_data::Dict{Symbol, Any}` boxes every value. Accessed in every step function for cached factorizations, history, iteration counts.

**Fix**: Add typed fields for the most-accessed items directly to `TimestepperState`:

```julia
mutable struct TimestepperState
    # ... existing fields ...
    pencil_linear_operator::Union{Nothing, PencilLinearOperator}
    lhs_caches::Dict{Symbol, Dict{Tuple{Int,Int}, Any}}
    rhs_caches::Dict{Symbol, Dict{Tuple{Int,Int}, Any}}
    F_history::Vector{Vector{ScalarField}}
    iteration_count::Int
end
```

Keep `timestepper_data::Dict{Symbol, Any}` for extension/user data, but move hot-path items out.

**Files**: `state.jl`, `step_pencil_imex.jl`, `step_rk.jl`, `pencil_operators.jl`

### 3c. Concrete Layout struct

**Problem**: `Layout` has `dist::Any` and `Tuple{Vararg{Int}}` fields — type-unstable on every access.

**Fix**: Parameterize on dimension:

```julia
struct Layout{N}
    dist::Distributor
    local_shape::NTuple{N, Int}
    global_shape::NTuple{N, Int}
    pencil::Union{Nothing, PencilArrays.Pencil}
end
```

**Files**: `distributor.jl`, `field_types.jl`

## Out of Scope

- **FieldPool**: Disabled due to aliasing bugs, marginal impact (~6 allocs/step) vs Tier 1 (~6000 allocs/step). Not worth the complexity.
- **GPU-specific optimizations**: Already uses workspace fields and kernel launches. No major allocation issues on GPU path.
- **FFTW plan allocation**: One-time setup cost, not per-timestep.
- **Public API changes**: All fixes are internal.

## Testing Strategy

- Run existing test suite (`test/runtests.jl`) — no regressions.
- Add `@allocated` tests for:
  - `_pencil_sbdf1_field!` inner kernel: should be 0 allocations
  - `step_pencil_sbdf2!` one full step: should be ≤ O(N_fields)
  - `copy_state` with `:c` layout: should allocate ~half vs current
- Benchmark before/after with `@btime` on a 2D Rayleigh-Benard problem (64×64).

## Success Criteria

- Zero heap allocations inside the wavenumber loop (Tier 1)
- Per-timestep allocations reduced from O(N_fields × N_stages × N_arrays) to O(N_fields)
- No test regressions
- Measurable reduction in GC time on 100-step benchmark
