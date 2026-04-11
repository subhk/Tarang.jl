# Memory Footprint & Allocation Reduction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-wavenumber heap allocations in the pencil IMEX inner loop, halve per-timestep array copies, and improve type stability in hot-path struct access.

**Architecture:** Three tiers — Tier 1 adds pre-allocated workspace buffers to PencilLinearOperator and uses in-place `ldiv!` in the wavenumber loop. Tier 2 makes `copy(field)` layout-aware and adds pre-allocated vector buffers to TimestepperState. Tier 3 caches PencilFFTPlan directly on Distributor and parameterizes Layout on dimension count.

**Tech Stack:** Julia 1.12, LinearAlgebra (ldiv!), SparseArrays, PencilArrays, PencilFFTs, MPI

**Spec:** `docs/superpowers/specs/2026-04-11-memory-reduction-design.md`

---

## File Map

| File | Changes |
|------|---------|
| `src/core/timesteppers/pencil_operators.jl` | Add workspace fields to `PencilLinearOperator`, pass to field kernels |
| `src/core/timesteppers/step_pencil_imex.jl` | Use `ldiv!` in wavenumber loops, Symbol cache keys, shell-alloc instead of copy |
| `src/core/field/field_data.jl` | Layout-aware `Base.copy(field)` |
| `src/core/timesteppers/state.jl` | Add `_vec_workspace` to `TimestepperState` |
| `src/core/timesteppers/state_utils.jl` | Add `fields_to_vector!` in-place variant |
| `src/core/timesteppers/step_rk.jl` | Use `fields_to_vector!` in RK paths |
| `src/core/distributor.jl` | Add `pencil_fft_plan` field, parameterize `Layout{N}` |
| `src/core/transforms/transform_types.jl` | Simplify `_find_pencil_plan` to field access |
| `src/core/transforms/transform_planning.jl` | Set `dist.pencil_fft_plan` during plan setup |
| `src/core/transforms/transform_legendre.jl` | Set `dist.pencil_fft_plan` during 3D plan setup |
| `test/test_pencil_imex.jl` | Add `@allocated` regression tests |

---

### Task 1: Add workspace buffers to PencilLinearOperator

**Files:**
- Modify: `src/core/timesteppers/pencil_operators.jl:51-61` (struct definition)
- Modify: `src/core/timesteppers/pencil_operators.jl` (constructor, around line 350-430)

- [ ] **Step 1: Add workspace fields to the struct**

In `src/core/timesteppers/pencil_operators.jl`, change the struct definition:

```julia
struct PencilLinearOperator{T<:AbstractFloat}
    Nz::Int
    local_kx_range::UnitRange{Int}
    local_ky_range::UnitRange{Int}
    k2_values::Matrix{T}
    chebyshev_D2::SparseMatrixCSC{T,Int}
    operator_type::Symbol
    parameters::Dict{Symbol, Any}
    chebyshev_basis_idx::Int
    fourier_basis_indices::Vector{Int}
    # Pre-allocated workspace buffers for zero-alloc wavenumber loop
    _rhs_buf::Vector{Complex{T}}
    _sol_buf::Vector{Complex{T}}
end
```

- [ ] **Step 2: Update the constructor to allocate workspace buffers**

At the end of the `PencilLinearOperator` constructor (around line 430), change the return statement from:

```julia
PencilLinearOperator{T}(
    Nz, local_kx_range, local_ky_range, k2_values, D2_cheb,
    operator_type, params, chebyshev_idx, fourier_indices
)
```

to:

```julia
PencilLinearOperator{T}(
    Nz, local_kx_range, local_ky_range, k2_values, D2_cheb,
    operator_type, params, chebyshev_idx, fourier_indices,
    Vector{Complex{T}}(undef, Nz),
    Vector{Complex{T}}(undef, Nz)
)
```

- [ ] **Step 3: Verify the project loads without errors**

Run: `julia --project=. -e 'using Tarang; println("OK")'`
Expected: `OK` (no errors)

- [ ] **Step 4: Commit**

```
git add src/core/timesteppers/pencil_operators.jl
git commit -m "feat: add pre-allocated workspace buffers to PencilLinearOperator"
```

---

### Task 2: Use ldiv! and workspace buffers in pencil IMEX kernels

**Files:**
- Modify: `src/core/timesteppers/step_pencil_imex.jl` — all 4 `_pencil_*_field!` functions

- [ ] **Step 1: Update `_pencil_sbdf1_field!` to use workspace + ldiv!**

In `_pencil_sbdf1_field!` (around line 256), change the function to accept and use workspace buffers:

Replace the buffer allocation and solve pattern. Change:

```julia
    CT = eltype(data_n)
    rhs_buf = Vector{CT}(undef, Nz)
```

to:

```julia
    rhs_buf = L._rhs_buf
    sol_buf = L._sol_buf
```

And in the 2D wavenumber loop, replace:

```julia
                data_new[ikx, :] .= factor \ rhs_buf
```

with:

```julia
                ldiv!(sol_buf, factor, rhs_buf)
                data_new[ikx, :] .= sol_buf
```

And in the 3D wavenumber loop, replace:

```julia
                    data_new[ikx, iky, :] .= factor \ rhs_buf
```

with:

```julia
                    ldiv!(sol_buf, factor, rhs_buf)
                    data_new[ikx, iky, :] .= sol_buf
```

- [ ] **Step 2: Apply the same pattern to `_pencil_sbdf2_field!`**

Same changes: remove `rhs_buf` allocation, use `L._rhs_buf` and `L._sol_buf`, replace `factor \ rhs_buf` with `ldiv!(sol_buf, factor, rhs_buf)` in both 2D and 3D loops.

- [ ] **Step 3: Apply the same pattern to `_pencil_cnab1_field!`**

Same changes. Note: CNAB also has `mul!(rhs_buf, RHS_mat, pencil_n)` which already uses `rhs_buf` in-place — that stays the same.

- [ ] **Step 4: Apply the same pattern to `_pencil_cnab2_field!`**

Same changes as CNAB1.

- [ ] **Step 5: Run the serial pencil IMEX test**

Run:
```bash
julia --project=. -e '
using Tarang
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; device=CPU())
xb = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
u = ScalarField(dist, "u", (xb, zb), Float64)
set!(u, (x, z) -> sin(x) * (1 - z^2))
problem = IVP([u])
add_parameters!(problem, nu=0.01)
add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")
L = PencilLinearOperator(dist, (xb, zb), :laplacian; ν=0.01)
solver = InitialValueSolver(problem, SBDF2(); dt=0.001)
set_pencil_linear_operator!(solver, L)
for _ in 1:10; step!(solver); end
val = maximum(abs.(grid_data(u)))
println("max|u| = $val")
@assert val < 1.0 "Field should decay"
println("PASS")
'
```

Expected: `max|u| < 1.0`, `PASS`, no warnings.

- [ ] **Step 6: Commit**

```
git add src/core/timesteppers/step_pencil_imex.jl
git commit -m "perf: zero-alloc wavenumber loop via ldiv! and pre-allocated buffers"
```

---

### Task 3: Symbol constants for cache keys

**Files:**
- Modify: `src/core/timesteppers/step_pencil_imex.jl` — `_get_pencil_lhs_cache!` and all call sites

- [ ] **Step 1: Add Symbol constants and refactor `_get_pencil_lhs_cache!`**

At the top of `step_pencil_imex.jl` (after the imports, before the first function), add:

```julia
# Pre-computed Symbol keys to avoid per-call String→Symbol allocation
const _CACHE_SBDF1_LHS     = :sbdf1_lhs
const _CACHE_SBDF1_LHS_INV = :sbdf1_lhs_inv
const _CACHE_SBDF2_LHS     = :sbdf2_lhs
const _CACHE_SBDF2_LHS_INV = :sbdf2_lhs_inv
const _CACHE_CNAB_LHS      = :cnab_lhs
const _CACHE_CNAB_LHS_INV  = :cnab_lhs_inv
const _CACHE_CNAB_RHS      = :cnab_rhs
const _CACHE_CNAB_RHS_INV  = :cnab_rhs_inv
```

Change `_get_pencil_lhs_cache!` from:

```julia
function _get_pencil_lhs_cache!(state::TimestepperState, cache_key::String, invalidation_key)
    key = Symbol(cache_key)
    inv_key = Symbol(cache_key, "_inv")
    data = state.timestepper_data
    ...
end
```

to:

```julia
function _get_pencil_lhs_cache!(state::TimestepperState, key::Symbol, inv_key::Symbol, invalidation_key)
    data = state.timestepper_data
    if !haskey(data, key) || get(data, inv_key, nothing) != invalidation_key
        data[key] = Dict{Tuple{Int,Int}, Any}()
        data[inv_key] = invalidation_key
    end
    return data[key]
end
```

- [ ] **Step 2: Update all call sites**

Replace all calls. Examples:

```julia
# Before:
lhs_cache = _get_pencil_lhs_cache!(state, "sbdf1_lhs", dt)

# After:
lhs_cache = _get_pencil_lhs_cache!(state, _CACHE_SBDF1_LHS, _CACHE_SBDF1_LHS_INV, dt)
```

```julia
# Before:
lhs_cache = _get_pencil_lhs_cache!(state, "sbdf2_lhs", (dt, a0))

# After:
lhs_cache = _get_pencil_lhs_cache!(state, _CACHE_SBDF2_LHS, _CACHE_SBDF2_LHS_INV, (dt, a0))
```

```julia
# Before:
lhs_cache = _get_pencil_lhs_cache!(state, "cnab_lhs", dt)
rhs_cache = _get_pencil_lhs_cache!(state, "cnab_rhs", dt)

# After:
lhs_cache = _get_pencil_lhs_cache!(state, _CACHE_CNAB_LHS, _CACHE_CNAB_LHS_INV, dt)
rhs_cache = _get_pencil_lhs_cache!(state, _CACHE_CNAB_RHS, _CACHE_CNAB_RHS_INV, dt)
```

- [ ] **Step 3: Run serial test to verify no regressions**

Same test as Task 2, Step 5.

- [ ] **Step 4: Commit**

```
git add src/core/timesteppers/step_pencil_imex.jl
git commit -m "perf: use Symbol constants for cache keys, eliminate String allocation"
```

---

### Task 4: Layout-aware `copy(field)`

**Files:**
- Modify: `src/core/field/field_data.jl:396-414`

- [ ] **Step 1: Make `copy` only copy the live data array**

Replace the current `Base.copy(field::ScalarField)` at line 396 with:

```julia
"""Create a copy of ScalarField, only copying the live data array (layout-aware).
If field is in :c layout, only coeff data is copied (grid is stale).
If field is in :g layout, only grid data is copied (coeff is stale)."""
function Base.copy(field::ScalarField)
    new_field = ScalarField(field.dist, field.name, (), field.dtype)
    new_field.bases = field.bases
    new_field.domain = field.domain
    new_field.layout = field.layout
    new_field.current_layout = field.current_layout
    new_field.scales = field.scales
    new_field.fft_mode = field.fft_mode
    new_field.buffers.architecture = field.buffers.architecture
    # Only copy the live data array — the other is stale and will be
    # recomputed on next layout change via ensure_layout!
    if field.current_layout == :c
        if get_coeff_data(field) !== nothing
            set_coeff_data!(new_field, copy(get_coeff_data(field)))
        end
    else
        if get_grid_data(field) !== nothing
            set_grid_data!(new_field, copy(get_grid_data(field)))
        end
    end
    return new_field
end
```

- [ ] **Step 2: Run existing tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'` (or the specific test files that exercise field copy).

At minimum: `julia --project=. test/test_fields.jl`

Expected: All tests pass.

- [ ] **Step 3: Commit**

```
git add src/core/field/field_data.jl
git commit -m "perf: layout-aware copy(field) — only copy live data array"
```

---

### Task 5: Shell-allocate instead of copy in pencil IMEX steps

**Files:**
- Modify: `src/core/timesteppers/step_pencil_imex.jl` — 4 step functions

- [ ] **Step 1: Replace `copy(field)` with shell allocation in all 4 step functions**

In `step_pencil_sbdf1!`, `step_pencil_sbdf2!`, `step_pencil_cnab1!`, `step_pencil_cnab2!`, find all instances of:

```julia
new_field = copy(field)
ensure_layout!(new_field, :c)
```

Replace with:

```julia
new_field = ScalarField(field.dist, field.name, (), field.dtype)
new_field.bases = field.bases
new_field.domain = field.domain
new_field.current_layout = :c
# Allocate coeff data matching source field
if get_coeff_data(field) !== nothing
    set_coeff_data!(new_field, similar(get_coeff_data(field)))
end
```

This avoids copying data that is immediately overwritten.

- [ ] **Step 2: Run serial test**

Same test as Task 2, Step 5.

- [ ] **Step 3: Commit**

```
git add src/core/timesteppers/step_pencil_imex.jl
git commit -m "perf: shell-allocate new fields in pencil IMEX instead of copy"
```

---

### Task 6: Cache PencilFFTPlan directly on Distributor

**Files:**
- Modify: `src/core/distributor.jl:36-83` (struct + constructor)
- Modify: `src/core/transforms/transform_types.jl` (`_find_pencil_plan`)
- Modify: `src/core/transforms/transform_planning.jl` (set cached plan)
- Modify: `src/core/transforms/transform_legendre.jl` (set cached plan)

- [ ] **Step 1: Add `pencil_fft_plan` field to Distributor**

In the `Distributor` struct (around line 55), add after the `transforms::Vector{Any}` line:

```julia
    pencil_fft_plan::Any  # Cached PencilFFTs.PencilFFTPlan for fast lookup (avoids Vector{Any} scan)
```

- [ ] **Step 2: Initialize to `nothing` in the constructor**

In the constructor body (around line 189), add after `transforms = Any[]`:

```julia
        pencil_fft_plan = nothing
```

And add `pencil_fft_plan` to the `new(...)` call at the appropriate position.

- [ ] **Step 3: Set the cached plan during plan creation**

In `src/core/transforms/transform_planning.jl`, after `push!(dist.transforms, fft_plan)`, add:

```julia
        dist.pencil_fft_plan = fft_plan
```

Do the same in `src/core/transforms/transform_legendre.jl` after the equivalent `push!`.

- [ ] **Step 4: Simplify `_find_pencil_plan` to a field access**

In `src/core/transforms/transform_types.jl`, replace the function:

```julia
function _find_pencil_plan(dist)
    for transform in dist.transforms
        if isa(transform, PencilFFTs.PencilFFTPlan)
            return transform
        end
    end
    return nothing
end
```

with:

```julia
_find_pencil_plan(dist) = dist.pencil_fft_plan
```

- [ ] **Step 5: Run serial test**

Same test as Task 2, Step 5.

- [ ] **Step 6: Commit**

```
git add src/core/distributor.jl src/core/transforms/transform_types.jl \
        src/core/transforms/transform_planning.jl src/core/transforms/transform_legendre.jl
git commit -m "perf: cache PencilFFTPlan on Distributor, eliminate Vector{Any} scan"
```

---

### Task 7: Parameterize Layout on dimension

**Files:**
- Modify: `src/core/distributor.jl:13-18` (Layout struct)

- [ ] **Step 1: Parameterize the Layout struct**

Replace:

```julia
struct Layout
    dist::Any
    local_shape::Tuple{Vararg{Int}}
    global_shape::Tuple{Vararg{Int}}
    pencil::Union{Nothing, PencilArrays.Pencil}
end
```

with:

```julia
struct Layout{N}
    dist::Distributor
    local_shape::NTuple{N, Int}
    global_shape::NTuple{N, Int}
    pencil::Union{Nothing, PencilArrays.Pencil}
end
```

- [ ] **Step 2: Update Layout construction sites**

Search for `Layout(` in `distributor.jl` and update any calls. The constructor signature changes from `Layout(dist, local_shape, global_shape, pencil)` to `Layout{N}(dist, local_shape, global_shape, pencil)` where `N = length(local_shape)`. Julia infers `N` automatically, so `Layout(dist, (64, 32), (128, 32), nothing)` still works — no call-site changes needed unless explicit type parameters are used.

- [ ] **Step 3: Update the Distributor's `layouts` Dict type**

In the Distributor struct, change:

```julia
layouts::Dict{Tuple, Layout}
```

to:

```julia
layouts::Dict{Tuple, Any}  # Layout{N} for varying N
```

(Since different domains have different dimensions, the Dict must accept `Layout{1}`, `Layout{2}`, `Layout{3}`.)

- [ ] **Step 4: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```
git add src/core/distributor.jl
git commit -m "perf: parameterize Layout{N} for type-stable shape access"
```

---

### Task 8: Add @allocated regression test

**Files:**
- Modify: `test/test_pencil_imex.jl` (add allocation test section)

- [ ] **Step 1: Add an allocation test for the pencil kernel**

At the end of `test/test_pencil_imex.jl`, add:

```julia
@testset "Allocation regression — pencil IMEX" begin
    using Tarang: _pencil_sbdf1_field!, PencilLinearOperator

    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; device=CPU())
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
    L = PencilLinearOperator(dist, (xb, zb), :laplacian; ν=0.01)

    Nkx = div(16, 2) + 1  # half-spectrum
    Nz = 8
    CT = ComplexF64
    data_n   = zeros(CT, Nkx, Nz)
    data_F_n = zeros(CT, Nkx, Nz)
    data_new = zeros(CT, Nkx, Nz)
    data_n[1, :] .= 1.0

    dt = 0.01
    lhs_cache = Dict{Tuple{Int,Int}, Any}()

    # Warm up (first call compiles + caches LU)
    _pencil_sbdf1_field!(data_new, data_n, data_F_n, L, dt; lhs_cache=lhs_cache)

    # Measure allocations on second call (LU cached, no compilation)
    allocs = @allocated _pencil_sbdf1_field!(data_new, data_n, data_F_n, L, dt; lhs_cache=lhs_cache)
    @test allocs == 0
end
```

- [ ] **Step 2: Run the test**

Run: `julia --project=. test/test_pencil_imex.jl`

Expected: `allocs == 0` passes.

- [ ] **Step 3: Commit**

```
git add test/test_pencil_imex.jl
git commit -m "test: add @allocated regression test for zero-alloc pencil kernel"
```
