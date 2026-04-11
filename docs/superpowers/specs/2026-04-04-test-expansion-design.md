# Test Expansion Design: Regression, Type Stability, and Coverage Gaps

**Date:** 2026-04-04
**Scope:** ~130 new tests across 3 categories
**Trigger:** 21 performance fixes across 30 source files in the Tarang.jl codebase

## Context

Two rounds of performance and type-stability fixes modified 30 source files touching arithmetic, field operations, transforms, timesteppers, operators, solvers, boundary conditions, and stochastic forcing. The existing test suite (~2,500 tests) has no type stability tests and no regression tests for the specific patterns changed.

## Goals (prioritized)

1. **Regression safety net** — verify each performance fix preserved correctness
2. **Type stability coverage** — `@inferred` tests on hot-path functions to catch future regressions
3. **Coverage gap filling** — test newly introduced utilities and dispatch paths

## Organization

- **Regression tests** (~60): added to existing test files where the code they test lives
- **Type stability tests** (~40): new dedicated `test_type_stability.jl` file (cross-cutting concern)
- **Coverage gap tests** (~30): added to existing test files + 1 possible new file

The new `test_type_stability.jl` is added to `TEST_FILES` in `runtests.jl`.

---

## Section 1: Regression Tests (~60 tests)

### `test_arithmetic.jl` — `@testset "Performance regression"`

| Test | What it verifies |
|------|-----------------|
| Field `+` correctness | `(a + b)` grid data ≈ `a_data .+ b_data`, result is fresh field (not a copy of `a`) |
| Field `-` correctness | `(a - b)` grid data ≈ `a_data .- b_data` |
| Field `*` (scalar) correctness | `(a * 2.0)` grid data ≈ `2.0 .* a_data` |
| Field `*` (field) correctness | `(a * b)` grid data ≈ `a_data .* b_data` |
| Commutative `*` | `2.0 * a ≈ a * 2.0` |
| Static field name | Result of `a + b` has name `_FIELD_ARITH_TMP_NAME`, not interpolated string |
| `_local_data` PencilArray | Returns `parent(pencil_array)` |
| `_local_data` plain Array | Returns the array itself |
| `combine_add` all pairs | (SF,SF), (SF,Number), (Number,SF), (Number,Number), (VF,VF), (VF,Number), (Number,VF) |
| `combine_multiply` all pairs | (SF,SF), (SF,Number), (Number,SF), (Number,Number), (VF,Number), (Number,VF) |
| `combine_multiply` VF×VF throws | ArgumentError |

### `test_solvers.jl` — `@testset "fields_to_vector regression"`

| Test | What it verifies |
|------|-----------------|
| Returns `Vector{ComplexF64}` | Type of return value |
| Roundtrip fidelity | `vector_to_fields(fields_to_vector(state), state)` ≈ original coeff data |
| Empty fields | Returns empty vector for empty input |

### `test_phi_functions.jl` — `@testset "Identity caching and power reuse"`

| Test | What it verifies |
|------|-----------------|
| Identity cache hit | `_get_identity_matrix(n, T)` returns same object on second call (`===`) |
| Small-norm φ functions | `phi_functions_matrix(A, dt)` for small `dt*A` matches Taylor series |
| Moderate-norm φ functions | `phi_functions_matrix(A, dt)` for moderate norm: `φ₀ ≈ exp(dt*A)` |

### `test_nonlinear.jl` — `@testset "Cache key regression"`

| Test | What it verifies |
|------|-----------------|
| Padded workspace caching | Second call returns same workspace object |
| Tuple key type | Key in `pencil_transforms` is a Tuple, not a String |
| Static product field name | Result field name is `"_nl_product"` |

### `test_stochastic_forcing.jl` — `@testset "work computation regression"`

| Test | What it verifies |
|------|-----------------|
| `work_stratonovich` correct value | Matches manual `sum(real.((prev .+ sol) ./ 2 .* conj.(forcing))) * dt / area` |
| `work_ito` correct value | Matches manual computation with drift |
| `work_stratonovich` nothing prevsol | Returns `zero(T)` |
| `work_ito` scaling | Work scales linearly with `dt` |

### `test_boundary_conditions.jl` — `@testset "BC cache tuple keys"`

| Test | What it verifies |
|------|-----------------|
| Tuple key storage | After `update_time_dependent_bcs!`, cache has tuple keys |
| Tuple key retrieval | `get_current_bc_value` returns correct value via tuple key |
| Robin BC components | Alpha, beta, value stored with `(:alpha, :beta, :value)` symbol keys |
| Cache clear | `empty!(bc_cache)` works with tuple-keyed dict |

---

## Section 2: Type Stability Tests (~40 tests)

### New file: `test_type_stability.jl`

All tests use `@inferred` or `@inferred Union{...}` for legitimate small unions.

**Field data access:**
- `get_grid_data(field)` — `Union{Nothing, AbstractArray}`
- `get_coeff_data(field)` — `Union{Nothing, AbstractArray}`
- `_local_data(plain_array)` — concrete array type
- `_local_data(pencil_array)` — concrete array type

**Arithmetic dispatch (compile-time resolution):**
- `combine_add(::ScalarField, ::ScalarField)` → ScalarField
- `combine_add(::Number, ::Number)` → Number
- `combine_multiply(::ScalarField, ::Number)` → ScalarField
- `combine_multiply(::Number, ::Number)` → Number

**Operator evaluate dispatch:**
- `_eval_operand(::ScalarField, :g)` → ScalarField
- `_eval_operand(::Float64, :g)` → Float64
- `_negate_result(::Number, :g)` → Number
- `_multiply_result(::Number, ::Number, :g)` → Number
- `_add_result(::Number, ::Number, :g)` → Number
- `_subtract_result(::Number, ::Number, :g)` → Number
- `_divide_result(::Number, ::Number, :g)` → Number

**History helpers:**
- `_push_trim!(::Vector{Int}, ::Int, ::Int)` → Vector{Int}
- `_prepend_trim!(::Vector{Int}, ::Int, ::Int)` → Vector{Int}

**Transform dispatch:**
- `_find_pencil_plan(dist)` — `Union{Nothing, PencilFFTPlan}` (or nothing if no plans)
- `_apply_forward(data, ::FourierTransform)` → AbstractArray
- `_apply_backward(data, ::FourierTransform)` → AbstractArray
- `_apply_forward(data, ::Transform)` fallback → same data type

**Solver/system:**
- `invoke_constructor(::Type{CPU}, (), (;))` → CPU
- `get_subdata(coeff_system, sp, ss)` → SubArray

**Derivative cache:**
- `_get_cached_deriv_mult(basis, N, L, order)` → `Vector{ComplexF64}`

---

## Section 3: Coverage Gap Tests (~30 tests)

### `test_solvers.jl` — `@testset "History helpers"`

| Test | What it verifies |
|------|-----------------|
| `_push_trim!` basic | Appends and trims to max length |
| `_push_trim!` max=1 | Single-element replacement (RK pattern) |
| `_push_trim!` below max | Appends without trimming |
| `_push_trim!` empty start | Grows from empty vector |
| `_prepend_trim!` basic | Prepends and trims from back |
| `_prepend_trim!` ordering | Newest element at index 1 |
| `_prepend_trim!` max=2 | Keeps exactly 2 elements |

### `test_solvers.jl` — `@testset "CoeffSystem typed keys"`

| Test | What it verifies |
|------|-----------------|
| Construction | CoeffSystem builds from subproblems |
| Key type | `keys(system.views)` are `Tuple{Subproblem, Subsystem}` |
| `get_subdata` returns view | Returned value is a `SubArray` |
| Write-read roundtrip | Write to view, read from `system.data` |

### `test_operators_basic.jl` — `@testset "Operator evaluate dispatch"`

| Test | What it verifies |
|------|-----------------|
| `_negate_result` ScalarField | Negated grid data ≈ `-original` |
| `_negate_result` Number | Returns `-n` |
| `_multiply_result` Number×ScalarField | Scaled grid data |
| `_multiply_result` ScalarField×ScalarField | Element-wise product |
| `_multiply_result` Number×Number | Returns product |
| `_multiply_result` unsupported | Throws ArgumentError |
| `_add_result` ScalarField pair | Sum of grid data |
| `_subtract_result` ScalarField pair | Difference of grid data |
| `_divide_result` ScalarField/Number | Divided grid data |

### `test_transforms.jl` — `@testset "Transform dispatch helpers"`

| Test | What it verifies |
|------|-----------------|
| `_apply_forward` FourierTransform | Produces transformed data |
| `_apply_backward` FourierTransform | Produces inverse-transformed data |
| Forward-backward roundtrip | `_apply_backward(_apply_forward(data, t), t) ≈ data` |
| Fallback dispatch | Unknown transform type returns data unchanged |
| `_find_pencil_plan` no plans | Returns `nothing` |

### `test_chebyshev.jl` — `@testset "Derivative multiplier caching"`

| Test | What it verifies |
|------|-----------------|
| Correct multipliers | Wavenumber values match manual `(im * k)^order` |
| Cache hit | Second call returns identical object (`===`) |
| Different order | Different `order` arg returns different multiplier |
| Tuple key in dict | `basis.transforms` has key `(:deriv_mult, N, order)` |

### `test_solvers.jl` — `@testset "Multistep solver correctness"`

Note: CNAB/SBDF coefficients are local variables inside step functions, not directly testable.
Instead, verify solver correctness indirectly: a simple ODE `du/dt = -u` with known solution
`u(t) = u₀ * exp(-t)` should produce correct results with each multistep method, confirming
the tuple coefficients are computed correctly.

| Test | What it verifies |
|------|-----------------|
| CNAB1 solver step | After N steps of `du/dt = -u`, solution ≈ `exp(-N*dt)` within tolerance |
| SBDF1 solver step | Same ODE, verify convergence |
| CNAB2 solver step | Same ODE, verify higher-order accuracy vs CNAB1 |
| SBDF2 solver step | Same ODE, verify convergence |

---

## Files Modified

| File | Action |
|------|--------|
| `test/test_type_stability.jl` | **New** — ~40 type stability tests |
| `test/test_arithmetic.jl` | Add ~15 regression tests |
| `test/test_solvers.jl` | Add ~20 regression + coverage tests |
| `test/test_phi_functions.jl` | Add ~5 regression tests |
| `test/test_nonlinear.jl` | Add ~5 regression tests |
| `test/test_stochastic_forcing.jl` | Add ~5 regression tests |
| `test/test_boundary_conditions.jl` | Add ~5 regression tests |
| `test/test_operators_basic.jl` | Add ~10 coverage tests |
| `test/test_transforms.jl` | Add ~5 coverage tests |
| `test/test_chebyshev.jl` | Add ~5 coverage tests |
| `test/runtests.jl` | Add `"test_type_stability.jl"` to `TEST_FILES` |

## Not in Scope

- GPU-specific tests (require CUDA setup)
- MPI tests (require mpiexec)
- Convergence/accuracy tests for timesteppers (existing optional suite covers this)
- Benchmark/performance timing tests (would need BenchmarkTools infrastructure)
- Refactoring existing tests
