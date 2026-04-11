# Allocation Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-timestep memory allocations by introducing a FieldPool for ScalarField reuse, in-place vector conversion, and pre-allocated FFT dealiasing buffers.

**Architecture:** A global `FieldPool` provides checkout/return semantics for temporary ScalarFields. All hot-path code (arithmetic, operators, RHS evaluation, derivatives) checks out fields from the pool instead of allocating. A generation counter provides debug-mode safety against use-after-return. Separately, `fields_to_vector`/`vector_to_fields` get in-place variants, and `PaddedDealiasingWorkspace` gets pre-allocated spectral buffers.

**Tech Stack:** Julia, FFTW, PencilArrays, AbstractFFTs

**Spec:** `docs/superpowers/specs/2026-04-04-allocation-reduction-design.md`

---

### Task 1: Create FieldPool Core (`field_pool.jl`)

**Files:**
- Create: `src/core/field_pool.jl`
- Test: `test/test_field_pool.jl`

- [ ] **Step 1: Write the failing test for FieldPool checkout/return**

Create `test/test_field_pool.jl`:
```julia
using Test
using Tarang

@testset "FieldPool" begin
    dist = Distributor(1, dtype=Float64)
    bases = (RealFourier(16, (0.0, 2π)),)

    pool = Tarang.FieldPool(dist)

    @testset "checkout allocates on first call" begin
        field = Tarang.checkout!(pool, bases, Float64)
        @test field isa ScalarField
        @test field._from_pool == true
        @test field._pool_generation == 1
        @test get_grid_data(field) !== nothing
        Tarang.return!(pool, field)
    end

    @testset "return and re-checkout reuses memory" begin
        field1 = Tarang.checkout!(pool, bases, Float64)
        grid_ptr = pointer(get_grid_data(field1))
        Tarang.return!(pool, field1)

        field2 = Tarang.checkout!(pool, bases, Float64)
        @test pointer(get_grid_data(field2)) == grid_ptr  # same memory
        @test field2._pool_generation == 2  # incremented
        Tarang.return!(pool, field2)
    end

    @testset "non-pool fields are rejected by return!" begin
        user_field = ScalarField(dist, "user", bases, Float64)
        @test user_field._from_pool == false
        @test_throws ArgumentError Tarang.return!(pool, user_field)
    end

    @testset "with_pool_field auto-returns" begin
        Tarang.with_pool_field(pool, bases, Float64) do field
            @test field._from_pool == true
            fill!(get_grid_data(field), 1.0)
        end
        # Field should be back in pool
        @test pool.in_use == 0
    end

    @testset "global pool access" begin
        Tarang.set_field_pool!(pool)
        @test Tarang.get_field_pool() === pool
        Tarang.set_field_pool!(nothing)
        @test Tarang.get_field_pool() === nothing
    end

    @testset "prewarm fills pool" begin
        Tarang.prewarm!(pool, bases, Float64, 4)
        key = Tarang.PoolKey(hash(bases), Float64)
        @test length(pool.available[key]) == 4
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_field_pool"])'`
Expected: FAIL — `FieldPool` not defined, `_from_pool` field not found on ScalarField.

- [ ] **Step 3: Add pool fields to ScalarField**

In `src/core/field.jl`, add two fields to the `ScalarField` struct (after `fft_mode::Symbol`, before the inner constructors):
```julia
    # Pool management (zero overhead for non-pool fields)
    _from_pool::Bool
    _pool_generation::Int
```

Update inner constructor 1 (line ~110) to include the new fields:
```julia
        field = new{T, SerialFieldStorage}(dist, name, bases, domain, dtype, storage, layout, current_layout, initial_scales, :auto, false, 0)
```

Update inner constructor 2 (line ~126) to include the new fields:
```julia
        new{T, S}(dist, name, bases, domain, dtype, storage, layout, :g, initial_scales, :auto, false, 0)
```

- [ ] **Step 4: Implement FieldPool**

Create `src/core/field_pool.jl`:
```julia
"""
    FieldPool — Pre-allocated ScalarField recycling for zero-allocation hot paths.

Fields are checked out for temporary use and returned when done.
After warmup, no further allocations occur.
"""

struct PoolKey
    bases_hash::UInt64
    dtype::DataType
end

PoolKey(bases::Tuple{Vararg{Basis}}, dtype::DataType) = PoolKey(hash(bases), dtype)

mutable struct FieldPool
    available::Dict{PoolKey, Vector{ScalarField}}
    dist::Distributor
    in_use::Int
    max_per_key::Int

    function FieldPool(dist::Distributor; max_per_key::Int=32)
        new(Dict{PoolKey, Vector{ScalarField}}(), dist, 0, max_per_key)
    end
end

"""
    checkout!(pool, bases, dtype) -> ScalarField

Get a recycled field from the pool, or allocate a new one if none available.
"""
function checkout!(pool::FieldPool, bases::Tuple{Vararg{Basis}}, dtype::DataType)
    key = PoolKey(bases, dtype)
    stack = get!(Vector{ScalarField}, pool.available, key)

    if !isempty(stack)
        field = pop!(stack)
    else
        field = ScalarField(pool.dist, "_pool", bases, dtype)
        field._from_pool = true
    end

    field._pool_generation += 1
    field._from_pool = true
    pool.in_use += 1
    return field
end

"""
    return!(pool, field)

Return a pool field for reuse. Resets layout to :g. Does NOT zero data.
Only pool-tagged fields are accepted.
"""
function return!(pool::FieldPool, field::ScalarField)
    if !field._from_pool
        throw(ArgumentError("Cannot return a non-pool field to FieldPool"))
    end

    field.current_layout = :g
    field.name = "_pool"

    key = PoolKey(field.bases, field.dtype)
    stack = get!(Vector{ScalarField}, pool.available, key)
    if length(stack) < pool.max_per_key
        push!(stack, field)
    end
    pool.in_use -= 1
    return nothing
end

"""
    with_pool_field(f, pool, bases, dtype)

Checkout a field, call f(field), then return it. Guarantees return even on error.
"""
function with_pool_field(f::Function, pool::FieldPool, bases::Tuple{Vararg{Basis}}, dtype::DataType)
    field = checkout!(pool, bases, dtype)
    try
        return f(field)
    finally
        return!(pool, field)
    end
end

"""
    prewarm!(pool, bases, dtype, count)

Pre-allocate `count` fields for the given key. Useful at solver init.
"""
function prewarm!(pool::FieldPool, bases::Tuple{Vararg{Basis}}, dtype::DataType, count::Int=8)
    for _ in 1:count
        field = ScalarField(pool.dist, "_pool", bases, dtype)
        field._from_pool = true
        return!(pool, field)
    end
end

# ---------------------------------------------------------------------------
# Global pool access (matches get_global_workspace() pattern)
# ---------------------------------------------------------------------------

const _FIELD_POOL = Ref{Union{Nothing, FieldPool}}(nothing)

"""Get the active field pool, or nothing if no solver is running."""
@inline get_field_pool() = _FIELD_POOL[]

"""Set the active field pool. Call with nothing to clear."""
set_field_pool!(pool::Union{Nothing, FieldPool}) = (_FIELD_POOL[] = pool)

"""
    checkout_or_alloc(bases, dtype, dist) -> ScalarField

Convenience: checkout from global pool if active, otherwise allocate normally.
"""
@inline function checkout_or_alloc(bases::Tuple{Vararg{Basis}}, dtype::DataType, dist::Distributor)
    pool = get_field_pool()
    if pool !== nothing
        return checkout!(pool, bases, dtype)
    else
        return ScalarField(dist, "_tmp", bases, dtype)
    end
end

"""
    maybe_return!(field::ScalarField)

Return field to global pool if it's a pool field. No-op otherwise.
"""
@inline function maybe_return!(field::ScalarField)
    if field._from_pool
        pool = get_field_pool()
        if pool !== nothing
            return!(pool, field)
        end
    end
    return nothing
end
```

- [ ] **Step 5: Include field_pool.jl in Tarang.jl**

In `src/Tarang.jl`, add after `include("core/field.jl")` (line 79):
```julia
include("core/field_pool.jl")
```

- [ ] **Step 6: Run tests to verify pool works**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_field_pool"])'`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/core/field_pool.jl src/core/field.jl src/Tarang.jl test/test_field_pool.jl
git commit -m "feat: add FieldPool for zero-allocation ScalarField recycling"
```

---

### Task 2: Integrate FieldPool into Solver Lifecycle

**Files:**
- Modify: `src/core/solvers.jl` (step! function, ~line 581)

- [ ] **Step 1: Write failing test for pool lifecycle**

Add to `test/test_field_pool.jl`:
```julia
@testset "Solver pool lifecycle" begin
    # Verify pool is set during solver stepping
    @test Tarang.get_field_pool() === nothing  # no solver running

    dist = Distributor(1, dtype=Float64)
    bases = (RealFourier(8, (0.0, 2π)),)
    u = ScalarField(dist, "u", bases, Float64)
    fill!(get_grid_data(u), 1.0)

    problem = Tarang.IVP(dist, [u], [0 * u]; dtype=Float64)
    solver = Tarang.InitialValueSolver(problem, RK111(); dt=0.01)

    # After step, pool should have been active (and cleaned up)
    step!(solver, 0.01)
    # Pool is cleared after step
    @test Tarang.get_field_pool() === nothing
end
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — pool not set during step!.

- [ ] **Step 3: Add pool init/teardown to step!**

In `src/core/solvers.jl`, in the `step!(solver::InitialValueSolver, dt::Float64)` function (~line 581), wrap the existing body:

```julia
function step!(solver::InitialValueSolver, dt::Float64=solver.dt)
    # Initialize field pool for this step
    pool = get_field_pool()
    pool_owner = false
    if pool === nothing
        pool = FieldPool(solver.problem.dist)
        set_field_pool!(pool)
        pool_owner = true
    end

    try
        # ... existing step! body unchanged ...
    finally
        if pool_owner
            set_field_pool!(nothing)
        end
    end
end
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_field_pool"])'`
Expected: PASS.

- [ ] **Step 5: Run existing solver tests for regression**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_solvers"])'`
Expected: All existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/solvers.jl test/test_field_pool.jl
git commit -m "feat: activate FieldPool during solver step lifecycle"
```

---

### Task 3: Integrate Pool into Arithmetic Operations

**Files:**
- Modify: `src/core/arithmetic.jl` (~lines 61-64, 135-143, 281-296, 305-318)

- [ ] **Step 1: Write failing allocation test**

Add to `test/test_field_pool.jl`:
```julia
@testset "Arithmetic uses pool" begin
    dist = Distributor(1, dtype=Float64)
    bases = (RealFourier(8, (0.0, 2π)),)
    pool = FieldPool(dist)
    set_field_pool!(pool)

    a = ScalarField(dist, "a", bases, Float64)
    b = ScalarField(dist, "b", bases, Float64)
    fill!(get_grid_data(a), 1.0)
    fill!(get_grid_data(b), 2.0)

    # After warmup, arithmetic should not allocate
    result1 = a + b  # warmup: populates pool
    maybe_return!(result1)

    alloc = @allocated begin
        result2 = a + b
        maybe_return!(result2)
    end
    @test alloc == 0

    set_field_pool!(nothing)
end
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `@allocated` is nonzero because arithmetic still uses `ScalarField()` constructor.

- [ ] **Step 3: Replace field allocation in arithmetic with pool checkout**

In `src/core/arithmetic.jl`, replace `ScalarField(...)` allocations with `checkout_or_alloc(...)`:

**`constant_field_like`** (~line 286):
```julia
function constant_field_like(field::ScalarField, value::Number)
    const_field = checkout_or_alloc(field.bases, field.dtype, field.dist)
    if field.scales !== nothing
        preset_scales!(const_field, field.scales)
    end
    ensure_layout!(const_field, :g)
    if get_grid_data(const_field) !== nothing
        fill!(get_grid_data(const_field), convert(field.dtype, value))
    end
    return const_field
end
```

**`add_vector_fields`** (~line 305):
Replace `VectorField(a.dist, a.coordsys, _ARITH_TMP_NAME, a.bases, a.dtype)` construction so that each component is a pool field:
```julia
function add_vector_fields(a::VectorField, b::VectorField)
    result = VectorField(a.dist, a.coordsys, _ARITH_TMP_NAME, a.bases, a.dtype)
    for i in 1:length(a.components)
        result[i] = combine_add(a.components[i], b.components[i])
    end
    return result
end
```
(VectorField shell is lightweight; the ScalarField components are pooled via `combine_add`.)

**`scale_vector_field`** (~line 312):
```julia
function scale_vector_field(field::VectorField, scalar)
    result = VectorField(field.dist, field.coordsys, _ARITH_TMP_NAME, field.bases, field.dtype)
    for i in 1:length(field.components)
        result[i] = field.components[i] * scalar
    end
    return result
end
```
(Components pooled via `combine_multiply`.)

**Power** (~line 359): Replace `ScalarField(a.dist, ...)` with `checkout_or_alloc(a.bases, a.dtype, a.dist)`.

**Negate** (~line 418+): Same pattern — `checkout_or_alloc`.

**Subtract** (~line 486+): Same pattern — `checkout_or_alloc`.

**Divide** (~line 595+): Same pattern — `checkout_or_alloc`.

- [ ] **Step 4: Run arithmetic tests**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_arithmetic"])'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/arithmetic.jl test/test_field_pool.jl
git commit -m "feat: use FieldPool in arithmetic operations"
```

---

### Task 4: Integrate Pool into Operator Evaluation

**Files:**
- Modify: `src/core/operators/evaluate.jl` (~lines 137, 210, 244)
- Modify: `src/core/operators/derivatives.jl` (~lines 51, 115, 163, 258)

- [ ] **Step 1: Replace field allocations in evaluate.jl**

In `evaluate_multiply` helpers (`_multiply_result` etc.), replace:
```julia
result = ScalarField(right.dist, "_mul", right.bases, right.dtype)
```
with:
```julia
result = checkout_or_alloc(right.bases, right.dtype, right.dist)
```

Apply the same pattern to `evaluate_add` (line ~210) and `evaluate_subtract` (line ~244), replacing `"_add"` and `"_sub"` field creation with `checkout_or_alloc`.

- [ ] **Step 2: Replace field allocations in derivatives.jl**

In `evaluate_differentiate` (~line 163), replace:
```julia
result = copy(operand)
```
with:
```julia
result = checkout_or_alloc(operand.bases, operand.dtype, operand.dist)
copy_field_data!(result, operand)
result.current_layout = operand.current_layout
```

In `evaluate_divergence` (~line 51), same pattern for the initial `copy(operand.components[1])`.

- [ ] **Step 3: Run operator tests**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_operators_basic"])'`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/core/operators/evaluate.jl src/core/operators/derivatives.jl
git commit -m "feat: use FieldPool in operator and derivative evaluation"
```

---

### Task 5: Integrate Pool into RHS Evaluation

**Files:**
- Modify: `src/core/timesteppers/state_utils.jl` (~lines 266-295, `create_rhs_zero_field`)

- [ ] **Step 1: Replace create_rhs_zero_field with pool checkout**

In `create_rhs_zero_field` (~line 266), replace the field creation with pool checkout + zeroing:

```julia
function create_rhs_zero_field(template_field::ScalarField)
    field = checkout_or_alloc(template_field.bases, template_field.dtype, template_field.dist)
    field.current_layout = template_field.current_layout

    # Zero the data arrays
    gd = get_grid_data(field)
    if gd !== nothing
        fill!(gd, zero(eltype(gd)))
    end
    cd = get_coeff_data(field)
    if cd !== nothing
        fill!(cd, zero(eltype(cd)))
    end

    # Copy scale information from template
    if template_field.scales !== nothing
        field.scales = template_field.scales
    end

    return field
end
```

- [ ] **Step 2: Run solver tests**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_solvers"])'`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/core/timesteppers/state_utils.jl
git commit -m "feat: use FieldPool in RHS evaluation"
```

---

### Task 6: Return Pool Intermediates in Future Evaluation

**Files:**
- Modify: `src/core/future.jl` (~lines 218-243, `evaluate` function)

- [ ] **Step 1: Add intermediate return logic to evaluate()**

In `future.jl`, modify the `evaluate` function to return pool fields from `_eval_buffer` after `operate()`:

```julia
function evaluate(f::Future; id=nothing, force::Bool=true)
    state = future_state(f)

    if state.store_last && !force && id !== nothing && state.last_id == id
        return state.last_out
    end

    # Return previous last_out if it was a pool field (being replaced)
    if state.store_last && state.last_out isa ScalarField
        maybe_return!(state.last_out)
    end

    buf = state._eval_buffer
    args = state.args
    n = length(args)
    if length(buf) != n
        resize!(buf, n)
    end
    @inbounds for i in 1:n
        buf[i] = evaluate_operand(args[i]; id=id, force=force)
    end
    result = operate(f, buf)

    # Return intermediate pool fields that are not the result
    @inbounds for i in 1:n
        val = buf[i]
        if val isa ScalarField && val !== result
            maybe_return!(val)
        end
    end

    if state.store_last && id !== nothing
        state.last_id = id
        state.last_out = result
    end

    return result
end
```

- [ ] **Step 2: Run solver and arithmetic tests**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_solvers", "test_arithmetic"])'`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/core/future.jl
git commit -m "feat: return pool intermediates after Future evaluation"
```

---

### Task 7: In-Place `vector_to_fields!`

**Files:**
- Modify: `src/core/solvers.jl` (~lines 1018-1046)
- Modify: `src/core/timesteppers/step_rk.jl` (callers)
- Modify: `src/core/timesteppers/step_multistep.jl` (callers)

- [ ] **Step 1: Add vector_to_fields! in-place variant**

In `src/core/solvers.jl`, after the existing `vector_to_fields` function, add:

```julia
"""
    vector_to_fields!(output::Vector{<:ScalarField}, vector, template)

In-place variant: writes vector data into pre-existing output fields.
No field allocation. Output fields must already have coeff data allocated.
"""
function vector_to_fields!(output::Vector{<:ScalarField}, vector::AbstractVector{<:Number},
                           template::Vector{<:ScalarField})
    offset = 1
    for (i, field) in enumerate(template)
        coeff_data = get_coeff_data(output[i])
        if coeff_data === nothing
            continue
        end
        local_data = get_local_data(coeff_data)
        n = length(local_data)
        if n > 0 && offset <= length(vector)
            end_idx = min(offset + n - 1, length(vector))
            copyto!(local_data, 1, vector, offset, end_idx - offset + 1)
            offset = end_idx + 1
        end
        output[i].current_layout = :c
    end
    return output
end
```

- [ ] **Step 2: Use vector_to_fields! in step_rk.jl**

In `_step_explicit_rk_cpu!` (~line 293), replace:
```julia
new_state = vector_to_fields(Y_vec, current_state)
```
with:
```julia
new_state = copy_state(current_state)
vector_to_fields!(new_state, Y_vec, current_state)
```

In `step_rk_imex!` (~line 139), replace:
```julia
new_state = vector_to_fields(X_new_vec, current_state)
```
with:
```julia
new_state = copy_state(current_state)
vector_to_fields!(new_state, X_new_vec, current_state)
```

- [ ] **Step 3: Use vector_to_fields! in step_multistep.jl**

Apply the same pattern to all `vector_to_fields(X_new, current_state)` calls in CNAB1 (~line 91), CNAB2, SBDF1-4.

- [ ] **Step 4: Run solver tests**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_solvers"])'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/solvers.jl src/core/timesteppers/step_rk.jl src/core/timesteppers/step_multistep.jl
git commit -m "feat: add in-place vector_to_fields! to avoid field allocation"
```

---

### Task 8: Pre-Allocated FFT Buffers in Dealiasing

**Files:**
- Modify: `src/core/nonlinear.jl` (~lines 113-129, 359-429)

- [ ] **Step 1: Add spectral buffers to PaddedDealiasingWorkspace**

In `src/core/nonlinear.jl`, add three buffer fields to `PaddedDealiasingWorkspace` (~line 113):

```julia
mutable struct PaddedDealiasingWorkspace{T<:AbstractFloat, A<:AbstractArray{Complex{T}}}
    original_shape::Tuple{Vararg{Int}}
    padded_shape::Tuple{Vararg{Int}}
    fourier_dims::Vector{Int}

    # Pre-allocated padded arrays
    padded1::A
    padded2::A
    padded_product::A

    # Pre-allocated spectral buffers (original size) — avoids fft() allocation
    spec1::A
    spec2::A
    spec_result::A

    # FFT plans
    plan_forward::AbstractFFTPlan
    plan_backward::AbstractFFTPlan

    # Architecture
    arch::AbstractArchitecture
end
```

- [ ] **Step 2: Update workspace construction**

In `_get_padded_workspace!` (~line 208), add buffer allocation:

```julia
    spec1 = zeros(_arch, Complex{T}, orig_t...)
    spec2 = zeros(_arch, Complex{T}, orig_t...)
    spec_result = zeros(_arch, Complex{T}, orig_t...)

    ws = PaddedDealiasingWorkspace{T, typeof(padded1)}(
        orig_t, pad_t, fourier_dims,
        padded1, padded2, padded_product,
        spec1, spec2, spec_result,
        plan_forward, plan_backward, _arch
    )
```

- [ ] **Step 3: Use pre-allocated buffers in evaluate_padded_multiply**

In `evaluate_padded_multiply` (~line 359), replace allocating FFT calls with in-place operations:

Replace lines 379-380:
```julia
    spec1 = fft(Complex{T}.(raw1_ws), ws.fourier_dims)
    spec2 = fft(Complex{T}.(raw2_ws), ws.fourier_dims)
```
with:
```julia
    ws.spec1 .= Complex{T}.(raw1_ws)
    ws.spec2 .= Complex{T}.(raw2_ws)
    ws.spec1 .= fft(ws.spec1, ws.fourier_dims)
    ws.spec2 .= fft(ws.spec2, ws.fourier_dims)
```

Replace line 397:
```julia
    spec_result = zeros(ws.arch, Complex{T}, ws.original_shape...)
```
with:
```julia
    fill!(ws.spec_result, zero(Complex{T}))
```

Replace line 398 and 407:
```julia
    _truncate_spectral!(ws.spec_result, ws.padded_product, ...)
    grid_result = ifft(ws.spec_result, ws.fourier_dims)
```
with:
```julia
    _truncate_spectral!(ws.spec_result, ws.padded_product, ws.original_shape, ws.padded_shape, ws.fourier_dims)
    ws.spec_result .= ifft(ws.spec_result, ws.fourier_dims)
```

And update the result writing to use `ws.spec_result` instead of `grid_result`.

- [ ] **Step 4: Run nonlinear tests**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_nonlinear"])'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/nonlinear.jl
git commit -m "feat: pre-allocated spectral buffers in dealiasing workspace"
```

---

### Task 9: Full Integration Test

**Files:**
- Test: `test/test_field_pool.jl` (extend)

- [ ] **Step 1: Write zero-allocation integration test**

Add to `test/test_field_pool.jl`:
```julia
@testset "Zero allocation after warmup" begin
    dist = Distributor(1, dtype=Float64)
    bases = (RealFourier(8, (0.0, 2π)),)
    u = ScalarField(dist, "u", bases, Float64)
    fill!(get_grid_data(u), sin.(range(0, 2π, length=8)))

    problem = Tarang.IVP(dist, [u], [0 * u]; dtype=Float64)
    solver = Tarang.InitialValueSolver(problem, RK111(); dt=0.01)

    # Warmup steps
    step!(solver, 0.01)
    step!(solver, 0.01)

    # Measure allocation on subsequent steps
    alloc = @allocated for _ in 1:5
        step!(solver, 0.01)
    end

    # Allow small allocation for GC bookkeeping, but field allocations should be zero
    @test alloc < 1024  # less than 1 KB for 5 steps
end
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Pkg; Pkg.test(test_args=["test_field_pool", "test_solvers", "test_arithmetic", "test_operators_basic"])'`
Expected: ALL PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_field_pool.jl
git commit -m "test: zero-allocation integration test for FieldPool"
```

---

## File Structure Summary

| File | Action | Responsibility |
|------|--------|---------------|
| `src/core/field_pool.jl` | CREATE | FieldPool struct, checkout!/return!/with_pool_field, global access |
| `src/core/field.jl` | MODIFY | Add `_from_pool`, `_pool_generation` to ScalarField |
| `src/core/arithmetic.jl` | MODIFY | Use `checkout_or_alloc` in combine_add/multiply/scale/etc. |
| `src/core/operators/evaluate.jl` | MODIFY | Use `checkout_or_alloc` in evaluate_multiply/add/subtract |
| `src/core/operators/derivatives.jl` | MODIFY | Use `checkout_or_alloc` in evaluate_differentiate/divergence |
| `src/core/future.jl` | MODIFY | Return pool intermediates after operate() |
| `src/core/timesteppers/state_utils.jl` | MODIFY | Pool-based create_rhs_zero_field |
| `src/core/solvers.jl` | MODIFY | Pool init/teardown in step!, vector_to_fields! |
| `src/core/timesteppers/step_rk.jl` | MODIFY | Use vector_to_fields! |
| `src/core/timesteppers/step_multistep.jl` | MODIFY | Use vector_to_fields! |
| `src/core/nonlinear.jl` | MODIFY | Add spec buffers to PaddedDealiasingWorkspace |
| `src/Tarang.jl` | MODIFY | Include field_pool.jl |
| `test/test_field_pool.jl` | CREATE | Unit + integration + allocation tests |
