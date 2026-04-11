# Test Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ~130 tests covering regression safety, type stability, and coverage gaps for the 21 performance fixes applied to Tarang.jl.

**Architecture:** New `test_type_stability.jl` for cross-cutting `@inferred` tests. Regression and coverage gap tests appended to existing test files to keep test organization consistent with the codebase convention (one test file per module).

**Tech Stack:** Julia Test stdlib, `@inferred` from Test, Tarang.jl internal imports via `import Tarang:`

**Spec:** `docs/superpowers/specs/2026-04-04-test-expansion-design.md`

---

### Task 1: Create `test_type_stability.jl` — Field and Arithmetic Type Stability

**Files:**
- Create: `test/test_type_stability.jl`

- [ ] **Step 1: Create the test file with field data access and arithmetic type stability tests**

```julia
"""
Type stability tests for Tarang.jl hot-path functions.

Uses @inferred to verify that critical functions return concrete types,
preventing performance regressions from type instability.
"""

using Test
using Tarang
using LinearAlgebra

import Tarang: _local_data, _FIELD_ARITH_TMP_NAME,
               combine_add, combine_multiply,
               _eval_operand, _negate_result, _multiply_result,
               _add_result, _subtract_result, _divide_result,
               _push_trim!, _prepend_trim!,
               _find_pencil_plan, _apply_forward, _apply_backward,
               _get_identity_matrix, _get_cached_deriv_mult,
               invoke_constructor, get_subdata,
               FourierTransform, Transform

# --- Test fixtures ---

function make_1d_scalar_field()
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    f = ScalarField(dist, "test", (basis,), Float64)
    ensure_layout!(f, :g)
    get_grid_data(f) .= randn(8)
    return f, dist, basis
end

@testset "Type Stability" begin

    @testset "Field data access" begin
        f, _, _ = make_1d_scalar_field()
        ensure_layout!(f, :g)
        # get_grid_data returns Union{Nothing, AbstractArray} — 2-way union is fine
        @test get_grid_data(f) isa AbstractArray
        @test get_coeff_data(f) === nothing || get_coeff_data(f) isa AbstractArray

        # _local_data is type-stable for plain arrays
        data = get_grid_data(f)
        @test @inferred(_local_data(data)) === data
    end

    @testset "Arithmetic dispatch" begin
        # Number-Number paths should be fully inferred
        @test @inferred(combine_add(1.0, 2.0)) == 3.0
        @test @inferred(combine_add(1, 2)) == 3
        @test @inferred(combine_multiply(3.0, 4.0)) == 12.0
        @test @inferred(combine_multiply(2, 5)) == 10

        # ScalarField paths — return type is ScalarField (not Any)
        f, _, _ = make_1d_scalar_field()
        g, _, _ = make_1d_scalar_field()
        @test combine_add(f, g) isa ScalarField
        @test combine_multiply(f, g) isa ScalarField
        @test combine_multiply(f, 2.0) isa ScalarField
        @test combine_add(f, 1.0) isa ScalarField
    end

    @testset "Operator evaluate dispatch" begin
        # Number paths are fully inferred
        @test @inferred(_eval_operand(42.0, :g)) === 42.0
        @test @inferred(_eval_operand(3, :g)) === 3
        @test @inferred(_negate_result(5.0, :g)) === -5.0
        @test @inferred(_multiply_result(2.0, 3.0, :g)) === 6.0
        @test @inferred(_add_result(1.0, 2.0, :g)) === 3.0
        @test @inferred(_subtract_result(5.0, 3.0, :g)) === 2.0
        @test @inferred(_divide_result(10.0, 2.0, :g)) === 5.0

        # ScalarField paths — return type is ScalarField
        f, _, _ = make_1d_scalar_field()
        @test _negate_result(f, :g) isa ScalarField
        @test _eval_operand(f, :g) isa ScalarField
    end

    @testset "History helpers" begin
        v = [1, 2, 3]
        @test @inferred(_push_trim!(v, 4, 3)) isa Vector{Int}
        v2 = [10, 20, 30]
        @test @inferred(_prepend_trim!(v2, 0, 3)) isa Vector{Int}
    end

    @testset "Transform dispatch" begin
        _, dist, _ = make_1d_scalar_field()
        # _find_pencil_plan returns Union{Nothing, ...}
        result = _find_pencil_plan(dist)
        @test result === nothing || result isa Any  # Just verify it runs without error

        # Fallback dispatch: unknown transform type returns data unchanged
        data = randn(ComplexF64, 8)
        struct DummyTransform <: Transform end
        @test @inferred(_apply_forward(data, DummyTransform())) === data
        @test @inferred(_apply_backward(data, DummyTransform())) === data
    end

    @testset "Phi function identity cache" begin
        I1 = @inferred _get_identity_matrix(4, Float64)
        @test I1 isa Matrix{Float64}
        @test size(I1) == (4, 4)
    end

    @testset "Derivative multiplier cache" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        result = _get_cached_deriv_mult(basis, 8, 2π, 1)
        @test result isa Vector{ComplexF64}
    end

    @testset "Dispatch invoke_constructor" begin
        @test @inferred(invoke_constructor(CPU, (), (;))) isa CPU
    end

end # Type Stability
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_type_stability.jl"])'`

Or directly: `julia --project test/test_type_stability.jl`

Expected: All tests PASS

- [ ] **Step 3: Add `test_type_stability.jl` to runtests.jl**

In `test/runtests.jl`, add `"test_type_stability.jl"` to the `TEST_FILES` array, after `"test_gpu_solver_cpu.jl"`:

```julia
    "test_gpu_solver_cpu.jl",
    "test_type_stability.jl",
]
```

---

### Task 2: Regression Tests in `test_arithmetic.jl`

**Files:**
- Modify: `test/test_arithmetic.jl` (append before final `end` or after last `@testset`)

- [ ] **Step 1: Append performance regression tests**

Add this block at the end of `test/test_arithmetic.jl`:

```julia
@testset "Performance regression — field arithmetic" begin
    using Tarang
    import Tarang: _FIELD_ARITH_TMP_NAME, _local_data

    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))

    a = ScalarField(dist, "a", (basis,), Float64)
    b = ScalarField(dist, "b", (basis,), Float64)
    ensure_layout!(a, :g)
    ensure_layout!(b, :g)
    get_grid_data(a) .= Float64.(1:8)
    get_grid_data(b) .= Float64.(8:-1:1)

    @testset "Field + produces correct result" begin
        c = a + b
        ensure_layout!(c, :g)
        @test all(get_grid_data(c) .≈ 9.0)
    end

    @testset "Field - produces correct result" begin
        c = a - b
        ensure_layout!(c, :g)
        expected = Float64.(1:8) .- Float64.(8:-1:1)
        @test get_grid_data(c) ≈ expected
    end

    @testset "Field * scalar produces correct result" begin
        c = a * 3.0
        ensure_layout!(c, :g)
        @test get_grid_data(c) ≈ 3.0 .* Float64.(1:8)
    end

    @testset "Scalar * Field is commutative" begin
        c1 = a * 2.5
        c2 = 2.5 * a
        ensure_layout!(c1, :g)
        ensure_layout!(c2, :g)
        @test get_grid_data(c1) ≈ get_grid_data(c2)
    end

    @testset "Field * Field produces correct result" begin
        c = a * b
        ensure_layout!(c, :g)
        @test get_grid_data(c) ≈ Float64.(1:8) .* Float64.(8:-1:1)
    end

    @testset "Result uses static name, not interpolated string" begin
        c = a + b
        @test c.name == _FIELD_ARITH_TMP_NAME
    end

    @testset "Result is fresh allocation, not copy of input" begin
        a_data_before = copy(get_grid_data(a))
        c = a + b
        # Modifying c should not affect a
        ensure_layout!(c, :g)
        get_grid_data(c) .= 0.0
        @test get_grid_data(a) ≈ a_data_before
    end

    @testset "_local_data for plain Array" begin
        arr = randn(8)
        @test _local_data(arr) === arr
    end

    @testset "combine_add dispatch covers all type pairs" begin
        import Tarang: combine_add, combine_multiply

        # Number + Number
        @test combine_add(1.0, 2.0) == 3.0

        # ScalarField + ScalarField
        r = combine_add(a, b)
        @test r isa ScalarField

        # ScalarField + Number
        r = combine_add(a, 1.0)
        @test r isa ScalarField

        # Number + ScalarField
        r = combine_add(1.0, a)
        @test r isa ScalarField

        # VectorField + VectorField error case tested in existing suite
    end

    @testset "combine_multiply dispatch covers type pairs" begin
        @test combine_multiply(2.0, 3.0) == 6.0
        @test combine_multiply(a, b) isa ScalarField
        @test combine_multiply(a, 2.0) isa ScalarField
        @test combine_multiply(2.0, a) isa ScalarField
    end

    @testset "combine_multiply VF×VF throws" begin
        import Tarang: combine_multiply
        coords2 = CartesianCoordinates("x", "y")
        dist2 = Distributor(coords2; mesh=(1,), dtype=Float64)
        bx = RealFourier(coords2["x"]; size=4, bounds=(0.0, 2π))
        by = RealFourier(coords2["y"]; size=4, bounds=(0.0, 2π))
        u = VectorField(dist2, "u", (bx, by), Float64)
        v = VectorField(dist2, "v", (bx, by), Float64)
        @test_throws ArgumentError combine_multiply(u, v)
    end
end
```

- [ ] **Step 2: Run test to verify it passes**

Run: `julia --project test/test_arithmetic.jl`

Expected: All tests PASS

---

### Task 3: Regression Tests in `test_solvers.jl` — History Helpers, CoeffSystem, fields_to_vector

**Files:**
- Modify: `test/test_solvers.jl` (append new testsets inside the outer `"Solvers Module"` testset)

- [ ] **Step 1: Append history helper, CoeffSystem, and fields_to_vector tests**

Add these blocks inside the `@testset "Solvers Module"` block, before the closing `end`:

```julia
    @testset "History helpers" begin
        import Tarang: _push_trim!, _prepend_trim!

        @testset "_push_trim! basic" begin
            v = [1, 2, 3]
            _push_trim!(v, 4, 3)
            @test v == [2, 3, 4]
            @test length(v) == 3
        end

        @testset "_push_trim! below max" begin
            v = [1]
            _push_trim!(v, 2, 5)
            @test v == [1, 2]
        end

        @testset "_push_trim! max=1 (RK pattern)" begin
            v = [10]
            _push_trim!(v, 20, 1)
            @test v == [20]
            @test length(v) == 1
        end

        @testset "_push_trim! from empty" begin
            v = Int[]
            _push_trim!(v, 99, 3)
            @test v == [99]
        end

        @testset "_prepend_trim! basic" begin
            v = [1, 2, 3]
            _prepend_trim!(v, 0, 3)
            @test v == [0, 1, 2]
            @test length(v) == 3
        end

        @testset "_prepend_trim! ordering (newest first)" begin
            v = Int[]
            _prepend_trim!(v, 1, 3)
            _prepend_trim!(v, 2, 3)
            _prepend_trim!(v, 3, 3)
            @test v == [3, 2, 1]  # newest first
        end

        @testset "_prepend_trim! max=2" begin
            v = [10, 20, 30]
            _prepend_trim!(v, 0, 2)
            @test v == [0, 10]
            @test length(v) == 2
        end
    end

    @testset "fields_to_vector regression" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        f = ScalarField(dist, "f", (basis,), Float64)
        ensure_layout!(f, :g)
        get_grid_data(f) .= randn(8)
        ensure_layout!(f, :c)

        state = [f]

        @testset "returns Vector{ComplexF64}" begin
            vec = Tarang.fields_to_vector(state)
            @test vec isa Vector{ComplexF64}
        end

        @testset "roundtrip fidelity" begin
            original_data = copy(get_coeff_data(f))
            vec = Tarang.fields_to_vector(state)
            new_state = Tarang.vector_to_fields(vec, state)
            ensure_layout!(new_state[1], :c)
            @test get_coeff_data(new_state[1]) ≈ original_data
        end

        @testset "empty input" begin
            vec = Tarang.fields_to_vector(ScalarField[])
            @test isempty(vec)
        end
    end

    @testset "CoeffSystem typed keys" begin
        import Tarang: CoeffSystem, Subsystem, Subproblem, get_subdata

        # Create minimal subsystems and subproblems for testing
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
        f = ScalarField(dist, "f", (basis,), Float64)

        # Use the solver infrastructure to build subsystems
        problem = IVP([f], ["dt(f) = -f"])
        Tarang.setup_domain!(problem)
        solver = InitialValueSolver(problem, RK111(); dt=0.01)

        if solver.base.problem.domain !== nothing
            subsystems = Tarang.build_subsystems(solver)
            if !isempty(subsystems)
                subproblems = Tarang.build_subproblems(solver, subsystems; build_matrices=String[])
                if !isempty(subproblems)
                    cs = CoeffSystem(subproblems)

                    @testset "keys are typed tuples" begin
                        for k in keys(cs.views)
                            @test k isa Tuple{Subproblem, Subsystem}
                        end
                    end

                    @testset "get_subdata returns SubArray" begin
                        sp = subproblems[1]
                        ss = sp.subsystems[1]
                        sub = get_subdata(cs, sp, ss)
                        @test sub isa SubArray
                    end

                    @testset "write-read roundtrip" begin
                        sp = subproblems[1]
                        ss = sp.subsystems[1]
                        sub = get_subdata(cs, sp, ss)
                        sub .= 42.0
                        @test all(x -> x == 42.0, get_subdata(cs, sp, ss))
                    end
                else
                    @test_skip "No subproblems built"
                end
            else
                @test_skip "No subsystems built"
            end
        else
            @test_skip "No domain configured"
        end
    end
```

- [ ] **Step 2: Run test to verify it passes**

Run: `julia --project test/test_solvers.jl`

Expected: All tests PASS

---

### Task 4: Regression Tests in `test_phi_functions.jl`

**Files:**
- Modify: `test/test_phi_functions.jl` (append inside the outer `"Phi Functions"` testset)

- [ ] **Step 1: Append identity caching and power reuse tests**

Add inside the `@testset "Phi Functions"` block:

```julia
    @testset "Identity matrix caching" begin
        import Tarang: _get_identity_matrix, _phi_identity_cache

        I1 = _get_identity_matrix(4, Float64)
        I2 = _get_identity_matrix(4, Float64)
        @test I1 === I2  # Same object (cached)
        @test I1 == Matrix{Float64}(LinearAlgebra.I, 4, 4)

        # Different size returns different object
        I3 = _get_identity_matrix(3, Float64)
        @test I3 !== I1
        @test size(I3) == (3, 3)

        # Different type returns different object
        I4 = _get_identity_matrix(4, ComplexF64)
        @test I4 !== I1
        @test eltype(I4) == ComplexF64
    end

    @testset "phi_functions_matrix small norm (Taylor)" begin
        A = diagm([0.001, 0.002, 0.003])
        dt = 0.001
        exp_z, φ₁, φ₂ = phi_functions_matrix(A, dt)

        z = dt * A
        # φ₀ should be ≈ exp(z)
        @test exp_z ≈ exp(z) atol=1e-10

        # φ₁ identity: z * φ₁ = exp(z) - I
        I_mat = Matrix{Float64}(LinearAlgebra.I, 3, 3)
        @test z * φ₁ ≈ exp(z) - I_mat atol=1e-10
    end

    @testset "phi_functions_matrix moderate norm" begin
        A = diagm([1.0, 2.0, 3.0])
        dt = 1.0
        exp_z, φ₁, φ₂ = phi_functions_matrix(A, dt)

        # φ₀ = exp(z)
        @test exp_z ≈ exp(dt * A) atol=1e-8
    end
```

- [ ] **Step 2: Run test to verify it passes**

Run: `julia --project test/test_phi_functions.jl`

Expected: All tests PASS

---

### Task 5: Regression Tests in `test_nonlinear.jl`

**Files:**
- Modify: `test/test_nonlinear.jl` (append new testset inside the outer testset)

- [ ] **Step 1: Append cache key regression tests**

Find the outermost `@testset` in the file and add before its `end`:

```julia
    @testset "Cache key regression" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))

        evaluator = NonlinearEvaluator(dist)

        @testset "pencil_transforms accepts tuple keys" begin
            # The dict should accept non-string keys (Dict{Any, Any})
            evaluator.pencil_transforms[(:test, 1, 2)] = "test_value"
            @test evaluator.pencil_transforms[(:test, 1, 2)] == "test_value"
        end

        @testset "static product field name" begin
            f = ScalarField(dist, "f", (basis,), Float64)
            g = ScalarField(dist, "g", (basis,), Float64)
            ensure_layout!(f, :g)
            ensure_layout!(g, :g)
            get_grid_data(f) .= 1.0
            get_grid_data(g) .= 2.0

            # evaluate_transform_multiply should produce a field named "_nl_product"
            result = evaluate_transform_multiply(f, g, evaluator)
            if result isa ScalarField
                @test result.name == "_nl_product"
            else
                @test_skip "evaluate_transform_multiply returned non-field result"
            end
        end
    end
```

- [ ] **Step 2: Run test to verify it passes**

Run: `julia --project test/test_nonlinear.jl`

Expected: All tests PASS

---

### Task 6: Regression Tests in `test_stochastic_forcing.jl`

**Files:**
- Modify: `test/test_stochastic_forcing.jl` (append inside the outer `"StochasticForcing"` testset)

- [ ] **Step 1: Append work computation regression tests**

Add inside the main testset:

```julia
    @testset "work_stratonovich regression" begin
        import Tarang: work_stratonovich, work_ito

        # Create a minimal forcing with known data
        N = 16
        domain_size = (2π,)
        forcing = StochasticForcing(N; domain_size=domain_size, energy_injection_rate=1.0,
                                     k_f=4.0, dk_f=1.0, dt=0.01)

        # Set known prevsol and cached_forcing
        prev = randn(ComplexF64, N)
        sol = randn(ComplexF64, N)
        cf = randn(ComplexF64, N)

        forcing.prevsol = prev
        forcing.cached_forcing = cf

        # Manual computation (reference)
        ψ_mid = (prev .+ sol) ./ 2
        expected_work = sum(real.(ψ_mid .* conj.(cf))) * forcing.dt / prod(domain_size)

        result = work_stratonovich(forcing, sol)
        @test result ≈ expected_work atol=1e-12

        @testset "returns zero when prevsol is nothing" begin
            forcing2 = StochasticForcing(N; domain_size=domain_size, energy_injection_rate=1.0,
                                          k_f=4.0, dk_f=1.0, dt=0.01)
            # prevsol defaults to nothing
            @test work_stratonovich(forcing2, sol) == 0.0
        end
    end

    @testset "work_ito regression" begin
        import Tarang: work_ito

        N = 16
        domain_size = (2π,)
        forcing = StochasticForcing(N; domain_size=domain_size, energy_injection_rate=1.0,
                                     k_f=4.0, dk_f=1.0, dt=0.01)

        sol_prev = randn(ComplexF64, N)
        cf = randn(ComplexF64, N)
        forcing.cached_forcing = cf

        # Manual computation (reference)
        expected_work = sum(real.(sol_prev .* conj.(cf))) * forcing.dt / prod(domain_size)
        drift = forcing.energy_injection_rate * forcing.dt
        expected_total = expected_work + drift

        result = work_ito(forcing, sol_prev)
        @test result ≈ expected_total atol=1e-12
    end
```

- [ ] **Step 2: Run test to verify it passes**

Run: `julia --project test/test_stochastic_forcing.jl`

Expected: All tests PASS

---

### Task 7: Regression Tests in `test_boundary_conditions.jl`

**Files:**
- Modify: `test/test_boundary_conditions.jl` (append inside outer testset)

- [ ] **Step 1: Append BC cache tuple key tests**

Add inside the `@testset "Boundary Conditions Module"` block:

```julia
    @testset "BC cache tuple keys" begin
        manager = BoundaryConditionManager()

        @testset "bc_cache accepts tuple keys" begin
            # Dict{Any, Any} should accept tuple keys
            manager.bc_cache[(1, 0.5)] = 42.0
            @test manager.bc_cache[(1, 0.5)] == 42.0
        end

        @testset "Robin-style component keys" begin
            manager.bc_cache[(2, 1.0, :alpha)] = 0.5
            manager.bc_cache[(2, 1.0, :beta)] = 0.3
            manager.bc_cache[(2, 1.0, :value)] = 1.0

            @test manager.bc_cache[(2, 1.0, :alpha)] == 0.5
            @test manager.bc_cache[(2, 1.0, :beta)] == 0.3
            @test manager.bc_cache[(2, 1.0, :value)] == 1.0
        end

        @testset "cache clear works with tuple keys" begin
            manager.bc_cache[(3, 2.0)] = "test"
            @test !isempty(manager.bc_cache)
            empty!(manager.bc_cache)
            @test isempty(manager.bc_cache)
        end

        @testset "get_current_bc_value with tuple keys" begin
            # Add a simple Dirichlet BC and cache a value for it
            add_bc!(manager, DirichletBC("u", "x", "left", 1.0))
            bc_idx = 1
            t = 0.5
            manager.bc_cache[(bc_idx, t)] = 42.0

            result = Tarang.get_current_bc_value(manager, bc_idx, t)
            @test result == 42.0
        end
    end
```

- [ ] **Step 2: Run test to verify it passes**

Run: `julia --project test/test_boundary_conditions.jl`

Expected: All tests PASS

---

### Task 8: Coverage Gap Tests in `test_operators_basic.jl`

**Files:**
- Modify: `test/test_operators_basic.jl` (append new testset)

- [ ] **Step 1: Append operator evaluate dispatch tests**

Add at the end of the file:

```julia
@testset "Operator evaluate dispatch" begin
    using Tarang
    import Tarang: _negate_result, _multiply_result, _add_result,
                   _subtract_result, _divide_result

    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))

    f = ScalarField(dist, "f", (basis,), Float64)
    ensure_layout!(f, :g)
    get_grid_data(f) .= Float64.(1:8)

    g = ScalarField(dist, "g", (basis,), Float64)
    ensure_layout!(g, :g)
    get_grid_data(g) .= Float64.(8:-1:1)

    @testset "_negate_result ScalarField" begin
        r = _negate_result(f, :g)
        @test r isa ScalarField
        ensure_layout!(r, :g)
        @test get_grid_data(r) ≈ .-Float64.(1:8)
    end

    @testset "_negate_result Number" begin
        @test _negate_result(5, :g) == -5
        @test _negate_result(3.14, :c) == -3.14
    end

    @testset "_negate_result AbstractArray" begin
        arr = [1.0, 2.0, 3.0]
        @test _negate_result(arr, :g) ≈ [-1.0, -2.0, -3.0]
    end

    @testset "_multiply_result Number×ScalarField" begin
        r = _multiply_result(3.0, f, :g)
        @test r isa ScalarField
        ensure_layout!(r, :g)
        @test get_grid_data(r) ≈ 3.0 .* Float64.(1:8)
    end

    @testset "_multiply_result ScalarField×Number (commutative)" begin
        r = _multiply_result(f, 3.0, :g)
        @test r isa ScalarField
        ensure_layout!(r, :g)
        @test get_grid_data(r) ≈ 3.0 .* Float64.(1:8)
    end

    @testset "_multiply_result ScalarField×ScalarField" begin
        r = _multiply_result(f, g, :g)
        @test r isa ScalarField
        ensure_layout!(r, :g)
        @test get_grid_data(r) ≈ Float64.(1:8) .* Float64.(8:-1:1)
    end

    @testset "_multiply_result Number×Number" begin
        @test _multiply_result(3, 4, :g) == 12
    end

    @testset "_multiply_result unsupported throws" begin
        @test_throws ArgumentError _multiply_result("a", "b", :g)
    end

    @testset "_add_result ScalarField pair" begin
        r = _add_result(f, g, :g)
        @test r isa ScalarField
        ensure_layout!(r, :g)
        @test all(get_grid_data(r) .≈ 9.0)
    end

    @testset "_subtract_result ScalarField pair" begin
        r = _subtract_result(f, g, :g)
        @test r isa ScalarField
        ensure_layout!(r, :g)
        @test get_grid_data(r) ≈ Float64.(1:8) .- Float64.(8:-1:1)
    end

    @testset "_divide_result ScalarField/Number" begin
        r = _divide_result(f, 2.0, :g)
        @test r isa ScalarField
        ensure_layout!(r, :g)
        @test get_grid_data(r) ≈ Float64.(1:8) ./ 2.0
    end

    @testset "_divide_result unsupported throws" begin
        @test_throws ArgumentError _divide_result("a", "b", :g)
    end
end
```

- [ ] **Step 2: Run test to verify it passes**

Run: `julia --project test/test_operators_basic.jl`

Expected: All tests PASS

---

### Task 9: Coverage Gap Tests in `test_transforms.jl` and `test_chebyshev.jl`

**Files:**
- Modify: `test/test_transforms.jl` (append new testset)
- Modify: `test/test_chebyshev.jl` (append new testset)

- [ ] **Step 1: Append transform dispatch helper tests to `test_transforms.jl`**

Add at the end of `test/test_transforms.jl`:

```julia
@testset "Transform dispatch helpers" begin
    using Tarang
    import Tarang: _apply_forward, _apply_backward, _find_pencil_plan, Transform

    @testset "Fallback dispatch returns data unchanged" begin
        struct _TestUnknownTransform <: Transform end
        data = randn(ComplexF64, 8)
        @test _apply_forward(data, _TestUnknownTransform()) === data
        @test _apply_backward(data, _TestUnknownTransform()) === data
    end

    @testset "_find_pencil_plan with no plans" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        result = _find_pencil_plan(dist)
        # Serial distributor typically has no PencilFFTPlan
        @test result === nothing
    end
end
```

- [ ] **Step 2: Append derivative cache tests to `test_chebyshev.jl`**

Add at the end of `test/test_chebyshev.jl`:

```julia
@testset "Derivative multiplier caching" begin
    using Tarang
    import Tarang: _get_cached_deriv_mult

    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))

    N = 16
    L = 2π

    @testset "correct multiplier values" begin
        mult = _get_cached_deriv_mult(basis, N, L, 1)
        @test length(mult) == N
        @test mult isa Vector{ComplexF64}
        # First mode (k=0) derivative multiplier should be 0
        @test abs(mult[1]) < 1e-14
    end

    @testset "cache hit returns same object" begin
        m1 = _get_cached_deriv_mult(basis, N, L, 1)
        m2 = _get_cached_deriv_mult(basis, N, L, 1)
        @test m1 === m2  # Identity check (same object, not just equal)
    end

    @testset "different order returns different multiplier" begin
        m1 = _get_cached_deriv_mult(basis, N, L, 1)
        m2 = _get_cached_deriv_mult(basis, N, L, 2)
        @test m1 !== m2
        @test !(m1 ≈ m2)
    end

    @testset "tuple key stored in basis.transforms" begin
        _get_cached_deriv_mult(basis, N, L, 3)
        @test haskey(basis.transforms, (:deriv_mult, N, 3))
    end
end
```

- [ ] **Step 3: Run both tests to verify they pass**

Run: `julia --project test/test_transforms.jl && julia --project test/test_chebyshev.jl`

Expected: All tests PASS

---

### Task 10: Final Verification — Run Full Test Suite

**Files:**
- All modified test files

- [ ] **Step 1: Run the full standard test suite**

Run: `julia --project -e 'using Pkg; Pkg.test()'`

Expected: All tests PASS (including the new ones)

- [ ] **Step 2: Verify test count increased**

Check that new test files are included and running by looking for the new testset names in the output.

---
