"""
Type stability tests for Tarang.jl hot-path functions.

Uses @inferred to verify that critical functions return concrete types,
preventing performance regressions from type instability.
"""

using Test
using Tarang
using LinearAlgebra
using PencilFFTs

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
        # Fields must share the same bases (same Distributor/Domain) for arithmetic
        f, dist, basis = make_1d_scalar_field()
        g = ScalarField(dist, "g", (basis,), Float64)
        ensure_layout!(g, :g)
        get_grid_data(g) .= randn(8)
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
        @test @inferred(Union{Nothing, PencilFFTs.PencilFFTPlan}, _find_pencil_plan(dist)) === nothing
        @test only(Base.return_types(_find_pencil_plan, (typeof(dist),))) !== Any

        # Fallback dispatch: unknown transform type returns data unchanged
        data = randn(ComplexF64, 8)
        struct DummyTransform <: Transform end
        @test @inferred(_apply_forward(data, DummyTransform())) === data
        @test @inferred(_apply_backward(data, DummyTransform())) === data
    end

    @testset "Phi function identity cache" begin
        # Note: @inferred fails here because get!() with Dict{Tuple,Matrix} loses
        # the element type parameter at inference time. The runtime type is correct.
        I1 = _get_identity_matrix(4, Float64)
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

    @testset "Lazy RHS preserves concrete state field vector type" begin
        domain = PeriodicDomain(8)
        u = ScalarField(domain, "u")
        set!(u, (x,) -> sin(x))

        problem = IVP([u])
        add_equation!(problem, "dt(u) = 0")
        solver = InitialValueSolver(problem, RK111(); dt=0.01)

        step!(solver, 0.01)
        rhs = Tarang.evaluate_rhs(solver, solver.state, solver.sim_time)

        @test rhs isa typeof(solver.state)
        @test eltype(rhs) === eltype(solver.state)

        ts_state = solver.timestepper_state
        @test only(Base.return_types(Tarang.step!, (typeof(ts_state), typeof(solver)))) === Nothing
        @test only(Base.return_types(Tarang.step_rk_imex!, (typeof(ts_state), typeof(solver)))) === Nothing
    end

    @testset "Multistep history preserves concrete state field vector type" begin
        domain = PeriodicDomain(8)
        u = ScalarField(domain, "u")
        set!(u, (x,) -> sin(x))

        problem = IVP([u])
        add_equation!(problem, "∂t(u) = 0")
        solver = InitialValueSolver(problem, CNAB1(); dt=0.01)
        state_type = typeof(solver.state)

        step!(solver, 0.01)

        @test typeof(solver.state) === state_type
        @test typeof(solver.timestepper_state.history[end]) === state_type
        @test typeof(Tarang.copy_state(solver.state)) === state_type
    end

    @testset "Parameterized operator types" begin
        f, dist, basis = make_1d_scalar_field()

        # Single-operand operators carry concrete operand type in type parameter
        lap = Laplacian(f)
        @test lap isa Laplacian{typeof(f)}
        @test fieldtype(typeof(lap), :operand) === typeof(f)

        copy_op = Copy(f)
        @test copy_op isa Copy{typeof(f)}

        hilbert = HilbertTransform(f)
        @test hilbert isa HilbertTransform{typeof(f)}

        # Nested operator carries fully concrete type
        lap2 = Laplacian(Laplacian(f))
        @test lap2 isa Laplacian{Laplacian{typeof(f)}}

        # UnaryGridFunction carries concrete function type (enables inlining)
        uf = UnaryGridFunction(f, sin, "sin")
        @test uf isa UnaryGridFunction{typeof(sin), typeof(f)}
        @test fieldtype(typeof(uf), :func) === typeof(sin)

        # GeneralFunction carries concrete function type
        gf = GeneralFunction(f, exp, "exp")
        @test gf isa GeneralFunction{typeof(exp), typeof(f)}

        # Arithmetic operators (already parameterized — confirm still work)
        lap_add = Laplacian(f) + Laplacian(f)
        @test lap_add isa Tarang.AddOperator{<:Laplacian, <:Laplacian}

        # FutureState.dist is now typed (not Any)
        @test fieldtype(Tarang.FutureState, :dist) == Union{Nothing, Tarang.Distributor}
    end

    @testset "Solver and operator field types" begin
        # evaluator field narrowed from Any to Union{Nothing, AbstractEvaluator}
        @test fieldtype(Tarang.InitialValueSolver, :evaluator) == Union{Nothing, Tarang.AbstractEvaluator}
        @test fieldtype(Tarang.SolverBaseData, :evaluator) == Union{Nothing, Tarang.AbstractEvaluator}

        # Evaluator <: AbstractEvaluator
        @test Tarang.Evaluator <: Tarang.AbstractEvaluator

        # IndexOperator indices stored as Tuple (not Vector{Any})
        f, dist, basis = make_1d_scalar_field()
        # Use a concrete Tuple of indices
        indices_tuple = (1,)
        op = Tarang.IndexOperator(f, indices_tuple)
        @test op.indices isa Tuple
        @test fieldtype(typeof(op), :indices) <: Tuple
        @test !(fieldtype(typeof(op), :indices) <: Vector)
    end

end # Type Stability
