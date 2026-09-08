using Test
using Tarang
using LinearAlgebra

function _stress_free_regression_problem(ndim, problem_type; wall=:stress_free, nz=20)
    labels = ndim == 2 ? ["x", "z"] : ["x", "y", "z"]
    coords = CartesianCoordinates(labels...)
    dist = Distributor(coords; dtype=Float64, device=CPU())
    tangential = Tuple(RealFourier(coords[label]; size=4, bounds=(0.0, 2π))
                       for label in labels[1:end-1])
    zb = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
    u = VectorField(dist, coords, "u", (tangential..., zb), Float64)
    tau1 = VectorField(dist, coords, "tau1", tangential, Float64)
    tau2 = VectorField(dist, coords, "tau2", tangential, Float64)
    problem = problem_type([u, tau1, tau2])
    lb = derivative_basis(zb, 2)
    add_parameters!(problem; l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2))
    lhs = problem_type === LinearBoundaryValueProblem ? "u" : "dt(u)"
    add_equation!(problem, "$lhs - lap(u) + l1 + l2 = 0")
    for z in (0.0, 1.0)
        if wall === :no_slip
            no_slip!(problem, "u", "z", z)
        else
            add_bc!(problem, stress_free_bc("u", "z", z; component_coordinates=labels))
        end
    end
    return (; problem, u, coords, zb)
end

@testset "Stress-free component boundary assembly" begin
    @testset "component equation validation retains constraint counts" begin
        case = _stress_free_regression_problem(2, InitialValueProblem)
        Tarang._merge_boundary_conditions!(case.problem)
        @test Tarang.validate_problem(case.problem)
        last_condition = pop!(case.problem.equations)
        @test_throws ArgumentError Tarang.validate_problem(case.problem)
        push!(case.problem.equations, last_condition, last_condition)
        @test_throws ArgumentError Tarang.validate_problem(case.problem)
    end

    for ndim in (2, 3), wall in (:no_slip, :stress_free)
        @testset "$ndim dimensions, $wall" begin
            case = _stress_free_regression_problem(ndim, LinearBoundaryValueProblem; wall)
            solver = BoundaryValueSolver(case.problem)
            expected_size = ndim * 22
            square = all(sp -> size(sp.L_min) == (expected_size, expected_size), solver.subproblems)
            @test square
            expected_equations = wall === :stress_free ? [ndim*20; ones(Int, 2ndim)] : [ndim*20, ndim, ndim]
            @test all(sp -> Tarang._subproblem_eqn_sizes(sp) == expected_equations, solver.subproblems)
            @test sum(eq["equation_size"] for eq in case.problem.equation_data) ==
                  sum(Tarang._coeff_space_dofs, case.problem.variables)
            if square
                solve!(solver)
                @test all(c -> all(iszero, c["g"]), case.u.components)
            end
        end
    end

    @testset "nested selectors retain parent component columns" begin
        for ndim in (2, 3)
            case = _stress_free_regression_problem(ndim, LinearBoundaryValueProblem)
            solver = BoundaryValueSolver(case.problem)
            sp = first(solver.subproblems)
            u = case.u
            for j in 1:ndim, wall in (0.0, 1.0), order in (0, 1)
                selected = component(u, j)
                if order == 1
                    selected = Differentiate(selected, case.coords["z"], 1)
                end
                outer = Interpolate(selected, case.coords["z"], wall)
                inner = order == 0 ? u : Differentiate(u, case.coords["z"], 1)
                reordered = component(Interpolate(inner, case.coords["z"], wall), j)
                # T_n(±1)=(±1)^n and dT_n/dz=2*n²*(±1)^(n-1)
                # on [0,1]. Every unselected parent component must remain zero.
                row = order == 0 ? [(2wall-1)^n for n in 0:19] :
                      [n == 0 ? 0.0 : 2n^2*(2wall-1)^(n-1) for n in 0:19]
                expected = zeros(ComplexF64, 1, ndim*20)
                expected[1, (j-1)*20+1:j*20] .= row
                @test expression_matrices(outer, sp, [u])[u] ≈ expected atol=1e-10 rtol=1e-12
                @test expression_matrices(reordered, sp, [u])[u] ≈ expected atol=1e-10 rtol=1e-12
                @test Tarang._expression_subproblem_dofs(sp, reordered) == 1
            end
            scalar_size = Tarang._coeff_space_dofs(u.components[1])
            for j in 1:ndim
                expected = zeros(ComplexF64, scalar_size, ndim*scalar_size)
                expected[:, (j-1)*scalar_size+1:j*scalar_size] .= Matrix{ComplexF64}(I, scalar_size, scalar_size)
                @test Tarang.build_expression_matrix_block(component(u, j), u,
                          scalar_size, ndim*scalar_size) == expected
            end
        end
    end

    @testset "nonzero tangential and normal heat modes" begin
        for ndim in (2, 3)
            case = _stress_free_regression_problem(ndim, InitialValueProblem)
            profiles = ndim == 2 ?
                ((x,z) -> cos(x)*cos(π*z), (x,z) -> sin(x)*sin(π*z)) :
                ((x,y,z) -> cos(x)*cos(π*z),
                 (x,y,z) -> sin(y)*cos(2π*z),
                 (x,y,z) -> cos(x+y)*sin(π*z))
            decay_rates = ndim == 2 ? [1+π^2, 1+π^2] : [1+π^2, 1+4π^2, 2+π^2]
            for (c, profile) in zip(case.u.components, profiles)
                set!(c, profile)
            end
            initial = [copy(c["g"]) for c in case.u.components]
            solver = InitialValueSolver(case.problem, RK222(); dt=1e-4)
            square = all(sp -> size(sp.L_min, 1) == size(sp.L_min, 2),
                         Tarang.compiled_subproblems(case.problem))
            @test square
            if square
                for _ in 1:2
                    step!(solver)
                end
                for j in 1:ndim
                    @test case.u.components[j]["g"] ≈ exp(-decay_rates[j]*solver.sim_time) .* initial[j] atol=1e-7 rtol=1e-7
                end
                normal = case.u.components[end]["g"]
                @test maximum(abs, selectdim(normal, ndim, 1)) < 1e-10
                @test maximum(abs, selectdim(normal, ndim, 20)) < 1e-10
                for c in case.u.components[1:end-1]
                    derivative = evaluate(Differentiate(c, case.coords["z"], 1), :g)["g"]
                    @test maximum(abs, selectdim(derivative, ndim, 1)) < 1e-9
                    @test maximum(abs, selectdim(derivative, ndim, 20)) < 1e-9
                end
            end
        end
    end
end
