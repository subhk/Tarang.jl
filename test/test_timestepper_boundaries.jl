using Test
using Tarang
using SparseArrays

@testset "Timestepper Boundary Helpers" begin
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x))

    problem = IVP([u])
    add_equation!(problem, "∂t(u) = 0")
    solver = InitialValueSolver(problem, RK111(); dt=0.01)
    ts_state = solver.timestepper_state
    current_state = solver.state
    n = sum(length(Tarang.get_coeff_data(field)) for field in current_state)

    @test isdefined(Tarang, :_distributed_field_path_required)
    @test isdefined(Tarang, :_timestepper_subproblems)
    @test isdefined(Tarang, :_imex_rk_explicit_fallback_reason)
    @test isdefined(Tarang, :_global_matrix_implicit_total_dofs)

    @test Tarang._distributed_field_path_required(current_state) == false
    @test Tarang._timestepper_subproblems(solver) === nothing
    @test Tarang._imex_rk_explicit_fallback_reason(ts_state, solver, current_state, nothing) === :missing_linear_operator
    @test Tarang._imex_rk_explicit_fallback_reason(ts_state, solver, current_state, spzeros(ComplexF64, n, n)) === :zero_linear_operator
    @test Tarang._global_matrix_implicit_total_dofs(solver) == n
end

@testset "Solver Step Orchestration Helpers" begin
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x))

    problem = IVP([u])
    add_equation!(problem, "∂t(u) = 0")
    solver = InitialValueSolver(problem, RK111(); dt=0.01)

    @test isdefined(Tarang, :_refresh_step_boundary_conditions!)
    @test isdefined(Tarang, :_ensure_timestepper_state!)
    @test isdefined(Tarang, :_sync_solver_from_timestepper!)
    @test isdefined(Tarang, :_advance_solver_clock!)

    @test solver.timestepper_state === nothing
    state = Tarang._ensure_timestepper_state!(solver, 0.02)
    @test state === solver.timestepper_state
    @test state.dt == 0.02
    @test solver.dt == 0.02

    state2 = Tarang._ensure_timestepper_state!(solver, 0.03)
    @test state2 === state
    @test state2.dt == 0.03
    @test solver.dt == 0.03
    @test last(state2.dt_history) == 0.03

    old_time = solver.sim_time
    old_iter = solver.iteration
    Tarang._advance_solver_clock!(solver, 0.03, 0.001)
    @test solver.sim_time == old_time + 0.03
    @test solver.iteration == old_iter + 1
    @test solver.performance_stats.total_steps >= 1
end

@testset "Timestep history update reuses bounded storage" begin
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x))

    state = Tarang.TimestepperState(RK111(), 0.01, ScalarField[u])
    Tarang.update_timestep_history!(state, 0.02)
    @test state.dt_history == [0.01, 0.02]

    alloc = @allocated Tarang.update_timestep_history!(state, 0.03)
    @test alloc == 0
    @test state.dt_history == [0.02, 0.03]
    @test Tarang.get_previous_timestep(state) == 0.02
end

@testset "Multistep startup methods advance to target order" begin
    function startup_solver(timestepper)
        domain = PeriodicDomain(8)
        u = ScalarField(domain, "u")
        set!(u, (x,) -> sin(x))

        problem = IVP([u])
        add_equation!(problem, "∂t(u) = 0")
        return InitialValueSolver(problem, timestepper; dt=0.01)
    end

    for (timestepper, iteration_key, nsteps) in (
        (CNAB2(), :cnab2_iteration, 2),
        (SBDF2(), :sbdf2_iteration, 2),
        (SBDF3(), :sbdf3_iteration, 3),
        (SBDF4(), :sbdf4_iteration, 4),
        (Tarang.MCNAB2(), :iteration, 2),
    )
        solver = startup_solver(timestepper)
        for _ in 1:nsteps
            step!(solver)
        end

        @test solver.timestepper_state.timestepper_data[iteration_key] == nsteps
    end
end

@testset "Variable timestep coefficient builders" begin
    dt = 0.02
    dt_prev = 0.01
    w1 = dt / dt_prev

    cnab2_a, cnab2_b, cnab2_c = Tarang._cnab2_coefs(dt, dt_prev)
    @test cnab2_a == (1.0 / dt, -1.0 / dt)
    @test cnab2_b == (0.5, 0.5)
    @test cnab2_c == (0.0, 1.0 + w1 / 2.0, -w1 / 2.0)

    @test applicable(Tarang._sbdf2_coefs, dt, dt_prev)
    if applicable(Tarang._sbdf2_coefs, dt, dt_prev)
        sbdf2_a, sbdf2_b, sbdf2_c = Tarang._sbdf2_coefs(dt, dt_prev)
        @test isapprox(sbdf2_a[1], (1.0 + 2.0 * w1) / ((1.0 + w1) * dt); atol=1e-14)
        @test isapprox(sbdf2_a[2], -(1.0 + w1) / dt; atol=1e-14)
        @test isapprox(sbdf2_a[3], w1^2 / ((1.0 + w1) * dt); atol=1e-14)
        @test sbdf2_b == (1.0,)
        @test sbdf2_c == (0.0, 1.0 + w1, -w1)
    end

    k2, k1, k0 = 0.03, 0.02, 0.01
    w2 = k2 / k1
    w1 = k1 / k0
    @test applicable(Tarang._sbdf3_coefs, k2, k1, k0)
    if applicable(Tarang._sbdf3_coefs, k2, k1, k0)
        sbdf3_a, sbdf3_b, sbdf3_c = Tarang._sbdf3_coefs(k2, k1, k0)
        expected_a = (
            (1 + w2 / (1 + w2) + w1 * w2 / (1 + w1 * (1 + w2))) / k2,
            (-1 - w2 - w1 * w2 * (1 + w2) / (1 + w1)) / k2,
            w2^2 * (w1 + 1 / (1 + w2)) / k2,
            -w1^3 * w2^2 * (1 + w2) / (1 + w1) / (1 + w1 + w1 * w2) / k2,
        )
        expected_c = (
            0.0,
            (1 + w2) * (1 + w1 * (1 + w2)) / (1 + w1),
            -w2 * (1 + w1 * (1 + w2)),
            w1 * w1 * w2 * (1 + w2) / (1 + w1),
        )
        @test all(isapprox.(sbdf3_a, expected_a; atol=1e-14))
        @test sbdf3_b == (1.0, 0.0, 0.0, 0.0)
        @test all(isapprox.(sbdf3_c, expected_c; atol=1e-14))
    end

    k3, k2, k1, k0 = 0.04, 0.03, 0.02, 0.01
    w3 = k3 / k2
    w2 = k2 / k1
    w1 = k1 / k0
    A1 = 1 + w1 * (1 + w2)
    A2 = 1 + w2 * (1 + w3)
    A3 = 1 + w1 * A2
    @test applicable(Tarang._sbdf4_coefs, k3, k2, k1, k0)
    if applicable(Tarang._sbdf4_coefs, k3, k2, k1, k0)
        sbdf4_a, sbdf4_b, sbdf4_c = Tarang._sbdf4_coefs(k3, k2, k1, k0)
        expected_a = (
            (1 + w3 / (1 + w3) + w2 * w3 / A2 + w1 * w2 * w3 / A3) / k3,
            (-1 - w3 * (1 + (w2 * (1 + w3) / (1 + w2)) * (1 + w1 * A2 / A1))) / k3,
            w3 * (w3 / (1 + w3) + (w2 * w3 * (A3 + w1)) / (1 + w1)) / k3,
            -(w2^3 * w3^2 * (1 + w3) * A3) / ((1 + w2) * A2 * k3),
            ((1 + w3) * A2 * w1^4 * w2^3 * w3^2) / ((1 + w1) * A1 * A3 * k3),
        )
        expected_c = (
            0.0,
            (w2 * (1 + w3) * ((1 + w3) * (A3 + w1) + (1 + w1) / w2)) / ((1 + w2) * A1),
            -(A2 * A3 * w3) / (1 + w1),
            (w2^2 * w3 * (1 + w3) * A3) / (1 + w2),
            -(w1^3 * w2^2 * w3 * (1 + w3) * A2) / ((1 + w1) * A1),
        )
        @test all(isapprox.(sbdf4_a, expected_a; atol=1e-14))
        @test sbdf4_b == (1.0, 0.0, 0.0, 0.0, 0.0)
        @test all(isapprox.(sbdf4_c, expected_c; atol=1e-14))
    end
end

@testset "State Vector Transport Helpers" begin
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> cos(x))
    fields = ScalarField[u]

    mode = isdefined(Tarang, :_state_vector_transport_mode) ?
           Tarang._state_vector_transport_mode : nothing

    @test mode !== nothing
    if mode !== nothing
        @test mode(ScalarField[]) === :empty
        @test mode(fields) === :local
    end

    vector = Tarang.fields_to_vector(fields)
    reusable = similar(vector)

    @test Tarang.fields_to_vector!(reusable, fields) === reusable
    @test reusable == vector

    copied = Tarang.vector_to_fields(vector, fields)
    @test Tarang.fields_to_vector(copied) == vector

    fill!(reusable, 2)
    Tarang.copy_solution_to_fields!(copied, reusable)
    @test Tarang.fields_to_vector(copied) == reusable
end

@testset "RHS Runtime Strategy Helpers" begin
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x))

    problem = IVP([u])
    add_equation!(problem, "∂t(u) = -u")
    solver = InitialValueSolver(problem, RK111(); dt=0.01)

    strategy = isdefined(Tarang, :_rhs_evaluation_strategy) ?
               Tarang._rhs_evaluation_strategy : nothing

    @test strategy !== nothing
    @test solver.rhs_plan !== nothing
    @test solver.rhs_plan.is_compiled

    if strategy !== nothing
        @test strategy(solver) === :lazy
        @test strategy(solver; buffered=true) === :lazy_buffered

        original_plan = solver.rhs_plan
        solver.rhs_plan = nothing
        @test strategy(solver) === :interpreted

        solver.rhs_plan = original_plan
        original_plan.is_compiled = false
        @test strategy(solver) === :interpreted
        original_plan.is_compiled = true
    end
end

@testset "Legacy Global Matrix Path Helpers" begin
    domain = PeriodicDomain(8)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x))

    problem = IVP([u])
    add_equation!(problem, "∂t(u) = 0")
    solver = InitialValueSolver(problem, CNAB1(); dt=0.01)

    matrices = isdefined(Tarang, :_global_matrix_implicit_matrices) ?
               Tarang._global_matrix_implicit_matrices : nothing
    missing_reason = isdefined(Tarang, :_global_matrix_implicit_missing_matrix_reason) ?
                     Tarang._global_matrix_implicit_missing_matrix_reason : nothing
    distributed_reason = isdefined(Tarang, :_global_matrix_implicit_distributed_fallback_reason) ?
                         Tarang._global_matrix_implicit_distributed_fallback_reason : nothing

    @test matrices !== nothing
    @test missing_reason !== nothing
    @test distributed_reason !== nothing

    if matrices !== nothing && missing_reason !== nothing && distributed_reason !== nothing
        L_matrix, M_matrix = matrices(solver)
        @test L_matrix !== nothing
        @test M_matrix !== nothing
        @test missing_reason(L_matrix, M_matrix) === nothing
        @test missing_reason(nothing, M_matrix) === :missing_linear_operator
        @test missing_reason(L_matrix, nothing) === :missing_mass_operator
        @test distributed_reason(solver.state) === nothing
    end
end
