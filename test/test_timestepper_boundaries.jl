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
