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
