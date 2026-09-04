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

    # If replacing only the algebraic row already makes M invertible, zero-M
    # columns are ordinary algebraic variables represented by that row. Pulling
    # their full L columns into differential rows changes the actual update.
    M = ComplexF64[1 0; 0 0]
    L = ComplexF64[0 1; 0 1]
    constrained = Tarang._constrained_mass_matrix(M, L, [2])
    @test Matrix(constrained) == ComplexF64[1 0; 0 1]

    # In a mixed DAE, row replacement can determine one algebraic variable
    # directly while another zero-M variable is a multiplier. Augment only the
    # latter; restoring both L columns would corrupt the u2 mass update.
    M_mixed = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
    L_mixed = ComplexF64[0 0 0 1; 0 0 1 0; 0 0 1 0; 1 0 0 0]
    constrained_mixed =
        Tarang._augmented_constrained_mass_matrix(M_mixed, L_mixed, [3, 4])
    @test Matrix(constrained_mixed) == ComplexF64[
        1 0 0 1;
        0 1 0 0;
        0 0 1 0;
        1 0 0 0
    ]
    @test Tarang._zero_mass_columns(sparse(M_mixed)) == [3, 4]

    # Projector pooling must inspect a sparse boundary-row slice without
    # materialising it for every Fourier mode. The zero-mass-column layout is
    # part of its identity, and those original algebraic coefficients remain in
    # the stored row so a genuinely mixed constraint gets the right residual.
    # Unrelated rows do not affect the shared constraint identity.
    L_constraint = sparse(ComplexF64[
        9 0 0 0 0 0;
        1 0 7 2 0 3;
        0 8 0 0 0 0;
        0 4 6 0 5 0
    ])
    constraint_rows = [2, 4]
    algebraic_columns = [3]
    expected_constraint = ComplexF64[
        1 0 7 2 0 3;
        0 4 6 0 5 0
    ]
    fingerprint = Tarang._constraint_projector_fingerprint(
        L_constraint, constraint_rows, algebraic_columns)
    materialized = Tarang._materialize_constraint(L_constraint, constraint_rows)
    @test materialized == expected_constraint
    @test Tarang._constraint_projector_matches(
        materialized, L_constraint, constraint_rows, algebraic_columns)

    L_same_constraint = copy(L_constraint)
    L_same_constraint[1, 6] = 11
    @test Tarang._constraint_projector_fingerprint(
        L_same_constraint, constraint_rows, algebraic_columns) == fingerprint
    @test Tarang._constraint_projector_matches(
        materialized, L_same_constraint, constraint_rows, algebraic_columns)

    L_different_algebraic_term = copy(L_constraint)
    L_different_algebraic_term[2, 3] = -19
    @test Tarang._constraint_projector_fingerprint(
        L_different_algebraic_term, constraint_rows, algebraic_columns) !=
        fingerprint
    @test !Tarang._constraint_projector_matches(
        materialized, L_different_algebraic_term, constraint_rows,
        algebraic_columns)

    L_different_constraint = copy(L_constraint)
    L_different_constraint[4, 6] = 12
    @test !Tarang._constraint_projector_matches(
        materialized, L_different_constraint, constraint_rows,
        algebraic_columns)

    # Warm the helpers, then guard against reintroducing a dense row-slice on a
    # pool hit. This bound is independent of the matrix width; current CPU use
    # is allocation-free, with a little tolerance for Julia runtime metadata.
    Tarang._constraint_projector_fingerprint(
        L_same_constraint, constraint_rows, algebraic_columns)
    Tarang._constraint_projector_matches(
        materialized, L_same_constraint, constraint_rows, algebraic_columns)
    fingerprint_alloc = @allocated Tarang._constraint_projector_fingerprint(
        L_same_constraint, constraint_rows, algebraic_columns)
    match_alloc = @allocated Tarang._constraint_projector_matches(
        materialized, L_same_constraint, constraint_rows, algebraic_columns)
    @test fingerprint_alloc <= 256
    @test match_alloc <= 256

    # A naturally global (no subproblems) full-rank DAE keeps the legacy
    # row-replacement path exercised end to end.
    dae_domain = PeriodicDomain(8)
    dae_u = ScalarField(dae_domain, "dae_u")
    dae_v = ScalarField(dae_domain, "dae_v")
    set!(dae_u, 0.0)
    set!(dae_v, 0.0)
    dae_problem = IVP([dae_u, dae_v])
    add_equation!(dae_problem, "dt(dae_u) + dae_u + dae_v = 1")
    add_equation!(dae_problem, "dae_v = 0")
    dae_ivp = InitialValueSolver(dae_problem, RK222(); dt=0.01)
    @test Tarang.compiled_subproblems(dae_problem) === nothing
    for _ in 1:100
        step!(dae_ivp)
    end
    ensure_layout!(dae_u, :g)
    ensure_layout!(dae_v, :g)
    @test maximum(abs, Array(get_grid_data(dae_u)) .- (1 - exp(-1))) < 2e-6
    @test maximum(abs, Array(get_grid_data(dae_v))) < 1e-12

    bad_u = ScalarField(dae_domain, "bad_u")
    bad_v = ScalarField(dae_domain, "bad_v")
    bad_problem = IVP([bad_u, bad_v])
    add_equation!(bad_problem, "dt(bad_u) + bad_u + bad_v = 0")
    add_equation!(bad_problem, "bad_v = 1")
    bad_ivp = InitialValueSolver(bad_problem, RK222(); dt=0.01)
    algebraic_error = try
        step!(bad_ivp)
        nothing
    catch err
        err
    end
    @test algebraic_error isa ArgumentError
    @test occursin("cannot project a nonzero algebraic right-hand side",
                   sprint(showerror, algebraic_error))
end

@testset "Subproblem RK final update preserves moving nonlinear boundary constraints" begin
    function moving_boundary_residual(; batched_modes::Bool)
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xbasis = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
        zbasis = ChebyshevT(coords["z"]; size=12, bounds=(0.0, 1.0))
        domain = Domain(dist, (xbasis, zbasis))

        b = ScalarField(domain, "b")
        gauge = ScalarField(dist, "gauge", (xbasis,), Float64)
        set!(gauge, 1.0)
        tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)
        tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)
        _, ez = unit_vector_fields(coords, dist)
        lift_basis = derivative_basis(zbasis, 1)
        tau_lift(A) = lift(A, lift_basis, -1)
        grad_b = grad(b) + ez * tau_lift(tau1)

        problem = IVP([b, gauge, tau1, tau2])
        add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
        add_equation!(problem,
                      "dt(b) - kappa*div(grad_b) + tau_lift(tau2) = 1 + b*b")
        # Algebraic-only row: analogous to the zero-Fourier pressure gauge in
        # an incompressible system. A moving BC elsewhere must not make the
        # differential-space projector demand rank from this row.
        add_equation!(problem, "gauge = 0")
        add_bc!(problem, "b(z=0) = sin(6.283185307*t)")
        add_bc!(problem, "b(z=1) = 0")

        dt = 0.01
        solver = InitialValueSolver(problem, RK222(); dt, batched_modes)
        step!(solver)

        if batched_modes
            @test !isempty(Tarang.active_mode_batches(solver))
        else
            @test isempty(Tarang.active_mode_batches(solver))
        end

        ensure_layout!(b, :g)
        ensure_layout!(gauge, :g)
        boundary = Array(get_grid_data(b))[:, 1]
        target = sin(6.283185307 * solver.sim_time)
        gauge_error = maximum(abs, Array(get_grid_data(gauge)))
        return max(maximum(abs, boundary .- target), gauge_error)
    end

    @test moving_boundary_residual(; batched_modes=false) < 1e-10
    @test moving_boundary_residual(; batched_modes=true) < 1e-10

    # A pure-Chebyshev system can legitimately carry redundant tau columns.
    # Applying the final constraint through an augmented mass system produces
    # enormous cancelling tau coefficients, so the CPU path must project only
    # through the differential variables with a smooth spectral lifting.
    function moving_redundant_tau_metrics()
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zbasis = ChebyshevT(coords["z"]; size=24, bounds=(0.0, 1.0))
        domain = Domain(dist, (zbasis,))

        b = ScalarField(domain, "b")
        set!(b, (z,) -> sin(π * z))
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lift_basis = derivative_basis(zbasis, 1)
        tau_lift(A) = lift(A, lift_basis, -1)

        problem = IVP([b, tau1, tau2])
        add_parameters!(problem; kappa=0.05, tau_lift)
        add_equation!(problem,
                      "dt(b) - kappa*lap(b) + tau_lift(tau1) + tau_lift(tau2) = 0")
        add_bc!(problem, "b(z=0) = sin(6.283185307*t)")
        add_bc!(problem, "b(z=1) = 0")

        dt = 0.002
        solver = InitialValueSolver(problem, RK222(); dt)
        for _ in 1:50
            step!(solver)
        end

        ensure_layout!(b, :g)
        values = vec(Array(get_grid_data(b)))
        target = sin(6.283185307 * solver.sim_time)
        boundary_error = max(abs(values[1] - target), abs(values[end]))
        return maximum(abs, values), boundary_error
    end

    field_max, boundary_error = moving_redundant_tau_metrics()
    @test isfinite(field_max)
    @test field_max < 2.0
    @test boundary_error < 1e-8

    # A single constraint row may contain both differential and algebraic
    # columns. Restore the algebraic stage value, but include its contribution
    # when computing the residual lifted through the differential field.
    function mixed_constraint_residual()
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zbasis = ChebyshevT(coords["z"]; size=12, bounds=(0.0, 1.0))
        domain = Domain(dist, (zbasis,))

        b = ScalarField(domain, "b")
        set!(b, (z,) -> z * (1 - z))
        a = ScalarField(dist, "a", (), Float64)
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lift_basis = derivative_basis(zbasis, 1)
        lift1 = lift(tau1, lift_basis, -1)
        lift2 = lift(tau2, lift_basis, -2)

        problem = IVP([b, a, tau1, tau2])
        add_parameters!(problem; lift1, lift2)
        add_equation!(problem, "dt(b) + lift1 + lift2 = 1 + b*b")
        add_bc!(problem, "b(z=0) = 0")
        add_bc!(problem, "b(z=1) = 0")
        add_bc!(problem, "integ(b) + a = 0.4")

        solver = InitialValueSolver(problem, RK222(); dt=0.01)
        step!(solver)
        sp = only(Tarang.compiled_subproblems(problem))
        for field in solver.state
            ensure_layout!(field, :c)
        end
        x = zeros(ComplexF64, size(sp.M_min, 2))
        alg_f = zeros(ComplexF64, size(sp.M_min, 1))
        Tarang.gather_inputs!(x, sp, solver.state)
        Tarang.gather_alg_F!(alg_f, sp)
        return maximum(abs,
            sp.L_min[sp.bc_rows, :] * x - alg_f[sp.bc_rows])
    end

    @test mixed_constraint_residual() < 1e-10

    # Static constraints need the same final projection. The RK mass update has
    # no algebraic rows, and a generic explicit forcing need not be tangent to
    # the wall trace even though every implicit stage satisfies it.
    function static_boundary_error(timestepper)
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zbasis = ChebyshevT(coords["z"]; size=12, bounds=(0.0, 1.0))
        domain = Domain(dist, (zbasis,))

        b = ScalarField(domain, "b")
        set!(b, 0.0)
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lift_basis = derivative_basis(zbasis, 1)
        tau_lift(A) = lift(A, lift_basis, -1)

        problem = IVP([b, tau1, tau2])
        add_parameters!(problem; tau_lift)
        add_equation!(problem,
                      "dt(b) + tau_lift(tau1) + tau_lift(tau2) = 1 + b*b")
        add_bc!(problem, "b(z=0) = 0")
        add_bc!(problem, "b(z=1) = 0")

        solver = InitialValueSolver(problem, timestepper; dt=0.01)
        step!(solver)
        ensure_layout!(b, :g)
        values = vec(Array(get_grid_data(b)))
        return max(abs(values[1]), abs(values[end]))
    end

    for timestepper in (RK222(), RK443(), Tarang.RKGFY(), RKSMR())
        @test static_boundary_error(timestepper) < 1e-10
    end

    # RK111 has c_explicit=[0] but c_implicit=[1]. Its RHS belongs to the old
    # time while the implicit boundary solve (and the algebraic state retained
    # from it) belongs to the final time.
    function rk111_moving_constraint_errors()
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zbasis = ChebyshevT(coords["z"]; size=12, bounds=(0.0, 1.0))
        domain = Domain(dist, (zbasis,))

        b = ScalarField(domain, "b")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lift_basis = derivative_basis(zbasis, 1)
        lift1 = lift(tau1, lift_basis, -1)
        lift2 = lift(tau2, lift_basis, -2)
        problem = IVP([b, tau1, tau2])
        add_parameters!(problem; kappa=0.05, lift1, lift2)
        add_equation!(problem, "dt(b) - kappa*lap(b) + lift1 + lift2 = 0")
        add_bc!(problem, "b(z=0) = sin(t)")
        add_bc!(problem, "b(z=1) = 0")

        solver = InitialValueSolver(problem, RK111(); dt=0.01)
        step!(solver)
        sp = only(Tarang.compiled_subproblems(problem))
        state = solver.timestepper_state
        last_stage = state.timestepper_data[:_sp_rk_RHS][1]
        alg_f = state.timestepper_data[:_sp_rk_ALG_F][1]
        stage_constraints = sp.L_min[sp.bc_rows, :] * last_stage
        stage_error = maximum(abs, stage_constraints .- alg_f[sp.bc_rows])

        ensure_layout!(b, :g)
        b_values = vec(Array(get_grid_data(b)))
        boundary_error = max(abs(b_values[1] - sin(solver.sim_time)),
                             abs(b_values[end]))

        for field in solver.state
            ensure_layout!(field, :c)
        end
        final_x = zeros(ComplexF64, size(sp.M_min, 2))
        Tarang.gather_inputs!(final_x, sp, solver.state)
        algebraic_columns = Tarang._zero_mass_columns(sp.M_min)
        algebraic_error = maximum(abs,
            final_x[algebraic_columns] .- last_stage[algebraic_columns])
        return stage_error, boundary_error, algebraic_error
    end

    rk111_stage_error, rk111_boundary_error, rk111_algebraic_error =
        rk111_moving_constraint_errors()
    @test rk111_stage_error < 1e-10
    @test rk111_boundary_error < 1e-10
    @test rk111_algebraic_error < 1e-12

    # Exact boundary values are not enough: replacing the weighted RK result
    # by a constrained last stage also satisfies these traces but loses the
    # method's temporal order in the interior. Use a manufactured solution
    # with a nonlinear-in-space component that the linear boundary lifting
    # cannot erase.
    function moving_boundary_error(timestepper, dt)
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zbasis = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
        domain = Domain(dist, (zbasis,))

        b = ScalarField(domain, "b")
        set!(b, (z,) -> 0.0)
        shape = ScalarField(domain, "shape")
        set!(shape, (z,) -> 1 - z + sin(π * z))
        exact = ScalarField(domain, "exact")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lift_basis = derivative_basis(zbasis, 1)
        tau_lift(A) = lift(A, lift_basis, -1)

        problem = IVP([b, tau1, tau2])
        add_parameters!(problem; shape, tau_lift)
        add_equation!(problem,
                      "dt(b) - b + tau_lift(tau1) + tau_lift(tau2) = shape")
        add_bc!(problem, "b(z=0) = exp(t) - 1")
        add_bc!(problem, "b(z=1) = 0")

        final_time = 0.4
        solver = InitialValueSolver(problem, timestepper; dt)
        for _ in 1:round(Int, final_time / dt)
            step!(solver)
        end

        set!(exact, (z,) -> (exp(final_time) - 1) * (1 - z + sin(π * z)))
        ensure_layout!(b, :g)
        ensure_layout!(exact, :g)
        return maximum(abs, Array(get_grid_data(b)) .- Array(get_grid_data(exact)))
    end

    for (timestepper, minimum_order) in ((RK222(), 1.7), (RK443(), 2.7))
        coarse_error = moving_boundary_error(timestepper, 0.04)
        fine_error = moving_boundary_error(timestepper, 0.02)
        @test fine_error < coarse_error
        @test log2(coarse_error / fine_error) > minimum_order
    end

    # A 0-D ScalarField can be used as a live boundary parameter without the
    # BC being classified as explicitly time-dependent. It is re-gathered each
    # step, and the final projection must enforce the newly supplied value.
    function mutable_boundary_values()
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zbasis = ChebyshevT(coords["z"]; size=12, bounds=(0.0, 1.0))
        domain = Domain(dist, (zbasis,))

        b = ScalarField(domain, "b")
        set!(b, 0.0)
        wall = ScalarField(dist, "wall", (), Float64)
        Tarang.set_grid_data!(wall, [0.0])
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lift_basis = derivative_basis(zbasis, 1)
        tau_lift(A) = lift(A, lift_basis, -1)

        problem = IVP([b, tau1, tau2])
        add_parameters!(problem; kappa=0.05, wall, tau_lift)
        add_equation!(problem,
                      "dt(b) - kappa*lap(b) + tau_lift(tau1) + tau_lift(tau2) = 0")
        add_bc!(problem, "b(z=0) = wall")
        add_bc!(problem, "b(z=1) = 0")

        solver = InitialValueSolver(problem, RK222(); dt=0.01)
        @test !Tarang.has_time_dependent_bcs(problem.bc_manager)
        @test !Tarang.alg_F_is_static(only(Tarang.compiled_subproblems(problem)))

        observed = Float64[]
        for target in (0.25, 0.4)
            Tarang.set_grid_data!(wall, [target])
            step!(solver)
            ensure_layout!(b, :g)
            push!(observed, vec(Array(get_grid_data(b)))[1])
        end
        return observed
    end

    @test mutable_boundary_values() ≈ [0.25, 0.4] atol=1e-10

    # Real coupled DAE coverage for the row-subset projector: the zero Fourier
    # mode contains a pressure gauge that is enforceable only through an
    # algebraic column, alongside physical wall constraints that do project
    # through differential velocity/buoyancy variables.
    function moving_rbc_constraints()
        Lx, Lz = 4.0, 1.0
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xbasis = RealFourier(coords["x"]; size=4, bounds=(0.0, Lx))
        zbasis = ChebyshevT(coords["z"]; size=6, bounds=(0.0, Lz))
        domain = Domain(dist, (xbasis, zbasis))

        p = ScalarField(domain, "p")
        b = ScalarField(domain, "b")
        u = VectorField(domain, "u")
        tau_p = ScalarField(dist, "tau_p", (), Float64)
        tau_b1 = ScalarField(dist, "tau_b1", (xbasis,), Float64)
        tau_b2 = ScalarField(dist, "tau_b2", (xbasis,), Float64)
        tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
        tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

        _, ez = unit_vector_fields(coords, dist)
        lift_basis = derivative_basis(zbasis, 1)
        tau_lift(A) = lift(A, lift_basis, -1)
        grad_u = grad(u) + ez * tau_lift(tau_u1)
        grad_b = grad(b) + ez * tau_lift(tau_b1)

        problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2])
        add_parameters!(problem; kappa=0.01, nu=0.01, Lz, ez,
                        grad_u, grad_b, tau_lift)
        add_equation!(problem, "trace(grad_u) + tau_p = 0")
        add_equation!(problem,
                      "dt(b) - kappa*div(grad_b) + tau_lift(tau_b2) = -u⋅grad(b)")
        add_equation!(problem,
                      "dt(u) - nu*div(grad_u) + grad(p) - b*ez + tau_lift(tau_u2) = -u⋅grad(u)")
        add_bc!(problem, "b(z=0) = sin(t)")
        add_bc!(problem, "u(z=0) = 0")
        add_bc!(problem, "b(z=Lz) = 0")
        add_bc!(problem, "u(z=Lz) = 0")
        add_bc!(problem, "integ(p) = 0")

        solver = InitialValueSolver(problem, RK222(); dt=0.01,
                                    batched_modes=false)
        step!(solver)

        ensure_layout!(b, :g)
        b_grid = Array(get_grid_data(b))
        @test maximum(abs, b_grid[:, 1] .- sin(solver.sim_time)) < 1e-10
        @test maximum(abs, b_grid[:, end]) < 1e-10
        for component in u.components
            ensure_layout!(component, :g)
            u_grid = Array(get_grid_data(component))
            @test maximum(abs, u_grid[:, 1]) < 1e-10
            @test maximum(abs, u_grid[:, end]) < 1e-10
        end
        @test abs(integrate(p)) < 1e-10
    end

    moving_rbc_constraints()

    function hydrostatic_rbc_pressure_error()
        Lx, Lz = 4.0, 1.0
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xbasis = RealFourier(coords["x"]; size=4, bounds=(0.0, Lx))
        zbasis = ChebyshevT(coords["z"]; size=6, bounds=(0.0, Lz))
        domain = Domain(dist, (xbasis, zbasis))

        p = ScalarField(domain, "p")
        b = ScalarField(domain, "b")
        u = VectorField(domain, "u")
        set!(p, (x, z) -> z - z^2 / 2 - 1 / 3)
        set!(b, (x, z) -> 1 - z)
        set!(u, ((x, z) -> 0.0, (x, z) -> 0.0))
        tau_p = ScalarField(dist, "tau_p", (), Float64)
        tau_b1 = ScalarField(dist, "tau_b1", (xbasis,), Float64)
        tau_b2 = ScalarField(dist, "tau_b2", (xbasis,), Float64)
        tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
        tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

        _, ez = unit_vector_fields(coords, dist)
        lift_basis = derivative_basis(zbasis, 1)
        tau_lift(A) = lift(A, lift_basis, -1)
        grad_u = grad(u) + ez * tau_lift(tau_u1)
        grad_b = grad(b) + ez * tau_lift(tau_b1)

        problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2])
        add_parameters!(problem; kappa=0.01, nu=0.01, Lz, ez,
                        grad_u, grad_b, tau_lift)
        add_equation!(problem, "trace(grad_u) + tau_p = 0")
        add_equation!(problem,
                      "dt(b) - kappa*div(grad_b) + tau_lift(tau_b2) = -u⋅grad(b)")
        add_equation!(problem,
                      "dt(u) - nu*div(grad_u) + grad(p) - b*ez + tau_lift(tau_u2) = -u⋅grad(u)")
        add_bc!(problem, "b(z=0) = 1")
        add_bc!(problem, "u(z=0) = 0")
        add_bc!(problem, "b(z=Lz) = 0")
        add_bc!(problem, "u(z=Lz) = 0")
        add_bc!(problem, "integ(p) = 0")

        solver = InitialValueSolver(problem, RK222(); dt=0.01,
                                    batched_modes=false)
        step!(solver)

        ensure_layout!(p, :g)
        ensure_layout!(b, :g)
        x_grid = vec(Array(Tarang.local_grid(xbasis, dist, 1)))
        z_grid = vec(Array(Tarang.local_grid(zbasis, dist, 1)))
        p_exact = [z - z^2 / 2 - 1 / 3 for _ in x_grid, z in z_grid]
        b_exact = [1 - z for _ in x_grid, z in z_grid]
        pressure_error = maximum(abs, Array(get_grid_data(p)) .- p_exact)
        buoyancy_error = maximum(abs, Array(get_grid_data(b)) .- b_exact)
        velocity_max = 0.0
        for component in u.components
            ensure_layout!(component, :g)
            velocity_max = max(velocity_max,
                               maximum(abs, Array(get_grid_data(component))))
        end
        return pressure_error, buoyancy_error, velocity_max
    end

    pressure_error, buoyancy_error, velocity_max =
        hydrostatic_rbc_pressure_error()
    @test pressure_error < 1e-10
    @test buoyancy_error < 1e-10
    @test velocity_max < 1e-10
end
