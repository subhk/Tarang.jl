using Test
using Tarang
using LinearAlgebra
using SparseArrays

const REVIEW_CASE = get(ENV, "TARANG_SOLVER_REVIEW_CASE", "all")
review_case(name) = REVIEW_CASE == "all" || REVIEW_CASE == name

struct NonIterableReviewNode end

struct MalformedReviewRHS <: Tarang.Operand
    operands::NonIterableReviewNode
end

struct StopAfterExplicitFirstStage <: Tarang.LazyFuture end

if review_case("imex_failure")
    @testset "IMEX singular stage never drops to explicit RK" begin
        dt = 0.1
        timestepper = Tarang.RKGFY()
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(Domain(dist, (basis,)), "u")
        u["g"] = sin.(range(0.0, 2π; length=9)[1:8])
        problem = IVP([u])
        add_equation!(problem, "∂t(u) + u = 0")
        solver = InitialValueSolver(problem, timestepper; dt)

        n = size(Tarang.compiled_problem(problem).mass_matrix, 1)
        stage_scale = dt * timestepper.A_implicit[2, 2]
        mass = spdiagm(0 => ones(ComplexF64, n))
        linear = spdiagm(0 => fill(ComplexF64(-inv(stage_scale)), n))
        @test iszero(norm(mass + stage_scale * linear, Inf))
        Tarang.set_compiled_matrices!(problem, linear, mass)

        exception = try
            step!(solver, dt)
            nothing
        catch err
            err
        end

        @test exception !== nothing
        message = lowercase(sprint(showerror, exception))
        @test occursin("imex", message)
        @test occursin("singular", message)
        @test solver.iteration == 0
    end
end

if review_case("rhs_parse")
    @testset "malformed RHS aborts solver construction" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(Domain(dist, (basis,)), "u")
        problem = IVP([u])
        add_equation!(problem, "∂t(u) = invsqrtlap()")

        exception = try
            InitialValueSolver(problem, RK222(); dt=1e-3)
            nothing
        catch err
            err
        end

        @test exception isa ArgumentError
        @test occursin("RHS", sprint(showerror, exception))
        @test occursin("invsqrtlap requires an operand", sprint(showerror, exception))
    end
end

if review_case("rhs_build")
    @testset "RHS matrix-expression build failures abort" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(Domain(dist, (basis,)), "u")
        problem = IVP([u])
        add_parameters!(problem; malformed_rhs=MalformedReviewRHS(NonIterableReviewNode()))
        add_equation!(problem, "0 = malformed_rhs")

        exception = try
            Tarang.build_matrix_expressions!(problem)
            nothing
        catch err
            err
        end

        @test exception isa ArgumentError
        message = sprint(showerror, exception)
        @test occursin("Failed to build matrix expressions", message)
        @test occursin("malformed_rhs", message)
    end
end

if review_case("etd_dae")
    @testset "ETD rejects singular-mass DAE systems" begin
        state = Tarang.TimestepperState(ETD_RK222(), 0.1, ScalarField[])
        linear = Matrix{ComplexF64}(I, 2, 2)
        mass = ComplexF64[1 0; 0 0]

        exception = try
            Tarang._get_linear_operator_eff!(state, linear, mass)
            nothing
        catch err
            err
        end

        @test exception isa ArgumentError
        message = sprint(showerror, exception)
        @test occursin("ETD", message)
        @test occursin("singular mass", lowercase(message))
    end
end

if review_case("subproblem_explicit_first")
    @testset "subproblem RK preserves the explicit first-stage state" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
        zbasis = ChebyshevT(coords["z"]; size=6, bounds=(0.0, 1.0))
        domain = Domain(dist, (zbasis,))
        b = ScalarField(domain, "b")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        ez = only(unit_vector_fields(coords, dist))
        lift_basis = derivative_basis(zbasis, 1)
        tau_lift(a) = lift(a, lift_basis, -1)
        grad_b = grad(b) + ez * tau_lift(tau1)

        problem = IVP([b, tau1, tau2])
        add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
        add_equation!(problem,
                      "∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = 0")
        add_bc!(problem, "b(z=0) = 0")
        add_bc!(problem, "b(z=1) = 0")
        solver = InitialValueSolver(problem, RKSMR(); dt=1e-3)
        sp = only(Tarang.compiled_subproblems(problem))

        mass = Matrix(sp.M_min)
        @test rank(mass) < size(mass, 2)
        tau_state = ComplexF64.(nullspace(mass)[:, 1])
        @test norm(tau_state) > 0
        @test norm(mass * tau_state) < 1e-13

        # Scatter the actual mass-null vector through the assembled subproblem.
        # Its nonzero component is held by a real 0-D tau stash, not a synthetic
        # buffer, and gather/scatter must round-trip it before the RK stage.
        Tarang.scatter_inputs(sp, tau_state, solver.state)
        @test any(stash -> norm(stash) > 0, values(sp.runtime.zero_dim_stash))
        gathered_before = similar(tau_state)
        Tarang.gather_inputs!(gathered_before, sp, solver.state)
        @test gathered_before ≈ tau_state atol=1e-13

        # Stop the real per-mode driver immediately after its first stage solve,
        # before any later RK stage or final update can overwrite the evidence.
        # Emptying the already-compiled IR disables the algebraic-refresh hook;
        # the assembled Subproblem matrices and lazy plan remain intact. Prime
        # the derived equation-space metadata first: whether it was populated
        # during construction used to depend on whether the CUDA extension had
        # already loaded, making this deliberate internal mutation order-sensitive.
        Tarang._subproblem_eqn_sizes(sp)
        empty!(problem.equation_data)
        solver.rhs_plan.exprs[1] = StopAfterExplicitFirstStage()
        state = Tarang.TimestepperState(solver.timestepper, solver.dt, solver.state)
        exception = try
            Tarang.step_subproblem_rk!(state, solver,
                                       Tarang.compiled_subproblems(problem))
            nothing
        catch err
            err
        end
        @test exception isa ArgumentError
        @test occursin("StopAfterExplicitFirstStage", sprint(showerror, exception))

        gathered_after = similar(tau_state)
        Tarang.gather_inputs!(gathered_after, sp, solver.state)
        @test gathered_after ≈ tau_state atol=1e-13
    end
end

if review_case("subproblem_explicit_first_batched")
    @testset "batched subproblem RK preserves the explicit first-stage state" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
        xbasis = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
        zbasis = ChebyshevT(coords["z"]; size=6, bounds=(0.0, 1.0))
        domain = Domain(dist, (xbasis, zbasis))
        b = ScalarField(domain, "b")
        tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)
        tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)
        _, ez = unit_vector_fields(coords, dist)
        lift_basis = derivative_basis(zbasis, 1)
        tau_lift(a) = lift(a, lift_basis, -1)
        grad_b = grad(b) + ez * tau_lift(tau1)

        problem = IVP([b, tau1, tau2])
        add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
        add_equation!(problem,
                      "∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = 0")
        add_bc!(problem, "b(z=0) = 0")
        add_bc!(problem, "b(z=1) = 0")
        solver = InitialValueSolver(problem, RKSMR(); dt=1e-3,
                                    batched_modes=true)
        subproblems = Tarang.compiled_subproblems(problem)
        state = Tarang.TimestepperState(solver.timestepper, solver.dt, solver.state)
        foreach(field -> ensure_layout!(field, :c), solver.state)
        plan = Tarang._sp_rk_batch_plan!(state, solver, subproblems, solver.state)
        @test plan !== nothing
        @test !isempty(plan.batches)

        sp_idx = first(plan.batches[1].sp_indices)
        sp = subproblems[sp_idx]
        mass = Matrix(sp.M_min)
        @test rank(mass) < size(mass, 2)
        tau_state = ComplexF64.(nullspace(mass)[:, 1])
        Tarang.scatter_inputs(sp, tau_state, solver.state)
        @test any(field -> maximum(abs, Tarang.get_coeff_data(field)) > 0,
                  (tau1, tau2))
        gathered_before = similar(tau_state)
        Tarang.gather_inputs!(gathered_before, sp, solver.state)
        @test gathered_before ≈ tau_state atol=1e-13

        foreach(Tarang._subproblem_eqn_sizes, subproblems)
        empty!(problem.equation_data)
        solver.rhs_plan.exprs[1] = StopAfterExplicitFirstStage()
        exception = try
            Tarang.step_subproblem_rk!(state, solver, subproblems)
            nothing
        catch err
            err
        end
        @test exception isa ArgumentError
        @test occursin("StopAfterExplicitFirstStage", sprint(showerror, exception))

        gathered_after = similar(tau_state)
        Tarang.gather_inputs!(gathered_after, sp, solver.state)
        @test gathered_after ≈ tau_state atol=1e-13
    end
end

if review_case("fourier_coeff_current")
    @testset "local Fourier differentiation honors coefficient-current storage" begin
        n = 16
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
        basis = RealFourier(coords["x"]; size=n, bounds=(0.0, 2π))
        domain = Domain(dist, (basis,))
        operand = ScalarField(domain, "operand")
        coefficient_source = ScalarField(domain, "coefficient_source")
        x = 2π .* collect(0:(n - 1)) ./ n

        operand["g"] = sin.(x)              # deliberately stale grid storage
        ensure_layout!(operand, :c)
        coefficient_source["g"] = sin.(3 .* x)
        ensure_layout!(coefficient_source, :c)
        copyto!(Tarang.get_coeff_data(operand),
                Tarang.get_coeff_data(coefficient_source))
        @test operand.current_layout == :c

        derivative = ScalarField(domain, "derivative")
        Tarang.evaluate_fourier_derivative!(derivative, operand, 1, 1, :g)
        @test Tarang.get_grid_data(derivative) ≈ 3 .* cos.(3 .* x) atol=1e-11
    end
end

if review_case("problem_reuse_bc")
    @testset "merging BC equations is idempotent for reused Problems" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
        basis = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
        u = ScalarField(Domain(dist, (basis,)), "u")
        problem = IVP([u])
        add_equation!(problem, "∂t(u) = 0")
        add_bc!(problem, "u(z=0) = 1")

        Tarang._merge_boundary_conditions!(problem)
        first_equations = copy(problem.equations)
        first_indices = copy(problem.bc_manager.bc_equation_indices)

        # Solver construction invokes the merge every time the same Problem is
        # reused.  Repeating it must only relink manager metadata.
        Tarang._merge_boundary_conditions!(problem)
        @test problem.equations == first_equations
        @test problem.bc_manager.bc_equation_indices == first_indices
    end
end

if review_case("deterministic_forcing")
    @testset "registered DeterministicForcing receives its field grid" begin
        n = 8
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64, device=Tarang.CPU())
        basis = RealFourier(coords["x"]; size=n, bounds=(0.0, 2π))
        u = ScalarField(Domain(dist, (basis,)), "u")
        problem = IVP([u])
        add_equation!(problem, "∂t(u) = 0")
        forcing = DeterministicForcing(
            (x, t, parameters) -> (@. parameters[:amplitude] * sin(x) + t),
            (n,);
            parameters=Dict(:amplitude => 2.0),
        )
        add_stochastic_forcing!(problem, :u, forcing)
        solver = InitialValueSolver(problem, RK222(); dt=0.05)

        Tarang._update_registered_forcings!(solver, 0.25, solver.dt)
        x = 2π .* collect(0:(n - 1)) ./ n
        @test forcing.cached_forcing ≈ @. 2.0 * sin(x) + 0.25
    end
end

if review_case("combined_schedules")
    @testset "Dictionary and Virtual schedules combine cadences with OR" begin
        dictionary = DictionaryHandler(cadence=10, sim_dt=0.1)
        @test Tarang.should_write(dictionary, 0.0, 0.05, 10)
        @test Tarang.should_write(dictionary, 0.0, 0.10, 1)
        @test !Tarang.should_write(dictionary, 0.0, 0.05, 1)

        mktempdir() do directory
            virtual = VirtualFileHandler(directory, "review_schedule";
                                         cadence=10, sim_dt=0.1)
            @test Tarang.should_write(virtual, 0.0, 0.05, 10)
            @test Tarang.should_write(virtual, 0.0, 0.10, 1)
            @test !Tarang.should_write(virtual, 0.0, 0.05, 1)
        end
    end
end
