using Test
using Tarang
using MPI

function _bc_context_problem(kind, bottom, top; bounds=(0.0, 1.0), parameters=NamedTuple(), three_d=false)
    coords = three_d ? CartesianCoordinates("x", "y", "z") : CartesianCoordinates("x", "z")
    dist = Distributor(coords; comm=MPI.COMM_SELF, dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=16, bounds=bounds)
    tangential = three_d ? (xb, RealFourier(coords["y"]; size=6, bounds=(0.0, 2π))) : (xb,)
    u = ScalarField(dist, "u", (tangential..., zb), Float64)
    tau1 = ScalarField(dist, "tau1", tangential, Float64)
    tau2 = ScalarField(dist, "tau2", tangential, Float64)
    lift_basis = derivative_basis(zb, 2)
    problem = kind([u, tau1, tau2])
    add_parameters!(problem; l1=lift(tau1, lift_basis, -1), l2=lift(tau2, lift_basis, -2), parameters...)
    equation = kind === InitialValueProblem ? "dt(u) - 0.1*lap(u) + l1 + l2 = 0" : "lap(u) + l1 + l2 = 0"
    add_equation!(problem, equation)
    add_bc!(problem, bottom)
    add_bc!(problem, top)
    return problem, u, coords
end

@testset "steady solvers evaluate spatial boundary values" begin
    xs = (0:7) .* (2π/8)
    zs = (1 .- cos.(π .* (0:15) ./ 15)) ./ 2
    cases = (
        ("raw Dirichlet", "u(z=0)=sin(x)", (x,z) -> sin(x)*sinh(1-z)/sinh(1)),
        ("structured Dirichlet", dirichlet_bc("u", "z", 0.0, "sin(x)"), (x,z) -> sin(x)*sinh(1-z)/sinh(1)),
        ("Neumann", "d(u,z)(z=0)=cos(x)", (x,z) -> -cos(x)*sinh(1-z)/cosh(1)),
        ("raw Robin", "1*u(z=0)+0.5*∂z(u)(z=0)=sin(x)",
            (x,z) -> sin(x)*sinh(1-z)/(sinh(1)-0.5cosh(1))),
        ("structured Robin", robin_bc("u", "z", 0.0, 1.0, 0.5, "sin(x)"),
            (x,z) -> sin(x)*sinh(1-z)/(sinh(1)-0.5cosh(1))),
    )
    for kind in (LinearBoundaryValueProblem, NonlinearBoundaryValueProblem), (name, bottom, exact) in cases
        @testset "$(nameof(kind)) $name" begin
            problem, u, _ = _bc_context_problem(kind, bottom, "u(z=1)=0")
            solve!(BoundaryValueSolver(problem))
            @test Array(grid_data!(u)) ≈ [exact(x,z) for x in xs, z in zs] atol=1e-9
        end
    end
end

@testset "raw and structured moving Robin values agree" begin
    boundaries = (
        "1*u(z=0)+0.5*∂z(u)(z=0)=2*t",
        "(1.0)*u(z=0) + (0.5)*d(u,z)(z=0) = 2*t",
        robin_bc("u", "z", 0.0, 1.0, 0.5, "2*t"),
        "h*u(z=0)+k*∂z(u)(z=0)=2*t",
        robin_bc("u", "z", 0.0, "h", "k", "2*t"),
    )
    for bottom in boundaries
        problem, u, coords = _bc_context_problem(InitialValueProblem, bottom, "u(z=1)=0";
                                                parameters=(h=1.0, k=0.5))
        solver = InitialValueSolver(problem, RK222(); dt=0.01)
        for _ in 1:3
            step!(solver)
        end
        values = Array(grid_data!(u))
        derivative = Array(grid_data!(evaluate(Differentiate(u, coords["z"], 1))))
        @test values[:,1] .+ 0.5 .* derivative[:,1] ≈ fill(2solver.sim_time, 8) atol=1e-10
        @test maximum(abs, values[:,end]) < 1e-10
        @test count(bc -> bc isa RobinBC, problem.bc_manager.conditions) == 1
    end
end

@testset "dynamic boundary expressions use registered parameters" begin
    for bottom in ("u(z=0)=sin(omega*t)", "u(z=0)=amp*cos(k*x)*sin(omega*t)")
        parameters = (omega=2.0, amp=1.25, k=1.0)
        problem, u, _ = _bc_context_problem(InitialValueProblem, bottom, "u(z=1)=0"; parameters)
        solver = InitialValueSolver(problem, RK222(); dt=0.01)
        for _ in 1:3
            step!(solver)
        end
        expected = occursin("amp", bottom) ? 1.25 .* cos.((0:7) .* (2π/8)) .* sin(2solver.sim_time) : fill(sin(2solver.sim_time), 8)
        @test Array(grid_data!(u))[:,1] ≈ expected atol=1e-10
    end
    problem, u, _ = _bc_context_problem(InitialValueProblem, "u(z=0)=amp*x+z+t", "u(z=1)=0";
        parameters=(amp=1.25, x=999.0, z=-100.0, t=99.0))
    solver = InitialValueSolver(problem, RK222(); dt=0.01)
    step!(solver)
    @test Array(grid_data!(u))[:,1] ≈ 1.25 .* (0:7) .* (2π/8) .+ solver.sim_time atol=1e-10
    # Time is authoritative even if an explicitly registered coordinate entry
    # happens to use the same name.
    @test evaluate_expression("t+x", 0.25, Dict("t" => 99.0, "x" => 2.0)) == 2.25
end

@testset "normal coordinates bind to symbolic walls on shifted intervals" begin
    parameters = (lower=-1.0, upper=2.0, z=100.0)
    for kind in (LinearBoundaryValueProblem, NonlinearBoundaryValueProblem)
        problem, u, _ = _bc_context_problem(kind, "u(z=lower)=z", "u(z=upper)=z";
                                           bounds=(-1.0, 2.0), parameters)
        solve!(BoundaryValueSolver(problem))
        zs = -1 .+ 3 .* (1 .- cos.(π .* (0:15) ./ 15)) ./ 2
        @test Array(grid_data!(u)) ≈ [z for _ in 1:8, z in zs] atol=1e-10
        @test get(problem.bc_manager.coordinate_fields, "z", Float64[]) ≈ zs
    end
end

@testset "normal binding preserves the two tangential axes of a 3D wall" begin
    problem, u, _ = _bc_context_problem(InitialValueProblem, "u(z=lower)=z", "u(z=upper)=z*sin(x)*cos(y)";
        bounds=(-1.0, 2.0), parameters=(lower=-1.0, upper=2.0), three_d=true)
    solver = InitialValueSolver(problem, RK222(); dt=0.01)
    for _ in 1:3
        step!(solver)
    end
    values = Array(grid_data!(u))
    @test values[:,:,1] ≈ fill(-1.0, 8, 6) atol=1e-10
    @test values[:,:,end] ≈ [2sin(x)*cos(y) for x in (0:7).*(2π/8), y in (0:5).*(2π/6)] atol=1e-10
    @test size(problem.bc_manager.coordinate_fields["x"]) == (8, 1)
    @test size(problem.bc_manager.coordinate_fields["y"]) == (1, 6)
    @test length(problem.bc_manager.coordinate_fields["z"]) == 16
end

@testset "boundary context preserves callbacks and their failures" begin
    manager = BoundaryConditionManager()
    coords = Dict{String,Any}("x" => [0.0, 0.5], "y" => [1.0, 2.0])
    space_value = SpaceDependentValue("", ["x", "y"], (x,y) -> x .+ y)
    bc = dirichlet_bc("u", "z", 2.0, space_value)
    @test evaluate_bc_value(manager, bc, 0.0, coords) == [1.0, 2.5]
    @test Set(keys(coords)) == Set(["x", "y"])

    full_coords = Dict{String,Any}("x" => [0.0, 0.5], "z" => [-1.0, 0.0, 2.0])
    callback = (t, c::AbstractDict) -> c["x"] .+ c["z"] .+ t
    moving = dirichlet_bc("u", "z", 2.0, callback; time_dependent=true)
    bound_result = try
        evaluate_bc_value(manager, moving, 0.25, full_coords)
    catch error
        error
    end
    @test bound_result == [2.25, 2.75]
    @test full_coords["z"] == [-1.0, 0.0, 2.0]

    sentinel = ErrorException("boundary callback failure")
    throwing = dirichlet_bc("u", "z", 2.0, (t, c::AbstractDict) -> throw(sentinel); time_dependent=true)
    caught = try
        evaluate_bc_value(manager, throwing, 0.0, full_coords)
        nothing
    catch error
        error
    end
    @test caught === sentinel
    @test_throws ArgumentError robin_bc("u", "z", 0.0, "1+t", 1.0, 0.0)
    @test_throws ArgumentError robin_bc("u", "z", 0.0, 1.0, "1+x", 0.0)
end
