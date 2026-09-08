using Test, Tarang

const _BCJL_OK = try
    @eval using JLArrays, GPUArrays
    true
catch
    false
end

if !_BCJL_OK
    @testset "GPU boundary buffer regressions on JLArray" begin
        @test_skip "JLArrays/GPUArrays unavailable"
    end
else
    include("gpu_boundary_jlarray_support.jl")
    GPUArrays.allowscalar(false)

    function bcjl_boundary_solver(; one_dimensional=false, moving=false, first_order=false)
        coords = one_dimensional ? CartesianCoordinates("z") : CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=_BCJL_ARCH)
        zb = ChebyshevT(coords["z"]; size=12, bounds=(0.0, 1.0))
        xb = one_dimensional ? nothing : RealFourier(coords["x"]; size=8, bounds=(0.0, 2pi))
        bases = one_dimensional ? (zb,) : (xb, zb)
        tau_bases = one_dimensional ? () : (xb,)
        u = ScalarField(Domain(dist, bases), "u")
        tau1 = ScalarField(dist, "tau1", tau_bases, Float64)
        tau2 = ScalarField(dist, "tau2", tau_bases, Float64)
        problem = (moving ? InitialValueProblem : LinearBoundaryValueProblem)([u, tau1, tau2])
        if first_order
            ez = last(unit_vector_fields(coords, dist))
            lb = derivative_basis(zb, 1)
            tau_lift(A) = lift(A, lb, -1)
            grad_u = grad(u) + ez*tau_lift(tau1)
            add_parameters!(problem; grad_u, tau_lift)
            lhs = "-0.1*div(grad_u) + tau_lift(tau2)"
        else
            lb = derivative_basis(zb, 2)
            add_parameters!(problem; l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2))
            lhs = "-0.1*lap(u) + l1 + l2"
        end
        add_equation!(problem, (moving ? "dt(u) + " : "") * lhs * (moving ? " = 1" : " = 0"))
        add_bc!(problem, moving ? "u(z=0) = 1+t" : "u(z=0) = 1")
        add_bc!(problem, moving ? "u(z=1) = 2+t" : "u(z=1) = 2")
        z = (1 .- cos.(pi .* (0:11) ./ 11)) ./ 2
        initial = one_dimensional ? 1 .+ z : [1+zj for xi in 1:8, zj in z]
        copyto!(get_grid_data(u), moving ? initial : zero(initial))
        solver = moving ? InitialValueSolver(problem, RK222(); dt=0.01,
                           matsolver=BoundaryJLHostLU, batched_modes=false) :
                          BoundaryValueSolver(problem; matsolver=BoundaryJLHostLU, batched_modes=false)
        return solver, u, initial
    end

    function bcjl_context_problem(kind, bottom, top; bounds=(0.0,1.0), parameters=NamedTuple())
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64, device=_BCJL_ARCH)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0,2pi))
        zb = ChebyshevT(coords["z"]; size=16, bounds=bounds)
        u = ScalarField(dist, "u", (xb,zb), Float64)
        tau1 = ScalarField(dist, "tau1", (xb,), Float64)
        tau2 = ScalarField(dist, "tau2", (xb,), Float64)
        lb = derivative_basis(zb,2)
        problem = kind([u,tau1,tau2])
        add_parameters!(problem; l1=lift(tau1,lb,-1), l2=lift(tau2,lb,-2), parameters...)
        add_equation!(problem, kind === InitialValueProblem ?
                      "dt(u)-0.1*lap(u)+l1+l2=0" : "lap(u)+l1+l2=0")
        add_bc!(problem,bottom)
        add_bc!(problem,top)
        return problem,u,coords
    end

    @testset "GPU boundary buffer regressions on JLArray" begin
        @testset "empty-basis scalar storage and unit vectors" begin
            for T in (Float64, ComplexF64)
                coords = CartesianCoordinates("x", "z")
                dist = Distributor(coords; dtype=T, device=_BCJL_ARCH)
                scalar = ScalarField(dist, "constant", (), T)
                @test get_grid_data(scalar) isa JLArray{T,1}
                @test get_coeff_data(scalar) isa JLArray{ComplexF64,1}
                @test isempty(get_grid_data(scalar))
                @test isempty(get_coeff_data(scalar))
                units = unit_vector_fields(coords, dist)
                for (i, unit) in enumerate(units), (j, component) in enumerate(unit.components)
                    @test get_grid_data(component) isa JLArray
                    @test Array(get_grid_data(component)) == T[i == j ? 1 : 0]
                    for clone in (copy(component), deepcopy(component))
                        @test get_grid_data(clone) isa JLArray
                        @test Array(get_grid_data(clone)) == Array(get_grid_data(component))
                        @test get_grid_data(clone) !== get_grid_data(component)
                    end
                end
            end
        end

        @testset "constant device field on an explicit RHS" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; dtype=Float64, device=_BCJL_ARCH)
            xb = RealFourier(coords["x"]; size=8, bounds=(0.0,2pi))
            u = ScalarField(Domain(dist,(xb,)),"u")
            a = ScalarField(dist,"a",(),Float64)
            Tarang.set_grid_data!(a,JLArray([3.0]))
            problem = InitialValueProblem([u])
            add_parameters!(problem;a)
            add_equation!(problem,"dt(u) = a")
            solver = InitialValueSolver(problem,RK111();dt=0.1)
            step!(solver)
            ensure_layout!(u,:g)
            @test Array(get_grid_data(u)) ≈ fill(0.3,8) atol=2e-12
        end

        for one_dimensional in (true, false), first_order in (false, true)
            @testset "steady solve: 1D=$one_dimensional unit-vector formulation=$first_order" begin
                solver, u, expected = bcjl_boundary_solver(;one_dimensional,first_order)
                @test !isempty(solver.subproblems)
                @test solve!(solver) === solver
                ensure_layout!(u,:g)
                @test get_grid_data(u) isa JLArray
                @test Array(get_grid_data(u)) ≈ expected atol=2e-10 rtol=2e-10
                for sp in solver.subproblems
                    @test Tarang.gather_inputs(sp,solver.state) isa JLArray
                end
            end
        end

        @testset "device BC gather, override, and moving final constraints" begin
            solver, u, initial = bcjl_boundary_solver(;moving=true)
            sps = solver.problem.parameters["subproblems"]
            for sp in sps
                host = zeros(ComplexF64,size(sp.L_min,1))
                device = JLArray(copy(host))
                Tarang.gather_alg_F!(host,sp)
                Tarang.gather_alg_F!(device,sp)
                @test Array(device) ≈ host
                host_rhs = fill(3.0+2im,length(host))
                device_rhs = JLArray(copy(host_rhs))
                Tarang.apply_bc_override!(host_rhs,host,sp,0.37)
                Tarang.apply_bc_override!(device_rhs,device,sp,0.37)
                @test Array(device_rhs) ≈ host_rhs
            end
            for _ in 1:10
                step!(solver)
            end
            ensure_layout!(u,:g)
            @test Array(get_grid_data(u)) ≈ initial .+ solver.sim_time atol=2e-10
        end

        @testset "parameterized spatial steady boundary values" begin
            xs = (0:7) .* (2pi/8)
            zs = (1 .- cos.(pi .* (0:15) ./ 15)) ./ 2
            cases = (
                ("Dirichlet", "u(z=0)=amp*sin(k*x)",
                 (x,z) -> 1.25sin(x)*sinh(1-z)/sinh(1)),
                ("Neumann", "d(u,z)(z=0)=amp*cos(k*x)",
                 (x,z) -> -1.25cos(x)*sinh(1-z)/cosh(1)),
            )
            for (name,bottom,exact) in cases
                @testset "$name" begin
                    problem,u,_ = bcjl_context_problem(LinearBoundaryValueProblem,bottom,"u(z=1)=0";
                                                      parameters=(amp=1.25,k=1.0))
                    solver = BoundaryValueSolver(problem; matsolver=BoundaryJLHostLU,batched_modes=false)
                    solve!(solver)
                    @test grid_data!(u) isa JLArray
                    @test Array(grid_data!(u)) ≈ [exact(x,z) for x in xs,z in zs] atol=1e-9
                end
            end
        end

        @testset "raw moving Robin uses parameters and projects final constraints" begin
            for bottom in ("1*u(z=0)+0.5*∂z(u)(z=0)=sin(omega*t)",
                           "(1.0)*u(z=0)+(0.5)*d(u,z)(z=0)=sin(omega*t)",
                           "h*u(z=0)+k*∂z(u)(z=0)=sin(omega*t)")
                problem,u,coords = bcjl_context_problem(InitialValueProblem,bottom,"u(z=1)=0";
                                                       parameters=(omega=2.0,h=1.0,k=0.5))
                solver = InitialValueSolver(problem,RK222();dt=0.01,
                                            matsolver=BoundaryJLHostLU,batched_modes=false)
                for _ in 1:3
                    step!(solver)
                end
                values = Array(grid_data!(u))
                # Evaluate only the residual oracle on the CPU twin. JLArray
                # has no Chebyshev derivative backend; the solve above stays
                # on its normal device dispatch path.
                reference = bcjl_twin(u)
                copyto!(grid_data!(reference), values)
                derivative = Array(grid_data!(evaluate(Differentiate(reference,coords["z"],1))))
                @test values[:,1] .+ 0.5 .* derivative[:,1] ≈ fill(sin(2solver.sim_time),8) atol=1e-10
                @test maximum(abs,values[:,end]) < 1e-10
                @test count(bc -> bc isa RobinBC,problem.bc_manager.conditions) == 1
            end
        end

        @testset "normal coordinate boundary values on a shifted interval" begin
            problem,u,_ = bcjl_context_problem(LinearBoundaryValueProblem,
                "u(z=lower)=z","u(z=upper)=z";bounds=(-1.0,2.0),
                parameters=(lower=-1.0,upper=2.0,z=100.0))
            solver = BoundaryValueSolver(problem; matsolver=BoundaryJLHostLU,batched_modes=false)
            solve!(solver)
            zs = -1 .+ 3 .* (1 .- cos.(pi .* (0:15) ./ 15)) ./ 2
            @test Array(grid_data!(u)) ≈ [z for _ in 1:8,z in zs] atol=1e-10
        end

        @testset "nonzero vector heat modes with stress-free device walls" begin
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; dtype=Float64, device=_BCJL_ARCH)
            xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 2pi))
            zb = ChebyshevT(coords["z"]; size=20, bounds=(0.0, 1.0))
            u = VectorField(dist, coords, "u", (xb, zb), Float64)
            tau1 = VectorField(dist, coords, "tau1", (xb,), Float64)
            tau2 = VectorField(dist, coords, "tau2", (xb,), Float64)
            problem = InitialValueProblem([u, tau1, tau2])
            lb = derivative_basis(zb, 2)
            add_parameters!(problem; l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2))
            add_equation!(problem, "dt(u) - lap(u) + l1 + l2 = 0")
            for wall in (0.0, 1.0)
                add_bc!(problem, stress_free_bc("u", "z", wall;
                                               component_coordinates=["x", "z"]))
            end
            xs = (0:3) .* (2pi/4)
            zs = (1 .- cos.(pi .* (0:19) ./ 19)) ./ 2
            initial = ([cos(x)*cos(pi*z) for x in xs, z in zs],
                       [sin(x)*sin(pi*z) for x in xs, z in zs])
            for (c, values) in zip(u.components, initial)
                copyto!(grid_data!(c), values)
            end
            solver = InitialValueSolver(problem, RK222(); dt=1e-4,
                                        matsolver=BoundaryJLHostLU, batched_modes=false)
            for _ in 1:2
                step!(solver)
            end
            for (c, values) in zip(u.components, initial)
                @test grid_data!(c) isa JLArray
                @test Array(grid_data!(c)) ≈ exp(-(1+pi^2)*solver.sim_time) .* values atol=1e-7 rtol=1e-7
            end
            @test maximum(abs, Array(grid_data!(u.components[2]))[:,[1,end]]) < 1e-10
        end
    end
end
