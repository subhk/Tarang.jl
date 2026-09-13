using Test, Tarang, LinearAlgebra, MPI

# Both domains reduce to a scalar ODE, while only the bounded domain uses the
# per-mode tau solver. The global path is an independent scheme reference.
function msrev_solver(stepper, dt; bounded=true)
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; dtype=Float64, architecture=CPU(), comm=MPI.COMM_SELF)
    if bounded
        basis = ChebyshevT(coords["z"]; size=20, bounds=(0.0, 1.0))
        u = ScalarField(Domain(dist, (basis,)), "u")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        lb2 = derivative_basis(basis, 2)
        problem = InitialValueProblem([u, tau1, tau2])
        add_parameters!(problem; lambda=0.3,
                        l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
        add_equation!(problem, "dt(u) + lambda*u + l1 + l2 = -u")
        add_bc!(problem, "u(z=0) = 0")
        add_bc!(problem, "u(z=1) = 0")
        set!(u, (z,) -> sin(pi*z))
    else
        basis = RealFourier(coords["z"]; size=8, bounds=(0.0, 2pi))
        u = ScalarField(Domain(dist, (basis,)), "u")
        problem = InitialValueProblem([u])
        add_parameters!(problem; lambda=0.3)
        add_equation!(problem, "dt(u) + lambda*u = -u")
        u["g"] .= 1.0
    end
    initial = copy(u["g"])
    solver = InitialValueSolver(problem, stepper; dt)
    return solver, u, initial
end

function msrev_run(stepper, dts; bounded=true)
    solver, u, initial = msrev_solver(stepper, first(dts); bounded)
    amplitudes = Float64[]
    for dt in dts
        step!(solver, dt)
        push!(amplitudes, dot(initial, u["g"]) / dot(initial, initial))
    end
    return amplitudes
end

@testset "MCNAB2/CNLF2 use their subproblem schemes after startup" begin
    for ts in (Tarang.MCNAB2(), Tarang.CNLF2())
        @testset "$(nameof(typeof(ts))) converges at second order" begin
            coarse = last(msrev_run(ts, fill(0.1, 10)))
            fine = last(msrev_run(ts, fill(0.05, 20)))
            exact = exp(-1.3)
            @test log2(abs(coarse-exact) / abs(fine-exact)) > 1.8
        end
    end

    # Include step-size changes: the subproblem path
    # must honor the selected formula, not substitute ordinary CNAB2.
    for ts in (Tarang.MCNAB2(), Tarang.CNLF2())
        dts = [0.03, 0.04, 0.05, 0.04, 0.025, 0.035, 0.045, 0.03]
        @testset "$(typeof(ts)) variable dt" begin
            bounded = msrev_run(ts, dts)
            global_reference = msrev_run(ts, dts; bounded=false)
            @test bounded ≈ global_reference atol=2e-12 rtol=2e-12
        end
    end
end
