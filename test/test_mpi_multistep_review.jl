using Test, Tarang, MPI, PencilArrays, LinearAlgebra

MPI.Initialized() || MPI.Init()
const MSREV_COMM = MPI.COMM_WORLD
const MSREV_RANK = MPI.Comm_rank(MSREV_COMM)
const MSREV_NP = MPI.Comm_size(MSREV_COMM)

if MSREV_NP < 2
    @warn "MPI multistep review regressions need at least two ranks"
    MPI.Finalize()
    exit(0)
end

function msrev_mpi_run(stepper, dts)
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    zb = ChebyshevT(coords["z"]; size=20, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2pi))
    u = ScalarField(Domain(dist, (zb, xb)), "u")
    tau1 = ScalarField(dist, "tau1", (), Float64)
    tau2 = ScalarField(dist, "tau2", (), Float64)
    lb2 = derivative_basis(zb, 2)
    problem = InitialValueProblem([u, tau1, tau2])
    add_parameters!(problem; lambda=0.3,
                    l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
    add_equation!(problem, "dt(u) + lambda*u + l1 + l2 = -u")
    add_bc!(problem, "u(z=0) = 0")
    add_bc!(problem, "u(z=1) = 0")
    zs = [0.5*(1-cos(pi*(j-1)/19)) for j in 1:20]
    xs = [2pi*(j-1)/8 for j in 1:8]
    initial = [sin(pi*z)*(1+0.5cos(x)) for z in zs, x in xs]
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    ax = PencilArrays.pencil(gd).axes_local
    parent(gd) .= initial[ax...]
    solver = InitialValueSolver(problem, stepper; dt=first(dts))
    for dt in dts
        step!(solver, dt)
    end
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    local_initial = initial[PencilArrays.pencil(gd).axes_local...]
    numerator = MPI.Allreduce(dot(local_initial, parent(gd)), MPI.SUM, MSREV_COMM)
    denominator = MPI.Allreduce(dot(local_initial, local_initial), MPI.SUM, MSREV_COMM)
    return numerator / denominator
end

@testset "MPI multistep review regressions" begin
@testset "MPI MCNAB2/CNLF2 subproblem order" begin
    for ts in (Tarang.MCNAB2(), Tarang.CNLF2())
        coarse = msrev_mpi_run(ts, fill(0.1, 10))
        fine = msrev_mpi_run(ts, fill(0.05, 20))
        exact = exp(-1.3)
        @test log2(abs(coarse-exact) / abs(fine-exact)) > 1.8
        dts = [0.02*(1+0.2sin(2pi*(i-1)/20)) for i in 1:20]
        variable = msrev_mpi_run(ts, dts)
        @test abs(variable-exp(-1.3sum(dts))) < 5e-4
    end
end

@testset "MPI public interpreted RHS works in multistep field history" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=ComplexF64, architecture=CPU())
    xb = ComplexFourier(coords["x"]; size=8, bounds=(0.0, 2pi))
    yb = ComplexFourier(coords["y"]; size=8, bounds=(0.0, 2pi))
    u = ScalarField(Domain(dist, (xb, yb)), "u")
    u["g"] .= 1.0+0im
    problem = InitialValueProblem([u])
    add_parameters!(problem; gamma=1im)
    add_equation!(problem, "dt(u) = gamma*u")
    solver = InitialValueSolver(problem, CNAB2(); dt=0.01, rhs_fallback=:interpreted)
    @test solver.rhs_plan === nothing || !solver.rhs_plan.is_compiled
    for _ in 1:20
        step!(solver)
    end
    err = MPI.Allreduce(maximum(abs.(u["g"] .- exp(0.2im))), MPI.MAX, MSREV_COMM)
    @test err < 1e-4
end
end

MPI.Barrier(MSREV_COMM)
MPI.Finalized() || MPI.Finalize()
