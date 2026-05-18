using Test
using Random
using MPI
using PencilArrays
using Tarang

if !MPI.Initialized()
    MPI.Init()
end

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs < 2
    rank == 0 && @warn "MPI stochastic forcing test requires at least 2 processes"
    exit(0)
end

rank == 0 && println("=" ^ 60)
rank == 0 && println("MPI Stochastic Forcing Tests")
rank == 0 && println("=" ^ 60)

@testset "MPI registered forcing participates in lazy RHS (rank=$rank)" begin
    N = 8
    dt = 0.01

    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
    xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xbasis, ybasis))

    q = ScalarField(domain, "q")
    forcing = StochasticForcing(
        field_size=(N, N),
        forcing_rate=0.1,
        k_forcing=3.0,
        dk_forcing=1.0,
        dt=dt,
        rng=MersenneTwister(42)
    )

    problem = IVP([q])
    add_equation!(problem, "∂t(q) = 0")
    add_stochastic_forcing!(problem, :q, forcing)

    solver = InitialValueSolver(problem, RK222(); dt=dt)
    @test solver.rhs_plan !== nothing
    @test solver.rhs_plan.is_compiled

    Tarang._update_registered_forcings!(solver, 0.0, dt)
    rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)
    ensure_layout!(rhs[1], :c)

    coeff_data = get_coeff_data(rhs[1])
    forcing_view = Tarang._matched_forcing_view(forcing, coeff_data)
    @test forcing_view !== nothing

    local_error = maximum(abs.(coeff_data .- forcing_view))
    global_error = MPI.Allreduce(local_error, MPI.MAX, comm)
    @test global_error < 1e-12

    coeff_storage = coeff_data isa PencilArrays.PencilArray ? parent(coeff_data) : coeff_data
    local_sum = sum(abs, coeff_storage)
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, comm)

    @test global_sum > 0
end

rank == 0 && println("MPI stochastic forcing tests completed")
