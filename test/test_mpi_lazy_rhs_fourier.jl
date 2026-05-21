using Test
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
    rank == 0 && @warn "MPI lazy RHS Fourier test requires at least 2 processes"
    exit(0)
end

rank == 0 && println("=" ^ 60)
rank == 0 && println("MPI Lazy RHS Fourier Tests")
rank == 0 && println("=" ^ 60)

function _local_grid_pair(field, N)
    data = get_grid_data(field)
    x_full = [2π * (i - 1) / N for i in 1:N]
    y_full = [2π * (j - 1) / N for j in 1:N]
    if data isa PencilArrays.PencilArray
        axes_local = PencilArrays.pencil(data).axes_local
        return x_full[axes_local[1]], y_full[axes_local[2]]
    end
    return x_full, y_full
end

_local_grid_data(field) = get_grid_data(field) isa PencilArrays.PencilArray ?
                          parent(get_grid_data(field)) : get_grid_data(field)

@testset "MPI lazy RHS differentiates later RealFourier axes (rank=$rank)" begin
    N = 32

    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
    xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    domain = Domain(dist, (xbasis, ybasis))

    q = ScalarField(domain, "q")
    problem = IVP([q])
    add_equation!(problem, "∂t(q) = d(q,y)")
    solver = InitialValueSolver(problem, SBDF1(); dt=1e-3)

    @test solver.rhs_plan !== nothing
    @test solver.rhs_plan.is_compiled

    ensure_layout!(q, :g)
    x, y = _local_grid_pair(q, N)
    _local_grid_data(q) .= @. sin(3 * x - 5 * y')

    rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
    ensure_layout!(rhs, :g)

    expected = @. -5 * cos(3 * x - 5 * y')
    local_error = maximum(abs.(_local_grid_data(rhs) .- expected))
    @test MPI.Allreduce(local_error, MPI.MAX, comm) < 1e-10
end

rank == 0 && println("MPI lazy RHS Fourier tests completed")
