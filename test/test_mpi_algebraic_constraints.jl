using Test
using MPI
using Tarang
using PencilArrays

if !MPI.Initialized()
    MPI.Init()
end

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs < 2
    rank == 0 && @warn "MPI algebraic constraint test requires at least 2 processes"
    exit(0)
end

rank == 0 && println("=" ^ 60)
rank == 0 && println("MPI Algebraic Constraint Tests")
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

function _local_grid_data(field)
    data = get_grid_data(field)
    return data isa PencilArrays.PencilArray ? parent(data) : data
end

@testset "MPI explicit fallback refreshes algebraic state (rank=$rank)" begin
    N = 16
    dt = 1e-3

    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
    xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xbasis, ybasis))

    q = ScalarField(domain, "q")
    ψ = ScalarField(domain, "ψ")
    u = VectorField(domain, "u")

    problem = IVP([q, ψ, u])
    add_equation!(problem, "∂t(q) = 0")
    add_equation!(problem, "Δ(ψ) - q = 0")
    add_equation!(problem, "u - skew(grad(ψ)) = 0")

    solver = InitialValueSolver(problem, SBDF2(); dt=dt)

    ensure_layout!(q, :g)
    x, y = _local_grid_pair(q, N)
    _local_grid_data(q) .= -2 .* sin.(x) .* cos.(y')

    step!(solver, dt)

    ensure_layout!(ψ, :g)
    ensure_layout!(u.components[1], :g)
    ensure_layout!(u.components[2], :g)

    local_ψ_error = maximum(abs.(_local_grid_data(ψ) .- sin.(x) .* cos.(y')))
    local_ux_error = maximum(abs.(_local_grid_data(u.components[1]) .- sin.(x) .* sin.(y')))
    local_uy_error = maximum(abs.(_local_grid_data(u.components[2]) .- cos.(x) .* cos.(y')))

    @test MPI.Allreduce(local_ψ_error, MPI.MAX, comm) < 1e-10
    @test MPI.Allreduce(local_ux_error, MPI.MAX, comm) < 1e-10
    @test MPI.Allreduce(local_uy_error, MPI.MAX, comm) < 1e-10
end

rank == 0 && println("MPI algebraic constraint tests completed")
