# Guard: 3D nonlinear IVP solve distributed == serial (np>=2; np=4 = 2D mesh).
#
# End-to-end integration check: a 3D Burgers solve (RK222 IMEX — nonlinear advection
# -u·∇u via the 3/2 padded distributed dealiasing + implicit ν∇²u) must give the SAME
# final state serial vs distributed. Stronger than the standalone-product guards: it
# exercises the dealiasing path INSIDE the timestepper RHS across many steps, catching
# dispatch/integration regressions the product unit-guards would miss. Round-7.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI 3D IVP dealiasing test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const N = 8
const NU = 0.05
const DT = 1e-3
const NSTEPS = 5
const SUMSQ_REF = 159.84007878174114
const MAX_REF   = 1.4991659750206703

@testset "Distributed 3D Burgers IVP == serial (np=$nprocs)" begin
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    zb = RealFourier(coords["z"]; size=N, bounds=(0.0, 2π), dealias=3/2)
    dom = Domain(dist, (xb, yb, zb))
    u = ScalarField(dom, "u")

    xg = [(i-1)*2π/N for i in 1:N]
    u0 = [sin(xg[i])*cos(xg[j]) + 0.5cos(xg[k])*sin(xg[i]) for i in 1:N, j in 1:N, k in 1:N]
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    if gd isa PencilArrays.PencilArray
        gv = PencilArrays.global_view(gd); for I in CartesianIndices(gv); gv[I] = u0[I]; end
    else
        Tarang.get_cpu_data(gd) .= u0
    end

    prob = IVP([u]); add_parameters!(prob; nu=NU)
    add_equation!(prob, "∂t(u) - nu*lap(u) = -u*∂x(u) - u*∂y(u) - u*∂z(u)")
    solver = InitialValueSolver(prob, RK222(); dt=DT)
    for _ in 1:NSTEPS
        step!(solver, DT)
    end

    ensure_layout!(u, :g)
    pg = get_grid_data(u)
    g = pg isa PencilArrays.PencilArray ? PencilArrays.gather(pg) : Array(Tarang.get_cpu_data(pg))
    if rank == 0
        @test isapprox(sum(abs2, g), SUMSQ_REF; rtol=1e-9)
        @test isapprox(maximum(g),  MAX_REF;   rtol=1e-9)
    end
end
