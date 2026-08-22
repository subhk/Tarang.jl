# Guard: 3-D Chebyshev-Fourier solve on a 2-D (pencil) process mesh.
#
# `dist.pencil_solve` decomposes the trailing Fourier axes while the PencilFFT
# output pencil decomposes (Chebyshev, first Fourier). On a 1-D process mesh those
# differ in the single decomposed slot, so one `PencilArrays.transpose!` sufficed.
# On a 2-D mesh they differ in BOTH slots and every fft<->solve transpose threw
#   ArgumentError: pencil decompositions must differ in at most one dimension.
#                  Got decomposed dimensions (1, 2) and (2, 3).
# so a 3-D Chebyshev-Fourier run could not use a pencil decomposition at all.
# `transpose_multistep!` hops through intermediate single-swap pencils instead.
#
# Reference = the serial (np=1) result of this exact problem.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs != 4
    rank == 0 && @warn "3D Cheb-Fourier pencil-mesh test needs exactly 4 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const SUMSQ_REF = 280.2120219162331
const MAX_REF   = 1.730722575835805

_loc(f) = get_grid_data(f) isa PencilArrays.PencilArray ? parent(get_grid_data(f)) : get_grid_data(f)
function _assign!(field, g)
    d = get_grid_data(field)
    d isa PencilArrays.PencilArray ? (parent(d) .= g[PencilArrays.pencil(d).axes_local...]) : (d .= g)
end

function _run(mesh)
    Nz, Nx, Ny, Lz, dt, nsteps = 12, 8, 8, 1.0, 1e-3, 10
    coords = CartesianCoordinates("z", "x", "y")
    dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=3/2)
    yb = RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2π), dealias=3/2)
    b = ScalarField(Domain(dist, (zb, xb, yb)), "b")
    tau1 = ScalarField(dist, "tau_b1", (), Float64)
    tau2 = ScalarField(dist, "tau_b2", (), Float64)
    ez = unit_vector_fields(coords, dist)[1]
    lift_basis = derivative_basis(zb, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * τ_lift(tau1)
    problem = IVP([b, tau1, tau2])
    add_parameters!(problem, kappa=0.1, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
    add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt=dt)
    xf = [2π*(i-1)/Nx for i in 1:Nx]; yf = [2π*(j-1)/Ny for j in 1:Ny]
    zf = [Lz/2*(1 - cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
    g0 = [sin(π*zf[iz]/Lz) * (1 + 0.5cos(3xf[ix]) + 0.3sin(3yf[iy]))
          for iz in 1:Nz, ix in 1:Nx, iy in 1:Ny]
    ensure_layout!(b, :g); _assign!(b, g0); ensure_layout!(b, :c)
    for _ in 1:nsteps; step!(solver, dt); end
    ensure_layout!(b, :g)
    lv = _loc(b)
    return (MPI.Allreduce(sum(abs2, lv), MPI.SUM, comm),
            MPI.Allreduce(maximum(abs, lv), MPI.MAX, comm),
            dist)
end

@testset "3D Cheb-Fourier IVP on a 2-D process mesh (np=4)" begin
    sumsq, bmax, dist = _run((2, 2))
    # The path only stays exercised while the two pencils really differ in BOTH
    # slots; if a future layout change makes them differ in one, this test would
    # quietly stop covering transpose_multistep!.
    fftdec = Tuple(PencilArrays.decomposition(dist.pencil_fft_output))
    slvdec = Tuple(PencilArrays.decomposition(dist.pencil_solve))
    @test length(fftdec) == 2
    @test count(i -> fftdec[i] != slvdec[i], 1:2) == 2
    @test isapprox(sumsq, SUMSQ_REF; rtol=1e-10)
    @test isapprox(bmax, MAX_REF; rtol=1e-10)
end

@testset "3D Cheb-Fourier IVP: 1-D slab mesh agrees (np=4)" begin
    sumsq, bmax, _ = _run((4,))
    @test isapprox(sumsq, SUMSQ_REF; rtol=1e-10)
    @test isapprox(bmax, MAX_REF; rtol=1e-10)
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
