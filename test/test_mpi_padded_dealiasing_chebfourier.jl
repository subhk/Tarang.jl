# Guard: distributed MIXED Chebyshev-Fourier padded dealiasing == serial (np>=2).
# Only the Fourier axes are 3/2-padded (transpose-pad when decomposed); the
# Chebyshev axis keeps its nodal grid — matching serial evaluate_padded_multiply
# (which pads only fourier_dims). Was distributed≠serial by ~0.5 (the Fourier axis
# is the decomposed one in a Cheb-first MPI layout). Round-7 audit 2026-06-23.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI Cheb-Fourier dealiasing test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const Nz = 8
const Nx = 12
const SUMSQ_REF = 16.432510723800412
const MAX_REF   = 1.6471787788618577

@testset "Distributed Cheb-Fourier padded dealiasing == serial (np=$nprocs)" begin
    coords = CartesianCoordinates("z", "x")            # Chebyshev FIRST (MPI layout)
    dist = Distributor(coords)
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=3/2)
    u = ScalarField(dist, "u", (zb, xb), Float64)
    ensure_layout!(u, :g)

    zf = [0.5*(1-cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
    xf = [(i-1)*2π/Nx for i in 1:Nx]
    u0 = [sin(π*zf[iz])*(cos(xf[ix]) + 0.5sin(3xf[ix])) for iz in 1:Nz, ix in 1:Nx]
    gd = get_grid_data(u)
    if gd isa PencilArrays.PencilArray
        gv = PencilArrays.global_view(gd); for I in CartesianIndices(gv); gv[I] = u0[I]; end
    else
        Tarang.get_cpu_data(gd) .= u0
    end

    ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=3/2)
    p = Tarang.evaluate_transform_multiply(u, u, ev; result_layout=:g)
    ensure_layout!(p, :g)
    pg = get_grid_data(p)
    g = pg isa PencilArrays.PencilArray ? PencilArrays.gather(pg) : Array(Tarang.get_cpu_data(pg))

    if rank == 0
        @test isapprox(sum(abs2, g), SUMSQ_REF; rtol=1e-10)
        @test isapprox(maximum(g),  MAX_REF;   rtol=1e-10)
    end
end
