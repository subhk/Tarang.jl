# Guard: distributed 3D MIXED Chebyshev-Fourier-Fourier padded dealiasing == serial.
#
# Regression for the round-7 bug where the result-alignment transpose threw
# `ArgumentError: pencil decompositions must differ in at most one dimension`
# (decomp (2,3) vs (3,2)) for a Cheb-first 3D field: matching only the local axis
# left res in the right decomp SET but wrong ORDER. Fixed by aligning res to the
# result grid pencil's EXACT ordered decomposition via a sequence of single swaps.
# This is the standard 3D channel-flow layout (Cheb wall-normal + 2 Fourier). np>=2.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI 3D mixed dealiasing test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const Nz = 6
const Ny = 8
const Nx = 8
const SUMSQ_REF = 49.79384474046044
const MAX_REF   = 1.5333421886891991

@testset "Distributed 3D Cheb-Fourier-Fourier dealiasing == serial (np=$nprocs)" begin
    coords = CartesianCoordinates("z", "y", "x")          # Chebyshev FIRST
    dist = Distributor(coords)
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    yb = RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2π), dealias=3/2)
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=3/2)
    u = ScalarField(Domain(dist, (zb, yb, xb)), "u")
    ensure_layout!(u, :g)

    zf = [0.5*(1-cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
    yf = [(j-1)*2π/Ny for j in 1:Ny]
    xf = [(i-1)*2π/Nx for i in 1:Nx]
    A = [sin(π*zf[iz])*(cos(yf[iy]) + 0.4sin(2xf[ix])) for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx]
    gd = get_grid_data(u)
    if gd isa PencilArrays.PencilArray
        gv = PencilArrays.global_view(gd); for I in CartesianIndices(gv); gv[I] = A[I]; end
    else
        Tarang.get_cpu_data(gd) .= A
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
