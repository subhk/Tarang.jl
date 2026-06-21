# Guard: distributed mixed Fourier-Chebyshev IVP solve (np >= 2).
#
# A Chebyshev(z)-first / Fourier(x) IVP with an IMPLICIT Laplacian and tau BCs.
# In coeff space the field's PencilArray is laid out (Fourier-axis LOCAL,
# Chebyshev-axis DECOMPOSED, with a permutation) and the distributed PencilFFT
# leaves the Chebyshev axis in GRID space — neither matches the per-Fourier-mode
# tau solve, which needs each rank to own a subset of x-modes plus FULL
# Chebyshev-COEFF z-columns. The fix transposes the coeff data into a
# Chebyshev-local "solve pencil" and applies the local Chebyshev DCT there before
# the gather (undoing both after the scatter). Before the fix this either crashed
# (DimensionMismatch) or silently produced garbage (the implicit diffusion solve
# operated on grid-space z data).
#
# The reference values are the SERIAL result of this exact problem. The MAX is
# reassociation-invariant (exact match); the SUM may reassociate by ~1 ulp.
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "Distributed Cheb-Fourier IVP test requires >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

# Serial reference (from the same code path at np=1).
const SUMSQ_REF = 33.04847505669661
const BMAX_REF = 1.430277722239972

_loc(f) = get_grid_data(f) isa PencilArrays.PencilArray ? parent(get_grid_data(f)) : get_grid_data(f)
function _assign_local!(field, gdata)
    data = get_grid_data(field)
    if data isa PencilArrays.PencilArray
        ax = PencilArrays.pencil(data).axes_local
        parent(data) .= gdata[ax...]
    else
        data .= gdata
    end
end

@testset "Distributed Cheb-Fourier IVP matches serial (rank=$rank)" begin
    kappa = 0.1; Lz = 1.0; dt = 1e-3; NSTEPS = 20; Nz = 12; Nx = 8
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
    xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    domain = Domain(dist, (zbasis, xbasis))
    b = ScalarField(domain, "b")
    tau_b1 = ScalarField(dist, "tau_b1", (), Float64)
    tau_b2 = ScalarField(dist, "tau_b2", (), Float64)
    ex, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * τ_lift(tau_b1)
    problem = IVP([b, tau_b1, tau_b2])
    add_parameters!(problem, kappa=kappa, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
    add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = 0")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt=dt)

    xfull = [2π*(i-1)/Nx for i in 1:Nx]
    zfull = [Lz/2*(1-cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
    b0(z, x) = sin(π*z/Lz)*(1 + 0.5*cos(2*x))
    gdata = [b0(zfull[iz], xfull[ix]) for iz in 1:Nz, ix in 1:Nx]
    ensure_layout!(b, :g); _assign_local!(b, gdata); ensure_layout!(b, :c)

    for _ in 1:NSTEPS
        step!(solver, dt)
    end

    ensure_layout!(b, :g); lv = _loc(b)
    sumsq = MPI.Allreduce(sum(abs2, lv), MPI.SUM, comm)
    bmax = MPI.Allreduce(maximum(abs, lv), MPI.MAX, comm)
    rank == 0 && println("  np=$nprocs sumsq=$sumsq bmax=$bmax (ref sumsq=$SUMSQ_REF bmax=$BMAX_REF)")

    # MAX is reassociation-invariant; SUM may reassociate by ~1 ulp.
    @test isapprox(bmax, BMAX_REF; atol=1e-10)
    @test isapprox(sumsq, SUMSQ_REF; atol=1e-6)
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
