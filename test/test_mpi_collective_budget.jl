# Guard: the COLLECTIVE budget of a distributed mixed Chebyshev-Fourier step.
#
# On a distributed mixed field, forward_transform!/backward_transform! each drag a
# `_apply_distributed_coupled_dct!` — TWO collective PencilArrays.transpose! calls (fft pencil
# → solve pencil → back). A derivative along a FOURIER axis does not need it: it multiplies by
# (ik)^order along that axis alone, and whether the Chebyshev axis holds spectral coefficients
# or grid values is irrelevant to that multiplier. So `_apply_lazy_diff!` skips the coupled DCT
# on both legs for a Fourier-axis derivative.
#
# That saving is INVISIBLE to every other kind of guard: the transposes are in-place, so nothing
# allocates, and the answers are identical either way (test_mpi_cheb_fourier_ivp_nonlinear
# already pins the values against a serial reference). Only a COUNT catches a regression.
#
#   RK222, -b*∂x(b) on a Chebyshev×Fourier domain, per step:
#     before: 9 coupled-DCT round-trips (18 collective transposes)
#     after:  3                          ( 6)
using Test
using Tarang
import MPI
using Tarang: PencilArrays
MPI.Initialized() || MPI.Init()
const NP   = MPI.Comm_size(MPI.COMM_WORLD)
const COMM = MPI.COMM_WORLD

function _build(stepper; Nz=12, Nx=16, dt=1e-3)
    coords = CartesianCoordinates("z", "x")
    dist = NP > 1 ? Distributor(coords; mesh=(NP,), dtype=Float64, architecture=CPU()) :
                    Distributor(coords; dtype=Float64, architecture=CPU())
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    domain = Domain(dist, (zb, xb))
    b = ScalarField(domain, "b")
    tau_b1 = ScalarField(dist, "tau_b1", (), Float64)
    tau_b2 = ScalarField(dist, "tau_b2", (), Float64)
    _, ez = unit_vector_fields(coords, dist)
    lb = derivative_basis(zb, 1)
    τ_lift(A) = lift(A, lb, -1)
    grad_b = grad(b) + ez * τ_lift(tau_b1)
    problem = IVP([b, tau_b1, tau_b2])
    add_parameters!(problem, kappa=0.1, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
    add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, stepper; dt=dt)

    zf = [0.5*(1-cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
    xf = [2π*(i-1)/Nx for i in 1:Nx]
    g = [sin(π*zf[i])*(1 + 0.5cos(2*xf[j])) for i in 1:Nz, j in 1:Nx]
    ensure_layout!(b, :g)
    d = get_grid_data(b)
    if d isa PencilArrays.PencilArray
        parent(d) .= g[PencilArrays.pencil(d).axes_local...]
    else
        d .= g
    end
    ensure_layout!(b, :c)
    solver
end

"""Coupled-DCT round-trips consumed by one step, after warm-up."""
function _coupled_dct_per_step(stepper)
    solver = _build(stepper)
    for _ in 1:6; step!(solver, 1e-3); end     # past SBDF2 startup
    Tarang.enable_transform_counts!(true)
    Tarang.reset_transform_counts!()
    step!(solver, 1e-3)
    c = Tarang.transform_counts()
    Tarang.enable_transform_counts!(false)
    c.coupled_dct
end

@testset "distributed mixed-basis collective budget (NP=$NP)" begin
    if NP == 1
        # Serial: the coupled DCT is a no-op (there is no solve pencil to transpose to).
        @test _coupled_dct_per_step(RK222()) == 0
    else
        rk = _coupled_dct_per_step(RK222())
        sb = _coupled_dct_per_step(SBDF2())

        # A Fourier-axis derivative must NOT pay a coupled-DCT round-trip. With the ∂x node
        # skipping it, RK222 is down to the state transforms alone. Unfused it was 9.
        @test rk <= 4          # measured 3
        @test sb <= 3          # measured 2

        # Sanity: the counter is actually wired up (a mixed distributed step DOES do some).
        @test rk >= 1
    end
end
