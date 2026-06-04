"""
End-to-end PDE solve test with known analytical solution.

Solves the 1D diffusion equation on a periodic domain:
    dt(u) = nu * lap(u)
with initial condition u(x, 0) = sin(x).

Analytical solution: u(x, t) = exp(-nu * k^2 * t) * sin(x), where k = 1.
"""

using Test
using Tarang

@testset "End-to-end PDE solve" begin

    # --- Parameters ---
    N  = 64                 # grid points
    nu = 0.05               # diffusion coefficient
    dt = 1e-3               # time step
    t_final = 1.0           # integration time
    k  = 1                  # wavenumber of initial condition

    # --- Domain and field ---
    domain = PeriodicDomain(N)
    u = ScalarField(domain, "u")

    # --- Problem setup ---
    problem = IVP([u])
    add_parameters!(problem, nu=nu)
    add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")

    # --- Initial condition: u(x, 0) = sin(x) ---
    set!(u, (x,) -> sin(x))

    # --- Solver ---
    solver = InitialValueSolver(problem, RK222(); dt=dt)

    # --- Integrate ---
    run!(solver; stop_time=t_final, progress=false)

    @test solver.sim_time >= t_final - dt   # confirm we reached the target time

    # --- Analytical solution ---
    # For the diffusion equation, a single Fourier mode decays as:
    #   u(x, t) = exp(-nu * k^2 * t) * sin(x)
    decay = exp(-nu * k^2 * t_final)

    # Get the numerical solution on the grid
    numerical = grid_data(u)

    # Build the expected solution on the same grid
    mesh = Tarang.create_meshgrid(domain; on_device=false)
    x = mesh["x"]
    analytical = decay .* sin.(x)

    # --- Comparison ---
    @test isapprox(numerical, analytical; rtol=1e-4, atol=1e-10)

    # Also verify that the peak amplitude decayed correctly
    @test isapprox(maximum(abs.(numerical)), decay; rtol=1e-4)
end

# Regression guard (2026-06-04): a PURE 1D-Chebyshev IVP (no Fourier axis) must
# build a per-mode tau subproblem and integrate the implicit Laplacian. Before the
# fix, `_try_build_subproblems!` only built for MIXED Fourier+Chebyshev domains, so
# a pure-Chebyshev problem built ZERO subproblems → the implicit operator was never
# applied → the field did not diffuse at all.
#     dt(u) = Δu,  u(z=0)=u(z=Lz)=0,  u(z,0)=sin(πz/Lz)
# decays as exp(-(π/Lz)² t) (the lowest Dirichlet-Laplacian mode).
@testset "End-to-end 1D pure-Chebyshev diffusion (implicit Laplacian)" begin
    Lz = 1.0
    coords = CartesianCoordinates("z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    zb = ChebyshevT(coords["z"]; size=32, bounds=(0.0, Lz))
    dom = Domain(dist, (zb,))
    u    = ScalarField(dom, "u")
    tau1 = ScalarField(dist, "tau1", (), Float64)
    tau2 = ScalarField(dist, "tau2", (), Float64)
    lb2  = derivative_basis(zb, 2)

    problem = IVP([u, tau1, tau2])
    add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
    add_equation!(problem, "dt(u) - Δ(u) - l1 - l2 = 0")
    add_bc!(problem, "u(z=0)   = 0")
    add_bc!(problem, "u(z=1.0) = 0")

    ensure_layout!(u, :g)
    zc = vec(Array(Tarang.local_grid(zb, dist, 1)))
    Tarang.get_grid_data(u) .= sin.(π .* zc ./ Lz)

    solver = InitialValueSolver(problem, RK222(); dt=1e-4)
    # the subproblem must actually be built (was 0 before the fix)
    @test length(get(problem.parameters, "subproblems", ())) >= 1

    t_final = 0.05
    for _ in 1:Int(round(t_final / 1e-4)); step!(solver); end

    ensure_layout!(u, :g)
    numerical = vec(Array(Tarang.get_grid_data(u)))
    analytical = exp(-(π / Lz)^2 * t_final) .* sin.(π .* zc ./ Lz)
    @test isapprox(numerical, analytical; rtol=1e-4, atol=1e-7)
    # must have actually decayed (catches the "no diffusion" failure mode)
    @test maximum(abs.(numerical)) < 0.95 * maximum(abs.(sin.(π .* zc ./ Lz)))
end
