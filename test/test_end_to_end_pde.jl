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
