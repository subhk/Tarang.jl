# 2D Quasi-Geostrophic Turbulence
#
# Evolves the potential vorticity (PV) equation on a doubly-periodic domain:
#
#     ∂t(q) + u·∇(q) = -ν * (-Δ)⁴ * q
#
# where q = Δ(ψ), u = skew(∇(ψ)), using 8th-order hyperviscosity.
#
# 2D turbulence exhibits:
#   - Inverse energy cascade (energy → large scales)
#   - Forward enstrophy cascade (enstrophy → small scales)
#
# To run:
#     julia --project=. examples/ivp/QG_turbulence.jl
#     mpiexec -n 4 julia --project=. examples/ivp/QG_turbulence.jl

using Tarang
using Printf
using Logging
global_logger(ConsoleLogger(stderr, Logging.Warn))

# ─── Parameters ───────────────────────────────────────────────
Nx, Ny        = 512, 512
nu            = 1e-20          # Hyperviscosity coefficient
stop_time     = 50.0
max_dt        = 0.001

# ─── Domain & Fields ──────────────────────────────────────────
domain = PeriodicDomain(Nx, Ny)
dist   = domain.dist

q     = ScalarField(domain, "q")          # Potential vorticity
ψ     = ScalarField(domain, "ψ")          # Streamfunction
u     = VectorField(domain, "u")          # Velocity
tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)

# ─── Problem ─────────────────────────────────────────────────
problem = IVP([q, ψ, u, tau_ψ])
add_parameters!(problem, nu=nu)

add_equation!(problem, "Δ(ψ) + tau_ψ - q  = 0")           # Poisson equation
add_equation!(problem, "u - skew(grad(ψ)) = 0")           # Velocity from ψ
add_equation!(problem, "∂t(q) + nu*Δ⁴(q)  = -u⋅∇(q)")     # PV evolution

add_bc!(problem, "integ(ψ) = 0")

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

@root_only diagnose(solver)

# ─── Initial Conditions ──────────────────────────────────────
fill_random!(q, "g"; seed=42, distribution="normal", scale=1.0)
ensure_layout!(q, :c)

# ─── CFL ─────────────────────────────────────────────────────
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)

# ─── Main Loop ────────────────────────────────────────────────
@root_only println("\nStarting QG turbulence simulation")

run!(solver;
     stop_time=stop_time,
     log_interval=100,
     callbacks=[
         on_interval(10) do s
             max_q = global_max(dist, abs.(grid_data(q)))
             @root_only @printf("  iter=%d, t=%.4f, dt=%.2e, max|q|=%.6f\n",
                                 s.iteration, s.sim_time, s.dt, max_q)
         end
     ])

@root_only println("Done!")
