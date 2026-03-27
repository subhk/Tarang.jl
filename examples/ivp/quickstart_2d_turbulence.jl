# Quickstart: 2D Decaying Turbulence
#
# Demonstrates a complete 2D turbulence simulation with energy diagnostics
# in ~25 lines of code. Uses the high-level API for minimal boilerplate.
#
#     julia --project=. examples/ivp/quickstart_2d_turbulence.jl

using Tarang
using Printf

# ─── Setup ────────────────────────────────────────────────────
N  = 128                           # Grid points per side
ν  = 1e-4                          # Viscosity

domain = PeriodicDomain(N, N)      # [0, 2π]² doubly-periodic

ψ = ScalarField(domain, "ψ")      # Streamfunction
q = ScalarField(domain, "q")      # Vorticity (q = Δψ)
u = VectorField(domain, "u")      # Velocity (u = ∇⊥ψ)

# ─── Equations ────────────────────────────────────────────────
problem = IVP([q, ψ, u])
add_parameters!(problem, nu=ν)

add_equation!(problem, "Δ(ψ) - q = 0")            # Poisson: q = Δψ
add_equation!(problem, "u - skew(grad(ψ)) = 0")   # Velocity: u = (-∂ψ/∂y, ∂ψ/∂x)
add_equation!(problem, "∂t(q) - nu*Δ(q) = -u⋅∇(q)")  # Vorticity transport

# ─── Initial Condition ───────────────────────────────────────
fill_random!(q, "g"; seed=42, distribution="normal", scale=1.0)

# ─── Solve ────────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=1e-3)
diagnose(solver)

@root_only println("\nRunning 2D decaying turbulence (N=$N, ν=$ν)...")
run!(solver;
     stop_time=10.0,
     log_interval=500,
     callbacks=[
         on_interval(100) do s
             enstrophy = sum(grid_data(q) .^ 2) / N^2
             @root_only @printf("  t=%.2f  enstrophy=%.4f\n", s.sim_time, enstrophy)
         end
     ])

@root_only println("Simulation complete!")
