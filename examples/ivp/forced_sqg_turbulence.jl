# Forced Surface Quasi-Geostrophic (SQG) Turbulence
#
# Evolves the surface buoyancy equation on a doubly-periodic domain:
#
#     ∂t(θ) + u·∇(θ) = -ν (-Δ)^α θ + F
#
# where ψ = (-Δ)^{-1/2} θ  (SQG inversion),  u = skew(∇ψ),
# and F is stochastic ring forcing.
#
# SQG turbulence produces:
#   - k^{-5/3} energy spectrum (forward cascade)
#   - Sharp fronts in surface buoyancy
#   - Filamentary structures
#
# The key difference from 2D turbulence: the streamfunction-vorticity
# relation uses the fractional Laplacian (-Δ)^{-1/2} instead of Δ^{-1}.
#
# To run:
#     julia --project=. examples/ivp/forced_sqg_turbulence.jl

using Tarang
using Printf
using Logging
global_logger(ConsoleLogger(stderr, Logging.Warn))

# ─── Parameters ───────────────────────────────────────────────
Nx, Ny   = 512, 512
Lx, Ly   = 2π, 2π
nu       = 1e-16             # Hyperdiffusion coefficient
alpha    = 4.0               # Dissipation exponent: (-Δ)^α
stop_time = 50.0
max_dt   = 2e-3

# Forcing parameters
k_f      = 10.0             # Central forcing wavenumber
dk_f     = 2.0              # Forcing bandwidth
ε        = 0.1              # Energy injection rate

# ─── Domain & Fields ──────────────────────────────────────────
domain = PeriodicDomain(Nx, Ny; L=(Lx, Ly))
dist   = domain.dist

θ     = ScalarField(domain, "θ")          # Surface buoyancy
ψ     = ScalarField(domain, "ψ")          # Streamfunction
u     = VectorField(domain, "u")          # Velocity
tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)

# ─── Stochastic Forcing ──────────────────────────────────────
forcing = StochasticForcing(
    field_size             = (Nx, Ny),
    domain_size            = (Lx, Ly),
    energy_injection_rate  = ε,
    k_forcing              = k_f,
    dk_forcing             = dk_f,
    dt                     = max_dt,
    spectrum_type          = :ring,
    enforce_hermitian      = true,
)

# ─── Problem ─────────────────────────────────────────────────
# SQG inversion: ψ = (-Δ)^{-1/2} θ  →  (-Δ)^{1/2} ψ = θ
# Using fraclap(ψ, 0.5) for (-Δ)^{1/2}
problem = IVP([θ, ψ, u, tau_ψ])
add_parameters!(problem, nu=nu, alpha=alpha)

add_equation!(problem, "∂t(θ) + nu*fraclap(θ, alpha) = -u⋅∇(θ)")  # Buoyancy evolution
add_equation!(problem, "fraclap(ψ, 0.5) + tau_ψ - θ = 0")         # SQG inversion
add_equation!(problem, "u - skew(grad(ψ)) = 0")                    # Velocity from ψ

add_bc!(problem, "integ(ψ) = 0")

# Register forcing on the buoyancy variable
add_stochastic_forcing!(problem, :θ, forcing)

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

# ─── Initial Conditions ──────────────────────────────────────
# Start from rest — forcing spins up the flow
# Optionally seed with small perturbation:
# fill_random!(θ, "g"; seed=42, distribution="normal", scale=0.01)
# ensure_layout!(θ, :c)

# ─── CFL ─────────────────────────────────────────────────────
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)

# ─── Main Loop ────────────────────────────────────────────────
@root_only println("Forced SQG Turbulence")
@root_only @printf("  N=%d×%d, ε=%.2e, k_f=%.1f, ν=%.1e, α=%.1f\n",
                    Nx, Ny, ε, k_f, nu, alpha)

run!(solver;
     stop_time=stop_time,
     log_interval=100,
     callbacks=[
         on_interval(20) do s
             ensure_layout!(θ, :g)
             max_θ = global_max(dist, abs.(get_grid_data(θ)))
             @root_only @printf("  iter=%d, t=%.2f, dt=%.2e, max|θ|=%.4f\n",
                                 s.iteration, s.sim_time, s.dt, max_θ)
         end
     ])

@root_only println("Done!")
