# Forced 2D Turbulence
#
# Stochastically forced 2D Navier-Stokes on a doubly-periodic domain:
#
#     ∂t(q) + u·∇(q) = -ν Δ⁴(q) + F
#
# where q = Δ(ψ), u = skew(∇ψ)), and F is white-in-time ring forcing
# injecting energy at wavenumber k_f.
#
# The forcing drives an inverse energy cascade (energy → large scales)
# and a forward enstrophy cascade (enstrophy → small scales), producing
# the classic k^{-5/3} and k^{-3} spectra.
#
# To run:
#     julia --project=. examples/ivp/forced_2d_turbulence.jl

using Tarang
using Printf
using Logging
global_logger(ConsoleLogger(stderr, Logging.Warn))

# ─── Parameters ───────────────────────────────────────────────
Nx, Ny   = 512, 512
Lx, Ly   = 2π, 2π
nu       = 1e-20             # Hyperviscosity coefficient (8th-order)
stop_time = 100.0
max_dt   = 5e-3

# Forcing parameters
k_f      = 8.0              # Central forcing wavenumber
dk_f     = 2.0              # Forcing bandwidth
ε        = 0.1              # Energy injection rate

# ─── Domain & Fields ──────────────────────────────────────────
domain = PeriodicDomain(Nx, Ny; L=(Lx, Ly))
dist   = domain.dist

q     = ScalarField(domain, "q")          # Vorticity
ψ     = ScalarField(domain, "ψ")          # Streamfunction
u     = VectorField(domain, "u")          # Velocity
tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)

# ─── Stochastic Forcing ──────────────────────────────────────
# Ring forcing in wavenumber space: energy injected in |k| ∈ [k_f - dk_f, k_f + dk_f]
# White-in-time: new random phase each timestep, amplitude set by ε
forcing = StochasticForcing(
    field_size             = (Nx, Ny),
    domain_size            = (Lx, Ly),
    energy_injection_rate  = ε,
    k_forcing              = k_f,
    dk_forcing             = dk_f,
    dt                     = max_dt,
    spectrum_type          = :ring,      # Isotropic ring in k-space
    enforce_hermitian      = true,       # Real-valued vorticity
)

# ─── Problem ─────────────────────────────────────────────────
problem = IVP([q, ψ, u, tau_ψ])
add_parameters!(problem, nu=nu)

add_equation!(problem, "∂t(q) + nu*Δ⁴(q)  = -u⋅∇(q)")     # PV evolution (forcing added below)
add_equation!(problem, "Δ(ψ) + tau_ψ - q  = 0")            # Poisson equation
add_equation!(problem, "u - skew(grad(ψ)) = 0")            # Velocity from ψ

add_bc!(problem, "integ(ψ) = 0")

# Register forcing on the vorticity variable
add_stochastic_forcing!(problem, :q, forcing)

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

# ─── Initial Conditions ──────────────────────────────────────
# Start from rest — the forcing will spin up the flow
# (No initial condition needed for forced turbulence)

# ─── CFL ─────────────────────────────────────────────────────
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)

# ─── Main Loop ────────────────────────────────────────────────
@root_only println("Forced 2D Turbulence")
@root_only @printf("  N=%d×%d, ε=%.2e, k_f=%.1f, ν=%.1e\n", Nx, Ny, ε, k_f, nu)

run!(solver;
     stop_time=stop_time,
     log_interval=100,
     callbacks=[
         on_interval(20) do s
             ensure_layout!(q, :g)
             max_q = global_max(dist, abs.(get_grid_data(q)))
             @root_only @printf("  iter=%d, t=%.2f, dt=%.2e, max|q|=%.4f\n",
                                 s.iteration, s.sim_time, s.dt, max_q)
         end
     ])

@root_only println("Done!")
