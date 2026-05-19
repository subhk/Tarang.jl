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
Nx       = 512
Ny       = Nx
Lx, Ly   = 2π, 2π
nu       = 1e-20
drag     = 1e-3
stop_time = 10000.0
stop_iteration = typemax(Int)
max_dt   = 1e-1

# Forcing parameters
k_f      = 5.0
dk_f     = 2.0
ε        = 0.1

# ─── Domain & Fields ──────────────────────────────────────────
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly), dealias=3/2)

domain = Domain(dist, (xbasis, ybasis))

# ─── Fields ───────────────────────────────────────────────────
b     = ScalarField(domain, "b")          # Surface buoyancy
ψ     = ScalarField(domain, "ψ")          # Streamfunction
u     = VectorField(domain, "u")          # Velocity

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
# SQG inversion: ψ = (-Δ)^(-1/2) θ. The inverse sets the mean mode to zero.
problem = IVP([b, ψ, u])
add_parameters!(problem, nu=nu, alpha=alpha)

add_equation!(problem, "∂t(b) + nu*fraclap(b, alpha) = -u⋅∇(θb)")   # Buoyancy evolution
add_equation!(problem, "ψ - invsqrtlap(θ) = 0")                    # SQG inversion
add_equation!(problem, "u - skew(grad(ψ)) = 0")                    # Velocity from ψ

# Register forcing on the buoyancy variable
add_stochastic_forcing!(problem, :b, forcing)

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

# ─── Initial Conditions ──────────────────────────────────────
fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3)

# ─── Output ──────────────────────────────────────────────────
output_path = "snapshots/sqg_snapshots"
snapshots = add_file_handler(output_path, dist, 
                             Dict("b" => b, "ψ" => ψ);
                             sim_dt=50.0, max_writes=100)

add_task!(snapshots, b; name="b")
add_task!(snapshots, ψ; name="psi")

# ─── CFL ─────────────────────────────────────────────────────
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)

# ─── Main Loop ────────────────────────────────────────────────
@root_only println("Forced SQG Turbulence")
@root_only @printf("  N=%d×%d, ε=%.2e, k_f=%.1f, ν=%.1e, α=%.1f\n",
                    Nx, Ny, ε, k_f, nu, alpha)
@root_only @printf("  dt≤%.2e, output_dt=%.2e, initial_noise=%.1e\n",
                    max_dt, output_dt, initial_noise)

wall_start = time()
process!(snapshots; iteration=solver.iteration, wall_time=0.0,
         sim_time=solver.sim_time, timestep=solver.dt)

while solver.sim_time < stop_time && solver.iteration < stop_iteration
    dt = min(compute_timestep(cfl), stop_time - solver.sim_time)
    step!(solver, dt)

    wall_time = time() - wall_start
    process!(snapshots; iteration=solver.iteration, wall_time=wall_time,
             sim_time=solver.sim_time, timestep=solver.dt)

    if solver.iteration % log_interval == 0
        ensure_layout!(θ, :g)
        max_θ = global_max(dist, abs.(get_grid_data(θ)))
        if !isfinite(max_θ)
            error("Non-finite surface buoyancy detected at iteration $(solver.iteration), t=$(solver.sim_time)")
        end
        @root_only @printf("  iter=%d, t=%.6f, dt=%.2e, max|θ|=%.4e\n",
                            solver.iteration, solver.sim_time, solver.dt, max_θ)
    end
end

close!(snapshots)

@root_only println("Done!")
