# Forced 2D Turbulence
#
# Stochastically forced 2D Navier-Stokes on a doubly-periodic domain:
#
#     ∂t(q) + u·∇(q) = -ν Δ⁴(q) - μq + F
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
Nx       = parse(Int, get(ENV, "TARANG_FORCED_2D_NX", "256"))
Ny       = parse(Int, get(ENV, "TARANG_FORCED_2D_NY", string(Nx)))
Lx, Ly   = 2π, 2π
nu       = parse(Float64, get(ENV, "TARANG_FORCED_2D_NU", "1e-16"))
drag     = parse(Float64, get(ENV, "TARANG_FORCED_2D_DRAG", "1e-2"))
stop_time = parse(Float64, get(ENV, "TARANG_FORCED_2D_STOP_TIME", "100.0"))
stop_iteration = parse(Int, get(ENV, "TARANG_FORCED_2D_STOP_ITERATION", string(typemax(Int))))
max_dt   = parse(Float64, get(ENV, "TARANG_FORCED_2D_MAX_DT", "1e-6"))

# Forcing parameters
k_f      = parse(Float64, get(ENV, "TARANG_FORCED_2D_KF", "50.0"))
dk_f     = parse(Float64, get(ENV, "TARANG_FORCED_2D_DKF", "2.0"))
ε        = parse(Float64, get(ENV, "TARANG_FORCED_2D_EPS", "1e-6"))

# ─── Domain & Fields ──────────────────────────────────────────
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly), dealias=3/2)

domain = Domain(dist, (xbasis, ybasis))

# ─── Fields ───────────────────────────────────────────────────
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
add_parameters!(problem, nu=nu, drag=drag)

add_equation!(problem, "∂t(q) + drag*q + nu*Δ⁴(q) = -u⋅∇(q)")  # PV evolution (forcing added below)
add_equation!(problem, "Δ(ψ) + tau_ψ - q  = 0")            # Poisson equation
add_equation!(problem, "u - skew(grad(ψ)) = 0")            # Velocity from ψ

add_bc!(problem, "integ(ψ) = 0")

# Register forcing on the vorticity variable
add_stochastic_forcing!(problem, :q, forcing)

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

# ─── Initial Conditions ──────────────────────────────────────
fill_random!(q, "g"; seed=42, distribution="normal", scale=1e-3)

# ─── Output ──────────────────────────────────────────────────
output_path = get(ENV, "TARANG_FORCED_2D_OUTPUT", "snapshots")
snapshots = add_file_handler(output_path, dist, Dict("q" => q); sim_dt=5.0, max_writes=50)
add_task!(snapshots, q; name="q")

# add_task!(snapshots, u.components[1]; name="ux")
# add_task!(snapshots, u.components[2]; name="uz")

# ─── CFL ─────────────────────────────────────────────────────
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)

# ─── Main Loop ────────────────────────────────────────────────
@root_only println("Forced 2D Turbulence")
@root_only @printf("  N=%d×%d, ε=%.2e, k_f=%.1f, ν=%.1e, μ=%.1e\n", Nx, Ny, ε, k_f, nu, drag)

wall_start = time()
process!(snapshots; iteration=solver.iteration, wall_time=0.0,
         sim_time=solver.sim_time, timestep=solver.dt)

while solver.sim_time < stop_time && solver.iteration < stop_iteration
    dt = min(compute_timestep(cfl), stop_time - solver.sim_time)
    step!(solver, dt)

    wall_time = time() - wall_start
    process!(snapshots; iteration=solver.iteration, wall_time=wall_time,
             sim_time=solver.sim_time, timestep=solver.dt)

    if solver.iteration % 20 == 0
        ensure_layout!(q, :g)
        max_q = global_max(dist, abs.(get_grid_data(q)))
        if !isfinite(max_q)
            error("Non-finite vorticity detected at iteration $(solver.iteration), t=$(solver.sim_time)")
        end
        @root_only @printf("  iter=%d, t=%.6f, dt=%.2e, max|q|=%.4e\n",
                            solver.iteration, solver.sim_time, solver.dt, max_q)
    end
end

close!(snapshots)

@root_only println("Done!")
