# Forced 2D Turbulence
#
# Stochastically forced 2D Navier-Stokes on a doubly-periodic domain:
#
#     ∂t(ζ) + u·∇(ζ) = -ν Δ⁴(ζ) - μζ + F
#
# where ζ = Δ(ψ), u = skew(∇ψ)), and F is white-in-time ring forcing
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
Nx       = 512
Ny       = Nx
Lx, Ly   = 2π, 2π
nu       = 1e-20
drag     = 1e-3
stop_time = 20000.0
stop_iteration = typemax(Int)
max_dt   = 1e-1

# Forcing parameters
k_f      = 50.0 
dk_f     = 2.0 
ε        = 600.0

# ─── Domain & Fields ──────────────────────────────────────────
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly), dealias=3/2)

domain = Domain(dist, (xbasis, ybasis))

# ─── Fields ───────────────────────────────────────────────────
ζ     = ScalarField(domain, "ζ")          # Vorticity
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
problem = IVP([ζ, ψ, u, tau_ψ])
add_parameters!(problem, nu=nu, drag=drag)

add_equation!(problem, "∂t(ζ) + drag*ζ + nu*Δ⁴(ζ) = -u⋅∇(ζ)")  # PV evolution (forcing added below)
add_equation!(problem, "Δ(ψ) + tau_ψ - ζ  = 0")                # Poisson equation
add_equation!(problem, "u - skew(grad(ψ)) = 0")                # Velocity from ψ

add_bc!(problem, "integ(ψ) = 0")

# Register forcing on the vorticity variable
add_stochastic_forcing!(problem, :ζ, forcing)

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

# ─── Initial Conditions ──────────────────────────────────────
fill_random!(ζ, "g"; seed=42, distribution="normal", scale=1e-2)

# ─── Output ──────────────────────────────────────────────────
snapshots = add_file_handler("2d_turb/2d_turb", dist,
                            Dict("ζ" => ζ, "ψ" => ψ);
                            sim_dt=50.0, max_writes=100)

add_task!(snapshots, ζ; name="ζ")
add_task!(snapshots, ψ; name="ψ")

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
    dt = compute_timestep(cfl)
    step!(solver, dt)

    wall_time = time() - wall_start
    process!(snapshots; iteration=solver.iteration, wall_time=wall_time,
             sim_time=solver.sim_time, timestep=solver.dt)

    if solver.iteration % 200 == 0
        ensure_layout!(ζ, :g)
        max_ζ = global_max(dist, abs.(get_grid_data(ζ)))
        if !isfinite(max_ζ)
            error("Non-finite vorticity detected at iteration $(solver.iteration), t=$(solver.sim_time)")
        end
        @root_only @printf("  iter=%d, t=%.6f, dt=%.2e, max|ζ|=%.4e\n",
                            solver.iteration, solver.sim_time, solver.dt, max_ζ)
    end
end

close!(snapshots)

@root_only println("Done!")
