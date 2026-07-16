# Forced 2D Turbulence
#
# Stochastically forced 2D Navier-Stokes on a doubly-periodic domain:
#
#     ∂t(ζ) + u·∇(ζ) = -ν Δ⁴(ζ) - μζ + F
#
# where ζ = Δ(ψ), u = skew(∇ψ), and F is white-in-time ring forcing
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
Nx       = parse(Int, get(ENV, "TARANG_FORCED_2D_NX", "512"))
Ny       = parse(Int, get(ENV, "TARANG_FORCED_2D_NY", string(Nx)))
Lx, Ly   = 2π, 2π
nu       = parse(Float64, get(ENV, "TARANG_FORCED_2D_NU", "1e-20"))
drag     = parse(Float64, get(ENV, "TARANG_FORCED_2D_DRAG", "1e-3"))
stop_time = parse(Float64, get(ENV, "TARANG_FORCED_2D_STOP_TIME", "20000.0"))
stop_iteration = parse(Int, get(ENV, "TARANG_FORCED_2D_STOP_ITERATION", string(typemax(Int))))
max_dt   = parse(Float64, get(ENV, "TARANG_FORCED_2D_MAX_DT", "1e-1"))
output_dt = parse(Float64, get(ENV, "TARANG_FORCED_2D_OUTPUT_DT", "50.0"))
initial_noise = parse(Float64, get(ENV, "TARANG_FORCED_2D_INITIAL_NOISE", "1e-2"))
log_interval = parse(Int, get(ENV, "TARANG_FORCED_2D_LOG_INTERVAL", "200"))
max_writes = parse(Int, get(ENV, "TARANG_FORCED_2D_MAX_WRITES", "100"))

# Forcing parameters
k_f      = parse(Float64, get(ENV, "TARANG_FORCED_2D_KF", "50.0"))
dk_f     = parse(Float64, get(ENV, "TARANG_FORCED_2D_DKF", "2.0"))
ε        = parse(Float64, get(ENV, "TARANG_FORCED_2D_EPSILON", "600.0"))

# ─── Architecture ─────────────────────────────────────────────
# CPU by default. Set TARANG_FORCED_2D_DEVICE=gpu to run on CUDA (needs a GPU +
# CUDA.jl); optionally pick a specific GPU with TARANG_FORCED_2D_GPU_ID.
const USE_GPU = lowercase(get(ENV, "TARANG_FORCED_2D_DEVICE", "cpu")) in ("gpu", "cuda")
gpu_id = parse(Int, get(ENV, "TARANG_FORCED_2D_GPU_ID", "0"))
if USE_GPU
    using CUDA
    CUDA.allowscalar(false)
end
device = USE_GPU ? GPU(device_id=gpu_id) : CPU()

# ─── Domain & Fields ──────────────────────────────────────────
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=device)

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly), dealias=3/2)

domain = Domain(dist, (xbasis, ybasis))

# ─── Fields ───────────────────────────────────────────────────
ζ     = ScalarField(domain, "ζ")          # Vorticity
ψ     = ScalarField(domain, "ψ")          # Streamfunction
u     = VectorField(domain, "u")          # Velocity
tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)

# ─── Stochastic Forcing ──────────────────────────────────────
# Gaussian ring forcing in wavenumber space, concentrated near |k| = k_f
# White-in-time: new random phase each timestep, amplitude set by ε
forcing = StochasticForcing(
    field_size             = (Nx, Ny),
    domain_size            = (Lx, Ly),
    energy_injection_rate  = ε,
    injection_metric       = :vorticity_kinetic, # ε is kinetic-energy injection
    k_forcing              = k_f,
    dk_forcing             = dk_f,
    dt                     = max_dt,
    spectrum_type          = :ring,      # Isotropic ring in k-space
    enforce_hermitian      = true,       # Real-valued vorticity
    architecture           = device,
)

# ─── Problem ─────────────────────────────────────────────────
problem = IVP([ζ, ψ, u, tau_ψ])
add_parameters!(problem, nu=nu, drag=drag)

# Keep every prognostic term on the explicit RHS.  Pure-Fourier GPU fields use
# Tarang's device-native explicit RK path (there is no global GPU DAE solve), so
# terms placed on the implicit LHS would otherwise be omitted by that fallback.
# At the default resolution/parameters the hyperviscous stability limit is much
# larger than max_dt, so explicit treatment is both correct and stable here.
add_equation!(problem, "∂t(ζ) = -u⋅∇(ζ) - drag*ζ - nu*Δ⁴(ζ)")  # forcing added below
add_equation!(problem, "Δ(ψ) + tau_ψ - ζ  = 0")                # Poisson equation
add_equation!(problem, "u - skew(grad(ψ)) = 0")                # Velocity from ψ

add_bc!(problem, "integ(ψ) = 0")

# Register forcing on the vorticity variable
add_stochastic_forcing!(problem, :ζ, forcing)

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

# ─── Initial Conditions ──────────────────────────────────────
fill_random!(ζ, "g"; seed=42, distribution="normal", scale=initial_noise)

# ─── Output ──────────────────────────────────────────────────
output_path = get(ENV, "TARANG_FORCED_2D_OUTPUT", "2d_turb/2d_turb")
snapshots = add_file_handler(output_path, solver;
                            sim_dt=output_dt, max_writes=max_writes)

add_task!(snapshots, ζ; name="ζ")
add_task!(snapshots, ψ; name="ψ")

# add_task!(snapshots, u.components[1]; name="ux")
# add_task!(snapshots, u.components[2]; name="uz")

# ─── CFL ─────────────────────────────────────────────────────
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)

# ─── Run ──────────────────────────────────────────────────────
@root_only println("Forced 2D Turbulence")
@root_only @printf("  N=%d×%d, ε=%.2e, k_f=%.1f, ν=%.1e, μ=%.1e\n", Nx, Ny, ε, k_f, nu, drag)
@root_only @printf("  dt≤%.2e, output_dt=%.2e, initial_noise=%.1e\n",
                    max_dt, output_dt, initial_noise)

# Diagnostics: print time / timestep / max|ζ| and guard against blow-up. The
# callback receives the solver; global_max is MPI-reduced (same value on all ranks).
function log_state(s)
    ensure_layout!(ζ, :g)
    max_ζ = global_max(dist, abs.(get_grid_data(ζ)))
    isfinite(max_ζ) || error("Non-finite vorticity at iter $(s.iteration), t=$(s.sim_time)")
    @root_only @printf("  iter=%d, t=%.6f, dt=%.2e, max|ζ|=%.4e\n",
                       s.iteration, s.sim_time, s.dt, max_ζ)
end

# run! drives the whole loop: CFL-adaptive dt, auto-writes `snapshots` (registered
# above by add_file_handler), fires the diagnostics callback every `log_interval`
# steps, and closes the handler at the end — no manual step!/process!/close!.
run!(solver; stop_time=stop_time, stop_iteration=stop_iteration, cfl=cfl,
     callbacks=[log_interval => log_state])

@root_only println("Done!")
