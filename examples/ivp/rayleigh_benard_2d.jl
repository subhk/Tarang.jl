# 2D Rayleigh-Benard Convection
#
# Simulates 2D horizontally-periodic Rayleigh-Benard convection.
# Non-dimensionalized using box height H and thermal diffusion time τ_κ = H²/κ:
#
#     length scale:      H
#     time scale:        τ_κ = H² / κ
#     velocity scale:    κ / H
#     pressure scale:    ρ(κ/H)²
#     temperature scale: ΔT = T_bottom - T_top
#
# With T̃ = (T - T_top)/ΔT ∈ [0, 1], the dimensionless equations are:
#
#     ∂t(u) + u⋅∇u = -∇p + Pr ∇²u + Ra·Pr T ẑ
#     ∂t(T) + u⋅∇T = ∇²T
#     ∇⋅u = 0
#
# where:
#     Pr = ν/κ              Prandtl number (viscous coeff in momentum)
#     Ra = gαΔT H³/(νκ)     Rayleigh number (buoyancy scale = Ra·Pr)
#
# The defining feature of the diffusive time scale is that the temperature
# equation has unit diffusivity — one time unit equals one thermal diffusion
# time, making the onset of convection and Nusselt number scaling natural.
#
# Boundary conditions (no-slip, fixed temperature):
#     T(z=0) = 1,  T(z=Lz) = 0,  u(z=0) = u(z=Lz) = 0
#
# First-order formulation for Chebyshev tau method:
#     ∇_T = ∇T + ẑ Lift(τ_T1),   ∇_u = ∇u + ẑ Lift(τ_u1)
#     tr(∇_u) + τ_p = 0
#     ∂t(T) - ∇⋅∇_T + Lift(τ_T2) = -u⋅∇T
#     ∂t(u) - Pr ∇⋅∇_u + ∇p - Ra·Pr T ẑ + Lift(τ_u2) = -u⋅∇u
#
# To run:
#     julia --project=. examples/ivp/rayleigh_benard_2d.jl

using Tarang
using Printf

# ─── Parameters ───────────────────────────────────────────────
Lx, Lz      = 4.0, 1.0
Nx, Nz      = 256, 64
Rayleigh    = 2e6
Prandtl     = 1.0
dealias     = 3/2
stop_time   = 25.0    # thermal diffusion times
max_dt      = 0.001   # small to resolve viscous/thermal transport

# Coefficients in the diffusive-time formulation:
#   ∂t(T) - ∇²T = -u⋅∇T             (unit diffusivity)
#   ∂t(u) - Pr ∇²u + ∇p - Ra·Pr T ẑ = -u⋅∇u
nu   = Prandtl                  # viscous coefficient (= Pr in diffusive units)
buoy = Rayleigh * Prandtl       # buoyancy forcing Ra·Pr

@root_only begin
    println("2D Rayleigh-Benard Convection (thermal-diffusive scaling)")
    @printf("  Ra=%.2e, Pr=%.1f\n", Rayleigh, Prandtl)
    @printf("  Coefficients: Pr=%.4f, Ra·Pr=%.2e\n", Prandtl, buoy)
    @printf("  Domain: %.1f × %.1f, Resolution: %d × %d\n", Lx, Lz, Nx, Nz)
    @printf("  Stop time: %.1f thermal diffusion times\n\n", stop_time)
end

# ─── Bases ────────────────────────────────────────────────────
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=dealias)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=dealias)

# ─── Fields ───────────────────────────────────────────────────
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")
T = ScalarField(domain, "T")        # Dimensionless temperature in [0, 1]
u = VectorField(domain, "u")

tau_p  = ScalarField(dist, "tau_p",  (), Float64)
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

# ─── Substitutions (first-order reduction) ───────────────────
ex, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A) = lift(A, lift_basis, -1)

# First-order gradient substitutions
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

# ─── Problem ─────────────────────────────────────────────────
problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])

add_parameters!(problem,
    nu=nu, buoy=buoy, ez=ez,
    grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

# Continuity (first-order form)
add_equation!(problem, "trace(grad_u) + tau_p = 0")

# Temperature: ∂t(T) - ∇²T = -u⋅∇T
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")

# Momentum: ∂t(u) - Pr∇²u + ∇p - Ra·Pr T ẑ = -u⋅∇u
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")

# Boundary conditions
add_bc!(problem, "T(z=0) = 1")       # hot bottom
add_bc!(problem, "T(z=1) = 0")      # cold top
add_bc!(problem, "u(z=0) = 0")       # no-slip bottom
add_bc!(problem, "u(z=1) = 0")      # no-slip top
add_bc!(problem, "integ(p) = 0")     # pressure gauge

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

@root_only diagnose(solver)

# ─── Output ──────────────────────────────────────────────────
snapshots = add_file_handler("snapshots", dist,
    Dict("T" => T, "ux" => u.components[1], "uz" => u.components[2]);
    sim_dt=0.1, max_writes=50)
add_task!(snapshots, T;               name="temperature")
add_task!(snapshots, u.components[1]; name="ux")
add_task!(snapshots, u.components[2]; name="uz")

# ─── Initial Conditions ──────────────────────────────────────
# Conduction profile: T(z) = 1 - z/Lz (linear from 1 at bottom to 0 at top),
# plus a small random perturbation to trigger convection.
x, z = local_grids(dist, xbasis, zbasis)
fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (Lz .- z')        # damp noise at walls
get_grid_data(T) .+= 1.0 .- z' ./ Lz          # add linear conduction profile
ensure_layout!(T, :c)                          # pre-compute coefficients for timestepper

# ─── CFL ─────────────────────────────────────────────────────
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)

# ─── Main Loop ────────────────────────────────────────────────
@root_only println("Starting main loop")

run!(solver;
     stop_time=stop_time,
     log_interval=100,
     callbacks=[
         on_interval(10) do s
             ensure_layout!(T, :g)
             max_T = global_max(dist, abs.(get_grid_data(T)))
             ensure_layout!(u.components[2], :g)
             max_uz = global_max(dist, abs.(get_grid_data(u.components[2])))
             @root_only @printf("  iter=%d, t=%.4e, dt=%.4e, max|T|=%.4f, max|uz|=%.4e\n",
                                 s.iteration, s.sim_time, s.dt, max_T, max_uz)
         end
     ])

@root_only println("Done!")
