# 2D Rayleigh-Benard Convection
#
# Simulates 2D horizontally-periodic Rayleigh-Benard convection.
# Non-dimensionalized using box height H and freefall time τ = √(H/(gαΔT)):
#
#     ∂t(u) + u⋅∇(u) = -∇p + b ẑ + ν ∇²u
#     ∂t(b) + u⋅∇(b) = κ ∇²b
#     ∇⋅u = 0
#
# where:
#     κ = (Ra Pr)^(-1/2)      thermal diffusivity
#     ν = (Ra/Pr)^(-1/2)      kinematic viscosity
#
# Boundary conditions (no-slip, fixed temperature):
#     b(z=0) = Lz,  b(z=Lz) = 0,  u(z=0) = u(z=Lz) = 0
#
# First-order formulation for Chebyshev tau method:
#     ∇_b = ∇b + ẑ Lift(τ_b1),   ∇_u = ∇u + ẑ Lift(τ_u1)
#     tr(∇_u) + τ_p = 0
#     ∂t(b) - κ ∇⋅∇_b + Lift(τ_b2) = -u⋅∇b
#     ∂t(u) - ν ∇⋅∇_u + ∇p - b ẑ + Lift(τ_u2) = -u⋅∇u
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
stop_time   = 50.0
max_dt      = 1e-2

kappa = (Rayleigh * Prandtl)^(-1/2)
nu    = (Rayleigh / Prandtl)^(-1/2)

@root_only begin
    println("2D Rayleigh-Benard Convection")
    @printf("  Ra=%.2e, Pr=%.1f, κ=%.6e, ν=%.6e\n", Rayleigh, Prandtl, kappa, nu)
    @printf("  Domain: %.1f × %.1f, Resolution: %d × %d\n\n", Lx, Lz, Nx, Nz)
end

# ─── Bases ────────────────────────────────────────────────────
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=dealias)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=dealias)

# ─── Fields ───────────────────────────────────────────────────
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")
b = ScalarField(domain, "b")
u = VectorField(domain, "u")

tau_p  = ScalarField(dist, "tau_p",  (), Float64)
tau_b1 = ScalarField(dist, "tau_b1", (xbasis,), Float64)
tau_b2 = ScalarField(dist, "tau_b2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

# ─── Substitutions (first-order reduction) ───────────────────
ex, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A) = lift(A, lift_basis, -1)

# First-order gradient substitutions
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_b = grad(b) + ez * τ_lift(tau_b1)

# ─── Problem ─────────────────────────────────────────────────
problem = IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2])

add_parameters!(problem,
    kappa=kappa, nu=nu, Lz=Lz, ez=ez,
    grad_u=grad_u, grad_b=grad_b, τ_lift=τ_lift)

# Continuity (first-order form)
add_equation!(problem, "trace(grad_u) + tau_p = 0")

# Buoyancy: ∂t(b) - κ∇²b = -u⋅∇b
add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -u⋅∇(b)")

# Momentum: ∂t(u) - ν∇²u + ∇p - b ẑ = -u⋅∇u
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - b*ez + τ_lift(tau_u2) = -u⋅∇(u)")

# Boundary conditions
add_bc!(problem, "b(z=0) = Lz")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "b(z=Lz) = 0")
add_bc!(problem, "u(z=Lz) = 0")
add_bc!(problem, "integ(p) = 0")

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

@root_only diagnose(solver)

# ─── Output ──────────────────────────────────────────────────
snapshots = add_file_handler("snapshots", dist,
    Dict("b" => b, "ux" => u.components[1], "uz" => u.components[2]);
    sim_dt=0.25, max_writes=50)
add_task!(snapshots, b;               name="buoyancy")
add_task!(snapshots, u.components[1]; name="ux")
add_task!(snapshots, u.components[2]; name="uz")

# ─── Initial Conditions ──────────────────────────────────────
x, z = local_grids(dist, xbasis, zbasis)
fill_random!(b, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(b) .*= z' .* (Lz .- z')       # Damp noise at walls
get_grid_data(b) .+= Lz .- z'               # Add linear background
ensure_layout!(b, :c)                        # Pre-compute coefficients for timestepper

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
             ensure_layout!(b, :g)
             max_b = global_max(dist, abs.(get_grid_data(b)))
             @root_only @printf("  iter=%d, t=%.4e, dt=%.4e, max|b|=%.4f\n",
                                 s.iteration, s.sim_time, s.dt, max_b)
         end
     ])

@root_only println("Done!")
