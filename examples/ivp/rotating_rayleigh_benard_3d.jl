# 3D Rotating Rayleigh-Benard Convection
#
# Non-dimensionalized using box height H and thermal diffusion time τ_κ = H²/κ.
# Standard rRBC formulation with E/Pr scaling on the momentum equation:
#
#     ∂t(θ) - Δ(θ) = -u⋅∇(θ)
#     E/Pr * ∂t(u) - E*Δ(u) + ∇(p) + ez×u = Ra*θ*ez - u⋅∇(u)
#     ∇⋅u = 0
#
# Control parameters:
#     Ra = g α ΔT H³ / (κ ν)       Rayleigh number
#     Pr = ν / κ                     Prandtl number
#     E  = ν / (2Ω H²)             Ekman number
#
# Note: the E/Pr factor on ∂t(u) arises from scaling time by τ_κ = H²/κ
# and velocity by κ/H, giving the standard rapidly-rotating convection
# formulation where rotation enters as an O(1) Coriolis term ez×u.
#
# Boundary conditions (no-slip, fixed temperature):
#     θ(z=0) = 1,  θ(z=1) = 0
#     u(z=0) = 0,  u(z=1) = 0
#     integ(p) = 0
#
# Uses first-order formulation with derivative basis lifts.
#
# To run:
#     julia --project=. examples/ivp/rotating_rayleigh_benard_3d.jl
#     mpiexec -n 4 julia --project=. examples/ivp/rotating_rayleigh_benard_3d.jl

using Tarang
using Printf

# ─── Parameters ───────────────────────────────────────────────
Lx, Ly, Lz = 2.0, 2.0, 1.0
Nx, Ny, Nz  = 64, 64, 32
Rayleigh    = 1e6
Prandtl     = 1.0
Ekman       = 1e-3
dealias     = 3/2
stop_time   = 100.0
max_dt      = 1e-3

Ra  = Rayleigh
Pr  = Prandtl
Ek  = Ekman
EPr = Ek / Pr          # E/Pr prefactor on ∂t(u)

@root_only begin
    println("3D Rotating Rayleigh-Benard Convection")
    @printf("  Ra=%.2e, Pr=%.1f, E=%.2e, E/Pr=%.2e\n", Ra, Pr, Ek, EPr)
    @printf("  Domain: %.1f×%.1f×%.1f, Resolution: %d×%d×%d\n\n", Lx, Ly, Lz, Nx, Ny, Nz)
end

# ─── Bases ────────────────────────────────────────────────────
coords = CartesianCoordinates("x", "y", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=dealias)
ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly), dealias=dealias)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=dealias)

# ─── Fields ───────────────────────────────────────────────────
domain = Domain(dist, (xbasis, ybasis, zbasis))

p = ScalarField(domain, "p")      # Modified pressure
θ = ScalarField(domain, "θ")      # Temperature perturbation
u = VectorField(domain, "u")      # Velocity

tau_p  = ScalarField(dist, "tau_p",  (), Float64)
tau_θ1 = ScalarField(dist, "tau_θ1", (xbasis, ybasis), Float64)
tau_θ2 = ScalarField(dist, "tau_θ2", (xbasis, ybasis), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis, ybasis), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis, ybasis), Float64)

# ─── Substitutions (first-order reduction) ───────────────────
ex, ey, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A) = lift(A, lift_basis, -1)

# First-order reduction
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_θ = grad(θ) + ez * τ_lift(tau_θ1)

# ─── Problem ─────────────────────────────────────────────────
problem = IVP([p, θ, u, tau_p, tau_θ1, tau_θ2, tau_u1, tau_u2])

add_parameters!(problem,
    Ra=Ra, Ek=Ek, EPr=EPr, Lz=Lz, ez=ez,
    grad_u=grad_u, grad_θ=grad_θ, τ_lift=τ_lift)

# Continuity (first-order form)
add_equation!(problem, "trace(grad_u) + tau_p = 0")

# Temperature: ∂t(θ) - Δθ = -u⋅∇θ
add_equation!(problem, "∂t(θ) - div(grad_θ) + τ_lift(tau_θ2) = -u⋅∇(θ)")

# Momentum: E/Pr ∂t(u) - E Δu + ∇p + ez×u = Ra θ ez - u⋅∇u
add_equation!(problem, "EPr*∂t(u) - Ek*div(grad_u) + ∇(p) + curl(u) + τ_lift(tau_u2) = Ra*θ*ez - u⋅∇(u)")

# Boundary conditions
add_bc!(problem, "θ(z=0) = Lz")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "θ(z=Lz) = 0")
add_bc!(problem, "u(z=Lz) = 0")
add_bc!(problem, "integ(p) = 0")

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, ETD_RK222(); dt=max_dt)

@root_only diagnose(solver)

# ─── Output ──────────────────────────────────────────────────
snapshots = add_file_handler("snapshots", dist,
    Dict("θ" => θ, "ux" => u.components[1], "uy" => u.components[2], "uz" => u.components[3]);
    sim_dt=1.0, max_writes=100)
add_task!(snapshots, θ;                name="temperature")
add_task!(snapshots, u.components[1];  name="ux")
add_task!(snapshots, u.components[2];  name="uy")
add_task!(snapshots, u.components[3];  name="uz")

# ─── Initial Conditions ──────────────────────────────────────
# Conductive profile θ = 1 - z, plus small noise
x, y, z = local_grids(dist, xbasis, ybasis, zbasis)
fill_random!(θ, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(θ) .*= reshape(z, 1, 1, :) .* (Lz .- reshape(z, 1, 1, :))
get_grid_data(θ) .+= Lz .- reshape(z, 1, 1, :)

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
             ensure_layout!(θ, :g)
             max_θ = global_max(dist, abs.(get_grid_data(θ)))
             @root_only @printf("  iter=%d, t=%.4e, dt=%.4e, max|θ|=%.4f\n",
                                 s.iteration, s.sim_time, s.dt, max_θ)
         end
     ])

@root_only println("Done!")
