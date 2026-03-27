# 2D Rayleigh-Benard Convection
#
# Simulates 2D horizontally-periodic Rayleigh-Benard convection.
# Non-dimensionalized using box height and freefall time:
#
#     kappa = (Rayleigh * Prandtl)^(-1/2)
#     nu = (Rayleigh / Prandtl)^(-1/2)
#
# To run:
#     julia --project=. examples/ivp/rayleigh_benard_2d.jl
#
# To run in parallel:
#     mpiexec -n 4 julia --project=. examples/ivp/rayleigh_benard_2d.jl

using Tarang
using Printf

# ─── Parameters ───────────────────────────────────────────────
Lx, Lz      = 4.0, 1.0
Nx, Nz      = 256, 64
Rayleigh    = 2e6
Prandtl     = 1.0
stop_time   = 50.0
max_dt      = 0.125

kappa = (Rayleigh * Prandtl)^(-1/2)
nu    = (Rayleigh / Prandtl)^(-1/2)

@root_only begin
    println("2D Rayleigh-Benard Convection")
    @printf("  Ra=%.2e, Pr=%.1f, κ=%.6e, ν=%.6e\n", Rayleigh, Prandtl, kappa, nu)
    @printf("  Domain: %.1f × %.1f, Resolution: %d × %d\n\n", Lx, Lz, Nx, Nz)
end

# ─── Domain & Fields ──────────────────────────────────────────
domain = ChannelDomain(Nx, Nz; Lx=Lx, Lz=Lz, dealias=3/2)
dist   = domain.dist
coords = dist.coordsys

p = ScalarField(domain, "p")               # Pressure
b = ScalarField(domain, "b")               # Buoyancy
u = VectorField(domain, "u")               # Velocity

# Tau fields for boundary conditions
tau_p  = ScalarField(dist, "tau_p",  (), Float64)
tau_b1 = ScalarField(dist, "tau_b1", (domain.bases[1],), Float64)
tau_b2 = ScalarField(dist, "tau_b2", (domain.bases[1],), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (domain.bases[1],), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (domain.bases[1],), Float64)

# ─── Problem ─────────────────────────────────────────────────
variables = [p, b,
             u.components[1], u.components[2],
             tau_p, tau_b1, tau_b2,
             tau_u1.components[1], tau_u1.components[2],
             tau_u2.components[1], tau_u2.components[2]]

problem = IVP(variables)
add_parameters!(problem, kappa=kappa, nu=nu, Lz=Lz)

# Continuity: ∇·u = 0
add_equation!(problem, "div(u) + tau_p = 0")

# Buoyancy: ∂b/∂t - κ∇²b = -u·∇b
add_equation!(problem, "∂t(b) - kappa*Δ(b) + lift(tau_b1, -1) + lift(tau_b2, -2) = -u⋅∇(b)")

# Momentum: ∂u/∂t - ν∇²u + ∇p = -u·∇u + b*ez
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) + lift(tau_u1, -1) + lift(tau_u2, -2) = -u⋅∇(u) + b*ez")

# Boundary conditions
fixed_value!(problem, "b", "z", 0.0, Lz)   # Hot bottom
fixed_value!(problem, "b", "z", Lz, 0.0)   # Cold top
no_slip!(problem, "u", "z", 0.0)            # No-slip bottom
no_slip!(problem, "u", "z", Lz)             # No-slip top
add_bc!(problem, "integ(p) = 0")            # Pressure gauge

# ─── Solver ───────────────────────────────────────────────────
solver = InitialValueSolver(problem, RK222(); dt=max_dt)
solver.stop_sim_time = stop_time

@root_only diagnose(solver)

# ─── Initial Conditions ──────────────────────────────────────
ensure_layout!(b, :g)
xb, zb = domain.bases
grids = local_grids(dist, xb, zb)
x_grid, z_grid = grids[1], grids[2]

b_data = get_grid_data(b)
for (iz, z) in enumerate(z_grid), ix in 1:length(x_grid)
    wall_factor = z * (Lz - z)
    b_data[ix, iz] = (Lz - z) + 1e-3 * randn() * wall_factor
end

# ─── CFL & Diagnostics ───────────────────────────────────────
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)

# ─── Main Loop ────────────────────────────────────────────────
run!(solver;
     stop_time=stop_time,
     log_interval=100,
     callbacks=[
         on_interval(10) do s
             ux_data = grid_data(u.components[1])
             uy_data = grid_data(u.components[2])
             vel2 = ux_data .^ 2 .+ uy_data .^ 2
             max_Re = sqrt(maximum(vel2)) / nu
             @root_only @printf("  t=%.4e, dt=%.4e, max(Re)=%.2f\n",
                                 s.sim_time, s.dt, max_Re)
         end
     ])

@root_only println("Done!")
