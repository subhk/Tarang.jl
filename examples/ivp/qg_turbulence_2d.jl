"""
Tarang script simulating 2D Quasi-Geostrophic turbulence on a doubly-periodic domain.
This script demonstrates solving a 2D Cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to NetCDF files. It should take about 2 cpu-minutes to run.

The system evolves the potential vorticity (PV) equation:

    dt(q) + u.grad(q) = -nu * (-Lap)^n * q

where q = Lap(psi) is the vorticity, psi is the streamfunction solved from the
Poisson equation, and u = skew(grad(psi)) is the divergence-free velocity.
We use 8th-order hyperviscosity (n=4) to provide small-scale dissipation while
minimizing impact on the inertial range.

For 2D turbulence, the system exhibits:
- Inverse energy cascade (energy -> large scales)
- Forward enstrophy cascade (enstrophy -> small scales)

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 julia --project=. examples/ivp/qg_turbulence_2d.jl
"""

using Tarang
using MPI
using Logging

# Initialize MPI
MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Parameters
Lx, Ly = 2π, 2π
Nx, Ny = 256, 256
nu = 1e-20                      # Hyperviscosity coefficient
dealias = 3/2
stop_sim_time = 50.0
timestepper = RK222
max_timestep = 0.01
dtype = Float64

# Bases
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=dtype)

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=dealias)
ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly), dealias=dealias)

# Fields
q     = ScalarField(dist, "q",     (xbasis, ybasis), dtype)        # Potential vorticity
ψ     = ScalarField(dist, "ψ",     (xbasis, ybasis), dtype)        # Streamfunction
u     = VectorField(dist, "u",     (xbasis, ybasis), dtype)        # Velocity
tau_ψ = ScalarField(dist, "tau_ψ", (), dtype)          # Tau for k=0 gauge

# Problem
problem = IVP([q, ψ, u, tau_ψ])

@substitutions problem begin
    "nu" => nu
end

@equations problem begin
    "Δ(ψ) + tau_ψ - q = 0"             # Streamfunction Poisson equation
    "u - skew(grad(ψ)) = 0"            # Velocity from streamfunction
    "∂t(q) + nu*Δ⁴(q) = -u⋅∇(q)"       # Vorticity evolution (8th-order hyperviscosity)
end

@bcs problem begin
    "integ(ψ) = 0"                     # Gauge: zero mean streamfunction
end

# Solver
solver = InitialValueSolver(problem, timestepper(); device="cpu")
solver.stop_sim_time = stop_sim_time

# Initial conditions: random vorticity filtered to wavenumber band k ∈ [3, 6]
fill_random!(q, "g"; seed=42+rank, distribution="normal", scale=1.0)
ensure_layout!(q, :c)

# kx, ky = Tarang.wavenumbers(xbasis), Tarang.wavenumbers(ybasis)
# for (iy, ky_val) in enumerate(ky), (ix, kx_val) in enumerate(kx)
#     k_mag = sqrt(kx_val^2 + ky_val^2)
#     q.data_c[ix, iy] *= (3 <= k_mag <= 6) / (1 + k_mag^2)  # Band-pass filter
# end
# Tarang.ensure_layout!(q, :g)

# Analysis
snapshots = add_file_handler("snapshots/snapshots", dist,
                Dict("q" => q, "psi" => ψ, "u" => u);
                sim_dt=0.25, max_writes=50)

add_task(snapshots, q; name="vorticity")
add_task(snapshots, ψ; name="streamfunction")

# CFL
cfl = CFL(solver; initial_dt=max_timestep, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)

add_velocity!(cfl, u)

# Flow properties
flow = GlobalFlowProperty(solver; cadence=10)

# Main loop
rank == 0 && @info "Starting main loop"
try
    while proceed(solver)
        timestep = compute_timestep(cfl)
        step!(solver, timestep)

        if (solver.iteration - 1) % 10 == 0
            ensure_layout!(q, :g)
            max_q = maximum(abs.(q.data_g))
            rank == 0 && @info "Iteration=$(solver.iteration), Time=$(solver.sim_time), dt=$timestep, max|q|=$max_q"
        end

        # Write snapshots
        wall_time = time()
        if should_write(snapshots, wall_time, solver.sim_time, solver.iteration)
            process!(snapshots; iteration=solver.iteration,
                           wall_time=wall_time, sim_time=solver.sim_time, timestep=timestep)
        end
    end
    
catch e
    @error "Exception raised, triggering end of main loop." exception=e
    rethrow(e)
finally
    log_stats(solver)
end

# Finalize MPI
MPI.Initialized() && !MPI.Finalized() && MPI.Finalize()
