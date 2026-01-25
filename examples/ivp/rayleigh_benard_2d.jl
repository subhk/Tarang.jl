"""
2D Rayleigh-Benard Convection

This script simulates 2D horizontally-periodic Rayleigh-Benard convection.
The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)^(-1/2)
    nu = (Rayleigh / Prandtl)^(-1/2)

To run:
    julia --project=. examples/ivp/rayleigh_benard_2d.jl

To run in parallel:
    mpiexec -n 4 julia --project=. examples/ivp/rayleigh_benard_2d.jl
"""

using Tarang
using MPI
using Random
using Printf

# Initialize MPI
if !MPI.Initialized()
    MPI.Init()
end

function main()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Parameters
    Lx, Lz = 4.0, 1.0           # Domain size
    Nx, Nz = 256, 64            # Resolution
    Rayleigh = 2e6              # Rayleigh number
    Prandtl = 1.0               # Prandtl number
    dealias = 3/2               # Dealiasing factor
    stop_sim_time = 50.0        # End time
    max_timestep = 0.125        # Maximum timestep
    dtype = Float64

    # Derived parameters
    kappa = (Rayleigh * Prandtl)^(-1/2)  # Thermal diffusivity
    nu = (Rayleigh / Prandtl)^(-1/2)     # Viscosity

    if rank == 0
        println("2D Rayleigh-Benard Convection")
        println("=============================")
        @printf("Domain: %.1f x %.1f\n", Lx, Lz)
        @printf("Resolution: %d x %d\n", Nx, Nz)
        @printf("Rayleigh number: %.2e\n", Rayleigh)
        @printf("Prandtl number: %.1f\n", Prandtl)
        @printf("Thermal diffusivity (kappa): %.6e\n", kappa)
        @printf("Viscosity (nu): %.6e\n", nu)
        println()
    end

    # Bases
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=dtype)

    xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=dealias)
    zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=dealias)

    # Fields
    p = ScalarField(dist, "p", (xbasis, zbasis), dtype)   # Pressure
    b = ScalarField(dist, "b", (xbasis, zbasis), dtype)   # Buoyancy
    u = VectorField(dist, coords, "u", (xbasis, zbasis), dtype)  # Velocity

    # Tau fields for boundary conditions
    tau_p = ScalarField(dist, "tau_p", (), dtype)                    # Pressure gauge
    tau_b1 = ScalarField(dist, "tau_b1", (xbasis,), dtype)           # Buoyancy BC 1
    tau_b2 = ScalarField(dist, "tau_b2", (xbasis,), dtype)           # Buoyancy BC 2
    tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), dtype)   # Velocity BC 1
    tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), dtype)   # Velocity BC 2

    # Problem setup
    # Collect all state variables
    variables = [p, b,
                 u.components[1], u.components[2],  # ux, uz
                 tau_p, tau_b1, tau_b2,
                 tau_u1.components[1], tau_u1.components[2],
                 tau_u2.components[1], tau_u2.components[2]]

    problem = IVP(variables)

    # Add substitutions
    Tarang.add_substitution!(problem, "kappa", kappa)
    Tarang.add_substitution!(problem, "nu", nu)
    Tarang.add_substitution!(problem, "Lz", Lz)

    # Equations (following first-order formulation with vector notation)
    # Continuity: ∇⋅u = 0
    Tarang.add_equation!(problem, "div(u) + tau_p = 0")

    # Buoyancy: ∂b/∂t - κ∇²b = -u⋅∇b
    Tarang.add_equation!(problem, "∂t(b) - kappa*Δ(b) + lift(tau_b2) = -u⋅∇(b)")

    # Momentum (vector form): ∂u/∂t - ν∇²u + ∇p = -u⋅∇u + b*ez
    # ez is the unit vector in z-direction (buoyancy acts vertically)
    Tarang.add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) + lift(tau_u2) = -u⋅∇(u) + b*ez")

    # Boundary conditions
    Tarang.add_bc!(problem, "b(z=0) = Lz")      # Hot bottom
    Tarang.add_bc!(problem, "b(z=Lz) = 0")      # Cold top
    Tarang.add_bc!(problem, "u(z=0) = 0")       # No-slip bottom (vector notation)
    Tarang.add_bc!(problem, "u(z=Lz) = 0")      # No-slip top (vector notation)
    # Pressure gauge: tau_p removes mathematical degeneracy, integ(p)=0 fixes physical gauge
    Tarang.add_bc!(problem, "integ(p) = 0")

    # Solver
    timestepper = RK222()
    solver = InitialValueSolver(problem, timestepper; device="cpu")
    solver.stop_sim_time = stop_sim_time

    if rank == 0
        println("Solver initialized")
        println("  Timestepper: RK222")
        @printf("  Stop time: %.1f\n", stop_sim_time)
        println()
    end

    # Initial conditions
    Tarang.ensure_layout!(b, :g)

    # Get local grids
    grids = Tarang.local_grids(dist, xbasis, zbasis)
    x_grid = grids[1]
    z_grid = grids[2]

    # Linear background profile: b = Lz - z
    for (iz, z_val) in enumerate(z_grid)
        for ix in 1:length(x_grid)
            b.data_g[ix, iz] = Lz - z_val
        end
    end

    # Add random noise damped at walls
    Random.seed!(42 + rank)  # Different seed per rank for parallel reproducibility
    noise_scale = 1e-3

    for (iz, z_val) in enumerate(z_grid)
        wall_factor = z_val * (Lz - z_val)  # Zero at walls
        for ix in 1:length(x_grid)
            b.data_g[ix, iz] += noise_scale * randn() * wall_factor
        end
    end

    # Compute global min/max across all MPI ranks
    b_min = global_min(dist, b.data_g)
    b_max = global_max(dist, b.data_g)
    if rank == 0
        println("Initial conditions set")
        @printf("  Buoyancy range: [%.4f, %.4f]\n", b_min, b_max)
        println()
    end

    # CFL condition
    cfl = CFL(solver; initial_dt=max_timestep, cadence=10, safety=0.5,
              threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
    Tarang.add_velocity!(cfl, u)

    # Flow properties
    flow = GlobalFlowProperty(solver; cadence=10)

    # Analysis output
    snapshots = Tarang.add_file_handler("snapshots", dist, Dict("b" => b, "u" => u);
                                        parallel="gather", max_writes=50)
    Tarang.add_task(snapshots, b; name="buoyancy")

    if rank == 0
        println("Starting main loop")
        println("==================")
    end

    # Main loop
    wall_time_start = time()

    try
        while Tarang.proceed(solver)
            # Compute adaptive timestep
            timestep = Tarang.compute_timestep(cfl)

            # Take a step
            Tarang.step!(solver, timestep)

            # Log progress
            if (solver.iteration - 1) % 10 == 0
                # Compute max Reynolds number: Re = |u| / nu
                Tarang.ensure_layout!(u.components[1], :g)
                Tarang.ensure_layout!(u.components[2], :g)

                vel_mag_sq = u.components[1].data_g.^2 .+ u.components[2].data_g.^2
                local_max_vel = sqrt(maximum(vel_mag_sq))

                # MPI reduction for global max
                global_max_vel = MPI.Allreduce(local_max_vel, MPI.MAX, comm)
                max_Re = global_max_vel / nu

                if rank == 0
                    elapsed = time() - wall_time_start
                    @printf("Iteration=%d, Time=%.4e, dt=%.4e, max(Re)=%.2f, wall_time=%.1fs\n",
                            solver.iteration, solver.sim_time, timestep, max_Re, elapsed)
                end
            end

            # Write snapshots
            if solver.iteration % 100 == 0
                wall_time = time() - wall_time_start
                Tarang.process!(snapshots; iteration=solver.iteration,
                               wall_time=wall_time, sim_time=solver.sim_time,
                               timestep=timestep)
            end
        end

        if rank == 0
            println()
            println("Simulation completed successfully!")
        end

    catch e
        if rank == 0
            println("Exception raised: $e")
        end
        rethrow(e)
    finally
        # Final statistics
        wall_time_total = time() - wall_time_start

        if rank == 0
            println()
            println("Final Statistics")
            println("================")
            @printf("Total iterations: %d\n", solver.iteration)
            @printf("Final simulation time: %.4f\n", solver.sim_time)
            @printf("Total wall time: %.2f seconds\n", wall_time_total)
            @printf("Iterations per second: %.2f\n", solver.iteration / wall_time_total)
            @printf("Simulation time per wall second: %.4f\n", solver.sim_time / wall_time_total)
        end
    end
end

# Run
main()

# Finalize MPI
if MPI.Initialized() && !MPI.Finalized()
    MPI.Finalize()
end
