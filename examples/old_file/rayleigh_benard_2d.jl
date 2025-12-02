"""
2D Rayleigh-Bénard Convection Example

Julia/Tarang implementation of the Dedalus Rayleigh-Bénard example
Demonstrates:
- 2D domain with horizontal periodicity and vertical boundaries
- MPI parallelization with both vertical and horizontal distribution
- PencilFFTs for 2D FFT operations
- IVP problem formulation and solving
- Analysis and output

Run with: mpiexec -n 4 julia rayleigh_benard_2d.jl
"""

using Tarang
using MPI
using Logging

# Initialize MPI
MPI.Init()

# Setup logging
setup_tarang_logging(level="INFO", mpi_aware=true)

function main()
    @info "Starting 2D Rayleigh-Bénard convection simulation"
    
    # Parameters
    Lx, Lz = 4.0, 1.0        # Domain size
    Nx, Nz = 256, 64         # Resolution
    Rayleigh = 2e6           # Rayleigh number
    Prandtl = 1.0            # Prandtl number
    dealias = 3.0/2.0        # Dealiasing factor
    stop_sim_time = 50.0     # Simulation time
    max_timestep = 0.125     # Maximum timestep
    dtype = Float64          # Data type
    
    # Derived parameters
    kappa = (Rayleigh * Prandtl)^(-1/2)  # Thermal diffusivity
    nu = (Rayleigh / Prandtl)^(-1/2)     # Viscosity
    
    @info "Problem parameters:"
    @info "  Domain: $Lx × $Lz"
    @info "  Resolution: $Nx × $Nz"
    @info "  Rayleigh number: $Rayleigh"
    @info "  Prandtl number: $Prandtl"
    @info "  Thermal diffusivity: $kappa"
    @info "  Viscosity: $nu"
    
    # Create coordinate system and distributor
    coords = CartesianCoordinates("x", "z")
    
    # Set up MPI mesh for both vertical and horizontal parallelization
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    
    # Create optimal 2D process mesh
    if size == 1
        mesh = (1, 1)
    elseif size == 2
        mesh = (1, 2)
    elseif size == 4
        mesh = (2, 2)
    elseif size == 8
        mesh = (2, 4)
    else
        # Find best factorization
        sqrt_size = Int(round(sqrt(size)))
        for i in sqrt_size:-1:1
            if size % i == 0
                mesh = (i, size ÷ i)
                break
            end
        end
    end
    
    @info "MPI configuration:"
    @info "  Total processes: $size"
    @info "  Process mesh: $mesh (vertical × horizontal)"
    @info "  Current rank: $rank"
    
    # Create distributor with 2D mesh
    dist = Distributor(coords, comm=comm, mesh=mesh, dtype=dtype)
    
    # Create bases
    x_basis = RealFourier(coords["x"], size=Nx, bounds=(0.0, Lx), dealias=dealias)
    z_basis = ChebyshevT(coords["z"], size=Nz, bounds=(0.0, Lz), dealias=dealias)
    
    @info "Bases:"
    @info "  x: RealFourier, size=$Nx, bounds=(0.0, $Lx)"
    @info "  z: ChebyshevT, size=$Nz, bounds=(0.0, $Lz)"
    @info "  PencilFFTs compatible: $(is_pencil_compatible((x_basis, z_basis)))"
    
    # Create domain
    domain = Domain(dist, (x_basis, z_basis))
    
    @info "Domain information:"
    @info "  Global shape: $(global_shape(domain))"
    @info "  Local shape: $(local_shape(domain))"
    @info "  Volume: $(volume(domain))"
    
    # Create fields
    p = ScalarField(dist, "pressure", (x_basis, z_basis), dtype)
    b = ScalarField(dist, "buoyancy", (x_basis, z_basis), dtype)
    u = VectorField(dist, coords, "velocity", (x_basis, z_basis), dtype)
    
    # Tau fields for boundary conditions (using lift method)
    tau_p  = ScalarField(dist, "tau_p", (), dtype)  # Pressure gauge
    tau_b1 = ScalarField(dist, "tau_b1", (x_basis,), dtype)  # Buoyancy BC 1
    tau_b2 = ScalarField(dist, "tau_b2", (x_basis,), dtype)  # Buoyancy BC 2
    tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,), dtype)  # Velocity BC 1
    tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,), dtype)  # Velocity BC 2
    
    @info "Created fields:"
    @info "  Scalar fields: p, b, tau_p, tau_b1, tau_b2"
    @info "  Vector fields: u, tau_u1, tau_u2"
    
    # Get coordinate grids
    x, z = local_grids(dist, x_basis, z_basis)
    
    @info "Local grids:"
    @info "  x: $(length(x)) points, range [$(minimum(x)), $(maximum(x))]"
    @info "  z: $(length(z)) points, range [$(minimum(z)), $(maximum(z))]"
    
    # Unit vectors
    ex, ez = unit_vector_fields(coords, dist)
    
    # Lift basis for boundary conditions
    lift_basis = derivative_basis(z_basis, 1)
    
    # Create problem
    variables = [p, b, u.components[1], u.components[2], tau_p, tau_b1, tau_b2, 
                tau_u1.components[1], tau_u1.components[2], tau_u2.components[1], tau_u2.components[2]]
    
    problem = IVP(variables)
    
    # Add substitutions to namespace
    add_substitution!(problem, "kappa", kappa)
    add_substitution!(problem, "nu", nu)
    add_substitution!(problem, "Lz", Lz)
    
    # Add equations following Dedalus conventions (nonlinear terms on RHS)
    add_equation!(problem, "div(u) + tau_p = 0")  # Continuity
    add_equation!(problem, "dt(b) - kappa*lap(b) + lift(tau_b2) = - u·grad(b)")  # Buoyancy (nonlinear advection on RHS)
    add_equation!(problem, "dt(u) - nu*lap(u) + grad(p) - b*ez + lift(tau_u2) = - u·grad(u)")  # Momentum (nonlinear advection on RHS)
    
    # Boundary conditions
    add_bc!(problem, "b(z=0) = Lz")      # Hot bottom
    add_bc!(problem, "b(z=Lz) = 0")      # Cold top
    add_bc!(problem, "u(z=0) = 0")       # No-slip bottom
    add_bc!(problem, "u(z=Lz) = 0")      # No-slip top
    add_bc!(problem, "integ(p) = 0")     # Pressure gauge
    
    @info "Problem setup:"
    @info "  Variables: $(length(problem.variables))"
    @info "  Equations: $(length(problem.equations))"
    @info "  Boundary conditions: $(length(problem.boundary_conditions))"
    
    # Validate problem
    try
        validate_problem(problem)
        @info "Problem validation: PASSED"
    catch e
        @error "Problem validation failed: $e"
        return
    end
    
    # Create solver
    timestepper = RK222()  # 2nd order Runge-Kutta
    solver = InitialValueSolver(problem, timestepper)
    solver.stop_sim_time = stop_sim_time
    
    @info "Solver configuration:"
    @info "  Timestepper: $(typeof(timestepper))"
    @info "  Stop time: $stop_sim_time"
    @info "  Max timestep: $max_timestep"
    
    # Initial conditions
    @info "Setting up initial conditions..."
    
    # Set buoyancy field: b = Lz - z + noise
    ensure_layout!(b, :g)
    b_data = b["g"]
    
    # Linear background profile
    for (i, z_val) in enumerate(z)
        for j in 1:length(x)
            b_data[j, i] = Lz - z_val
        end
    end
    
    # Add random noise
    Random.seed!(42)  # For reproducibility
    noise_scale = 1e-3
    noise = noise_scale * randn(size(b_data))
    
    # Damp noise at walls
    for (i, z_val) in enumerate(z)
        wall_factor = z_val * (Lz - z_val)  # Zero at walls, max at center
        for j in 1:length(x)
            noise[j, i] *= wall_factor
        end
    end
    
    b_data .+= noise
    
    @info "Initial conditions set:"
    @info "  Buoyancy range: [$(minimum(b_data)), $(maximum(b_data))]"
    
    # Create CFL condition
    cfl = CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, 
             threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
    add_velocity!(cfl, u)
    
    @info "CFL condition configured"
    
    # Flow properties for monitoring
    flow = GlobalFlowProperty(solver, cadence=10)
    add_property!(flow, "sqrt(u·u)/nu", "Re")  # Reynolds number
    
    # Analysis output
    evaluator = create_evaluator(solver)
    snapshots = add_file_handler(evaluator, "snapshots", sim_dt=0.25, max_writes=50)
    add_task!(snapshots, b, name="buoyancy")
    add_task!(snapshots, curl(u), name="vorticity")  # This would need proper 2D curl implementation
    
    @info "Analysis configured:"
    @info "  Snapshot interval: 0.25 time units"
    @info "  Maximum snapshots: 50"
    
    # Main time-stepping loop
    @info "Starting main simulation loop..."
    
    wall_time_start = time()
    
    try
        while proceed(solver)
            # Compute adaptive timestep
            timestep = compute_timestep(cfl)
            
            # Take step
            step!(solver, timestep)
            
            # Evaluate analysis
            wall_time = time() - wall_time_start
            evaluate_handlers!(evaluator, wall_time, solver.sim_time, solver.iteration)
            
            # Log progress
            if (solver.iteration - 1) % 10 == 0
                max_Re = max(flow, "Re")
                elapsed = time() - wall_time_start
                
                @info "Progress: iteration=$(solver.iteration), " *
                      "time=$(round(solver.sim_time, digits=4)), " *
                      "dt=$(round(timestep, digits=6)), " *
                      "max(Re)=$(round(max_Re, digits=2)), " *
                      "wall_time=$(round(elapsed, digits=2))s"
                      
                # Estimate completion time
                if solver.sim_time > 0
                    rate = solver.sim_time / elapsed
                    remaining_time = (stop_sim_time - solver.sim_time) / rate
                    @info "  Estimated completion in $(round(remaining_time, digits=1)) seconds"
                end
            end
        end
        
        @info "Simulation completed successfully!"
        
    catch e
        @error "Simulation error: $e"
        @error "Exception raised, triggering end of main loop."
        rethrow(e)
    finally
        # Performance statistics
        wall_time_total = time() - wall_time_start
        
        @info "Final statistics:"
        @info "  Total iterations: $(solver.iteration)"
        @info "  Final simulation time: $(solver.sim_time)"
        @info "  Total wall time: $(round(wall_time_total, digits=2)) seconds"
        @info "  Iterations per second: $(round(solver.iteration / wall_time_total, digits=2))"
        @info "  Time units per wall second: $(round(solver.sim_time / wall_time_total, digits=2))"
        
        if MPI.Initialized()
            # Gather statistics from all processes
            local_stats = [solver.iteration, wall_time_total]
            all_stats = MPI.Gather(local_stats, 0, comm)
            
            if rank == 0
                total_iterations = sum(stat[1] for stat in all_stats)
                avg_wall_time = sum(stat[2] for stat in all_stats) / length(all_stats)
                
                @info "MPI performance:"
                @info "  Total iterations across all processes: $(Int(total_iterations))"
                @info "  Average wall time per process: $(round(avg_wall_time, digits=2)) seconds"
                @info "  Parallel efficiency: $(round(wall_time_total / avg_wall_time * 100, digits=1))%"
            end
        end
        
        log_stats(solver)
    end
end

# Run the simulation
if abspath(PROGRAM_FILE) == @__FILE__
    main()
    
    # Finalize MPI
    MPI.Finalize()
end