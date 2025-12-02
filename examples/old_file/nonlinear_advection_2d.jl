"""
2D Nonlinear Advection Example with PencilArrays and PencilFFTs

This example demonstrates the nonlinear term evaluation capabilities of Tarang.jl,
specifically showing how nonlinear terms like u·∇u are handled efficiently using:
- PencilArrays for distributed memory parallelization  
- PencilFFTs for parallel 2D FFT operations
- Proper dealiasing using the 3/2 rule
- Both horizontal and vertical parallelization

The example solves a 2D incompressible flow with nonlinear advection:
∂u/∂t + (u·∇)u = -∇p + ν∇²u
∇·u = 0

Run with: mpiexec -n 4 julia nonlinear_advection_2d.jl
"""

using Tarang
using MPI
using Random
using Logging

# Initialize MPI
MPI.Init()

function main()
    @info "Starting 2D nonlinear advection example with PencilArrays/PencilFFTs"
    
    # Parameters
    Lx, Ly = 2π, 2π       # Domain size (periodic)
    Nx, Ny = 128, 128     # Resolution
    nu = 1e-3             # Viscosity
    dealias = 3.0/2.0     # Dealiasing factor (3/2 rule)
    stop_sim_time = 1.0   # Simulation time
    max_timestep = 0.01   # Maximum timestep
    dtype = Float64       # Data type
    
    # MPI setup for 2D parallelization
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    
    # Optimal 2D process mesh for PencilFFTs
    if size == 1
        mesh = (1, 1)
    elseif size == 2
        mesh = (1, 2)  # Horizontal parallelization
    elseif size == 4
        mesh = (2, 2)  # Both horizontal and vertical
    elseif size == 8
        mesh = (2, 4)  # More horizontal parallelization
    else
        sqrt_size = Int(round(sqrt(size)))
        for i in sqrt_size:-1:1
            if size % i == 0
                mesh = (i, size ÷ i)
                break
            end
        end
    end
    
    @info "MPI configuration for nonlinear terms:"
    @info "  Total processes: $size"
    @info "  Process mesh: $mesh (enables both vertical and horizontal parallelization)"
    @info "  Current rank: $rank"
    
    # Create coordinate system and distributor
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords, comm=comm, mesh=mesh, dtype=dtype)
    
    # Create bases (periodic in both directions for optimal FFT performance)
    x_basis = RealFourier(coords["x"], size=Nx, bounds=(0.0, Lx), dealias=dealias)
    y_basis = RealFourier(coords["y"], size=Ny, bounds=(0.0, Ly), dealias=dealias)
    
    @info "Bases configuration:"
    @info "  x: RealFourier, size=$Nx, bounds=(0.0, $Lx)"
    @info "  y: RealFourier, size=$Ny, bounds=(0.0, $Ly)"
    @info "  Dealiasing factor: $dealias"
    @info "  PencilFFTs compatible: $(is_pencil_compatible((x_basis, y_basis)))"
    
    # Create domain
    domain = Domain(dist, (x_basis, y_basis))
    
    @info "Domain information:"
    @info "  Global shape: $(global_shape(domain))"
    @info "  Local shape: $(local_shape(domain))"
    @info "  Optimal for 2D PencilFFTs: $(is_pencil_compatible((x_basis, y_basis)))"
    
    # Create fields
    p = ScalarField(dist, "pressure", (x_basis, y_basis), dtype)
    u = VectorField(dist, coords, "velocity", (x_basis, y_basis), dtype)
    omega = ScalarField(dist, "vorticity", (x_basis, y_basis), dtype)  # For analysis
    
    # Pressure gauge condition (needed for incompressible flow)
    tau_p = ScalarField(dist, "tau_p", (), dtype)
    
    @info "Created fields for nonlinear advection:"
    @info "  Velocity field: u (2-component vector)"
    @info "  Pressure field: p (scalar)"
    @info "  Vorticity field: ω (scalar, for analysis)"
    
    # Get coordinate grids
    x, y = local_grids(dist, x_basis, y_basis)
    
    # Create problem
    variables = [u.components[1], u.components[2], p, tau_p]
    problem = IVP(variables)
    
    # Add substitutions
    add_substitution!(problem, "nu", nu)
    
    # Create nonlinear operators using the new nonlinear terms module
    @info "Setting up nonlinear operators..."
    
    # Nonlinear momentum term: (u·∇)u
    nonlinear_u = nonlinear_momentum(u)
    
    # Add equations following Dedalus conventions:
    # LHS: Linear terms only (time derivatives, linear operators)
    # RHS: Nonlinear terms, source terms, etc.
    
    # Momentum equation: ∂u/∂t + ∇p - ν∇²u = -(u·∇)u
    add_equation!(problem, "dt(u) + grad(p) - nu*lap(u) = -(u·grad(u))")  # Nonlinear term on RHS
    add_equation!(problem, "div(u) + tau_p = 0")                          # Incompressibility constraint
    
    # Boundary conditions (periodic, automatically satisfied)
    add_bc!(problem, "integ(p) = 0")  # Pressure gauge
    
    @info "Problem setup with nonlinear terms:"
    @info "  Variables: $(length(problem.variables))"
    @info "  Equations: $(length(problem.equations))"
    @info "  Nonlinear operators: 1 (nonlinear momentum advection)"
    @info "  Dealiasing: $dealias rule for nonlinear term evaluation"
    
    # Create solver with appropriate timestepper for nonlinear problems
    timestepper = RK443()  # 4th order Runge-Kutta (good for nonlinear problems)
    solver = InitialValueSolver(problem, timestepper)
    solver.stop_sim_time = stop_sim_time
    
    @info "Solver configuration:"
    @info "  Timestepper: $(typeof(timestepper)) (suitable for nonlinear problems)"
    @info "  Stop time: $stop_sim_time"
    @info "  Max timestep: $max_timestep"
    
    # Initial conditions: Create a vortical flow structure
    @info "Setting up initial conditions with vortical structure..."
    
    # Set initial velocity field with Taylor-Green vortex pattern
    ensure_layout!(u.components[1], :g)  # u_x component
    ensure_layout!(u.components[2], :g)  # u_y component
    
    u_data = u.components[1]["g"]
    v_data = u.components[2]["g"]
    
    # Taylor-Green vortex initial condition
    for (i, x_val) in enumerate(x), (j, y_val) in enumerate(y)
        u_data[i, j] =  sin(x_val) * cos(y_val)  # u_x
        v_data[i, j] = -cos(x_val) * sin(y_val)  # u_y
    end
    
    # Add small perturbation to trigger nonlinear instability
    Random.seed!(42 + rank)  # Different seed per process
    perturbation_scale = 0.01
    
    for (i, x_val) in enumerate(x), (j, y_val) in enumerate(y)
        # Add small-scale random perturbation
        u_data[i, j] += perturbation_scale * randn() * sin(2*x_val) * sin(2*y_val)
        v_data[i, j] += perturbation_scale * randn() * cos(2*x_val) * cos(2*y_val)
    end
    
    @info "Initial conditions set:"
    @info "  Base flow: Taylor-Green vortex"
    @info "  Perturbations: Small-scale random (scale: $perturbation_scale)"
    @info "  u_x range: [$(minimum(u_data)), $(maximum(u_data))]"
    @info "  u_y range: [$(minimum(v_data)), $(maximum(v_data))]"
    
    # Create CFL condition
    cfl = CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.3,  # Lower safety for nonlinear
             threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
    add_velocity!(cfl, u)
    
    @info "CFL condition configured for nonlinear problem"
    
    # Flow properties for monitoring nonlinear effects
    flow = GlobalFlowProperty(solver, cadence=10)
    add_property!(flow, "sqrt(u·u)", "velocity_magnitude")
    add_property!(flow, "u·u/2", "kinetic_energy")
    add_property!(flow, "enstrophy", "enstrophy")  # ω²/2, measures vorticity
    
    # Analysis output with enhanced monitoring
    evaluator = create_evaluator(solver)
    
    # Snapshots for visualization
    snapshots = add_file_handler(evaluator, "nonlinear_snapshots", sim_dt=0.1, max_writes=10)
    add_task!(snapshots, u.components[1], name="u_x")
    add_task!(snapshots, u.components[2], name="u_y")
    add_task!(snapshots, omega, name="vorticity")  # Will be computed from curl(u)
    
    # Performance monitoring specifically for nonlinear terms
    nonlinear_stats = NonlinearPerformanceStats()
    
    @info "Analysis configured:"
    @info "  Snapshot interval: 0.1 time units"
    @info "  Maximum snapshots: 10"
    @info "  Monitoring: velocity magnitude, kinetic energy, enstrophy"
    @info "  Nonlinear performance tracking: enabled"
    
    # Main time-stepping loop with nonlinear term monitoring
    @info "Starting main simulation loop with nonlinear advection..."
    
    wall_time_start = time()
    
    try
        while proceed(solver)
            # Compute adaptive timestep
            timestep = compute_timestep(cfl)
            
            # Time nonlinear evaluation (this happens inside the timestepper)
            nl_start = time()
            
            # Take step (this will trigger nonlinear term evaluation)
            step!(solver, timestep)
            
            nl_time = time() - nl_start
            nonlinear_stats.total_evaluations += 1
            nonlinear_stats.total_time += nl_time
            
            # Compute vorticity for analysis: ω = ∂v/∂x - ∂u/∂y
            if solver.iteration % 10 == 0
                # This demonstrates nonlinear operator evaluation
                omega_op = curl(u)  # This will use the 2D curl implementation
                omega_result = evaluate_operator(omega_op)
                if isa(omega_result, ScalarField)
                    omega = omega_result
                end
            end
            
            # Evaluate analysis
            wall_time = time() - wall_time_start
            evaluate_handlers!(evaluator, wall_time, solver.sim_time, solver.iteration)
            
            # Log progress with nonlinear-specific metrics
            if (solver.iteration - 1) % 10 == 0
                max_vel = max(flow, "velocity_magnitude")
                kinetic_energy = max(flow, "kinetic_energy")
                elapsed = time() - wall_time_start
                
                @info "Progress: iteration=$(solver.iteration), " *
                      "time=$(round(solver.sim_time, digits=4)), " *
                      "dt=$(round(timestep, digits=6)), " *
                      "max|u|=$(round(max_vel, digits=3)), " *
                      "KE=$(round(kinetic_energy, digits=3)), " *
                      "wall_time=$(round(elapsed, digits=2))s"
                      
                # Nonlinear evaluation statistics
                if nonlinear_stats.total_evaluations > 0
                    avg_nl_time = nonlinear_stats.total_time / nonlinear_stats.total_evaluations
                    @info "  Nonlinear evaluation: $(nonlinear_stats.total_evaluations) calls, " *
                          "avg_time=$(round(avg_nl_time*1000, digits=3))ms per call"
                end
                
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
        
        # Nonlinear performance analysis
        log_nonlinear_performance(nonlinear_stats)
        
        if MPI.Initialized()
            # Gather statistics from all processes for parallel efficiency analysis
            local_stats = [solver.iteration, wall_time_total, nonlinear_stats.total_time]
            all_stats = MPI.Gather(local_stats, 0, comm)
            
            if rank == 0
                total_iterations = sum(stat[1] for stat in all_stats)
                avg_wall_time = sum(stat[2] for stat in all_stats) / length(all_stats)
                avg_nl_time = sum(stat[3] for stat in all_stats) / length(all_stats)
                
                @info "MPI performance for nonlinear problem:"
                @info "  Total iterations across all processes: $(Int(total_iterations))"
                @info "  Average wall time per process: $(round(avg_wall_time, digits=2)) seconds"
                @info "  Average nonlinear time per process: $(round(avg_nl_time, digits=2)) seconds"
                @info "  Nonlinear overhead: $(round(100*avg_nl_time/max(avg_wall_time, 1e-10), digits=1))%"
                @info "  Parallel efficiency: $(round(wall_time_total / avg_wall_time * 100, digits=1))%"
            end
        end
        
        log_stats(solver)
    end
end

# Enhanced logging for nonlinear problems
function setup_nonlinear_logging()
    """Setup enhanced logging for nonlinear term evaluation"""
    
    # Enable debug logging for nonlinear evaluation if requested
    if get(ENV, "VARUNA_DEBUG_NONLINEAR", "false") == "true"
        @info "Enabling debug logging for nonlinear terms"
        global_logger(SimpleLogger(stderr, Logging.Debug))
    end
end

# Run the simulation
if abspath(PROGRAM_FILE) == @__FILE__
    setup_nonlinear_logging()
    main()
    
    # Finalize MPI
    MPI.Finalize()
end