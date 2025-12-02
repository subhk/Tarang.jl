"""
3D Taylor-Green Vortex Simulation

A classic test case for 3D incompressible Navier-Stokes equations demonstrating:
- Full 3D domain with triple-periodic boundary conditions
- 3D PencilFFTs for optimal parallel performance
- 3D process mesh decomposition (e.g., 2×2×2 for 8 processes)
- Complete 3D vector calculus operations (grad, div, curl)
- Energy cascade and enstrophy evolution
- Advanced 3D analysis and visualization

The Taylor-Green vortex exhibits transition from laminar to turbulent flow,
making it an excellent benchmark for 3D turbulence simulations.

Run with: mpiexec -n 8 julia taylor_green_vortex_3d.jl
"""

using Tarang
using MPI
using Logging
using Random

# Initialize MPI
MPI.Init()

# Setup logging
setup_tarang_logging(level="INFO", mpi_aware=true)

function main()
    @info "Starting 3D Taylor-Green Vortex simulation"
    
    # Parameters
    N = 128                    # Resolution in each direction (N³ total)
    Re = 1600.0               # Reynolds number
    nu = 1.0 / Re             # Kinematic viscosity  
    stop_sim_time = 20.0      # Simulation time (about 2 eddy turnover times)
    max_timestep = 0.01       # Maximum timestep
    dealias = 3.0/2.0         # Dealiasing factor
    dtype = Float64           # Data type
    
    @info "Problem parameters:"
    @info "  Resolution: $N³ = $(N^3) grid points"
    @info "  Reynolds number: $Re"
    @info "  Viscosity: $nu"
    @info "  Domain: [0,2π]³ (triply periodic)"
    @info "  Stop time: $stop_sim_time"
    
    # Create 3D coordinate system and distributor
    coords = CartesianCoordinates("x", "y", "z")
    
    # Set up optimal 3D MPI mesh
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    
    # Create 3D process mesh (auto-optimized)
    mesh = create_3d_process_mesh(size)
    
    @info "MPI configuration:"
    @info "  Total processes: $size"
    @info "  3D process mesh: $mesh (nx × ny × nz)"
    @info "  Current rank: $rank"
    @info "  Local domain size: ~$(div(N, mesh[1])) × $(div(N, mesh[2])) × $(div(N, mesh[3]))"
    
    # Create distributor with 3D mesh
    dist = Distributor(coords, comm=comm, mesh=mesh, dtype=dtype)
    
    # Create 3D Taylor-Green domain: [0,2π]³ with triple Fourier basis
    domain = taylor_green_vortex_domain(dist, N, dtype=dtype, dealias=dealias)
    
    @info "Domain configuration:"
    @info "  Global shape: $(global_shape(domain))"
    @info "  Local shape: $(local_shape(domain))"
    @info "  Volume: $(volume(domain))"
    @info "  3D PencilFFTs optimal: $(is_3d_pencil_optimal(domain.bases))"
    
    # Performance analysis
    analyze_3d_performance(domain, 4)  # u (3 components) + p = 4 fields
    
    # Create fields for 3D Navier-Stokes
    fields = create_navier_stokes_3d_fields(domain)
    u = fields["velocity"]
    p = fields["pressure"]
    tau_u = fields["tau_velocity"]
    tau_p = fields["tau_pressure"]
    
    @info "Created 3D fields:"
    @info "  Velocity: $(u.name) (3D vector field)"
    @info "  Pressure: $(p.name) (scalar field)"
    @info "  Tau terms: $(tau_u.name), $(tau_p.name)"
    
    # Get coordinate grids
    x, y, z = local_grids(dist, domain.bases...)
    
    @info "Local coordinate ranges:"
    @info "  x: [$(minimum(x)), $(maximum(x))] ($(length(x)) points)"
    @info "  y: [$(minimum(y)), $(maximum(y))] ($(length(y)) points)" 
    @info "  z: [$(minimum(z)), $(maximum(z))] ($(length(z)) points)"
    
    # Set up 3D Taylor-Green initial conditions
    @info "Setting up Taylor-Green initial conditions..."
    
    # Classic Taylor-Green vortex initial condition:
    # u = sin(x)cos(y)cos(z)
    # v = -cos(x)sin(y)cos(z) 
    # w = 0
    # p = (1/4)[cos(2x) + cos(2y)][cos(2z) + 2]
    
    ensure_layout!(u.components[1], :g)  # u-component
    ensure_layout!(u.components[2], :g)  # v-component  
    ensure_layout!(u.components[3], :g)  # w-component
    ensure_layout!(p, :g)                # pressure
    
    u_data = u.components[1]["g"]
    v_data = u.components[2]["g"]
    w_data = u.components[3]["g"]
    p_data = p["g"]
    
    # Set initial velocity field
    for (k, z_val) in enumerate(z), (j, y_val) in enumerate(y), (i, x_val) in enumerate(x)
        u_data[i, j, k] = sin(x_val) * cos(y_val) * cos(z_val)
        v_data[i, j, k] = -cos(x_val) * sin(y_val) * cos(z_val)
        w_data[i, j, k] = 0.0
        p_data[i, j, k] = 0.25 * (cos(2*x_val) + cos(2*y_val)) * (cos(2*z_val) + 2)
    end
    
    # Verify initial conditions
    u_max = maximum(abs.(u_data))
    v_max = maximum(abs.(v_data))
    w_max = maximum(abs.(w_data))
    
    # Reduce across MPI processes
    if MPI.Initialized()
        u_max = MPI.Allreduce(u_max, MPI.MAX, comm)
        v_max = MPI.Allreduce(v_max, MPI.MAX, comm)
        w_max = MPI.Allreduce(w_max, MPI.MAX, comm)
    end
    
    @info "Initial conditions set:"
    @info "  max(|u|): $u_max"
    @info "  max(|v|): $v_max" 
    @info "  max(|w|): $w_max"
    
    # Check divergence-free condition
    div_u = evaluate_operator(div(u))
    ensure_layout!(div_u, :g)
    div_max = maximum(abs.(div_u["g"]))
    if MPI.Initialized()
        div_max = MPI.Allreduce(div_max, MPI.MAX, comm)
    end
    @info "  max(|∇⋅u|): $div_max (should be ~machine epsilon)"
    
    # Create 3D Navier-Stokes problem
    variables = [u.components[1], u.components[2], u.components[3], p, tau_p]
    problem = IVP(variables)
    
    # Add 3D Navier-Stokes equations
    add_equation!(problem, "dt(u) - nu*lap(u) + grad(p) = -u·grad(u)")     # Momentum
    add_equation!(problem, "div(u) + tau_p = 0")                           # Continuity
    add_equation!(problem, "integ(p) = 0")                                 # Pressure gauge
    
    # Set parameters
    add_substitution!(problem, "nu", nu)
    
    @info "Problem setup:"
    @info "  Variables: $(length(problem.variables))"
    @info "  Equations: $(length(problem.equations))"
    @info "  3D Navier-Stokes with periodic BCs"
    
    # Validate problem
    try
        validate_problem(problem)
        @info "Problem validation: PASSED"
    catch e
        @error "Problem validation failed: $e"
        return
    end
    
    # Create solver with appropriate timestepper for 3D
    timestepper = RK443()  # 4th order for accuracy in turbulent flow
    solver = InitialValueSolver(problem, timestepper)
    solver.stop_sim_time = stop_sim_time
    
    @info "Solver configuration:"
    @info "  Timestepper: $(typeof(timestepper)) (4th order)"
    @info "  Stop time: $stop_sim_time"
    @info "  Max timestep: $max_timestep"
    
    # Create CFL condition for 3D
    cfl = CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2,  # Lower safety for 3D
             threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
    add_velocity!(cfl, u)
    
    @info "CFL condition configured for 3D"
    
    # 3D flow properties for monitoring
    flow = GlobalFlowProperty(solver, cadence=10)
    add_property!(flow, "sqrt(u·u)", "velocity_rms")
    add_property!(flow, "sqrt((curl(u))·(curl(u)))", "enstrophy")
    add_property!(flow, "sqrt(u·u)/nu", "Re_local")
    
    # Analysis output
    evaluator = create_evaluator(solver)
    
    # Snapshots (less frequent due to 3D data size)
    snapshots = add_file_handler(evaluator, "snapshots_3d", sim_dt=1.0, max_writes=20)
    add_task!(snapshots, u, name="velocity")
    add_task!(snapshots, p, name="pressure")
    add_task!(snapshots, curl(u), name="vorticity")
    
    # Scalars (more frequent)
    scalars = add_file_handler(evaluator, "scalars", sim_dt=0.1, max_writes=200)
    add_task!(scalars, "kinetic_energy", "0.5 * integ(u·u)")
    add_task!(scalars, "enstrophy", "0.5 * integ((curl(u))·(curl(u)))")
    add_task!(scalars, "dissipation", "nu * integ(grad(u)·grad(u))")
    
    @info "Analysis configured:"
    @info "  3D snapshots: every 1.0 time units (max 20)"
    @info "  Scalars: every 0.1 time units (max 200)"
    @info "  Total data: ~$(round(N^3 * 4 * sizeof(Float64) * 20 / 1024^3, digits=2)) GB"
    
    # Main 3D simulation loop
    @info "Starting 3D Taylor-Green vortex evolution..."
    
    wall_time_start = time()
    
    try
        while proceed(solver)
            # Compute adaptive timestep (important for 3D turbulence)
            timestep = compute_timestep(cfl)
            
            # Take step
            step!(solver, timestep)
            
            # Evaluate analysis
            wall_time = time() - wall_time_start
            evaluate_handlers!(evaluator, wall_time, solver.sim_time, solver.iteration)
            
            # Log progress (less frequent due to 3D overhead)
            if (solver.iteration - 1) % 50 == 0
                velocity_rms = volume_average(flow, "velocity_rms")
                enstrophy = volume_average(flow, "enstrophy")
                re_local = max(flow, "Re_local")
                
                elapsed = time() - wall_time_start
                
                @info "3D Progress: iter=$(solver.iteration), " *
                      "t=$(round(solver.sim_time, digits=3)), " *
                      "dt=$(round(timestep, digits=5)), " *
                      "u_rms=$(round(velocity_rms, digits=3)), " *
                      "enstrophy=$(round(enstrophy, digits=3)), " *
                      "Re_max=$(round(re_local, digits=1))"
                      
                @info "  Wall time: $(round(elapsed, digits=1))s, " *
                      "rate: $(round(solver.sim_time / elapsed, digits=3)) sim/wall"
                      
                # Estimate completion time
                if solver.sim_time > 0
                    remaining_sim = stop_sim_time - solver.sim_time
                    rate = solver.sim_time / elapsed
                    remaining_wall = remaining_sim / rate
                    @info "  ETA: $(round(remaining_wall / 60, digits=1)) minutes"
                end
            end
        end
        
        @info "3D Taylor-Green vortex simulation completed successfully!"
        
    catch e
        @error "3D simulation error: $e"
        rethrow(e)
    finally
        # Final performance statistics
        wall_time_total = time() - wall_time_start
        
        @info "Final 3D performance statistics:"
        @info "  Total iterations: $(solver.iteration)"
        @info "  Final simulation time: $(solver.sim_time)"
        @info "  Total wall time: $(round(wall_time_total / 60, digits=2)) minutes"
        @info "  Iterations per second: $(round(solver.iteration / wall_time_total, digits=2))"
        @info "  Time units per wall hour: $(round(solver.sim_time / (wall_time_total / 3600), digits=2))"
        
        # 3D-specific performance metrics
        grid_points = N^3
        dof = 4 * grid_points  # 3 velocity + 1 pressure
        @info "  Grid points: $(N^3)"
        @info "  Degrees of freedom: $dof"
        @info "  Grid points per second: $(round(grid_points * solver.iteration / wall_time_total / 1e6, digits=2))M"
        
        # MPI performance analysis
        if MPI.Initialized() && size > 1
            # Gather statistics from all processes
            local_stats = [solver.iteration, wall_time_total, grid_points / size]
            all_stats = MPI.Gather(local_stats, 0, comm)
            
            if rank == 0
                total_iterations = sum(stat[1] for stat in all_stats)
                avg_wall_time = sum(stat[2] for stat in all_stats) / length(all_stats)
                total_grid_points_per_proc = sum(stat[3] for stat in all_stats) / length(all_stats)
                
                efficiency = wall_time_total / avg_wall_time
                
                @info "3D MPI performance:"
                @info "  Total iterations across processes: $(Int(total_iterations))"
                @info "  Average wall time per process: $(round(avg_wall_time / 60, digits=2)) min"
                @info "  Parallel efficiency: $(round(efficiency * 100, digits=1))%"
                @info "  Strong scaling efficiency: $(round(efficiency / size * 100, digits=1))%"
                @info "  Grid points per process: $(round(total_grid_points_per_proc / 1e6, digits=2))M"
            end
        end
        
        log_stats(solver)
    end
end

# Run the 3D simulation
if abspath(PROGRAM_FILE) == @__FILE__
    main()
    
    # Finalize MPI
    MPI.Finalize()
end