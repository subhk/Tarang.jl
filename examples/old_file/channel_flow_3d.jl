"""
3D Turbulent Channel Flow Simulation

Demonstrates 3D Tarang capabilities with a practical engineering application:
- 3D domain: periodic in x,y and Chebyshev in z (wall-normal)
- Mixed Fourier-Chebyshev spectral methods
- Wall boundary conditions (no-slip at top/bottom)
- Pressure-driven flow with body force
- 3D turbulence statistics and analysis

This is a canonical test case for wall-bounded turbulence.

Run with: mpiexec -n 8 julia channel_flow_3d.jl
"""

using Tarang
using MPI
using Random

MPI.Init()
setup_tarang_logging(level="INFO", mpi_aware=true)

function main()
    @info "Starting 3D turbulent channel flow simulation"
    
    # Parameters
    Re_tau = 180.0           # Friction Reynolds number
    Lx, Ly = 4π, 2π         # Domain size (streamwise, spanwise)
    Lz = 2.0                # Channel height
    Nx, Ny, Nz = 128, 128, 64  # Resolution
    nu = 1.0/2000           # Kinematic viscosity
    stop_sim_time = 50.0    # Simulation time
    
    @info "Channel flow parameters:"
    @info "  Friction Reynolds number: $Re_tau"
    @info "  Domain size: $Lx × $Ly × $Lz"
    @info "  Resolution: $Nx × $Ny × $Nz"
    @info "  Viscosity: $nu"
    
    # Create 3D channel domain
    coords = CartesianCoordinates("x", "y", "z")
    
    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    mesh = create_3d_process_mesh(size)
    
    dist = Distributor(coords, comm=comm, mesh=mesh)
    domain = channel_flow_3d_domain(dist, Lx, Ly, Lz, Nx, Ny, Nz)
    
    @info "3D channel domain created with PencilFFTs support"
    
    # Create fields
    fields = create_navier_stokes_3d_fields(domain)
    u = fields["velocity"]
    p = fields["pressure"]
    
    # Set up initial conditions (laminar Poiseuille + perturbations)
    x, y, z = local_grids(dist, domain.bases...)
    
    ensure_layout!(u.components[1], :g)  # streamwise (x)
    ensure_layout!(u.components[2], :g)  # spanwise (y)
    ensure_layout!(u.components[3], :g)  # wall-normal (z)
    
    ux_data = u.components[1]["g"]
    uy_data = u.components[2]["g"]
    uz_data = u.components[3]["g"]
    
    # Laminar profile: u(z) = U_max * (1 - (z/h)²)
    U_max = 1.5  # Maximum centerline velocity
    h = Lz / 2   # Half-channel height
    
    Random.seed!(42)
    noise_amplitude = 0.1
    
    for (k, z_val) in enumerate(z), (j, y_val) in enumerate(y), (i, x_val) in enumerate(x)
        # Mean flow (Poiseuille profile)
        z_norm = (z_val - h) / h  # Normalize to [-1, 1]
        u_mean = U_max * (1 - z_norm^2)
        
        # Add turbulent perturbations
        ux_data[i, j, k] = u_mean + noise_amplitude * randn() * (1 - abs(z_norm))
        uy_data[i, j, k] = noise_amplitude * randn() * (1 - abs(z_norm))
        uz_data[i, j, k] = noise_amplitude * randn() * (1 - abs(z_norm))
    end
    
    @info "Initial conditions set: laminar + random perturbations"
    
    # Create problem with channel flow equations
    variables = [u.components[1], u.components[2], u.components[3], p]
    problem = IVP(variables)
    
    # Add equations with body force to drive the flow
    body_force = 1.0  # Pressure gradient
    add_equation!(problem, "dt(u) - nu*lap(u) + grad(p) = -u·grad(u) + body_force*ex")
    add_equation!(problem, "div(u) = 0")
    
    # Boundary conditions: no-slip at walls (z = 0, Lz)
    add_bc!(problem, "u(z=0) = 0")
    add_bc!(problem, "u(z=$Lz) = 0")
    add_bc!(problem, "integ(p) = 0")
    
    add_substitution!(problem, "nu", nu)
    add_substitution!(problem, "body_force", body_force)
    add_substitution!(problem, "Lz", Lz)
    
    @info "3D channel flow problem configured"
    
    # Solver
    timestepper = RK222()
    solver = InitialValueSolver(problem, timestepper)
    solver.stop_sim_time = stop_sim_time
    
    # Analysis
    evaluator = create_evaluator(solver)
    
    # Statistics
    stats = add_file_handler(evaluator, "channel_stats", sim_dt=1.0)
    add_task!(stats, "mean_u", "integ(u)/volume")
    add_task!(stats, "wall_shear", "nu*grad(u)(z=0)")
    add_task!(stats, "cf", "wall_shear/(0.5*U_bulk²)")  # Friction coefficient
    
    @info "Starting 3D channel flow simulation..."
    
    try
        while proceed(solver) && solver.iteration < 1000  # Limit for demo
            step!(solver, 0.01)
            
            if solver.iteration % 100 == 0
                @info "3D channel: iter=$(solver.iteration), t=$(round(solver.sim_time, digits=2))"
                monitor_3d_memory_usage(MPI.Comm_rank(comm))
            end
        end
        
        @info "3D channel flow simulation completed"
        
    finally
        log_stats(solver)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
    MPI.Finalize()
end