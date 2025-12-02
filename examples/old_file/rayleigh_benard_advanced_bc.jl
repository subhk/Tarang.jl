"""
2D Rayleigh-Bénard Convection with Advanced Boundary Conditions

This example demonstrates the enhanced boundary condition system
by solving the classic Rayleigh-Bénard convection problem with
structured boundary condition specification.

Problem:
- 2D incompressible Boussinesq convection
- Hot bottom, cold top (Dirichlet temperature BCs)
- No-slip velocity boundary conditions
- Periodic in horizontal direction

Enhanced features:
- Structured boundary condition specification
- Automatic tau field generation
- Advanced BC validation and error checking

Run with: mpiexec -n 4 julia rayleigh_benard_advanced_bc.jl
"""

using Tarang
using MPI
using Random

MPI.Init()
setup_tarang_logging(level="INFO", mpi_aware=true)

function main()
    @info "Starting 2D Rayleigh-Bénard with Advanced Boundary Conditions"
    
    # Parameters
    Lx, Lz = 4.0, 1.0
    Nx, Nz = 256, 64
    Rayleigh = 2e6
    Prandtl = 1.0
    dealias = 3/2
    stop_sim_time = 10.0  # Shorter for demo
    max_timestep = 0.125
    
    # Derived parameters
    κ = (Rayleigh * Prandtl)^(-1/2)
    ν = (Rayleigh / Prandtl)^(-1/2)
    
    @info "Problem parameters:"
    @info "  Domain: $Lx × $Lz"
    @info "  Resolution: $Nx × $Nz"
    @info "  Rayleigh: $Rayleigh"
    @info "  Prandtl: $Prandtl"
    @info "  Thermal diffusivity κ: $κ"
    @info "  Kinematic viscosity ν: $ν"
    
    # Coordinate system and distributor
    coords = CartesianCoordinates("x", "z")
    comm = MPI.COMM_WORLD
    dist = Distributor(coords, comm=comm)
    
    # Create bases
    x_basis = RealFourier(coords["x"], size=Nx, bounds=(0.0, Lx), dealias=dealias)
    z_basis = ChebyshevT(coords["z"], size=Nz, bounds=(0.0, Lz), dealias=dealias)
    
    # Create domain
    domain = Domain(dist, (x_basis, z_basis))
    
    @info "Domain created:"
    @info "  Global shape: $(global_shape(domain))"
    @info "  Local shape: $(local_shape(domain))"
    
    # Create fields
    u = VectorField(dist, coords, "velocity", (x_basis, z_basis))
    p = ScalarField(dist, "pressure", (x_basis, z_basis))
    b = ScalarField(dist, "buoyancy", (x_basis, z_basis))
    
    # Create tau fields for boundary condition enforcement
    # Using the advanced BC system, we'll set these up systematically
    
    # Velocity boundary conditions (no-slip at top and bottom)
    tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,))  # Bottom no-slip
    tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,))  # Top no-slip
    
    # Buoyancy boundary conditions (Dirichlet at top and bottom)
    tau_b1 = ScalarField(dist, "tau_b1", (x_basis,))  # Bottom temperature
    tau_b2 = ScalarField(dist, "tau_b2", (x_basis,))  # Top temperature
    
    # Pressure gauge
    tau_p = ScalarField(dist, "tau_p")
    
    @info "Fields created:"
    @info "  Primary fields: velocity (vector), pressure (scalar), buoyancy (scalar)"
    @info "  Tau fields: 4 boundary + 1 gauge = 5 total"
    
    # Set up initial conditions
    x, z = local_grids(dist, x_basis, z_basis)
    
    # Initialize buoyancy with small random perturbations
    ensure_layout!(b, :g)
    Random.seed!(42)
    b_data = b["g"]
    
    for (j, z_val) in enumerate(z), (i, x_val) in enumerate(x)
        # Linear background + small perturbations
        b_background = Lz - z_val  # Linear profile from 1 at bottom to 0 at top
        perturbation = 0.01 * randn() * sin(π * z_val / Lz)
        b_data[i, j] = b_background + perturbation
    end
    
    # Initialize velocity to zero
    ensure_layout!(u.components[1], :g)
    ensure_layout!(u.components[2], :g)
    u.components[1]["g"] .= 0.0
    u.components[2]["g"] .= 0.0
    
    # Initialize pressure to zero
    ensure_layout!(p, :g)
    p["g"] .= 0.0
    
    @info "Initial conditions set with random perturbations"
    
    # Create problem with advanced boundary condition system
    variables = [
        u.components[1], u.components[2], p, b,  # Primary variables
        tau_p,                                    # Pressure gauge
        tau_u1.components[1], tau_u1.components[2],  # Bottom velocity tau
        tau_u2.components[1], tau_u2.components[2],  # Top velocity tau
        tau_b1, tau_b2                            # Buoyancy tau fields
    ]
    
    problem = IVP(variables)
    
    @info "Setting up equations with advanced boundary conditions..."
    
    # Register tau fields with the BC manager
    register_tau_field!(problem, "tau_u1_x", tau_u1.components[1])
    register_tau_field!(problem, "tau_u1_z", tau_u1.components[2])
    register_tau_field!(problem, "tau_u2_x", tau_u2.components[1])
    register_tau_field!(problem, "tau_u2_z", tau_u2.components[2])
    register_tau_field!(problem, "tau_b1", tau_b1)
    register_tau_field!(problem, "tau_b2", tau_b2)
    register_tau_field!(problem, "tau_p", tau_p)
    
    # Main equations with lift terms for boundary condition enforcement
    # First-order form equations (following Dedalus pattern)
    
    # Momentum equation with lift terms for no-slip BCs
    add_equation!(problem, "dt(u) - ν*lap(u) + grad(p) - b*ez + " *
                          "lift(tau_u1) + lift(tau_u2) = -u·grad(u)")
    
    # Continuity equation
    add_equation!(problem, "div(u) + tau_p = 0")
    
    # Buoyancy equation with lift terms for temperature BCs
    add_equation!(problem, "dt(b) - κ*lap(b) + lift(tau_b1) + lift(tau_b2) = -u·grad(b)")
    
    # Add boundary conditions using the advanced system
    @info "Adding structured boundary conditions..."
    
    # No-slip velocity boundary conditions
    add_dirichlet_bc!(problem, "velocity", "z", 0.0, 0.0, tau_field="tau_u1")
    add_dirichlet_bc!(problem, "velocity", "z", Lz, 0.0, tau_field="tau_u2")
    
    # Temperature boundary conditions (hot bottom, cold top)
    add_dirichlet_bc!(problem, "buoyancy", "z", 0.0, Lz, tau_field="tau_b1")  # Hot bottom
    add_dirichlet_bc!(problem, "buoyancy", "z", Lz, 0.0, tau_field="tau_b2")  # Cold top
    
    # Pressure gauge condition
    add_equation!(problem, "integ(p) = 0")
    
    # Add parameter substitutions
    add_substitution!(problem, "ν", ν)
    add_substitution!(problem, "κ", κ)
    add_substitution!(problem, "Lz", Lz)
    add_substitution!(problem, "ez", [0.0, 1.0])  # Unit vector in z direction
    
    @info "Problem setup complete:"
    @info "  Total variables: $(length(problem.variables))"
    @info "  Equations: $(length(problem.equations))"
    @info "  Legacy BC strings: $(length(problem.boundary_conditions))" 
    @info "  Advanced BCs: $(length(problem.bc_manager.conditions))"
    
    # Show boundary condition summary
    bc_counts = get_bc_count_by_type(problem.bc_manager)
    @info "Boundary condition summary:"
    for (bc_type, count) in bc_counts
        @info "  $bc_type: $count"
    end
    
    @info "Required tau fields: $(get_required_tau_fields(problem.bc_manager))"
    
    # Validate problem with enhanced BC validation
    @info "Validating problem formulation..."
    try
        validate_problem(problem)
        @info "✓ Problem validation PASSED (including advanced BCs)"
    catch e
        @error "Problem validation FAILED: $e"
        return
    end
    
    # Create solver
    timestepper = RK222()
    solver = InitialValueSolver(problem, timestepper)
    solver.stop_sim_time = stop_sim_time
    
    @info "Solver created with $(typeof(timestepper)) timestepper"
    
    # Create CFL condition
    cfl = CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.4,
             threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
    add_velocity!(cfl, u)
    
    # Flow properties for monitoring
    flow = GlobalFlowProperty(solver, cadence=10)
    add_property!(flow, "sqrt(u·u)", "velocity_rms")
    add_property!(flow, "sqrt(grad(b)·grad(b))", "thermal_gradient")
    add_property!(flow, "integ(b)", "total_buoyancy")
    
    @info "Starting Rayleigh-Bénard simulation with advanced BCs..."
    
    try
        iteration_count = 0
        while proceed(solver) && iteration_count < 100  # Limited for demo
            dt = compute_timestep(cfl)
            step!(solver, dt)
            iteration_count += 1
            
            if iteration_count % 20 == 0
                velocity_rms = volume_average(flow, "velocity_rms")
                thermal_grad = volume_average(flow, "thermal_gradient")
                total_buoyancy = volume_average(flow, "total_buoyancy")
                
                @info "Progress: iter=$iteration_count, t=$(round(solver.sim_time, digits=3)), " *
                      "dt=$(round(dt, digits=5))"
                @info "  |u|_rms=$(round(velocity_rms, digits=4)), " *
                      "|∇b|=$(round(thermal_grad, digits=4)), " *
                      "∫b=$(round(total_buoyancy, digits=4))"
            end
        end
        
        @info "✓ Rayleigh-Bénard simulation with advanced BCs completed successfully!"
        @info "Final time: $(solver.sim_time), iterations: $iteration_count"
        
    catch e
        @error "Simulation failed: $e"
        rethrow(e)
    finally
        log_stats(solver)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
    MPI.Finalize()
end