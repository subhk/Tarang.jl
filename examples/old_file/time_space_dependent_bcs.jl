"""
Time and Space Dependent Boundary Conditions Example

This example demonstrates Tarang.jl's advanced boundary condition system
with time and spatially dependent boundary conditions, similar to Dedalus.

Examples include:
1. Time-dependent Dirichlet BCs: u(x,0,t) = sin(2πt)
2. Space-dependent Neumann BCs: ∂u/∂n = x² + y²  
3. Combined time+space BCs: u(0,y,t) = sin(ωt)*exp(-y²)
4. Moving boundary problems
5. Oscillating wall problems

Run with: julia time_space_dependent_bcs.jl
"""

using Tarang
using MPI
using Random

MPI.Init()

function example_time_dependent_dirichlet()
    """
    1D Heat equation with time-dependent Dirichlet boundary conditions
    
    Problem: ∂u/∂t = κ∇²u on [0,1] × [0,T]
    BCs: u(0,t) = sin(2πt)    (time-dependent)
         u(1,t) = cos(πt)     (time-dependent) 
    IC: u(x,0) = sin(πx)
    """
    
    @info "=== Time-Dependent Dirichlet BC Example ==="
    
    # Parameters
    L = 1.0
    N = 64
    κ = 0.1
    T = 2.0
    
    # Create coordinate system
    coords = CartesianCoordinates("x")
    dist = Distributor(coords)
    
    # Create basis
    x_basis = ChebyshevT(coords["x"], size=N, bounds=(0.0, L))
    
    # Create fields
    u = ScalarField(dist, "u", (x_basis,))
    
    # Create tau fields for time-dependent BCs
    tau_left = ScalarField(dist, "tau_left")
    tau_right = ScalarField(dist, "tau_right")
    
    # Create problem
    variables = [u, tau_left, tau_right]
    problem = IVP(variables)
    
    # Set up time variable
    set_time_variable!(problem.bc_manager, "t")
    
    # Add main equation
    add_equation!(problem, "dt(u) - κ*lap(u) + lift(tau_left, -1) + lift(tau_right, -2) = 0")
    
    # Add time-dependent boundary conditions
    add_dirichlet_bc!(problem, "u", "x", 0.0, "sin(2*pi*t)", tau_field="tau_left", time_dependent=true)
    add_dirichlet_bc!(problem, "u", "x", L, "cos(pi*t)", tau_field="tau_right", time_dependent=true)
    
    # Add substitutions
    add_substitution!(problem, "κ", κ)
    add_substitution!(problem, "L", L)
    add_substitution!(problem, "pi", π)
    
    @info "Time-dependent Dirichlet problem setup:"
    @info "  Time-dependent BCs: $(length(problem.bc_manager.time_dependent_bcs))"
    @info "  BC update required: $(requires_bc_update(problem.bc_manager))"
    @info "  Has time-dependent BCs: $(has_time_dependent_bcs(problem.bc_manager))"
    
    # Set initial conditions
    x = local_grids(dist, x_basis)[1]
    ensure_layout!(u, :g)
    u["g"] .= sin.(π * x)
    
    @info "✓ Time-dependent Dirichlet BC example set up successfully"
    return problem
end

function example_space_dependent_neumann()
    """
    2D Poisson equation with space-dependent Neumann boundary conditions
    
    Problem: ∇²u = f on [0,2π] × [0,1]
    BCs: u(x,0) = 0                    (Dirichlet bottom)
         u(x,1) = 0                    (Dirichlet top)
         ∂u/∂x(0,y) = y²               (space-dependent Neumann left)
         ∂u/∂x(2π,y) = sin(πy)         (space-dependent Neumann right)
    """
    
    @info "=== Space-Dependent Neumann BC Example ==="
    
    # Parameters
    Lx, Ly = 2π, 1.0
    Nx, Ny = 64, 32
    
    # Create coordinates
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    
    # Create bases
    x_basis = RealFourier(coords["x"], size=Nx, bounds=(0.0, Lx))
    y_basis = ChebyshevT(coords["y"], size=Ny, bounds=(0.0, Ly))
    
    # Create fields
    u = ScalarField(dist, "u", (x_basis, y_basis))
    f = ScalarField(dist, "f", (x_basis, y_basis))
    
    # Tau fields for BCs
    tau_bottom = ScalarField(dist, "tau_bottom", (x_basis,))
    tau_top = ScalarField(dist, "tau_top", (x_basis,))
    tau_left = ScalarField(dist, "tau_left", (y_basis,))
    tau_right = ScalarField(dist, "tau_right", (y_basis,))
    
    # Set up coordinate fields for space-dependent BCs
    x, y = local_grids(dist, x_basis, y_basis)
    add_coordinate_field!(problem.bc_manager, "x", x)
    add_coordinate_field!(problem.bc_manager, "y", y)
    
    # Create problem
    variables = [u, tau_bottom, tau_top, tau_left, tau_right]
    problem = LBVP(variables)
    
    # Main equation
    add_equation!(problem, "lap(u) + lift(tau_bottom, -1) + lift(tau_top, -2) + " *
                          "lift(tau_left, -1) + lift(tau_right, -2) = f")
    
    # Add space-dependent boundary conditions
    add_dirichlet_bc!(problem, "u", "y", 0.0, 0.0, tau_field="tau_bottom")
    add_dirichlet_bc!(problem, "u", "y", Ly, 0.0, tau_field="tau_top")
    add_neumann_bc!(problem, "u", "x", 0.0, "y^2", tau_field="tau_left", space_dependent=true)
    add_neumann_bc!(problem, "u", "x", Lx, "sin(pi*y)", tau_field="tau_right", space_dependent=true)
    
    # Set forcing
    ensure_layout!(f, :g)
    f["g"] .= sin.(x) .* cos.(y)
    
    add_substitution!(problem, "Lx", Lx)
    add_substitution!(problem, "Ly", Ly)
    add_substitution!(problem, "pi", π)
    
    @info "Space-dependent Neumann problem setup:"
    @info "  Space-dependent BCs: $(length(problem.bc_manager.space_dependent_bcs))"
    @info "  Has space-dependent BCs: $(has_space_dependent_bcs(problem.bc_manager))"
    
    @info "✓ Space-dependent Neumann BC example set up successfully"
    return problem
end

function example_time_space_dependent_mixed()
    """
    2D time-dependent problem with mixed time/space dependent BCs
    
    Problem: ∂u/∂t = κ∇²u + f on [0,1]² × [0,T]
    BCs: u(0,y,t) = sin(ωt)*exp(-y²)     (time+space dependent)
         u(1,y,t) = cos(ωt)*y             (time+space dependent)
         u(x,0,t) = sin(2πt)*x            (time+space dependent)
         u(x,1,t) = exp(-t)*cos(πx)       (time+space dependent)
    """
    
    @info "=== Combined Time+Space Dependent BC Example ==="
    
    # Parameters
    L = 1.0
    Nx, Ny = 32, 32
    κ = 0.05
    ω = 2π
    
    # Create coordinates
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    
    # Create bases
    x_basis = ChebyshevT(coords["x"], size=Nx, bounds=(0.0, L))
    y_basis = ChebyshevT(coords["y"], size=Ny, bounds=(0.0, L))
    
    # Create fields
    u = ScalarField(dist, "u", (x_basis, y_basis))
    f = ScalarField(dist, "f", (x_basis, y_basis))
    
    # Create tau fields
    tau_left = ScalarField(dist, "tau_left", (y_basis,))
    tau_right = ScalarField(dist, "tau_right", (y_basis,))
    tau_bottom = ScalarField(dist, "tau_bottom", (x_basis,))
    tau_top = ScalarField(dist, "tau_top", (x_basis,))
    
    # Create problem
    variables = [u, tau_left, tau_right, tau_bottom, tau_top]
    problem = IVP(variables)
    
    # Set up time and coordinate fields
    set_time_variable!(problem.bc_manager, "t")
    x, y = local_grids(dist, x_basis, y_basis)
    add_coordinate_field!(problem.bc_manager, "x", x)
    add_coordinate_field!(problem.bc_manager, "y", y)
    
    # Main equation
    add_equation!(problem, "dt(u) - κ*lap(u) + lift(tau_left, -1) + lift(tau_right, -2) + " *
                          "lift(tau_bottom, -1) + lift(tau_top, -2) = f")
    
    # Add combined time+space dependent boundary conditions
    add_dirichlet_bc!(problem, "u", "x", 0.0, "sin(ω*t)*exp(-y^2)", 
                     tau_field="tau_left", time_dependent=true, space_dependent=true)
    add_dirichlet_bc!(problem, "u", "x", L, "cos(ω*t)*y", 
                     tau_field="tau_right", time_dependent=true, space_dependent=true)
    add_dirichlet_bc!(problem, "u", "y", 0.0, "sin(2*pi*t)*x", 
                     tau_field="tau_bottom", time_dependent=true, space_dependent=true)
    add_dirichlet_bc!(problem, "u", "y", L, "exp(-t)*cos(pi*x)", 
                     tau_field="tau_top", time_dependent=true, space_dependent=true)
    
    # Set forcing to zero
    ensure_layout!(f, :g)
    f["g"] .= 0.0
    
    # Add substitutions
    add_substitution!(problem, "κ", κ)
    add_substitution!(problem, "ω", ω)
    add_substitution!(problem, "L", L)
    add_substitution!(problem, "pi", π)
    
    @info "Combined time+space dependent problem setup:"
    @info "  Time-dependent BCs: $(length(problem.bc_manager.time_dependent_bcs))"
    @info "  Space-dependent BCs: $(length(problem.bc_manager.space_dependent_bcs))"
    @info "  Total BCs: $(length(problem.bc_manager.conditions))"
    
    # Set initial conditions
    ensure_layout!(u, :g)
    u["g"] .= exp.(-((x .- 0.5).^2 .+ (y .- 0.5).^2))  # Gaussian initial condition
    
    @info "✓ Combined time+space dependent BC example set up successfully"
    return problem
end

function example_oscillating_boundary()
    """
    Example inspired by oscillating wall problems (like Stokes' second problem)
    
    Problem: ∂u/∂t = ν∂²u/∂y² on [0,∞) × [0,T]
    BC: u(0,t) = U₀*cos(ωt)    (oscillating wall)
        u(∞,t) = 0             (far field)
    IC: u(y,0) = 0
    """
    
    @info "=== Oscillating Boundary Example (Stokes' Second Problem) ==="
    
    # Parameters
    L = 10.0  # Truncated "infinite" domain
    N = 64
    ν = 1.0   # Kinematic viscosity
    U₀ = 1.0  # Oscillation amplitude
    ω = 2π    # Oscillation frequency
    
    # Create coordinate system
    coords = CartesianCoordinates("y")
    dist = Distributor(coords)
    
    # Create basis
    y_basis = ChebyshevT(coords["y"], size=N, bounds=(0.0, L))
    
    # Create fields
    u = ScalarField(dist, "u", (y_basis,))
    
    # Create tau fields
    tau_wall = ScalarField(dist, "tau_wall")      # Oscillating wall
    tau_far = ScalarField(dist, "tau_far")        # Far field
    
    # Create problem
    variables = [u, tau_wall, tau_far]
    problem = IVP(variables)
    
    # Set up time variable
    set_time_variable!(problem.bc_manager, "t")
    
    # Main equation (1D diffusion)
    add_equation!(problem, "dt(u) - ν*d2(u) + lift(tau_wall, -1) + lift(tau_far, -2) = 0")
    
    # Add time-dependent oscillating wall BC
    add_dirichlet_bc!(problem, "u", "y", 0.0, "U₀*cos(ω*t)", 
                     tau_field="tau_wall", time_dependent=true)
    add_dirichlet_bc!(problem, "u", "y", L, 0.0, tau_field="tau_far")
    
    # Add substitutions
    add_substitution!(problem, "ν", ν)
    add_substitution!(problem, "U₀", U₀)
    add_substitution!(problem, "ω", ω)
    add_substitution!(problem, "L", L)
    
    # Initial condition (zero everywhere)
    ensure_layout!(u, :g)
    u["g"] .= 0.0
    
    @info "Oscillating boundary problem setup:"
    @info "  Wall velocity: U₀=$(U₀) * cos(ω*t), ω=$(ω)"
    @info "  Reynolds number: Re = U₀*L/ν = $(U₀*L/ν)"
    @info "  Time-dependent BCs: $(length(problem.bc_manager.time_dependent_bcs))"
    
    @info "✓ Oscillating boundary example set up successfully"
    return problem
end

function example_bc_evaluation_demo()
    """
    Demonstrate boundary condition evaluation at different times
    """
    
    @info "=== BC Evaluation Demonstration ==="
    
    # Create a simple manager for testing
    manager = BoundaryConditionManager()
    
    # Set up time variable
    set_time_variable!(manager, "t")
    
    # Create some test BCs
    bc1 = dirichlet_bc("u", "x", 0.0, "sin(2*pi*t)", time_dependent=true)
    bc2 = dirichlet_bc("v", "x", 1.0, "cos(t)*x", time_dependent=true, space_dependent=true)
    bc3 = neumann_bc("w", "y", 0.0, "exp(-t)", time_dependent=true)
    
    # Add to manager
    add_bc!(manager, bc1)
    add_bc!(manager, bc2)  
    add_bc!(manager, bc3)
    
    @info "Evaluating BCs at different times:"
    
    # Test evaluation at different times
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    coords = Dict("x" => 0.5, "y" => 0.3)
    
    for t in times
        val1 = evaluate_bc_value(manager, bc1, t)
        val2 = evaluate_bc_value(manager, bc2, t, coords)
        val3 = evaluate_bc_value(manager, bc3, t)
        
        @info "  t=$t: BC1=$(round(val1, digits=4)), BC2=$(val2), BC3=$(round(val3, digits=4))"
        
        # Update time-dependent BCs
        update_time_dependent_bcs!(manager, t)
    end
    
    @info "✓ BC evaluation demonstration completed"
    return manager
end

function main()
    """Run all time/space dependent boundary condition examples"""
    
    @info "Starting Time and Space Dependent Boundary Condition Examples"
    
    try
        # Example 1: Time-dependent Dirichlet
        prob1 = example_time_dependent_dirichlet()
        @info "✓ Time-dependent Dirichlet example completed"
        println()
        
        # Example 2: Space-dependent Neumann
        prob2 = example_space_dependent_neumann()
        @info "✓ Space-dependent Neumann example completed"
        println()
        
        # Example 3: Combined time+space dependent
        prob3 = example_time_space_dependent_mixed()
        @info "✓ Combined time+space dependent example completed"
        println()
        
        # Example 4: Oscillating boundary
        prob4 = example_oscillating_boundary()
        @info "✓ Oscillating boundary example completed"
        println()
        
        # Example 5: BC evaluation demonstration
        mgr = example_bc_evaluation_demo()
        @info "✓ BC evaluation demonstration completed"
        println()
        
        @info "All time/space dependent BC examples completed successfully!"
        
        # Summary
        @info "Summary of capabilities demonstrated:"
        @info "  ✓ Time-dependent Dirichlet BCs: u(boundary,t) = f(t)"
        @info "  ✓ Space-dependent Neumann BCs: ∂u/∂n = g(x,y,z)"
        @info "  ✓ Combined time+space BCs: u = h(t,x,y,z)"
        @info "  ✓ Oscillating wall problems"
        @info "  ✓ Real-time BC evaluation and updating"
        @info "  ✓ Automatic dependency detection"
        @info "  ✓ Multiple coordinate systems support"
        
    catch e
        @error "Example failed with error: $e"
        rethrow(e)
    end
end

# Run examples if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
    MPI.Finalize()
end