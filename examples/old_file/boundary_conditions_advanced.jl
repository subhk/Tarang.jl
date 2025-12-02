"""
Advanced Boundary Condition Examples for Tarang.jl

This example demonstrates the comprehensive boundary condition system
with various types of boundary conditions:
- Dirichlet boundary conditions
- Neumann boundary conditions  
- Robin (mixed) boundary conditions
- Stress-free boundary conditions
- Custom boundary conditions

Run with: julia boundary_conditions_advanced.jl
"""

using Tarang
using MPI

# Initialize MPI
MPI.Init()

function example_dirichlet_neumann_poisson()
    """
    2D Poisson equation with mixed Dirichlet/Neumann boundary conditions
    
    Problem: ∇²u = f on [0,2π] × [0,1]
    BCs: u(x,0) = g₁(x)        (Dirichlet bottom)
         u(x,1) = g₂(x)        (Dirichlet top)  
         ∂u/∂x(0,y) = h₁(y)    (Neumann left)
         ∂u/∂x(2π,y) = h₂(y)   (Neumann right)
    """
    
    @info "=== Dirichlet-Neumann Poisson Example ==="
    
    # Parameters
    Lx, Ly = 2π, 1.0
    Nx, Ny = 64, 32
    
    # Create coordinate system and distributor
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    
    # Create bases
    x_basis = RealFourier(coords["x"], size=Nx, bounds=(0.0, Lx))
    y_basis = ChebyshevT(coords["y"], size=Ny, bounds=(0.0, Ly))
    
    # Create domain
    domain = Domain(dist, (x_basis, y_basis))
    
    # Create fields
    u = ScalarField(dist, "u", (x_basis, y_basis))
    f = ScalarField(dist, "f", (x_basis, y_basis))  # Forcing
    
    # Create tau fields for boundary conditions
    tau_b1 = ScalarField(dist, "tau_b1", (x_basis,))  # Bottom Dirichlet
    tau_b2 = ScalarField(dist, "tau_b2", (x_basis,))  # Top Dirichlet
    tau_l = ScalarField(dist, "tau_l", (y_basis,))    # Left Neumann
    tau_r = ScalarField(dist, "tau_r", (y_basis,))    # Right Neumann
    
    # Set up forcing and boundary conditions
    x, y = local_grids(dist, x_basis, y_basis)
    ensure_layout!(f, :g)
    f["g"] .= sin.(2*x) .* cos.(3*y)  # Random forcing
    
    # Boundary condition values
    g1 = sin.(x)          # Bottom boundary
    g2 = -sin.(x)         # Top boundary  
    h1 = cos.(y)          # Left boundary (Neumann)
    h2 = -cos.(y)         # Right boundary (Neumann)
    
    # Create problem
    variables = [u, tau_b1, tau_b2, tau_l, tau_r]
    problem = LBVP(variables)
    
    # Add main equation with lift terms for boundary conditions
    add_equation!(problem, "lap(u) + lift(tau_b1, -1) + lift(tau_b2, -2) + " *
                          "lift(tau_l, -1) + lift(tau_r, -2) = f")
    
    # Add boundary conditions using advanced BC system
    add_dirichlet_bc!(problem, "u", "y", 0.0, "g1", tau_field="tau_b1")
    add_dirichlet_bc!(problem, "u", "y", Ly, "g2", tau_field="tau_b2") 
    add_neumann_bc!(problem, "u", "x", 0.0, "h1", tau_field="tau_l")
    add_neumann_bc!(problem, "u", "x", Lx, "h2", tau_field="tau_r")
    
    # Add substitutions
    add_substitution!(problem, "g1", g1)
    add_substitution!(problem, "g2", g2)
    add_substitution!(problem, "h1", h1)
    add_substitution!(problem, "h2", h2)
    add_substitution!(problem, "Ly", Ly)
    add_substitution!(problem, "Lx", Lx)
    
    @info "Problem setup complete:"
    @info "  Variables: $(length(problem.variables))"
    @info "  Equations: $(length(problem.equations))"
    @info "  Boundary conditions: $(length(problem.boundary_conditions))"
    @info "  Advanced BCs: $(length(problem.bc_manager.conditions))"
    
    # Validate problem
    validate_problem(problem)
    @info "Problem validation: PASSED"
    
    return problem
end

function example_robin_boundary_conditions()
    """
    1D diffusion-reaction equation with Robin boundary conditions
    
    Problem: -d²u/dx² + u = f on [0,1]
    BCs: -k₁ du/dx + α₁ u = β₁ at x=0  (Robin left)
         k₂ du/dx + α₂ u = β₂  at x=1  (Robin right)
    """
    
    @info "=== Robin Boundary Conditions Example ==="
    
    # Parameters
    L = 1.0
    N = 64
    k1, k2 = 0.5, 1.0      # Diffusion coefficients
    α1, α2 = 1.0, 2.0      # Robin coefficients
    β1, β2 = 0.0, 1.0      # Robin values
    
    # Create coordinate system
    coords = CartesianCoordinates("x")
    dist = Distributor(coords)
    
    # Create basis
    x_basis = ChebyshevT(coords["x"], size=N, bounds=(0.0, L))
    
    # Create fields
    u = ScalarField(dist, "u", (x_basis,))
    f = ScalarField(dist, "f", (x_basis,))
    
    # Tau fields for Robin BCs
    tau_l = ScalarField(dist, "tau_left")   # Constant tau field
    tau_r = ScalarField(dist, "tau_right")  # Constant tau field
    
    # Set forcing
    x = local_grids(dist, x_basis)[1]
    ensure_layout!(f, :g)
    f["g"] .= exp.(-5*(x .- 0.5).^2)  # Gaussian forcing
    
    # Create problem
    variables = [u, tau_l, tau_r]
    problem = LBVP(variables)
    
    # Main equation
    add_equation!(problem, "-d2(u) + u + lift(tau_l, -1) + lift(tau_r, -2) = f")
    
    # Robin boundary conditions
    add_robin_bc!(problem, "u", "x", 0.0, α1, -k1, β1, tau_field="tau_left")
    add_robin_bc!(problem, "u", "x", L, α2, k2, β2, tau_field="tau_right")
    
    # Add substitutions
    add_substitution!(problem, "k1", k1)
    add_substitution!(problem, "k2", k2)
    add_substitution!(problem, "α1", α1)
    add_substitution!(problem, "α2", α2)
    add_substitution!(problem, "β1", β1)
    add_substitution!(problem, "β2", β2)
    add_substitution!(problem, "L", L)
    
    @info "Robin BC problem setup:"
    @info "  Left BC: $(α1)*u - $(k1)*du/dx = $(β1) at x=0"
    @info "  Right BC: $(α2)*u + $(k2)*du/dx = $(β2) at x=1"
    
    validate_problem(problem)
    @info "Robin BC validation: PASSED"
    
    return problem
end

function example_stress_free_navier_stokes()
    """
    2D incompressible Navier-Stokes with stress-free boundary conditions
    
    Problem: ∂u/∂t - ν∇²u + ∇p = -u·∇u
             ∇·u = 0
    BCs: Stress-free at top and bottom (z = 0, Lz)
         Periodic in x
    """
    
    @info "=== Stress-Free Navier-Stokes Example ==="
    
    # Parameters
    Lx, Lz = 4.0, 1.0
    Nx, Nz = 128, 32
    Re = 1000.0
    ν = 1.0 / Re
    
    # Create coordinates and distributor
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords)
    
    # Create bases
    x_basis = RealFourier(coords["x"], size=Nx, bounds=(0.0, Lx))
    z_basis = ChebyshevT(coords["z"], size=Nz, bounds=(0.0, Lz))
    
    # Create domain
    domain = Domain(dist, (x_basis, z_basis))
    
    # Create fields
    u = VectorField(dist, coords, "velocity", (x_basis, z_basis))
    p = ScalarField(dist, "pressure", (x_basis, z_basis))
    
    # Tau fields for stress-free BCs
    tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,))  # Bottom stress-free
    tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,))  # Top stress-free
    tau_p = ScalarField(dist, "tau_p")  # Pressure gauge
    
    # Create IVP problem (time-dependent)
    variables = [u.components[1], u.components[2], p, 
                tau_u1.components[1], tau_u1.components[2],
                tau_u2.components[1], tau_u2.components[2], tau_p]
    problem = IVP(variables)
    
    # Navier-Stokes equations with tau terms
    add_equation!(problem, "dt(u) - ν*lap(u) + grad(p) + lift(tau_u1) + lift(tau_u2) = -u·grad(u)")
    add_equation!(problem, "div(u) + tau_p = 0")
    add_equation!(problem, "integ(p) = 0")  # Pressure gauge
    
    # Stress-free boundary conditions using advanced BC system
    add_stress_free_bc!(problem, "velocity", "z", 0.0, 
                       tau_fields=["tau_u1_x", "tau_u1_z"])
    add_stress_free_bc!(problem, "velocity", "z", Lz,
                       tau_fields=["tau_u2_x", "tau_u2_z"])
    
    # Add substitutions
    add_substitution!(problem, "ν", ν)
    add_substitution!(problem, "Lz", Lz)
    add_substitution!(problem, "Re", Re)
    
    @info "Stress-free Navier-Stokes setup:"
    @info "  Reynolds number: $Re"
    @info "  Domain: [$Lx × $Lz] with stress-free top/bottom"
    @info "  Advanced BC count: $(length(problem.bc_manager.conditions))"
    
    # Note: This example shows the setup - actual solving would require
    # proper solver implementation
    
    return problem
end

function example_custom_boundary_conditions()
    """
    Example with custom boundary condition expressions
    """
    
    @info "=== Custom Boundary Conditions Example ==="
    
    # Simple 1D problem with custom BC
    coords = CartesianCoordinates("x")
    dist = Distributor(coords)
    x_basis = ChebyshevT(coords["x"], size=32, bounds=(0.0, 1.0))
    
    u = ScalarField(dist, "u", (x_basis,))
    tau_custom = ScalarField(dist, "tau_custom")
    
    variables = [u, tau_custom]
    problem = LBVP(variables)
    
    # Main equation
    add_equation!(problem, "-d2(u) + lift(tau_custom, -1) = sin(π*x)")
    
    # Custom boundary condition: u(0) + 2*du/dx(0) = 1
    custom_bc = custom_bc("u(x=0) + 2*dx(u)(x=0) = 1", tau_fields=["tau_custom"])
    add_bc!(problem, custom_bc)
    
    # Standard Dirichlet at other end
    add_dirichlet_bc!(problem, "u", "x", 1.0, 0.0)
    
    @info "Custom BC problem setup with mixed custom/standard BCs"
    
    return problem
end

function demonstrate_bc_validation()
    """
    Demonstrate boundary condition validation and error checking
    """
    
    @info "=== Boundary Condition Validation Demo ==="
    
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    
    x_basis = RealFourier(coords["x"], size=32, bounds=(0.0, 2π))
    y_basis = ChebyshevT(coords["y"], size=16, bounds=(0.0, 1.0))
    
    u = ScalarField(dist, "u", (x_basis, y_basis))
    problem = LBVP([u])
    
    # Add equation
    add_equation!(problem, "lap(u) = 0")
    
    # Try adding BCs and show validation
    @info "Adding Dirichlet BCs..."
    add_dirichlet_bc!(problem, "u", "y", 0.0, 1.0)
    add_dirichlet_bc!(problem, "u", "y", 1.0, 0.0)
    
    @info "Boundary conditions summary:"
    bc_counts = get_bc_count_by_type(problem.bc_manager)
    for (bc_type, count) in bc_counts
        @info "  $bc_type: $count"
    end
    
    @info "Required tau fields: $(get_required_tau_fields(problem.bc_manager))"
    
    # Test validation
    try
        validate_problem(problem)
        @info "Problem validation: PASSED"
    catch e
        @error "Problem validation failed: $e"
    end
    
    return problem
end

function main()
    """Run all boundary condition examples"""
    
    @info "Starting Advanced Boundary Condition Examples"
    
    try
        # Example 1: Mixed Dirichlet/Neumann Poisson
        prob1 = example_dirichlet_neumann_poisson()
        @info "✓ Dirichlet-Neumann Poisson example completed"
        
        # Example 2: Robin boundary conditions
        prob2 = example_robin_boundary_conditions()
        @info "✓ Robin boundary conditions example completed"
        
        # Example 3: Stress-free Navier-Stokes
        prob3 = example_stress_free_navier_stokes()
        @info "✓ Stress-free Navier-Stokes example completed"
        
        # Example 4: Custom boundary conditions
        prob4 = example_custom_boundary_conditions()
        @info "✓ Custom boundary conditions example completed"
        
        # Example 5: Validation demonstration
        prob5 = demonstrate_bc_validation()
        @info "✓ Boundary condition validation demo completed"
        
        @info "All advanced boundary condition examples completed successfully!"
        
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