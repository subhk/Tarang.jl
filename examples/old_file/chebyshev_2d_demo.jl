"""
2D Chebyshev Spectral Method Demonstration

This example demonstrates using Chebyshev polynomials in both spatial directions,
following the Dedalus pattern for multi-dimensional spectral methods.

Shows:
- ChebyshevT basis construction in x and y directions
- Multi-dimensional domain creation
- Boundary condition application on Chebyshev domains
- Sample 2D problem setup (Poisson equation)

Run with:
    julia --project=.. chebyshev_2d_demo.jl
"""

include("../src/core/coordinates.jl")
include("../src/core/basis.jl")
include("../src/core/domain.jl")
include("../src/core/distributor.jl")
include("../src/core/boundary_conditions.jl")

function demonstrate_2d_chebyshev()
    """Demonstrate 2D Chebyshev spectral method setup"""
    
    println("2D Chebyshev Spectral Method Demonstration")
    println("=" ^ 50)
    
    # Create coordinate system
    println("\n1. Setting up coordinate system...")
    coords = CoordinateSystem("cartesian")
    add_coordinate!(coords, "x", -1.0, 1.0)
    add_coordinate!(coords, "y", -1.0, 1.0)
    
    println("   Created Cartesian coordinates:")
    println("   - x: [-1, 1]")
    println("   - y: [-1, 1]")
    
    # Create Chebyshev bases for both directions
    println("\n2. Creating Chebyshev bases...")
    
    # Resolution parameters
    Nx, Ny = 32, 32
    
    # Create bases
    x_basis = ChebyshevT(coords["x"], size=Nx, bounds=(-1.0, 1.0))
    y_basis = ChebyshevT(coords["y"], size=Ny, bounds=(-1.0, 1.0))
    
    println("   Created ChebyshevT bases:")
    println("   - x direction: $(x_basis.meta.size) points, bounds $(x_basis.meta.bounds)")
    println("   - y direction: $(y_basis.meta.size) points, bounds $(y_basis.meta.bounds)")
    
    # Create distributor (for parallel support)
    println("\n3. Setting up distributor...")
    dist = Distributor(coords, comm_size=1)  # Single processor for demo
    
    # Create 2D domain with both Chebyshev bases
    println("\n4. Creating 2D Chebyshev domain...")
    domain = Domain(dist, (x_basis, y_basis))
    
    println("   Domain properties:")
    println("   - Dimensions: $(domain.dim)")
    println("   - Bases: $(length(domain.bases))")
    println("   - Global shape: $(global_shape(domain))")
    println("   - Is compound domain: $(is_compound(domain))")
    
    # Show basis information
    println("\n   Basis details:")
    for (i, basis) in enumerate(domain.bases)
        println("   - Basis $i: $(typeof(basis)) on $(basis.meta.element_label)")
        println("     Size: $(basis.meta.size), Bounds: $(basis.meta.bounds)")
    end
    
    return domain, x_basis, y_basis
end

function demonstrate_2d_chebyshev_bcs(domain, x_basis, y_basis)
    """Demonstrate boundary conditions on 2D Chebyshev domain"""
    
    println("\n5. Setting up boundary conditions for 2D Chebyshev...")
    
    # Create boundary condition manager
    bc_manager = BoundaryConditionManager()
    
    # Boundary conditions for a 2D problem (e.g., Poisson equation: ∇²u = f)
    # Dirichlet BCs on all boundaries
    
    println("   Adding Dirichlet boundary conditions:")
    
    # Left boundary (x = -1): u(-1, y) = sin(π*y)
    bc_left = dirichlet_bc("u", "x", -1.0, "sin(pi*y)", space_dependent=true)
    add_bc!(bc_manager, bc_left)
    println("   - Left (x=-1): u = sin(π*y)")
    
    # Right boundary (x = 1): u(1, y) = -sin(π*y)  
    bc_right = dirichlet_bc("u", "x", 1.0, "-sin(pi*y)", space_dependent=true)
    add_bc!(bc_manager, bc_right)
    println("   - Right (x=1): u = -sin(π*y)")
    
    # Bottom boundary (y = -1): u(x, -1) = 0
    bc_bottom = dirichlet_bc("u", "y", -1.0, 0.0)
    add_bc!(bc_manager, bc_bottom)
    println("   - Bottom (y=-1): u = 0")
    
    # Top boundary (y = 1): u(x, 1) = 0
    bc_top = dirichlet_bc("u", "y", 1.0, 0.0)
    add_bc!(bc_manager, bc_top)
    println("   - Top (y=1): u = 0")
    
    println("\n   Boundary condition summary:")
    println("   - Total BCs: $(length(bc_manager.conditions))")
    println("   - Space-dependent BCs: $(length(bc_manager.space_dependent_bcs))")
    println("   - Has space dependencies: $(has_space_dependent_bcs(bc_manager))")
    
    return bc_manager
end

function demonstrate_chebyshev_properties()
    """Demonstrate key properties of Chebyshev spectral methods"""
    
    println("\n6. Chebyshev spectral method properties...")
    
    println("   Key advantages:")
    println("   - Exponential convergence for smooth functions")
    println("   - Natural clustering near boundaries")
    println("   - Efficient transforms via FFT")
    println("   - Well-suited for boundary value problems")
    
    println("\n   Chebyshev grid points:")
    N = 8  # Small example for display
    
    # Chebyshev points of the first kind: x_k = cos(π*k/N) for k = 0, ..., N
    points = [cos(π * k / N) for k in 0:N]
    println("   - For N=$N points in [-1,1]:")
    for (i, x) in enumerate(points)
        println("     x[$i] = $(round(x, digits=4))")
    end
    
    println("\n   Grid spacing characteristics:")
    println("   - Denser near boundaries (±1)")
    println("   - Sparser in domain interior")
    println("   - Optimal for resolving boundary layers")
end

function demonstrate_integration_weights(domain)
    """Show integration weights for 2D Chebyshev domain"""
    
    println("\n7. Integration weights for 2D Chebyshev domain...")
    
    weights = integration_weights(domain)
    
    println("   Integration method:")
    for (i, basis) in enumerate(domain.bases)
        w = weights[i]
        println("   - $(basis.meta.element_label) direction: $(length(w)) weights")
        println("     Range: [$(round(minimum(w), digits=6)), $(round(maximum(w), digits=6))]")
        
        if isa(basis, ChebyshevT)
            println("     Method: Clenshaw-Curtis quadrature")
            println("     Boundary weight factor: 0.5")
        end
    end
    
    # Total domain volume
    vol = volume(domain)
    println("\n   Domain volume: $vol")
    println("   (Should equal 4.0 for [-1,1] × [-1,1] domain)")
end

function main()
    """Main demonstration function"""
    
    try
        # Demonstrate 2D Chebyshev setup
        domain, x_basis, y_basis = demonstrate_2d_chebyshev()
        
        # Show boundary condition handling
        bc_manager = demonstrate_2d_chebyshev_bcs(domain, x_basis, y_basis)
        
        # Demonstrate Chebyshev properties
        demonstrate_chebyshev_properties()
        
        # Show integration capabilities
        demonstrate_integration_weights(domain)
        
        println("\n" * "=" ^ 50)
        println("2D Chebyshev demonstration completed successfully!")
        
        println("\nKey capabilities demonstrated:")
        println("  - ChebyshevT basis in multiple directions")
        println("  - Multi-dimensional domain construction")
        println("  - Boundary condition application")
        println("  - Integration weight computation")
        println("  - Grid point distribution")
        
        println("\nThis confirms Tarang.jl supports Chebyshev spectral methods")
        println("in both spatial directions, following the Dedalus architecture.")
        
        return true
        
    catch e
        println("Error in 2D Chebyshev demonstration: $e")
        rethrow(e)
    end
end

# Run demonstration if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end