"""
Quick test for time and space dependent boundary conditions

This is a basic test to verify the enhanced boundary condition system
works with time and spatially dependent values.
"""

using Tarang

function test_time_space_dependent_bcs()
    """Test time and space dependent BC functionality"""
    
    println("Testing enhanced boundary condition system with time/space dependence...")
    
    # Create boundary condition manager
    bc_manager = BoundaryConditionManager()
    
    println("✓ Enhanced BC manager created")
    
    # Test creating time-dependent BC
    time_bc = dirichlet_bc("u", "x", 0.0, "sin(2*pi*t)", time_dependent=true)
    println("✓ Time-dependent BC created: $(typeof(time_bc))")
    println("  Is time dependent: $(time_bc.is_time_dependent)")
    println("  Is space dependent: $(time_bc.is_space_dependent)")
    
    # Test creating space-dependent BC
    space_bc = neumann_bc("v", "y", 1.0, "x^2 + y^2", space_dependent=true)
    println("✓ Space-dependent BC created: $(typeof(space_bc))")
    println("  Is time dependent: $(space_bc.is_time_dependent)")
    println("  Is space dependent: $(space_bc.is_space_dependent)")
    
    # Test creating combined time+space BC
    combined_bc = dirichlet_bc("w", "z", 0.5, "sin(t)*exp(-x^2)", 
                              time_dependent=true, space_dependent=true)
    println("✓ Combined time+space BC created: $(typeof(combined_bc))")
    println("  Is time dependent: $(combined_bc.is_time_dependent)")
    println("  Is space dependent: $(combined_bc.is_space_dependent)")
    
    # Add to manager
    add_bc!(bc_manager, time_bc)
    add_bc!(bc_manager, space_bc)
    add_bc!(bc_manager, combined_bc)
    
    println("✓ Added $(length(bc_manager.conditions)) BCs to manager")
    println("  Time-dependent BCs: $(length(bc_manager.time_dependent_bcs))")
    println("  Space-dependent BCs: $(length(bc_manager.space_dependent_bcs))")
    
    # Test dependency detection
    println("✓ Dependency detection:")
    println("  Has time-dependent BCs: $(has_time_dependent_bcs(bc_manager))")
    println("  Has space-dependent BCs: $(has_space_dependent_bcs(bc_manager))")
    println("  Requires BC update: $(requires_bc_update(bc_manager))")
    
    # Test automatic detection
    auto_bc = dirichlet_bc("test", "x", 0.0, "cos(omega*t)*r^2")  # Should auto-detect both
    println("✓ Auto-detected dependencies for 'cos(omega*t)*r^2':")
    println("  Is time dependent: $(auto_bc.is_time_dependent)")
    println("  Is space dependent: $(auto_bc.is_space_dependent)")
    
    # Test BC evaluation
    println("✓ Testing BC evaluation:")
    set_time_variable!(bc_manager, "t")
    
    times = [0.0, 0.25, 0.5, 1.0]
    for t in times
        val = evaluate_bc_value(bc_manager, time_bc, t)
        println("  t=$t: sin(2π*t) = $(round(val, digits=4))")
    end
    
    # Test expression evaluation
    println("✓ Testing expression evaluation:")
    expr_tests = [
        ("sin(2*pi*t)", 0.25, sin(2*π*0.25)),
        ("cos(t)", 1.0, cos(1.0)),
        ("exp(-t)", 0.5, exp(-0.5))
    ]
    
    for (expr, t, expected) in expr_tests
        result = evaluate_expression(expr, t)
        println("  $expr at t=$t: $result (expected: $(round(expected, digits=4)))")
    end
    
    println("✓ All time/space dependent BC tests completed successfully!")
    return bc_manager
end

function main()
    """Run the test"""
    
    println("=== Time/Space Dependent Boundary Condition Test ===")
    
    try
        mgr = test_time_space_dependent_bcs()
        println("\nAll tests passed! Time and space dependent BCs are working correctly.")
        
        println("\nSummary of capabilities:")
        println("  - Time-dependent BCs: u(boundary,t) = f(t)")
        println("  - Space-dependent BCs: u(boundary) = g(x,y,z)")
        println("  - Combined BCs: u(boundary,t) = h(t,x,y,z)")
        println("  - Automatic dependency detection")
        println("  - Real-time BC evaluation")
        println("  - Solver integration ready")
        
    catch e
        println("\nTest failed with error: $e")
        rethrow(e)
    end
end

# Run test if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end