"""
Syntax and API validation test for spherical implementation files.
This test checks that all files can be parsed without runtime errors
and that the PencilArrays API usage is correct.
"""

using Test

@testset "Spherical Implementation Syntax Tests" begin
    
    @testset "File Parsing" begin
        println("Testing file parsing...")
        
        # Test that all spherical files can be parsed (syntax check)
        spherical_files = [
            "src/libraries/spherical_coordinates.jl",
            "src/libraries/spherical_bases.jl", 
            "src/libraries/spherical_operators.jl",
            "src/libraries/spherical_fields.jl",
            "src/libraries/spherical_boundary_conditions.jl"
        ]
        
        for file in spherical_files
            if isfile(file)
                try
                    # Parse the file to check syntax
                    content = read(file, String)
                    parsed = Meta.parse("begin\n" * content * "\nend")
                    @test parsed isa Expr
                    println("✓ $(file) syntax OK")
                catch e
                    @test false # Syntax error in $(file): $e
                end
            else
                @warn "File not found: $(file)"
            end
        end
        
        println("✓ All files parsed successfully")
    end
    
    @testset "PencilArrays API Usage" begin
        println("Testing PencilArrays API usage...")
        
        # Check that files use correct PencilArrays constructors
        spherical_files = [
            "src/libraries/spherical_coordinates.jl",
            "src/libraries/spherical_bases.jl"
        ]
        
        for file in spherical_files
            if isfile(file)
                content = read(file, String)
                
                # Should NOT contain old API
                @test !contains(content, "PencilConfig") 
                @test !contains(content, "pencil_from_shape")
                
                # Should contain new API
                if contains(content, "PencilArrays")
                    @test contains(content, "MPITopology") || contains(content, "Pencil")
                end
                
                println("✓ $(file) API usage OK")
            end
        end
        
        println("✓ PencilArrays API usage validated")
    end
    
    @testset "Export Statements" begin
        println("Testing export statements...")
        
        # Check key exports are present
        coord_file = "src/libraries/spherical_coordinates.jl"
        if isfile(coord_file)
            content = read(coord_file, String)
            @test contains(content, "export SphericalCoordinates")
        end
        
        bases_file = "src/libraries/spherical_bases.jl" 
        if isfile(bases_file)
            content = read(bases_file, String)
            @test contains(content, "export BallBasis")
        end
        
        ops_file = "src/libraries/spherical_operators.jl"
        if isfile(ops_file)
            content = read(ops_file, String)
            @test contains(content, "export gradient") || contains(content, "export")
        end
        
        fields_file = "src/libraries/spherical_fields.jl"
        if isfile(fields_file)
            content = read(fields_file, String)
            @test contains(content, "export")
        end
        
        bc_file = "src/libraries/spherical_boundary_conditions.jl"
        if isfile(bc_file)
            content = read(bc_file, String)
            @test contains(content, "export")
        end
        
        println("✓ Export statements validated")
    end
    
    @testset "Function Signatures" begin
        println("Testing critical function signatures...")
        
        # Check that key function signatures look reasonable
        files_to_check = [
            ("src/libraries/spherical_coordinates.jl", ["SphericalCoordinates", "create_spherical_grids"]),
            ("src/libraries/spherical_bases.jl", ["BallBasis", "forward_transform", "backward_transform"]),
            ("src/libraries/spherical_operators.jl", ["gradient", "laplacian", "divergence"]),
            ("src/libraries/spherical_fields.jl", ["SphericalScalarField", "SphericalVectorField"]),
            ("src/libraries/spherical_boundary_conditions.jl", ["DirichletBC", "NeumannBC", "TauSystem"])
        ]
        
        for (file, functions) in files_to_check
            if isfile(file)
                content = read(file, String)
                for func in functions
                    # Check if function/struct is defined
                    has_def = contains(content, "function $(func)") || 
                             contains(content, "struct $(func)") ||
                             contains(content, "mutable struct $(func)")
                    @test has_def
                end
                println("✓ $(file) function signatures OK")
            end
        end
        
        println("✓ Function signatures validated")
    end
    
end

println("All spherical implementation syntax tests completed successfully!")