"""
Test suite for coords.jl

Tests:
1. CartesianCoordinates creation (1D, 2D, 3D)
2. Coordinate name access (string indexing, integer indexing, .names)
3. Dimension checking
4. Invalid input errors (duplicate names, out-of-bounds access)
5. Coordinate equality and right-handedness
"""

using Test

@testset "Coordinate Systems" begin
    using Tarang

    # ========================================================================
    # CartesianCoordinates Construction
    # ========================================================================

    @testset "CartesianCoordinates Construction" begin
        @testset "1D Cartesian" begin
            coords = CartesianCoordinates("x")
            @test coords.dim == 1
            @test coords.names == ["x"]
            @test length(coords.coords) == 1
            @test coords.curvilinear == false
            @test coords.right_handed === nothing  # only set for 3D
        end

        @testset "2D Cartesian" begin
            coords = CartesianCoordinates("x", "z")
            @test coords.dim == 2
            @test coords.names == ["x", "z"]
            @test length(coords.coords) == 2
            @test coords.curvilinear == false
            @test coords.right_handed === nothing
        end

        @testset "3D Cartesian" begin
            coords = CartesianCoordinates("x", "y", "z")
            @test coords.dim == 3
            @test coords.names == ["x", "y", "z"]
            @test length(coords.coords) == 3
            @test coords.curvilinear == false
            @test coords.right_handed == true
        end

        @testset "3D Cartesian left-handed" begin
            coords = CartesianCoordinates("x", "y", "z"; right_handed=false)
            @test coords.right_handed == false
        end
    end

    # ========================================================================
    # Coordinate Access
    # ========================================================================

    @testset "Coordinate Access" begin
        coords = CartesianCoordinates("x", "y", "z")

        @testset "String indexing" begin
            cx = coords["x"]
            @test cx isa Tarang.Coordinate
            @test cx.name == "x"

            cy = coords["y"]
            @test cy.name == "y"

            cz = coords["z"]
            @test cz.name == "z"
        end

        @testset "Integer indexing" begin
            @test coords[1].name == "x"
            @test coords[2].name == "y"
            @test coords[3].name == "z"
        end

        @testset "Coordinate back-reference to parent" begin
            cx = coords["x"]
            @test cx.coordsys === coords
            @test cx.dim == 1
            @test cx.curvilinear == false
        end
    end

    # ========================================================================
    # Equality
    # ========================================================================

    @testset "Coordinate Equality" begin
        c1 = CartesianCoordinates("x", "y")
        c2 = CartesianCoordinates("x", "y")
        c3 = CartesianCoordinates("x", "z")

        @test c1 == c2
        @test c1 != c3
    end

    # ========================================================================
    # Invalid Input Errors
    # ========================================================================

    @testset "Invalid Input Errors" begin
        @testset "Duplicate coordinate names" begin
            @test_throws ArgumentError CartesianCoordinates("x", "x")
        end

        @testset "Invalid string key" begin
            coords = CartesianCoordinates("x", "y")
            @test_throws KeyError coords["z"]
        end

        @testset "Out-of-bounds integer key" begin
            coords = CartesianCoordinates("x", "y")
            @test_throws BoundsError coords[0]
            @test_throws BoundsError coords[3]
        end
    end
end
