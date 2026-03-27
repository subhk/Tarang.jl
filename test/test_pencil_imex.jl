"""
Test suite for PencilLinearOperator and pencil-based IMEX methods.

Tests the Chebyshev-Fourier IMEX implementation for:
1. PencilLinearOperator construction
2. Pencil LHS matrix building
3. Pencil compatibility checks
"""

using Test
using LinearAlgebra
using SparseArrays

# Include Tarang (adjust path as needed)
using Tarang

@testset "PencilLinearOperator" begin

    @testset "2D Fourier-Chebyshev construction" begin
        # Create a 2D domain: Fourier in x, Chebyshev in z
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)

        # Bases
        Nx, Nz = 32, 16
        x_basis = ComplexFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=1.0)
        z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(-1.0, 1.0), dealias=1.0)

        # Create PencilLinearOperator
        L = PencilLinearOperator(dist, (x_basis, z_basis), :laplacian; ν=1e-2)

        @test L.Nz == Nz
        @test L.chebyshev_basis_idx == 2
        @test length(L.fourier_basis_indices) == 1
        @test L.fourier_basis_indices[1] == 1
        @test size(L.k2_values, 1) == Nx
        @test L.parameters[:ν] == 1e-2
    end

    @testset "3D Fourier-Fourier-Chebyshev construction" begin
        # Create a 3D domain: Fourier in x,y, Chebyshev in z
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; dtype=Float64)

        # Bases
        Nx, Ny, Nz = 16, 16, 12
        x_basis = ComplexFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=1.0)
        y_basis = ComplexFourier(coords["y"]; size=Ny, bounds=(0.0, 2π), dealias=1.0)
        z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(-1.0, 1.0), dealias=1.0)

        # Create PencilLinearOperator
        L = PencilLinearOperator(dist, (x_basis, y_basis, z_basis), :laplacian; ν=1e-3)

        @test L.Nz == Nz
        @test L.chebyshev_basis_idx == 3
        @test length(L.fourier_basis_indices) == 2
        @test size(L.k2_values) == (Nx, Ny)
    end

    @testset "Pencil LHS matrix construction" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)

        Nx, Nz = 8, 8
        x_basis = ComplexFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=1.0)
        z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(-1.0, 1.0), dealias=1.0)

        ν = 0.1
        L = PencilLinearOperator(dist, (x_basis, z_basis), :laplacian; ν=ν)

        # Build LHS matrix for k²=1, dt=0.1, γ=0.5
        k2 = 1.0
        dt = 0.1
        γ = 0.5

        LHS = build_pencil_lhs_matrix(L, k2, dt, γ)

        @test size(LHS) == (Nz, Nz)
        @test issparse(LHS)

        # Check that diagonal contains the expected term: (1 + dt*γ*ν*k²)
        # Note: D² has zeros on diagonal for Chebyshev, so diagonal should be ~expected_diag_term
        expected_diag_term = 1 + dt * γ * ν * k2
        @test LHS[1, 1] ≈ expected_diag_term atol=1e-10
    end

    @testset "Pencil cache functionality" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)

        Nx, Nz = 8, 8
        x_basis = ComplexFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=1.0)
        z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(-1.0, 1.0), dealias=1.0)

        L = PencilLinearOperator(dist, (x_basis, z_basis), :laplacian; ν=0.1)
        cache = PencilLHSCache{Float64}(L)

        dt, γ = 0.1, 0.5

        # First call - should compute factorization
        factor1 = get_pencil_lhs_factor!(cache, 1, 1, dt, γ)
        @test factor1 !== nothing
        @test cache.dt == dt
        @test cache.γ == γ

        # Second call with same parameters - should reuse
        factor2 = get_pencil_lhs_factor!(cache, 1, 1, dt, γ)
        @test factor1 === factor2  # Same object (cache hit)

        # Call with different parameters - should recompute
        factor3 = get_pencil_lhs_factor!(cache, 1, 1, dt * 2, γ)
        @test factor3 !== nothing
        @test cache.dt == dt * 2
    end
end

@testset "Pencil compatibility checks" begin
    @testset "is_pencil_imex_compatible" begin
        coords2d = CartesianCoordinates("x", "z")

        # Fourier + Chebyshev - compatible
        x_fourier = ComplexFourier(coords2d["x"]; size=16, bounds=(0.0, 2π), dealias=1.0)
        z_cheb = ChebyshevT(coords2d["z"]; size=8, bounds=(-1.0, 1.0), dealias=1.0)

        @test is_pencil_imex_compatible((x_fourier, z_cheb)) == true

        # Pure Fourier - not compatible (no Chebyshev)
        z_fourier = ComplexFourier(coords2d["z"]; size=16, bounds=(0.0, 2π), dealias=1.0)
        @test is_pencil_imex_compatible((x_fourier, z_fourier)) == false

        # Pure Chebyshev - not compatible (no Fourier)
        x_cheb = ChebyshevT(coords2d["x"]; size=8, bounds=(-1.0, 1.0), dealias=1.0)
        @test is_pencil_imex_compatible((x_cheb, z_cheb)) == false
    end

    @testset "has_chebyshev_basis" begin
        coords = CartesianCoordinates("x", "y")

        fourier_x = ComplexFourier(coords["x"]; size=16, bounds=(0.0, 2π), dealias=1.0)
        fourier_y = ComplexFourier(coords["y"]; size=16, bounds=(0.0, 2π), dealias=1.0)
        cheb_y = ChebyshevT(coords["y"]; size=8, bounds=(-1.0, 1.0), dealias=1.0)

        @test has_chebyshev_basis((fourier_x, fourier_y)) == false
        @test has_chebyshev_basis((fourier_x, cheb_y)) == true
    end
end

@testset "Hyperviscosity support" begin
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)

    Nx, Nz = 8, 8
    x_basis = ComplexFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=1.0)
    z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(-1.0, 1.0), dealias=1.0)

    # Test hyperviscosity operator construction
    L = PencilLinearOperator(dist, (x_basis, z_basis), :hyperviscosity; ν=1e-4, order=2)

    @test L.operator_type == :hyperviscosity
    @test L.parameters[:order] == 2
    @test L.parameters[:ν] == 1e-4

    # Build LHS with hyperviscosity
    k2 = 1.0
    dt = 0.1
    γ = 0.5

    LHS = build_pencil_lhs_matrix(L, k2, dt, γ)
    @test size(LHS) == (Nz, Nz)
end
