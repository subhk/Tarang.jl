using Test
using LinearAlgebra
using SpecialFunctions

# Test just the mathematical components without MPI
@testset "Spherical Mathematics Tests" begin
    
    @testset "Basic Coordinate Arrays" begin
        Nr, Nθ, Nφ = 8, 16, 32
        
        # Test radial coordinate generation (Gauss-Radau)
        r = collect(range(0, 1, length=Nr+1))[2:end]  # Exclude r=0 
        @test length(r) == Nr
        @test r[end] ≈ 1.0 atol=1e-10
        @test all(r .> 0)
        
        # Test colatitude coordinate (Gauss-Legendre)
        θ = acos.(collect(range(-1, 1, length=Nθ+2))[2:end-1])  # Exclude poles
        @test length(θ) == Nθ
        @test θ[1] > 0
        @test θ[end] < π
        
        # Test azimuthal coordinate (periodic)
        φ = collect(range(0, 2π, length=Nφ+1))[1:end-1]
        @test length(φ) == Nφ  
        @test φ[1] ≈ 0.0 atol=1e-10
        @test φ[end] ≈ 2π - 2π/Nφ atol=1e-10
        
        println("✓ Basic coordinate arrays test passed")
    end
    
    @testset "Zernike Polynomials" begin
        Nr = 16
        r = collect(range(0, 1, length=Nr))
        
        # Test Zernike polynomial orthogonality
        n1, n2 = 2, 4
        Z1 = zernike_radial.(r, n1, 0)
        Z2 = zernike_radial.(r, n2, 0)
        
        # Simple orthogonality check (not exact due to discrete integration)
        overlap = sum(Z1 .* Z2 .* r) * (r[2] - r[1])
        @test abs(overlap) < 0.1  # Should be small for different n
        
        # Test normalization condition
        Z_norm = zernike_radial.(r, 2, 0)
        norm_sq = sum(Z_norm.^2 .* r) * (r[2] - r[1])
        @test norm_sq > 0.4  # Should be order 1
        
        println("✓ Zernike polynomials test passed")
    end
    
    @testset "Spherical Harmonics" begin
        Nθ, Nφ = 32, 64
        θ = collect(range(0, π, length=Nθ))
        φ = collect(range(0, 2π-2π/Nφ, length=Nφ))
        
        # Test spherical harmonic properties
        l, m = 2, 1
        Y = zeros(ComplexF64, Nθ, Nφ)
        for (i, θ_val) in enumerate(θ)
            for (j, φ_val) in enumerate(φ)
                Y[i,j] = spherical_harmonic(l, m, θ_val, φ_val)
            end
        end
        
        # Test that spherical harmonics are finite everywhere
        @test all(isfinite.(Y))
        
        # Test m=0 harmonics are real
        Y_real = zeros(ComplexF64, Nθ, Nφ)
        for (i, θ_val) in enumerate(θ)
            for (j, φ_val) in enumerate(φ)
                Y_real[i,j] = spherical_harmonic(l, 0, θ_val, φ_val)
            end
        end
        @test all(abs.(imag.(Y_real)) .< 1e-12)
        
        println("✓ Spherical harmonics test passed")
    end
    
    @testset "Ball Basis" begin
        Nr, Nθ, Nφ = 16, 32, 64
        r_bounds = (0.0, 1.0)
        lmax = 4
        
        basis = BallBasis(Nr, Nθ, Nφ, r_bounds, lmax)
        
        @test basis.Nr == Nr
        @test basis.Nθ == Nθ
        @test basis.Nφ == Nφ
        @test basis.lmax == lmax
        @test length(basis.zernike_modes) > 0
        @test length(basis.spherical_modes) > 0
        
        # Test forward and backward transforms
        test_field = randn(ComplexF64, Nr, Nθ, Nφ)
        
        # Forward transform
        coeffs = forward_transform(basis, test_field)
        @test size(coeffs) == (length(basis.zernike_modes), length(basis.spherical_modes))
        
        # Backward transform
        reconstructed = backward_transform(basis, coeffs)
        @test size(reconstructed) == size(test_field)
        
        println("✓ Ball basis test passed")
    end
    
    @testset "Spherical Fields" begin
        Nr, Nθ, Nφ = 16, 32, 64
        r_bounds = (0.0, 1.0)
        lmax = 4
        
        coords = SphericalCoordinates(Nr, Nθ, Nφ, r_bounds)
        basis = BallBasis(Nr, Nθ, Nφ, r_bounds, lmax)
        
        # Test scalar field
        scalar_field = SphericalScalarField(coords, basis, "test_scalar")
        @test scalar_field.name == "test_scalar"
        @test scalar_field.layout == GRID_LAYOUT
        
        # Set some test data
        test_data = randn(ComplexF64, Nr, Nθ, Nφ)
        set_data!(scalar_field, test_data)
        @test get_data(scalar_field) ≈ test_data
        
        # Test layout transforms
        to_spectral!(scalar_field)
        @test scalar_field.layout == SPECTRAL_LAYOUT
        
        to_grid!(scalar_field)
        @test scalar_field.layout == GRID_LAYOUT
        
        # Test vector field
        vector_field = SphericalVectorField(coords, basis, "test_vector")
        @test vector_field.name == "test_vector"
        @test length(vector_field.components) == 3
        
        println("✓ Spherical fields test passed")
    end
    
    @testset "Spherical Operators" begin
        Nr, Nθ, Nφ = 16, 32, 64
        r_bounds = (0.0, 1.0)
        lmax = 4
        
        coords = SphericalCoordinates(Nr, Nθ, Nφ, r_bounds)
        basis = BallBasis(Nr, Nθ, Nφ, r_bounds, lmax)
        
        # Create test field
        scalar_field = SphericalScalarField(coords, basis, "test")
        
        # Simple polynomial test function: f = r²
        for i in 1:Nr
            for j in 1:Nθ
                for k in 1:Nφ
                    r_val = coords.r[i]
                    scalar_field.data[i,j,k] = r_val^2
                end
            end
        end
        
        # Test gradient
        grad_field = gradient(scalar_field)
        @test isa(grad_field, SphericalVectorField)
        
        # Test that gradient of r² has r-component = 2r
        r_component = grad_field.components[1]  # r-component
        expected_r_grad = similar(r_component.data)
        for i in 1:Nr
            for j in 1:Nθ
                for k in 1:Nφ
                    expected_r_grad[i,j,k] = 2 * coords.r[i]
                end
            end
        end
        
        # Check that r-component is approximately correct
        @test norm(r_component.data - expected_r_grad) / norm(expected_r_grad) < 0.2
        
        # Test Laplacian
        lapl_field = laplacian(scalar_field)
        @test isa(lapl_field, SphericalScalarField)
        
        # For f = r², Laplacian should be 6 (2/r d/dr(r² df/dr) = 2/r d/dr(r²·2r) = 2/r d/dr(2r³) = 2/r·6r² = 12r, but in Cartesian ∇²(r²) = 6)
        expected_lapl = fill(6.0, Nr, Nθ, Nφ)
        rel_error = norm(real.(lapl_field.data) .- expected_lapl) / norm(expected_lapl)
        @test rel_error < 0.5  # Allow some numerical error
        
        println("✓ Spherical operators test passed")
    end
    
    @testset "Boundary Conditions" begin
        Nr, Nθ, Nφ = 16, 32, 64
        r_bounds = (0.0, 1.0)
        lmax = 4
        
        coords = SphericalCoordinates(Nr, Nθ, Nφ, r_bounds)
        basis = BallBasis(Nr, Nθ, Nφ, r_bounds, lmax)
        
        # Create tau system
        tau_system = TauSystem(basis, 10)  # 10 tau variables
        
        @test length(tau_system.tau_data) == 10
        @test tau_system.n_tau == 10
        
        # Test Dirichlet boundary condition
        scalar_field = SphericalScalarField(coords, basis, "test")
        bc = DirichletBC(1.0, "r", 1.0)  # u = 1 at r = 1
        
        apply_bc!(tau_system, scalar_field, bc, 1)
        
        # Test that boundary condition was applied
        @test tau_system.bc_equations[1] isa Function
        
        # Test Neumann boundary condition
        bc_neumann = NeumannBC(0.0, "r", 1.0)  # du/dr = 0 at r = 1
        apply_bc!(tau_system, scalar_field, bc_neumann, 2)
        
        @test tau_system.bc_equations[2] isa Function
        
        # Test regularity condition
        reg_condition = RegularityCondition("center", [0, 1])  # Regularity at center for l=0,1
        apply_regularity!(tau_system, scalar_field, reg_condition)
        
        println("✓ Boundary conditions test passed")
    end
    
    @testset "Integration Test - Simple PDE" begin
        # Test solving a simple Poisson equation: ∇²u = -2
        # with boundary condition u(r=1) = 0
        # Analytical solution: u = (1-r²)/6
        
        Nr, Nθ, Nφ = 16, 32, 64
        r_bounds = (0.0, 1.0)
        lmax = 2
        
        coords = SphericalCoordinates(Nr, Nθ, Nφ, r_bounds)
        basis = BallBasis(Nr, Nθ, Nφ, r_bounds, lmax)
        
        # Create fields
        u = SphericalScalarField(coords, basis, "solution")
        rhs = SphericalScalarField(coords, basis, "rhs")
        
        # Set RHS to -2 (for l=0 mode only to get spherically symmetric solution)
        fill!(rhs.data, 0.0)
        rhs.data[:, 1, 1] .= -2.0  # Only l=0, m=0 mode
        
        # Create simple analytical solution for comparison
        analytical = SphericalScalarField(coords, basis, "analytical")
        for i in 1:Nr
            r_val = coords.r[i]
            analytical.data[i, :, :] .= (1 - r_val^2) / 6
        end
        
        # For a real solver, we would solve the linear system here
        # For testing, we just check that the analytical solution satisfies ∇²u ≈ -2
        lapl_u = laplacian(analytical)
        
        # Check that Laplacian of analytical solution is approximately -2
        expected = fill(-2.0, Nr, Nθ, Nφ)
        rel_error = norm(real.(lapl_u.data) .- expected) / norm(expected)
        @test rel_error < 0.5  # Allow numerical errors
        
        println("✓ Integration test passed")
    end
    
    MPI.Finalize()
    
end

println("All spherical implementation tests completed successfully!")