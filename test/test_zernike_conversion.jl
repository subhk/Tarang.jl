using Test
using LinearAlgebra
using SparseArrays
using SpecialFunctions

include("../src/libraries/spherical_bases.jl")

@testset "Zernike Conversion Matrix Tests" begin
    
    @testset "Basic Matrix Properties" begin
        println("Testing basic conversion matrix properties...")
        
        n_max, l_max = 8, 4
        alpha, beta = 0.5, 0.5
        T = Float64
        
        # Build conversion matrix
        C = build_zernike_conversion_matrix(n_max, l_max, alpha, beta, T)
        
        # Test matrix dimensions
        @test size(C) == (n_max + 1, n_max + 1)
        
        # Test that matrix elements are finite
        @test all(isfinite.(C))
        
        # Test sparsity structure (should be mostly zeros)
        nnz_ratio = nnz(C) / (size(C,1) * size(C,2))
        @test nnz_ratio < 0.5  # Should be reasonably sparse
        
        # Test diagonal dominance (diagonal should be 1.0)
        for i in 1:min(size(C)...)
            @test abs(C[i,i] - 1.0) < 1e-10
        end
        
        println("✓ Basic matrix properties test passed")
    end
    
    @testset "Jacobi Conversion Coefficients" begin
        println("Testing Jacobi conversion coefficients...")
        
        alpha, beta = 1.0, 0.5
        T = Float64
        
        # Test diagonal terms
        @test jacobi_conversion_coefficient(0, 0, alpha, beta, T) ≈ 1.0
        @test jacobi_conversion_coefficient(1, 1, alpha, beta, T) ≈ 1.0
        @test jacobi_conversion_coefficient(5, 5, alpha, beta, T) ≈ 1.0
        
        # Test out-of-bounds
        @test jacobi_conversion_coefficient(3, 5, alpha, beta, T) ≈ 0.0
        @test jacobi_conversion_coefficient(-1, 0, alpha, beta, T) ≈ 0.0
        @test jacobi_conversion_coefficient(0, -1, alpha, beta, T) ≈ 0.0
        
        # Test first sub-diagonal terms
        coeff_10 = jacobi_conversion_coefficient(1, 0, alpha, beta, T)
        @test isfinite(coeff_10)
        @test abs(coeff_10) < 10  # Should be bounded
        
        coeff_21 = jacobi_conversion_coefficient(2, 1, alpha, beta, T)
        @test isfinite(coeff_21)
        
        println("✓ Jacobi conversion coefficients test passed")
    end
    
    @testset "Regularity Conditions" begin
        println("Testing regularity conditions...")
        
        n_max, l_max = 6, 3
        alpha, beta = 0.0, 0.0  # Legendre case
        T = Float64
        
        C = build_zernike_conversion_matrix(n_max, l_max, alpha, beta, T)
        
        # Check that irregular modes (n-l odd) are properly handled
        for n in 0:n_max
            for l in 0:min(l_max, n)
                if (n - l) % 2 != 0
                    # This row should have been zeroed out or properly handled
                    row_sum = sum(abs, C[n+1, :])
                    @test row_sum < 1e-10 || row_sum > 0.5  # Either zeroed or normalized
                end
            end
        end
        
        println("✓ Regularity conditions test passed")
    end
    
    @testset "Helper Functions" begin
        println("Testing helper functions...")
        
        T = Float64
        
        # Test binomial coefficient
        @test binomial_coefficient(5, 0, T) ≈ 1.0
        @test binomial_coefficient(5, 5, T) ≈ 1.0
        @test binomial_coefficient(5, 2, T) ≈ 10.0
        @test binomial_coefficient(4, 2, T) ≈ 6.0
        @test binomial_coefficient(3, 4, T) ≈ 0.0  # k > n
        
        # Test power ratio
        alpha, beta = 1.0, 0.5
        @test power_ratio(alpha, beta, 0, T) ≈ 1.0
        @test power_ratio(alpha, beta, 1, T) ≈ (alpha + beta + 2) / 2
        
        pr2 = power_ratio(alpha, beta, 2, T)
        expected = ((alpha + beta + 2) / 2) * ((alpha + beta + 3) / 2)
        @test pr2 ≈ expected
        
        # Test conversion recursion
        n, k = 3, 1
        coeff = compute_conversion_recursion(n, k, alpha, beta, T)
        @test isfinite(coeff)
        
        println("✓ Helper functions test passed")
    end
    
    @testset "Matrix Orthogonality Properties" begin
        println("Testing matrix orthogonality properties...")
        
        n_max, l_max = 4, 2
        alpha, beta = 0.5, 0.5
        T = Float64
        
        C = build_zernike_conversion_matrix(n_max, l_max, alpha, beta, T)
        
        # Test that the matrix preserves some structure
        # For conversion matrices, C should satisfy certain orthogonality-like properties
        
        # Check row normalization (each row should have reasonable norm)
        for i in 1:size(C, 1)
            row_norm = norm(C[i, :])
            if row_norm > eps(T)
                @test 0.5 < row_norm < 2.0  # Should be reasonably normalized
            end
        end
        
        # Check that matrix is not degenerate
        det_C = det(Matrix(C))
        @test abs(det_C) > 1e-10  # Should be non-singular
        
        println("✓ Matrix orthogonality properties test passed")
    end
    
    @testset "Parameter Sensitivity" begin
        println("Testing parameter sensitivity...")
        
        n_max, l_max = 4, 2
        T = Float64
        
        # Test different parameter combinations
        params = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        
        for (alpha, beta) in params
            C = build_zernike_conversion_matrix(n_max, l_max, alpha, beta, T)
            
            # Basic sanity checks
            @test all(isfinite.(C))
            @test size(C) == (n_max + 1, n_max + 1)
            
            # Diagonal should be identity-like
            for i in 1:min(size(C)...)
                @test abs(C[i,i] - 1.0) < 1e-8 || abs(C[i,i]) < 1e-8  # Either 1 or 0
            end
        end
        
        println("✓ Parameter sensitivity test passed")
    end
    
end

println("All Zernike conversion matrix tests completed successfully!")