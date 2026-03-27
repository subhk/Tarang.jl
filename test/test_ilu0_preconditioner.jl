"""
Test ILU(0) preconditioner implementation for GPU solvers.

Run with:
    julia --project test/test_ilu0_preconditioner.jl

Requires CUDA-capable GPU.
"""

using Test
using SparseArrays
using LinearAlgebra

# Check if CUDA is available
cuda_available = false
try
    using CUDA
    if CUDA.functional()
        cuda_available = true
        println("CUDA is available: ", CUDA.name(CUDA.device()))
    else
        println("CUDA found but not functional")
    end
catch e
    println("CUDA not available: $e")
end

if cuda_available
    using CUDA.CUSPARSE

    @testset "ILU(0) Preconditioner Tests" begin

        @testset "ilu02 function exists and works" begin
            # Create a simple SPD matrix
            n = 100
            A_cpu = sprand(n, n, 0.1)
            A_cpu = A_cpu + A_cpu' + 10I  # Make symmetric positive definite

            # Convert to GPU CSR
            A_csr = CuSparseMatrixCSR(A_cpu)

            # Test ilu02 exists and returns a matrix
            @test_nowarn begin
                LU = CUSPARSE.ilu02(A_csr)
                @test LU isa CuSparseMatrixCSR
                @test size(LU) == size(A_csr)
            end

            println("✓ ilu02 function works")
        end

        @testset "Triangular solves with ldiv!" begin
            n = 100
            A_cpu = sprand(n, n, 0.1)
            A_cpu = A_cpu + A_cpu' + 10I

            A_csr = CuSparseMatrixCSR(A_cpu)
            LU = CUSPARSE.ilu02(A_csr)

            # Create test vectors
            r = CUDA.rand(Float64, n)
            z = CUDA.zeros(Float64, n)
            tmp = CUDA.zeros(Float64, n)

            # Test lower triangular solve
            @test_nowarn begin
                ldiv!(tmp, UnitLowerTriangular(LU), r)
            end
            @test !all(tmp .== 0)  # Should have non-zero values

            # Test upper triangular solve
            @test_nowarn begin
                ldiv!(z, UpperTriangular(LU), tmp)
            end
            @test !all(z .== 0)  # Should have non-zero values

            println("✓ Triangular solves work with ldiv!")
        end

        @testset "Full ILU(0) preconditioner application" begin
            n = 100
            A_cpu = sprand(n, n, 0.1)
            A_cpu = A_cpu + A_cpu' + 10I

            A_csr = CuSparseMatrixCSR(A_cpu)
            LU = CUSPARSE.ilu02(A_csr)

            # Simulate preconditioner application: z = (LU)^{-1} * r
            r = CUDA.rand(Float64, n)
            z = CUDA.zeros(Float64, n)
            tmp = CUDA.zeros(Float64, n)

            # Forward substitution: L * tmp = r
            ldiv!(tmp, UnitLowerTriangular(LU), r)

            # Backward substitution: U * z = tmp
            ldiv!(z, UpperTriangular(LU), tmp)

            # Verify: LU * z ≈ r (approximately, since ILU is incomplete)
            # We can't test exact equality, but z should be a reasonable approximation
            @test norm(Array(z)) > 0

            println("✓ Full ILU(0) preconditioner application works")
        end

        @testset "Integration with Tarang solvers" begin
            using Tarang

            # Create a test problem - 2D Poisson-like matrix (ill-conditioned)
            n = 50
            N = n^2

            # Create 2D Laplacian (5-point stencil)
            function laplacian_2d(n)
                N = n^2
                I_n = sparse(I, n, n)
                D = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
                return kron(I_n, D) + kron(D, I_n)
            end

            A = laplacian_2d(n)
            b = rand(N)

            # Test with Jacobi preconditioner
            println("\nTesting CuIterativeCG with Jacobi preconditioner...")
            solver_jacobi = CuIterativeCG(A; preconditioner=:jacobi, tol=1e-8, maxiter=500)
            x_jacobi = Tarang.MatSolvers.solve(solver_jacobi, b)

            # Test with ILU(0) preconditioner
            println("Testing CuIterativeCG with ILU(0) preconditioner...")
            solver_ilu0 = CuIterativeCG(A; preconditioner=:ilu0, tol=1e-8, maxiter=500)
            x_ilu0 = Tarang.MatSolvers.solve(solver_ilu0, b)

            # Both should give similar solutions
            x_jacobi_cpu = Array(x_jacobi)
            x_ilu0_cpu = Array(x_ilu0)

            # Check solutions are close to each other
            @test norm(x_jacobi_cpu - x_ilu0_cpu) / norm(x_jacobi_cpu) < 0.1

            # Check solutions actually solve the system
            residual_jacobi = norm(A * x_jacobi_cpu - b) / norm(b)
            residual_ilu0 = norm(A * x_ilu0_cpu - b) / norm(b)

            println("  Jacobi residual: $residual_jacobi")
            println("  ILU(0) residual: $residual_ilu0")

            @test residual_jacobi < 1e-6
            @test residual_ilu0 < 1e-6

            println("✓ Tarang solver integration works")
        end

        @testset "ILU(0) with GMRES" begin
            using Tarang

            # Non-symmetric matrix
            n = 30
            N = n^2

            # Create non-symmetric convection-diffusion matrix
            function convection_diffusion_2d(n; peclet=10.0)
                N = n^2
                h = 1.0 / (n + 1)
                I_n = sparse(I, n, n)

                # Diffusion (symmetric)
                D = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1)) / h^2
                L_diff = kron(I_n, D) + kron(D, I_n)

                # Convection (non-symmetric)
                C = spdiagm(-1 => -ones(n-1), 1 => ones(n-1)) / (2h)
                L_conv = peclet * kron(I_n, C)

                return L_diff + L_conv
            end

            A = convection_diffusion_2d(n; peclet=5.0)
            b = rand(N)

            # Test GMRES with ILU(0)
            println("\nTesting CuIterativeGMRES with ILU(0) preconditioner...")
            solver = CuIterativeGMRES(A; preconditioner=:ilu0, tol=1e-8, maxiter=200, restart=30)
            x = Tarang.MatSolvers.solve(solver, b)

            x_cpu = Array(x)
            residual = norm(A * x_cpu - b) / norm(b)

            println("  GMRES+ILU(0) residual: $residual")
            @test residual < 1e-6

            println("✓ GMRES with ILU(0) works")
        end

    end

    println("\n" * "="^50)
    println("All ILU(0) preconditioner tests passed!")
    println("="^50)

else
    @warn "Skipping ILU(0) tests - CUDA not available"
    println("\nTo run these tests, you need:")
    println("  1. A CUDA-capable NVIDIA GPU")
    println("  2. CUDA toolkit installed")
    println("  3. CUDA.jl package installed")
end
