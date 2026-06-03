"""
Test suite for phi_functions.jl — ETD φ function computation.

Tests:
1. Scalar φ functions at z=0 (known limits)
2. Scalar φ functions at moderate z (direct formula)
3. Scalar φ functions satisfy recurrence φₖ(z) = (φₖ₋₁(z) - 1/(k-1)!) / z
4. Matrix φ functions for diagonal matrix (compare to scalar)
5. Matrix φ functions for small matrix (Taylor branch)
6. Roundtrip identity: exp(z) = φ₀(z), φ₁(z)*z = exp(z)-1
"""

using Test
using LinearAlgebra
using Tarang

import Tarang: phi_functions, phi_functions_matrix,
               phiv_vector, expv_krylov, _phi_functions_pade, _phi_via_eigen,
               _phi_functions_krylov

@testset "Phi Functions" begin

    @testset "Scalar φ at z=0 (Taylor branch)" begin
        φ₀, φ₁, φ₂, φ₃ = phi_functions(0.0)
        @test φ₀ ≈ 1.0       # exp(0) = 1
        @test φ₁ ≈ 1.0       # lim_{z→0} (exp(z)-1)/z = 1
        @test φ₂ ≈ 0.5       # lim_{z→0} (exp(z)-1-z)/z² = 1/2
        @test φ₃ ≈ 1/6       # lim_{z→0} = 1/3! = 1/6
    end

    @testset "Scalar φ at small z (Taylor branch)" begin
        z = 1e-10
        φ₀, φ₁, φ₂, φ₃ = phi_functions(z)
        @test φ₀ ≈ exp(z) atol=1e-14
        @test φ₁ ≈ 1.0 atol=1e-8
        @test φ₂ ≈ 0.5 atol=1e-8
        @test φ₃ ≈ 1/6 atol=1e-8
    end

    @testset "Scalar φ at moderate z (direct formula)" begin
        for z in [-10.0, -1.0, -0.1, 0.1, 1.0, 5.0]
            φ₀, φ₁, φ₂, φ₃ = phi_functions(z)

            # Verify definitions
            @test φ₀ ≈ exp(z)
            @test φ₁ ≈ (exp(z) - 1) / z
            @test φ₂ ≈ (exp(z) - 1 - z) / z^2
            @test φ₃ ≈ (exp(z) - 1 - z - z^2/2) / z^3
        end
    end

    @testset "Scalar φ recurrence relation" begin
        # φₖ(z) = (φₖ₋₁(z) - 1/(k-1)!) / z
        for z in [-5.0, -1.0, 0.5, 2.0, 10.0]
            φ₀, φ₁, φ₂, φ₃ = phi_functions(z)

            # φ₁ = (φ₀ - 1/0!) / z = (exp(z) - 1) / z
            @test φ₁ ≈ (φ₀ - 1) / z rtol=1e-10

            # φ₂ = (φ₁ - 1/1!) / z
            @test φ₂ ≈ (φ₁ - 1) / z rtol=1e-10

            # φ₃ = (φ₂ - 1/2!) / z
            @test φ₃ ≈ (φ₂ - 0.5) / z rtol=1e-10
        end
    end

    @testset "Matrix φ for diagonal matrix" begin
        # For diagonal A, matrix φ functions should match scalar φ applied element-wise
        λ = [-5.0, -1.0, -0.1]
        A = diagm(λ)
        dt = 0.1

        exp_hA, φ₁_hA, φ₂_hA = phi_functions_matrix(A, dt)

        for (i, λᵢ) in enumerate(λ)
            z = dt * λᵢ
            φ₀_s, φ₁_s, φ₂_s, _ = phi_functions(z)

            @test exp_hA[i, i] ≈ φ₀_s rtol=1e-8
            @test φ₁_hA[i, i] ≈ φ₁_s rtol=1e-8
            @test φ₂_hA[i, i] ≈ φ₂_s rtol=1e-8
        end

        # Off-diagonal should be zero for diagonal input
        for i in 1:3, j in 1:3
            if i != j
                @test abs(exp_hA[i, j]) < 1e-10
            end
        end
    end

    @testset "Matrix φ for small matrix (Taylor branch)" begin
        # Very small matrix → Taylor expansion path
        A = 1e-10 * [1.0 0.5; 0.5 1.0]
        dt = 1.0

        exp_hA, φ₁_hA, φ₂_hA = phi_functions_matrix(A, dt)

        I_mat = Matrix{Float64}(I, 2, 2)

        # For very small z, exp(z) ≈ I + z
        @test isapprox(exp_hA, I_mat + dt * A; atol=1e-8)

        # φ₁ ≈ I for very small z
        @test isapprox(φ₁_hA, I_mat; atol=1e-8)

        # φ₂ ≈ I/2 for very small z
        @test isapprox(φ₂_hA, I_mat / 2; atol=1e-8)
    end

    @testset "Matrix φ identity: exp(hA) = I + hA*φ₁(hA)" begin
        A = [-3.0 1.0; 0.5 -2.0]
        dt = 0.5

        exp_hA, φ₁_hA, _ = phi_functions_matrix(A, dt)

        I_mat = Matrix{Float64}(I, 2, 2)
        z = dt * A

        # exp(z) = I + z * φ₁(z)
        reconstructed = I_mat + z * φ₁_hA
        @test isapprox(exp_hA, reconstructed; rtol=1e-8)
    end

    @testset "Negative eigenvalues decay" begin
        # For diffusion-like operator with negative eigenvalues,
        # exp(hA) should have spectral radius < 1
        A = diagm([-1.0, -10.0, -100.0])
        dt = 0.01

        exp_hA, _, _ = phi_functions_matrix(A, dt)

        # All diagonal entries should be decaying exponentials
        @test exp_hA[1, 1] ≈ exp(-0.01)
        @test exp_hA[2, 2] ≈ exp(-0.1)
        @test exp_hA[3, 3] ≈ exp(-1.0)
    end

    @testset "Identity matrix caching" begin
        import Tarang: _get_identity_matrix, _phi_identity_cache

        I1 = _get_identity_matrix(4, Float64)
        I2 = _get_identity_matrix(4, Float64)
        @test I1 === I2  # Same object (cached)
        @test I1 == Matrix{Float64}(LinearAlgebra.I, 4, 4)

        # Different size returns different object
        I3 = _get_identity_matrix(3, Float64)
        @test I3 !== I1
        @test size(I3) == (3, 3)

        # Different type returns different object
        I4 = _get_identity_matrix(4, ComplexF64)
        @test I4 !== I1
        @test eltype(I4) == ComplexF64
    end

    @testset "phi_functions_matrix small norm (Taylor)" begin
        A = diagm([0.001, 0.002, 0.003])
        dt = 0.001
        exp_z, φ₁, φ₂ = phi_functions_matrix(A, dt)

        z = dt * A
        @test exp_z ≈ exp(z) atol=1e-10

        I_mat = Matrix{Float64}(LinearAlgebra.I, 3, 3)
        @test z * φ₁ ≈ exp(z) - I_mat atol=1e-10
    end

    @testset "phi_functions_matrix moderate norm" begin
        A = diagm([1.0, 2.0, 3.0])
        dt = 1.0
        exp_z, φ₁, φ₂ = phi_functions_matrix(A, dt)
        @test exp_z ≈ exp(dt * A) atol=1e-8
    end

    # Per-eigenvalue scalar oracle for a diagonal matrix's φ functions.
    _phi_diag(λs, dt, k) = diagm([phi_functions(dt * l)[k + 1] for l in λs])

    @testset "Singular matrix routes through _phi_via_eigen (λ=0 removable)" begin
        # A zero eigenvalue makes z \\ (exp(z)-I) singular; phi_functions_matrix
        # must fall back to the eigen-based evaluation (φ₁(0)=1, φ₂(0)=1/2).
        λs = [0.0, -1.0, -3.0]
        A = diagm(λs); dt = 1.0
        exp_z, φ₁, φ₂ = phi_functions_matrix(A, dt)
        @test exp_z ≈ _phi_diag(λs, dt, 0) rtol=1e-10
        @test φ₁    ≈ _phi_diag(λs, dt, 1) rtol=1e-10
        @test φ₂    ≈ _phi_diag(λs, dt, 2) rtol=1e-10
        # the λ=0 mode hits the removable-singularity limits exactly
        @test φ₁[1, 1] ≈ 1.0   rtol=1e-12
        @test φ₂[1, 1] ≈ 0.5   rtol=1e-12

        # call _phi_via_eigen directly on a non-diagonal (but diagonalizable) z
        z = [0.0 0.2; 0.0 -1.5]
        ez, p1, p2 = _phi_via_eigen(z, Matrix{Float64}(LinearAlgebra.I, 2, 2))
        @test ez ≈ exp(z) rtol=1e-10
        @test z * p1 ≈ exp(z) - LinearAlgebra.I rtol=1e-9
    end

    @testset "_phi_functions_pade matches dense exp/φ" begin
        z = [0.3 0.1; -0.2 0.4]
        exp_z, φ₁, φ₂ = _phi_functions_pade(z)
        # Padé [3/3] with scaling-and-squaring — moderate accuracy
        @test exp_z ≈ exp(z) atol=1e-6
        @test z * φ₁ ≈ exp_z - LinearAlgebra.I atol=1e-7
        @test z * φ₂ ≈ φ₁ - LinearAlgebra.I atol=1e-7
    end

    @testset "Large/stiff norm routes through _phi_functions_krylov" begin
        λs = [-60.0, -70.0, -80.0]      # norm > 50 → Krylov branch
        A = diagm(λs)
        exp_z, φ₁, φ₂ = phi_functions_matrix(A, 1.0)
        @test exp_z ≈ _phi_diag(λs, 1.0, 0) rtol=1e-8
        @test φ₁    ≈ _phi_diag(λs, 1.0, 1) rtol=1e-8
    end

    @testset "expv_krylov and phiv_vector match dense oracle" begin
        C = [-2.0 0.5; 0.1 -3.0]
        b = [1.0, 2.0]
        t = 0.7
        # exp(tC)·b
        @test expv_krylov(t, C, b) ≈ exp(t * C) * b rtol=1e-10
        # phiv_vector columns: [φ₀(tC)b, φ₁(tC)b, φ₂(tC)b]
        pv = phiv_vector(t, C, b, 2)
        @test pv[:, 1] ≈ exp(t * C) * b rtol=1e-10
        φ₁C = (t .* C) \ (exp(t * C) - LinearAlgebra.I)
        @test pv[:, 2] ≈ φ₁C * b rtol=1e-9
    end
end
