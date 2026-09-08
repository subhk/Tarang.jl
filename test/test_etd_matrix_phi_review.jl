using Test
using Tarang
using LinearAlgebra
using SparseArrays

@testset "ETD matrix phi functions preserve singular couplings" begin
    @testset "nilpotent Jordan blocks" begin
        for T in (Float64, ComplexF64), n in (2, 3), dt in (1e-10, 0.1, 2.0, 100.0)
            A = zeros(T, n, n)
            for i in 1:n-1
                A[i, i + 1] = 1
            end
            original = copy(A)
            z = dt .* A
            identity = Matrix{T}(I, n, n)
            # A^3 = 0, so these Taylor polynomials are exact for either size.
            expected = (identity + z + z*z/2,
                        identity + z/2 + z*z/6,
                        identity/2 + z/6 + z*z/24)
            actual = Tarang.phi_functions_matrix(A, dt)
            for k in 1:3
                @test actual[k] ≈ expected[k] rtol=1e-12 atol=1e-13
            end
            @test A == original
        end
    end

    @testset "near-singular and nonnormal matrices" begin
        # Independent block-exponential oracle: its top row of n×n blocks is
        # [exp(z), phi1(z), phi2(z)], with no division or eigenvector inverse.
        for A in ([1e-12 1.0; 0.0 -1e-12],
                  [-0.3 1.0; 0.0 -0.3],
                  [-30.0 100.0; 0.0 -30.0],
                  ComplexF64[-0.3+0.4im 1.0; 0.0 -0.3+0.4im])
            n = size(A, 1)
            z = 0.25 .* A
            block = zeros(eltype(z), 3n, 3n)
            block[1:n, 1:n] .= z
            for i in 1:2n
                block[i, i + n] = 1
            end
            oracle = exp(block)
            actual = Tarang.phi_functions_matrix(A, 0.25)
            for k in 1:3
                @test actual[k] ≈ oracle[1:n, (k-1)*n+1:k*n] rtol=1e-12 atol=1e-13
            end
        end
    end

    @testset "singular diffusion and dense-size safeguard" begin
        Q = [1.0 1.0 1.0; 1.0 -1.0 0.0; 1.0 1.0 -2.0]
        Q = Matrix(qr(Q).Q)
        for eigenvalues in ([0.0, -2.0, -20.0], [0.0, -100.0, -1000.0]),
            basis in (Matrix{Float64}(I, 3, 3), Q)
            A = basis * Diagonal(eigenvalues) * basis'
            expected = (basis * Diagonal(exp.(eigenvalues)) * basis',
                        basis * Diagonal([iszero(z) ? 1.0 : expm1(z)/z for z in eigenvalues]) * basis',
                        basis * Diagonal([iszero(z) ? 0.5 : (expm1(z)-z)/z^2 for z in eigenvalues]) * basis')
            actual = Tarang.phi_functions_matrix(A, 1.0)
            for k in 1:3
                @test actual[k] ≈ expected[k] rtol=1e-11 atol=1e-12
            end
        end
        @test_throws ArgumentError Tarang.phi_functions_matrix(spzeros(4097, 4097), 0.1)
    end

    @testset "coupled ETD equations with time-dependent forcing" begin
        # u_t = v, v_t = t*cos(x), u(0)=0, v(0)=cos(x).
        # The linear operator has a nilpotent Jordan block in every
        # Fourier mode. Linear-in-time forcing also exercises phi1 and phi2.
        for timestepper in (ETD_RK222(), ETD_CNAB2(), ETD_SBDF2())
            domain = PeriodicDomain(8)
            u = ScalarField(domain, "u")
            v = ScalarField(domain, "v")
            set!(u, (x,) -> 0.0)
            set!(v, (x,) -> cos(x))
            problem = InitialValueProblem([u, v])
            add_equation!(problem, "dt(u) - v = 0")
            add_equation!(problem, "dt(v) = 0")
            forcing = DeterministicForcing((x, t, p) -> t .* cos.(x), (8,))
            add_stochastic_forcing!(problem, :v, forcing)
            solver = InitialValueSolver(problem, timestepper; dt=0.1)
            for dt in (0.1, 0.05, 0.15)
                step!(solver, dt)
            end
            t = solver.sim_time
            x = collect(0:7) .* (2π/8)
            @test u["g"] ≈ (t + t^3/6) .* cos.(x) atol=1e-11
            @test v["g"] ≈ (1 + t^2/2) .* cos.(x) atol=1e-11
            @test haskey(solver.timestepper_state.timestepper_data, :etd_phi)
        end
    end
end
