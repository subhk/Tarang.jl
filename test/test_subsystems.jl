using Test
using Tarang

@testset "Subsystems" begin
    @testset "Group Normalization and Coefficient Shapes" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=10, bounds=(0.0, 2π))

        field = ScalarField(dist, "u", (xb, yb), Float64)
        problem = IVP([field])

        struct DummyBase
            matrix_coupling::Vector{Bool}
        end

        struct DummySolver
            problem::Problem
            base::DummyBase
        end

        solver = DummySolver(problem, DummyBase(fill(true, dist.dim)))
        subsystem = Subsystem(solver)

        @test subsystem.group == (nothing, nothing)

        expected_shape = (div(xb.meta.size, 2) + 1, div(yb.meta.size, 2) + 1)
        @test coeff_shape(subsystem, field.domain) == expected_shape
        @test coeff_size(subsystem, field.domain) == prod(expected_shape)
    end

    @testset "Per-subproblem field sizing" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 4.0))
        zb = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
        domain = Domain(dist, (xb, zb))

        # 2D scalar field on full domain
        b = ScalarField(domain, "b")
        # 2D vector field on full domain (2 components for 2D coords)
        u = VectorField(domain, "u")
        # 1D tau field — Fourier only, no Chebyshev basis
        tau_b = ScalarField(dist, "tau_b", (xb,), Float64)
        # 0D tau field — no bases at all
        tau_p = ScalarField(dist, "tau_p", (), Float64)

        problem = IVP([b, u, tau_b, tau_p])

        # Build a mock subproblem with group (5, nothing):
        #   Fourier mode 5 (separable), Chebyshev fully coupled
        struct MockSolverBase
            matrix_coupling::Vector{Bool}
        end
        struct MockSolver
            problem::Problem
            base::MockSolverBase
        end
        # Fourier separable (false), Chebyshev coupled (true)
        solver = MockSolver(problem, MockSolverBase([false, true]))
        subsys = Subsystem(solver, (5, nothing))
        sp = Subproblem(solver, (subsys,), (5, nothing))

        # 2D scalar: separable Fourier (1) * coupled Chebyshev (Nz=8) = 8
        @test subproblem_field_size(sp, b) == 8
        # 2D vector (2 components in 2D): 2 * Nz = 16
        @test subproblem_field_size(sp, u) == 16
        # 1D tau (Fourier only, separable): 1 DOF per mode
        @test subproblem_field_size(sp, tau_b) == 1
        # 0D tau (no bases): 1
        @test subproblem_field_size(sp, tau_p) == 1
    end
end
