using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang

function _mdoc_fields(; comm=MPI.COMM_WORLD)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, comm=comm)
    bases = Tuple(RealFourier(coords[name]; size=8, bounds=(0.0, 2π)) for name in ("x", "y"))
    domain = Domain(dist, bases)
    u, v = ScalarField(domain, "u"), ScalarField(domain, "v")
    x = reshape(vec(Tarang.local_grid(bases[1], dist, 1)), :, 1)
    return u, v, x
end

@testset "MPI diagonal implicit operator composition" begin
    @testset "matrix-free eligibility stays scoped" begin
        u, _, _ = _mdoc_fields()
        @test !Tarang._mpi_matrix_free_fourier_state([u], SBDF3())
        coords = u.dist.coordsys
        cheb = ChebyshevT(coords["x"]; size=8, bounds=(0.0, 1.0))
        mixed = ScalarField(Domain(u.dist, (cheb, u.bases[2])), "mixed")
        @test !Tarang._mpi_matrix_free_fourier_state([u, mixed], RK222())
        tau = ScalarField(u.dist, "tau", (), Float64)
        @test !Tarang._mpi_matrix_free_fourier_state([u, tau], RK222())
    end

    @testset "matrix-free construction preserves serial results" begin
        for scheme in (RK222(), RK443(), SBDF2()), equation in (
            "dt(u) - lap(u)/8 = -u/4", "dt(u) = -u/4", "dt(u) = 0")
            results = Any[]
            for comm in (MPI.COMM_SELF, MPI.COMM_WORLD)
                u, _, x = _mdoc_fields(; comm)
                parent(u["g"]) .= cos.(2 .* x)
                problem = InitialValueProblem([u])
                add_equation!(problem, equation)
                solver = InitialValueSolver(problem, scheme; dt=1e-3)
                matrix_free = MPI.Comm_size(comm) > 1
                @test solver.execution_plan.assembled_global_matrices == !matrix_free
                @test !isempty(problem.equation_data)
                if matrix_free
                    @test Tarang._get_problem_matrix(problem, "L_matrix") === nothing
                    @test Tarang._get_problem_matrix(problem, "M_matrix") === nothing
                    @test Tarang.compiled_subproblems(problem) === nothing
                end
                for dt in (1e-3, 1.5e-3, 8e-4)
                    step!(solver, dt)
                end
                push!(results, Tarang.gather_array(u.dist, Tarang.grid_data!(u)))
            end
            @test results[1] ≈ results[2] rtol=1e-10 atol=1e-12
        end
    end

    @testset "public schemes support scalar division" begin
        for scheme in (RK222(), RK443(), SBDF2()), expression in ("u/2", "-lap(u)/8")
            u, _, x = _mdoc_fields()
            parent(u["g"]) .= cos.(2 .* x)
            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) + $expression = 0")
            solver = InitialValueSolver(problem, scheme; dt=1e-3)
            for _ in 1:3
                step!(solver, 1e-3)
            end
            @test maximum(abs, parent(u["g"]) .- exp(-0.5 * 3e-3) .* cos.(2 .* x)) < 1e-6
        end
    end

    @testset "MPI refuses attached operators instead of ignoring them" begin
        if MPI.Comm_size(MPI.COMM_WORLD) > 1
            for scheme in (RK222(), RK443(), SBDF2())
                u, _, x = _mdoc_fields()
                parent(u["g"]) .= cos.(2 .* x)
                problem = InitialValueProblem([u])
                add_equation!(problem, "dt(u) = 0")
                solver = InitialValueSolver(problem, scheme; dt=1e-3)
                operator = SpectralLinearOperator(u, :laplacian; ν=0.5)
                @test_throws ArgumentError set_spectral_linear_operator!(solver, operator)
                @test Tarang._get_spectral_linear_operator(solver) === nothing

                # Direct parameter registration must not bypass the step guard.
                problem.parameters["spectral_linear_operator"] = operator
                before = copy(parent(u["g"]))
                @test_throws ArgumentError step!(solver, 1e-3)
                @test solver.iteration == 0
                @test solver.sim_time == 0.0
                @test parent(u["g"]) == before
            end
        else
            @test_skip "requires multiple MPI ranks"
        end
    end

    diagonal_schemes = (Tarang.DiagonalIMEX_RK222(), Tarang.DiagonalIMEX_RK443(), Tarang.DiagonalIMEX_SBDF2())
    # Serial ETD uses the full matrix and legitimately supports cross-field L.
    # Its per-mode refusal applies only to the distributed ETD implementation.
    schemes = MPI.Comm_size(MPI.COMM_WORLD) > 1 ?
              (diagonal_schemes..., ETD_RK222(), ETD_CNAB2(), ETD_SBDF2()) : diagonal_schemes
    for scheme in schemes
        @testset "$(nameof(typeof(scheme)))" begin
            u, _, x = _mdoc_fields()
            parent(u["g"]) .= cos.(2 .* x)
            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) + lap(lap(u)) = 0")
            solver = InitialValueSolver(problem, scheme; dt=1e-4)
            step!(solver, 1e-4)
            # Even the SBDF1 startup error is <2e-6 at this step size; dropping
            # the inner Laplacian instead produces growth and an error >1e-3.
            @test maximum(abs, parent(u["g"]) .- exp(-16e-4) .* cos.(2 .* x)) < 2e-6

            for expression in ("lap(v)", "fraclap(v, 0.5)")
                u, v, x = _mdoc_fields()
                fill!(u["g"], 0.0)
                parent(v["g"]) .= cos.(x)
                problem = InitialValueProblem([u, v])
                add_equation!(problem, "dt(u) + $expression = 0")
                add_equation!(problem, "dt(v) = 0")
                solver = InitialValueSolver(problem, scheme; dt=0.01)
                err = try
                    step!(solver, 0.01)
                    nothing
                catch error
                    error
                end
                @test err isa Exception
                @test err === nothing || occursin("diagonal", sprint(showerror, err))
                @test solver.sim_time == 0.0
                @test solver.iteration == 0
                @test all(iszero, parent(u["g"]))
            end
        end
    end
end

MPI.Barrier(MPI.COMM_WORLD)
