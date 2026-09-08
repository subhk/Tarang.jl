using Test
using Tarang

function _doc_fields(; N=16)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xb,))
    return ScalarField(domain, "u"), ScalarField(domain, "v")
end

@testset "Diagonal implicit operator composition" begin
    @testset "nested Laplacian dissipates the represented Fourier mode" begin
        for scheme in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
            u, _ = _doc_fields()
            set!(u, (x,) -> cos(2x))
            problem = InitialValueProblem([u])
            add_equation!(problem, "dt(u) + lap(lap(u)) = 0")
            solver = InitialValueSolver(problem, scheme; dt=0.001)
            for _ in 1:100
                step!(solver, 0.001)
            end
            x = 2π .* (0:15) ./ 16
            expected = exp(-16 * 0.1) .* cos.(2 .* x)
            @test maximum(abs, Array(u["g"]) .- expected) < 1e-4
        end
    end

    @testset "cross-field diffusion is refused before a step advances" begin
        for scheme in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2()),
            expression in ("lap(v)", "fraclap(v, 0.5)", "lap(u + v)")
            u, v = _doc_fields()
            fill!(u["g"], 0.0)
            set!(v, (x,) -> cos(x))
            problem = InitialValueProblem([u, v])
            add_equation!(problem, "dt(u) + $expression = 0")
            add_equation!(problem, "dt(v) = 0")
            solver = InitialValueSolver(problem, scheme; dt=0.01)
            before = [copy(Array(field["g"])) for field in (u, v)]
            @test_throws ArgumentError step!(solver, 0.01)
            @test solver.sim_time == 0.0
            @test solver.iteration == 0
            @test [Array(field["g"]) for field in (u, v)] == before
        end
    end

    @testset "scalar, signed, and fractional operands compose" begin
        u, v = _doc_fields()
        namespace = Dict{String, Any}("u" => u, "v" => v, "x" => u.dist.coordsys["x"])
        k = Float64.(0:8)
        cases = (
            ("lap(lap(u))", k.^4),
            ("lap(-2*u)", 2 .* k.^2),
            ("lap(2*(u + u))", -4 .* k.^2),
            ("lap(u - 2*u)", k.^2),
            ("fraclap(fraclap(u, 0.5), 1.5)", k.^4),
            ("fraclap(lap(u), 0.5)", -k.^3),
            ("lap(fraclap(u, 0.5))", -k.^3),
            ("fraclap(-2*u, 0.5)", -2 .* k),
            ("fraclap(fraclap(u, -0.5), 0.5)", Float64.(k .> 0)),
        )
        for (expression, expected) in cases
            op = Tarang.parse_expression(expression, namespace)
            actual = Tarang._diagonal_Lhat_from_expr(op, u)
            @test actual !== nothing
            actual === nothing || @test Array(actual) ≈ expected
        end

        for expression in ("lap(v)", "fraclap(v, 0.5)", "lap(u + v)",
                           "lap(u*v)", "lap(d(v, x))")
            op = Tarang.parse_expression(expression, namespace)
            @test Tarang._diagonal_Lhat_from_expr(op, u) === nothing
        end
    end
end
