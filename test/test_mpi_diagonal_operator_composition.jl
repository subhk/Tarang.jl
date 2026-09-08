using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang

function _mdoc_fields()
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64)
    bases = Tuple(RealFourier(coords[name]; size=8, bounds=(0.0, 2π)) for name in ("x", "y"))
    domain = Domain(dist, bases)
    u, v = ScalarField(domain, "u"), ScalarField(domain, "v")
    x = reshape(vec(Tarang.local_grid(bases[1], dist, 1)), :, 1)
    return u, v, x
end

@testset "MPI diagonal implicit operator composition" begin
    diagonal_schemes = (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
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
