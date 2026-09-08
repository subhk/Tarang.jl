using Test, Tarang, MPI, PencilArrays
MPI.Initialized() || MPI.Init()

const TMGM_COMM = MPI.COMM_WORLD
const TMGM_NP = MPI.Comm_size(TMGM_COMM)

function tmgm_solver(ts, equation)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, device=CPU())
    bases = (RealFourier(coords["x"]; size=8, bounds=(0.0, 2π)),
             RealFourier(coords["y"]; size=8, bounds=(0.0, 2π)))
    u = ScalarField(Domain(dist, bases), "u")
    fill!(grid_data!(u), 1.0)
    problem = InitialValueProblem([u])
    add_equation!(problem, equation)
    return InitialValueSolver(problem, ts; dt=0.01), u
end

@testset "MPI timestepper mass operators are never silently discarded" begin
    if TMGM_NP < 2
        @test_skip "Requires at least two MPI ranks"
    else
        steppers = (RK111(), RK222(), RK443(), RKSMR(), Tarang.RKGFY(), Tarang.RK443_IMEX(),
                    CNAB1(), CNAB2(), SBDF1(), SBDF2(), SBDF3(), SBDF4(),
                    ETD_RK222(), ETD_CNAB2(), ETD_SBDF2(), Tarang.MCNAB2(), Tarang.CNLF2(),
                    DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
        for ts in steppers
            @testset "$(nameof(typeof(ts)))" begin
                solver, u = tmgm_solver(ts, "2*dt(u) = -u")
                before = copy(parent(grid_data!(u)))
                err = try
                    step!(solver)
                    nothing
                catch ex
                    ex
                end
                @test err isa ArgumentError
                @test occursin("non-identity mass", err === nothing ? "" : sprint(showerror, err))
                @test parent(grid_data!(u)) == before
                @test solver.sim_time == 0
                @test solver.iteration == 0

                if Tarang.supports_distributed_diagonal_imex(ts)
                    implicit_solver, implicit_u = tmgm_solver(ts, "2*dt(u) + u = 0")
                    @test_throws ArgumentError step!(implicit_solver)
                    @test all(isone, parent(grid_data!(implicit_u)))
                end
            end
        end
    end
end

MPI.Barrier(TMGM_COMM)
MPI.Finalized() || MPI.Finalize()
