# Guard: a distributed explicit RHS written with lap()/div() must compile AND must
# integrate to the same answer as serial.
#
# Before the vector/tensor operators were added to translate_to_lazy, `dt(q) = nu*lap(q)`
# could not compile, so the solver dropped to the interpreted RHS evaluator. Under MPI
# that path evaluated to ZERO: the field was completely FROZEN — 300 steps of strong
# diffusion left sumsq exactly at its initial value at np=2 and np=4, while np=1 decayed
# correctly. A silent wrong answer, not just a slow one. This pins it.
#
# Reference is the serial (np=1) result of the same problem, which was correct all along.
using Test
using Tarang
import MPI
MPI.Initialized() || MPI.Init()
const NP = MPI.Comm_size(MPI.COMM_WORLD)
const COMM = MPI.COMM_WORLD

_raw(f) = (d = get_grid_data(f); d isa Tarang.PencilArrays.PencilArray ? parent(d) : d)

# Serial reference: 300 steps of dt(q) = 0.5*lap(q) from the IC below, N=64, dt=1e-3.
const SUMSQ_IC  = 2560.0
const SUMSQ_REF = 482.4622053646085

function _run(rhs_str; N=64, steps=300, dt=1e-3)
    coords = CartesianCoordinates("x", "y")
    dist = NP > 1 ? Distributor(coords; mesh=(NP,), dtype=Float64, architecture=CPU()) :
                    Distributor(coords; dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    q = ScalarField(domain, "q")
    problem = IVP([q]); add_parameters!(problem, nu=0.5)
    add_equation!(problem, "dt(q) = $rhs_str")
    solver = InitialValueSolver(problem, RK222(); dt=dt)

    ensure_layout!(q, :g)
    dd = get_grid_data(q)
    ax = dd isa Tarang.PencilArrays.PencilArray ? Tarang.PencilArrays.pencil(dd).axes_local : (1:N, 1:N)
    xs = [2π*(i-1)/N for i in ax[1]]; ys = [2π*(j-1)/N for j in ax[2]]
    _raw(q) .= [sin(2x - y) + 0.5cos(x + 3y) for x in xs, y in ys]
    ensure_layout!(q, :c)

    ss0 = MPI.Allreduce(sum(abs2, _raw(q)), MPI.SUM, COMM)
    for _ in 1:steps; step!(solver, dt); end
    ensure_layout!(q, :g)
    ss1 = MPI.Allreduce(sum(abs2, _raw(q)), MPI.SUM, COMM)
    solver.rhs_plan.is_compiled, ss0, ss1
end

@testset "distributed lap/div on the explicit RHS (NP=$NP)" begin
    @testset "lap(q) compiles under MPI" begin
        compiled, ss0, ss1 = _run("nu*lap(q)")
        @test compiled
        @test isapprox(ss0, SUMSQ_IC; rtol=1e-12)      # IC is what we think it is
        @test !isapprox(ss1, ss0; rtol=1e-6)           # the field actually EVOLVED (was frozen)
        @test isapprox(ss1, SUMSQ_REF; rtol=1e-9)      # and matches the serial reference
    end

    @testset "div(grad(q)) compiles under MPI and matches lap(q)" begin
        compiled, _, ss1 = _run("nu*div(grad(q))")
        @test compiled
        @test isapprox(ss1, SUMSQ_REF; rtol=1e-9)
    end

    # A distributed CHEBYSHEV axis is the opposite case: the lazy derivative works in
    # COEFFICIENT space, where the Chebyshev axis is the DECOMPOSED one, so it cannot serve
    # this at all. The interpreted derivative works in GRID space, where that axis is LOCAL —
    # so it is correct, and the translator must DECLINE and leave it in place. Compiling it
    # (as the first version of the lap/div translation did) turned a correct run into a
    # hard error on step 1.
    @testset "lap() on a distributed Chebyshev axis declines to compile, stays correct" begin
        Nz, Nx = 12, 8
        coords = CartesianCoordinates("z", "x")
        dist = NP > 1 ? Distributor(coords; mesh=(NP,), dtype=Float64, architecture=CPU()) :
                        Distributor(coords; dtype=Float64, architecture=CPU())
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
        domain = Domain(dist, (zb, xb))
        q = ScalarField(domain, "q")

        zf = [0.5*(1-cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
        xf = [2π*(i-1)/Nx for i in 1:Nx]
        g0 = [sin(π*zf[i])*(1 + 0.5cos(2*xf[j])) for i in 1:Nz, j in 1:Nx]
        ge = [-π^2*sin(π*zf[i])*(1 + 0.5cos(2*xf[j])) - 2*sin(π*zf[i])*cos(2*xf[j])
              for i in 1:Nz, j in 1:Nx]
        ensure_layout!(q, :g)
        dd = get_grid_data(q)
        sl(g) = dd isa Tarang.PencilArrays.PencilArray ? g[Tarang.PencilArrays.pencil(dd).axes_local...] : g
        _raw(q) .= sl(g0)
        ensure_layout!(q, :c)

        problem = IVP([q]); add_parameters!(problem, nu=1.0)
        add_equation!(problem, "dt(q) = nu*lap(q)")
        solver = InitialValueSolver(problem, RK222(); dt=1e-4)
        if NP > 1
            @test !solver.rhs_plan.is_compiled     # declined — the lazy path cannot serve it
        else
            @test solver.rhs_plan.is_compiled      # serial: compiles, and is correct
        end

        F = Tarang.evaluate_rhs(solver, solver.state, 0.0)   # must NOT throw
        f = F[1]; ensure_layout!(f, :g)
        err = MPI.Allreduce(maximum(abs.(_raw(f) .- sl(ge))), MPI.MAX, COMM)
        @test err < 1e-4     # spectral truncation at Nz=12; identical at np=1/2/4
    end
end
