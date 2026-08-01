# Regression: explicit multistep on a distributed pure-Fourier problem.
#
# An MPI pure-Fourier solver builds no per-mode subproblems and cannot factorize
# the global matrix, so every multistep method used to reach
# `_global_multistep_distributed_fallback!` and THROW — even for a problem with
# no implicit linear operator at all, where there is nothing to drop. The
# matrix-free field path now serves those: it is the same code path a GPU run
# takes, exercised here with real transforms and real decomposition.
#
# SBDF2 keeps its existing distributed diagonal-IMEX path and is checked here
# only to prove this change left it alone.

using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NP = MPI.Comm_size(COMM)

if NP < 2
    RANK == 0 && @warn "MPI explicit-multistep field-path test needs at least two ranks"
    MPI.Finalize()
    exit(0)
end

const KAPPA = 0.02
const NX = 16
const NY = 16

# dt(u) = kappa*lap(u) with the Laplacian on the RIGHT-hand side: explicit, so
# there is no implicit operator to preserve. u0 = sin(x)cos(y) is a single
# eigenmode, hence spatially exact and decaying as exp(-2*kappa*t).
function _build(stepper, dt; equation="dt(u) = kappa*lap(u)")
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=NX, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=NY, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    u = ScalarField(domain, "u")
    problem = IVP([u])
    add_parameters!(problem, kappa=KAPPA)
    add_equation!(problem, equation)
    solver = InitialValueSolver(problem, stepper; dt)

    xs = [2π * (i - 1) / NX for i in 1:NX]
    ys = [2π * (j - 1) / NY for j in 1:NY]
    initial = [sin(x) * cos(y) for x in xs, y in ys]
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    ax = PencilArrays.pencil(gd).axes_local
    parent(gd) .= initial[ax...]
    ensure_layout!(u, :c)
    return solver, initial
end

function _decay_error(stepper, dt; tfinal=0.5)
    solver, initial = _build(stepper, dt)
    for _ in 1:round(Int, tfinal / dt)
        step!(solver, dt)
    end
    exact = initial .* exp(-2 * KAPPA * tfinal)
    u = solver.state[1]
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    ax = PencilArrays.pencil(gd).axes_local
    local_error = maximum(abs.(parent(gd) .- exact[ax...]))
    return MPI.Allreduce(local_error, MPI.MAX, COMM)
end

# The same problem on one rank (COMM_SELF), which assembles global matrices and
# therefore runs the reference global-matrix multistep. Returns the full grid.
function _serial_reference(stepper, dt; tfinal=0.5)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, architecture=CPU(), comm=MPI.COMM_SELF)
    xb = RealFourier(coords["x"]; size=NX, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=NY, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    u = ScalarField(domain, "u")
    set!(u, (x, y) -> sin(x) * cos(y))
    problem = IVP([u])
    add_parameters!(problem, kappa=KAPPA)
    add_equation!(problem, "dt(u) = kappa*lap(u)")
    solver = InitialValueSolver(problem, stepper; dt)
    for _ in 1:round(Int, tfinal / dt)
        step!(solver, dt)
    end
    v = solver.state[1]
    ensure_layout!(v, :g)
    return Array(get_grid_data(v))
end

function _distributed_vs_serial(stepper, dt; tfinal=0.5)
    solver, _ = _build(stepper, dt)
    for _ in 1:round(Int, tfinal / dt)
        step!(solver, dt)
    end
    reference = _serial_reference(stepper, dt; tfinal)
    u = solver.state[1]
    ensure_layout!(u, :g)
    gd = get_grid_data(u)
    ax = PencilArrays.pencil(gd).axes_local
    local_diff = maximum(abs.(parent(gd) .- reference[ax...]))
    return MPI.Allreduce(local_diff, MPI.MAX, COMM)
end

@testset "Distributed explicit multistep == serial reference (rank=$RANK)" begin
    # Before the field path these threw ArgumentError("... has no distributed
    # diagonal-IMEX implementation ...") on the first step. The serial run takes
    # the global-matrix path, so this pins the field path to the reference
    # implementation rather than to a convergence rate — on this single-eigenmode
    # problem SBDF3/SBDF4 hit roundoff before their asymptotic rate is visible.
    for (stepper, dt) in ((CNAB1(), 0.01), (CNAB2(), 0.01),
                          (SBDF1(), 0.01), (SBDF3(), 0.01), (SBDF4(), 0.01))
        name = string(nameof(typeof(stepper)))
        diff = _distributed_vs_serial(stepper, dt)
        RANK == 0 && @info "distributed vs serial" name diff
        @test diff < 1e-12
    end
end

@testset "Multistep beats first order — the collapse guard (rank=$RANK)" begin
    # The defect: every method degraded to forward Euler. With L = 0, CNAB1 IS
    # forward Euler, so it is the exact "collapsed" reference to beat.
    dt = 0.025
    e_euler = _decay_error(CNAB1(), dt)
    for (stepper, order) in ((CNAB2(), 2), (SBDF3(), 3), (SBDF4(), 4))
        name = string(nameof(typeof(stepper)))
        e = _decay_error(stepper, dt)
        RANK == 0 && @info "vs forward Euler" name e e_euler ratio=e_euler/e
        @test e < e_euler / 10
    end

    # CNAB2 is well conditioned at these step sizes, so its rate is measurable.
    rate = log2(_decay_error(CNAB2(), 0.025) / _decay_error(CNAB2(), 0.0125))
    RANK == 0 && @info "distributed CNAB2 convergence" rate
    @test rate > 1.8
end

@testset "The distributed path really is the field path (rank=$RANK)" begin
    solver, _ = _build(CNAB2(), 0.01)
    step!(solver, 0.01)
    state = solver.timestepper_state
    # History deques only exist if `_step_explicit_multistep_field!` ran.
    @test haskey(state.timestepper_data, :explicit_field_ms_X)
    @test haskey(state.timestepper_data, :explicit_field_ms_F)
end

@testset "MPI distributed diagonal-IMEX SBDF2 is unchanged (rank=$RANK)" begin
    e_coarse = _decay_error(SBDF2(), 0.02)
    e_fine = _decay_error(SBDF2(), 0.01)
    rate = log2(e_coarse / e_fine)
    RANK == 0 && @info "distributed SBDF2" e_coarse e_fine rate
    @test rate > 1.5
end

@testset "An implicit operator is still refused, never dropped (rank=$RANK)" begin
    # Same equation with the Laplacian on the LEFT: implicit. The field path must
    # decline so the existing loud refusal / diagonal-IMEX path keeps precedence.
    solver, _ = _build(CNAB2(), 0.01; equation="dt(u) - kappa*lap(u) = 0")
    Tarang._ensure_timestepper_state!(solver, 0.01)
    @test Tarang._problem_has_implicit_linear_term(solver)
    @test !Tarang._explicit_multistep_field_eligible(solver)
end

MPI.Barrier(COMM)
MPI.Finalized() || MPI.Finalize()
