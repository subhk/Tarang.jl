# Guard: the distributed diagonal-IMEX steppers must not leave the shared lazy-RHS
# output buffer in :c layout.
#
# evaluate_rhs returns `solver.rhs_plan.output_fields` — a SHARED, REUSED buffer,
# not a copy. The distributed diagonal steppers need the RHS in coefficient space,
# and used to get it by calling ensure_layout!(buffer, :c) directly on that buffer.
# That leaves the buffer :c-flagged. The next evaluate_rhs writes it through
# ensure_layout!(out, :g) (lazy_rhs.jl), which then pays a full distributed
# backward_transform! (a PencilFFTs ldiv! = MPI all-to-all) of stale coefficients
# that are immediately overwritten. Pure waste, and it scales with rank count.
#
# The fix transforms the buffer but restores its :g flag once its coefficients have
# been consumed: forward_transform! reads the grid array and writes the coeff array
# (PencilFFTs mul! is out-of-place), so the grid data is still valid and the :g flag
# is honest. The first testset pins that precondition — if PencilFFTs ever starts
# destroying its input, the fix is unsound and this test says so.
#
# This is alloc-invisible (ldiv! is in-place), so the O(1)-alloc guard in
# test_mpi_diagonal_imex_alloc.jl cannot catch it. Hence a separate layout guard.
using Test
using Tarang
import MPI
MPI.Initialized() || MPI.Init()
const NP = MPI.Comm_size(MPI.COMM_WORLD)

_raw(f) = (d = get_grid_data(f); d isa Tarang.PencilArrays.PencilArray ? parent(d) : d)
_mkdist(coords) = NP > 1 ? Distributor(coords; mesh=(NP,), dtype=Float64, architecture=CPU()) :
                           Distributor(coords; dtype=Float64, architecture=CPU())

function _ic!(f, N, fn)
    ensure_layout!(f, :g)
    dd = get_grid_data(f)
    ax = dd isa Tarang.PencilArrays.PencilArray ? Tarang.PencilArrays.pencil(dd).axes_local : (1:N, 1:N)
    xs = [2π*(i-1)/N for i in ax[1]]; ys = [2π*(j-1)/N for j in ax[2]]
    _raw(f) .= fn.(xs, ys')
    ensure_layout!(f, :c)
end

# Two-field coupled system: the cross-field advection is non-diagonal in a pure-Fourier
# basis, so it lands in the explicit RHS (F != 0) and the RHS buffer is genuinely used.
function _build(stepper; N=32, dt=1e-3)
    coords = CartesianCoordinates("x", "y"); dist = _mkdist(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    u = ScalarField(domain, "u"); v = ScalarField(domain, "v")
    problem = IVP([u, v]); add_parameters!(problem, nu=0.03, c=0.7)
    add_equation!(problem, "dt(u) - nu*lap(u) = -c*d(v,x)")
    add_equation!(problem, "dt(v) - nu*lap(v) = c*d(u,x)")
    solver = InitialValueSolver(problem, stepper; dt=dt)
    _ic!(u, N, (x,y)->sin(x)+0.4cos(2x+y)); _ic!(v, N, (x,y)->cos(x-2y)+0.3sin(3x))
    solver
end

# Mixed Chebyshev-Fourier: routed to the SUBPROBLEM steppers (coupled axis ⇒ tau solve),
# not the diagonal ones. The explicit term differentiates along the FOURIER axis — a
# distributed derivative along the Chebyshev axis is unsupported and now errors loudly.
function _build_cheb(stepper; Nz=12, Nx=8, dt=1e-3)
    coords = CartesianCoordinates("z", "x"); dist = _mkdist(coords)
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    domain = Domain(dist, (zb, xb))
    b = ScalarField(domain, "b")
    tau_b1 = ScalarField(dist, "tau_b1", (), Float64)
    tau_b2 = ScalarField(dist, "tau_b2", (), Float64)
    ex, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zb, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * τ_lift(tau_b1)
    problem = IVP([b, tau_b1, tau_b2])
    add_parameters!(problem, kappa=0.1, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
    add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, stepper; dt=dt)
    zf = [0.5*(1-cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
    xf = [2π*(i-1)/Nx for i in 1:Nx]
    g = [sin(π*zf[iz])*(1 + 0.5cos(2*xf[ix])) for iz in 1:Nz, ix in 1:Nx]
    ensure_layout!(b, :g)
    d = get_grid_data(b)
    if d isa Tarang.PencilArrays.PencilArray
        parent(d) .= g[Tarang.PencilArrays.pencil(d).axes_local...]
    else
        d .= g
    end
    ensure_layout!(b, :c)
    solver
end

@testset "distributed diagonal-IMEX: shared RHS buffer left at :g (NP=$NP)" begin
    # Precondition the fix rests on: a forward transform does not destroy the grid data,
    # so re-flagging the buffer :g after reading its coefficients is honest.
    @testset "forward_transform! preserves the grid array" begin
        solver = _build(RK222())
        f = solver.rhs_plan.output_fields[1]
        ensure_layout!(f, :g)
        before = copy(_raw(f))
        forward_transform!(f, :c)
        f.current_layout = :g
        @test _raw(f) == before
    end

    # The release may ONLY touch the compiled plan's own buffer. On the interpreted
    # fallback, evaluate_rhs returns a fresh vector whose entries can be the PROBLEM'S
    # fields by identity (evaluate_solver_expression returns a field-valued expression
    # as-is, so `dt(u) = f` yields F[i] === f). Re-flagging those would mutate a field
    # this stepper does not own — and would silently invalidate a source term whose
    # coefficients are updated between steps.
    @testset "release is gated on buffer ownership" begin
        solver = _build(RK222())
        buf = solver.rhs_plan.output_fields
        @test solver.rhs_plan.is_compiled

        # A field the stepper does not own: never re-flagged, even when :c.
        foreign = solver.state[1]
        ensure_layout!(foreign, :c)
        Tarang._release_rhs_buffer!([foreign], solver)
        @test foreign.current_layout === :c

        # The plan's own buffer: released.
        for f in buf; ensure_layout!(f, :c); end
        Tarang._release_rhs_buffer!(buf, solver)
        @test all(f -> f.current_layout === :g, buf)
    end

    if NP > 1
        # The distributed diagonal path is only taken at nprocs > 1.
        for stepper in (RK222(), SBDF2(), ETD_RK222())
            name = string(nameof(typeof(stepper)))
            @testset "$name leaves no :c-flagged RHS buffer" begin
                solver = _build(stepper)
                for _ in 1:4; step!(solver, 1e-3); end   # past SBDF2 startup
                layouts = [f.current_layout for f in solver.rhs_plan.output_fields]
                @test all(==(:g), layouts)
            end
        end

        # The SUBPROBLEM steppers (mixed Chebyshev-Fourier — the main distributed
        # workload) have the same shared buffer and the same hazard: to_solve_layout!
        # leaves it :c, so the next stage's evaluate_rhs_buffered backward-transforms
        # coefficients it is about to overwrite. RK222 → step_subproblem_rk.jl,
        # SBDF2 → step_subproblem_multistep.jl.
        for stepper in (RK222(), SBDF2())
            name = string(nameof(typeof(stepper)))
            @testset "subproblem $name leaves no :c-flagged RHS buffer" begin
                solver = _build_cheb(stepper)
                for _ in 1:4; step!(solver, 1e-3); end
                written = [f.current_layout for (i, f) in enumerate(solver.rhs_plan.output_fields)
                           if solver.rhs_plan.exprs[i] !== nothing]
                @test !isempty(written)
                @test all(==(:g), written)
            end
        end
    end
end
