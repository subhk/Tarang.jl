# Guard: the TRANSFORM BUDGET of a step.
#
# Spectral transforms are the dominant cost (each is an MPI all-to-all when distributed), and a
# redundant one is INVISIBLE to an allocation guard: PencilFFTs `mul!`/`ldiv!` are in-place, so a
# wasted distributed FFT allocates NOTHING. A real regression of exactly this kind shipped and
# survived an alloc guard; it was caught only by hand-counting. Hence this test.
#
# Two properties are pinned:
#   1. lap() is FUSED: a Laplacian costs ONE forward + ONE backward transform per RHS evaluation,
#      no matter how many dimensions. Expanding it into one LazyDiff per axis (the naive
#      translation) costs a full grid->coeff->grid round-trip PER AXIS.
#   2. Scratch fields are returned to the pool flagged :g. A :c-flagged scratch makes the next
#      borrower's `ensure_layout!(scratch, :g)` a full BACKWARD transform of coefficients it is
#      about to overwrite — which silently ate the entire saving of (1) when it regressed.
using Test
using Tarang

const N = 16

function _solver(rhs::String; stepper=RK222())
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (xb, yb))
    u = ScalarField(domain, "u")
    problem = IVP([u]); add_parameters!(problem, nu=0.01)
    add_equation!(problem, "dt(u) = $rhs")
    solver = InitialValueSolver(problem, stepper; dt=1e-4)
    ensure_layout!(u, :g)
    xs = [2π*(i-1)/N for i in 1:N]
    get_grid_data(u) .= [sin(x) + 0.3cos(2y) for x in xs, y in xs]
    ensure_layout!(u, :c)
    solver
end

"""Transforms consumed by one step, after warm-up."""
function _count(rhs::String)
    solver = _solver(rhs)
    for _ in 1:6; step!(solver, 1e-4); end
    Tarang.enable_transform_counts!(true)
    Tarang.reset_transform_counts!()
    step!(solver, 1e-4)
    c = Tarang.transform_counts()
    Tarang.enable_transform_counts!(false)
    @test solver.rhs_plan.is_compiled      # a fallback would make the count meaningless
    c.forward + c.backward
end

@testset "lazy RHS transform budget" begin
    one_deriv = _count("-u*d(u,x)")                 # 1 ∂ node — the unit of comparison
    lap       = _count("nu*lap(u)")                 # 2 ∂ nodes sharing one operand — FUSED

    # A 2-D Laplacian must not cost more than a single first derivative: both are one
    # operand, one forward, one backward. Unfused it was 6 transforms/step more.
    @test lap <= one_deriv

    # And it must not scale with dimension: the fused node applies both axes' multipliers
    # inside a single coefficient-space visit.
    @test lap < 20     # measured 14; the unfused translation was 23

    # Two derivatives of the same operand that are NOT summed (u*∂x(u) + u*∂y(u)) cannot be
    # fused into one output — they legitimately need one BACKWARD transform each. But they must
    # share the operand's coefficients, so the FORWARD count must not grow with them.
    two_indep = _count("-u*d(u,x) - u*d(u,y)")
    @test two_indep > lap
end

@testset "sibling ∂ nodes share one forward transform" begin
    # Every advection term is several derivatives of ONE operand. Each ∂ node used to
    # forward-transform that operand again, so the forward count grew with the number of nodes
    # (measured: 6 / 9 / 12 forwards for 1 / 2 / 3 nodes). The per-evaluation cache in
    # LazyWorkspace makes the operand's coefficients shared, so FORWARDS ARE NOW CONSTANT and
    # only the backwards (one per node, genuinely needed — each node produces a different field)
    # grow.
    #
    # Zero-allocation, value-identical saving ⇒ invisible to every guard except a count.
    fwd(rhs) = begin
        solver = _solver(rhs)
        for _ in 1:6; step!(solver, 1e-4); end
        Tarang.enable_transform_counts!(true)
        Tarang.reset_transform_counts!()
        step!(solver, 1e-4)
        c = Tarang.transform_counts()
        Tarang.enable_transform_counts!(false)
        @test solver.rhs_plan.is_compiled
        c.forward
    end

    f1 = fwd("-u*d(u,x)")
    f2 = fwd("-u*d(u,x) - u*d(u,y)")           # 2 ∂ nodes, same operand
    f3 = fwd("-u*d(u,x) - u*d(u,y) - u*d(u,x)") # 3 ∂ nodes, same operand

    @test f2 == f1      # was f1 + 3
    @test f3 == f1      # was f1 + 6
end
