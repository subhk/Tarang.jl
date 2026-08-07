"""
A field handed back by `u * v` must not be a buffer the nonlinear pool can reissue.

THE BUG THIS PINS. `evaluate_transform_multiply` writes its dealiased result into
`_NL_RESULT_POOL`, a rotating pool of `_NL_RESULT_POOL_SIZE` (8) buffers. The
index `_NL_RESULT_IDX` is global and the pool hangs off the `Distributor`'s
`NonlinearEvaluator`, so every caller on a domain shares one rotation — including
the solver RHS.

`Base.:*(::ScalarField, ::ScalarField)` returned that buffer straight to user
code. Holding nine products silently rewrote the first: measured max error 99 on
an N=128 Fourier grid, no error and no warning. Because the index is shared with
the RHS, a user holding ONE product also lost it after eight internal products —
so this reached ordinary `u*v; step!(solver)` code, not just contrived loops.

This is the same failure family as `test_deriv_pool_ownership.jl`: a rotating
pool plus a caller that RETAINS the result. Growing the pool is not a fix, since
nothing bounds how many products a caller may hold. `own=true` (the default on
`evaluate_transform_multiply`) copies out via `_own_borrowed_field`; `own=false`
is the explicit opt-in for callers that consume the result immediately, which is
what keeps the hot RHS path allocation-free.

The tests check identity AND value: aliasing is only visible in values.
"""

using Test
using Tarang

# N=32 keeps prod(size) well past the 64-element gate in `Base.:*` while staying
# cheap; a single Fourier axis is enough to exercise the padded-dealias path.
function _nl_pool_domain(N = 32)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    bases = (RealFourier(coords["x"]; size = N, bounds = (0.0, 2π)),
             RealFourier(coords["y"]; size = N, bounds = (0.0, 2π)))
    return coords, dist, bases
end

function _nl_pool_field(dist, bases, name, amp)
    f = ScalarField(dist, name, bases, Float64)
    set!(f, (x, y) -> amp * sin(x))
    return f
end

@testset "many live products from Base.:* stay independent" begin
    _, dist, bases = _nl_pool_domain()
    npool = Tarang._NL_RESULT_POOL_SIZE

    # Amplitudes, not wavenumbers: on this grid a high-k factor would fail on
    # spectral resolution and say nothing about buffer aliasing.
    fields = [_nl_pool_field(dist, bases, "f$k", Float64(k)) for k in 1:(npool + 4)]

    products = [fields[k] * fields[k] for k in eachindex(fields)]
    for p in products
        ensure_layout!(p, :g)
    end

    @test length(unique(objectid.(products))) == length(products)

    # (k sin x)^2 = k^2 (1 - cos 2x)/2. Both k=0 and k=2 are resolved at N=32 and
    # survive 3/2-rule dealiasing, so the product is exact to roundoff. A pool
    # collision showed up as product k carrying amplitude of a later k.
    N = 32
    g = [2π * (m - 1) / N for m in 1:N]
    for (k, p) in enumerate(products)
        got = Array(get_grid_data(p))
        want = [k^2 * sin(x)^2 for x in g, _ in g]
        @test maximum(abs, got .- want) < 1e-10
    end
end

@testset "a held product survives later products" begin
    _, dist, bases = _nl_pool_domain()
    npool = Tarang._NL_RESULT_POOL_SIZE

    u = _nl_pool_field(dist, bases, "u", 1.0)
    held = u * u
    ensure_layout!(held, :g)
    snapshot = copy(Array(get_grid_data(held)))

    # Churn the pool well past a full rotation without touching `held`.
    for k in 1:(2 * npool)
        w = _nl_pool_field(dist, bases, "w$k", Float64(k))
        p = w * w
        ensure_layout!(p, :g)
    end

    ensure_layout!(held, :g)
    @test maximum(abs, Array(get_grid_data(held)) .- snapshot) < 1e-12
end

@testset "evaluate_transform_multiply owns its result by default" begin
    _, dist, bases = _nl_pool_domain()
    ev = Tarang._get_evaluator(dist)
    npool = Tarang._NL_RESULT_POOL_SIZE

    u = _nl_pool_field(dist, bases, "u", 1.0)
    v = _nl_pool_field(dist, bases, "v", 2.0)

    owned = [Tarang.evaluate_transform_multiply(u, v, ev) for _ in 1:(npool + 2)]
    @test length(unique(objectid.(owned))) == npool + 2

    # Defaulting to owned is the point: a caller that forgets the keyword gets a
    # slower correct answer, never a silently wrong one.
    first_vals = copy(Array(get_grid_data(owned[1])))
    for _ in 1:(2 * npool)
        Tarang.evaluate_transform_multiply(u, v, ev; own = false)
    end
    @test maximum(abs, Array(get_grid_data(owned[1])) .- first_vals) < 1e-12
end

@testset "own=false still borrows from the pool" begin
    # Ownership is taken at the boundary, not by removing pooling. If borrowing
    # ever stops recycling, the hot-path `own=false` annotations become dead
    # weight and should go.
    _, dist, bases = _nl_pool_domain()
    ev = Tarang._get_evaluator(dist)
    npool = Tarang._NL_RESULT_POOL_SIZE

    u = _nl_pool_field(dist, bases, "u", 1.0)
    v = _nl_pool_field(dist, bases, "v", 2.0)

    seen = UInt64[]
    for _ in 1:(npool + 2)
        push!(seen, objectid(Tarang.evaluate_transform_multiply(u, v, ev; own = false)))
    end
    @test length(unique(seen)) <= npool
end

@testset "dot and cross results are owned" begin
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    N = 16
    bases = ntuple(i -> RealFourier(coords[["x", "y", "z"][i]]; size = N, bounds = (0.0, 2π)), 3)

    mkvec = (name, amp) -> begin
        w = VectorField(dist, coords, name, bases, Float64)
        set!(w.components[1], (x, y, z) -> amp * sin(x))
        set!(w.components[2], (x, y, z) -> amp * sin(y))
        set!(w.components[3], (x, y, z) -> amp * sin(z))
        w
    end

    # Hold several cross products at once: each holds 3 components and each
    # component consumed 2 pooled products while being built.
    crosses = [Tarang.evaluate_vector_cross_product(mkvec("a$k", Float64(k)),
                                                    mkvec("b$k", 1.0)) for k in 1:4]
    ids = UInt64[]
    for c in crosses, i in 1:3
        push!(ids, objectid(c.components[i]))
    end
    @test length(unique(ids)) == 12

    dots = [Tarang.evaluate_vector_dot_product(mkvec("c$k", Float64(k)), mkvec("d$k", 1.0))
            for k in 1:4]
    @test length(unique(objectid.(dots))) == 4
end

@testset "_own_borrowed_field preserves layout" begin
    _, dist, bases = _nl_pool_domain(8)
    f = ScalarField(dist, "f", bases, Float64)
    set!(f, (x, y) -> sin(x))

    ensure_layout!(f, :c)
    owned_c = Tarang._own_borrowed_field(f)
    @test owned_c.current_layout === :c
    @test maximum(abs, Array(get_coeff_data(owned_c)) .- Array(get_coeff_data(f))) < 1e-12

    ensure_layout!(f, :g)
    owned_g = Tarang._own_borrowed_field(f)
    @test owned_g.current_layout === :g
    @test maximum(abs, Array(get_grid_data(owned_g)) .- Array(get_grid_data(f))) < 1e-12

    # A copy, not a view.
    get_grid_data(owned_g)[1] += 1.0
    @test Array(get_grid_data(owned_g))[1] != Array(get_grid_data(f))[1]
end
