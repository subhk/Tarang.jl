"""
A field handed back by `grad` must not be a buffer the derivative pool can reissue.

THE BUG THIS PINS. `evaluate_differentiate` returns memory borrowed from a global
rotating pool (`_DERIV_RESULT_POOL`, `_DERIV_RESULT_IDX`), which hands the same
slot to the next caller after `_DERIV_RESULT_POOL_SIZE` checkouts. `grad` stored
those buffers directly into the `VectorField` / `TensorField` it returned, so the
container went on referencing pool slots after the pool had moved on.

A 3-D vector gradient is a 3x3 Jacobian: 9 slots held live. Two live vector
gradients need 18, against a pool of 16. So

    gu = grad(u); gv = grad(v)      # both live

silently rewrote two components of `gu` with values belonging to `gv` — measured
max error 3.0 on `T[1,2]`, with no error raised and no warning. That is this
project's dominant failure shape: not a crash, a plausible wrong number.

The pool size had already been raised 8 -> 16 after exactly this failure inside a
single tensor (`T[3,3]` overwriting `T[1,1]`). Raising it again would only move
the threshold, because nothing bounds how many results a caller may hold live.
The fix takes ownership instead: `_own_borrowed_field` copies out of the pool, so
correctness no longer depends on the pool size at all.

These tests therefore check both halves — that results are independent objects
AND that their VALUES are right, since an aliasing bug is only visible in values.
"""

using Test
using Tarang

function _pool_test_domain(N = 8)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    bases = ntuple(i -> RealFourier(coords[["x", "y", "z"][i]]; size = N, bounds = (0.0, 2π)), 3)
    return coords, dist, bases
end

"""u = (sin x, sin 2y, sin 3z); Jacobian is diagonal with (cos x, 2cos 2y, 3cos 3z)."""
function _pool_test_vector(dist, coords, bases, name, phase)
    u = VectorField(dist, coords, name, bases, Float64)
    if phase === :sin
        set!(u.components[1], (x, y, z) -> sin(x))
        set!(u.components[2], (x, y, z) -> sin(2y))
        set!(u.components[3], (x, y, z) -> sin(3z))
    else
        set!(u.components[1], (x, y, z) -> cos(x))
        set!(u.components[2], (x, y, z) -> cos(2y))
        set!(u.components[3], (x, y, z) -> cos(3z))
    end
    return u
end

@testset "two live vector gradients do not share pool slots" begin
    coords, dist, bases = _pool_test_domain()
    u = _pool_test_vector(dist, coords, bases, "u", :sin)
    v = _pool_test_vector(dist, coords, bases, "v", :cos)

    # grad(u) computed with nothing else live — the uncontended reference.
    alone = Tarang.evaluate_gradient(Tarang.Gradient(u, coords), :g)
    reference = Array{Array{Float64, 3}}(undef, 3, 3)
    for i in 1:3, j in 1:3
        reference[i, j] = copy(Array(get_grid_data(alone.components[i, j])))
    end

    # Now hold two gradients live at once: 18 components against a 16-slot pool.
    gu = Tarang.evaluate_gradient(Tarang.Gradient(u, coords), :g)
    gv = Tarang.evaluate_gradient(Tarang.Gradient(v, coords), :g)

    # Every component of both tensors must be its own field.
    ids = UInt64[]
    for i in 1:3, j in 1:3
        push!(ids, objectid(gu.components[i, j]))
        push!(ids, objectid(gv.components[i, j]))
    end
    @test length(unique(ids)) == 18

    # And grad(u) must be unchanged by grad(v) existing. This is the assertion
    # that actually failed before the fix (max error 3.0); the identity check
    # above would not have caught a copy that shared underlying storage.
    for i in 1:3, j in 1:3
        got = Array(get_grid_data(gu.components[i, j]))
        @test maximum(abs, got .- reference[i, j]) < 1e-12
    end
end

@testset "vector gradient values are correct with contention" begin
    coords, dist, bases = _pool_test_domain()
    u = _pool_test_vector(dist, coords, bases, "u", :sin)
    v = _pool_test_vector(dist, coords, bases, "v", :cos)

    gu = Tarang.evaluate_gradient(Tarang.Gradient(u, coords), :g)
    gv = Tarang.evaluate_gradient(Tarang.Gradient(v, coords), :g)

    N = 8
    g = [2π * (k - 1) / N for k in 1:N]
    # T[i,j] = d u_j / d x_i, so the Jacobian of (sin x, sin 2y, sin 3z) is
    # diagonal. A pool collision showed up precisely as a nonzero off-diagonal.
    expected_u = Dict((1, 1) => [cos(x)      for x in g, _ in g, _ in g],
                      (2, 2) => [2cos(2y)    for _ in g, y in g, _ in g],
                      (3, 3) => [3cos(3z)    for _ in g, _ in g, z in g])
    expected_v = Dict((1, 1) => [-sin(x)     for x in g, _ in g, _ in g],
                      (2, 2) => [-2sin(2y)   for _ in g, y in g, _ in g],
                      (3, 3) => [-3sin(3z)   for _ in g, _ in g, z in g])

    for i in 1:3, j in 1:3
        gu_ij = Array(get_grid_data(gu.components[i, j]))
        gv_ij = Array(get_grid_data(gv.components[i, j]))
        @test maximum(abs, gu_ij .- get(expected_u, (i, j), zeros(N, N, N))) < 1e-10
        @test maximum(abs, gv_ij .- get(expected_v, (i, j), zeros(N, N, N))) < 1e-10
    end
end

@testset "many live scalar gradients stay independent" begin
    coords, dist, bases = _pool_test_domain()

    # 10 scalar gradients = 30 components, comfortably past the pool size.
    # The fields differ by AMPLITUDE, not wavenumber: on an N=8 grid anything
    # above the Nyquist mode (4) is unrepresentable, so `sin(k*x)` for k>=4 would
    # fail on spectral aliasing and say nothing about buffer aliasing.
    grads = map(1:10) do k
        f = ScalarField(dist, "f$k", bases, Float64)
        set!(f, (x, y, z) -> k * sin(x))
        (k, Tarang.evaluate_gradient(Tarang.Gradient(f, coords), :g))
    end

    ids = UInt64[]
    for (_, gf) in grads, i in 1:3
        push!(ids, objectid(gf.components[i]))
    end
    @test length(unique(ids)) == 30

    N = 8
    g = [2π * (m - 1) / N for m in 1:N]
    for (k, gf) in grads
        dx = Array(get_grid_data(gf.components[1]))
        @test maximum(abs, dx .- [k * cos(x) for x in g, _ in g, _ in g]) < 1e-9
        # The other two directions are identically zero for f = k sin(x); a
        # collision with another gradient in the batch would show up here.
        @test maximum(abs, Array(get_grid_data(gf.components[2]))) < 1e-9
        @test maximum(abs, Array(get_grid_data(gf.components[3]))) < 1e-9
    end
end

@testset "the pool is still a pool" begin
    # Ownership is taken by `grad`, not by removing pooling: a bare
    # `evaluate_differentiate` still returns borrowed memory, which is what makes
    # the transient single-derivative case allocation-free. If this ever stops
    # being true the ownership copies above become dead weight and should go.
    coords, dist, bases = _pool_test_domain()
    f = ScalarField(dist, "f", bases, Float64)
    set!(f, (x, y, z) -> sin(x))
    coord = coords["x"]

    seen = UInt64[]
    for _ in 1:(Tarang._DERIV_RESULT_POOL_SIZE + 1)
        d = Tarang.evaluate_differentiate(Tarang.Differentiate(f, coord, 1), :g)
        push!(seen, objectid(d))
    end
    @test length(unique(seen)) <= Tarang._DERIV_RESULT_POOL_SIZE
end
