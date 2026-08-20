"""
The state-arithmetic helpers must not force a layout.

`axpy_state!`, `add_scaled_state!`/`add_scaled_state` and
`linear_combination_state!` all compute a LINEAR COMBINATION, which commutes with
the spectral transform: it is equally valid in grid space or coefficient space so
long as every operand is in the SAME one. They used to call
`ensure_layout!(…, :g)` unconditionally, so handing them coefficient-space fields
paid a backward transform per operand on entry — and, since the RHS evaluators
want `:c`, a forward transform right back on the next call.

Two things are pinned here:

1. **No forced transforms.** Operands already in `:c` stay in `:c` and the call
   costs zero transforms (counted, not timed — `Tarang.transform_counts()`).
2. **Same answer either way.** A combination computed in coefficient space must
   agree with the same combination computed in grid space. This is the assertion
   that would catch `_arith_layout` returning `:c` for operands whose buffers do
   not actually line up, which is the only way the optimisation could go wrong.
"""

using Test
using Tarang

function _mk(name, coords, dist, bases)
    f = ScalarField(dist, name, bases)
    fill_random!(f)
    return f
end

@testset "state arithmetic follows the operands' layout" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
    xb = RealFourier(coords.coords[1], size=16, bounds=(0.0, 2pi))
    yb = ComplexFourier(coords.coords[2], size=8, bounds=(0.0, 2pi))
    bases = (xb, yb)

    @testset "coefficient operands cost no transforms" begin
        u = _mk("u", coords, dist, bases)
        v = _mk("v", coords, dist, bases)
        ensure_layout!(u, :c); ensure_layout!(v, :c)

        Tarang.enable_transform_counts!(true)
        Tarang.reset_transform_counts!()
        Tarang.axpy_state!(0.5, [u], [v])
        c = Tarang.transform_counts()
        Tarang.enable_transform_counts!(false)

        @test c.forward == 0
        @test c.backward == 0
        @test u.current_layout == :c
        @test v.current_layout == :c
    end

    @testset "grid operands still work in grid space" begin
        u = _mk("u", coords, dist, bases)
        v = _mk("v", coords, dist, bases)
        ensure_layout!(u, :g); ensure_layout!(v, :g)

        Tarang.enable_transform_counts!(true)
        Tarang.reset_transform_counts!()
        Tarang.axpy_state!(0.5, [u], [v])
        c = Tarang.transform_counts()
        Tarang.enable_transform_counts!(false)

        @test c.forward == 0
        @test c.backward == 0
        @test v.current_layout == :g
    end

    @testset "mixed layouts fall back to grid, not to a wrong answer" begin
        u = _mk("u", coords, dist, bases)
        v = _mk("v", coords, dist, bases)
        ensure_layout!(u, :c); ensure_layout!(v, :g)
        Tarang.axpy_state!(0.5, [u], [v])
        # `_arith_layout` refuses `:c` unless EVERY operand is already there.
        @test u.current_layout == :g
        @test v.current_layout == :g
    end

    @testset "coefficient-space result equals grid-space result" begin
        # Same inputs, same operation, two layouts. The transform is linear, so
        # the two must agree to roundoff.
        for op in (:axpy, :linear_combination, :add_scaled)
            u1 = _mk("u", coords, dist, bases); v1 = _mk("v", coords, dist, bases)
            u2 = copy(u1); v2 = copy(v1)

            ensure_layout!(u1, :c); ensure_layout!(v1, :c)
            ensure_layout!(u2, :g); ensure_layout!(v2, :g)

            if op === :axpy
                Tarang.axpy_state!(0.75, [u1], [v1])
                Tarang.axpy_state!(0.75, [u2], [v2])
                got, want = v1, v2
            elseif op === :linear_combination
                d1 = _mk("d", coords, dist, bases); d2 = copy(d1)
                ensure_layout!(d1, :c); ensure_layout!(d2, :g)
                Tarang.linear_combination_state!([d1], 2.0, [u1], -0.5, [v1])
                Tarang.linear_combination_state!([d2], 2.0, [u2], -0.5, [v2])
                got, want = d1, d2
            else
                d1 = _mk("d", coords, dist, bases); d2 = copy(d1)
                ensure_layout!(d1, :c); ensure_layout!(d2, :g)
                Tarang.add_scaled_state!([d1], [u1], [v1], 0.25)
                Tarang.add_scaled_state!([d2], [u2], [v2], 0.25)
                got, want = d1, d2
            end

            # Compare in ONE layout so the check is not itself layout-dependent.
            ensure_layout!(got, :g); ensure_layout!(want, :g)
            @test Array(get_grid_data(got)) ≈ Array(get_grid_data(want)) atol=1e-12
        end
    end
end
