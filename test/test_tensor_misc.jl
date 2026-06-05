"""
Test suite for src/core/operators/tensor/tensor_misc.jl

Covers the three internal functions in that file:
  - Tarang.compute_grid_spacing(basis, dist, axis)
  - Tarang.evaluate_outer(Outer(u, v), layout)
  - Tarang.evaluate_advective_cfl(AdvectiveCFL(velocity, coords), layout)

Oracles are derived directly from the function bodies so they match the
exact convention used in the source (see comments at each testset).
"""

using Test
using Tarang

# ============================================================================
# compute_grid_spacing
# ============================================================================

@testset "compute_grid_spacing" begin

    @testset "RealFourier: dx = L / N" begin
        # Source convention: FourierBasis returns L/N (uniform), where
        # L = bounds[2] - bounds[1].
        for N in (8, 16, 32), L in (1.0, 2.0, 2π)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; dtype=Float64)
            xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L), dtype=Float64)

            dx = Tarang.compute_grid_spacing(xb, dist, 1)

            @test dx isa Real
            @test isapprox(dx, L / N; rtol=1e-12)
        end
    end

    @testset "RealFourier: nonzero left bound uses span L = b - a" begin
        N = 20
        a, b = -1.0, 3.0   # L = 4
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(a, b), dtype=Float64)

        dx = Tarang.compute_grid_spacing(xb, dist, 1)
        @test isapprox(dx, (b - a) / N; rtol=1e-12)
    end

    @testset "ChebyshevT: analytic stretched node spacing vector" begin
        # Source convention for ChebyshevT:
        #   stretch = COV.stretch = (b - a) / 2   (native bounds (-1, 1))
        #   theta_i = pi*(i + 0.5)/N   for i = 0..N-1
        #   spacing_i = stretch * sin(theta_i) * pi / N
        for N in (8, 16), (a, b) in ((-1.0, 1.0), (0.0, 2.0), (-2.0, 5.0))
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; dtype=Float64)
            cb = ChebyshevT(coords["x"]; size=N, bounds=(a, b), dtype=Float64)

            spacing = Tarang.compute_grid_spacing(cb, dist, 1)

            @test spacing isa AbstractVector
            @test length(spacing) == N

            stretch = (b - a) / 2
            i = collect(0:(N - 1))
            theta = pi .* (i .+ 0.5) ./ N
            expected = stretch .* sin.(theta) .* pi ./ N

            @test isapprox(spacing, expected; rtol=1e-12)
            # Non-uniform: interior nodes are coarser than near-boundary ones.
            @test maximum(spacing) > minimum(spacing)
            @test all(spacing .> 0)
        end
    end

    @testset "nothing basis branch is unreachable via typed dispatch" begin
        # compute_grid_spacing is typed as (basis::Basis, dist, axis::Int).
        # `nothing` is not a Basis, so the `if basis === nothing` guard inside
        # the body is dead code: calling with nothing raises MethodError rather
        # than returning Inf. Documenting the actual (typed) behavior here.
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        @test_throws MethodError Tarang.compute_grid_spacing(nothing, dist, 1)
    end
end

# ============================================================================
# evaluate_outer
# ============================================================================

@testset "evaluate_outer" begin

    @testset "T_ij = u_i .* v_j on 2D grid" begin
        N = 8
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dtype=Float64)
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dtype=Float64)

        u = VectorField(dist, coords, "u", (xb, yb), Float64)
        v = VectorField(dist, coords, "v", (xb, yb), Float64)

        # Give each component distinct known grid data.
        mesh = Tarang.create_meshgrid(u.components[1].domain)
        x, y = mesh["x"], mesh["y"]

        Tarang.ensure_layout!(u.components[1], :g)
        Tarang.ensure_layout!(u.components[2], :g)
        Tarang.ensure_layout!(v.components[1], :g)
        Tarang.ensure_layout!(v.components[2], :g)

        u1 = @. sin(x) + 2.0
        u2 = @. cos(y) + 3.0
        v1 = @. sin(2x) * cos(y) + 1.5
        v2 = @. cos(3y) + 0.5

        Tarang.get_grid_data(u.components[1]) .= u1
        Tarang.get_grid_data(u.components[2]) .= u2
        Tarang.get_grid_data(v.components[1]) .= v1
        Tarang.get_grid_data(v.components[2]) .= v2

        T = Tarang.evaluate_outer(Tarang.Outer(u, v))

        @test size(T.components) == (2, 2)

        comps_u = (u1, u2)
        comps_v = (v1, v2)
        for i in 1:2, j in 1:2
            Tarang.ensure_layout!(T.components[i, j], :g)
            got = Tarang.get_grid_data(T.components[i, j])
            expected = comps_u[i] .* comps_v[j]
            @test isapprox(got, expected; rtol=1e-10, atol=1e-12)
        end
    end

    @testset "outer of a field with itself (1D)" begin
        N = 8
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dtype=Float64)

        u = VectorField(dist, coords, "u", (xb,), Float64)
        Tarang.ensure_layout!(u.components[1], :g)
        mesh = Tarang.create_meshgrid(u.components[1].domain)
        x = mesh["x"]
        u1 = @. 1.0 + 0.5 * sin(x)
        Tarang.get_grid_data(u.components[1]) .= u1

        T = Tarang.evaluate_outer(Tarang.Outer(u, u))
        @test size(T.components) == (1, 1)
        Tarang.ensure_layout!(T.components[1, 1], :g)
        @test isapprox(Tarang.get_grid_data(T.components[1, 1]), u1 .* u1;
                       rtol=1e-10, atol=1e-12)
    end

    @testset "non-VectorField operands throw ArgumentError" begin
        N = 8
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dtype=Float64)

        s = ScalarField(dist, "s", (xb,), Float64)
        u = VectorField(dist, coords, "u", (xb,), Float64)

        @test_throws ArgumentError Tarang.evaluate_outer(Tarang.Outer(s, u))
        @test_throws ArgumentError Tarang.evaluate_outer(Tarang.Outer(u, s))
    end
end

# ============================================================================
# evaluate_advective_cfl
# ============================================================================

@testset "evaluate_advective_cfl" begin

    @testset "1D Fourier: f = |u| / dx" begin
        # Source formula: f = sum_i |u_i| / dx_i.
        # For RealFourier, dx = L / N (a scalar), so f is uniform = |u| / dx.
        for N in (8, 16), L in (1.0, 2π), umag in (0.5, -2.0, 3.0)
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; dtype=Float64)
            xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L), dtype=Float64)

            u = VectorField(dist, coords, "u", (xb,), Float64)
            Tarang.ensure_layout!(u.components[1], :g)
            fill!(Tarang.get_grid_data(u.components[1]), umag)

            result = Tarang.evaluate_advective_cfl(Tarang.AdvectiveCFL(u, coords))
            Tarang.ensure_layout!(result, :g)

            dx = L / N
            expected = abs(umag) / dx
            got = Tarang.get_grid_data(result)
            @test all(isapprox.(got, expected; rtol=1e-10))
        end
    end

    @testset "2D Fourier: f = |u|/dx + |v|/dy (uniform field)" begin
        Nx, Ny = 16, 8
        Lx, Ly = 2.0, 3.0
        umag, vmag = 1.5, -2.5

        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dtype=Float64)
        yb = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly), dtype=Float64)

        u = VectorField(dist, coords, "u", (xb, yb), Float64)
        Tarang.ensure_layout!(u.components[1], :g)
        Tarang.ensure_layout!(u.components[2], :g)
        fill!(Tarang.get_grid_data(u.components[1]), umag)
        fill!(Tarang.get_grid_data(u.components[2]), vmag)

        result = Tarang.evaluate_advective_cfl(Tarang.AdvectiveCFL(u, coords))
        Tarang.ensure_layout!(result, :g)

        dx = Lx / Nx
        dy = Ly / Ny
        expected = abs(umag) / dx + abs(vmag) / dy
        got = Tarang.get_grid_data(result)
        @test all(isapprox.(got, expected; rtol=1e-10))
    end

    @testset "2D Fourier: spatially varying field, pointwise oracle" begin
        Nx, Ny = 16, 16
        Lx, Ly = 2π, 2π

        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dtype=Float64)
        yb = RealFourier(coords["y"]; size=Ny, bounds=(0.0, Ly), dtype=Float64)

        u = VectorField(dist, coords, "u", (xb, yb), Float64)
        Tarang.ensure_layout!(u.components[1], :g)
        Tarang.ensure_layout!(u.components[2], :g)

        mesh = Tarang.create_meshgrid(u.components[1].domain)
        x, y = mesh["x"], mesh["y"]
        ucomp = @. 2.0 + sin(x)
        vcomp = @. -1.0 + cos(y)
        Tarang.get_grid_data(u.components[1]) .= ucomp
        Tarang.get_grid_data(u.components[2]) .= vcomp

        result = Tarang.evaluate_advective_cfl(Tarang.AdvectiveCFL(u, coords))
        Tarang.ensure_layout!(result, :g)

        dx = Lx / Nx
        dy = Ly / Ny
        expected = abs.(ucomp) ./ dx .+ abs.(vcomp) ./ dy
        @test isapprox(Tarang.get_grid_data(result), expected; rtol=1e-10, atol=1e-12)
    end

    @testset "non-VectorField operand throws ArgumentError" begin
        N = 8
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dtype=Float64)
        s = ScalarField(dist, "s", (xb,), Float64)

        @test_throws ArgumentError Tarang.evaluate_advective_cfl(Tarang.AdvectiveCFL(s, coords))
    end

    @testset "non-grid layout warns and still computes in :g" begin
        N = 8
        L = 2π
        umag = 2.0
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L), dtype=Float64)

        u = VectorField(dist, coords, "u", (xb,), Float64)
        Tarang.ensure_layout!(u.components[1], :g)
        fill!(Tarang.get_grid_data(u.components[1]), umag)

        # Passing :c triggers the @warn branch but result is still grid-space.
        result = @test_logs (:warn,) match_mode=:any Tarang.evaluate_advective_cfl(
            Tarang.AdvectiveCFL(u, coords), :c)
        Tarang.ensure_layout!(result, :g)
        expected = abs(umag) / (L / N)
        @test all(isapprox.(Tarang.get_grid_data(result), expected; rtol=1e-10))
    end
end
