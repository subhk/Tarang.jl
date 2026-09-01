using Test
using Tarang
using InteractiveUtils

@testset "Field storage type stability" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    yb = ComplexFourier(coords["y"]; size=16, bounds=(0.0, 2π))

    @testset "Phase 1: no nothing in storage" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        @test Tarang.get_grid_data(u) !== nothing
        @test Tarang.get_coeff_data(u) !== nothing

        tau = ScalarField(dist, "tau", (), Float64)
        @test Tarang.get_grid_data(tau) !== nothing
        @test length(Tarang.get_grid_data(tau)) == 0
        @test Tarang.get_coeff_data(tau) !== nothing
        @test length(Tarang.get_coeff_data(tau)) == 0
    end

    @testset "0-D spectral filters are no-ops" begin
        tau = ScalarField(dist, "tau_filter", (), Float64)
        grid_before = copy(Tarang.get_grid_data(tau))
        coeff_before = copy(Tarang.get_coeff_data(tau))
        layout_before = tau.current_layout

        @test apply_spectral_cutoff!(tau, 2 / 3) === tau
        @test Tarang.apply_3d_dealiasing!(tau, 1.5) === tau
        @test Tarang.apply_basic_dealiasing!(tau, 1.5) === tau
        @test Tarang.get_grid_data(tau) == grid_before
        @test Tarang.get_coeff_data(tau) == coeff_before
        @test tau.current_layout === layout_before

        k_squared = Tarang.compute_wavenumber_squared_grid(tau)
        @test isempty(k_squared)
        @test eltype(k_squared) === Float64

        evaluator = NonlinearEvaluator(dist)
        temp_tau = get_temp_field(evaluator, tau, "tau_filter_scratch")
        @test isempty(temp_tau.bases)
        @test temp_tau.dtype === Float64
    end

    @testset "Phase 2: field array type fixed at construction" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        gtype = typeof(Tarang.get_grid_data(u))
        Tarang.synchronize_field_architecture!(u; arch=dist.architecture)
        @test typeof(Tarang.get_grid_data(u)) === gtype
    end

    @testset "Phase 3: get_grid_data is type-stable" begin
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)
        # @inferred throws unless the inferred return type is concrete — asserts
        # the SerialFieldStorage{G,C} parametrization payoff.
        @test (@inferred Tarang.get_grid_data(u); true)
        @test (@inferred Tarang.get_coeff_data(u); true)
    end

    @testset "field metadata follows canonical coordinate order" begin
        mixed_coords = CartesianCoordinates("x", "z")
        mixed_dist = Distributor(mixed_coords; dtype=Float64, device=CPU())
        mixed_xb = RealFourier(mixed_coords["x"]; size=8, bounds=(0.0, 2π))
        mixed_zb = ChebyshevT(mixed_coords["z"]; size=9, bounds=(-1.0, 1.0))

        # Callers may supply bases out of coordinate order. Domain owns the
        # canonical order, and every field shape/transform must follow it.
        canonical_u = ScalarField(mixed_dist, "canonical", (mixed_xb, mixed_zb), Float64)
        u = ScalarField(mixed_dist, "mixed", (mixed_zb, mixed_xb), Float64)
        @test u.domain === canonical_u.domain
        ensure_layout!(u, :g)
        @test u.bases == u.domain.bases == (mixed_xb, mixed_zb)
        @test u.layout.global_shape == size(Tarang.get_grid_data(u)) == (8, 9)
        @test size(Tarang.get_coeff_data(u)) == (5, 9)
        @test get_scaled_shape(u) == (8, 9)

        x = reshape(Tarang.local_grid(mixed_xb, mixed_dist, 1.0;
                                      move_to_arch=false), :, 1)
        z = reshape(Tarang.local_grid(mixed_zb, mixed_dist, 1.0;
                                      move_to_arch=false), 1, :)
        data = @. sin(2x) * (1 + z - 0.25z^2)
        Tarang.get_grid_data(u) .= data
        ensure_layout!(u, :c)
        ensure_layout!(u, :g)
        @test isapprox(Tarang.get_grid_data(u), data; rtol=1e-10, atol=1e-11)

        v = VectorField(mixed_dist, "v", (mixed_zb, mixed_xb), Float64)
        @test v.bases == v.domain.bases == (mixed_xb, mixed_zb)
        @test all(component -> component.bases == v.bases, v.components)
        @test all(component -> size(Tarang.get_grid_data(component)) == (8, 9),
                  v.components)
        @test all(component -> size(Tarang.get_coeff_data(component)) == (5, 9),
                  v.components)
        for (i, component) in enumerate(v.components)
            Tarang.get_grid_data(component) .= i .* data
        end
        ensure_layout!(v, :c)
        ensure_layout!(v, :g)
        @test all(i -> isapprox(Tarang.get_grid_data(v.components[i]), i .* data;
                                rtol=1e-10, atol=1e-11), eachindex(v.components))

        tensor = TensorField(mixed_dist, "T", (mixed_zb, mixed_xb), Float64)
        @test tensor.bases == tensor.domain.bases == (mixed_xb, mixed_zb)
        @test all(component -> component.bases == tensor.bases, tensor.components)
        @test all(component -> size(Tarang.get_grid_data(component)) == (8, 9),
                  tensor.components)
        @test all(component -> size(Tarang.get_coeff_data(component)) == (5, 9),
                  tensor.components)
        for (i, component) in enumerate(tensor.components)
            Tarang.get_grid_data(component) .= i .* data
        end
        ensure_layout!(tensor, :c)
        ensure_layout!(tensor, :g)
        @test all(i -> isapprox(Tarang.get_grid_data(tensor.components[i]), i .* data;
                                rtol=1e-10, atol=1e-11), eachindex(tensor.components))
    end

    @testset "copy preserves live data and scales" begin
        # Regression guard (review C3): copy must not crash on a scaled field, and
        # must duplicate (not alias) the live-layout data. The off-layout array is
        # kept full-size (not a 0-sized placeholder) so a later ensure_layout!/
        # transform of the copy can plan its FFT.
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :c)
        Tarang.get_coeff_data(u) .= rand(ComplexF64, size(Tarang.get_coeff_data(u)))
        c = copy(u)
        @test Tarang.get_coeff_data(c) == Tarang.get_coeff_data(u)   # live layout copied
        @test Tarang.get_coeff_data(c) !== Tarang.get_coeff_data(u)  # not aliased
        ensure_layout!(c, :g)  # must be able to plan the backward transform
        @test c.current_layout == :g

        us = ScalarField(dist, "us", (xb, yb), Float64)
        set_scales!(us, (1.5, 1.5))
        ensure_layout!(us, :g)
        cs = copy(us)                                                # must not BoundsError (C3)
        @test size(Tarang.get_grid_data(cs)) == size(Tarang.get_grid_data(us))
    end

    @testset "scaled RealFourier round-trip" begin
        # Backward transform must size the irfft from the SCALED grid (scale ×
        # basis.meta.size), not the base size, mirroring the scale-correct forward.
        # Regression for the rfft-axis detection / irfft-size bug surfaced by the
        # storage parametrization (transform_fourier.jl backward path).
        u = ScalarField(dist, "u", (xb, yb), Float64)
        set_scales!(u, (1.5, 1.5))
        ensure_layout!(u, :g)
        orig = copy(Tarang.get_grid_data(u))
        orig .= rand(size(orig)...)
        Tarang.get_grid_data(u) .= orig
        ensure_layout!(u, :c)
        ensure_layout!(u, :g)
        @test isapprox(Tarang.get_grid_data(u), orig; rtol=1e-10)
    end
end
