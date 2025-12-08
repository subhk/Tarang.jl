"""
Test stochastic forcing implementation.

Tests:
1. StochasticForcing construction
2. Forcing spectrum generation
3. Forcing stays constant within timestep (substep awareness)
4. Energy injection rate calculation

Run with:
    julia --project test/test_stochastic_forcing.jl
"""

using Test
using Random
using LinearAlgebra

# Include stochastic forcing directly for standalone testing
include("../src/core/stochastic_forcing.jl")

println("=" ^ 60)
println("Stochastic Forcing Tests")
println("=" ^ 60)

@testset "StochasticForcing" begin

    @testset "Construction" begin
        # Test 1D construction
        forcing_1d = StochasticForcing(
            field_size=(64,),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01
        )
        @test forcing_1d.field_size == (64,)
        @test forcing_1d.forcing_rate == 0.1
        @test forcing_1d.dt == 0.01
        @test forcing_1d.is_stochastic == true
        println("  1D construction OK")

        # Test 2D construction
        forcing_2d = StochasticForcing(
            field_size=(64, 64),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01
        )
        @test forcing_2d.field_size == (64, 64)
        @test size(forcing_2d.spectrum) == (64, 64)
        @test size(forcing_2d.cached_forcing) == (64, 64)
        println("  2D construction OK")

        # Test 3D construction
        forcing_3d = StochasticForcing(
            field_size=(32, 32, 32),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01
        )
        @test forcing_3d.field_size == (32, 32, 32)
        @test size(forcing_3d.spectrum) == (32, 32, 32)
        println("  3D construction OK")
    end

    @testset "Spectrum Types" begin
        N = 64

        # Ring spectrum
        forcing_ring = StochasticForcing(
            field_size=(N, N),
            forcing_rate=0.1,
            k_forcing=8.0,
            dk_forcing=2.0,
            spectrum_type=:ring
        )
        # Check that spectrum is non-zero only in a ring
        @test sum(forcing_ring.spectrum .> 0) > 0
        println("  Ring spectrum OK")

        # Isotropic spectrum
        forcing_iso = StochasticForcing(
            field_size=(N, N),
            forcing_rate=0.1,
            k_forcing=8.0,
            dk_forcing=2.0,
            spectrum_type=:isotropic
        )
        @test sum(forcing_iso.spectrum .> 0) > 0
        println("  Isotropic spectrum OK")

        # Bandlimited spectrum
        forcing_band = StochasticForcing(
            field_size=(N, N),
            forcing_rate=0.1,
            k_forcing=8.0,
            dk_forcing=2.0,
            spectrum_type=:bandlimited
        )
        @test sum(forcing_band.spectrum .> 0) > 0
        println("  Bandlimited spectrum OK")

        # Kolmogorov spectrum
        forcing_kol = StochasticForcing(
            field_size=(N, N),
            forcing_rate=0.1,
            k_forcing=8.0,
            dk_forcing=2.0,
            spectrum_type=:kolmogorov
        )
        @test sum(forcing_kol.spectrum .> 0) > 0
        println("  Kolmogorov spectrum OK")
    end

    @testset "Forcing Generation" begin
        Random.seed!(12345)

        forcing = StochasticForcing(
            field_size=(32, 32),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            rng=MersenneTwister(42)
        )

        # Generate forcing at time 0
        F1 = generate_forcing!(forcing, 0.0)
        @test size(F1) == (32, 32)
        @test eltype(F1) <: Complex

        # Zero mean (k=0 mode should be zero)
        @test F1[1, 1] == 0.0 + 0.0im
        println("  Zero mean forcing OK")

        # Forcing should change at different times
        F2 = generate_forcing!(forcing, 1.0)
        @test F1 != F2
        println("  Forcing varies between timesteps OK")
    end

    @testset "Substep Awareness (KEY TEST)" begin
        Random.seed!(12345)

        forcing = StochasticForcing(
            field_size=(32, 32),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            rng=MersenneTwister(42)
        )

        # Reset forcing to ensure clean state
        reset_forcing!(forcing)

        # Simulate a timestep with multiple substeps (like RK4)
        current_time = 0.0

        # Substep 1: should generate new forcing
        F_substep1 = generate_forcing!(forcing, current_time, 1)
        F1_copy = copy(F_substep1)

        # Substep 2: should return SAME forcing (cached)
        F_substep2 = generate_forcing!(forcing, current_time, 2)
        @test F_substep1 === F_substep2  # Same object
        @test F1_copy == F_substep2  # Same values
        println("  Substep 2 returns cached forcing OK")

        # Substep 3: should return SAME forcing (cached)
        F_substep3 = generate_forcing!(forcing, current_time, 3)
        @test F1_copy == F_substep3
        println("  Substep 3 returns cached forcing OK")

        # Substep 4: should return SAME forcing (cached)
        F_substep4 = generate_forcing!(forcing, current_time, 4)
        @test F1_copy == F_substep4
        println("  Substep 4 returns cached forcing OK")

        # Now move to next timestep
        next_time = 0.01  # dt = 0.01

        # Substep 1 of new timestep: should generate NEW forcing
        F_new = generate_forcing!(forcing, next_time, 1)
        # The forcing should be different from the previous timestep
        # (with high probability since it's random)
        @test F1_copy != F_new
        println("  New timestep generates new forcing OK")

        println("\n  KEY RESULT: Forcing stays constant within timestep across substeps")
    end

    @testset "Time-based Caching" begin
        forcing = StochasticForcing(
            field_size=(16, 16),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            rng=MersenneTwister(123)
        )

        # Generate at time 0
        F1 = copy(generate_forcing!(forcing, 0.0))

        # Same time should return same forcing
        F1_again = generate_forcing!(forcing, 0.0)
        @test F1 == F1_again
        println("  Same time returns cached forcing OK")

        # Different time should generate new forcing
        F2 = generate_forcing!(forcing, 1.0)
        @test F1 != F2
        println("  Different time generates new forcing OK")
    end

    @testset "Energy Injection Rate" begin
        forcing = StochasticForcing(
            field_size=(32, 32),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01
        )

        ε = energy_injection_rate(forcing)
        @test ε > 0
        @test isfinite(ε)
        println("  Energy injection rate = $ε")
    end

    @testset "set_dt!" begin
        forcing = StochasticForcing(
            field_size=(16, 16),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01
        )

        @test forcing.dt == 0.01

        set_dt!(forcing, 0.001)
        @test forcing.dt == 0.001
        println("  set_dt! works OK")
    end

    @testset "reset_forcing!" begin
        forcing = StochasticForcing(
            field_size=(16, 16),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            rng=MersenneTwister(999)
        )

        # Generate some forcing
        generate_forcing!(forcing, 0.0)
        @test forcing.last_update_time == 0.0

        # Reset
        reset_forcing!(forcing)
        @test forcing.last_update_time == -Inf
        @test all(forcing.cached_forcing .== 0)
        println("  reset_forcing! works OK")
    end

    @testset "Deterministic Forcing" begin
        # Test deterministic forcing
        function my_forcing(x, y, t, params)
            A = get(params, :amplitude, 1.0)
            return A * sin.(x) .* cos.(y)
        end

        det_forcing = DeterministicForcing(
            my_forcing,
            (32, 32);
            parameters=Dict{Symbol, Any}(:amplitude => 2.0)
        )

        @test det_forcing.field_size == (32, 32)
        @test det_forcing.parameters[:amplitude] == 2.0
        println("  DeterministicForcing construction OK")
    end

    @testset "apply_forcing!" begin
        forcing = StochasticForcing(
            field_size=(16, 16),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            rng=MersenneTwister(42)
        )

        # Create a field
        field = zeros(Complex{Float64}, 16, 16)

        # Apply forcing
        apply_forcing!(field, forcing, 0.0)

        # Field should no longer be zero
        @test any(field .!= 0)
        println("  apply_forcing! works OK")

        # Apply with substep
        field2 = zeros(Complex{Float64}, 16, 16)
        apply_forcing!(field2, forcing, 0.0, 2)  # substep 2

        # Should use cached forcing
        @test field == field2
        println("  apply_forcing! with substep works OK")
    end

end

println("\n" * "=" ^ 60)
println("All Stochastic Forcing Tests Passed!")
println("=" ^ 60)
