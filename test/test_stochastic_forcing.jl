"""
Test stochastic forcing implementation for CPU and GPU.

Tests:
1. StochasticForcing construction (CPU and GPU)
2. Forcing spectrum generation
3. Forcing stays constant within timestep (substep awareness)
4. Energy injection rate calculation
5. Hermitian symmetry enforcement
6. GPU compatibility

Run with:
    julia --project test/test_stochastic_forcing.jl
"""

using Test
using Random
using LinearAlgebra
using Tarang

println("=" ^ 60)
println("Stochastic Forcing Tests")
println("=" ^ 60)

@testset "StochasticForcing" begin

    @testset "CPU Construction" begin
        # Test 1D construction
        forcing_1d = StochasticForcing(
            field_size=(64,),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            architecture=CPU()
        )
        @test forcing_1d.field_size == (64,)
        @test forcing_1d.forcing_rate == 0.1
        @test forcing_1d.dt == 0.01
        @test forcing_1d.is_stochastic == true
        @test forcing_1d.is_gpu == false
        println("  1D CPU construction OK")

        # Test 2D construction
        forcing_2d = StochasticForcing(
            field_size=(64, 64),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            architecture=CPU()
        )
        @test forcing_2d.field_size == (64, 64)
        @test size(forcing_2d.spectrum) == (64, 64)
        @test size(forcing_2d.cached_forcing) == (64, 64)
        @test forcing_2d.is_gpu == false
        println("  2D CPU construction OK")

        # Test 3D construction
        forcing_3d = StochasticForcing(
            field_size=(32, 32, 32),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            architecture=CPU()
        )
        @test forcing_3d.field_size == (32, 32, 32)
        @test size(forcing_3d.spectrum) == (32, 32, 32)
        @test forcing_3d.is_gpu == false
        println("  3D CPU construction OK")
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

        # Isotropic spectrum (alias for ring)
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

        # Copy F1 before generating F2 (generate_forcing! returns a reference to cached_forcing)
        F1_copy = copy(F1)

        # Forcing should change at different times
        F2 = generate_forcing!(forcing, 1.0)
        @test F1_copy != F2  # Compare copy, not the reference
        println("  Forcing varies between timesteps OK")
    end

    @testset "Hermitian Symmetry" begin
        # Test that generated forcing has Hermitian symmetry
        forcing = StochasticForcing(
            field_size=(16, 16),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            rng=MersenneTwister(999)
        )

        F = generate_forcing!(forcing, 0.0)

        # Check Hermitian symmetry: F(-k) = F(k)*
        nx, ny = size(F)
        max_error = 0.0
        for j in 1:ny
            for i in 1:nx
                ci = i == 1 ? 1 : nx + 2 - i
                cj = j == 1 ? 1 : ny + 2 - j
                error = abs(F[i, j] - conj(F[ci, cj]))
                max_error = max(max_error, error)
            end
        end
        @test max_error < 1e-10
        println("  Hermitian symmetry OK (max error: $max_error)")

        # Check that self-conjugate modes are real
        @test abs(imag(F[1, 1])) < 1e-14  # (0, 0) mode
        # Nyquist modes
        nyq_x = nx ÷ 2 + 1
        nyq_y = ny ÷ 2 + 1
        @test abs(imag(F[nyq_x, 1])) < 1e-10  # (Nx/2, 0)
        @test abs(imag(F[1, nyq_y])) < 1e-10  # (0, Ny/2)
        @test abs(imag(F[nyq_x, nyq_y])) < 1e-10  # (Nx/2, Ny/2)
        println("  Self-conjugate modes are real OK")
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

    @testset "Work Calculation" begin
        forcing = StochasticForcing(
            field_size=(16, 16),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            rng=MersenneTwister(123)
        )

        # Generate forcing
        F = generate_forcing!(forcing, 0.0)

        # Create mock solution arrays
        sol_prev = randn(ComplexF64, 16, 16)
        sol_next = sol_prev .+ 0.01 .* F  # Simple forward Euler

        # Store previous solution
        store_prevsol!(forcing, sol_prev)

        # Compute Stratonovich work
        W_strat = work_stratonovich(forcing, sol_next)
        @test isfinite(W_strat)
        println("  Stratonovich work = $W_strat")

        # Compute Itô work
        W_ito = work_ito(forcing, sol_prev)
        @test isfinite(W_ito)
        println("  Itô work = $W_ito")

        # Instantaneous power
        P = instantaneous_power(forcing, sol_prev)
        @test isfinite(P)
        println("  Instantaneous power = $P")
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

    @testset "1D Forcing" begin
        forcing_1d = StochasticForcing(
            field_size=(64,),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            rng=MersenneTwister(42)
        )

        F = generate_forcing!(forcing_1d, 0.0)
        @test size(F) == (64,)
        @test F[1] == 0.0 + 0.0im  # Zero mean

        # Check 1D Hermitian symmetry
        n = length(F)
        max_error = 0.0
        for i in 2:(n÷2)
            j = n + 2 - i
            error = abs(F[i] - conj(F[j]))
            max_error = max(max_error, error)
        end
        @test max_error < 1e-10
        println("  1D forcing with Hermitian symmetry OK")
    end

    @testset "3D Forcing" begin
        forcing_3d = StochasticForcing(
            field_size=(16, 16, 16),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=0.01,
            rng=MersenneTwister(42)
        )

        F = generate_forcing!(forcing_3d, 0.0)
        @test size(F) == (16, 16, 16)
        @test F[1, 1, 1] == 0.0 + 0.0im  # Zero mean
        println("  3D forcing generation OK")

        # Check 3D Hermitian symmetry (sample check)
        nx, ny, nz = size(F)
        @test abs(F[2, 3, 4] - conj(F[nx, ny-1, nz-2])) < 1e-10
        println("  3D Hermitian symmetry OK")
    end

end

# GPU tests (only run if CUDA is available)
@testset "GPU Stochastic Forcing" begin
    # Check if CUDA is available
    gpu_available = false
    try
        using CUDA
        if CUDA.functional()
            gpu_available = true
        end
    catch
        # CUDA not available
    end

    if gpu_available
        println("\nGPU tests (CUDA available):")

        @testset "GPU Construction" begin
            forcing_gpu = StochasticForcing(
                field_size=(32, 32),
                forcing_rate=0.1,
                k_forcing=4.0,
                dk_forcing=2.0,
                dt=0.01,
                architecture=GPU(),
                rng=MersenneTwister(42)
            )

            @test forcing_gpu.is_gpu == true
            @test size(forcing_gpu.cached_forcing) == (32, 32)
            println("    GPU construction OK")
        end

        @testset "GPU Forcing Generation" begin
            forcing_gpu = StochasticForcing(
                field_size=(32, 32),
                forcing_rate=0.1,
                k_forcing=4.0,
                dk_forcing=2.0,
                dt=0.01,
                architecture=GPU(),
                rng=MersenneTwister(42)
            )

            F = generate_forcing!(forcing_gpu, 0.0)
            @test size(F) == (32, 32)

            # Transfer to CPU for checking
            F_cpu = Array(F)
            @test F_cpu[1, 1] == 0.0 + 0.0im  # Zero mean
            println("    GPU forcing generation OK")

            # Check Hermitian symmetry
            nx, ny = size(F_cpu)
            max_error = 0.0
            for j in 1:ny
                for i in 1:nx
                    ci = i == 1 ? 1 : nx + 2 - i
                    cj = j == 1 ? 1 : ny + 2 - j
                    error = abs(F_cpu[i, j] - conj(F_cpu[ci, cj]))
                    max_error = max(max_error, error)
                end
            end
            @test max_error < 1e-10
            println("    GPU Hermitian symmetry OK")
        end

        @testset "GPU apply_forcing!" begin
            forcing_gpu = StochasticForcing(
                field_size=(16, 16),
                forcing_rate=0.1,
                k_forcing=4.0,
                dk_forcing=2.0,
                dt=0.01,
                architecture=GPU(),
                rng=MersenneTwister(42)
            )

            # Create GPU field
            field_gpu = CUDA.zeros(ComplexF64, 16, 16)

            # Apply forcing
            apply_forcing!(field_gpu, forcing_gpu, 0.0)

            # Check on CPU
            field_cpu = Array(field_gpu)
            @test any(field_cpu .!= 0)
            println("    GPU apply_forcing! OK")
        end

        @testset "GPU Work Calculation" begin
            forcing_gpu = StochasticForcing(
                field_size=(16, 16),
                forcing_rate=0.1,
                k_forcing=4.0,
                dk_forcing=2.0,
                dt=0.01,
                architecture=GPU(),
                rng=MersenneTwister(123)
            )

            # Generate forcing
            F = generate_forcing!(forcing_gpu, 0.0)

            # Create mock GPU solution arrays
            sol_cpu = randn(ComplexF64, 16, 16)
            sol_prev = CuArray(sol_cpu)
            sol_next = sol_prev .+ 0.01f0 .* F

            # Store previous solution
            store_prevsol!(forcing_gpu, sol_prev)

            # Compute Stratonovich work
            W_strat = work_stratonovich(forcing_gpu, sol_next)
            @test isfinite(W_strat)
            println("    GPU Stratonovich work = $W_strat")

            # Compute Itô work
            W_ito = work_ito(forcing_gpu, sol_prev)
            @test isfinite(W_ito)
            println("    GPU Itô work = $W_ito")
        end

    else
        println("\nGPU tests skipped (CUDA not available)")
        @test_skip "GPU construction"
        @test_skip "GPU forcing generation"
        @test_skip "GPU apply_forcing!"
        @test_skip "GPU work calculation"
    end
end

println("\n" * "=" ^ 60)
println("All Stochastic Forcing Tests Completed!")
println("=" ^ 60)
