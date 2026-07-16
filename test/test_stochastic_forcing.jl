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

    @testset "Physical spectrum normalization and validation" begin
        ε = 0.125

        # Tarang stores unnormalized full FFT coefficients, so Parseval's
        # normalization is M⁻² and must not depend on grid resolution.
        for field_size in ((16,), (16, 16), (8, 8, 8), (32, 32))
            forcing = StochasticForcing(
                field_size=field_size,
                energy_injection_rate=ε,
                k_forcing=3.0,
                spectrum_type=:lowk,
                injection_metric=:direct,
            )
            M = prod(field_size)
            physical_rate = sum(abs2, forcing.spectrum) / (2M^2)
            @test physical_rate ≈ ε rtol=5e-14
            @test forcing.injection_metric === :direct
        end

        vorticity_forcing = StochasticForcing(
            field_size=(32, 32),
            energy_injection_rate=ε,
            k_forcing=6.0,
            dk_forcing=1.0,
            spectrum_type=:band,
            injection_metric=:vorticity_kinetic,
        )
        kx, ky = vorticity_forcing.wavenumbers
        weighted_power = sum(CartesianIndices(vorticity_forcing.spectrum)) do I
            k2 = kx[I[1]]^2 + ky[I[2]]^2
            iszero(k2) ? 0.0 : abs2(vorticity_forcing.spectrum[I]) / k2
        end
        @test weighted_power / (2prod(vorticity_forcing.field_size)^2) ≈ ε rtol=5e-14
        @test vorticity_forcing.injection_metric === :vorticity_kinetic

        direct_vorticity_forcing = StochasticForcing(
            field_size=(32, 32),
            energy_injection_rate=ε,
            k_forcing=6.0,
            dk_forcing=1.0,
            spectrum_type=:band,
            injection_metric=:direct,
        )
        @test forcing_enstrophy_injection_rate(direct_vorticity_forcing) ≈ ε rtol=5e-14

        M_vorticity = prod(vorticity_forcing.field_size)
        manual_enstrophy_rate = sum(abs2, vorticity_forcing.spectrum) / (2M_vorticity^2)
        @test forcing_enstrophy_injection_rate(vorticity_forcing) ≈
              manual_enstrophy_rate rtol=5e-14
        @test manual_enstrophy_rate > ε

        ring = StochasticForcing(
            field_size=(32,),
            energy_injection_rate=ε,
            k_forcing=8.0,
            dk_forcing=2.0,
            spectrum_type=:ring,
        )
        @test abs2(ring.spectrum[11]) / abs2(ring.spectrum[9]) ≈ exp(-1 / 2) rtol=5e-14

        kolmogorov = StochasticForcing(
            field_size=(32,),
            energy_injection_rate=ε,
            k_forcing=8.0,
            dk_forcing=2.0,
            spectrum_type=:kolmogorov,
        )
        expected_kolmogorov_power_ratio = (6 / 8) * exp(-1 / 2)
        @test abs2(kolmogorov.spectrum[7]) / abs2(kolmogorov.spectrum[9]) ≈
              expected_kolmogorov_power_ratio rtol=5e-14

        common = (field_size=(16, 16), k_forcing=4.0, dk_forcing=1.0)
        @test_throws ArgumentError StochasticForcing(; common..., injection_metric=:unknown)
        @test_throws ArgumentError StochasticForcing(; common..., energy_injection_rate=-0.1)
        @test_throws ArgumentError StochasticForcing(; common..., forcing_rate=-0.1)
        overriding_alias = @test_logs (:warn, r"Both forcing_rate and energy_injection_rate") StochasticForcing(
            ; common..., energy_injection_rate=-0.1, forcing_rate=0.1
        )
        @test overriding_alias.energy_injection_rate == 0.1
        @test overriding_alias.forcing_rate == 0.1
        @test_throws ArgumentError StochasticForcing(
            field_size=(8, 8),
            energy_injection_rate=0.1,
            k_forcing=100.0,
            dk_forcing=0.25,
            spectrum_type=:band,
        )

        empty_zero_rate = StochasticForcing(
            field_size=(8, 8),
            energy_injection_rate=0.0,
            k_forcing=100.0,
            dk_forcing=0.25,
            spectrum_type=:band,
        )
        @test iszero(sum(abs2, empty_zero_rate.spectrum))
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

    @testset "Self-conjugate projection preserves power" begin
        data_1d = zeros(ComplexF64, 8)
        data_1d[5] = 3 + 4im
        Tarang._enforce_hermitian_1d!(data_1d)
        @test data_1d[5] ≈ 3sqrt(2) + 0im

        data_2d = zeros(ComplexF64, 8, 6)
        data_2d[5, 4] = 3 + 4im
        Tarang._enforce_hermitian_2d!(data_2d)
        @test data_2d[5, 4] ≈ 3sqrt(2) + 0im

        data_3d = zeros(ComplexF64, 8, 6, 4)
        data_3d[5, 4, 3] = 3 + 4im
        Tarang._enforce_hermitian_3d!(data_3d)
        @test data_3d[5, 4, 3] ≈ 3sqrt(2) + 0im
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
        generate_forcing!(forcing, 0.0)
        @test forcing.last_update_time == 0.0
        @test any(!iszero, forcing.cached_forcing)

        set_dt!(forcing, 0.001)
        @test forcing.dt == 0.001
        @test forcing.last_update_time == -Inf
        @test all(iszero, forcing.cached_forcing)

        generate_forcing!(forcing, 0.0)
        @test forcing.last_update_time == 0.0
        @test any(!iszero, forcing.cached_forcing)
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

    @testset "Work diagnostics support half-spectrum targets" begin
        dt = 0.01
        field_size = (8, 8)
        half_size = (5, 8)
        multiplicity = reshape([1.0, 2.0, 2.0, 2.0, 1.0], :, 1)

        for metric in (:direct, :vorticity_kinetic)
            forcing = StochasticForcing(
                field_size=field_size,
                forcing_rate=0.1,
                injection_metric=metric,
                k_forcing=2.0,
                dk_forcing=0.1,
                spectrum_type=:band,
                dt=dt,
                rng=MersenneTwister(42),
            )
            generate_forcing!(forcing, 0.0)
            forcing_view = Tarang._matched_forcing_view(forcing, half_size)
            rng = MersenneTwister(7)
            sol_prev = randn(rng, ComplexF64, half_size)
            sol_next = sol_prev .+ dt .* forcing_view
            store_prevsol!(forcing, sol_prev)

            metric_weight = [
                metric === :direct ? 1.0 :
                    (iszero(kx^2 + ky^2) ? 0.0 : inv(kx^2 + ky^2))
                for kx in forcing.wavenumbers[1][1:half_size[1]],
                    ky in forcing.wavenumbers[2]
            ]
            weights = multiplicity .* metric_weight ./ prod(field_size)^2
            pairing(a) = sum(weights .* real.(a .* conj.(forcing_view)))

            @test work_stratonovich(forcing, sol_next) ≈
                  pairing((sol_prev .+ sol_next) ./ 2) * dt
            @test work_ito(forcing, sol_prev) ≈
                  pairing(sol_prev) * dt + forcing.energy_injection_rate * dt
            @test instantaneous_power(forcing, sol_prev) ≈ pairing(sol_prev)
        end
        println("  Work diagnostics use matched forcing views OK")
    end

    @testset "Odd half-spectrum retains doubled final mode" begin
        forcing = StochasticForcing(
            field_size=(7,), forcing_rate=0.0, k_forcing=2.0,
            dk_forcing=0.1, spectrum_type=:band, dt=0.1,
        )
        fill!(forcing.cached_forcing, 0)
        forcing.cached_forcing[4] = 1
        forcing.cached_forcing[5] = 1
        sol = ComplexF64[0, 0, 0, 1]

        manual = 2 / 7^2
        @test instantaneous_power(forcing, sol) ≈ manual
        @test instantaneous_power(forcing, sol) != 1 / 7^2
    end

    @testset "CPU work diagnostics are allocation-free after warmup" begin
        forcing = StochasticForcing(
            field_size=(8, 8), forcing_rate=0.1, k_forcing=2.0,
            dk_forcing=0.1, spectrum_type=:band, dt=0.01,
            rng=MersenneTwister(91),
        )
        generate_forcing!(forcing, 0.0)
        sol = randn(MersenneTwister(92), ComplexF64, 5, 8)
        store_prevsol!(forcing, sol)
        work_stratonovich(forcing, sol)
        work_ito(forcing, sol)
        instantaneous_power(forcing, sol)

        allocations = (
            @allocated(work_stratonovich(forcing, sol)),
            @allocated(work_ito(forcing, sol)),
            @allocated(instantaneous_power(forcing, sol)),
        )
        @test allocations == (0, 0, 0)
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

    @testset "registered forcing participates in lazy RHS" begin
        N = 8
        dt = 0.01
        domain = PeriodicDomain(N, N)
        q = ScalarField(domain, "q")

        forcing = StochasticForcing(
            field_size=(N, N),
            forcing_rate=0.1,
            k_forcing=3.0,
            dk_forcing=1.0,
            dt=dt,
            rng=MersenneTwister(42)
        )

        problem = IVP([q])
        add_equation!(problem, "∂t(q) = 0")
        add_stochastic_forcing!(problem, :q, forcing)

        solver = InitialValueSolver(problem, RK222(); dt=dt)
        @test solver.rhs_plan !== nothing
        @test solver.rhs_plan.is_compiled

        Tarang._update_registered_forcings!(solver, 0.0, dt)
        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        ensure_layout!(rhs[1], :c)

        rhs_data = get_coeff_data(rhs[1])
        forcing_view = Tarang._matched_forcing_view(forcing, size(rhs_data))

        @test forcing_view !== nothing
        @test maximum(abs.(rhs_data)) > 0
        @test rhs_data ≈ forcing_view
        println("  Registered forcing is included in compiled lazy RHS")
    end

    @testset "registered forcing resolves flattened solver-state indices" begin
        N = 8
        dt = 0.01
        domain = PeriodicDomain(N, N)
        u = VectorField(domain, "u")
        q = ScalarField(domain, "q")

        forcing = StochasticForcing(
            field_size=(N, N),
            forcing_rate=0.1,
            k_forcing=3.0,
            dk_forcing=1.0,
            dt=dt,
            rng=MersenneTwister(42)
        )

        problem = IVP([u, q])
        add_equation!(problem, "∂t(u) = 0")
        add_equation!(problem, "∂t(q) = 0")
        add_stochastic_forcing!(problem, :q, forcing)

        @test haskey(problem.stochastic_forcings, 3)

        solver = InitialValueSolver(problem, RK222(); dt=dt)
        @test [field.name for field in solver.state] == ["u_x", "u_y", "q"]

        Tarang._update_registered_forcings!(solver, 0.0, dt)
        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)
        foreach(field -> ensure_layout!(field, :c), rhs)
        rhs_data = map(get_coeff_data, rhs)
        forcing_view = Tarang._matched_forcing_view(forcing, size(rhs_data[3]))

        @test all(iszero, rhs_data[1])
        @test all(iszero, rhs_data[2])
        @test forcing_view !== nothing
        @test rhs_data[3] ≈ forcing_view

        @testset "component names map to component state indices" begin
            component_problem = IVP([u, q])
            add_stochastic_forcing!(component_problem, :u_x, forcing)
            @test haskey(component_problem.stochastic_forcings, 1)
        end

        @testset "vector container names are ambiguous" begin
            ambiguous_problem = IVP([u, q])
            @test_throws ArgumentError add_stochastic_forcing!(ambiguous_problem, :u, forcing)
        end
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

    # ------------------------------------------------------------------
    # Regression tests for the shared spectral work pairing.
    # Verifies numerical answers against direct Parseval-normalized references.
    # ------------------------------------------------------------------
    @testset "work_stratonovich regression" begin
        T = Float64
        dt = 0.01
        domain_size = (2π, 2π)
        M2 = 16^4

        forcing = StochasticForcing(
            field_size=(16, 16),
            forcing_rate=0.1,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=dt,
            domain_size=domain_size,
            rng=MersenneTwister(77)
        )

        # Generate forcing so cached_forcing is populated
        generate_forcing!(forcing, 0.0)
        cf = forcing.cached_forcing

        # Create deterministic solution arrays
        Random.seed!(42)
        prev = randn(ComplexF64, 16, 16)
        sol  = randn(ComplexF64, 16, 16)

        # Store previous solution
        store_prevsol!(forcing, prev)

        # -- Test 1: numerical value matches direct spectral reference --
        W = work_stratonovich(forcing, sol)

        # Reference: full-spectrum Parseval pairing.
        ref = sum(real.((prev .+ sol) ./ 2 .* conj.(cf))) * dt / M2
        @test W ≈ ref atol=1e-12 rtol=1e-12
        println("  work_stratonovich matches Parseval reference OK")

        # -- Test 2: returns zero(T) when prevsol is nothing --
        # Temporarily set prevsol to nothing
        forcing.prevsol = nothing
        W_nil = work_stratonovich(forcing, sol)
        @test W_nil === zero(T)
        println("  work_stratonovich returns zero when prevsol===nothing OK")
    end

    @testset "work_ito regression" begin
        T = Float64
        dt = 0.005
        domain_size = (2π, 2π)
        M2 = 16^4
        eps_rate = 0.25  # non-default energy injection rate

        forcing = StochasticForcing(
            field_size=(16, 16),
            forcing_rate=eps_rate,
            k_forcing=4.0,
            dk_forcing=2.0,
            dt=dt,
            domain_size=domain_size,
            rng=MersenneTwister(88)
        )

        # Generate forcing so cached_forcing is populated
        generate_forcing!(forcing, 0.0)
        cf = forcing.cached_forcing

        # Deterministic previous-solution array
        Random.seed!(99)
        sol_prev = randn(ComplexF64, 16, 16)

        # -- Test: numerical value matches Parseval reference with drift --
        W = work_ito(forcing, sol_prev)

        # Reference: broadcast-based Ito work + drift correction
        work_sum = sum(real.(sol_prev .* conj.(cf)))
        drift    = eps_rate * dt
        ref      = work_sum * dt / M2 + drift
        @test W ≈ ref atol=1e-12 rtol=1e-12
        println("  work_ito matches Parseval reference with drift OK")
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

        @testset "GPU self-conjugate projection preserves power" begin
            arch = GPU()
            for (shape, index) in (((8,), (5,)),
                                   ((8, 6), (5, 4)),
                                   ((8, 6, 4), (5, 4, 3)))
                data_cpu = zeros(ComplexF64, shape)
                data_cpu[index...] = 3 + 4im
                data_gpu = CUDA.CuArray(data_cpu)
                Tarang._enforce_hermitian_symmetry!(data_gpu, arch)
                CUDA.synchronize()
                @test Array(data_gpu)[index...] ≈ 3sqrt(2) + 0im
            end
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

        @testset "GPU work diagnostics match manual half-spectrum references" begin
            field_size = (8, 8)
            half_size = (5, 8)
            dt = 0.01
            multiplicity = reshape([1.0, 2.0, 2.0, 2.0, 1.0], :, 1)
            kx = Float64[0, 1, 2, 3, 4]
            ky = Float64[0, 1, 2, 3, 4, -3, -2, -1]
            forcing_cpu = ComplexF64[
                (0.2i - 0.1j) + (0.03i * j)im
                for i in 1:field_size[1], j in 1:field_size[2]
            ]
            forcing_view = @view forcing_cpu[1:half_size[1], :]
            sol_prev_cpu = ComplexF64[
                (0.15i + 0.07j) - (0.02i * j)im
                for i in 1:half_size[1], j in 1:half_size[2]
            ]
            sol_next_cpu = sol_prev_cpu .+ dt .* forcing_view

            for metric in (:direct, :vorticity_kinetic)
                forcing = StochasticForcing(
                    field_size=field_size, forcing_rate=0.1,
                    injection_metric=metric, k_forcing=2.0,
                    dk_forcing=0.1, spectrum_type=:band, dt=dt,
                    architecture=GPU(), rng=MersenneTwister(42),
                )
                copyto!(forcing.cached_forcing, CUDA.CuArray(forcing_cpu))
                sol_prev = CUDA.CuArray(sol_prev_cpu)
                sol_next = CUDA.CuArray(sol_next_cpu)
                store_prevsol!(forcing, sol_prev)

                metric_weight = [
                    metric === :direct ? 1.0 :
                        (iszero(x^2 + y^2) ? 0.0 : inv(x^2 + y^2))
                    for x in kx, y in ky
                ]
                weights = multiplicity .* metric_weight ./ prod(field_size)^2
                pairing(a) = sum(weights .* real.(a .* conj.(forcing_view)))

                @test work_stratonovich(forcing, sol_next) ≈
                      pairing((sol_prev_cpu .+ sol_next_cpu) ./ 2) * dt
                @test work_ito(forcing, sol_prev) ≈
                      pairing(sol_prev_cpu) * dt + forcing.energy_injection_rate * dt
                @test instantaneous_power(forcing, sol_prev) ≈ pairing(sol_prev_cpu)
                cached_weights = forcing.diagnostic_weights
                @test cached_weights !== nothing
                instantaneous_power(forcing, sol_prev)
                @test forcing.diagnostic_weights === cached_weights
            end
        end

    else
        println("\nGPU tests skipped (CUDA not available)")
        @test_skip "GPU construction"
        @test_skip "GPU forcing generation"
        @test_skip "GPU exact self-conjugate projection"
        @test_skip "GPU apply_forcing!"
        @test_skip "GPU work calculation"
        @test_skip "GPU manual work diagnostics"
    end
end

println("\n" * "=" ^ 60)
println("All Stochastic Forcing Tests Completed!")
println("=" ^ 60)
