using Test
using Random
using Tarang

@testset "Separable stochastic forcing" begin
    coords = CartesianCoordinates("z")
    zbasis = ChebyshevT(coords["z"]; size=10, bounds=(0.0, 1.0))

    constructor(; profile=(z -> z * (1 - z)), profile_basis=zbasis,
                injection_metric=:direct) = SeparableStochasticForcing(
        fourier_size=(8,),
        chebyshev_basis=profile_basis,
        chebyshev_profile=profile,
        domain_size=(2pi,),
        energy_injection_rate=0.2,
        injection_metric=injection_metric,
        k_forcing=2.0,
        dk_forcing=0.5,
        dt=1e-2,
        architecture=CPU(),
        rng=MersenneTwister(42),
    )

    @testset "construction and profile validation" begin
        forcing = constructor()
        @test forcing isa StochasticForcingType
        @test size(forcing.cached_forcing) == (8, 10)
        @test size(forcing.fourier_realization) == (8,)
        @test length(forcing.chebyshev_profile) == 10
        @test forcing.injection_metric === :direct
        @test forcing.field_size == (8, 10)
        @test forcing.fourier_size == (8,)
        @test forcing.domain_size == (2pi,)
        @test forcing.is_stochastic
        @test !forcing.is_gpu

        @test_throws ArgumentError constructor(injection_metric=:vorticity_kinetic)
        @test_throws ArgumentError constructor(profile=zeros(10))
        @test_throws ArgumentError constructor(profile=fill(Inf, 10))
        @test_throws DimensionMismatch constructor(profile=ones(9))

        other_coords = CartesianCoordinates("z")
        wrong_basis = ChebyshevU(other_coords["z"]; size=10, bounds=(0.0, 1.0))
        @test_throws ArgumentError constructor(profile_basis=wrong_basis)
    end

    @testset "generation and lifecycle" begin
        forcing = constructor()
        generated = generate_forcing!(forcing, 0.0, 1)

        expected = reshape(forcing.fourier_realization, 8, 1) .*
                   reshape(forcing.chebyshev_profile, 1, 10)
        @test generated === forcing.cached_forcing
        @test generated == expected
        @test all(iszero, @view generated[1, :])
        @test forcing.fourier_realization[2:4] ≈
              conj.(forcing.fourier_realization[8:-1:6])
        @test isreal(forcing.fourier_realization[5])

        first_realization = copy(generated)
        @test generate_forcing!(forcing, 0.0, 1) === generated
        @test generate_forcing!(forcing, 0.0, 2) === generated
        @test generated == first_realization

        generate_forcing!(forcing, 1.0, 1)
        @test generated != first_realization
        @test get_forcing_spectrum(forcing) === forcing.forcing_spectrum
        @test get_cached_forcing(forcing) === forcing.cached_forcing
        @test mean_energy_injection_rate(forcing) == 0.2
        @test energy_injection_rate(forcing) == 0.2

        set_dt!(forcing, 2e-2)
        @test forcing.dt == 2e-2
        @test all(iszero, forcing.cached_forcing)
        generate_forcing!(forcing, 2.0, 1) # warm compilation
        allocations = @allocated generate_forcing!(forcing, 3.0, 1)
        @test allocations <= 4096

        reset_forcing!(forcing)
        @test forcing.last_update_time == -Inf
        @test all(iszero, forcing.cached_forcing)
    end

    @testset "registered Fourier--Chebyshev RHS" begin
        nx, nz = 8, 10
        dt = 1e-2
        coords2 = CartesianCoordinates("x", "z")
        dist = Distributor(coords2; dtype=Float64, device=CPU())
        xbasis = RealFourier(coords2["x"]; size=nx, bounds=(0.0, 2pi))
        zbasis2 = ChebyshevT(coords2["z"]; size=nz, bounds=(0.0, 1.0))
        domain = Domain(dist, (xbasis, zbasis2))

        b = ScalarField(domain, "b")
        tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)
        tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)
        _, ez = unit_vector_fields(coords2, dist)
        lift_basis = derivative_basis(zbasis2, 1)
        tau_lift(A) = lift(A, lift_basis, -1)
        grad_b = grad(b) + ez * tau_lift(tau1)

        problem = IVP([b, tau1, tau2])
        add_parameters!(problem; grad_b, tau_lift)
        add_equation!(problem, "∂t(b) - div(grad_b) + tau_lift(tau2) = 0")
        add_bc!(problem, "b(z=0) = 0")
        add_bc!(problem, "b(z=1) = 0")

        forcing = SeparableStochasticForcing(
            fourier_size=(nx,),
            chebyshev_basis=zbasis2,
            chebyshev_profile=z -> z * (1 - z),
            domain_size=(2pi,),
            energy_injection_rate=0.2,
            k_forcing=2.0,
            dk_forcing=0.5,
            dt=dt,
            architecture=CPU(),
            rng=MersenneTwister(42),
        )
        add_stochastic_forcing!(problem, :b, forcing)
        solver = InitialValueSolver(problem, RK222(); dt)
        @test solver.rhs_plan !== nothing
        @test solver.rhs_plan.is_compiled

        Tarang._update_registered_forcings!(solver, 0.0, dt)
        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
        ensure_layout!(rhs, :c)
        rhs_data = get_coeff_data(rhs)
        forcing_view = Tarang._matched_forcing_view(forcing, rhs_data)

        half_nx = div(nx, 2) + 1
        @test size(rhs_data) == (half_nx, nz)
        @test forcing_view !== nothing
        @test rhs_data ≈ forcing_view
        @test forcing_view == @view(forcing.cached_forcing[1:half_nx, :])
        @test Tarang._matched_forcing_view(forcing, (half_nx, nz - 1)) === nothing

        initial_coeffs = copy(get_coeff_data(b))
        first_forcing = copy(forcing.cached_forcing)
        step!(solver, dt)
        ensure_layout!(b, :c)
        @test all(isfinite, get_coeff_data(b))
        @test get_coeff_data(b) != initial_coeffs
        @test forcing.cached_forcing == first_forcing

        step!(solver, dt)
        @test forcing.cached_forcing != first_forcing
        @test solver.iteration == 2
    end


    @testset "registered 3D Fourier--Fourier--Chebyshev IVP" begin
        nx, ny, nz = 8, 8, 10
        dt = 1e-3
        coords3 = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords3; dtype=Float64, device=CPU())
        xbasis = RealFourier(coords3["x"]; size=nx, bounds=(0.0, 2pi))
        ybasis = RealFourier(coords3["y"]; size=ny, bounds=(0.0, 2pi))
        zbasis3 = ChebyshevT(coords3["z"]; size=nz, bounds=(0.0, 1.0))
        domain = Domain(dist, (xbasis, ybasis, zbasis3))

        b = ScalarField(domain, "b3")
        tau1 = ScalarField(dist, "tau31", (xbasis, ybasis), Float64)
        tau2 = ScalarField(dist, "tau32", (xbasis, ybasis), Float64)
        _, _, ez = unit_vector_fields(coords3, dist)
        lift_basis = derivative_basis(zbasis3, 1)
        tau_lift3(A) = lift(A, lift_basis, -1)
        grad_b3 = grad(b) + ez * tau_lift3(tau1)

        problem = IVP([b, tau1, tau2])
        add_parameters!(problem; kappa=0.1, grad_b3, tau_lift3)
        add_equation!(problem,
                      "∂t(b3) - kappa*div(grad_b3) + tau_lift3(tau32) = 0")
        add_bc!(problem, "b3(z=0) = 0")
        add_bc!(problem, "b3(z=1) = 0")

        forcing = SeparableStochasticForcing(
            fourier_size=(nx, ny),
            chebyshev_basis=zbasis3,
            chebyshev_profile=z -> z * (1 - z),
            domain_size=(2pi, 2pi),
            energy_injection_rate=0.05,
            k_forcing=2.0,
            dk_forcing=0.5,
            dt=dt,
            architecture=CPU(),
            rng=MersenneTwister(2026),
        )
        add_stochastic_forcing!(problem, :b3, forcing)
        solver = InitialValueSolver(problem, RK222(); dt)
        Tarang._update_registered_forcings!(solver, 0.0, dt)

        rhs = Tarang.evaluate_rhs(solver, solver.state, 0.0)[1]
        ensure_layout!(rhs, :c)
        rhs_data = get_coeff_data(rhs)
        forcing_view = Tarang._matched_forcing_view(forcing, rhs_data)
        @test size(rhs_data) == (div(nx, 2) + 1, ny, nz)
        @test forcing_view !== nothing
        @test rhs_data ≈ forcing_view

        initial = copy(get_coeff_data(b))
        step!(solver, dt)
        ensure_layout!(b, :c)
        @test get_coeff_data(b) != initial
        @test all(isfinite, get_coeff_data(b))
        @test solver.iteration == 1
    end
end
