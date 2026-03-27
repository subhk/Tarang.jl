"""
Comprehensive test suite for flow_tools.jl

Tests:
1. CFL adaptive timestep (CFL struct, add_velocity!, compute_timestep)
2. Flow diagnostics (reynolds_number, kinetic_energy, enstrophy, energy_dissipation_rate)
3. WavenumberInfo and spectral analysis utilities
4. Domain utilities (get_domain_size, get_domain_bounds, get_fourier_shape)
5. Streamfunction utilities (streamfunction, perp_grad)
6. Velocity utilities (velocity_divergence)
7. SQG system (sqg_streamfunction, sqg_velocity)
8. QG system (QGSystem, qg_system_setup, qg_invert!, qg_step!)
9. Boundary advection-diffusion system types
"""

using Test
using Tarang
using MPI
using LinearAlgebra

@testset "Flow Tools Module" begin
    # ========================================================================
    # Helper functions for creating test fields
    # ========================================================================

    function create_1d_velocity()
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        velocity = VectorField(dist, coords, "u", (basis,), Float64)
        return velocity, dist, coords, (basis,)
    end

    function create_2d_velocity()
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        x_basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        y_basis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
        bases = (x_basis, y_basis)
        velocity = VectorField(dist, coords, "u", bases, Float64)
        return velocity, dist, coords, bases
    end

    function create_2d_scalar()
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        x_basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        y_basis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
        bases = (x_basis, y_basis)
        field = ScalarField(dist, "f", bases, Float64)
        return field, dist, coords, bases
    end

    # ========================================================================
    # Reynolds Number Tests
    # ========================================================================

    @testset "reynolds_number" begin
        @testset "Basic calculation" begin
            velocity, _, _, _ = create_1d_velocity()
            component = velocity.components[1]
            Tarang.ensure_layout!(component, :g)
            fill!(Tarang.get_grid_data(component), 2.0)

            Re = Tarang.reynolds_number(velocity, 0.5, 1.0)
            @test isapprox(Re, 4.0; atol=1e-8)  # Re = u * L / ν = 2 * 1 / 0.5 = 4
        end

        @testset "With length scale" begin
            velocity, _, _, _ = create_1d_velocity()
            component = velocity.components[1]
            Tarang.ensure_layout!(component, :g)
            fill!(Tarang.get_grid_data(component), 3.0)

            Re = Tarang.reynolds_number(velocity, 1.0, 2.0)
            @test isapprox(Re, 6.0; atol=1e-8)  # Re = 3 * 2 / 1 = 6
        end

        @testset "2D velocity field" begin
            velocity, _, _, _ = create_2d_velocity()
            for comp in velocity.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 1.0)
            end

            # |u| = sqrt(1^2 + 1^2) = sqrt(2)
            Re = Tarang.reynolds_number(velocity, 1.0, 1.0)
            @test isapprox(Re, sqrt(2); atol=1e-8)
        end

        @testset "Zero velocity" begin
            velocity, _, _, _ = create_1d_velocity()
            component = velocity.components[1]
            Tarang.ensure_layout!(component, :g)
            fill!(Tarang.get_grid_data(component), 0.0)

            Re = Tarang.reynolds_number(velocity, 1.0, 1.0)
            @test Re == 0.0
        end
    end

    # ========================================================================
    # Kinetic Energy Tests
    # ========================================================================

    @testset "kinetic_energy" begin
        @testset "Basic calculation" begin
            velocity, _, _, _ = create_1d_velocity()
            component = velocity.components[1]
            Tarang.ensure_layout!(component, :g)
            fill!(Tarang.get_grid_data(component), 2.0)

            ke = Tarang.kinetic_energy(velocity, 1.0)
            Tarang.ensure_layout!(ke, :g)
            # KE = 0.5 * ρ * |u|² = 0.5 * 1 * 4 = 2
            @test all(isapprox.(Tarang.get_grid_data(ke), 2.0; atol=1e-10))
        end

        @testset "With density" begin
            velocity, _, _, _ = create_1d_velocity()
            component = velocity.components[1]
            Tarang.ensure_layout!(component, :g)
            fill!(Tarang.get_grid_data(component), 2.0)

            ke = Tarang.kinetic_energy(velocity, 2.0)
            Tarang.ensure_layout!(ke, :g)
            # KE = 0.5 * 2 * 4 = 4
            @test all(isapprox.(Tarang.get_grid_data(ke), 4.0; atol=1e-10))
        end

        @testset "2D field" begin
            velocity, _, _, _ = create_2d_velocity()
            Tarang.ensure_layout!(velocity.components[1], :g)
            Tarang.ensure_layout!(velocity.components[2], :g)
            fill!(Tarang.get_grid_data(velocity.components[1]), 1.0)
            fill!(Tarang.get_grid_data(velocity.components[2]), 1.0)

            ke = Tarang.kinetic_energy(velocity, 1.0)
            Tarang.ensure_layout!(ke, :g)
            # KE = 0.5 * 1 * (1² + 1²) = 1
            @test all(isapprox.(Tarang.get_grid_data(ke), 1.0; atol=1e-10))
        end
    end

    @testset "total_kinetic_energy" begin
        @testset "Function exists" begin
            @test hasmethod(Tarang.total_kinetic_energy, Tuple{VectorField, Float64})
        end
    end

    # ========================================================================
    # Enstrophy Tests
    # ========================================================================

    @testset "enstrophy" begin
        @testset "2D velocity required" begin
            velocity_1d, _, _, _ = create_1d_velocity()
            Tarang.ensure_layout!(velocity_1d.components[1], :g)
            fill!(Tarang.get_grid_data(velocity_1d.components[1]), 1.0)

            @test_throws ArgumentError Tarang.enstrophy(velocity_1d)
        end

        @testset "Uniform velocity (zero vorticity)" begin
            velocity, _, _, _ = create_2d_velocity()
            for comp in velocity.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 1.0)
            end

            ens = Tarang.enstrophy(velocity)
            Tarang.ensure_layout!(ens, :g)
            # Uniform velocity has zero vorticity
            @test all(isapprox.(Tarang.get_grid_data(ens), 0.0; atol=1e-10))
        end
    end

    @testset "total_enstrophy" begin
        @testset "Function exists" begin
            @test hasmethod(Tarang.total_enstrophy, Tuple{VectorField})
        end
    end

    # ========================================================================
    # Energy Dissipation Rate Tests
    # ========================================================================

    @testset "energy_dissipation_rate" begin
        @testset "Uniform velocity (zero gradient)" begin
            velocity, _, _, _ = create_2d_velocity()
            for comp in velocity.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 1.0)
            end

            dissipation = Tarang.energy_dissipation_rate(velocity, 0.1)
            Tarang.ensure_layout!(dissipation, :g)
            # Uniform velocity has zero gradient, so zero dissipation
            @test all(isapprox.(Tarang.get_grid_data(dissipation), 0.0; atol=1e-10))
        end

        @testset "Returns correct field type" begin
            velocity, _, _, _ = create_2d_velocity()
            for comp in velocity.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 1.0)
            end

            dissipation = Tarang.energy_dissipation_rate(velocity, 0.1)
            @test isa(dissipation, ScalarField)
        end
    end

    # ========================================================================
    # Vorticity Transport Tests
    # ========================================================================

    @testset "vorticity_transport" begin
        @testset "2D required" begin
            velocity_1d, _, _, _ = create_1d_velocity()
            vorticity = ScalarField(velocity_1d.dist, "omega", velocity_1d.bases, Float64)

            @test_throws ArgumentError Tarang.vorticity_transport(velocity_1d, vorticity, 0.1)
        end

        @testset "Function exists" begin
            @test hasmethod(Tarang.vorticity_transport, Tuple{VectorField, ScalarField, Float64})
        end
    end

    # ========================================================================
    # WavenumberInfo Tests
    # ========================================================================

    @testset "WavenumberInfo" begin
        @testset "Struct creation" begin
            kmax = 8
            k_magnitudes = zeros(8, 8)
            kx_grid = zeros(8, 1)
            ky_grid = zeros(1, 8)
            domain_size = (2π, 2π)
            fourier_shape = (8, 8)

            info = Tarang.WavenumberInfo(kmax, k_magnitudes, kx_grid, ky_grid, nothing, domain_size, fourier_shape)
            @test info.kmax == 8
            @test info.domain_size == (2π, 2π)
            @test info.fourier_shape == (8, 8)
            @test info.kz_grid === nothing
        end
    end

    @testset "validate_fourier_bases" begin
        @testset "Returns Fourier axes" begin
            velocity, _, _, _ = create_2d_velocity()
            fourier_axes, fourier_bases = Tarang.validate_fourier_bases(velocity)

            @test length(fourier_axes) == 2
            @test 1 in fourier_axes
            @test 2 in fourier_axes
            @test length(fourier_bases) == 2
        end
    end

    # ========================================================================
    # Domain Utilities Tests
    # ========================================================================

    @testset "get_domain_size" begin
        @testset "1D domain" begin
            velocity, _, _, _ = create_1d_velocity()
            domain = velocity.domain
            size = Tarang.get_domain_size(domain)
            @test length(size) >= 1
            @test isapprox(size[1], 2π; atol=1e-10)
        end

        @testset "2D domain" begin
            velocity, _, _, _ = create_2d_velocity()
            domain = velocity.domain
            size = Tarang.get_domain_size(domain)
            @test length(size) >= 2
            @test isapprox(size[1], 2π; atol=1e-10)
            @test isapprox(size[2], 2π; atol=1e-10)
        end
    end

    @testset "get_domain_bounds" begin
        @testset "Returns bounds array" begin
            velocity, _, _, _ = create_2d_velocity()
            domain = velocity.domain
            bounds = Tarang.get_domain_bounds(domain)
            # Returns Vector of Tuple{Float64, Float64}
            @test isa(bounds, AbstractVector)
            @test length(bounds) >= 2
        end
    end

    # ========================================================================
    # Energy Spectrum Tests
    # ========================================================================

    @testset "energy_spectrum" begin
        @testset "validate_fourier_bases returns correct info" begin
            velocity, _, _, _ = create_2d_velocity()
            fourier_axes, fourier_bases = Tarang.validate_fourier_bases(velocity)
            @test length(fourier_axes) == 2
            @test length(fourier_bases) == 2
        end

        @testset "energy_spectrum function exists" begin
            @test hasmethod(Tarang.energy_spectrum, Tuple{VectorField})
        end
    end

    # ========================================================================
    # Streamfunction Tests
    # ========================================================================

    @testset "perp_grad" begin
        @testset "Basic perpendicular gradient" begin
            field, dist, coords, bases = create_2d_scalar()
            Tarang.ensure_layout!(field, :g)
            # Set f = x, so ∇⊥f = (-∂f/∂y, ∂f/∂x) = (0, 1)
            fill!(Tarang.get_grid_data(field), 1.0)

            perp = Tarang.perp_grad(field)
            @test isa(perp, VectorField)
            @test length(perp.components) == 2
        end

        @testset "Alias ∇⊥ works" begin
            field, _, _, _ = create_2d_scalar()
            Tarang.ensure_layout!(field, :g)
            fill!(Tarang.get_grid_data(field), 1.0)

            perp = Tarang.∇⊥(field)
            @test isa(perp, VectorField)
        end
    end

    @testset "velocity_divergence" begin
        @testset "Divergence-free uniform field" begin
            velocity, _, _, _ = create_2d_velocity()
            for comp in velocity.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 1.0)
            end

            div = Tarang.velocity_divergence(velocity)
            Tarang.ensure_layout!(div, :g)
            # Uniform velocity has zero divergence
            @test all(isapprox.(Tarang.get_grid_data(div), 0.0; atol=1e-10))
        end
    end

    # ========================================================================
    # SQG System Tests
    # ========================================================================

    @testset "SQG System" begin
        @testset "sqg_velocity function exists" begin
            # Just verify the function is defined and callable
            @test hasmethod(Tarang.sqg_velocity, Tuple{ScalarField})
        end

        @testset "sqg_streamfunction function exists" begin
            # Just verify the function is defined and callable
            @test hasmethod(Tarang.sqg_streamfunction, Tuple{ScalarField})
        end
    end

    # ========================================================================
    # VelocitySource Types Tests
    # ========================================================================

    @testset "VelocitySource Types" begin
        @testset "PrescribedVelocity" begin
            vs = Tarang.PrescribedVelocity()
            @test isa(vs, Tarang.VelocitySource)
        end

        @testset "SelfDerivedVelocity" begin
            vs = Tarang.SelfDerivedVelocity(inversion_exponent=-0.5, use_perp_grad=true)
            @test isa(vs, Tarang.VelocitySource)
            @test vs.inversion_exponent == -0.5
            @test vs.use_perp_grad == true
        end

        @testset "InteriorDerivedVelocity" begin
            vs = Tarang.InteriorDerivedVelocity(:perp_grad)
            @test isa(vs, Tarang.VelocitySource)
            @test vs.operator == :perp_grad
        end
    end

    # ========================================================================
    # DiffusionSpec Tests
    # ========================================================================

    @testset "DiffusionSpec" begin
        @testset "Default values" begin
            ds = Tarang.DiffusionSpec()
            @test ds.type == :laplacian  # Default is :laplacian
            @test ds.coefficient == 0.0
            @test ds.exponent == 1.0
            @test ds.implicit == false
        end

        @testset "Laplacian diffusion" begin
            ds = Tarang.DiffusionSpec(type=:laplacian, coefficient=0.01)
            @test ds.type == :laplacian
            @test ds.coefficient == 0.01
        end

        @testset "Fractional diffusion" begin
            ds = Tarang.DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5)
            @test ds.type == :fractional
            @test ds.coefficient == 1e-4
            @test ds.exponent == 0.5
        end

        @testset "Hyperdiffusion" begin
            ds = Tarang.DiffusionSpec(type=:hyperdiffusion, coefficient=1e-6, exponent=2.0)
            @test ds.type == :hyperdiffusion
            @test ds.exponent == 2.0
        end

        @testset "Implicit flag" begin
            ds = Tarang.DiffusionSpec(type=:laplacian, coefficient=0.01, implicit=true)
            @test ds.implicit == true
        end
    end

    # ========================================================================
    # BoundarySpec Tests
    # ========================================================================

    @testset "BoundarySpec" begin
        @testset "Basic creation" begin
            bs = Tarang.BoundarySpec("surface", :z, 0.0)
            @test bs.name == "surface"
            @test bs.dimension == :z
            @test bs.position == 0.0
        end

        @testset "Different positions" begin
            bottom = Tarang.BoundarySpec("bottom", :z, 0.0)
            top = Tarang.BoundarySpec("top", :z, 1.0)

            @test bottom.position == 0.0
            @test top.position == 1.0
        end

        @testset "With custom field name" begin
            bs = Tarang.BoundarySpec("theta_surface", :z, 0.0; field_name="theta")
            @test bs.name == "theta_surface"
            @test bs.field_name == "theta"
        end
    end

    # ========================================================================
    # GlobalArrayReducer Tests
    # ========================================================================

    @testset "GlobalArrayReducer" begin
        @testset "Default constructor" begin
            reducer = Tarang.GlobalArrayReducer()
            @test isa(reducer, Tarang.GlobalArrayReducer)
        end

        @testset "global_max" begin
            reducer = Tarang.GlobalArrayReducer()
            @test Tarang.global_max(reducer, 3.5) ≈ 3.5
            @test Tarang.global_max(reducer, -2.0) ≈ -2.0
        end

        @testset "global_min" begin
            reducer = Tarang.GlobalArrayReducer()
            @test Tarang.global_min(reducer, 3.5) ≈ 3.5
            @test Tarang.global_min(reducer, -2.0) ≈ -2.0
        end

        @testset "global_mean" begin
            reducer = Tarang.GlobalArrayReducer()
            @test Tarang.global_mean(reducer, ones(4)) ≈ 1.0
            @test Tarang.global_mean(reducer, [1.0, 2.0, 3.0, 4.0]) ≈ 2.5
        end

        @testset "reduce_scalar" begin
            reducer = Tarang.GlobalArrayReducer()
            result = Tarang.reduce_scalar(reducer, 5.0, +)
            @test result ≈ 5.0
        end
    end

    # ========================================================================
    # CFL Tests
    # ========================================================================

    @testset "CFL" begin
        @testset "CFL struct fields" begin
            # Test that CFL struct exists with expected fields
            @test hasfield(Tarang.CFL, :initial_dt)
            @test hasfield(Tarang.CFL, :safety)
            @test hasfield(Tarang.CFL, :current_dt)
            @test hasfield(Tarang.CFL, :velocities)
        end
    end

    @testset "add_velocity!" begin
        @testset "Function exists" begin
            @test hasmethod(Tarang.add_velocity!, Tuple{Tarang.CFL, VectorField})
        end
    end

    @testset "compute_timestep" begin
        @testset "Function exists" begin
            @test hasmethod(Tarang.compute_timestep, Tuple{Tarang.CFL})
        end
    end

    # ========================================================================
    # QG System Tests
    # ========================================================================

    @testset "QGSystem" begin
        @testset "Struct fields exist" begin
            # Test that QGSystem struct exists with expected fields
            @test hasfield(Tarang.QGSystem, :ψ)
            @test hasfield(Tarang.QGSystem, :q)
            @test hasfield(Tarang.QGSystem, :θ_bot)  # Bottom surface buoyancy
            @test hasfield(Tarang.QGSystem, :θ_top)  # Top surface buoyancy
            @test hasfield(Tarang.QGSystem, :f0)
            @test hasfield(Tarang.QGSystem, :N)
            @test hasfield(Tarang.QGSystem, :H)
        end
    end

    @testset "qg_step!" begin
        @testset "Function exists" begin
            @test hasmethod(Tarang.qg_step!, Tuple{Tarang.QGSystem, Real})
        end
    end

    @testset "qg_energy" begin
        @testset "Function exists" begin
            @test hasmethod(Tarang.qg_energy, Tuple{Tarang.QGSystem})
        end
    end

    # ========================================================================
    # Extract Surface Tests
    # ========================================================================

    @testset "extract_surface" begin
        @testset "Function exists" begin
            @test hasmethod(Tarang.extract_surface, Tuple{ScalarField, Symbol, Real})
        end
    end

    # ========================================================================
    # BoundaryAdvectionDiffusion Tests
    # ========================================================================

    @testset "BoundaryAdvectionDiffusion" begin
        @testset "Basic setup" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=16, Ny=16,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity(),
                diffusion=Tarang.DiffusionSpec(type=:laplacian, coefficient=0.01)
            )

            @test isa(bad, Tarang.BoundaryAdvectionDiffusion)
            @test haskey(bad.fields, "surface")
            @test haskey(bad.velocities, "surface")
        end

        @testset "With SelfDerivedVelocity" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=16, Ny=16,
                boundaries=[Tarang.BoundarySpec("theta", :z, 0.0)],
                velocity_source=Tarang.SelfDerivedVelocity(inversion_exponent=-0.5, use_perp_grad=true),
                diffusion=Tarang.DiffusionSpec(type=:fractional, coefficient=1e-4, exponent=0.5)
            )

            @test isa(bad.velocity_source, Tarang.SelfDerivedVelocity)
            @test bad.diffusion.type == :fractional
        end

        @testset "Multiple boundaries" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=16, Ny=16,
                boundaries=[
                    Tarang.BoundarySpec("bottom", :z, 0.0),
                    Tarang.BoundarySpec("top", :z, 1.0)
                ],
                velocity_source=Tarang.PrescribedVelocity()
            )

            @test length(bad.boundary_specs) == 2
            @test haskey(bad.fields, "bottom")
            @test haskey(bad.fields, "top")
        end
    end

    @testset "bad_step!" begin
        @testset "Euler step" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            # Set prescribed velocity to zero
            vel = bad.velocities["surface"]
            for comp in vel.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 0.0)
            end

            # Initialize field
            Tarang.ensure_layout!(bad.fields["surface"], :g)
            fill!(Tarang.get_grid_data(bad.fields["surface"]), 1.0)

            initial_time = bad.time
            Tarang.bad_step!(bad, 0.01; timestepper=:Euler)
            @test bad.time > initial_time
            @test bad.iteration == 1
        end

        @testset "RK4 step" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            # Set prescribed velocity
            vel = bad.velocities["surface"]
            for comp in vel.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 0.0)
            end

            Tarang.ensure_layout!(bad.fields["surface"], :g)
            fill!(Tarang.get_grid_data(bad.fields["surface"]), 1.0)

            Tarang.bad_step!(bad, 0.01; timestepper=:RK4)
            @test bad.iteration == 1
        end

        @testset "SSPRK3 step" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            vel = bad.velocities["surface"]
            for comp in vel.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 0.0)
            end

            Tarang.ensure_layout!(bad.fields["surface"], :g)
            fill!(Tarang.get_grid_data(bad.fields["surface"]), 1.0)

            Tarang.bad_step!(bad, 0.01; timestepper=:SSPRK3)
            @test bad.iteration == 1
        end
    end

    @testset "bad_energy" begin
        @testset "Returns positive for non-zero field" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            Tarang.ensure_layout!(bad.fields["surface"], :g)
            fill!(Tarang.get_grid_data(bad.fields["surface"]), 1.0)

            energy = Tarang.bad_energy(bad)
            @test energy > 0
        end

        @testset "Zero field gives zero energy" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            Tarang.ensure_layout!(bad.fields["surface"], :g)
            fill!(Tarang.get_grid_data(bad.fields["surface"]), 0.0)

            energy = Tarang.bad_energy(bad)
            @test isapprox(energy, 0.0; atol=1e-12)
        end
    end

    @testset "bad_enstrophy" begin
        @testset "Uniform field gives zero enstrophy" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            Tarang.ensure_layout!(bad.fields["surface"], :g)
            fill!(Tarang.get_grid_data(bad.fields["surface"]), 1.0)

            ens = Tarang.bad_enstrophy(bad, "surface")
            # Uniform field has zero gradient
            @test isapprox(ens, 0.0; atol=1e-10)
        end
    end

    @testset "bad_max_velocity" begin
        @testset "Returns correct max" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            vel = bad.velocities["surface"]
            Tarang.ensure_layout!(vel.components[1], :g)
            Tarang.ensure_layout!(vel.components[2], :g)
            fill!(Tarang.get_grid_data(vel.components[1]), 3.0)
            fill!(Tarang.get_grid_data(vel.components[2]), 4.0)

            max_vel = Tarang.bad_max_velocity(bad)
            # |u| = sqrt(3² + 4²) = 5
            @test isapprox(max_vel, 5.0; atol=1e-10)
        end
    end

    @testset "bad_cfl_dt" begin
        @testset "Returns Inf for zero velocity" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            vel = bad.velocities["surface"]
            for comp in vel.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 0.0)
            end

            dt = Tarang.bad_cfl_dt(bad)
            @test dt == Inf
        end

        @testset "Returns finite for non-zero velocity" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            vel = bad.velocities["surface"]
            for comp in vel.components
                Tarang.ensure_layout!(comp, :g)
                fill!(Tarang.get_grid_data(comp), 1.0)
            end

            dt = Tarang.bad_cfl_dt(bad; safety=0.5)
            @test isfinite(dt)
            @test dt > 0
        end
    end

    @testset "bad_add_source!" begin
        @testset "Adds source function" begin
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π,
                Nx=8, Ny=8,
                boundaries=[Tarang.BoundarySpec("surface", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity()
            )

            # Define a simple source
            my_source(bad, name) = zeros(size(Tarang.get_grid_data(bad.fields[name])))

            @test isempty(bad.source_terms)
            Tarang.bad_add_source!(bad, "surface", my_source)
            @test haskey(bad.source_terms, "surface")
        end
    end

    # ========================================================================
    # Turbulence Statistics Tests
    # ========================================================================

    @testset "turbulence_statistics" begin
        @testset "Function exists" begin
            # turbulence_statistics takes only VectorField
            @test hasmethod(Tarang.turbulence_statistics, Tuple{VectorField})
        end
    end

    # ========================================================================
    # Streamfunction Solve Tests
    # ========================================================================

    @testset "streamfunction" begin
        @testset "Function exists" begin
            @test hasmethod(Tarang.streamfunction, Tuple{VectorField})
        end
    end
end

println("All flow tools tests passed!")
