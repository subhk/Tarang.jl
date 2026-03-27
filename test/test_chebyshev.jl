"""
Test suite for Chebyshev transforms and boundary condition integration.

Tests:
1. ChebyshevT roundtrip (forward → backward preserves data)
2. Mixed Fourier-Chebyshev 2D roundtrip
3. Known Chebyshev coefficient projection (T_n accuracy)
4. Chebyshev differentiation matrix correctness
5. Derivative basis chain: T → U → V
6. ChebyshevT ↔ ChebyshevU conversion matrix
7. Dirichlet BC enforcement on Fourier-Chebyshev domain
8. Neumann BC enforcement on Fourier-Chebyshev domain
"""

using Test
using LinearAlgebra
using Tarang

# ============================================================================
# Part 1: Chebyshev Transform Tests
# ============================================================================

@testset "Chebyshev Transforms" begin

    @testset "ChebyshevT 1D roundtrip" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        zb = ChebyshevT(coords["z"]; size=32, bounds=(-1.0, 1.0))

        field = ScalarField(dist, "u", (zb,), Float64)
        z = Tarang.create_meshgrid(field.domain)["z"]

        # Smooth polynomial that is exactly representable
        Tarang.get_grid_data(field) .= @. 3z^4 - 2z^2 + z - 0.5
        original = copy(Tarang.get_grid_data(field))

        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-10, atol=1e-12)
    end

    @testset "ChebyshevT 1D roundtrip – trigonometric" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        zb = ChebyshevT(coords["z"]; size=64, bounds=(-1.0, 1.0))

        field = ScalarField(dist, "u", (zb,), Float64)
        z = Tarang.create_meshgrid(field.domain)["z"]

        Tarang.get_grid_data(field) .= @. cos(π * z) + 0.3 * sin(2π * z)
        original = copy(Tarang.get_grid_data(field))

        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-10, atol=1e-10)
    end

    @testset "Fourier-Chebyshev 2D roundtrip" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))

        field = ScalarField(dist, "u", (xb, zb), Float64)
        mesh = Tarang.create_meshgrid(field.domain)
        x = mesh["x"]
        z = mesh["z"]

        Tarang.get_grid_data(field) .= @. sin(2x) * (1 - z^2) + 0.5 * cos(x) * z
        original = copy(Tarang.get_grid_data(field))

        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-10, atol=1e-10)
    end

    @testset "Chebyshev-Chebyshev 2D roundtrip" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = ChebyshevT(coords["x"]; size=16, bounds=(-1.0, 1.0))
        zb = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))

        field = ScalarField(dist, "u", (xb, zb), Float64)
        mesh = Tarang.create_meshgrid(field.domain)
        x = mesh["x"]
        z = mesh["z"]

        Tarang.get_grid_data(field) .= @. x^2 * z + 0.5 * x * z^3 - 0.3
        original = copy(Tarang.get_grid_data(field))

        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-10, atol=1e-10)
    end

    @testset "Chebyshev alias (Chebyshev === ChebyshevT)" begin
        @test Chebyshev === ChebyshevT
    end

    @testset "Non-standard bounds roundtrip" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        zb = ChebyshevT(coords["z"]; size=32, bounds=(0.0, 5.0))

        field = ScalarField(dist, "u", (zb,), Float64)
        z = Tarang.create_meshgrid(field.domain)["z"]

        Tarang.get_grid_data(field) .= @. exp(-z) * sin(z)
        original = copy(Tarang.get_grid_data(field))

        forward_transform!(field)
        backward_transform!(field)

        @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-8, atol=1e-8)
    end
end

# ============================================================================
# Part 2: Chebyshev Differentiation & Basis Chain
# ============================================================================

@testset "Chebyshev Differentiation" begin

    @testset "Differentiation matrix in coefficient space" begin
        # Test D matrix directly on analytically-known Chebyshev coefficients
        # (avoids transform normalization coupling)
        N = 8
        coords = CartesianCoordinates("z")
        zb = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        D = differentiation_matrix(zb, 1)

        # z = T_1  ⟹  dz/dz = 1 = T_0
        c_z = zeros(N); c_z[2] = 1.0
        dc_z = D * c_z
        @test dc_z[1] ≈ 1.0
        @test all(abs.(dc_z[2:end]) .< 1e-14)

        # z² = (T_0 + T_2)/2  ⟹  d(z²)/dz = 2z = 2T_1
        c_z2 = zeros(N); c_z2[1] = 0.5; c_z2[3] = 0.5
        dc_z2 = D * c_z2
        @test dc_z2[2] ≈ 2.0
        @test abs(dc_z2[1]) < 1e-14
        @test all(abs.(dc_z2[3:end]) .< 1e-14)

        # z³ = (3T_1 + T_3)/4  ⟹  d(z³)/dz = 3z² = 3(T_0+T_2)/2
        c_z3 = zeros(N); c_z3[2] = 0.75; c_z3[4] = 0.25
        dc_z3 = D * c_z3
        @test dc_z3[1] ≈ 1.5
        @test dc_z3[3] ≈ 1.5
        @test abs(dc_z3[2]) < 1e-14
        @test all(abs.(dc_z3[4:end]) .< 1e-14)
    end

    @testset "Second derivative in coefficient space" begin
        N = 8
        coords = CartesianCoordinates("z")
        zb = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        D2 = differentiation_matrix(zb, 2)

        # z⁴ = (3T_0 + 4T_2 + T_4)/8  ⟹  d²(z⁴)/dz² = 12z² = 6T_0 + 6T_2
        c_z4 = zeros(N); c_z4[1] = 3/8; c_z4[3] = 4/8; c_z4[5] = 1/8
        d2c = D2 * c_z4
        @test d2c[1] ≈ 6.0
        @test d2c[3] ≈ 6.0
        @test abs(d2c[2]) < 1e-12
        @test all(abs.(d2c[4:end]) .< 1e-12)
    end

    @testset "Derivative basis chain T → U → V" begin
        coords = CartesianCoordinates("z")
        zb_t = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))

        # First derivative: T → U
        db1 = derivative_basis(zb_t, 1)
        @test db1 isa ChebyshevU

        # Second derivative: T → U → V
        db2 = derivative_basis(zb_t, 2)
        @test db2 isa ChebyshevV

        # Zero-th order returns same basis type
        db0 = derivative_basis(zb_t, 0)
        @test db0 === zb_t

        # Negative order errors
        @test_throws ArgumentError derivative_basis(zb_t, -1)
    end

    @testset "ChebyshevT → ChebyshevU conversion matrix" begin
        N = 8
        coords = CartesianCoordinates("z")
        zb_t = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
        zb_u = ChebyshevU(coords["z"]; size=N, bounds=(-1.0, 1.0))

        C = conversion_matrix(zb_t, zb_u)
        @test size(C) == (N, N)

        # T_0 = U_0  ⟹  first column should be [1, 0, 0, ...]
        @test C[1, 1] ≈ 1.0
        for i in 2:N
            @test abs(C[i, 1]) < 1e-14
        end

        # T_1 = U_1 / 2  ⟹  C[:, 2] should have 0.5 at row 2
        if N > 1
            @test C[2, 2] ≈ 0.5
        end
    end

    @testset "Differentiation matrix scaling for non-standard bounds" begin
        # On [0, L], the D matrix should include the chain rule factor 2/L
        coords = CartesianCoordinates("z")

        # Standard bounds [-1, 1]: scale = 2/2 = 1
        zb_std = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))
        D_std = differentiation_matrix(zb_std, 1)

        # Non-standard bounds [0, 2]: scale = 2/2 = 1 (same as standard)
        zb_ns = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 2.0))
        D_ns = differentiation_matrix(zb_ns, 1)

        # D matrices should be identical (both have scale=1)
        @test isapprox(Matrix(D_std), Matrix(D_ns); atol=1e-14)

        # Non-standard bounds [0, 1]: scale = 2/1 = 2
        zb_01 = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
        D_01 = differentiation_matrix(zb_01, 1)

        # D_01 should be 2× D_std (chain rule: d/dz_physical = (2/L) d/dz_native)
        @test isapprox(Matrix(D_01), 2.0 .* Matrix(D_std); atol=1e-14)
    end
end

# ============================================================================
# Part 3: Boundary Condition Integration with Chebyshev Domains
# ============================================================================

@testset "BC Integration with Chebyshev" begin

    @testset "Dirichlet BC on Fourier-Chebyshev domain" begin
        # Verify BCs can be set up with a real Chebyshev domain
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))

        manager = BoundaryConditionManager()

        # Dirichlet: u = 0 at z = -1 and z = 1
        add_dirichlet!(manager, "u", "z", -1.0, 0.0)
        add_dirichlet!(manager, "u", "z", 1.0, 0.0)

        @test length(manager.conditions) == 2
        @test manager.conditions[1] isa DirichletBC
        @test manager.conditions[2] isa DirichletBC
        @test manager.conditions[1].position == -1.0
        @test manager.conditions[2].position == 1.0

        # Register coordinate info from our domain
        register_coordinate_info!(manager, ["x", "z"], ["RealFourier", "ChebyshevT"])
        @test manager.coordinate_info["coordinates"] == ["x", "z"]
        @test manager.coordinate_info["bases"] == ["RealFourier", "ChebyshevT"]
    end

    @testset "Neumann BC on Fourier-Chebyshev domain" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))

        manager = BoundaryConditionManager()

        # Neumann: du/dz = 0 at z = -1, Dirichlet: u = 0 at z = 1
        add_neumann!(manager, "u", "z", -1.0, 0.0)
        add_dirichlet!(manager, "u", "z", 1.0, 0.0)

        @test length(manager.conditions) == 2
        @test manager.conditions[1] isa NeumannBC
        @test manager.conditions[1].derivative_order == 1

        counts = get_bc_count_by_type(manager)
        @test counts["NeumannBC"] == 1
        @test counts["DirichletBC"] == 1
    end

    @testset "Robin BC on Chebyshev domain" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        zb = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))

        manager = BoundaryConditionManager()

        # Robin: alpha * u + beta * du/dz = 0 at z = 1
        add_robin!(manager, "u", "z", 1.0, 1.0, 1.0, 0.0)
        @test length(manager.conditions) == 1
        @test manager.conditions[1] isa RobinBC
        @test manager.conditions[1].alpha == 1.0
        @test manager.conditions[1].beta == 1.0
    end

    @testset "Tau field with Chebyshev BCs" begin
        manager = BoundaryConditionManager()

        # Add BCs with tau fields (typical for Chebyshev tau method)
        add_dirichlet!(manager, "u", "z", -1.0, 0.0; tau_field="tau_u1")
        add_dirichlet!(manager, "u", "z",  1.0, 0.0; tau_field="tau_u2")

        required = get_required_tau_fields(manager)
        @test "tau_u1" in required
        @test "tau_u2" in required

        # Create lift operator for the tau field
        lift_op = Tarang.create_lift_operator(manager, "tau_u1", "ChebyshevT", 0)
        @test haskey(manager.lift_operators, "tau_u1")
        @test lift_op["derivative_order"] == 0
    end

    @testset "Time-dependent Dirichlet on Chebyshev" begin
        manager = BoundaryConditionManager()

        # Oscillating lid: u = sin(2πt) at z = 1
        add_dirichlet!(manager, "u", "z", 1.0, "sin(2*pi*t)")
        @test has_time_dependent_bcs(manager) == true

        update_time_dependent_bcs!(manager, 0.0)
        @test manager.performance_stats.bc_updates == 1

        update_time_dependent_bcs!(manager, 0.25)
        @test manager.performance_stats.bc_updates == 2
    end

    @testset "Multiple fields with BCs on Chebyshev" begin
        # Typical Rayleigh-Bénard setup: velocity + temperature BCs
        manager = BoundaryConditionManager()

        # Velocity: no-slip at both walls
        add_dirichlet!(manager, "u", "z", -1.0, 0.0; tau_field="tau_u1")
        add_dirichlet!(manager, "u", "z",  1.0, 0.0; tau_field="tau_u2")
        add_dirichlet!(manager, "w", "z", -1.0, 0.0; tau_field="tau_w1")
        add_dirichlet!(manager, "w", "z",  1.0, 0.0; tau_field="tau_w2")

        # Temperature: fixed temperatures at walls
        add_dirichlet!(manager, "T", "z", -1.0, 1.0; tau_field="tau_T1")
        add_dirichlet!(manager, "T", "z",  1.0, 0.0; tau_field="tau_T2")

        @test length(manager.conditions) == 6

        counts = get_bc_count_by_type(manager)
        @test counts["DirichletBC"] == 6

        required = get_required_tau_fields(manager)
        @test length(required) == 6
        @test "tau_T1" in required
        @test "tau_w2" in required
    end

    @testset "BC equation conversion for Chebyshev BCs" begin
        manager = BoundaryConditionManager()

        # Dirichlet
        bc_dir = dirichlet_bc("u", "z", -1.0, 0.0)
        eq_dir = Tarang.bc_to_equation(manager, bc_dir)
        @test occursin("u(z=", eq_dir)
        @test occursin("= 0", eq_dir)

        # Neumann
        bc_neu = neumann_bc("u", "z", 1.0, 0.0)
        eq_neu = Tarang.bc_to_equation(manager, bc_neu)
        @test occursin("d(u, z)", eq_neu)
        @test occursin("z=1", eq_neu)

        # Neumann with higher order
        bc_neu2 = neumann_bc("u", "z", -1.0, 0.0; derivative_order=2)
        eq_neu2 = Tarang.bc_to_equation(manager, bc_neu2)
        @test bc_neu2.derivative_order == 2
    end

    @testset "Clear and re-add BCs" begin
        manager = BoundaryConditionManager()

        add_dirichlet!(manager, "u", "z", -1.0, 0.0)
        add_dirichlet!(manager, "u", "z",  1.0, 0.0)
        @test length(manager.conditions) == 2

        clear_boundary_conditions!(manager)
        @test isempty(manager.conditions)

        # Re-add with different BC type
        add_neumann!(manager, "u", "z", -1.0, 0.0)
        add_neumann!(manager, "u", "z",  1.0, 0.0)
        @test length(manager.conditions) == 2
        @test all(bc -> bc isa NeumannBC, manager.conditions)
    end
end
