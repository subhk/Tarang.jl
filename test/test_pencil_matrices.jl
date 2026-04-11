using Test
using Tarang
using LinearAlgebra

@testset "Pencil Matrices" begin

@testset "Pencil Laplacian block" begin
    # Create a Chebyshev basis for testing
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)
    zbasis = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))

    # At kx=0: Laplacian is pure D²_z
    lap0 = Tarang.pencil_laplacian_block(0.0, zbasis)
    @test size(lap0) == (16, 16)
    # Eigenvalues of D² should be real and ≤ 0
    evals0 = eigvals(lap0)
    @test all(real.(evals0) .<= 1e-10)

    # At kx=2π: Laplacian is -kx² I + D²_z, should be more negative
    kx = 2π
    lap_k = Tarang.pencil_laplacian_block(kx, zbasis)
    @test size(lap_k) == (16, 16)
    evals_k = eigvals(lap_k)
    @test all(real.(evals_k) .< real(evals0[1]))  # more negative
end

@testset "Pencil gather/scatter roundtrip" begin
    domain = ChannelDomain(32, 8; Lx=2π, Lz=1.0)
    u = ScalarField(domain, "u")
    fill_random!(u, "g"; seed=42, distribution="normal", scale=1.0)
    ensure_layout!(u, :c)
    original_data = copy(get_coeff_data(u))

    state = [u]
    xb, zb = domain.bases
    pms = PencilMatrixSystem(state, xb, zb)

    # Scatter to pencils
    pencils = Tarang.state_to_pencils(state, pms)
    @test length(pencils) == pms.n_pencils
    @test length(pencils[1]) == pms.pencil_size

    # Gather back
    state2 = [copy(u)]
    Tarang.pencils_to_state!(state2, pencils, pms)

    # Data should match
    @test get_coeff_data(state2[1]) ≈ original_data
end

@testset "PencilMatrixSystem constructor" begin
    domain = ChannelDomain(64, 16; Lx=4.0, Lz=1.0)
    p = ScalarField(domain, "p")
    b = ScalarField(domain, "b")

    state = [p, b]
    xb, zb = domain.bases
    pms = PencilMatrixSystem(state, xb, zb)

    @test pms.n_pencils == 64 ÷ 2 + 1  # RealFourier: N/2+1 = 33
    @test pms.n_cheb == 16
    @test pms.n_scalar_fields == 2
    @test pms.pencil_size == 2 * 16  # 32
    @test length(pms.L_pencils) == pms.n_pencils
    @test size(pms.L_pencils[1]) == (32, 32)
    @test pms.field_offsets == [0, 16]
    @test pms.kx_values[1] ≈ 0.0
    @test pms.kx_values[2] ≈ 2π / 4.0
end

end  # top-level testset
