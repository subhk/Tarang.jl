using Test
using Tarang
using LinearAlgebra

@testset "PencilSystem" begin

@testset "Constructor sizing" begin
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
    domain = Domain(dist, (xb, zb))

    p = ScalarField(domain, "p")
    b = ScalarField(domain, "b")
    u = VectorField(domain, "u")
    tau_p = ScalarField(dist, "tau_p", (), Float64)

    problem = IVP([p, b, u, tau_p])

    ps = PencilSystem(problem, xb, zb)

    @test ps.n_pencils == 32 ÷ 2 + 1  # 17
    @test ps.n_cheb == 16
    @test ps.n_vars == 4

    # p: 1 comp × 16 = 16
    # b: 1 comp × 16 = 16
    # u: 2 comp × 16 = 32
    # tau_p: 1 comp × 1 = 1
    @test ps.var_dofs == [16, 16, 32, 1]
    @test ps.var_n_comp == [1, 1, 2, 1]
    @test ps.var_comp_size == [16, 16, 16, 1]
    @test ps.var_offsets == [0, 16, 32, 64]
    @test ps.pencil_size == 65
    @test size(ps.L_pencils[1]) == (65, 65)
end

@testset "Gather/scatter roundtrip" begin
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
    domain = Domain(dist, (xb, zb))

    b = ScalarField(domain, "b")
    u = VectorField(domain, "u")

    fill_random!(b, "g"; seed=42, distribution="normal", scale=1.0)
    ensure_layout!(b, :c)
    original_b = copy(get_coeff_data(b))

    fill_random!(u.components[1], "g"; seed=43, distribution="normal", scale=1.0)
    fill_random!(u.components[2], "g"; seed=44, distribution="normal", scale=1.0)
    ensure_layout!(u.components[1], :c)
    ensure_layout!(u.components[2], :c)
    original_ux = copy(get_coeff_data(u.components[1]))
    original_uz = copy(get_coeff_data(u.components[2]))

    problem = IVP([b, u])
    ps = PencilSystem(problem, xb, zb)

    # Scatter
    pencils = vars_to_pencils(problem.variables, ps)
    @test length(pencils) == ps.n_pencils
    @test length(pencils[1]) == ps.pencil_size

    # Check non-zero
    total = sum(sum(abs.(p)) for p in pencils)
    @test total > 0

    # Gather back
    pencils_to_vars!(problem.variables, pencils, ps)

    @test get_coeff_data(b) ≈ original_b
    @test get_coeff_data(u.components[1]) ≈ original_ux
    @test get_coeff_data(u.components[2]) ≈ original_uz
end

end
