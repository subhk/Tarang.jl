using Test
using Tarang
using LinearAlgebra

@testset "Pencil Matrices" begin

function scalar_diffusion_pencil_system(; Nx=16, Nz=8, ν=0.1)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64)
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    domain = Domain(dist, (xb, zb))

    u = ScalarField(domain, "u")
    problem = IVP([u])
    add_parameters!(problem, ν=ν)
    add_equation!(problem, "∂t(u) - ν*Δ(u) = 0")
    Tarang.build_matrix_expressions!(problem)

    ps = PencilSystem(problem, xb, zb)
    Tarang.build_pencil_system_matrices!(ps, problem, zb)
    return ps, problem, u
end

@testset "PencilSystem matrix assembly" begin
    ps, _, _ = scalar_diffusion_pencil_system()

    @test ps.n_pencils == 16 ÷ 2 + 1
    @test ps.pencil_size == 8
    @test length(ps.L_pencils) == ps.n_pencils
    @test length(ps.M_pencils) == ps.n_pencils

    for M in ps.M_pencils
        @test M ≈ Matrix{ComplexF64}(I, ps.pencil_size, ps.pencil_size)
    end

    @test all(L -> size(L) == (ps.pencil_size, ps.pencil_size), ps.L_pencils)
    @test any(L -> norm(L) > 0, ps.L_pencils)
    @test norm(ps.L_pencils[2] - ps.L_pencils[1]) > 0
end

@testset "PencilSystem cached LHS solve" begin
    ps, _, _ = scalar_diffusion_pencil_system()
    rhs = [fill(ComplexF64(k), ps.pencil_size) for k in 1:ps.n_pencils]

    lhs1 = Tarang.get_pencil_lhs!(ps, 1, 0.01, 0.5)
    lhs2 = Tarang.get_pencil_lhs!(ps, 1, 0.01, 0.5)
    @test lhs1 === lhs2
    @test haskey(ps.lhs_cache, (1, 0.01, 0.5))

    solved = Tarang.solve_pencil_system!(ps, rhs, 0.01, 0.5)
    @test length(solved) == ps.n_pencils
    @test all(v -> length(v) == ps.pencil_size, solved)
    @test all(v -> all(isfinite, real.(v)) && all(isfinite, imag.(v)), solved)
end

@testset "PencilSystem gather/scatter roundtrip" begin
    domain = ChannelDomain(32, 8; Lx=2π, Lz=1.0)
    u = ScalarField(domain, "u")
    fill_random!(u, "g"; seed=42, distribution="normal", scale=1.0)
    ensure_layout!(u, :c)
    original_data = copy(get_coeff_data(u))

    state = [u]
    xb, zb = domain.bases
    problem = IVP(state)
    ps = PencilSystem(problem, xb, zb)

    # Scatter to pencils
    pencils = vars_to_pencils(problem.variables, ps)
    @test length(pencils) == ps.n_pencils
    @test length(pencils[1]) == ps.pencil_size

    # Gather back
    state2 = [copy(u)]
    problem2 = IVP(state2)
    pencils_to_vars!(problem2.variables, pencils, ps)

    # Data should match
    @test get_coeff_data(state2[1]) ≈ original_data
end

@testset "PencilSystem constructor" begin
    domain = ChannelDomain(64, 16; Lx=4.0, Lz=1.0)
    p = ScalarField(domain, "p")
    b = ScalarField(domain, "b")

    state = [p, b]
    xb, zb = domain.bases
    problem = IVP(state)
    ps = PencilSystem(problem, xb, zb)

    @test ps.n_pencils == 64 ÷ 2 + 1  # RealFourier: N/2+1 = 33
    @test ps.n_cheb == 16
    @test ps.n_vars == 2
    @test ps.pencil_size == 2 * 16  # 32
    @test length(ps.L_pencils) == ps.n_pencils
    @test size(ps.L_pencils[1]) == (32, 32)
    @test ps.var_offsets == [0, 16]
    @test ps.kx_values[1] ≈ 0.0
    @test ps.kx_values[2] ≈ 2π / 4.0
end

end  # top-level testset
