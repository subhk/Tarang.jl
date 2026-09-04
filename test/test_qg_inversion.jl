using Test
using Tarang

@testset "QG inversion solves the tau LBVP with surface buoyancy forcing" begin
    qg = qg_system_setup(
        Lx = 2π, Ly = 2π, H = 1.0,
        Nx = 4, Ny = 4, Nz = 6,
        f0 = 2.0, N = 1.0,
    )

    # Manufactured streamfunction ψ = cos(x) * (z² + z).  For S=(f₀/N)²=4,
    # q = ∂xxψ + S∂zzψ and θ = (f₀/N)∂zψ at each surface.
    set!(qg.q, (x, y, z) -> cos(x) * (8 - z^2 - z))
    set!(qg.θ_bot, (x, y) -> 2cos(x))
    set!(qg.θ_top, (x, y) -> 6cos(x))

    ψ = qg_invert!(qg)
    ensure_layout!(ψ, :g)

    X, _, Z = Tarang.local_grids(ψ.dist, ψ.bases...)
    got = Array(get_grid_data(ψ))
    expected = [cos(x) * (z^2 + z) for x in vec(X), _ in axes(got, 2), z in vec(Z)]

    @test maximum(abs, got) > 1
    @test got ≈ expected atol = 1e-9 rtol = 1e-9

    dψdz = Tarang.evaluate_differentiate(
        Tarang.Differentiate(ψ, ψ.dist.coordsys["z"], 1), :g,
    )
    ensure_layout!(dψdz, :g)
    derivative_grid = Array(get_grid_data(dψdz))
    surface_mode = repeat(reshape(cos.(vec(X)), :, 1), 1, size(got, 2))

    @test derivative_grid[:, :, 1] ≈ surface_mode atol = 1e-9 rtol = 1e-9
    @test derivative_grid[:, :, end] ≈ 3surface_mode atol = 1e-9 rtol = 1e-9

    # The public coupled step must reuse the corrected inversion instead of
    # rebuilding a rectangular BVP.
    qg_step!(qg, 1e-4; timestepper=:Euler)
    ensure_layout!(qg.θ_bot, :g)
    ensure_layout!(qg.θ_top, :g)
    @test all(isfinite, Array(get_grid_data(qg.θ_bot)))
    @test all(isfinite, Array(get_grid_data(qg.θ_top)))

    # Reuse the cached solver for a horizontally uniform solution.  The
    # Neumann--Neumann zero mode needs an explicit streamfunction gauge:
    # ψ = z² + z - 5/6 has zero volume mean, q = S∂zzψ = 8, and the same
    # two boundary derivatives as above.
    set!(qg.q, (x, y, z) -> 8.0)
    set!(qg.θ_bot, (x, y) -> 2.0)
    set!(qg.θ_top, (x, y) -> 6.0)

    ψ_dc = qg_invert!(qg)
    ensure_layout!(ψ_dc, :g)
    got_dc = Array(get_grid_data(ψ_dc))
    expected_dc = [z^2 + z - 5 / 6
                   for _ in axes(got_dc, 1), _ in axes(got_dc, 2), z in vec(Z)]

    @test got_dc ≈ expected_dc atol = 1e-9 rtol = 1e-9
    @test abs(integrate(ψ_dc)) < 1e-10
end
