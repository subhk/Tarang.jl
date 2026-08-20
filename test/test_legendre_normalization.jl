"""
Legendre's stored coefficients are ORTHONORMAL; its classical matrices are not.

`setup_legendre_transform!` (transform_planning.jl) normalizes with
`sqrt((2n+1)/2)`, so a Legendre field stores coefficients of `P̃ₙ = γₙ Pₙ`. But
`differentiation_matrix` implements the classical recurrence for un-normalized
`Pₙ`, and `evaluate_basis` returns un-normalized `Pₙ` values. Both of those are
deliberately classical — other tests pin that self-consistency
(`test_cov_basis_operators.jl` asserts `Dn·(B c) == B·(D c)`, and
`test_lift_convert.jl` uses `evaluate_basis` as an oracle *because* it is
independent of the transform's convention).

The bug was that nothing bridged the two. Every consumer that applied a
classical matrix to STORED coefficients was silently wrong for Legendre — and
"silently" is the whole problem: no error, no warning, a plausible number, and
the solver reporting success.

Measured before the fix:
  * `differentiation_matrix(Legendre) * coeffs`  → max err 3.25 (interpreted: 6.5e-13)
  * a `u(z=0)` BC row from `evaluate_basis`      → 0.761 for a field whose u(0) is 0
  * an LBVP `Δu = -2, u(0)=u(L)=0`               → max err 0.199 on amplitude 0.248,
                                                    ~80% relative, no error raised

The bridge is `stored_basis_scaling` / `spectral_derivative_matrix` /
`evaluate_stored_basis` in basis_operators.jl. These tests pin the bridge and the
end-to-end result, and — importantly — pin that ChebyshevT is UNCHANGED, since a
bridge that quietly rescaled Chebyshev would break everything else.
"""

using Test
using Tarang
using LinearAlgebra

@testset "Legendre orthonormal-coefficient bridge" begin
    N = 12
    f(z)  = z^3 - 0.3z
    fp(z) = 3z^2 - 0.3

    @testset "LBVP is exact — the silent 80%-error regression" begin
        # Δu = -2 on z ∈ [0, Lz], u(0) = u(Lz) = 0  ->  u = z(Lz - z).
        # Before the bridge: ChebyshevT exact to 1.4e-16, Legendre off by 0.199 on
        # an amplitude-0.248 answer, with the solver reporting success.
        Lz, Nx, Nz = 1.0, 8, 16
        for (name, mk) in (("ChebyshevT", c -> ChebyshevT(c; size=Nz, bounds=(0.0, Lz))),
                           ("Legendre",   c -> Legendre(c;   size=Nz, bounds=(0.0, Lz))))
            # Field names must match the equation string's identifiers — the parser
            # resolves `u` / `l1` by NAME, so a per-basis suffix silently fails to bind.
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; dtype=Float64)
            xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2pi))
            zb = mk(coords["z"])
            dom = Domain(dist, (xb, zb))
            u    = ScalarField(dom, "u")
            tau1 = ScalarField(dist, "tau1", (xb,), Float64)
            tau2 = ScalarField(dist, "tau2", (xb,), Float64)
            lb2  = Tarang.derivative_basis(zb, 2)
            prob = Tarang.LBVP([u, tau1, tau2])
            add_parameters!(prob; Lz=Lz, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
            Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = -2")
            Tarang.add_bc!(prob, "u(z=0) = 0")
            Tarang.add_bc!(prob, "u(z=Lz) = 0")

            Tarang.solve!(Tarang.BoundaryValueSolver(prob))
            ensure_layout!(u, :g)
            zc = vec(Array(Tarang.local_grid(zb, dist, 1)))
            got = Tarang.get_cpu_data(get_grid_data(u))
            exact = [zc[iz] * (Lz - zc[iz]) for _ in 1:size(got, 1), iz in 1:size(got, 2)]
            @test maximum(abs.(got .- exact)) < 1e-12
            # Guard against the assertion passing on an all-zero solve.
            @test maximum(abs.(got)) > 0.2
        end
    end

    @testset "stored_basis_scaling matches the transform" begin
        coords = CartesianCoordinates("z")
        lb = Legendre(coords["z"]; size=N, bounds=(0.0, 2.0))
        cb = ChebyshevT(coords["z"]; size=N, bounds=(0.0, 2.0))

        # Exactly `setup_legendre_transform!`'s `normalization`.
        @test Tarang.stored_basis_scaling(lb, N) ≈ [sqrt((2n + 1) / 2) for n in 0:(N - 1)]
        # Chebyshev's transform is un-normalized: the bridge must be the identity,
        # or every Chebyshev path in the package shifts underneath it.
        @test all(isone, Tarang.stored_basis_scaling(cb, N))
        @test Tarang.spectral_derivative_matrix(cb, 1) == Tarang.differentiation_matrix(cb, 1)
    end

    @testset "spectral_derivative_matrix agrees with the interpreted recurrence" begin
        for (name, mk) in (("ChebyshevT", c -> ChebyshevT(c; size=N, bounds=(0.0, 2.0))),
                           ("ChebyshevU", c -> ChebyshevU(c; size=N, bounds=(0.0, 2.0))),
                           ("Legendre",   c -> Legendre(c;   size=N, bounds=(0.0, 2.0))))
            coords = CartesianCoordinates("z")
            dist = Distributor(coords; dtype=Float64)
            zb = mk(coords["z"])
            q = ScalarField(dist, "q_$name", (zb,), Float64)
            zg = collect(Tarang.local_grid(zb, dist, 1.0))
            ensure_layout!(q, :g); get_grid_data(q) .= f.(zg); ensure_layout!(q, :c)

            D = Tarang.spectral_derivative_matrix(zb, 1)
            r = ScalarField(dist, "r_$name", (zb,), Float64)
            ensure_layout!(r, :c)
            get_coeff_data(r) .= D * get_coeff_data(q)
            ensure_layout!(r, :g)
            @test maximum(abs.(get_grid_data(r) .- fp.(zg))) < 1e-10
        end
    end

    @testset "evaluate_stored_basis reconstructs the field" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; dtype=Float64)
        zb = Legendre(coords["z"]; size=N, bounds=(0.0, 2.0))
        q = ScalarField(dist, "q", (zb,), Float64)
        zg = collect(Tarang.local_grid(zb, dist, 1.0))
        ensure_layout!(q, :g); get_grid_data(q) .= f.(zg); ensure_layout!(q, :c)
        c = copy(get_coeff_data(q))

        B̃ = Matrix{Float64}(Tarang.evaluate_stored_basis(zb, zg, 0:(N - 1)))
        @test maximum(abs.(B̃ * c .- f.(zg))) < 1e-10

        # The un-normalized values must NOT reconstruct it — if this ever passes,
        # the two conventions have silently merged and the bridge is a no-op.
        B = Matrix{Float64}(Tarang.evaluate_basis(zb, zg, 0:(N - 1)))
        @test maximum(abs.(B * c .- f.(zg))) > 1e-3

        # A point (Dirichlet) BC functional: u(z=0) must be f(0).
        row = Matrix{Float64}(Tarang.evaluate_stored_basis(zb, [0.0], 0:(N - 1)))
        @test abs((row * c)[1] - f(0.0)) < 1e-10
    end

end
