# Guard tests for the 2026-06-19 CPU correctness audit (11 confirmed fixes).
# Each testset pins the corrected numerical behavior so the bug cannot silently
# return. See memory/project_cpu_audit_2026_06_19.md for the full analysis.

using Test
using LinearAlgebra
using Tarang

# Helper: collocation coefficients of monomial x^k on a basis's own nodes, then
# verify D*coeffs evaluates to the analytic derivative on test points.
function _diffmat_max_error(basis, order::Int)
    N = basis.meta.size
    a, b = basis.meta.bounds
    nodes = Tarang._native_grid(basis, 1.0)
    nodes_phys = [a + (x + 1) * (b - a) / 2 for x in nodes]
    Bn = Matrix{Float64}(Tarang.evaluate_basis(basis, nodes_phys, 0:(N - 1)))
    test = collect(range(a + 0.05 * (b - a), b - 0.05 * (b - a); length=21))
    Bt = Matrix{Float64}(Tarang.evaluate_basis(basis, test, 0:(N - 1)))
    D = Matrix(differentiation_matrix(basis, order))
    maxerr = 0.0
    for k in order:(N - 2)            # polynomials whose order-th derivative is resolved
        c = Bn \ (nodes_phys .^ k)
        approx = Bt * (D * c)
        exact = k == order ? fill(Float64(factorial(k)), length(test)) :
                (factorial(k) / factorial(k - order)) .* (test .^ (k - order))
        maxerr = max(maxerr, maximum(abs.(approx .- exact)))
    end
    return maxerr
end

@testset "CPU audit 2026-06-19 guard tests" begin

    # #1 differentiation_matrix correct for ChebyshevU/V/Ultraspherical/Jacobi
    @testset "#1 differentiation_matrix general Jacobi family" begin
        coords = CartesianCoordinates("z")
        for mk in (z -> ChebyshevU(z; size=10, bounds=(-1.0, 1.0)),
                   z -> ChebyshevV(z; size=10, bounds=(-1.0, 1.0)),
                   z -> Ultraspherical(z; size=10, bounds=(-1.0, 1.0), alpha=1.0),
                   z -> Jacobi(z; size=10, bounds=(0.0, 3.0), a=0.3, b=0.7))
            b = mk(coords["z"])
            @test _diffmat_max_error(b, 1) < 1e-9
        end
        # second derivative too (was even more wrong)
        @test _diffmat_max_error(ChebyshevU(coords["z"]; size=10, bounds=(-1.0, 1.0)), 2) < 1e-7
        # control: ChebyshevT / Legendre stay exact
        @test _diffmat_max_error(ChebyshevT(coords["z"]; size=10, bounds=(-1.0, 1.0)), 1) < 1e-10
        @test _diffmat_max_error(Legendre(coords["z"]; size=10, bounds=(-1.0, 1.0)), 1) < 1e-10
    end

    # #3 integration weights exact for the non-T/Legendre Jacobi family
    @testset "#3 integration weights ChebyshevU/V/Ultraspherical/Jacobi" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        for mk in (z -> ChebyshevU(z; size=16, bounds=(-1.0, 1.0)),
                   z -> ChebyshevV(z; size=16, bounds=(-1.0, 1.0)),
                   z -> Ultraspherical(z; size=16, bounds=(-1.0, 1.0), alpha=1.5),
                   z -> Jacobi(z; size=16, bounds=(-1.0, 1.0), a=0.5, b=0.5))
            zb = mk(coords["z"])
            field = ScalarField(Domain(dist, (zb,)), "u")
            ensure_layout!(field, :g)
            zc = create_meshgrid(field.domain)
            zv = vec(Array(zc[collect(keys(zc))[1]]))
            get_grid_data(field) .= reshape(zv .^ 2, size(get_grid_data(field)))
            @test isapprox(integrate(field), 2 / 3; atol=1e-10)   # ∫_{-1}^1 z^2 dz
        end
    end

    # #4 3/2-dealias does NOT halve the even-N Nyquist mode
    @testset "#4 dealiased product preserves even-N Nyquist" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        ev = NonlinearEvaluator(dist; dealiasing_factor=1.5)
        f1 = ScalarField(Domain(dist, (xb,)), "f1")
        f2 = ScalarField(Domain(dist, (xb,)), "f2")
        ensure_layout!(f1, :g); ensure_layout!(f2, :g)
        xc = create_meshgrid(f1.domain); xv = vec(Array(xc[collect(keys(xc))[1]]))
        get_grid_data(f1) .= reshape(cos.(2 .* xv), size(get_grid_data(f1)))
        get_grid_data(f2) .= reshape(cos.(2 .* xv), size(get_grid_data(f2)))
        res = evaluate_transform_multiply(f1, f2, ev)
        ensure_layout!(res, :g)
        g = vec(Array(get_grid_data(res)))
        # cos(2x)^2 = 1/2 + 1/2 cos(4x); on the 8-pt grid that is [1,0,1,0,...].
        @test maximum(abs.(g .- Float64[isodd(i) ? 1.0 : 0.0 for i in 1:length(g)])) < 1e-9
    end

    # #5 complex-valued Chebyshev derivative does not crash and is correct
    @testset "#5 complex Chebyshev derivative" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh=(1,), dtype=ComplexF64)
        zb = ChebyshevT(coords["z"]; size=24, bounds=(-1.0, 1.0))
        f = ScalarField(Domain(dist, (zb,)), "f")
        ensure_layout!(f, :g)
        zc = create_meshgrid(f.domain); zv = vec(Array(zc[collect(keys(zc))[1]]))
        get_grid_data(f) .= reshape(sin.(zv) .+ im .* cos.(zv), size(get_grid_data(f)))
        r = evaluate(Tarang.Differentiate(f, coords["z"], 1), :g)
        got = vec(Array(get_grid_data(r)))
        exact = cos.(zv) .- im .* sin.(zv)         # d/dz [sin z + i cos z]
        @test maximum(abs.(got .- exact)) < 1e-8
    end

    # #6 Chebyshev derivative requested in :c layout on a mixed Fourier×Cheb domain
    @testset "#6 :c-layout Chebyshev derivative on mixed Fourier×Cheb" begin
        coords = CartesianCoordinates("x", "z")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=24, bounds=(0.0, 2.0))
        u = ScalarField(Domain(dist, (xb, zb)), "u")
        ensure_layout!(u, :g)
        get_grid_data(u) .= reshape(rand(16 * 24), size(get_grid_data(u)))
        r = evaluate(Tarang.Differentiate(u, coords["z"], 1), :c)   # must not throw
        @test r.current_layout == :c
        cd = get_coeff_data(r)
        @test size(cd, 1) == 16 ÷ 2 + 1            # rfft-halved Fourier axis
        @test eltype(cd) <: Complex
    end

    # #7 spectral resample_1d! downsample preserves the new Nyquist mode
    @testset "#7 resample_1d! downsample Nyquist" begin
        old = Float64[1, 0, -1, 0, 1, 0, -1, 0]    # pure frequency f=2
        newd = zeros(Float64, 4)
        Tarang.resample_1d!(newd, old)
        @test maximum(abs.(newd .- Float64[1, -1, 1, -1])) < 1e-10
    end

    # #8 enstrophy path: the squared-magnitude helper must not inject a 1/2
    @testset "#8 calculate_spectral_kinetic_energy apply_half kwarg" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        v = VectorField(dist, coords, "v", (xb, yb), Float64)
        mg = create_meshgrid(v.components[1].domain)
        X = vec(Array(mg["x"])); Y = vec(Array(mg["y"]))
        ensure_layout!(v.components[1], :g); ensure_layout!(v.components[2], :g)
        get_grid_data(v.components[1]) .= reshape(sin.(X) .* cos.(Y), size(get_grid_data(v.components[1])))
        get_grid_data(v.components[2]) .= reshape(cos.(X) .* sin.(Y), size(get_grid_data(v.components[2])))
        half = Tarang.calculate_spectral_kinetic_energy(v; apply_conjugate_symmetry=true, apply_half=true)
        full = Tarang.calculate_spectral_kinetic_energy(v; apply_conjugate_symmetry=true, apply_half=false)
        @test maximum(abs.(full .- 2.0 .* half)) < 1e-10
    end

    # #9 validate_streamfunction matches the perp_grad / streamfunction convention
    @testset "#9 validate_streamfunction perp_grad convention" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
        ψ = ScalarField(Domain(dist, (xb, yb)), "psi")
        ensure_layout!(ψ, :g)
        mg = create_meshgrid(ψ.domain)
        X = vec(Array(mg["x"])); Y = vec(Array(mg["y"]))
        get_grid_data(ψ) .= reshape(sin.(X) .* cos.(Y), size(get_grid_data(ψ)))
        u = Tarang.perp_grad(ψ)
        res = Tarang.validate_streamfunction(u, ψ; tolerance=1e-6)
        @test res.valid
        @test res.u_error < 1e-6
        @test res.v_error < 1e-6
    end

    # #10 ComplexFourier interpolation keeps the imaginary part
    @testset "#10 ComplexFourier interpolation imaginary part" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=ComplexF64)
        xb = ComplexFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        g = ScalarField(Domain(dist, (xb,)), "g")
        ensure_layout!(g, :g)
        xc = create_meshgrid(g.domain); xv = vec(Array(xc[collect(keys(xc))[1]]))
        get_grid_data(g) .= reshape(im .* sin.(xv), size(get_grid_data(g)))
        for p in (0.37, 1.9, 4.2)
            val = evaluate(Tarang.Interpolate(g, coords["x"], p), :g)
            v = val isa AbstractArray ? first(val) : val
            @test isapprox(v, im * sin(p); atol=1e-8)
        end
    end

    # #11 grid-space resample of a non-Fourier field uses the basis-aware spectral
    # path, not periodic-FFT interpolation
    @testset "#11 grid-space Chebyshev resample (basis-aware)" begin
        coords = CartesianCoordinates("z")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        zb = ChebyshevT(coords["z"]; size=16, bounds=(-1.0, 1.0))
        f = ScalarField(Domain(dist, (zb,)), "f")
        ensure_layout!(f, :g)
        zc = create_meshgrid(f.domain); zv = vec(Array(zc[collect(keys(zc))[1]]))
        testfn(x) = exp(x) * (1 + x^2) + 0.3x          # analytic ⇒ spectrally resolved
        get_grid_data(f) .= reshape(testfn.(zv), size(get_grid_data(f)))
        Tarang.change_scales!(f, 1.5)
        ensure_layout!(f, :g)
        got = vec(Array(get_grid_data(f)))
        new_nodes = Tarang._native_grid(zb, 1.5)        # bounds (-1,1): ref == physical
        @test length(got) == ceil(Int, 16 * 1.5)
        @test maximum(abs.(got .- testfn.(new_nodes))) < 1e-9
    end

    # #2 streamfunction Jacobi Poisson solver is convergent (finite, not NaN/Inf)
    # on a non-all-Fourier domain — the corrected diagonal makes it diag-dominant.
    @testset "#2 streamfunction Jacobi solver converges (finite)" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = ChebyshevT(coords["y"]; size=16, bounds=(0.0, 1.0))
        ψ0 = ScalarField(Domain(dist, (xb, yb)), "psi0")
        ensure_layout!(ψ0, :g)
        mg = create_meshgrid(ψ0.domain)
        X = vec(Array(mg["x"])); Y = vec(Array(mg["y"]))
        get_grid_data(ψ0) .= reshape(sin.(X) .* sin.(π .* Y), size(get_grid_data(ψ0)))
        u = Tarang.perp_grad(ψ0)
        psi = Tarang.streamfunction(u; boundary_condition=:no_slip, gauge_condition=true)
        ensure_layout!(psi, :g)
        @test all(isfinite, get_grid_data(psi))
    end

end
