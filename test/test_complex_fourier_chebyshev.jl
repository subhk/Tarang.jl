# Guard: a ComplexFourier axis in a Fourier x Chebyshev problem.
#
# The per-mode implicit operator picks its wavenumber array with
#   isa(basis, RealFourier) && first_real_fourier_axis ? wavenumbers_rfft : wavenumbers_fft
# (`_subproblem_kx`, matrices_subproblem_helpers.jl). A ComplexFourier basis lands
# in the `wavenumbers_fft` arm, which only had a RealFourier method — so ANY
# Fourier x Chebyshev problem with a ComplexFourier axis died at solver
# construction with `MethodError: no method matching wavenumbers_fft(::ComplexFourier)`.
# A ComplexFourier axis is always stored as a full-length fft spectrum, so its
# FFT-layout wavenumbers are exactly its native ones.
using Test
using Tarang

@testset "wavenumbers_fft covers ComplexFourier" begin
    coords = CartesianCoordinates("x")
    for N in (8, 9)
        b = ComplexFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        @test Tarang.wavenumbers_fft(b) == Tarang.wavenumbers(b)
        @test length(Tarang.wavenumbers_fft(b)) == N
    end
end

# Same physical problem, once with a RealFourier periodic axis and once with a
# ComplexFourier one. The initial condition is real and periodic, so the two
# discretizations represent the same function and must agree to roundoff.
function _fc_diffusion(xkind; Nz=12, Nx=8, Lz=1.0, dt=1e-3, nsteps=20)
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
    xb = xkind === :real ? RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π)) :
                           ComplexFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    b = ScalarField(Domain(dist, (zb, xb)), "b")
    tau1 = ScalarField(dist, "tau_b1", (), Float64)
    tau2 = ScalarField(dist, "tau_b2", (), Float64)
    _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zb, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * τ_lift(tau1)
    problem = IVP([b, tau1, tau2])
    add_parameters!(problem, kappa=0.1, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
    add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = 0")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt=dt)

    xf = [2π * (i - 1) / Nx for i in 1:Nx]
    zf = [Lz / 2 * (1 - cos(π * (k - 1) / (Nz - 1))) for k in 1:Nz]
    ensure_layout!(b, :g)
    get_grid_data(b) .= [sin(π * zf[iz] / Lz) * (1 + 0.5cos(2xf[ix]) + 0.3sin(xf[ix]))
                         for iz in 1:Nz, ix in 1:Nx]
    ensure_layout!(b, :c)
    for _ in 1:nsteps; step!(solver, dt); end
    ensure_layout!(b, :g)
    return real.(Array(get_grid_data(b))), zf, Lz, dt * nsteps
end

@testset "Fourier x Chebyshev IVP: ComplexFourier axis == RealFourier axis" begin
    ref, zf, Lz, tend = _fc_diffusion(:real)
    got, _, _, _ = _fc_diffusion(:complex)
    @test size(got) == size(ref)
    @test isapprox(got, ref; atol=1e-12)

    # ...and both solve the right problem: the x-mean is the decaying z-mode.
    decay = exp(-0.1 * (π / Lz)^2 * tend)
    analytic = [sin(π * zf[iz] / Lz) * decay for iz in eachindex(zf)]
    for field in (ref, got)
        xmean = vec(sum(field, dims=2)) ./ size(field, 2)
        @test isapprox(xmean, analytic; rtol=1e-5)
    end
end

@testset "3D Fourier x Fourier x Chebyshev with a ComplexFourier axis" begin
    Nz, Nx, Ny, Lz, dt, nsteps = 12, 8, 8, 1.0, 1e-3, 10
    function run3d(ykind)
        coords = CartesianCoordinates("z", "x", "y")
        dist = Distributor(coords; dtype=Float64, architecture=CPU())
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
        yb = ykind === :real ? RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2π)) :
                               ComplexFourier(coords["y"]; size=Ny, bounds=(0.0, 2π))
        b = ScalarField(Domain(dist, (zb, xb, yb)), "b")
        tau1 = ScalarField(dist, "tau_b1", (), Float64)
        tau2 = ScalarField(dist, "tau_b2", (), Float64)
        ez = unit_vector_fields(coords, dist)[1]
        lift_basis = derivative_basis(zb, 1)
        τ_lift(A) = lift(A, lift_basis, -1)
        grad_b = grad(b) + ez * τ_lift(tau1)
        problem = IVP([b, tau1, tau2])
        add_parameters!(problem, kappa=0.1, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
        add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = 0")
        add_bc!(problem, "b(z=0) = 0")
        add_bc!(problem, "b(z=1) = 0")
        solver = InitialValueSolver(problem, RK222(); dt=dt)
        xf = [2π * (i - 1) / Nx for i in 1:Nx]
        yf = [2π * (j - 1) / Ny for j in 1:Ny]
        zf = [Lz / 2 * (1 - cos(π * (k - 1) / (Nz - 1))) for k in 1:Nz]
        ensure_layout!(b, :g)
        get_grid_data(b) .= [sin(π * zf[iz] / Lz) * (1 + 0.5cos(2xf[ix]) + 0.3sin(yf[iy]))
                             for iz in 1:Nz, ix in 1:Nx, iy in 1:Ny]
        ensure_layout!(b, :c)
        for _ in 1:nsteps; step!(solver, dt); end
        ensure_layout!(b, :g)
        return real.(Array(get_grid_data(b)))
    end
    @test isapprox(run3d(:complex), run3d(:real); atol=1e-12)
end
