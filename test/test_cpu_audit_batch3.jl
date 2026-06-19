"""
Guard tests for CPU correctness audit batch 3 (2026-06-19, fix/cpu-correctness-batch2).

Six wrong-answer bugs found by an ultracode multi-subsystem audit + adversarial
verification, all FIXED and verified here against analytic oracles:

  C1 (HIGH) matrices_subproblem_operators.jl — Interpolate BC row hardcoded the
       ChebyshevT recurrence T_n(ξ) for EVERY Jacobi coupled basis, so Dirichlet/
       point BCs on ChebyshevU/V/Ultraspherical/Legendre/Jacobi enforced the wrong
       functional (U_n(±1)=(±1)^n(n+1) ≠ T_n(±1)). Fixed via basis-aware evaluate_basis.
  C2 (MED) matrices_subproblem_operators.jl — Integrate-constraint weight row
       hardcoded ∫T_n=L/2·2/(1-n²) (SIGN-FLIPPED for non-Chebyshev). Fixed via the
       exact basis-aware spectral row wₙ=Σ_j q_j φ_n(z_j).
  C3 (MED) problem_matrices_spectral.jl — nested/composed differential operators on
       the implicit LHS dropped all but the OUTERMOST derivative, so ∂x(∂x(u)) was
       assembled as (ik)¹ not (ik)²=-k² (a diffusion term silently became advection).
       Fixed by composing D_outer · matrix(operand).
  C4 (MED) flow_tools_spectra.jl — 3D enstrophy_spectrum used a mode-count kmax
       ceiling against PHYSICAL |k|, dropping resolved vorticity modes on L<2π.
  C5 (MED) flow_tools_spectra.jl — scalar power_spectrum reported mode-number
       wavenumbers (omitted 2π/L), disagreeing with energy_spectrum on non-2π domains.
  C7 (LOW) boundary_conditions/construction.jl — time/space dependency detection
       (\\bt\\b regex) missed implicit-multiplication forms like "sin(2t)"/"2x",
       freezing dynamic BCs at their build-time value. Fixed via AST free-symbols.

  C6 (MED) les_models.jl — AMD eddy-diffusivity numerator dropped the
       velocity-gradient/scalar-gradient DOUBLE contraction (summed only one
       velocity component, contracting the scaled velocity gradient with the same
       scalar-gradient direction twice). Now the full Abkar-Moin (2016) eq 2.7
       form over all velocity components — a BREAKING signature change (2D now
       takes 4 velocity gradients, 3D takes 9), applied with user approval.
"""

using Test
using Tarang
using LinearAlgebra

# Manufactured Poisson LBVP Δu+lift(τ1,-1)+lift(τ2,-2)=-2, u(0)=u(Lz)=0,
# exact u(z)=z(1-z); returns the max nodal error on the coupled `mkz` basis.
function _b3_bvp_err(mkz)
    coords = CartesianCoordinates("x", "z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 2π))
    zb = mkz(coords["z"])
    dom = Domain(dist, (xb, zb))
    u    = ScalarField(dom, "u")
    tau1 = ScalarField(dist, "tau1", (xb,), Float64)
    tau2 = ScalarField(dist, "tau2", (xb,), Float64)
    lb2  = derivative_basis(zb, 2)
    prob = Tarang.LBVP([u, tau1, tau2])
    add_parameters!(prob; Lz=1.0, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
    Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = -2")
    Tarang.add_bc!(prob, "u(z=0) = 0")
    Tarang.add_bc!(prob, "u(z=Lz) = 0")
    solver = Tarang.BoundaryValueSolver(prob)
    Tarang.solve!(solver)
    ensure_layout!(u, :g)
    zc = vec(Array(Tarang.local_grid(zb, dist, 1)))
    g  = Array(Tarang.get_grid_data(u))
    return maximum(abs.(g[1, :] .- zc .* (1.0 .- zc)))
end

# Diagonal of the implicit L matrix for a 1D pure-Fourier IVP equation.
function _b3_ldiag(eqstr)
    coords = CartesianCoordinates("x")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    xb  = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
    dom = Domain(dist, (xb,))
    u   = ScalarField(dom, "u")
    prob = Tarang.IVP([u])
    Tarang.add_equation!(prob, eqstr)
    L, _, _ = Tarang.build_matrices(prob)
    return diag(Matrix(L))
end

@testset "CPU audit batch 3" begin

    @testset "C1 Interpolate BC row is basis-aware (ChebyshevU)" begin
        eU = _b3_bvp_err(z -> ChebyshevU(z; size=16, bounds=(0.0, 1.0)))
        eT = _b3_bvp_err(z -> ChebyshevT(z; size=16, bounds=(0.0, 1.0)))
        @test eU < 1e-8          # was 0.125 (BCs enforced with wrong T_n row)
        @test eT < 1e-8          # ChebyshevT path unchanged
    end

    @testset "C2 Integrate-constraint weights are basis-aware (ChebyshevU)" begin
        cz   = CartesianCoordinates("z")
        zbU  = ChebyshevU(cz["z"]; size=6, bounds=(0.0, 1.0))
        Nz   = 6
        q    = Tarang.get_integration_weights(zbU)
        xref = Tarang._native_grid(zbU, 1.0)
        znod = (1.0 / 2) .* (xref .+ 1)                       # physical nodes on [0,1]
        V    = Tarang.evaluate_basis(zbU, znod, 0:(Nz-1))
        wfix = vec(permutedims(q) * V)                       # ∫U_n dz over [0,1]
        wexa = [iseven(n) ? 1.0 / (n + 1) : 0.0 for n in 0:(Nz-1)]
        @test maximum(abs.(wfix .- wexa)) < 1e-9
        # The old hardcoded ChebyshevT weights are demonstrably wrong (sign flip).
        wold = [iseven(n) ? (n == 0 ? 1.0 : 1.0 / (1 - n^2)) : 0.0 for n in 0:(Nz-1)]
        @test maximum(abs.(wold .- wexa)) > 0.1
    end

    @testset "C3 nested differential operators keep all derivatives" begin
        dnest = _b3_ldiag("dt(u) + ∂x(∂x(u)) = 0")
        dlap  = _b3_ldiag("dt(u) + lap(u) = 0")
        kp    = Float64[0, 1, 2, 3, 4]
        @test maximum(abs.(abs.(real.(dnest)) .- kp .^ 2)) < 1e-9   # 2nd order, not 1st
        @test maximum(abs.(imag.(dnest))) < 1e-9                    # real, not imaginary i·k
        @test maximum(abs.(dnest .- dlap)) < 1e-9                   # ≡ lap(u)
    end

    @testset "C4 3D enstrophy_spectrum keeps high-k modes on L<2π" begin
        N = 16; L = Float64(π)
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(1, 1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        zb = RealFourier(coords["z"]; size=N, bounds=(0.0, L))
        u  = VectorField(dist, "u", (xb, yb, zb), Float64)
        xs = collect(range(0, L, length=N + 1))[1:N]
        ensure_layout!(u.components[2], :g)
        g2 = Tarang.get_grid_data(u.components[2])
        for i in 1:N, j in 1:N, k in 1:N
            g2[i, j, k] = cos(2π * 6 * xs[i] / L)     # physical kx = 12 > N/2 = 8
        end
        for c in (1, 3)
            ensure_layout!(u.components[c], :g); fill!(Tarang.get_grid_data(u.components[c]), 0.0)
        end
        ps = enstrophy_spectrum(u)
        @test maximum(ps.k) >= 12       # physical ceiling reaches the mode (was 8)
        @test sum(ps.power) > 0         # vorticity retained (was ≈ 0)
    end

    @testset "C5 scalar power_spectrum reports physical wavenumbers" begin
        N = 32; L = 4π                  # k0 = 2π/L = 0.5
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, L))
        f  = ScalarField(dist, "f", (xb, yb), Float64)
        ensure_layout!(f, :g)
        xs = collect(range(0, L, length=N + 1))[1:N]
        gd = Tarang.get_grid_data(f)
        for i in 1:N, j in 1:N
            gd[i, j] = cos(2 * xs[i])   # physical wavenumber 2 (mode number n=4)
        end
        ps = power_spectrum(f)
        @test isapprox(ps.k[argmax(ps.power)], 2.0; atol=0.6)   # physical 2, not mode 4
    end

    @testset "C7 BC dependency detection handles implicit multiplication" begin
        @test Tarang.is_time_dependent("sin(2t)") == true
        @test Tarang.is_time_dependent("0.5t") == true
        @test Tarang.is_space_dependent("2x") == true
        @test Tarang.is_time_dependent("sin(2*t)") == true     # regression
        @test Tarang.is_time_dependent("sin(t)") == true       # regression
        @test Tarang.is_space_dependent("exp(-x^2)") == true   # regression
        @test Tarang.is_time_dependent("tau") == false         # no false positive
        @test Tarang.is_time_dependent("theta") == false       # no false positive
        @test !Tarang.is_time_dependent("0.0")
        @test !Tarang.is_space_dependent("0.0")
    end

    @testset "C6 AMD eddy diffusivity is the full double contraction" begin
        # κₑ† = -[ Σ_k δ_k²(∂_k b) Σ_i (∂_k u_i)(∂_i b) ] / |∇b|² (Abkar-Moin 2016 eq 2.7).
        # Oracle = explicit double sum over k AND velocity component i; the old
        # single-component formula cannot reproduce it for these distinct values.
        fs = (2, 2)
        m2 = AMDModel(filter_width=(1.0, 1.0), field_size=fs)
        du2 = ((0.3, 0.2), (-0.1, -0.4))     # du[i][k] = ∂u_i/∂x_k, i,k ∈ {x,y}
        b2  = (0.4, 0.5)
        compute_eddy_diffusivity!(m2,
            fill(du2[1][1], fs), fill(du2[1][2], fs),
            fill(du2[2][1], fs), fill(du2[2][2], fs),
            fill(b2[1], fs), fill(b2[2], fs))
        num2 = -sum(b2[k] * sum(du2[i][k] * b2[i] for i in 1:2) for k in 1:2)
        exp2 = m2.C * num2 / (b2[1]^2 + b2[2]^2)
        @test exp2 > 0
        @test all(isapprox.(get_eddy_diffusivity(m2), max(0, exp2); atol=1e-12))

        fs3 = (2, 2, 2)
        m3 = AMDModel(filter_width=(1.0, 1.0, 1.0), field_size=fs3)
        du3 = ((0.3, 0.2, 0.1), (-0.1, -0.4, 0.2), (0.05, -0.15, 0.25))
        b3  = (0.4, 0.5, -0.3)
        compute_eddy_diffusivity!(m3,
            fill(du3[1][1], fs3), fill(du3[1][2], fs3), fill(du3[1][3], fs3),
            fill(du3[2][1], fs3), fill(du3[2][2], fs3), fill(du3[2][3], fs3),
            fill(du3[3][1], fs3), fill(du3[3][2], fs3), fill(du3[3][3], fs3),
            fill(b3[1], fs3), fill(b3[2], fs3), fill(b3[3], fs3))
        num3 = -sum(b3[k] * sum(du3[i][k] * b3[i] for i in 1:3) for k in 1:3)
        exp3 = m3.C * num3 / (b3[1]^2 + b3[2]^2 + b3[3]^2)
        @test exp3 > 0
        @test all(isapprox.(get_eddy_diffusivity(m3), max(0, exp3); atol=1e-12))
    end

    @testset "R2-1 Poisson constraint solve respects the source sign" begin
        # RHS of Δ(ψ)+others=0 is -(others). The source-field shortcut used to strip
        # the sign, so Δ(ψ)+q=0 and Δ(ψ)-q=0 both solved to ψ=-q/k² — correct only for
        # the -source convention. Source q=cos(2x) (kx=2,ky=0 ⇒ Δ=-4) ⇒ ψ=∓cos(2x)/4.
        function _poisson_psi(eqstr)
            Nx, Ny = 16, 16
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; dtype=Float64, device=CPU())
            xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2pi), dealias=3/2)
            yb = RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2pi), dealias=3/2)
            dom = Domain(dist, (xb, yb))
            q = ScalarField(dom, "q"); psi = ScalarField(dom, "psi"); u = VectorField(dom, "u")
            tau_psi = ScalarField(dist, "tau_psi", (), Float64)
            prob = Tarang.IVP([q, psi, u, tau_psi])
            Tarang.add_equation!(prob, "∂t(q) = 0")
            Tarang.add_equation!(prob, eqstr)
            Tarang.add_equation!(prob, "u - skew(grad(psi)) = 0")
            Tarang.add_bc!(prob, "integ(psi) = 0")
            solver = Tarang.InitialValueSolver(prob, SBDF1(); dt=1e-3)
            x = Tarang.get_grid_coordinates(dom; on_device=false)["x"]
            q["g"] = [cos(2 * x[i]) for i in 1:Nx, j in 1:Ny]
            Tarang.evaluate_rhs(solver, solver.state, 0.0)
            ensure_layout!(psi, :g)
            return Array(get_grid_data(psi)), [cos(2 * x[i]) / 4 for i in 1:Nx, j in 1:Ny]
        end
        psiA, q4 = _poisson_psi("Δ(psi) + tau_psi - q = 0")   # Δψ=q ⇒ ψ=-q/4
        psiB, _  = _poisson_psi("Δ(psi) + tau_psi + q = 0")   # Δψ=-q ⇒ ψ=+q/4
        @test maximum(abs.(psiA .- (.-q4))) < 1e-9            # -source stays correct
        @test maximum(abs.(psiB .- q4)) < 1e-9               # +source now +q/4, not -q/4
    end

    @testset "R3-1 DiffusionSpec(implicit=true) applies diffusion, not silently dropped" begin
        # The explicit boundary-advection module has no implicit solver, so implicit=true
        # used to DROP the diffusion term entirely (field bit-for-bit unchanged). It is now
        # applied explicitly (with a one-time warning) — identical to implicit=false.
        function _bad_diffuse(implicit_flag)
            bad = Tarang.boundary_advection_diffusion_setup(
                Lx=2π, Ly=2π, Nx=16, Ny=16,
                boundaries=[Tarang.BoundarySpec("s", :z, 0.0)],
                velocity_source=Tarang.PrescribedVelocity(),
                diffusion=Tarang.DiffusionSpec(type=:laplacian, coefficient=0.5, implicit=implicit_flag))
            f = bad.fields["s"]; ensure_layout!(f, :g)
            gd = Tarang.get_grid_data(f)
            for i in 1:16, j in 1:16
                gd[i, j] = sin((i - 1) * 2π / 16) * cos((j - 1) * 2π / 16)
            end
            init = copy(Array(gd))
            for _ in 1:5
                Tarang.bad_step!(bad, 0.01)
            end
            ensure_layout!(f, :g)
            return init, Array(Tarang.get_grid_data(f))
        end
        initT, finalT = _bad_diffuse(true)
        _, finalF = _bad_diffuse(false)
        @test maximum(abs.(finalT .- initT)) > 1e-3       # diffusion actually applied (was 0 change)
        @test maximum(abs.(finalT .- finalF)) < 1e-12     # implicit=true ≡ explicit treatment now
    end
end
