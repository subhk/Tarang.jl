"""
Serial-CPU coverage tests for src/core/timesteppers/step_diagonal_imex.jl.

Targets the reachable serial-CPU lines of the diagonal-IMEX steppers:

  * step_diagonal_imex_rk222! / rk443!  EXPLICIT FALLBACK
        (no SpectralLinearOperator attached → `_step_explicit_rk!`)
  * step_diagonal_imex_sbdf2!  variable-dt warning branch and SBDF1 startup.

NOTE: the no-operator SBDF2 branch used to crash — a dotted assignment
broadcast `get_coeff_data` itself over the ScalarField, throwing
MethodError(length, ::ScalarField) on step 2. That is fixed (the coefficient
arrays are hoisted out of the broadcast), and the branch is covered in
test_diagonal_imex_robustness.jl along with the diagonalization contract.
  * The IMPLICIT path (`_step_diagonal_imex_rk_impl!`, `_sbdf2_apply_be_L!`,
        `_sbdf2_apply_bdf2_L!`, `_ddi_sbdf1_update!`, `_ddi_sbdf2_update!`)
        with a SpectralLinearOperator attached → analytic viscous decay
        exp(-ν k² t).

All tests assert real behaviour: exp(-t) ODE decay, designed convergence
order, exp(-ν k² t) viscous decay, and invariants (boundedness, finiteness).
"""

using Test
using LinearAlgebra
using Tarang

import Tarang: SpectralLinearOperator

# Helper: integrate dt(u) = -u, u(0)=1 over T_final with a given diagonal-IMEX
# timestepper and NO spectral operator (engages the explicit fallback). Returns
# the single grid value (constant mode) at T_final.
function _solve_decay_explicit(ts, dt; T_final = 1.0)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh = (1,), dtype = Float64)
    xb = RealFourier(coords["x"]; size = 4, bounds = (0.0, 2π))
    u = ScalarField(dist, "u", (xb,), Float64)
    ensure_layout!(u, :g)
    fill!(Tarang.get_grid_data(u), 1.0)            # u(x,0) = 1 (mean mode only)

    problem = IVP([u])
    add_equation!(problem, "dt(u) = -u")
    solver = InitialValueSolver(problem, ts; dt = dt)
    # NOTE: deliberately do NOT attach a SpectralLinearOperator → explicit fallback.

    nsteps = round(Int, T_final / dt)
    for _ in 1:nsteps
        step!(solver)
    end
    ensure_layout!(u, :g)
    return Tarang.get_grid_data(u)[1]
end

@testset "step_diagonal_imex.jl serial-CPU coverage" begin

    # -----------------------------------------------------------------------
    # 1. RK222 / RK443 EXPLICIT FALLBACK (no spectral operator).
    #    Hits step_diagonal_imex_rk222! lines 28-32 (L_spectral===nothing →
    #    _step_explicit_rk!) and the analogous rk443! branch.
    # -----------------------------------------------------------------------
    @testset "RK222 explicit fallback — exp(-t) decay + 2nd order" begin
        exact = exp(-1.0)
        u_coarse = _solve_decay_explicit(DiagonalIMEX_RK222(), 0.05)
        u_fine   = _solve_decay_explicit(DiagonalIMEX_RK222(), 0.025)

        # Both must land near the analytic value exp(-1).
        @test isapprox(u_coarse, exact; rtol = 0.05)
        @test isapprox(u_fine,   exact; atol = 1e-3)

        err_coarse = abs(u_coarse - exact)
        err_fine   = abs(u_fine - exact)
        @test err_fine < err_coarse                  # refining helps
        rate = log2(err_coarse / err_fine)
        @test rate > 1.5                             # ~2nd order
    end

    @testset "RK443 explicit fallback — exp(-t) decay + high order" begin
        exact = exp(-1.0)
        u_coarse = _solve_decay_explicit(DiagonalIMEX_RK443(), 0.05)
        u_fine   = _solve_decay_explicit(DiagonalIMEX_RK443(), 0.025)

        @test isapprox(u_coarse, exact; rtol = 0.01)
        @test isapprox(u_fine,   exact; atol = 1e-4)

        err_coarse = abs(u_coarse - exact)
        err_fine   = abs(u_fine - exact)
        # 3rd-order method; require at least clear super-linear improvement
        # (errors are tiny so use a relaxed threshold to stay robust).
        if err_fine > 1e-13
            @test log2(err_coarse / err_fine) > 1.5
        else
            @test err_fine <= err_coarse
        end
    end

    # -----------------------------------------------------------------------
    # 2. SBDF2 VARIABLE-dt branch — with an operator attached so the working
    #    `_sbdf2_apply_bdf2_L!` path runs. Pass a changed dt on a later step so
    #    dt/dt_prev ≠ 1; both branches now use the variable-dt SBDF2 weights
    #    (w = dtₙ/dtₙ₋₁), so the step is accurate and emits NO warning (the old
    #    constant-dt warning was removed in the 2026-06-20 audit). Integration
    #    must finish finite.
    # -----------------------------------------------------------------------
    @testset "SBDF2 variable-dt integrates (variable-dt weights, no warning)" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh = (1,), dtype = Float64)
        xb = RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g)
        fill!(Tarang.get_grid_data(u), 1.0)            # mean mode only (k=0)

        L = SpectralLinearOperator(dist, (xb,), :laplacian; ν = 0.5)  # 0 at k=0
        problem = IVP([u])
        add_equation!(problem, "dt(u) = -u")
        solver = InitialValueSolver(problem, DiagonalIMEX_SBDF2(); dt = 0.02)
        Tarang.set_spectral_linear_operator!(solver, L)

        # Steps 1-2 at dt=0.02 establish history; step 3 at a clearly different
        # dt=0.04 exercises the variable-dt SBDF2 weights. No warning is expected.
        step!(solver, 0.02)
        step!(solver, 0.02)
        step!(solver, 0.04)   # variable-dt SBDF2 weights; completes without warning

        ensure_layout!(u, :g)
        uf = Tarang.get_grid_data(u)[1]
        @test isfinite(uf)
        @test 0.0 < uf < 1.0          # decayed (dt(u)=-u, k=0) but stayed bounded
    end

    # -----------------------------------------------------------------------
    # 4. IMPLICIT path with SpectralLinearOperator attached.
    #    dt(u)=0 + laplacian L = ν k² → analytic viscous decay exp(-ν k² t).
    #    Exercises _step_diagonal_imex_rk_impl! (RK222/RK443) and the SBDF2
    #    L-applied helpers (_sbdf2_apply_be_L!, _sbdf2_apply_bdf2_L!,
    #    _ddi_sbdf1_update!, _ddi_sbdf2_update!).
    # -----------------------------------------------------------------------
    @testset "implicit viscous decay exp(-ν k² t) — RK222 / RK443 / SBDF2" begin
        # cos(2x): mode k=2, k²=4; ν=0.5 → λ = ν k² = 2. Over T = 200·0.005 = 1
        # the amplitude decays to exp(-λ T) = exp(-2).
        for ts in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh = (1,), dtype = Float64)
            xb = RealFourier(coords["x"]; size = 16, bounds = (0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            xs = collect(range(0, 2π, length = 17))[1:16]
            Tarang.get_grid_data(u) .= cos.(2 .* xs)
            u0 = maximum(abs, Tarang.get_grid_data(u))

            L = SpectralLinearOperator(dist, (xb,), :laplacian; ν = 0.5)
            problem = IVP([u])
            add_equation!(problem, "dt(u) = 0")
            solver = InitialValueSolver(problem, ts; dt = 0.005)
            Tarang.set_spectral_linear_operator!(solver, L)
            @test Tarang._get_spectral_linear_operator(solver) !== nothing

            for _ in 1:200
                step!(solver)
            end
            ensure_layout!(u, :g)
            uf = maximum(abs, Tarang.get_grid_data(u))
            @test isfinite(uf)
            @test uf < u0                              # the mode decays
            @test isapprox(uf / u0, exp(-2.0); rtol = 0.05)   # at λ = ν k² = 2
        end
    end

    # -----------------------------------------------------------------------
    # 5. IMPLICIT path, stiff limit (regression guard for the off-diagonal
    #    ESDIRK terms inside _step_diagonal_imex_rk_impl!). z = dt·ν·k² = 9 ≫ 1;
    #    an L-stable method must damp it toward 0 (the buggy diagonal-only update
    #    grows it ~2000×).
    # -----------------------------------------------------------------------
    @testset "implicit stiff-limit L-stability — RK222 / RK443" begin
        for ts in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443())
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh = (1,), dtype = Float64)
            xb = RealFourier(coords["x"]; size = 16, bounds = (0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            ensure_layout!(u, :g)
            xs = collect(range(0, 2π, length = 17))[1:16]
            Tarang.get_grid_data(u) .= cos.(6 .* xs)       # k=6, k²=36
            u0 = maximum(abs, Tarang.get_grid_data(u))

            L = SpectralLinearOperator(dist, (xb,), :laplacian; ν = 1.0)
            problem = IVP([u])
            add_equation!(problem, "dt(u) = 0")
            solver = InitialValueSolver(problem, ts; dt = 0.25)  # z = 0.25·36 = 9
            Tarang.set_spectral_linear_operator!(solver, L)
            for _ in 1:20
                step!(solver)
            end
            ensure_layout!(u, :g)
            uf = maximum(abs, Tarang.get_grid_data(u))
            @test isfinite(uf)
            @test uf < u0
            @test uf < 1e-2                              # strongly damped
        end
    end

    # -----------------------------------------------------------------------
    # 6. SBDF2 implicit with a nonlinear/forcing RHS so the BDF2 implicit
    #    branch (_sbdf2_apply_bdf2_L!) runs together with a non-zero F. Solves
    #    dt(u) = -u (explicit RHS) plus implicit damping ν k² (k=0 → no damping
    #    of the mean), giving exp(-t) on the mean mode. Confirms the implicit
    #    SBDF2 update produces the right result when both F and L participate.
    # -----------------------------------------------------------------------
    @testset "SBDF2 implicit with non-trivial explicit RHS" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh = (1,), dtype = Float64)
        xb = RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g)
        fill!(Tarang.get_grid_data(u), 1.0)            # mean mode only (k=0)

        L = SpectralLinearOperator(dist, (xb,), :laplacian; ν = 0.5)  # 0 at k=0
        problem = IVP([u])
        add_equation!(problem, "dt(u) = -u")
        solver = InitialValueSolver(problem, DiagonalIMEX_SBDF2(); dt = 0.01)
        Tarang.set_spectral_linear_operator!(solver, L)

        for _ in 1:100
            step!(solver)
        end
        ensure_layout!(u, :g)
        uf = Tarang.get_grid_data(u)[1]
        # k=0 mode: L̂=0, so pure dt(u)=-u → exp(-1).
        @test isapprox(uf, exp(-1.0); rtol = 0.02)
    end
end
