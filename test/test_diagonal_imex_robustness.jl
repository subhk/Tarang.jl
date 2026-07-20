"""
Robustness tests for the serial DiagonalIMEX timesteppers and SpectralLinearOperator.

Companion to test_diagonal_imex.jl (which pins the happy path with an explicitly
attached operator). This file guards three defects that all produced a WRONG
ANSWER or a crash rather than a diagnostic:

D1. `DiagonalIMEX_SBDF2` died on step 2 of every run whose implicit operator was
    not an attached `SpectralLinearOperator`. The no-operator SBDF2 branch wrote
    `@. d = ((1+w)*get_coeff_data(field_n) - ...)`; `@.` dots EVERY call in the
    expression, so `get_coeff_data` was broadcast over a `ScalarField`, and
    `Base.broadcastable`'s `collect` fallback needs `length(::ScalarField)` —
    a method that does not exist. Fixed by hoisting the coefficient arrays out of
    the broadcast into a function barrier (`_ddi_sbdf2_update_noL!`).

D2. `DiagonalIMEX_RK222` / `RK443` logged at `@debug` (invisible) and silently
    took a fully EXPLICIT step whenever no operator was attached, DROPPING the
    equation's implicit operator: pure diffusion returned its undamped initial
    condition (1.0) instead of exp(-1). The steppers now derive the diagonal
    operator from the equation's own `L` term, and refuse loudly when it is not
    diagonalizable — matching the MPI sibling `_get_distributed_diagonal_Lhats!`.

D3. An unrecognised `operator_type` fell through to `zero(T)`, producing an
    ALL-ZERO `SpectralLinearOperator`: the implicit term vanished with no
    complaint (a viscous run silently became inviscid). Unknown symbols now
    raise an `ArgumentError` naming the valid set.
"""

using Test
using LinearAlgebra
using Tarang

import Tarang: SpectralLinearOperator, SPECTRAL_OPERATOR_TYPES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""1-D periodic Fourier setup: returns (dist, basis)."""
function _dimex_setup(N::Int)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh = (1,), dtype = Float64)
    xb = RealFourier(coords["x"]; size = N, bounds = (0.0, 2π))
    return dist, xb
end

_dimex_grid(N::Int) = collect(range(0, 2π, length = N + 1))[1:N]

"""
Integrate the pure-diffusion problem `∂t(u) - Δu = 0` from `u(x,0) = cos(x)`.

Mode k=1 decays at rate k² = 1, so `max|u|` at `T = 1` is exactly `exp(-1)`.
The implicit operator lives ONLY in the equation's `L` term — nothing is
attached with `set_spectral_linear_operator!`. Any scheme that drops `L` returns
the initial amplitude 1.0 instead.
"""
function _diffusion_amplitude(ts; dt = 0.001, N = 16, T = 1.0)
    dist, xb = _dimex_setup(N)
    u = ScalarField(dist, "u", (xb,), Float64)
    ensure_layout!(u, :g)
    Tarang.get_grid_data(u) .= cos.(_dimex_grid(N))

    problem = IVP([u])
    add_equation!(problem, "dt(u) - lap(u) = 0")
    solver = InitialValueSolver(problem, ts; dt = dt)

    for _ in 1:round(Int, T / dt)
        step!(solver)
    end
    ensure_layout!(u, :g)
    return maximum(abs, Tarang.get_grid_data(u))
end

@testset "DiagonalIMEX robustness (D1/D2/D3)" begin

# ===========================================================================
# D1 — DiagonalIMEX_SBDF2 multistep branch
# ===========================================================================
@testset "D1: DiagonalIMEX_SBDF2 survives the multistep branch" begin
    # Step 1 uses the SBDF1 startup branch; step 2 is the first to enter the
    # SBDF2 multistep branch — precisely where the run used to die with
    # `MethodError: no method matching length(::ScalarField)`.
    N = 16
    dist, xb = _dimex_setup(N)
    u = ScalarField(dist, "u", (xb,), Float64)
    ensure_layout!(u, :g)
    Tarang.get_grid_data(u) .= cos.(_dimex_grid(N))

    problem = IVP([u])
    add_equation!(problem, "dt(u) - lap(u) = 0")
    solver = InitialValueSolver(problem, DiagonalIMEX_SBDF2(); dt = 0.05)

    for _ in 1:5
        step!(solver)                      # crashed on iteration 2 before the fix
    end
    ensure_layout!(u, :g)
    @test all(isfinite, Tarang.get_grid_data(u))
    @test solver.iteration == 5
end

@testset "D1: DiagonalIMEX_SBDF2 matches the reference SBDF2" begin
    # Not merely "it stops crashing": the two schemes solve the SAME variable-dt
    # SBDF2 update, so on this diagonalizable problem they must agree closely,
    # and both must sit near the analytic value.
    dt = 0.05
    exact_2 = exp(-2 * dt)                 # amplitude of cos(x) after 2 steps

    function two_steps(ts)
        N = 16
        dist, xb = _dimex_setup(N)
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g)
        Tarang.get_grid_data(u) .= cos.(_dimex_grid(N))
        problem = IVP([u])
        add_equation!(problem, "dt(u) - lap(u) = 0")
        solver = InitialValueSolver(problem, ts; dt = dt)
        step!(solver)
        step!(solver)
        ensure_layout!(u, :g)
        return maximum(abs, Tarang.get_grid_data(u))
    end

    u_diag = two_steps(DiagonalIMEX_SBDF2())
    u_ref  = two_steps(SBDF2())

    @test isfinite(u_diag)
    @test isapprox(u_diag, u_ref; rtol = 1e-10)   # same scheme, same answer
    @test isapprox(u_diag, exact_2; rtol = 0.05)  # and both track the exact decay
    @test u_diag < 1.0                            # actually diffused

    # Over a full unit of time the two remain in agreement.
    @test isapprox(_diffusion_amplitude(DiagonalIMEX_SBDF2()),
                   _diffusion_amplitude(SBDF2()); rtol = 1e-8)
end

# ===========================================================================
# D2 — the implicit operator must never be dropped in silence
# ===========================================================================
@testset "D2: diagonalizable problem decays at exp(-1) under all three schemes" begin
    # THE regression number. Before the fix RK222/RK443 returned 1.00000000
    # (no decay at all) and SBDF2 crashed; the exact answer is exp(-1).
    exact = exp(-1.0)
    for ts in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
        amp = _diffusion_amplitude(ts)
        @test isfinite(amp)
        @test amp < 1.0                              # the operator was applied at all
        @test isapprox(amp, exact; rtol = 1e-4)      # and applied correctly
    end
end

@testset "D2: derived operator agrees with an explicitly attached one" begin
    # `∂t(u) + u = 0` has L = u, i.e. a constant damping L̂ ≡ 1 for every mode.
    # Deriving it from the equation must give the same answer as handing the
    # scheme the equivalent operator by hand (:custom with all-ones).
    function damped(ts; attach::Bool)
        N = 8
        dist, xb = _dimex_setup(N)
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g)
        fill!(Tarang.get_grid_data(u), 1.0)
        problem = IVP([u])
        add_equation!(problem, attach ? "dt(u) = 0" : "dt(u) + u = 0")
        solver = InitialValueSolver(problem, ts; dt = 0.001)
        if attach
            ensure_layout!(u, :c)
            n = length(Tarang.get_coeff_data(u))
            L = SpectralLinearOperator(dist, (xb,), :custom; coefficients = ones(n))
            Tarang.set_spectral_linear_operator!(solver, L)
        end
        for _ in 1:1000
            step!(solver)
        end
        ensure_layout!(u, :g)
        return Tarang.get_grid_data(u)[1]
    end

    for ts in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
        derived  = damped(ts; attach = false)
        attached = damped(ts; attach = true)
        @test isapprox(derived, attached; rtol = 1e-10)
        @test isapprox(derived, exp(-1.0); rtol = 1e-4)
    end
end

@testset "D2: a non-diagonalizable implicit term raises, never runs explicitly" begin
    # (a) Cross-field COUPLING: L = v is not a per-mode multiplier on u. Folding
    #     it in as a constant would silently integrate a different equation.
    function coupled_solver(ts)
        N = 8
        dist, xb = _dimex_setup(N)
        u = ScalarField(dist, "u", (xb,), Float64)
        v = ScalarField(dist, "v", (xb,), Float64)
        ensure_layout!(u, :g); fill!(Tarang.get_grid_data(u), 1.0)
        ensure_layout!(v, :g); fill!(Tarang.get_grid_data(v), 1.0)
        problem = IVP([u, v])
        add_equation!(problem, "dt(u) - v = 0")
        add_equation!(problem, "v = 0")
        return InitialValueSolver(problem, ts; dt = 0.01)
    end

    for ts in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
        solver = coupled_solver(ts)
        @test_throws ArgumentError step!(solver)
    end

    # The message must name the scheme and point at a usable alternative,
    # rather than leaving the user to discover the missing physics themselves.
    err = try
        step!(coupled_solver(DiagonalIMEX_RK222())); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("DiagonalIMEX_RK222", msg)
    @test occursin("not diagonal", msg)
    @test occursin("RK222, RK443", msg)          # non-diagonal alternatives
    @test occursin("set_spectral_linear_operator!", msg)

    # (b) Derivative of ANOTHER field. `∂x(u)` on the stepped field itself IS
    #     diagonal (multiplier ik), but `∂x(v)` couples u to v and is not — the
    #     `is_self` guard in `_accumulate_diagonal_term!` is what separates them.
    let N = 8
        dist, xb = _dimex_setup(N)
        u = ScalarField(dist, "u", (xb,), Float64)
        v = ScalarField(dist, "v", (xb,), Float64)
        ensure_layout!(u, :g); fill!(Tarang.get_grid_data(u), 1.0)
        ensure_layout!(v, :g); fill!(Tarang.get_grid_data(v), 1.0)
        problem = IVP([u, v])
        add_equation!(problem, "dt(u) - d(v,x) = 0")   # Differentiate(v), operand ≠ u
        add_equation!(problem, "v = 0")
        solver = InitialValueSolver(problem, DiagonalIMEX_RK443(); dt = 0.01)
        @test_throws ArgumentError step!(solver)
    end

    # (b') The self-derivative counterpart must still be ACCEPTED — advection
    #     `∂t(u) + ∂x(u) = 0` has the diagonal multiplier ik and must not be
    #     swept up by the new refusal.
    let N = 16
        dist, xb = _dimex_setup(N)
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g)
        Tarang.get_grid_data(u) .= cos.(_dimex_grid(N))
        problem = IVP([u])
        add_equation!(problem, "dt(u) + d(u,x) = 0")
        solver = InitialValueSolver(problem, DiagonalIMEX_RK443(); dt = 0.002)
        for _ in 1:500                                  # advect by t = 1.0
            step!(solver)
        end
        ensure_layout!(u, :g)
        got = Tarang.get_grid_data(u)
        @test all(isfinite, got)
        @test isapprox(got, cos.(_dimex_grid(N) .- 1.0); atol = 1e-3)  # phase shift, no decay
    end

    # (c) Non-Fourier direction: a diagonal per-mode solve does not exist for a
    #     Chebyshev Laplacian, so the scheme must refuse instead of running
    #     without the viscous term.
    let N = 16
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh = (1,), dtype = Float64)
        xb = Chebyshev(coords["x"]; size = N, bounds = (-1.0, 1.0))
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g); fill!(Tarang.get_grid_data(u), 1.0)
        problem = IVP([u])
        add_equation!(problem, "dt(u) - lap(u) = 0")
        solver = InitialValueSolver(problem, DiagonalIMEX_RK222(); dt = 0.01)
        @test_throws ArgumentError step!(solver)
    end

    # (d) Spatially varying coefficient — a diagonal operator structurally
    #     cannot hold one (`SpectralLinearOperator.ν` is a scalar `Real`). This
    #     must fail somewhere in the pipeline, never integrate silently.
    #     The exact exception type belongs to the equation-parsing layer, so
    #     assert only that it raises.
    let N = 8
        dist, xb = _dimex_setup(N)
        xs = _dimex_grid(N)
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g); Tarang.get_grid_data(u) .= cos.(xs)
        nu = ScalarField(dist, "nu", (xb,), Float64)
        ensure_layout!(nu, :g); Tarang.get_grid_data(nu) .= 1.0 .+ 0.5 .* sin.(xs)
        problem = IVP([u])
        problem.parameters["nu"] = nu
        add_equation!(problem, "dt(u) - nu*lap(u) = 0")
        raised = try
            solver = InitialValueSolver(problem, DiagonalIMEX_RK222(); dt = 0.01)
            step!(solver)
            false
        catch
            true
        end
        @test raised
    end
end

@testset "D2: a genuinely explicit problem still falls back (and stays correct)" begin
    # `dt(u) = -u` keeps the linear term on the RHS, so L is a ZeroOperator:
    # there is no implicit term to drop and an explicit step is legitimate.
    # This must NOT raise — the fix has to separate "no implicit term" from
    # "implicit term I cannot diagonalize".
    exact = exp(-1.0)
    for ts in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
        N = 4
        dist, xb = _dimex_setup(N)
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g)
        fill!(Tarang.get_grid_data(u), 1.0)
        problem = IVP([u])
        add_equation!(problem, "dt(u) = -u")
        solver = InitialValueSolver(problem, ts; dt = 0.005)
        for _ in 1:200
            step!(solver)
        end
        ensure_layout!(u, :g)
        uf = Tarang.get_grid_data(u)[1]
        @test isfinite(uf)
        @test isapprox(uf, exact; rtol = 1e-3)
    end
end

@testset "D2: an attached operator still wins over the derived one" begin
    # Back-compat: `set_spectral_linear_operator!` remains authoritative, so the
    # documented workflow (`dt(u) = 0` plus an attached Laplacian) is unchanged.
    N = 16
    dist, xb = _dimex_setup(N)
    u = ScalarField(dist, "u", (xb,), Float64)
    ensure_layout!(u, :g)
    Tarang.get_grid_data(u) .= cos.(2 .* _dimex_grid(N))

    L = SpectralLinearOperator(dist, (xb,), :laplacian; ν = 0.5)
    problem = IVP([u])
    add_equation!(problem, "dt(u) = 0")
    solver = InitialValueSolver(problem, DiagonalIMEX_RK222(); dt = 0.005)
    Tarang.set_spectral_linear_operator!(solver, L)
    for _ in 1:200
        step!(solver)
    end
    ensure_layout!(u, :g)
    @test isapprox(maximum(abs, Tarang.get_grid_data(u)), exp(-2.0); rtol = 0.05)
end

@testset "D2: implicit and explicit terms combine as in the reference schemes" begin
    # `dt(u) + u = -0.5*u*u` splits into an implicit L (u) and an explicit
    # nonlinear F. Exact solution of u' = -u - u²/2, u(0)=1.
    exact = 1.0 / (1.5 * exp(1.0) - 0.5)
    function mixed(ts; dt = 0.002)
        N = 8
        dist, xb = _dimex_setup(N)
        u = ScalarField(dist, "u", (xb,), Float64)
        ensure_layout!(u, :g)
        fill!(Tarang.get_grid_data(u), 1.0)
        problem = IVP([u])
        add_equation!(problem, "dt(u) + u = -0.5*u*u")
        solver = InitialValueSolver(problem, ts; dt = dt)
        for _ in 1:round(Int, 1.0 / dt)
            step!(solver)
        end
        ensure_layout!(u, :g)
        return Tarang.get_grid_data(u)[1]
    end

    @test isapprox(mixed(DiagonalIMEX_RK222()), mixed(RK222()); rtol = 1e-8)
    @test isapprox(mixed(DiagonalIMEX_SBDF2()), mixed(SBDF2()); rtol = 1e-8)
    for ts in (DiagonalIMEX_RK222(), DiagonalIMEX_RK443(), DiagonalIMEX_SBDF2())
        @test isapprox(mixed(ts), exact; rtol = 1e-4)
    end
end

# ===========================================================================
# D3 — SpectralLinearOperator operator_type validation
# ===========================================================================
@testset "D3: unknown operator_type raises instead of yielding a zero operator" begin
    dist, xb = _dimex_setup(8)

    # `:viscosity` reads like a plausible name and silently produced
    # `extrema(L.coefficients) == (0.0, 0.0)` — an inviscid run, no complaint.
    for bad in (:viscosity, :diffusion, :hyper, :Laplacian, :bogus)
        @test_throws ArgumentError SpectralLinearOperator(dist, (xb,), bad; ν = 0.5)
    end

    err = try
        SpectralLinearOperator(dist, (xb,), :viscosity; ν = 0.5); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("viscosity", msg)                 # names the offending symbol
    for valid in SPECTRAL_OPERATOR_TYPES             # and lists every valid one
        @test occursin(":$(valid)", msg)
    end

    # `:custom` without coefficients is the same failure mode by another route.
    @test_throws ArgumentError SpectralLinearOperator(dist, (xb,), :custom; ν = 0.5)

    # The low-level kernel must refuse too, not silently return zero.
    @test_throws ArgumentError Tarang._spectral_operator_value(4.0, :viscosity, 0.5, 1, Float64)
end

@testset "D3: every valid operator_type builds a NON-ZERO operator" begin
    dist, xb = _dimex_setup(8)
    ν = 0.5

    for sym in SPECTRAL_OPERATOR_TYPES
        L = if sym === :custom
            SpectralLinearOperator(dist, (xb,), :custom; coefficients = fill(2.5, 5))
        else
            SpectralLinearOperator(dist, (xb,), sym; ν = ν, order = 2)
        end
        @test L.operator_type === sym
        @test any(!iszero, L.coefficients)            # the D3 failure mode
        @test all(isfinite, L.coefficients)
        @test all(real.(L.coefficients) .>= -1e-14)   # damping, not growth
    end

    # Spot-check the actual analytic values on the rfft layout k = 0,1,2,3,4.
    k = 0:4
    @test SpectralLinearOperator(dist, (xb,), :laplacian; ν = ν).coefficients ≈ ν .* k .^ 2
    @test SpectralLinearOperator(dist, (xb,), :hyperviscosity; ν = ν, order = 2).coefficients ≈
          ν .* (k .^ 2) .^ 2
    @test SpectralLinearOperator(dist, (xb,), :biharmonic; ν = ν).coefficients ≈ ν .* k .^ 4
end

@testset "D3: the documented symbol set is the enforced one" begin
    # SPECTRAL_OPERATOR_TYPES is the single source of truth shared by the
    # constructor's validation and the per-mode kernel.
    @test :laplacian in SPECTRAL_OPERATOR_TYPES
    @test :hyperviscosity in SPECTRAL_OPERATOR_TYPES
    @test :biharmonic in SPECTRAL_OPERATOR_TYPES
    @test :custom in SPECTRAL_OPERATOR_TYPES
    @test !(:viscosity in SPECTRAL_OPERATOR_TYPES)
    @test length(SPECTRAL_OPERATOR_TYPES) == 4
end

end # testset
