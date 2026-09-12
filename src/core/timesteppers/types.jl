# =============================================================================
# IMEX Runge-Kutta Methods 
# =============================================================================
# These are the default RK methods convention where
# linear terms (LHS) are treated implicitly and nonlinear terms (RHS) explicitly.
#
# Naming: RKabc where a=stages, b=explicit order, c=implicit order
# - RK111: 1st order IMEX (two-row tableau, one implicit solve per step)
# - RK222: 2-stage, 2nd order IMEX
# - RK443: 4-stage, 3rd order IMEX
# =============================================================================

struct RK111 <: TimeStepper
    """
    1st-order IMEX Runge-Kutta: backward Euler for the implicit (linear) part,
    forward Euler for the explicit part — Ascher, Ruuth & Spiteri (1997) (1,1,1),
    stored in the two-row form with an explicit initial stage:

    Explicit:          Implicit:
    0 | 0  0           0 | 0  0
    1 | 1  0           1 | 0  1
    --|------          --|------
      | 1  0             | 0  1

    One implicit solve per step:
        (M + dt·L) X_{n+1} = M X_n + dt·F(X_n, t)
    Both tableaux are stiffly accurate (b = last row of A), so the final mass
    update reproduces the last stage exactly.

    Why this form and not the single-row `A_exp = [0], b_exp = [1]` it used to be:
    that tableau evaluates F only AFTER the implicit stage — a Lie splitting
    X_{n+1} = X_1 + dt·F(X_1), X_1 the implicit solve — whose explicit part is
    not stiffly accurate (Σ A_exp[end,:] = 0 ≠ Σ b_exp = 1). On the tau /
    subproblem path the stage satisfies the boundary conditions but the final
    update then adds dt·F, so an explicit forcing that is not tangent to the
    boundary leaves an O(dt) boundary residual on EVERY step; the CPU final
    projector lifted that residual into the interior and the scheme did not
    converge at all (measured on a manufactured moving-boundary problem: error
    0.97 at dt=0.04 AND at dt=0.02, against 2.7e-2 → 1.3e-2 with this tableau).
    On pure-Fourier problems the two forms agree to roundoff.
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RK111()
        stages = 2
        # Explicit tableau (forward Euler, explicit first stage), c = [0, 1]
        A_explicit = [0.0 0.0;
                      1.0 0.0]
        b_explicit = [1.0, 0.0]
        c_explicit = [0.0, 1.0]
        # Implicit tableau (backward Euler on the second stage), c = [0, 1]
        A_implicit = [0.0 0.0;
                      0.0 1.0]
        b_implicit = [0.0, 1.0]
        c_implicit = [0.0, 1.0]
        new(stages, A_explicit, b_explicit, c_explicit, A_implicit, b_implicit, c_implicit)
    end
end

struct RK222 <: TimeStepper
    """Standard RK222. The tableau includes the initial stage.

    The explicit and implicit weights equal their final tableau rows.
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RK222()
        g = 1 - 1 / sqrt(2)
        d = 1 - 1 / (2g)
        A = [0.0 0 0; g 0 0; d 1-d 0]
        H = [0.0 0 0; 0 g 0; 0 1-g g]
        c = [0.0, g, 1.0]
        return new(3, A, A[end, :], c, H, H[end, :], copy(c))
    end
end

struct RK443 <: TimeStepper
    """Standard RK443. The tableau includes the initial stage.

    The explicit and implicit weights equal their final tableau rows.
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RK443()
        A = [0.0 0 0 0 0; 1/2 0 0 0 0; 11/18 1/18 0 0 0;
             5/6 -5/6 1/2 0 0; 1/4 7/4 3/4 -7/4 0]
        H = [0.0 0 0 0 0; 0 1/2 0 0 0; 0 1/6 1/2 0 0;
             0 -1/2 1/2 1/2 0; 0 3/2 -3/2 1/2 1/2]
        c = [0.0, 1/2, 2/3, 1/2, 1.0]
        return new(5, A, A[end, :], c, H, H[end, :], copy(c))
    end
end

# =============================================================================
# Multistep IMEX Methods
# =============================================================================

struct CNAB1 <: TimeStepper
    # Crank-Nicolson Adams-Bashforth 1st order
    stages::Int
    implicit_coefficient::Float64
    explicit_coefficients::Vector{Float64}

    function CNAB1()
        stages = 1
        implicit_coeff = 0.5  # Crank-Nicolson
        explicit_coeffs = [1.0]  # Forward Euler
        new(stages, implicit_coeff, explicit_coeffs)
    end
end

struct CNAB2 <: TimeStepper
    # Crank-Nicolson Adams-Bashforth 2nd order
    stages::Int
    implicit_coefficient::Float64
    explicit_coefficients::Vector{Float64}

    function CNAB2()
        stages = 2
        implicit_coeff = 0.5  # Crank-Nicolson
        explicit_coeffs = [1.5, -0.5]  # Adams-Bashforth 2
        new(stages, implicit_coeff, explicit_coeffs)
    end
end

# =============================================================================
# Semi-implicit Backwards Differentiation Formulas
# =============================================================================

struct SBDF1 <: TimeStepper
    # 1st order backwards differentiation formula
    order::Int
    coefficients::Vector{Float64}

    function SBDF1()
        order = 1
        coeffs = [1.0, -1.0]  # BDF1 coefficients
        new(order, coeffs)
    end
end

struct SBDF2 <: TimeStepper
    # 2nd order backwards differentiation formula
    order::Int
    coefficients::Vector{Float64}

    function SBDF2()
        order = 2
        coeffs = [3.0/2.0, -2.0, 1.0/2.0]  # BDF2 coefficients
        new(order, coeffs)
    end
end

struct SBDF3 <: TimeStepper
    # 3rd order backwards differentiation formula
    order::Int
    coefficients::Vector{Float64}

    function SBDF3()
        order = 3
        coeffs = [11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0]  # BDF3 coefficients
        new(order, coeffs)
    end
end

struct SBDF4 <: TimeStepper
    # 4th order backwards differentiation formula
    order::Int
    coefficients::Vector{Float64}

    function SBDF4()
        order = 4
        coeffs = [25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0]  # BDF4 coefficients
        new(order, coeffs)
    end
end

# =============================================================================
# Exponential Time Differencing (ETD) Methods
# =============================================================================

struct ETD_RK222 <: TimeStepper
    # 2nd-order exponential Runge-Kutta method
    stages::Int

    function ETD_RK222()
        stages = 2
        new(stages)
    end
end

struct ETD_CNAB2 <: TimeStepper
    # 2nd-order exponential Crank-Nicolson Adams-Bashforth
    stages::Int
    implicit_coefficient::Float64
    explicit_coefficients::Vector{Float64}

    function ETD_CNAB2()
        stages = 2
        implicit_coeff = 0.5  # Crank-Nicolson
        explicit_coeffs = [1.5, -0.5]  # Adams-Bashforth 2
        new(stages, implicit_coeff, explicit_coeffs)
    end
end

struct ETD_SBDF2 <: TimeStepper
    # 2nd-order exponential semi-implicit BDF
    order::Int
    coefficients::Vector{Float64}

    function ETD_SBDF2()
        order = 2
        coeffs = [3.0/2.0, -2.0, 1.0/2.0]  # BDF2 coefficients
        new(order, coeffs)
    end
end

# =============================================================================
# Global-Matrix Timesteppers
# =============================================================================

"""Modified CNAB2 with three implicit time levels and AB2 extrapolation."""
struct MCNAB2 <: TimeStepper
    stages::Int
    explicit_coefficients::Vector{Float64}

    function MCNAB2()
        new(2, [1.5, -0.5])
    end
end

struct CNLF2 <: TimeStepper
    """
    Crank-Nicolson Leapfrog 2nd order (also known as CNLF or Adam-Bashforth Leapfrog).

    Uses leapfrog for explicit extrapolation with Crank-Nicolson implicit treatment.

    This is a 2-step method that uses centered differences for explicit treatment:
    Implicit: Crank-Nicolson (θ = 0.5)
    Explicit: Leapfrog (centered 2-step extrapolation)

    Formula: (1 + θ*dt*L) X^{n+1} = (1 - (1-θ)*dt*L) X^{n-1} + 2*dt*F^n

    Variable dt: the stepper generalizes the stencils with exact nonuniform
    Lagrange weights (Wang 2008 eqn 2.11) and stays 2nd order through smooth or
    isolated dt changes. RAPIDLY ALTERNATING dt degrades it toward 1st order —
    the leapfrog parasitic mode has amplification |−w₁²| per step (w₁ = dt
    ratio), only marginally stable when the ratio oscillates. Prefer CNAB2 or
    SBDF2 under aggressive adaptive stepping.
    """
    stages::Int
    implicit_coefficient::Float64
    explicit_coefficients::Vector{Float64}

    function CNLF2()
        stages = 2
        implicit_coeff = 0.5  # Crank-Nicolson
        explicit_coeffs = [2.0, 0.0, 0.0]  # Leapfrog uses F^n only with factor 2
        new(stages, implicit_coeff, explicit_coeffs)
    end
end

struct RKSMR <: TimeStepper
    """
    Spalart–Moser–Rogers (SMR) semi-implicit (IMEX) Runge–Kutta scheme.

    The classic three-substep low-storage IMEX-RK of Spalart, Moser & Rogers
    (J. Comput. Phys. 1991), the workhorse time integrator for incompressible
    spectral DNS (Kim–Moin–Moser channel flow and descendants). It treats the
    nonlinear/advective term `F` EXPLICITLY (3rd-order RK, with a two-substep
    Adams-Bashforth-like blend) and the stiff linear term `L` (viscous diffusion)
    IMPLICITLY (Crank–Nicolson-like, 2nd order, stiffly stable but not L-stable):

        (M - dt·β_k L) y^k = y^{k-1} + dt[γ_k F^{k-1} + ζ_k F^{k-2} + dt·α_k L y^{k-1}]
        γ = (8/15, 5/12, 3/4),  ζ = (0, -17/60, -5/12)
        α = (29/96, -3/40, 1/6), β = (37/160, 5/24, 1/6),  α_k+β_k = γ_k+ζ_k

    Here it is stored in the equivalent 4-stage ESDIRK additive-Runge–Kutta
    (ARK) Butcher form so it shares the generic IMEX driver `step_rk_imex!`
    (M/L matrices, per-mode subproblems, distributed diagonal IMEX, fallbacks)
    with RK222/RK443. Stage 1 is the trivial explicit-first stage (= yⁿ); stages
    2–4 are the three SMR substeps. Cumulative explicit/implicit coefficients are
    the running sums of (γ,ζ) and (α,β). Stiffly accurate: b = last stage row.

    Properties:
    - 3rd-order accurate for the explicit (nonlinear) part, 2nd-order for the
      implicit (linear) part — the standard SMR accuracy profile.
    - Implicit linear treatment is stable for diffusion-dominated problems
      (previously this method silently ran fully explicit and blew up on stiff L).
      Its stiff-limit amplification is `87/185 ≈ 0.47027`, so very stiff modes
      remain bounded but are not annihilated in one step as an L-stable method
      would do.
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RKSMR()
        stages = 4
        γ = (8/15, 5/12, 3/4)
        ζ = (0.0, -17/60, -5/12)
        α = (29/96, -3/40, 1/6)
        β = (37/160, 5/24, 1/6)

        # Explicit ARK tableau — cumulative running sums of (γ, ζ).
        A_explicit = [
            0.0          0.0          0.0   0.0;
            γ[1]         0.0          0.0   0.0;
            γ[1]+ζ[2]    γ[2]         0.0   0.0;
            γ[1]+ζ[2]    γ[2]+ζ[3]    γ[3]  0.0
        ]
        # Implicit ARK tableau — cumulative running sums of (α, β); β on diagonal.
        A_implicit = [
            0.0     0.0          0.0          0.0;
            α[1]    β[1]         0.0          0.0;
            α[1]    β[1]+α[2]    β[2]         0.0;
            α[1]    β[1]+α[2]    β[2]+α[3]    β[3]
        ]
        # Stiffly accurate: weights = last stage row → y^{n+1} = final substep.
        b_explicit = A_explicit[4, :]
        b_implicit = A_implicit[4, :]
        c_explicit = [0.0, γ[1], γ[1]+ζ[2]+γ[2], 1.0]   # [0, 8/15, 2/3, 1]
        c_implicit = copy(c_explicit)
        new(stages, A_explicit, b_explicit, c_explicit, A_implicit, b_implicit, c_implicit)
    end
end

struct RKGFY <: TimeStepper
    """Standard RKGFY. The tableau includes the initial stage.

    The explicit and implicit weights equal their final tableau rows.
    """
    stages::Int
    # Explicit Butcher tableau
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    # Implicit Butcher tableau (DIRK - diagonal implicit)
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RKGFY()
        A = [0.0 0 0; 1 0 0; 1/2 1/2 0]
        H = [0.0 0 0; 1/2 1/2 0; 1/2 0 1/2]
        c = [0.0, 1.0, 1.0]
        return new(3, A, A[end, :], c, H, H[end, :], copy(c))
    end
end

struct RK443_IMEX <: TimeStepper
    """
    4-stage 3rd-order IMEX Runge-Kutta method.

    Uses the same Ascher-Ruuth-Spiteri coefficients as RK443.
    This type exists as an alias for use in contexts where the "_IMEX" suffix
    makes the intent clearer.

    Properties:
    - L-stable implicit part (stiff decay)
    - 3rd order accuracy for both parts
    - ESDIRK structure (explicit first stage, same diagonal thereafter)
    - Stiffly accurate (last row of A_implicit equals b_implicit)
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RK443_IMEX()
        # Use the same Ascher-Ruuth-Spiteri coefficients as RK443
        # (same as RK443)
        rk = RK443()
        new(rk.stages, rk.A_explicit, rk.b_explicit, rk.c_explicit,
            rk.A_implicit, rk.b_implicit, rk.c_implicit)
    end
end

# =============================================================================
# Diagonal IMEX Methods (GPU-native)
# =============================================================================

"""
    DiagonalIMEX_RK222 <: TimeStepper

Internal 2nd-order IMEX Runge-Kutta with diagonal spectral implicit treatment.
Users select `RK222()`; GPU diagonal execution is chosen automatically.

For pseudospectral methods where the linear operator is diagonal in
Fourier space, this method avoids sparse matrix solves entirely.

The implicit step (I + dt*γ*L)⁻¹ * RHS becomes element-wise division:
    û_new = RHS ./ (1 .+ dt * γ .* L_diagonal)

This stays 100% on GPU with no CPU transfers.

# Usage
```julia
ts = RK222()
L = SpectralLinearOperator(dist, bases, :hyperviscosity; ν=1e-10, order=4)

# Set up solver with spectral operator
solver = InitialValueSolver(problem, ts)
set_spectral_linear_operator!(solver, L)
```
"""
struct DiagonalIMEX_RK222 <: TimeStepper
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    b_implicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}  # Full ESDIRK implicit tableau (off-diagonal terms required for L-stability)
    γ::Float64  # Implicit diagonal coefficient (A_implicit[s,s] for s≥2)

    function DiagonalIMEX_RK222()
        rk = RK222()
        new(rk.stages, rk.A_explicit, rk.b_explicit, rk.b_implicit,
            rk.c_explicit, rk.A_implicit, rk.A_implicit[2, 2])
    end
end

"""
    DiagonalIMEX_RK443 <: TimeStepper

Internal 3rd-order IMEX Runge-Kutta with diagonal spectral implicit treatment.
Users select `RK443()`; GPU diagonal execution is chosen automatically.

Uses the same explicit and implicit tableaux as RK443. This ensures the
IMEX coupling conditions are satisfied for 3rd-order accuracy.

Higher-order version for better accuracy with larger timesteps.
"""
struct DiagonalIMEX_RK443 <: TimeStepper
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    b_implicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}       # Full ESDIRK implicit tableau (off-diagonal terms required for L-stability)
    A_implicit_diag::Vector{Float64}  # Diagonal implicit coefficients (= diag(A_implicit))

    function DiagonalIMEX_RK443()
        rk = RK443()
        new(rk.stages, rk.A_explicit, rk.b_explicit, rk.b_implicit,
            rk.c_explicit, rk.A_implicit, diag(rk.A_implicit))
    end
end

"""
    DiagonalIMEX_SBDF2 <: TimeStepper

Internal 2nd-order SBDF with diagonal spectral implicit treatment.
Users select `SBDF2()`; GPU diagonal execution is chosen automatically.

Multi-step method that's efficient for steady-state problems.
"""
struct DiagonalIMEX_SBDF2 <: TimeStepper
    order::Int

    function DiagonalIMEX_SBDF2()
        new(2)
    end
end

"""Both additive tableaux return their last stage as the advanced solution."""
function _rk_stiffly_accurate(ts)
    return ts.c_explicit[end] == 1 &&
           ts.b_explicit == ts.A_explicit[end, :] &&
           ts.b_implicit == ts.A_implicit[end, :]
end
