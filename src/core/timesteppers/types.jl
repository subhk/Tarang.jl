# =============================================================================
# IMEX Runge-Kutta Methods 
# =============================================================================
# These are the default RK methods convention where
# linear terms (LHS) are treated implicitly and nonlinear terms (RHS) explicitly.
#
# Naming: RKabc where a=stages, b=explicit order, c=implicit order
# - RK111: 1-stage, 1st order IMEX
# - RK222: 2-stage, 2nd order IMEX
# - RK443: 4-stage, 3rd order IMEX
# =============================================================================

struct RK111 <: TimeStepper
    """
    1-stage, 1st order IMEX Runge-Kutta (Backward Euler / Forward Euler).

    Following spectral convention:
    - Implicit: Backward Euler (treats linear terms)
    - Explicit: Forward Euler (treats nonlinear terms)

    Butcher tableaux:
    Explicit:          Implicit:
    0 |               1 | 1
    --|--             --|--
      | 1               | 1
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RK111()
        stages = 1
        # Explicit tableau (Forward Euler)
        A_explicit = reshape([0.0], 1, 1)
        b_explicit = [1.0]
        c_explicit = [0.0]
        # Implicit tableau (Backward Euler)
        A_implicit = reshape([1.0], 1, 1)
        b_implicit = [1.0]
        c_implicit = [1.0]
        new(stages, A_explicit, b_explicit, c_explicit, A_implicit, b_implicit, c_implicit)
    end
end

struct RK222 <: TimeStepper
    """
    2-stage, 2nd order IMEX Runge-Kutta.

    Based on Ascher, Ruuth, Spiteri (1997) ARK2(2,2,2).
    "Implicit-Explicit Runge-Kutta Methods for Time-Dependent PDEs"

    γ = 1 - 1/√2 ≈ 0.29289321881345254

    Properties:
    - L-stable implicit part (stiff decay)
    - 2nd order accuracy for both parts
    - SDIRK structure (same diagonal)

    Butcher tableaux (3-stage form, shared abscissae c = [0, γ, 1]):
    Explicit:                  Implicit (ESDIRK):
    0   | 0    0    0          0   | 0    0    0
    γ   | γ    0    0          γ   | 0    γ    0
    1   | γ    1-γ  0          1   | 0    1-γ  γ
    ----|------------          ----|------------
        | 0    1-γ  γ              | 0    1-γ  γ
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RK222()
        # ARS(2,2,2) — Ascher, Ruuth, Spiteri (1997). 3-stage form. The explicit
        # and implicit tableaux MUST share the abscissae c=[0,γ,1] for the IMEX
        # combination to be 2nd order (the explicit and implicit stages have to be
        # evaluated at the same stage times). The previous 2-stage form had
        # c_explicit=[0,1] ≠ c_implicit=[γ,1], which silently dropped it to 1st order.
        stages = 3
        γ = 1 - 1/√2  # ≈ 0.29289321881345254

        # Explicit tableau (ERK), c = [0, γ, 1]
        A_explicit = [
            0.0    0.0      0.0;
            γ      0.0      0.0;
            γ      1.0-γ    0.0
        ]
        b_explicit = [0.0, 1.0-γ, γ]
        c_explicit = [0.0, γ, 1.0]

        # Implicit tableau (ESDIRK, explicit first stage, same γ diagonal), c = [0, γ, 1]
        A_implicit = [
            0.0    0.0      0.0;
            0.0    γ        0.0;
            0.0    1.0-γ    γ
        ]
        b_implicit = [0.0, 1.0-γ, γ]
        c_implicit = [0.0, γ, 1.0]

        new(stages, A_explicit, b_explicit, c_explicit, A_implicit, b_implicit, c_implicit)
    end
end

struct RK443 <: TimeStepper
    """
    4-stage, 3rd order IMEX Runge-Kutta.

    ARK built on Alexander's (1977) 3-stage L-stable SDIRK3 embedded as ESDIRK:
    γ ≈ 0.435866521508459 is the root of x³ − 3x² + (3/2)x − 1/6, with
    c = [0, γ, (1+γ)/2, 1] and the classical b₂, b₃ weights. (Note: this is
    NOT Kennedy & Carpenter's ARK3(2)4L[2]SA, whose c₂ = 2γ, nor ARS(4,4,3);
    the full order-3 ARK coupling conditions are nonetheless satisfied —
    verified numerically to ≤1e-11.)

    Properties:
    - L-stable implicit part (stiff decay)
    - 3rd order accuracy for both parts
    - 4 stages for improved stability region
    - ESDIRK structure (explicit first stage, same diagonal thereafter)

    This is the recommended timestepper for most problems with both
    stiff linear terms (diffusion) and nonlinear advection.
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RK443()
        stages = 4

        # Alexander SDIRK3-based ARK coefficients (see docstring)
        # γ is the SDIRK diagonal value: root of x³ − 3x² + (3/2)x − 1/6
        γ = 0.4358665215084590  # Root for L-stability

        # Explicit tableau (ERK part)
        A_explicit = [
            0.0                  0.0                  0.0                  0.0;
            0.4358665215084590   0.0                  0.0                  0.0;
            0.3212788860285571   0.3966543747256624   0.0                  0.0;
            -0.105858296071263   0.5529291479590279   0.5529291481122351   0.0
        ]
        b_explicit = [0.0, 1.208496649176010, -0.6443631706844688, 0.4358665215084590]
        c_explicit = [0.0, γ, 0.7179332607542195, 1.0]

        # Implicit tableau (ESDIRK part - same γ on diagonal after first stage)
        A_implicit = [
            0.0   0.0   0.0   0.0;
            0.0   γ     0.0   0.0;
            0.0   0.2820667392457805  γ     0.0;
            0.0   1.208496649176010  -0.6443631706844688  γ
        ]
        b_implicit = [0.0, 1.208496649176010, -0.6443631706844688, γ]
        c_implicit = [0.0, γ, 0.7179332607542195, 1.0]

        new(stages, A_explicit, b_explicit, c_explicit, A_implicit, b_implicit, c_implicit)
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

struct MCNAB2 <: TimeStepper
    """
    Crank-Nicolson / Adams-Bashforth — 2nd order at the default θ = 1/2.

    Implicit: θ-weighted treatment of the linear operator (θ on level n+1, 1−θ on n).
    Explicit: 2nd-order Adams-Bashforth extrapolation of the nonlinear term.

    θ = 1/2 (default) is Crank-Nicolson and is **2nd order**. θ > 1/2 adds damping
    for stiff linear terms but is only **1st order** — this is a 2-level θ-method,
    NOT the 3-level Ascher-Ruuth-Wetton "modified CNAB" (stencil 1/2+γ, 1/2−2γ, γ)
    that retains 2nd order with damping. Use θ = 1/2 unless you specifically want the
    extra (1st-order) damping.
    """
    stages::Int
    implicit_coefficient::Float64  # θ-weight: 0.5 = Crank-Nicolson (2nd order); >0.5 = damped, 1st order
    explicit_coefficients::Vector{Float64}

    function MCNAB2(theta::Float64=0.5)
        stages = 2
        implicit_coeff = theta  # Modified CN parameter
        explicit_coeffs = [1.5, -0.5]  # Adams-Bashforth 2
        new(stages, implicit_coeff, explicit_coeffs)
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
    IMPLICITLY (Crank–Nicolson, 2nd order, L-stable for the linear part):

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
    """
    General Framework Runge-Kutta IMEX method (RKGFY).

    This is the ARK (Additive Runge-Kutta) form for IMEX problems.

    Implements the 2nd-order L-stable IMEX scheme from Ascher, Ruuth, Spiteri (1997):
    "Implicit-Explicit Runge-Kutta Methods for Time-Dependent PDEs"

    The method uses:
    - Explicit tableau for nonlinear/advection terms
    - Implicit (DIRK) tableau for stiff linear terms

    This is a 3-stage, 2nd-order method with good stability properties.
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
        stages = 3

        # Ascher-Ruuth-Spiteri ARK2(2,2,2) coefficients
        # Reference: Ascher, Ruuth, Spiteri (1997), Table 1
        # γ = (2 - √2) / 2 ≈ 0.2928932..., δ = 1 - 1/(2γ) ≈ -0.7071...
        γ = (2.0 - sqrt(2.0)) / 2.0
        δ = 1.0 - 1.0 / (2.0 * γ)

        # Explicit tableau (lower triangular, zeros on diagonal)
        A_explicit = [
            0.0     0.0     0.0;
            γ       0.0     0.0;
            δ       1.0-δ   0.0
        ]
        b_explicit = [0.0, 1.0-γ, γ]
        c_explicit = [0.0, γ, 1.0]

        # Implicit tableau (DIRK - γ on diagonal)
        A_implicit = [
            0.0     0.0     0.0;
            0.0     γ       0.0;
            0.0     1.0-γ   γ
        ]
        b_implicit = [0.0, 1.0-γ, γ]
        c_implicit = [0.0, γ, 1.0]

        new(stages, A_explicit, b_explicit, c_explicit, A_implicit, b_implicit, c_implicit)
    end
end

struct RK443_IMEX <: TimeStepper
    """
    4-stage 3rd-order IMEX Runge-Kutta method.

    Uses the same Alexander-SDIRK3-based ARK coefficients as RK443.
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
        # Use the same Alexander-SDIRK3-based ARK coefficients as RK443
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

2nd-order IMEX Runge-Kutta with diagonal spectral implicit treatment.

For pseudospectral methods where the linear operator is diagonal in
Fourier space, this method avoids sparse matrix solves entirely.

The implicit step (I + dt*γ*L)⁻¹ * RHS becomes element-wise division:
    û_new = RHS ./ (1 .+ dt * γ .* L_diagonal)

This stays 100% on GPU with no CPU transfers.

# Usage
```julia
ts = DiagonalIMEX_RK222()
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
        # Ascher-Ruuth-Spiteri (1997) ARS(2,2,2) — a matched, L-stable, 2nd-order
        # IMEX pair. The implicit part is solved DIAGONALLY per Fourier mode (vs a
        # global matrix), but it must carry the FULL ESDIRK tableau: omitting the
        # off-diagonal implicit terms makes R(z)→1−1/γ≈−2.41 (|R|>1 for dt·λ≳4.8),
        # i.e. unstable in the stiff limit — the opposite of L-stable.
        stages = 3
        γ = 1 - 1/√2  # ≈ 0.29289321881345254

        # Explicit tableau (ERK), c = [0, γ, 1]
        A_explicit = [
            0.0    0.0      0.0;
            γ      0.0      0.0;
            γ      1.0-γ    0.0
        ]
        b_explicit = [0.0, 1.0-γ, γ]
        c_explicit = [0.0, γ, 1.0]

        # Implicit tableau (ESDIRK, explicit first stage, same γ diagonal), c = [0, γ, 1]
        A_implicit = [
            0.0    0.0      0.0;
            0.0    γ        0.0;
            0.0    1.0-γ    γ
        ]
        b_implicit = [0.0, 1.0-γ, γ]

        new(stages, A_explicit, b_explicit, b_implicit, c_explicit, A_implicit, γ)
    end
end

"""
    DiagonalIMEX_RK443 <: TimeStepper

3rd-order IMEX Runge-Kutta with diagonal spectral implicit treatment.

Uses the same Alexander-SDIRK3-based ARK explicit tableau as RK443
paired with its ESDIRK diagonal for the implicit part. This ensures the
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
        stages = 4
        γ = 0.4358665215084590  # L-stable SDIRK3 root (Alexander)

        # Alexander-SDIRK3-based ARK explicit tableau (matches RK443)
        A_explicit = [
            0.0                  0.0                  0.0                  0.0;
            0.4358665215084590   0.0                  0.0                  0.0;
            0.3212788860285571   0.3966543747256624   0.0                  0.0;
            -0.105858296071263   0.5529291479590279   0.5529291481122351   0.0
        ]
        b_explicit = [0.0, 1.208496649176010, -0.6443631706844688, 0.4358665215084590]
        b_implicit = [0.0, 1.208496649176010, -0.6443631706844688, γ]  # Implicit weights from ESDIRK tableau
        c_explicit = [0.0, γ, 0.7179332607542195, 1.0]

        # Full ESDIRK implicit tableau (matches RK443; off-diagonal terms are
        # essential — dropping them is the stiff-limit instability bug).
        A_implicit = [
            0.0   0.0                 0.0                  0.0;
            0.0   γ                   0.0                  0.0;
            0.0   0.2820667392457805  γ                    0.0;
            0.0   1.208496649176010  -0.6443631706844688   γ
        ]
        A_implicit_diag = [0.0, γ, γ, γ]

        new(stages, A_explicit, b_explicit, b_implicit, c_explicit, A_implicit, A_implicit_diag)
    end
end

"""
    DiagonalIMEX_SBDF2 <: TimeStepper

2nd-order SBDF with diagonal spectral implicit treatment.

Multi-step method that's efficient for steady-state problems.
"""
struct DiagonalIMEX_SBDF2 <: TimeStepper
    order::Int

    function DiagonalIMEX_SBDF2()
        new(2)
    end
end
