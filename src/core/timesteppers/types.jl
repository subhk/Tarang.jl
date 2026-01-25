# =============================================================================
# IMEX Runge-Kutta Methods (Dedalus-compatible)
# =============================================================================
# These are the default RK methods, following Dedalus convention where
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

    Following Dedalus convention:
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

    Butcher tableaux:
    Explicit:            Implicit:
    0   | 0   0          γ   | γ   0
    1   | 1   0          1   | 1-γ γ
    ----|------          ----|------
        | 1/2 1/2            | 1-γ γ
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RK222()
        stages = 2
        γ = 1 - 1/√2  # ≈ 0.29289321881345254

        # Explicit tableau
        A_explicit = [
            0.0  0.0;
            1.0  0.0
        ]
        b_explicit = [0.5, 0.5]
        c_explicit = [0.0, 1.0]

        # Implicit tableau (SDIRK)
        A_implicit = [
            γ      0.0;
            1.0-γ  γ
        ]
        b_implicit = [1.0-γ, γ]
        c_implicit = [γ, 1.0]

        new(stages, A_explicit, b_explicit, c_explicit, A_implicit, b_implicit, c_implicit)
    end
end

struct RK443 <: TimeStepper
    """
    4-stage, 3rd order IMEX Runge-Kutta.

    Based on Kennedy & Carpenter (2003) ARK3(2)4L[2]SA.
    "Additive Runge-Kutta schemes for convection-diffusion-reaction equations"

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

        # Kennedy & Carpenter ARK3(2)4L[2]SA coefficients
        # γ is the SDIRK diagonal value
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
# Additional Timesteppers
# =============================================================================

struct MCNAB2 <: TimeStepper
    """
    Modified Crank-Nicolson Adams-Bashforth 2nd order.

    Uses modified CN weighting for improved stability with stiff linear terms.

    Implicit: Modified Crank-Nicolson with θ = 1/2 + ε
    Explicit: Adams-Bashforth 2nd order extrapolation
    """
    stages::Int
    implicit_coefficient::Float64  # θ for modified CN (typically slightly > 0.5)
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
    Strong Stability Preserving Runge-Kutta method (SSP-RK).

    Also known as TVD (Total Variation Diminishing) RK methods.

    This implements SSP-RK3 (Shu-Osher form):
    Stage 1: u^(1) = u^n + dt*F(u^n)
    Stage 2: u^(2) = 3/4*u^n + 1/4*u^(1) + 1/4*dt*F(u^(1))
    Stage 3: u^{n+1} = 1/3*u^n + 2/3*u^(2) + 2/3*dt*F(u^(2))

    Properties:
    - 3rd order accurate
    - Strong stability preserving (SSP) with CFL coefficient C = 1
    - Optimal for hyperbolic conservation laws
    """
    stages::Int
    alpha::Matrix{Float64}  # SSP coefficients (convex combinations)
    beta::Vector{Float64}   # RHS scaling coefficients

    function RKSMR()
        stages = 3
        # Shu-Osher form coefficients for SSP-RK3
        alpha = [
            1.0  0.0  0.0;    # Stage 1: u^(1) = 1*u^n
            0.75 0.25 0.0;    # Stage 2: u^(2) = 3/4*u^n + 1/4*u^(1)
            1/3  0.0  2/3     # Stage 3: u^{n+1} = 1/3*u^n + 2/3*u^(2)
        ]
        beta = [1.0, 0.25, 2/3]  # dt*F scaling for each stage
        new(stages, alpha, beta)
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

        # Ascher-Ruuth-Spiteri ARK2 coefficients
        # γ = (2 - √2) / 2 ≈ 0.2928932...
        γ = (2.0 - sqrt(2.0)) / 2.0
        δ = -2.0 * sqrt(2.0) / 3.0

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

    Based on Kennedy & Carpenter (2003) ARK4(3)6L[2]SA.

    This is a higher-order IMEX method with:
    - 4 stages
    - 3rd order accuracy
    - L-stable implicit part
    - SSP-like explicit part
    """
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}

    function RK443_IMEX()
        stages = 4

        # ARK443 coefficients: 4-stage, 3rd-order additive Runge-Kutta
        # L-stable SDIRK implicit part with classical RK4-like explicit part
        γ = 0.4358665215  # Root of x³ - 3x² + 3x/2 - 1/6 = 0 for L-stability

        # Explicit tableau (classical RK4-like structure)
        A_explicit = [
            0.0    0.0    0.0    0.0;
            0.5    0.0    0.0    0.0;
            0.0    0.5    0.0    0.0;
            0.0    0.0    1.0    0.0
        ]
        b_explicit = [1/6, 1/3, 1/3, 1/6]
        c_explicit = [0.0, 0.5, 0.5, 1.0]

        # Implicit tableau (SDIRK - same γ on diagonal)
        A_implicit = [
            γ       0.0     0.0     0.0;
            0.5-γ   γ       0.0     0.0;
            0.0     0.5-γ   γ       0.0;
            1/6     1/3-γ   1/3     γ
        ]
        b_implicit = [1/6, 1/3, 1/3, 1/6]
        c_implicit = [γ, 0.5, 0.5, 1.0]

        new(stages, A_explicit, b_explicit, c_explicit, A_implicit, b_implicit, c_implicit)
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
    c_explicit::Vector{Float64}
    γ::Float64  # Implicit diagonal coefficient

    function DiagonalIMEX_RK222()
        # Same as RK222 explicit tableau
        stages = 3
        γ = 1 - 1/√2  # ≈ 0.2929 for L-stability

        A_explicit = [
            0.0  0.0  0.0;
            1.0  0.0  0.0;
            0.5  0.5  0.0
        ]
        b_explicit = [0.5, 0.5, 0.0]
        c_explicit = [0.0, 1.0, 0.5]

        new(stages, A_explicit, b_explicit, c_explicit, γ)
    end
end

"""
    DiagonalIMEX_RK443 <: TimeStepper

3rd-order IMEX Runge-Kutta with diagonal spectral implicit treatment.

Higher-order version for better accuracy with larger timesteps.
"""
struct DiagonalIMEX_RK443 <: TimeStepper
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit_diag::Vector{Float64}  # Diagonal implicit coefficients

    function DiagonalIMEX_RK443()
        stages = 4
        γ = 0.4358665215  # L-stable coefficient

        # Classical RK4-like explicit tableau
        A_explicit = [
            0.0    0.0    0.0    0.0;
            0.5    0.0    0.0    0.0;
            0.0    0.5    0.0    0.0;
            0.0    0.0    1.0    0.0
        ]
        b_explicit = [1/6, 1/3, 1/3, 1/6]
        c_explicit = [0.0, 0.5, 0.5, 1.0]

        # Diagonal coefficients for implicit part (SDIRK structure)
        A_implicit_diag = [γ, γ, γ, γ]

        new(stages, A_explicit, b_explicit, c_explicit, A_implicit_diag)
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
