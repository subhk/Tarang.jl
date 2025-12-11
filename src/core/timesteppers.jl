"""
Timestepping schemes for initial value problems
"""

using LinearAlgebra
using LinearAlgebra: BLAS
using LoopVectorization  # For SIMD loops
using ExponentialUtilities  # For Krylov-based φ functions

abstract type TimeStepper end

# Exponential Time Differencing (ETD) utility functions
"""
Compute φ functions for exponential time differencing methods.

φ₀(z) = exp(z)
φ₁(z) = (exp(z) - 1) / z
φ₂(z) = (exp(z) - 1 - z) / z²
φ₃(z) = (exp(z) - 1 - z - z²/2) / z³

These functions handle the z ≈ 0 case using Taylor expansions.
"""
function phi_functions(z::Number)
    if abs(z) < 1e-8
        # Use Taylor expansions for small z to avoid numerical issues
        φ₀ = 1 + z + z^2/2 + z^3/6 + z^4/24
        φ₁ = 1 + z/2 + z^2/6 + z^3/24 + z^4/120
        φ₂ = 1/2 + z/6 + z^2/24 + z^3/120 + z^4/720
        φ₃ = 1/6 + z/24 + z^2/120 + z^3/720 + z^4/5040
    else
        exp_z = exp(z)
        φ₀ = exp_z
        φ₁ = (exp_z - 1) / z
        φ₂ = (exp_z - 1 - z) / z^2
        φ₃ = (exp_z - 1 - z - z^2/2) / z^3
    end
    return φ₀, φ₁, φ₂, φ₃
end

function phi_functions_matrix(A::AbstractMatrix, dt::Float64)
    """Compute matrix φ functions for exponential integrators"""

    z = dt * A

    # Check matrix norm for stability
    z_norm = norm(z)

    if z_norm < 1e-8
        # Use Taylor expansions for small matrices (numerically stable)
        I = Matrix{eltype(A)}(LinearAlgebra.I, size(A, 1), size(A, 1))

        # Taylor series: φ₀(z) = I + z + z²/2 + z³/6 + z⁴/24
        exp_z = I + z + (z^2)/2 + (z^3)/6 + (z^4)/24

        # Taylor series: φ₁(z) = I + z/2 + z²/6 + z³/24 + z⁴/120
        φ₁ = I + z/2 + (z^2)/6 + (z^3)/24 + (z^4)/120

        # Taylor series: φ₂(z) = I/2 + z/6 + z²/24 + z³/120 + z⁴/720
        φ₂ = I/2 + z/6 + (z^2)/24 + (z^3)/120 + (z^4)/720

        return exp_z, φ₁, φ₂

    elseif z_norm < 50.0
        # Use matrix exponential for moderate matrices
        I = Matrix{eltype(A)}(LinearAlgebra.I, size(A, 1), size(A, 1))

        try
            exp_z = exp(z)

            # Use stable computation
            φ₁ = _compute_phi1_stable(z, exp_z, I)
            φ₂ = _compute_phi2_stable(z, exp_z, I, φ₁)

            return exp_z, φ₁, φ₂

        catch e
            @warn "Matrix exponential failed: $e, using Padé approximation"
            return _phi_functions_pade(z)
        end
    else
        # Use Krylov subspace methods for large/stiff matrices
        @warn "Matrix is large or stiff (norm=$z_norm), using Krylov approximation"
        return _phi_functions_krylov(z)
    end
end

function _compute_phi1_stable(z, exp_z, I)
    """Stable computation of φ₁"""
    z_norm = norm(z)
    if z_norm < 1e-2
        # Use series expansion for better accuracy
        return I + z/2 + z^2/6 + z^3/24 + z^4/120
    else
        return (exp_z - I) * inv(z)
    end
end

function _compute_phi2_stable(z, exp_z, I, φ₁)
    """
    Stable computation of φ₂(z) = (exp(z) - 1 - z) / z² = (φ₁(z) - I) / z

    Reference: Cox & Matthews (2002), Hochbruck & Ostermann (2010)
    """
    z_norm = norm(z)
    if z_norm < 1e-2
        # Use series expansion for better accuracy near z=0
        # φ₂(z) = 1/2 + z/6 + z²/24 + z³/120 + z⁴/720 + O(z⁵)
        return I/2 + z/6 + z^2/24 + z^3/120 + z^4/720
    else
        # Standard formula: φ₂ = (φ₁ - I) / z
        return (φ₁ - I) * inv(z)
    end
end

function _phi_functions_pade(z)
    """Padé approximation fallback for φ functions"""
    I = Matrix{eltype(z)}(LinearAlgebra.I, size(z, 1), size(z, 1))

    # Simple Padé [1/1] approximation for demonstration
    # In practice, would use higher-order approximations
    exp_z = (I + z/2) * inv(I - z/2)  # Padé [1/1] for exp
    φ₁ = inv(z) * (exp_z - I)
    φ₂ = inv(z^2) * (exp_z - I - z)

    return exp_z, φ₁, φ₂
end

function _phi_functions_krylov(A::AbstractMatrix, krylov_dim::Int=30)
    """
    Krylov subspace approximation for φ functions using ExponentialUtilities.jl.

    Uses the phiv function which computes [φ₀(A)b, φ₁(A)b, ..., φₖ(A)b] efficiently
    via Krylov subspace methods (Arnoldi iteration).

    For matrix φ functions, we compute φₖ(A) by applying to identity vectors.
    """
    n = size(A, 1)
    I_mat = Matrix{eltype(A)}(LinearAlgebra.I, n, n)

    # Allocate result matrices
    exp_A = similar(I_mat)
    φ₁ = similar(I_mat)
    φ₂ = similar(I_mat)

    # Use ExponentialUtilities.phiv to compute φ functions column by column
    # phiv(t, A, b, k) returns [φ₀(tA)b, φ₁(tA)b, ..., φₖ(tA)b]
    # We use t=1 since A already contains the timestep scaling

    try
        for j in 1:n
            # Unit vector e_j
            e_j = zeros(eltype(A), n)
            e_j[j] = one(eltype(A))

            # Compute φ functions applied to e_j using Krylov methods
            # phiv returns a matrix where columns are φ₀(A)e_j, φ₁(A)e_j, φ₂(A)e_j
            phi_result = phiv(1.0, A, e_j, 2; m=min(krylov_dim, n))

            # Extract columns for each φ function
            exp_A[:, j] = phi_result[:, 1]  # φ₀(A)e_j = exp(A)e_j
            φ₁[:, j] = phi_result[:, 2]     # φ₁(A)e_j
            φ₂[:, j] = phi_result[:, 3]     # φ₂(A)e_j
        end

        return exp_A, φ₁, φ₂

    catch e
        @warn "Krylov φ computation failed: $e, falling back to direct method"
        # Fallback to direct computation
        try
            exp_A = exp(A)
            φ₁ = (exp_A - I_mat) * inv(A)
            φ₂ = (exp_A - I_mat - A) * inv(A^2)
            return exp_A, φ₁, φ₂
        catch e2
            @error "All φ function computations failed: $e2"
            return I_mat, I_mat, I_mat/2
        end
    end
end

function phiv_vector(t::Real, A::AbstractMatrix, b::AbstractVector, k::Int; m::Int=30)
    """
    Compute [φ₀(tA)b, φ₁(tA)b, ..., φₖ(tA)b] using Krylov subspace methods.

    This is a convenience wrapper around ExponentialUtilities.phiv for
    computing φ-function vector products efficiently.

    Arguments:
    - t: Time scaling factor
    - A: Matrix (typically the linear operator L)
    - b: Vector to apply φ functions to
    - k: Maximum φ index to compute (computes φ₀ through φₖ)
    - m: Krylov subspace dimension (default 30)

    Returns:
    - Matrix of size (n, k+1) where column j+1 contains φⱼ(tA)b
    """
    return phiv(t, A, b, k; m=min(m, length(b)))
end

function expv_krylov(t::Real, A::AbstractMatrix, b::AbstractVector; m::Int=30)
    """
    Compute exp(tA)b using Krylov subspace methods.

    More efficient than computing exp(tA) and then multiplying by b,
    especially for large sparse matrices.

    Arguments:
    - t: Time scaling factor
    - A: Matrix (typically the linear operator L)
    - b: Vector to apply exponential to
    - m: Krylov subspace dimension (default 30)

    Returns:
    - Vector exp(tA)b
    """
    return expv(t, A, b; m=min(m, length(b)))
end

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

# Implicit-explicit methods
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

# Semi-implicit backwards differentiation formulas
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

# Exponential Time Differencing (ETD) methods
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

# ============================================================================
# Additional Timesteppers
# ============================================================================

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

        # Simplified ARK coefficients for 4-stage 3rd-order
        # Using a simplified L-stable DIRK
        γ = 0.4358665215  # Root of x^3 - 3x^2 + 3x/2 - 1/6 = 0

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

# Timestepper state management with workspace optimization
mutable struct TimestepperState
    timestepper::TimeStepper
    dt::Float64
    history::Vector{Vector{ScalarField}}
    dt_history::Vector{Float64}  # Track timestep history for variable timesteps
    stage::Int
    timestepper_data::Dict{String, Any}  # Additional data for specific timesteppers

    # Pre-allocated workspace fields for zero-allocation time-stepping
    workspace_fields::Vector{ScalarField}  # Reusable scratch fields
    workspace_allocated::Bool

    # Stochastic forcing support (following GeophysicalFlows.jl pattern)
    # Forcing is computed ONCE at the beginning of each timestep and stays constant
    # across all substeps (important for Stratonovich calculus correctness)
    forcing::Union{Nothing, Any}  # StochasticForcing or nothing
    current_substep::Int  # Track which substep we're in (1-indexed)
    forcing_generated::Bool  # Flag to track if forcing was generated this timestep

    function TimestepperState(timestepper::TimeStepper, dt::Float64, initial_state::Vector{ScalarField};
                              forcing=nothing)
        history = [copy.(initial_state)]
        dt_history = [dt]  # Initialize with current timestep
        timestepper_data = Dict{String, Any}()

        # Pre-allocate workspace fields based on timestepper requirements
        n_fields = length(initial_state)
        n_workspace = _workspace_count(timestepper) * n_fields
        workspace_fields = ScalarField[]

        # Pre-allocate workspace fields matching the initial state structure
        for _ in 1:n_workspace
            for field in initial_state
                ws_field = ScalarField(field.dist, "workspace", field.bases, field.dtype)
                push!(workspace_fields, ws_field)
            end
        end

        new(timestepper, dt, history, dt_history, 0, timestepper_data, workspace_fields, true,
            forcing, 1, false)
    end
end

"""
    set_forcing!(state::TimestepperState, forcing)

Set the stochastic forcing configuration for the timestepper.
The forcing will be generated once per timestep and held constant across substeps.
"""
function set_forcing!(state::TimestepperState, forcing)
    state.forcing = forcing
    state.forcing_generated = false
end

"""
    update_forcing!(state::TimestepperState, sim_time::Float64)

Generate new forcing realization at the beginning of a timestep.
This should be called ONCE at the start of step!, not at each substep.

Following GeophysicalFlows.jl pattern:
- Forcing is white in time but spatially correlated
- For Stratonovich calculus, forcing must be constant within each timestep
"""
function update_forcing!(state::TimestepperState, sim_time::Float64)
    if state.forcing !== nothing && !state.forcing_generated
        # Update dt in forcing if it changed
        if hasfield(typeof(state.forcing), :dt) && state.forcing.dt != state.dt
            state.forcing.dt = state.dt
        end
        # Generate new forcing realization
        generate_forcing!(state.forcing, sim_time)
        state.forcing_generated = true
    end
end

"""
    reset_forcing_flag!(state::TimestepperState)

Reset the forcing generation flag. Call this at the END of each timestep
to prepare for the next forcing generation.
"""
function reset_forcing_flag!(state::TimestepperState)
    state.forcing_generated = false
    state.current_substep = 1
end

"""
    get_cached_forcing(state::TimestepperState)

Get the cached forcing array. Returns nothing if no forcing is configured.
"""
function get_cached_forcing(state::TimestepperState)
    if state.forcing !== nothing
        return state.forcing.cached_forcing
    end
    return nothing
end

"""
    _update_registered_forcings!(solver::InitialValueSolver, sim_time::Float64, dt::Float64)

Generate new forcing realizations for all forcings registered via `add_stochastic_forcing!`.
Called ONCE at the beginning of each timestep to ensure Stratonovich calculus correctness.
"""
function _update_registered_forcings!(solver::InitialValueSolver, sim_time::Float64, dt::Float64)
    problem = solver.problem

    # Check if problem has stochastic_forcings field (only IVP does)
    if !hasfield(typeof(problem), :stochastic_forcings)
        return
    end

    # Generate forcing for each registered forcing
    for (var_idx, forcing) in problem.stochastic_forcings
        # Update dt if it changed
        if hasfield(typeof(forcing), :dt) && forcing.dt != dt
            forcing.dt = dt
        end

        # Generate new forcing realization
        # Substep=1 ensures forcing is actually regenerated (not just cached)
        generate_forcing!(forcing, sim_time, 1)
    end
end

"""
    _workspace_count(timestepper)

Return the number of workspace field sets needed for a timestepper.
"""
# IMEX RK methods (stages * 3 for X_stages, F_exp, F_imp)
function _workspace_count(::RK111)
    return 3  # 1 stage: X_stages, F_exp, F_imp
end

function _workspace_count(::RK222)
    return 6  # 2 stages: 2 * (X_stages, F_exp, F_imp)
end

function _workspace_count(::RK443)
    return 12  # 4 stages: 4 * (X_stages, F_exp, F_imp)
end

function _workspace_count(::Union{CNAB1, CNAB2})
    return 2
end

function _workspace_count(::Union{SBDF1, SBDF2, SBDF3, SBDF4})
    return 2
end

function _workspace_count(::Union{ETD_RK222, ETD_CNAB2, ETD_SBDF2})
    return 3
end

function _workspace_count(::TimeStepper)
    return 2  # Default fallback
end

"""
    get_workspace_field!(state::TimestepperState, template::ScalarField, idx::Int)

Get a pre-allocated workspace field, or allocate one if needed.
"""
function get_workspace_field!(state::TimestepperState, template::ScalarField, idx::Int)
    if idx <= length(state.workspace_fields)
        ws = state.workspace_fields[idx]
        # Reset to grid layout
        ws.current_layout = :g
        return ws
    else
        # Fallback: allocate new field (should rarely happen)
        return ScalarField(template.dist, "workspace", template.bases, template.dtype)
    end
end

"""
    copy_field_data!(dest::ScalarField, src::ScalarField)

Copy field data in-place without allocation.
"""
function copy_field_data!(dest::ScalarField, src::ScalarField)
    ensure_layout!(src, :g)
    ensure_layout!(dest, :g)
    copyto!(dest.data_g, src.data_g)
    dest.current_layout = src.current_layout
end

# Timestepping implementation
function step!(state::TimestepperState, solver::InitialValueSolver)
    """
    Advance solution by one timestep.

    IMPORTANT for stochastic forcing (following GeophysicalFlows.jl pattern):
    - Forcing is generated ONCE at the beginning of the timestep
    - Forcing stays CONSTANT across all substeps (RK stages)
    - This is essential for correct Stratonovich calculus interpretation
    """

    # Generate stochastic forcing ONCE at the beginning of the timestep
    # This ensures forcing is constant across all substeps (critical for Stratonovich)
    update_forcing!(state, solver.sim_time)

    # Also generate forcing for any forcings registered via add_stochastic_forcing!
    # These are stored in problem.stochastic_forcings Dict
    _update_registered_forcings!(solver, solver.sim_time, state.dt)

    state.current_substep = 1  # Reset substep counter

    # IMEX Runge-Kutta methods (Dedalus-compatible)
    if isa(state.timestepper, RK111)
        step_rk_imex!(state, solver)
    elseif isa(state.timestepper, RK222)
        step_rk_imex!(state, solver)
    elseif isa(state.timestepper, RK443)
        step_rk_imex!(state, solver)
    # Multistep IMEX methods
    elseif isa(state.timestepper, CNAB1)
        step_cnab1!(state, solver)
    elseif isa(state.timestepper, CNAB2)
        step_cnab2!(state, solver)
    elseif isa(state.timestepper, SBDF1)
        step_sbdf1!(state, solver)
    elseif isa(state.timestepper, SBDF2)
        step_sbdf2!(state, solver)
    elseif isa(state.timestepper, SBDF3)
        step_sbdf3!(state, solver)
    elseif isa(state.timestepper, SBDF4)
        step_sbdf4!(state, solver)
    elseif isa(state.timestepper, ETD_RK222)
        step_etd_rk222!(state, solver)
    elseif isa(state.timestepper, ETD_CNAB2)
        step_etd_cnab2!(state, solver)
    elseif isa(state.timestepper, ETD_SBDF2)
        step_etd_sbdf2!(state, solver)
    elseif isa(state.timestepper, MCNAB2)
        step_mcnab2!(state, solver)
    elseif isa(state.timestepper, CNLF2)
        step_cnlf2!(state, solver)
    elseif isa(state.timestepper, RKSMR)
        step_rksmr!(state, solver)
    elseif isa(state.timestepper, RKGFY)
        step_rkgfy!(state, solver)
    elseif isa(state.timestepper, RK443_IMEX)
        step_rk443_imex!(state, solver)
    else
        throw(ArgumentError("Unknown timestepper type: $(typeof(state.timestepper))"))
    end

    # Reset forcing flag at the end of timestep (prepare for next forcing generation)
    reset_forcing_flag!(state)
end

# =============================================================================
# IMEX Runge-Kutta Step Function (Dedalus-compatible ARK scheme)
# =============================================================================

"""
    step_rk_imex!(state::TimestepperState, solver::InitialValueSolver)

Generic IMEX Runge-Kutta timestep following Dedalus's Additive Runge-Kutta (ARK) scheme.

This implements the standard IMEX-RK algorithm:
- Explicit tableau treats nonlinear terms (RHS from equations)
- Implicit tableau treats linear terms (LHS operators, typically diffusion)

For each stage s:
    X_s = X_n + dt * sum_{j<s} A_exp[s,j] * F_exp[j]
              + dt * sum_{j<s} A_imp[s,j] * F_imp[j]
              + dt * A_imp[s,s] * F_imp[s]

Where the last term leads to an implicit solve:
    (I - dt * A_imp[s,s] * L) * X_s = RHS

For ESDIRK methods, A_exp[s,s] = 0 (explicitly), and A_imp is lower triangular
with constant diagonal γ, making it "singly diagonally implicit".

References:
- Kennedy & Carpenter (2003): "Additive Runge-Kutta schemes for CDR equations"
- Ascher, Ruuth, Spiteri (1997): IMEX RK methods for convection-diffusion
- Dedalus source: timestepping.py, RungeKuttaIMEX class
"""
function step_rk_imex!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    stages = ts.stages

    # Extract Butcher tableaux
    A_exp = ts.A_explicit
    A_imp = ts.A_implicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit
    c = ts.c_explicit  # Stage times (same for both tableaux)

    # Check if L_matrix is available for implicit treatment
    has_implicit = haskey(solver.problem.parameters, "L_matrix")
    L_matrix = has_implicit ? solver.problem.parameters["L_matrix"] : nothing
    M_matrix = haskey(solver.problem.parameters, "M_matrix") ?
               solver.problem.parameters["M_matrix"] : nothing

    # If no linear operator provided, fall back to explicit-only treatment
    if !has_implicit
        @debug "IMEX RK: No L_matrix found, treating all terms explicitly"
        _step_rk_imex_explicit_fallback!(state, solver)
        return
    end

    # Stage values and derivatives storage
    # X_stages[s] = solution at stage s
    # F_exp[s] = explicit RHS (nonlinear terms) at stage s
    # F_imp[s] = implicit RHS (L * X_s) at stage s
    X_stages = Vector{Vector{ScalarField}}(undef, stages)
    F_exp = Vector{Vector{ScalarField}}(undef, stages)
    F_imp = Vector{Vector{ScalarField}}(undef, stages)

    # Initialize stage 0 (current solution)
    X_stages[1] = current_state  # Stage 1 starts from current state for ESDIRK

    # Get diagonal coefficient (γ) for implicit solve
    # For ESDIRK methods, all diagonal entries are the same
    γ = A_imp[1, 1]  # First diagonal entry

    # Precompute LU factorization for implicit solve: (M - dt*γ*L)
    # This assumes a single linear operator for all fields
    n = size(L_matrix, 1)
    elem_type = eltype(L_matrix)

    # Check if L_matrix is effectively zero (no linear terms)
    L_is_zero = (L_matrix isa SparseMatrixCSC && nnz(L_matrix) == 0) ||
                (norm(L_matrix, Inf) < 1e-14)

    if L_is_zero
        # No linear terms - fall back to explicit treatment
        @debug "IMEX RK: L_matrix is zero, falling back to explicit treatment"
        _step_rk_imex_explicit_fallback!(state, solver)
        return
    end

    if M_matrix !== nothing
        LHS_matrix = M_matrix - dt * γ * L_matrix
    else
        # Create identity matrix matching L_matrix element type
        I_mat = Matrix{elem_type}(I, n, n)
        LHS_matrix = I_mat - dt * γ * L_matrix
    end

    # Convert to dense if sparse has too few entries (helps with LU stability)
    if LHS_matrix isa SparseMatrixCSC && nnz(LHS_matrix) < n
        LHS_matrix = Matrix(LHS_matrix)
    end

    lhs_factorization = lu(LHS_matrix)

    # Loop over stages
    for s in 1:stages
        state.current_substep = s

        # Compute explicit contribution from previous stages
        # sum_{j<s} A_exp[s,j] * F_exp[j]
        if s == 1
            # First stage: evaluate RHS at current state
            F_exp[1] = evaluate_rhs(solver, current_state, t + c[1] * dt)

            # For ESDIRK, first stage may be explicit (A_imp[1,1] = 0 or γ)
            if abs(γ) < 1e-14
                # Fully explicit first stage
                X_stages[1] = current_state
                F_imp[1] = _apply_linear_operator(L_matrix, current_state)
            else
                # Implicit first stage - need to solve
                rhs_vector = fields_to_vector(current_state)

                # Add explicit contribution: dt * A_exp[1,1] * F_exp[1] (usually 0 for ESDIRK)
                if abs(A_exp[1, 1]) > 1e-14
                    F_exp_vec = fields_to_vector(F_exp[1])
                    rhs_vector .+= dt * A_exp[1, 1] .* F_exp_vec
                end

                # Solve: (M - dt*γ*L) * X_1 = M * X_n + dt * explicit_terms
                if M_matrix !== nothing
                    rhs_vector = M_matrix * rhs_vector
                end
                X1_vector = lhs_factorization \ rhs_vector
                X_stages[1] = vector_to_fields(X1_vector, current_state)
                F_imp[1] = _apply_linear_operator(L_matrix, X_stages[1])
            end
        else
            # Stages s > 1: build RHS from previous stages
            rhs_vector = fields_to_vector(current_state)

            # Add explicit contributions: sum_{j=1}^{s-1} A_exp[s,j] * F_exp[j]
            for j in 1:(s-1)
                if abs(A_exp[s, j]) > 1e-14
                    F_exp_vec = fields_to_vector(F_exp[j])
                    rhs_vector .+= dt * A_exp[s, j] .* F_exp_vec
                end
            end

            # Add implicit contributions from previous stages: sum_{j=1}^{s-1} A_imp[s,j] * F_imp[j]
            for j in 1:(s-1)
                if abs(A_imp[s, j]) > 1e-14
                    F_imp_vec = fields_to_vector(F_imp[j])
                    rhs_vector .+= dt * A_imp[s, j] .* F_imp_vec
                end
            end

            # Apply mass matrix if present
            if M_matrix !== nothing
                X_n_vec = fields_to_vector(current_state)
                rhs_vector = M_matrix * X_n_vec + rhs_vector - M_matrix * X_n_vec
                # Actually: RHS = M * (X_n + dt * sums) for general M
                # For M=I this simplifies
            end

            # Solve implicit system: (M - dt*γ*L) * X_s = RHS
            Xs_vector = lhs_factorization \ rhs_vector
            X_stages[s] = vector_to_fields(Xs_vector, current_state)

            # Evaluate RHS at new stage
            F_exp[s] = evaluate_rhs(solver, X_stages[s], t + c[s] * dt)
            F_imp[s] = _apply_linear_operator(L_matrix, X_stages[s])
        end
    end

    # Final update using b weights
    # X_{n+1} = X_n + dt * sum_s (b_exp[s] * F_exp[s] + b_imp[s] * F_imp[s])
    final_vector = fields_to_vector(current_state)
    for s in 1:stages
        if abs(b_exp[s]) > 1e-14
            F_exp_vec = fields_to_vector(F_exp[s])
            final_vector .+= dt * b_exp[s] .* F_exp_vec
        end
        if abs(b_imp[s]) > 1e-14
            F_imp_vec = fields_to_vector(F_imp[s])
            final_vector .+= dt * b_imp[s] .* F_imp_vec
        end
    end

    new_state = vector_to_fields(final_vector, current_state)

    push!(state.history, new_state)

    # Keep only necessary history
    if length(state.history) > 1
        popfirst!(state.history)
    end
end

"""
    _apply_linear_operator(L_matrix, state)

Apply the linear operator L to a state (collection of fields).
Returns F_imp = L * X as a vector of fields.
"""
function _apply_linear_operator(L_matrix::AbstractMatrix, state::Vector{ScalarField})
    X_vector = fields_to_vector(state)
    LX_vector = L_matrix * X_vector
    return vector_to_fields(LX_vector, state)
end

"""
    _step_rk_imex_explicit_fallback!(state, solver)

Fallback to fully explicit RK when no L_matrix is provided.
Uses only the explicit tableau coefficients.
"""
function _step_rk_imex_explicit_fallback!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    stages = ts.stages

    A = ts.A_explicit
    b = ts.b_explicit
    c = ts.c_explicit

    # Stage derivatives storage
    k = Vector{Vector{ScalarField}}(undef, stages)

    # Compute stages
    for s in 1:stages
        state.current_substep = s

        if s == 1
            # First stage at current state
            k[1] = evaluate_rhs(solver, current_state, t + c[1] * dt)
        else
            # Build intermediate state
            temp_state = ScalarField[]
            for (i, field) in enumerate(current_state)
                temp_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
                ensure_layout!(field, :g)
                ensure_layout!(temp_field, :g)

                # X_temp = X_n + dt * sum_{j<s} A[s,j] * k[j]
                temp_field.data_g .= field.data_g
                for j in 1:(s-1)
                    if abs(A[s, j]) > 1e-14
                        ensure_layout!(k[j][i], :g)
                        temp_field.data_g .+= dt * A[s, j] .* k[j][i].data_g
                    end
                end
                push!(temp_state, temp_field)
            end
            k[s] = evaluate_rhs(solver, temp_state, t + c[s] * dt)
        end
    end

    # Final update: X_{n+1} = X_n + dt * sum_s b[s] * k[s]
    new_state = ScalarField[]
    for (i, field) in enumerate(current_state)
        new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
        ensure_layout!(field, :g)
        ensure_layout!(new_field, :g)

        new_field.data_g .= field.data_g
        for s in 1:stages
            if abs(b[s]) > 1e-14
                ensure_layout!(k[s][i], :g)
                new_field.data_g .+= dt * b[s] .* k[s][i].data_g
            end
        end
        push!(new_state, new_field)
    end

    push!(state.history, new_state)

    if length(state.history) > 1
        popfirst!(state.history)
    end
end

# Implicit-explicit methods
function step_cnab1!(state::TimestepperState, solver::InitialValueSolver)
    """
    Crank-Nicolson Adams-Bashforth 1st order following Tarang MultistepIMEX implementation.
    
    Based on Tarang timesteppers:95-188 MultistepIMEX.step method:
    - Proper coefficient computation: a[0] = 1/dt, a[1] = -1/dt, b[0] = 1/2, b[1] = 1/2, c[1] = 1
    - RHS construction: c[1]*F[0] - a[1]*MX[0] - b[1]*LX[0] (following lines 156-166)
    - LHS solution: (a[0]*M + b[0]*L).X = RHS (following lines 174-184)
    - Proper state rotation and history management
    """
    
    current_state = state.history[end]
    dt = state.dt
    
    # Initialize history arrays if needed (following Tarang MultistepIMEX.__init__)
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end
    
    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "CNAB1 requires L_matrix and M_matrix, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end
    
    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]
    
    # Get CNAB1 coefficients following Tarang (timesteppers:206-220)
    a = [1.0/dt, -1.0/dt]  # a[0], a[1]
    b = [0.5, 0.5]         # b[0], b[1] 
    c = [0.0, 1.0]         # c[0], c[1]
    
    try
        # Step 1: Convert current state to vector (following Tarang gather_inputs)
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0] (following Tarang lines 142-147)
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step (following Tarang lines 149-153)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history (following Tarang lines 124-126)
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]
        
        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)
        
        # Keep only needed history for CNAB1 (amax=1, bmax=1, cmax=1)
        while length(MX_history) > 1; pop!(MX_history); end
        while length(LX_history) > 1; pop!(LX_history); end
        while length(F_history) > 1; pop!(F_history); end
        
        # Step 5: Build RHS following Tarang exactly (timesteppers:156-166)
        # RHS = c[1] * F[0] - a[1] * MX[0] - b[1] * LX[0]
        rhs = c[2] * F_history[1]  # c[1] * F[0] (using 1-based indexing)
        if length(MX_history) >= 1  # a[1] term
            rhs .-= a[2] * MX_history[1]  # -a[1] * MX[0]
        end
        if length(LX_history) >= 1  # b[1] term
            rhs .-= b[2] * LX_history[1]  # -b[1] * LX[0]
        end
        
        # Step 6: Build and solve LHS system (following Tarang lines 174-184)
        # (a[0] * M + b[0] * L).X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix  # a[0] * M + b[0] * L
        X_new = LHS_matrix \ rhs
        
        # Step 7: Update state (following Tarang scatter_inputs)
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1
        
        @debug "CNAB1 step completed: dt=$dt, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"
        
    catch e
        @warn "CNAB1 failed: $e, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end
    
    # Keep reasonable history length
    if length(state.history) > 3
        popfirst!(state.history)
    end
end

function step_cnab2!(state::TimestepperState, solver::InitialValueSolver)
    """
    Crank-Nicolson Adams-Bashforth 2nd order following Tarang MultistepIMEX implementation.
    
    Based on Tarang timesteppers:95-188 MultistepIMEX.step method:
    - Variable timestep coefficients: w1 = k1/k0, c[1] = 1 + w1/2, c[2] = -w1/2 (lines 276-290)
    - Full RHS construction: c[1]*F[0] + c[2]*F[1] - a[1]*MX[0] - b[1]*LX[0] (lines 156-166)
    - Proper history management with rotation for MX, LX, F arrays (lines 124-126)
    - Falls back to CNAB1 for iteration < 1 (line 274)
    """
    
    current_state = state.history[end]
    dt = state.dt
    
    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end
    
    iteration = state.timestepper_data["iteration"]
    
    # Check if we have enough history for CNAB2 (following Tarang line 274)
    if iteration < 1 || length(state.history) < 2
        @debug "CNAB2 requires iteration >= 1, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end
    
    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "CNAB2 requires L_matrix and M_matrix, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end
    
    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]
    
    # Get timestep history for variable timestep (following Tarang lines 280-281)
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous
    
    # Get CNAB2 coefficients following Tarang exactly (timesteppers:283-288)
    a = [1.0/dt_current, -1.0/dt_current]  # a[0], a[1]
    b = [0.5, 0.5]                         # b[0], b[1]
    c = [0.0, 1.0 + w1/2.0, -w1/2.0]      # c[0], c[1], c[2]
    
    @debug "CNAB2 variable timestep: dt_current=$dt_current, dt_previous=$dt_previous, w1=$w1"
    
    try
        # Step 1: Convert current state to vector
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0] (following Tarang lines 142-147)
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step (following Tarang lines 149-153)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history (following Tarang lines 124-126)
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]
        
        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)
        
        # Keep only needed history for CNAB2 (amax=2, bmax=2, cmax=2)
        while length(MX_history) > 2; pop!(MX_history); end
        while length(LX_history) > 2; pop!(LX_history); end
        while length(F_history) > 2; pop!(F_history); end
        
        # Step 5: Build RHS following Tarang exactly (timesteppers:156-166)
        # RHS = c[1] * F[0] + c[2] * F[1] - a[1] * MX[0] - b[1] * LX[0]
        rhs = c[2] * F_history[1]  # c[1] * F[0]
        if length(F_history) >= 2  # c[2] term
            rhs .+= c[3] * F_history[2]  # c[2] * F[1]
        end
        if length(MX_history) >= 1  # a[1] term
            rhs .-= a[2] * MX_history[1]  # -a[1] * MX[0]
        end
        if length(LX_history) >= 1  # b[1] term
            rhs .-= b[2] * LX_history[1]  # -b[1] * LX[0]
        end
        
        # Step 6: Build and solve LHS system (following Tarang lines 174-184)
        # (a[0] * M + b[0] * L).X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix  # a[0] * M + b[0] * L
        X_new = LHS_matrix \ rhs
        
        # Step 7: Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1
        
        @debug "CNAB2 step completed: dt=$dt_current, w1=$w1, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"
        
    catch e
        @warn "CNAB2 failed: $e, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end
    
    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

# BDF methods
function step_sbdf1!(state::TimestepperState, solver::InitialValueSolver)
    """
    Semi-implicit BDF1 (backward Euler) following Tarang MultistepIMEX implementation.
    
    Based on Tarang timesteppers:224-252 SBDF1 coefficients:
    - a[0] = 1/k0, a[1] = -1/k0 (BDF1 time derivative)
    - b[0] = 1 (fully implicit, not Crank-Nicolson 1/2)
    - c[1] = 1 (forward Euler explicit)
    
    Implicit: 1st-order BDF (backward Euler)
    Explicit: 1st-order extrapolation (forward Euler)
    """
    
    current_state = state.history[end]
    dt = state.dt
    
    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end
    
    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "SBDF1 requires L_matrix and M_matrix, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end
    
    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]
    
    # Get SBDF1 coefficients following Tarang exactly (timesteppers:247-250)
    a = [1.0/dt, -1.0/dt]  # a[0], a[1] - BDF1 time derivative
    b = [1.0]              # b[0] - fully implicit (not 1/2 like CNAB)
    c = [0.0, 1.0]         # c[0], c[1] - forward Euler explicit
    
    try
        # Step 1: Convert current state to vector
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0] (following Tarang MultistepIMEX pattern)
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]
        
        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)
        
        # Keep only needed history for SBDF1 (amax=1, bmax=1, cmax=1)
        while length(MX_history) > 1; pop!(MX_history); end
        while length(LX_history) > 1; pop!(LX_history); end
        while length(F_history) > 1; pop!(F_history); end
        
        # Step 5: Build RHS following Tarang MultistepIMEX pattern
        # RHS = c[1] * F[0] - a[1] * MX[0] - 0 * LX[0] (since bmax=1, no b[1] term)
        rhs = c[2] * F_history[1]  # c[1] * F[0]
        if length(MX_history) >= 1  # a[1] term
            rhs .-= a[2] * MX_history[1]  # -a[1] * MX[0]
        end
        # No b[1] term for SBDF1 since bmax=1
        
        # Step 6: Build and solve LHS system
        # (a[0] * M + b[0] * L).X = RHS  ->  (1/dt * M + 1 * L).X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix  # a[0] * M + b[0] * L
        X_new = LHS_matrix \ rhs
        
        # Step 7: Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1
        
        @debug "SBDF1 step completed: dt=$dt, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"
        
    catch e
        @warn "SBDF1 failed: $e, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end
    
    # Keep reasonable history length
    if length(state.history) > 3
        popfirst!(state.history)
    end
end

function step_sbdf2!(state::TimestepperState, solver::InitialValueSolver)
    """
    Semi-implicit BDF2 following Tarang MultistepIMEX implementation.
    
    Based on Tarang timesteppers:333-367 SBDF2 coefficients:
    - Variable timestep with w1 = k1/k0
    - a[0] = (1 + 2*w1) / (1 + w1) / k1
    - a[1] = -(1 + w1) / k1  
    - a[2] = w1^2 / (1 + w1) / k1
    - b[0] = 1, c[1] = 1 + w1, c[2] = -w1
    - Falls back to SBDF1 for iteration < 1
    
    Implicit: 2nd-order BDF
    Explicit: 2nd-order extrapolation
    """
    
    current_state = state.history[end]
    dt = state.dt
    
    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end
    
    iteration = state.timestepper_data["iteration"]
    
    # Check if we have enough history for SBDF2 (following Tarang line 350)
    if iteration < 1 || length(state.history) < 2
        @debug "SBDF2 requires iteration >= 1, falling back to SBDF1"
        step_sbdf1!(state, solver)
        return
    end
    
    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "SBDF2 requires L_matrix and M_matrix, falling back to SBDF1"
        step_sbdf1!(state, solver)
        return
    end
    
    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]
    
    # Get timestep history for variable timestep (following Tarang lines 357-358)
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous
    
    # Get SBDF2 coefficients following Tarang exactly (timesteppers:360-365)
    a = [(1.0 + 2.0*w1) / (1.0 + w1) / dt_current,  # a[0]
         -(1.0 + w1) / dt_current,                    # a[1]
         w1^2 / (1.0 + w1) / dt_current]              # a[2]
    b = [1.0]                                         # b[0] - fully implicit
    c = [0.0, 1.0 + w1, -w1]                        # c[0], c[1], c[2]
    
    @debug "SBDF2 variable timestep: dt_current=$dt_current, dt_previous=$dt_previous, w1=$w1"
    
    try
        # Step 1: Convert current state to vector
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0]
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]
        
        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)
        
        # Keep only needed history for SBDF2 (amax=2, bmax=2, cmax=2)
        while length(MX_history) > 2; pop!(MX_history); end
        while length(LX_history) > 2; pop!(LX_history); end
        while length(F_history) > 2; pop!(F_history); end
        
        # Step 5: Build RHS following Tarang MultistepIMEX pattern
        # RHS = c[1]*F[0] + c[2]*F[1] - a[1]*MX[0] - a[2]*MX[1] - 0*LX terms (bmax=1)
        rhs = c[2] * F_history[1]  # c[1] * F[0]
        if length(F_history) >= 2  # c[2] term
            rhs .+= c[3] * F_history[2]  # c[2] * F[1]
        end
        if length(MX_history) >= 1  # a[1] term
            rhs .-= a[2] * MX_history[1]  # -a[1] * MX[0]
        end
        if length(MX_history) >= 2  # a[2] term
            rhs .-= a[3] * MX_history[2]  # -a[2] * MX[1]
        end
        # No b[1], b[2] terms since bmax=1 for SBDF2
        
        # Step 6: Build and solve LHS system
        # (a[0] * M + b[0] * L).X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix  # a[0] * M + b[0] * L
        X_new = LHS_matrix \ rhs
        
        # Step 7: Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1
        
        @debug "SBDF2 step completed: dt=$dt_current, w1=$w1, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"
        
    catch e
        @warn "SBDF2 failed: $e, falling back to SBDF1"
        step_sbdf1!(state, solver)
        return
    end
    
    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

function step_sbdf3!(state::TimestepperState, solver::InitialValueSolver)
    """
    Semi-implicit BDF3 following Tarang implementation.
    
    Tarang coefficients (timesteppers:425-447):
    For iteration >= 2: uses complex 3rd-order BDF coefficients
    For iteration < 2: falls back to SBDF2
    
    Implicit: 3rd-order BDF
    Explicit: 3rd-order extrapolation
    """
    
    # Check if we have enough history for SBDF3
    if length(state.history) < 3
        @debug "SBDF3 requires 3 history states, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
    current_state = state.history[end]
    dt = state.dt
    
    # Get timestep history for variable timestep ratios (Tarang pattern)
    if length(state.dt_history) < 3
        @warn "SBDF3 requires 3 timestep history, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
    k2 = state.dt_history[end]     # current timestep
    k1 = state.dt_history[end-1]   # previous timestep  
    k0 = state.dt_history[end-2]   # timestep before that
    
    # Compute timestep ratios following Tarang (timesteppers:435-436)
    w2 = k2 / k1
    w1 = k1 / k0
    
    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "SBDF3 requires L_matrix and M_matrix, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]
    
    try
        # Get SBDF3 coefficients following Tarang exactly (timesteppers:438-445)
        a = zeros(4)
        b = zeros(4)
        c = zeros(4)
        
        a[1] = (1 + w2/(1 + w2) + w1*w2/(1 + w1*(1 + w2))) / k2
        a[2] = (-1 - w2 - w1*w2*(1 + w2)/(1 + w1)) / k2
        a[3] = w2^2 * (w1 + 1/(1 + w2)) / k2
        a[4] = -w1^3 * w2^2 * (1 + w2) / (1 + w1) / (1 + w1 + w1*w2) / k2
        b[1] = 1
        c[2] = (1 + w2)*(1 + w1*(1 + w2)) / (1 + w1)
        c[3] = -w2*(1 + w1*(1 + w2))
        c[4] = w1*w1*w2*(1 + w2) / (1 + w1)
        
        # Evaluate RHS terms at current and previous times
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_prev1 = evaluate_rhs(solver, state.history[end-1], solver.sim_time - k2)
        F_prev2 = evaluate_rhs(solver, state.history[end-2], solver.sim_time - k2 - k1)
        
        # Convert states to vectors
        X_current = fields_to_vector(current_state)
        X_prev1 = fields_to_vector(state.history[end-1])
        X_prev2 = fields_to_vector(state.history[end-2])
        F_current_vec = fields_to_vector(F_current)
        F_prev1_vec = fields_to_vector(F_prev1)
        F_prev2_vec = fields_to_vector(F_prev2)
        
        # Build RHS following Tarang multistep pattern
        # RHS = sum(cj * F(n-j)) - sum(aj * M.X(n-j)) (j >= 2 for a)
        rhs = (c[2] * F_current_vec + c[3] * F_prev1_vec + c[4] * F_prev2_vec - 
               a[2] * (M_matrix * X_current) - a[3] * (M_matrix * X_prev1) - a[4] * (M_matrix * X_prev2))
        
        # Build and solve LHS system: (a0 M + b0 L).X(n+1) = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix
        
        # Solve linear system
        X_new = LHS_matrix \ rhs
        
        # Convert back to fields and update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        
        @debug "SBDF3 step completed: dt=$k2, w2=$w2, w1=$w1, |X_new|=$(norm(X_new))"
        
    catch e
        @warn "SBDF3 failed: $e, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
    # Keep only necessary history for SBDF3
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

function step_sbdf4!(state::TimestepperState, solver::InitialValueSolver)
    """
    Semi-implicit BDF4 following Tarang implementation.
    
    Tarang coefficients (timesteppers:466-495):
    For iteration >= 3: uses complex 4th-order BDF coefficients  
    For iteration < 3: falls back to SBDF3
    
    Implicit: 4th-order BDF
    Explicit: 4th-order extrapolation
    """
    
    # Check if we have enough history for SBDF4
    if length(state.history) < 4
        @debug "SBDF4 requires 4 history states, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end
    
    current_state = state.history[end]
    dt = state.dt
    
    # Get timestep history for variable timestep ratios
    if length(state.dt_history) < 4
        @warn "SBDF4 requires 4 timestep history, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end
    
    k3 = state.dt_history[end]     # current timestep
    k2 = state.dt_history[end-1]   # previous timestep
    k1 = state.dt_history[end-2]   # timestep before that
    k0 = state.dt_history[end-3]   # timestep 3 back
    
    # Compute timestep ratios following Tarang (timesteppers:476-478)
    w3 = k3 / k2
    w2 = k2 / k1
    w1 = k1 / k0
    
    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "SBDF4 requires L_matrix and M_matrix, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end
    
    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]
    
    try
        # Get SBDF4 coefficients following Tarang exactly (timesteppers:480-494)
        A1 = 1 + w1*(1 + w2)
        A2 = 1 + w2*(1 + w3) 
        A3 = 1 + w1*A2
        
        a = zeros(5)
        b = zeros(5)
        c = zeros(5)
        
        a[1] = (1 + w3/(1 + w3) + w2*w3/A2 + w1*w2*w3/A3) / k3
        a[2] = (-1 - w3*(1 + w2*(1 + w3)/(1 + w2)*(1 + w1*A2/A1))) / k3
        a[3] = w3 * (w3/(1 + w3) + w2*w3*(A3 + w1)/(1 + w1)) / k3
        a[4] = -w2^3 * w3^2 * (1 + w3) / (1 + w2) * A3 / A2 / k3
        a[5] = (1 + w3) / (1 + w1) * A2 / A1 * w1^4 * w2^3 * w3^2 / A3 / k3
        b[1] = 1
        c[2] = w2 * (1 + w3) / (1 + w2) * ((1 + w3)*(A3 + w1) + (1 + w1)/w2) / A1
        c[3] = -A2 * A3 * w3 / (1 + w1)
        c[4] = w2^2 * w3 * (1 + w3) / (1 + w2) * A3
        c[5] = -w1^3 * w2^2 * w3 * (1 + w3) / (1 + w1) * A2 / A1
        
        # Evaluate RHS terms at current and previous times
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_prev1 = evaluate_rhs(solver, state.history[end-1], solver.sim_time - k3)
        F_prev2 = evaluate_rhs(solver, state.history[end-2], solver.sim_time - k3 - k2)
        F_prev3 = evaluate_rhs(solver, state.history[end-3], solver.sim_time - k3 - k2 - k1)
        
        # Convert states to vectors
        X_current = fields_to_vector(current_state)
        X_prev1 = fields_to_vector(state.history[end-1])
        X_prev2 = fields_to_vector(state.history[end-2])
        X_prev3 = fields_to_vector(state.history[end-3])
        F_current_vec = fields_to_vector(F_current)
        F_prev1_vec = fields_to_vector(F_prev1)
        F_prev2_vec = fields_to_vector(F_prev2)
        F_prev3_vec = fields_to_vector(F_prev3)
        
        # Build RHS following Tarang multistep pattern
        rhs = (c[2] * F_current_vec + c[3] * F_prev1_vec + c[4] * F_prev2_vec + c[5] * F_prev3_vec - 
               a[2] * (M_matrix * X_current) - a[3] * (M_matrix * X_prev1) - 
               a[4] * (M_matrix * X_prev2) - a[5] * (M_matrix * X_prev3))
        
        # Build and solve LHS system
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix
        
        # Solve linear system
        X_new = LHS_matrix \ rhs
        
        # Convert back to fields and update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)
        
        push!(state.history, new_state)
        
        @debug "SBDF4 step completed: dt=$k3, w3=$w3, w2=$w2, w1=$w1, |X_new|=$(norm(X_new))"
        
    catch e
        @warn "SBDF4 failed: $e, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end
    
    # Keep only necessary history for SBDF4
    if length(state.history) > 5
        popfirst!(state.history)
    end
end

# Exponential Time Differencing methods
function step_etd_rk222!(state::TimestepperState, solver::InitialValueSolver)
    """
    2nd-order exponential Runge-Kutta method (ETDRK2).

    Standard formulation from Cox-Matthews (2002), Eq. 22:
    Stage 1 (predictor): a_n = exp(hL)u_n
                         c = a_n + h*φ₁(hL)*N(u_n)
    Stage 2 (corrector): u_{n+1} = a_n + h*φ₁(hL)*N(c)

    where:
    - φ₁(z) = (exp(z) - 1)/z
    - N(u) is the nonlinear term
    - L is the linear operator

    This is the standard ETD2RK method from the literature. The predictor c uses
    the nonlinear term at u_n, and the corrector uses only N(c), providing
    second-order accuracy via proper exponential integration.

    References:
    - Cox & Matthews (2002), "Exponential Time Differencing for Stiff Systems",
      J. Comput. Phys. 176, 430-455, Equation 22
    - Hochbruck & Ostermann (2010), "Exponential integrators", Acta Numerica 19, 209-286
    - Kassam & Trefethen (2005), "Fourth-Order Time Stepping for Stiff PDEs",
      SIAM J. Sci. Comput. 26(4), 1214-1233
    """

    current_state = state.history[end]
    dt = state.dt

    # Get linear operator from solver
    if !haskey(solver.problem.parameters, "L_matrix")
        @warn "ETD_RK222 requires L_matrix for linear operator, falling back to RK222"
        step_rk222!(state, solver)
        return
    end

    L_matrix = solver.problem.parameters["L_matrix"]

    try
        # Compute matrix exponentials and φ functions
        exp_hL, φ₁_hL, _ = phi_functions_matrix(L_matrix, dt)

        # Convert state to vector form
        X₀ = fields_to_vector(current_state)

        # Compute exponential propagator: a_n = exp(hL)*u_n
        a_n = exp_hL * X₀

        # Stage 1 (predictor): Evaluate nonlinear term N(u_n) at current state
        F₀ = evaluate_rhs(solver, current_state, solver.sim_time)
        N_u_n = fields_to_vector(F₀)

        # Predictor: c = a_n + h*φ₁(hL)*N(u_n)
        c = a_n + dt * (φ₁_hL * N_u_n)

        # Convert back to field form for nonlinear evaluation
        temp_state = copy.(current_state)
        copy_solution_to_fields!(temp_state, c)

        # Stage 2 (corrector): Evaluate N(c) at predicted state
        F_c = evaluate_rhs(solver, temp_state, solver.sim_time + dt)
        N_c = fields_to_vector(F_c)

        # Final update (standard ETDRK2 formula):
        # u_{n+1} = a_n + h*φ₁(hL)*N(c)
        # This is the Cox-Matthews Eq. 22 formulation
        X_new = a_n + dt * (φ₁_hL * N_c)

        # Update state
        X_new_cpu = X_new
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new_cpu)

        push!(state.history, new_state)

        @debug "ETDRK2 step completed: dt=$dt, |X_new|=$(norm(X_new_cpu))"

    catch e
        @warn "ETD-RK222 failed: $e, falling back to RK222"
        step_rk222!(state, solver)
        return
    end

    # Keep only necessary history
    if length(state.history) > 1
        popfirst!(state.history)
    end
end

function step_etd_cnab2!(state::TimestepperState, solver::InitialValueSolver)
    """
    2nd-order exponential Adams-Bashforth method (ETDAB2/ETD-CNAB2).

    Formulation:
    u_{n+1} = exp(hL)u_n + h*φ₁(hL)*N_AB2

    where N_AB2 is the 2nd-order Adams-Bashforth extrapolation:
    N_AB2 = (1 + w/2)*N(u_n) - (w/2)*N(u_{n-1})
    w = h_n / h_{n-1} (timestep ratio for variable timesteps)

    Linear treatment: Exact via exponential propagator exp(hL)
    Nonlinear treatment: Explicit 2nd-order Adams-Bashforth extrapolation

    Note: This method is called ETD-CNAB2 following Tarang naming convention,
    but it uses exponential treatment (not Crank-Nicolson) for the linear operator.
    The "CNAB" refers to the multistep structure, not implicit treatment.

    References:
    - Hochbruck & Ostermann (2010), "Exponential integrators"
    - Cox & Matthews (2002), "Exponential Time Differencing for Stiff Systems"
    """

    current_state = state.history[end]
    dt = state.dt

    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "F_history")
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end

    iteration = state.timestepper_data["iteration"]

    # Check if we have enough history for 2-step Adams-Bashforth
    if iteration < 1 || length(state.history) < 2
        @debug "ETD-CNAB2 requires iteration >= 1, falling back to ETDRK2 for startup"
        step_etd_rk222!(state, solver)
        return
    end

    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "ETD-CNAB2 requires L_matrix and M_matrix, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous

    try
        # Compute exponential integrators
        exp_hL, φ₁_hL, _ = phi_functions_matrix(L_matrix, dt_current)

        # Convert current state to vector
        X_current = fields_to_vector(current_state)

        # Evaluate nonlinear term N(u_n)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)

        # Rotate and store history
        F_history = state.timestepper_data["F_history"]
        pushfirst!(F_history, F_current_vec)

        # Keep only needed history for Adams-Bashforth 2
        while length(F_history) > 2; pop!(F_history); end

        # Adams-Bashforth 2nd-order extrapolation coefficients (variable timestep)
        # N_AB2 = c₁*N(u_n) + c₂*N(u_{n-1})
        c₁ = 1.0 + w1/2.0  # Current step weight
        c₂ = -w1/2.0       # Previous step weight

        # Build Adams-Bashforth extrapolated nonlinear term
        F_extrap = c₁ * F_history[1]
        if length(F_history) >= 2
            F_extrap .+= c₂ * F_history[2]
        end

        # Exponential time differencing step with Adams-Bashforth extrapolation:
        # u_{n+1} = exp(hL)u_n + h*φ₁(hL)*N_AB2
        X_new = exp_hL * X_current + dt_current * (φ₁_hL * F_extrap)

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1

        @debug "ETDAB2 step completed: dt=$dt_current, w1=$w1, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"

    catch e
        @warn "ETD-CNAB2 failed: $e, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

function step_etd_sbdf2!(state::TimestepperState, solver::InitialValueSolver)
    """
    2nd-order Exponential Time Differencing Multistep Method (ETD-MS2).

    For the ODE: u'(t) = Lu + N(u), this implements a proper 2-step exponential
    multistep method derived from the variation-of-constants formula.

    The variation-of-constants formula gives:
        u(tₙ₊₁) = exp(hL)u(tₙ) + ∫₀ʰ exp((h-τ)L) N(u(tₙ+τ)) dτ

    For a 2-step method, we interpolate N using values at tₙ and tₙ₋₁:
        N(tₙ + τ) ≈ N(uₙ) + (τ/h)[N(uₙ) - N(uₙ₋₁)]  (linear interpolation)

    Substituting and integrating exactly gives the ETD multistep formula:
        u_{n+1} = exp(hL)uₙ + h[b₁(hL)Nₙ + b₀(hL)Nₙ₋₁]

    where the coefficient functions are:
        b₁(z) = φ₁(z) + φ₂(z) = (exp(z) - 1)/z + (exp(z) - 1 - z)/z²
        b₀(z) = -φ₂(z) = -(exp(z) - 1 - z)/z²

    This is the ETD analog of Adams-Bashforth 2, providing 2nd-order accuracy
    with exact linear propagation.

    For variable timesteps (w = hₙ/hₙ₋₁):
        b₁(z) = φ₁(z) + (1/w)φ₂(z)
        b₀(z) = -(1/w)φ₂(z)

    References:
    - Hochbruck & Ostermann (2010), "Exponential integrators", Acta Numerica 19, 209-286
    - Cox & Matthews (2002), "Exponential Time Differencing for Stiff Systems"
    - Beylkin, Keiser, & Vozovoi (1998), "A new class of time discretization schemes"
    """

    current_state = state.history[end]
    dt = state.dt

    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "F_history")
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end

    iteration = state.timestepper_data["iteration"]

    # Check if we have enough history for 2-step method
    if iteration < 1 || length(state.history) < 2
        @debug "ETD-SBDF2 requires iteration >= 1, falling back to ETDRK2 for startup"
        step_etd_rk222!(state, solver)
        state.timestepper_data["iteration"] = get(state.timestepper_data, "iteration", 0) + 1
        return
    end

    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix")
        @warn "ETD-SBDF2 requires L_matrix, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end

    L_matrix = solver.problem.parameters["L_matrix"]

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w = dt_current / dt_previous

    try
        # Compute exponential integrators: exp(hL), φ₁(hL), φ₂(hL)
        exp_hL, φ₁_hL, φ₂_hL = phi_functions_matrix(L_matrix, dt_current)

        # Convert current state to vector
        X_current = fields_to_vector(current_state)

        # Evaluate nonlinear term N(uₙ) at current state
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)

        # Rotate and store history
        F_history = state.timestepper_data["F_history"]
        pushfirst!(F_history, F_current_vec)

        # Keep only needed history for 2-step method
        while length(F_history) > 2
            pop!(F_history)
        end

        # Get previous nonlinear term N(uₙ₋₁)
        Nₙ = F_history[1]      # N(uₙ)
        Nₙ₋₁ = F_history[2]    # N(uₙ₋₁)

        # Compute ETD multistep coefficients (variable timestep version)
        # b₁(z) = φ₁(z) + (1/w)φ₂(z)  -- coefficient for Nₙ
        # b₀(z) = -(1/w)φ₂(z)         -- coefficient for Nₙ₋₁
        inv_w = 1.0 / w

        # Linear propagation: exp(hL)uₙ
        X_propagated = exp_hL * X_current

        # Nonlinear contribution using ETD coefficients:
        # h[b₁(hL)Nₙ + b₀(hL)Nₙ₋₁] = h[(φ₁ + φ₂/w)Nₙ - (φ₂/w)Nₙ₋₁]
        #                          = h[φ₁Nₙ + (φ₂/w)(Nₙ - Nₙ₋₁)]

        # Compute the nonlinear contributions
        φ₁_Nₙ = φ₁_hL * Nₙ
        φ₂_diff = φ₂_hL * (Nₙ - Nₙ₋₁)

        # Full update: u_{n+1} = exp(hL)uₙ + h[φ₁Nₙ + (φ₂/w)(Nₙ - Nₙ₋₁)]
        X_new = X_propagated + dt_current * (φ₁_Nₙ + inv_w * φ₂_diff)

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1

        @debug "ETD-MS2 step completed: dt=$dt_current, w=$w, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"

    catch e
        @warn "ETD-SBDF2 failed: $e, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end

    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

# ============================================================================
# Additional Timestepper Step Functions
# ============================================================================

function step_mcnab2!(state::TimestepperState, solver::InitialValueSolver)
    """
    Modified Crank-Nicolson Adams-Bashforth 2nd order step.

    Uses modified θ parameter for the implicit Crank-Nicolson treatment.

    The modification uses θ slightly different from 0.5 to improve stability
    for certain stiff problems while maintaining 2nd order accuracy.

    Formula:
    (M + θ*dt*L) X^{n+1} = (M - (1-θ)*dt*L) X^n + dt*(c₁*F^n + c₂*F^{n-1})

    where c₁ = 1.5, c₂ = -0.5 are Adams-Bashforth 2 coefficients.
    """

    current_state = state.history[end]
    dt = state.dt
    θ = state.timestepper.implicit_coefficient

    # Initialize history arrays if needed
    if !haskey(state.timestepper_data, "MX_history")
        state.timestepper_data["MX_history"] = []
        state.timestepper_data["LX_history"] = []
        state.timestepper_data["F_history"] = []
        state.timestepper_data["iteration"] = 0
    end

    iteration = state.timestepper_data["iteration"]

    # Check if we have enough history for MCNAB2
    if iteration < 1 || length(state.history) < 2
        @debug "MCNAB2 requires iteration >= 1, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end

    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "MCNAB2 requires L_matrix and M_matrix, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous

    # MCNAB2 coefficients with modified θ
    # a coefficients for time derivative (same as CNAB2)
    a = [1.0/dt_current, -1.0/dt_current]
    # b coefficients for implicit treatment with modified θ
    b = [θ, 1.0 - θ]
    # c coefficients for Adams-Bashforth 2 extrapolation (variable timestep)
    c = [0.0, 1.0 + w1/2.0, -w1/2.0]

    try
        # Convert current state to vector
        X_current = fields_to_vector(current_state)

        # Compute M.X[0] and L.X[0]
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current

        # Evaluate F(X[0]) at current time step
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)

        # Rotate and store history
        MX_history = state.timestepper_data["MX_history"]
        LX_history = state.timestepper_data["LX_history"]
        F_history = state.timestepper_data["F_history"]

        pushfirst!(MX_history, MX_current)
        pushfirst!(LX_history, LX_current)
        pushfirst!(F_history, F_current_vec)

        # Keep only needed history
        while length(MX_history) > 2; pop!(MX_history); end
        while length(LX_history) > 2; pop!(LX_history); end
        while length(F_history) > 2; pop!(F_history); end

        # Build RHS: c[1]*F[0] + c[2]*F[1] - a[1]*MX[0] - b[1]*LX[0]
        rhs = c[2] * F_history[1]
        if length(F_history) >= 2
            rhs .+= c[3] * F_history[2]
        end
        rhs .-= a[2] * MX_history[1]
        rhs .-= b[2] * LX_history[1]

        # Build and solve LHS: (a[0]*M + b[0]*L) X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix
        X_new = LHS_matrix \ rhs

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1

        @debug "MCNAB2 step completed: dt=$dt_current, θ=$θ, iteration=$(state.timestepper_data["iteration"])"

    catch e
        @warn "MCNAB2 failed: $e, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    # Keep reasonable history length
    if length(state.history) > 4
        popfirst!(state.history)
    end
end

function step_cnlf2!(state::TimestepperState, solver::InitialValueSolver)
    """
    Crank-Nicolson Leapfrog 2nd order step.

    Uses leapfrog (centered) treatment for explicit terms with Crank-Nicolson
    for implicit terms.

    This is a 3-level method that uses X^{n-1}, X^n, and computes X^{n+1}.

    Formula:
    (M + θ*dt*L) X^{n+1} = (M - (1-θ)*dt*L) X^{n-1} + 2*dt*F^n

    where θ = 0.5 for standard Crank-Nicolson.

    Note: Leapfrog can have computational mode issues; Robert-Asselin filter
    may be needed for long integrations.
    """

    current_state = state.history[end]
    dt = state.dt
    θ = state.timestepper.implicit_coefficient

    # Initialize history if needed
    if !haskey(state.timestepper_data, "iteration")
        state.timestepper_data["iteration"] = 0
    end

    iteration = state.timestepper_data["iteration"]

    # CNLF requires X^{n-1}, so need at least 2 history states
    if iteration < 1 || length(state.history) < 2
        @debug "CNLF2 requires 2 history states, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end

    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "CNLF2 requires L_matrix and M_matrix, falling back to RK222"
        step_rk222!(state, solver)
        return
    end

    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]

    try
        # Get X^n and X^{n-1}
        X_current = fields_to_vector(current_state)
        X_previous = fields_to_vector(state.history[end-1])

        # Evaluate F^n at current state
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)

        # Build RHS: (M - (1-θ)*dt*L) X^{n-1} + 2*dt*F^n
        rhs = (M_matrix - (1.0 - θ) * dt * L_matrix) * X_previous + 2.0 * dt * F_current_vec

        # Build and solve LHS: (M + θ*dt*L) X^{n+1} = RHS
        LHS_matrix = M_matrix + θ * dt * L_matrix
        X_new = LHS_matrix \ rhs

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1

        @debug "CNLF2 step completed: dt=$dt, θ=$θ, iteration=$(state.timestepper_data["iteration"])"

    catch e
        @warn "CNLF2 failed: $e, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    # Keep reasonable history length (CNLF needs 2 previous states)
    if length(state.history) > 3
        popfirst!(state.history)
    end
end

function step_rksmr!(state::TimestepperState, solver::InitialValueSolver)
    """
    Strong Stability Preserving Runge-Kutta 3rd order step (SSP-RK3).

    This is the Shu-Osher form of SSP-RK3, optimal for hyperbolic PDEs.

    Shu-Osher form:
    Stage 1: u^(1) = u^n + dt*F(u^n)
    Stage 2: u^(2) = 3/4*u^n + 1/4*u^(1) + 1/4*dt*F(u^(1))
    Stage 3: u^{n+1} = 1/3*u^n + 2/3*u^(2) + 2/3*dt*F(u^(2))

    Properties:
    - 3rd order accurate
    - SSP with CFL coefficient C = 1
    - TVD (Total Variation Diminishing) for scalar conservation laws
    """

    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    alpha = state.timestepper.alpha
    beta = state.timestepper.beta

    try
        # Stage 1: u^(1) = u^n + dt*F(u^n)
        F0 = evaluate_rhs(solver, current_state, t)
        u1 = add_scaled_state(current_state, F0, dt * beta[1])

        # Stage 2: u^(2) = 3/4*u^n + 1/4*u^(1) + 1/4*dt*F(u^(1))
        F1 = evaluate_rhs(solver, u1, t + dt)
        # u^(2) = alpha[2,1]*u^n + alpha[2,2]*u^(1) + beta[2]*dt*F(u^(1))
        u2 = ScalarField[]
        for (i, field0) in enumerate(current_state)
            field1 = u1[i]
            new_field = ScalarField(field0.dist, field0.name, field0.bases, field0.dtype)
            ensure_layout!(field0, :g)
            ensure_layout!(field1, :g)
            ensure_layout!(F1[i], :g)
            ensure_layout!(new_field, :g)

            new_field.data_g .= alpha[2,1] .* field0.data_g .+
                                alpha[2,2] .* field1.data_g .+
                                dt * beta[2] .* F1[i].data_g
            push!(u2, new_field)
        end

        # Stage 3: u^{n+1} = 1/3*u^n + 2/3*u^(2) + 2/3*dt*F(u^(2))
        F2 = evaluate_rhs(solver, u2, t + 0.5*dt)  # Approximate midpoint
        new_state = ScalarField[]
        for (i, field0) in enumerate(current_state)
            field2 = u2[i]
            new_field = ScalarField(field0.dist, field0.name, field0.bases, field0.dtype)
            ensure_layout!(field0, :g)
            ensure_layout!(field2, :g)
            ensure_layout!(F2[i], :g)
            ensure_layout!(new_field, :g)

            new_field.data_g .= alpha[3,1] .* field0.data_g .+
                                alpha[3,3] .* field2.data_g .+
                                dt * beta[3] .* F2[i].data_g
            push!(new_state, new_field)
        end

        push!(state.history, new_state)

        @debug "RKSMR (SSP-RK3) step completed: dt=$dt"

    catch e
        @warn "RKSMR failed: $e, falling back to RK443"
        step_rk443!(state, solver)
        return
    end

    # Keep only necessary history
    if length(state.history) > 2
        popfirst!(state.history)
    end
end

function step_rkgfy!(state::TimestepperState, solver::InitialValueSolver)
    """
    General Framework IMEX Runge-Kutta step (ARK2).

    This implements the Ascher-Ruuth-Spiteri ARK2 method.

    For the system: du/dt = L*u + N(u)
    where L is the stiff linear part and N is the nonlinear part.

    Each stage solves:
    (I - γ*dt*L) * k_i = L*Y_i + N(Y_i)

    where γ is the DIRK diagonal coefficient and Y_i are the stage values.
    """

    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    A_exp = state.timestepper.A_explicit
    b_exp = state.timestepper.b_explicit
    c_exp = state.timestepper.c_explicit
    A_imp = state.timestepper.A_implicit
    b_imp = state.timestepper.b_implicit
    c_imp = state.timestepper.c_implicit
    stages = state.timestepper.stages

    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "RKGFY requires L_matrix and M_matrix, falling back to RK443"
        step_rk443!(state, solver)
        return
    end

    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]

    try
        # Convert initial state to vector
        X0 = fields_to_vector(current_state)
        n = length(X0)
        I_mat = Matrix{eltype(L_matrix)}(LinearAlgebra.I, n, n)

        # Store stage values and RHS evaluations
        K_exp = zeros(eltype(X0), n, stages)  # Explicit RHS at each stage
        K_imp = zeros(eltype(X0), n, stages)  # Implicit RHS at each stage
        Y = zeros(eltype(X0), n, stages)      # Stage values

        # Stage 1 (initial stage)
        Y[:, 1] .= X0
        F0 = evaluate_rhs(solver, current_state, t + c_exp[1]*dt)
        K_exp[:, 1] .= fields_to_vector(F0)
        K_imp[:, 1] .= L_matrix * X0

        # Stages 2 to s
        for i in 2:stages
            # Build stage value Y_i from previous stages
            Y_i = copy(X0)

            # Explicit contributions: sum_{j=1}^{i-1} a^E_{ij} * k^E_j
            for j in 1:(i-1)
                if A_exp[i,j] != 0.0
                    Y_i .+= dt * A_exp[i,j] .* K_exp[:, j]
                end
            end

            # Implicit contributions from previous stages: sum_{j=1}^{i-1} a^I_{ij} * k^I_j
            for j in 1:(i-1)
                if A_imp[i,j] != 0.0
                    Y_i .+= dt * A_imp[i,j] .* K_imp[:, j]
                end
            end

            # Solve implicit stage: (I - γ*dt*L) * Y_i_new = Y_i
            γ = A_imp[i,i]
            if γ != 0.0
                LHS = I_mat - γ * dt * L_matrix
                Y_i = LHS \ Y_i
            end

            Y[:, i] .= Y_i

            # Evaluate RHS at stage value
            temp_state = copy.(current_state)
            copy_solution_to_fields!(temp_state, Y_i)
            F_i = evaluate_rhs(solver, temp_state, t + c_exp[i]*dt)
            K_exp[:, i] .= fields_to_vector(F_i)
            K_imp[:, i] .= L_matrix * Y_i
        end

        # Final update: X_{n+1} = X_n + dt * sum_i (b^E_i * k^E_i + b^I_i * k^I_i)
        X_new = copy(X0)
        for i in 1:stages
            if b_exp[i] != 0.0
                X_new .+= dt * b_exp[i] .* K_exp[:, i]
            end
            if b_imp[i] != 0.0
                X_new .+= dt * b_imp[i] .* K_imp[:, i]
            end
        end

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)

        @debug "RKGFY (ARK2) step completed: dt=$dt"

    catch e
        @warn "RKGFY failed: $e, falling back to RK443"
        step_rk443!(state, solver)
        return
    end

    # Keep only necessary history
    if length(state.history) > 2
        popfirst!(state.history)
    end
end

function step_rk443_imex!(state::TimestepperState, solver::InitialValueSolver)
    """
    4-stage 3rd-order IMEX Runge-Kutta step.

    Similar to RKGFY but with 4 stages for 3rd order accuracy.

    Uses the same ARK framework with:
    - Explicit tableau for nonlinear terms
    - SDIRK (singly diagonal implicit) tableau for linear terms
    """

    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time

    A_exp = state.timestepper.A_explicit
    b_exp = state.timestepper.b_explicit
    c_exp = state.timestepper.c_explicit
    A_imp = state.timestepper.A_implicit
    b_imp = state.timestepper.b_implicit
    c_imp = state.timestepper.c_implicit
    stages = state.timestepper.stages

    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "RK443_IMEX requires L_matrix and M_matrix, falling back to RK443"
        step_rk443!(state, solver)
        return
    end

    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]

    try
        # Convert initial state to vector
        X0 = fields_to_vector(current_state)
        n = length(X0)
        I_mat = Matrix{eltype(L_matrix)}(LinearAlgebra.I, n, n)

        # Store stage values and RHS evaluations
        K_exp = zeros(eltype(X0), n, stages)
        K_imp = zeros(eltype(X0), n, stages)
        Y = zeros(eltype(X0), n, stages)

        # For SDIRK, compute LU factorization of (I - γ*dt*L) once
        γ = A_imp[1,1]  # Same γ on diagonal for SDIRK
        if γ != 0.0
            LHS = I_mat - γ * dt * L_matrix
            LHS_factored = lu(LHS)
        else
            LHS_factored = nothing
        end

        # Stage 1
        if γ != 0.0 && LHS_factored !== nothing
            Y[:, 1] .= LHS_factored \ X0
        else
            Y[:, 1] .= X0
        end

        temp_state = copy.(current_state)
        copy_solution_to_fields!(temp_state, Y[:, 1])
        F0 = evaluate_rhs(solver, temp_state, t + c_exp[1]*dt)
        K_exp[:, 1] .= fields_to_vector(F0)
        K_imp[:, 1] .= L_matrix * Y[:, 1]

        # Stages 2 to s
        for i in 2:stages
            # Build RHS from previous stages
            rhs = copy(X0)

            for j in 1:(i-1)
                if A_exp[i,j] != 0.0
                    rhs .+= dt * A_exp[i,j] .* K_exp[:, j]
                end
                if A_imp[i,j] != 0.0
                    rhs .+= dt * A_imp[i,j] .* K_imp[:, j]
                end
            end

            # Solve implicit stage
            if A_imp[i,i] != 0.0 && LHS_factored !== nothing
                Y[:, i] .= LHS_factored \ rhs
            else
                Y[:, i] .= rhs
            end

            # Evaluate RHS
            temp_state = copy.(current_state)
            copy_solution_to_fields!(temp_state, Y[:, i])
            F_i = evaluate_rhs(solver, temp_state, t + c_exp[i]*dt)
            K_exp[:, i] .= fields_to_vector(F_i)
            K_imp[:, i] .= L_matrix * Y[:, i]
        end

        # Final update
        X_new = copy(X0)
        for i in 1:stages
            if b_exp[i] != 0.0
                X_new .+= dt * b_exp[i] .* K_exp[:, i]
            end
            if b_imp[i] != 0.0
                X_new .+= dt * b_imp[i] .* K_imp[:, i]
            end
        end

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)

        @debug "RK443_IMEX step completed: dt=$dt"

    catch e
        @warn "RK443_IMEX failed: $e, falling back to RK443"
        step_rk443!(state, solver)
        return
    end

    # Keep only necessary history
    if length(state.history) > 2
        popfirst!(state.history)
    end
end

# Helper functions
function evaluate_rhs(solver::InitialValueSolver, state::Vector{ScalarField}, time::Float64)
    """
    Evaluate right-hand side of differential equations following Tarang pattern.
    
    Based on Tarang MultistepIMEX.step method (timesteppers:149-153):
    - evaluator.evaluate_scheduled(iteration=iteration, wall_time=wall_time, sim_time=sim_time, timestep=dt)
    - evaluator.require_coeff_space(F_fields)  
    - sp.gather_outputs(F_fields, out=F0.get_subdata(sp))
    
    This evaluates the F expressions from problem.equation_data for each equation.
    """
    
    problem = solver.problem
    rhs = ScalarField[]
    
    try
        # Set current state fields to the provided state for evaluation
        # This mimics how Tarang sets the current field values before evaluation
        for (i, field) in enumerate(state)
            if i <= length(problem.variables)
                # Update the problem variable with current state data
                # Handle different field data structures
                if hasfield(typeof(problem.variables[i]), :data)
                    problem.variables[i].data = field.data
                elseif hasfield(typeof(problem.variables[i]), :data_g) && hasfield(typeof(field), :data_g)
                    problem.variables[i].data_g = field.data_g
                elseif hasfield(typeof(problem.variables[i]), :data_c) && hasfield(typeof(field), :data_c)
                    problem.variables[i].data_c = field.data_c
                end
                
                # Ensure correct layout for evaluation
                ensure_layout!(field, :g)  # Start in grid space for nonlinear evaluation
            end
        end
        
        # Update time parameter if it exists (like Tarang sim_time updates)
        if hasfield(typeof(problem), :time) && problem.time !== nothing
            # Update time field value for time-dependent expressions
            if hasfield(typeof(problem.time), :data)
                problem.time.data = time
            elseif hasfield(typeof(problem.time), :value)
                problem.time.value = time
            end
        end
        
        # Evaluate each equation's RHS (F expression) following Tarang pattern
        if hasfield(typeof(problem), :equation_data) && !isempty(problem.equation_data)
            for (eq_idx, eq_data) in enumerate(problem.equation_data)
                if haskey(eq_data, "F_expr") && eq_data["F_expr"] !== nothing
                    # Evaluate the F expression symbolically
                    try
                        rhs_field = evaluate_expression(eq_data["F_expr"], problem.variables)
                        
                        # Ensure correct field properties
                        if isa(rhs_field, ScalarField)
                            # Convert to coefficient space as done in Tarang
                            ensure_layout!(rhs_field, :c)
                            push!(rhs, rhs_field)
                        else
                            @warn "F expression $eq_idx did not evaluate to ScalarField, creating zero field"
                            rhs_field = create_zero_field(state[eq_idx])
                            push!(rhs, rhs_field)
                        end
                        
                    catch e
                        @warn "Failed to evaluate F expression for equation $eq_idx: $e"
                        # Create zero field as fallback
                        rhs_field = create_zero_field(state[eq_idx])
                        push!(rhs, rhs_field)
                    end
                else
                    # No F_expr means zero RHS (e.g., ∂u/∂t = 0 or purely linear equation)
                    @debug "No F_expr found for equation $eq_idx, using zero field"
                    rhs_field = create_zero_field(state[eq_idx])
                    push!(rhs, rhs_field)
                end
            end
        else
            @warn "No equation_data found in problem, creating zero fields"
            # Fallback: create zero fields matching the state
            for field in state
                rhs_field = create_zero_field(field)
                push!(rhs, rhs_field)
            end
        end
        
        @debug "Evaluated RHS for $(length(rhs)) equations at time $time"

        # Add stochastic forcing if registered (automatic handling)
        # Forcing was already generated once at the start of the timestep by step!()
        # The same cached forcing is reused for all RK substeps (Stratonovich correct)
        if hasfield(typeof(problem), :stochastic_forcings) && !isempty(problem.stochastic_forcings)
            for (var_idx, forcing) in problem.stochastic_forcings
                if var_idx <= length(rhs)
                    rhs_field = rhs[var_idx]
                    # Get cached forcing (already generated at timestep start)
                    F = forcing.cached_forcing

                    # Add forcing to RHS in coefficient space
                    ensure_layout!(rhs_field, :c)
                    if size(rhs_field.data_c) == size(F)
                        rhs_field.data_c .+= F
                        @debug "Added stochastic forcing to equation $var_idx"
                    else
                        @warn "Forcing size $(size(F)) doesn't match RHS size $(size(rhs_field.data_c)) for equation $var_idx"
                    end
                end
            end
        end

    catch e
        @error "RHS evaluation failed: $e"
        # Fallback: create zero fields
        for field in state
            rhs_field = create_zero_field(field)
            push!(rhs, rhs_field)
        end
    end

    return rhs
end

function create_zero_field(template_field::ScalarField)
    """Create a zero field matching the template field properties"""
    rhs_field = ScalarField(template_field.dist, "rhs_$(template_field.name)", template_field.bases, template_field.dtype)
    ensure_layout!(rhs_field, :c)  # Coefficient space following Tarang
    fill!(rhs_field.data_c, 0.0)
    return rhs_field
end

# Expression evaluation is handled by the complete implementation in solvers.jl
# which supports the operator tree structure used in equation parsing

function add_scaled_state(state1::Vector{ScalarField}, state2::Vector{ScalarField}, scale::Float64)
    """Compute state1 + scale * state2 - OPTIMIZED version"""
    result = ScalarField[]

    for (i, field1) in enumerate(state1)
        field2 = state2[i]
        new_field = ScalarField(field1.dist, field1.name, field1.bases, field1.dtype)

        ensure_layout!(field1, :g)
        ensure_layout!(field2, :g)
        ensure_layout!(new_field, :g)

        # Use optimized in-place operations
        n = length(field1.data_g)
        if n > 2000
            # BLAS-based: y = α*x + β*y via axpby! or copy + axpy
            copyto!(new_field.data_g, field1.data_g)
            BLAS.axpy!(scale, field2.data_g, new_field.data_g)
        elseif n > 100
            # LoopVectorization for medium arrays
            scale_local = scale
            @turbo for j in eachindex(new_field.data_g, field1.data_g, field2.data_g)
                new_field.data_g[j] = field1.data_g[j] + scale_local * field2.data_g[j]
            end
        else
            # Broadcasting for small arrays
            new_field.data_g .= field1.data_g .+ scale .* field2.data_g
        end
        push!(result, new_field)
    end

    return result
end

"""
    add_scaled_state!(dest::Vector{ScalarField}, state1::Vector{ScalarField},
                      state2::Vector{ScalarField}, scale::Float64)

In-place version: dest = state1 + scale * state2
"""
function add_scaled_state!(dest::Vector{ScalarField}, state1::Vector{ScalarField},
                           state2::Vector{ScalarField}, scale::Float64)
    for (i, field1) in enumerate(state1)
        field2 = state2[i]
        dest_field = dest[i]

        ensure_layout!(field1, :g)
        ensure_layout!(field2, :g)
        ensure_layout!(dest_field, :g)

        n = length(field1.data_g)
        if n > 2000
            copyto!(dest_field.data_g, field1.data_g)
            BLAS.axpy!(scale, field2.data_g, dest_field.data_g)
        elseif n > 100
            scale_local = scale
            @turbo for j in eachindex(dest_field.data_g, field1.data_g, field2.data_g)
                dest_field.data_g[j] = field1.data_g[j] + scale_local * field2.data_g[j]
            end
        else
            dest_field.data_g .= field1.data_g .+ scale .* field2.data_g
        end
    end
end

"""
    axpy_state!(scale::Float64, x::Vector{ScalarField}, y::Vector{ScalarField})

In-place AXPY: y = y + scale * x
"""
function axpy_state!(scale::Float64, x::Vector{ScalarField}, y::Vector{ScalarField})
    for i in eachindex(x, y)
        ensure_layout!(x[i], :g)
        ensure_layout!(y[i], :g)

        n = length(x[i].data_g)
        if n > 2000
            BLAS.axpy!(scale, x[i].data_g, y[i].data_g)
        elseif n > 100
            scale_local = scale
            @turbo for j in eachindex(y[i].data_g, x[i].data_g)
                y[i].data_g[j] += scale_local * x[i].data_g[j]
            end
        else
            y[i].data_g .+= scale .* x[i].data_g
        end
    end
end

"""
    linear_combination_state!(dest::Vector{ScalarField}, α::Float64, a::Vector{ScalarField},
                              β::Float64, b::Vector{ScalarField})

In-place linear combination: dest = α*a + β*b
"""
function linear_combination_state!(dest::Vector{ScalarField}, α::Float64, a::Vector{ScalarField},
                                   β::Float64, b::Vector{ScalarField})
    for i in eachindex(dest, a, b)
        ensure_layout!(a[i], :g)
        ensure_layout!(b[i], :g)
        ensure_layout!(dest[i], :g)

        n = length(a[i].data_g)
        if n > 100
            α_local, β_local = α, β
            @turbo for j in eachindex(dest[i].data_g, a[i].data_g, b[i].data_g)
                dest[i].data_g[j] = α_local * a[i].data_g[j] + β_local * b[i].data_g[j]
            end
        else
            dest[i].data_g .= α .* a[i].data_g .+ β .* b[i].data_g
        end
    end
end

function copy_state(state::Vector{ScalarField})
    """Create a deep copy of state"""
    new_state = ScalarField[]
    
    for field in state
        new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
        ensure_layout!(field, :g)
        ensure_layout!(new_field, :g)
        new_field.data_g .= field.data_g
        push!(new_state, new_field)
    end
    
    return new_state
end

function get_previous_timestep(state::TimestepperState)
    """Get previous timestep for variable timestep handling"""
    if length(state.dt_history) >= 2
        return state.dt_history[end-1]
    elseif length(state.dt_history) >= 1
        # If only one timestep in history, use current timestep as fallback
        return state.dt_history[end]
    else
        # Fallback to current timestep
        return state.dt
    end
end

function update_timestep_history!(state::TimestepperState, dt::Float64)
    """Update timestep history following Tarang deque rotation pattern"""
    # Update current timestep
    state.dt = dt
    
    # Add to history (following Tarang rotation)
    push!(state.dt_history, dt)
    
    # Keep only necessary timestep history (limit to what multistep methods need)
    max_history = get_max_timestep_history(state.timestepper)
    if length(state.dt_history) > max_history
        popfirst!(state.dt_history)
    end
end

function get_max_timestep_history(timestepper::TimeStepper)
    """Get maximum timestep history needed for timestepper"""
    if isa(timestepper, Union{CNAB1, SBDF1})
        return 2  # Current + 1 previous
    elseif isa(timestepper, Union{CNAB2, SBDF2, ETD_CNAB2, ETD_SBDF2, MCNAB2, CNLF2})
        return 3  # Current + 2 previous
    elseif isa(timestepper, Union{SBDF3})
        return 4  # Current + 3 previous
    elseif isa(timestepper, Union{SBDF4})
        return 5  # Current + 4 previous
    elseif isa(timestepper, Union{ETD_RK222, RKSMR, RKGFY, RK443_IMEX})
        return 2  # Current + 1 previous for RK/exponential methods
    else
        return 2  # Default for explicit methods
    end
end

# Helper functions for exponential integrators are defined in solvers.jl:
# - fields_to_vector: Convert fields to coefficient-space vector
# - copy_solution_to_fields!: Copy solution vector back to fields

