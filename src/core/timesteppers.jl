"""
Timestepping schemes for initial value problems
"""

using LinearAlgebra
using LinearAlgebra: BLAS
using SparseArrays: SparseMatrixCSC, nnz
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
        n_workspace_sets = _workspace_count(timestepper)
        workspace_fields = ScalarField[]

        # Pre-allocate workspace fields matching the initial state structure
        for _ in 1:n_workspace_sets
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

# Mass-matrix helpers for exponential integrators
function _get_mass_factor!(state::TimestepperState, M_matrix::AbstractMatrix)
    cache = state.timestepper_data
    if !haskey(cache, "M_factor") || get(cache, "M_factor_source", nothing) !== M_matrix
        cache["M_factor"] = factorize(M_matrix)
        cache["M_factor_source"] = M_matrix
        cache["L_eff"] = nothing
        cache["L_eff_source"] = nothing
    end
    return cache["M_factor"]
end

function _get_linear_operator_eff!(state::TimestepperState, L_matrix::AbstractMatrix,
                                   M_matrix::Union{Nothing, AbstractMatrix})
    if M_matrix === nothing
        return -L_matrix, nothing
    end

    M_factor = _get_mass_factor!(state, M_matrix)
    cache = state.timestepper_data
    if !haskey(cache, "L_eff") || get(cache, "L_eff_source", nothing) !== L_matrix
        cache["L_eff"] = M_factor \ L_matrix
        cache["L_eff_source"] = L_matrix
    end

    return -cache["L_eff"], M_factor
end

function _apply_mass_inverse(M_factor, vec::AbstractVector)
    return M_factor === nothing ? vec : (M_factor \ vec)
end

"""
    _get_problem_matrix(problem, key)

Fetch a matrix from `problem.parameters` ensuring it resides on CPU memory.
If the stored matrix is a GPU array, it is copied back to CPU once and the
problem parameter is updated in place so subsequent calls reuse the CPU copy.
"""
function _get_problem_matrix(problem::Problem, key::AbstractString)
    params = problem.parameters
    key_str = key isa String ? key : String(key)
    if !haskey(params, key_str)
        return nothing
    end
    matrix = params[key_str]
    return _ensure_cpu_matrix!(params, key_str, matrix)
end

function _ensure_cpu_matrix!(params::Dict{String, Any}, key::String, matrix)
    if matrix isa AbstractArray && is_gpu_array(matrix)
        cpu_matrix = Array(matrix)
        params[key] = cpu_matrix
        return cpu_matrix
    end
    return matrix
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
    _update_temporal_filters!(solver::InitialValueSolver, dt::Float64)

Update all temporal filters registered via `add_temporal_filter!`.
Called at the END of each timestep after the solution has been advanced.
"""
function _update_temporal_filters!(solver::InitialValueSolver, dt::Float64)
    problem = solver.problem

    # Check if problem has temporal_filters field (only IVP does)
    if !hasfield(typeof(problem), :temporal_filters)
        return
    end

    # Skip if no filters registered
    if isempty(problem.temporal_filters)
        return
    end

    # Get variable name to field mapping
    var_map = Dict{String, Any}()
    for var in problem.variables
        if hasproperty(var, :name)
            var_map[getfield(var, :name)] = var
        end
    end

    # Update each registered filter
    for (filter_name, filter_info) in problem.temporal_filters
        filter = filter_info.filter
        source_sym = filter_info.source
        source_name = String(source_sym)

        if haskey(var_map, source_name)
            source_var = var_map[source_name]
            # Get the physical space data from the variable
            if source_var isa ScalarField
                ensure_layout!(source_var, :g)
                data = get_grid_data(source_var)
            elseif hasproperty(source_var, :data_g)
                data = getproperty(source_var, :data_g)
            elseif hasproperty(source_var, :data)
                data = getfield(source_var, :data)
            else
                @warn "Cannot find data for variable $source_name"
                continue
            end

            # Update the filter with current data
            # Use try-catch in case filter types differ
            try
                update!(filter, data, dt)
            catch e
                @warn "Failed to update temporal filter :$filter_name: $e"
            end
        end
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

GPU-aware: Uses copyto!() which works on both CPU and GPU arrays.
Both source and destination should be on the same architecture.
"""
function copy_field_data!(dest::ScalarField, src::ScalarField)
    ensure_layout!(src, :g)
    ensure_layout!(dest, :g)
    # copyto! works on both CPU and GPU arrays
    copyto!(get_grid_data(dest), get_grid_data(src))
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

    # Update temporal filters with the new solution
    _update_temporal_filters!(solver, state.dt)

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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")

    # If no linear operator provided, fall back to explicit-only treatment
    if L_matrix === nothing
        @debug "IMEX RK: No L_matrix found, treating all terms explicitly"
        _step_rk_imex_explicit_fallback!(state, solver)
        return
    end

    # Check if L_matrix is effectively zero (no linear terms)
    L_is_zero = (L_matrix isa SparseMatrixCSC && nnz(L_matrix) == 0) ||
                (norm(L_matrix, Inf) < 1e-14)

    if L_is_zero
        @debug "IMEX RK: L_matrix is zero, falling back to explicit treatment"
        _step_rk_imex_explicit_fallback!(state, solver)
        return
    end

    X_n_vec = fields_to_vector(current_state)
    MX_n_vec = M_matrix === nothing ? X_n_vec : (M_matrix * X_n_vec)

    F_exp_vecs = Vector{Vector{eltype(X_n_vec)}}(undef, stages)
    F_imp_vecs = Vector{Vector{eltype(X_n_vec)}}(undef, stages)
    lhs_cache = Dict{Float64, Any}()

    # Loop over stages
    for s in 1:stages
        state.current_substep = s

        rhs_vec = copy(MX_n_vec)

        # Add explicit contributions from previous stages
        for j in 1:(s-1)
            if abs(A_exp[s, j]) > 1e-14
                rhs_vec .+= dt * A_exp[s, j] .* F_exp_vecs[j]
            end
        end

        # Subtract implicit contributions from previous stages (L on LHS)
        for j in 1:(s-1)
            if abs(A_imp[s, j]) > 1e-14
                rhs_vec .-= dt * A_imp[s, j] .* F_imp_vecs[j]
            end
        end

        a_ii = A_imp[s, s]
        if M_matrix === nothing
            if abs(a_ii) < 1e-14
                Xs_vec = rhs_vec
            else
                lhs = get!(lhs_cache, a_ii) do
                    I + dt * a_ii * L_matrix
                end
                Xs_vec = lhs \ rhs_vec
            end
        else
            if abs(a_ii) < 1e-14
                Xs_vec = M_matrix \ rhs_vec
            else
                lhs = get!(lhs_cache, a_ii) do
                    M_matrix + dt * a_ii * L_matrix
                end
                Xs_vec = lhs \ rhs_vec
            end
        end

        Xs_fields = vector_to_fields(Xs_vec, current_state)
        F_exp_fields = evaluate_rhs(solver, Xs_fields, t + c[s] * dt)
        F_exp_vecs[s] = fields_to_vector(F_exp_fields)
        F_imp_vecs[s] = L_matrix * Xs_vec
    end

    # Final update using b weights
    rhs_vec = copy(MX_n_vec)
    for s in 1:stages
        if abs(b_exp[s]) > 1e-14
            rhs_vec .+= dt * b_exp[s] .* F_exp_vecs[s]
        end
        if abs(b_imp[s]) > 1e-14
            rhs_vec .-= dt * b_imp[s] .* F_imp_vecs[s]
        end
    end

    X_new_vec = M_matrix === nothing ? rhs_vec : (M_matrix \ rhs_vec)
    new_state = vector_to_fields(X_new_vec, current_state)

    push!(state.history, new_state)

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
    _step_explicit_rk!(state, solver, A, b, c)

Generic explicit Runge-Kutta step for X' = M^-1 F(X).
"""
function _step_explicit_rk!(state::TimestepperState, solver::InitialValueSolver,
                            A::AbstractMatrix, b::AbstractVector, c::AbstractVector)
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    stages = length(b)

    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")

    X_n_vec = fields_to_vector(current_state)
    k_vecs = Vector{Vector{eltype(X_n_vec)}}(undef, stages)

    for s in 1:stages
        state.current_substep = s

        Y_vec = copy(X_n_vec)
        for j in 1:(s-1)
            if abs(A[s, j]) > 1e-14
                Y_vec .+= dt * A[s, j] .* k_vecs[j]
            end
        end

        stage_state = vector_to_fields(Y_vec, current_state)
        F_stage = evaluate_rhs(solver, stage_state, t + c[s] * dt)
        F_vec = fields_to_vector(F_stage)
        k_vecs[s] = M_matrix === nothing ? F_vec : (M_matrix \ F_vec)
    end

    X_new_vec = copy(X_n_vec)
    for s in 1:stages
        if abs(b[s]) > 1e-14
            X_new_vec .+= dt * b[s] .* k_vecs[s]
        end
    end

    new_state = vector_to_fields(X_new_vec, current_state)
    push!(state.history, new_state)

    if length(state.history) > 1
        popfirst!(state.history)
    end
end

"""
    _step_rk_imex_explicit_fallback!(state, solver)

Fallback to fully explicit RK when no L_matrix is provided.
Uses only the explicit tableau coefficients.
"""
function _step_rk_imex_explicit_fallback!(state::TimestepperState, solver::InitialValueSolver)
    ts = state.timestepper
    _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
end

# Explicit RK fallbacks used by other timesteppers
function step_rk111!(state::TimestepperState, solver::InitialValueSolver)
    ts = RK111()
    _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
end

function step_rk222!(state::TimestepperState, solver::InitialValueSolver)
    ts = RK222()
    _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
end

function step_rk443!(state::TimestepperState, solver::InitialValueSolver)
    ts = RK443()
    _step_explicit_rk!(state, solver, ts.A_explicit, ts.b_explicit, ts.c_explicit)
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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "CNAB1 requires L_matrix and M_matrix, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end

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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "CNAB2 requires L_matrix and M_matrix, falling back to CNAB1"
        step_cnab1!(state, solver)
        return
    end
    
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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "SBDF1 requires L_matrix and M_matrix, falling back to forward Euler"
        step_rk111!(state, solver)
        return
    end
    
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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "SBDF2 requires L_matrix and M_matrix, falling back to SBDF1"
        step_sbdf1!(state, solver)
        return
    end
    
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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "SBDF3 requires L_matrix and M_matrix, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "SBDF4 requires L_matrix and M_matrix, falling back to SBDF3"
        step_sbdf3!(state, solver)
        return
    end
    
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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    if L_matrix === nothing
        @warn "ETD_RK222 requires L_matrix for linear operator, falling back to RK222"
        step_rk222!(state, solver)
        return
    end

    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    L_linear, M_factor = _get_linear_operator_eff!(state, L_matrix, M_matrix)

    try
        # Compute matrix exponentials and φ functions
        exp_hL, φ₁_hL, _ = phi_functions_matrix(L_linear, dt)

        # Convert state to vector form
        X₀ = fields_to_vector(current_state)

        # Compute exponential propagator: a_n = exp(hL)*u_n
        a_n = exp_hL * X₀

        # Stage 1 (predictor): Evaluate nonlinear term N(u_n) at current state
        F₀ = evaluate_rhs(solver, current_state, solver.sim_time)
        N_u_n = _apply_mass_inverse(M_factor, fields_to_vector(F₀))

        # Predictor: c = a_n + h*φ₁(hL)*N(u_n)
        c = a_n + dt * (φ₁_hL * N_u_n)

        # Convert back to field form for nonlinear evaluation
        temp_state = copy.(current_state)
        copy_solution_to_fields!(temp_state, c)

        # Stage 2 (corrector): Evaluate N(c) at predicted state
        F_c = evaluate_rhs(solver, temp_state, solver.sim_time + dt)
        N_c = _apply_mass_inverse(M_factor, fields_to_vector(F_c))

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

    # Keep only necessary history (retain one previous state for multistep startups)
    max_history = get_max_timestep_history(state.timestepper)
    if length(state.history) > max_history
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
        state.timestepper_data["iteration"] = get(state.timestepper_data, "iteration", 0) + 1
        return
    end

    # Get linear operator from solver
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    if L_matrix === nothing
        @warn "ETD-CNAB2 requires L_matrix, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    L_linear, M_factor = _get_linear_operator_eff!(state, L_matrix, M_matrix)

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous

    try
        # Compute exponential integrators
        exp_hL, φ₁_hL, _ = phi_functions_matrix(L_linear, dt_current)

        # Convert current state to vector
        X_current = fields_to_vector(current_state)

        # Evaluate nonlinear term N(u_n)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = _apply_mass_inverse(M_factor, fields_to_vector(F_current))

        # Rotate and store history
        F_history = state.timestepper_data["F_history"]
        pushfirst!(F_history, F_current_vec)

        # Keep only needed history for Adams-Bashforth 2
        while length(F_history) > 2; pop!(F_history); end
        if length(F_history) < 2 && length(state.history) >= 2
            prev_state = state.history[end-1]
            F_prev = evaluate_rhs(solver, prev_state, solver.sim_time - dt_previous)
            push!(F_history, _apply_mass_inverse(M_factor, fields_to_vector(F_prev)))
        end

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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    if L_matrix === nothing
        @warn "ETD-SBDF2 requires L_matrix, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end

    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    L_linear, M_factor = _get_linear_operator_eff!(state, L_matrix, M_matrix)

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w = dt_current / dt_previous

    try
        # Compute exponential integrators: exp(hL), φ₁(hL), φ₂(hL)
        exp_hL, φ₁_hL, φ₂_hL = phi_functions_matrix(L_linear, dt_current)

        # Convert current state to vector
        X_current = fields_to_vector(current_state)

        # Evaluate nonlinear term N(uₙ) at current state
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = _apply_mass_inverse(M_factor, fields_to_vector(F_current))

        # Rotate and store history
        F_history = state.timestepper_data["F_history"]
        pushfirst!(F_history, F_current_vec)

        # Keep only needed history for 2-step method
        while length(F_history) > 2
            pop!(F_history)
        end
        if length(F_history) < 2 && length(state.history) >= 2
            prev_state = state.history[end-1]
            F_prev = evaluate_rhs(solver, prev_state, solver.sim_time - dt_previous)
            push!(F_history, _apply_mass_inverse(M_factor, fields_to_vector(F_prev)))
        end
        if length(F_history) < 2
            @debug "ETD-SBDF2 missing previous RHS, falling back to ETDRK2"
            step_etd_rk222!(state, solver)
            state.timestepper_data["iteration"] = get(state.timestepper_data, "iteration", 0) + 1
            return
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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "MCNAB2 requires L_matrix and M_matrix, falling back to CNAB2"
        step_cnab2!(state, solver)
        return
    end

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
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    if L_matrix === nothing || M_matrix === nothing
        @warn "CNLF2 requires L_matrix and M_matrix, falling back to RK222"
        step_rk222!(state, solver)
        return
    end

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

            get_grid_data(new_field) .= alpha[2,1] .* get_grid_data(field0) .+
                                alpha[2,2] .* get_grid_data(field1) .+
                                dt * beta[2] .* get_grid_data(F1[i])
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

            get_grid_data(new_field) .= alpha[3,1] .* get_grid_data(field0) .+
                                alpha[3,3] .* get_grid_data(field2) .+
                                dt * beta[3] .* get_grid_data(F2[i])
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
    # Reuse the generic IMEX RK implementation for consistent M/L handling
    try
        step_rk_imex!(state, solver)
    catch e
        @warn "RKGFY failed: $e, falling back to RK443"
        step_rk443!(state, solver)
    end
end

function step_rk443_imex!(state::TimestepperState, solver::InitialValueSolver)
    # Reuse the generic IMEX RK implementation for consistent M/L handling
    try
        step_rk_imex!(state, solver)
    catch e
        @warn "RK443_IMEX failed: $e, falling back to RK443"
        step_rk443!(state, solver)
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
        # Sync current state into problem variables for expression evaluation
        sync_state_to_problem!(problem, state)
        
        # Update time parameter if it exists (like Tarang sim_time updates)
        if hasfield(typeof(problem), :time) && problem.time !== nothing
            # Update time field value for time-dependent expressions
            if problem.time isa ScalarField
                if get_grid_data(problem.time) !== nothing
                    ensure_layout!(problem.time, :g)
                    fill!(get_grid_data(problem.time), time)
                elseif get_coeff_data(problem.time) !== nothing
                    ensure_layout!(problem.time, :c)
                    fill!(get_coeff_data(problem.time), time)
                elseif hasfield(typeof(problem), :parameters)
                    problem.parameters["t"] = time
                end
            elseif hasfield(typeof(problem.time), :data)
                problem.time.data = time
            elseif hasfield(typeof(problem.time), :value)
                problem.time.value = time
            end
        end
        
        # Evaluate each equation's RHS (F expression) following Tarang pattern
        if hasfield(typeof(problem), :equation_data) && !isempty(problem.equation_data)
            for (eq_idx, eq_data) in enumerate(problem.equation_data)
                template = state[min(eq_idx, length(state))]
                expr = if haskey(eq_data, "F_expr") && eq_data["F_expr"] !== nothing
                    eq_data["F_expr"]
                else
                    get(eq_data, "F", ZeroOperator())
                end

                try
                    rhs_field = evaluate_solver_expression(expr, problem.variables; layout=:g, template=template)
                    if isa(rhs_field, ScalarField)
                        ensure_layout!(rhs_field, :c)
                        push!(rhs, rhs_field)
                    else
                        @warn "RHS expression $eq_idx did not evaluate to ScalarField, using zero field"
                        push!(rhs, create_rhs_zero_field(template))
                    end
                catch e
                    @warn "Failed to evaluate RHS expression for equation $eq_idx: $e"
                    push!(rhs, create_rhs_zero_field(template))
                end
            end
        else
            @warn "No equation_data found in problem, creating zero fields"
            for field in state
                push!(rhs, create_rhs_zero_field(field))
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
                    F_view = _matched_forcing_view(forcing, size(get_coeff_data(rhs_field)))

                    # Add forcing to RHS in coefficient space
                    ensure_layout!(rhs_field, :c)
                    if F_view !== nothing
                        get_coeff_data(rhs_field) .+= F_view
                        @debug "Added stochastic forcing to equation $var_idx"
                    else
                        @warn "Forcing size $(size(forcing.cached_forcing)) doesn't match RHS size $(size(get_coeff_data(rhs_field))) for equation $var_idx"
                    end
                end
            end
        end

    catch e
        @error "RHS evaluation failed: $e"
        # Fallback: create zero fields
        for field in state
            rhs_field = create_rhs_zero_field(field)
            push!(rhs, rhs_field)
        end
    end

    return rhs
end

function create_rhs_zero_field(template_field::ScalarField)
    """Create a zero RHS field matching the template field properties"""
    rhs_field = ScalarField(template_field.dist, "rhs_$(template_field.name)", template_field.bases, template_field.dtype)
    ensure_layout!(rhs_field, :c)  # Coefficient space following Tarang
    fill!(get_coeff_data(rhs_field), zero(eltype(get_coeff_data(rhs_field))))
    return rhs_field
end

# Expression evaluation is handled by the complete implementation in solvers.jl
# which supports the operator tree structure used in equation parsing

function add_scaled_state(state1::Vector{ScalarField}, state2::Vector{ScalarField}, scale::Float64)
    """Compute state1 + scale * state2 - OPTIMIZED version (GPU-aware)"""
    result = ScalarField[]

    for (i, field1) in enumerate(state1)
        field2 = state2[i]
        new_field = ScalarField(field1.dist, field1.name, field1.bases, field1.dtype)

        ensure_layout!(field1, :g)
        ensure_layout!(field2, :g)
        ensure_layout!(new_field, :g)

        # Check if on GPU - use broadcasting for GPU arrays (CUDA.jl optimizes this)
        if is_gpu_array(get_grid_data(field1))
            # GPU path: use broadcasting (CUDA.jl handles optimization)
            get_grid_data(new_field) .= get_grid_data(field1) .+ scale .* get_grid_data(field2)
        else
            # CPU path: use optimized in-place operations
            n = length(get_grid_data(field1))
            use_blas = n > 2000 &&
                       get_grid_data(field1) isa StridedArray &&
                       get_grid_data(field2) isa StridedArray &&
                       get_grid_data(new_field) isa StridedArray
            if use_blas
                # BLAS-based: y = α*x + β*y via axpby! or copy + axpy
                copyto!(get_grid_data(new_field), get_grid_data(field1))
                BLAS.axpy!(scale, get_grid_data(field2), get_grid_data(new_field))
            elseif n > 100
                # LoopVectorization for medium arrays
                scale_local = scale
                new_data = get_grid_data(new_field)
                data1 = get_grid_data(field1)
                data2 = get_grid_data(field2)
                @turbo for j in eachindex(new_data, data1, data2)
                    new_data[j] = data1[j] + scale_local * data2[j]
                end
            else
                # Broadcasting for small arrays
                get_grid_data(new_field) .= get_grid_data(field1) .+ scale .* get_grid_data(field2)
            end
        end
        push!(result, new_field)
    end

    return result
end

"""
    add_scaled_state!(dest::Vector{ScalarField}, state1::Vector{ScalarField},
                      state2::Vector{ScalarField}, scale::Float64)

In-place version: dest = state1 + scale * state2 (GPU-aware)
"""
function add_scaled_state!(dest::Vector{ScalarField}, state1::Vector{ScalarField},
                           state2::Vector{ScalarField}, scale::Float64)
    for (i, field1) in enumerate(state1)
        field2 = state2[i]
        dest_field = dest[i]

        ensure_layout!(field1, :g)
        ensure_layout!(field2, :g)
        ensure_layout!(dest_field, :g)

        # Check if on GPU - use broadcasting for GPU arrays
        if is_gpu_array(get_grid_data(field1))
            # GPU path: use broadcasting (CUDA.jl handles optimization)
            get_grid_data(dest_field) .= get_grid_data(field1) .+ scale .* get_grid_data(field2)
        else
            # CPU path: use optimized operations
            n = length(get_grid_data(field1))
            use_blas = n > 2000 &&
                       get_grid_data(field1) isa StridedArray &&
                       get_grid_data(field2) isa StridedArray &&
                       get_grid_data(dest_field) isa StridedArray
            if use_blas
                copyto!(get_grid_data(dest_field), get_grid_data(field1))
                BLAS.axpy!(scale, get_grid_data(field2), get_grid_data(dest_field))
            elseif n > 100
                scale_local = scale
                dest_data = get_grid_data(dest_field)
                data1 = get_grid_data(field1)
                data2 = get_grid_data(field2)
                @turbo for j in eachindex(dest_data, data1, data2)
                    dest_data[j] = data1[j] + scale_local * data2[j]
                end
            else
                get_grid_data(dest_field) .= get_grid_data(field1) .+ scale .* get_grid_data(field2)
            end
        end
    end
end

"""
    axpy_state!(scale::Float64, x::Vector{ScalarField}, y::Vector{ScalarField})

In-place AXPY: y = y + scale * x (GPU-aware)
"""
function axpy_state!(scale::Float64, x::Vector{ScalarField}, y::Vector{ScalarField})
    for i in eachindex(x, y)
        ensure_layout!(x[i], :g)
        ensure_layout!(y[i], :g)

        # Check if on GPU - use broadcasting for GPU arrays
        if is_gpu_array(get_grid_data(x[i]))
            # GPU path: use broadcasting (CUDA.jl handles optimization)
            get_grid_data(y[i]) .+= scale .* get_grid_data(x[i])
        else
            # CPU path: use optimized operations
            n = length(get_grid_data(x[i]))
            use_blas = n > 2000 &&
                       get_grid_data(x[i]) isa StridedArray &&
                       get_grid_data(y[i]) isa StridedArray
            if use_blas
                BLAS.axpy!(scale, get_grid_data(x[i]), get_grid_data(y[i]))
            elseif n > 100
                scale_local = scale
                y_data = get_grid_data(y[i])
                x_data = get_grid_data(x[i])
                @turbo for j in eachindex(y_data, x_data)
                    y_data[j] += scale_local * x_data[j]
                end
            else
                get_grid_data(y[i]) .+= scale .* get_grid_data(x[i])
            end
        end
    end
end

"""
    linear_combination_state!(dest::Vector{ScalarField}, α::Float64, a::Vector{ScalarField},
                              β::Float64, b::Vector{ScalarField})

In-place linear combination: dest = α*a + β*b (GPU-aware)
"""
function linear_combination_state!(dest::Vector{ScalarField}, α::Float64, a::Vector{ScalarField},
                                   β::Float64, b::Vector{ScalarField})
    for i in eachindex(dest, a, b)
        ensure_layout!(a[i], :g)
        ensure_layout!(b[i], :g)
        ensure_layout!(dest[i], :g)

        # Check if on GPU - use broadcasting for GPU arrays
        if is_gpu_array(get_grid_data(a[i]))
            # GPU path: use broadcasting (CUDA.jl handles optimization)
            get_grid_data(dest[i]) .= α .* get_grid_data(a[i]) .+ β .* get_grid_data(b[i])
        else
            # CPU path: use optimized operations
            n = length(get_grid_data(a[i]))
            if n > 100
                α_local, β_local = α, β
                dest_data = get_grid_data(dest[i])
                a_data = get_grid_data(a[i])
                b_data = get_grid_data(b[i])
                @turbo for j in eachindex(dest_data, a_data, b_data)
                    dest_data[j] = α_local * a_data[j] + β_local * b_data[j]
                end
            else
                get_grid_data(dest[i]) .= α .* get_grid_data(a[i]) .+ β .* get_grid_data(b[i])
            end
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
        get_grid_data(new_field) .= get_grid_data(field)
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

# ============================================================================
# Exports
# ============================================================================

# Export abstract type
export TimeStepper

# Export explicit Runge-Kutta schemes
export RK111, RK222, RK443

# Export IMEX (Implicit-Explicit) schemes
export CNAB1, CNAB2  # Crank-Nicolson Adams-Bashforth
export SBDF1, SBDF2, SBDF3, SBDF4  # Semi-implicit BDF
export MCNAB2  # Modified Crank-Nicolson Adams-Bashforth
export CNLF2   # Crank-Nicolson Leap-Frog
export RK443_IMEX  # 4th order IMEX Runge-Kutta

# Export exponential time differencing schemes
export ETD_RK222, ETD_CNAB2, ETD_SBDF2

# Export specialized schemes
export RKSMR  # Runge-Kutta for stiff problems
export RKGFY  # Runge-Kutta Guo-Feng-Yang

# Export timestepper state management
export TimestepperState
export set_forcing!, update_forcing!, reset_forcing_flag!, get_cached_forcing

# Export step functions
export step!
export step_rk_imex!, step_cnab1!, step_cnab2!
export step_sbdf1!, step_sbdf2!, step_sbdf3!, step_sbdf4!
export step_etd_rk222!, step_etd_cnab2!, step_etd_sbdf2!
export step_mcnab2!, step_cnlf2!, step_rksmr!, step_rkgfy!, step_rk443_imex!
export evaluate_rhs

# Export timestep history management
export update_timestep_history!, get_previous_timestep, get_max_timestep_history

# Export state utilities
export copy_state, copy_field_data!
export add_scaled_state, add_scaled_state!, axpy_state!, linear_combination_state!

# Export φ functions for exponential integrators
export phi_functions, phi_functions_matrix, phiv_vector, expv_krylov

# Export workspace management
export _workspace_count, get_workspace_field!
