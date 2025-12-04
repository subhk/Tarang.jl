"""
Timestepping schemes for initial value problems

Translated from dedalus/core/timesteppers.py
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
    """Stable computation of φ₂"""
    z_norm = norm(z)
    if z_norm < 1e-2
        # Use series expansion for better accuracy  
        return I/2 + z/6 + z^2/24 + z^3/120 + z^4/720
    else
        return (φ₁ - inv(z)) * inv(z)
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

# Explicit Runge-Kutta methods
struct RK111 <: TimeStepper
    # Forward Euler
    stages::Int
    coefficients::Matrix{Float64}
    
    function RK111()
        stages = 1
        coefficients = reshape([1.0], 1, 1)
        new(stages, coefficients)
    end
end

struct RK222 <: TimeStepper
    # 2nd order Runge-Kutta (midpoint method)
    stages::Int
    coefficients::Matrix{Float64}
    
    function RK222()
        stages = 2
        coefficients = [0.0 1.0; 0.5 0.0]
        new(stages, coefficients)
    end
end

struct RK443 <: TimeStepper
    # 4th order Runge-Kutta
    stages::Int
    coefficients::Matrix{Float64}
    
    function RK443()
        stages = 4
        # Classical RK4 coefficients
        coefficients = [
            0.0  1.0  0.0  0.0;
            0.5  0.0  1.0  0.0;
            0.5  0.0  0.0  1.0;
            1.0  0.0  0.0  0.0
        ]
        new(stages, coefficients)
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

# Timestepper state management
mutable struct TimestepperState
    timestepper::TimeStepper
    dt::Float64
    history::Vector{Vector{ScalarField}}
    dt_history::Vector{Float64}  # Track timestep history for variable timesteps
    stage::Int
    timestepper_data::Dict{String, Any}  # Additional data for specific timesteppers

    function TimestepperState(timestepper::TimeStepper, dt::Float64, initial_state::Vector{ScalarField})
        history = [copy.(initial_state)]
        dt_history = [dt]  # Initialize with current timestep
        timestepper_data = Dict{String, Any}()
        new(timestepper, dt, history, dt_history, 0, timestepper_data)
    end
end

# Timestepping implementation
function step!(state::TimestepperState, solver::InitialValueSolver)
    """Advance solution by one timestep"""
    
    if isa(state.timestepper, RK111)
        step_rk111!(state, solver)
    elseif isa(state.timestepper, RK222)
        step_rk222!(state, solver)
    elseif isa(state.timestepper, RK443)
        step_rk443!(state, solver)
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
    else
        throw(ArgumentError("Unknown timestepper type: $(typeof(state.timestepper))"))
    end
end

# Explicit Runge-Kutta implementations
function step_rk111!(state::TimestepperState, solver::InitialValueSolver)
    """Forward Euler step"""
    current_state = state.history[end]
    dt = state.dt
    
    # Evaluate RHS: du/dt = F(u)
    rhs = evaluate_rhs(solver, current_state, solver.sim_time)
    
    # Forward Euler: u^{n+1} = u^n + dt * F(u^n)
    new_state = ScalarField[]
    for (i, field) in enumerate(current_state)
        new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
        ensure_layout!(field, :g)
        ensure_layout!(rhs[i], :g)
        ensure_layout!(new_field, :g)
        
        # Forward Euler: u^{n+1} = u^n + dt * F(u^n)
        # Multi-tier implementation: BLAS > LoopVectorization > Broadcasting
        n = length(field.data_g)
        new_field.data_g .= field.data_g  # Copy initial state
        
        if n > 2000
            BLAS.axpy!(dt, rhs[i].data_g, new_field.data_g)  # BLAS for very large arrays
        elseif n > 100
            # LoopVectorization for medium arrays
            @turbo for j in eachindex(new_field.data_g, rhs[i].data_g)
                new_field.data_g[j] += dt * rhs[i].data_g[j]
            end
        else
            new_field.data_g .+= dt .* rhs[i].data_g  # Broadcasting for small arrays
        end
        push!(new_state, new_field)
    end
    
    push!(state.history, new_state)
    
    # Keep only necessary history
    if length(state.history) > 1
        popfirst!(state.history)
    end
end

function step_rk222!(state::TimestepperState, solver::InitialValueSolver)
    """2nd order Runge-Kutta (midpoint method)"""
    current_state = state.history[end]
    dt = state.dt
    
    # Stage 1: k1 = F(t, u^n)
    k1 = evaluate_rhs(solver, current_state, solver.sim_time)
    
    # Stage 2: u_temp = u^n + dt/2 * k1
    temp_state = ScalarField[]
    for (i, field) in enumerate(current_state)
        temp_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
        ensure_layout!(field, :g)
        ensure_layout!(k1[i], :g)
        ensure_layout!(temp_field, :g)
        
        temp_field.data_g .= field.data_g .+ (dt/2) .* k1[i].data_g
        push!(temp_state, temp_field)
    end
    
    # Stage 2: k2 = F(t + dt/2, u_temp)
    k2 = evaluate_rhs(solver, temp_state, solver.sim_time + dt/2)
    
    # Final update: u^{n+1} = u^n + dt * k2
    new_state = ScalarField[]
    for (i, field) in enumerate(current_state)
        new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
        ensure_layout!(field, :g)
        ensure_layout!(k2[i], :g)
        ensure_layout!(new_field, :g)
        
        new_field.data_g .= field.data_g .+ dt .* k2[i].data_g
        push!(new_state, new_field)
    end
    
    push!(state.history, new_state)
    
    # Keep only necessary history
    if length(state.history) > 1
        popfirst!(state.history)
    end
end

function step_rk443!(state::TimestepperState, solver::InitialValueSolver)
    """4th order Runge-Kutta"""
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    
    # Stage 1: k1 = F(t, u^n)
    k1 = evaluate_rhs(solver, current_state, t)
    
    # Stage 2: u_temp = u^n + dt/2 * k1
    temp_state1 = add_scaled_state(current_state, k1, dt/2)
    k2 = evaluate_rhs(solver, temp_state1, t + dt/2)
    
    # Stage 3: u_temp = u^n + dt/2 * k2
    temp_state2 = add_scaled_state(current_state, k2, dt/2)
    k3 = evaluate_rhs(solver, temp_state2, t + dt/2)
    
    # Stage 4: u_temp = u^n + dt * k3
    temp_state3 = add_scaled_state(current_state, k3, dt)
    k4 = evaluate_rhs(solver, temp_state3, t + dt)
    
    # Final update: u^{n+1} = u^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_state = ScalarField[]
    for (i, field) in enumerate(current_state)
        new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
        ensure_layout!(field, :g)
        ensure_layout!(new_field, :g)
        
        # RK4 combination
        rhs_combined = (k1[i].data_g .+ 2 .* k2[i].data_g .+ 2 .* k3[i].data_g .+ k4[i].data_g) ./ 6
        new_field.data_g .= field.data_g .+ dt .* rhs_combined
        push!(new_state, new_field)
    end
    
    push!(state.history, new_state)
    
    # Keep only necessary history
    if length(state.history) > 1
        popfirst!(state.history)
    end
end

# Implicit-explicit methods
function step_cnab1!(state::TimestepperState, solver::InitialValueSolver)
    """
    Crank-Nicolson Adams-Bashforth 1st order following Dedalus MultistepIMEX implementation.
    
    Based on Dedalus timesteppers.py:95-188 MultistepIMEX.step method:
    - Proper coefficient computation: a[0] = 1/dt, a[1] = -1/dt, b[0] = 1/2, b[1] = 1/2, c[1] = 1
    - RHS construction: c[1]*F[0] - a[1]*MX[0] - b[1]*LX[0] (following lines 156-166)
    - LHS solution: (a[0]*M + b[0]*L).X = RHS (following lines 174-184)
    - Proper state rotation and history management
    """
    
    current_state = state.history[end]
    dt = state.dt
    
    # Initialize history arrays if needed (following Dedalus MultistepIMEX.__init__)
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
    
    # Get CNAB1 coefficients following Dedalus (timesteppers.py:206-220)
    a = [1.0/dt, -1.0/dt]  # a[0], a[1]
    b = [0.5, 0.5]         # b[0], b[1] 
    c = [0.0, 1.0]         # c[0], c[1]
    
    try
        # Step 1: Convert current state to vector (following Dedalus gather_inputs)
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0] (following Dedalus lines 142-147)
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step (following Dedalus lines 149-153)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history (following Dedalus lines 124-126)
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
        
        # Step 5: Build RHS following Dedalus exactly (timesteppers.py:156-166)
        # RHS = c[1] * F[0] - a[1] * MX[0] - b[1] * LX[0]
        rhs = c[2] * F_history[1]  # c[1] * F[0] (using 1-based indexing)
        if length(MX_history) >= 1  # a[1] term
            rhs .-= a[2] * MX_history[1]  # -a[1] * MX[0]
        end
        if length(LX_history) >= 1  # b[1] term
            rhs .-= b[2] * LX_history[1]  # -b[1] * LX[0]
        end
        
        # Step 6: Build and solve LHS system (following Dedalus lines 174-184)
        # (a[0] * M + b[0] * L).X = RHS
        LHS_matrix = a[1] * M_matrix + b[1] * L_matrix  # a[0] * M + b[0] * L
        X_new = LHS_matrix \ rhs
        
        # Step 7: Update state (following Dedalus scatter_inputs)
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
    Crank-Nicolson Adams-Bashforth 2nd order following Dedalus MultistepIMEX implementation.
    
    Based on Dedalus timesteppers.py:95-188 MultistepIMEX.step method:
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
    
    # Check if we have enough history for CNAB2 (following Dedalus line 274)
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
    
    # Get timestep history for variable timestep (following Dedalus lines 280-281)
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous
    
    # Get CNAB2 coefficients following Dedalus exactly (timesteppers.py:283-288)
    a = [1.0/dt_current, -1.0/dt_current]  # a[0], a[1]
    b = [0.5, 0.5]                         # b[0], b[1]
    c = [0.0, 1.0 + w1/2.0, -w1/2.0]      # c[0], c[1], c[2]
    
    @debug "CNAB2 variable timestep: dt_current=$dt_current, dt_previous=$dt_previous, w1=$w1"
    
    try
        # Step 1: Convert current state to vector
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0] (following Dedalus lines 142-147)
        MX_current = M_matrix * X_current
        LX_current = L_matrix * X_current
        
        # Step 3: Evaluate F(X[0]) at current time step (following Dedalus lines 149-153)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)
        
        # Step 4: Rotate and store history (following Dedalus lines 124-126)
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
        
        # Step 5: Build RHS following Dedalus exactly (timesteppers.py:156-166)
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
        
        # Step 6: Build and solve LHS system (following Dedalus lines 174-184)
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
    Semi-implicit BDF1 (backward Euler) following Dedalus MultistepIMEX implementation.
    
    Based on Dedalus timesteppers.py:224-252 SBDF1 coefficients:
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
    
    # Get SBDF1 coefficients following Dedalus exactly (timesteppers.py:247-250)
    a = [1.0/dt, -1.0/dt]  # a[0], a[1] - BDF1 time derivative
    b = [1.0]              # b[0] - fully implicit (not 1/2 like CNAB)
    c = [0.0, 1.0]         # c[0], c[1] - forward Euler explicit
    
    try
        # Step 1: Convert current state to vector
        X_current = fields_to_vector(current_state)
        
        # Step 2: Compute M.X[0] and L.X[0] (following Dedalus MultistepIMEX pattern)
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
        
        # Step 5: Build RHS following Dedalus MultistepIMEX pattern
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
    Semi-implicit BDF2 following Dedalus MultistepIMEX implementation.
    
    Based on Dedalus timesteppers.py:333-367 SBDF2 coefficients:
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
    
    # Check if we have enough history for SBDF2 (following Dedalus line 350)
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
    
    # Get timestep history for variable timestep (following Dedalus lines 357-358)
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w1 = dt_current / dt_previous
    
    # Get SBDF2 coefficients following Dedalus exactly (timesteppers.py:360-365)
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
        
        # Step 5: Build RHS following Dedalus MultistepIMEX pattern
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
    Semi-implicit BDF3 following Dedalus implementation.
    
    Dedalus coefficients (timesteppers.py:425-447):
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
    
    # Get timestep history for variable timestep ratios (Dedalus pattern)
    if length(state.dt_history) < 3
        @warn "SBDF3 requires 3 timestep history, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end
    
    k2 = state.dt_history[end]     # current timestep
    k1 = state.dt_history[end-1]   # previous timestep  
    k0 = state.dt_history[end-2]   # timestep before that
    
    # Compute timestep ratios following Dedalus (timesteppers.py:435-436)
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
        # Get SBDF3 coefficients following Dedalus exactly (timesteppers.py:438-445)
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
        
        # Build RHS following Dedalus multistep pattern
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
    Semi-implicit BDF4 following Dedalus implementation.
    
    Dedalus coefficients (timesteppers.py:466-495):
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
    
    # Compute timestep ratios following Dedalus (timesteppers.py:476-478)
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
        # Get SBDF4 coefficients following Dedalus exactly (timesteppers.py:480-494)
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
        
        # Build RHS following Dedalus multistep pattern
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

    Note: This method is called ETD-CNAB2 following Dedalus naming convention,
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
    2nd-order exponential multistep method (ETDBDF2-style).

    Formulation (simplified exponential BDF2):
    u_{n+1} = a₀*exp(hL)u_n + a₁*exp(2hL)u_{n-1} + h*φ₁(hL)*N_BDF2

    where:
    - a₀, a₁ are BDF2-derived coefficients adjusted for variable timesteps
    - N_BDF2 = (1 + w)*N(u_n) - w*N(u_{n-1}) is BDF2 extrapolation
    - w = h_n / h_{n-1} is the timestep ratio

    Linear treatment: Exact via exponential propagator
    Nonlinear treatment: BDF2-style extrapolation

    Note: This is a simplified exponential BDF variant. True exponential BDF2
    methods (ETDBDF2) are more complex and require additional φ functions.
    This implementation provides good stability with BDF-like damping properties.

    References:
    - Hochbruck & Ostermann (2010), "Exponential integrators"
    - Kassam & Trefethen (2005), "Fourth-order time-stepping for stiff PDEs"
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
        return
    end

    # Get matrices from solver
    if !haskey(solver.problem.parameters, "L_matrix") || !haskey(solver.problem.parameters, "M_matrix")
        @warn "ETD-SBDF2 requires L_matrix and M_matrix, falling back to SBDF2"
        step_sbdf2!(state, solver)
        return
    end

    L_matrix = solver.problem.parameters["L_matrix"]
    M_matrix = solver.problem.parameters["M_matrix"]

    # Get timestep history for variable timestep
    dt_current = dt
    dt_previous = get_previous_timestep(state)
    w = dt_current / dt_previous

    try
        # Compute exponential integrators for current and previous timesteps
        exp_hL, φ₁_hL, _ = phi_functions_matrix(L_matrix, dt_current)

        # For the previous state, we need exp(h_previous * L) applied to u_{n-1}
        # In the exponential BDF framework, we need proper weighting

        # Convert states to vectors
        X_current = fields_to_vector(current_state)
        X_previous = fields_to_vector(state.history[end-1])

        # Evaluate nonlinear term N(u_n)
        F_current = evaluate_rhs(solver, current_state, solver.sim_time)
        F_current_vec = fields_to_vector(F_current)

        # Rotate and store history
        F_history = state.timestepper_data["F_history"]
        pushfirst!(F_history, F_current_vec)

        # Keep only needed history for BDF2
        while length(F_history) > 2; pop!(F_history); end

        # BDF2-style extrapolation coefficients for nonlinear terms
        # N_BDF2 = c₁*N(u_n) + c₂*N(u_{n-1})
        c₁ = 1.0 + w         # Current step weight (BDF2 extrapolation)
        c₂ = -w              # Previous step weight

        # Build BDF2-extrapolated nonlinear term
        F_extrap = c₁ * F_history[1]
        if length(F_history) >= 2
            F_extrap .+= c₂ * F_history[2]
        end

        # Exponential BDF2-style coefficients for linear part
        # These provide the implicit stability of BDF2 via exponential propagation
        # Simplified form: a₀ ~ (1+2w)/(1+w), a₁ ~ -w²/(1+w)
        a₀ = (1.0 + 2.0*w) / (1.0 + w)
        a₁ = -w * w / (1.0 + w)

        # Compute exponential propagation with BDF2 weighting:
        # u_{n+1} = a₀*exp(hL)u_n + a₁*exp(hL)u_{n-1} + h*φ₁(hL)*N_BDF2
        #
        # Note: a₁*exp(hL)u_{n-1} ≈ a₁*u_{n-1} for small hL, but we use
        # exponential propagation for consistency
        X_propagated = a₀ * (exp_hL * X_current) + a₁ * (exp_hL * X_previous)

        # Add nonlinear contribution
        X_new = X_propagated + dt_current * (φ₁_hL * F_extrap)

        # Update state
        new_state = copy.(current_state)
        copy_solution_to_fields!(new_state, X_new)

        push!(state.history, new_state)
        state.timestepper_data["iteration"] += 1

        @debug "ETDBDF2 step completed: dt=$dt_current, w=$w, a₀=$a₀, a₁=$a₁, iteration=$(state.timestepper_data["iteration"]), |X_new|=$(norm(X_new))"

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

# Helper functions
function evaluate_rhs(solver::InitialValueSolver, state::Vector{ScalarField}, time::Float64)
    """
    Evaluate right-hand side of differential equations following Dedalus pattern.
    
    Based on Dedalus MultistepIMEX.step method (timesteppers.py:149-153):
    - evaluator.evaluate_scheduled(iteration=iteration, wall_time=wall_time, sim_time=sim_time, timestep=dt)
    - evaluator.require_coeff_space(F_fields)  
    - sp.gather_outputs(F_fields, out=F0.get_subdata(sp))
    
    This evaluates the F expressions from problem.equation_data for each equation.
    """
    
    problem = solver.problem
    rhs = ScalarField[]
    
    try
        # Set current state fields to the provided state for evaluation
        # This mimics how Dedalus sets the current field values before evaluation
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
        
        # Update time parameter if it exists (like Dedalus sim_time updates)
        if hasfield(typeof(problem), :time) && problem.time !== nothing
            # Update time field value for time-dependent expressions
            if hasfield(typeof(problem.time), :data)
                problem.time.data = time
            elseif hasfield(typeof(problem.time), :value)
                problem.time.value = time
            end
        end
        
        # Evaluate each equation's RHS (F expression) following Dedalus pattern
        if hasfield(typeof(problem), :equation_data) && !isempty(problem.equation_data)
            for (eq_idx, eq_data) in enumerate(problem.equation_data)
                if haskey(eq_data, "F_expr") && eq_data["F_expr"] !== nothing
                    # Evaluate the F expression symbolically
                    try
                        rhs_field = evaluate_expression(eq_data["F_expr"], problem.variables)
                        
                        # Ensure correct field properties
                        if isa(rhs_field, ScalarField)
                            # Convert to coefficient space as done in Dedalus
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
                    @warn "No F_expr found for equation $eq_idx, creating zero field"
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
    ensure_layout!(rhs_field, :c)  # Coefficient space following Dedalus
    fill!(rhs_field.data_c, 0.0)
    return rhs_field
end

# Expression evaluation is handled by the complete implementation in solvers.jl
# which supports the operator tree structure used in equation parsing

function add_scaled_state(state1::Vector{ScalarField}, state2::Vector{ScalarField}, scale::Float64)
    """Compute state1 + scale * state2"""
    result = ScalarField[]
    
    for (i, field1) in enumerate(state1)
        field2 = state2[i]
        new_field = ScalarField(field1.dist, field1.name, field1.bases, field1.dtype)
        
        ensure_layout!(field1, :g)
        ensure_layout!(field2, :g)
        ensure_layout!(new_field, :g)
        
        new_field.data_g .= field1.data_g .+ scale .* field2.data_g
        push!(result, new_field)
    end
    
    return result
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
    """Update timestep history following Dedalus deque rotation pattern"""
    # Update current timestep
    state.dt = dt
    
    # Add to history (following Dedalus rotation)
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
    elseif isa(timestepper, Union{CNAB2, SBDF2, ETD_CNAB2, ETD_SBDF2})
        return 3  # Current + 2 previous  
    elseif isa(timestepper, Union{SBDF3})
        return 4  # Current + 3 previous
    elseif isa(timestepper, Union{SBDF4})
        return 5  # Current + 4 previous
    elseif isa(timestepper, Union{ETD_RK222})
        return 2  # Current + 1 previous for exponential methods
    else
        return 2  # Default for explicit methods
    end
end

# Helper functions for exponential integrators are defined in solvers.jl:
# - fields_to_vector: Convert fields to coefficient-space vector
# - copy_solution_to_fields!: Copy solution vector back to fields

