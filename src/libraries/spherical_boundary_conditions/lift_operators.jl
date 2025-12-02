"""
Lift operators for spherical boundary conditions following dedalus approach.

Implements the lift operator system for integrating tau terms with weak formulations:
- Surface lift operators for boundary enforcement at r=R
- Basis lifting for spectral tau method
- Matrix construction for lift operations
"""

using LinearAlgebra
using SparseArrays
include("utilities.jl")

export LiftOperator, LiftTerm
export build_lift_matrices, build_surface_lift_matrix
export compute_lift_weight, get_lift_mode_number, get_robin_lift_modes
export apply_dirichlet_lift!, apply_neumann_lift!, apply_robin_lift!, apply_regularity_lift!

"""
Lift operator for tau method implementation.

Represents the lifting of tau variables from boundary to domain following
dedalus generalized tau approach.
"""
struct LiftOperator{T<:Real}
    basis_type::Symbol                  # :ball, :sphere, :shell
    lift_mode::Int                     # Mode number to lift to (-1 for highest)
    lift_matrices::Dict{Symbol, SparseMatrixCSC{Complex{T}, Int}}
    boundary_location::Symbol          # :surface, :center
    
    function LiftOperator{T}(basis_type::Symbol, lift_mode::Int, 
                           boundary_location::Symbol=:surface) where T<:Real
        lift_matrices = Dict{Symbol, SparseMatrixCSC{Complex{T}, Int}}()
        new{T}(basis_type, lift_mode, lift_matrices, boundary_location)
    end
end

"""
Lift term combining lift operator with tau variable.

Represents the term Lift(tau_var) that appears in weak formulations.
"""
struct LiftTerm{T<:Real}
    lift_operator::LiftOperator{T}
    tau_variable_name::String
    coefficient::Complex{T}
    
    function LiftTerm{T}(lift_op::LiftOperator{T}, tau_name::String, 
                        coeff::Complex{T}=Complex{T}(1)) where T<:Real
        new{T}(lift_op, tau_name, coeff)
    end
end

"""
    build_lift_matrices(basis, T) -> Dict{Symbol, SparseMatrixCSC}

Build lift matrices for different boundary condition types.

Following dedalus approach, lift matrices map tau variables to domain modes:
L[bc_type][mode_idx, tau_idx] represents lifting tau_idx to domain mode_idx
for boundary condition type bc_type.

# Arguments
- `basis`: Ball basis specification 
- `T::Type`: Numeric precision type

# Returns
Dictionary with lift matrices for :dirichlet, :neumann, :robin, :regularity
"""
function build_lift_matrices(basis::BallBasis{T}, ::Type{T}) where T<:Real
    lift_matrices = Dict{Symbol, SparseMatrixCSC{Complex{T}, Int}}()
    
    # Get basis dimensions
    n_modes = basis.n_total_modes
    n_radial = basis.n_radial
    
    # Build surface lift matrix (most common case)
    lift_matrices[:dirichlet] = build_surface_lift_matrix(basis, -1, T)  # k=-1 (highest mode)
    lift_matrices[:neumann] = build_surface_lift_matrix(basis, -2, T)    # k=-2 (second highest)
    
    # Robin condition requires TWO separate lift matrices following dedalus approach
    lift_matrices[:robin_value] = build_surface_lift_matrix(basis, -1, T)  # k=-1 for value term
    lift_matrices[:robin_deriv] = build_surface_lift_matrix(basis, -2, T)  # k=-2 for derivative term
    
    lift_matrices[:regularity] = build_regularity_lift_matrix(basis, T)   # Center/pole regularity
    
    return lift_matrices
end

"""
    build_surface_lift_matrix(basis, k, T) -> SparseMatrixCSC

Build lift matrix for surface boundary conditions at r=R.

The lift matrix L satisfies: (Lifted_field)[mode_idx] = L[mode_idx, tau_idx] * tau[tau_idx]
where the lifting maps tau variables to the k-th highest radial mode.

# Mathematical Background
For ball basis with Zernike polynomials Z_n^l(r):
- k=-1 lifts to highest radial mode n_max for each (l,m)
- k=-2 lifts to second-highest mode n_max-2 for each (l,m)

The lift weight comes from boundary evaluation: Z_n^l(r=1) = 1
"""
function build_surface_lift_matrix(basis::BallBasis{T}, k::Int, ::Type{T}) where T<:Real
    n_modes = basis.n_total_modes
    n_tau = basis.n_boundary_modes  # Number of boundary tau variables
    
    # Initialize sparse matrix
    I = Int[]
    J = Int[]
    V = Complex{T}[]
    
    mode_idx = 0
    tau_idx = 0
    
    # Iterate over spectral modes
    for l in 0:basis.l_max
        for m in -l:l
            tau_idx += 1  # Each (l,m) gets one tau variable
            
            for n in 0:basis.n_max
                mode_idx += 1
                
                # Determine lift mode based on k
                lift_n = if k == -1
                    basis.n_max  # Highest radial mode
                elseif k == -2
                    max(0, basis.n_max - 2)  # Second highest
                else
                    max(0, basis.n_max + k)  # k-th from top
                end
                
                # Only lift to specified radial mode
                if n == lift_n
                    # Compute lift weight
                    weight = compute_lift_weight(mode_idx, lift_n, basis, T)
                    
                    if abs(weight) > 1e-14  # Avoid numerical zeros
                        push!(I, mode_idx)
                        push!(J, tau_idx)
                        push!(V, weight)
                    end
                end
                
                if mode_idx > n_modes
                    break
                end
            end
            
            if mode_idx > n_modes
                break
            end
        end
        
        if mode_idx > n_modes
            break
        end
    end
    
    return sparse(I, J, V, n_modes, n_tau)
end

"""
    build_regularity_lift_matrix(basis, T) -> SparseMatrixCSC

Build lift matrix for regularity conditions (center/poles).

Regularity lifting is more complex as it involves mode coupling:
- Center regularity: lifts to l=0 modes only
- Pole regularity: lifts to m=0 modes with proper weights
"""
function build_regularity_lift_matrix(basis::BallBasis{T}, ::Type{T}) where T<:Real
    n_modes = basis.n_total_modes
    n_reg_tau = count_regularity_modes(basis)  # Number of regularity tau variables
    
    I = Int[]
    J = Int[]
    V = Complex{T}[]
    
    mode_idx = 0
    reg_tau_idx = 0
    
    for l in 0:basis.l_max
        for m in -l:l
            for n in 0:basis.n_max
                mode_idx += 1
                
                # Center regularity: only l=0 modes
                if l == 0 && m == 0
                    reg_tau_idx += 1
                    
                    # Weight from Zernike evaluation at center
                    weight = evaluate_zernike_at_center(n, l, T)
                    
                    # Apply Y_0^0 normalization
                    weight *= T(1) / sqrt(4 * π)
                    
                    if abs(weight) > 1e-14
                        push!(I, mode_idx)
                        push!(J, reg_tau_idx)
                        push!(V, Complex{T}(weight))
                    end
                end
                
                # Pole regularity: only m=0 modes  
                if m == 0 && l > 0
                    pole_tau_idx = reg_tau_idx + l  # Separate tau for each l
                    
                    # Weight from spherical harmonic at poles
                    weight = sqrt((2*l + 1) / (4 * π))  # Y_l^0 normalization
                    
                    if abs(weight) > 1e-14
                        push!(I, mode_idx)
                        push!(J, min(pole_tau_idx, n_reg_tau))  # Bounds check
                        push!(V, Complex{T}(weight))
                    end
                end
                
                if mode_idx > n_modes
                    break
                end
            end
            
            if mode_idx > n_modes
                break
            end
        end
        
        if mode_idx > n_modes
            break
        end
    end
    
    return sparse(I, J, V, n_modes, n_reg_tau)
end

"""
    count_regularity_modes(basis) -> Int

Count number of regularity tau variables needed.

Center regularity needs one tau per n for l=0.
Pole regularity needs one tau per l for m=0.
"""
function count_regularity_modes(basis::BallBasis{T}) where T<:Real
    center_modes = basis.n_max + 1  # n = 0, 1, ..., n_max for l=0, m=0
    pole_modes = basis.l_max        # l = 1, 2, ..., l_max for m=0
    return center_modes + pole_modes
end

"""
    compute_lift_weight(mode_idx, lift_mode, basis, T) -> Complex{T}

Compute lift weight for mapping tau variable to spectral mode.

The lift weight depends on boundary evaluation of basis functions:
- Surface lift: Z_n^l(r=1) = 1 for all modes
- Derivative lift: ∂Z_n^l/∂r|_{r=1} depends on n,l
- Regularity lift: special evaluation at center/poles
"""
function compute_lift_weight(mode_idx::Int, lift_mode::Int, lift_basis::Any, ::Type{T}) where T<:Real
    # Get (n,l,m) indices for this mode
    n, l, m = get_mode_indices(mode_idx, lift_basis)
    
    if lift_mode == -1
        # Surface Dirichlet: Z_n^l(r=1) = 1
        return Complex{T}(1)
        
    elseif lift_mode == -2  
        # Surface Neumann: ∂Z_n^l/∂r|_{r=1}
        deriv_val = evaluate_zernike_derivative_at_boundary(n, 1, T)
        return Complex{T}(deriv_val)
        
    elseif lift_mode == 0
        # Center regularity
        if l == 0 && m == 0
            center_val = evaluate_zernike_at_center(n, 0, T) 
            return Complex{T}(center_val / sqrt(4 * π))  # Y_0^0 factor
        else
            return Complex{T}(0)
        end
        
    else
        # General case: evaluate at specified radial mode
        return Complex{T}(1)  # Default uniform weight
    end
end

"""
    get_mode_indices(mode_idx, basis) -> (n, l, m)

Extract (n,l,m) quantum numbers from linear mode index.

Assumes standard ordering: iterate l, then m∈[-l:l], then n∈[0:n_max].
"""
function get_mode_indices(mode_idx::Int, basis::BallBasis{T}) where T<:Real
    idx = mode_idx - 1  # Convert to 0-based indexing
    
    for l in 0:basis.l_max
        modes_per_l = 2*l + 1  # m ∈ [-l:l]
        radial_modes = basis.n_max + 1  # n ∈ [0:n_max]
        modes_this_l = modes_per_l * radial_modes
        
        if idx < modes_this_l
            # Found the right l
            remaining = idx
            n = remaining ÷ modes_per_l
            m_idx = remaining % modes_per_l
            m = m_idx - l  # Convert to m ∈ [-l:l]
            
            return n, l, m
        end
        
        idx -= modes_this_l
    end
    
    error("Mode index $mode_idx exceeds basis size")
end

"""
    apply_dirichlet_lift!(spectral_data, tau_coeffs, lift_matrix)

Apply Dirichlet lift operation: field += Lift(tau) for Dirichlet BC.

Modifies spectral_data in place by adding lifted tau contributions.
"""
function apply_dirichlet_lift!(spectral_data::Vector{Complex{T}}, tau_coeffs::Vector{Complex{T}},
                              lift_matrix::SparseMatrixCSC{Complex{T}, Int}) where T<:Real
    # Lift operation: spectral_data += L * tau_coeffs
    lifted_contribution = lift_matrix * tau_coeffs
    spectral_data .+= lifted_contribution
end

"""
    apply_neumann_lift!(spectral_data, tau_coeffs, lift_matrix)

Apply Neumann lift operation: field += Lift(tau) for Neumann BC.

Uses derivative-based lift matrix for boundary condition enforcement.
"""
function apply_neumann_lift!(spectral_data::Vector{Complex{T}}, tau_coeffs::Vector{Complex{T}},
                            lift_matrix::SparseMatrixCSC{Complex{T}, Int}) where T<:Real
    # Neumann lift accounts for derivative boundary condition
    lifted_contribution = lift_matrix * tau_coeffs
    spectral_data .+= lifted_contribution
end

"""
    apply_robin_lift!(spectral_data, tau_coeffs_value, tau_coeffs_deriv, value_lift_matrix, deriv_lift_matrix, a_coeff, b_coeff)

Apply Robin lift operation: field += Lift(tau) for Robin BC following dedalus approach.

Robin conditions a*f + b*df/dr = c require TWO separate tau terms and lift matrices:
- tau_value with lift to k=-1 (highest radial mode) for the f term
- tau_deriv with lift to k=-2 (second highest radial mode) for the df/dr term

This follows dedalus pattern: lap(u) + a*lift(tau_1,-1) + b*lift(tau_2,-2) = RHS
where the boundary condition becomes: a*u + b*du/dr = c at the boundary.

# Mathematical Background
The Robin BC a⋅f + b⋅∂f/∂r = c at r=R is enforced by:
1. Adding lift terms to PDE: PDE + a⋅Lift(tau_1,-1) + b⋅Lift(tau_2,-2) = RHS  
2. The lift weights are: 
   - Lift(tau_1,-1): lifts to highest radial mode with weight Z_n^l(R) = 1
   - Lift(tau_2,-2): lifts to second-highest mode with weight ∂Z_n^l/∂r|_R
3. Boundary constraint: a⋅f(R) + b⋅∂f/∂r|_R = c determines tau coefficients
"""
function apply_robin_lift!(spectral_data::Vector{Complex{T}}, 
                          tau_coeffs_value::Vector{Complex{T}}, tau_coeffs_deriv::Vector{Complex{T}},
                          value_lift_matrix::SparseMatrixCSC{Complex{T}, Int}, 
                          deriv_lift_matrix::SparseMatrixCSC{Complex{T}, Int},
                          a_coeff::T, b_coeff::T) where T<:Real
    
    # Apply value term: a * Lift(tau_value, -1) 
    # This lifts to highest radial mode (k=-1)
    value_contribution = Complex{T}(a_coeff) * (value_lift_matrix * tau_coeffs_value)
    spectral_data .+= value_contribution
    
    # Apply derivative term: b * Lift(tau_deriv, -2)
    # This lifts to second-highest radial mode (k=-2) 
    deriv_contribution = Complex{T}(b_coeff) * (deriv_lift_matrix * tau_coeffs_deriv)
    spectral_data .+= deriv_contribution
end

# Simplified version for backward compatibility (single tau variable)
function apply_robin_lift!(spectral_data::Vector{Complex{T}}, tau_coeffs::Vector{Complex{T}},
                          lift_matrix::SparseMatrixCSC{Complex{T}, Int},
                          a_coeff::T, b_coeff::T) where T<:Real
    
    @warn "Using simplified Robin lift - full implementation requires separate value and derivative tau terms"
    
    # Simplified Robin BC: a*u + b*du/dr = g at boundary
    # For single tau approximation, we assume the lift matrix captures the appropriate
    # boundary mode behavior that combines value and derivative effects
    
    # Mathematical approach: The Robin BC a*u + b*∂u/∂r = g becomes
    # a linear combination of tau contributions at the boundary
    # 
    # Following Dedalus approach: The lift operation should preserve the
    # linear combination a*coeff + b*coeff for proper boundary enforcement
    
    # Compute effective weight based on Robin BC coefficients
    # This represents the combined effect: (a + b*∂/∂r) applied to lifted tau
    if abs(a_coeff) > eps(T) || abs(b_coeff) > eps(T)
        # For Robin BC: the effective weight is the linear combination of coefficients
        # scaled by the characteristic boundary operator norm
        boundary_scale = sqrt(a_coeff^2 + b_coeff^2)  # L2 norm of Robin operator
        
        # Normalize to prevent numerical issues while preserving the Robin balance
        if boundary_scale > eps(T)
            # Weighted combination preserving Robin BC structure: a + b*(boundary_gradient_scale)
            # For spherical coordinates at r=R, the gradient scale is typically O(1/R)
            # We use a unit scale as approximation for the simplified version
            gradient_scale = T(1)  # Simplified: should be related to 1/R or basis derivative scaling
            combined_weight = Complex{T}(a_coeff + b_coeff * gradient_scale)
        else
            combined_weight = Complex{T}(1)  # Fallback
        end
    else
        combined_weight = Complex{T}(0)  # Both coefficients zero
    end
    
    # Apply lift with Robin-weighted tau contribution
    # Note: This is still an approximation - proper Robin BC needs two tau variables
    lifted_contribution = combined_weight * (lift_matrix * tau_coeffs)
    spectral_data .+= lifted_contribution
end

"""
    apply_regularity_lift!(spectral_data, tau_coeffs, lift_matrix)

Apply regularity lift operation: field += Lift(tau) for regularity enforcement.

Uses specialized lift matrix for center/pole regularity conditions.
"""
function apply_regularity_lift!(spectral_data::Vector{Complex{T}}, tau_coeffs::Vector{Complex{T}},
                               lift_matrix::SparseMatrixCSC{Complex{T}, Int}) where T<:Real
    # Regularity lift ensures proper behavior at singular points
    lifted_contribution = lift_matrix * tau_coeffs
    spectral_data .+= lifted_contribution
end

"""
    get_lift_mode_number(bc) -> Int

Determine lift mode number for given boundary condition type.

Following dedalus convention:
- Dirichlet: lift to k=-1 (highest radial mode)
- Neumann: lift to k=-2 (second highest radial mode)  
- Robin: lift to k=-1 (mixed condition)
- Regularity: lift to k=0 (special center/pole modes)
"""
function get_lift_mode_number(bc::SphericalBoundaryCondition{T}) where T<:Real
    if isa(bc, DirichletBC)
        return -1  # Highest radial mode
    elseif isa(bc, NeumannBC)
        return -2  # Second highest radial mode
    elseif isa(bc, RobinBC)
        return -1  # Mixed condition, use highest for primary tau (value term)
                   # Note: Robin BC actually requires TWO tau variables:
                   # tau_value (mode -1) and tau_deriv (mode -2)
    elseif isa(bc, RegularityBC)
        return 0   # Special regularity modes
    else
        error("Unknown boundary condition type: $(typeof(bc))")
    end
end

"""
    get_robin_lift_modes() -> Tuple{Int, Int}

Return both lift mode numbers for Robin boundary conditions.
Robin BC requires two tau variables with different lift modes.
"""
function get_robin_lift_modes()
    return (-1, -2)  # (value_mode, derivative_mode)
end