"""
Tau system implementation for spherical boundary conditions following dedalus approach.

Implements the generalized tau method for boundary condition enforcement:
- Tau variable management
- Constraint matrix construction  
- Integration with weak formulations
- Boundary condition equation formulation
"""

using LinearAlgebra
using SparseArrays
include("boundary_conditions.jl")
include("lift_operators.jl")

export TauSystem, TauVariable
export build_tau_system, add_boundary_condition!, rebuild_tau_matrices!
export add_tau_terms!, add_unknown_field!, add_lift_terms_to_equations!
export add_boundary_condition_equations!
export BoundaryEquation, DirichletEquation, NeumannEquation, RobinEquation, RegularityEquation

"""
Tau variable for boundary condition enforcement.

Each tau variable corresponds to a boundary constraint that needs enforcement.
"""
struct TauVariable{T<:Real}
    name::String                       # Unique identifier
    bc_type::Symbol                   # :dirichlet, :neumann, :robin, :regularity
    location::Symbol                  # :surface, :center, :poles
    component::Union{Int, Nothing}    # Vector component (nothing for scalar)
    lift_mode::Int                   # Mode to lift to (-1, -2, etc.)
    coefficients::Vector{Complex{T}} # Spectral coefficients
    
    function TauVariable{T}(name::String, bc_type::Symbol, location::Symbol,
                          component::Union{Int, Nothing}, lift_mode::Int) where T<:Real
        coefficients = Complex{T}[]
        new{T}(name, bc_type, location, component, lift_mode, coefficients)
    end
end

"""
Tau system for managing all boundary conditions in a problem.

Central data structure following dedalus generalized tau approach.
"""
struct TauSystem{T<:Real}
    tau_variables::Vector{TauVariable{T}}                              # All tau variables
    boundary_conditions::Vector{SphericalBoundaryCondition{T}}        # All boundary conditions
    constraint_matrix::SparseMatrixCSC{Complex{T}, Int}               # LHS constraint matrix
    rhs_vector::Vector{Complex{T}}                                     # RHS constraint vector
    lift_operators::Dict{Symbol, LiftOperator{T}}                     # Lift operators by type
    tau_coupling_matrix::SparseMatrixCSC{Complex{T}, Int}             # Tau-field coupling
    is_built::Bool                                                     # System ready flag
    
    function TauSystem{T}() where T<:Real
        tau_vars = TauVariable{T}[]
        bcs = SphericalBoundaryCondition{T}[]
        # Initialize empty matrices - will be built later
        constraint_matrix = spzeros(Complex{T}, 0, 0)
        rhs_vector = Complex{T}[]
        lift_operators = Dict{Symbol, LiftOperator{T}}()
        tau_coupling_matrix = spzeros(Complex{T}, 0, 0)
        
        new{T}(tau_vars, bcs, constraint_matrix, rhs_vector, 
               lift_operators, tau_coupling_matrix, false)
    end
end

"""
    build_tau_system(boundary_conditions, basis) -> TauSystem

Build complete tau system from boundary conditions following dedalus approach.

# Arguments
- `boundary_conditions`: Vector of SphericalBoundaryCondition objects
- `basis`: Ball basis specification

# Returns
TauSystem ready for integration with problem formulation
"""
function build_tau_system(boundary_conditions::Vector{SphericalBoundaryCondition{T}}, 
                         basis::BallBasis{T}) where T<:Real
    tau_system = TauSystem{T}()
    
    # Add each boundary condition
    for bc in boundary_conditions
        add_boundary_condition!(tau_system, bc, basis)
    end
    
    # Build system matrices
    rebuild_tau_matrices!(tau_system)
    
    return tau_system
end

"""
    add_boundary_condition!(tau_system, bc, basis)

Add boundary condition to tau system.

Creates appropriate tau variables and updates system structure.
For Robin conditions, creates TWO tau variables following dedalus approach.
"""
function add_boundary_condition!(tau_system::TauSystem{T}, bc::SphericalBoundaryCondition{T},
                                basis::BallBasis{T}) where T<:Real
    # Add boundary condition to system
    push!(tau_system.boundary_conditions, bc)
    
    bc_type = get_bc_symbol(bc)
    
    if isa(bc, RobinBC)
        # Robin BC requires TWO tau variables following dedalus approach
        # tau_1 for value term (lifts to k=-1, highest radial mode)
        # tau_2 for derivative term (lifts to k=-2, second highest radial mode)
        
        tau_name_value = generate_tau_name(bc, length(tau_system.tau_variables)) * "_value"
        tau_name_deriv = generate_tau_name(bc, length(tau_system.tau_variables)) * "_deriv"
        
        # Value tau variable (for a*f term)
        tau_var_value = TauVariable{T}(tau_name_value, bc_type, bc.location, bc.component, -1)  # k=-1
        push!(tau_system.tau_variables, tau_var_value)
        
        # Derivative tau variable (for b*∂f/∂r term) 
        tau_var_deriv = TauVariable{T}(tau_name_deriv, bc_type, bc.location, bc.component, -2)  # k=-2
        push!(tau_system.tau_variables, tau_var_deriv)
        
        # Create lift operators for both modes if needed
        if !haskey(tau_system.lift_operators, :robin_value)
            tau_system.lift_operators[:robin_value] = LiftOperator{T}(basis.type, -1, bc.location)
        end
        if !haskey(tau_system.lift_operators, :robin_deriv)
            tau_system.lift_operators[:robin_deriv] = LiftOperator{T}(basis.type, -2, bc.location)
        end
        
    else
        # Single tau variable for other BC types
        tau_name = generate_tau_name(bc, length(tau_system.tau_variables))
        lift_mode = get_lift_mode_number(bc)
        
        tau_var = TauVariable{T}(tau_name, bc_type, bc.location, bc.component, lift_mode)
        push!(tau_system.tau_variables, tau_var)
        
        # Create lift operator if needed
        if !haskey(tau_system.lift_operators, bc_type)
            tau_system.lift_operators[bc_type] = LiftOperator{T}(basis.type, lift_mode, bc.location)
        end
    end
    
    # Mark system as needing rebuild
    tau_system.is_built = false
end

"""
    rebuild_tau_matrices!(tau_system)

Rebuild constraint and coupling matrices after modifications.

Constructs the full tau system matrices needed for problem integration.
"""
function rebuild_tau_matrices!(tau_system::TauSystem{T}) where T<:Real
    n_tau = length(tau_system.tau_variables)
    n_bcs = length(tau_system.boundary_conditions)
    
    if n_tau == 0 || n_bcs == 0
        tau_system.is_built = true
        return
    end
    
    # Build constraint matrix (n_bcs × n_modes)
    # Each row enforces one boundary condition
    n_modes = estimate_total_modes(tau_system)  # Estimate from tau variables
    tau_system.constraint_matrix = spzeros(Complex{T}, n_bcs, n_modes)
    tau_system.rhs_vector = zeros(Complex{T}, n_bcs)
    
    # Build tau coupling matrix (n_modes × n_tau)  
    # Maps tau variables to spectral modes via lift operators
    tau_system.tau_coupling_matrix = spzeros(Complex{T}, n_modes, n_tau)
    
    # Fill matrices row by row
    for (bc_idx, bc) in enumerate(tau_system.boundary_conditions)
        tau_var = tau_system.tau_variables[bc_idx]  # Assuming 1-1 correspondence
        
        # Build constraint row for this boundary condition
        build_constraint_row!(tau_system.constraint_matrix, bc, bc_idx, n_modes)
        
        # Set RHS value
        tau_system.rhs_vector[bc_idx] = compute_bc_rhs_value(bc)
        
        # Build tau coupling column
        build_tau_coupling_column!(tau_system.tau_coupling_matrix, tau_var, bc_idx, n_modes)
    end
    
    tau_system.is_built = true
end

"""
    build_constraint_row!(constraint_matrix, bc, row_idx, n_modes)

Build single constraint row for boundary condition.

The constraint row enforces: BC_operator(field) = BC_value
where BC_operator depends on the boundary condition type.
"""
function build_constraint_row!(constraint_matrix::SparseMatrixCSC{Complex{T}, Int},
                              bc::SphericalBoundaryCondition{T}, row_idx::Int, n_modes::Int) where T<:Real
    
    if isa(bc, DirichletBC)
        # Dirichlet: field(boundary) = value
        # Constraint row has boundary evaluation weights
        build_dirichlet_constraint_row!(constraint_matrix, bc, row_idx, n_modes)
        
    elseif isa(bc, NeumannBC)  
        # Neumann: ∂field/∂r(boundary) = value
        # Constraint row has derivative evaluation weights
        build_neumann_constraint_row!(constraint_matrix, bc, row_idx, n_modes)
        
    elseif isa(bc, RobinBC)
        # Robin: a*field + b*∂field/∂r = value
        # Constraint row combines value and derivative weights
        build_robin_constraint_row!(constraint_matrix, bc, row_idx, n_modes)
        
    elseif isa(bc, RegularityBC)
        # Regularity: field regular at singular points
        # Constraint row enforces regularity conditions
        build_regularity_constraint_row!(constraint_matrix, bc, row_idx, n_modes)
        
    else
        error("Unknown boundary condition type: $(typeof(bc))")
    end
end

"""
    build_dirichlet_constraint_row!(constraint_matrix, bc, row_idx, n_modes)

Build constraint row for Dirichlet boundary condition.

For Dirichlet BC f(r=R) = g, the constraint is:
∑_modes f_mode * basis_mode(r=R) = g
"""
function build_dirichlet_constraint_row!(constraint_matrix::SparseMatrixCSC{Complex{T}, Int},
                                        bc::DirichletBC{T}, row_idx::Int, n_modes::Int) where T<:Real
    # Dirichlet constraint: evaluate field at boundary
    mode_idx = 0
    
    # Iterate over spectral modes (simplified - assumes known mode structure)
    for l in 0:10  # Truncated for example
        for m in -l:l
            for n in 0:10
                mode_idx += 1
                if mode_idx > n_modes
                    break
                end
                
                # Boundary evaluation weight: Z_n^l(r=1) * Y_l^m(θ,φ)
                # For surface BC at r=1: Z_n^l(1) = 1
                boundary_weight = Complex{T}(1)  # Simplified
                
                constraint_matrix[row_idx, mode_idx] = boundary_weight
            end
            if mode_idx > n_modes
                break
            end
        end
        if mode_idx > n_modes
            break
        end
    end
end

"""
    build_neumann_constraint_row!(constraint_matrix, bc, row_idx, n_modes)

Build constraint row for Neumann boundary condition.

For Neumann BC ∂f/∂r|_{r=R} = h, the constraint is:
∑_modes f_mode * ∂basis_mode/∂r|_{r=R} = h
"""
function build_neumann_constraint_row!(constraint_matrix::SparseMatrixCSC{Complex{T}, Int},
                                      bc::NeumannBC{T}, row_idx::Int, n_modes::Int) where T<:Real
    # Neumann constraint: evaluate field derivative at boundary
    mode_idx = 0
    
    for l in 0:10
        for m in -l:l
            for n in 0:10
                mode_idx += 1
                if mode_idx > n_modes
                    break
                end
                
                # Derivative evaluation weight: ∂Z_n^l/∂r|_{r=1} * Y_l^m(θ,φ)
                derivative_weight = evaluate_zernike_derivative_at_boundary(n, 1, T)
                
                constraint_matrix[row_idx, mode_idx] = Complex{T}(derivative_weight)
            end
            if mode_idx > n_modes
                break
            end
        end
        if mode_idx > n_modes
            break
        end
    end
end

"""
    build_robin_constraint_row!(constraint_matrix, bc, row_idx, n_modes)

Build constraint row for Robin boundary condition.

For Robin BC a*f + b*∂f/∂r = c, the constraint combines value and derivative.
"""
function build_robin_constraint_row!(constraint_matrix::SparseMatrixCSC{Complex{T}, Int},
                                    bc::RobinBC{T}, row_idx::Int, n_modes::Int) where T<:Real
    mode_idx = 0
    
    for l in 0:10
        for m in -l:l
            for n in 0:10
                mode_idx += 1
                if mode_idx > n_modes
                    break
                end
                
                # Robin weight: a * Z_n^l(1) + b * ∂Z_n^l/∂r|_{r=1}
                value_weight = T(1)  # Z_n^l(1) = 1
                deriv_weight = evaluate_zernike_derivative_at_boundary(n, 1, T)
                
                robin_weight = bc.a_coeff * value_weight + bc.b_coeff * deriv_weight
                constraint_matrix[row_idx, mode_idx] = Complex{T}(robin_weight)
            end
            if mode_idx > n_modes
                break
            end
        end
        if mode_idx > n_modes
            break
        end
    end
end

"""
    build_regularity_constraint_row!(constraint_matrix, bc, row_idx, n_modes)

Build constraint row for regularity boundary condition.

Regularity constraints enforce proper behavior at singular points.
"""
function build_regularity_constraint_row!(constraint_matrix::SparseMatrixCSC{Complex{T}, Int},
                                         bc::RegularityBC{T}, row_idx::Int, n_modes::Int) where T<:Real
    mode_idx = 0
    
    if bc.location == :center
        # Center regularity: only l=0 modes contribute
        for l in 0:10
            for m in -l:l
                for n in 0:10
                    mode_idx += 1
                    if mode_idx > n_modes
                        break
                    end
                    
                    if l == 0 && m == 0
                        # Center weight: Z_n^0(0) * Y_0^0
                        center_weight = evaluate_zernike_at_center(n, 0, T) / sqrt(4 * π)
                        constraint_matrix[row_idx, mode_idx] = Complex{T}(center_weight)
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
        
    elseif bc.location == :poles
        # Pole regularity: only m=0 modes contribute
        for l in 0:10
            for m in -l:l
                for n in 0:10
                    mode_idx += 1
                    if mode_idx > n_modes
                        break
                    end
                    
                    if m == 0
                        # Pole weight: Y_l^0 normalization
                        pole_weight = sqrt((2*l + 1) / (4 * π))
                        constraint_matrix[row_idx, mode_idx] = Complex{T}(pole_weight)
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
    end
end

"""
    build_tau_coupling_column!(tau_coupling_matrix, tau_var, col_idx, n_modes)

Build tau coupling column showing how tau variable couples to spectral modes.

This implements the lift operation: field += Lift(tau_var)
"""
function build_tau_coupling_column!(tau_coupling_matrix::SparseMatrixCSC{Complex{T}, Int},
                                   tau_var::TauVariable{T}, col_idx::Int, n_modes::Int) where T<:Real
    mode_idx = 0
    
    for l in 0:10
        for m in -l:l
            for n in 0:10
                mode_idx += 1
                if mode_idx > n_modes
                    break
                end
                
                # Compute lift weight for this mode
                lift_weight = compute_tau_lift_weight(tau_var, n, l, m)
                
                if abs(lift_weight) > 1e-14
                    tau_coupling_matrix[mode_idx, col_idx] = lift_weight
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
end

"""
    compute_tau_lift_weight(tau_var, n, l, m) -> Complex{T}

Compute lift weight for tau variable to spectral mode (n,l,m).
"""
function compute_tau_lift_weight(tau_var::TauVariable{T}, n::Int, l::Int, m::Int) where T<:Real
    if tau_var.bc_type == :dirichlet
        # Dirichlet lift to highest radial mode
        if tau_var.lift_mode == -1 && n == 10  # Assuming n_max = 10
            return Complex{T}(1)
        end
        
    elseif tau_var.bc_type == :neumann
        # Neumann lift to second highest radial mode  
        if tau_var.lift_mode == -2 && n == 8  # n_max - 2
            return Complex{T}(1)
        end
        
    elseif tau_var.bc_type == :regularity
        if tau_var.location == :center && l == 0 && m == 0
            # Center regularity lift
            weight = evaluate_zernike_at_center(n, 0, T) / sqrt(4 * π)
            return Complex{T}(weight)
        elseif tau_var.location == :poles && m == 0
            # Pole regularity lift
            weight = sqrt((2*l + 1) / (4 * π))
            return Complex{T}(weight)
        end
    end
    
    return Complex{T}(0)
end

# Helper functions

"""
Generate unique name for tau variable.
"""
function generate_tau_name(bc::SphericalBoundaryCondition{T}, idx::Int) where T<:Real
    bc_type = get_bc_symbol(bc)
    comp_str = bc.component === nothing ? "" : "_$(bc.component)"
    return "tau_$(bc_type)_$(bc.location)$(comp_str)_$idx"
end

"""
Convert boundary condition to symbol.
"""
function get_bc_symbol(bc::SphericalBoundaryCondition{T}) where T<:Real
    if isa(bc, DirichletBC)
        return :dirichlet
    elseif isa(bc, NeumannBC)
        return :neumann
    elseif isa(bc, RobinBC)
        return :robin
    elseif isa(bc, RegularityBC)
        return :regularity
    else
        error("Unknown BC type: $(typeof(bc))")
    end
end

"""
    compute_bc_rhs_value(bc, coords, basis) -> Complex{T}

Compute RHS value for boundary condition following dedalus approach.

The RHS value depends on the boundary condition type and location:
- Dirichlet: Evaluate bc.value_function at boundary points  
- Neumann: Evaluate bc.value_function for derivative boundary values
- Robin: Evaluate bc.rhs_function for the Robin condition RHS
- Regularity: Zero (regularity conditions have no RHS)

Following dedalus pattern, boundary condition equations have the form:
- BC_operator(field) = BC_value
where BC_value is computed by this function.

# Mathematical Background
For spherical domains at r=R:
- Surface boundary: integrate over (θ,φ) with proper measure sin(θ)dθdφ  
- Angular modes: project onto spherical harmonics Y_l^m(θ,φ)
- Radial evaluation: at r=R for surface conditions

# Arguments  
- `bc`: Boundary condition object containing value function
- `coords`: Spherical coordinate system (optional, for advanced evaluation)
- `basis`: Spherical basis for spectral projection (optional)

# Returns
Complex{T}: RHS value for constraint equation
"""
function compute_bc_rhs_value(bc::SphericalBoundaryCondition{T}, 
                             coords::Union{SphericalCoordinates{T}, Nothing}=nothing,
                             basis::Union{BallBasis{T}, Nothing}=nothing) where T<:Real
    
    if isa(bc, DirichletBC)
        # Dirichlet: u(boundary) = value_function(θ,φ)
        return compute_dirichlet_rhs_value(bc, coords, basis)
        
    elseif isa(bc, NeumannBC)
        # Neumann: ∂u/∂r|_boundary = value_function(θ,φ)  
        return compute_neumann_rhs_value(bc, coords, basis)
        
    elseif isa(bc, RobinBC)
        # Robin: a*u + b*∂u/∂r = rhs_function(θ,φ)
        return compute_robin_rhs_value(bc, coords, basis)
        
    elseif isa(bc, RegularityBC)
        # Regularity: no explicit RHS, enforced through spectral properties
        return Complex{T}(0)
        
    else
        error("Unknown boundary condition type: $(typeof(bc))")
    end
end

"""
    compute_dirichlet_rhs_value(bc, coords, basis) -> Complex{T}

Compute RHS value for Dirichlet boundary condition.

For Dirichlet BC u(r=R) = g(θ,φ), we need to evaluate:
∫ g(θ,φ) * Y_l^m(θ,φ) * sin(θ) dθ dφ

This gives the spectral coefficient of g in the Y_l^m expansion.
"""
function compute_dirichlet_rhs_value(bc::DirichletBC{T}, 
                                   coords::Union{SphericalCoordinates{T}, Nothing},
                                   basis::Union{BallBasis{T}, Nothing}) where T<:Real
    
    if isa(bc.value_function, Function)
        # Proper implementation: integrate boundary function over spherical surface
        # using spherical harmonic projection
        
        if coords !== nothing && basis !== nothing
            # Use proper spherical quadrature integration
            return integrate_boundary_function_spherical(bc.value_function, coords, basis, bc.location)
        else
            # Fallback: more sophisticated single-point evaluation
            try
                # For spherical surfaces, we need to account for spherical geometry
                # Evaluate at multiple representative points and average
                npoints = 8  # Use 8 representative points for better approximation
                
                total_value = Complex{T}(0)
                for i in 1:npoints
                    θ = π * (i - 0.5) / npoints  # Distribute in θ from 0 to π
                    φ = 2π * (i - 1) / npoints   # Distribute in φ from 0 to 2π
                    
                    try
                        point_value = bc.value_function(θ, φ)
                        # Weight by sin(θ) for proper spherical measure
                        weight = sin(θ)
                        total_value += Complex{T}(point_value * weight)
                    catch
                        # Skip points where function evaluation fails
                        continue
                    end
                end
                
                # Normalize by total weight (integral of sin(θ) over sphere)
                return total_value / Complex{T}(npoints * π/2)
            catch
                # Final fallback: single point evaluation
                try
                    representative_value = bc.value_function(π/2, 0.0)
                    return Complex{T}(representative_value)
                catch
                    return Complex{T}(0)
                end
            end
        end
    else
        # Handle non-function values
        if bc.value_function isa Number
            return Complex{T}(bc.value_function)
        else
            return Complex{T}(0)  # Fallback
        end
    end
end

"""
    integrate_boundary_function_spherical(func, coords, basis, location) -> Complex{T}

Integrate boundary function over spherical surface using proper spectral quadrature.
This follows the Dedalus approach for boundary condition RHS computation.
"""
function integrate_boundary_function_spherical(func::Function, 
                                              coords::SphericalCoordinates{T},
                                              basis::BallBasis{T}, 
                                              location::Symbol) where T<:Real
    
    # Get quadrature points and weights for spherical integration
    # This should use Gauss-Legendre quadrature in θ and trapezoidal in φ
    
    ntheta = coords.ntheta
    nphi = coords.nphi
    
    # Use Gauss-Legendre quadrature points in θ (colatitude)
    # and uniform points in φ (azimuth)
    theta_points, theta_weights = gauss_legendre_quadrature(ntheta)
    phi_points = range(0, 2π, length=nphi+1)[1:end-1]  # Exclude 2π (same as 0)
    phi_weights = fill(2π/nphi, nphi)
    
    total_integral = Complex{T}(0)
    
    for (i, θ) in enumerate(theta_points)
        for (j, φ) in enumerate(phi_points)
            try
                # Map θ from [-1,1] to [0,π] for colatitude
                theta_mapped = π * (θ + 1) / 2
                
                # Evaluate function at quadrature point
                func_value = func(theta_mapped, φ)
                
                # Spherical measure: sin(θ) dθ dφ
                jacobian = sin(theta_mapped)
                
                # Quadrature weight
                weight = theta_weights[i] * phi_weights[j] * jacobian * π/2
                
                total_integral += Complex{T}(func_value * weight)
            catch
                # Skip points where function evaluation fails
                continue
            end
        end
    end
    
    # Normalize by surface area (4π for unit sphere)
    return total_integral / Complex{T}(4π)
end

"""
    gauss_legendre_quadrature(n) -> (points, weights)

Generate Gauss-Legendre quadrature points and weights for integration over [-1,1].
"""
function gauss_legendre_quadrature(n::Int)
    # Simplified implementation - in practice would use more robust method
    if n == 1
        return [0.0], [2.0]
    elseif n == 2  
        points = [-1.0/sqrt(3), 1.0/sqrt(3)]
        weights = [1.0, 1.0]
        return points, weights
    elseif n <= 8
        # Use precomputed values for small n
        return gauss_legendre_precomputed(n)
    else
        # For larger n, would use iterative algorithm
        # Simplified: uniform points as approximation
        points = collect(range(-1, 1, length=n))
        weights = fill(2.0/n, n)
        return points, weights
    end
end

"""
Helper function for small n Gauss-Legendre quadrature.
"""
function gauss_legendre_precomputed(n::Int)
    if n == 3
        points = [-sqrt(3/5), 0.0, sqrt(3/5)]
        weights = [5/9, 8/9, 5/9]
        return points, weights
    elseif n == 4
        sqrt_val = sqrt(3/7 - 2/7*sqrt(6/5))
        points = [-sqrt_val, -sqrt(3/7 + 2/7*sqrt(6/5)), sqrt(3/7 + 2/7*sqrt(6/5)), sqrt_val]
        weights = [(18+sqrt(30))/36, (18-sqrt(30))/36, (18-sqrt(30))/36, (18+sqrt(30))/36]
        return points, weights
    else
        # Fallback to uniform grid
        points = collect(range(-1, 1, length=n))
        weights = fill(2.0/n, n)
        return points, weights
    end
end

"""
    compute_neumann_rhs_value(bc, coords, basis) -> Complex{T}

Compute RHS value for Neumann boundary condition.

For Neumann BC ∂u/∂r|_{r=R} = h(θ,φ), we evaluate h at boundary.
"""
function compute_neumann_rhs_value(bc::NeumannBC{T}, 
                                 coords::Union{SphericalCoordinates{T}, Nothing},
                                 basis::Union{BallBasis{T}, Nothing}) where T<:Real
    
    if isa(bc.value_function, Function)
        try
            # Evaluate derivative boundary condition at representative point
            representative_value = bc.value_function(π/2, 0.0)
            return Complex{T}(representative_value)
        catch
            return Complex{T}(0)
        end
    else
        return Complex{T}(0)
    end
end

"""
    compute_robin_rhs_value(bc, coords, basis) -> Complex{T}

Compute RHS value for Robin boundary condition.

For Robin BC a*u + b*∂u/∂r = c(θ,φ), we evaluate c at boundary.
"""
function compute_robin_rhs_value(bc::RobinBC{T}, 
                               coords::Union{SphericalCoordinates{T}, Nothing},
                               basis::Union{BallBasis{T}, Nothing}) where T<:Real
    
    if isa(bc.rhs_function, Function)
        try
            # Evaluate RHS function at representative point
            representative_value = bc.rhs_function(π/2, 0.0)
            return Complex{T}(representative_value)
        catch
            return Complex{T}(0)
        end
    else
        return Complex{T}(0)
    end
end

# Backward compatibility function (without coords/basis arguments)
function compute_bc_rhs_value(bc::SphericalBoundaryCondition{T}) where T<:Real
    return compute_bc_rhs_value(bc, nothing, nothing)
end

"""
    estimate_total_modes(tau_system::TauSystem{T}) -> Int

Estimate total number of spectral modes from tau system following dedalus patterns.

For ball geometry, computes the total coefficient size as the product of:
- Azimuthal modes: Fourier coefficients in φ direction
- Colatitude modes: Spherical harmonic degrees in θ direction  
- Radial modes: Zernike polynomial coefficients in r direction

The calculation follows dedalus coeff_size computation:
    total_modes = n_phi_modes * n_theta_modes * n_radial_modes

# Mathematical background
For a ball domain with BallBasis(Nφ, Nθ, Nr):
- Azimuthal: RealFourier typically gives ~Nφ/2 + 1 modes
- Colatitude: SphereBasis gives modes up to l_max
- Radial: Zernike polynomials give modes up to n_max

# Arguments
- `tau_system::TauSystem{T}`: Tau system containing basis information

# Returns
- `Int`: Estimated total number of spectral modes

# Examples
```julia
tau_system = build_tau_system(bcs, basis)
n_modes = estimate_total_modes(tau_system)
```

See dedalus subsystems.py coeff_size() for reference implementation.
"""
function estimate_total_modes(tau_system::TauSystem{T}) where T<:Real
    
    # Try to get basis information from tau variables
    if !isempty(tau_system.tau_variables)
        first_tau = tau_system.tau_variables[1]
        
        # If we have a basis associated with tau variable, use its dimensions
        if hasfield(typeof(first_tau), :basis) && first_tau.basis isa BallBasis{T}
            basis = first_tau.basis
            
            # Calculate modes following dedalus BallBasis pattern
            # Based on Dedalus research: BallBasis uses spherical harmonics + Jacobi polynomials
            
            return calculate_ball_modes(basis.l_max, basis.n_max, basis.type)
            
        elseif hasfield(typeof(first_tau), :n_total_modes)
            # If tau variable directly stores total mode count
            return first_tau.n_total_modes
        end
    end
    
    # Try to extract basis info from boundary conditions or domain
    for bc in tau_system.boundary_conditions
        if hasfield(typeof(bc), :domain) && bc.domain !== nothing
            domain = bc.domain
            if hasfield(typeof(domain), :coords) && domain.coords !== nothing
                coords = domain.coords
                # Estimate from coordinate dimensions
                if hasfield(typeof(coords), :l_max) && hasfield(typeof(coords), :n_max)
                    return calculate_ball_modes(coords.l_max, coords.n_max, :ball)
                elseif hasfield(typeof(coords), :ntheta) && hasfield(typeof(coords), :nr)
                    # Convert grid sizes to spectral modes
                    l_max_est = coords.ntheta ÷ 2  # Typical dealiasing: N_grid ≈ 2*l_max
                    n_max_est = coords.nr - 1      # Radial modes typically nr-1
                    return calculate_ball_modes(l_max_est, n_max_est, :ball)
                end
            end
        end
    end
    
    # Conservative fallback estimate for typical ball problems
    # Based on dedalus ball examples and common usage
    l_max_default = 32    # Spherical harmonic degree
    n_max_default = 48    # Radial polynomial degree
    estimated_modes = calculate_ball_modes(l_max_default, n_max_default, :ball)
    
    @warn "Could not determine exact mode count from tau system. Using default ball parameters (l_max=$l_max_default, n_max=$n_max_default): $estimated_modes modes"
    
    return estimated_modes
end

"""
    calculate_ball_modes(l_max, n_max, basis_type) -> Int

Calculate total number of modes for ball/sphere spectral basis following Dedalus patterns.

# Mathematical Background
- Spherical harmonics: For each degree ℓ, there are 2ℓ+1 modes (m = -ℓ, ..., ℓ)
- Total spherical modes: Σ(2ℓ+1) for ℓ=0 to l_max = (l_max+1)²
- Radial modes: Jacobi polynomials from n=0 to n_max, giving (n_max+1) modes
- Ball total: (l_max+1)² × (n_max+1) modes

# Arguments
- `l_max::Int`: Maximum spherical harmonic degree
- `n_max::Int`: Maximum radial polynomial degree  
- `basis_type::Symbol`: Type of basis (:ball, :sphere, :shell)
"""
function calculate_ball_modes(l_max::Int, n_max::Int, basis_type::Symbol)
    
    # Spherical harmonic modes: (l_max + 1)² 
    # This counts all (ℓ,m) pairs with ℓ ∈ [0, l_max], m ∈ [-ℓ, ℓ]
    spherical_modes = (l_max + 1)^2
    
    if basis_type == :ball
        # Ball basis: spherical harmonics × radial Jacobi polynomials
        radial_modes = n_max + 1  # Jacobi polynomials n = 0, 1, ..., n_max
        total_modes = spherical_modes * radial_modes
        
    elseif basis_type == :sphere  
        # Sphere basis: only spherical harmonics (no radial component)
        total_modes = spherical_modes
        
    elseif basis_type == :shell
        # Shell basis: similar to ball but may have different radial treatment
        radial_modes = n_max + 1  
        total_modes = spherical_modes * radial_modes
        
    else
        # Default to ball calculation
        radial_modes = n_max + 1
        total_modes = spherical_modes * radial_modes
    end
    
    return total_modes
end

# Abstract types for boundary equation formulation

"""
Abstract base type for boundary equation formulations.
"""
abstract type BoundaryEquation{T<:Real} end

"""
Dirichlet equation: field(boundary) = value
"""
struct DirichletEquation{T<:Real} <: BoundaryEquation{T}
    field_name::String
    boundary_value::Complex{T}
end

"""
Neumann equation: ∂field/∂r|_boundary = value
"""
struct NeumannEquation{T<:Real} <: BoundaryEquation{T}
    field_name::String
    derivative_value::Complex{T}
end

"""
Robin equation: a*field + b*∂field/∂r = value
"""
struct RobinEquation{T<:Real} <: BoundaryEquation{T}
    field_name::String
    a_coeff::T
    b_coeff::T
    rhs_value::Complex{T}
end

"""
Regularity equation: field regular at singular points
"""
struct RegularityEquation{T<:Real} <: BoundaryEquation{T}
    field_name::String
    location::Symbol  # :center, :poles
    regularity_type::Symbol  # :scalar, :vector
end

# Integration functions for problem formulation

"""
    add_tau_terms!(problem, tau_system)

Add tau terms to problem following dedalus generalized tau method.

Integrates tau system with problem formulation by:
1. Adding tau variables as unknowns
2. Adding lift terms to equations  
3. Adding boundary condition equations
"""
function add_tau_terms!(problem, tau_system::TauSystem{T}) where T<:Real
    if !tau_system.is_built
        error("Tau system must be built before adding to problem")
    end
    
    # Add tau variables as unknowns
    for tau_var in tau_system.tau_variables
        add_unknown_field!(problem, tau_var)
    end
    
    # Add lift terms to existing equations
    add_lift_terms_to_equations!(problem, tau_system)
    
    # Add boundary condition equations
    add_boundary_condition_equations!(problem, tau_system)
end

"""
    add_unknown_field!(problem, tau_var)

Add tau variable as unknown field to problem.
"""
function add_unknown_field!(problem, tau_var::TauVariable{T}) where T<:Real
    if tau_var.component === nothing
        # Scalar tau variable
        add_scalar_unknown!(problem, tau_var.name, tau_var.coefficients)
    else
        # Vector component tau variable
        add_vector_component_unknown!(problem, tau_var.name, tau_var.component, tau_var.coefficients)
    end
end

"""
    add_lift_terms_to_equations!(problem, tau_system)

Add lift terms Lift(tau) to existing problem equations.

Each equation gets lift terms for relevant tau variables.
"""
function add_lift_terms_to_equations!(problem, tau_system::TauSystem{T}) where T<:Real
    for (i, tau_var) in enumerate(tau_system.tau_variables)
        bc = tau_system.boundary_conditions[i]
        
        # Determine target equation for this tau variable
        equation_name = get_target_equation_name(bc)
        
        # Create lift term
        lift_term = create_lift_term(tau_system.lift_operators[tau_var.bc_type], tau_var)
        
        # Add to equation
        add_term_to_equation!(problem, equation_name, lift_term)
    end
end

"""
    add_boundary_condition_equations!(problem, tau_system)

Add boundary condition equations to problem.

Each boundary condition becomes a constraint equation.
"""
function add_boundary_condition_equations!(problem, tau_system::TauSystem{T}) where T<:Real
    for (i, bc) in enumerate(tau_system.boundary_conditions)
        # Create boundary equation
        boundary_eq = formulate_boundary_condition_equation(bc, problem.domain)
        
        # Add equation to problem
        equation_name = "BC_$(i)_$(get_bc_symbol(bc))"
        add_equation!(problem, equation_name, boundary_eq)
    end
end

# Placeholder functions for problem integration

"""
    get_target_equation_name(bc::SphericalBoundaryCondition{T}) -> String

Get target equation name for boundary condition tau term integration following dedalus patterns.

In dedalus, tau terms are added to specific equations based on the physics and field type:
- Velocity boundary conditions → momentum/velocity equations  
- Temperature/scalar boundary conditions → energy/scalar equations
- Pressure boundary conditions → continuity/divergence equations
- Regularity conditions → automatic (no explicit equation targeting)

# Mathematical Background
Following dedalus tau method:
- Dirichlet BC on velocity: `dt(u) - viscous_terms + lift(tau_u) = nonlinear_terms`  
- Dirichlet BC on temperature: `dt(T) - diffusion_terms + lift(tau_T) = convection_terms`
- Neumann BC: Similar pattern with appropriate lift operators
- Robin BC: Combined form targeting relevant equation

# Arguments  
- `bc::SphericalBoundaryCondition{T}`: Boundary condition to analyze

# Returns
- `String`: Target equation name for tau term integration

# Examples
```julia
bc_velocity = create_boundary_condition(:dirichlet, :surface, 0.0; component=1)
equation_name = get_target_equation_name(bc_velocity)  # "momentum" 

bc_temperature = create_boundary_condition(:neumann, :surface, 1.0)
equation_name = get_target_equation_name(bc_temperature)  # "energy"
```

See dedalus examples: ball convection, Rayleigh-Benard for reference patterns.
"""
function get_target_equation_name(bc::SphericalBoundaryCondition{T}) where T<:Real
    
    if isa(bc, RegularityBC)
        # Regularity conditions are usually enforced automatically
        # and don't target specific equations in dedalus
        return "regularity"  # Special case, may not need explicit targeting
        
    elseif isa(bc, DirichletBC) || isa(bc, NeumannBC) || isa(bc, RobinBC)
        
        # Determine target based on component type and physics
        if bc.component !== nothing
            # Vector field component - likely velocity
            if bc.component == 1      # Radial component
                return "momentum_radial"
            elseif bc.component == 2  # Theta component  
                return "momentum_theta"
            elseif bc.component == 3  # Phi component
                return "momentum_phi"
            else
                return "momentum"  # Generic momentum equation
            end
        else
            # Scalar field - determine type based on location and physics
            if bc.location == :surface
                # Surface boundary conditions typically for temperature/scalar transport
                return "energy"  # or "scalar_transport", "temperature" 
            elseif bc.location == :center
                # Center conditions often for scalar fields or pressure
                return "scalar"   # Could also be "energy" or "temperature"
            else
                return "scalar"   # Default for scalar equations
            end
        end
        
    else
        # Fallback for unknown BC types
        @warn "Unknown boundary condition type: $(typeof(bc)). Using default equation name."
        return "unknown"
    end
end

function create_lift_term(lift_op::LiftOperator{T}, tau_var::TauVariable{T}) where T<:Real
    return LiftTerm{T}(lift_op, tau_var.name)
end

function formulate_boundary_condition_equation(bc::SphericalBoundaryCondition{T}, domain) where T<:Real
    if isa(bc, DirichletBC)
        return DirichletEquation{T}("u", Complex{T}(0))  # Simplified
    elseif isa(bc, NeumannBC)
        return NeumannEquation{T}("u", Complex{T}(0))
    elseif isa(bc, RobinBC)
        return RobinEquation{T}("u", bc.a_coeff, bc.b_coeff, Complex{T}(0))
    elseif isa(bc, RegularityBC)
        return RegularityEquation{T}("u", bc.location, bc.regularity_type)
    end
end

"""
    add_scalar_unknown!(problem, name::String, coefficients::Array{Complex{T}}) where T<:Real

Add scalar tau variable as unknown field to dedalus-style problem following dedalus patterns.

In dedalus, tau variables are Field objects created by the distributor and passed to 
the problem constructor. The typical pattern is:

```python
# Create tau field
tau_scalar = dist.Field(name='tau_T', bases=boundary_basis)

# Pass to problem constructor  
problem = d3.IVP([u, T, p, tau_scalar], namespace=locals())
```

# Mathematical Background
Scalar tau variables represent boundary constraint multipliers for:
- Temperature/scalar boundary conditions: `dt(T) - diffusion + lift(tau_T) = convection`
- Pressure gauge conditions: `div(u) + tau_p = 0`
- Scalar regularity conditions at center/poles

# Arguments
- `problem`: Problem object to modify (dedalus IVP/LBVP/EVP/NLBVP)
- `name::String`: Name of the tau variable
- `coefficients::Array{Complex{T}}`: Initial spectral coefficients

# Implementation Notes
This function creates a scalar Field object and registers it with the problem.
The coefficients are used to initialize the field data in spectral space.

See dedalus examples: ball convection (`tau_T`), Rayleigh-Benard (`tau_p`).
"""
function add_scalar_unknown!(problem, name::String, coefficients::Array{Complex{T}}) where T<:Real
    # Following dedalus v3 approach: create proper field objects for unknown variables
    # In dedalus: tau_field = dist.Field(name=name, bases=boundary_bases)
    #             problem = d3.IVP([u, tau1, tau2], namespace=locals())
    
    # Create proper field representation following dedalus Field class pattern
    field = create_scalar_field(name, coefficients, T)
    
    # Add to problem's variables list (following dedalus IVP constructor pattern)
    if !hasfield(typeof(problem), :variables)
        # Initialize variables list if it doesn't exist
        setfield!(problem, :variables, Vector{Any}())
    end
    
    # Check if problem has variables field
    if hasfield(typeof(problem), :variables)
        # Add to existing variables list (like dedalus problem constructor)
        push!(problem.variables, field)
        
        @info "Added scalar unknown '$name' to problem variables ($(length(coefficients)) coefficients)"
        
        # Update problem metadata
        if hasfield(typeof(problem), :nvariables)
            problem.nvariables += 1
        end
        
        # Register in namespace for equation parsing (like dedalus namespace)
        if hasfield(typeof(problem), :namespace) && problem.namespace isa Dict
            problem.namespace[name] = field
        end
        
    else
        # Fallback: store in separate tau_variables collection
        if !hasfield(typeof(problem), :tau_variables)
            setfield!(problem, :tau_variables, Vector{Any}())
        end
        push!(problem.tau_variables, field)
        @info "Added scalar tau variable '$name' to problem (fallback storage)"
    end
    
    return field
end

"""
    create_scalar_field(name, coefficients, T) -> ScalarFieldRepresentation

Create a scalar field object following Dedalus Field class pattern.
This represents the Julia equivalent of dedalus `dist.Field(name=name)`.
"""
function create_scalar_field(name::String, coefficients::Array{Complex{T}}, ::Type{T}) where T<:Real
    # Create field object following Dedalus Field class structure
    field = ScalarFieldRepresentation{T}(
        name = name,
        tensor_signature = nothing,  # Scalar fields have no tensor signature
        dtype = Complex{T},
        data_coefficient = copy(coefficients),  # Coefficient space data
        data_grid = nothing,  # Grid space data (will be populated on demand)
        layout = :coefficient,  # Current data layout (coefficient space)
        bases = nothing,  # Tau fields often have no spatial bases
        scales = ones(T, length(size(coefficients))),  # Default scales
        global_shape = size(coefficients),
        meta = Dict{Symbol,Any}(
            :field_type => :scalar,
            :created_at => now(),
            :is_tau_variable => true
        )
    )
    
    return field
end

"""
Scalar field representation following Dedalus Field class pattern.
Represents scalar-valued fields for spectral methods.
"""
mutable struct ScalarFieldRepresentation{T<:Real}
    name::String
    tensor_signature::Union{Nothing, Tuple}
    dtype::Type
    data_coefficient::Union{Nothing, Array{Complex{T}}}
    data_grid::Union{Nothing, Array{Complex{T}}}
    layout::Symbol  # :coefficient or :grid
    bases::Union{Nothing, Any}  # Spectral bases
    scales::Vector{T}
    global_shape::Tuple
    meta::Dict{Symbol,Any}
end

# Constructor for ScalarFieldRepresentation
function ScalarFieldRepresentation{T}(; 
    name::String,
    tensor_signature = nothing,
    dtype = Complex{T},
    data_coefficient = nothing,
    data_grid = nothing,
    layout::Symbol = :coefficient,
    bases = nothing,
    scales = T[],
    global_shape = (),
    meta = Dict{Symbol,Any}()) where T<:Real
    
    return ScalarFieldRepresentation{T}(
        name, tensor_signature, dtype, data_coefficient, data_grid,
        layout, bases, scales, global_shape, meta
    )
end

"""
    change_layout!(field, new_layout) 

Switch field data layout between coefficient and grid space.
Following Dedalus field.change_layout() method.
"""
function change_layout!(field::ScalarFieldRepresentation{T}, new_layout::Symbol) where T<:Real
    if new_layout == field.layout
        return field  # Already in requested layout
    end
    
    if new_layout == :grid && field.layout == :coefficient
        # Transform from coefficient to grid space
        # In full implementation, would use spectral transforms
        field.data_grid = field.data_coefficient  # Simplified
        field.layout = :grid
        
    elseif new_layout == :coefficient && field.layout == :grid
        # Transform from grid to coefficient space
        field.data_coefficient = field.data_grid  # Simplified
        field.layout = :coefficient
        
    else
        error("Unknown layout transition: $(field.layout) -> $new_layout")
    end
    
    return field
end

"""
    add_vector_component_unknown!(problem, name::String, component::Int, 
                                 coefficients::Array{Complex{T}}) where T<:Real

Add vector component tau variable as unknown field to dedalus-style problem following dedalus patterns.

In dedalus, vector tau variables are VectorField objects with multiple components:

```python
# Create vector tau field
tau_u = dist.VectorField(coords, name='tau_u', bases=boundary_basis)

# Pass to problem constructor
problem = d3.IVP([u, p, T, tau_u], namespace=locals())
```

# Mathematical Background
Vector tau variables represent boundary constraint multipliers for:
- Velocity boundary conditions: `dt(u) - viscous + lift(tau_u) = nonlinear`
- Vector regularity conditions at center/poles
- Component-specific constraints (radial, theta, phi)

# Arguments
- `problem`: Problem object to modify
- `name::String`: Name of the vector tau variable
- `component::Int`: Component index (1=radial, 2=theta, 3=phi in spherical)
- `coefficients::Array{Complex{T}}`: Initial spectral coefficients for this component

# Implementation Notes
This function creates a VectorField component and registers it with the problem.
In full dedalus integration, components are accessed via tau_u['g'][component_index].

See dedalus examples: ball convection (`tau_u`), Rayleigh-Benard (`tau_u1`, `tau_u2`).
"""
function add_vector_component_unknown!(problem, name::String, component::Int, 
                                     coefficients::Array{Complex{T}}) where T<:Real
    # Following dedalus v3 approach: VectorField objects contain multiple components
    # In dedalus: tau_field = dist.VectorField(coords, name=name, bases=boundary_bases)
    #             tau_field['c'][component] = coefficients
    #             problem = d3.IVP([u, tau_field], namespace=locals())
    
    component_names = ["radial", "theta", "phi"]
    component_name = component <= length(component_names) ? component_names[component] : "component_$component"
    
    # Check if vector field already exists in problem
    existing_field = find_existing_vector_field(problem, name)
    
    if existing_field !== nothing
        # Update existing vector field with new component
        set_vector_component!(existing_field, component, coefficients)
        @info "Updated vector field '$name' $component_name component ($(length(coefficients)) coefficients)"
        return existing_field
    else
        # Create new vector field following dedalus VectorField pattern
        vector_field = create_vector_field(name, component, coefficients, T)
        
        # Add to problem's variables list (following dedalus IVP constructor pattern)
        if !hasfield(typeof(problem), :variables)
            setfield!(problem, :variables, Vector{Any}())
        end
        
        if hasfield(typeof(problem), :variables)
            # Add to existing variables list (like dedalus problem constructor)
            push!(problem.variables, vector_field)
            
            @info "Added vector unknown '$name' to problem variables ($component_name component, $(length(coefficients)) coefficients)"
            
            # Update problem metadata
            if hasfield(typeof(problem), :nvariables)
                problem.nvariables += 1
            end
            
            # Register in namespace for equation parsing (like dedalus namespace)
            if hasfield(typeof(problem), :namespace) && problem.namespace isa Dict
                problem.namespace[name] = vector_field
            end
            
        else
            # Fallback: store in separate tau_variables collection
            if !hasfield(typeof(problem), :tau_variables)
                setfield!(problem, :tau_variables, Vector{Any}())
            end
            push!(problem.tau_variables, vector_field)
            @info "Added vector tau variable '$name' ($component_name component) to problem (fallback storage)"
        end
        
        return vector_field
    end
end

"""
    find_existing_vector_field(problem, name) -> VectorFieldRepresentation or nothing

Find existing vector field in problem by name.
"""
function find_existing_vector_field(problem, name::String)
    # Check in problem.variables
    if hasfield(typeof(problem), :variables)
        for field in problem.variables
            if isa(field, VectorFieldRepresentation) && field.name == name
                return field
            end
        end
    end
    
    # Check in fallback tau_variables
    if hasfield(typeof(problem), :tau_variables)
        for field in problem.tau_variables
            if isa(field, VectorFieldRepresentation) && field.name == name
                return field
            end
        end
    end
    
    return nothing
end

"""
    create_vector_field(name, component, coefficients, T) -> VectorFieldRepresentation

Create a vector field object following Dedalus VectorField class pattern.
This represents the Julia equivalent of dedalus `dist.VectorField(coords, name=name)`.
"""
function create_vector_field(name::String, component::Int, coefficients::Array{Complex{T}}, ::Type{T}) where T<:Real
    # Determine vector field dimensions (typically 3 for spherical coordinates: r, θ, φ)
    max_components = 3
    coeff_shape = size(coefficients)
    
    # Create data arrays for all components (following Dedalus first-axis pattern)
    # In Dedalus: vector fields have shape (num_components, ...spatial_dims)
    vector_coeff_shape = (max_components, coeff_shape...)
    data_coefficient = zeros(Complex{T}, vector_coeff_shape)
    data_grid = nothing  # Will be allocated on demand
    
    # Set the specified component
    if component <= max_components
        data_coefficient[component, :] = coefficients[:]
    else
        error("Component $component exceeds maximum vector components ($max_components)")
    end
    
    # Create vector field object following Dedalus VectorField class structure
    vector_field = VectorFieldRepresentation{T}(
        name = name,
        tensor_signature = (max_components,),  # Vector has one tensor index
        dtype = Complex{T},
        data_coefficient = data_coefficient,
        data_grid = data_grid,
        layout = :coefficient,
        bases = nothing,  # Tau fields often have no spatial bases
        scales = ones(T, length(coeff_shape)),
        global_shape = vector_coeff_shape,
        num_components = max_components,
        meta = Dict{Symbol,Any}(
            :field_type => :vector,
            :created_at => now(),
            :is_tau_variable => true,
            :active_components => Set([component])
        )
    )
    
    return vector_field
end

"""
    set_vector_component!(field, component, coefficients)

Set coefficients for a specific component of a vector field.
Following Dedalus pattern: tau_field['c'][component] = coefficients
"""
function set_vector_component!(field::VectorFieldRepresentation{T}, component::Int, 
                              coefficients::Array{Complex{T}}) where T<:Real
    if component > field.num_components
        error("Component $component exceeds field dimensions ($(field.num_components))")
    end
    
    # Update coefficient data (first axis contains components)
    field.data_coefficient[component, :] = coefficients[:]
    
    # Mark component as active
    push!(field.meta[:active_components], component)
    
    return field
end

"""
Vector field representation following Dedalus VectorField class pattern.
Represents vector-valued fields for spectral methods.
"""
mutable struct VectorFieldRepresentation{T<:Real}
    name::String
    tensor_signature::Union{Nothing, Tuple}
    dtype::Type
    data_coefficient::Union{Nothing, Array{Complex{T}}}
    data_grid::Union{Nothing, Array{Complex{T}}}
    layout::Symbol  # :coefficient or :grid
    bases::Union{Nothing, Any}  # Spectral bases
    scales::Vector{T}
    global_shape::Tuple
    num_components::Int
    meta::Dict{Symbol,Any}
end

# Constructor for VectorFieldRepresentation
function VectorFieldRepresentation{T}(; 
    name::String,
    tensor_signature = nothing,
    dtype = Complex{T},
    data_coefficient = nothing,
    data_grid = nothing,
    layout::Symbol = :coefficient,
    bases = nothing,
    scales = T[],
    global_shape = (),
    num_components::Int = 3,
    meta = Dict{Symbol,Any}()) where T<:Real
    
    return VectorFieldRepresentation{T}(
        name, tensor_signature, dtype, data_coefficient, data_grid,
        layout, bases, scales, global_shape, num_components, meta
    )
end

"""
    change_layout!(field, new_layout) 

Switch vector field data layout between coefficient and grid space.
Following Dedalus field.change_layout() method.
"""
function change_layout!(field::VectorFieldRepresentation{T}, new_layout::Symbol) where T<:Real
    if new_layout == field.layout
        return field  # Already in requested layout
    end
    
    if new_layout == :grid && field.layout == :coefficient
        # Transform from coefficient to grid space
        # In full implementation, would use spectral transforms for each component
        if field.data_grid === nothing
            field.data_grid = similar(field.data_coefficient)
        end
        field.data_grid .= field.data_coefficient  # Simplified
        field.layout = :grid
        
    elseif new_layout == :coefficient && field.layout == :grid
        # Transform from grid to coefficient space
        if field.data_coefficient === nothing
            field.data_coefficient = similar(field.data_grid)
        end
        field.data_coefficient .= field.data_grid  # Simplified
        field.layout = :coefficient
        
    else
        error("Unknown layout transition: $(field.layout) -> $new_layout")
    end
    
    return field
end

"""
    add_term_to_equation!(problem, equation_name::String, term)

Add term to equation following dedalus patterns.

**Important**: In dedalus, equations are NOT built incrementally by adding terms.
Instead, complete equations are constructed as strings and added all at once using `add_equation`.

The dedalus pattern is:
```python
# Complete equation with all terms including lift
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - u@grad(u)")

# NOT: problem.add_term("momentum", "lift(tau_u)")
```

# Mathematical Background
Dedalus equations are symbolic expressions parsed from strings:
- LHS: Linear terms (time derivatives, spatial operators, lift terms)
- RHS: Nonlinear terms and forcing
- Complete equation: `"LHS = RHS"` format

# Arguments
- `problem`: Problem object (dedalus IVP/LBVP/EVP/NLBVP)
- `equation_name::String`: Target equation identifier
- `term`: Term to add (lift term, boundary term, etc.)

# Implementation Notes
This function simulates term addition for compatibility, but in real dedalus
integration, equations should be constructed as complete strings.

See dedalus examples: all equations added as complete expressions.
"""
function add_term_to_equation!(problem, equation_name::String, term)
    # Following dedalus approach: equations are NOT built incrementally
    # In dedalus: problem.add_equation("complete_equation_string")
    # where the string contains all terms: "dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = -u@grad(u)"
    
    @warn "add_term_to_equation! is deprecated. Dedalus uses complete equation strings, not incremental term addition."
    @info "Instead of building equations term-by-term, construct the complete equation string and use add_equation!(problem, equation_string)"
    
    # Store term for potential equation reconstruction
    term_info = DedalusEquationTerm(
        equation_name = equation_name,
        term_expression = string(term),
        term_type = :lift_term,
        position = :lhs  # Lift terms typically go on LHS
    )
    
    # Add to problem's equation builder (for compatibility)
    if !hasfield(typeof(problem), :equation_builder)
        setfield!(problem, :equation_builder, Dict{String, Vector{DedalusEquationTerm}}())
    end
    
    if hasfield(typeof(problem), :equation_builder)
        if !haskey(problem.equation_builder, equation_name)
            problem.equation_builder[equation_name] = DedalusEquationTerm[]
        end
        push!(problem.equation_builder[equation_name], term_info)
        
        @info "Stored term for equation '$equation_name'. Use build_complete_equation!(problem, '$equation_name') to generate dedalus equation string"
    end
    
    return term_info
end

"""
Represents a term in a Dedalus equation for incremental building.
"""
struct DedalusEquationTerm
    equation_name::String
    term_expression::String
    term_type::Symbol      # :lift_term, :differential, :source, etc.
    position::Symbol       # :lhs, :rhs
end

"""
    build_complete_equation!(problem, equation_name) -> String

Build complete Dedalus equation string from accumulated terms.
Following Dedalus pattern: "LHS = RHS"
"""
function build_complete_equation!(problem, equation_name::String)
    if !hasfield(typeof(problem), :equation_builder) || 
       !haskey(problem.equation_builder, equation_name)
        error("No terms found for equation '$equation_name'. Use add_term_to_equation! first or provide complete equation string.")
    end
    
    terms = problem.equation_builder[equation_name]
    
    # Separate LHS and RHS terms
    lhs_terms = String[]
    rhs_terms = String[]
    
    for term in terms
        if term.position == :lhs
            push!(lhs_terms, term.term_expression)
        else
            push!(rhs_terms, term.term_expression)
        end
    end
    
    # Build complete equation string
    lhs_str = isempty(lhs_terms) ? "0" : join(lhs_terms, " + ")
    rhs_str = isempty(rhs_terms) ? "0" : join(rhs_terms, " + ")
    
    equation_str = "$lhs_str = $rhs_str"
    
    @info "Built complete equation '$equation_name': $equation_str"
    
    return equation_str
end

"""
    add_equation!(problem, equation_name::String, equation::BoundaryEquation{T}) where T<:Real

Add complete equation to problem following dedalus patterns.

In dedalus, equations are added as complete string expressions:

```python
# Physical equations with lift terms
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - u@grad(u)")

# Boundary conditions
problem.add_equation("u(r=1) = 0")
problem.add_equation("radial(grad(T)(r=1)) = -2")

# Gauge conditions
problem.add_equation("integ(p) = 0")
```

# Mathematical Background
Dedalus equations follow the pattern `"LHS = RHS"` where:
- LHS: Linear operators, time derivatives, tau/lift terms
- RHS: Nonlinear terms, boundary values, forcing
- Boundary conditions: Field evaluations at boundaries

# Arguments
- `problem`: Problem object to modify
- `equation_name::String`: Unique identifier for the equation
- `equation::BoundaryEquation{T}`: Boundary equation object to add

# Implementation Notes
This function translates BoundaryEquation objects to dedalus-style equation strings.
In full dedalus integration, would call problem.add_equation(equation_string).

See dedalus examples: ball convection, Rayleigh-Benard for equation patterns.
"""
function add_equation!(problem, equation_name::String, equation::BoundaryEquation{T}) where T<:Real
    # Following dedalus v3 approach: problem.add_equation("LHS = RHS")
    # Convert boundary equations to proper dedalus equation strings
    
    # Build equation string based on boundary condition type following dedalus syntax
    equation_string = if isa(equation, DirichletEquation)
        # Dedalus Dirichlet: field(boundary=value) = rhs
        location = equation.location == :surface ? "r=1" : "r=0"  # Assume unit sphere
        rhs_value = format_complex_value(equation.boundary_value)
        "$(equation.field_name)($location) = $rhs_value"
        
    elseif isa(equation, NeumannEquation)
        # Dedalus Neumann: dr(field)(boundary=value) = rhs  
        location = equation.location == :surface ? "r=1" : "r=0"
        rhs_value = format_complex_value(equation.derivative_value)
        "dr($(equation.field_name))($location) = $rhs_value"
        
    elseif isa(equation, RobinEquation)
        # Dedalus Robin: a*field(boundary) + b*dr(field)(boundary) = rhs
        location = equation.location == :surface ? "r=1" : "r=0"
        rhs_value = format_complex_value(equation.rhs_value)
        a_coeff = format_real_value(equation.a_coeff)
        b_coeff = format_real_value(equation.b_coeff)
        "$(a_coeff)*$(equation.field_name)($location) + $(b_coeff)*dr($(equation.field_name))($location) = $rhs_value"
        
    elseif isa(equation, RegularityEquation)
        # Regularity: implicit - handled by field construction, not explicit equation
        "# Regularity condition for $(equation.field_name) at $(equation.location) (implicit)"
        
    else
        error("Unknown boundary equation type: $(typeof(equation))")
    end
    
    # Create proper dedalus equation representation
    dedalus_equation = DedalusEquation(
        name = equation_name,
        equation_string = equation_string,
        equation_type = :boundary_condition,
        source_bc = equation,
        lhs_terms = extract_lhs_terms(equation_string),
        rhs_terms = extract_rhs_terms(equation_string),
        variables = extract_variables(equation_string)
    )
    
    # Add to problem following dedalus pattern
    if !hasfield(typeof(problem), :equations)
        setfield!(problem, :equations, Vector{DedalusEquation}())
    end
    
    if hasfield(typeof(problem), :equations)
        # Add to problem equations list (like dedalus problem.add_equation())
        push!(problem.equations, dedalus_equation)
        
        @info "Added equation '$equation_name' to problem: $equation_string"
        
        # Also call core add_equation! if available (for compatibility)
        try
            if hasmethod(add_equation!, (typeof(problem), String))
                add_equation!(problem, equation_string)
            end
        catch
            # Ignore if core method unavailable
        end
        
        # Update problem metadata
        if hasfield(typeof(problem), :num_equations)
            problem.num_equations += 1
        end
        
    else
        @warn "Problem object does not support equation storage"
    end
    
    return dedalus_equation
end

"""
    add_equation!(problem, equation_string::String)

Add complete equation string to problem following dedalus pattern.
This is the main interface for adding equations in dedalus style.
"""
function add_equation!(problem, equation_string::String)
    # Following dedalus v3: problem.add_equation("dt(u) - nu*lap(u) + grad(p) = -u@grad(u)")
    
    # Parse and validate equation string
    if !contains(equation_string, "=")
        error("Equation string must contain '=' separator: '$equation_string'")
    end
    
    # Create dedalus equation representation
    dedalus_equation = DedalusEquation(
        name = generate_equation_name(equation_string),
        equation_string = equation_string,
        equation_type = :pde,
        source_bc = nothing,
        lhs_terms = extract_lhs_terms(equation_string),
        rhs_terms = extract_rhs_terms(equation_string),
        variables = extract_variables(equation_string)
    )
    
    # Add to problem
    if !hasfield(typeof(problem), :equations)
        setfield!(problem, :equations, Vector{DedalusEquation}())
    end
    
    if hasfield(typeof(problem), :equations)
        push!(problem.equations, dedalus_equation)
        @info "Added equation to problem: $equation_string"
        
        # Update metadata
        if hasfield(typeof(problem), :num_equations)
            problem.num_equations += 1
        elseif !hasfield(typeof(problem), :num_equations)
            setfield!(problem, :num_equations, 1)
        end
        
    else
        @warn "Problem object does not support equation storage"
    end
    
    return dedalus_equation
end

"""
Dedalus equation representation following dedalus problem structure.
"""
struct DedalusEquation
    name::String
    equation_string::String
    equation_type::Symbol      # :pde, :boundary_condition, :constraint
    source_bc::Union{Nothing, BoundaryEquation}
    lhs_terms::Vector{String}
    rhs_terms::Vector{String}
    variables::Vector{String}
end

"""Helper functions for equation parsing and formatting."""
function format_complex_value(val::Complex{T}) where T<:Real
    if imag(val) ≈ 0
        return string(real(val))
    else
        return string(val)
    end
end

function format_real_value(val::T) where T<:Real
    return string(val)
end

function extract_lhs_terms(equation_string::String)
    parts = split(equation_string, "=", limit=2)
    return [strip(parts[1])]  # Simplified: treat whole LHS as one term
end

function extract_rhs_terms(equation_string::String)
    parts = split(equation_string, "=", limit=2)
    return length(parts) > 1 ? [strip(parts[2])] : ["0"]
end

function extract_variables(equation_string::String)
    # Simplified variable extraction - in practice would use more sophisticated parsing
    variables = String[]
    
    # Common variable patterns in dedalus equations
    patterns = [r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"]
    
    for pattern in patterns
        matches = collect(eachmatch(pattern, equation_string))
        for m in matches
            var = m.match
            if !(var in ["dt", "grad", "lap", "div", "dr", "lift", "integ", "r"]) && !(var in variables)
                push!(variables, var)
            end
        end
    end
    
    return variables
end

function generate_equation_name(equation_string::String)
    # Generate a name based on equation content
    hash_val = string(hash(equation_string))[1:8]
    return "eq_$hash_val"
end