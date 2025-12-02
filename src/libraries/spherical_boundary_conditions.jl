"""
Spherical Boundary Conditions Module

Complete boundary condition system for spherical ball domains following dedalus approach.
Provides modular organization with separate files for different functionality:

- boundary_conditions.jl: Core BC types and construction
- utilities.jl: Mathematical utilities (Zernike, spherical harmonics)  
- regularity.jl: Center and pole regularity enforcement
- lift_operators.jl: Tau method lift operators
- tau_system.jl: Complete tau system implementation

# Usage Example

```julia
using SphericalBoundaryConditions

# Create boundary conditions
bc1 = create_boundary_condition(:dirichlet, :surface, 0.0)
bc2 = create_boundary_condition(:regularity, :center, nothing; regularity_type=:scalar)

# Build tau system
tau_system = build_tau_system([bc1, bc2], basis)

# Add to problem
add_tau_terms!(problem, tau_system)

# Enforce regularity
enforce_regularity!(field, coords)
```

Based on dedalus spectral methods and boundary condition system.
Organized following dedalus modular structure for maintainability.
"""

module SphericalBoundaryConditions

using LinearAlgebra
using SparseArrays

# Define types that need to be available across modules
# These would typically come from the main codebase
struct SphericalCoordinates{T<:Real}
    r_min::T
    r_max::T
end

struct SphericalScalarField{T<:Real}
    data_spectral::Vector{Complex{T}}
    data_grid::Array{Complex{T}, 3}  # [phi, theta, r]
    coords::SphericalCoordinates{T}
end

struct SphericalVectorField{T<:Real}
    data_spectral::Array{Complex{T}, 2}  # [mode, component]
    data_grid::Array{Complex{T}, 4}      # [phi, theta, r, component]
    coords::SphericalCoordinates{T}
end

struct BallBasis{T<:Real}
    type::Symbol
    n_total_modes::Int
    n_radial::Int
    n_boundary_modes::Int
    l_max::Int
    n_max::Int
end

struct BallDomain{T<:Real}
    basis::BallBasis{T}
    coords::SphericalCoordinates{T}
end

# Include sub-modules
include("spherical_boundary_conditions/boundary_conditions.jl")
include("spherical_boundary_conditions/utilities.jl") 
include("spherical_boundary_conditions/regularity.jl")
include("spherical_boundary_conditions/lift_operators.jl")
include("spherical_boundary_conditions/tau_system.jl")

# Re-export all public interface from sub-modules

# From boundary_conditions.jl
export SphericalBoundaryCondition
export DirichletBC, NeumannBC, RobinBC, RegularityBC
export create_boundary_condition

# From utilities.jl  
export evaluate_zernike_at_center, evaluate_zernike_at_boundary
export evaluate_zernike_derivative_at_boundary, compute_zernike_radial, compute_zernike_radial_derivative
export evaluate_spin_weighted_harmonic_at_pole

# From regularity.jl
export enforce_regularity!, enforce_scalar_regularity!, enforce_vector_regularity!
export apply_center_regularity_scalar!, apply_pole_regularity_scalar!
export compute_regular_value_at_center, compute_regular_value_at_pole
export compute_radial_regular_value, compute_angular_regular_value_at_pole

# From lift_operators.jl
export LiftOperator, LiftTerm
export build_lift_matrices, build_surface_lift_matrix
export compute_lift_weight, get_lift_mode_number, get_robin_lift_modes
export apply_dirichlet_lift!, apply_neumann_lift!, apply_robin_lift!, apply_regularity_lift!

# From tau_system.jl
export TauSystem, TauVariable
export build_tau_system, add_boundary_condition!, rebuild_tau_matrices!
export add_tau_terms!, add_unknown_field!, add_lift_terms_to_equations!
export add_boundary_condition_equations!
export BoundaryEquation, DirichletEquation, NeumannEquation, RobinEquation, RegularityEquation
export compute_bc_rhs_value, estimate_total_modes, get_target_equation_name
export add_scalar_unknown!, add_vector_component_unknown!
export add_term_to_equation!, add_equation!

# High-level convenience functions

"""
    apply_boundary_conditions!(field, boundary_conditions, coords)

High-level function to apply multiple boundary conditions to a field.

# Arguments
- `field`: SphericalScalarField or SphericalVectorField to modify
- `boundary_conditions`: Vector of boundary condition objects  
- `coords`: SphericalCoordinates for the domain

# Example
```julia
bcs = [
    create_boundary_condition(:dirichlet, :surface, 0.0),
    create_boundary_condition(:regularity, :center, nothing; regularity_type=:scalar)
]

apply_boundary_conditions!(field, bcs, coords)
```
"""
function apply_boundary_conditions!(field::Union{SphericalScalarField{T}, SphericalVectorField{T}},
                                   boundary_conditions::Vector{SphericalBoundaryCondition{T}},
                                   coords::SphericalCoordinates{T}) where T<:Real
    for bc in boundary_conditions
        apply_single_boundary_condition!(field, bc, coords)
    end
end

"""
    apply_single_boundary_condition!(field, bc, coords)

Apply single boundary condition to field.
"""
function apply_single_boundary_condition!(field::Union{SphericalScalarField{T}, SphericalVectorField{T}},
                                         bc::SphericalBoundaryCondition{T},
                                         coords::SphericalCoordinates{T}) where T<:Real
    if isa(bc, RegularityBC)
        # Regularity conditions use specialized enforcement
        enforce_regularity!(field, coords)
    else
        # Other boundary conditions would be handled by tau system
        # This is a simplified placeholder - full implementation would
        # require integration with the tau system
        @warn "Direct boundary condition application not fully implemented. Use tau system integration."
    end
end

end  # module SphericalBoundaryConditions