"""
Core boundary condition types and construction functions for spherical domains.

Defines the fundamental boundary condition types (Dirichlet, Neumann, Robin, Regularity)
and their construction functions following dedalus patterns.
"""

using LinearAlgebra

export SphericalBoundaryCondition
export DirichletBC, NeumannBC, RobinBC, RegularityBC
export create_boundary_condition

"""
Abstract base type for spherical boundary conditions.
"""
abstract type SphericalBoundaryCondition{T<:Real} end

"""
Dirichlet Boundary Condition: f(r=R, θ, φ) = g(θ, φ)
"""
struct DirichletBC{T<:Real} <: SphericalBoundaryCondition{T}
    location::Symbol                    # :surface, :center
    component::Union{Int, Nothing}      # Vector component (nothing for scalar)
    value_function::Function           # g(θ, φ) or constant
    
    function DirichletBC{T}(location::Symbol, value_function::Function; 
                           component::Union{Int, Nothing}=nothing) where T<:Real
        new{T}(location, component, value_function)
    end
end

# Convenience constructors
DirichletBC(location::Symbol, value_function::Function, ::Type{T}=Float64; kwargs...) where T = 
    DirichletBC{T}(location, value_function; kwargs...)

DirichletBC(location::Symbol, value::Number, ::Type{T}=Float64; kwargs...) where T = 
    DirichletBC{T}(location, (θ, φ) -> T(value); kwargs...)

"""
Neumann Boundary Condition: ∂f/∂r|_{r=R} = h(θ, φ)
"""
struct NeumannBC{T<:Real} <: SphericalBoundaryCondition{T}
    location::Symbol
    component::Union{Int, Nothing}
    value_function::Function
    
    function NeumannBC{T}(location::Symbol, value_function::Function; 
                         component::Union{Int, Nothing}=nothing) where T<:Real
        new{T}(location, component, value_function)
    end
end

NeumannBC(location::Symbol, value_function::Function, ::Type{T}=Float64; kwargs...) where T = 
    NeumannBC{T}(location, value_function; kwargs...)

NeumannBC(location::Symbol, value::Number, ::Type{T}=Float64; kwargs...) where T = 
    NeumannBC{T}(location, (θ, φ) -> T(value); kwargs...)

"""
Robin Boundary Condition: a⋅f + b⋅∂f/∂r = c at r=R
"""
struct RobinBC{T<:Real} <: SphericalBoundaryCondition{T}
    location::Symbol
    component::Union{Int, Nothing}
    a_coeff::T                         # Coefficient of f
    b_coeff::T                         # Coefficient of ∂f/∂r
    rhs_function::Function             # Right-hand side c(θ, φ)
    
    function RobinBC{T}(location::Symbol, a_coeff::T, b_coeff::T, rhs_function::Function;
                       component::Union{Int, Nothing}=nothing) where T<:Real
        new{T}(location, component, a_coeff, b_coeff, rhs_function)
    end
end

RobinBC(location::Symbol, a::T, b::T, rhs::Function, ::Type{T}=Float64; kwargs...) where T = 
    RobinBC{T}(location, a, b, rhs; kwargs...)

"""
Regularity Boundary Condition: Automatic regularity enforcement
"""
struct RegularityBC{T<:Real} <: SphericalBoundaryCondition{T}
    location::Symbol                    # :center, :poles
    regularity_type::Symbol            # :scalar, :vector_radial, :vector_angular
    component::Union{Int, Nothing}
    
    function RegularityBC{T}(location::Symbol, regularity_type::Symbol;
                            component::Union{Int, Nothing}=nothing) where T<:Real
        new{T}(location, regularity_type, component)
    end
end

RegularityBC(location::Symbol, regularity_type::Symbol, ::Type{T}=Float64; kwargs...) where T = 
    RegularityBC{T}(location, regularity_type; kwargs...)

"""
    create_boundary_condition(bc_type, location, value; kwargs...) -> SphericalBoundaryCondition

Factory function for creating boundary conditions following dedalus patterns.

# Arguments
- `bc_type::Symbol`: Type of boundary condition (:dirichlet, :neumann, :robin, :regularity)
- `location::Symbol`: Boundary location (:surface, :center, :poles) 
- `value`: Boundary condition value (function or number)

# Keywords
- `component::Union{Int, Nothing}`: Vector component for vector fields
- `a_coeff::T`: Robin condition coefficient for field term (Robin only)
- `b_coeff::T`: Robin condition coefficient for derivative term (Robin only) 
- `regularity_type::Symbol`: Type of regularity condition (Regularity only)

# Examples
```julia
# Dirichlet: u = 0 on surface
bc1 = create_boundary_condition(:dirichlet, :surface, 0.0)

# Neumann: ∂u/∂r = 1 on surface  
bc2 = create_boundary_condition(:neumann, :surface, 1.0)

# Robin: u + ∂u/∂r = sin(θ) on surface
bc3 = create_boundary_condition(:robin, :surface, sin; a_coeff=1.0, b_coeff=1.0)

# Regularity at center
bc4 = create_boundary_condition(:regularity, :center, nothing; regularity_type=:scalar)
```
"""
function create_boundary_condition(bc_type::Symbol, location::Symbol, value::T; 
                                  component::Union{Int, Nothing}=nothing,
                                  a_coeff::T=T(1), b_coeff::T=T(1),
                                  regularity_type::Symbol=:scalar) where T<:Real
    
    if bc_type == :dirichlet
        return DirichletBC{T}(location, (θ, φ) -> T(value); component=component)
        
    elseif bc_type == :neumann
        return NeumannBC{T}(location, (θ, φ) -> T(value); component=component)
        
    elseif bc_type == :robin
        return RobinBC{T}(location, a_coeff, b_coeff, (θ, φ) -> T(value); component=component)
        
    elseif bc_type == :regularity
        return RegularityBC{T}(location, regularity_type; component=component)
        
    else
        error("Unknown boundary condition type: $bc_type. Valid types: :dirichlet, :neumann, :robin, :regularity")
    end
end

# Separate method for Function values
function create_boundary_condition(bc_type::Symbol, location::Symbol, value::Function, ::Type{T}=Float64; 
                                  component::Union{Int, Nothing}=nothing,
                                  a_coeff::T=T(1), b_coeff::T=T(1),
                                  regularity_type::Symbol=:scalar) where T<:Real
    
    if bc_type == :dirichlet
        return DirichletBC{T}(location, value; component=component)
        
    elseif bc_type == :neumann
        return NeumannBC{T}(location, value; component=component)
        
    elseif bc_type == :robin
        return RobinBC{T}(location, a_coeff, b_coeff, value; component=component)
        
    elseif bc_type == :regularity
        return RegularityBC{T}(location, regularity_type; component=component)
        
    else
        error("Unknown boundary condition type: $bc_type. Valid types: :dirichlet, :neumann, :robin, :regularity")
    end
end

# Overload for different value types
function create_boundary_condition(bc_type::Symbol, location::Symbol, value::Function; kwargs...) 
    # Extract numeric type from kwargs if possible, default to Float64
    T = Float64  # Could be more sophisticated type inference
    return create_boundary_condition(bc_type, location, value, T; kwargs...)
end

create_boundary_condition(bc_type::Symbol, location::Symbol, value::Number; kwargs...) = 
    create_boundary_condition(bc_type, location, Float64(value); kwargs...)

create_boundary_condition(bc_type::Symbol, location::Symbol, ::Nothing; kwargs...) = 
    create_boundary_condition(bc_type, location, 0.0; kwargs...)