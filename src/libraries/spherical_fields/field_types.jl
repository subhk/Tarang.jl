"""
Spherical Field Types and Layout System

Core type definitions for spherical ball domain fields:
- SphericalScalarField: Scalar fields on ball domains
- SphericalVectorField: Vector fields with proper component handling
- SphericalTensorField: Tensor fields with full index management
- LayoutManager: Manages data layout transformations
- SphericalFieldSystem: System for managing multiple fields

Based on dedalus/core/field.py with spherical-specific extensions.
"""

using PencilArrays
using PencilFFTs
using LinearAlgebra
using SparseArrays
using MPI
using StaticArrays

"""
Spherical Field Layout Types

Defines different data layouts for spherical fields:
- :grid - Physical space (r, θ, φ) coordinates
- :spectral - Spectral space (n, l, m) coefficients  
- :mixed - Partially transformed states
"""
@enum SphericalLayout begin
    GRID_LAYOUT       # (φ, θ, r) physical coordinates
    SPECTRAL_LAYOUT   # (m, l, n) spectral coefficients
    MIXED_PHI         # FFT in φ, grid in θ,r
    MIXED_THETA       # FFT in φ, SHT in θ, grid in r
end

"""
Layout Manager for Spherical Fields

Handles layout transformations and data organization for distributed spherical fields.
"""
struct LayoutManager{T<:Real}
    ball_basis::BallBasis{T}
    current_layout::Ref{SphericalLayout}
    
    # Layout-specific shapes
    grid_shape::Tuple{Int,Int,Int}        # (nphi, ntheta, nr)
    spectral_shape::Tuple{Int,Int,Int}    # (nm, nl, nn)
    
    # Transform workspace
    transform_buffers::Dict{String, Array{Complex{T}}}
    
    function LayoutManager{T}(ball_basis::BallBasis{T}) where T<:Real
        grid_shape = (ball_basis.nphi, ball_basis.ntheta, ball_basis.nr)
        spectral_shape = (ball_basis.nphi÷2+1, ball_basis.l_max+1, ball_basis.n_max+1)
        
        local_grid_shape = PencilArrays.size_local(ball_basis.pencil)
        
        transform_buffers = Dict{String, Array{Complex{T}}}(
            "grid_buffer" => zeros(Complex{T}, local_grid_shape...),
            "spectral_buffer" => zeros(Complex{T}, spectral_shape...),
            "phi_buffer" => zeros(Complex{T}, local_grid_shape...),
            "theta_buffer" => zeros(Complex{T}, local_grid_shape...)
        )
        
        new{T}(ball_basis, Ref(GRID_LAYOUT), grid_shape, spectral_shape, transform_buffers)
    end
end

"""
Spherical Scalar Field

Represents scalar fields f(r,θ,φ) on ball domains with complete transform support.
"""
mutable struct SphericalScalarField{T<:Real}
    # Field identification
    name::String
    domain::BallDomain{T}
    layout_manager::LayoutManager{T}
    
    # Data storage (distributed via PencilArrays)
    data_grid::Array{Complex{T},3}        # Grid space data
    data_spectral::Array{Complex{T},3}    # Spectral coefficients
    
    # Field properties
    dtype::DataType                       # Complex{T} or T
    tensor_signature::Vector{Int}         # Empty for scalar
    regularity_conditions::Dict{String, Any}
    
    # Metadata
    scales::Vector{T}                     # Scaling factors
    constant::Bool                        # Whether field is constant
    
    function SphericalScalarField{T}(name::String, domain::BallDomain{T}; 
                                   dtype::DataType=Complex{T},
                                   scales::Vector{T}=ones(T,3),
                                   constant::Bool=false) where T<:Real
        
        layout_manager = LayoutManager{T}(
            BallBasis{T}(domain.coords.nr-1, domain.coords.ntheta÷2, 
                        domain.coords.nr, domain.coords.ntheta, domain.coords.nphi)
        )
        
        # Initialize data arrays
        local_shape = PencilArrays.size_local(layout_manager.ball_basis.pencil)
        data_grid = zeros(Complex{T}, local_shape...)
        data_spectral = zeros(Complex{T}, layout_manager.spectral_shape...)
        
        # Setup regularity conditions
        regularity_conditions = Dict{String, Any}(
            "pole_regularity" => true,
            "center_regularity" => true,
            "surface_bc" => nothing
        )
        
        new{T}(name, domain, layout_manager, data_grid, data_spectral, dtype,
               Int[], regularity_conditions, scales, constant)
    end
end

"""
Spherical Vector Field

Represents vector fields F⃗(r,θ,φ) with proper component transformation and regularity.
"""
mutable struct SphericalVectorField{T<:Real}
    # Field identification  
    name::String
    domain::BallDomain{T}
    layout_manager::LayoutManager{T}
    
    # Data storage for 3 vector components
    data_grid::Array{Complex{T},4}        # (3, φ, θ, r) grid space
    data_spectral::Array{Complex{T},4}    # (3, m, l, n) spectral space
    data_spin::Array{Complex{T},4}        # (3, φ, θ, r) spin components
    data_regularity::Array{Complex{T},4}  # (3, φ, θ, r) regularity components
    
    # Component representation
    current_representation::Symbol        # :coordinate, :spin, or :regularity
    
    # Field properties
    dtype::DataType
    tensor_signature::Vector{Int}         # [1] for vector
    regularity_conditions::Dict{String, Any}
    
    # Transform matrices (from coordinate system)
    U_forward::SMatrix{3,3,Complex{T}}    # Coord → spin
    U_backward::SMatrix{3,3,Complex{T}}   # Spin → coord
    Q_forward::SMatrix{3,3,Complex{T}}    # Spin → regularity
    Q_backward::SMatrix{3,3,Complex{T}}   # Regularity → spin
    
    # Metadata
    scales::Vector{T}
    constant::Bool
    
    function SphericalVectorField{T}(name::String, domain::BallDomain{T}; 
                                   dtype::DataType=Complex{T},
                                   scales::Vector{T}=ones(T,3),
                                   constant::Bool=false) where T<:Real
        
        layout_manager = LayoutManager{T}(
            BallBasis{T}(domain.coords.nr-1, domain.coords.ntheta÷2, 
                        domain.coords.nr, domain.coords.ntheta, domain.coords.nphi)
        )
        
        # Initialize data arrays
        local_shape = PencilArrays.size_local(layout_manager.ball_basis.pencil)
        data_grid = zeros(Complex{T}, 3, local_shape...)
        data_spectral = zeros(Complex{T}, 3, layout_manager.spectral_shape...)
        data_spin = zeros(Complex{T}, 3, local_shape...)
        data_regularity = zeros(Complex{T}, 3, local_shape...)
        
        # Get transform matrices from coordinate system
        coords = domain.coords
        U_forward = coords.U_forward
        U_backward = coords.U_backward
        Q_forward = coords.Q_forward  
        Q_backward = coords.Q_backward
        
        # Setup regularity conditions for vector field
        regularity_conditions = Dict{String, Any}(
            "pole_regularity" => true,
            "center_regularity" => true,
            "surface_bc" => nothing,
            "component_regularity" => [true, true, true]  # Per-component regularity
        )
        
        new{T}(name, domain, layout_manager, data_grid, data_spectral, data_spin, data_regularity,
               :coordinate, dtype, [1], regularity_conditions,
               U_forward, U_backward, Q_forward, Q_backward, scales, constant)
    end
end

"""
Spherical Tensor Field

Represents tensor fields T(r,θ,φ) with full index management and component transformations.
"""
mutable struct SphericalTensorField{T<:Real}
    # Field identification
    name::String
    domain::BallDomain{T}
    layout_manager::LayoutManager{T}
    
    # Tensor properties
    tensor_rank::Int
    tensor_signature::Vector{Int}         # Index signature
    component_count::Int                  # Total number of components
    
    # Data storage
    data_grid::Array{Complex{T}}          # Multi-dimensional grid space
    data_spectral::Array{Complex{T}}      # Multi-dimensional spectral space
    
    # Component transformations
    component_transforms::Vector{SMatrix{3,3,Complex{T}}}
    
    # Field properties
    dtype::DataType
    regularity_conditions::Dict{String, Any}
    scales::Vector{T}
    constant::Bool
    
    function SphericalTensorField{T}(name::String, domain::BallDomain{T}, tensor_signature::Vector{Int}; 
                                   dtype::DataType=Complex{T},
                                   scales::Vector{T}=ones(T,3),
                                   constant::Bool=false) where T<:Real
        
        tensor_rank = length(tensor_signature)
        component_count = 3^tensor_rank
        
        layout_manager = LayoutManager{T}(
            BallBasis{T}(domain.coords.nr-1, domain.coords.ntheta÷2, 
                        domain.coords.nr, domain.coords.ntheta, domain.coords.nphi)
        )
        
        # Initialize data arrays with proper tensor dimensions
        local_shape = PencilArrays.size_local(layout_manager.ball_basis.pencil)
        tensor_shape_grid = (fill(3, tensor_rank)..., local_shape...)
        tensor_shape_spectral = (fill(3, tensor_rank)..., layout_manager.spectral_shape...)
        
        data_grid = zeros(Complex{T}, tensor_shape_grid...)
        data_spectral = zeros(Complex{T}, tensor_shape_spectral...)
        
        # Setup component transformations
        coords = domain.coords
        component_transforms = fill(coords.U_forward, tensor_rank)
        
        # Regularity conditions for tensor
        regularity_conditions = Dict{String, Any}(
            "pole_regularity" => true,
            "center_regularity" => true,
            "surface_bc" => nothing,
            "tensor_regularity" => fill(true, component_count)
        )
        
        new{T}(name, domain, layout_manager, tensor_rank, tensor_signature, component_count,
               data_grid, data_spectral, component_transforms,
               dtype, regularity_conditions, scales, constant)
    end
end

"""
Spherical Field System

Manages collections of spherical fields and their interactions.
"""
struct SphericalFieldSystem{T<:Real}
    domain::BallDomain{T}
    fields::Dict{String, Union{SphericalScalarField{T}, SphericalVectorField{T}, SphericalTensorField{T}}}
    global_layout::Ref{SphericalLayout}
    
    # System-wide properties
    time::Ref{T}
    iteration::Ref{Int}
    
    function SphericalFieldSystem{T}(domain::BallDomain{T}) where T<:Real
        fields = Dict{String, Union{SphericalScalarField{T}, SphericalVectorField{T}, SphericalTensorField{T}}}()
        new{T}(domain, fields, Ref(GRID_LAYOUT), Ref(T(0)), Ref(0))
    end
end

# Convenience constructors
SphericalScalarField(name::String, domain::BallDomain{T}; kwargs...) where T = 
    SphericalScalarField{T}(name, domain; kwargs...)

SphericalVectorField(name::String, domain::BallDomain{T}; kwargs...) where T = 
    SphericalVectorField{T}(name, domain; kwargs...)

SphericalTensorField(name::String, domain::BallDomain{T}, tensor_signature::Vector{Int}; kwargs...) where T = 
    SphericalTensorField{T}(name, domain, tensor_signature; kwargs...)

SphericalFieldSystem(domain::BallDomain{T}) where T = SphericalFieldSystem{T}(domain)