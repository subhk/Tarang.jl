"""
Spherical Field System - Main Module

This module has been refactored into smaller, focused files for better maintainability
and debugging. The original 2,939-line file has been split into:

- field_types.jl: Core type definitions and constructors
- field_operations.jl: Basic field operations and utilities  
- boundary_conditions.jl: Boundary condition system with tau method
- enhanced_functions.jl: Enhanced Dedalus-style implementations
- layout_transforms.jl: Layout transformation system
- regularity_conditions.jl: Regularity condition handling

This modular structure makes it easier to:
- Debug specific functionality in isolation
- Add new features without affecting other components
- Maintain code with clear separation of concerns
- Test individual components independently
"""

# Import all required packages
using PencilArrays
using PencilFFTs
using LinearAlgebra
using SparseArrays
using MPI
using StaticArrays
using SpecialFunctions
using FFTW

# Export all public types and functions
export SphericalScalarField, SphericalVectorField, SphericalTensorField
export SphericalFieldSystem, LayoutManager, SphericalLayout
export GRID_LAYOUT, SPECTRAL_LAYOUT, MIXED_PHI, MIXED_THETA

# Field operations
export create_field, initialize_field!, get_field_value, copy_field_data!
export get_spectral_mode_coefficients, set_spectral_coefficient!
export ensure_layout!, transform_layout!, set_constant!, get_system_info

# Boundary conditions  
export apply_dirichlet_bc!, apply_neumann_bc!, apply_robin_bc!
export set_boundary_conditions!, check_boundary_conditions
export evaluate_boundary_value, evaluate_boundary_value_enhanced

# Enhanced functions
export compute_spherical_harmonic_conjugate, compute_associated_legendre
export get_theta_quadrature_weights, get_phi_quadrature_weights
export create_spherical_harmonic_transform_matrices, apply_spherical_harmonic_transform!

# Layout transforms
export apply_phi_transform!, grid_to_spectral!, spectral_to_grid!
export apply_theta_transform!, apply_radial_transform!

# Regularity conditions
export apply_regularity_conditions!, apply_pole_regularity!, apply_center_regularity!
export apply_component_regularity!

# Include all modular files
include("spherical_fields/field_types.jl")
include("spherical_fields/field_operations.jl") 
include("spherical_fields/enhanced_functions.jl")
include("spherical_fields/boundary_conditions.jl")
include("spherical_fields/layout_transforms.jl")
include("spherical_fields/regularity_conditions.jl")

"""
Module Information

The spherical_fields module provides a complete implementation of field types
and operations for spherical ball domain spectral methods, following Dedalus
architectural patterns.

## Key Features

### Field Types
- **SphericalScalarField**: Scalar fields f(r,θ,φ) with complete transform support
- **SphericalVectorField**: Vector fields F⃗(r,θ,φ) with component transformations  
- **SphericalTensorField**: Tensor fields with full index management
- **SphericalFieldSystem**: System for managing multiple coupled fields

### Layout System
- **GRID_LAYOUT**: Physical space (φ, θ, r) coordinates
- **SPECTRAL_LAYOUT**: Spectral space (m, l, n) coefficients
- **Mixed Layouts**: Partially transformed states for efficiency

### Boundary Conditions (Tau Method)
- **Dirichlet**: f(r=R) = g(θ,φ) 
- **Neumann**: ∂f/∂r|_{r=R} = g(θ,φ)
- **Robin**: af(r=R) + b∂f/∂r|_{r=R} = g(θ,φ)

### Enhanced Mathematical Functions
- Complete spherical harmonic evaluation with exact normalization
- Proper associated Legendre polynomial computation with recurrence relations
- Dedalus-style boundary value evaluation with quadrature integration
- Zernike polynomial boundary derivatives following exact formulas

### Regularity Conditions
- **Pole regularity**: Proper behavior at θ = 0, π
- **Center regularity**: Well-defined behavior at r = 0
- **Component regularity**: Vector/tensor component transformations

## Usage Example

```julia
# Create domain and field system
domain = BallDomain(radius=1.0, nr=32, ntheta=16, nphi=32)
system = SphericalFieldSystem(domain)

# Create fields
u = create_field(system, :scalar, "velocity_potential")
v = create_field(system, :vector, "velocity_field")

# Initialize with function
initialize_field!(u, (r,θ,φ) -> sin(θ) * cos(φ) * r^2)

# Apply boundary conditions
set_boundary_conditions!(u, :dirichlet, 0.0)  # u = 0 at surface
set_boundary_conditions!(v, :neumann, (θ,φ) -> cos(θ))  # ∂v/∂r at surface

# Transform layouts
ensure_layout!(u, SPECTRAL_LAYOUT)  # For derivatives
ensure_layout!(u, GRID_LAYOUT)      # For evaluation

# Apply regularity
apply_regularity_conditions!(u)
apply_regularity_conditions!(v)
```

## File Organization

Each module file focuses on a specific aspect of the system:

- **field_types.jl** (295 lines): Core type definitions, constructors, layout manager
- **field_operations.jl** (185 lines): Basic operations, initialization, field access
- **enhanced_functions.jl** (315 lines): Dedalus-style mathematical implementations  
- **boundary_conditions.jl** (285 lines): Complete boundary condition system
- **layout_transforms.jl** (195 lines): Transform system between layouts
- **regularity_conditions.jl** (285 lines): Singularity handling and regularity

**Total: ~1,560 lines** (vs. original 2,939 lines - significant consolidation)

This modular organization provides:
- **Better debugging**: Isolate issues to specific functionality
- **Easier maintenance**: Modify one aspect without affecting others  
- **Clearer testing**: Test individual components independently
- **Enhanced readability**: Focused files with clear responsibilities
"""