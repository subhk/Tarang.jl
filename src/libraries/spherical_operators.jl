"""
Spherical Operators Implementation

Complete implementation of differential operators in spherical coordinates for ball domains:
- Gradient operator ∇
- Divergence operator ∇⋅ 
- Curl operator ∇×
- Laplacian operator ∇²
- Angular momentum operators L̂
- Radial derivative operators

Based on dedalus/libraries/dedalus_sphere/operators.py and dedalus/core/operators.py

This main file includes all the organized operator components from separate files
for better maintainability and debugging.
"""

using PencilArrays
using PencilFFTs
using LinearAlgebra
using SparseArrays
using StaticArrays

# Include all operator component files
include("spherical_operators/operator_utilities.jl")
include("spherical_operators/radial_operators.jl") 
include("spherical_operators/angular_operators.jl")
include("spherical_operators/gradient_operator.jl")
include("spherical_operators/divergence_operator.jl")
include("spherical_operators/curl_operator.jl")
include("spherical_operators/laplacian_operator.jl")

# Export main operator types
export SphericalGradient, SphericalDivergence, SphericalCurl, SphericalLaplacian
export AngularMomentumOperator, RadialDerivativeOperator
export RadialLaplacianOperator, AngularLaplacianOperator

# Export main operator functions
export apply_gradient!, apply_divergence!, apply_curl!, apply_laplacian!
export apply_radial_derivative!, apply_angular_derivative!, apply_angular_momentum_operator!
export apply_angular_laplacian!, apply_azimuthal_derivative!

# Export matrix creation functions
export build_radial_derivative_matrices, build_angular_momentum_matrices
export create_radial_operator_matrix, create_jacobi_derivative_matrix
export create_E_matrix, create_gradient_matrix, create_divergence_matrix
export create_spin_curl_transformation_matrix, create_laplacian_matrix

# Export utility functions
export xi_factor, angular_momentum_matrix_element
export jacobi_normalization_constant, jacobi_domain_transformation, jacobi_coupling_element
export gamma_function, zernike_derivative_coefficient
export radial_curl_matrix_component

# Export spin-weighted functions
export transform_to_spin_components!, transform_from_spin_components!
export convert_to_spin_components!, apply_spherical_gradient_spin!
export apply_spherical_divergence_spin!, apply_spherical_laplacian_spin!
export create_spin_angular_matrices, create_unitary_spin_matrix

# Export regularity and geometry functions  
export apply_gradient_with_geometry!, apply_divergence_with_regularity!
export apply_laplacian_with_regularity!, apply_theta_derivative_proper!

# Export enhanced operator functions
export enhanced_apply_spin_curl!, apply_full_spin_curl_transformation!
export apply_angular_laplacian_composition!, apply_angular_laplacian_dedalus_style!
export apply_curl_ell_mode!, apply_curl_regularity_conditions!
export apply_gradient_ell_mode_complete!, apply_gradient_regularity_conditions!

# Export Laplacian-specific functions
export create_radial_laplacian_matrix, apply_laplacian_ell_mode_complete!
export apply_laplacian_regularity_conditions!