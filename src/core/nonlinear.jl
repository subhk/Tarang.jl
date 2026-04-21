"""
Nonlinear term evaluation using PencilArrays and PencilFFTs

This module implements efficient evaluation of nonlinear terms in spectral methods,
designed for Julia with PencilArrays/PencilFFTs.
Supports both 2D and 3D parallelization with proper dealiasing.

Key features:
- Transform-based multiplication for nonlinear terms (u·∇u, etc.)
- Automatic dealiasing using 3/2 rule
- MPI parallelization through PencilArrays
- Efficient memory management and reuse
- Support for various nonlinear operators
"""

include("nonlinear/nonlinear_core.jl")

include("nonlinear/nonlinear_padding.jl")
include("nonlinear/nonlinear_transforms.jl")

include("nonlinear/nonlinear_dealiasing.jl")

# Runtime allocation and pencil-compatibility helpers live out of line so
# the remaining code reads as the nonlinear execution path.
include("nonlinear/nonlinear_pencil_utils.jl")

include("nonlinear/nonlinear_evaluation.jl")

# ============================================================================
# Exports
# ============================================================================

# Export types
export NonlinearOperator, AdvectionOperator, NonlinearAdvectionOperator, ConvectiveOperator
export NonlinearEvaluator, NonlinearPerformanceStats

# Export constructor functions
export advection, nonlinear_momentum, convection

# Export evaluation functions
export evaluate_nonlinear_term, evaluate_transform_multiply, evaluate_operator
export evaluate_vector_dot_product, evaluate_vector_cross_product

# Export dealiasing functions
export apply_basic_dealiasing!, apply_spectral_cutoff!, get_dealiasing_cutoffs
export apply_spherical_spectral_cutoff!

# Export utility functions
export get_nonlinear_transform, setup_nonlinear_transforms!
export get_temp_field, clear_temp_fields!, get_temp_array
export log_nonlinear_performance

# Export GPU helper functions for masks (useful for custom dealiasing)
export create_dealiasing_mask, create_spherical_mask

# Export pencil compatibility functions
export get_pencil_compatible_data, set_pencil_compatible_data!
export compute_local_shape, compute_local_range, is_shape_compatible
