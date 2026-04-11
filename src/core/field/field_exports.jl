# ============================================================================
# Exports
# ============================================================================

# Export types
export Operand, ScalarField, VectorField, TensorField, LockedField

# Export data allocation and management
export allocate_data!, get_local_array_size, get_process_coordinate,
       get_local_range, global_to_local_index, local_to_global_index,
       get_global_size, get_global_sizes, validate_decomposition_convention

# Export scale functions
export preset_scales!, get_scaled_shape, get_coefficient_shape,
       require_scales!, dealias_scales, apply_dealiasing_scales!,
       set_scales!, change_scales!

# Export data resampling
export resample_grid_data!, resample_1d!, resample_linear_1d!,
       resample_2d!, resample_3d!, resample_nearest!

# Export data access functions
export get_local_data, set_local_data!, get_data

# Export layout and transform functions
export ensure_layout!, require_grid_space!, require_coeff_space!,
       towards_grid_space!, towards_coeff_space!,
       forward_transform!, backward_transform!,
       forward_transform_axis!, backward_transform_axis!

# Export field operations
export fill_random!, integrate

# Export I/O functions
export save_field, load_field!

# Export optimization functions
export has_spectral_bases, apply_dealiasing_to_product!, apply_spectral_cutoff!,
       low_pass_filter!, high_pass_filter!

# Export shape functions
export get_global_grid_shape, get_basis_grid_size, get_global_coeff_shape,
       get_basis_coeff_size, get_local_grid_shape, get_local_coeff_shape,
       get_grid_layout_info

# Export vectorized operations
export vectorized_add!, vectorized_sub!, vectorized_mul!, vectorized_scale!,
       vectorized_axpy!, vectorized_linear_combination!, fast_axpy!

# Export LockedField functions
export unlock

# Export coordinate utilities
export unit_vector_fields
