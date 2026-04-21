# ============================================================================
# Exports
# ============================================================================

# Export configuration
export SolverConfig, SUBSYSTEM_GROUP

# Export main types
export Subsystem, Subproblem, NCCData

# Export subsystem construction
export build_subsystems, build_subproblems
export scalar_components, scalar_field_dofs
export infer_problem_dtype, infer_problem_dist
export compute_variable_ranges, compute_equation_ranges, compute_matrix_group

# Export subsystem methods
export coeff_slices, coeff_shape, coeff_size
export field_slices, field_shape, field_size, field_domain, subproblem_field_size
export gather, scatter

# Export subproblem methods
export check_condition, valid_modes
export gather_inputs, scatter_inputs, gather_eqn_F!, gather_alg_F!, apply_bc_override!

# Export matrix building
export build_matrices!, build_subproblem_matrices
export expression_matrices, gather_ncc_coeffs!
export get_valid_modes, get_valid_modes_var
export compute_update_rank

# Export permutation functions
export left_permutation, right_permutation, perm_matrix
export get_var_dim

# Export matrix utilities
export drop_empty_rows, zeros_with_pattern, expand_pattern, apply_sparse

# Export NCC functions
export build_ncc_matrix, cartesian_mode_matrix

# Export coupling analysis
export get_matrix_coupling, analyze_operator_coupling
export get_separable_dim_size, generate_mode_groups
