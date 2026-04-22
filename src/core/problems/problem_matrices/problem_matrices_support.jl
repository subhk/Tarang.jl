"""
    Problem matrix support

This file contains field-size helpers, equation-condition utilities, and
small shared block constructors for problem-level matrix assembly.
"""

# Supporting functions for matrix building

function field_dofs(field::ScalarField)
    # Tau variables (empty bases): always 1 DOF, consistent with
    # _coeff_space_dofs, compute_field_vector_size, extract_field_data_for_vector
    if isempty(field.bases)
        return 1
    end
    if get_coeff_data(field) !== nothing
        return length(get_coeff_data(field))
    elseif get_grid_data(field) !== nothing
        return length(get_grid_data(field))
    else
        total = 1
        for basis in field.bases
            if basis !== nothing
                total *= basis.meta.size
            end
        end
        return total
    end
end

field_dofs(field::VectorField) = sum(field_dofs(comp) for comp in field.components)
field_dofs(field::TensorField) = sum(field_dofs(comp) for comp in vec(field.components))

"""
    _coeff_space_dofs(var)

Compute GLOBAL coefficient-space DOF count from basis metadata, WITHOUT
transforming the field or reading field data (which may be a local MPI slice).

Julia's `rfft` only halves the FIRST RealFourier dimension.  Subsequent
RealFourier axes use regular FFT (full size N).  This matches
`coefficient_shape_mpi` logic and is correct for both serial and MPI modes.
"""
function _coeff_space_dofs(var::ScalarField)
    # Compute from basis metadata — always global, works for serial + MPI
    if !isempty(var.bases)
        total = 1
        first_rf = true
        for basis in var.bases
            if basis !== nothing
                if isa(basis, RealFourier) && first_rf
                    total *= div(basis.meta.size, 2) + 1
                    first_rf = false
                else
                    total *= basis.meta.size
                end
            end
        end
        return total
    end

    # Tau variables (empty bases): always 1 DOF (0D scalar gauge variable).
    # Don't check data — its allocation state varies between matrix-build
    # and stepping, causing off-by-1 mismatches. The fallback in
    # compute_field_vector_size also returns 1 for empty-bases fields.
    return 1
end
_coeff_space_dofs(var::VectorField) = sum(_coeff_space_dofs(c) for c in var.components)
_coeff_space_dofs(var::TensorField) = sum(_coeff_space_dofs(c) for c in vec(var.components))

"""Compute size (degrees of freedom) of field or equation data"""
function compute_field_size(field_or_data)
    if isa(field_or_data, Dict)
        if haskey(field_or_data, "equation_size")
            return field_or_data["equation_size"]
        elseif haskey(field_or_data, "equation_variables")
            vars = field_or_data["equation_variables"]
            if isa(vars, Vector)
                return sum(field_dofs(var) for var in vars)
            end
        elseif haskey(field_or_data, "variables")
            vars = field_or_data["variables"]
            if isa(vars, Vector)
                return sum(field_dofs(var) for var in vars)
            end
        end
        return 0
    elseif isa(field_or_data, ScalarField)
        return field_dofs(field_or_data)
    elseif isa(field_or_data, VectorField) || isa(field_or_data, TensorField)
        return field_dofs(field_or_data)
    elseif hasfield(typeof(field_or_data), :buffers) && get_coeff_data(field_or_data) !== nothing
        return length(get_coeff_data(field_or_data))
    elseif hasfield(typeof(field_or_data), :buffers) && get_grid_data(field_or_data) !== nothing
        return length(get_grid_data(field_or_data))
    else
        return 0
    end
end

"""
    Check if equation should be included in matrix assembly.

    An equation is included if:
    1. It has valid matrix expressions (M, L, or F)
    2. It is marked as enabled (if "enabled" key exists)
    3. It has a valid condition (if "condition" key exists)
    4. It references at least one problem variable
    5. The equation is well-formed (not flagged as invalid)

    Following patterns where equations can be conditionally
    included/excluded based on wavenumber, problem parameters, etc.
    """
function check_equation_condition(eq_data::Dict)

    # Check if equation is explicitly disabled
    if haskey(eq_data, "enabled") && !eq_data["enabled"]
        @debug "Equation excluded: explicitly disabled" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check if equation has a condition function that evaluates to false
    if haskey(eq_data, "condition")
        condition = eq_data["condition"]
        if isa(condition, Bool)
            if !condition
                @debug "Equation excluded: condition is false" eq_index=get(eq_data, "equation_index", 0)
                return false
            end
        elseif isa(condition, Function)
            # Condition is a function - evaluate it
            try
                result = condition(eq_data)
                if !result
                    @debug "Equation excluded: condition function returned false" eq_index=get(eq_data, "equation_index", 0)
                    return false
                end
            catch e
                @warn "Equation condition evaluation failed, including equation" exception=e
            end
        end
    end

    # Check if equation is flagged as invalid
    if get(eq_data, "is_invalid", false)
        @debug "Equation excluded: flagged as invalid" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check if equation has any matrix content
    has_M = haskey(eq_data, "M") && !is_zero_expression(eq_data["M"])
    has_L = haskey(eq_data, "L") && !is_zero_expression(eq_data["L"])
    has_F = haskey(eq_data, "F") && !is_zero_expression(eq_data["F"])

    if !has_M && !has_L && !has_F
        @debug "Equation excluded: no matrix content (M, L, F all zero/missing)" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check equation size
    eq_size = get(eq_data, "equation_size", 0)
    if eq_size <= 0
        @debug "Equation excluded: equation_size <= 0" eq_index=get(eq_data, "equation_index", 0)
        return false
    end

    # Check wavenumber conditions (for spectral problems)
    if haskey(eq_data, "valid_modes")
        valid_modes = eq_data["valid_modes"]
        current_mode = get(eq_data, "current_mode", nothing)
        if current_mode !== nothing && !in(current_mode, valid_modes)
            @debug "Equation excluded: mode not in valid_modes" current_mode valid_modes
            return false
        end
    end

    # Check for wavenumber-based conditions (k=0 special handling, etc.)
    if haskey(eq_data, "exclude_k_zero") && eq_data["exclude_k_zero"]
        wavenumber = get(eq_data, "wavenumber", nothing)
        if wavenumber !== nothing
            # Check if all wavenumber components are zero
            if isa(wavenumber, Number) && wavenumber == 0
                @debug "Equation excluded: k=0 mode excluded" eq_index=get(eq_data, "equation_index", 0)
                return false
            elseif isa(wavenumber, Tuple) && all(k -> k == 0, wavenumber)
                @debug "Equation excluded: k=(0,...,0) mode excluded" eq_index=get(eq_data, "equation_index", 0)
                return false
            end
        end
    end

    # Check for gauge conditions (pressure gauge, etc.)
    if haskey(eq_data, "is_gauge_condition") && eq_data["is_gauge_condition"]
        # Gauge conditions may have special handling
        gauge_mode = get(eq_data, "gauge_mode", nothing)
        current_mode = get(eq_data, "current_mode", nothing)

        if gauge_mode !== nothing && current_mode !== nothing
            if gauge_mode != current_mode
                # Only include gauge condition for specific mode
                return false
            end
        end
    end

    # Check if this is a boundary condition equation
    if haskey(eq_data, "is_boundary_condition") && eq_data["is_boundary_condition"]
        # Boundary conditions are always included if they're valid
        bc_valid = get(eq_data, "bc_valid", true)
        if !bc_valid
            @debug "Equation excluded: boundary condition marked invalid"
            return false
        end
    end

    # All checks passed
    return true
end

"""
    Check if equation data is structurally valid.
    Returns (is_valid::Bool, error_message::Union{String,Nothing})
    """
function is_equation_valid(eq_data::Dict)

    # Must have equation string
    if !haskey(eq_data, "equation_string")
        return (false, "Missing equation_string")
    end

    # Must have LHS
    if !haskey(eq_data, "lhs")
        return (false, "Missing LHS expression")
    end

    # Check for parse errors
    if haskey(eq_data, "parse_error")
        return (false, "Parse error: $(eq_data["parse_error"])")
    end

    # Check LHS structure if we have the expression
    lhs = eq_data["lhs"]
    if lhs !== nothing
        is_valid_lhs, lhs_info = is_proper_lhs_structure(lhs)
        if !is_valid_lhs
            return (false, "Invalid LHS structure: $(lhs_info[:error_message])")
        end
    end

    return (true, nothing)
end

"""
    Set a condition for equation inclusion in matrix assembly.
    """
function set_equation_condition!(eq_data::Dict, condition::Union{Bool, Function})
    eq_data["condition"] = condition
end

"""Enable an equation for matrix assembly."""
function enable_equation!(eq_data::Dict)
    eq_data["enabled"] = true
end

"""Disable an equation from matrix assembly."""
function disable_equation!(eq_data::Dict)
    eq_data["enabled"] = false
end

"""
    Set the valid wavenumber modes for this equation.
    The equation will only be included for these modes.
    """
function set_valid_modes!(eq_data::Dict, modes::Union{Vector, Set, AbstractRange})
    eq_data["valid_modes"] = Set(modes)
end

"""
    Exclude this equation from k=0 (homogeneous) mode.
    Useful for gauge conditions in incompressible flow problems.
    """
function exclude_k_zero!(eq_data::Dict, exclude::Bool=true)
    eq_data["exclude_k_zero"] = exclude
end

"""Get matrix expression from equation data"""
function get_matrix_expression(eq_data::Dict, matrix_name::String)
    return get(eq_data, matrix_name, nothing)
end

"""Check if expression is effectively zero"""
function is_zero_expression(expr)
    return isa(expr, ZeroOperator) || expr === nothing
end

@inline _zero_block(eqn_size::Int, var_size::Int) = spzeros(ComplexF64, eqn_size, var_size)

function _identity_block(eqn_size::Int, var_size::Int; scale::Number=1.0)
    if eqn_size == 0 || var_size == 0
        return _zero_block(eqn_size, var_size)
    end
    diag_len = min(eqn_size, var_size)
    vals = fill(ComplexF64(scale), diag_len)
    return spdiagm(eqn_size, var_size, 0 => vals)
end
