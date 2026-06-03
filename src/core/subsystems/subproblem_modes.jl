"""
    check_condition(sp::Subproblem, eq_data)

Check if equation condition is satisfied for this subproblem.
Following subsystems:494-495.
"""
function check_condition(sp::Subproblem, eq_data::Dict)
    condition = get(eq_data, "condition", "true")
    if condition == "true" || condition === nothing || condition == true
        return true
    end
    if condition == "false" || condition == false
        return false
    end

    # Evaluate condition expression with subproblem group dictionary
    # The condition is typically a string expression involving group indices
    # For example: "nx != 0" or "kx == 0 && ky == 0"
    group_dict = sp.group_dict

    # Simple boolean conditions
    if isa(condition, Bool)
        return condition
    end

    # String conditions - evaluate against the subproblem's group dictionary
    if isa(condition, String)
        return _eval_condition_str(strip(condition), group_dict)
    end

    return true
end

"""
    _eval_condition_str(condition_str, group_dict) -> Bool

Evaluate an equation condition expression (e.g. `"nx != 0"`, `"kx == 0 && ky == 0"`)
against a subproblem's group dictionary. Supports `&&`/`||` compounds (recursively)
and simple `==`/`!=` integer comparisons keyed by group-variable name. Unparseable
expressions default to `true` (equation included) with a loud warning.
"""
function _eval_condition_str(condition_str::AbstractString, group_dict)
    condition_str = strip(condition_str)

    # Compound conditions — recurse on each part (NOTE: recurse on the string
    # evaluator, not check_condition, which expects a Subproblem).
    if occursin("&&", condition_str)
        return all(_eval_condition_str(p, group_dict) for p in split(condition_str, "&&"))
    elseif occursin("||", condition_str)
        return any(_eval_condition_str(p, group_dict) for p in split(condition_str, "||"))
    end

    # Simple comparisons like "nx != 0", "kx == 0"
    if occursin("!=", condition_str)
        parts = split(condition_str, "!=")
        if length(parts) == 2
            var_name = strip(parts[1])
            value = tryparse(Int, strip(parts[2]))
            if value !== nothing && haskey(group_dict, var_name)
                return group_dict[var_name] != value
            elseif value !== nothing && haskey(group_dict, Symbol(var_name))
                return group_dict[Symbol(var_name)] != value
            end
        end
    elseif occursin("==", condition_str)
        parts = split(condition_str, "==")
        if length(parts) == 2
            var_name = strip(parts[1])
            value = tryparse(Int, strip(parts[2]))
            if value !== nothing && haskey(group_dict, var_name)
                return group_dict[var_name] == value
            elseif value !== nothing && haskey(group_dict, Symbol(var_name))
                return group_dict[Symbol(var_name)] == value
            end
        end
    end

    # Unparseable condition — warn loudly rather than silently assuming true
    @warn "Could not parse condition expression: '$condition_str'. Assuming true (equation included)." maxlog=3
    return true
end

"""
    valid_modes(sp::Subproblem, field, valid_modes_array)

Get valid modes for field in this subproblem.
Following subsystems:476-478.
"""
function valid_modes(sp::Subproblem, field, valid_modes_array)
    if valid_modes_array === nothing
        # All modes valid by default
        return ones(Bool, field_size(sp, field))
    end
    slices = field_slices(sp, field)
    return valid_modes_array[slices...]
end
