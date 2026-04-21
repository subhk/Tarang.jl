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

    # String conditions - evaluate with group variables
    if isa(condition, String)
        condition_str = strip(condition)

        # Handle compound conditions with && and ||
        if occursin("&&", condition_str)
            parts = split(condition_str, "&&")
            return all(check_condition(strip(p), group_dict) for p in parts)
        elseif occursin("||", condition_str)
            parts = split(condition_str, "||")
            return any(check_condition(strip(p), group_dict) for p in parts)
        end

        # Parse simple comparisons like "nx != 0", "kx == 0"
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
    end

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
