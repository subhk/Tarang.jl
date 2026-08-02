"""
    Forcing-vector construction for the global matrix system.

This file previously also carried `process_lhs_operator!`, `process_rhs_operator!`,
`evaluate_rhs_scalar` and `find_variable_index` under a "kept for compatibility"
banner. They formed a closed cluster that only ever called each other — nothing in
`src/`, `ext/`, `test/`, `docs/` or `scripts/` reached any of them, and none was
exported. They were removed rather than repaired: `evaluate_rhs_scalar` returned a
plausible `0.0` for anything it could not evaluate, which is exactly the
silent-wrong-value shape this project keeps finding, and unreachable code carrying
that shape is a trap for whoever wires it up next.
"""

function build_forcing_vector(problem::Problem, eqn_sizes::Vector{Int}, total_size::Int)
    
    F_vector = zeros(ComplexF64, total_size)
    
    i0 = 0
    for (eq_idx, eq_data) in enumerate(problem.equation_data)
        eqn_size = eqn_sizes[eq_idx]
        if eqn_size > 0
            rhs_expr = get(eq_data, "F", ZeroOperator())
            
            # Evaluate RHS expression to get forcing values
            if isa(rhs_expr, ConstantOperator)
                F_vector[i0+1:i0+eqn_size] .= rhs_expr.value
            elseif isa(rhs_expr, ZeroOperator)
                F_vector[i0+1:i0+eqn_size] .= 0.0
            else
                # Anything that is not a bare constant is written as ZERO here: this
                # builder can place a scalar, not evaluate a field or an expression.
                # That silently drops the forcing, so it must never be the last word —
                # `_global_forcing_dropped_equations` re-derives exactly this condition
                # and `solve_linear!` refuses before consuming the vector. Building it
                # is still correct: most problems solve per-mode and never read it.
                F_vector[i0+1:i0+eqn_size] .= 0.0
            end
        end
        i0 += eqn_size
    end
    
    return F_vector
end


"""
    _global_forcing_dropped_equations(problem) -> Vector{Int}

Equation indices whose right-hand side `build_forcing_vector` wrote as zero because
it is neither a `ConstantOperator` nor a `ZeroOperator`.

The global forcing vector can only carry a scalar per equation block. A field- or
expression-valued RHS is therefore dropped, and the resulting solve is wrong by
exactly that forcing — with no error, since a zero is a perfectly plausible answer.

This re-derives the condition rather than recording it during the build, so the two
cannot drift apart: both read `eq_data["F"]` and ask the same question. Callers that
are about to CONSUME the global vector must check this first; building it is harmless
because the per-mode subproblem path, which most problems take, never reads it.
"""
function _global_forcing_dropped_equations(problem::Problem)
    dropped = Int[]
    for (eq_idx, eq_data) in enumerate(problem.equation_data)
        rhs_expr = get(eq_data, "F", ZeroOperator())
        isa(rhs_expr, ConstantOperator) && continue
        isa(rhs_expr, ZeroOperator) && continue
        push!(dropped, eq_idx)
    end
    return dropped
end
