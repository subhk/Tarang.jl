"""
Tools for equation parsing.
"""

export split_equation, split_call, lambdify_functions

"""
    split_equation(equation::AbstractString)

Split an equation string into left- and right-hand sides, tracking parenthesis
depth so that keyword arguments (e.g. ``f(x=1)``) do not trigger false splits.

Throws `SymbolicParsingError` if there is not exactly one top-level equals sign.
"""
function split_equation(equation::AbstractString)
    parentheses = 0
    top_level_equals = Int[]

    for (i, character) in enumerate(equation)
        if character == '('
            parentheses += 1
        elseif character == ')'
            parentheses -= 1
        elseif character == '=' && parentheses == 0
            push!(top_level_equals, i)
        end
    end

    if isempty(top_level_equals)
        throw(SymbolicParsingError("Equation contains no top-level equals signs: $equation"))
    elseif length(top_level_equals) > 1
        throw(SymbolicParsingError("Equation contains multiple top-level equals signs: $equation"))
    end

    i = top_level_equals[1]
    lhs = strip(equation[1:i-1])
    rhs = strip(equation[i+1:end])
    return lhs, rhs
end

"""
    split_call(call::AbstractString)

Split a function-style string ``"f(x, y)"`` into a head ``"f"`` and a tuple of
argument names. Returns ``(call, ())`` if the string does not have call syntax.
"""
function split_call(call::AbstractString)
    match = match(r"^(.+)\((.*)\)$"s, call)
    if match === nothing
        return call, ()
    else
        head = strip(match.captures[1])
        argstring = strip(match.captures[2])
        if isempty(argstring)
            return head, ()
        else
            args = tuple((strip(arg) for arg in split(argstring, ','))...)
            return head, args
        end
    end
end

"""
    lambdify_functions(call::AbstractString, result::AbstractString)

Convert a math-style definition ``"f(x, y)"``/``"x*y"`` into a Julia anonymous
function encoded as a string ``"(x,y) -> x*y"``. Returns the original result for
non-call statements to preserve standard semantics.
"""
function lambdify_functions(call::AbstractString, result::AbstractString)
    head, args = split_call(call)
    if isempty(args)
        return head, result
    else
        argstring = join(args, ",")
        lambda_expr = "($argstring) -> $result"
        return head, lambda_expr
    end
end
