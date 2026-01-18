"""
Tools for equation parsing.
"""

export split_equation, split_call, split_arguments, lambdify_functions

"""
    split_equation(equation::AbstractString)

Split an equation string into left- and right-hand sides, tracking bracket
depth so that keyword arguments (e.g. ``f(x=1)``) and array indices do not
trigger false splits.

Tracks parentheses `()`, square brackets `[]`, and curly braces `{}`.

Throws `SymbolicParsingError` if there is not exactly one top-level equals sign.
"""
function split_equation(equation::AbstractString)
    stack = Char[]  # Track bracket types
    top_level_equals = Int[]  # Store byte indices for proper Unicode handling

    # Use eachindex to get proper byte indices for Unicode strings
    idx = firstindex(equation)
    while idx <= lastindex(equation)
        character = equation[idx]
        if character == '(' || character == '[' || character == '{'
            push!(stack, character)
        elseif character == ')' || character == ']' || character == '}'
            if isempty(stack)
                throw(SymbolicParsingError("Unmatched closing bracket in equation: $equation"))
            end
            opener = pop!(stack)
            if (character == ')' && opener != '(') ||
               (character == ']' && opener != '[') ||
               (character == '}' && opener != '{')
                throw(SymbolicParsingError("Mismatched brackets in equation: $equation"))
            end
        elseif character == '=' && isempty(stack)
            push!(top_level_equals, idx)
        end
        idx = nextind(equation, idx)
    end

    if !isempty(stack)
        throw(SymbolicParsingError("Unmatched opening bracket in equation: $equation"))
    end

    if isempty(top_level_equals)
        throw(SymbolicParsingError("Equation contains no top-level equals signs: $equation"))
    elseif length(top_level_equals) > 1
        throw(SymbolicParsingError("Equation contains multiple top-level equals signs: $equation"))
    end

    i = top_level_equals[1]
    lhs = strip(equation[1:prevind(equation, i)])
    rhs = strip(equation[nextind(equation, i):end])
    return lhs, rhs
end

"""
    split_call(call::AbstractString)

Split a function-style string ``"f(x, y)"`` into a head ``"f"`` and a tuple of
argument names. Returns ``(call, ())`` if the string does not have call syntax.

Handles nested parentheses correctly, e.g., ``"f(g(x, y), z)"`` splits into
``("f", ("g(x, y)", "z"))``.
"""
function split_call(call::AbstractString)
    last_non_ws = findlast(ch -> !isspace(ch), call)
    last_non_ws === nothing && return call, ()

    stack = Char[]
    first_open = nothing
    first_close = nothing

    idx = firstindex(call)
    while idx <= lastindex(call)
        character = call[idx]
        if character == '(' || character == '[' || character == '{'
            if isempty(stack) && first_open === nothing && character == '('
                first_open = idx
            end
            push!(stack, character)
        elseif character == ')' || character == ']' || character == '}'
            if isempty(stack)
                throw(SymbolicParsingError("Unmatched closing bracket in call: $call"))
            end
            opener = pop!(stack)
            if (character == ')' && opener != '(') ||
               (character == ']' && opener != '[') ||
               (character == '}' && opener != '{')
                throw(SymbolicParsingError("Mismatched brackets in call: $call"))
            end
            if isempty(stack) && first_open !== nothing && first_close === nothing && character == ')'
                first_close = idx
            end
        end
        idx = nextind(call, idx)
    end

    if !isempty(stack)
        throw(SymbolicParsingError("Unmatched opening bracket in call: $call"))
    end

    if first_open === nothing || first_close === nothing || first_close != last_non_ws
        return call, ()
    end

    head = strip(call[1:prevind(call, first_open)])
    if isempty(head)
        return call, ()
    end

    argstring = strip(call[nextind(call, first_open):prevind(call, first_close)])
    if isempty(argstring)
        return head, ()
    end

    # Split arguments respecting parenthesis depth
    args = split_arguments(argstring)
    return head, Tuple(args)
end

"""
    split_arguments(argstring::AbstractString) -> Vector{String}

Split a comma-separated argument string respecting parenthesis depth.
E.g., ``"g(x, y), z"`` splits into ``["g(x, y)", "z"]``.
"""
function split_arguments(argstring::AbstractString)
    args = String[]
    current_arg = IOBuffer()
    stack = Char[]

    idx = firstindex(argstring)
    while idx <= lastindex(argstring)
        char = argstring[idx]
        if char == '(' || char == '[' || char == '{'
            push!(stack, char)
            write(current_arg, char)
        elseif char == ')' || char == ']' || char == '}'
            if isempty(stack)
                throw(SymbolicParsingError("Unmatched closing bracket in arguments: $argstring"))
            end
            opener = pop!(stack)
            if (char == ')' && opener != '(') ||
               (char == ']' && opener != '[') ||
               (char == '}' && opener != '{')
                throw(SymbolicParsingError("Mismatched brackets in arguments: $argstring"))
            end
            write(current_arg, char)
        elseif char == ',' && isempty(stack)
            # Top-level comma - end of argument
            arg = strip(String(take!(current_arg)))
            if !isempty(arg)
                push!(args, arg)
            end
        else
            write(current_arg, char)
        end
        idx = nextind(argstring, idx)
    end

    if !isempty(stack)
        throw(SymbolicParsingError("Unmatched opening bracket in arguments: $argstring"))
    end

    # Don't forget the last argument
    last_arg = strip(String(take!(current_arg)))
    if !isempty(last_arg)
        push!(args, last_arg)
    end

    return args
end

"""
    lambdify_functions(call::AbstractString, result::AbstractString)

Convert a math-style definition ``"f(x, y)"``/``"x*y"`` into a Julia anonymous
function encoded as a string ``"(x,y) -> x*y"``. Returns the original result for
non-call statements to preserve standard semantics.
"""
function lambdify_functions(call::AbstractString, result::AbstractString)
    head, args = split_call(call)
    # Check if this is actually a function call by comparing head to original
    # For "variable", head == "variable" (not a call)
    # For "f(x)", head == "f" != "f(x)" (is a call)
    # For "f()", head == "f" != "f()" (is a zero-arg call)
    if strip(head) == strip(call)
        # Not a call - return unchanged
        return head, result
    else
        # Is a call - create lambda
        argstring = join(args, ",")
        lambda_expr = "($argstring) -> $result"
        return head, lambda_expr
    end
end
