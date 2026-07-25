"""Typed equation and compiled-problem state owned by the problem layer."""

abstract type Problem end

"""
    EquationIR

Named intermediate representation for one parsed equation. The common matrix
and RHS slots are real fields, so internal code no longer depends on unchecked
string keys. `AbstractDict` compatibility is retained temporarily for existing
extensions and downstream code; uncommon metadata lives in `metadata`.
"""
mutable struct EquationIR <: AbstractDict{String, Any}
    mass::Any
    linear::Any
    forcing::Any
    forcing_expr::Any
    lhs::Any
    equation_size::Int
    metadata::Dict{String, Any}
end

EquationIR() = EquationIR(nothing, nothing, nothing, nothing, nothing, 0,
                          Dict{String, Any}())

function EquationIR(values::AbstractDict)
    ir = EquationIR()
    for (key, value) in values
        ir[string(key)] = value
    end
    return ir
end

# `Vector{EquationIR}` calls `convert` when legacy downstream code pushes a
# `Dict`. Julia's generic AbstractDict conversion tries to reconstruct this
# fixed-key compatibility wrapper and can report a spurious key collision.
Base.convert(::Type{EquationIR}, values::AbstractDict) = EquationIR(values)

@inline function _equation_ir_field(key::String)
    key == "M" && return :mass
    key == "L" && return :linear
    key == "F" && return :forcing
    key == "F_expr" && return :forcing_expr
    key == "lhs" && return :lhs
    key == "equation_size" && return :equation_size
    return nothing
end

function Base.getindex(ir::EquationIR, key::String)
    field = _equation_ir_field(key)
    field === nothing && return ir.metadata[key]
    return getfield(ir, field)
end

function Base.setindex!(ir::EquationIR, value, key::String)
    field = _equation_ir_field(key)
    if field === nothing
        ir.metadata[key] = value
    elseif field === :equation_size
        setfield!(ir, field, Int(value))
    else
        setfield!(ir, field, value)
    end
    return value
end

Base.haskey(ir::EquationIR, key::String) =
    _equation_ir_field(key) !== nothing || haskey(ir.metadata, key)

Base.get(ir::EquationIR, key::String, default) =
    haskey(ir, key) ? ir[key] : default

function Base.delete!(ir::EquationIR, key::String)
    field = _equation_ir_field(key)
    if field === nothing
        delete!(ir.metadata, key)
    elseif field === :equation_size
        ir.equation_size = 0
    else
        setfield!(ir, field, nothing)
    end
    return ir
end

const _EQUATION_IR_KEYS = ("M", "L", "F", "F_expr", "lhs", "equation_size")

Base.length(ir::EquationIR) = length(_EQUATION_IR_KEYS) + length(ir.metadata)
Base.keys(ir::EquationIR) = (collect(_EQUATION_IR_KEYS)..., keys(ir.metadata)...)

function Base.iterate(ir::EquationIR, state::Int=1)
    all_keys = keys(ir)
    state > length(all_keys) && return nothing
    key = all_keys[state]
    return key => ir[key], state + 1
end

"""
    RuntimeCacheContext

Problem-scoped caches whose lifetime is tied to compiled solver artifacts.
This prevents temporary arrays and transforms from leaking through the public
parameter dictionary or being shared accidentally between independent runs.
"""
mutable struct RuntimeCacheContext
    bc_rfft::IdDict{Any, Any}
    workspaces::Dict{Symbol, Any}
end

RuntimeCacheContext() = RuntimeCacheContext(IdDict{Any, Any}(),
                                            Dict{Symbol, Any}())

function Base.empty!(context::RuntimeCacheContext)
    empty!(context.bc_rfft)
    empty!(context.workspaces)
    return context
end

"""
    CompiledProblem

Solver-owned artifacts produced from a problem formulation. Keeping matrices,
subproblems, and runtime caches here separates compilation state from user
parameters and gives it an explicit lifecycle.
"""
mutable struct CompiledProblem
    linear_matrix::Union{Nothing, AbstractMatrix}
    mass_matrix::Union{Nothing, AbstractMatrix}
    forcing_vector::Union{Nothing, AbstractVector}
    subsystems::Union{Nothing, Tuple}
    subproblems::Union{Nothing, Tuple}
    coeff_system::Any
    caches::RuntimeCacheContext
end

CompiledProblem() = CompiledProblem(nothing, nothing, nothing, nothing, nothing,
                                    nothing, RuntimeCacheContext())

compiled_problem(problem::Problem) = problem.compiled

function set_compiled_matrices!(problem::Problem, linear, mass, forcing=nothing)
    compiled = compiled_problem(problem)
    compiled.linear_matrix = linear
    compiled.mass_matrix = mass
    compiled.forcing_vector = forcing

    # One-release compatibility mirror. Internal code reads `compiled`; this
    # keeps existing downstream inspection of `problem.parameters` working.
    problem.parameters["L_matrix"] = linear
    problem.parameters["M_matrix"] = mass
    forcing === nothing || (problem.parameters["F_vector"] = forcing)
    return compiled
end

function set_compiled_subproblems!(problem::Problem, subproblems;
                                   subsystems=nothing, coeff_system=nothing)
    compiled = compiled_problem(problem)
    compiled.subproblems = subproblems
    subsystems === nothing || (compiled.subsystems = subsystems)
    coeff_system === nothing || (compiled.coeff_system = coeff_system)
    problem.parameters["subproblems"] = subproblems
    return compiled
end

compiled_subproblems(problem::Problem) = compiled_problem(problem).subproblems

function reset_compiled_problem!(problem::Problem)
    compiled = compiled_problem(problem)
    compiled.linear_matrix = nothing
    compiled.mass_matrix = nothing
    compiled.forcing_vector = nothing
    compiled.subsystems = nothing
    compiled.subproblems = nothing
    compiled.coeff_system = nothing
    empty!(compiled.caches)
    for key in ("L_matrix", "M_matrix", "F_vector", "subproblems")
        pop!(problem.parameters, key, nothing)
    end
    return compiled
end
