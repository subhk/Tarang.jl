"""
Matrix solver registry inspired by ``dedalus/libraries/matsolvers.py``.

This module provides Julia-friendly wrappers over linear-system solvers so
other components can select a solver by name or by passing a constructor.
"""

module MatSolvers

export AbstractMatSolver, register_solver, get_solver, solver_instance, solve,
       DummySolver, DenseLUSolver, SparseLUSolver, SOLVER_REGISTRY

using SparseArrays
using LinearAlgebra
using SuiteSparse

const SOLVER_REGISTRY = Dict{String, Type}()

abstract type AbstractMatSolver end

struct DummySolver <: AbstractMatSolver
end

function DummySolver(matrix::AbstractMatrix; kwargs...)
    return DummySolver()
end

solve(::DummySolver, rhs) = zero(eltype(rhs)) .* rhs

struct DenseLUSolver{T} <: AbstractMatSolver
    lu::LinearAlgebra.LU{T, Matrix{T}}
end

function DenseLUSolver(matrix::AbstractMatrix; kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    return DenseLUSolver{T}(lu(Matrix{T}(matrix)))
end

solve(s::DenseLUSolver, rhs) = s.lu \ rhs

struct SparseLUSolver{T} <: AbstractMatSolver
    factor::SuiteSparse.UMFPACK.UmfpackFactorization{T, Int}
end

function SparseLUSolver(matrix::SparseMatrixCSC; kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    return SparseLUSolver{T}(lu(SparseMatrixCSC{T, Int}(matrix)))
end

solve(s::SparseLUSolver, rhs) = s.factor \ rhs

function register_solver(name::AbstractString, solver_type::Type)
    SOLVER_REGISTRY[lowercase(String(name))] = solver_type
    return solver_type
end

register_solver("dummy", DummySolver)
register_solver("dense", DenseLUSolver)
register_solver("lu", DenseLUSolver)
register_solver("sparse", SparseLUSolver)

function get_solver(name_or_type)
    if name_or_type isa AbstractString
        solver_type = get(SOLVER_REGISTRY, lowercase(String(name_or_type)), nothing)
        solver_type === nothing && throw(ArgumentError("Unknown matrix solver: $(name_or_type)"))
        return solver_type
    elseif name_or_type <: AbstractMatSolver
        return name_or_type
    else
        throw(ArgumentError("Unsupported solver reference: $(name_or_type)"))
    end
end

function solver_instance(name_or_type, matrix; kwargs...)
    solver_type = get_solver(name_or_type)
    return solver_type(matrix; kwargs...)
end

end # module
