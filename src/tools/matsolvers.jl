"""
Matrix solver registry.

This module provides Julia-friendly wrappers over linear-system solvers so
other components can select a solver by name or by passing a constructor.
"""

module MatSolvers

export AbstractMatSolver, register_solver, get_solver, solver_instance, solve,
       DummySolver, DenseLUSolver, SparseLUSolver, WoodburySolver,
       BandedLUSolver, BlockDiagonalSolver, SPQRSolver, SOLVER_REGISTRY

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
    factor::Any  # Use a generic factorization type for compatibility across Julia versions
end

function SparseLUSolver(matrix::SparseMatrixCSC; kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    return SparseLUSolver{T}(lu(SparseMatrixCSC{T, Int}(matrix)))
end

function SparseLUSolver(matrix::AbstractMatrix; kwargs...)
    return SparseLUSolver(sparse(matrix); kwargs...)
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
    elseif name_or_type isa Symbol
        # Convert Symbol to String and lookup
        solver_type = get(SOLVER_REGISTRY, lowercase(String(name_or_type)), nothing)
        solver_type === nothing && throw(ArgumentError("Unknown matrix solver: $(name_or_type)"))
        return solver_type
    elseif name_or_type isa Type && name_or_type <: AbstractMatSolver
        return name_or_type
    else
        throw(ArgumentError("Unsupported solver reference: $(name_or_type)"))
    end
end

function solver_instance(name_or_type, matrix; kwargs...)
    solver_type = get_solver(name_or_type)
    return solver_type(matrix; kwargs...)
end

# ============================================================================
# Woodbury Solver for bordered/augmented systems
# ============================================================================

"""
    WoodburySolver <: AbstractMatSolver

Solver for bordered matrices using the Woodbury matrix identity.
Efficiently solves systems of the form (A + UCV)x = b where:
- A is the base matrix (sparse, factorizable)
- U is n×k (k << n)
- C is k×k
- V is k×n

The Woodbury formula: (A + UCV)^(-1) = A^(-1) - A^(-1)U(C^(-1) + VA^(-1)U)^(-1)VA^(-1)

This is useful for:
- Bordered systems arising from boundary conditions
- Low-rank updates to sparse matrices
- Tau formulations in spectral methods

Following Dedalus matsolvers:Woodbury pattern.
"""
struct WoodburySolver{T, S<:AbstractMatSolver} <: AbstractMatSolver
    base_solver::S           # Solver for base matrix A
    U::Matrix{T}             # Left border (n×k)
    V::Matrix{T}             # Right border (k×n)
    capacitance::Matrix{T}   # C^(-1) + V*A^(-1)*U (k×k)
    cap_lu::LinearAlgebra.LU{T, Matrix{T}}  # LU factorization of capacitance
end

"""
    WoodburySolver(A, U, V; C=I, base_solver=:sparse)

Construct a Woodbury solver for the bordered matrix A + UCV.

# Arguments
- `A`: Base matrix (n×n)
- `U`: Left border matrix (n×k)
- `V`: Right border matrix (k×n)
- `C`: Center matrix (k×k), default is identity
- `base_solver`: Solver type for base matrix A

# Example
```julia
# Solve (A + uv')x = b where u, v are vectors
solver = WoodburySolver(A, reshape(u, :, 1), reshape(v, 1, :))
x = solve(solver, b)
```
"""
function WoodburySolver(A::AbstractMatrix, U::AbstractMatrix, V::AbstractMatrix;
                        C::Union{Nothing, AbstractMatrix}=nothing,
                        base_solver=:sparse)
    T = promote_type(eltype(A), eltype(U), eltype(V), ComplexF64)
    n = size(A, 1)
    k = size(U, 2)

    @assert size(A, 2) == n "A must be square"
    @assert size(U, 1) == n "U must have n rows"
    @assert size(V, 2) == n "V must have n columns"
    @assert size(V, 1) == k "V must have k rows matching U columns"

    # Convert to appropriate types
    U_t = Matrix{T}(U)
    V_t = Matrix{T}(V)

    # Build base solver
    base = solver_instance(base_solver, A)

    # Compute A^(-1)U by solving A * (A^(-1)U) = U
    AinvU = zeros(T, n, k)
    for j in 1:k
        AinvU[:, j] = solve(base, U_t[:, j])
    end

    # Compute capacitance matrix: C^(-1) + V*A^(-1)*U
    if C === nothing
        cap = Matrix{T}(I, k, k) + V_t * AinvU
    else
        C_t = Matrix{T}(C)
        cap = inv(C_t) + V_t * AinvU
    end

    # Factorize capacitance matrix
    cap_lu = lu(cap)

    return WoodburySolver{T, typeof(base)}(base, U_t, V_t, cap, cap_lu)
end

"""
    solve(ws::WoodburySolver, rhs)

Solve (A + UCV)x = b using the Woodbury formula.
"""
function solve(ws::WoodburySolver, rhs::AbstractVector)
    # Step 1: y = A^(-1)b
    y = solve(ws.base_solver, rhs)

    # Step 2: z = V*y
    z = ws.V * y

    # Step 3: w = cap^(-1) * z
    w = ws.cap_lu \ z

    # Step 4: u = A^(-1)*U*w
    Uw = ws.U * w
    u = solve(ws.base_solver, Uw)

    # Step 5: x = y - u
    return y - u
end

# Note: WoodburySolver is NOT registered in the solver registry because it requires
# special constructor arguments (A, U, V). It should be constructed directly:
#   solver = WoodburySolver(A, U, V)
# rather than through solver_instance(:woodbury, matrix)

# ============================================================================
# Banded solvers
# ============================================================================

"""
    BandedLUSolver <: AbstractMatSolver

LU factorization for banded matrices using LAPACK's banded routines (gbtrf/gbtrs).
More efficient than general sparse for problems with limited bandwidth.

The matrix is stored in LAPACK's banded format for efficient factorization and solve.
"""
struct BandedLUSolver{T} <: AbstractMatSolver
    AB::Matrix{T}           # Banded storage format (2*kl + ku + 1) × n
    n::Int                  # Matrix dimension
    kl::Int                 # Number of subdiagonals (lower bandwidth)
    ku::Int                 # Number of superdiagonals (upper bandwidth)
    ipiv::Vector{LinearAlgebra.BlasInt}  # Pivot indices from factorization
end

function BandedLUSolver(matrix::AbstractMatrix;
                        lower_bandwidth::Int=-1,
                        upper_bandwidth::Int=-1,
                        kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)
    @assert size(matrix, 2) == n "Matrix must be square"

    # Auto-detect bandwidth if not provided
    if lower_bandwidth < 0 || upper_bandwidth < 0
        lower_bandwidth, upper_bandwidth = detect_bandwidth(matrix)
    end
    kl = lower_bandwidth
    ku = upper_bandwidth

    # Convert to LAPACK banded format
    # LAPACK requires (2*kl + ku + 1) rows for gbtrf (extra kl rows for fill-in during pivoting)
    ldab = 2*kl + ku + 1
    AB = zeros(T, ldab, n)

    # Fill the banded storage: element A[i,j] goes to AB[kl + ku + 1 + i - j, j]
    for j in 1:n
        for i in max(1, j - ku):min(n, j + kl)
            AB[kl + ku + 1 + i - j, j] = matrix[i, j]
        end
    end

    # Perform LU factorization using LAPACK gbtrf
    # gbtrf!(kl, ku, m, AB) returns (AB, ipiv)
    AB, ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, n, AB)

    return BandedLUSolver{T}(AB, n, kl, ku, ipiv)
end

function detect_bandwidth(A::AbstractMatrix)
    n = size(A, 1)
    lower = 0
    upper = 0

    for j in 1:size(A, 2)
        for i in 1:size(A, 1)
            if !iszero(A[i, j])
                lower = max(lower, i - j)
                upper = max(upper, j - i)
            end
        end
    end

    return lower, upper
end

function solve(s::BandedLUSolver{T}, rhs::AbstractVector) where T
    # Convert rhs to appropriate type and make a copy (LAPACK overwrites)
    x = Vector{T}(rhs)

    # Solve using LAPACK gbtrs (banded triangular solve)
    # gbtrs!(trans, kl, ku, m, AB, ipiv, B) returns B
    LinearAlgebra.LAPACK.gbtrs!('N', s.kl, s.ku, s.n, s.AB, s.ipiv, x)

    return x
end

register_solver("banded", BandedLUSolver)

# ============================================================================
# Block diagonal solver
# ============================================================================

"""
    BlockDiagonalSolver <: AbstractMatSolver

Solver for block diagonal matrices where each block can be solved independently.
"""
struct BlockDiagonalSolver{T} <: AbstractMatSolver
    block_solvers::Vector{Any}
    block_sizes::Vector{Int}
end

function BlockDiagonalSolver(matrix::AbstractMatrix;
                             block_sizes::Vector{Int}=Int[],
                             kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)

    if isempty(block_sizes)
        # Assume single block
        block_sizes = [n]
    end

    @assert sum(block_sizes) == n "Block sizes must sum to matrix dimension"

    # Extract and factorize each block
    block_solvers = Any[]
    offset = 0
    for bs in block_sizes
        block = matrix[offset+1:offset+bs, offset+1:offset+bs]
        push!(block_solvers, lu(Matrix{T}(block)))
        offset += bs
    end

    return BlockDiagonalSolver{T}(block_solvers, block_sizes)
end

function solve(s::BlockDiagonalSolver{T}, rhs::AbstractVector) where T
    result = similar(rhs, T)
    offset = 0

    for (i, bs) in enumerate(s.block_sizes)
        rhs_block = rhs[offset+1:offset+bs]
        result[offset+1:offset+bs] = s.block_solvers[i] \ rhs_block
        offset += bs
    end

    return result
end

register_solver("block", BlockDiagonalSolver)
register_solver("blockdiagonal", BlockDiagonalSolver)

# ============================================================================
# SPQR (SuiteSparse QR) solver
# ============================================================================

"""
    SPQRSolver <: AbstractMatSolver

QR-based sparse solver using SuiteSparse SPQR.
Good for overdetermined or rank-deficient systems.
"""
struct SPQRSolver{T} <: AbstractMatSolver
    qr_factor::Any
end

function SPQRSolver(matrix::SparseMatrixCSC; kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    return SPQRSolver{T}(qr(SparseMatrixCSC{T, Int}(matrix)))
end

function SPQRSolver(matrix::AbstractMatrix; kwargs...)
    return SPQRSolver(sparse(matrix); kwargs...)
end

solve(s::SPQRSolver, rhs) = s.qr_factor \ rhs

register_solver("spqr", SPQRSolver)
register_solver("qr", SPQRSolver)

end # module
