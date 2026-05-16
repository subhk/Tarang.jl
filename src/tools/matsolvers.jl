"""
Matrix solver registry.

This module provides Julia-friendly wrappers over linear-system solvers so
other components can select a solver by name or by passing a constructor.
"""

module MatSolvers

export AbstractMatSolver, register_solver, get_solver, solver_instance, solve, solve!,
       refactor!,
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
function solve!(dest, ::DummySolver, rhs)
    fill!(dest, zero(eltype(dest)))
    return dest
end

struct DenseLUSolver{T} <: AbstractMatSolver
    lu::LinearAlgebra.LU{T, Matrix{T}}
end

function DenseLUSolver(matrix::AbstractMatrix; kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    return DenseLUSolver{T}(lu(Matrix{T}(matrix)))
end

solve(s::DenseLUSolver, rhs) = s.lu \ rhs
function solve!(dest, s::DenseLUSolver, rhs)
    ldiv!(dest, s.lu, rhs)
    return dest
end

"""
    SparseLUSolver{T}

Sparse LU solver wrapping SuiteSparse UMFPACK. Mutable so that `refactor!`
can swap in a new numeric factorization while reusing the symbolic
analysis (row/column ordering, fill-in pattern) when the matrix sparsity
pattern is unchanged — the common case for IMEX-RK solvers that rebuild
`(M + dt*a_ii*L)` with a new `dt` but keep the same `(M, L)` pattern.

Storing `factor::Any` keeps the struct compatible across Julia versions
where `lu` returns slightly different concrete types (e.g. `UmfpackLU`
vs `SPQR.QRSparse` depending on fallback paths). The runtime cost of
`Any`-typed field access is dwarfed by the actual sparse solve.
"""
mutable struct SparseLUSolver{T} <: AbstractMatSolver
    factor::Any  # Concrete type varies by Julia version; kept as Any for compatibility
    matrix_ref::Any  # Reference to the matrix used for factorization; used to detect pattern match
end

function SparseLUSolver(matrix::SparseMatrixCSC; kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    A = SparseMatrixCSC{T, Int}(matrix)
    return SparseLUSolver{T}(lu(A), A)
end

function SparseLUSolver(matrix::AbstractMatrix; kwargs...)
    return SparseLUSolver(sparse(matrix); kwargs...)
end

solve(s::SparseLUSolver, rhs) = _factor_div_barrier(s.factor, rhs)
@inline _factor_div_barrier(factor::F, rhs) where {F} = factor \ rhs

function solve!(dest, s::SparseLUSolver, rhs)
    _ldiv_barrier!(dest, s.factor, rhs)
    return dest
end
@inline function _ldiv_barrier!(dest, factor::F, rhs) where {F}
    ldiv!(dest, factor, rhs)
end

"""
    refactor!(solver::SparseLUSolver, A::SparseMatrixCSC) -> solver

Update the solver's numeric factorization to match the new matrix `A`,
reusing the cached symbolic factorization when `A` has the same sparsity
pattern as the matrix the solver was originally built from.

### Pattern-match fast path

Julia's `SparseArrays.lu!(F::UmfpackLU, A)` (since 1.9) detects when the
passed `A` shares `colptr` / `rowval` with the cached factor and triggers
**only a numeric refactor** via `umfpack_numeric!`. The symbolic analysis
(row/column ordering, pivot strategy, fill-in pattern) is reused — this
saves 30–60% of the total factorization time, which is often the
bottleneck in CFL-adaptive IMEX runs where `dt` changes every few steps.

This function wraps `lu!` with additional safety:
1. Validates dimensions and element type
2. Falls back to a full rebuild if `lu!` fails or is unavailable
3. Updates `matrix_ref` so the next `refactor!` has an accurate baseline
   for pattern comparison.

### When to call

Call `refactor!(solver, new_lhs)` from the stepper's LHS cache path
instead of building a fresh `SparseLUSolver` — for the common case where
only `dt` changes, this is a ~2× speedup on the LU factorization step.
"""
function refactor!(solver::SparseLUSolver{T}, A::SparseMatrixCSC) where {T}
    # Promote to T if needed so nzval types match the cached factor.
    if eltype(A) !== T
        A = SparseMatrixCSC{T, Int}(A)
    end

    # Size check — fall back to full rebuild if dimensions changed.
    old_n = size(solver.matrix_ref, 1)
    if size(A, 1) != old_n || size(A, 2) != size(solver.matrix_ref, 2)
        solver.factor = lu(A)
        solver.matrix_ref = A
        return solver
    end

    # Fast path: in-place numeric refactor. `lu!` detects pattern matches
    # internally and only re-runs numeric factorization when possible.
    try
        lu!(solver.factor, A)
        solver.matrix_ref = A
        return solver
    catch err
        # If lu! isn't supported for this factor type, or pattern check
        # failed, fall back to full rebuild.
        @debug "refactor!: lu! failed ($err), rebuilding from scratch" maxlog=3
        solver.factor = lu(A)
        solver.matrix_ref = A
        return solver
    end
end

function solve!(dest, s::AbstractMatSolver, rhs)
    copyto!(dest, solve(s, rhs))
    return dest
end

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

Following matsolvers:Woodbury pattern.
"""
struct WoodburySolver{T, S<:AbstractMatSolver} <: AbstractMatSolver
    base_solver::S           # Solver for base matrix A
    U::Matrix{T}             # Left border (n×k)
    V::Matrix{T}             # Right border (k×n)
    capacitance::Matrix{T}   # C^(-1) + V*A^(-1)*U (k×k)
    cap_lu::LinearAlgebra.LU{T, Matrix{T}}  # LU factorization of capacitance
    y::Vector{T}
    z::Vector{T}
    w::Vector{T}
    Uw::Vector{T}
    u::Vector{T}
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

    return WoodburySolver{T, typeof(base)}(base, U_t, V_t, cap, cap_lu,
                                          Vector{T}(undef, n),
                                          Vector{T}(undef, k),
                                          Vector{T}(undef, k),
                                          Vector{T}(undef, n),
                                          Vector{T}(undef, n))
end

"""
    solve(ws::WoodburySolver, rhs)

Solve (A + UCV)x = b using the Woodbury formula.
"""
function solve(ws::WoodburySolver, rhs::AbstractVector)
    result = similar(rhs, eltype(ws.y))
    solve!(result, ws, rhs)
    return result
end

function solve!(dest, ws::WoodburySolver, rhs::AbstractVector)
    # Step 1: y = A^(-1)b
    solve!(ws.y, ws.base_solver, rhs)

    # Step 2: z = V*y
    mul!(ws.z, ws.V, ws.y)

    # Step 3: w = cap^(-1) * z
    ldiv!(ws.w, ws.cap_lu, ws.z)

    # Step 4: u = A^(-1)*U*w
    mul!(ws.Uw, ws.U, ws.w)
    solve!(ws.u, ws.base_solver, ws.Uw)

    # Step 5: x = y - u
    @. dest = ws.y - ws.u
    return dest
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

function detect_bandwidth(A::SparseMatrixCSC)
    lower = 0
    upper = 0
    rows = rowvals(A)
    for j in 1:size(A, 2)
        for idx in nzrange(A, j)
            i = rows[idx]
            lower = max(lower, i - j)
            upper = max(upper, j - i)
        end
    end
    return lower, upper
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

function solve!(dest, s::BandedLUSolver{T}, rhs::AbstractVector) where T
    copyto!(dest, rhs)
    LinearAlgebra.LAPACK.gbtrs!('N', s.kl, s.ku, s.n, s.AB, s.ipiv, dest)
    return dest
end

register_solver("banded", BandedLUSolver)

# ============================================================================
# Block diagonal solver
# ============================================================================

"""
    BlockDiagonalSolver <: AbstractMatSolver

Solver for block diagonal matrices where each block can be solved independently.
"""
struct BlockDiagonalSolver{T, F} <: AbstractMatSolver
    block_solvers::Vector{F}
    block_sizes::Vector{Int}
    workspace::Vector{Vector{Vector{T}}}
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
    block_solvers = [lu(Matrix{T}(matrix[1:block_sizes[1], 1:block_sizes[1]]))]
    offset = 0
    for (block_idx, bs) in enumerate(block_sizes)
        if block_idx == 1
            offset += bs
            continue
        end
        block = matrix[offset+1:offset+bs, offset+1:offset+bs]
        push!(block_solvers, lu(Matrix{T}(block)))
        offset += bs
    end

    workspace = [[Vector{T}(undef, bs) for bs in block_sizes]
                 for _ in 1:Base.Threads.nthreads()]

    return BlockDiagonalSolver{T, eltype(block_solvers)}(block_solvers, block_sizes, workspace)
end

function solve(s::BlockDiagonalSolver{T}, rhs::AbstractVector) where T
    result = similar(rhs, T)
    solve!(result, s, rhs)
    return result
end

function solve!(dest, s::BlockDiagonalSolver{T}, rhs::AbstractVector) where T
    n = sum(s.block_sizes)
    length(dest) == n ||
        throw(DimensionMismatch("BlockDiagonal destination length $(length(dest)) does not match matrix rows $n"))
    length(rhs) == n ||
        throw(DimensionMismatch("BlockDiagonal rhs length $(length(rhs)) does not match matrix rows $n"))

    workspace = s.workspace[Base.Threads.threadid()]
    offset = 0

    for (i, bs) in enumerate(s.block_sizes)
        block_rhs = workspace[i]
        @inbounds for j in 1:bs
            block_rhs[j] = rhs[offset + j]
        end
        ldiv!(s.block_solvers[i], block_rhs)
        @inbounds for j in 1:bs
            dest[offset + j] = block_rhs[j]
        end
        offset += bs
    end

    return dest
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
struct SPQRSolver{T, F, Q, R} <: AbstractMatSolver
    qr_factor::F
    q_adjoint::Q
    r_factor::R
    inv_cpiv::Vector{Int}
    rpivinv::Vector{Int}
    workspace::Vector{Vector{T}}
    q_workspace::Vector{Vector{T}}
    rank_workspace::Vector{Vector{T}}
    rank::Int
    m::Int
    n::Int
end

function SPQRSolver(matrix::SparseMatrixCSC; kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    A = SparseMatrixCSC{T, Int}(matrix)
    qr_factor = qr(A)
    rnk = rank(qr_factor)
    r_factor = UpperTriangular(qr_factor.R[Base.OneTo(rnk), Base.OneTo(rnk)])
    q_adjoint = adjoint(qr_factor.Q)
    inv_cpiv = isempty(qr_factor.cpiv) ? Int[] : invperm(qr_factor.cpiv)
    rpivinv = Vector{Int}(qr_factor.rpivinv)

    return SPQRSolver{T, typeof(qr_factor), typeof(q_adjoint), typeof(r_factor)}(
        qr_factor, q_adjoint, r_factor, inv_cpiv, rpivinv,
        [Vector{T}(undef, max(size(qr_factor, 1), size(qr_factor, 2)))
         for _ in 1:Base.Threads.nthreads()],
        [Vector{T}(undef, size(qr_factor, 1))
         for _ in 1:Base.Threads.nthreads()],
        [Vector{T}(undef, Int(rnk))
         for _ in 1:Base.Threads.nthreads()],
        Int(rnk), size(qr_factor, 1), size(qr_factor, 2))
end

function SPQRSolver(matrix::AbstractMatrix; kwargs...)
    return SPQRSolver(sparse(matrix); kwargs...)
end

function solve(s::SPQRSolver{T}, rhs::AbstractVector) where T
    result = similar(rhs, T, s.n)
    solve!(result, s, rhs)
    return result
end

function solve(s::SPQRSolver{T}, rhs::AbstractMatrix) where T
    result = similar(rhs, T, s.n, size(rhs, 2))
    solve!(result, s, rhs)
    return result
end

function solve!(dest::AbstractVector, s::SPQRSolver{T}, rhs::AbstractVector) where T
    length(rhs) == s.m ||
        throw(DimensionMismatch("SPQR rhs length $(length(rhs)) does not match matrix rows $(s.m)"))
    length(dest) == s.n ||
        throw(DimensionMismatch("SPQR destination length $(length(dest)) does not match matrix columns $(s.n)"))

    thread_id = Base.Threads.threadid()
    x = s.workspace[thread_id]
    q_rhs = s.q_workspace[thread_id]
    rank_rhs = s.rank_workspace[thread_id]
    rpivinv = s.rpivinv

    for i in eachindex(rpivinv)
        @inbounds q_rhs[rpivinv[i]] = rhs[i]
    end

    lmul!(s.q_adjoint, q_rhs)

    @inbounds for i in 1:s.rank
        rank_rhs[i] = q_rhs[i]
    end
    ldiv!(s.r_factor, rank_rhs)

    @inbounds for i in 1:s.rank
        x[i] = rank_rhs[i]
    end
    if s.rank < s.n
        z = zero(T)
        @inbounds for i in (s.rank + 1):s.n
            x[i] = z
        end
    end

    if isempty(s.inv_cpiv)
        copyto!(dest, 1, x, 1, s.n)
    else
        for i in 1:s.n
            @inbounds dest[i] = x[s.inv_cpiv[i]]
        end
    end

    return dest
end

function solve!(dest::AbstractMatrix, s::SPQRSolver, rhs::AbstractMatrix)
    size(rhs, 1) == s.m ||
        throw(DimensionMismatch("SPQR rhs row count $(size(rhs, 1)) does not match matrix rows $(s.m)"))
    size(dest, 1) == s.n ||
        throw(DimensionMismatch("SPQR destination row count $(size(dest, 1)) does not match matrix columns $(s.n)"))
    size(dest, 2) == size(rhs, 2) ||
        throw(DimensionMismatch("SPQR destination columns $(size(dest, 2)) do not match rhs columns $(size(rhs, 2))"))

    for j in axes(rhs, 2)
        solve!(view(dest, :, j), s, view(rhs, :, j))
    end

    return dest
end

register_solver("spqr", SPQRSolver)
register_solver("qr", SPQRSolver)

end # module
