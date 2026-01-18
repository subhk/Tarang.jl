"""
GPU Matrix Solvers for Tarang.jl

This module provides GPU-accelerated linear solvers using CUDA.jl.
These can be beneficial for:
- Large problems (>500K unknowns)
- Iterative solvers with many RHS vectors
- Problems where matrix stays constant across many solves

## Solver Types

1. **CuSolverLU**: Direct LU factorization on GPU using cuSOLVER
   - Best for: Medium-large dense or sparse systems
   - Note: May be slower than CPU for small sparse matrices

2. **CuIterativeCG**: Conjugate Gradient on GPU
   - Best for: Large symmetric positive definite systems
   - Supports preconditioners: `:jacobi`, `:ilu0`, `:none`, or custom

3. **CuIterativeGMRES**: GMRES on GPU
   - Best for: Large non-symmetric systems
   - Supports preconditioners: `:jacobi`, `:ilu0`, `:none`, or custom

4. **CuDenseLU**: Dense LU on GPU
   - Best for: Small-medium dense systems
   - Very fast for matrices that fit in GPU memory

## Preconditioners

| Preconditioner | Description | Best For |
|----------------|-------------|----------|
| `:jacobi`      | Diagonal (Jacobi) preconditioner | Diagonally dominant matrices |
| `:ilu0`        | ILU(0) via cuSPARSE | Ill-conditioned matrices, strong off-diagonal coupling |
| `:ic0`         | IC(0) via cuSPARSE | Symmetric positive definite (SPD) matrices |
| `:none`        | No preconditioning | Well-conditioned problems |
| Custom function| User-defined M^{-1} | Specialized applications |

**ILU(0)** (Incomplete LU with zero fill-in) is significantly more effective than Jacobi
for tough matrices, often reducing iteration counts by 5-10x on ill-conditioned problems.

**IC(0)** (Incomplete Cholesky with zero fill-in) is the symmetric variant, optimal for
SPD matrices. It uses half the memory of ILU(0) and provides better numerical stability.

## When to Use GPU vs CPU Solvers

| Problem Size | Matrix Type | Recommended Solver |
|--------------|-------------|--------------------|
| < 10K        | Sparse      | CPU SparseLU       |
| < 10K        | Dense       | GPU CuDenseLU      |
| 10K - 100K   | Sparse      | CPU SparseLU or GPU CuIterative |
| 10K - 100K   | Dense       | GPU CuDenseLU      |
| > 100K       | Sparse      | GPU CuIterative    |
| > 100K       | Dense       | GPU CuDenseLU (if memory permits) |

## Usage

```julia
using CUDA
using Tarang

# Direct GPU solver
solver = CuSolverLU(A_sparse)
x = solve(solver, b)

# Iterative GPU solver with Jacobi preconditioner
solver = CuIterativeCG(A_sparse; preconditioner=:jacobi, tol=1e-10)
x = solve(solver, b)

# Iterative GPU solver with ILU(0) preconditioner (better for tough matrices)
solver = CuIterativeCG(A_sparse; preconditioner=:ilu0, tol=1e-10)
x = solve(solver, b)

# Iterative GPU solver with IC(0) preconditioner (optimal for SPD matrices)
solver = CuIterativeCG(A_spd; preconditioner=:ic0, tol=1e-10)
x = solve(solver, b)

# GMRES with ILU(0) for non-symmetric systems
solver = CuIterativeGMRES(A_sparse; preconditioner=:ilu0, tol=1e-10)
x = solve(solver, b)

# Dense GPU solver
solver = CuDenseLU(A_dense)
x = solve(solver, b)
```
"""

# Only load GPU solvers if CUDA is available
const CUDA_AVAILABLE = Ref(false)

"""
    _init_gpu_solvers!()

Initialize GPU matrix solvers if CUDA is available.
Called from the main module __init__ function.
"""
function _init_gpu_solvers!()
    try
        @eval using CUDA
        if CUDA.functional()
            CUDA_AVAILABLE[] = true
            @info "GPU matrix solvers enabled (CUDA available)"
            _register_gpu_helpers()
            _register_gpu_solvers()
        else
            @debug "CUDA found but not functional - GPU solvers disabled"
        end
    catch e
        @debug "CUDA not available - GPU solvers disabled: $e"
    end
end

# Import from parent MatSolvers module
using ..MatSolvers: AbstractMatSolver, register_solver, SOLVER_REGISTRY

# Import triangular types for ILU/IC preconditioner application
using LinearAlgebra: UnitLowerTriangular, LowerTriangular, UpperTriangular, ldiv!, tril

# ============================================================================
# GPU Helper Functions (Abstract CUDA-specific calls for consistency)
# ============================================================================
# These helpers centralize CUDA-specific operations, making the code:
# 1. More consistent with the architecture abstraction pattern
# 2. Easier to adapt for different GPU backends in the future
# 3. Cleaner and more maintainable

"""
    _gpu_zeros(T::Type, dims...)

Allocate a zero-filled GPU array. Wrapper around CUDA.zeros for consistency.
"""
function _gpu_zeros end

"""
    _gpu_array(data::AbstractArray, T::Type)

Convert array to GPU array with specified element type.
"""
function _gpu_array end

"""
    _gpu_sparse_csr(A::SparseMatrixCSC, T::Type)

Convert sparse matrix to GPU CSR format for cuSPARSE operations.
"""
function _gpu_sparse_csr end

"""
    _gpu_axpy!(α, x, y)

GPU AXPY operation: y = α*x + y. Uses CUDA.axpy! or custom kernel.
"""
function _gpu_axpy! end

"""
    _is_gpu_array(a)

Check if array is on GPU.
"""
function _is_gpu_array end

"""
    _gpu_ilu0(A_csr)

Compute ILU(0) factorization on GPU CSR matrix using cuSPARSE.
"""
function _gpu_ilu0 end

"""
    _gpu_ic0(A_csr)

Compute IC(0) (Incomplete Cholesky) factorization on GPU CSR matrix using cuSPARSE.
For symmetric positive definite matrices only.
"""
function _gpu_ic0 end

# Register GPU helper implementations when CUDA is loaded
function _register_gpu_helpers()
    @eval begin
        _gpu_zeros(T::Type, dims...) = CUDA.zeros(T, dims...)
        _gpu_array(data::AbstractArray, T::Type) = CuVector{T}(data)
        _gpu_array(data::AbstractMatrix, T::Type) = CuMatrix{T}(data)
        _gpu_sparse_csr(A::SparseMatrixCSC, T::Type) = CUDA.CUSPARSE.CuSparseMatrixCSR(SparseMatrixCSC{T, Int32}(A))
        _gpu_axpy!(α, x, y) = CUDA.axpy!(α, x, y)
        _is_gpu_array(a) = a isa CuArray
        _is_gpu_array(::Any) = false
        _gpu_ilu0(A_csr) = CUDA.CUSPARSE.ilu02(A_csr)
        _gpu_ic0(A_csr) = CUDA.CUSPARSE.ic02(A_csr)
    end
end

# ============================================================================
# Preconditioner infrastructure (must be defined before iterative solvers)
# ============================================================================

abstract type AbstractPreconditioner end

struct NoPreconditioner <: AbstractPreconditioner end

struct DiagonalPreconditioner{T} <: AbstractPreconditioner
    diag_inv::Any  # CuVector{T}
end

struct CustomPreconditioner{F} <: AbstractPreconditioner
    apply!::F
end

"""
    ILU0Preconditioner{T} <: AbstractPreconditioner

Incomplete LU factorization with zero fill-in (ILU(0)) preconditioner on GPU.
Uses cuSPARSE for efficient factorization and triangular solves.

ILU(0) is significantly more effective than Jacobi for:
- Ill-conditioned matrices
- Problems with strong off-diagonal coupling
- Reducing CG/GMRES iteration counts on tough matrices

The factorization computes L and U such that A ≈ LU, where:
- L is lower triangular with unit diagonal
- U is upper triangular
- Only nonzero positions from original A are filled

Application involves two triangular solves: z = U^{-1}(L^{-1}r)
"""
struct ILU0Preconditioner{T} <: AbstractPreconditioner
    # LU factors stored in CSR format (L below diagonal, U on and above diagonal)
    LU_csr::Any         # CuSparseMatrixCSR{T} with LU factors
    # Workspace for intermediate triangular solve result
    tmp::Any            # CuVector{T}
    # Matrix dimensions
    n::Int
end

"""
    IC0Preconditioner{T} <: AbstractPreconditioner

Incomplete Cholesky factorization with zero fill-in (IC(0)) preconditioner on GPU.
Uses cuSPARSE for efficient factorization and triangular solves.

IC(0) is the symmetric version of incomplete factorization, specifically for
symmetric positive definite (SPD) matrices. It is more efficient than ILU(0)
when the matrix is known to be SPD because:
- Only one triangular factor L is computed (A ≈ LL^T)
- Memory usage is halved compared to storing both L and U
- Better numerical stability for SPD systems

**Important**: The input matrix MUST be symmetric positive definite.
Using IC(0) on a non-SPD matrix may produce incorrect results or fail.

The factorization computes L such that A ≈ LL^T, where:
- L is lower triangular with explicit diagonal
- Only nonzero positions from the lower triangle of A are filled

Application involves two triangular solves: z = L^{-T}(L^{-1}r)
"""
struct IC0Preconditioner{T} <: AbstractPreconditioner
    # L factor stored in CSR format (lower triangular)
    L_csr::Any          # CuSparseMatrixCSR{T} with L factor
    # Workspace for intermediate triangular solve result
    tmp::Any            # CuVector{T}
    # Matrix dimensions
    n::Int
end

# ============================================================================
# GPU Dense LU Solver
# ============================================================================

"""
    CuDenseLU <: AbstractMatSolver

Dense LU factorization on GPU using cuSOLVER.
Efficient for small-medium dense matrices.

The matrix is kept on GPU, and factorization uses CUDA's cusolverDnXgetrf.
"""
struct CuDenseLU{T} <: AbstractMatSolver
    A_gpu::Any          # CuMatrix{T}
    lu_factor::Any      # Factorization result
    ipiv::Any           # CuVector{Int32} pivot indices
    workspace::Any      # GPU workspace buffer
    n::Int
end

function CuDenseLU(matrix::AbstractMatrix; kwargs...)
    if !CUDA_AVAILABLE[]
        error("CUDA not available. Use CPU solver instead.")
    end

    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)
    @assert size(matrix, 2) == n "Matrix must be square"

    # Transfer matrix to GPU using helper
    A_gpu = _gpu_array(matrix, T)

    # Allocate pivot vector on GPU using helper
    ipiv = _gpu_zeros(Int32, n)

    # Perform LU factorization on GPU
    # cusolverDnXgetrf computes LU factorization in-place
    CUDA.CUSOLVER.getrf!(A_gpu, ipiv)

    return CuDenseLU{T}(A_gpu, nothing, ipiv, nothing, n)
end

function MatSolvers.solve(s::CuDenseLU{T}, rhs::AbstractVector) where T
    # Transfer RHS to GPU if needed using helper
    b_gpu = _is_gpu_array(rhs) ? copy(rhs) : _gpu_array(rhs, T)

    # Solve using pre-factored LU
    # cusolverDnXgetrs solves A*X = B using LU factorization
    CUDA.CUSOLVER.getrs!('N', s.A_gpu, s.ipiv, b_gpu)

    # Return result (caller decides if they want CPU or GPU)
    return b_gpu
end

# ============================================================================
# GPU Sparse LU Solver (via cuSOLVER)
# ============================================================================

"""
    CuSparseLU <: AbstractMatSolver

Sparse LU solve on GPU via cuSOLVER's QR-based sparse solver. Each `solve` call runs
`csrlsvqr!` on the CSR matrix; this avoids dense fallbacks and works for general sparse
systems. For matrices that change frequently, consider CuIterative* solvers instead.

Note: GPU sparse LU can be slower than CPU for small-medium matrices
due to the irregular memory access patterns. Benchmark before using.
"""
struct CuSparseLU{T} <: AbstractMatSolver
    A_csr::Any  # CUDA.CUSPARSE.CuSparseMatrixCSR{T, Int32}
    tol::Float64
    reorder::Bool
end

function CuSparseLU(matrix::SparseMatrixCSC; tol::Real=1e-12, reorder::Bool=true, kwargs...)
    if !CUDA_AVAILABLE[]
        error("CUDA not available. Use CPU solver instead.")
    end

    T = promote_type(eltype(matrix), ComplexF64)
    # Convert to GPU CSR using helper
    A_csr = _gpu_sparse_csr(matrix, T)
    return CuSparseLU{T}(A_csr, Float64(tol), reorder)
end

function CuSparseLU(matrix::AbstractMatrix; kwargs...)
    return CuSparseLU(sparse(matrix); kwargs...)
end

function MatSolvers.solve(s::CuSparseLU{T}, rhs::AbstractVector) where T
    # Transfer RHS to GPU using helper
    b_gpu = _gpu_array(rhs, T)
    x_gpu = similar(b_gpu)
    singularity = Ref{Int32}(-1)
    CUDA.CUSOLVER.csrlsvqr!(s.A_csr, b_gpu, x_gpu;
                            tol=s.tol, reorder=s.reorder, singularity=singularity)
    if singularity[] >= 0
        @warn "CuSparseLU: matrix may be singular" row=singularity[]
    end
    return x_gpu
end

# ============================================================================
# GPU Iterative Solver: Conjugate Gradient
# ============================================================================

"""
    CuIterativeCG <: AbstractMatSolver

Conjugate Gradient solver on GPU.
Best for large symmetric positive definite systems.

The matrix-vector products are performed on GPU using cuSPARSE,
which is highly efficient for sparse matrices.

# Arguments
- `matrix`: Sparse matrix (must be SPD for convergence)
- `tol`: Convergence tolerance (default: 1e-10)
- `maxiter`: Maximum iterations (default: 1000)
- `preconditioner`: Preconditioner definition:
  - `:jacobi` (default): Diagonal preconditioner, good for diagonally dominant matrices
  - `:ilu0`: ILU(0) via cuSPARSE, much better for ill-conditioned matrices
  - `:ic0`: IC(0) via cuSPARSE, optimal for SPD matrices (recommended for CG)
  - `:none`: No preconditioning
  - `Vector`: Explicit diagonal inverse values
  - `Function`: Custom preconditioner `(out, r) -> out = M^{-1}*r`

# Example
```julia
# For well-conditioned SPD matrices
solver = CuIterativeCG(A; preconditioner=:jacobi)

# For ill-conditioned matrices (reduces iterations significantly)
solver = CuIterativeCG(A; preconditioner=:ilu0, tol=1e-12)

# For SPD matrices (most efficient, uses half the memory of ILU(0))
solver = CuIterativeCG(A; preconditioner=:ic0, tol=1e-12)
```
"""
mutable struct CuIterativeCG{T,P<:AbstractPreconditioner} <: AbstractMatSolver
    A_csr::Any              # CuSparseMatrixCSR for SpMV
    preconditioner::P
    tol::Float64
    maxiter::Int
    n::Int
    # Workspace vectors (pre-allocated for efficiency)
    r::Any                  # Residual
    p::Any                  # Search direction
    Ap::Any                 # Matrix-vector product
    z::Any                  # Preconditioned residual
end

function CuIterativeCG(matrix::AbstractMatrix;
                       tol::Real=1e-10,
                       maxiter::Int=1000,
                       preconditioner::Symbol=:jacobi,
                       kwargs...)
    if !CUDA_AVAILABLE[]
        error("CUDA not available. Use CPU solver instead.")
    end

    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)

    # Convert to CSR on GPU using helper
    A_sparse = matrix isa SparseMatrixCSC ? matrix : sparse(matrix)
    A_csr = _gpu_sparse_csr(A_sparse, T)

    # Build preconditioner
    M_inv = _build_preconditioner(A_sparse, preconditioner, T)

    # Pre-allocate workspace on GPU using helper
    r = _gpu_zeros(T, n)
    p = _gpu_zeros(T, n)
    Ap = _gpu_zeros(T, n)
    z = _gpu_zeros(T, n)

    return CuIterativeCG{T, typeof(M_inv)}(A_csr, M_inv, Float64(tol), maxiter, n, r, p, Ap, z)
end

function MatSolvers.solve(s::CuIterativeCG{T}, rhs::AbstractVector) where T
    # Transfer RHS to GPU if needed using helper
    b = _is_gpu_array(rhs) ? rhs : _gpu_array(rhs, T)

    # Initial guess: x = 0
    x = _gpu_zeros(T, s.n)

    # r = b - A*x = b (since x=0)
    copyto!(s.r, b)

    # Apply preconditioner: z = M^{-1} * r
    _apply_preconditioner!(s.z, s.preconditioner, s.r)

    # p = z
    copyto!(s.p, s.z)

    # rz_old = r' * z
    rz_old = dot(s.r, s.z)

    for iter in 1:s.maxiter
        # Ap = A * p
        mul!(s.Ap, s.A_csr, s.p)

        # alpha = rz_old / (p' * Ap)
        pAp = dot(s.p, s.Ap)
        if abs(pAp) < 1e-30
            @warn "CG breakdown at iteration $iter: p'Ap ≈ 0"
            break
        end
        alpha = rz_old / pAp

        # x = x + alpha * p (using helper for GPU AXPY)
        _gpu_axpy!(alpha, s.p, x)

        # r = r - alpha * Ap (using helper for GPU AXPY)
        _gpu_axpy!(-alpha, s.Ap, s.r)

        # Check convergence
        r_norm = norm(s.r)
        if r_norm < s.tol
            @debug "CG converged in $iter iterations, residual = $r_norm"
            return x
        end

        # z = M^{-1} * r
        _apply_preconditioner!(s.z, s.preconditioner, s.r)

        # rz_new = r' * z
        rz_new = dot(s.r, s.z)

        # beta = rz_new / rz_old
        beta = rz_new / rz_old

        # p = z + beta * p
        s.p .= s.z .+ beta .* s.p

        rz_old = rz_new
    end

    @warn "CG did not converge in $(s.maxiter) iterations"
    return x
end

_apply_preconditioner!(z, M_inv, r) = error("Unsupported preconditioner type $(typeof(M_inv))")

# ============================================================================
# GPU Iterative Solver: GMRES
# ============================================================================

"""
    CuIterativeGMRES <: AbstractMatSolver

GMRES solver on GPU.
Best for large non-symmetric systems.

Uses restarted GMRES with configurable restart parameter.
Matrix-vector products use cuSPARSE for efficiency.

# Arguments
- `matrix`: Sparse matrix (can be non-symmetric)
- `tol`: Convergence tolerance (default: 1e-10)
- `maxiter`: Maximum iterations (default: 1000)
- `restart`: Restart parameter (default: 30)
- `preconditioner`: Preconditioner definition:
  - `:jacobi` (default): Diagonal preconditioner, good for diagonally dominant matrices
  - `:ilu0`: ILU(0) via cuSPARSE, much better for ill-conditioned matrices
  - `:ic0`: IC(0) via cuSPARSE, for SPD matrices only (use CG instead if SPD)
  - `:none`: No preconditioning
  - `Vector`: Explicit diagonal inverse values
  - `Function`: Custom preconditioner `(out, r) -> out = M^{-1}*r`

# Example
```julia
# Standard GMRES with Jacobi preconditioning
solver = CuIterativeGMRES(A; preconditioner=:jacobi, restart=50)

# For tough non-symmetric systems (recommended)
solver = CuIterativeGMRES(A; preconditioner=:ilu0, tol=1e-12)
```
"""
mutable struct CuIterativeGMRES{T,P<:AbstractPreconditioner} <: AbstractMatSolver
    A_csr::Any              # CuSparseMatrixCSR
    preconditioner::P
    tol::Float64
    maxiter::Int
    restart::Int            # Restart parameter
    n::Int
end

function CuIterativeGMRES(matrix::AbstractMatrix;
                          tol::Real=1e-10,
                          maxiter::Int=1000,
                          restart::Int=30,
                          preconditioner::Symbol=:jacobi,
                          kwargs...)
    if !CUDA_AVAILABLE[]
        error("CUDA not available. Use CPU solver instead.")
    end

    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)

    # Convert to CSR on GPU using helper
    A_sparse = matrix isa SparseMatrixCSC ? matrix : sparse(matrix)
    A_csr = _gpu_sparse_csr(A_sparse, T)

    M_inv = _build_preconditioner(A_sparse, preconditioner, T)

    return CuIterativeGMRES{T, typeof(M_inv)}(A_csr, M_inv, Float64(tol), maxiter, restart, n)
end

function MatSolvers.solve(s::CuIterativeGMRES{T}, rhs::AbstractVector) where T
    # Transfer RHS to GPU using helper
    b = _is_gpu_array(rhs) ? rhs : _gpu_array(rhs, T)

    n = s.n
    m = s.restart
    x = _gpu_zeros(T, n)

    # Outer iteration (restarts)
    for outer in 1:div(s.maxiter, m)
        # r = b - A*x
        r = b - s.A_csr * x

        # Apply preconditioner
        tmp = similar(r)
        _apply_preconditioner!(tmp, s.preconditioner, r)
        r = tmp

        beta = norm(r)
        if beta < s.tol
            @debug "GMRES converged in $(outer * m) iterations"
            return x
        end

        # Arnoldi process
        V = [r / beta]  # Krylov basis vectors
        H = _gpu_zeros(T, m + 1, m)  # Hessenberg matrix

        for j in 1:m
            # w = A * v_j
            w = s.A_csr * V[j]

            # Apply preconditioner (left preconditioning)
            w_tmp = similar(w)
            _apply_preconditioner!(w_tmp, s.preconditioner, w)
            w = w_tmp

            # Modified Gram-Schmidt
            for i in 1:j
                H[i, j] = dot(V[i], w)
                w = w - H[i, j] * V[i]
            end
            H[j + 1, j] = norm(w)

            if abs(H[j + 1, j]) < 1e-14
                break
            end

            push!(V, w / H[j + 1, j])
        end

        # Solve least squares: min ||H*y - beta*e1||
        e1 = _gpu_zeros(T, m + 1)
        e1[1] = beta

        # Use QR factorization (transfer to CPU for small system)
        H_cpu = Array(H)
        e1_cpu = Array(e1)
        y_cpu = H_cpu \ e1_cpu
        y = _gpu_array(y_cpu[1:length(V)-1], T)

        # Update solution: x = x + V * y (using helper for GPU AXPY)
        for i in 1:length(y)
            _gpu_axpy!(y[i], V[i], x)
        end
    end

    @warn "GMRES did not converge in $(s.maxiter) iterations"
    return x
end

# ============================================================================
# Hybrid CPU-GPU Solver
# ============================================================================

"""
    HybridSolver <: AbstractMatSolver

Automatically selects CPU or GPU solver based on problem characteristics.

Decision criteria:
- Matrix size: GPU for n > threshold
- Matrix density: GPU for dense, CPU for very sparse
- Available GPU memory
"""
struct HybridSolver{T} <: AbstractMatSolver
    cpu_solver::Any
    gpu_solver::Any
    use_gpu::Bool
    threshold::Int
    mode::Symbol
end

function HybridSolver(matrix::AbstractMatrix;
                      threshold::Int=50000,
                      gpu_solver_type::Symbol=:cg,
                      mode::Symbol=:auto,
                      gpu_preconditioner=:jacobi,
                      kwargs...)
    T = promote_type(eltype(matrix), ComplexF64)
    n = size(matrix, 1)

    mode ∈ (:auto, :cpu, :gpu) || throw(ArgumentError("HybridSolver mode must be :auto, :cpu, or :gpu"))

    # Build CPU solver (always available)
    cpu_solver = SparseLUSolver(matrix)

    # Decide whether to use GPU
    use_gpu = false
    gpu_solver = nothing

    use_gpu_requested = (mode === :gpu) || (mode === :auto && CUDA_AVAILABLE[] && n > threshold)

    if use_gpu_requested
        try
            if gpu_solver_type == :cg
                gpu_solver = CuIterativeCG(matrix; preconditioner=gpu_preconditioner, kwargs...)
            elseif gpu_solver_type == :gmres
                gpu_solver = CuIterativeGMRES(matrix; preconditioner=gpu_preconditioner, kwargs...)
            elseif gpu_solver_type == :lu
                gpu_solver = CuDenseLU(matrix)
            end
            use_gpu = true
            @info "HybridSolver: Using GPU $(gpu_solver_type) (mode=$(mode), n=$n, threshold=$threshold)"
        catch e
            @warn "GPU solver creation failed, using CPU: $e"
        end
    else
        @debug "HybridSolver: Using CPU solver (mode=$(mode), n=$n, threshold=$threshold)"
    end

    return HybridSolver{T}(cpu_solver, gpu_solver, use_gpu, threshold, mode)
end

function MatSolvers.solve(s::HybridSolver, rhs::AbstractVector)
    if s.use_gpu && s.gpu_solver !== nothing
        result = solve(s.gpu_solver, rhs)
        # Return as CPU array for compatibility using helper
        return _is_gpu_array(result) ? Array(result) : result
    else
        return solve(s.cpu_solver, rhs)
    end
end

# ============================================================================
# Registration
# ============================================================================

function _register_gpu_solvers()
    register_solver("cuda_dense", CuDenseLU)
    register_solver("cuda_lu", CuDenseLU)
    register_solver("cuda_sparse", CuSparseLU)
    register_solver("cuda_cg", CuIterativeCG)
    register_solver("cuda_gmres", CuIterativeGMRES)
    register_solver("gpu", HybridSolver)
    register_solver("hybrid", HybridSolver)
    @debug "Registered GPU matrix solvers"
end

# Export GPU solvers
export CuDenseLU, CuSparseLU, CuIterativeCG, CuIterativeGMRES, HybridSolver
export CUDA_AVAILABLE

# ============================================================================
# Preconditioner helper functions
# ============================================================================

function _build_preconditioner(A::SparseMatrixCSC, preconditioner, T::Type)
    if preconditioner isa Symbol
        if preconditioner === :none
            return NoPreconditioner()
        elseif preconditioner === :jacobi
            diag_A = diag(A)
            diag_inv = similar(diag_A, T)
            @inbounds for i in eachindex(diag_A)
                d = diag_A[i]
                diag_inv[i] = abs(d) > 1e-14 ? T(1) / T(d) : zero(T)
            end
            # Transfer diagonal to GPU using helper
            return DiagonalPreconditioner(_gpu_array(diag_inv, T))
        elseif preconditioner === :ilu0
            return _build_ilu0_preconditioner(A, T)
        elseif preconditioner === :ic0
            return _build_ic0_preconditioner(A, T)
        else
            error("Unsupported preconditioner symbol $(preconditioner). Use :none, :jacobi, :ilu0, :ic0, a vector, or a custom function.")
        end
    elseif preconditioner === nothing
        return NoPreconditioner()
    elseif preconditioner isa AbstractVector
        length(preconditioner) == size(A, 1) || error("Preconditioner vector has wrong length")
        # Transfer preconditioner vector to GPU using helper
        return DiagonalPreconditioner(_gpu_array(preconditioner, T))
    elseif preconditioner isa Function
        return CustomPreconditioner(preconditioner)
    else
        error("Unsupported preconditioner type $(typeof(preconditioner))")
    end
end

"""
    _build_ilu0_preconditioner(A::SparseMatrixCSC, T::Type)

Build an ILU(0) preconditioner using cuSPARSE.
Computes incomplete LU factorization with zero fill-in on GPU.
"""
function _build_ilu0_preconditioner(A::SparseMatrixCSC, T::Type)
    n = size(A, 1)
    @assert size(A, 2) == n "Matrix must be square for ILU(0)"

    # Convert to CSR format on GPU using helper
    A_csr = _gpu_sparse_csr(A, T)

    # Compute ILU(0) factorization using helper
    # ilu02 returns the matrix with L and U factors:
    # - L is stored below diagonal (with implicit unit diagonal)
    # - U is stored on and above diagonal
    LU_csr = _gpu_ilu0(A_csr)

    # Allocate workspace for triangular solve intermediate using helper
    tmp = _gpu_zeros(T, n)

    return ILU0Preconditioner{T}(LU_csr, tmp, n)
end

"""
    _build_ic0_preconditioner(A::SparseMatrixCSC, T::Type)

Build an IC(0) (Incomplete Cholesky) preconditioner using cuSPARSE.
Computes incomplete Cholesky factorization with zero fill-in on GPU.

**Important**: The input matrix must be symmetric positive definite (SPD).
The function extracts the lower triangular part of A for factorization.
"""
function _build_ic0_preconditioner(A::SparseMatrixCSC, T::Type)
    n = size(A, 1)
    @assert size(A, 2) == n "Matrix must be square for IC(0)"

    # For IC(0), we need to work with a symmetric matrix
    # cuSPARSE ic02 expects the lower triangular part in CSR format
    # Extract lower triangular part to ensure symmetry
    A_lower = tril(A)

    # Convert to CSR format on GPU using helper
    A_csr = _gpu_sparse_csr(A_lower, T)

    # Compute IC(0) factorization using helper
    # ic02 returns the matrix with L factor:
    # - L is stored in the lower triangular part with explicit diagonal
    # - A ≈ LL^T
    L_csr = _gpu_ic0(A_csr)

    # Allocate workspace for triangular solve intermediate using helper
    tmp = _gpu_zeros(T, n)

    return IC0Preconditioner{T}(L_csr, tmp, n)
end

function _apply_preconditioner!(z, ::NoPreconditioner, r)
    copyto!(z, r)
end

function _apply_preconditioner!(z, M::DiagonalPreconditioner, r)
    z .= M.diag_inv .* r
end

function _apply_preconditioner!(z, M::CustomPreconditioner, r)
    M.apply!(z, r)
end

"""
    _apply_preconditioner!(z, M::ILU0Preconditioner, r)

Apply ILU(0) preconditioner: z = (LU)^{-1} * r

This performs two triangular solves:
1. L * tmp = r  (lower triangular, unit diagonal)
2. U * z = tmp  (upper triangular)

Uses cuSPARSE triangular solves via ldiv! for GPU-accelerated operations.
For CSR format: L has unit diagonal (below), U has non-unit diagonal (on and above).
"""
function _apply_preconditioner!(z, M::ILU0Preconditioner{T}, r) where T
    # For CSR matrices from ilu02:
    # - L is lower triangular with implicit unit diagonal
    # - U is upper triangular with explicit diagonal

    # Step 1: Forward substitution - solve L * tmp = r
    # UnitLowerTriangular indicates L has unit diagonal (ones on diagonal, not stored)
    ldiv!(M.tmp, UnitLowerTriangular(M.LU_csr), r)

    # Step 2: Backward substitution - solve U * z = tmp
    # UpperTriangular indicates U has explicit diagonal values
    ldiv!(z, UpperTriangular(M.LU_csr), M.tmp)
end

"""
    _apply_preconditioner!(z, M::IC0Preconditioner, r)

Apply IC(0) preconditioner: z = (LL^T)^{-1} * r

This performs two triangular solves:
1. L * tmp = r   (forward substitution, lower triangular)
2. L^T * z = tmp (backward substitution, upper triangular)

Uses cuSPARSE triangular solves via ldiv! for GPU-accelerated operations.
For CSR format from ic02: L has explicit diagonal values.

Note: IC(0) is more efficient than ILU(0) for SPD matrices because:
- Only one factor needs to be stored (L instead of L and U)
- The transpose L^T can be computed implicitly from L
"""
function _apply_preconditioner!(z, M::IC0Preconditioner{T}, r) where T
    # For CSR matrices from ic02:
    # - L is lower triangular with explicit diagonal

    # Step 1: Forward substitution - solve L * tmp = r
    # LowerTriangular indicates L has explicit diagonal values
    ldiv!(M.tmp, LowerTriangular(M.L_csr), r)

    # Step 2: Backward substitution - solve L^T * z = tmp
    # For L^T, we use the transpose. With CSR, transpose gives us upper triangular.
    # cuSPARSE handles the transpose efficiently.
    ldiv!(z, LowerTriangular(M.L_csr)', M.tmp)
end
