# GPU Solvers: Preconditioning

`CuIterativeCG` and `CuIterativeGMRES` support several preconditioners to accelerate
convergence on GPU.

## Available Preconditioners

| Preconditioner | Value | Best For |
|----------------|--------|----------|
| Jacobi (diagonal) | `:jacobi` | Diagonally dominant matrices (default) |
| ILU(0) | `:ilu0` | Ill-conditioned matrices, strong off-diagonal coupling |
| IC(0) | `:ic0` | Symmetric positive definite (SPD) matrices |
| None | `:none` | Well-conditioned problems |

The `preconditioner` keyword is declared `::Symbol`, so these four values are the only
ones accepted. Passing a vector of diagonal values, a custom function, or `nothing` is
rejected at the call boundary, before the constructor body runs:

```julia
CuIterativeCG(A; preconditioner=d_inv)   # TypeError: in keyword argument preconditioner,
                                         # expected Symbol, got a value of type Vector{Float64}
```

## Requirements

These solvers live behind the CUDA package extension and are gated on
`Tarang.CUDA_AVAILABLE[]`. Loading both packages activates the extension in either
load order. A working NVIDIA driver/device is still required for allocation and solve:

```julia
using CUDA, Tarang
@assert CUDA.functional()
@assert Tarang.CUDA_AVAILABLE[]
```

When the CUDA extension is unavailable, every constructor below raises

```
ERROR: CUDA not available. Use CPU solver instead.
```

For a CPU-only environment, use the CPU solvers, which take the same
`(solver, rhs)` call shape:

```julia
using SparseArrays, LinearAlgebra, Tarang

A = sprand(1_000, 1_000, 5e-3)
A = A + A' + 10I               # symmetric positive definite
b = rand(size(A, 1))

solver = Tarang.MatSolvers.SparseLUSolver(A)
x = Tarang.MatSolvers.solve(solver, b)

norm(A * x - b) / norm(b)      # approximately machine precision
```

`solve` is **not** exported by `Tarang`; call it as `Tarang.solve` (or
`Tarang.MatSolvers.solve`). The same goes for `SparseLUSolver`.

## Jacobi Preconditioner (Default)

Simple diagonal scaling using the inverse of the matrix diagonal:

```julia
using CUDA, SparseArrays, LinearAlgebra, Tarang

A = sprand(2_000, 2_000, 2e-3)
A = A + A' + 10I               # symmetric positive definite
b = rand(size(A, 1))

# Jacobi preconditioner (default)
solver = CuIterativeCG(A; preconditioner=:jacobi, tol=1e-10)
x = Tarang.solve(solver, b)
```

Good for diagonally dominant matrices where the diagonal captures most of the matrix structure.

## ILU(0) Preconditioner (Recommended for Tough Problems)

Incomplete LU factorization with zero fill-in. Uses cuSPARSE for GPU-accelerated
factorization and triangular solves.

```julia
using CUDA, SparseArrays, LinearAlgebra, Tarang

# Create a challenging problem (2D Laplacian)
function laplacian_2d(n)
    I_n = sparse(I, n, n)
    D = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
    return kron(I_n, D) + kron(D, I_n)
end

n = 32
A = laplacian_2d(n)            # 1_024 x 1_024, SPD smoke problem
b = rand(size(A, 1))

# ILU(0) preconditioner - much better for ill-conditioned matrices
solver = CuIterativeCG(A; preconditioner=:ilu0, tol=1e-10)
x = Tarang.solve(solver, b)
```

**Benefits of ILU(0):**
- Captures matrix structure beyond just the diagonal, so it typically converges in far fewer
  iterations than Jacobi on ill-conditioned problems
- Efficient GPU implementation via cuSPARSE
- Zero fill-in: the factors reuse the sparsity pattern of `A`

**When to use ILU(0):**
- Ill-conditioned matrices
- Problems with strong off-diagonal coupling
- When Jacobi converges slowly or not at all
- Large sparse systems from PDE discretizations

## IC(0) Preconditioner (Optimal for SPD Matrices)

Incomplete Cholesky factorization with zero fill-in. The symmetric variant of ILU(0),
specifically optimized for symmetric positive definite (SPD) matrices.

```julia
using CUDA, SparseArrays, LinearAlgebra, Tarang

function laplacian_2d(n)
    I_n = sparse(I, n, n)
    D = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
    return kron(I_n, D) + kron(D, I_n)
end

n = 32
A = laplacian_2d(n)            # this is SPD
b = rand(size(A, 1))

# IC(0) preconditioner - optimal for SPD matrices
solver = CuIterativeCG(A; preconditioner=:ic0, tol=1e-10)
x = Tarang.solve(solver, b)
```

**Benefits of IC(0) over ILU(0) for SPD matrices:**
- Lower memory: IC(0) factorizes `tril(A)` and stores only `L`, where ILU(0) stores `L` and `U`
  together. For the 2D Laplacian above that is 3,008 stored entries against 4,992.
- Better numerical stability for symmetric systems
- Preserves symmetry properties

**When to use IC(0):**
- Matrix is known to be symmetric positive definite
- Using Conjugate Gradient solver (CG requires SPD matrix anyway)
- Memory is a concern
- Poisson/Laplacian problems, diffusion equations, mass matrices

**Important:** IC(0) requires the matrix to be SPD. Using it on a non-SPD matrix
may produce incorrect results or fail. If unsure about SPD property, use ILU(0) instead.

## GMRES with ILU(0)

For non-symmetric systems, use `CuIterativeGMRES`:

```julia
using CUDA, SparseArrays, LinearAlgebra, Tarang

# Non-symmetric convection-diffusion problem
function convection_diffusion_1d(n; peclet=10.0)
    h = 1.0 / (n + 1)
    diffusion  = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1)) ./ h^2
    convection = spdiagm(-1 => -ones(n-1), 1 => ones(n-1)) .* (peclet / (2h))
    return diffusion + convection
end

A = convection_diffusion_1d(2_000)   # non-symmetric smoke problem
b = rand(size(A, 1))

# GMRES with ILU(0) preconditioning
solver = CuIterativeGMRES(A;
    preconditioner=:ilu0,
    tol=1e-10,
    restart=50,    # Restart parameter
    maxiter=500
)
x = Tarang.solve(solver, b)
```

## Preconditioner Summary

```text
# Allowed preconditioner values (Symbol only):
solver = CuIterativeCG(A; preconditioner=...)

# :none      - No preconditioning
# :jacobi    - Diagonal inverse from diag(A) [default]
# :ilu0      - ILU(0) via cuSPARSE (recommended for tough non-SPD problems)
# :ic0       - IC(0) via cuSPARSE (optimal for SPD matrices, lower memory than ILU(0))
```

**Quick selection guide:**
- SPD matrix + CG solver: Use `:ic0` (best performance and memory)
- Non-symmetric matrix + GMRES: Use `:ilu0`
- Unsure if SPD: Use `:ilu0` (works for all matrices)
- Simple/well-conditioned: Use `:jacobi` (fastest setup)

## See Also

- [GPU Solvers Overview](../pages/solvers.md): Solver types and selection
- [Optimization](../pages/optimization.md): Performance tuning
