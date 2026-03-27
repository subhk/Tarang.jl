# GPU Solvers: Preconditioning

`CuIterativeCG` and `CuIterativeGMRES` support multiple preconditioner options to accelerate
convergence on GPU.

## Available Preconditioners

| Preconditioner | Symbol | Best For |
|----------------|--------|----------|
| Jacobi (diagonal) | `:jacobi` | Diagonally dominant matrices |
| ILU(0) | `:ilu0` | Ill-conditioned matrices, strong off-diagonal coupling |
| IC(0) | `:ic0` | Symmetric positive definite (SPD) matrices |
| None | `:none` | Well-conditioned problems |
| Custom | `Function` | Specialized applications |

## Jacobi Preconditioner (Default)

Simple diagonal scaling using the inverse of the matrix diagonal:

```julia
using CUDA, SparseArrays, Tarang

A = sprand(10_000, 10_000, 1e-3)
A = A + A' + 10I  # Make SPD

# Jacobi preconditioner (default)
solver = CuIterativeCG(A; preconditioner=:jacobi, tol=1e-10)
x = solve(solver, b)
```

Good for diagonally dominant matrices where the diagonal captures most of the matrix structure.

## ILU(0) Preconditioner (Recommended for Tough Problems)

Incomplete LU factorization with zero fill-in. Uses cuSPARSE for GPU-accelerated
factorization and triangular solves.

```julia
using CUDA, SparseArrays, Tarang

# Create a challenging problem (2D Laplacian)
n = 100
N = n^2
function laplacian_2d(n)
    I_n = sparse(I, n, n)
    D = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
    return kron(I_n, D) + kron(D, I_n)
end

A = laplacian_2d(n)
b = rand(N)

# ILU(0) preconditioner - much better for ill-conditioned matrices
solver = CuIterativeCG(A; preconditioner=:ilu0, tol=1e-10)
x = solve(solver, b)
```

**Benefits of ILU(0):**
- Significantly reduces iteration count (often 5-10x fewer iterations than Jacobi)
- Captures matrix structure beyond just the diagonal
- Efficient GPU implementation via cuSPARSE
- Same memory footprint as original matrix (zero fill-in)

**When to use ILU(0):**
- Ill-conditioned matrices
- Problems with strong off-diagonal coupling
- When Jacobi converges slowly or not at all
- Large sparse systems from PDE discretizations

## IC(0) Preconditioner (Optimal for SPD Matrices)

Incomplete Cholesky factorization with zero fill-in. The symmetric variant of ILU(0),
specifically optimized for symmetric positive definite (SPD) matrices.

```julia
using CUDA, SparseArrays, Tarang

# Create an SPD matrix (e.g., from 2D Laplacian)
n = 100
N = n^2
function laplacian_2d(n)
    I_n = sparse(I, n, n)
    D = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
    return kron(I_n, D) + kron(D, I_n)
end

A = laplacian_2d(n)  # This is SPD
b = rand(N)

# IC(0) preconditioner - optimal for SPD matrices
solver = CuIterativeCG(A; preconditioner=:ic0, tol=1e-10)
x = solve(solver, b)
```

**Benefits of IC(0) over ILU(0) for SPD matrices:**
- Uses half the memory (stores only L, not L and U)
- Better numerical stability for symmetric systems
- Slightly faster factorization and application
- Preserves symmetry properties

**When to use IC(0):**
- Matrix is known to be symmetric positive definite
- Using Conjugate Gradient solver (CG requires SPD matrix anyway)
- Memory is a concern (half the storage of ILU(0))
- Poisson/Laplacian problems, diffusion equations, mass matrices

**Important:** IC(0) requires the matrix to be SPD. Using it on a non-SPD matrix
may produce incorrect results or fail. If unsure about SPD property, use ILU(0) instead.

## GMRES with ILU(0)

For non-symmetric systems, use `CuIterativeGMRES`:

```julia
using CUDA, SparseArrays, Tarang

# Non-symmetric convection-diffusion problem
A = # ... your non-symmetric matrix ...
b = rand(size(A, 1))

# GMRES with ILU(0) preconditioning
solver = CuIterativeGMRES(A;
    preconditioner=:ilu0,
    tol=1e-10,
    restart=50,    # Restart parameter
    maxiter=500
)
x = solve(solver, b)
```

## Custom Preconditioners

For specialized applications, provide a custom function:

```julia
using CUDA, SparseArrays, Tarang

A = sprand(10_000, 10_000, 1e-3)
A = A + A' + 10I

# Custom preconditioner: in-place function on CuArrays
function my_preconditioner!(out, r)
    # Example: scale by a user-defined factor
    @. out = r / 10
end

solver = CuIterativeCG(A; preconditioner=my_preconditioner!)
x = solve(solver, b)
```

**Custom preconditioner requirements:**
- Function signature: `(out::CuVector, r::CuVector) -> Nothing`
- Operates in-place on GPU arrays
- Should approximate `out = M⁻¹ * r` where `M ≈ A`

## Explicit Diagonal Preconditioner

Provide your own diagonal inverse values:

```julia
using CUDA, SparseArrays, Tarang

A = # ... your matrix ...

# Compute custom diagonal inverse
d = diag(A)
d_inv = 1.0 ./ d

# Use as preconditioner
solver = CuIterativeCG(A; preconditioner=d_inv)
```

## Performance Comparison

Typical iteration counts for a 2D Poisson problem (N=10,000):

| Preconditioner | Iterations | Relative Time | Memory |
|----------------|------------|---------------|--------|
| None | 500+ | Baseline | 1x |
| Jacobi | 150-200 | ~0.4x | 1x |
| ILU(0) | 20-40 | ~0.1x | 2x |
| IC(0) | 15-35 | ~0.08x | 1x |

*Note: Actual performance depends on matrix structure and condition number.*
*IC(0) is only valid for SPD matrices but offers similar convergence to ILU(0) with half the memory.*

## Preconditioner Summary

```julia
# Allowed preconditioner values:
solver = CuIterativeCG(A; preconditioner=...)

# :none      - No preconditioning
# :jacobi    - Diagonal inverse from diag(A) [default]
# :ilu0      - ILU(0) via cuSPARSE (recommended for tough non-SPD problems)
# :ic0       - IC(0) via cuSPARSE (optimal for SPD matrices, half memory of ILU)
# Vector     - Explicit diagonal inverse values (copied to GPU)
# Function   - Custom (out, r) -> out = M⁻¹*r on GPU arrays
```

**Quick selection guide:**
- SPD matrix + CG solver: Use `:ic0` (best performance and memory)
- Non-symmetric matrix + GMRES: Use `:ilu0`
- Unsure if SPD: Use `:ilu0` (works for all matrices)
- Simple/well-conditioned: Use `:jacobi` (fastest setup)

## See Also

- [GPU Solvers Overview](../pages/solvers.md): Solver types and selection
- [Optimization](../pages/optimization.md): Performance tuning
