# The Tau Method for Boundary Conditions

The tau method is a powerful spectral technique for enforcing boundary conditions in differential equations. This page provides a detailed explanation accessible to those new to spectral methods.

## Why Do We Need the Tau Method?

### The Boundary Condition Problem in Spectral Methods

Consider solving a differential equation like the heat equation:

```math
\frac{\partial u}{\partial t} = \kappa \frac{\partial^2 u}{\partial z^2}, \quad z \in [0, 1]
```

with boundary conditions u(0,t) = 0 and u(1,t) = 0.

In **finite difference methods**, applying boundary conditions is straightforward: you simply set the values at the boundary grid points:
```
u[1] = 0      # at z = 0
u[N] = 0      # at z = 1
```

In **spectral methods**, we represent the solution as a sum of basis functions:
```math
u(z) = \sum_{n=0}^{N-1} a_n T_n(z)
```

where T_n(z) are Chebyshev polynomials and a_n are the spectral coefficients we need to find.

**The challenge**: The spectral coefficients a_n are global—they affect the solution everywhere, not just at specific points. We cannot simply "set" boundary values; instead, we must find coefficients that simultaneously:
1. Satisfy the PDE in the interior
2. Satisfy the boundary conditions at the edges

### The Core Idea of the Tau Method

The tau method solves this by a clever trade-off:

> **Key Insight**: We have N unknowns (spectral coefficients a₀, a₁, ..., a_{N-1}) and need N equations. Instead of using N equations from the PDE, we use (N-k) equations from the PDE and k equations from the boundary conditions.

For a second-order equation with 2 boundary conditions (k=2):
- Equations 0 through N-3: Come from the PDE
- Equations N-2 and N-1: Come from the boundary conditions

## Understanding Through a Simple Example

### Step 1: The Continuous Problem

Solve the steady diffusion equation:
```math
\frac{d^2 u}{d z^2} = f(z), \quad u(0) = 0, \quad u(1) = 0
```

### Step 2: Spectral Representation

Expand u(z) in Chebyshev polynomials:
```math
u(z) = \sum_{n=0}^{N-1} a_n T_n(z)
```

Recall key properties of Chebyshev polynomials:
- T_n(z) = cos(n · arccos(z)) for z ∈ [-1, 1]
- T_n(1) = 1 for all n
- T_n(-1) = (-1)^n

### Step 3: The Differentiation Matrix

Taking derivatives in spectral space:
```math
\frac{d^2 u}{dz^2} = \sum_{n=0}^{N-1} a_n \frac{d^2 T_n}{dz^2} = \sum_{n=0}^{N-1} b_n T_n(z)
```

The coefficients b_n are related to a_n through the **differentiation matrix** D²:
```math
\mathbf{b} = D^{(2)} \mathbf{a}
```

### Step 4: The PDE in Spectral Space (Without Tau)

Matching coefficients of T_n(z) on both sides of the PDE gives N equations:
```math
\sum_{m=0}^{N-1} D^{(2)}_{nm} a_m = f_n, \quad n = 0, 1, ..., N-1
```

where f_n are the spectral coefficients of f(z).

**Problem**: These N equations determine all N coefficients, leaving no freedom to satisfy boundary conditions!

### Step 5: The Tau Modification

**Replace the last two equations** with boundary conditions:

Original system:
```
Row 0:   D²₀₀·a₀ + D²₀₁·a₁ + ... = f₀    ← PDE
Row 1:   D²₁₀·a₀ + D²₁₁·a₁ + ... = f₁    ← PDE
  ⋮
Row N-3: D²_{N-3,0}·a₀ + ... = f_{N-3}   ← PDE
Row N-2: D²_{N-2,0}·a₀ + ... = f_{N-2}   ← PDE  ✗ REPLACE
Row N-1: D²_{N-1,0}·a₀ + ... = f_{N-1}   ← PDE  ✗ REPLACE
```

Modified system (tau method):
```
Row 0:   D²₀₀·a₀ + D²₀₁·a₁ + ... = f₀    ← PDE
Row 1:   D²₁₀·a₀ + D²₁₁·a₁ + ... = f₁    ← PDE
  ⋮
Row N-3: D²_{N-3,0}·a₀ + ... = f_{N-3}   ← PDE
Row N-2: T₀(-1)·a₀ + T₁(-1)·a₁ + ... = 0  ← BC at z=0
Row N-1: T₀(1)·a₀ + T₁(1)·a₁ + ... = 0   ← BC at z=1
```

Since T_n(1) = 1 and T_n(-1) = (-1)^n:
```
Row N-2: a₀ - a₁ + a₂ - a₃ + ... = 0     ← u(-1) = 0
Row N-1: a₀ + a₁ + a₂ + a₃ + ... = 0     ← u(+1) = 0
```

## Visual Representation

```
┌─────────────────────────────────────────┐
│           Original System               │
│  ┌─────────────────────────────────┐   │
│  │  PDE Row 0                      │   │
│  │  PDE Row 1                      │   │
│  │  PDE Row 2                      │   │
│  │       ⋮                         │   │
│  │  PDE Row N-3                    │   │
│  │  PDE Row N-2  ─────┐            │   │
│  │  PDE Row N-1  ─────┤            │   │
│  └────────────────────┼────────────┘   │
│                       │                 │
│                       ▼                 │
│           Tau Modified System           │
│  ┌─────────────────────────────────┐   │
│  │  PDE Row 0                      │   │
│  │  PDE Row 1                      │   │
│  │  PDE Row 2                      │   │
│  │       ⋮                         │   │
│  │  PDE Row N-3                    │   │
│  │  BC at z = 0  (replaces N-2)    │   │
│  │  BC at z = 1  (replaces N-1)    │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Why Replace the *Highest* Order Equations?

The spectral coefficients a_n for large n correspond to fine-scale (high-frequency) features of the solution. By modifying these high-order equations:

1. **Smooth solutions are preserved**: Low-order coefficients (capturing the main shape) are determined by the PDE
2. **Boundary conditions are enforced**: High-order coefficients adjust to match boundaries
3. **Spectral accuracy maintained**: For smooth solutions, high-order coefficients are already small

This is related to the concept of **tau error**: the residual in the original PDE equations that we replaced. For smooth solutions, this error is spectrally small (decays faster than any polynomial).

## Comparison with Other Methods

### Tau Method vs. Collocation Method

| Aspect | Tau Method | Collocation Method |
|--------|------------|-------------------|
| Where BCs applied | In spectral space | At physical grid points |
| Matrix structure | Modified rows | Natural incorporation |
| Implementation | Replace equations | Evaluate at boundaries |
| Accuracy | Spectral | Spectral |
| Conditioning | Can be challenging | Generally better |

**Collocation** evaluates the PDE at specific grid points (e.g., Gauss-Lobatto points) and applies BCs directly at boundary points. It's often easier to implement but mathematically equivalent for many problems.

### Tau Method vs. Galerkin Method

| Aspect | Tau Method | Galerkin Method |
|--------|------------|-----------------|
| Basis functions | Standard (Chebyshev, etc.) | Modified to satisfy BCs |
| BC handling | Via tau equations | Built into basis |
| Flexibility | Any BC type | Requires basis construction |
| Common use | General purpose | Specific BC types |

**Galerkin** uses basis functions that inherently satisfy the boundary conditions. For example, using sin(nπz) automatically satisfies u(0) = u(1) = 0. This is elegant but requires constructing appropriate bases for each BC type.

## Types of Boundary Conditions

### Dirichlet (Value Specified)

```math
u(z_{boundary}) = g
```

Tau equation:
```math
\sum_{n=0}^{N-1} a_n T_n(z_{boundary}) = g
```

### Neumann (Derivative Specified)

```math
\frac{du}{dz}\bigg|_{z_{boundary}} = g
```

Tau equation:
```math
\sum_{n=0}^{N-1} a_n T'_n(z_{boundary}) = g
```

where T'_n(z) is the derivative of the Chebyshev polynomial.

### Robin (Mixed)

```math
\alpha u(z_{boundary}) + \beta \frac{du}{dz}\bigg|_{z_{boundary}} = g
```

Tau equation:
```math
\alpha \sum_{n=0}^{N-1} a_n T_n(z_{boundary}) + \beta \sum_{n=0}^{N-1} a_n T'_n(z_{boundary}) = g
```

## Implementation in Tarang.jl

### Automatic Tau Method

Tarang.jl handles the tau method automatically when you specify boundary conditions:

```julia
using Tarang

# Define problem
problem = LBVP([u])

# Add PDE
add_equation!(problem, lap(u) - f)

# Add boundary conditions - tau method applied automatically
add_dirichlet_bc!(problem, u, "z", :left, 0.0)   # u(z=0) = 0
add_dirichlet_bc!(problem, u, "z", :right, 0.0)  # u(z=1) = 0

# Solve
solver = BoundaryValueSolver(problem)
solve!(solver)
```

### What Happens Internally

1. **Build PDE matrix**: Constructs the differentiation matrices
2. **Identify tau rows**: Determines which rows to replace based on BC count
3. **Build BC rows**: Constructs boundary condition equations
4. **Replace rows**: Substitutes tau rows with BC rows
5. **Solve**: Standard linear algebra solve

### Checking Boundary Condition Satisfaction

```julia
# After solving, verify BCs are satisfied
u_at_left = evaluate(u, z=0.0)
u_at_right = evaluate(u, z=1.0)

@assert abs(u_at_left - 0.0) < 1e-10 "Left BC not satisfied"
@assert abs(u_at_right - 0.0) < 1e-10 "Right BC not satisfied"
```

## Number of Tau Terms Required

The number of tau terms equals the number of boundary conditions, which depends on the order of the differential operator:

| Operator | Order | BCs Needed | Tau Terms |
|----------|-------|------------|-----------|
| ∂u/∂z | 1st | 1 | 1 |
| ∂²u/∂z² | 2nd | 2 | 2 |
| ∂⁴u/∂z⁴ | 4th | 4 | 4 |
| ∇² (2D) | 2nd in each | 2 per direction | 2 per direction |

## Mathematical Details

### The Tau Error

By replacing PDE equations with BCs, we introduce a "tau error" in the original equations. The modified solution u_τ satisfies:

```math
\mathcal{L}[u_\tau] = f + \sum_{k=1}^{K} \tau_k \phi_k(z)
```

where:
- L is the differential operator
- τ_k are the tau corrections (unknown)
- φ_k(z) are basis functions (typically highest-order Chebyshev polynomials)

The key result: **For smooth solutions, the tau error is spectrally small**—it decays faster than any polynomial in N.

### Condition Number Considerations

The tau method can lead to ill-conditioned matrices, especially for:
- High-order problems (4th order and above)
- Large N (many spectral modes)
- Stiff problems

Strategies to improve conditioning:
1. **Row scaling**: Scale tau rows to have similar magnitude to PDE rows
2. **Preconditioning**: Use appropriate preconditioners
3. **Alternative formulations**: Use integral formulations or different bases

## Worked Example: Heat Equation

### Problem Setup

```math
\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial z^2}, \quad z \in [-1, 1]
```

with:
- u(-1, t) = 0 (left boundary)
- u(+1, t) = 0 (right boundary)
- u(z, 0) = sin(πz) (initial condition)

### Spectral Formulation

1. Expand: u(z,t) = Σ aₙ(t) Tₙ(z)

2. Galerkin projection of PDE:
```math
\frac{d a_n}{dt} = \nu \sum_m D^{(2)}_{nm} a_m, \quad n = 0, ..., N-3
```

3. Boundary conditions (tau):
```math
\sum_m a_m T_m(-1) = 0 \quad \text{(row N-2)}
```
```math
\sum_m a_m T_m(+1) = 0 \quad \text{(row N-1)}
```

### Matrix Form

```math
\frac{d\mathbf{a}}{dt} = \mathbf{M}^{-1} \mathbf{L} \mathbf{a}
```

where M is the tau-modified mass matrix and L is the tau-modified operator.

## Common Pitfalls and Solutions

### Pitfall 1: Wrong Number of BCs

**Problem**: Specifying too few or too many boundary conditions.

**Solution**: Match the number of BCs to the operator order in each direction.

### Pitfall 2: Incompatible BCs

**Problem**: Boundary conditions that are inconsistent with the PDE.

**Solution**: Check physical consistency; ensure BCs don't over-constrain the problem.

### Pitfall 3: Ill-Conditioning

**Problem**: Matrix becomes nearly singular for large N.

**Solution**: Use row scaling, consider alternative formulations, or reduce N.

### Pitfall 4: Tau Error Accumulation

**Problem**: In time-dependent problems, tau errors can accumulate.

**Solution**: Use appropriate time-stepping schemes; IMEX methods handle this well.

## Historical Note

The tau method was introduced by **Cornelius Lanczos** in 1938 as a way to obtain approximate solutions to differential equations using polynomial expansions. It was later refined and popularized in the context of spectral methods by researchers including **Steven Orszag** and **David Gottlieb** in the 1970s-80s.

The name "tau" (τ) comes from Lanczos's notation for the error terms introduced by truncating the polynomial expansion and enforcing boundary conditions.

## References

### Textbooks

1. **Canuto, C., Hussaini, M.Y., Quarteroni, A., & Zang, T.A.** (2006). *Spectral Methods: Fundamentals in Single Domains*. Springer.
   - Chapter 3: Tau and Galerkin methods in detail
   - Excellent mathematical rigor

2. **Boyd, J.P.** (2001). *Chebyshev and Fourier Spectral Methods* (2nd ed.). Dover.
   - Chapter 6: Boundary conditions
   - Freely available online, very accessible

3. **Trefethen, L.N.** (2000). *Spectral Methods in MATLAB*. SIAM.
   - Chapter 7: Boundary value problems
   - Practical implementation focus

4. **Peyret, R.** (2002). *Spectral Methods for Incompressible Viscous Flow*. Springer.
   - Application to fluid dynamics
   - Detailed treatment of Navier-Stokes

### Key Papers

5. **Lanczos, C.** (1938). "Trigonometric interpolation of empirical and analytical functions." *Journal of Mathematics and Physics*, 17(1-4), 123-199.
   - Original tau method paper

6. **Gottlieb, D., & Orszag, S.A.** (1977). *Numerical Analysis of Spectral Methods: Theory and Applications*. SIAM.
   - Classic reference for spectral methods

7. **Orszag, S.A.** (1971). "Accurate solution of the Orr-Sommerfeld stability equation." *Journal of Fluid Mechanics*, 50(4), 689-703.
   - Influential application to hydrodynamic stability

### Online Resources

8. **Dedalus Project Documentation**: [https://dedalus-project.readthedocs.io/](https://dedalus-project.readthedocs.io/)
   - Modern spectral solver using tau method
   - Excellent tutorials and examples

9. **Lloyd N. Trefethen's Spectral Methods Course Notes**: Available at [https://people.maths.ox.ac.uk/trefethen/](https://people.maths.ox.ac.uk/trefethen/)

## See Also

- [Boundary Conditions Tutorial](../tutorials/boundary_conditions.md): Practical BC examples
- [Bases](bases.md): Spectral bases (Chebyshev, Fourier, etc.)
- [Solvers](solvers.md): Using the tau method in solvers
- [API: Problems](../api/problems.md): Adding boundary conditions programmatically
