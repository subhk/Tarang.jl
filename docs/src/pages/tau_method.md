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

where $T_n(z)$ are Chebyshev polynomials and $a_n$ are the spectral coefficients we need to find.

**The challenge**: The spectral coefficients $a_n$ are global—they affect the solution everywhere, not just at specific points. We cannot simply "set" boundary values; instead, we must find coefficients that simultaneously:
1. Satisfy the PDE in the interior
2. Satisfy the boundary conditions at the edges

### The Core Idea of the Tau Method

The tau method solves this by a clever trade-off:

> **Key Insight**: We have N unknowns (spectral coefficients $a_0$, $a_1$, ..., $a_{N-1}$) and need N equations. Instead of using N equations from the PDE, we use (N-k) equations from the PDE and k equations from the boundary conditions.

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

Expand $u(z)$ in Chebyshev polynomials:
```math
u(z) = \sum_{n=0}^{N-1} a_n T_n(z)
```

Recall key properties of Chebyshev polynomials:
- $T_n(z) = cos(n · arccos(z))$ for $z$ ∈ [-1, 1]
- $T_n(1) = 1$ for all $n$
- $T_n(-1) = (-1)^n$

### Step 3: The Differentiation Matrix

Taking derivatives in spectral space:
```math
\frac{d^2 u}{dz^2} = \sum_{n=0}^{N-1} a_n \frac{d^2 T_n}{dz^2} = \sum_{n=0}^{N-1} b_n T_n(z)
```

The coefficients $b_n$ are related to a_n through the **differentiation matrix** D²:
```math
\mathbf{b} = D^{(2)} \mathbf{a}
```

### Step 4: The PDE in Spectral Space (Without Tau)

Matching coefficients of $T_n(z)$ on both sides of the PDE gives N equations:
```math
\sum_{m=0}^{N-1} D^{(2)}_{nm} a_m = f_n, \quad n = 0, 1, ..., N-1
```

where $f_n$ are the spectral coefficients of $f(z)$.

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

Since $T_n(1)$ = 1 and $T_n(-1) = (-1)^n$:
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

The spectral coefficients $a_n$ for large $n$ correspond to fine-scale (high-frequency) features of the solution. By modifying these high-order equations:

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

**Galerkin** uses basis functions that inherently satisfy the boundary conditions. For example, using sin(nπz) automatically satisfies $u(0) = u(1) = 0$. This is elegant but requires constructing appropriate bases for each BC type.

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

where $T'_n(z)$ is the derivative of the Chebyshev polynomial.

### Robin (Mixed)

```math
\alpha u(z_{boundary}) + \beta \frac{du}{dz}\bigg|_{z_{boundary}} = g
```

Tau equation:
```math
\alpha \sum_{n=0}^{N-1} a_n T_n(z_{boundary}) + \beta \sum_{n=0}^{N-1} a_n T'_n(z_{boundary}) = g
```

## Implementation in Tarang.jl

Tarang.jl follows the **Dedalus approach**: users must explicitly create tau fields and add them to equations using the `lift()` operator. This design provides full transparency and control over how boundary conditions interact with your equations.

### Required Steps

To apply boundary conditions in Tarang.jl, you must:

1. **Create tau fields** - One for each boundary condition
2. **Add tau fields to the problem** - Include them as variables
3. **Add lift(tau) terms to equations** - Use the `lift()` operator
4. **Specify tau_field in boundary conditions** - Link BCs to their tau fields

### Complete Example

```julia
using Tarang

# Create domain
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords)

xbasis = RealFourier(coords["x"]; size=64, bounds=(0, 2π))
zbasis = ChebyshevT(coords["z"]; size=64, bounds=(0, 1))

# Main field
u = ScalarField(dist, "u", (xbasis, zbasis))

# Step 1: Create tau fields (one per boundary condition)
# Tau fields live on the "other" bases (here: xbasis only, not zbasis)
tau_u1 = ScalarField(dist, "tau_u1", (xbasis,))  # For BC at z=0
tau_u2 = ScalarField(dist, "tau_u2", (xbasis,))  # For BC at z=1

# Step 2: Add all fields (including tau) to problem
problem = LBVP([u, tau_u1, tau_u2])

# Step 3: Add equation with lift() operators
# The lift() terms place tau values at specific spectral modes
add_equation!(problem,
    lap(u) + lift(tau_u1, zbasis, -1) + lift(tau_u2, zbasis, -2) - f
)

# Step 4: Add boundary conditions
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "u(z=1) = 0")

# Solve
solver = BoundaryValueSolver(problem)
solve!(solver)
```

### The `lift()` Operator

The `lift()` operator "lifts" a lower-dimensional tau field into the full domain by placing its values at specific spectral modes:

```julia
lift(tau_field, basis, n)
```

**Arguments:**
- `tau_field` - The tau field (lives on reduced bases)
- `basis` - The basis along which to lift (typically the non-periodic direction)
- `n` - Which spectral mode to modify

**Mode indexing (`n`):**
- `n = 0` → First mode (mode 0)
- `n = -1` → Last mode (N-1)
- `n = -2` → Second-to-last mode (N-2)

**Why specific modes?** The tau method works by replacing the highest-order spectral equations with boundary conditions. The `lift()` operator places tau contributions at these modes.

### Comparison: Tarang vs. Dedalus

Tarang.jl closely follows the Dedalus design pattern:

| Feature | Dedalus (Python) | Tarang.jl (Julia) |
|---------|------------------|-------------------|
| Create tau field | `tau_p = dist.Field(name='tau_p', bases=xbasis)` | `tau_p = ScalarField(dist, "tau_p", (xbasis,))` |
| Lift operator | `lift(tau_p, -1)` | `lift(tau_p, zbasis, -1)` |
| Add to equation | `"lap(u) + lift(tau_p, -1) = f"` | `lap(u) + lift(tau_p, zbasis, -1) - f` |
| Add BC | `problem.add_bc("u(z=0) = 0")` | `add_equation!(problem, "u(z=0) = 0")` |

**Dedalus example:**
```python
# Dedalus (Python)
tau_u1 = dist.Field(name='tau_u1', bases=xbasis)
tau_u2 = dist.Field(name='tau_u2', bases=xbasis)

problem.add_equation("lap(u) + lift(tau_u1, -1) + lift(tau_u2, -2) = f")
problem.add_bc("u(z=0) = 0")
problem.add_bc("u(z=1) = 0")
```

**Tarang.jl equivalent:**
```julia
# Tarang.jl (Julia)
tau_u1 = ScalarField(dist, "tau_u1", (xbasis,))
tau_u2 = ScalarField(dist, "tau_u2", (xbasis,))

problem = LBVP([u, tau_u1, tau_u2])
add_equation!(problem, lap(u) + lift(tau_u1, zbasis, -1) + lift(tau_u2, zbasis, -2) - f)
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "u(z=1) = 0")
```

### Why Explicit Tau Fields?

The Dedalus approach (explicit tau fields) has several advantages:

1. **Transparency**: You see exactly how BCs interact with equations
2. **Debugging**: Easy to inspect tau field values after solving
3. **Flexibility**: Custom tau arrangements for coupled systems
4. **Consistency**: Same pattern for all problem types (IVP, BVP, EVP)
5. **Documentation**: Code is self-documenting about BC structure

### What Happens Internally

1. **Build PDE matrix**: Constructs differentiation matrices for all operators
2. **Process lift operators**: Inserts tau contributions at specified spectral modes
3. **Build BC equations**: Constructs boundary condition rows linking to tau fields
4. **Assemble system**: Combines PDE + lift + BC into linear system
5. **Solve**: Standard linear algebra solve
6. **Extract solution**: Both u and tau values are solved simultaneously

### Checking Boundary Condition Satisfaction

```julia
# After solving, verify BCs are satisfied
u_at_left = evaluate(u, z=0.0)
u_at_right = evaluate(u, z=1.0)

@assert abs(u_at_left - 0.0) < 1e-10 "Left BC not satisfied"
@assert abs(u_at_right - 0.0) < 1e-10 "Right BC not satisfied"

# You can also inspect tau field values
println("tau_u1 values: ", tau_u1.data_c)
println("tau_u2 values: ", tau_u2.data_c)
```

## Number of Tau Terms Required

The number of tau terms equals the number of boundary conditions, which depends on the order of the differential operator:

| Operator | Order | BCs Needed | Tau Terms |
|----------|-------|------------|-----------|
| ∂u/∂z | 1st | 1 | 1 |
| ∂²u/∂z² | 2nd | 2 | 2 |
| ∂⁴u/∂z⁴ | 4th | 4 | 4 |
| ∇² (2D) | 2nd in each | 2 per direction | 2 per direction |

## Pressure in Incompressible Flows

When solving incompressible Navier-Stokes equations, the pressure requires special treatment because it appears only as a gradient. The divergence equation `div(u) = 0` becomes degenerate at the mean Fourier mode (k=0).

### The Tau Solution

Add a `tau_p` term to the continuity equation to remove the mathematical degeneracy:

```julia
# Continuity equation with tau_p
add_equation!(problem, "div(u) + tau_p = 0")

# Boundary conditions
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
```

**Why `tau_p` is needed:**
- The divergence equation `div(u) = 0` has a null space at k=0
- `tau_p` is a scalar constant that absorbs this degeneracy
- Without it, the system matrix becomes singular

### Complete Example

```julia
# Vector velocity field
u = VectorField(dist, coords, "u", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))

# Tau fields
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,))
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,))
tau_p = ScalarField(dist, "tau_p", ())  # Scalar constant

# Problem with all fields
problem = IVP([u, p, tau_u1, tau_u2, tau_p])

# Equations
add_equation!(problem, "div(u) + tau_p = 0")  # tau_p removes degeneracy
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) + lift(tau_u2) = -u⋅∇(u)")

# Boundary conditions
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
```

**Note:** The pressure is determined up to an arbitrary constant. For most simulations, only the pressure gradient matters, so this ambiguity doesn't affect the physics. If you need absolute pressure values, you can manually subtract the mean after solving.

## First-Order Formulation with Derivative Basis

For better conditioning and stability, Dedalus recommends converting second-order equations to first-order form using auxiliary gradient variables with tau corrections.

### The Derivative Basis

When differentiating Chebyshev T polynomials, the result is expressed in terms of Chebyshev U polynomials (the "derivative basis"):

```julia
# Get the derivative basis for ChebyshevT
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))
lift_basis = derivative_basis(z_basis)  # Returns ChebyshevU
```

This is mathematically rigorous: ∂/∂z(T_n) is proportional to U_{n-1}.

### First-Order Formulation (Dedalus Style)

Instead of directly using the Laplacian, introduce an auxiliary gradient variable with tau correction:

```julia
# Tau fields
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,))  # For gradient substitution
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,))  # For evolution equation

# Lift basis from derivative basis
lift_basis = derivative_basis(z_basis)

# First-order gradient substitution with tau correction
# grad_u = ∇(u) + ez*lift(tau_u1, lift_basis, -1)
add_substitution!(problem, "grad_u", "∇(u) + ez*lift(tau_u1)")

# Continuity: trace(grad_u) + tau_p = 0
add_equation!(problem, "trace(grad_u) + tau_p = 0")

# Momentum using div(grad_u) instead of Δ(u)
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) + lift(tau_u2) = -u⋅∇(u)")
```

### Why First-Order is Better

| Aspect | Second-Order (Δ) | First-Order (div(grad)) |
|--------|------------------|-------------------------|
| Tau fields needed | 1 per equation | 2 per equation |
| Matrix conditioning | Can be poor for high N | Better conditioning |
| BC flexibility | Standard | More control |
| Dedalus default | No | Yes |

### When to Use Each

**Second-Order (simpler)**:
- Quick prototyping
- Moderate resolution (N < 128)
- Low-order problems (2nd order PDEs)

**First-Order (recommended)**:
- Production runs
- High resolution (N > 128)
- High-order problems (4th order, biharmonic)
- When conditioning issues arise

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
