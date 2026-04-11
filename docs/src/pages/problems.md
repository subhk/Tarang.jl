# Problems

Problems define the PDE system to be solved, including equations and boundary conditions.

## Problem Types

### IVP - Initial Value Problem

Time-dependent PDEs with initial conditions.

```julia
using Tarang

# Create IVP
problem = IVP([u, v, p, T])

# Add evolution equations (linear terms on LHS, nonlinear on RHS)
add_equation!(problem, "∂t(u) - nu*Δ(u) = -u*∂x(u)")
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u*∂x(T)")
```

### LBVP - Linear Boundary Value Problem

Steady-state linear PDEs.

```julia
# Create LBVP
problem = LBVP([phi])

# Add equations
add_equation!(problem, "Δ(phi) = rho")
```

### NLBVP - Nonlinear Boundary Value Problem

Steady-state nonlinear PDEs.

```julia
# Create NLBVP
problem = NLBVP([u, p])

# Add nonlinear equations
add_equation!(problem, "-nu*Δ(u) + ∂x(p) = -u*∂x(u)")
```

### EVP - Eigenvalue Problem

Linear stability and eigenvalue analysis.

```julia
# Create EVP with eigenvalue name
evp = EVP([u_hat, p_hat]; eigenvalue=:sigma)

# Add eigenvalue equations
add_equation!(evp, "sigma*u_hat = Δ(u_hat)")
```

## Adding Equations

### Equation Syntax

```julia
# Format: "LHS = RHS" (linear terms on LHS, nonlinear on RHS)
add_equation!(problem, "∂t(u) - nu*Δ(u) = rhs_expression")

# Multiple terms
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∂x(p) = -u*∂x(u)")
```

### Equation Sizing

The solver automatically determines each equation's row count in the system matrix from the expression's output type (scalar, vector, tensor, etc.). Equations can be added in any order — no specific ordering is required.

```julia
problem = IVP([q, ψ, u, tau_ψ])

# Any order is fine:
add_equation!(problem, "Δ(ψ) + tau_ψ - q = 0")           # scalar → D rows
add_equation!(problem, "u - skew(grad(ψ)) = 0")           # vector → 2D rows
add_equation!(problem, "∂t(q) + nu*Δ⁴(q) = -u⋅∇(q)")     # scalar → D rows
add_bc!(problem, "integ(ψ) = 0")                           # constraint → 1 row
```

### Supported Operations

- Derivatives: `∂x`, `∂y`, `∂z`, `∂t`, `Δ`, `∇`, `div`, `curl`
- Arithmetic: `+`, `-`, `*`, `/`
- Functions: `sin`, `cos`, `exp`, `sqrt`
- Parameters: Any name in `problem.namespace`

### Parameters

```julia
# Define parameters
problem.namespace["nu"] = 0.01
problem.namespace["Ra"] = 1e6
problem.namespace["Pr"] = 1.0

# Use in equations
add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")
add_equation!(problem, "∂t(T) - Δ(T) = Ra*Pr*w")
```

## First-Order Formulation

For problems involving Chebyshev bases, Tarang supports a first-order reduction (tau method) that replaces `Δ(f)` with `div(grad_f)` where `grad_f` includes a tau-lifting term. This ensures correct boundary condition enforcement.

```julia
# Derivative basis and lift closure
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A) = lift(A, lift_basis, -1)

# First-order gradient substitutions
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_b = grad(b) + ez * τ_lift(tau_b1)

# Equations use div(grad_f) instead of Δ(f)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -u⋅∇(b)")
```

The advection operator `u⋅∇(f)` is automatically expanded component-wise to `Σᵢ uᵢ ∂ᵢf`, so it works for both scalar and vector fields without manual expansion.

## Boundary Conditions

Boundary conditions use the same `add_equation!` function as PDEs. The syntax `field(coord=value)` is auto-detected and converted to the appropriate boundary condition.

### Dirichlet (Value)

```julia
# u = value at location
add_equation!(problem, "u(z=0) = 0")  # u=0 at z=0
add_equation!(problem, "T(z=1) = 0")  # T=0 at z=1
```

### Neumann (Derivative)

```julia
# du/dz = value at location
add_equation!(problem, "∂z(T)(z=1) = 0")  # ∂T/∂z=0 at z=1
```

### Robin (Mixed)

```julia
# α*u + β*du/dn = γ
add_equation!(problem, "1.0*T(z=0) + 1.0*∂z(T)(z=0) = 0")
```

### Stress-Free

```julia
# du/dz = 0 (free surface)
add_equation!(problem, "∂z(u)(z=1) = 0")
```

### No-Slip

```julia
# u = 0 (solid wall)
add_equation!(problem, "u(z=0) = 0")
```

## Problem Validation

```julia
# Check problem is well-posed
is_valid = validate_problem(problem)

# Reports issues:
# - Missing boundary conditions
# - Incompatible equation counts
# - Parameter issues
```

## Common Problem Patterns

### Heat Equation

```julia
problem = IVP([T])
problem.namespace["kappa"] = 0.01
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")
add_equation!(problem, "T(z=0) = 1")
add_equation!(problem, "T(z=1) = 0")
```

### Incompressible Navier-Stokes

```julia
problem = IVP([ux, uz, p])
problem.namespace["nu"] = 0.01

# Momentum (linear LHS, nonlinear RHS)
add_equation!(problem, "∂t(ux) - nu*Δ(ux) + ∂x(p) = -ux*∂x(ux) - uz*∂z(ux)")
add_equation!(problem, "∂t(uz) - nu*Δ(uz) + ∂z(p) = -ux*∂x(uz) - uz*∂z(uz)")

# Continuity
add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")

# No-slip walls
for field in ["ux", "uz"]
    add_equation!(problem, "$(field)(z=0) = 0")
    add_equation!(problem, "$(field)(z=1) = 0")
end
```

### Rayleigh-Bénard

```julia
problem = IVP([ux, uz, p, T])
problem.namespace["Ra"] = 1e6
problem.namespace["Pr"] = 1.0

add_equation!(problem, "∂t(ux) - Pr*Δ(ux) + ∂x(p) = -ux*∂x(ux) - uz*∂z(ux)")
add_equation!(problem, "∂t(uz) - Pr*Δ(uz) + ∂z(p) - Ra*Pr*T = -ux*∂x(uz) - uz*∂z(uz)")
add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")
add_equation!(problem, "∂t(T) - Δ(T) = -ux*∂x(T) - uz*∂z(T)")

# Boundary conditions
add_equation!(problem, "ux(z=0) = 0")
add_equation!(problem, "ux(z=1) = 0")
add_equation!(problem, "uz(z=0) = 0")
add_equation!(problem, "uz(z=1) = 0")
add_equation!(problem, "T(z=0) = 1")  # Hot bottom
add_equation!(problem, "T(z=1) = 0")  # Cold top
```

### Poisson Equation (BVP)

```julia
problem = LBVP([phi])
add_equation!(problem, "Δ(phi) = rho")
add_equation!(problem, "phi(z=0) = 0")
add_equation!(problem, "phi(z=1) = 0")
```

## See Also

- [Solvers](solvers.md): Solving problems
- [Boundary Conditions Tutorial](../tutorials/boundary_conditions.md): Detailed BC guide
- [API: Problems](../api/problems.md): Complete reference
