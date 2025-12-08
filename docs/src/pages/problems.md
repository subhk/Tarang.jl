# Problems

Problems define the PDE system to be solved, including equations and boundary conditions.

## Problem Types

### IVP - Initial Value Problem

Time-dependent PDEs with initial conditions.

```julia
using Tarang

# Create IVP
problem = IVP([u, v, p, T])

# Add evolution equations
add_equation!(problem, "∂ₜ(u) = -u*∂x(u) + nu*Δ(u)")
add_equation!(problem, "∂ₜ(T) = -u*∂x(T) + kappa*Δ(T)")
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
add_equation!(problem, "u*∂x(u) + ∂x(p) = nu*Δ(u)")
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
# Format: "LHS = RHS"
add_equation!(problem, "∂ₜ(u) = rhs_expression")

# Multiple terms
add_equation!(problem, "∂ₜ(u) + u*∂x(u) = nu*Δ(u) - ∂x(p)")
```

### Supported Operations

- Derivatives: `∂x`, `∂y`, `∂z`, `∂ₜ`, `Δ`, `∇`, `div`, `curl`
- Arithmetic: `+`, `-`, `*`, `/`
- Functions: `sin`, `cos`, `exp`, `sqrt`
- Parameters: Any name in `problem.parameters`

### Parameters

```julia
# Define parameters
problem.parameters["nu"] = 0.01
problem.parameters["Ra"] = 1e6
problem.parameters["Pr"] = 1.0

# Use in equations
add_equation!(problem, "∂ₜ(u) = nu*Δ(u)")
add_equation!(problem, "∂ₜ(T) = Ra*Pr*w + Δ(T)")
```

## Boundary Conditions

### Dirichlet (Value)

```julia
# u = value at location
add_dirichlet_bc!(problem, "u(z=0) = 0")  # u=0 at z=0
add_dirichlet_bc!(problem, "T(z=1) = 0")  # T=0 at z=1
```

### Neumann (Derivative)

```julia
# du/dz = value at location
add_neumann_bc!(problem, "∂z(T)(z=1) = 0")  # ∂T/∂z=0 at z=1
```

### Robin (Mixed)

```julia
# α*u + β*du/dn = γ
add_robin_bc!(problem, "1.0*T(z=0) + 1.0*∂z(T)(z=0) = 0")
```

### Stress-Free

```julia
# du/dz = 0 (free surface)
add_stress_free_bc!(problem, "u(z=1)")
```

### No-Slip

```julia
# u = 0 (solid wall)
add_no_slip_bc!(problem, "u(z=0)")
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
problem.parameters["kappa"] = 0.01
add_equation!(problem, "∂ₜ(T) = kappa*Δ(T)")
add_dirichlet_bc!(problem, "T(z=0) = 1")
add_dirichlet_bc!(problem, "T(z=1) = 0")
```

### Incompressible Navier-Stokes

```julia
problem = IVP([ux, uz, p])
problem.parameters["nu"] = 0.01

# Momentum
add_equation!(problem, "∂ₜ(ux) + ux*∂x(ux) + uz*∂z(ux) + ∂x(p) = nu*Δ(ux)")
add_equation!(problem, "∂ₜ(uz) + ux*∂x(uz) + uz*∂z(uz) + ∂z(p) = nu*Δ(uz)")

# Continuity
add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")

# No-slip walls
for field in ["ux", "uz"]
    add_dirichlet_bc!(problem, "$(field)(z=0) = 0")
    add_dirichlet_bc!(problem, "$(field)(z=1) = 0")
end
```

### Rayleigh-Bénard

```julia
problem = IVP([ux, uz, p, T])
problem.parameters["Ra"] = 1e6
problem.parameters["Pr"] = 1.0

add_equation!(problem, "∂ₜ(ux) + ux*∂x(ux) + uz*∂z(ux) + ∂x(p) = Pr*Δ(ux)")
add_equation!(problem, "∂ₜ(uz) + ux*∂x(uz) + uz*∂z(uz) + ∂z(p) = Pr*Δ(uz) + Ra*Pr*T")
add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")
add_equation!(problem, "∂ₜ(T) + ux*∂x(T) + uz*∂z(T) = Δ(T)")

# Boundary conditions
add_dirichlet_bc!(problem, "ux(z=0) = 0")
add_dirichlet_bc!(problem, "ux(z=1) = 0")
add_dirichlet_bc!(problem, "uz(z=0) = 0")
add_dirichlet_bc!(problem, "uz(z=1) = 0")
add_dirichlet_bc!(problem, "T(z=0) = 1")  # Hot bottom
add_dirichlet_bc!(problem, "T(z=1) = 0")  # Cold top
```

### Poisson Equation (BVP)

```julia
problem = LBVP([phi])
add_equation!(problem, "Δ(phi) = rho")
add_dirichlet_bc!(problem, "phi(z=0) = 0")
add_dirichlet_bc!(problem, "phi(z=1) = 0")
```

## See Also

- [Solvers](solvers.md): Solving problems
- [Boundary Conditions Tutorial](../tutorials/boundary_conditions.md): Detailed BC guide
- [API: Problems](../api/problems.md): Complete reference
