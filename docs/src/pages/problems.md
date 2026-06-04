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

Steady-state linear PDEs. Boundary conditions in a bounded (Chebyshev/Jacobi)
direction use the **tau method**: declare one `tau` variable per boundary
condition, lift each into the bulk equation via
`lift(tau, derivative_basis(basis, 2), -k)` (registered with `add_parameters!`),
and declare the boundary conditions with `add_bc!` (not `add_equation!`).

```julia
# 2D Poisson  Δu = -2,  u = 0 on both z walls  (solution u = z(Lz - z))
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 2π))   # periodic (separable)
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))  # bounded  (coupled)
dom = Domain(dist, (xb, zb))

u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)           # one tau per BC
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = LBVP([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
ensure_layout!(u, :g)            # scatter writes coefficients; switch to grid
```

!!! note "1D pure-Chebyshev BVP"
    Verified BVPs include at least one separable (Fourier) direction. A pure
    single-axis Chebyshev BVP currently mis-scatters the solution; add a Fourier
    direction or use the EVP path for 1D spectra.

### NLBVP - Nonlinear Boundary Value Problem

Steady-state nonlinear PDEs. Same tau-method boundary handling as the LBVP
(tau variables + `lift` + `add_bc!`). Put the nonlinear terms on the right-hand
side; the solver linearizes them with a symbolic Frechet derivative and runs a
per-Fourier-mode Newton iteration.

```julia
# Manufactured nonlinearity  Δu = u² + g,  with g = -2 - u_exact²  so u_exact = z(Lz - z)
# domain / fields / taus as in the LBVP example above (u, tau1, tau2, lb2)
g = ScalarField(dom, "g"); ensure_layout!(g, :g)
zg = create_meshgrid(dom; on_device=false)["z"]
get_grid_data(g) .= -2 .- (zg .* (1.0 .- zg)).^2

problem = NLBVP([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2), g=g)
add_equation!(problem, "Δ(u) + l1 + l2 = u*u + g")   # nonlinearity on the RHS
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solver.tolerance = 1e-10
ensure_layout!(u, :g); get_grid_data(u) .= 0.0       # initial guess
solve!(solver)
ensure_layout!(u, :g)
```

### EVP - Eigenvalue Problem

Linear stability and eigenvalue analysis. Tarang solves the generalized problem
`L x = σ M x`, where the mass matrix `M` is assembled from the **time-derivative
terms** `dt(·)`: the eigenvalue *replaces* the time derivative (`dt(u) → σ u`).
Keep the `dt(u)` term in the equation — do **not** multiply the eigenvalue symbol
into it (`σ*u = …` builds an empty `M` and returns no eigenvalues). Boundary
conditions use the same tau method (`tau` vars + `lift` + `add_bc!`).

```julia
# 1D diffusion eigenproblem  σu = Δu, Dirichlet;  eigenvalues σ_n = -(nπ/Lz)²
coords = CartesianCoordinates("z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
zb     = ChebyshevT(coords["z"]; size=32, bounds=(0.0, 1.0))
dom    = Domain(dist, (zb,))

u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (), Float64)
tau2 = ScalarField(dist, "tau2", (), Float64)
lb2  = derivative_basis(zb, 2)

evp = EVP([u, tau1, tau2]; eigenvalue=:σ)
add_parameters!(evp; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(evp, "dt(u) - Δ(u) - l1 - l2 = 0")   # dt(u) → σu marks M
add_bc!(evp, "u(z=0)   = 0")
add_bc!(evp, "u(z=1.0) = 0")

solver = EigenvalueSolver(evp; nev=5, which=:SM)   # 5 smallest-magnitude
eigenvalues, eigenvectors = solve!(solver)
# |eigenvalues| ≈ (nπ)² = 9.87, 39.48, 88.83, ...
```

`EigenvalueSolver` accepts only `nev`, `which` (∈ `:LM :SM :LR :SR :LI :SI`),
`target`, and `matsolver`.

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

A steady BVP needs the tau method, and verified domains include at least one
separable (Fourier) direction — a pure 1D Chebyshev Poisson currently
mis-scatters. Here is the 2D form (`Δu = -2`, `u = 0` on both `z` walls):

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 2π))
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))
dom = Domain(dist, (xb, zb))

u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = LBVP([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = BoundaryValueSolver(problem); solve!(solver); ensure_layout!(u, :g)
```

## See Also

- [Solvers](solvers.md): Solving problems
- [Boundary Conditions Tutorial](../tutorials/boundary_conditions.md): Detailed BC guide
- [API: Problems](../api/problems.md): Complete reference
