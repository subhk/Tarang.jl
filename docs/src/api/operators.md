# Operators API

Operators compute derivatives and other mathematical operations on fields. Tarang.jl provides a symbolic syntax for natural mathematical notation in equations.

## Overview

Tarang.jl supports:
- **Differential operators**: grad (∇), div, curl, lap (Δ, ∇²)
- **Coordinate derivatives**: dx, dy, dz, dr, etc.
- **Time derivatives**: dt (∂ₜ)
- **Field operations**: dot (⋅), cross (×)
- **Custom operators**: User-defined operations

## Unicode Operators

Tarang.jl supports Unicode mathematical symbols for more readable code:

| ASCII | Unicode | Description |
|-------|---------|-------------|
| `grad(f)` | `∇(f)` | Gradient |
| `lap(f)` | `Δ(f)` or `∇²(f)` | Laplacian |
| `dt(f)` | `∂ₜ(f)` | Time derivative |
| `dot(u, v)` | `u ⋅ v` | Dot product |
| `cross(u, v)` | `u × v` | Cross product |

**Example** - Navier-Stokes with Unicode:
```julia
# Traditional ASCII
add_equation!(problem, "dt(u) + dot(u, grad(u)) = -grad(p) + nu*lap(u)")

# With Unicode (more readable)
add_equation!(problem, "∂ₜ(u) + u⋅∇(u) = -∇(p) + nu*Δ(u)")
```

**Typing Unicode in Julia**:
- `∇` : Type `\nabla` then press Tab
- `Δ` : Type `\Delta` then press Tab
- `∂ₜ` : Type `\partial` Tab `\_t` Tab
- `⋅` : Type `\cdot` then press Tab
- `×` : Type `\times` then press Tab

---

## Differential Operators

### Gradient (grad / ∇)

Computes the gradient of a scalar field, returning a vector field.

**Syntax**:
```julia
# In equations
add_equation!(problem, "∇(p)")  # or "grad(p)"

# Programmatic
∇p = ∇(p)  # or grad(p)
```

**Definitions**:

**Cartesian**:
```math
\nabla p = \frac{\partial p}{\partial x}\hat{x} + \frac{\partial p}{\partial y}\hat{y} + \frac{\partial p}{\partial z}\hat{z}
```

**Spherical**:
```math
\nabla p = \frac{\partial p}{\partial r}\hat{r} + \frac{1}{r}\frac{\partial p}{\partial \theta}\hat{\theta} + \frac{1}{r\sin\theta}\frac{\partial p}{\partial \phi}\hat{\phi}
```

**Examples**:

```julia
# 2D gradient
coords = CartesianCoordinates("x", "z")
problem = IVP([u, w, p])

# Pressure gradient in momentum equation
add_equation!(problem, "∂ₜ(u) = -∇(p)")

# Expands to:
# dt(u_x) = -dx(p)
# dt(u_z) = -dz(p)
```

```julia
# 3D gradient with custom usage
∇T = ∇(T)  # Returns VectorField
# Components: ∇T.components[1] = dx(T), etc.
```

**Return type**: VectorField

---

### Divergence (div)

Computes the divergence of a vector field, returning a scalar field.

**Syntax**:
```julia
# In equations
add_equation!(problem, "div(u) = 0")

# Programmatic
div_u = div(u)
```

**Definitions**:

**Cartesian**:
```math
\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z}
```

**Spherical**:
```math
\nabla \cdot \mathbf{u} = \frac{1}{r^2}\frac{\partial (r^2 u_r)}{\partial r} + \frac{1}{r\sin\theta}\frac{\partial (\sin\theta\, u_\theta)}{\partial \theta} + \frac{1}{r\sin\theta}\frac{\partial u_\phi}{\partial \phi}
```

**Examples**:

```julia
# Incompressibility constraint
problem = IVP([u, v, w, p])
add_equation!(problem, "div(u) = 0")
```

```julia
# Mass conservation with source
add_equation!(problem, "div(u) = S")
```

**Return type**: ScalarField

---

### Curl (curl)

Computes the curl of a vector field, returning a vector field.

**Syntax**:
```julia
# In equations
add_equation!(problem, "omega = curl(u)")

# Programmatic
ω = curl(u)
```

**Definitions**:

**Cartesian (3D)**:
```math
\nabla \times \mathbf{u} = \left(\frac{\partial u_z}{\partial y} - \frac{\partial u_y}{\partial z}\right)\hat{x} + \left(\frac{\partial u_x}{\partial z} - \frac{\partial u_z}{\partial x}\right)\hat{y} + \left(\frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}\right)\hat{z}
```

**2D (returns scalar vorticity)**:
```math
\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
```

**Examples**:

```julia
# Vorticity equation (3D)
problem = IVP([u, omega])
add_equation!(problem, "dt(omega) = curl(u × omega)")
```

```julia
# 2D vorticity
coords = CartesianCoordinates("x", "y")
problem = IVP([u, v, omega])
add_equation!(problem, "omega = dx(v) - dy(u)")
```

**Return type**: VectorField (3D) or ScalarField (2D)

---

### Laplacian (lap)

Computes the Laplacian (second derivative) of a field.

**Syntax**:
```julia
# In equations
add_equation!(problem, "∂ₜ(T) = kappa*Δ(T)")

# Programmatic
∇²T = Δ(T)
```

**Definitions**:

**Cartesian**:
```math
\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}
```

**Spherical**:
```math
\nabla^2 f = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial f}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta\frac{\partial f}{\partial \theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}
```

**Examples**:

```julia
# Diffusion equation
add_equation!(problem, "∂ₜ(T) = kappa*Δ(T)")
```

```julia
# Viscous term in Navier-Stokes
add_equation!(problem, "∂ₜ(u) = nu*Δ(u) - ∇(p)")
```

```julia
# Poisson equation (BVP)
problem = LBVP([phi])
add_equation!(problem, "Δ(phi) = rho")
```

**Works on**: ScalarField, VectorField (applies componentwise)

---

## Coordinate Derivatives

### First Derivatives

Partial derivatives with respect to coordinate directions.

**Syntax**:
```julia
dx(field)   # ∂/∂x
dy(field)   # ∂/∂y
dz(field)   # ∂/∂z
dr(field)   # ∂/∂r (spherical/polar)
dtheta(field)  # ∂/∂θ
dphi(field)    # ∂/∂φ
```

**Examples**:

```julia
# Advection term
add_equation!(problem, "∂ₜ(T) = -u*dx(T) - w*dz(T)")
```

```julia
# Shear
add_equation!(problem, "S = dx(u) + dz(w)")
```

```julia
# Custom derivative
dudz = dz(u)  # Returns field with ∂u/∂z
```

**Implementation**:
- **Fourier**: Multiplication by ik in spectral space
- **Chebyshev/Legendre**: Sparse matrix multiplication using recurrence relations

---

### Higher Derivatives

Multiple derivatives can be composed:

**Syntax**:
```julia
# Second derivatives
dx(dx(T))   # ∂²T/∂x²
dz(dz(T))   # ∂²T/∂z²

# Mixed derivatives
dx(dz(T))   # ∂²T/∂x∂z

# Higher order
dx(dx(dx(T)))  # ∂³T/∂x³
```

**Examples**:

```julia
# Biharmonic operator
add_equation!(problem, "lap(lap(psi)) = omega")

# Equivalent to:
add_equation!(problem, "dx(dx(dx(dx(psi)))) + 2*dx(dx(dz(dz(psi)))) + dz(dz(dz(dz(psi)))) = omega")
```

```julia
# Hyperdiffusion (for numerical stability)
add_equation!(problem, "∂ₜ(T) = -nu4*Δ(Δ(T))")
```

---

## Time Derivatives

### dt / ∂ₜ Operator

Time derivative for initial value problems.

**Syntax**:
```julia
dt(field)   # ASCII
∂ₜ(field)   # Unicode (type \partial Tab \_t Tab)
```

**Examples**:

```julia
# Evolution equations
add_equation!(problem, "∂ₜ(u) = -u*dx(u) + nu*Δ(u)")
add_equation!(problem, "∂ₜ(T) = -u*dx(T) + kappa*Δ(T)")
```

**Note**: Only use in IVP (Initial Value Problems). Not valid for BVP or EVP.

---

## Vector Operations

### Dot Product / Advection

**Syntax**:
```julia
# In equations - use vector notation directly
add_equation!(problem, "∂ₜ(T) = -u⋅∇(T)")

# For vector advection (nonlinear term)
add_equation!(problem, "∂ₜ(u) = -u⋅∇(u)")
```

**Example**:

```julia
# Scalar advection: -u·∇T
add_equation!(problem, "∂ₜ(T) = -u⋅∇(T)")

# Vector advection (Navier-Stokes nonlinear term)
add_equation!(problem, "∂ₜ(u) - nu*Δ(u) + ∇(p) = -u⋅∇(u)")
```

---

### Cross Product

**Syntax**:
```julia
# Use helper function
u_cross_omega = cross_product(u, omega)

# Or component form for specific cases
```

**Example**:

```julia
# Vorticity equation: ∂ω/∂t = ∇×(u×ω)
problem = IVP([ux, uy, uz, omega_x, omega_y, omega_z])

# Compute u × ω
u_cross_omega = cross_product(u, omega)

# Then: ∇×(u×ω)
add_equation!(problem, "dt(omega) = curl(u_cross_omega)")
```

---

## Composite Operators

Combine operators for complex expressions.

### Vector Laplacian

```julia
# For vector field u
# ∇²u = (∇²u_x, ∇²u_y, ∇²u_z)

add_equation!(problem, "∂ₜ(u) = nu*Δ(u)")
# Automatically applies componentwise
```

### Advection Operator

```julia
# u·∇u (nonlinear advection) - use vector notation directly

# Navier-Stokes momentum equation:
add_equation!(problem, "∂ₜ(u) - nu*Δ(u) + ∇(p) = -u⋅∇(u)")

# Scalar advection:
add_equation!(problem, "∂ₜ(T) - kappa*Δ(T) = -u⋅∇(T)")

# With buoyancy (Boussinesq):
add_equation!(problem, "∂ₜ(u) - nu*Δ(u) + ∇(p) = -u⋅∇(u) + Ra*T*ez")
```

### Strain Rate Tensor

```julia
# S_ij = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)

function strain_rate_tensor(u)
    # Returns TensorField
    S = TensorField(u.distributor, u.coords, "S", u.bases, symmetric=true)

    S[1,1] = dx(u.components[1])
    S[1,2] = 0.5 * (dx(u.components[2]) + dy(u.components[1]))
    S[2,2] = dy(u.components[2])
    # ... etc

    return S
end
```

---

## Operator Properties

### Linearity

All operators are linear:

```julia
# ∇(αf + βg) = α∇f + β∇g
∇(alpha*f + beta*g) == alpha*∇(f) + beta*∇(g)
```

### Commutativity

Partial derivatives commute:

```julia
# ∂²f/∂x∂z = ∂²f/∂z∂x
dx(dz(f)) == dz(dx(f))
```

### Product Rule

```julia
# ∇(fg) = f∇g + g∇f
∇(f*g) == f*∇(g) + g*∇(f)
```

---

## Custom Operators

### Defining Custom Operators

```julia
# Define helper function
function my_operator(field, param)
    # Custom operation
    result = dx(field) + param * Δ(field)
    return result
end

# Use in equations
add_equation!(problem, "∂ₜ(T) = my_operator(T, kappa)")
```

### Registered Custom Operators

For operators used frequently in equations:

```julia
# Register with problem
function register_operator!(problem, name, func)
    problem.operators[name] = func
end

# Example: Stokes operator
stokes(u, nu) = -nu*Δ(u) + ∇(p)

register_operator!(problem, "stokes", stokes)

# Use in equations
add_equation!(problem, "∂ₜ(u) = stokes(u, nu)")
```

---

## Equation Parsing

### Symbolic Syntax

Tarang.jl parses equation strings into operator applications:

```julia
# String equation
add_equation!(problem, "∂ₜ(u) + u*dx(u) = nu*Δ(u) - dx(p)")

# Parsed as:
# LHS: dt(u) + u*dx(u)
# RHS: nu*lap(u) - dx(p)
```

**Supported operations**:
- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Parentheses: `(`, `)`
- Functions: `sin`, `cos`, `exp`, etc.

### Operator Evaluation

Operators are evaluated in spectral space when possible:

```julia
# dx(u): Multiply by ik in Fourier space
# Δ(u): Multiply by -k² in Fourier space
# Nonlinear terms: Transform to grid space, evaluate, transform back
```

---

## Performance Tips

### Minimize Grid-Spectral Transforms

```julia
# Bad: Multiple transforms
add_equation!(problem, "∂ₜ(T) = -u*dx(T)")  # Transforms for each term

# Better: Group operations
# Tarang automatically optimizes transform grouping
```

### Precompute Common Terms

```julia
# If using same derivative multiple times
dudx = dx(u)

add_equation!(problem, "term1 = dudx")
add_equation!(problem, "term2 = w*dudx")
```

### Use Sparse Differentiation

Chebyshev/Legendre derivatives are sparse matrix operations - very efficient.

---

## See Also

- [Fields](fields.md): Field types that operators act on
- [Problems](problems.md): Using operators in equations
- [Domains](domains.md): Spatial discretization for operators
- [Bases](bases.md): Spectral bases for differentiation
