# Operators API

Operators compute derivatives and other mathematical operations on fields. Tarang.jl provides a symbolic syntax for natural mathematical notation in equations.

## Overview

Tarang.jl supports:
- **Differential operators**: grad (∇), div, curl, lap (Δ, ∇²)
- **Coordinate derivatives**: ∂x, ∂y, ∂z, ∂r, etc.
- **Time derivatives**: ∂t
- **Field operations**: dot (⋅), cross (×)
- **Custom operators**: User-defined operations

## Unicode Operators

Tarang.jl uses Unicode mathematical symbols for readable, publication-quality code:

| Syntax | Description | Typing |
|--------|-------------|--------|
| `∇(f)` | Gradient | `\nabla` Tab |
| `Δ(f)` or `∇²(f)` | Laplacian | `\Delta` Tab |
| `∂t(f)` | Time derivative | `\partial` Tab `t` |
| `∂x(f)` | x-derivative | `\partial` Tab `x` |
| `∂y(f)` | y-derivative | `\partial` Tab `y` |
| `∂z(f)` | z-derivative | `\partial` Tab `z` |
| `∂r(f)` | r-derivative | `\partial` Tab `r` |
| `u ⋅ v` | Dot product | `\cdot` Tab |
| `u × v` | Cross product | `\times` Tab |

**Example** - Navier-Stokes equation:
```julia
add_equation!(problem, "∂t(u) + u⋅∇(u) = -∇(p) + nu*Δ(u)")
```

**Typing Unicode in Julia**:
- `∇` : Type `\nabla` then press Tab
- `Δ` : Type `\Delta` then press Tab
- `∂t` : Type `\partial` Tab `\_t` Tab
- `∂x` : Type `\partial` Tab `x`
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
add_equation!(problem, "∂t(u) = -∇(p)")

# Expands to:
# ∂t(u_x) = -∂x(p)
# ∂t(u_z) = -∂z(p)
```

```julia
# 3D gradient with custom usage
∇T = ∇(T)  # Returns VectorField
# Components: ∇T.components[1] = ∂x(T), etc.
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
add_equation!(problem, "∂t(omega) = curl(u × omega)")
```

```julia
# 2D vorticity
coords = CartesianCoordinates("x", "y")
problem = IVP([u, v, omega])
add_equation!(problem, "omega = ∂x(v) - ∂y(u)")
```

**Return type**: VectorField (3D) or ScalarField (2D)

---

### Laplacian (lap)

Computes the Laplacian (second derivative) of a field.

**Syntax**:
```julia
# In equations
add_equation!(problem, "∂t(T) = kappa*Δ(T)")

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
add_equation!(problem, "∂t(T) = kappa*Δ(T)")
```

```julia
# Viscous term in Navier-Stokes
add_equation!(problem, "∂t(u) = nu*Δ(u) - ∇(p)")
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
∂x(field)   # ∂/∂x
∂y(field)   # ∂/∂y
∂z(field)   # ∂/∂z
∂r(field)   # ∂/∂r (spherical/polar)
∂θ(field)   # ∂/∂θ
∂φ(field)   # ∂/∂φ
```

**Examples**:

```julia
# Advection term
add_equation!(problem, "∂t(T) = -u*∂x(T) - w*∂z(T)")
```

```julia
# Shear
add_equation!(problem, "S = ∂x(u) + ∂z(w)")
```

```julia
# Custom derivative
dudz = ∂z(u)  # Returns field with ∂u/∂z
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
∂x(∂x(T))   # ∂²T/∂x²
∂z(∂z(T))   # ∂²T/∂z²

# Mixed derivatives
∂x(∂z(T))   # ∂²T/∂x∂z

# Higher order
∂x(∂x(∂x(T)))  # ∂³T/∂x³
```

**Examples**:

```julia
# Biharmonic operator
add_equation!(problem, "Δ(Δ(psi)) = omega")

# Equivalent to:
add_equation!(problem, "∂x(∂x(∂x(∂x(psi)))) + 2*∂x(∂x(∂z(∂z(psi)))) + ∂z(∂z(∂z(∂z(psi)))) = omega")
```

```julia
# Hyperdiffusion (for numerical stability)
add_equation!(problem, "∂t(T) = -nu4*Δ(Δ(T))")
```

---

## Time Derivatives

### dt / ∂t Operator

Time derivative for initial value problems.

**Syntax**:
```julia
dt(field)   # ASCII
∂t(field)   # Unicode (type \partial Tab \_t Tab)
```

**Examples**:

```julia
# Evolution equations
add_equation!(problem, "∂t(u) = -u*∂x(u) + nu*Δ(u)")
add_equation!(problem, "∂t(T) = -u*∂x(T) + kappa*Δ(T)")
```

**Note**: Only use in IVP (Initial Value Problems). Not valid for BVP or EVP.

---

## Vector Operations

### Dot Product / Advection

**Syntax**:
```julia
# In equations - use vector notation directly
add_equation!(problem, "∂t(T) = -u⋅∇(T)")

# For vector advection (nonlinear term)
add_equation!(problem, "∂t(u) = -u⋅∇(u)")
```

**Example**:

```julia
# Scalar advection: -u·∇T
add_equation!(problem, "∂t(T) = -u⋅∇(T)")

# Vector advection (Navier-Stokes nonlinear term)
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) = -u⋅∇(u)")
```

---

### Cross Product

**Syntax**:
```julia
# Use cross function or × operator
u_cross_omega = cross(u, omega)
u_cross_omega = u × omega  # Unicode: \times Tab
```

**Example**:

```julia
# Vorticity equation: ∂ω/∂t = ∇×(u×ω)
problem = IVP([ux, uy, uz, omega_x, omega_y, omega_z])

# Compute u × ω
u_cross_omega = cross(u, omega)
# Or equivalently:
u_cross_omega = u × omega

# Then: ∇×(u×ω)
add_equation!(problem, "∂t(omega) = curl(u_cross_omega)")
```

---

## Composite Operators

Combine operators for complex expressions.

### Vector Laplacian

```julia
# For vector field u
# ∇²u = (∇²u_x, ∇²u_y, ∇²u_z)

add_equation!(problem, "∂t(u) = nu*Δ(u)")
# Automatically applies componentwise
```

### Advection Operator

```julia
# u·∇u (nonlinear advection) - use vector notation directly

# Navier-Stokes momentum equation:
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) = -u⋅∇(u)")

# Scalar advection:
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")

# With buoyancy (Boussinesq):
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) = -u⋅∇(u) + Ra*T*ez")
```

### Strain Rate Tensor

```julia
# S_ij = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)

function strain_rate_tensor(u)
    # Returns TensorField
    S = TensorField(u.distributor, u.coords, "S", u.bases, symmetric=true)

    S[1,1] = ∂x(u.components[1])
    S[1,2] = 0.5 * (∂x(u.components[2]) + ∂y(u.components[1]))
    S[2,2] = ∂y(u.components[2])
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
∂x(∂z(f)) == ∂z(∂x(f))
```

### Product Rule

```julia
# ∇(fg) = f∇g + g∇f
∇(f*g) == f*∇(g) + g*∇(f)
```

---

## Custom Operators

### Defining Helper Functions

You can define helper functions that compose built-in operators:

```julia
# Define helper function
function my_diffusion(field, kappa)
    # Custom operation combining derivatives
    return kappa * Δ(field)
end

# Use programmatically
diffusion_term = my_diffusion(T, kappa)
```

### Using Built-in Operators in Equations

The equation parser recognizes all registered operators. Use them directly in equation strings:

```julia
# Available operators in equations:
# grad, div, curl, lap (or Δ), dt (or ∂t), d
# integrate, average, interpolate, convert, lift
# sin, cos, tan, exp, log, sqrt, abs, tanh

# Example: diffusion equation
add_equation!(problem, "∂t(T) = kappa*Δ(T)")

# Example: advection-diffusion
add_equation!(problem, "∂t(T) + u⋅∇(T) = kappa*Δ(T)")
```

For complex expressions, compute terms programmatically and use the result:

```julia
# Compute complex term
rhs_term = kappa * Δ(T) - u * ∂x(T)

# Add to field's RHS
T.rhs .+= rhs_term.data
```

---

## Equation Parsing

### Symbolic Syntax

Tarang.jl parses equation strings into operator applications:

```julia
# String equation
add_equation!(problem, "∂t(u) + u*∂x(u) = nu*Δ(u) - ∂x(p)")

# Parsed as:
# LHS: ∂t(u) + u*∂x(u)
# RHS: nu*Δ(u) - ∂x(p)
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
# ∂x(u): Multiply by ik in Fourier space
# Δ(u): Multiply by -k² in Fourier space
# Nonlinear terms: Transform to grid space, evaluate, transform back
```

---

## Performance Tips

### Minimize Grid-Spectral Transforms

```julia
# Bad: Multiple transforms
add_equation!(problem, "∂t(T) = -u*∂x(T)")  # Transforms for each term

# Better: Group operations
# Tarang automatically optimizes transform grouping
```

### Precompute Common Terms

```julia
# If using same derivative multiple times
dudx = ∂x(u)

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
