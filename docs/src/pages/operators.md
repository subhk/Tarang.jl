# Operators

Operators perform mathematical operations on fields, including differentiation and vector calculus.

## Differential Operators

### First Derivatives

```julia
# Syntax in equations
dx(field)   # ∂/∂x
dy(field)   # ∂/∂y
dz(field)   # ∂/∂z

# Example
add_equation!(problem, "dt(T) = -u*dx(T) - w*dz(T)")
```

### Second Derivatives

```julia
# Composed derivatives
dx(dx(field))   # ∂²/∂x²
dz(dz(field))   # ∂²/∂z²
dx(dz(field))   # ∂²/∂x∂z
```

### Laplacian

```julia
# lap(f) = ∇²f
lap(field)

# Example: Diffusion equation
add_equation!(problem, "dt(T) = kappa*lap(T)")
```

### Fractional Laplacian

The fractional Laplacian `(-Δ)^α` generalizes the Laplacian to non-integer powers:

```julia
# fraclap(f, α) = (-Δ)^α f
fraclap(field, 0.5)    # Square root: (-Δ)^(1/2)
fraclap(field, -0.5)   # Inverse square root: (-Δ)^(-1/2)
fraclap(field, 1.0)    # Standard Laplacian: (-Δ)^1 = -Δ
fraclap(field, 2.0)    # Biharmonic: (-Δ)^2 = Δ²

# Convenience functions
sqrtlap(field)         # Same as fraclap(f, 0.5)
invsqrtlap(field)      # Same as fraclap(f, -0.5)
```

**In spectral space**: Multiplication by `|k|^(2α)` where `k` is the wavenumber.

**Applications**:
- **SQG dynamics**: `ψ = (-Δ)^(-1/2) θ` for streamfunction from buoyancy
- **Fractional diffusion**: `∂θ/∂t = -κ(-Δ)^α θ` for anomalous diffusion
- **Hyperviscosity**: `(-Δ)^n` for numerical dissipation at small scales

```julia
# SQG buoyancy equation with fractional dissipation
add_equation!(problem, "∂ₜ(θ) = -u⋅∇(θ) + κ*fraclap(θ, 0.5)")

# Can also be used on LHS (implicit treatment)
add_equation!(problem, "∂ₜ(θ) + κ*fraclap(θ, 0.5) = -u⋅∇(θ)")
```

## Vector Calculus

### Gradient

Converts scalar to vector field.

```julia
# grad(p) = ∇p
add_equation!(problem, "dt(u) = -grad(p)")

# Components:
# ∂p/∂x, ∂p/∂y, ∂p/∂z
```

### Divergence

Converts vector to scalar field.

```julia
# div(u) = ∇·u
add_equation!(problem, "div(u) = 0")

# Expands to:
# ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z = 0
```

### Curl

For 3D vector fields (returns vector):

```julia
# curl(u) = ∇×u
omega = curl(u)
```

For 2D (returns scalar vorticity):

```julia
# ω = ∂v/∂x - ∂u/∂y
add_equation!(problem, "omega = dx(v) - dy(u)")
```

### Perpendicular Gradient

For 2D flows, the perpendicular gradient creates a divergence-free velocity from a streamfunction:

```julia
# perp_grad(ψ) = ∇⊥ψ = (-∂ψ/∂y, ∂ψ/∂x)
u = perp_grad(psi)

# This gives: u_x = -∂ψ/∂y, u_y = ∂ψ/∂x
# Automatically satisfies: ∇·u = 0
```

**Applications**:
- **2D incompressible flow**: `u = ∇⊥ψ` from streamfunction
- **SQG velocity**: `u = ∇⊥((-Δ)^(-1/2) θ)`
- **QG geostrophic flow**: `u = ∇⊥ψ`

## Time Derivatives

```julia
# dt(field) for IVP equations
add_equation!(problem, "dt(u) = rhs")

# Only valid in Initial Value Problems
```

## Unicode Operators

Tarang supports Unicode mathematical notation for cleaner, more readable code:

| Unicode | ASCII Equivalent | Description |
|---------|------------------|-------------|
| `∇` | `grad` | Gradient |
| `Δ` | `lap` | Laplacian |
| `∇²` | `lap` | Laplacian (alternative) |
| `∂ₜ` | `dt` | Time derivative |
| `⋅` | `dot` | Dot product |
| `×` | `cross` | Cross product |
| `∇⊥` | `perp_grad` | Perpendicular gradient |
| `Δᵅ` | `fraclap` | Fractional Laplacian |

### In Equations

```julia
# Traditional ASCII syntax
add_equation!(problem, "dt(u) + u*dx(u) = nu*lap(u) - grad(p)")

# Unicode syntax (equivalent)
add_equation!(problem, "∂ₜ(u) + u⋅∇(u) = nu*Δ(u) - ∇(p)")
```

### In Code

```julia
# ASCII
velocity = grad(pressure)
vorticity = curl(velocity)

# Unicode (equivalent)
velocity = ∇(pressure)
vorticity = curl(velocity)

# Mixed - use what's clearest
u = ∇⊥(ψ)        # Perpendicular gradient
dissipation = Δ(T)  # Laplacian
```

### Typing Unicode in Julia

In Julia REPL or editors with Julia support:

| Symbol | Type |
|--------|------|
| `∇` | `\nabla` + Tab |
| `Δ` | `\Delta` + Tab |
| `∂` | `\partial` + Tab |
| `⋅` | `\cdot` + Tab |
| `×` | `\times` + Tab |
| `⊥` | `\perp` + Tab |
| `α` | `\alpha` + Tab |
| `½` | `\^1` + Tab, then type `/2` |

## Using Operators in Equations

### Symbolic Syntax

```julia
# Equations are strings parsed symbolically
add_equation!(problem, "dt(T) + u*dx(T) = kappa*lap(T)")

# Supports:
# - Addition/subtraction: +, -
# - Multiplication: *
# - Division: /
# - Parentheses: (, )
# - Function calls: sin, cos, exp, etc.
```

### Common Patterns

```julia
# Advection
"u*dx(f) + w*dz(f)"

# Diffusion
"nu*lap(u)"

# Pressure gradient
"dx(p)"  # or "grad(p)" for vector

# Navier-Stokes viscous term
"nu*(dx(dx(u)) + dz(dz(u)))"
# or simply:
"nu*lap(u)"
```

## Implementation Details

### Fourier Differentiation

For Fourier bases, differentiation is multiplication by wavenumber:

```math
\frac{\partial}{\partial x} \sum_k \hat{f}_k e^{ikx} = \sum_k (ik) \hat{f}_k e^{ikx}
```

### Chebyshev Differentiation

Uses recurrence relations:

```math
\frac{dT_n}{dx} = n U_{n-1}(x)
```

Implemented as sparse matrix operations.

### Computational Cost

| Operation | Cost |
|-----------|------|
| Fourier derivative | O(N) |
| Chebyshev derivative | O(N²) sparse |
| Laplacian | Same as 2× first derivative |

## Custom Operators

### Helper Functions

```julia
function advection(u, field)
    # u·∇f
    result = u.components[1] * dx(field)
    for i in 2:length(u.components)
        result += u.components[i] * d[i](field)
    end
    return result
end
```

### Registered Operators

```julia
# Register custom operator with problem
problem.operators["advect"] = advection

# Use in equations
add_equation!(problem, "dt(T) = -advect(u, T)")
```

## Performance Tips

1. **Minimize transforms**: Group operations in same space
2. **Use Laplacian**: `lap(f)` is optimized vs `dx(dx(f)) + dz(dz(f))`
3. **Spectral derivatives**: Free in Fourier, cheap in Chebyshev
4. **Nonlinear terms**: Require transform to grid space

## See Also

- [Fields](fields.md): What operators act on
- [Problems](problems.md): Using operators in PDEs
- [Surface Dynamics](../tutorials/surface_dynamics.md): SQG, QG, and boundary advection-diffusion
- [API: Operators](../api/operators.md): Complete reference
