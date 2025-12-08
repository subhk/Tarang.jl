# Operators

Operators perform mathematical operations on fields, including differentiation and vector calculus.

## Differential Operators

### First Derivatives

```julia
# Syntax in equations
∂x(field)   # ∂/∂x
∂y(field)   # ∂/∂y
∂z(field)   # ∂/∂z

# Example
add_equation!(problem, "∂t(T) = -u*∂x(T) - w*∂z(T)")
```

### Second Derivatives

```julia
# Composed derivatives
∂x(∂x(field))   # ∂²/∂x²
∂z(∂z(field))   # ∂²/∂z²
∂x(∂z(field))   # ∂²/∂x∂z
```

### Laplacian

```julia
# Δ(f) = ∇²f
Δ(field)

# Example: Diffusion equation
add_equation!(problem, "∂t(T) = kappa*Δ(T)")
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
add_equation!(problem, "∂t(θ) = -u⋅∇(θ) + κ*fraclap(θ, 0.5)")

# Can also be used on LHS (implicit treatment)
add_equation!(problem, "∂t(θ) + κ*fraclap(θ, 0.5) = -u⋅∇(θ)")
```

### Hyperviscosity (Higher-Order Laplacian)

For turbulence simulations, hyperviscosity provides selective dissipation at small scales while preserving large-scale dynamics:

```julia
# General form: hyperlap(f, n) = (-Δ)^n = |k|^(2n) in Fourier space
hyperlap(field, 2)   # Biharmonic: (-Δ)² = |k|⁴
hyperlap(field, 4)   # 8th-order: (-Δ)⁴ = |k|⁸
hyperlap(field, 8)   # 16th-order: (-Δ)⁸ = |k|¹⁶

# Unicode shortcuts (preferred)
Δ²(field)   # Biharmonic (4th-order derivative)
Δ⁴(field)   # 8th-order derivative
Δ⁶(field)   # 12th-order derivative
Δ⁸(field)   # 16th-order derivative
```

**In spectral space**: Multiplication by `|k|^(2n)` - very efficient for Fourier bases.

**Usage in equations**:

```julia
# 2D turbulence with biharmonic hyperviscosity
add_equation!(problem, "∂t(ω) = -u⋅∇(ω) - ν₄*Δ²(ω)")

# 3D turbulence with 8th-order hyperviscosity
add_equation!(problem, "∂t(u) = -u⋅∇(u) - ∇(p) - ν₈*Δ⁴(u)")

# General n-th order using hyperlap
add_equation!(problem, "∂t(u) = -u⋅∇(u) - ν*hyperlap(u, 4)")
```

**Why use hyperviscosity?**

| Order | Operator | Spectral | Use Case |
|-------|----------|----------|----------|
| 2 | `Δ` | `-k²` | Standard viscosity |
| 4 | `Δ²` | `k⁴` | Mild scale separation |
| 8 | `Δ⁴` | `k⁸` | Strong scale separation |
| 16 | `Δ⁸` | `k¹⁶` | Extreme Reynolds numbers |

Higher orders concentrate dissipation at the smallest resolved scales, extending the inertial range.

**Typing Unicode**:

| Symbol | Type |
|--------|------|
| `Δ²` | `\Delta` + Tab, `\^2` + Tab |
| `Δ⁴` | `\Delta` + Tab, `\^4` + Tab |
| `Δ⁶` | `\Delta` + Tab, `\^6` + Tab |
| `Δ⁸` | `\Delta` + Tab, `\^8` + Tab |

## Vector Calculus

### Gradient

Converts scalar to vector field.

```julia
# grad(p) = ∇p
add_equation!(problem, "∂t(u) = -grad(p)")

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
add_equation!(problem, "omega = ∂x(v) - ∂y(u)")
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
# ∂t(field) for IVP equations
add_equation!(problem, "∂t(u) = rhs")

# Only valid in Initial Value Problems
```

## Unicode Operators

Tarang supports Unicode mathematical notation for cleaner, more readable code:

| Unicode | ASCII Equivalent | Description |
|---------|------------------|-------------|
| `∇` | `grad` | Gradient |
| `Δ` | `lap` | Laplacian |
| `∇²` | `lap` | Laplacian (alternative) |
| `∂t` | `dt` | Time derivative |
| `⋅` | `dot` | Dot product |
| `×` | `cross` | Cross product |
| `∇⊥` | `perp_grad` | Perpendicular gradient |
| `Δᵅ` | `fraclap` | Fractional Laplacian |

### In Equations

```julia
# Unicode syntax
add_equation!(problem, "∂t(u) + u⋅∇(u) = nu*Δ(u) - ∇(p)")
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
add_equation!(problem, "∂t(T) + u*∂x(T) = kappa*Δ(T)")

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
"u*∂x(f) + w*∂z(f)"

# Diffusion
"nu*Δ(u)"

# Pressure gradient
"∂x(p)"  # or "∇(p)" for vector

# Navier-Stokes viscous term
"nu*(∂x(∂x(u)) + ∂z(∂z(u)))"
# or simply:
"nu*Δ(u)"
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
    result = u.components[1] * ∂x(field)
    for i in 2:length(u.components)
        result += u.components[i] * d[i](field)
    end
    return result
end
```

### Using Built-in Operators in Equations

The equation parser recognizes all built-in operators:

```julia
# Available operators in equations:
# grad, div, curl, lap (or Δ), dt (or ∂t), d
# integrate, average, interpolate, convert, lift
# sin, cos, tan, exp, log, sqrt, abs, tanh

# Use operators directly
add_equation!(problem, "∂t(T) = -ux*∂x(T) - uz*∂z(T) + kappa*Δ(T)")
```

## Performance Tips

1. **Minimize transforms**: Group operations in same space
2. **Use Laplacian**: `Δ(f)` is optimized vs `∂x(∂x(f)) + ∂z(∂z(f))`
3. **Spectral derivatives**: Free in Fourier, cheap in Chebyshev
4. **Nonlinear terms**: Require transform to grid space

## See Also

- [Fields](fields.md): What operators act on
- [Problems](problems.md): Using operators in PDEs
- [Surface Dynamics](../tutorials/surface_dynamics.md): SQG, QG, and boundary advection-diffusion
- [API: Operators](../api/operators.md): Complete reference
