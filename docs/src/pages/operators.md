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

## Time Derivatives

```julia
# dt(field) for IVP equations
add_equation!(problem, "dt(u) = rhs")

# Only valid in Initial Value Problems
```

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
- [API: Operators](../api/operators.md): Complete reference
