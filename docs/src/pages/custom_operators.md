# Custom Operators

Creating custom differential and integral operators.

## Overview

Beyond built-in operators (grad, div, curl, lap), you can define custom operators for specialized physics.

## Helper Functions

### Simple Custom Operators

```julia
# Advection operator: u·∇f
function advection(u, f)
    result = u.components[1].data .* ∂x(f).data
    for i in 2:length(u.components)
        result .+= u.components[i].data .* d_operators[i](f).data
    end
    return result
end
```

### Using in Equations

```julia
# Option 1: Expand manually
Tarang.add_equation!(problem, "∂ₜ(T) = -ux*∂x(T) - uz*∂z(T)")

# Option 2: Use helper in equation string
# (requires registration)
```

## Registering Operators

```julia
# Register with problem
function register_operator!(problem, name, func)
    problem.operators[name] = func
end

# Define operator
my_advect(u, f) = "ux*∂x($f) + uz*∂z($f)"

register_operator!(problem, "advect", my_advect)

# Use in equation
Tarang.add_equation!(problem, "∂ₜ(T) = advect(u, T)")
```

## Spectral Differentiation

### Fourier Derivative

```julia
function fourier_derivative(field, basis, order=1)
    k = get_wavenumbers(basis)
    Tarang.ensure_layout!(field, :c)

    # Multiply by (ik)^order
    field.data_c .*= (1im .* k) .^ order

    return field
end
```

### Chebyshev Derivative

```julia
function chebyshev_derivative(field, basis)
    Tarang.ensure_layout!(field, :c)

    # Use differentiation matrix
    D = chebyshev_diff_matrix(basis.size)
    field.data_c = D * field.data_c

    return field
end
```

## Integral Operators

### Spatial Integration

```julia
function integrate(field, dim)
    Tarang.ensure_layout!(field, :g)

    if dim == 1  # x-direction
        dx = field.bases[1].length / field.bases[1].size
        return sum(field.data_g, dims=1) * dx
    elseif dim == 2  # z-direction
        # Chebyshev: use quadrature weights
        weights = chebyshev_weights(field.bases[2])
        return sum(field.data_g .* weights', dims=2)
    end
end
```

### Running Average

```julia
function running_average(field, window)
    Tarang.ensure_layout!(field, :g)

    # Convolution with box filter
    # ...
end
```

## Vector Calculus

### Strain Rate Tensor

```julia
function strain_rate(u)
    # S_ij = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i)
    S = TensorField(u.dist, u.coords, "S", u.bases; symmetric=true)

    ux, uz = u.components[1], u.components[2]

    S[1,1] = ∂x(ux)
    S[1,2] = 0.5 * (∂x(uz) + ∂z(ux))
    S[2,2] = ∂z(uz)

    return S
end
```

### Vorticity (2D)

```julia
function vorticity_2d(u)
    ux, uz = u.components[1], u.components[2]
    return ∂x(uz) - ∂z(ux)
end
```

### Helicity (3D)

```julia
function helicity(u)
    omega = ∇×(u)

    # H = u · ω
    H = u.components[1].data .* omega.components[1].data
    for i in 2:3
        H .+= u.components[i].data .* omega.components[i].data
    end

    return H
end
```

## Nonlinear Operators

### Convective Derivative

```julia
function convective_derivative(u, f)
    # (u·∇)f
    result = zeros(size(f.data_g))

    for (i, comp) in enumerate(u.components)
        Tarang.ensure_layout!(comp, :g)
        df = d_operators[i](f)
        Tarang.ensure_layout!(df, :g)
        result .+= comp.data_g .* df.data_g
    end

    return result
end
```

### Nonlinear Term (u·∇u)

```julia
function nonlinear_advection(u)
    # Returns vector field
    result = similar(u)

    for (j, uj) in enumerate(u.components)
        result.components[j].data_g .= convective_derivative(u, uj)
    end

    return result
end
```

## Physics-Specific Operators

### Coriolis Force

```julia
function coriolis(u, Omega)
    # 2Ω × u
    # For rotation about z-axis:
    fx = -2 * Omega * u.components[2]  # -2Ω*v
    fy =  2 * Omega * u.components[1]  # +2Ω*u
    fz = 0

    return (fx, fy, fz)
end
```

### Lorentz Force (MHD)

```julia
function lorentz_force(J, B)
    # J × B
    return cross_product(J, B)
end
```

## Tips

### Performance

- Keep operations in same space (grid or spectral)
- Minimize transforms
- Pre-compute constant factors

### Validation

- Test against analytical solutions
- Check symmetries
- Verify conservation properties

## See Also

- [Operators](operators.md): Built-in operators
- [Fields](fields.md): Field types
- [API: Operators](../api/operators.md): Reference
