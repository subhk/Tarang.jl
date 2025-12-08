# Tutorial: Boundary Conditions

This tutorial covers the different types of boundary conditions available in Tarang.jl and how to apply them effectively.

## Overview

Boundary conditions (BCs) are essential for well-posed PDE problems. Tarang.jl supports:

- **Dirichlet**: Specify field value at boundary
- **Neumann**: Specify derivative at boundary
- **Robin**: Linear combination of value and derivative
- **Periodic**: Automatic for Fourier bases
- **Custom**: User-defined conditions

## Dirichlet Boundary Conditions

Fix the value of a field at the boundary.

### Basic Usage

```julia
# Syntax: add_dirichlet_bc!(problem, "field(coord=location) = value")

# Temperature fixed at bottom wall
add_dirichlet_bc!(problem, "T(z=0) = 1")  # T = 1 at z = 0

# Temperature fixed at top wall
add_dirichlet_bc!(problem, "T(z=1) = 0")  # T = 0 at z = 1
```

### No-Slip Velocity

For viscous flows at solid walls:

```julia
# Velocity = 0 at both walls
add_dirichlet_bc!(problem, "ux(z=0) = 0")  # u = 0 at z = 0
add_dirichlet_bc!(problem, "ux(z=1) = 0")  # u = 0 at z = 1
add_dirichlet_bc!(problem, "uz(z=0) = 0")  # w = 0 at z = 0
add_dirichlet_bc!(problem, "uz(z=1) = 0")  # w = 0 at z = 1
```

### Time-Dependent Boundary Conditions

```julia
# Moving wall (Couette flow)
U_wall = 1.0
add_dirichlet_bc!(problem, "ux(z=1) = $U_wall")

# Oscillating boundary
function oscillating_bc(t)
    return sin(2π * t)
end

# Use with solver callback to update BC value
```

## Neumann Boundary Conditions

Specify the derivative (flux) at the boundary.

### Basic Usage

```julia
# Syntax: add_neumann_bc!(problem, "d<coord>(field)(coord=location) = value")

# Zero heat flux (insulating wall)
add_neumann_bc!(problem, "dz(T)(z=1) = 0")  # dT/dz = 0 at z = 1

# Specified heat flux
q = 1.0  # Heat flux value
add_neumann_bc!(problem, "dz(T)(z=0) = $q")  # dT/dz = q at z = 0
```

### Stress-Free Conditions

For free surfaces or slip boundaries:

```julia
# Stress-free surface (du/dz = 0)
add_neumann_bc!(problem, "dz(ux)(z=1) = 0")

# Using the convenience function
add_stress_free_bc!(problem, "ux(z=1)")
```

## Robin Boundary Conditions

Linear combination of Dirichlet and Neumann conditions:

$\alpha u + \beta \frac{\partial u}{\partial n} = \gamma$

### Basic Usage

```julia
# Syntax: add_robin_bc!(problem, "alpha*field(coord=loc) + beta*d<coord>(field)(coord=loc) = value")

# Convective heat transfer: h*T + k*dT/dn = h*T_ambient
h = 10.0   # Heat transfer coefficient
k = 1.0    # Thermal conductivity
T_amb = 0.0
add_robin_bc!(problem, "$(h)*T(z=1) + $(k)*dz(T)(z=1) = $(h*T_amb)")
```

### Impedance Boundary Conditions

For wave problems:

```julia
# ∂u/∂n + Z*u = 0 (absorbing boundary)
Z = 1.0  # Impedance
add_robin_bc!(problem, "$(Z)*u(z=1) + 1.0*dz(u)(z=1) = 0")
```

## Periodic Boundary Conditions

Periodic boundaries are automatically handled by Fourier bases.

```julia
# Fourier basis implies periodicity
x_basis = RealFourier(coords["x"], size=128, bounds=(0.0, 2π))

# No boundary conditions needed for x-direction!
# The Fourier representation enforces u(x=0) = u(x=2π)
```

!!! warning "Mixing Periodic and Non-Periodic"
    When using Fourier (periodic) and Chebyshev (non-periodic) bases together, only apply boundary conditions to the non-periodic directions.

## Tau Method Implementation

Tarang.jl uses the tau method to enforce boundary conditions spectrally. This replaces the highest-order spectral coefficients with constraint equations.

### How It Works

For a Chebyshev expansion with N modes:
1. The last few equations become BC constraints
2. "Tau" correction terms absorb the mismatch

```julia
# The tau method is handled automatically
# When you add BCs, Tarang modifies the operator matrices

add_dirichlet_bc!(problem, "T(z=0) = 1")
# Internally: replaces one spectral equation with T(z=0) = 1
```

### Number of Boundary Conditions

For well-posed problems:
- 1st-order operator (∂/∂z): 1 BC needed
- 2nd-order operator (∂²/∂z²): 2 BCs needed
- 4th-order operator (biharmonic): 4 BCs needed

```julia
# Diffusion equation: lap(T) - 2nd order in z
# Needs 2 BCs in z direction
add_dirichlet_bc!(problem, "T(z=0) = $T_bottom")
add_dirichlet_bc!(problem, "T(z=1) = $T_top")
```

## Common Patterns

### Channel Flow

```julia
# Pressure-driven channel flow
coords = CartesianCoordinates("x", "z")

# No-slip at walls
add_dirichlet_bc!(problem, "ux(z=0) = 0")
add_dirichlet_bc!(problem, "ux(z=1) = 0")
add_dirichlet_bc!(problem, "uz(z=0) = 0")
add_dirichlet_bc!(problem, "uz(z=1) = 0")

# Pressure gradient as forcing (not a BC)
problem.parameters["dpdx"] = -1.0
```

### Thermal Convection

```julia
# Rayleigh-Bénard setup
# Fixed temperatures at top and bottom
add_dirichlet_bc!(problem, "T(z=0) = 1")  # Hot bottom
add_dirichlet_bc!(problem, "T(z=1) = 0")  # Cold top

# No-slip velocity
add_dirichlet_bc!(problem, "ux(z=0) = 0")
add_dirichlet_bc!(problem, "ux(z=1) = 0")
add_dirichlet_bc!(problem, "uz(z=0) = 0")
add_dirichlet_bc!(problem, "uz(z=1) = 0")
```

### Free-Slip Boundaries

```julia
# Free-slip (stress-free) at top, no-slip at bottom
add_dirichlet_bc!(problem, "ux(z=0) = 0")              # No-slip
add_stress_free_bc!(problem, "ux(z=1)")   # Stress-free
add_dirichlet_bc!(problem, "uz(z=0) = 0")              # No penetration
add_dirichlet_bc!(problem, "uz(z=1) = 0")              # No penetration
```

## Validation

### Checking BC Satisfaction

```julia
# After solving, verify BCs are satisfied
function check_dirichlet_bc(field, coord, location, expected_value; tol=1e-10)
    to_grid!(field)
    data = get_grid_data(field)

    # Get boundary values
    if coord == "z" && location == 0.0
        bc_values = data[:, 1]
    elseif coord == "z" && location == 1.0
        bc_values = data[:, end]
    end

    error = maximum(abs.(bc_values .- expected_value))
    @assert error < tol "BC error: $error"

    return error
end

# Usage
check_dirichlet_bc(T, "z", 0.0, 1.0)
```

## Troubleshooting

### Over-Specified System

**Problem**: Too many boundary conditions cause singular matrices.

**Solution**: Match the number of BCs to the operator order in each direction.

### Under-Specified System

**Problem**: Not enough BCs lead to non-unique solutions.

**Solution**: Add appropriate boundary conditions for the problem physics.

### BC Order Matters

**Problem**: BCs must be consistent with problem physics.

**Example**: For Navier-Stokes, you cannot specify both velocity and stress at the same boundary in the same direction.

## See Also

- [Problems API](../api/problems.md): Problem definition
- [Bases API](../api/bases.md): Spectral basis selection
- [2D Rayleigh-Bénard Tutorial](ivp_2d_rbc.md): Complete example with BCs
