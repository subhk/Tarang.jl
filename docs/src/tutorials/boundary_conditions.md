# Tutorial: Boundary Conditions

This tutorial covers the different types of boundary conditions available in Tarang.jl and how to apply them effectively using the tau method with explicit tau fields.

## Overview

Boundary conditions (BCs) are essential for well-posed PDE problems. Tarang.jl supports:

- **Dirichlet**: Specify field value at boundary
- **Neumann**: Specify derivative at boundary
- **Robin**: Linear combination of value and derivative
- **Periodic**: Automatic for Fourier bases

## The Tau Method Approach

Tarang.jl follows the [Dedalus](https://dedalus-project.readthedocs.io/) approach for handling boundary conditions. Users must **explicitly create tau fields** and add them to equations using the `lift()` operator.

### Why Explicit Tau Fields?

1. **Clarity**: The mathematical structure is visible in your code
2. **Flexibility**: Full control over tau placement
3. **Debugging**: Easy to inspect tau field values
4. **Consistency**: Matches the mathematical formulation

### Required Steps

For any problem with non-periodic boundary conditions:

1. **Create tau fields** - One per boundary condition
2. **Add tau fields to the problem** - Include them in the field list
3. **Add lift() terms to equations** - Place tau contributions at specific modes
4. **Specify boundary conditions** - Link each BC to its tau field

## Complete Example: Poisson Equation

Let's solve the Poisson equation $\nabla^2 u = f$ with Dirichlet BCs:

```julia
using Tarang

# Create coordinates and bases
coords = CartesianCoordinates("x", "z")
x_basis = RealFourier(coords["x"], size=64, bounds=(0.0, 2π))
z_basis = Chebyshev(coords["z"], size=64, bounds=(0.0, 1.0))

# Create distributor and fields
dist = Distributor([x_basis, z_basis])
u = ScalarField(dist, "u")
f = ScalarField(dist, "f")  # Source term

# Step 1: Create tau fields (one per BC)
# These live on the x-basis only (boundary is a line in 2D)
tau_u1 = ScalarField(dist, "tau_u1", (x_basis,))  # For BC at z=0
tau_u2 = ScalarField(dist, "tau_u2", (x_basis,))  # For BC at z=1

# Step 2: Add ALL fields to problem (including tau fields)
problem = LBVP([u, tau_u1, tau_u2])

# Step 3: Add equation with lift() operators
# lift(tau, basis, mode) places tau contribution at specified spectral mode
add_equation!(problem,
    lap(u) + lift(tau_u1, z_basis, -1) + lift(tau_u2, z_basis, -2) - f
)

# Step 4: Add boundary conditions with explicit tau_field parameter
add_dirichlet_bc!(problem, u, "z", :left, 0.0; tau_field="tau_u1")   # u(z=0) = 0
add_dirichlet_bc!(problem, u, "z", :right, 0.0; tau_field="tau_u2")  # u(z=1) = 0

# Solve
solver = LinearBVPSolver(problem)
solve!(solver)
```

## Dirichlet Boundary Conditions

Fix the value of a field at the boundary.

### Basic Setup

```julia
# Create tau fields for each Dirichlet BC
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,))
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,))

# Add to problem
problem = LBVP([T, tau_T1, tau_T2])

# Add equation with lift terms
add_equation!(problem,
    lap(T) + lift(tau_T1, z_basis, -1) + lift(tau_T2, z_basis, -2) - source
)

# Boundary conditions
add_dirichlet_bc!(problem, T, "z", :left, 1.0; tau_field="tau_T1")   # T(z=0) = 1
add_dirichlet_bc!(problem, T, "z", :right, 0.0; tau_field="tau_T2")  # T(z=1) = 0
```

### No-Slip Velocity (Multiple Components)

For viscous flows at solid walls, each velocity component needs its own tau fields:

```julia
# Tau fields for ux
tau_ux1 = ScalarField(dist, "tau_ux1", (x_basis,))
tau_ux2 = ScalarField(dist, "tau_ux2", (x_basis,))

# Tau fields for uz
tau_uz1 = ScalarField(dist, "tau_uz1", (x_basis,))
tau_uz2 = ScalarField(dist, "tau_uz2", (x_basis,))

# Include all tau fields in problem
problem = IVP([ux, uz, p, tau_ux1, tau_ux2, tau_uz1, tau_uz2])

# Momentum equation for ux with lift terms
add_equation!(problem,
    dt(ux) - ν*lap(ux) + dx(p) + lift(tau_ux1, z_basis, -1) + lift(tau_ux2, z_basis, -2)
)

# Momentum equation for uz with lift terms
add_equation!(problem,
    dt(uz) - ν*lap(uz) + dz(p) + lift(tau_uz1, z_basis, -1) + lift(tau_uz2, z_basis, -2)
)

# No-slip boundary conditions
add_dirichlet_bc!(problem, ux, "z", :left, 0.0; tau_field="tau_ux1")
add_dirichlet_bc!(problem, ux, "z", :right, 0.0; tau_field="tau_ux2")
add_dirichlet_bc!(problem, uz, "z", :left, 0.0; tau_field="tau_uz1")
add_dirichlet_bc!(problem, uz, "z", :right, 0.0; tau_field="tau_uz2")
```

## Neumann Boundary Conditions

Specify the derivative (flux) at the boundary.

### Basic Setup

```julia
# Tau fields for Neumann BCs
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,))
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,))

problem = LBVP([T, tau_T1, tau_T2])

add_equation!(problem,
    lap(T) + lift(tau_T1, z_basis, -1) + lift(tau_T2, z_basis, -2) - source
)

# Neumann: specify derivative at boundary
add_neumann_bc!(problem, T, "z", :left, 1.0; tau_field="tau_T1")   # dT/dz(z=0) = 1
add_neumann_bc!(problem, T, "z", :right, 0.0; tau_field="tau_T2")  # dT/dz(z=1) = 0
```

### Stress-Free Conditions

For free surfaces or slip boundaries (∂u/∂z = 0):

```julia
# Mixed: no-slip at bottom, stress-free at top
add_dirichlet_bc!(problem, ux, "z", :left, 0.0; tau_field="tau_ux1")   # No-slip
add_neumann_bc!(problem, ux, "z", :right, 0.0; tau_field="tau_ux2")    # Stress-free
```

## Robin Boundary Conditions

Linear combination: $\alpha u + \beta \frac{\partial u}{\partial n} = \gamma$

```julia
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,))
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,))

problem = LBVP([T, tau_T1, tau_T2])

add_equation!(problem,
    lap(T) + lift(tau_T1, z_basis, -1) + lift(tau_T2, z_basis, -2) - source
)

# Convective heat transfer: h*T + k*dT/dn = h*T_ambient
h = 10.0   # Heat transfer coefficient
k = 1.0    # Thermal conductivity
T_amb = 25.0

# Robin BC at top: α=h, β=k, γ=h*T_amb
add_robin_bc!(problem, T, "z", :right, h, k, h*T_amb; tau_field="tau_T2")

# Dirichlet at bottom
add_dirichlet_bc!(problem, T, "z", :left, 100.0; tau_field="tau_T1")
```

## Periodic Boundary Conditions

Periodic boundaries are automatically handled by Fourier bases - no tau fields needed!

```julia
# Fourier basis implies periodicity
x_basis = RealFourier(coords["x"], size=128, bounds=(0.0, 2π))

# No boundary conditions or tau fields needed for x-direction
# The Fourier representation automatically enforces u(x=0) = u(x=2π)
```

!!! warning "Mixing Periodic and Non-Periodic"
    When using Fourier (periodic) and Chebyshev (non-periodic) bases together, only create tau fields and boundary conditions for the non-periodic directions.

## The lift() Operator

The `lift()` operator places tau corrections at specific spectral modes:

```julia
lift(tau_field, basis, mode_index)
```

- `tau_field`: The tau field to lift
- `basis`: The non-periodic basis (e.g., Chebyshev)
- `mode_index`: Which spectral mode (-1 = last, -2 = second-to-last)

### Mode Selection Guidelines

| Operator Order | Number of BCs | lift() modes |
|---------------|---------------|--------------|
| 1st (∂/∂z)    | 1             | -1           |
| 2nd (∂²/∂z²)  | 2             | -1, -2       |
| 4th (∇⁴)      | 4             | -1, -2, -3, -4 |

### Example: Fourth-Order Problem

For a biharmonic equation (∇⁴u = f) with 4 boundary conditions:

```julia
# Four tau fields needed
tau_u1 = ScalarField(dist, "tau_u1", (x_basis,))
tau_u2 = ScalarField(dist, "tau_u2", (x_basis,))
tau_u3 = ScalarField(dist, "tau_u3", (x_basis,))
tau_u4 = ScalarField(dist, "tau_u4", (x_basis,))

problem = LBVP([u, tau_u1, tau_u2, tau_u3, tau_u4])

add_equation!(problem,
    lap(lap(u)) + lift(tau_u1, z_basis, -1) + lift(tau_u2, z_basis, -2)
               + lift(tau_u3, z_basis, -3) + lift(tau_u4, z_basis, -4) - f
)

# Clamped beam: u = 0 and du/dz = 0 at both ends
add_dirichlet_bc!(problem, u, "z", :left, 0.0; tau_field="tau_u1")
add_neumann_bc!(problem, u, "z", :left, 0.0; tau_field="tau_u2")
add_dirichlet_bc!(problem, u, "z", :right, 0.0; tau_field="tau_u3")
add_neumann_bc!(problem, u, "z", :right, 0.0; tau_field="tau_u4")
```

## Complete Examples

### Channel Flow (Poiseuille Flow)

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
x_basis = RealFourier(coords["x"], size=64, bounds=(0.0, 2π))
z_basis = Chebyshev(coords["z"], size=64, bounds=(0.0, 1.0))
dist = Distributor([x_basis, z_basis])

# Fields
ux = ScalarField(dist, "ux")
uz = ScalarField(dist, "uz")
p = ScalarField(dist, "p")

# Tau fields for velocity BCs
tau_ux1 = ScalarField(dist, "tau_ux1", (x_basis,))
tau_ux2 = ScalarField(dist, "tau_ux2", (x_basis,))
tau_uz1 = ScalarField(dist, "tau_uz1", (x_basis,))
tau_uz2 = ScalarField(dist, "tau_uz2", (x_basis,))

# Tau for pressure gauge
tau_p = ScalarField(dist, "tau_p", (x_basis,))

# Parameters
ν = 0.01
dpdx = -1.0

problem = IVP([ux, uz, p, tau_ux1, tau_ux2, tau_uz1, tau_uz2, tau_p])

# Equations with lift terms
add_equation!(problem, dt(ux) - ν*lap(ux) + dx(p)
    + lift(tau_ux1, z_basis, -1) + lift(tau_ux2, z_basis, -2) + dpdx)
add_equation!(problem, dt(uz) - ν*lap(uz) + dz(p)
    + lift(tau_uz1, z_basis, -1) + lift(tau_uz2, z_basis, -2))
add_equation!(problem, dx(ux) + dz(uz) + lift(tau_p, z_basis, -1))

# No-slip at both walls
add_dirichlet_bc!(problem, ux, "z", :left, 0.0; tau_field="tau_ux1")
add_dirichlet_bc!(problem, ux, "z", :right, 0.0; tau_field="tau_ux2")
add_dirichlet_bc!(problem, uz, "z", :left, 0.0; tau_field="tau_uz1")
add_dirichlet_bc!(problem, uz, "z", :right, 0.0; tau_field="tau_uz2")

# Pressure gauge
add_dirichlet_bc!(problem, p, "z", :left, 0.0; tau_field="tau_p")
```

### Rayleigh-Bénard Convection

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
x_basis = RealFourier(coords["x"], size=128, bounds=(0.0, 4.0))
z_basis = Chebyshev(coords["z"], size=64, bounds=(0.0, 1.0))
dist = Distributor([x_basis, z_basis])

# Fields
u = ScalarField(dist, "u")    # Horizontal velocity
w = ScalarField(dist, "w")    # Vertical velocity
p = ScalarField(dist, "p")    # Pressure
T = ScalarField(dist, "T")    # Temperature

# Tau fields (2 per field with 2nd-order z-derivative)
tau_u1 = ScalarField(dist, "tau_u1", (x_basis,))
tau_u2 = ScalarField(dist, "tau_u2", (x_basis,))
tau_w1 = ScalarField(dist, "tau_w1", (x_basis,))
tau_w2 = ScalarField(dist, "tau_w2", (x_basis,))
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,))
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,))
tau_p = ScalarField(dist, "tau_p", (x_basis,))

# Parameters
Ra = 1e6   # Rayleigh number
Pr = 1.0   # Prandtl number

problem = IVP([u, w, p, T, tau_u1, tau_u2, tau_w1, tau_w2, tau_T1, tau_T2, tau_p])

# Momentum equations
add_equation!(problem, dt(u) - Pr*lap(u) + dx(p)
    + lift(tau_u1, z_basis, -1) + lift(tau_u2, z_basis, -2))
add_equation!(problem, dt(w) - Pr*lap(w) + dz(p) - Ra*Pr*T
    + lift(tau_w1, z_basis, -1) + lift(tau_w2, z_basis, -2))

# Continuity
add_equation!(problem, dx(u) + dz(w) + lift(tau_p, z_basis, -1))

# Temperature equation
add_equation!(problem, dt(T) - lap(T)
    + lift(tau_T1, z_basis, -1) + lift(tau_T2, z_basis, -2))

# Boundary conditions
# No-slip walls
add_dirichlet_bc!(problem, u, "z", :left, 0.0; tau_field="tau_u1")
add_dirichlet_bc!(problem, u, "z", :right, 0.0; tau_field="tau_u2")
add_dirichlet_bc!(problem, w, "z", :left, 0.0; tau_field="tau_w1")
add_dirichlet_bc!(problem, w, "z", :right, 0.0; tau_field="tau_w2")

# Fixed temperature
add_dirichlet_bc!(problem, T, "z", :left, 1.0; tau_field="tau_T1")   # Hot bottom
add_dirichlet_bc!(problem, T, "z", :right, 0.0; tau_field="tau_T2")  # Cold top

# Pressure gauge
add_dirichlet_bc!(problem, p, "z", :left, 0.0; tau_field="tau_p")
```

## Validation

### Checking BC Satisfaction

```julia
# After solving, verify BCs are satisfied
function check_dirichlet_bc(field, coord, location, expected_value; tol=1e-10)
    to_grid!(field)
    data = get_grid_data(field)

    # Get boundary values
    if coord == "z" && location == :left
        bc_values = data[:, 1]
    elseif coord == "z" && location == :right
        bc_values = data[:, end]
    end

    error = maximum(abs.(bc_values .- expected_value))
    @assert error < tol "BC error: $error"

    return error
end

# Usage
check_dirichlet_bc(T, "z", :left, 1.0)
check_dirichlet_bc(T, "z", :right, 0.0)
```

## Troubleshooting

### Missing Tau Fields

**Error**: `ArgumentError: Missing tau field specifications for boundary conditions`

**Solution**: Create tau fields and pass them via the `tau_field` parameter:

```julia
# Wrong:
add_dirichlet_bc!(problem, u, "z", :left, 0.0)  # Missing tau_field!

# Correct:
tau_u1 = ScalarField(dist, "tau_u1", (x_basis,))
add_dirichlet_bc!(problem, u, "z", :left, 0.0; tau_field="tau_u1")
```

### Over-Specified System

**Problem**: Too many boundary conditions cause singular matrices.

**Solution**: Match the number of BCs (and tau fields) to the operator order in each direction.

### Under-Specified System

**Problem**: Not enough BCs lead to non-unique solutions.

**Solution**: Add appropriate boundary conditions for the problem physics.

### Tau Field Dimension Mismatch

**Problem**: Tau field has wrong dimensionality.

**Solution**: Tau fields should live on the *boundary*, not the full domain:

```julia
# For a 2D problem with x-basis (periodic) and z-basis (non-periodic):
# Boundaries are at constant z, so tau fields live on x-basis only
tau_u = ScalarField(dist, "tau_u", (x_basis,))  # Correct: 1D on x

# NOT:
tau_u = ScalarField(dist, "tau_u", (x_basis, z_basis))  # Wrong: 2D
```

## See Also

- [Tau Method (Advanced)](../pages/tau_method.md): Mathematical details
- [Problems API](../api/problems.md): Problem definition
- [Bases API](../api/bases.md): Spectral basis selection
- [2D Rayleigh-Bénard Tutorial](ivp_2d_rbc.md): Complete example with BCs
