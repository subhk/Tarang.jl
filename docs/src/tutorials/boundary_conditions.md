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
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

# Create distributor and fields
dist = Distributor(coords)
u = ScalarField(dist, "u", (x_basis, z_basis))
f = ScalarField(dist, "f", (x_basis, z_basis))  # Source term

# Step 1: Create tau fields (one per BC)
# These live on the x-basis only (boundary is a line in 2D)
tau_u1 = ScalarField(dist, "tau_u1", (x_basis,))  # For BC at z=0
tau_u2 = ScalarField(dist, "tau_u2", (x_basis,))  # For BC at z=1

# Step 2: Add ALL fields to problem (including tau fields)
problem = LBVP([u, tau_u1, tau_u2])

# Step 3: Add substitution for source term
add_substitution!(problem, "f", f)

# Step 4: Add equation with lift() operators (Dedalus-style string format)
add_equation!(problem, "Δ(u) + lift(tau_u1) + lift(tau_u2) = f")

# Step 5: Add boundary conditions
add_bc!(problem, "u(z=0) = 0")   # u(z=0) = 0
add_bc!(problem, "u(z=1) = 0")   # u(z=1) = 0

# Solve
solver = BoundaryValueSolver(problem)
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

# Add equation with lift terms (Dedalus-style string format)
add_equation!(problem, "Δ(T) + lift(tau_T1) + lift(tau_T2) = source")

# Boundary conditions
add_bc!(problem, "T(z=0) = 1")   # T(z=0) = 1
add_bc!(problem, "T(z=1) = 0")   # T(z=1) = 0
```

### No-Slip Velocity (Vector Fields)

For viscous flows at solid walls, use vector fields for compact notation:

```julia
# Vector field for velocity
u = VectorField(dist, coords, "u", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))

# Vector tau fields for BCs at each wall
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,))  # Wall at z=0
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,))  # Wall at z=1
tau_p = ScalarField(dist, "tau_p", ())

# Pass vector fields directly to problem
problem = IVP([u, p, tau_u1, tau_u2, tau_p])

# Add substitutions
add_substitution!(problem, "nu", nu)

# Momentum equation (single vector equation)
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) + lift(tau_u2) = -u⋅∇(u)")

# Continuity with tau_p (removes degeneracy)
add_equation!(problem, "div(u) + tau_p = 0")

# No-slip boundary conditions (vector notation)
add_bc!(problem, "u(z=0) = 0")   # No-slip bottom (all components)
add_bc!(problem, "u(z=1) = 0")   # No-slip top (all components)
```

## Neumann Boundary Conditions

Specify the derivative (flux) at the boundary.

### Basic Setup

```julia
# Tau fields for Neumann BCs
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,))
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,))

problem = LBVP([T, tau_T1, tau_T2])

add_equation!(problem, "Δ(T) + lift(tau_T1) + lift(tau_T2) = source")

# Neumann: specify derivative at boundary
add_bc!(problem, "∂z(T)(z=0) = 1")   # ∂T/∂z(z=0) = 1
add_bc!(problem, "∂z(T)(z=1) = 0")   # ∂T/∂z(z=1) = 0
```

### Stress-Free Conditions

For free surfaces or slip boundaries (∂u/∂z = 0):

```julia
# Mixed: no-slip at bottom, stress-free at top
add_bc!(problem, "u_x(z=0) = 0")      # No-slip at bottom
add_bc!(problem, "∂z(u_x)(z=1) = 0")  # Stress-free at top (∂u/∂z = 0)
```

## Robin Boundary Conditions

Linear combination: $\alpha u + \beta \frac{\partial u}{\partial n} = \gamma$

```julia
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,))
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,))

problem = LBVP([T, tau_T1, tau_T2])

# Add parameters
add_substitution!(problem, "h", 10.0)   # Heat transfer coefficient
add_substitution!(problem, "k", 1.0)    # Thermal conductivity
add_substitution!(problem, "T_amb", 25.0)

add_equation!(problem, "Δ(T) + lift(tau_T1) + lift(tau_T2) = source")

# Convective heat transfer at top: h*T + k*dT/dn = h*T_ambient
add_bc!(problem, "h*T(z=1) + k*∂z(T)(z=1) = h*T_amb")

# Dirichlet at bottom
add_bc!(problem, "T(z=0) = 100")
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

The `lift()` operator places tau corrections at specific spectral modes. In the string equation format:

```julia
"Δ(u) + lift(tau_u1) + lift(tau_u2) = f"
```

The tau field name in the lift() operator should match the tau field name you created.

### Mode Selection Guidelines

| Operator Order | Number of BCs | Number of lift() terms |
|---------------|---------------|------------------------|
| 1st (∂/∂z)    | 1             | 1                      |
| 2nd (∂²/∂z²)  | 2             | 2                      |
| 4th (∇⁴)      | 4             | 4                      |

### Example: Fourth-Order Problem

For a biharmonic equation (∇⁴u = f) with 4 boundary conditions:

```julia
# Four tau fields needed
tau_u1 = ScalarField(dist, "tau_u1", (x_basis,))
tau_u2 = ScalarField(dist, "tau_u2", (x_basis,))
tau_u3 = ScalarField(dist, "tau_u3", (x_basis,))
tau_u4 = ScalarField(dist, "tau_u4", (x_basis,))

problem = LBVP([u, tau_u1, tau_u2, tau_u3, tau_u4])

# Biharmonic equation with all four lift terms
add_equation!(problem, "Δ(Δ(u)) + lift(tau_u1) + lift(tau_u2) + lift(tau_u3) + lift(tau_u4) = f")

# Clamped beam: u = 0 and du/dz = 0 at both ends
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "∂z(u)(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "∂z(u)(z=1) = 0")
```

## Complete Examples

### Channel Flow (Poiseuille Flow)

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
x_basis = RealFourier(coords["x"]; size=64, bounds=(0.0, 2π))
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))
dist = Distributor(coords)

# Vector velocity field
u = VectorField(dist, coords, "u", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))

# Vector tau fields for velocity BCs at each wall
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,))  # Wall at z=0
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,))  # Wall at z=1

# Tau for pressure (removes degeneracy)
tau_p = ScalarField(dist, "tau_p", ())

# Parameters
nu = 0.01
dpdx = -1.0

# Create problem with all fields
problem = IVP([u, p, tau_u1, tau_u2, tau_p])

# Add parameter substitutions
add_substitution!(problem, "nu", nu)
add_substitution!(problem, "dpdx", dpdx)

# Momentum equation (vector form) - dpdx is the driving pressure gradient
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) + lift(tau_u2) = -u⋅∇(u) - dpdx*ex")

# Continuity with tau_p (removes degeneracy)
add_equation!(problem, "div(u) + tau_p = 0")

# No-slip at both walls (vector notation)
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
```

### Rayleigh-Bénard Convection

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
x_basis = RealFourier(coords["x"]; size=128, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))
dist = Distributor(coords)

# Vector velocity field and scalar fields
u = VectorField(dist, coords, "u", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))
T = ScalarField(dist, "T", (x_basis, z_basis))

# Vector tau fields for velocity BCs
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,))  # Wall at z=0
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,))  # Wall at z=1

# Scalar tau fields for temperature BCs
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,))  # BC at z=0
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,))  # BC at z=1

# Tau for pressure (removes degeneracy)
tau_p = ScalarField(dist, "tau_p", ())

# Parameters
Ra = 1e6   # Rayleigh number
Pr = 1.0   # Prandtl number

# Create problem with all fields
problem = IVP([u, p, T, tau_u1, tau_u2, tau_T1, tau_T2, tau_p])

# Add parameter substitutions
add_substitution!(problem, "Ra", Ra)
add_substitution!(problem, "Pr", Pr)

# Momentum equation (vector form with buoyancy)
# ez is the unit vector in z-direction
add_equation!(problem, "∂t(u) - Pr*Δ(u) + ∇(p) + lift(tau_u2) = -u⋅∇(u) + Ra*Pr*T*ez")

# Continuity with tau_p (removes degeneracy)
add_equation!(problem, "div(u) + tau_p = 0")

# Temperature equation
add_equation!(problem, "∂t(T) - Δ(T) + lift(tau_T2) = -u⋅∇(T)")

# Boundary conditions (vector notation for velocity)
add_bc!(problem, "u(z=0) = 0")   # No-slip bottom
add_bc!(problem, "u(z=1) = 0")   # No-slip top

# Fixed temperature
add_bc!(problem, "T(z=0) = 1")   # Hot bottom
add_bc!(problem, "T(z=1) = 0")   # Cold top
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

**Solution**: Create tau fields and include lift() terms in your equations:

```julia
# Wrong: No tau fields or lift terms
add_equation!(problem, "Δ(u) = f")
add_bc!(problem, "u(z=0) = 0")

# Correct: Create tau fields and add lift terms
tau_u1 = ScalarField(dist, "tau_u1", (x_basis,))
tau_u2 = ScalarField(dist, "tau_u2", (x_basis,))
add_equation!(problem, "Δ(u) + lift(tau_u1) + lift(tau_u2) = f")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
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
