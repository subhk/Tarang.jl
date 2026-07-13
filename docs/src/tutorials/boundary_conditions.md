# Tutorial: Boundary Conditions

This tutorial covers the different types of boundary conditions available in Tarang.jl and how to apply them effectively using the tau method with explicit tau fields.

## Overview

Boundary conditions (BCs) are essential for well-posed PDE problems. Tarang.jl supports:

- **Dirichlet**: Specify field value at boundary
- **Neumann**: Specify derivative at boundary
- **Robin**: Linear combination of value and derivative
- **Periodic**: Automatic for Fourier bases

## The Tau Method Approach

Tarang.jl uses an explicit tau-method approach for handling boundary conditions. Users must **explicitly create tau fields** and add them to equations using the `lift()` operator.

### Why Explicit Tau Fields?

1. **Clarity**: The mathematical structure is visible in your code
2. **Flexibility**: Full control over tau placement
3. **Debugging**: Easy to inspect tau field values
4. **Consistency**: Matches the mathematical formulation

### Required Steps

For any problem with non-periodic boundary conditions:

1. **Create tau fields** — one scalar tau DOF per scalar constraint row (a BC on a
   `VectorField` contributes one row per component, so it needs a `VectorField` tau)
2. **Add tau fields to the problem** — include them in the field list
3. **Add lift() terms to equations** — place tau contributions at specific modes
4. **Specify boundary conditions** — add the algebraic constraint rows. Tau columns and
   BC rows couple through the assembled matrix, not through a name-based link between a
   BC and "its" tau field

Because there is no name-based pairing, the thing that keeps the system square is a
**count**. Every variable you declare must be matched by an equation or a BC; Tarang
checks this and rejects the problem otherwise. A tau field that you declare but never
lift into an equation is an error, not a harmless extra.

The wording of that rejection depends on the problem type, because a BVP is allowed to
carry extra equations (its BCs) while an IVP is not:

| problem | check | message |
|---|---|---|
| `LBVP` / `NLBVP` | `n_equations >= n_variables` | `Number of equations (2) is less than number of variables (3)` |
| `IVP` / `EVP` | `n_equations == n_variables` | `Number of equations (7) does not match number of variables (5)` |

## Complete Example: Poisson Equation

Let's solve the Poisson equation $\nabla^2 u = -2$ with Dirichlet BCs `u = 0` on
both `z` walls. This is a verified, runnable LBVP; the analytic solution is
`u = z(Lz - z)` and the solver reproduces it to machine precision.

The tau method needs **one tau variable per boundary condition**. Each tau is
lifted into the bulk equation with `lift(tau, derivative_basis(z_basis, 2), -k)`
and registered as a parameter via `add_parameters!`. The lift order index `-1`
targets the last coefficient row and `-2` the second-to-last; a 2nd-order problem
therefore uses two taus at orders `-1` and `-2`.

```julia
using Tarang

# Create coordinates, distributor, and bases
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
x_basis = RealFourier(coords["x"]; size=4,  bounds=(0.0, 2π))   # periodic (separable)
z_basis = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))  # bounded  (coupled)
dom = Domain(dist, (x_basis, z_basis))

# Field to solve for
u = ScalarField(dom, "u")

# Step 1: Create tau fields (one per BC)
# These live on the x-basis only (boundary is a line in 2D)
tau1 = ScalarField(dist, "tau1", (x_basis,), Float64)  # For BC at z=0
tau2 = ScalarField(dist, "tau2", (x_basis,), Float64)  # For BC at z=1

# Step 2: Add ALL fields to problem (including tau fields)
problem = LBVP([u, tau1, tau2])

# Step 3: Register the lifted tau terms as parameters
#         (use the 2nd-derivative basis for a 2nd-order problem)
lb2 = derivative_basis(z_basis, 2)
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))

# Step 4: Add equation referencing the lifted tau parameters
add_equation!(problem, "Δ(u) + l1 + l2 = -2")

# Step 5: Add boundary conditions
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

# Solve
solver = BoundaryValueSolver(problem)
solve!(solver)
ensure_layout!(u, :g)   # solve! writes coefficients; switch to grid space to read u
```

The solution matches `z(1 - z)` at every grid point to `1.4e-16`, and both walls
come out at exactly `0.0`.

!!! note "1D pure-Chebyshev BVP"
    The example above keeps a Fourier `x` axis, but a pure single-axis Chebyshev
    BVP (no Fourier) also works — drop the `x` axis and define the `tau` variables
    on `()`. The solver builds one coupled tau subproblem over the Chebyshev
    spectrum.

## Dirichlet Boundary Conditions

Fix the value of a field at the boundary.

### Basic Setup

Continuing with the `dist`, `x_basis`, `z_basis` and `dom` from above:

```julia
T = ScalarField(dom, "T")

# Create tau fields for each Dirichlet BC
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,), Float64)

# Add to problem
problem = LBVP([T, tau_T1, tau_T2])

# Register the lifted tau terms (2nd-order problem -> 2nd-derivative basis)
lb2 = derivative_basis(z_basis, 2)
add_parameters!(problem; l1=lift(tau_T1, lb2, -1), l2=lift(tau_T2, lb2, -2), source=0.0)

# Add equation referencing the lifted tau parameters
add_equation!(problem, "Δ(T) + l1 + l2 = source")

# Boundary conditions
add_bc!(problem, "T(z=0) = 1")   # hot wall
add_bc!(problem, "T(z=1) = 0")   # cold wall

solver = BoundaryValueSolver(problem)
solve!(solver)
ensure_layout!(T, :g)
```

With a zero source this is the conduction profile `T = 1 - z` (recovered to
`2.5e-16`); the walls come out at exactly `1.0` and `0.0`.

### No-Slip Velocity (Vector Fields)

For viscous flows at solid walls, use vector fields for compact notation.

The idiomatic form is a **first-order reduction**: rather than lifting two taus into a
single second-order operator, introduce the gradient as a substitution and lift one tau
into it. A second-order momentum equation then needs **two** velocity tau fields: one
lifted into the *gradient* (`tau_u1`, giving `grad_u = ∇u + ẑ·lift(τ)`), one lifted into
the *equation* (`tau_u2`). Both lifts use the **first-derivative** basis,
`derivative_basis(z_basis, 1)` — see "Which derivative_basis order?" below. The
divergence constraint gets its own gauge tau, `tau_p`, paid for by an `integ(p)`
condition.

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
x_basis = RealFourier(coords["x"]; size=8,  bounds=(0.0, 2π))
z_basis = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))
dom = Domain(dist, (x_basis, z_basis))

p = ScalarField(dom, "p")
u = VectorField(dom, "u")

# Tau fields: scalar gauge for the pressure, one vector tau per lift site
tau_p  = ScalarField(dist, "tau_p", (), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,), Float64)

# First-order reduction: grad_u = ∇u + ẑ·lift(tau_u1)
ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(z_basis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u     = grad(u) + ez * τ_lift(tau_u1)

problem = IVP([p, u, tau_p, tau_u1, tau_u2])
add_parameters!(problem, nu=0.01, ez=ez, grad_u=grad_u, τ_lift=τ_lift)

# Continuity with tau_p (removes the pressure degeneracy)
add_equation!(problem, "trace(grad_u) + tau_p = 0")

# Momentum (single vector equation)
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) + τ_lift(tau_u2) = -u⋅∇(u)")

# No-slip boundary conditions (vector notation — all components at once)
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "integ(p) = 0")   # pressure gauge, pays for tau_p

solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# Start from a decaying shear layer
ux = u.components[1]
x, z = local_grids(dist, x_basis, z_basis)
ensure_layout!(ux, :g)
get_grid_data(ux) .= sin.(π .* z')
ensure_layout!(ux, :c)

run!(solver; stop_iteration=20, progress=false)
```

After 20 steps the no-slip walls hold to `1.1e-16` and the interior shear layer
has decayed slightly (`max|u_x| = 0.985`, from 1.0).

## Neumann Boundary Conditions

Specify the derivative (flux) at the boundary.

### Basic Setup

```julia
T = ScalarField(dom, "T")
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,), Float64)

problem = LBVP([T, tau_T1, tau_T2])

# Register the lifted tau terms (2nd-order problem -> 2nd-derivative basis)
lb2 = derivative_basis(z_basis, 2)
add_parameters!(problem; l1=lift(tau_T1, lb2, -1), l2=lift(tau_T2, lb2, -2), source=0.0)

add_equation!(problem, "Δ(T) + l1 + l2 = source")

# Neumann: prescribed flux at z=0, Dirichlet at z=1
add_bc!(problem, "∂z(T)(z=0) = 1")
add_bc!(problem, "T(z=1) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
ensure_layout!(T, :g)
```

The exact solution is `T = z - 1`, and the solve returns it to `2.5e-16`.

Pair each Neumann condition with something that pins the level of the solution.
Two Neumann conditions on a Laplacian leave the solution defined only up to a
constant; the tau solve will still return *an* answer, but it is not the one you
meant.

### Per-Component Conditions (Stress-Free Walls)

A BC applies to a **whole variable**. `add_bc!(problem, "u(z=0) = 0")` constrains
every component of the `VectorField` `u`, and there is no string syntax that
reaches into a single component: `"u_x(z=0) = 0"` does not resolve to a component
of `u`, so the four per-component strings are counted as four extra constraint
rows and the problem fails validation (`Number of equations (7) does not match
number of variables (5)`).

To impose different conditions on different components — a stress-free wall,
where `∂u_x/∂z = 0` but `u_z = 0` — declare the components as separate
`ScalarField`s, each with its own tau pair:

```julia
ux = ScalarField(dom, "ux")
uz = ScalarField(dom, "uz")

tx1 = ScalarField(dist, "tx1", (x_basis,), Float64)
tx2 = ScalarField(dist, "tx2", (x_basis,), Float64)
tz1 = ScalarField(dist, "tz1", (x_basis,), Float64)
tz2 = ScalarField(dist, "tz2", (x_basis,), Float64)

problem = LBVP([ux, uz, tx1, tx2, tz1, tz2])
lb2 = derivative_basis(z_basis, 2)
add_parameters!(problem; lx1=lift(tx1, lb2, -1), lx2=lift(tx2, lb2, -2),
                         lz1=lift(tz1, lb2, -1), lz2=lift(tz2, lb2, -2))

add_equation!(problem, "Δ(ux) + lx1 + lx2 = -2")
add_equation!(problem, "Δ(uz) + lz1 + lz2 = 0")

add_bc!(problem, "ux(z=0) = 0")       # no-slip at the bottom
add_bc!(problem, "∂z(ux)(z=1) = 0")   # stress-free at the top
add_bc!(problem, "uz(z=0) = 0")
add_bc!(problem, "uz(z=1) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
ensure_layout!(ux, :g)
```

The exact solution is `ux = 2z - z²` (zero value at the bottom, zero slope at the
top), recovered to `1.7e-16`.

## Robin Boundary Conditions

Linear combination: $\alpha u + \beta \frac{\partial u}{\partial n} = \gamma$

Parameters registered with `add_parameters!` may be used both as **coefficients**
on the left-hand side of a BC and inside a **constant right-hand side** — the
product `h*T_amb` is folded to a number when the BC row is built.

```julia
T = ScalarField(dom, "T")
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,), Float64)

problem = LBVP([T, tau_T1, tau_T2])

# Register the lifted tau terms and the scalar parameters
lb2 = derivative_basis(z_basis, 2)
add_parameters!(problem; l1=lift(tau_T1, lb2, -1), l2=lift(tau_T2, lb2, -2),
                         h=10.0, k=1.0, T_amb=25.0, source=0.0)

add_equation!(problem, "Δ(T) + l1 + l2 = source")

# Convective heat transfer at top: h*T + k*dT/dz = h*T_ambient
add_bc!(problem, "h*T(z=1) + k*∂z(T)(z=1) = h*T_amb")

# Dirichlet at bottom
add_bc!(problem, "T(z=0) = 100")

solver = BoundaryValueSolver(problem)
solve!(solver)
ensure_layout!(T, :g)
```

The solve returns `T = 100 - 68.18 z`, and the Robin residual
`h·T(1) + k·T'(1)` comes out at exactly `250.0 = h·T_amb`.

!!! warning "A name in a BC must be a registered parameter"
    The folding only reaches numbers and names passed to `add_parameters!`. A
    plain Julia global — `Tbot = 3.0` in your script, never registered — is
    **not** in scope: the BC is enforced as **zero**. It is at least loud about
    it (`Unknown variable: Tbot`, then `right-hand side … enforced as ZERO`).
    See [Constants and parameters in BC expressions](#Constants-and-parameters-in-BC-expressions)
    for the full rules.

## Periodic Boundary Conditions

Periodic boundaries are automatically handled by Fourier bases - no tau fields needed!

```julia
# Fourier basis implies periodicity
periodic_x = RealFourier(coords["x"]; size=128, bounds=(0.0, 2π))

# No boundary conditions or tau fields needed for x-direction
# The Fourier representation automatically enforces u(x=0) = u(x=2π)
```

!!! warning "Mixing Periodic and Non-Periodic"
    When using Fourier (periodic) and Chebyshev (non-periodic) bases together, only create tau fields and boundary conditions for the non-periodic directions.

## The lift() Operator

The `lift(tau, derivative_basis(basis, order), -k)` operator places a tau
correction at a specific coefficient row of the bounded (Chebyshev/Jacobi) basis.
You build the lifted terms once and register them as parameters, then reference
those parameter names in the equation string:

```julia
u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (x_basis,), Float64)
tau2 = ScalarField(dist, "tau2", (x_basis,), Float64)
problem = LBVP([u, tau1, tau2])

lb2 = derivative_basis(z_basis, 2)   # direct 2nd-order equation -> 2nd-derivative basis
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
```

The order index `-1` lifts into the last coefficient row, `-2` the
second-to-last, and so on. The number of taus (and lift terms) must match the
operator order in the bounded direction.

### Which derivative_basis order?

The lift basis order follows the order of the operator you are lifting **into**, and
that depends on which formulation you wrote:

| Formulation | Lift basis | Tau layout |
|---|---|---|
| Direct 2nd-order equation, e.g. `Δ(u) + l1 + l2 = f` | `derivative_basis(z_basis, 2)` | two taus, lifted at `-1` and `-2` |
| First-order reduction, `grad_u = grad(u) + ez*lift(tau_u1, …, -1)` then `div(grad_u)` | `derivative_basis(z_basis, 1)` | two taus, each lifted at `-1` — one into the gradient, one into the equation |
| Direct 4th-order equation (∇⁴) | `derivative_basis(z_basis, 4)` | four taus, lifted at `-1 … -4` |

The first-order reduction is **order 1, not 2** — this is the form used in the IVP
examples below and in the Rayleigh-Bénard tutorial. Both formulations spend the same
budget: two taus and two BCs per field per wall pair.

!!! note "The basis argument is bookkeeping; the mode index is what assembles"
    `subproblem_matrix(::Lift, …)` builds the lift column from the subproblem's own
    Chebyshev basis and the mode index `-k` alone — the `derivative_basis(…)` object you
    pass is not consulted when the matrix is assembled. Passing the wrong order therefore
    does **not** raise an error, and the 2nd-order Poisson example above returns a
    bit-identical answer with `derivative_basis(z_basis, 1)`. Get the order right anyway:
    it documents the formulation, and it is what the rest of the framework (and any
    future matrix builder) will read.

### Mode Selection Guidelines

| Operator Order | Number of BCs | Number of lift() terms | derivative_basis order | Lift indices  |
|----------------|---------------|------------------------|------------------------|---------------|
| 1st (∂/∂z)     | 1             | 1                      | 1                      | -1            |
| 2nd (∂²/∂z²)   | 2             | 2                      | 2                      | -1, -2        |
| 4th (∇⁴)       | 4             | 4                      | 4                      | -1, -2, -3, -4|

### Example: Fourth-Order Problem

For a biharmonic equation (∇⁴u = f) with 4 boundary conditions:

```julia
u = ScalarField(dom, "u")

# Four tau fields needed
tau_u1 = ScalarField(dist, "tau_u1", (x_basis,), Float64)
tau_u2 = ScalarField(dist, "tau_u2", (x_basis,), Float64)
tau_u3 = ScalarField(dist, "tau_u3", (x_basis,), Float64)
tau_u4 = ScalarField(dist, "tau_u4", (x_basis,), Float64)

problem = LBVP([u, tau_u1, tau_u2, tau_u3, tau_u4])

# Register the lifted tau terms (4th-order problem -> 4th-derivative basis)
lb4 = derivative_basis(z_basis, 4)
add_parameters!(problem; l1=lift(tau_u1, lb4, -1), l2=lift(tau_u2, lb4, -2),
                         l3=lift(tau_u3, lb4, -3), l4=lift(tau_u4, lb4, -4), f=1.0)

# Biharmonic equation with all four lifted tau terms
add_equation!(problem, "Δ(Δ(u)) + l1 + l2 + l3 + l4 = f")

# Clamped beam: u = 0 and du/dz = 0 at both ends
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "∂z(u)(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "∂z(u)(z=1) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
ensure_layout!(u, :g)
```

The clamped beam with a unit load has the exact solution `u = z²(1-z)²/24`, which
the solve reproduces to `1.4e-18`.

## Complete Examples

### Channel Flow (Poiseuille Flow)

A channel driven by a constant pressure gradient. The forcing enters the explicit
right-hand side as a parameter times a unit vector, `- dpdx*ex`; the scalar coefficient
is honoured (doubling `dpdx` doubles the start-up velocity).

```julia
using Tarang

Lx, Lz = 2π, 1.0
Nx, Nz = 16, 16
nu   = 0.01
dpdx = -1.0

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
x_basis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx))
z_basis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz))
dom = Domain(dist, (x_basis, z_basis))

p = ScalarField(dom, "p")
u = VectorField(dom, "u")

# First-order tau fields
tau_p  = ScalarField(dist, "tau_p", (), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,), Float64)

# First-order viscous operator
ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(z_basis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u     = grad(u) + ez * τ_lift(tau_u1)

problem = IVP([u, p, tau_u1, tau_u2, tau_p])
add_parameters!(problem, nu=nu, dpdx=dpdx, ex=ex, grad_u=grad_u, τ_lift=τ_lift)

# Continuity with tau_p (removes degeneracy)
add_equation!(problem, "trace(grad_u) + tau_p = 0")

# Momentum (vector form) — dpdx is the driving pressure gradient
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) + τ_lift(tau_u2) = -u⋅∇(u) - dpdx*ex")

# No-slip at both walls (vector notation)
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "integ(p) = 0")   # pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
run!(solver; stop_iteration=50, progress=false)
```

Starting from rest, the flow is still in its linear start-up phase after 50 steps
(`t = 0.05`): the measured peak is `max u_x = 0.0500`, matching `|dpdx|·t = 0.05`, and
no-slip holds at both walls to `7e-18`. The steady Poiseuille profile it is heading for
is `u_x = |dpdx|·z(Lz - z) / (2ν)`, which takes `t ≫ Lz²/ν` to establish.

!!! note "Two things to know about this forcing"
    * A **non-uniform** body force cannot be written as `parameter * unit_vector`. Build a
      real `VectorField` on the domain, fill its components in grid space, and register it
      with `add_parameters!(problem, F=F)`; then write `… = -u⋅∇(u) + F`.
    * Naming the parameter `dpdx` trips a false-positive parser warning —
      `Linear term on RHS: 'p' appears linearly on RHS` — because the name *contains* the
      variable name `p`. It is harmless (the run above is correct); rename the parameter
      (e.g. `fx`) if you want a clean log.

### Rayleigh-Bénard Convection

The full convection problem: momentum, continuity and temperature, with a
first-order reduction on both `u` and `T`.

```julia
using Tarang

Lx, Lz = 4.0, 1.0
Nx, Nz = 32, 16
Ra, Pr = 2e4, 1.0

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
x_basis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
z_basis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=3/2)
dom = Domain(dist, (x_basis, z_basis))

p = ScalarField(dom, "p")
T = ScalarField(dom, "T")
u = VectorField(dom, "u")

tau_p  = ScalarField(dist, "tau_p",  (), Float64)
tau_T1 = ScalarField(dist, "tau_T1", (x_basis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (x_basis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (x_basis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (x_basis,), Float64)

# First-order substitutions (lift basis order 1)
ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(z_basis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])
add_parameters!(problem, nu=Pr, buoy=Ra*Pr, ez=ez,
                grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")

# Boundary conditions (vector notation for velocity)
add_bc!(problem, "u(z=0) = 0")   # No-slip bottom
add_bc!(problem, "u(z=1) = 0")   # No-slip top
add_bc!(problem, "T(z=0) = 1")   # Hot bottom
add_bc!(problem, "T(z=1) = 0")   # Cold top
add_bc!(problem, "integ(p) = 0") # Pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=1e-4)

# Conduction profile + damped noise
x, z = local_grids(dist, x_basis, z_basis)
fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (Lz .- z')
get_grid_data(T) .+= 1.0 .- z' ./ Lz
ensure_layout!(T, :c)

run!(solver; stop_iteration=20, progress=false)
```

After 20 steps the buoyancy is driving a weak vertical flow
(`max|u_z| = 7.6e-3`), and the wall temperatures hold to machine precision:
`max|T(z=0) - 1| = 0.0`, `max|T(z=Lz)| = 1.6e-17`.

!!! warning "Run this example serially"
    A `(Fourier x, Chebyshev z)` domain cannot be decomposed as written: Tarang stops at
    `np ≥ 2` with *"the decomposed (trailing) axis/axes [2] are non-Fourier … a Chebyshev
    axis cannot be decomposed"*, because the DCT needs the whole `z` axis local to each
    rank. Putting the Chebyshev axis first gets past that guard, but the 1-D boundary tau
    fields then hit the "MPI parallelization is not supported for 1D problems" guard. The
    LBVPs above (Poisson, Neumann, Robin, biharmonic) are unaffected — they are solved per
    Fourier mode.

## Time- and Space-Dependent Boundary Conditions

An `IVP` supports BCs whose value varies in time, space, or both. The BC
expression string is re-evaluated at solver build time (for space-dependent) and
at every time step / RK stage (for time-dependent), and the result is projected
onto the appropriate Fourier mode for each per-subproblem solve.

Nothing extra is required — no `add_coordinate_field!` call, no `set_time_variable!` — the solver auto-registers coordinate arrays from the problem's bases and uses `t` as the default time variable.

!!! note "Time-stepped problems only"
    This machinery is driven by the stepper. In an `LBVP`/`NLBVP` a
    space-dependent BC string is currently evaluated as **zero** — use a constant
    BC there.

### Time-dependent BC (scalar)

```julia
add_bc!(problem, "T(z=0) = 1.0 + 0.1 * cos(2*pi*t)")
```

The BC value is re-evaluated **at every RK stage time** `t + c[i]·dt`, so multi-stage methods retain full formal order of accuracy even for rapidly-varying BCs. Internally this goes through `update_time_dependent_bcs!(bc_manager, stage_time)` followed by `_apply_bc_values_to_equations!`, and the resulting `ConstantOperator(value)` is written into `equation_data[eq_idx]["F"]`.

Measured on a 2D diffusion problem after 100 steps to `t = 0.1`, the enforced
wall value tracks `1 + 0.1·cos(2πt)` to `2.2e-16`.

### Space-dependent BC (1D variation in 2D problem)

```julia
add_bc!(problem, "T(z=0) = 1.0 + 0.1 * sin(2*pi*x/4.0)")
add_bc!(problem, "T(z=1) = 0")
```

At solver build, `_apply_bc_values_to_equations!(solver, 0.0)` evaluates `"1.0 + 0.1*sin(2*pi*x/4.0)"` against the auto-registered global `x` coordinate array, yielding an `Nx`-long grid-space array. This is wrapped in an `ArrayOperator` and stored in `equation_data[eq_idx]["F"]`. At each stepper call, `gather_alg_F!` runs `_bc_array_projection` on this array — taking an unnormalized `FFTW.rfft` and extracting each subproblem's own Fourier-mode coefficient.

The bottom-wall temperature is thus enforced at `T(x, z=0) = 1 + 0.1·sin(2πx/Lx)`
in grid space — verified to `3.3e-15`.

### Space + time dependent BC

```julia
add_bc!(problem, "T(z=0) = 1.0 + 0.1 * sin(2*pi*x/4.0) * cos(2*pi*t)")
```

Combines both paths. The spatial pattern is re-projected at each stage time, so the stepper always sees the correct instantaneous BC. This is what enables oscillating boundary temperature patterns, traveling thermal waves at the wall, etc. Measured after 100 steps to `t = 0.1`, the enforced wall value tracks `1 + 0.1·sin(2πx/Lx)·cos(2πt)` to `2.0e-15`.

### 3D problems: both periodic axes

In a 3D problem with two periodic axes `(x, y)`, a BC expression may vary along
`x`, along `y`, or along both.

A 1D expression in `x` alone is broadcast along `y` — "uniform in `y`":

```julia
# 3D problem; this is constant in y by design
add_bc!(problem, "T(z=0) = sin(2*pi*x/4.0)")
```

The broadcast-and-FFT machinery places `rfft(sin(2π x/Lx))[k_x] · Ny` at the DC
`y` mode and zero at non-DC `y` modes, matching the grid-space meaning exactly.
Measured error `3.3e-16` on an 8×8×8 `RealFourier` × `RealFourier` ×
`ChebyshevT` problem.

A genuinely two-axis expression works as well: it is evaluated as an `(Nx, Ny)`
grid-space array, and each `(kx, ky)` subproblem extracts its own coefficient.
Either of these is a valid single `z=0` condition:

```julia
# varies along y only
add_bc!(problem, "T(z=0) = cos(2*pi*y/2.0)")

# varies along both axes (use ONE of these, not both)
# add_bc!(problem, "T(z=0) = sin(2*pi*x/4.0) * cos(2*pi*y/2.0)")
```

On the same 8×8×8 problem the enforced wall matches the intended grid-space
pattern to `3.3e-16` for the `y`-only form and `4.4e-16` for the `x·y` product.

!!! warning "Bake the box size in as a literal, not as `Lx`/`Ly`"
    Note the `2.0` and `4.0` above rather than `Ly` and `Lx`. This is the
    coordinate-expression rule from the table below, and it bites hardest here.
    `"T(z=0) = cos(2*pi*y/Ly)"` does **not** fail because of the `y` axis — it
    fails because `Ly` is a *name*, and a coordinate expression cannot see the
    problem's parameters. It is enforced as **zero**, and says so (`Unknown
    variable: Ly`, then `right-hand side … enforced as ZERO`). Registering `Ly`
    with `add_parameters!` does not help; interpolate the number into the string
    instead.

### Constants and parameters in BC expressions

A BC right-hand side is read in one of two ways, and it matters which. A
**constant** right-hand side is folded to a number — arithmetic over numeric
literals and registered parameters is fine. An **expression** right-hand side
(one that mentions a coordinate or `t`) goes through a different evaluator, and
that one sees the coordinate arrays and `t` but **not** the problem's parameters:

| right-hand side | how it is read | works? |
|---|---|---|
| `"T(z=0) = 1"` | bare literal | ✅ |
| `"T(z=0) = Tbot"`, with `add_parameters!(problem, Tbot=3.0)` | bare parameter name | ✅ |
| `"T(z=0) = 10*25"`, `"= h*T_amb"`, `"= 1/Re"` | compound constant, folded | ✅ |
| `"T(z=0) = Tbot"`, with `Tbot` an unregistered Julia global | unknown name | ❌ warns, enforced as **zero** |
| `"T(z=0) = 1.0 + 0.1*sin(2*pi*x/4.0)"` | coordinate/time expression, numeric literals | ✅ |
| `"T(z=0) = 1.0 + amplitude*sin(2*pi*x/Lx)"` | coordinate expression naming parameters | ❌ warns, enforced as **zero** |

So a *constant* BC may name parameters freely:

```julia
add_parameters!(problem, h=10.0, k=1.0, T_amb=25.0)
add_bc!(problem, "h*T(z=1) + k*∂z(T)(z=1) = h*T_amb")
```

but an *expression* BC may not — bake its constants in with Julia string
interpolation instead:

```julia
Lx = 4.0
amplitude = 0.1
add_bc!(problem, "T(z=0) = 1.0 + $amplitude * sin(2*pi*x/$Lx)")
```

An unsupported right-hand side is enforced as zero, but it says so: you get
`Unknown variable: amplitude` followed by `Boundary condition right-hand side of
type … is not supported and is being enforced as ZERO`. Treat that warning as an
error — the solve will otherwise "succeed" against the wrong boundary condition.

### Common pitfalls

- **Missing coordinate field** — for a BC to reference `x`, the problem must have a basis with `element_label == "x"`. This is automatic if you build bases with `coords["x"]`, but if you have multiple coordinate systems the auto-registration may pick the wrong `x`. A `"space-dependent BCs detected but no coordinate fields registered"` warning at solver build means the auto-registration didn't find a matching basis.
- **Time variable other than `t`** — the parser and the runtime evaluator both hard-code `t` as the time symbol. Custom time variable names aren't fully supported; use `t` in your BC expressions.
- **Non-constant custom function** — the BC string evaluator whitelists `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `sign`, `floor`, `ceil`, `round`, `rem`, `min`, `max`, and hyperbolic variants. Arbitrary Julia functions in BC strings are not supported; if you need a custom function, build it into the value at simulation-script level (outside the string) and register via `add_parameters!`.
- **BCs must be declared with `add_bc!`** — passing a boundary condition to `add_equation!` does not register it with the BC manager. A constant BC happens to come out right; anything space- or time-dependent is then silently not enforced.

## Validation

### Checking BC Satisfaction

```julia
# After solving, verify BCs are satisfied
function check_dirichlet_bc(field, coord, location, expected_value; tol=1e-10)
    ensure_layout!(field, :g)
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

Run against the Rayleigh-Bénard solver above, this returns `0.0` at the hot wall
and `1.6e-17` at the cold wall.

## Troubleshooting

### Missing or Unused Tau Fields

There is no automatic name-based pairing between a BC and a tau field. A missing lift
column, an unused tau variable, or the wrong number of scalar tau DOFs surfaces instead
as a **counting or shape failure**:

| mistake | what you actually see (in the `LBVP`s above) |
|---|---|
| BCs but no tau fields and no lift terms | `DimensionMismatch` during matrix construction (the BC rows have no tau columns to land in, so the system is not square) |
| a tau field declared but never lifted | `ArgumentError: Problem validation failed: Number of equations (3) is less than number of variables (4)` |
| a BC forgotten | `ArgumentError: Problem validation failed: Number of equations (2) is less than number of variables (3)` |

In an `IVP` the same two mistakes read `Number of equations (n) does not match number of
variables (m)` instead — that is the message the per-component `VectorField` BC above
trips.

**Solution**: create the required tau fields, include every one in the problem variables,
and reference every one through a lift term:

```julia
# Wrong: no tau fields, no lift terms — the BCs have nothing to act through
#     problem = LBVP([u])
#     add_equation!(problem, "Δ(u) = -2")
#     add_bc!(problem, "u(z=0) = 0")

# Correct: create tau fields, register lifted tau parameters, add them to the equation
u = ScalarField(dom, "u")
tau_u1 = ScalarField(dist, "tau_u1", (x_basis,), Float64)
tau_u2 = ScalarField(dist, "tau_u2", (x_basis,), Float64)

problem = LBVP([u, tau_u1, tau_u2])
lb2 = derivative_basis(z_basis, 2)
add_parameters!(problem; l1=lift(tau_u1, lb2, -1), l2=lift(tau_u2, lb2, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
```

### Over-Specified System

**Problem**: Too many boundary conditions cause singular matrices.

**Solution**: Match the number of BCs (and tau fields) to the operator order in each direction.

### Under-Specified System

**Error** (in an `LBVP`): `ArgumentError: Problem validation failed: Number of equations (4) is less than number of variables (5)`

**Solution**: Every variable needs an equation. Bulk equations and BCs both count,
so a tau field you declared but never lifted into an equation — or a BC you forgot
— shows up here. Count them: `n_equations + n_bcs == n_variables`.

### Tau Field Dimension Mismatch

**Problem**: Tau field has wrong dimensionality.

**Solution**: Tau fields should live on the *boundary*, not the full domain:

```julia
# For a 2D problem with x-basis (periodic) and z-basis (non-periodic):
# Boundaries are at constant z, so tau fields live on x-basis only
tau_u = ScalarField(dist, "tau_u", (x_basis,), Float64)  # Correct: 1D on x

# NOT:
tau_u = ScalarField(dist, "tau_u", (x_basis, z_basis), Float64)  # Wrong: 2D
```

## See Also

- [Tau Method (Advanced)](../pages/tau_method.md): Mathematical details
- [Problems API](../api/problems.md): Problem definition
- [Bases API](../api/bases.md): Spectral basis selection
- [2D Rayleigh-Bénard Tutorial](ivp_2d_rbc.md): Complete example with BCs
