# Problems API

Problems define the PDE systems to be solved, including equations, parameters, and boundary conditions. Tarang.jl supports Initial Value Problems (IVP), Boundary Value Problems (LBVP/NLBVP), and Eigenvalue Problems (EVP).

## Problem Types

### Initial Value Problems (IVP)

Time-evolution problems where PDEs are integrated forward in time from initial conditions.

**Constructor**:
```julia
IVP(fields::Vector{<:AbstractField})
```

**Arguments**:
- `fields`: Vector of fields to solve for

**Examples**:

```julia
# 2D Navier-Stokes
problem = IVP([ux, uz, p])

# With additional scalars
problem = IVP([ux, uz, p, T, S])
```

**Use cases**:
- Fluid dynamics (Navier-Stokes)
- Heat/mass diffusion
- Wave propagation
- Reaction-diffusion systems
- Turbulence simulations

---

### Linear Boundary Value Problems (LBVP)

Steady-state linear problems with boundary conditions.

**Constructor**:
```julia
LBVP(fields::Vector{<:AbstractField})
```

**Arguments**:
- `fields`: Vector of fields to solve for

**Boundary conditions (tau method)**: a bounded (Chebyshev/Jacobi) direction
needs explicit `tau` variables, one per boundary condition, lifted into the bulk
equation via `lift(tau, derivative_basis(basis, 2), -k)`. Boundary conditions are
declared with `add_bc!` (not `add_equation!`). The solver builds one square tau
subproblem per separable Fourier mode.

**Example** (2D Poisson `Δu = -2`, `u = 0` on both `z` walls; solution
`u = z(Lz - z)`, verified to machine precision):

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 2π))   # periodic (separable)
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))  # bounded  (coupled)
dom = Domain(dist, (xb, zb))

u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)           # one tau per BC
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = LBVP([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
ensure_layout!(u, :g)            # scatter writes coefficients; switch to grid
```

!!! note "1D pure-Chebyshev BVP"
    The per-mode solve is verified for domains with at least one separable
    (Fourier) direction. A pure single-axis Chebyshev BVP currently mis-scatters
    the solution; add a (size-1 or larger) Fourier direction, or use the EVP path
    for 1D spectra.

**Use cases**:
- Steady heat conduction
- Poisson/Laplace equations
- Stokes flow
- Electrostatics

---

### Nonlinear Boundary Value Problems (NLBVP)

Steady-state nonlinear problems with boundary conditions.

**Constructor**:
```julia
NLBVP(fields::Vector{<:AbstractField})
```

**Arguments**:
- `fields`: Vector of fields to solve for

Same tau-method boundary handling as the LBVP (tau variables + `lift` + `add_bc!`).
The nonlinear terms go on the right-hand side; the solver linearizes them with a
symbolic Frechet derivative and runs a per-Fourier-mode Newton iteration (the
Jacobian `dF = ∂(LHS − RHS)/∂u` is rebuilt each iteration with the current state).

**Example** (manufactured quadratic nonlinearity `Δu = u² + g`, with
`g = -2 - u_exact²` so `u_exact = z(Lz - z)`; verified to ~1e-12):

```julia
# domain / fields / taus as in the LBVP example above (u, tau1, tau2, lb2)
g = ScalarField(dom, "g"); ensure_layout!(g, :g)
zg = create_meshgrid(dom; on_device=false)["z"]
get_grid_data(g) .= -2 .- (zg .* (1.0 .- zg)).^2

problem = NLBVP([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2), g=g)
add_equation!(problem, "Δ(u) + l1 + l2 = u*u + g")   # nonlinearity on the RHS
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solver.tolerance = 1e-10
ensure_layout!(u, :g); get_grid_data(u) .= 0.0       # initial guess
solve!(solver)
ensure_layout!(u, :g)
```

**Use cases**:
- Steady nonlinear flows
- Bifurcation analysis
- Multiple steady states

**Solution method**: per-Fourier-mode Newton iteration with a symbolic (Frechet)
Jacobian.

---

### Eigenvalue Problems (EVP)

Linear eigenvalue problems for stability analysis and normal modes.

**Constructor**:
```julia
EVP(fields::Vector{<:AbstractField}; eigenvalue::Symbol)
```

**Arguments**:
- `fields`: Vector of fields to solve for (plus the `tau` variables for BCs)
- `eigenvalue`: Symbol naming the eigenvalue (declarative; e.g. `:σ`, `:λ`)

**Eigenvalue convention**: the generalized problem is `L x = λ M x`, where the
mass matrix `M` is assembled from the **time-derivative terms** `dt(·)`. The
eigenvalue *replaces* the time derivative (`dt(u) → λ u`), so a normal-mode
ansatz `u ~ e^{λ t}` is written by keeping the `dt(u)` term in the equation. Do
**not** multiply the eigenvalue symbol into the equation (`σ*u = …` builds an
empty `M` and returns no eigenvalues). Boundary conditions use the same tau
method as the BVP (`tau` vars + `lift` + `add_bc!`).

**Example** (1D diffusion eigenproblem `λu = Δu`, Dirichlet; eigenvalues are the
Dirichlet-Laplacian values `λ_n = -(nπ/Lz)²`, verified to ~1e-13):

```julia
coords = CartesianCoordinates("z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
zb     = ChebyshevT(coords["z"]; size=32, bounds=(0.0, 1.0))
dom    = Domain(dist, (zb,))

u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (), Float64)
tau2 = ScalarField(dist, "tau2", (), Float64)
lb2  = derivative_basis(zb, 2)

problem = EVP([u, tau1, tau2]; eigenvalue=:σ)
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "dt(u) - Δ(u) - l1 - l2 = 0")   # dt(u) → λu marks M
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = EigenvalueSolver(problem; nev=5, which=:SM)   # 5 smallest-magnitude
eigenvalues, eigenvectors = solve!(solver)
# |eigenvalues| ≈ (nπ)² = 9.87, 39.48, 88.83, ...
```

The solver solves `eigen(L, M)` per Fourier mode on the square tau matrices and
filters spurious eigenvalues from the singular mass matrix (the BC/tau rows).
`which` accepts `:LM` `:SM` `:LR` `:SR` `:LI` `:SI`, or pass `target=…` to order
by proximity to a shift.

**Use cases**:
- Linear stability analysis
- Growth rates and frequencies
- Normal mode decomposition
- Critical parameter values

---

## Adding Equations

### add_equation!

Add a PDE to the problem.

**Syntax**:
```julia
add_equation!(problem, equation_string::String)
```

**Arguments**:
- `problem`: Problem object (IVP, LBVP, NLBVP, or EVP)
- `equation_string`: String equation using symbolic syntax

**Examples**:

#### Simple Equations

```julia
# Diffusion
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")

# Wave equation
add_equation!(problem, "∂t(∂t(u)) - c^2*Δ(u) = 0")

# Poisson
add_equation!(problem, "Δ(phi) = rho")
```

#### Complex Equations

```julia
# Navier-Stokes momentum
add_equation!(problem, "∂t(u) + ∂x(p) - nu*Δ(u) = -u*∂x(u) - w*∂z(u)")

# Energy equation with dissipation
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u*∂x(T) - w*∂z(T) + Q")

# With parameters
add_equation!(problem, "∂t(T) - kappa*Δ(T) = Ra*Pr*w")
```

#### Using Fields and Parameters

```julia
# Fields are referenced by name
add_equation!(problem, "∂t(u) = -u*∂x(u)")  # u is a field

# Parameters from problem.namespace
problem.namespace["nu"] = 0.01
problem.namespace["Ra"] = 1e6
add_equation!(problem, "∂t(u) - nu*Δ(u) = Ra*T")
```

---

## Parameters

### Setting Parameters

```julia
# Create problem
problem = IVP([u, p, T])

# Set dimensionless parameters
problem.namespace["Re"] = 1000.0      # Reynolds number
problem.namespace["Pr"] = 0.7         # Prandtl number
problem.namespace["Ra"] = 1e6         # Rayleigh number

# Set physical parameters
problem.namespace["nu"] = 1e-3        # Kinematic viscosity
problem.namespace["kappa"] = 1e-3     # Thermal diffusivity
problem.namespace["g"] = 9.81         # Gravitational acceleration
```

### Using Parameters in Equations

```julia
# Reference by name in equations
add_equation!(problem, "∂t(u) - nu*Δ(u) = -u*∂x(u)")
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u*∂x(T)")

# Dimensionless formulation
add_equation!(problem, "∂t(u) - (1/Re)*Δ(u) = -u*∂x(u)")
add_equation!(problem, "∂t(T) - (1/(Re*Pr))*Δ(T) = -u*∂x(T) + Ra*Pr*w")
```

### Modifying Parameters

```julia
# Change parameter value
problem.namespace["Ra"] = 1e7

# Access parameter
Ra = problem.namespace["Ra"]

# Iterate over parameters
for (name, value) in problem.namespace
    println("$name = $value")
end
```

---

## Boundary Conditions

Tarang has a dedicated `add_bc!` function for boundary conditions. It's the preferred API — use it instead of `add_equation!` for anything that's `field(coord=value) = ...`.

### `add_bc!`

```julia
add_bc!(problem::Problem, bc::String)
```

Adds a boundary condition to a problem. The BC string must match one of these forms:

- **Dirichlet**: `"field(coord=position) = value"` (e.g. `"T(z=0) = 1"`)
- **Neumann**: `"∂coord(field)(coord=position) = value"` (e.g. `"∂z(T)(z=0) = 0"`)
- **Integral constraint**: `"integ(field) = value"` (e.g. `"integ(p) = 0"` as a pressure gauge)

`add_bc!` does two things:

1. Pushes the raw string into `problem.boundary_conditions` (so that `_merge_boundary_conditions!` adds it as an equation at solver-build time for the parser to see).
2. Parses the string via `parse_bc_string` / `parse_neumann_bc_string` and registers a concrete `DirichletBC` or `NeumannBC` object in `problem.bc_manager.conditions`, with `is_time_dependent` / `is_space_dependent` flags auto-detected from the value expression. This second step is what enables time- and space-dependent BC refresh — without it, the BC is treated as a plain algebraic constraint with no runtime value updates.

Integral constraints like `"integ(p) = 0"` don't match the `field(coord=pos)` pattern; they flow through as raw strings to the equation parser and are handled by valid-mode filtering in `subsystems.jl`.

### Dirichlet (fixed value)

```julia
# Constant value
add_bc!(problem, "T(z=0) = 1")
add_bc!(problem, "T(z=Lz) = 0")

# No-slip velocity (vector BC — applies to all components)
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=Lz) = 0")
```

### Time-dependent BC

```julia
add_bc!(problem, "T(z=0) = 1.0 + 0.1*cos(2*pi*t)")
```

The BC value is re-evaluated at every RK stage time `t + c[i]·dt`, so multi-stage methods retain full formal order of accuracy. No extra configuration required — `t` is the default time variable and is recognized automatically.

### Space-dependent BC

```julia
add_bc!(problem, "T(z=0) = 1.0 + 0.1*sin(2*pi*x/Lx)")
```

The BC expression is evaluated against the global `x` coordinate grid at solver-build time, producing a 1D array. At each stepper call, the array is projected onto each subproblem's Fourier mode via an unnormalized RFFT. Coordinate names (`x`, `y`, `z`, ...) are auto-registered from the problem's bases — no `add_coordinate_field!` call needed.

For user parameters in the expression, register them via `add_parameters!`:

```julia
add_parameters!(problem, Lx=4.0, amplitude=0.1)
add_bc!(problem, "T(z=0) = 1.0 + amplitude*sin(2*pi*x/Lx)")
```

Or bake the numeric value into the string via Julia interpolation:

```julia
Lx = 4.0
add_bc!(problem, "T(z=0) = 1.0 + 0.1*sin(2*pi*x/$Lx)")
```

### Space + time dependent BC

```julia
add_bc!(problem, "T(z=0) = 1.0 + 0.1*sin(2*pi*x/Lx)*cos(2*pi*t)")
```

The spatial pattern is projected onto Fourier modes and re-evaluated at each stage time, combining both refresh paths.

### Neumann (fixed derivative)

```julia
# Insulating boundary
add_bc!(problem, "∂z(T)(z=0) = 0")

# Fixed flux
add_bc!(problem, "∂z(T)(z=Lz) = -1")
```

Neumann BCs detect space/time dependency in exactly the same way as Dirichlet. The `∂coord(field)` prefix is what tells `_register_string_bc!` to route through `parse_neumann_bc_string`.

### Pressure gauge

For incompressible flow, pressure is defined only up to a constant. Fix the gauge with an integral constraint:

```julia
tau_p = ScalarField(dist, "tau_p", (), Float64)
# ... add to problem.variables ...
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_bc!(problem, "integ(p) = 0")
```

The `integ(p) = 0` constraint is trivially zero at non-DC Fourier modes and is filtered by valid-mode filtering in the subproblem builder — it's only active at the DC mode, where it fixes the pressure gauge.

### Legacy: `add_equation!` for BCs

`add_equation!(problem, "T(z=0) = 1")` still works — the equation parser detects the `field(coord=value)` syntax and converts it internally. However, this path **does not** register the BC in `bc_manager.conditions`, so time- and space-dependent BC refresh is disabled. For constant BCs it's equivalent to `add_bc!`; for anything else, prefer `add_bc!`.

---

### Neumann (Fixed Derivative)

```julia
# Insulating boundary (zero heat flux)
add_equation!(problem, "∂z(T)(z=0) = 0")  # ∂T/∂z(z=0) = 0

# Fixed flux
add_equation!(problem, "∂z(T)(z=1) = -1")  # ∂T/∂z(z=1) = -1

# Time-dependent flux
add_equation!(problem, "∂x(phi)(x=0) = sin(omega*t)")
```

---

### Robin (Mixed)

Robin boundary condition: α*f + β*df/dn = value

```julia
# Convective heat transfer: h*T + k*∂T/∂n = h*T_ambient
h, k = 10.0, 1.0
T_ambient = 300.0
add_equation!(problem, "$(h)*T(z=1) + $(k)*∂z(T)(z=1) = $(h*T_ambient)")

# Radiation boundary: T + ε*∂T/∂n = 0
add_equation!(problem, "1.0*T(x=0) + $(epsilon)*∂x(T)(x=0) = 0")
```

---

### Stress-Free

Stress-free boundary condition for fluid mechanics: u_normal = 0, ∂u_tangential/∂n = 0

```julia
# Stress-free top and bottom (free-slip)
add_equation!(problem, "w(z=0) = 0")
add_equation!(problem, "∂z(u)(z=0) = 0")
add_equation!(problem, "∂z(v)(z=0) = 0")
add_equation!(problem, "w(z=1) = 0")
add_equation!(problem, "∂z(u)(z=1) = 0")
add_equation!(problem, "∂z(v)(z=1) = 0")
```

---

### No-Slip

No-slip boundary condition for fluid mechanics: all velocity components = 0 at the boundary.

```julia
# No-slip at walls
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "v(z=0) = 0")
add_equation!(problem, "w(z=0) = 0")
add_equation!(problem, "u(z=1) = 0")
add_equation!(problem, "v(z=1) = 0")
add_equation!(problem, "w(z=1) = 0")
```

---

### Custom Boundary Conditions

Use `add_equation!` for any custom boundary condition:

```julia
# Custom combinations
add_equation!(problem, "u(z=0) + 2*∂z(u)(z=0) = 1")

# Coupling between fields
add_equation!(problem, "T(x=0) = 2*S(x=0)")

# Complex expressions
add_equation!(problem, "∂x(p)(z=0) + omega^2*u(z=0) = 0")
```

---

## Problem Properties

### Accessing Fields

```julia
# Get list of fields
fields = problem.fields

# Iterate over fields
for field in problem.fields
    println(field.name)
end

# Find field by name
u = find_field(problem, "u")
```

### Accessing Equations

```julia
# Get equations
equations = problem.equations

# Number of equations
n_eqs = length(problem.equations)

# Print equations
for (i, eq) in enumerate(problem.equations)
    println("Equation $i: $eq")
end
```

### Accessing Boundary Conditions

```julia
# Get boundary conditions
bcs = problem.boundary_conditions

# Number of BCs
n_bcs = length(problem.boundary_conditions)

# Print BCs
for bc in problem.boundary_conditions
    println("$(bc.field) at $(bc.coord)=$(bc.position): $(bc.type)")
end
```

---

## Problem Validation

### validate_problem

Check problem for consistency and completeness.

**Syntax**:
```julia
validate_problem(problem)
```

**Checks**:
- Number of equations matches number of unknowns
- Boundary conditions are sufficient
- Parameters are defined
- Field dimensions match
- Operator applications are valid

**Example**:

```julia
problem = IVP([u, v, p])
add_equation!(problem, "∂t(u) + ∂x(p) - nu*Δ(u) = -u*∂x(u) - v*∂z(u)")
add_equation!(problem, "∂t(v) + ∂z(p) - nu*Δ(v) = -u*∂x(v) - v*∂z(v)")
add_equation!(problem, "∂x(u) + ∂z(v) = 0")

# Add boundary conditions
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "v(z=0) = 0")
# ... more BCs

# Validate
validate_problem(problem)  # Throws error if invalid
```

---

## Problem Substitutions

### Substitution Variables

Define intermediate variables for readability:

```julia
# Define substitution
add_parameters!(problem, omega="∂x(v) - ∂z(u)")

# Use in equations
add_equation!(problem, "∂t(omega) - nu*Δ(omega) = -u*∂x(omega) - v*∂z(omega)")
```

### Common Substitutions

```julia
# Vorticity
add_parameters!(problem, omega="∂x(v) - ∂z(u)")

# Kinetic energy
add_parameters!(problem, KE="0.5*(u^2 + v^2 + w^2)")

# Strain rate
add_parameters!(problem, S11="∂x(u)", S12="0.5*(∂x(v) + ∂z(u))")
```

---

## Advanced Features

### Tau Method for Boundary Conditions

Tarang.jl uses the tau method for enforcing boundary conditions in spectral methods:

```julia
# Tau fields are created automatically
# Lift operations handle boundary condition enforcement

# User typically doesn't interact with tau fields directly
# But they appear in the linear operator for non-periodic bases
```

### Constraints

Add algebraic constraints to the system:

```julia
# Mean value constraint
add_constraint!(problem, "mean(p) = 0")

# Integral constraint
add_constraint!(problem, "integral(T) = 1.0")
```

---

## Complete Examples

### 2D Rayleigh-Bénard Convection

```julia
# Create fields
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

domain = Domain(dist, (x_basis, z_basis))

u = VectorField(dist, coords, "u", (x_basis, z_basis))
p = ScalarField(dist, "p", (x_basis, z_basis))
T = ScalarField(dist, "T", (x_basis, z_basis))

# Create problem
problem = IVP([u.components[1], u.components[2], p, T])

# Parameters
problem.namespace["Ra"] = 1e6
problem.namespace["Pr"] = 1.0

# Equations
add_equation!(problem, "∂t(u) + ∂x(p) - Pr*Δ(u) = -u*∂x(u) - w*∂z(u)")
add_equation!(problem, "∂t(w) + ∂z(p) - Pr*Δ(w) - Ra*Pr*T = -u*∂x(w) - w*∂z(w)")
add_equation!(problem, "∂x(u) + ∂z(w) = 0")
add_equation!(problem, "∂t(T) - Δ(T) = -u*∂x(T) - w*∂z(T)")

# Boundary conditions
add_equation!(problem, "u(z=0) = 0")
add_equation!(problem, "w(z=0) = 0")
add_equation!(problem, "T(z=0) = 1")

add_equation!(problem, "u(z=1) = 0")
add_equation!(problem, "w(z=1) = 0")
add_equation!(problem, "T(z=1) = 0")

# Validate
validate_problem(problem)
```

---

## See Also

- [Solvers](solvers.md): Solving problems with different algorithms
- [Operators](operators.md): Mathematical operators for equations
- [Fields](fields.md): Field types used in problems
- [Timesteppers](timesteppers.md): Time integration schemes
