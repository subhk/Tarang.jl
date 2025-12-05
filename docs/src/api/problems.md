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

**Examples**:

```julia
# Poisson equation
problem = LBVP([phi])
add_equation!(problem, "lap(phi) = rho")

# Stokes flow
problem = LBVP([u, v, p])
add_equation!(problem, "-nu*lap(u) + dx(p) = fx")
add_equation!(problem, "-nu*lap(v) + dz(p) = fz")
add_equation!(problem, "dx(u) + dz(v) = 0")
```

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

**Examples**:

```julia
# Steady Navier-Stokes
problem = NLBVP([u, v, p])
add_equation!(problem, "u*dx(u) + v*dz(u) = -dx(p) + nu*lap(u)")
add_equation!(problem, "u*dx(v) + v*dz(v) = -dz(p) + nu*lap(v)")
add_equation!(problem, "dx(u) + dz(v) = 0")
```

**Use cases**:
- Steady nonlinear flows
- Bifurcation analysis
- Multiple steady states

**Solution method**: Newton iteration or similar nonlinear solvers

---

### Eigenvalue Problems (EVP)

Linear eigenvalue problems for stability analysis and normal modes.

**Constructor**:
```julia
EVP(fields::Vector{<:AbstractField}; eigenvalue::Symbol)
```

**Arguments**:
- `fields`: Vector of fields to solve for
- `eigenvalue`: Symbol for eigenvalue (e.g., `:sigma`, `:lambda`, `:omega`)

**Examples**:

```julia
# Rayleigh-Bénard stability
problem = EVP([u, v, p, T], eigenvalue=:sigma)
add_equation!(problem, "sigma*u = -u0*dx(u) - v*dx(u0) - dx(p) + Pr*lap(u) + Pr*Ra*T")
add_equation!(problem, "sigma*v = -u0*dx(v) - v*dz(u0) - dz(p) + Pr*lap(v)")
add_equation!(problem, "sigma*T = -u0*dx(T) - v + lap(T)")
add_equation!(problem, "dx(u) + dz(v) = 0")
```

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
add_equation!(problem, "dt(T) = kappa*lap(T)")

# Wave equation
add_equation!(problem, "dt(dt(u)) = c^2*lap(u)")

# Poisson
add_equation!(problem, "lap(phi) = rho")
```

#### Complex Equations

```julia
# Navier-Stokes momentum
add_equation!(problem, "dt(u) + u*dx(u) + w*dz(u) = -dx(p) + nu*lap(u)")

# Energy equation with dissipation
add_equation!(problem, "dt(T) + u*dx(T) + w*dz(T) = kappa*lap(T) + Q")

# With parameters
add_equation!(problem, "dt(T) = kappa*lap(T) + Ra*Pr*w")
```

#### Using Fields and Parameters

```julia
# Fields are referenced by name
add_equation!(problem, "dt(u) = -u*dx(u)")  # u is a field

# Parameters from problem.parameters
problem.parameters["nu"] = 0.01
problem.parameters["Ra"] = 1e6
add_equation!(problem, "dt(u) = nu*lap(u) + Ra*T")
```

---

## Parameters

### Setting Parameters

```julia
# Create problem
problem = IVP([u, p, T])

# Set dimensionless parameters
problem.parameters["Re"] = 1000.0      # Reynolds number
problem.parameters["Pr"] = 0.7         # Prandtl number
problem.parameters["Ra"] = 1e6         # Rayleigh number

# Set physical parameters
problem.parameters["nu"] = 1e-3        # Kinematic viscosity
problem.parameters["kappa"] = 1e-3     # Thermal diffusivity
problem.parameters["g"] = 9.81         # Gravitational acceleration
```

### Using Parameters in Equations

```julia
# Reference by name in equations
add_equation!(problem, "dt(u) = -u*dx(u) + nu*lap(u)")
add_equation!(problem, "dt(T) = -u*dx(T) + kappa*lap(T)")

# Dimensionless formulation
add_equation!(problem, "dt(u) = -u*dx(u) + (1/Re)*lap(u)")
add_equation!(problem, "dt(T) = -u*dx(T) + (1/(Re*Pr))*lap(T) + Ra*Pr*w")
```

### Modifying Parameters

```julia
# Change parameter value
problem.parameters["Ra"] = 1e7

# Access parameter
Ra = problem.parameters["Ra"]

# Iterate over parameters
for (name, value) in problem.parameters
    println("$name = $value")
end
```

---

## Boundary Conditions

### add_dirichlet_bc!

Add Dirichlet (fixed value) boundary condition.

**Syntax**:
```julia
add_dirichlet_bc!(problem, field_name, coord_name, position, value)
```

**Arguments**:
- `problem`: Problem object
- `field_name`: Name of field (String)
- `coord_name`: Coordinate direction (String)
- `position`: Location (Float64)
- `value`: Boundary value (Float64, String expression, or Function)

**Examples**:

```julia
# Constant value
add_dirichlet_bc!(problem, "T", "z", 0.0, 1.0)  # T(z=0) = 1
add_dirichlet_bc!(problem, "T", "z", 1.0, 0.0)  # T(z=1) = 0

# No-slip velocity
add_dirichlet_bc!(problem, "u", "z", 0.0, 0.0)  # u(z=0) = 0

# Time-dependent
add_dirichlet_bc!(problem, "u", "x", 0.0, "sin(omega*t)")

# Space-dependent
add_dirichlet_bc!(problem, "T", "z", 0.0, "1.0 - x^2")

# Function
bc_func(x, y, t) = sin(2π*x) * cos(omega*t)
add_dirichlet_bc!(problem, "u", "z", 0.0, bc_func)
```

---

### add_neumann_bc!

Add Neumann (fixed derivative) boundary condition.

**Syntax**:
```julia
add_neumann_bc!(problem, field_name, coord_name, position, value)
```

**Arguments**: Same as Dirichlet

**Examples**:

```julia
# Insulating boundary (zero heat flux)
add_neumann_bc!(problem, "T", "z", 0.0, 0.0)  # dT/dz(z=0) = 0

# Fixed flux
add_neumann_bc!(problem, "T", "z", 1.0, -1.0)  # dT/dz(z=1) = -1

# Time-dependent flux
add_neumann_bc!(problem, "phi", "x", 0.0, "sin(omega*t)")
```

---

### add_robin_bc!

Add Robin (mixed) boundary condition: α*f + β*df/dn = value

**Syntax**:
```julia
add_robin_bc!(problem, field_name, coord_name, position, alpha, beta, value)
```

**Arguments**:
- `alpha`: Coefficient for field value
- `beta`: Coefficient for derivative
- `value`: Right-hand side value

**Examples**:

```julia
# Convective heat transfer: h*T + k*dT/dn = h*T_ambient
h, k = 10.0, 1.0
T_ambient = 300.0
add_robin_bc!(problem, "T", "z", 1.0, h, k, h*T_ambient)

# Radiation boundary: T + ε*dT/dn = 0
add_robin_bc!(problem, "T", "x", 0.0, 1.0, epsilon, 0.0)
```

---

### add_stress_free_bc!

Add stress-free boundary condition for fluid mechanics: u_normal = 0, ∂u_tangential/∂n = 0

**Syntax**:
```julia
add_stress_free_bc!(problem, field_name, coord_name, position)
```

**Examples**:

```julia
# Stress-free top and bottom (free-slip)
add_stress_free_bc!(problem, "u", "z", 0.0)
add_stress_free_bc!(problem, "u", "z", 1.0)

# Equivalent to:
# w(z=0) = 0, du/dz(z=0) = 0, dv/dz(z=0) = 0
# w(z=1) = 0, du/dz(z=1) = 0, dv/dz(z=1) = 0
```

---

### add_bc!

Add custom boundary condition using equation syntax.

**Syntax**:
```julia
add_bc!(problem, equation_string::String)
```

**Examples**:

```julia
# Custom combinations
add_bc!(problem, "u(z=0) + 2*dz(u)(z=0) = 1")

# Coupling between fields
add_bc!(problem, "T(x=0) = 2*S(x=0)")

# Complex expressions
add_bc!(problem, "dx(p)(z=0) + omega^2*u(z=0) = 0")
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
add_equation!(problem, "dt(u) = -u*dx(u) - v*dz(u) - dx(p) + nu*lap(u)")
add_equation!(problem, "dt(v) = -u*dx(v) - v*dz(v) - dz(p) + nu*lap(v)")
add_equation!(problem, "dx(u) + dz(v) = 0")

# Add boundary conditions
add_dirichlet_bc!(problem, "u", "z", 0.0, 0.0)
add_dirichlet_bc!(problem, "v", "z", 0.0, 0.0)
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
add_substitution!(problem, "omega", "dx(v) - dz(u)")

# Use in equations
add_equation!(problem, "dt(omega) = -u*dx(omega) - v*dz(omega) + nu*lap(omega)")
```

### Common Substitutions

```julia
# Vorticity
add_substitution!(problem, "omega", "dx(v) - dz(u)")

# Kinetic energy
add_substitution!(problem, "KE", "0.5*(u^2 + v^2 + w^2)")

# Strain rate
add_substitution!(problem, "S11", "dx(u)")
add_substitution!(problem, "S12", "0.5*(dx(v) + dz(u))")
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
problem.parameters["Ra"] = 1e6
problem.parameters["Pr"] = 1.0

# Equations
add_equation!(problem, "dt(u) + u*dx(u) + w*dz(u) + dx(p) = Pr*lap(u)")
add_equation!(problem, "dt(w) + u*dx(w) + w*dz(w) + dz(p) = Pr*lap(w) + Ra*Pr*T")
add_equation!(problem, "dx(u) + dz(w) = 0")
add_equation!(problem, "dt(T) + u*dx(T) + w*dz(T) = lap(T)")

# Boundary conditions
add_dirichlet_bc!(problem, "u", "z", 0.0, 0.0)
add_dirichlet_bc!(problem, "w", "z", 0.0, 0.0)
add_dirichlet_bc!(problem, "T", "z", 0.0, 1.0)

add_dirichlet_bc!(problem, "u", "z", 1.0, 0.0)
add_dirichlet_bc!(problem, "w", "z", 1.0, 0.0)
add_dirichlet_bc!(problem, "T", "z", 1.0, 0.0)

# Validate
validate_problem(problem)
```

---

## See Also

- [Solvers](solvers.md): Solving problems with different algorithms
- [Operators](operators.md): Mathematical operators for equations
- [Fields](fields.md): Field types used in problems
- [Timesteppers](timesteppers.md): Time integration schemes
