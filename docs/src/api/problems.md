# Problems API

Problems define the PDE systems to be solved, including equations, parameters, and boundary conditions. Tarang.jl supports Initial Value Problems (IVP), Boundary Value Problems (LBVP/NLBVP), and Eigenvalue Problems (EVP).

## Problem Types

### Initial Value Problems (IVP)

Time-evolution problems where PDEs are integrated forward in time from initial conditions.

**Constructor**:
```julia
IVP(variables::Vector{<:Operand})
```

**Arguments**:
- `variables`: Vector of the unknowns — `ScalarField`s, `VectorField`s, and any `tau` variables required by the boundary conditions

**Example** (2D advection-diffusion on a doubly-periodic domain):

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
zb = RealFourier(coords["z"]; size=16, bounds=(0.0, 2π))
dom = Domain(dist, (xb, zb))

T = ScalarField(dom, "T")
u = VectorField(dom, "u")

problem = IVP([T, u])
add_parameters!(problem, kappa=0.05, nu=0.05)
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")

set!(T, (x, z) -> sin(x) * cos(z))
set!(u.components[1], (x, z) -> 0.5)

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
run!(solver; stop_iteration=20, progress=false)
```

**Use cases**:
- Fluid dynamics (Navier-Stokes)
- Heat/mass diffusion
- Reaction-diffusion systems
- Turbulence simulations

---

### Linear Boundary Value Problems (LBVP)

Steady-state linear problems with boundary conditions.

**Constructor**:
```julia
LBVP(variables::Vector{<:Operand})
```

**Arguments**:
- `variables`: Vector of unknowns (fields plus the `tau` variables)

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
    A pure single-axis Chebyshev BVP (no Fourier direction) works too — just drop
    the `x` axis and put the `tau` variables on `()`. The solver builds a single
    coupled tau subproblem over the Chebyshev spectrum.

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
NLBVP(variables::Vector{<:Operand})
```

**Arguments**:
- `variables`: Vector of unknowns (fields plus the `tau` variables)

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
EVP(variables::Vector{<:Operand}; eigenvalue::Symbol)
```

**Arguments**:
- `variables`: Vector of unknowns (plus the `tau` variables for BCs)
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
# |eigenvalues| ≈ (nπ)² = 9.8696, 39.4784, 88.8264, ...
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

The parser reads the string against the problem namespace: the problem's own
variables, everything registered with [`add_parameters!`](#Parameters), the
coordinate names taken from the bases, the time variable `t`, and the operator
registry (`Δ`/`lap`, `∇`/`grad`, `div`, `curl`, `∂x`/`∂z`/…, `∂t`/`dt`, `lift`,
`trace`, `integ`, …). A name that is none of these is not an error: the parser
warns `Unknown variable: …` and substitutes an opaque placeholder, so watch for
that warning.

**Split convention**: linear terms that should be treated implicitly go on the
**left**; the explicit (nonlinear, forcing) terms go on the **right**. Putting a
linear operator on the RHS is legal but triggers a warning, because IMEX
timesteppers only apply the implicit solve to the LHS.

**Examples**:

#### Simple equations

```julia
# Diffusion (IVP)
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")

# Poisson (LBVP; `l1`, `l2` are the lifted tau terms)
add_equation!(problem, "Δ(phi) + l1 + l2 = rho")
```

!!! warning "Only first-order time derivatives"
    `∂t(u)` is the only time derivative the mass matrix understands. A
    second-order form such as `"∂t(∂t(u)) - c^2*Δ(u) = 0"` is *not* rejected —
    it is silently integrated as if it were first order (measured: it decays
    like the heat equation instead of oscillating). Reduce the equation to a
    first-order system instead:

    ```julia
    problem = IVP([u, v])
    add_parameters!(problem, c2=1.0)
    add_equation!(problem, "∂t(u) - v = 0")
    add_equation!(problem, "∂t(v) - c2*Δ(u) = 0")
    ```

    (RUN: 1D, 32 modes, RK222, dt=1e-3 to t=1 — matches `cos(x)cos(t)` to 3e-8.)

#### Coupled equations

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb  = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
zb  = RealFourier(coords["z"]; size=16, bounds=(0.0, 2π))
dom = Domain(dist, (xb, zb))

u = ScalarField(dom, "u")   # x-velocity
w = ScalarField(dom, "w")   # z-velocity
T = ScalarField(dom, "T")

problem = IVP([u, w, T])
add_parameters!(problem, nu=0.01, kappa=0.01, Ra=100.0, Pr=1.0)

# Momentum: buoyancy is linear in T, so it belongs on the LHS with the diffusion
add_equation!(problem, "∂t(u) - nu*Δ(u) = -u*∂x(u) - w*∂z(u)")
add_equation!(problem, "∂t(w) - nu*Δ(w) - Ra*Pr*T = -u*∂x(w) - w*∂z(w)")
# Energy
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u*∂x(T) - w*∂z(T)")

solver = InitialValueSolver(problem, RK222(); dt=1e-4)
```

Vector notation works too, and is preferred when the unknown is a `VectorField`
— one equation then contributes one row block per component:

```julia
T = ScalarField(dom, "T")
u = VectorField(dom, "u")

problem = IVP([T, u])
add_parameters!(problem, nu=0.01, kappa=0.01)
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*Δ(u) = -u⋅∇(u)")
```

The incompressible form (`∇(p)` in the momentum equation plus `div(u) = 0`) needs
the tau machinery and a pressure gauge — see the
[Rayleigh-Bénard example](#2D-Rayleigh-Bénard-convection) at the end of this page.

---

## Parameters

### Setting parameters

Use `add_parameters!`. Every name registered this way becomes visible to the
equation parser, and to *constant* boundary-condition values — but **not** to
space- or time-dependent BC values (see
[Boundary Conditions](#Boundary-Conditions) below).

```julia
problem = IVP([u, w, T])

# Dimensionless numbers
add_parameters!(problem, Re=1000.0, Pr=0.7, Ra=1e6)

# Physical parameters
add_parameters!(problem, nu=1e-3, kappa=1e-3, g=9.81)
```

Values may be numbers, fields, operator expressions, or plain Julia functions —
anything the parser can splice into the expression tree. (They are stored in
`problem.namespace`, which is a `Dict{String, Any}`; `problem.namespace["nu"] =
1e-3` is equivalent to `add_parameters!(problem, nu=1e-3)`, but `add_parameters!`
is the supported spelling.)

### Using parameters in equations

```julia
add_equation!(problem, "∂t(u) - nu*Δ(u) = -u*∂x(u)")
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u*∂x(T)")

# Dimensionless formulation
add_equation!(problem, "∂t(u) - (1/Re)*Δ(u) = -u*∂x(u)")
add_equation!(problem, "∂t(T) - (1/(Re*Pr))*Δ(T) - Ra*Pr*w = -u*∂x(T)")
```

### Accessing and modifying parameters

Parameters are read back from the namespace, and re-registering a name overwrites
it. Values are baked into the matrices when the solver is built, so change them
*before* constructing the solver.

```julia
Ra = problem.namespace["Ra"]          # read
add_parameters!(problem, Ra=1e7)      # overwrite (before InitialValueSolver)
```

!!! note
    `problem.namespace` is not a list of your parameters: it is pre-populated
    with the operator registry, the coordinate names, and the time variable
    (~60 entries before you add anything). Iterating it prints all of that. The
    `problem.parameters` field is a separate, unused dict — it stays empty.

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
add_bc!(problem, "T(z=1) = 0")

# No-slip velocity (vector BC — applies to all components)
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
```

A **constant** BC value may be a literal, or a name registered with
`add_parameters!`. A bare Julia global is **not** visible to the parser — it warns
`Unknown variable: Tbot` and silently applies `0`:

```julia
add_parameters!(problem, Tbot=3.0)
add_bc!(problem, "T(z=0) = Tbot")     # 3.0, exactly
```

### Time-dependent BC

```julia
add_bc!(problem, "T(z=0) = 1.0 + 0.1*cos(2*pi*t)")
```

The BC value is re-evaluated at every RK stage time `t + c[i]·dt`, so multi-stage methods retain full formal order of accuracy. No extra configuration required — `t` is the default time variable and is recognized automatically.

### Space-dependent BC

```julia
add_bc!(problem, "T(z=0) = 1.0 + 0.1*sin(2*pi*x/4)")
```

The BC expression is evaluated against the global `x` coordinate grid at solver-build time, producing a 1D array. At each stepper call, the array is projected onto each subproblem's Fourier mode via an unnormalized RFFT. Coordinate names (`x`, `y`, `z`, ...) are auto-registered from the problem's bases — no `add_coordinate_field!` call needed.

### Space + time dependent BC

```julia
add_bc!(problem, "T(z=0) = 1.0 + 0.1*sin(2*pi*x/4)*cos(2*pi*t)")
```

The spatial pattern is projected onto Fourier modes and re-evaluated at each stage time, combining both refresh paths.

!!! warning "Non-constant BC values take literals only — not parameter names"
    The refresh evaluator for space- and time-dependent BC values has its own
    scope: `t`, `pi`, `e`, and the coordinate grids. Names you registered with
    `add_parameters!` are **not** in it. `add_parameters!(problem, Lx=4.0)` +
    `"T(z=0) = 1.0 + 0.1*sin(2*pi*x/Lx)"` warns
    *"BC expression evaluation failed … Unknown variable: Lx"* and the boundary
    value comes out wrong (measured error 1.1 on an O(1) BC). Bake the number in
    with a literal or with Julia string interpolation:

    ```julia
    Lx = 4.0
    add_bc!(problem, "T(z=0) = 1.0 + 0.1*sin(2*pi*x/$Lx)*cos(2*pi*t)")   # exact
    ```

    (RUN: literal and `$`-interpolated forms enforce the BC to 1e-16; the
    parameter-name form is off by 1.1.)

### Neumann (fixed derivative)

```julia
# Insulating boundary
add_bc!(problem, "∂z(T)(z=0) = 0")

# Fixed flux
add_bc!(problem, "∂z(T)(z=1) = -1")

# Time-dependent flux
add_bc!(problem, "∂z(T)(z=1) = sin(2*pi*t)")
```

Neumann BCs detect space/time dependency in exactly the same way as Dirichlet. The `∂coord(field)` prefix is what tells `_register_string_bc!` to route through `parse_neumann_bc_string`.

### Robin (mixed)

Robin conditions `α·f + β·∂f/∂n = value` are written as a single BC string mixing
the interpolated field and its interpolated derivative:

```julia
add_bc!(problem, "u(z=1.0) + 2*∂z(u)(z=1.0) = -2")
```

(RUN: `Δu = -2` on `z ∈ [0,1]` with `u(z=0) = 0` and the Robin condition above
recovers `u = z(1-z)` to 8e-17.)

A Robin string does not match the Dirichlet/Neumann patterns, so it is *not*
registered in `bc_manager.conditions` — it goes through the raw-string equation
path. That is fine for a constant right-hand side, but a time- or
space-dependent Robin value will **not** be refreshed during the run.

### Stress-free / free-slip

```julia
add_bc!(problem, "w(z=0) = 0")
add_bc!(problem, "∂z(u)(z=0) = 0")
add_bc!(problem, "w(z=1) = 0")
add_bc!(problem, "∂z(u)(z=1) = 0")
```

### No-slip

All velocity components vanish at the wall. On a `VectorField` a single BC covers
every component:

```julia
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
```

Component-by-component (scalar velocity fields) it is one BC per component:

```julia
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "w(z=0) = 0")
```

There are also named helpers for the common cases, which build the BC object
directly: `no_slip!(problem, "u", "z", 0.0)`, `fixed_value!(problem, "T", "z", 0.0, 1.0)`,
`free_slip!(problem, "u", "z", 0.0)`, `insulating!(problem, "T", "z", 1.0)`.

### Pressure gauge

For incompressible flow, pressure is defined only up to a constant. Fix the gauge with an integral constraint:

```julia
tau_p = ScalarField(dist, "tau_p", (), Float64)   # add tau_p to the IVP variables
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_bc!(problem, "integ(p) = 0")
```

The `integ(p) = 0` constraint is trivially zero at non-DC Fourier modes and is filtered by valid-mode filtering in the subproblem builder — it's only active at the DC mode, where it fixes the pressure gauge.

### Custom boundary conditions

Any linear combination of interpolated fields and interpolated derivatives works,
including combinations that couple two fields at a wall — **provided both fields are
problem variables and every field term is on the left-hand side**. Each interpolated
field term becomes a column of the BC row, so it has to be an unknown:

```julia
# T and S are both unknowns; S is pinned to 1, and T is tied to it at the wall
problem = LBVP([T, S, tau_T1, tau_T2, tau_S1, tau_S2])
add_parameters!(problem; lT1=lift(tau_T1, lb2, -1), lT2=lift(tau_T2, lb2, -2),
                         lS1=lift(tau_S1, lb2, -1), lS2=lift(tau_S2, lb2, -2))
add_equation!(problem, "Δ(T) + lT1 + lT2 = 0")
add_equation!(problem, "Δ(S) + lS1 + lS2 = 0")
add_bc!(problem, "S(z=0) = 1")
add_bc!(problem, "S(z=1) = 1")               # ⇒ S ≡ 1
add_bc!(problem, "T(z=0) - 2*S(z=0) = 0")    # coupled BC: T(z=0) = 2·S(z=0)
add_bc!(problem, "T(z=1) = 0")
```

(RUN: recovers `S ≡ 1` exactly and `T = 2(1-z)` to 2e-16.)

!!! warning "A coupled BC only sees the problem's variables"
    If `S` is a *known* field handed to the problem through `add_parameters!` rather
    than declared as a variable, the `S(z=0)` term has no column to land in: it is
    dropped without a warning and the condition silently degrades to `T(z=0) = 0`
    (measured `T ≡ 0` instead of `2(1-z)`). For a known field, evaluate it yourself
    and write the resulting number on the right — `add_bc!(problem, "T(z=0) = 2")`.

!!! warning "Field terms on the RHS of a BC are dropped"
    The right-hand side of a BC must be a constant, a compound constant, or a
    coordinate/time expression — never a field. Writing the same coupling as
    `"T(z=0) = 2*S(z=0)"` still builds and solves, but the solver warns
    *"Boundary condition right-hand side … is not supported and is being enforced as
    ZERO"* and discards the term — measured `T ≡ 0` instead of `2(1-z)`. Keep every
    field reference on the left of the `=`.

### Legacy: `add_equation!` for BCs

`add_equation!(problem, "T(z=0) = 1")` still works — the equation parser detects the `field(coord=value)` syntax and converts it internally. However, this path **does not** register the BC in `bc_manager.conditions`, so time- and space-dependent BC refresh is disabled. For constant BCs it's equivalent to `add_bc!`; for anything else, prefer `add_bc!`.

---

## Problem Properties

### Accessing variables

```julia
# The unknowns, as declared
vars = problem.variables

for v in problem.variables
    println(v.name)
end

# Scalar names, flattening VectorField components ("u_x", "u_z", ...)
get_variable_names(problem)
get_variable_count(problem)
```

### Accessing equations

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

### Accessing boundary conditions

`problem.boundary_conditions` is a `Vector{String}` — the raw strings you passed
to `add_bc!`. The structured objects (with `field` / `coordinate` / `position` /
`value`) live in `problem.bc_manager.conditions`:

```julia
# Raw strings
for bc in problem.boundary_conditions
    println(bc)
end

# Parsed BC objects (DirichletBC / NeumannBC)
for bc in problem.bc_manager.conditions
    println("$(bc.field) at $(bc.coordinate)=$(bc.position): $(bc.value)")
end
```

Only BCs matching the Dirichlet/Neumann patterns appear in `bc_manager.conditions`;
integral constraints and Robin combinations exist only as raw strings.

---

## Problem Validation

### validate_problem

A coarse pre-flight check on the problem's bookkeeping.

**Syntax**:
```julia
validate_problem(problem)   # returns true, or throws ArgumentError
```

**Checks**:
- There is at least one variable and at least one equation
- IVP/EVP: `length(problem.equations) == length(problem.variables)`
- LBVP/NLBVP: `length(problem.equations) >= length(problem.variables)`, and at least one boundary condition exists
- Any BC objects registered in `bc_manager` are self-consistent

**Example**:

```julia
T = ScalarField(dom, "T")
u = VectorField(dom, "u")

problem = IVP([T, u])                                  # 2 declared variables
add_parameters!(problem, kappa=0.05, nu=0.05)
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)") # 2 equations
add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")

validate_problem(problem)   # true
```

!!! warning "It counts declared operands, not degrees of freedom"
    A `VectorField` counts as **one** variable, and boundary conditions added
    with `add_bc!` are **not** counted as equations. A well-posed tau-method
    problem therefore trips the count check — e.g. `LBVP([u, tau1, tau2])` with
    one bulk equation and two `add_bc!` calls throws
    *"Number of equations (1) is less than number of variables (3)"*. The BCs are
    merged into `problem.equations` when the solver is built, so the very same call
    passes afterwards. Build the solver instead: `BoundaryValueSolver` /
    `InitialValueSolver` do the real structural checks.

---

## Problem Substitutions

Give a name to an operator expression and reuse it in equations. Substitutions
are **operator objects**, not strings:

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb  = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
zb  = RealFourier(coords["z"]; size=16, bounds=(0.0, 2π))
dom = Domain(dist, (xb, zb))

u = ScalarField(dom, "u")
w = ScalarField(dom, "w")

# Vorticity ω = ∂x(w) - ∂z(u), built once as an operator
omega = d(w, coords["x"], 1) - d(u, coords["z"], 1)

problem = IVP([u, w])
add_parameters!(problem, nu=0.01, omega=omega)
add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")
add_equation!(problem, "∂t(w) - nu*Δ(w) = omega")
```

The same mechanism carries the tau-lift and first-order-reduction expressions of
the tau method (`grad_u = grad(u) + ez*lift(tau_u1, lift_basis, -1)`), and a plain
Julia function may be registered and called from an equation string
(`add_parameters!(problem, τ_lift=A -> lift(A, lift_basis, -1))`, then
`"… + τ_lift(tau_T2) = …"`).

!!! warning "A substitution given as a String evaluates to zero"
    `add_parameters!(problem, omega="∂x(w) - ∂z(u)")` is accepted, but the string
    is never parsed: the lazy RHS refuses to compile the equation, the solver
    falls back to the interpreted evaluator, warns
    *"Unsupported expression type in evaluate_solver_expression: String"* on every
    stage, and the term contributes **zero** (measured from the same initial
    condition after 10 steps: `max|w| = 0.0` for the string, `1.0e-2` for the
    operator object). Always pass the operator object.

---

## Advanced Features

### Tau method for boundary conditions

Tau variables are **not** created automatically. In a bounded (Chebyshev/Jacobi)
direction you declare one tau variable per boundary condition, include it in the
problem's variable list, and lift it into the bulk equation:

```julia
tau1 = ScalarField(dist, "tau1", (xb,), Float64)   # per-Fourier-mode tau (serial)
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = LBVP([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
```

The lifted taus add the extra degrees of freedom that make each per-mode
subproblem square, so the boundary conditions can be enforced exactly. See
[Tau Method](../pages/tau_method.md) for the full treatment, and the LBVP example
above for a complete run.

### Constraints

There is no `add_constraint!`. Integral constraints are boundary conditions: use
`add_bc!(problem, "integ(p) = 0")` (see [Pressure gauge](#Pressure-gauge) above).

---

## Complete Examples

### 2D Rayleigh-Bénard convection

The full tau-method skeleton: first-order reduction, per-mode tau variables,
`add_bc!` boundary conditions, and a pressure gauge. (RUN: 16×12, 20 steps of
RK222; BCs satisfied to 2e-16.)

```julia
using Tarang

Lx, Lz = 4.0, 1.0
Nx, Nz = 16, 12
Rayleigh, Prandtl = 2e4, 1.0

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))       # serial: Fourier first, Chebyshev last

p = ScalarField(domain, "p")
T = ScalarField(domain, "T")
u = VectorField(domain, "u")

# tau variables: rank-reduced fields carrying the Fourier basis (one tau per x-mode)
tau_p  = ScalarField(dist, "tau_p",  (), Float64)          # scalar gauge unknown
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

# first-order reduction: grad_X = ∇X + ẑ·lift(τ)
ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])
add_parameters!(problem, nu=Prandtl, buoy=Rayleigh*Prandtl, ez=ez,
                grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")

add_bc!(problem, "T(z=0) = 1")
add_bc!(problem, "T(z=1) = 0")
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "integ(p) = 0")      # pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=1e-4)

# initial condition: conduction profile + damped noise
x, z = local_grids(dist, xbasis, zbasis)
fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (1.0 .- z')
get_grid_data(T) .+= 1.0 .- z'
ensure_layout!(T, :c)

run!(solver; stop_iteration=20, progress=false)
```

!!! warning "This example is serial-only"
    The `-u⋅∇(T)` / `-u⋅∇(u)` advection expands to a Chebyshev derivative on the
    explicit side, which the distributed RHS cannot evaluate (each rank owns only
    part of the Chebyshev spectrum). Under MPI the first step errors out. See
    [Running with MPI](../getting_started/running_with_mpi.md).

---

## See Also

- [Solvers](solvers.md): Solving problems with different algorithms
- [Operators](operators.md): Mathematical operators for equations
- [Fields](fields.md): Field types used in problems
- [Timesteppers](timesteppers.md): Time integration schemes
