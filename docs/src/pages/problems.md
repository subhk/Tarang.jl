# Problems

Problems define the PDE system to be solved, including equations and boundary conditions.

## Problem Types

### IVP - Initial Value Problem

Time-dependent PDEs with initial conditions. Every unknown — including any `tau`
variables — is listed in the `IVP([...])` constructor, and the number of equations
(evolution equations **plus** boundary conditions) must equal the number of variables.

```julia
using Tarang

coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
bx = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π), dealias=3/2)
by = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π), dealias=3/2)
dom = Domain(dist, (bx, by))

s = ScalarField(dom, "s")
u = VectorField(dom, "u")

problem = IVP([s, u])
add_parameters!(problem, nu=0.05)

# Linear terms on the LHS (treated implicitly), nonlinear terms on the RHS
add_equation!(problem, "∂t(s) - nu*Δ(s) = -u⋅∇(s)")
add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
```

A fully periodic (all-Fourier) problem like this needs no boundary conditions and no
`tau` variables. A bounded Chebyshev/Jacobi direction needs both — see *Boundary
Conditions* below.

### LBVP - Linear Boundary Value Problem

Steady-state linear PDEs. Boundary conditions in a bounded (Chebyshev/Jacobi)
direction use the **tau method**: declare one `tau` variable per boundary
condition, lift each into the bulk equation via
`lift(tau, derivative_basis(basis, 2), -k)` (registered with `add_parameters!`),
and declare the boundary conditions with `add_bc!` (not `add_equation!`).

```julia
# 2D Poisson  Δu = -2,  u = 0 on both z walls  (solution u = z(Lz - z))
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
    A pure single-axis Chebyshev BVP (no Fourier direction) works too: drop the
    `x` axis and define the `tau` variables on `()`. The solver builds one coupled
    tau subproblem over the Chebyshev spectrum.

### NLBVP - Nonlinear Boundary Value Problem

Steady-state nonlinear PDEs. Same tau-method boundary handling as the LBVP
(tau variables + `lift` + `add_bc!`). Put the nonlinear terms on the right-hand
side; the solver linearizes them with a symbolic Frechet derivative and runs a
per-Fourier-mode Newton iteration.

```julia
# Manufactured nonlinearity  Δu = u² + g,  with g = -2 - u_exact²  so u_exact = z(Lz - z)
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

### EVP - Eigenvalue Problem

Linear stability and eigenvalue analysis. Tarang solves the generalized problem
`L x = σ M x`, where the mass matrix `M` is assembled from the **time-derivative
terms** `dt(·)`: the eigenvalue *replaces* the time derivative (`dt(u) → σ u`).
Keep the `dt(u)` term in the equation — do **not** multiply the eigenvalue symbol
into it (`σ*u = …` builds an empty `M` and returns no eigenvalues). Boundary
conditions use the same tau method (`tau` vars + `lift` + `add_bc!`).

```julia
# 1D diffusion eigenproblem  σu = Δu, Dirichlet;  eigenvalues σ_n = -(nπ/Lz)²
coords = CartesianCoordinates("z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
zb     = ChebyshevT(coords["z"]; size=32, bounds=(0.0, 1.0))
dom    = Domain(dist, (zb,))

u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (), Float64)
tau2 = ScalarField(dist, "tau2", (), Float64)
lb2  = derivative_basis(zb, 2)

evp = EVP([u, tau1, tau2]; eigenvalue=:σ)
add_parameters!(evp; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(evp, "dt(u) - Δ(u) - l1 - l2 = 0")   # dt(u) → σu marks M
add_bc!(evp, "u(z=0)   = 0")
add_bc!(evp, "u(z=1.0) = 0")

solver = EigenvalueSolver(evp; nev=5, which=:SM)   # 5 smallest-magnitude
eigenvalues, eigenvectors = solve!(solver)
# |eigenvalues| ≈ (nπ)² = 9.87, 39.48, 88.83, ...
```

`EigenvalueSolver` accepts only `nev`, `which` (∈ `:LM :SM :LR :SR :LI :SI`),
`target`, and `matsolver`.

## Adding Equations

### Equation Syntax

An equation is a `"LHS = RHS"` string. Everything linear goes on the LHS, where it is
stepped implicitly; the nonlinear terms go on the RHS, where they are stepped explicitly.

```julia
# scalar advection-diffusion: diffusion implicit, advection explicit
add_equation!(problem, "∂t(s) - nu*Δ(s) = -u⋅∇(s)")

# any number of linear terms may share the LHS
add_equation!(problem, "∂t(s) - nu*Δ(s) + ∂x(s) = -u⋅∇(s)")
```

### Equation Sizing

The solver automatically determines each equation's row count in the system matrix from the expression's output type (scalar, vector, tensor, etc.). Equations can be added in any order — no specific ordering is required.

```julia
# q, ψ: ScalarFields;  u: VectorField;  tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)
problem = IVP([q, ψ, u, tau_ψ])
add_parameters!(problem, nu=1e-6)

# Any order is fine:
add_equation!(problem, "Δ(ψ) + tau_ψ - q = 0")           # scalar → D rows
add_equation!(problem, "u - skew(grad(ψ)) = 0")           # vector → 2D rows
add_equation!(problem, "∂t(q) + nu*Δ⁴(q) = -u⋅∇(q)")     # scalar → D rows
add_bc!(problem, "integ(ψ) = 0")                           # constraint → 1 row
```

### Supported Operations

- Derivatives: `∂x`, `∂y`, `∂z`, `∂t` (or `dt`), `Δ` (or `lap`), `∇`, `div`, `curl`,
  `Δ⁴` (hyperdiffusion), and the advection shorthand `u⋅∇(f)`
- Tensor/algebraic: `trace`, `skew`, `grad`, `lift`, `integ`
- Arithmetic: `+`, `-`, `*`, `/`, `^`
- Functions: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `tanh`
- Parameters: any name registered with `add_parameters!` (i.e. present in
  `problem.namespace`) — a bare Julia global is *not* visible to an equation string

### Parameters

Equation strings resolve names against `problem.namespace`. They cannot see plain Julia
globals, so every constant, field or operator an equation refers to must be registered
first — with `add_parameters!` (preferred) or by writing into the namespace directly.

```julia
add_parameters!(problem; nu=0.01, Ra=1e6, Pr=1.0)   # preferred
problem.namespace["nu"] = 0.01                       # equivalent, direct dict access

add_equation!(problem, "∂t(u) - nu*Δ(u) = 0")
```

Keep parameter-times-field terms on the LHS: a linear term on the RHS (`= Ra*Pr*w`)
breaks the IMEX splitting, and `add_equation!` warns about it.

## First-Order Formulation

For problems involving Chebyshev bases, Tarang supports a first-order reduction (tau method) that replaces `Δ(f)` with `div(grad_f)` where `grad_f` includes a tau-lifting term. This ensures correct boundary condition enforcement.

```julia
# Derivative basis and lift closure
ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A)  = lift(A, lift_basis, -1)

# First-order gradient substitutions
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_b = grad(b) + ez * τ_lift(tau_b1)

# The substitutions are Julia objects, so they must be registered before the equation
# strings can name them
add_parameters!(problem, kappa=0.1, grad_u=grad_u, grad_b=grad_b, τ_lift=τ_lift)

# Equations use div(grad_f) instead of Δ(f)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -u⋅∇(b)")
```

Each `τ_lift(tau_*)` term carries one tau variable, and each tau variable pays for one
boundary condition. The Rayleigh-Bénard pattern below is a complete system in this form.

The advection operator `u⋅∇(f)` is automatically expanded component-wise to `Σᵢ uᵢ ∂ᵢf`, so it works for both scalar and vector fields without manual expansion.

## Boundary Conditions

Boundary conditions are declared with **`add_bc!`**, not `add_equation!`. The syntax
`field(coord=value)` is auto-detected and converted to the appropriate condition.

Two rules govern every BC:

1. **Each boundary condition needs its own `tau` variable.** A BC adds one row to the
   system, and the solver requires as many equations (PDEs + BCs) as variables. A wall
   problem with two conditions per field carries two tau variables per field, lifted into
   the bulk equation (see the [tau method](tau_method.md)); without them, solver
   construction throws `Problem validation failed: Number of equations ... does not match
   number of variables`.
2. **The value must be a literal or a registered parameter.** BC strings resolve names
   against `problem.namespace`, so a plain Julia global warns `Unknown variable` and is
   silently enforced as `0`. Use a literal, or register the name with `add_parameters!`.

Declaring a BC with `add_equation!` also registers it as an equation row, but it never
reaches the BC manager, so a space- or time-dependent value is never refreshed and is
enforced as zero. Always use `add_bc!`.

### Dirichlet (Value)

```julia
# field = value at location
add_bc!(problem, "T(z=0) = 1")   # T = 1 at z = 0
add_bc!(problem, "T(z=1) = 0")   # T = 0 at z = 1
```

### Neumann (Derivative)

```julia
# ∂field/∂z = value at location
add_bc!(problem, "∂z(T)(z=0) = 1")   # ∂T/∂z = 1 at z = 0
```

### Robin (Mixed)

```julia
# α*T + β*∂T/∂n = γ
add_bc!(problem, "1.0*T(z=0) + 1.0*∂z(T)(z=0) = 0")

# The same condition as a structured object: robin_bc(field, coord, position, α, β, γ)
add_bc!(problem, robin_bc("T", "z", 0.0, 1.0, 1.0, 0.0))
```

### No-Slip and Stress-Free Walls

```julia
add_bc!(problem, "u(z=0) = 0")       # no-slip:     u = 0 at a solid wall
add_bc!(problem, "∂z(u)(z=1) = 0")   # stress-free: ∂u/∂z = 0 at a free surface
```

### Named Helpers

The common physical conditions have wrappers that build the same BCs:

```julia
no_slip!(problem, "u", "z", 0.0)            # u = 0        at z = 0
free_slip!(problem, "u", "z", 1.0)          # ∂u/∂z = 0    at z = 1
fixed_value!(problem, "T", "z", 0.0, 1.0)   # T = 1        at z = 0
insulating!(problem, "T", "z", 1.0)         # ∂T/∂z = 0    at z = 1
```

### Gauge Constraints

A pressure-like field defined only up to a constant needs a gauge condition, which is
also declared with `add_bc!` and also consumes a tau variable:

```julia
add_bc!(problem, "integ(p) = 0")
```

## Problem Validation

`validate_problem` returns `true` for a well-posed system and otherwise throws an
`ArgumentError` listing every problem it found (missing variables or equations, a
mismatched equation count, invalid boundary conditions). It runs automatically inside
every solver constructor, so a badly-posed problem fails at solver construction — you
rarely need to call it yourself.

```julia
validate_problem(problem)   # true, or throws ArgumentError
```

The equation count check is exact for IVPs and EVPs (`#equations + #BCs == #variables`)
and a lower bound for BVPs (`#equations + #BCs >= #variables`). Boundary conditions are
merged into the equation list when the solver is built, which is why every BC must be
paired with a tau variable.

## Common Problem Patterns

All three wall-bounded patterns below share one skeleton: a periodic Fourier direction
`x`, a bounded Chebyshev direction `z`, one tau variable per boundary condition, and BCs
declared with `add_bc!`.

### Heat Equation

Two Dirichlet walls, so two tau variables lifted into the diffusion term. Lifting into a
second-order operator (`Δ`) uses `derivative_basis(zb, 2)` at lift orders `-1` and `-2`.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=8,  bounds=(0.0, 2π))
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))
dom = Domain(dist, (xb, zb))

T    = ScalarField(dom, "T")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)   # one tau per BC
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = IVP([T, tau1, tau2])
add_parameters!(problem; kappa=0.01, l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "∂t(T) - kappa*Δ(T) + l1 + l2 = 0")
add_bc!(problem, "T(z=0) = 1")   # hot bottom
add_bc!(problem, "T(z=1) = 0")   # cold top

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
set!(T, 0.0)
run!(solver; stop_iteration=10, progress=false)
```

### Incompressible Navier-Stokes

Velocity is a single `VectorField` (not per-component scalars), and the equations use the
first-order tau form: `div(grad_u)` in place of `Δ(u)`, with `tau_u1` inside `grad_u` and
`tau_u2` lifted into the momentum equation. `tau_p` is the gauge unknown that pays for
`integ(p) = 0`.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 4.0), dealias=3/2)
zb = ChebyshevT(coords["z"];  size=12, bounds=(0.0, 1.0), dealias=3/2)
dom = Domain(dist, (xb, zb))

p = ScalarField(dom, "p")
u = VectorField(dom, "u")
tau_p  = ScalarField(dist, "tau_p", (), Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xb,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xb,), Float64)

ex, ez     = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zb, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_u     = grad(u) + ez * τ_lift(tau_u1)

problem = IVP([p, u, tau_p, tau_u1, tau_u2])
add_parameters!(problem, nu=0.01, grad_u=grad_u, τ_lift=τ_lift)
add_equation!(problem, "trace(grad_u) + tau_p = 0")                                 # continuity
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) + τ_lift(tau_u2) = -u⋅∇(u)")  # momentum
add_bc!(problem, "u(z=0) = 0")     # no-slip walls
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "integ(p) = 0")   # pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=1e-4)
```

### Rayleigh-Bénard

Navier-Stokes plus a buoyant temperature field: two more tau variables (`tau_T1`,
`tau_T2`) for the two temperature walls.

```julia
Rayleigh, Prandtl = 2e4, 1.0

# coords / dist / xb / zb / dom / p / u / tau_p / tau_u1 / tau_u2 / ez / τ_lift / grad_u
# exactly as in the Navier-Stokes pattern above
T      = ScalarField(dom, "T")
tau_T1 = ScalarField(dist, "tau_T1", (xb,), Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xb,), Float64)
grad_T = grad(T) + ez * τ_lift(tau_T1)

problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])
add_parameters!(problem, nu=Prandtl, buoy=Rayleigh*Prandtl, ez=ez,
                grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")
add_bc!(problem, "T(z=0) = 1")   # hot bottom
add_bc!(problem, "T(z=1) = 0")   # cold top
add_bc!(problem, "u(z=0) = 0")
add_bc!(problem, "u(z=1) = 0")
add_bc!(problem, "integ(p) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-4)
```

!!! warning "Chebyshev advection is serial-only"
    The `-u⋅∇(u)` / `-u⋅∇(T)` terms differentiate along the Chebyshev axis on the explicit
    side, which a distributed run cannot do — each rank owns only part of that axis. Both
    patterns run in serial; under MPI the first step raises an error. See
    [Parallelism](parallelism.md).

### Poisson Equation (BVP)

A steady BVP needs the tau method. The 2D form below (`Δu = -2`, `u = 0` on both
`z` walls) is shown; a pure 1D Chebyshev Poisson works the same way — drop the `x`
axis and put the `tau` variables on `()`.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 2π))
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))
dom = Domain(dist, (xb, zb))

u    = ScalarField(dom, "u")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = LBVP([u, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
add_equation!(problem, "Δ(u) + l1 + l2 = -2")
add_bc!(problem, "u(z=0)   = 0")
add_bc!(problem, "u(z=1.0) = 0")

solver = BoundaryValueSolver(problem); solve!(solver); ensure_layout!(u, :g)
```

## See Also

- [Solvers](solvers.md): Solving problems
- [Boundary Conditions Tutorial](../tutorials/boundary_conditions.md): Detailed BC guide
- [API: Problems](../api/problems.md): Complete reference
