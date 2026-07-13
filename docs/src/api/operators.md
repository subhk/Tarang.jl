# Operators API

Operators compute derivatives and other mathematical operations on fields. Tarang.jl provides a symbolic syntax for natural mathematical notation in equations.

## Overview

Tarang.jl supports:
- **Differential operators**: grad (∇), divergence (div), curl, lap (Δ, ∇²)
- **Coordinate derivatives**: `∂x`, `∂y`, `∂z`, … in equation strings; `d(field, coord, order)` in Julia code
- **Time derivatives**: ∂t
- **Field operations**: dot (⋅), cross (×)
- **Custom operators**: helper functions that compose the built-in operators

Operators are **lazy**: `∇(T)` builds a `Gradient` object, it does not compute anything.
Call `evaluate` to get a field back:

```julia
∇T = ∇(T)            # Gradient{ScalarField{...}}  — an unevaluated operator
∇T = evaluate(∇(T))  # VectorField                 — the computed gradient
```

Inside an equation string (`add_equation!`) you never call `evaluate` — the solver does it.

## Unicode Operators

Tarang.jl uses Unicode mathematical symbols for readable, publication-quality code.
Some names are **callable Julia functions**, others exist **only inside equation strings** —
the last column says which:

| Syntax | Description | Typing | Available as |
|--------|-------------|--------|--------------|
| `∇(f)` | Gradient | `\nabla` Tab | Julia function + equation string (`grad`) |
| `Δ(f)` or `∇²(f)` | Laplacian | `\Delta` Tab | Julia function + equation string (`lap`, `Δ`) |
| `∂t(f)` | Time derivative | `\partial` Tab `t` | Julia function + equation string |
| `∂x(f)` | x-derivative | `\partial` Tab `x` | **equation string only** |
| `∂y(f)` | y-derivative | `\partial` Tab `y` | **equation string only** |
| `∂z(f)` | z-derivative | `\partial` Tab `z` | **equation string only** |
| `u ⋅ v` | Dot product | `\cdot` Tab | Julia function + equation string |
| `u × v` | Cross product | `\times` Tab | Julia function + equation string |

`∂x`, `∂y`, `∂z` (and `∂<name>` for any coordinate in your coordinate system) are tokens the
equation parser recognizes — they are **not bound in the `Tarang` module**, so `dudx = ∂x(u)`
in Julia code is an `UndefVarError`. The callable derivative is `d(field, coord, order)`
(see **Coordinate Derivatives** below).

**Example** - advection-diffusion:
```julia
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")
```

**Typing Unicode in Julia**:
- `∇` : Type `\nabla` then press Tab
- `Δ` : Type `\Delta` then press Tab
- `∂t` : Type `\partial` Tab `\_t` Tab
- `∂x` : Type `\partial` Tab `x`
- `⋅` : Type `\cdot` then press Tab
- `×` : Type `\times` then press Tab

---

## Setup used by the examples below

Every programmatic example on this page runs against this 2-D periodic domain:

```julia
using Tarang

coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
bx = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi))
by = RealFourier(coords["y"]; size=16, bounds=(0.0, 2pi))
domain = Domain(dist, (bx, by))

T = ScalarField(domain, "T")
u = VectorField(domain, "u")
set!(T, (x, y) -> sin(x) * cos(y))
set!(u.components[1], (x, y) -> sin(y))
set!(u.components[2], (x, y) -> cos(x))

x, y = local_grids(dist, bx, by)   # grid vectors, for checking results
```

---

## Differential Operators

### Gradient (grad / ∇)

Computes the gradient of a scalar field, returning a vector field.

**Syntax**:
```julia
# In equations
add_equation!(problem, "∂t(T) = -u⋅∇(T)")   # or "grad(T)"

# Programmatic
∇T = evaluate(∇(T))    # or evaluate(grad(T))  -> VectorField
```

**Definitions**:

**Cartesian**:
```math
\nabla p = \frac{\partial p}{\partial x}\hat{x} + \frac{\partial p}{\partial y}\hat{y} + \frac{\partial p}{\partial z}\hat{z}
```

**Spherical**:
```math
\nabla p = \frac{\partial p}{\partial r}\hat{r} + \frac{1}{r}\frac{\partial p}{\partial \theta}\hat{\theta} + \frac{1}{r\sin\theta}\frac{\partial p}{\partial \phi}\hat{\phi}
```

**Example**:

```julia
∇T = evaluate(∇(T))                       # VectorField, 2 components
ensure_layout!(∇T.components[1], :g)
get_grid_data(∇T.components[1])           # ∂T/∂x = cos(x)cos(y), error 2.1e-15
```

The components live on the *evaluated* field: `∇(T).components` does not exist, because
`∇(T)` is still an operator.

Applied to a `VectorField`, `grad` returns a `TensorField` with `(∇u)[i,j] = ∂u_j/∂x_i`
(the first index is the derivative direction).

**Return type**: `Gradient` operator; evaluates to `VectorField` (from a scalar) or
`TensorField` (from a vector).

---

### Divergence (div)

Computes the divergence of a vector field, returning a scalar field.

**Syntax**:
```julia
# In equations
add_equation!(problem, "div(grad_phi) + τ_lift(tau2) = rho")

# Programmatic — the callable is `divergence` (alias `div_op`).
# `div` is Julia's integer division; `div(u)` is a MethodError.
div_u = evaluate(divergence(u))
```

**Definitions**:

**Cartesian**:
```math
\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z}
```

**Spherical**:
```math
\nabla \cdot \mathbf{u} = \frac{1}{r^2}\frac{\partial (r^2 u_r)}{\partial r} + \frac{1}{r\sin\theta}\frac{\partial (\sin\theta\, u_\theta)}{\partial \theta} + \frac{1}{r\sin\theta}\frac{\partial u_\phi}{\partial \phi}
```

**Example**:

```julia
div_u = evaluate(divergence(u))       # ScalarField
ensure_layout!(div_u, :g)
maximum(abs, get_grid_data(div_u))    # u = (sin y, cos x) is divergence-free: 0.0
```

`evaluate` does not compose `Divergence` with an unevaluated `Gradient`
(`evaluate(divergence(grad(T)))` throws). Use `Δ(T)` for the Laplacian, or evaluate the
inner operator first. Inside an *equation string*, `div(...)` of a gradient expression is
supported on the implicit side — that is the tau-method form used below and in
[Tau method](../pages/tau_method.md).

**Return type**: `Divergence` operator; evaluates to `ScalarField`.

---

### Curl (curl)

Computes the curl of a vector field.

**Syntax**:
```julia
ω = evaluate(curl(u))
```

**Definitions**:

**Cartesian (3D)**:
```math
\nabla \times \mathbf{u} = \left(\frac{\partial u_z}{\partial y} - \frac{\partial u_y}{\partial z}\right)\hat{x} + \left(\frac{\partial u_x}{\partial z} - \frac{\partial u_z}{\partial x}\right)\hat{y} + \left(\frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}\right)\hat{z}
```

**2D (returns scalar vorticity)**:
```math
\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
```

**Example** — 2-D vorticity diagnostic:

```julia
ω = evaluate(curl(u))                 # 2D -> ScalarField
ensure_layout!(ω, :g)
get_grid_data(ω)                      # -sin(x) - cos(y), error 3.0e-15
```

**Vorticity evolution.** The rotational form `∇×(u×ω)` is **not supported**: a
`CrossProduct` cannot be an operand of `curl` (`curl(u × ω)` throws
`FieldError: type CrossProduct has no field dist`). Write the vorticity equation in
advective form instead — it compiles and steps:

```julia
omega = ScalarField(domain, "omega")     # in 2D the vorticity is a SCALAR
set!(omega, (x, y) -> -sin(x) - cos(y))  # = curl(u) for the u above

problem = IVP([omega, u])
add_parameters!(problem, nu=0.01)
add_equation!(problem, "∂t(omega) - nu*Δ(omega) = -u⋅∇(omega)")
add_equation!(problem, "∂t(u) - nu*Δ(u) = -u⋅∇(u)")

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
solver.rhs_plan.is_compiled              # true
run!(solver; stop_iteration=5, progress=false)   # max|omega| = 1.9999

# `run!` evolves `u` as well, so restore the setup's velocity for the examples below:
set!(u.components[1], (x, y) -> sin(y))
set!(u.components[2], (x, y) -> cos(x))
```

There is no vortex-stretching term in 2D, and you cannot write one: with a *scalar* `omega`,
`omega⋅∇(u)` is a scalar dotted with a tensor, and the solver rejects it on the first step
with `UnrecognizedRHSExpression`. Stretching only makes sense on a 3-D domain, where
`omega` is a `VectorField`; there the same equation with `+ omega⋅∇(u)` compiles and steps.

**Return type**: `Curl` operator; evaluates to `ScalarField` (2D) or `VectorField` (3D).

---

### Laplacian (lap)

Computes the Laplacian (second derivative) of a field.

**Syntax**:
```julia
# In equations
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")

# Programmatic
ΔT = evaluate(Δ(T))    # or lap(T), or ∇²(T)
```

**Definitions**:

**Cartesian**:
```math
\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}
```

**Spherical**:
```math
\nabla^2 f = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial f}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta\frac{\partial f}{\partial \theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}
```

**Examples**:

```julia
# Programmatic: Δ(sin(x)cos(y)) = -2 sin(x)cos(y), error 1.1e-14
ΔT = evaluate(Δ(T))
```

```julia
# Diffusion / advection-diffusion equation
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")
```

```julia
# Poisson equation (LBVP) — Δφ = ρ with Dirichlet BCs on a Fourier × Chebyshev domain.
# A BVP needs boundary conditions, and the Chebyshev axis needs tau terms.
# NOTE: this block builds its OWN domain, so it uses fresh names (`pcoords`, `pdist`, …)
# and leaves the 2-D setup above intact for the later sections.
pcoords = CartesianCoordinates("x", "z")
pdist   = Distributor(pcoords; dtype=Float64, device=CPU())
xb = RealFourier(pcoords["x"]; size=16, bounds=(0.0, 2pi))
zb = ChebyshevT(pcoords["z"];  size=24, bounds=(0.0, 1.0))
pdomain = Domain(pdist, (xb, zb))

phi  = ScalarField(pdomain, "phi")
rho  = ScalarField(pdomain, "rho")
tau1 = ScalarField(pdist, "tau1", (xb,), Float64)
tau2 = ScalarField(pdist, "tau2", (xb,), Float64)

xg, zg = local_grids(pdist, xb, zb)
ensure_layout!(rho, :g)
get_grid_data(rho) .= -(1 + pi^2) .* cos.(xg) .* sin.(pi .* zg')  # exact: φ = cos(x) sin(πz)
ensure_layout!(rho, :c)

ex, ez     = unit_vector_fields(pcoords, pdist)
lift_basis = derivative_basis(zb, 1)
τ_lift(A)  = lift(A, lift_basis, -1)
grad_phi   = grad(phi) + ez * τ_lift(tau1)

problem = LBVP([phi, tau1, tau2])
add_parameters!(problem, rho=rho, grad_phi=grad_phi, τ_lift=τ_lift)
add_equation!(problem, "div(grad_phi) + τ_lift(tau2) = rho")
add_bc!(problem, "phi(z=0) = 0")
add_bc!(problem, "phi(z=1) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)                       # max error vs cos(x)sin(πz): 4.4e-16
```

**Works on**: ScalarField, VectorField (applies componentwise)

---

## Coordinate Derivatives

### First Derivatives

Partial derivatives with respect to coordinate directions.

**In equation strings** — `∂<coord>` for any coordinate name in your coordinate system:

```julia
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -0.5*∂x(T) - 0.2*∂y(T)")
```

**In Julia code** — `∂x` is *not* a function. Use `d(field, coord, order)` (constructor
alias `Differentiate`), and `evaluate` it:

```julia
dTdx = evaluate(d(T, coords["x"], 1))     # ∂T/∂x = cos(x)cos(y), error 2.1e-15
d2T  = evaluate(d(T, coords["x"], 2))     # ∂²T/∂x² = -sin(x)cos(y), error 8.9e-15
```

**Implementation**:
- **Fourier**: Multiplication by ik in spectral space
- **Chebyshev/Legendre**: Sparse matrix multiplication using recurrence relations

---

### Higher Derivatives

**In equation strings** derivatives compose directly:

```julia
# Second and mixed derivatives
add_equation!(problem, "∂t(T) = ∂x(∂y(T))")            # compiles; RHS = ∂²T/∂x∂y

# Biharmonic / hyperdiffusion
add_equation!(problem, "∂t(psi) + nu4*Δ(Δ(psi)) = 0")
```

**In Julia code** an unevaluated `Differentiate` cannot be differentiated again. Either
raise the order, or evaluate the inner derivative first:

```julia
d2T   = evaluate(d(T, coords["x"], 2))                       # ∂²T/∂x²
dxT   = evaluate(d(T, coords["x"], 1))
dxdyT = evaluate(d(dxT, coords["y"], 1))                     # ∂²T/∂x∂y, error 3.1e-15
# evaluate(d(d(T, coords["x"], 1), coords["y"], 1))          # ArgumentError
```

---

## Time Derivatives

### dt / ∂t Operator

Time derivative for initial value problems.

**Syntax**:
```julia
dt(field)   # ASCII
∂t(field)   # Unicode (type \partial Tab \_t Tab)
```

**Examples**:

```julia
# Evolution equations
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*Δ(u) = -u⋅∇(u)")
```

**Note**: Only use in IVP (Initial Value Problems). Not valid for BVP or EVP.

---

## Spectral Operators

### Hilbert Transform (hilbert)

Applies the Hilbert transform in spectral space: each Fourier mode `k` is
multiplied by `-i·sign(k)` (the `k = 0` mode maps to 0). Useful for analytic
signals, envelope/phase diagnostics, and 90° phase shifts along a Fourier axis.

**Syntax**:

```julia
Hf = evaluate(hilbert(f))                      # ScalarField -> ScalarField
```

```julia
add_equation!(problem, "∂t(T) = hilbert(T)")   # parseable in equation strings
```

**Properties** (all verified on a 32-point `RealFourier` axis):
- `H[sin x] = -cos x` (error 5.6e-16), and `H[H[f]] = -f` for zero-mean `f` (error 3.3e-16).
- It is applied to **every** Fourier axis of the field, not just the first: on the 2-D
  domain above, `hilbert(sin(x)cos(y))` returns `-cos(x)sin(y)` (error 8.9e-16).
- Scalar fields only (a `VectorField` operand throws).

**Limitations**:
- `hilbert` in an equation RHS is **not** handled by the compiled ("lazy") RHS: the solver
  logs a warning and the whole problem falls back to the slower interpreted evaluator
  (`solver.rhs_plan.is_compiled == false`).
- Under MPI it throws if a **Fourier** axis is the decomposed one — the `-i·sign(k)`
  multiplier needs global wavenumbers, which a decomposed axis does not expose.

---

## Vector Operations

### Dot Product / Advection

**In equations** — use vector notation directly:
```julia
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")    # scalar advection
add_equation!(problem, "∂t(u) - nu*Δ(u) = -u⋅∇(u)")       # vector advection
```

**Programmatic**:
```julia
adv = evaluate(u ⋅ ∇(T))     # ScalarField, u·∇T — error ~1e-15 against the exact field
```

---

### Cross Product

The cross product is a 3-D operation, so it needs a 3-D domain. As with the Poisson block,
this one uses fresh names so it does not disturb the 2-D setup used by the sections below:

```julia
coords3 = CartesianCoordinates("x", "y", "z")
dist3   = Distributor(coords3; dtype=Float64, device=CPU())
domain3 = Domain(dist3, (RealFourier(coords3["x"]; size=8, bounds=(0.0, 2pi)),
                         RealFourier(coords3["y"]; size=8, bounds=(0.0, 2pi)),
                         RealFourier(coords3["z"]; size=8, bounds=(0.0, 2pi))))
u3 = VectorField(domain3, "u")
ω3 = VectorField(domain3, "omega")
set!(u3.components[1], (x, y, z) -> sin(y))     # u = (sin y, 0, 0)
set!(ω3.components[3], (x, y, z) -> cos(x))     # ω = (0, 0, cos x)

uxω = evaluate(cross(u3, ω3))                   # or: evaluate(u3 × ω3)  -> VectorField
ensure_layout!(uxω.components[2], :g)
get_grid_data(uxω.components[2])                # -sin(y)cos(x), error 7.8e-16
```

`u × ω` is also parseable inside an equation string, but the term is not compiled by the
lazy RHS (the solver falls back to the interpreted evaluator). It **cannot** be nested
inside `curl` — see **Curl** above for the supported vorticity form.

---

## Composite Operators

Combine operators for complex expressions.

### Vector Laplacian

```julia
# For vector field u
# ∇²u = (∇²u_x, ∇²u_y, ∇²u_z)

add_equation!(problem, "∂t(u) - nu*Δ(u) = -u⋅∇(u)")
# Δ automatically applies componentwise
```

### Advection Operator

```julia
# Scalar advection:
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")

# Vector advection (Navier-Stokes nonlinear term):
add_equation!(problem, "∂t(u) - nu*Δ(u) = -u⋅∇(u)")
```

The incompressible momentum equation with a pressure gradient needs the tau/BC machinery —
see the [Rayleigh-Bénard tutorial](../tutorials/ivp_2d_rbc.md) for the full form.

### Strain Rate Tensor

`grad` of a `VectorField` evaluates to a `TensorField` with `(∇u)[i,j] = ∂u_j/∂x_i`.
Build the symmetric part from its components — arithmetic on `ScalarField`s is eager, so
each expression below is a real field:

```julia
∇u  = evaluate(grad(u))                                  # TensorField
S11 = ∇u.components[1, 1]
S12 = 0.5 * (∇u.components[1, 2] + ∇u.components[2, 1])  # ScalarField
S22 = ∇u.components[2, 2]
```

For `u = (sin y, cos x)`: `S11 = S22 = 0` exactly, and `S12 = 0.5(cos y - sin x)` to
1.3e-15.

`S[i, j]` is shorthand for `S.components[i, j]`, for reading *and* for assignment, so you
can also collect the components into a `TensorField` of your own:

```julia
S = TensorField(dist, "S", (bx, by), Float64)
S[1, 2] = S12
S[2, 1] = S12                    # S[1,2] === S.components[1,2]
```

There is no `symmetric=` keyword on the `TensorField` constructor (`MethodError`).

---

## Operator Properties

The identities below were checked numerically on the 2-D setup, with a second scalar field

```julia
g = ScalarField(domain, "g")
set!(g, (x, y) -> cos(2x))
```

The quoted figure is the measured maximum difference.

### Linearity

```julia
# ∇(αf + βg) = α∇f + β∇g          — max difference 3.6e-15
lhs = evaluate(∇(2.0 * T + 3.0 * g))
gT, gg = evaluate(∇(T)), evaluate(∇(g))
# lhs.components[i] ≈ 2.0 * gT.components[i] + 3.0 * gg.components[i]
```

### Commutativity

```julia
# ∂²f/∂x∂y = ∂²f/∂y∂x             — max difference 3.4e-15
a = evaluate(d(evaluate(d(T, coords["x"], 1)), coords["y"], 1))
b = evaluate(d(evaluate(d(T, coords["y"], 1)), coords["x"], 1))
```

### Product Rule

```julia
# ∇(fg) = f∇g + g∇f               — max difference ~2e-15
lhs = evaluate(∇(T * g))
# lhs.components[i] ≈ T * gg.components[i] + g * gT.components[i]
```

---

## Custom Operators

### Defining Helper Functions

Helper functions that compose built-in operators return operator objects, which you can
register as a parameter and then name inside an equation string:

```julia
hyperdiffusion(field, k4) = k4 * Δ(Δ(field))

problem = IVP([T])
add_parameters!(problem, hyper_T = hyperdiffusion(T, 1e-4))
add_equation!(problem, "∂t(T) + hyper_T = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
run!(solver; stop_iteration=10, progress=false)
solver.rhs_plan.is_compiled      # true
```

This is the supported way to inject a hand-built term: an equation string can only see
fields of the problem and names registered with `add_parameters!` — it cannot see plain
Julia globals, and there is no `field.rhs` / `field.data` to write into.

### Using Built-in Operators in Equations

The equation parser recognizes all registered operators. Use them directly in equation strings:

```julia
# Available operators in equations:
# grad, div, curl, lap (or Δ), dt (or ∂t), d, ∂<coord>
# integrate, average, interpolate, convert, lift, trace, hilbert
# sin, cos, tan, exp, log, sqrt, abs, tanh

# Example: diffusion equation
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")

# Example: advection-diffusion
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")
```

---

## Equation Parsing

### Symbolic Syntax

Tarang.jl parses equation strings into operator applications:

```julia
# String equation
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -0.5*∂x(T) - 0.2*∂y(T)")

# Parsed as:
# LHS: ∂t(T) - kappa*Δ(T)
# RHS: -0.5*∂x(T) - 0.2*∂y(T)
```

**Supported operations**:
- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Parentheses: `(`, `)`
- Functions: `sin`, `cos`, `exp`, etc.

### Operator Evaluation

Operators are evaluated in spectral space when possible:

```julia
# ∂x(u): Multiply by ik in Fourier space
# Δ(u): Multiply by -k² in Fourier space
# Nonlinear terms: Transform to grid space, evaluate, transform back
```

---

## Performance Tips

### Check that the RHS compiled

The explicit side of every equation is compiled into a type-specialized plan. If any term
cannot be compiled, the *whole solver* silently drops to the ~100×-slower interpreted
evaluator (after a loud `@warn`). Check it:

```julia
solver = InitialValueSolver(problem, RK222(); dt=1e-3)
solver.rhs_plan.is_compiled      # true for every equation on this page except `hilbert`
                                 # (a bare `u × ω` term does not compile either)
```

### Keep non-Fourier derivatives off the explicit side

A Chebyshev/Legendre derivative in an RHS is either declined by the compiler (Legendre,
and any non-Fourier axis under MPI) or — distributed — errors at the first step, because a
rank owns only part of that axis. Express those terms with `grad`/`div` on the implicit
(L) side of the equation. Fourier derivatives are fine anywhere: they are a diagonal `ik`
multiply.

### Use sparse differentiation

Chebyshev/Legendre derivatives are sparse matrix operations - very efficient.

---

## See Also

- [Fields](fields.md): Field types that operators act on
- [Problems](problems.md): Using operators in equations
- [Domains](domains.md): Spatial discretization for operators
- [Bases](bases.md): Spectral bases for differentiation
