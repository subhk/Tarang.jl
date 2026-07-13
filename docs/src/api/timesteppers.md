# Timesteppers API

Time integration schemes for evolving PDEs in time.

Tarang's timesteppers are **IMEX** schemes: the linear operator `L` (everything on the
left-hand side of an equation) is advanced implicitly, and the nonlinear right-hand side
is advanced explicitly.

## TimeStepper Types

### Abstract Type

All timesteppers inherit from the abstract type `Tarang.TimeStepper`. It is **not
exported** — write it qualified if you need to refer to it (e.g. in a type annotation):

```julia
using Tarang

supertype(RK222)                  # Tarang.TimeStepper
RK222() isa Tarang.TimeStepper    # true
```

The concrete schemes below **are** exported, unless the section says otherwise.

## IMEX Runge-Kutta

### RK111

1st-order IMEX Runge-Kutta.

```julia
RK111()
```

**Properties**:
- Order: 1
- Stages: 1 (all implicit)
- Implicit part: Backward Euler for linear terms
- Explicit part: Forward Euler for nonlinear terms
- Workspace field sets: 3 (stage state, explicit RHS, implicit RHS)

### RK222

2nd-order IMEX Runge-Kutta (Ascher, Ruuth & Spiteri 1997, ARS(2,2,2)) with three stored
stages: an explicit first stage and two diagonally implicit stages.

```julia
RK222()
```

**Properties**:
- Order: 2
- Stages: 3 — an ESDIRK form whose *first* stage is explicit, so only 2 of the 3 stages
  require an implicit solve. (`RK222().stages == 3`; the explicit and implicit tableaux
  share the abscissae `c = [0, γ, 1]`, which is what makes the IMEX pair 2nd order.)
- Implicit part: L-stable ESDIRK with `γ = 1 - 1/sqrt(2)` on stages 2 and 3
- Explicit part: matched three-row explicit tableau
- Workspace field sets: 6 (stage state, explicit RHS, and implicit RHS storage)

**Recommended for**: General purpose problems.

### RK443

3rd-order, 4-stage IMEX Runge-Kutta (Kennedy & Carpenter ARK3(2)4L[2]SA).

```julia
RK443()
```

**Properties**:
- Order: 3
- Stages: 4 — ESDIRK, first stage explicit, so 3 implicit solves per step
- Implicit part: L-stable ESDIRK for linear terms
- Explicit part: 4-stage explicit RK for nonlinear terms
- Workspace field sets: 12 (three field sets per stage)

**Recommended for**: High accuracy requirements.

## IMEX Multistep Methods

### CNAB (Crank-Nicolson Adams-Bashforth)

```julia
CNAB1()  # 1st order
CNAB2()  # 2nd order
```

**Treatment**:
- Linear terms: Crank-Nicolson (implicit)
- Nonlinear terms: Adams-Bashforth (explicit)

**Properties**:
- CNAB1: Order 1, 1 history level
- CNAB2: Order 2, 2 history levels

### SBDF (Semi-implicit BDF)

```julia
SBDF1()  # 1st order
SBDF2()  # 2nd order
SBDF3()  # 3rd order
SBDF4()  # 4th order
```

**Treatment**:
- Linear terms: BDF (implicit)
- Nonlinear terms: Extrapolation (explicit)

**Properties**:
| Method | Order | History Levels | Stability |
|--------|-------|----------------|-----------|
| SBDF1 | 1 | 1 | A-stable |
| SBDF2 | 2 | 2 | A-stable |
| SBDF3 | 3 | 3 | A(α)-stable |
| SBDF4 | 4 | 4 | A(α)-stable |

Multistep schemes start from a single state, so the history has to be seeded. `CNAB2` and
`SBDF2` bootstrap with their own 1st-order member (`CNAB1` / `SBDF1`), so their first step
is reduced order; `SBDF3` / `SBDF4` self-start with an order-3 RK443 step on the
global-matrix path, which keeps the one-time startup error from capping the global order
(on the per-subproblem path they still bootstrap from SBDF2/SBDF1 and are order-capped
at 2). Only SBDF1/SBDF2 are A-stable: the higher-order members trade stability angle for
formal order, so they suit *smooth, accuracy-limited* integration rather than the stiffest
linear terms.

### MCNAB2 / CNLF2

Additional two-step multistep schemes. **Neither is exported** — they must be qualified:

```julia
Tarang.MCNAB2()   # θ-weighted Crank-Nicolson + Adams-Bashforth 2
Tarang.CNLF2()    # Crank-Nicolson Leapfrog (CN implicit + centered leapfrog explicit)
```

`MCNAB2` accepts an optional CN weight, `Tarang.MCNAB2(theta)` (default `0.5`). This is a
**two-level θ-method**, not the three-level Ascher-Ruuth-Wetton "modified CNAB": at
`θ = 0.5` it is Crank-Nicolson and 2nd order, but any `θ > 0.5` adds damping at the cost of
dropping to **1st order**. Use the default unless you specifically want that trade.

`CNLF2` is 2nd order and 2-step.

## Exponential Time Differencing (ETD)

Exponential integrators that treat the linear term *exactly* via the matrix
exponential, which is well suited to very stiff linear operators.

```julia
ETD_RK222()  # 2nd-order exponential Runge-Kutta
ETD_CNAB2()  # 2nd-order exponential Crank-Nicolson Adams-Bashforth
ETD_SBDF2()  # 2nd-order exponential semi-implicit BDF
```

**Properties**:
- Order: 2
- Linear term: exact exponential propagation (via φ-functions)
- Nonlinear term: explicit (RK / Adams-Bashforth / BDF extrapolation)
- Best for problems where the linear part dominates the stiffness.

## Specialized IMEX-RK

Additive Runge-Kutta (ARK) schemes. `RKSMR` is exported; `RKGFY` and `RK443_IMEX` are
**not**, and must be qualified.

### RKSMR

```julia
RKSMR()
```

The Spalart–Moser–Rogers scheme, the workhorse IMEX integrator of incompressible spectral
DNS. Its nonlinear/explicit part is **third order** and its linear/implicit part
(Crank-Nicolson on the viscous term) is **second order** — that asymmetric accuracy profile
is the standard SMR one, not an implementation limitation.

It is stored in a four-stage additive-RK (ESDIRK) tableau — a trivial explicit first stage
plus the three SMR substeps — so it shares the same generic IMEX runtime paths as `RK222`
and `RK443`.

### RKGFY / RK443_IMEX

```julia
Tarang.RKGFY()       # 3-stage 2nd-order L-stable ARK (Ascher-Ruuth-Spiteri 1997)
Tarang.RK443_IMEX()  # Same Kennedy-Carpenter coefficients as RK443; the suffix clarifies IMEX intent
```

### Diagonal IMEX

```julia
DiagonalIMEX_RK222()  # 2nd order
DiagonalIMEX_RK443()  # 3rd order
DiagonalIMEX_SBDF2()  # 2nd order multistep
```

The implicit solve is **diagonal in spectral space** — the linear operator is applied per
Fourier mode (`(I + γ·dt·L̂)⁻¹` per wavenumber) rather than through a global matrix solve.
Much cheaper for pure-Fourier / diagonalizable linear terms, and because the solve is an
element-wise division there is no sparse factorization to move off-device on GPU. It
requires the implicit operator to be diagonal in the chosen basis.

**These schemes do not read the implicit operator from your equation.** They take it from a
`SpectralLinearOperator` that you attach to the solver with `set_spectral_linear_operator!`:

```julia
using Tarang

coords = CartesianCoordinates("x")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb     = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))

u = ScalarField(dist, "u", (xb,), Float64)
x = local_grid(xb, dist, 1)
ensure_layout!(u, :g)
get_grid_data(u) .= cos.(2 .* x)          # a single mode, k = 2

problem = IVP([u])
add_equation!(problem, "∂t(u) = 0")       # the linear term lives in L, NOT in the equation

solver = InitialValueSolver(problem, DiagonalIMEX_RK222(); dt=0.005)
L = SpectralLinearOperator(dist, (xb,), :laplacian; ν=0.5)   # L̂(k) = ν k²
set_spectral_linear_operator!(solver, L)

run!(solver; stop_iteration=200, progress=false)   # t = 1.0

ensure_layout!(u, :g)
maximum(abs, get_grid_data(u))   # 0.13533 ≈ exp(-ν k² t) = exp(-2) = 0.13534
```

`SpectralLinearOperator(dist, bases, kind; ν, order)` supports `:laplacian` (`L̂ = ν k²`),
`:hyperviscosity` (`L̂ = ν (k²)^order`), `:biharmonic` (`L̂ = ν k⁴`), and `:custom`
(pass `coefficients` yourself).

!!! warning "Attach the operator, or the linear term is lost"
    If no `SpectralLinearOperator` is attached, `DiagonalIMEX_RK222` / `DiagonalIMEX_RK443`
    fall back to a **fully explicit** RK step, and the implicit `L` written in the equation
    is silently dropped: running `∂t(u) - nu*lap(u) = 0` with `DiagonalIMEX_RK222` and no
    attached operator leaves `u` completely undiffused (`max|u|` stays at its initial
    value). `DiagonalIMEX_SBDF2` errors outright on the same path. If you want the
    equation's `L` treated implicitly without extra setup, use `RK222` / `SBDF2` instead.

## Usage with Solver

```julia
using Tarang

coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb     = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
yb     = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
domain = Domain(dist, (xb, yb))

u = ScalarField(domain, "u")
problem = IVP([u])
add_parameters!(problem, nu=0.1)
add_equation!(problem, "∂t(u) - nu*lap(u) = 0")
set!(u, (x, y) -> sin(x) * cos(y))

# Create solver with a timestepper — note the `()`
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# Drive it with run!
run!(solver; stop_iteration=50, progress=false)

ensure_layout!(u, :g)
maximum(abs, get_grid_data(u))   # 0.99004983, matching exp(-2νt) to 1.6e-11
```

Or drive the loop yourself:

```julia
solver = InitialValueSolver(problem, SBDF2(); dt=1e-3)
while solver.sim_time < 0.05
    step!(solver)
end
```

## Timestepper State

### TimestepperState

`Tarang.TimestepperState` holds everything a scheme needs between steps: the state history,
the timestep history (for variable-`dt` multistep schemes), pre-allocated workspace fields,
and any forcing. It is **not exported** and is managed automatically by the solver — you
reach it as `solver.timestepper_state` and normally never touch it.

```julia
fieldnames(Tarang.TimestepperState)
# (:timestepper, :dt, :history, :dt_history, :stage, :timestepper_data,
#  :workspace_fields, :workspace_allocated, :forcing, :current_substep, :forcing_generated)
```

## Stability

### Explicit (advective) restriction

The explicit half of every IMEX scheme is limited by the advective CFL condition:

```math
\Delta t < C \frac{\Delta x}{u_{max}}
```

The admissible factor `C` depends on the PDE, the spatial discretization and the scheme;
Tarang does not define universal CFL constants for individual steppers. Rather than
hard-coding `C`, use the `CFL` controller, which recomputes `dt` from the grid spacing and
the current velocity field with a `safety` factor. It is constructed from the **solver**
(never from a problem), and its velocities must be `VectorField`s:

```julia
v = VectorField(domain, "v")
set!(v.components[1], (x, y) -> 0.5)

prob = IVP([u, v])
add_parameters!(prob, nu=0.1)
add_equation!(prob, "∂t(u) - nu*lap(u) = -v⋅∇(u)")
add_equation!(prob, "∂t(v) - nu*lap(v) = 0")
set!(u, (x, y) -> sin(x) * cos(y))

solver = InitialValueSolver(prob, RK222(); dt=1e-3)

cfl = CFL(solver; initial_dt=1e-3, cadence=5, safety=0.4, max_dt=0.01)
add_velocity!(cfl, v)

run!(solver; stop_iteration=10, cfl=cfl, progress=false)
solver.dt   # updated by the controller
```

### Implicit (diffusive) restriction

Because the linear operator is advanced implicitly, the *diffusive* stability restriction
(`Δt ≲ Δx²/ν`) is removed. Implicit linear treatment removes the explicit diffusive limit,
but it does **not** remove the advective CFL limit above: only advective CFL constrains
`dt`. That is the whole point of the IMEX splitting, and it is why all of Tarang's schemes
are IMEX.

## Method Selection Guide

| Problem Type | Method | Reason |
|--------------|--------|--------|
| General purpose | RK222 | Balance of cost/accuracy |
| High accuracy | RK443 | More stages, higher order |
| Diffusion-dominated | SBDF2 | A-stable implicit diffusion, one solve per step |
| Smooth high-order integration | RK443 or SBDF3/SBDF4 | Higher formal order; the multistep members need startup history and are only A(α)-stable |
| Very stiff, linear-dominated | ETD_RK222 | Exact exponential propagation of L |
| Classic incompressible DNS IMEX | RKSMR | SMR explicit-third / implicit-second accuracy profile |
| Diagonal Fourier linear operator / GPU | DiagonalIMEX family | Per-mode implicit division avoids a global sparse solve (needs an attached `SpectralLinearOperator`) |

## Performance

### Computational Cost

The RHS is evaluated once per stage; an implicit solve is needed on each stage whose
implicit diagonal coefficient `A_implicit[s,s]` is nonzero (the ESDIRK first stage is free).
Each IMEX-RK step then finishes with one mass-matrix solve for the update; the
factorizations of `(M + a·dt·L)` and `M` are cached across steps and reused as long as `dt`
is unchanged.

| Method | Stages | RHS Evaluations | Implicit Solves |
|--------|--------|-----------------|-----------------|
| RK111 | 1 | 1 | 1 |
| RK222 | 3 (1st explicit) | 3 | 2 |
| RK443 | 4 (1st explicit) | 4 | 3 |
| RKSMR | 4 (1st explicit) | 4 | 3 |
| CNAB2 | — | 1 | 1 |
| SBDF2 | — | 1 | 1 |

The multistep rows are the per-step cost once the history ring is full; the startup steps
cost the same but run at reduced order.

For the `DiagonalIMEX_*` schemes the stage counts are the same, but each "implicit solve"
is an element-wise division by `(1 + a·dt·L̂(k))` instead of a sparse factorization.

### Memory Requirements

Workspace storage is preallocated per timestepper type and reused: these are scratch
buffers, not fresh state copies allocated on every step. Workspace field *sets* are
allocated per state field (each set is one full field):

| Method | Workspace Sets |
|--------|----------------|
| RK111 | 3 |
| RK222 | 6 |
| RK443 | 12 |
| CNAB1 / CNAB2 | 2 |
| SBDF1 – SBDF4 | 2 |
| ETD_RK222 / ETD_CNAB2 / ETD_SBDF2 | 3 |
| DiagonalIMEX_* | 4 |
| everything else (RKSMR, MCNAB2, CNLF2, RKGFY, RK443_IMEX) | 2 |

The multistep schemes additionally retain previous state and RHS levels in a history ring
(2–5 levels, depending on order), which the RK schemes do not.

## See Also

- [Solvers](solvers.md): Using timesteppers
- [User Guide: Timesteppers](../pages/timesteppers.md): Conceptual guide
