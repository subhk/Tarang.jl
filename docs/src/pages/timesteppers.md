# Time Steppers

Time steppers integrate PDEs forward in time.

## Overview

Tarang.jl provides several families of time integration schemes:

| Family | Type | Best For |
|--------|------|----------|
| RK | IMEX Runge-Kutta | General problems |
| CNAB | Multistep IMEX | Moderately stiff |
| SBDF | Multistep IMEX | Stiff problems |
| ETD | Exponential Time Differencing | Very stiff linear terms |

## IMEX Runge-Kutta

All RK methods use Additive Runge-Kutta (ARK) schemes:
- Linear terms (LHS) are treated **implicitly**
- Nonlinear terms (RHS) are treated **explicitly**

This allows stable integration of stiff problems with larger timesteps.

### RK111

First-order IMEX method (Backward Euler / Forward Euler).

```julia
timestepper = RK111()
```

- **Implicit part**: Backward Euler (linear terms)
- **Explicit part**: Forward Euler (nonlinear terms)
- **Accuracy**: O(Δt)
- **Use case**: Testing, simple problems

### RK222

Second-order IMEX Runge-Kutta using a three-stage matched ARS tableau (an explicit first stage followed by two diagonally implicit stages).

```julia
timestepper = RK222()
```

- **Implicit part**: ESDIRK with γ = 1 - 1/√2 on stages 2 and 3
- **Explicit part**: matched three-row explicit tableau with `c = [0, γ, 1]`
- **Accuracy**: O(Δt²)
- **Stability**: L-stable
- **Use case**: General purpose (recommended)

### RK443

Third-order, 4-stage IMEX Runge-Kutta (Kennedy & Carpenter ARK3(2)4L[2]SA).

```julia
timestepper = RK443()
```

- **Implicit part**: 4-stage ESDIRK (γ ≈ 0.4359)
- **Explicit part**: 4-stage explicit RK
- **Accuracy**: O(Δt³)
- **Stability**: L-stable
- **Use case**: High accuracy requirements, stiff problems

### RKSMR

The Spalart–Moser–Rogers semi-implicit scheme, stored as an equivalent four-stage additive Runge-Kutta tableau.

```julia
timestepper = RKSMR()
```

- **Explicit part**: third-order treatment of nonlinear/advection terms
- **Implicit part**: second-order treatment of the linear term
- **Runtime path**: the same generic IMEX driver used by `RK222` and `RK443`
- **Use case**: incompressible spectral DNS where the classic SMR accuracy profile is desired

## Multistep IMEX Methods

Implicit-Explicit methods treat stiff (linear) terms implicitly and non-stiff (nonlinear) terms explicitly.

### CNAB (Crank-Nicolson Adams-Bashforth)

```julia
CNAB1()  # 1st order
CNAB2()  # 2nd order (recommended)
```

- **Linear terms**: Crank-Nicolson (implicit)
- **Nonlinear terms**: Adams-Bashforth (explicit)
- **Use case**: Diffusion-dominated flows

### SBDF (Semi-implicit BDF)

```julia
SBDF1()  # 1st order
SBDF2()  # 2nd order (recommended)
SBDF3()  # 3rd order
SBDF4()  # 4th order
```

- **Linear terms**: BDF (implicit)
- **Nonlinear terms**: Extrapolation (explicit)
- **Use case**: Stiff problems, high Re flows

## How Boundary Conditions Are Enforced During Time Stepping

Tarang's stepper is aware of the fact that boundary-condition rows are *algebraic* (have no time derivative), which makes the full system a **differential-algebraic equation** (DAE) rather than a pure ODE. If you write a BC like `T(z=0) = 1`, the corresponding row of the combined `M·dX/dt + L·X = F` system has `M_row = 0`, and the raw accumulated IMEX-RK stage RHS formula produces the **wrong scaling factor** for that row.

Concretely, for RK222 (`γ = 1 − 1/√2 ≈ 0.293`), the naive formula enforces `L_row·X = (1/γ)·F_BC = (2+√2)·F_BC ≈ 3.414·F_BC` instead of `F_BC`. A `T(z=0) = 1` BC would produce `T(z=0) ≈ 3.414` in grid space.

### The `apply_bc_override!` fix

`step_subproblem_rk!` handles this by **overriding** the BC-row entries of the stage RHS at each stage:

```
# (pseudo-code — internal stepper logic, not user API)

# After the normal accumulation:
rhs = M*X_n + dt * Σ (A^E[i,j]*F_j − A^I[i,j]*L*X_j)

# Override BC rows with the correct direct-enforcement value:
for r in sp.bc_rows
    rhs[r] = dt * a_ii * F_BC[r]
end
```

After this override, the stage solve `(M + dt·a_ii·L)·X = rhs` gives `dt·a_ii·L_row·X = dt·a_ii·F_BC` → `L_row·X = F_BC` directly, at every stage. Since the singular-`M` final-update path keeps the last stage's value, this guarantees the BC is correctly enforced in the final solution.

`sp.bc_rows` comes from the subproblem builder's equation-size classification (rows with `eq_size < Nz` — small algebraic rows like wall BCs and gauge constraints). The override only touches those rows; larger `Nz`-sized algebraic rows like `trace(grad_u) + tau_p = 0` (continuity) have `F = 0` and ride the normal accumulation path without issue.

`step_subproblem_multistep!` does the same thing for CNAB / SBDF — the override coefficient there is `b[0]` (the implicit-diagonal coefficient of the multistep formula) instead of `dt·a_ii`, but the mechanism is identical.

### Time- and space-dependent BCs

For BCs that vary with time (`T(z=0) = sin(t)`) or space (`T(z=0) = sin(2πx/Lx)`) or both, the stepper refreshes the BC F value **before each stage solve**:

1. `update_time_dependent_bcs!(bc_manager, t + c[i]·dt)` — evaluates the BC expression string at the stage time.
2. `_apply_bc_values_to_equations!(solver, stage_time)` — writes the fresh value into `equation_data[eq_idx]["F"]` (as `ConstantOperator` for scalar values or `ArrayOperator` for spatially-varying ones).
3. `gather_alg_F!` picks up the new F on the next read.

This per-stage refresh (gated on `has_time_dependent_bcs`, so it's free when BCs are constant) preserves the stepper's formal order of accuracy for rapidly-varying BCs. Without it, a multi-stage RK method would freeze the BC at the `t+dt` value and see `O(dt)` error at intermediate stages, silently dropping the order from 2 to 1.

### Multistep steppers fall through to the same code path

RK dispatch via `step_rk_imex!` → `step_subproblem_rk!` is the obvious path, but `CNAB1`/`CNAB2`/`SBDF1..4` also dispatch through the subproblem path whenever subproblems are available (`problem.parameters["subproblems"] !== nothing`). The dispatch happens inside each `step_<scheme>!` function: it computes the scheme's `(a, b, c)` coefficient tuples and calls the generic `step_subproblem_multistep!(state, solver, sps, a, b, c)`.

This means every IMEX stepper that supports the subproblem path gets DAE-correct BC handling automatically. The global-matrix path is still used when subproblem decomposition is unavailable, including pure-periodic/global systems. Inhomogeneous tau BCs should use the subproblem path; the global multistep path does not carry those algebraic BC values through the same override machinery.

### Why not just always use the override

The override path is only correct for algebraic rows with `F ≠ 0` that we want to enforce directly. Applying it to PDE rows (which have `M ≠ 0`) would break the time integration. The row classification is done once per subproblem at matrix-build time — `sp.bc_rows` lists exactly the rows where the override is needed, and nothing else.

## Exponential Time Differencing (ETD)

Exponential integrators solve the linear part of the ODE **exactly** using matrix exponentials, while treating nonlinear terms explicitly. This is fundamentally different from IMEX methods.

### How ETD Works

For the semi-linear ODE:

```math
\frac{du}{dt} = Lu + N(u)
```

where L is the linear operator and N(u) is the nonlinear term, ETD methods use the variation of constants formula:

```math
u(t_{n+1}) = e^{h L} u(t_n) + \int_0^h e^{(h-\tau)L} N(u(t_n + \tau)) d\tau
```

The key insight is that `exp(hL)` propagates the linear part **exactly**, eliminating stability constraints from stiff linear terms entirely.

### ETD vs IMEX

| Aspect | IMEX | ETD |
|--------|------|-----|
| Linear terms | Implicit (approximate) | Exponential (exact) |
| Linear solve | Yes, every step | No (but needs exp(hL)) |
| Stiffness | Limited by approximation | No limit from L |
| Best for | General stiff problems | Diagonal L (Fourier) |

### ETD_RK222

Second-order exponential Runge-Kutta method (Cox-Matthews 2002).

```julia
timestepper = ETD_RK222()
```

**Algorithm** (Cox-Matthews 2002, eq. 22):
```
Stage 1 (predictor): a_n = exp(hL) * u_n
                     c   = a_n + h * phi_1(hL) * N(u_n)
Stage 2 (corrector): u_{n+1} = c + h * phi_2(hL) * (N(c) - N(u_n))
```

where phi_1(z) = (exp(z) - 1) / z and phi_2(z) = (exp(z) - 1 - z) / z².

The corrector needs **both** `N(u_n)` and `N(c)`, and the difference must be weighted by
`phi_2` — a corrector that applies `phi_1` to `N(c)` alone is only first order.

- **Linear part**: Exact via exp(hL) and phi functions
- **Nonlinear part**: Explicit 2-stage Runge-Kutta
- **Accuracy**: O(dt^2)
- **Stability**: Unconditionally stable for linear terms
- **Use case**: Very stiff diffusion, reaction-diffusion

### ETD_CNAB2

Second-order exponential Adams-Bashforth method.

```julia
timestepper = ETD_CNAB2()
```

**Algorithm:**
```
w = h_n / h_{n-1}          (timestep ratio; w = 1 for constant dt)
u_{n+1} = exp(hL) * u_n
          + h * phi_1(hL) * N(u_n)
          + h * w * phi_2(hL) * (N(u_n) - N(u_{n-1}))
```

Despite the name, the linear term is **not** treated with Crank-Nicolson — "CNAB" refers only
to the two-step structure. Note that the history term carries `phi_2`, not `phi_1`: applying a
single `phi_1` to a plain AB2 extrapolation `(3/2)N_n − (1/2)N_{n−1}` is second order only in
the `hL → 0` limit, and loses accuracy exactly in the stiff regime ETD exists for.

- **Linear part**: Exact via exponential propagator
- **Nonlinear part**: Adams-Bashforth extrapolation
- **Accuracy**: O(dt^2)
- **Startup**: Uses ETD_RK222 for first step
- **Use case**: Long time integrations with stiff diffusion

### ETD_SBDF2

Second-order exponential multistep method.

```julia
timestepper = ETD_SBDF2()
```

**Algorithm:**
Uses the same implemented variable-step two-step exponential update as `ETD_CNAB2`:
```
w = h_n / h_{n-1}
u_{n+1} = exp(hL) * u_n + h * phi_1(hL) * N_n
          + h * w * phi_2(hL) * (N_n - N_{n-1})
```

This is the same second-order exponential-multistep update as `ETD_CNAB2` — both are the ETD
analogue of AB2, obtained by interpolating `N` linearly between `t_{n-1}` and `t_n` and
integrating the variation-of-constants formula exactly. The two schemes differ only in
bookkeeping; pick either.

- **Linear part**: Exact via exponential + phi functions
- **Nonlinear part**: BDF-style extrapolation
- **Accuracy**: O(dt^2)
- **Startup**: Uses ETD_RK222 for first step
- **Use case**: Stiff problems requiring multistep efficiency

### Phi Functions

ETD methods use phi functions defined as:

```math
\phi_0(z) = e^z
```
```math
\phi_1(z) = \frac{e^z - 1}{z}
```
```math
\phi_2(z) = \frac{e^z - 1 - z}{z^2}
```

Scalar arguments use a Taylor expansion near zero (`|z| < 1e-8`), where the `e^z − 1 − z`
numerator cancels catastrophically against the `z^2` denominator.

For the matrix case, `phi_functions_matrix(L, dt)` computes all three at once from the assembled
`L_matrix`, choosing its method by the operator norm `‖dt·L‖`:

| `‖dt·L‖` | method |
|---|---|
| `< 1e-8` | Taylor series |
| `< 50` | dense matrix exponential plus stable φ solves, with an eigendecomposition fallback when `z` is singular (the `k=0` mode of a pure diffusion operator makes it so) |
| `≥ 50` | Krylov approximation, column by column, via `ExponentialUtilities.phiv` |

Matrix ETD is rejected above size 4096 because it requires dense O(n²) storage — see the serial
size limit below.

The Krylov branch announces itself with `Warning: Matrix is large or stiff (norm=...), using
Krylov approximation`. It is not an error — a stiff `L` is the whole point of ETD, so any useful
ETD run will trip it (the 1-D heat example below has `‖dt·L‖ ≈ 269` and still reproduces the
analytic solution to 8 digits).

The result is cached and only recomputed when `dt` changes.

### When to Use ETD

**Ideal for:**
- Fourier-based spectral methods (L is diagonal)
- Very stiff diffusion (high viscosity/diffusivity)
- Reaction-diffusion equations
- Problems where IMEX requires very small dt

**Less suitable for:**
- Non-diagonal L (Chebyshev with BCs) - use IMEX instead
- Large serial problems (see the size limit below)
- Problems with complex boundary conditions

### Setup (none required)

You do **not** build or install a linear operator. `InitialValueSolver` runs
`build_solver_matrices!`, which assembles `L_matrix`, `M_matrix` and `F_vector` from the
equation strings and stores them in `problem.parameters`; the ETD steppers read `L_matrix`
from there (and use `M_matrix` when present). Anything you write on the **left-hand side** of the
equation is the linear operator L that gets propagated exactly; anything on the right-hand side is
the nonlinear term N. So writing `add_equation!(problem, "∂t(T) - nu*lap(T) = 0")` is all it takes
for `ETD_RK222()` to propagate `nu·∇²` with `exp(hL)` — see the complete example below.

Because the solver assembles these matrices itself, application code should **not** invent or
manually insert a separate `"L_matrix"`: a hand-built matrix will not share the solver's state
ordering, and the stepper will silently propagate the wrong operator.

### Size limit (serial)

In serial the exponential is taken of the **dense global** matrix, so the cost is O(n³) in the
total number of coefficients. Above `n = 4096` degrees of freedom the phi-function builder
refuses, and the stepper emits

> `Warning: ETD-RK222 failed: ArgumentError("ETD matrix exponential requires dense O(n²) storage
> but n=8193 is too large ..."), falling back to RK222`

and carries on with RK222. If you asked for ETD and see that warning, you are not getting ETD —
reduce the resolution or switch to SBDF2.

### Under MPI (pure-Fourier)

The serial size limit does **not** apply to a distributed pure-Fourier problem: a global dense
matrix exponential cannot run on distributed field vectors, so the distributed path steps each
Fourier mode with its own scalar exponential and never forms a global matrix (verified: a 128×128
Fourier heat problem, 8320 coefficients, steps at np=2 with no fallback).

The trade-off is that **all three** exported ETD types (`ETD_RK222`, `ETD_CNAB2`, `ETD_SBDF2`)
dispatch to the same distributed per-mode ETD-RK2 implementation on that path — the distinction
between the exponential-RK and the exponential-multistep variants is lost. If that distinction
matters to you, use the serial global-matrix path.

### Example: 1D Heat Equation with ETD

The linear part is propagated exactly, so for a purely linear problem ETD is exact at *any*
timestep — `dt = 0.1` here is far past the explicit diffusive stability limit and still lands on
the analytic answer.

```julia
using Tarang

# Fourier domain (L is diagonal)
coords  = CartesianCoordinates("x")
dist    = Distributor(coords; dtype=Float64, device=CPU())
x_basis = RealFourier(coords["x"]; size=64, bounds=(0.0, 2*pi))
domain  = Domain(dist, (x_basis,))

T = ScalarField(domain, "T")

problem = IVP([T])
add_parameters!(problem, nu=1.0)
add_equation!(problem, "∂t(T) - nu*lap(T) = 0")     # L is built from this line

x, = local_grids(dist, x_basis)
ensure_layout!(T, :g)
get_grid_data(T) .= sin.(x)
ensure_layout!(T, :c)

# ETD solver - no stability limit from diffusion!
solver = InitialValueSolver(problem, ETD_RK222(); dt=0.1)

for _ in 1:10          # t = 1.0
    step!(solver)
end

ensure_layout!(T, :g)
maximum(abs, get_grid_data(T))      # 0.36787944 == exp(-nu*1.0), to 8 digits
```

### References

- Cox, S. M., & Matthews, P. C. (2002). "Exponential Time Differencing for Stiff Systems". J. Comput. Phys. 176, 430-455.
- Hochbruck, M., & Ostermann, A. (2010). "Exponential integrators". Acta Numerica 19, 209-286.
- Kassam, A.-K., & Trefethen, L. N. (2005). "Fourth-Order Time Stepping for Stiff PDEs". SIAM J. Sci. Comput. 26(4), 1214-1233.

## Choosing a Timestepper

### By Problem Stiffness

| Stiffness | Indicator | Recommended |
|-----------|-----------|-------------|
| Mild | Explicit CFL dominates | RK222 |
| Moderate linear stiffness | Implicit linear solve permits a larger step | RK443, CNAB2, SBDF2, RKSMR |
| Smooth solution needing higher temporal order | Fixed/slowly varying step and adequate startup history | RK443, SBDF3, SBDF4 |
| Very stiff, manageable global matrix | Dense exponential is affordable | ETD_RK222, ETD_CNAB2, ETD_SBDF2 |
| Pure-Fourier diagonal linear operator | Per-mode implicit division is available | `DiagonalIMEX_RK222`, `DiagonalIMEX_RK443`, `DiagonalIMEX_SBDF2` |

### By Physics

| Problem Type | Recommended |
|--------------|-------------|
| General purpose | RK222, RK443 |
| Diffusion-dominated | CNAB2, SBDF2 |
| High Rayleigh number | RK222, RK443, or SBDF2 with CFL control |
| Turbulence | RK443, RKSMR, or SBDF2 with CFL control |
| Reaction-diffusion (Fourier) | ETD_RK222 |
| Very stiff diffusion (Fourier) | ETD_SBDF2 |

## Stability Analysis

### CFL Condition (Explicit)

For explicit methods, the timestep is limited by:

```math
\Delta t < \frac{C \cdot \Delta x}{u_{max}}
```

where C is the CFL number (typically 0.5-1.0).

### Diffusive Stability (IMEX)

IMEX methods allow larger timesteps by treating diffusion implicitly:

```math
\Delta t < \frac{C \cdot \Delta x}{u_{max}}
```

only constrained by advection, not diffusion.

!!! warning "Explicitly-treated diffusion still limits dt"
    That holds only for diffusion the timestepper actually treats implicitly. A
    **spatially varying** coefficient — most importantly an LES eddy viscosity νₑ —
    cannot go down the implicit path, so it is stepped explicitly and imposes

    ```math
    \Delta t \le \frac{1}{2 \nu_{max} \sum_i \Delta x_i^{-2}}
    ```

    (equivalently `Δx²/(2dν)` on an isotropic `d`-dimensional grid). The `CFL`
    controller does **not** enforce this unless you register the coefficient with
    `add_diffusivity!(cfl, get_eddy_viscosity(model))`.

## Adaptive Time Stepping

### With CFL

`CFL` is constructed from the **solver** (not the problem), and velocities are registered as
`VectorField`s. The canonical use is to hand the controller to `run!`, which recomputes `dt`
every `cadence` iterations and updates `solver.dt` for you:

```julia
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

cfl = CFL(solver; initial_dt=1e-3, cadence=5, safety=0.4, max_dt=0.01)
add_velocity!(cfl, u)                      # u must be a VectorField

run!(solver; stop_iteration=10, cfl=cfl, progress=false)
solver.dt                                  # updated by the controller
```

If you drive the loop yourself, call `compute_timestep` and pass the result to `step!`:

```julia
while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
end
```

The remaining knobs are `threshold`, `max_change` and `min_change` (all *ratios* limiting how
fast `dt` may change between recomputations). There is no `min_dt` field; the current step is
`cfl.current_dt`.

Registering velocities alone gives an **advection-only** step. Add
`add_diffusivity!(cfl, ν)` for anything diffusive you integrate explicitly — see
[Diffusive Limit](solvers.md#Diffusive-Limit-for-Explicit-Diffusion).

### Manual Adjustment

```julia
# Reduce timestep if solver becomes unstable
if any(isnan, get_grid_data(field))
    solver.dt *= 0.5
    error("NaN detected, reduce timestep")
end
```

## Multi-Step Methods

SBDF and CNAB methods require previous time level data:

```julia
# First step uses lower order
# SBDF2 uses SBDF1 for first step
# Automatic in Tarang

solver = InitialValueSolver(problem, SBDF2(); dt=0.001)
# First step: SBDF1
# Subsequent steps: SBDF2
```

## Performance Comparison

The RK counts below are the number of stages actually driven per step. Note that `RK222` is a
**three**-stage ESDIRK tableau (explicit first stage), so it costs three RHS evaluations, not two.

| Method | RHS evaluations/step | Implicit/exponential work | Memory |
|--------|----------------------|---------------------------|--------|
| RK111 | 1 | 1 implicit stage solve | Medium |
| RK222 | 3 | 2 implicit stage solves (the first stage is explicit) | Medium |
| RK443 | 4 | 3 implicit stage solves | Higher |
| RKSMR | 4 | 3 implicit stage solves | Higher |
| CNAB2 | 1 | 1 implicit solve after startup | Medium |
| SBDF2 | 1 | 1 implicit solve after startup | Medium |
| ETD_RK222 | 2 | cached `exp(hL)`, `phi_1`, and `phi_2` actions | Dense n×n (serial) |
| ETD_CNAB2 / ETD_SBDF2 | 1 | cached exponential/φ actions after ETD-RK2 startup | Dense n×n (serial) |

The implicit factorization is cached and reused while `dt` and the operator are unchanged, so the
factorization is paid once, not per step. Two consequences:

- **Adaptive steps can cost more than fixed steps**, because changing `dt` invalidates the cached
  factorization. (This is why `CFL` has a `threshold` ratio — it refuses to commit a new `dt` for
  a change smaller than that, rather than refactorizing every recomputation.)
- For ETD, the O(n³) exponential is likewise paid once per distinct `dt`, but the dense n×n
  storage is permanent — which is why serial ETD is capped at n = 4096 coefficients.

## Example Usage

The timestepper is the only thing that changes between the three families — the problem, the
fields and the loop are identical. This 2-D advection-diffusion problem runs with any of them:

```julia
using Tarang

coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb     = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi), dealias=3/2)
yb     = RealFourier(coords["y"]; size=16, bounds=(0.0, 2pi), dealias=3/2)
domain = Domain(dist, (xb, yb))

s = ScalarField(domain, "s")
u = VectorField(domain, "u")

problem = IVP([s, u])
add_parameters!(problem, nu=0.05)
add_equation!(problem, "∂t(s) - nu*lap(s) = -u⋅∇(s)")   # LHS implicit, RHS explicit
add_equation!(problem, "∂t(u) - nu*lap(u) = 0")

set!(s, (x, y) -> sin(x) * cos(y))
set!(u.components[1], (x, y) -> 0.5)

# Pick one:
#   RK443()      -- IMEX Runge-Kutta, stiff diffusion handled implicitly
#   SBDF2()      -- multistep IMEX, one RHS evaluation per step
#   ETD_RK222()  -- exponential, no stability limit from the linear term
solver = InitialValueSolver(problem, RK443(); dt=1e-3)

t_end = 0.005
while solver.sim_time < t_end
    step!(solver)
end
```

Swapping `RK443()` for `SBDF2()` or `ETD_RK222()` changes nothing else and, at this timestep,
agrees to 6 digits (`max|s| = 0.999497` after 5 steps). In particular, the ETD stepper needs no
extra setup: the problem equations already define the assembled linear operator it propagates.

## Stochastic Forcing

For turbulence simulations requiring external energy injection, see [Stochastic Forcing](stochastic_forcing.md).

Key points:
- Forcing is generated once per timestep, constant across substeps
- Required for proper Stratonovich calculus treatment
- Supports ring, isotropic, and custom forcing spectra

Register the forcing on the **problem**, against the variable it drives, before building the
solver — `add_stochastic_forcing!` is the supported entry point, and the stepper picks the
forcing up from there.

```julia
s = ScalarField(domain, "s")                      # starts at zero

problem = IVP([s])
add_parameters!(problem, nu=0.01)
add_equation!(problem, "∂t(s) - nu*lap(s) = 0")   # no explicit source term

forcing = StochasticForcing(
    field_size   = (16, 16),        # grid shape of the forced field
    domain_size  = (2pi, 2pi),
    forcing_rate = 0.1,
    k_forcing    = 4.0,
    dk_forcing   = 1.0,
    dt           = 1e-3,
)
add_stochastic_forcing!(problem, :s, forcing)     # :s is the variable's name

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
run!(solver; stop_iteration=20, progress=false)   # forcing drives s off zero (max|s| ~ 4e-3)
```

The magnitude varies from run to run — the forcing is white in time. Pass an explicit `rng=` to
`StochasticForcing` if you need a reproducible realization.

## See Also

- [Solvers](solvers.md): Using time steppers with solvers
- [CFL and Analysis](analysis.md): Adaptive time stepping
- [Stochastic Forcing](stochastic_forcing.md): External forcing for turbulence
- [API: Timesteppers](../api/timesteppers.md): Complete reference
