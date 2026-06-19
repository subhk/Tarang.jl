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

```julia
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

**Algorithm:**
```
Stage 1 (predictor): a_n = exp(hL) * u_n
                     c = a_n + h * phi_1(hL) * N(u_n)
Stage 2 (corrector): u_{n+1} = c + h * phi_2(hL) * (N(c) - N(u_n))
```

where phi_1(z) = (exp(z) - 1) / z

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
w = h_n / h_{n-1}
u_{n+1} = exp(hL) * u_n
          + h * phi_1(hL) * N(u_n)
          + h * w * phi_2(hL) * (N(u_n) - N(u_{n-1}))
```

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

Scalar and small-norm matrix arguments use Taylor expansions near zero. Moderate dense matrices use a matrix exponential plus stable φ solves (with an eigen fallback for singular operators); sufficiently stiff matrices use the Krylov implementation. Matrix ETD is rejected above size 4096 because it requires dense storage.

### When to Use ETD

**Ideal for:**
- Fourier-based spectral methods (L is diagonal)
- Very stiff diffusion (high viscosity/diffusivity)
- Reaction-diffusion equations
- Problems where IMEX requires very small dt

**Less suitable for:**
- Non-diagonal L (Chebyshev with BCs) - use IMEX instead
- Large dense systems (exp(hL) is expensive)
- Problems with complex boundary conditions

### Requirements

ETD methods require the solver's assembled `"L_matrix"`; an assembled `"M_matrix"` is used when present. `InitialValueSolver` builds these matrices from the problem equations, so application code should not invent or manually insert a separate matrix with a different state ordering.

```julia
# After defining the IVP equations, choose an ETD stepper normally.
solver = InitialValueSolver(problem, ETD_RK222(); dt=1e-2)
```

For MPI pure-Fourier problems, the three exported ETD types currently dispatch to the distributed per-mode ETD-RK2 implementation because a global dense matrix exponential cannot run on distributed field vectors. If retaining the exact distinction between ETD multistep variants matters, use the serial global-matrix path.

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

## Adaptive Time Stepping

### With CFL

```julia
cfl = CFL(solver; initial_dt=solver.dt, safety=0.5)
add_velocity!(cfl, u)

while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
end
```

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

| Method | RHS evaluations/step | Implicit/exponential work |
|--------|----------------------|---------------------------|
| RK111 | 1 | 1 implicit stage solve |
| RK222 | 3 | 2 implicit stage solves (the first stage is explicit) |
| RK443 | 4 | 3 implicit stage solves |
| RKSMR | 4 | 3 implicit stage solves |
| CNAB2 | 1 | 1 implicit solve after startup |
| SBDF2 | 1 | 1 implicit solve after startup |
| ETD_RK222 | 2 | cached `exp(hL)`, `phi_1`, and `phi_2` actions |
| ETD_CNAB2 / ETD_SBDF2 | 1 | cached exponential/φ actions after ETD-RK2 startup |

The implicit factorization is cached and reused while `dt` and the operator are unchanged. Adaptive steps can therefore cost more than fixed steps because changing `dt` invalidates the cached factorization.

## Example Usage

### Runge-Kutta IMEX

```julia
using Tarang

# Problem setup
problem = IVP([u, p])
# ... add equations ...

# IMEX RK solver - handles stiff diffusion implicitly
solver = InitialValueSolver(problem, RK443(); dt=1e-3)

# Larger timestep possible due to implicit treatment of linear terms
while solver.sim_time < t_end
    step!(solver)
end
```

### Multistep IMEX (SBDF)

```julia
# Same problem, multistep IMEX solver
solver = InitialValueSolver(problem, SBDF2(); dt=1e-3)

# Larger timestep possible
while solver.sim_time < t_end
    step!(solver)
end
```

### Exponential Time Differencing (ETD)

```julia
using LinearAlgebra

# The problem equations define the assembled linear operator.
solver = InitialValueSolver(problem, ETD_RK222(); dt=0.1)

# Can use much larger timesteps when L is very stiff
while solver.sim_time < t_end
    step!(solver)
end
```

## Stochastic Forcing

For turbulence simulations requiring external energy injection, see [Stochastic Forcing](stochastic_forcing.md).

Key points:
- Forcing is generated once per timestep, constant across substeps
- Required for proper Stratonovich calculus treatment
- Supports ring, isotropic, and custom forcing spectra

```julia
# Quick example
forcing = StochasticForcing(
    field_size = (64, 64),
    forcing_rate = 0.1,
    k_forcing = 4.0,
    dt = solver.dt
)
set_forcing!(solver.timestepper_state, forcing)
```

## See Also

- [Solvers](solvers.md): Using time steppers with solvers
- [CFL and Analysis](analysis.md): Adaptive time stepping
- [Stochastic Forcing](stochastic_forcing.md): External forcing for turbulence
- [API: Timesteppers](../api/timesteppers.md): Complete reference
