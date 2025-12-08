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

Following Dedalus convention, all RK methods use Additive Runge-Kutta (ARK) schemes:
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

Second-order, 2-stage IMEX Runge-Kutta (Ascher, Ruuth, Spiteri 1997).

```julia
timestepper = RK222()
```

- **Implicit part**: 2-stage SDIRK (γ = 1 - 1/√2)
- **Explicit part**: 2-stage explicit RK
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
Stage 2 (corrector): u_{n+1} = a_n + h * phi_1(hL) * N(c)
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
u_{n+1} = exp(hL) * u_n + h * phi_1(hL) * N_AB2
N_AB2 = (3/2) * N(u_n) - (1/2) * N(u_{n-1})
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
Uses proper 2-step exponential integration with phi_1 and phi_2 functions:
```
u_{n+1} = exp(hL) * u_n + h * phi_1(hL) * N_n + h^2 * phi_2(hL) * (N_n - N_{n-1}) / h
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

These are computed stably even for small z using Taylor series or Krylov methods.

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

ETD methods require:
1. The linear operator L stored as `"L_matrix"` in problem parameters
2. Optionally, mass matrix M as `"M_matrix"`

```julia
# Setup L matrix for ETD
problem.parameters["L_matrix"] = build_linear_operator(domain)

# Use ETD solver
solver = InitialValueSolver(problem, ETD_RK222(); dt=1e-2)
```

### Example: 1D Heat Equation with ETD

```julia
using Tarang

# Setup Fourier domain (L is diagonal)
coords = CartesianCoordinates("x")
dist = Distributor(coords; mesh=(1,))
x_basis = RealFourier(coords["x"]; size=128, bounds=(0.0, 2*pi))
domain = Domain(dist, (x_basis,))

# Temperature field
T = ScalarField(dist, "T", domain.bases, Float64)

# Build diagonal diffusion operator in Fourier space
# For Fourier: d^2/dx^2 -> -k^2
kx = wavenumbers(x_basis)
L_diag = -nu * kx.^2  # Diagonal in spectral space
problem.parameters["L_matrix"] = Diagonal(L_diag)

# ETD solver - no stability limit from diffusion!
solver = InitialValueSolver(problem, ETD_RK222(); dt=0.1)

while solver.sim_time < t_end
    step!(solver)
end
```

### References

- Cox, S. M., & Matthews, P. C. (2002). "Exponential Time Differencing for Stiff Systems". J. Comput. Phys. 176, 430-455.
- Hochbruck, M., & Ostermann, A. (2010). "Exponential integrators". Acta Numerica 19, 209-286.
- Kassam, A.-K., & Trefethen, L. N. (2005). "Fourth-Order Time Stepping for Stiff PDEs". SIAM J. Sci. Comput. 26(4), 1214-1233.

## Choosing a Timestepper

### By Problem Stiffness

| Stiffness | Indicator | Recommended |
|-----------|-----------|-------------|
| Mild | Moderate Re | RK222, RK443 |
| Moderate | Higher Re | CNAB2, SBDF2 |
| Stiff | High Re, requires tiny Δt | SBDF3, SBDF4 |
| Very stiff (diagonal L) | Extreme diffusion | ETD_RK222, ETD_SBDF2 |

### By Physics

| Problem Type | Recommended |
|--------------|-------------|
| General purpose | RK222, RK443 |
| Diffusion-dominated | CNAB2, SBDF2 |
| High Rayleigh number | SBDF2, SBDF3 |
| Turbulence | RK443 or SBDF2 |
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
cfl = CFL(problem, safety=0.5)
add_velocity!(cfl, u)

while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
end
```

### Manual Adjustment

```julia
# Reduce timestep if solver becomes unstable
if any(isnan, field.data_g)
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

| Method | Evaluations/Step | Memory | Stability |
|--------|------------------|--------|-----------|
| RK111 | 1 + solve | Medium | Good (L-stable) |
| RK222 | 2 + solve | Medium | Very good (L-stable) |
| RK443 | 4 + solve | Higher | Best (L-stable) |
| CNAB2 | 1 + solve | Medium | Very good |
| SBDF2 | 1 + solve | Medium | Excellent |
| ETD_RK222 | 2 + exp(hL) | Medium | Exact (linear) |
| ETD_CNAB2 | 1 + exp(hL) | Medium | Exact (linear) |
| ETD_SBDF2 | 1 + exp(hL) | Medium | Exact (linear) |

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

# For Fourier-based problems with diagonal L
# Build the linear operator matrix
L_matrix = build_diffusion_operator(domain)  # Diagonal in spectral space
problem.parameters["L_matrix"] = L_matrix

# ETD solver - no stability limit from stiff linear terms!
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
