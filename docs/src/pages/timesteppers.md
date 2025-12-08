# Time Steppers

Time steppers integrate PDEs forward in time.

## Overview

Tarang.jl provides several families of time integration schemes:

| Family | Type | Best For |
|--------|------|----------|
| RK | IMEX Runge-Kutta | General problems |
| CNAB | Multistep IMEX | Moderately stiff |
| SBDF | Multistep IMEX | Stiff problems |

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

## Choosing a Timestepper

### By Problem Stiffness

| Stiffness | Indicator | Recommended |
|-----------|-----------|-------------|
| Mild | Moderate Re | RK222, RK443 |
| Moderate | Higher Re | CNAB2, SBDF2 |
| Stiff | High Re, requires tiny Δt | SBDF3, SBDF4 |

### By Physics

| Problem Type | Recommended |
|--------------|-------------|
| General purpose | RK222, RK443 |
| Diffusion-dominated | CNAB2, SBDF2 |
| High Rayleigh number | SBDF2, SBDF3 |
| Turbulence | RK443 or SBDF2 |

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
