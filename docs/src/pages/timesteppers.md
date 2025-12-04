# Time Steppers

Time steppers integrate PDEs forward in time.

## Overview

Tarang.jl provides several families of time integration schemes:

| Family | Type | Best For |
|--------|------|----------|
| RK | Explicit | Non-stiff problems |
| CNAB | IMEX | Moderately stiff |
| SBDF | IMEX | Stiff problems |

## Explicit Runge-Kutta

### RK111 (Forward Euler)

First-order explicit method.

```julia
timestepper = RK111()
```

- **Stability**: CFL limited
- **Accuracy**: O(Δt)
- **Use case**: Testing, simple problems

### RK222

Second-order, 2-stage Runge-Kutta.

```julia
timestepper = RK222()
```

- **Stability**: Better than RK111
- **Accuracy**: O(Δt²)
- **Use case**: General purpose (recommended)

### RK443

Fourth-order, 4-stage Runge-Kutta.

```julia
timestepper = RK443()
```

- **Stability**: Good
- **Accuracy**: O(Δt⁴)
- **Use case**: High accuracy requirements

## IMEX Methods

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
| Non-stiff | Low Re, large Δt stable | RK222, RK443 |
| Mild | Moderate Re | CNAB2, SBDF2 |
| Stiff | High Re, requires tiny Δt | SBDF3, SBDF4 |

### By Physics

| Problem Type | Recommended |
|--------------|-------------|
| Advection-dominated | RK222, RK443 |
| Diffusion-dominated | CNAB2, SBDF2 |
| High Rayleigh number | SBDF2, SBDF3 |
| Turbulence | RK443 (explicit) or SBDF2 |

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
| RK111 | 1 | Low | Poor |
| RK222 | 2 | Medium | Good |
| RK443 | 4 | Higher | Best (explicit) |
| CNAB2 | 1 | Medium | Very good |
| SBDF2 | 1 | Medium | Excellent |

## Example Usage

### Explicit (RK)

```julia
using Tarang

# Problem setup
problem = IVP([u, p])
# ... add equations ...

# Explicit solver
solver = InitialValueSolver(problem, RK443(); dt=1e-4)

# Small timestep required
while solver.sim_time < t_end
    step!(solver)
end
```

### IMEX (SBDF)

```julia
# Same problem, IMEX solver
solver = InitialValueSolver(problem, SBDF2(); dt=1e-3)

# Larger timestep possible
while solver.sim_time < t_end
    step!(solver)
end
```

## See Also

- [Solvers](solvers.md): Using time steppers with solvers
- [CFL and Analysis](analysis.md): Adaptive time stepping
- [API: Timesteppers](../api/timesteppers.md): Complete reference
