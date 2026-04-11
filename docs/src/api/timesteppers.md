# Timesteppers API

Time integration schemes for evolving PDEs in time.

## TimeStepper Types

### Abstract Type

```julia
abstract type TimeStepper end
```

All timesteppers inherit from this abstract type.

## IMEX Runge-Kutta

### RK111

1st-order IMEX Runge-Kutta.

```julia
RK111()
```

**Properties**:
- Order: 1
- Stages: 1
- Implicit part: Backward Euler for linear terms
- Explicit part: Forward Euler for nonlinear terms
- Memory: Minimal

### RK222

2nd-order, 2-stage IMEX Runge-Kutta.

```julia
RK222()
```

**Properties**:
- Order: 2
- Stages: 2
- Implicit part: 2-stage DIRK for linear terms
- Explicit part: 2-stage explicit RK for nonlinear terms
- Memory: 2 state copies

**Recommended for**: General purpose problems.

### RK443

3rd-order, 4-stage IMEX Runge-Kutta.

```julia
RK443()
```

**Properties**:
- Order: 3
- Stages: 4
- Implicit part: 4-stage DIRK for linear terms
- Explicit part: 4-stage explicit RK for nonlinear terms
- Memory: 4 state copies

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

## Usage with Solver

```julia
# Create solver with timestepper
solver = InitialValueSolver(problem, RK222(); dt=0.001)

# Or IMEX
solver = InitialValueSolver(problem, SBDF2(); dt=0.01)

# Time stepping
while solver.sim_time < t_end
    step!(solver)
end
```

## Timestepper State

### TimestepperState

Internal state for multi-step methods.

```julia
struct TimestepperState
    timestepper::TimeStepper
    dt::Float64
    history::Vector  # Previous time levels
end
```

Managed automatically by the solver.

## Stability Regions

### Explicit Methods

CFL condition limits timestep:

```math
\Delta t < C \frac{\Delta x}{u_{max}}
```

- RK111: C ≈ 1.0
- RK222: C ≈ 1.0
- RK443: C ≈ 1.4

### IMEX Methods

Diffusive stability removed:

```math
\Delta t < C \frac{\Delta x}{u_{max}}
```

Only advective CFL, not diffusive.

## Method Selection Guide

| Problem Type | Method | Reason |
|--------------|--------|--------|
| General purpose | RK222 | Balance of cost/accuracy |
| High accuracy | RK443 | More stages, higher order |
| Diffusion-dominated | SBDF2 | Implicit diffusion |
| Very stiff | SBDF3/SBDF4 | Strong stability |

## Performance

### Computational Cost

| Method | RHS Evaluations | Linear Solves |
|--------|-----------------|---------------|
| RK111 | 1 | 0 |
| RK222 | 2 | 0 |
| RK443 | 4 | 0 |
| CNAB2 | 1 | 1 |
| SBDF2 | 1 | 1 |

### Memory Requirements

| Method | State Copies |
|--------|--------------|
| RK111 | 1 |
| RK222 | 2 |
| RK443 | 4 |
| SBDF2 | 2 (history) |
| SBDF4 | 4 (history) |

## See Also

- [Solvers](solvers.md): Using timesteppers
- [User Guide: Timesteppers](../pages/timesteppers.md): Conceptual guide
