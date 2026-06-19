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

2nd-order IMEX Runge-Kutta with three stored stages: an explicit first stage and two diagonally implicit stages.

```julia
RK222()
```

**Properties**:
- Order: 2
- Stages: 3 (`c = [0, γ, 1]`)
- Implicit part: ESDIRK with `γ = 1 - 1/sqrt(2)` on stages 2 and 3
- Explicit part: matched three-row explicit tableau
- Workspace field sets: 6 (stage state, explicit RHS, and implicit RHS storage)

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

### RKSMR

```julia
RKSMR()
```

The Spalart–Moser–Rogers scheme is represented as a four-stage additive RK
tableau. Its nonlinear/explicit part is third order and its linear/implicit part
is second order. It uses the same generic IMEX runtime paths as `RK222` and
`RK443`.

### Diagonal IMEX

```julia
DiagonalIMEX_RK222()  # 2nd order
DiagonalIMEX_RK443()  # 3rd order
DiagonalIMEX_SBDF2()  # 2nd order multistep
```

**Properties**: the implicit solve is **diagonal in spectral space** — the linear
operator is applied per Fourier mode (`(I + γ·dt·L̂)⁻¹` per wavenumber) rather than
through a global matrix solve. Much cheaper for pure-Fourier / diagonalizable
linear terms; requires the implicit operator to be diagonal in the chosen basis.

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

### Advective CFL

Explicitly treated advection/nonlinear terms still limit the timestep:

```math
\Delta t < C \frac{\Delta x}{u_{max}}
```

The admissible CFL factor depends on the PDE, spatial discretization, and chosen
scheme; Tarang does not define universal constants for individual steppers.

### IMEX Methods

Implicit linear treatment removes the corresponding explicit diffusive limit,
but it does not remove the advective CFL limit:

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
| Smooth high-order integration | RK443 or SBDF3/SBDF4 | Higher formal order; multistep methods require startup history |
| Classic incompressible DNS IMEX | RKSMR | SMR explicit-third/implicit-second accuracy profile |
| Diagonal Fourier linear operator | Diagonal IMEX family | Per-mode implicit division avoids a global sparse solve |

## Performance

### Computational Cost

| Method | RHS Evaluations | Implicit Solves |
|--------|-----------------|-----------------|
| RK111 | 1 | 1 |
| RK222 | 3 | 2 |
| RK443 | 4 | 3 |
| RKSMR | 4 | 3 |
| CNAB2 | 1 | 1 after startup |
| SBDF2 | 1 | 1 after startup |

### Memory Requirements

Workspace storage is preallocated by timestepper type. The RK implementations
reserve three field sets per stage (`X_stage`, explicit RHS, implicit RHS), while
multistep methods use history rings and two general workspace field sets. These
are reusable buffers, not fresh state copies allocated on every step.

## See Also

- [Solvers](solvers.md): Using timesteppers
- [User Guide: Timesteppers](../pages/timesteppers.md): Conceptual guide
