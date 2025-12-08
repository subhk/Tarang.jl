# Stochastic Forcing

Stochastic forcing is essential for studying turbulence, where energy must be continuously injected to maintain a statistically steady state. Tarang.jl provides a stochastic forcing implementation following the approach used in [GeophysicalFlows.jl](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/stochastic_forcing/).

## Overview

The forcing ξ(x,t) is:
- **White in time**: decorrelated between timesteps
- **Spatially correlated**: determined by a forcing spectrum Q(k)
- **Zero mean**: ⟨ξ(x,t)⟩ = 0

The correlation structure is:
```math
\langle \xi(\mathbf{x},t) \xi(\mathbf{x}',t') \rangle = Q(\mathbf{x}-\mathbf{x}') \delta(t-t')
```

## Key Design: Constant Forcing Within Timesteps

**Critical for correctness**: When using multi-stage time integrators (RK2, RK4, SBDF), the forcing must stay constant across all substeps within a single timestep. This is required for proper Stratonovich calculus treatment.

```
Timestep n:     [substep 1] → [substep 2] → [substep 3] → [substep 4]
                     ↑              ↑              ↑              ↑
Forcing:          F_n           F_n           F_n           F_n    (SAME!)

Timestep n+1:   [substep 1] → [substep 2] → [substep 3] → [substep 4]
                     ↑              ↑              ↑              ↑
Forcing:        F_{n+1}       F_{n+1}       F_{n+1}       F_{n+1}  (NEW, but constant)
```

Tarang.jl handles this automatically in the timestepper.

## Creating Stochastic Forcing

### Basic Usage

```julia
using Tarang

# Create stochastic forcing for a 2D field
forcing = StochasticForcing(
    field_size = (64, 64),    # Grid dimensions
    forcing_rate = 0.1,        # Energy injection rate ε
    k_forcing = 4.0,           # Central forcing wavenumber
    dk_forcing = 2.0,          # Bandwidth around k_forcing
    dt = 0.001                 # Timestep (for Stratonovich scaling)
)
```

### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `field_size` | Tuple | Size of the field (Nx,) or (Nx, Ny) or (Nx, Ny, Nz) | Required |
| `forcing_rate` | Real | Energy injection rate ε | 1.0 |
| `k_forcing` | Real | Central forcing wavenumber | 4.0 |
| `dk_forcing` | Real | Forcing bandwidth | 2.0 |
| `dt` | Real | Timestep for Stratonovich scaling | 0.01 |
| `is_stochastic` | Bool | Enable stochastic forcing | true |
| `spectrum_type` | Symbol | Type of forcing spectrum | `:ring` |
| `spectrum` | Array | Custom spectrum (overrides spectrum_type) | nothing |
| `rng` | AbstractRNG | Random number generator | GLOBAL_RNG |
| `dtype` | Type | Floating point type | Float64 |

## Forcing Spectrum Types

### Ring Forcing (`:ring`)

Forces modes in an annulus around `k_forcing`:

```julia
forcing = StochasticForcing(
    field_size = (128, 128),
    k_forcing = 8.0,
    dk_forcing = 2.0,
    spectrum_type = :ring
)
```

This forces all modes with `|k_forcing - dk_forcing| < |k| < |k_forcing + dk_forcing|`.

### Isotropic Forcing (`:isotropic`)

Gaussian envelope centered on `k_forcing`:

```julia
forcing = StochasticForcing(
    field_size = (128, 128),
    k_forcing = 8.0,
    dk_forcing = 2.0,
    spectrum_type = :isotropic
)
```

The spectrum follows:
```math
\sqrt{Q(k)} \propto \exp\left(-\frac{(|k| - k_f)^2}{2 \, \Delta k^2}\right)
```

### Bandlimited Forcing (`:bandlimited`)

Sharp cutoff band forcing:

```julia
forcing = StochasticForcing(
    field_size = (128, 128),
    spectrum_type = :bandlimited
)
```

### Kolmogorov Forcing (`:kolmogorov`)

Large-scale forcing typical for turbulence studies:

```julia
forcing = StochasticForcing(
    field_size = (128, 128),
    k_forcing = 4.0,
    spectrum_type = :kolmogorov
)
```

### Custom Spectrum

You can provide your own spectrum:

```julia
# Custom spectrum array
custom_spectrum = zeros(64, 64)
# ... fill in your spectrum ...

forcing = StochasticForcing(
    field_size = (64, 64),
    spectrum = custom_spectrum
)
```

## Integrating with Timesteppers

### Attaching Forcing to Timestepper

```julia
# Create problem and solver
problem = IVP([vorticity])
# ... add equations ...

solver = InitialValueSolver(problem, RK443(); dt=0.001)

# Create and attach forcing
forcing = StochasticForcing(
    field_size = size(vorticity.data_c),
    forcing_rate = 0.1,
    k_forcing = 4.0,
    dk_forcing = 2.0,
    dt = solver.dt
)

# Attach to timestepper state
set_forcing!(solver.timestepper_state, forcing)
```

### Automatic Forcing Updates

Once attached, forcing is handled automatically:
- New forcing generated at the START of each `step!()`
- Forcing remains constant during all substeps
- Reset at the END of each `step!()` for next timestep

```julia
while solver.sim_time < t_end
    step!(solver)  # Forcing handled automatically
end
```

### Manual Forcing Application

For custom timestepping, you can control forcing manually:

```julia
# Get current cached forcing
F = get_cached_forcing(solver.timestepper_state)

# Apply to a field (in spectral space)
apply_forcing!(field_spectral, forcing, current_time)

# Or with substep awareness
apply_forcing!(field_spectral, forcing, current_time, substep_number)
```

## Generating Forcing

### Basic Generation

```julia
# Generate new forcing at time t
F = generate_forcing!(forcing, t)
```

### Substep-Aware Generation

```julia
# Substep 1: generates new forcing
F1 = generate_forcing!(forcing, t, 1)

# Substeps 2, 3, 4: return cached forcing (no regeneration)
F2 = generate_forcing!(forcing, t, 2)  # Same as F1
F3 = generate_forcing!(forcing, t, 3)  # Same as F1
F4 = generate_forcing!(forcing, t, 4)  # Same as F1

# New timestep: generates new forcing
F_new = generate_forcing!(forcing, t + dt, 1)  # Different from F1
```

## Diagnostics

### Energy Injection Rate

The mean energy injection rate:

```julia
ε = energy_injection_rate(forcing)
println("Mean energy injection rate: $ε")
```

### Instantaneous Power

Compute instantaneous power input to a field:

```julia
P = instantaneous_power(forcing, vorticity_spectral)
println("Instantaneous power: $P")
```

## Utility Functions

### Update Timestep

If the timestep changes during simulation:

```julia
set_dt!(forcing, new_dt)
```

### Reset Forcing

Force regeneration on next access:

```julia
reset_forcing!(forcing)
```

### Get Forcing in Real Space

For visualization (requires manual FFT):

```julia
F_real = get_forcing_real(forcing)  # Returns real part of cached forcing
```

## Complete Example: Forced 2D Turbulence

```julia
using Tarang
using Random

# Set random seed for reproducibility
Random.seed!(12345)

# Domain setup
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords)

Nx, Ny = 256, 256
Lx, Ly = 2π, 2π

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0, Lx))
ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0, Ly))

domain = Domain(dist, (xbasis, ybasis))

# Vorticity field
ω = ScalarField(dist, "omega", (xbasis, ybasis))

# Stochastic forcing at intermediate scales
forcing = StochasticForcing(
    field_size = (Nx, Ny),
    forcing_rate = 0.1,       # Energy injection rate
    k_forcing = 8.0,          # Force at k ~ 8
    dk_forcing = 2.0,         # Bandwidth
    dt = 1e-3,
    spectrum_type = :ring,
    rng = MersenneTwister(42)  # For reproducibility
)

# Problem setup
problem = IVP([ω])
# ... add vorticity equation with forcing ...

# Solver with forcing attached
solver = InitialValueSolver(problem, RK443(); dt=1e-3)
set_forcing!(solver.timestepper_state, forcing)

# Time integration
t_end = 10.0
while solver.sim_time < t_end
    step!(solver)

    # Periodic diagnostics
    if mod(solver.iteration, 100) == 0
        ε = energy_injection_rate(forcing)
        P = instantaneous_power(forcing, ω.data_c)
        println("t = $(solver.sim_time), ε = $ε, P = $P")
    end
end
```

## Deterministic Forcing

For comparison or testing, deterministic forcing is also available:

```julia
# Define forcing function
function my_forcing(x, y, t, params)
    A = params[:amplitude]
    k = params[:wavenumber]
    return A * sin.(k * x) .* cos.(k * y)
end

# Create deterministic forcing
det_forcing = DeterministicForcing(
    my_forcing,
    (64, 64);
    parameters = Dict{Symbol, Any}(
        :amplitude => 0.1,
        :wavenumber => 4.0
    )
)
```

## Mathematical Background

### Stratonovich Calculus

For stochastic differential equations of the form:
```math
\frac{\partial u}{\partial t} = \mathcal{L}[u] + \mathcal{N}[u] + \xi
```

Using Stratonovich calculus, the numerical forcing is:
```math
F = \sqrt{\frac{Q}{\Delta t}} \cdot e^{2\pi i \phi}
```

where φ is uniformly distributed random phase.

### Energy Balance

In statistically steady state:
```math
\varepsilon_{forcing} = \varepsilon_{dissipation}
```

The energy injection rate is controlled by the `forcing_rate` parameter.

## References

1. GeophysicalFlows.jl documentation: [Stochastic Forcing](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/stochastic_forcing/)
2. Constantinou, N. C., & Hogg, A. M. (2021). "Intrinsic oceanic decadal variability of upper-ocean heat content"

## See Also

- [Timesteppers](timesteppers.md): Time integration methods
- [Solvers](solvers.md): Using forcing with IVP solvers
- [API: Timesteppers](../api/timesteppers.md): Complete API reference
