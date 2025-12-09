# Stochastic Forcing

> **TL;DR**: Stochastic forcing injects energy at a controlled rate into your simulation. Create it with `StochasticForcing(field_size=(Nx, Ny), energy_injection_rate=ε, k_forcing=k_f)`, then let the timestepper handle it automatically.

---

## Quick Start

```julia
using Tarang

# 1. Create stochastic forcing
forcing = StochasticForcing(
    field_size = (256, 256),        # Grid dimensions
    energy_injection_rate = 0.1,    # Target ε
    k_forcing = 10.0,               # Force at |k| ≈ 10
    dk_forcing = 2.0,               # Bandwidth
    dt = 0.001                      # Timestep
)

# 2. In your time loop
store_prevsol!(forcing, ψ_hat)         # Store ψⁿ
generate_forcing!(forcing, t, 1)        # Generate F̂
apply_forcing!(rhs_hat, forcing, t, 1)  # Add to RHS
# ... advance solution ...
W = work_stratonovich(forcing, ψ_hat)   # Compute work done
```

---

## Why Stochastic Forcing?

In turbulence simulations, energy cascades from large to small scales (3D) or from injection to both ends (2D). Without continuous energy input, the flow dies out. Stochastic forcing:

- Maintains a **statistically steady state**
- Provides a **controlled energy injection rate** ε
- Creates **physically realistic** turbulent fluctuations
- Is **white in time** (decorrelated between timesteps)

---

## Mathematical Background

### Forcing Statistics

The forcing ξ(x,t) satisfies:

| Property | Formula |
|----------|---------|
| Zero mean | ⟨ξ(x,t)⟩ = 0 |
| White in time | ⟨ξ(x,t) ξ(x',t')⟩ = Q(x-x') δ(t-t') |
| Power spectrum | ⟨ξ̂(k,t) ξ̂*(k',t')⟩ = Q̂(k) δ(k-k') δ(t-t') |

The **forcing spectrum** Q̂(k) controls which scales receive energy.

### Energy Injection Rate

The mean energy injection rate is:

```math
\varepsilon = \int \frac{d^d k}{(2\pi)^d} \, \frac{\hat{Q}(k)}{2|k|^2}
```

Tarang normalizes the spectrum to match your specified `energy_injection_rate`.

### Ring Forcing Spectrum

The default "ring" forcing concentrates energy around a central wavenumber k_f:

```math
\hat{Q}(k) \propto \exp\left(-\frac{(|k| - k_f)^2}{2\delta_f^2}\right)
```

where δ_f is the bandwidth (`dk_forcing`).

### Numerical Implementation

For discrete time with step dt:

```math
\hat{F}(k) = \frac{\sqrt{\hat{Q}(k)}}{\sqrt{dt}} \cdot e^{2\pi i \cdot \text{rand}()}
```

The √dt scaling ensures correct variance for the Wiener process.

---

## Choosing a Spectrum Type

| Type | Use Case | Formula |
|------|----------|---------|
| `:ring` | 2D/3D turbulence at specific scale | Gaussian around \|k\| = k_f |
| `:band` | Sharp wavenumber band | \|k\| ∈ [k_f - δ_f, k_f + δ_f] |
| `:lowk` | Large-scale forcing | \|k\| < k_f |
| `:kolmogorov` | Kolmogorov cascade studies | Large-scale, smooth |

**Decision flowchart:**

```
Want specific injection scale?
├── Yes → :ring (default, most common)
│          └── Need sharp cutoff? → :band
└── No  → :lowk (force all large scales)
          └── Studying Kolmogorov cascade? → :kolmogorov
```

---

## API Reference

### Constructor

```julia
StochasticForcing(;
    field_size,                              # Required: (Nx,) or (Nx, Ny) or (Nx, Ny, Nz)
    domain_size = ntuple(i -> 2π, N),       # Domain extent
    energy_injection_rate = 1.0,             # Target ε
    k_forcing = 4.0,                         # Central wavenumber
    dk_forcing = 1.0,                        # Bandwidth
    dt = 0.01,                               # Timestep
    spectrum_type = :ring,                   # Spectrum shape
    rng = Random.GLOBAL_RNG,                 # RNG for reproducibility
    dtype = Float64                          # Precision
)
```

### Forcing Generation

| Function | Purpose |
|----------|---------|
| `generate_forcing!(forcing, t, substep)` | Generate/cache forcing at time t |
| `apply_forcing!(rhs, forcing, t, substep)` | Add forcing to RHS array |
| `reset_forcing!(forcing)` | Clear cache, force regeneration |
| `set_dt!(forcing, dt)` | Update timestep (for adaptive dt) |

### Work Calculation

| Function | Calculus | Formula |
|----------|----------|---------|
| `work_stratonovich(forcing, sol)` | Stratonovich | -⟨(ψⁿ + ψⁿ⁺¹)/2 · F̂*⟩ |
| `work_ito(forcing, sol_prev)` | Itô | -⟨ψⁿ · F̂*⟩ + ε·dt |
| `store_prevsol!(forcing, sol)` | Helper | Store ψⁿ for Stratonovich |

### Diagnostics

| Function | Returns |
|----------|---------|
| `mean_energy_injection_rate(forcing)` | Target ε |
| `instantaneous_power(forcing, sol)` | Current power P(t) |
| `forcing_enstrophy_injection_rate(forcing)` | η (2D only) |
| `get_forcing_spectrum(forcing)` | √Q̂(k) array |
| `get_cached_forcing(forcing)` | Current F̂(k) |

---

## Stratonovich vs Itô Calculus

### Why Stratonovich?

Tarang uses **Stratonovich calculus** because:

1. **Chain rule works normally** - d(f(X)) = f'(X) dX, just like regular calculus
2. **Physical systems converge to it** - colored noise with τ→0 gives Stratonovich
3. **Same formulas for stochastic and deterministic** - easier to verify

### Work Calculation

The work done by forcing over one timestep differs between interpretations:

**Stratonovich** (uses midpoint):
```math
W = -\left\langle \frac{\psi^n + \psi^{n+1}}{2} \cdot \hat{F}^* \right\rangle \cdot dt
```

**Itô** (uses initial value + drift):
```math
W = -\left\langle \psi^n \cdot \hat{F}^* \right\rangle \cdot dt + \varepsilon \cdot dt
```

Both give the same ensemble average ⟨W⟩ = ε·dt.

---

## Multi-Stage Timesteppers

**Critical**: Forcing must stay constant across all substeps within a timestep.

```
Timestep n:     [stage 1] → [stage 2] → [stage 3] → [stage 4]
                     ↑           ↑           ↑           ↑
Forcing:           F_n         F_n         F_n         F_n   ← SAME!

Timestep n+1:   [stage 1] → [stage 2] → [stage 3] → [stage 4]
                     ↑           ↑           ↑           ↑
Forcing:         F_{n+1}     F_{n+1}     F_{n+1}     F_{n+1} ← NEW
```

Tarang handles this automatically via the `substep` argument:

```julia
# Stage 1: generates NEW forcing
generate_forcing!(forcing, t, 1)

# Stages 2, 3, 4: returns CACHED forcing
generate_forcing!(forcing, t, 2)  # same as stage 1
generate_forcing!(forcing, t, 3)  # same as stage 1
generate_forcing!(forcing, t, 4)  # same as stage 1

# Next timestep: generates NEW forcing
generate_forcing!(forcing, t + dt, 1)  # different!
```

---

## Complete Example: 2D Forced Turbulence

```julia
using Tarang
using Random

# Reproducibility
Random.seed!(42)
rng = MersenneTwister(42)

# Domain setup
Nx, Ny = 256, 256
Lx, Ly = 2π, 2π
dt = 1e-3

# Create stochastic forcing
forcing = StochasticForcing(
    field_size = (Nx, Ny),
    domain_size = (Lx, Ly),
    energy_injection_rate = 0.1,    # ε = 0.1
    k_forcing = 10.0,               # Force at |k| ≈ 10
    dk_forcing = 2.0,               # Bandwidth
    dt = dt,
    spectrum_type = :ring,
    rng = rng
)

# Initialize vorticity (in spectral space)
ω_hat = zeros(ComplexF64, Nx, Ny)

# Time integration
t_end = 10.0
t = 0.0
total_work = 0.0

while t < t_end
    # Store previous solution (for Stratonovich work)
    store_prevsol!(forcing, ω_hat)

    # Generate forcing for this timestep
    F_hat = generate_forcing!(forcing, t, 1)

    # Your time integration here...
    # rhs = compute_rhs(ω_hat)
    # rhs .+= F_hat
    # ω_hat_new = advance(ω_hat, rhs, dt)

    # Compute work done (Stratonovich)
    W = work_stratonovich(forcing, ω_hat)
    total_work += W

    # Diagnostics
    if mod(round(Int, t/dt), 1000) == 0
        ε_mean = mean_energy_injection_rate(forcing)
        P = instantaneous_power(forcing, ω_hat)
        println("t = $t, ε = $ε_mean, P = $P, ∫W = $total_work")
    end

    t += dt
end
```

---

## Adaptive Timestepping

When using adaptive dt, update the forcing:

```julia
# CFL gives new timestep
dt_new = compute_timestep(cfl)

# Update forcing scaling
set_dt!(forcing, dt_new)

# Continue normally
generate_forcing!(forcing, t, 1)
```

The `set_dt!` call rescales the cached spectrum for the new dt.

---

## Reproducibility

For reproducible results, seed the RNG:

```julia
using Random

# Option 1: Global seed
Random.seed!(12345)
forcing = StochasticForcing(field_size = (64, 64), ...)

# Option 2: Dedicated RNG (recommended for parallel)
rng = MersenneTwister(12345)
forcing = StochasticForcing(field_size = (64, 64), ..., rng = rng)
```

---

## Energy Budget Verification

Verify energy balance in your simulation:

```julia
# Theoretical injection rate
ε_target = forcing.energy_injection_rate

# Measured from work
W_total = 0.0
for step in 1:n_steps
    store_prevsol!(forcing, ψ_hat)
    # ... advance ...
    W_total += work_stratonovich(forcing, ψ_hat)
end
ε_measured = W_total / (n_steps * dt)

# These should match (within statistical fluctuations)
@assert abs(ε_measured - ε_target) / ε_target < 0.1
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Energy grows unbounded | ε too high or dissipation too low | Reduce `energy_injection_rate` or increase viscosity |
| Forcing has no effect | Wrong spectrum normalization | Check `energy_injection_rate > 0` |
| Noise at all scales | `k_forcing` too high | Reduce `k_forcing` to target larger scales |
| Results not reproducible | RNG not seeded | Pass seeded RNG to constructor |
| Wrong energy injection rate | dt changed without `set_dt!` | Call `set_dt!(forcing, dt_new)` |

---

## See Also

- [Timesteppers](timesteppers.md) - Time integration methods
- [Solvers](solvers.md) - Using forcing with IVP solvers
- [API: Stochastic Forcing](../api/stochastic_forcing.md) - Complete API reference

## References

1. [GeophysicalFlows.jl Stochastic Forcing](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/stochastic_forcing/)
2. [FourierFlows.jl](https://github.com/FourierFlows/FourierFlows.jl)
3. Constantinou, N. C., & Hogg, A. M. (2021). "Intrinsic oceanic decadal variability"
