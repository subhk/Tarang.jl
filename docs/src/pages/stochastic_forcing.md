# Stochastic Forcing

This page demonstrates how to set up a forced-dissipative 2D turbulence simulation, following the approach from [GeophysicalFlows.jl](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/stochastic_forcing/).

---

## TL;DR - Automatic Forcing (Recommended)

The simplest way to add stochastic forcing is using `add_stochastic_forcing!`:

```julia
# 1. Create forcing
forcing = StochasticForcing(
    field_size = (256, 256),
    energy_injection_rate = 0.1,
    k_forcing = 10.0,
    dt = dt
)

# 2. Register with problem by variable symbol - forcing is handled automatically!
add_stochastic_forcing!(problem, :ω, forcing)

# 3. Just call step! - no manual forcing management needed
for step in 1:nsteps
    step!(solver)  # Forcing generated & applied internally
end
```

The timestepper automatically:
- Generates forcing ONCE at the start of each timestep
- Uses the SAME forcing value across all RK substeps
- Adds forcing to the RHS of the specified equation

This ensures correct **Stratonovich calculus** without any manual intervention.

---

## Why Stochastic Forcing Needs Special Handling

White-in-time noise requires careful handling for correct statistics:

1. **Stratonovich vs Itô**: Physical systems use Stratonovich calculus where the chain rule works normally
2. **Constant within timestep**: For Stratonovich correctness, forcing must be constant across all RK substeps
3. **√dt scaling**: Discrete forcing needs `F = √(Q̂/dt) · noise` to give correct variance

If you wrote forcing as a regular RHS term, it would be evaluated at each RK stage with different random values - this gives **wrong statistics** (Itô instead of Stratonovich).

---

## Complete Example: Forced 2D Turbulence

We simulate the 2D vorticity equation with stochastic forcing and linear drag:

```math
\frac{\partial \omega}{\partial t} + J(\psi, \omega) = -\mu \omega + \nu \nabla^2 \omega + \xi
```

where ξ is white-in-time, spatially-correlated stochastic forcing.

```julia
using Tarang
using Random
using Printf

# ============================================================
# 1. Physical and Numerical Parameters
# ============================================================

# Grid
n = 256                     # Resolution
L = 2π                      # Domain size
dt = 0.005                  # Timestep
nsteps = 4000               # Total steps

# Dissipation
ν = 2e-7                    # Viscosity (hyperviscosity coefficient)
μ = 1e-1                    # Linear drag coefficient

# Forcing parameters
forcing_wavenumber = 14.0 * 2π/L    # k_f: force at this scale
forcing_bandwidth = 1.5 * 2π/L      # δ_f: width of forcing ring
ε = 0.1                              # Energy injection rate

# ============================================================
# 2. Set Up Domain and Fields
# ============================================================

coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; mesh=(1, 1))

xbasis = RealFourier(coords["x"]; size=n, bounds=(0.0, L))
ybasis = RealFourier(coords["y"]; size=n, bounds=(0.0, L))
domain = Domain(dist, (xbasis, ybasis))

# Vorticity field
ω = ScalarField(dist, "omega", (xbasis, ybasis))

# ============================================================
# 3. Create Stochastic Forcing
# ============================================================

# The forcing is concentrated in a ring around |k| = k_f
# with Gaussian profile: exp(-(|k| - k_f)² / 2δ_f²)

forcing = StochasticForcing(
    field_size = (n, n),
    domain_size = (L, L),
    energy_injection_rate = ε,
    k_forcing = forcing_wavenumber,
    dk_forcing = forcing_bandwidth,
    dt = dt,
    spectrum_type = :ring,
    rng = MersenneTwister(1234)     # For reproducibility
)

println("Forcing setup:")
println("  k_forcing = $(forcing_wavenumber) (mode ≈ $(forcing_wavenumber * L / 2π))")
println("  bandwidth = $(forcing_bandwidth)")
println("  ε = $(ε)")

# ============================================================
# 4. Set Up Problem and Solver
# ============================================================

problem = IVP([ω])

# Vorticity equation: ∂ω/∂t = -μω + ν∇²ω - J(ψ,ω) + ξ
# Note: Forcing ξ is NOT in the equation string - it's added automatically!
add_equation!(problem, "∂t(ω) + μ*ω - ν*Δ(ω) = -J(ψ, ω)")
problem.parameters["ν"] = ν
problem.parameters["μ"] = μ

# Register stochastic forcing - it will be added to ω's RHS automatically
add_stochastic_forcing!(problem, :ω, forcing)

solver = InitialValueSolver(problem, RK443(); dt=dt, device="cpu")

# ============================================================
# 5. Diagnostics: Energy and Enstrophy
# ============================================================

function compute_energy(ω_hat, grid)
    # E = ½⟨|∇ψ|²⟩ = ½ ∑_k |ω̂_k|² / |k|²
    E = 0.0
    for j in 1:size(ω_hat, 2), i in 1:size(ω_hat, 1)
        k2 = grid.kx[i]^2 + grid.ky[j]^2
        if k2 > 0
            E += abs2(ω_hat[i,j]) / k2
        end
    end
    return 0.5 * E / prod(size(ω_hat))
end

function compute_enstrophy(ω_hat)
    # Z = ½⟨ω²⟩ = ½ ∑_k |ω̂_k|²
    return 0.5 * sum(abs2, ω_hat) / prod(size(ω_hat))
end

# ============================================================
# 6. Time Integration Loop
# ============================================================

# Storage for diagnostics
t_save = Float64[]
E_save = Float64[]
Z_save = Float64[]
work_save = Float64[]

println("\nStarting simulation...")
println("=" ^ 60)

for step in 1:nsteps
    # Store previous solution for Stratonovich work calculation (optional diagnostic)
    store_prevsol!(forcing, ω.data_c)

    # Advance one timestep - forcing is generated and applied automatically!
    step!(solver)

    # Compute work done by forcing (Stratonovich) - optional diagnostic
    W = work_stratonovich(forcing, ω.data_c)

    # Periodic diagnostics
    if step % 100 == 0 || step == 1
        E = compute_energy(ω.data_c, domain.grid)
        Z = compute_enstrophy(ω.data_c)

        push!(t_save, solver.sim_time)
        push!(E_save, E)
        push!(Z_save, Z)
        push!(work_save, W)

        @printf("step %4d, t = %6.3f, E = %.2e, Z = %.2e, W = %+.2e\n",
                step, solver.sim_time, E, Z, W)
    end
end

println("=" ^ 60)
println("Simulation complete!")
println("Final time: $(solver.sim_time)")
println("Mean energy injection rate: $(mean(work_save ./ dt))")
```

---

## Understanding the Example

### Forcing Spectrum

The forcing concentrates energy input around a specific wavenumber:

```
          │
    Q̂(k)  │      ╱╲
          │    ╱    ╲
          │  ╱        ╲
          │╱            ╲
          └──────────────────► |k|
                k_f
               ←─────→
              2*δ_f (width)
```

Modes with `|k| ≈ k_f` receive most of the energy. This creates a **dual cascade**:
- Energy cascades to large scales (inverse cascade)
- Enstrophy cascades to small scales (forward cascade)

### Energy Balance

In steady state:

```math
\varepsilon_{\text{forcing}} = \varepsilon_{\text{drag}} + \varepsilon_{\text{viscous}}
```

The linear drag μ removes energy at large scales, preventing condensation.

---

## Step-by-Step Breakdown

### Step 1: Choose Forcing Parameters

```julia
# Where to inject energy (wavenumber)
forcing_wavenumber = 14.0 * 2π/L  # Mode k ≈ 14

# How spread out the forcing is
forcing_bandwidth = 1.5 * 2π/L    # A few modes wide

# How much energy to inject per unit time
ε = 0.1
```

**Guidelines:**
- `k_forcing`: Set to intermediate scales (not too large, not too small)
- `dk_forcing`: Usually 1-3 in mode units; too narrow causes intermittency
- `ε`: Balance with dissipation; if energy grows unbounded, increase μ or ν

### Step 2: Create the Forcing Object

```julia
forcing = StochasticForcing(
    field_size = (n, n),              # Match your grid
    domain_size = (L, L),             # Match your domain
    energy_injection_rate = ε,        # Normalized automatically
    k_forcing = forcing_wavenumber,
    dk_forcing = forcing_bandwidth,
    dt = dt,                          # For √dt scaling
    spectrum_type = :ring,            # Isotropic ring
    rng = MersenneTwister(1234)       # Reproducibility
)
```

### Step 3: Apply Forcing in Time Loop

```julia
for step in 1:nsteps
    # 1. Store ψⁿ for Stratonovich work
    store_prevsol!(forcing, ω.data_c)

    # 2. Generate F̂ (new forcing each timestep)
    F_hat = generate_forcing!(forcing, t, 1)

    # 3. Add to your RHS: rhs .+= F_hat
    apply_forcing!(rhs, forcing, t, 1)

    # 4. Advance your solution
    step!(solver)

    # 5. Compute work done
    W = work_stratonovich(forcing, ω.data_c)
end
```

### Step 4: Verify Energy Budget

```julia
# Average over many timesteps in steady state
ε_measured = mean(work_array) / dt

# Should match target
@assert abs(ε_measured - ε) / ε < 0.1  "Energy budget not balanced!"
```

---

## Spectrum Types

### Ring Forcing (`:ring`) - Default

Forces modes in a Gaussian ring around `|k| = k_f`:

```julia
forcing = StochasticForcing(
    field_size = (256, 256),
    k_forcing = 10.0,
    dk_forcing = 2.0,
    spectrum_type = :ring
)
```

Best for: Standard 2D/3D turbulence, dual cascade studies

### Band Forcing (`:band`)

Sharp cutoff band `|k| ∈ [k_f - δ_f, k_f + δ_f]`:

```julia
forcing = StochasticForcing(
    field_size = (256, 256),
    k_forcing = 10.0,
    dk_forcing = 2.0,
    spectrum_type = :band
)
```

Best for: Controlled experiments with exact spectral support

### Low-k Forcing (`:lowk`)

Force all modes with `|k| < k_f`:

```julia
forcing = StochasticForcing(
    field_size = (256, 256),
    k_forcing = 5.0,
    spectrum_type = :lowk
)
```

Best for: Large-scale forcing, simple turbulence setups

### Kolmogorov Forcing (`:kolmogorov`)

Smooth large-scale forcing for Kolmogorov cascade:

```julia
forcing = StochasticForcing(
    field_size = (256, 256),
    k_forcing = 4.0,
    spectrum_type = :kolmogorov
)
```

Best for: Kolmogorov-Obukhov theory verification

---

## Mathematical Background

### Forcing Statistics

The forcing ξ(x,t) is a random field with:

| Property | Formula |
|----------|---------|
| Zero mean | ⟨ξ(x,t)⟩ = 0 |
| White in time | ⟨ξ(x,t) ξ(x',t')⟩ = Q(x-x') δ(t-t') |
| Power spectrum | ⟨ξ̂(k) ξ̂*(k')⟩ = Q̂(k) δ(k-k') |

### Numerical Implementation

For discrete time with step dt:

```math
\hat{F}(k) = \sqrt{\frac{\hat{Q}(k)}{dt}} \cdot e^{2\pi i \cdot \text{rand}()}
```

The √dt scaling gives correct variance: ⟨|F̂|²⟩ · dt = Q̂(k)

### Energy Injection Rate

```math
\varepsilon = \int \frac{d^2 k}{(2\pi)^2} \, \frac{\hat{Q}(k)}{2|k|^2}
```

Tarang normalizes the spectrum to achieve the requested `energy_injection_rate`.

### Stratonovich vs Itô

**Why Stratonovich?**
- Chain rule works normally: d(f(X)) = f'(X) dX
- Physical limit of colored noise with τ→0
- Same formulas for stochastic and deterministic forcing

**Work calculation:**

| Calculus | Formula |
|----------|---------|
| Stratonovich | W = -⟨(ψⁿ + ψⁿ⁺¹)/2 · F̂*⟩ |
| Itô | W = -⟨ψⁿ · F̂*⟩ + ε·dt |

---

## Multi-Stage Timesteppers

For RK4, SBDF, and other multi-stage methods, forcing must stay constant within a timestep:

```
Timestep n:     [stage 1] → [stage 2] → [stage 3] → [stage 4]
Forcing:           F_n         F_n         F_n         F_n

Timestep n+1:   [stage 1] → [stage 2] → [stage 3] → [stage 4]
Forcing:         F_{n+1}     F_{n+1}     F_{n+1}     F_{n+1}
```

Tarang handles this via the `substep` argument:

```julia
# Substep 1: generates NEW forcing
generate_forcing!(forcing, t, 1)

# Substeps 2-4: returns CACHED forcing
generate_forcing!(forcing, t, 2)  # Same as substep 1
generate_forcing!(forcing, t, 3)  # Same as substep 1
generate_forcing!(forcing, t, 4)  # Same as substep 1
```

---

## Adaptive Timestepping

When dt changes, update the forcing:

```julia
dt_new = compute_timestep(cfl)
set_dt!(forcing, dt_new)
generate_forcing!(forcing, t, 1)
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Energy grows unbounded | ε too high or dissipation too low | Increase μ or ν, or decrease ε |
| No turbulence develops | ε too low | Increase `energy_injection_rate` |
| Forcing at wrong scales | k_f mismatched | Check units: k_f should be in radians, not mode number |
| Results not reproducible | RNG not seeded | Pass `rng = MersenneTwister(seed)` |
| Jerky dynamics | dt too large | Reduce timestep or forcing bandwidth |

---

## API Reference Summary

### Constructor

```julia
StochasticForcing(;
    field_size,                          # (Nx, Ny) or (Nx, Ny, Nz)
    domain_size = (2π, 2π, ...),        # (Lx, Ly, ...)
    energy_injection_rate = 1.0,         # ε
    k_forcing = 4.0,                     # k_f
    dk_forcing = 1.0,                    # δ_f
    dt = 0.01,
    spectrum_type = :ring,               # :ring, :band, :lowk, :kolmogorov
    rng = Random.GLOBAL_RNG,
    dtype = Float64
)
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `generate_forcing!(f, t, substep)` | Generate/cache forcing |
| `apply_forcing!(rhs, f, t, substep)` | Add forcing to RHS |
| `store_prevsol!(f, sol)` | Store ψⁿ for work calculation |
| `work_stratonovich(f, sol)` | Compute Stratonovich work |
| `set_dt!(f, dt)` | Update timestep |
| `mean_energy_injection_rate(f)` | Get target ε |
| `instantaneous_power(f, sol)` | Get current P(t) |

---

## See Also

- [Timesteppers](timesteppers.md) - Time integration methods
- [Solvers](solvers.md) - Using forcing with IVP solvers
- [API: Stochastic Forcing](../api/stochastic_forcing.md) - Complete API reference

## References

1. [GeophysicalFlows.jl Stochastic Forcing](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/stochastic_forcing/)
2. [GeophysicalFlows.jl 2D NS Example](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/literated/twodnavierstokes_stochasticforcing/)
3. Constantinou, N. C., & Hogg, A. M. (2021). "Intrinsic oceanic decadal variability"
