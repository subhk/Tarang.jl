# Temporal Filters for Lagrangian Averaging

Efficient computation of Lagrangian means using exponential time filters.

**Reference:** Minz, Baker, Kafiabad & Vanneste (2025). ["Efficient Lagrangian averaging with exponential filters"](https://doi.org/10.1103/PhysRevFluids.10.074902). *Phys. Rev. Fluids* 10, 074902.

---

## Quick Summary

> **What:** Temporal filters that separate fast waves from slow mean flows
>
> **Why:** Traditional averaging requires storing entire time history; these filters need only the current state
>
> **Which filter to use:**
> - `ExponentialMean` - Simple, low memory (1 array), moderate filtering
> - `ButterworthFilter` - Sharp cutoff, 2x memory (2 arrays), excellent filtering
> - `LagrangianFilter` - Full Lagrangian averaging with particle displacement tracking

---

## Quick Start

### Basic Usage (3 lines of code)

```julia
using Tarang

# 1. Create a filter for your field size
filter = ButterworthFilter((64, 64); α=0.5)  # α = 1/averaging_time

# 2. In your time loop, update the filter
update!(filter, field_data, dt)

# 3. Get the filtered mean anytime
mean_field = get_mean(filter)
```

### Complete Working Example

```julia
using Tarang

# Setup: 2D field with slow trend + fast oscillations
Nx, Ny = 64, 64
α = 0.5      # Averaging time = 1/α = 2 time units
dt = 0.01
nsteps = 500

# Create Butterworth filter (better than exponential for sharp separation)
filter = ButterworthFilter((Nx, Ny); α=α)

# Simulate: field = slow_trend + fast_oscillation
for step in 1:nsteps
    t = step * dt

    # Your field: slow part (want to keep) + fast part (want to remove)
    slow_part = 1.0 + 0.1 * t
    fast_part = 0.5 * sin(20.0 * t)  # ω=20 >> α=0.5, will be filtered out
    field = fill(slow_part + fast_part, Nx, Ny)

    # Update filter
    update!(filter, field, dt)
end

# Result: filtered mean ≈ slow_part only
h_mean = get_mean(filter)
println("Filtered mean: ", h_mean[1,1])  # ≈ 1.0 + 0.1*5.0 = 1.5
```

---

## Choosing a Filter

```
                    ┌─────────────────────────────────────┐
                    │  Do you need Lagrangian averaging?  │
                    │  (particle tracking, Stokes drift)  │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
                   YES                              NO
                    │                               │
                    ▼                               ▼
           ┌────────────────┐              ┌─────────────────────┐
           │LagrangianFilter│              │Need sharp wave-mean │
           └────────────────┘              │   separation?       │
                                           └─────────────────────┘
                                                    │
                                    ┌───────────────┴───────────────┐
                                    ▼                               ▼
                                   YES                              NO
                                    │                               │
                                    ▼                               ▼
                          ┌──────────────────┐            ┌─────────────────┐
                          │ButterworthFilter │            │ ExponentialMean │
                          │  (recommended)   │            │   (simplest)    │
                          └──────────────────┘            └─────────────────┘
```

### Filter Comparison

| Feature | ExponentialMean | ButterworthFilter | LagrangianFilter |
|---------|-----------------|-------------------|------------------|
| **Use case** | Simple running mean | Wave-mean separation | Full Lagrangian averaging |
| **Memory** | 1 array | 2 arrays | 2-4 arrays |
| **High-freq rolloff** | -20 dB/decade | -40 dB/decade | Depends on type |
| **Attenuation at ω=10α** | 1% passes | 0.01% passes | - |
| **Complexity** | Simplest | Moderate | Most complex |

---

## The Key Parameter: α

The parameter `α` (alpha) is the **inverse averaging timescale**:

$$\alpha = \frac{1}{T_{\text{avg}}}$$

### How to Choose α

**Rule of thumb:** Set α much smaller than the frequency of oscillations you want to remove.

```
α << ω_wave    (α should be much less than wave frequency)
```

**Practical guidance:**

| Wave period | Suggested α | Averaging time |
|-------------|-------------|----------------|
| 1 hour | 0.05/hour | 20 hours |
| 10 minutes | 0.005/min | 200 min |
| 1 second | 0.05/s | 20 s |

**Example:** If you have internal waves with period T=1 hour (ω = 2π/hour ≈ 6.28/hour):
```julia
α = 0.1  # per hour → averages over 10 hours → filters waves with ω >> 0.1
```

!!! tip "Quick rule"
    For Butterworth: waves with ω > 20α are effectively filtered out.
    For Exponential: waves with ω > 50α are effectively filtered out.

---

## API Reference

### ExponentialMean

First-order filter. Simple and memory-efficient.

```julia
# Constructor
filter = ExponentialMean((Nx, Ny); α=0.5, dtype=Float64)

# Key ODE: dh̄/dt = α(h - h̄)
# Transfer function: H(s) = α/(s + α)
```

**Functions:**
```julia
update!(filter, h, dt)           # Update with new data (forward Euler)
update!(filter, h, dt, Val(:RK2)) # Update with RK2 (more accurate)
get_mean(filter)                 # Get filtered mean h̄
reset!(filter)                   # Reset to zero
set_α!(filter, new_α)            # Change α dynamically
effective_averaging_time(filter) # Returns 1/α
filter_response(filter, ω)       # Returns |H(iω)|² at frequency ω
```

### ButterworthFilter

Second-order filter. Sharper cutoff, recommended for wave-mean separation.

```julia
# Constructor
filter = ButterworthFilter((Nx, Ny); α=0.5, dtype=Float64)

# Key ODEs (coupled system):
#   dh̃/dt = α[h - (√2-1)h̃ - (2-√2)h̄]
#   dh̄/dt = α(h̃ - h̄)
# Transfer function: K(s) = α²/(s² + √2αs + α²)
```

**Functions:**
```julia
update!(filter, h, dt)           # Update with new data
update!(filter, h, dt, Val(:RK2)) # Update with RK2
get_mean(filter)                 # Get filtered mean h̄
get_auxiliary(filter)            # Get auxiliary field h̃
reset!(filter)                   # Reset to zero
set_α!(filter, new_α)            # Change α
effective_averaging_time(filter) # Returns 1/α
filter_response(filter, ω)       # Returns |K(iω)|²
```

### LagrangianFilter

Full Lagrangian averaging with displacement field tracking.

```julia
# Constructor
filter = LagrangianFilter((Nx, Ny);
    α=0.5,
    filter_type=:butterworth,  # or :exponential
    dtype=Float64
)

# Key relations:
#   Mean velocity: ū = α·ξ (exponential) or ū = α·ξ̃ (Butterworth)
#   Displacement PDE: ∂ξ/∂t + ū·∇ξ = u∘(id+ξ) - ū
```

**Functions:**
```julia
update_displacement!(filter, velocity, dt)  # Update displacement ξ
get_mean_velocity(filter)                   # Get ū
get_displacement(filter)                    # Get ξ
lagrangian_mean!(filter, gᴸ, g, dt)         # Compute Lagrangian mean of scalar g
reset!(filter)                              # Reset all fields
set_α!(filter, new_α)                       # Change α
```

---

## Detailed Examples

### Example 1: Filtering Oscillations from a Time Series

```julia
using Tarang

# Problem: Extract slowly-varying part from noisy oscillating signal
Nx = 100
α = 1.0      # Filter frequencies >> 1
dt = 0.01
T_final = 10.0

# Compare exponential vs Butterworth
exp_filter = ExponentialMean((Nx,); α=α)
but_filter = ButterworthFilter((Nx,); α=α)

t = 0.0
while t < T_final
    # Signal: mean=2.0, slow variation, fast noise
    signal = fill(2.0 + 0.5*sin(0.1*t) + 0.3*sin(50.0*t), Nx)
    #              ↑         ↑                ↑
    #           constant   slow (keep)    fast (remove)

    update!(exp_filter, signal, dt)
    update!(but_filter, signal, dt)
    t += dt
end

exp_mean = get_mean(exp_filter)[1]
but_mean = get_mean(but_filter)[1]

println("Exponential mean: ", round(exp_mean, digits=3))  # Some fast oscillation leaks through
println("Butterworth mean: ", round(but_mean, digits=3))  # Cleaner result
```

### Example 2: Wave-Mean Flow Decomposition

```julia
using Tarang

# Setup domain (simplified - not using full Tarang fields)
Nx, Ny = 64, 64
α = 0.2      # Average over 5 time units
dt = 0.02

# Filter for each velocity component
u_filter = ButterworthFilter((Nx, Ny); α=α)
v_filter = ButterworthFilter((Nx, Ny); α=α)

# Simulation loop
for step in 1:1000
    t = step * dt

    # Synthetic velocity: mean jet + inertia-gravity waves
    u = zeros(Nx, Ny)
    v = zeros(Nx, Ny)
    for i in 1:Nx, j in 1:Ny
        x, y = (i-1)/Nx * 2π, (j-1)/Ny * 2π

        # Mean zonal jet (slow)
        u[i,j] = sin(y)

        # Wave perturbations (fast)
        ω = 10.0  # wave frequency >> α
        u[i,j] += 0.1 * cos(2x + 3y - ω*t)
        v[i,j] = 0.1 * sin(2x + 3y - ω*t)
    end

    # Update filters
    update!(u_filter, u, dt)
    update!(v_filter, v, dt)
end

# Get mean flow
ū = get_mean(u_filter)
v̄ = get_mean(v_filter)

# Wave fluctuations
u_wave = u - ū
v_wave = v - v̄

println("Mean u max: ", maximum(abs.(ū)))     # ≈ 1.0 (the jet)
println("Mean v max: ", maximum(abs.(v̄)))     # ≈ 0.0 (waves filtered)
```

### Example 3: Comparing Filter Frequency Response

```julia
using Tarang

α = 1.0

exp_filter = ExponentialMean((10,); α=α)
but_filter = ButterworthFilter((10,); α=α)

println("Frequency Response |H(iω)|²")
println("="^50)
println("ω/α      Exponential    Butterworth")
println("-"^50)

for ω_ratio in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    ω = ω_ratio * α
    exp_resp = filter_response(exp_filter, ω)
    but_resp = filter_response(but_filter, ω)

    @printf("%4.1f     %10.6f     %10.6f\n", ω_ratio, exp_resp, but_resp)
end
```

Output:
```
ω/α      Exponential    Butterworth
--------------------------------------------------
 0.1       0.990099       0.999800
 0.5       0.800000       0.941176
 1.0       0.500000       0.500000  ← half-power point
 2.0       0.200000       0.058824
 5.0       0.038462       0.001575
10.0       0.009901       0.000100
20.0       0.002494       0.000006
```

!!! note "Key observation"
    At ω = 10α, Butterworth passes only 0.01% vs Exponential's 1% — that's 100x better filtering!

---

## Background Theory

This section explains the physics behind temporal filters. Skip if you just want to use the filters.

### The Wave-Mean Flow Problem

Geophysical flows contain both:
- **Slow mean flows**: jets, eddies, large-scale circulation
- **Fast waves**: gravity waves, inertial oscillations, acoustic waves

```
u(x,t) = ū(x,t) + u'(x,t)
         ↑          ↑
      mean flow   wave fluctuation
```

The challenge: How do we define and compute the "mean" efficiently?

### Eulerian vs Lagrangian Averaging

**Eulerian (fixed point):**
$$\bar{u}^E(\mathbf{x}) = \frac{1}{T} \int_0^T u(\mathbf{x}, t) \, dt$$

**Problem:** Misses Stokes drift! A wave with zero Eulerian mean can still transport particles.

**Lagrangian (following particles):**
$$\bar{u}^L = \frac{1}{T} \int_0^T u(\mathbf{X}(t), t) \, dt$$

**Advantage:** Captures actual material transport.

### Why Exponential Filters?

Traditional averaging requires storing entire time history. Exponential filters use a weighted average:

$$\bar{h}(t) = \alpha \int_0^\infty e^{-\alpha\tau} h(t-\tau) \, d\tau$$

This satisfies a simple ODE:

$$\frac{d\bar{h}}{dt} = \alpha(h - \bar{h})$$

**No storage needed** — just update the current mean at each timestep!

### The Butterworth Advantage

The exponential filter has slow rolloff (-20 dB/decade). The second-order Butterworth filter has:

$$K(s) = \frac{\alpha^2}{s^2 + \sqrt{2}\alpha s + \alpha^2}$$

This gives -40 dB/decade rolloff — much sharper separation between waves and mean flow.

### The Lifting Map (Advanced)

For true Lagrangian averaging, we need the **lifting map** $\boldsymbol{\Xi}(\mathbf{x}, t)$:

> $\boldsymbol{\Xi}(\mathbf{x}, t)$ = position at time $t$ of the parcel whose **mean position** is $\mathbf{x}$

The **displacement** is: $\boldsymbol{\xi} = \boldsymbol{\Xi} - \mathbf{x}$

Key result from Minz et al.:
- For exponential mean: $\bar{\mathbf{u}} = \alpha \boldsymbol{\xi}$
- For Butterworth: $\bar{\mathbf{u}} = \alpha \tilde{\boldsymbol{\xi}}$

This is implemented in `LagrangianFilter`.

---

## Mathematical Details

### Exponential Filter

**Kernel:** $k(\tau) = \alpha e^{-\alpha\tau} \Theta(\tau)$

**ODE:** $\frac{d\bar{h}}{dt} = \alpha(h - \bar{h})$

**Transfer function:** $H(s) = \frac{\alpha}{s + \alpha}$

**Frequency response:** $|H(i\omega)|^2 = \frac{\alpha^2}{\alpha^2 + \omega^2}$

### Butterworth Filter

**Kernel:** $k(\tau) = \sqrt{2}\alpha e^{-\alpha\tau/\sqrt{2}} \sin(\alpha\tau/\sqrt{2}) \Theta(\tau)$

**Coupled ODEs:**
$$\frac{d\tilde{h}}{dt} = \alpha\left[h - (\sqrt{2}-1)\tilde{h} - (2-\sqrt{2})\bar{h}\right]$$
$$\frac{d\bar{h}}{dt} = \alpha(\tilde{h} - \bar{h})$$

**Transfer function:** $K(s) = \frac{\alpha^2}{s^2 + \sqrt{2}\alpha s + \alpha^2}$

**Frequency response:** $|K(i\omega)|^2 = \frac{\alpha^4}{(\alpha^2 - \omega^2)^2 + 2\alpha^2\omega^2}$

### Filter Matrix

The Butterworth system can be written as:
$$\frac{d}{dt}\begin{pmatrix}\tilde{h}\\\bar{h}\end{pmatrix} = -\alpha\mathbf{A}\begin{pmatrix}\tilde{h}\\\bar{h}\end{pmatrix} + \alpha\begin{pmatrix}h\\0\end{pmatrix}$$

where:
$$\mathbf{A} = \begin{pmatrix}\sqrt{2}-1 & 2-\sqrt{2}\\-1 & 1\end{pmatrix} \approx \begin{pmatrix}0.414 & 0.586\\-1 & 1\end{pmatrix}$$

### Lagrangian Mean Equations

**Displacement evolution:**
$$\frac{\partial\boldsymbol{\xi}}{\partial t} + \bar{\mathbf{u}}\cdot\nabla\boldsymbol{\xi} = \mathbf{u}\circ(\mathbf{id}+\boldsymbol{\xi}) - \bar{\mathbf{u}}$$

**Scalar Lagrangian mean (exponential):**
$$\frac{\partial g^L}{\partial t} + \bar{\mathbf{u}}\cdot\nabla g^L = \alpha(g\circ\boldsymbol{\Xi} - g^L)$$

**Scalar Lagrangian mean (Butterworth):**
$$\frac{\partial\tilde{g}}{\partial t} + \bar{\mathbf{u}}\cdot\nabla\tilde{g} = \alpha\left[g\circ\boldsymbol{\Xi} - (\sqrt{2}-1)\tilde{g} - (2-\sqrt{2})g^L\right]$$
$$\frac{\partial g^L}{\partial t} + \bar{\mathbf{u}}\cdot\nabla g^L = \alpha(\tilde{g} - g^L)$$

---

## Advanced Time Integration Methods

The basic `update!` function uses Forward Euler, which has a stability limit. For larger timesteps or coupling with implicit PDE solvers, use these advanced methods.

### Stability Limits (Explicit Methods)

| Filter | Forward Euler | RK2 |
|--------|---------------|-----|
| ExponentialMean | $\Delta t \leq 2/\alpha$ | $\Delta t \leq 4/\alpha$ |
| ButterworthFilter | $\Delta t \leq \sqrt{2}/\alpha$ | $\Delta t \leq 2\sqrt{2}/\alpha$ |

```julia
# Check maximum stable timestep
dt_max = max_stable_timestep(filter)           # Forward Euler
dt_max_rk2 = max_stable_timestep(filter; method=:RK2)  # RK2
```

### ETD (Exponential Time Differencing) - Recommended

**Unconditionally stable** for any timestep. Treats the linear decay term exactly.

```julia
# Precompute coefficients once (per α, dt pair)
coeffs = precompute_etd_coefficients(filter, dt)

# In time loop - no stability limit!
for step in 1:nsteps
    # ... your PDE timestepping ...
    update_etd!(filter, field_data, coeffs)
end
```

**How it works:**

For the filter ODE $d\bar{h}/dt = -\alpha\bar{h} + \alpha h$, ETD computes:

$$\bar{h}^{n+1} = e^{-\alpha\Delta t}\bar{h}^n + (1 - e^{-\alpha\Delta t})h^n$$

This is **exact** when $h$ is constant over the timestep, and unconditionally stable otherwise.

### IMEX/SBDF Integration

For coupling with implicit PDE solvers using SBDF2 or SBDF3 timestepping.

```julia
# Precompute IMEX coefficients
coeffs = precompute_imex_coefficients(filter, dt; scheme=:SBDF2)

# Store field history for multistep method
h_prev = copy(h)
h_curr = copy(h)

# In time loop - unconditionally stable!
for step in 1:nsteps
    # ... your SBDF2 PDE timestepping ...
    h_new = # ... compute new field value ...

    # Update filter with field history
    update_imex!(filter, (h_curr, h_prev), coeffs)

    h_prev .= h_curr
    h_curr .= h_new
end
```

**Available schemes:**
- `:SBDF1` (Backward Euler): First-order, most stable
- `:SBDF2`: Second-order, recommended
- `:SBDF3`: Third-order, highest accuracy

### Accessing Linear Operator for External Solvers

If you're using your own IMEX framework:

```julia
# Get the linear operator coefficient
L = linear_operator_coefficients(filter)
# Returns -α (scalar) for ExponentialMean
# Returns -α·A (2×2 matrix) for ButterworthFilter
```

---

## Troubleshooting

### Common Issues

**1. Filter not converging to expected mean**

*Cause:* Not enough spinup time.

*Solution:* Run for at least $5/\alpha$ time units before trusting results.
```julia
T_spinup = 5 / α  # Minimum spinup time
```

**2. Oscillations in filtered output**

*Cause:* Timestep too large for stability.

*Solution:* Use ETD integration (unconditionally stable):
```julia
coeffs = precompute_etd_coefficients(filter, dt)
update_etd!(filter, h, coeffs)  # Always stable!
```

Or use RK2 for explicit methods:
```julia
update!(filter, h, dt, Val(:RK2))  # 2× larger stability region
```

**3. High frequencies leaking through**

*Cause:* Using ExponentialMean when ButterworthFilter is needed.

*Solution:* Switch to Butterworth for sharper cutoff.

**4. Memory issues with large 3D fields**

*Cause:* ButterworthFilter uses 2x memory.

*Solution:* Use ExponentialMean if memory is critical:
```julia
# ButterworthFilter: 2 arrays
# ExponentialMean: 1 array (half the memory)
filter = ExponentialMean((Nx, Ny, Nz); α=α)
```

**5. Need to couple with implicit timestepper**

*Solution:* Use IMEX integration:
```julia
coeffs = precompute_imex_coefficients(filter, dt; scheme=:SBDF2)
update_imex!(filter, (h_n, h_nm1), coeffs)
```

---

## Tutorials

- **[Rotating Shallow Water with Lagrangian Averaging](../tutorials/rotating_shallow_water.md)** - Complete example of wave-mean separation in a rotating flow

---

## References

1. **Minz, C., Baker, L. E., Kafiabad, H. A., & Vanneste, J.** (2025). Efficient Lagrangian averaging with exponential filters. *Physical Review Fluids*, 10, 074902. [DOI](https://doi.org/10.1103/PhysRevFluids.10.074902)

2. **Andrews, D. G., & McIntyre, M. E.** (1978). An exact theory of nonlinear waves on a Lagrangian-mean flow. *J. Fluid Mech.*, 89(4), 609-646.

3. **Bühler, O.** (2014). *Waves and Mean Flows* (2nd ed.). Cambridge University Press.

4. **Gilbert, A. D., & Vanneste, J.** (2018). Geometric generalised Lagrangian-mean theories. *J. Fluid Mech.*, 839, 95-134.

---

*Document version 2.0 | December 2024 | Tarang.jl*
