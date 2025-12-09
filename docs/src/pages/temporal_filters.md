# Temporal Filters for Lagrangian Averaging

A comprehensive guide to efficient computation of Lagrangian means using exponential time filters.

**Based on:** Minz, Baker, Kafiabad & Vanneste (2025). "Efficient Lagrangian averaging with exponential filters". *Physical Review Fluids* 10, 074902. [DOI: 10.1103/PhysRevFluids.10.074902](https://doi.org/10.1103/PhysRevFluids.10.074902)

---

## Table of Contents

1. [Introduction: The Wave-Mean Flow Problem](#1-introduction-the-wave-mean-flow-problem)
2. [Eulerian vs Lagrangian Averaging](#2-eulerian-vs-lagrangian-averaging)
3. [Exponential Time Filters](#3-exponential-time-filters)
4. [First-Order Exponential Mean Filter](#4-first-order-exponential-mean-filter)
5. [Second-Order Butterworth Filter](#5-second-order-butterworth-filter)
6. [Lagrangian Averaging Framework](#6-lagrangian-averaging-framework)
7. [Implementation in Tarang.jl](#7-implementation-in-tarangjl)
8. [Practical Examples](#8-practical-examples)
9. [Parameter Selection Guide](#9-parameter-selection-guide)
10. [Mathematical Derivations](#10-mathematical-derivations)
11. [References](#11-references)

---

## 1. Introduction: The Wave-Mean Flow Problem

### 1.1 Physical Motivation

Many geophysical and astrophysical flows contain motions on multiple timescales:

| System | Slow Motion | Fast Oscillations |
|--------|-------------|-------------------|
| Ocean | Currents, eddies | Internal gravity waves |
| Atmosphere | Jet streams, weather systems | Gravity waves, acoustic waves |
| Stellar interiors | Differential rotation | Gravity waves, oscillations |
| Plasma | Bulk flows | Alfvén waves, magnetosonic waves |

**The fundamental question:** How do fast waves interact with slow mean flows?

### 1.2 Why This Matters

Understanding wave-mean flow interaction is crucial for:

- **Climate modeling**: Gravity waves transport momentum and affect large-scale circulation
- **Ocean mixing**: Internal waves drive turbulent mixing in the deep ocean
- **Weather prediction**: Wave drag parameterizations in numerical weather models
- **Astrophysics**: Angular momentum transport in stellar and planetary interiors

### 1.3 The Challenge

To study wave-mean flow interaction, we need to **decompose** the flow into:

```
u(x,t) = ū(x,t) + u'(x,t)
         ↑          ↑
      mean flow   wave fluctuation
```

But how do we define the "mean"? This is where Lagrangian averaging becomes essential.

---

## 2. Eulerian vs Lagrangian Averaging

### 2.1 Eulerian Mean (Fixed Point)

The Eulerian mean averages over time at a **fixed spatial location**:

$$\bar{u}^E(\mathbf{x}) = \lim_{T \to \infty} \frac{1}{T} \int_0^T u(\mathbf{x}, t) \, dt$$

**Problem:** Consider a simple oscillating flow $u = U_0 \sin(kx - \omega t)$. At any fixed point, the Eulerian mean is zero: $\bar{u}^E = 0$.

But particles oscillating in this wave experience a **net drift** (Stokes drift) that the Eulerian mean completely misses!

### 2.2 Lagrangian Mean (Following Particles)

The Lagrangian mean averages **following fluid parcels**:

$$\bar{u}^L = \lim_{T \to \infty} \frac{1}{T} \int_0^T u(\mathbf{X}(t), t) \, dt$$

where $\mathbf{X}(t)$ is the trajectory of a fluid parcel.

**Advantage:** Captures the actual transport experienced by material elements.

### 2.3 Stokes Drift: A Concrete Example

Consider surface gravity waves on the ocean. A floating object doesn't just bob up and down—it drifts slowly in the direction of wave propagation.

```
    Wave direction →

    ~~~~~~~~~~~~~~~~~~~~  Surface

    Particle path:     ○→○→○→○
                      ↗     ↘
                     ○       ○
                      ↖     ↙
                       ○←←←○

    The orbit doesn't close → net drift to the right
```

The **Stokes drift velocity** is:

$$u_S = \bar{u}^L - \bar{u}^E$$

For deep water waves: $u_S = \omega k a^2 e^{2kz}$ where $a$ is wave amplitude.

### 2.4 The Generalized Lagrangian Mean (GLM)

The GLM theory (Andrews & McIntyre, 1978) provides a rigorous framework for Lagrangian averaging that:

1. Preserves conservation laws
2. Gives exact wave-action conservation
3. Yields mean equations that can be solved

**Key insight from Minz et al. (2025):** We can compute Lagrangian means **efficiently** using temporal filters, without explicitly tracking particle trajectories.

---

## 3. Exponential Time Filters

### 3.1 The Basic Idea

Instead of averaging over all past time (which requires storing the entire history), we use a **weighted average** that emphasizes recent times:

$$\bar{h}(t) = \int_0^\infty k(\tau) \, h(t - \tau) \, d\tau$$

where $k(\tau)$ is a **filter kernel** that decays for large $\tau$.

### 3.2 Why Exponential Kernels?

The exponential kernel $k(\tau) = \alpha e^{-\alpha \tau}$ is special because:

1. **Causal:** Only uses past data ($\tau > 0$)
2. **Normalized:** $\int_0^\infty k(\tau) d\tau = 1$
3. **Evolutionary:** Can be computed by solving an ODE (no storage needed!)
4. **Simple parameter:** Single parameter $\alpha$ controls the averaging time

### 3.3 Filter Parameter $\alpha$

The parameter $\alpha$ is the **inverse averaging timescale**:

- **Averaging time:** $T_{avg} = 1/\alpha$
- **Cutoff frequency:** $\omega_c = \alpha$

**Physical interpretation:**
- Large $\alpha$ → short memory → less filtering → captures faster variations
- Small $\alpha$ → long memory → more filtering → only slow variations remain

### 3.4 Frequency Domain View

In the frequency domain, filtering becomes multiplication:

$$\hat{\bar{h}}(\omega) = H(\omega) \cdot \hat{h}(\omega)$$

where $H(\omega)$ is the **transfer function**. The key properties are:

| Frequency | Response | Physical meaning |
|-----------|----------|------------------|
| $\omega \ll \alpha$ | $|H| \approx 1$ | Slow variations pass through |
| $\omega = \alpha$ | $|H|^2 = 0.5$ | Half-power (cutoff) |
| $\omega \gg \alpha$ | $|H| \ll 1$ | Fast oscillations filtered out |

---

## 4. First-Order Exponential Mean Filter

### 4.1 Definition

The first-order exponential mean is defined by the convolution:

$$\bar{h}(t) = \alpha \int_0^\infty e^{-\alpha \tau} h(t - \tau) \, d\tau \tag{Minz Eq. 5}$$

### 4.2 Evolution Equation

**Key result:** Instead of computing the integral, we solve the ODE:

$$\frac{d\bar{h}}{dt} = \alpha(h - \bar{h}) \tag{Minz Eq. 7}$$

**Derivation:** Differentiate the convolution integral:
$$\frac{d\bar{h}}{dt} = \alpha h(t) - \alpha \cdot \alpha \int_0^\infty e^{-\alpha\tau} h(t-\tau) d\tau = \alpha h - \alpha \bar{h}$$

### 4.3 Transfer Function

Taking the Laplace transform of the ODE with $s = i\omega$:

$$s\hat{\bar{h}} = \alpha(\hat{h} - \hat{\bar{h}})$$

Solving for the transfer function $H(s) = \hat{\bar{h}}/\hat{h}$:

$$H(s) = \frac{\alpha}{s + \alpha} \tag{Minz Eq. 6}$$

### 4.4 Frequency Response

The **power response** (how much power passes through at each frequency):

$$|H(i\omega)|^2 = \frac{\alpha^2}{\alpha^2 + \omega^2}$$

| Frequency | Power Response | dB |
|-----------|----------------|-----|
| $\omega = 0$ | 1.00 | 0 dB |
| $\omega = \alpha$ | 0.50 | -3 dB |
| $\omega = 2\alpha$ | 0.20 | -7 dB |
| $\omega = 10\alpha$ | 0.01 | -20 dB |

**Rolloff rate:** -20 dB/decade (first-order filter)

### 4.5 Physical Interpretation

The ODE $d\bar{h}/dt = \alpha(h - \bar{h})$ describes **exponential relaxation**:

- If $h > \bar{h}$: the mean increases toward $h$
- If $h < \bar{h}$: the mean decreases toward $h$
- Rate of relaxation: $\alpha$

This is like a spring pulling $\bar{h}$ toward $h$ with stiffness $\alpha$.

### 4.6 Numerical Implementation

**Forward Euler:**
$$\bar{h}^{n+1} = \bar{h}^n + \Delta t \cdot \alpha (h^n - \bar{h}^n)$$

**Stability:** Requires $\Delta t < 2/\alpha$ for stability.

**RK2 (Midpoint):**
```
k₁ = α(hⁿ - h̄ⁿ)
k₂ = α(hⁿ - (h̄ⁿ + Δt/2 · k₁))
h̄ⁿ⁺¹ = h̄ⁿ + Δt · k₂
```

---

## 5. Second-Order Butterworth Filter

### 5.1 Motivation

The first-order filter has a gradual frequency rolloff. For better wave-mean separation, we want:

1. **Flat passband:** Low frequencies pass through unchanged
2. **Sharp cutoff:** Clear separation between pass and stop bands
3. **Steep rolloff:** Strong attenuation of high frequencies

The **Butterworth filter** achieves "maximally flat" frequency response.

### 5.2 Transfer Function

The second-order Butterworth low-pass filter has transfer function:

$$K(s) = \frac{\alpha^2}{s^2 + \sqrt{2}\alpha s + \alpha^2} \tag{Minz Eq. 24}$$

This is derived from the general Butterworth form by setting the cutoff frequency to $\alpha$.

### 5.3 Frequency Response

$$|K(i\omega)|^2 = \frac{\alpha^4}{(\alpha^2 - \omega^2)^2 + 2\alpha^2\omega^2}$$

| Frequency | Power Response | dB |
|-----------|----------------|-----|
| $\omega = 0$ | 1.0000 | 0 dB |
| $\omega = \alpha$ | 0.5000 | -3 dB |
| $\omega = 2\alpha$ | 0.0588 | -12 dB |
| $\omega = 10\alpha$ | 0.0001 | -40 dB |

**Rolloff rate:** -40 dB/decade (second-order filter)

### 5.4 Comparison: Exponential vs Butterworth

```
Power Response |H(ω)|²

1.0 ├─────────╲
    │          ╲
0.8 │           ╲
    │            ╲
0.6 │             ╲    ← Exponential (gradual)
    │              ╲
0.4 │          ╲    ╲
    │           ╲    ╲
0.2 │            ╲    ╲
    │             ╲    ╲
0.0 ├──────────────╲────╲───────────
    0      α      2α    5α    10α  → ω

           Butterworth ↗
           (sharp cutoff)
```

At $\omega = 10\alpha$:
- Exponential: 1% power passes through
- Butterworth: 0.01% power passes through (100× better!)

### 5.5 Time Domain Kernel

The inverse Laplace transform of $K(s)$ gives the impulse response:

$$k(t) = \sqrt{2}\alpha \exp\left(-\frac{\alpha t}{\sqrt{2}}\right) \sin\left(\frac{\alpha t}{\sqrt{2}}\right) \Theta(t) \tag{Minz Eq. 25}$$

where $\Theta(t)$ is the Heaviside step function.

**Key features:**
1. Starts at zero (not at maximum like exponential)
2. Peaks at $t \approx \sqrt{2}/\alpha$
3. Has oscillatory character (the sine term)
4. Decays exponentially

### 5.6 Coupled Evolution Equations

The Butterworth filter requires **two state variables**: $\tilde{h}$ (auxiliary) and $\bar{h}$ (mean).

From partial fraction decomposition of $K(s)$ (see Section 10 for derivation):

$$\frac{d\tilde{h}}{dt} = \alpha \left[ h - (\sqrt{2}-1)\tilde{h} - (2-\sqrt{2})\bar{h} \right] \tag{Minz Eq. 30a}$$

$$\frac{d\bar{h}}{dt} = \alpha (\tilde{h} - \bar{h}) \tag{Minz Eq. 30b}$$

### 5.7 Matrix Form

Define state vector $\mathbf{y} = (\tilde{h}, \bar{h})^T$:

$$\frac{d\mathbf{y}}{dt} = -\alpha \mathbf{A} \mathbf{y} + \alpha \begin{pmatrix} h \\ 0 \end{pmatrix}$$

where the **filter matrix** is:

$$\mathbf{A} = \begin{pmatrix} \sqrt{2}-1 & 2-\sqrt{2} \\ -1 & 1 \end{pmatrix} \approx \begin{pmatrix} 0.414 & 0.586 \\ -1 & 1 \end{pmatrix} \tag{Minz Eq. 31}$$

### 5.8 Eigenvalue Analysis

The eigenvalues of $\mathbf{A}$ are:

$$\lambda_{\pm} = \frac{\sqrt{2} \pm i\sqrt{2}}{2}$$

The complex eigenvalues explain:
- Real part $(\sqrt{2}/2)$: exponential decay rate
- Imaginary part $(±\sqrt{2}/2)$: oscillation in the kernel

### 5.9 Numerical Implementation

**Forward Euler:**
```julia
h̃_new = h̃ + Δt · α · (h - A₁₁·h̃ - A₁₂·h̄)
h̄_new = h̄ + Δt · α · (h̃ - h̄)
```

where $A_{11} = \sqrt{2}-1 \approx 0.414$ and $A_{12} = 2-\sqrt{2} \approx 0.586$.

---

## 6. Lagrangian Averaging Framework

### 6.1 The Lifting Map

The key to efficient Lagrangian averaging is the **lifting map** $\boldsymbol{\Xi}(\mathbf{x}, t)$:

$$\boldsymbol{\Xi}(\mathbf{x}, t) = \text{position at time } t \text{ of the parcel whose \textbf{mean} position is } \mathbf{x}$$

**Note:** This is the inverse of the usual Lagrangian labeling!

### 6.2 Displacement Field

The **displacement** $\boldsymbol{\xi}$ is the deviation from the mean position:

$$\boldsymbol{\xi}(\mathbf{x}, t) = \boldsymbol{\Xi}(\mathbf{x}, t) - \mathbf{x} \tag{Minz Eq. 10}$$

For periodic motions (waves), $\boldsymbol{\xi}$ is periodic in time with zero mean.

### 6.3 Velocity Composition

The actual velocity at the displaced position is:

$$\mathbf{u} \circ \boldsymbol{\Xi} = \mathbf{u}(\mathbf{x} + \boldsymbol{\xi}, t)$$

This is the velocity experienced by the particle whose mean position is $\mathbf{x}$.

### 6.4 Mean Velocity Relations

**For exponential mean:**

$$\bar{\mathbf{u}} = \alpha \boldsymbol{\xi} \tag{Minz Eq. 12}$$

The mean velocity is simply proportional to the displacement!

**Derivation:** The lifting map definition gives:
$$\frac{\partial \boldsymbol{\Xi}}{\partial t} = \mathbf{u} \circ \boldsymbol{\Xi}$$

Taking the exponential mean:
$$\frac{\partial \bar{\boldsymbol{\Xi}}}{\partial t} = \overline{\mathbf{u} \circ \boldsymbol{\Xi}}$$

Since $\bar{\boldsymbol{\Xi}} = \mathbf{x}$ (mean position is $\mathbf{x}$ by definition), we need
$$\bar{\mathbf{u}}^L = \overline{\mathbf{u} \circ \boldsymbol{\Xi}}$$

The exponential filter then gives $\bar{\mathbf{u}} = \alpha\boldsymbol{\xi}$.

**For Butterworth filter:**

$$\bar{\mathbf{u}} = \alpha \tilde{\boldsymbol{\xi}} \tag{Minz Eq. 38a}$$

$$\tilde{\mathbf{u}} = \alpha \left[ \boldsymbol{\xi} - (\sqrt{2}-1)\tilde{\boldsymbol{\xi}} \right]$$

### 6.5 Displacement Evolution

The displacement satisfies:

$$\frac{\partial \boldsymbol{\xi}}{\partial t} + \bar{\mathbf{u}} \cdot \nabla \boldsymbol{\xi} = \mathbf{u} \circ (\mathbf{id} + \boldsymbol{\xi}) - \bar{\mathbf{u}} \tag{Minz Eq. 13/38b}$$

**Physical interpretation:**
- LHS: Material derivative of displacement following the mean flow
- RHS: Difference between actual velocity (at displaced position) and mean velocity

### 6.6 Lagrangian Mean of Scalars

For a scalar field $g$ (temperature, salinity, vorticity, etc.), the Lagrangian mean $g^L$ satisfies:

**Exponential mean:**
$$\frac{\partial g^L}{\partial t} + \bar{\mathbf{u}} \cdot \nabla g^L = \alpha (g \circ \boldsymbol{\Xi} - g^L) \tag{Minz Eq. 14}$$

**Butterworth (coupled system):**
$$\frac{\partial \tilde{g}}{\partial t} + \bar{\mathbf{u}} \cdot \nabla \tilde{g} = \alpha \left[ g \circ \boldsymbol{\Xi} - (\sqrt{2}-1)\tilde{g} - (2-\sqrt{2})g^L \right] \tag{Minz Eq. 37a}$$

$$\frac{\partial g^L}{\partial t} + \bar{\mathbf{u}} \cdot \nabla g^L = \alpha (\tilde{g} - g^L) \tag{Minz Eq. 37b}$$

### 6.7 Complete Algorithm

```
At each timestep:
1. Compute u∘(id + ξ) by interpolating velocity to displaced positions
2. Update displacement ξ (and ξ̃ for Butterworth)
3. Compute mean velocity ū = α·ξ (or α·ξ̃ for Butterworth)
4. For each scalar g to average:
   a. Compute g∘Ξ by interpolation
   b. Update gᴸ (and g̃ for Butterworth)
```

---

## 7. Implementation in Tarang.jl

### 7.1 Available Types

```julia
# Abstract base type
abstract type TemporalFilter end

# First-order exponential filter
struct ExponentialMean{T, N} <: TemporalFilter
    α::T                    # Inverse averaging timescale
    h̄::Array{T, N}          # Filtered mean
    field_size::NTuple{N, Int}
end

# Second-order Butterworth filter
struct ButterworthFilter{T, N} <: TemporalFilter
    α::T                    # Inverse averaging timescale
    h̃::Array{T, N}          # Auxiliary field
    h̄::Array{T, N}          # Filtered mean
    A::SMatrix{2, 2, T}     # Filter matrix
    field_size::NTuple{N, Int}
end

# Full Lagrangian filter with displacement tracking
struct LagrangianFilter{T, N, F} <: TemporalFilter
    temporal_filter::F      # Underlying filter (Exp or But)
    ξ::Array{T}             # Displacement field
    ξ̃::Union{Array{T}, Nothing}  # Auxiliary displacement
    ū::Array{T}             # Mean velocity
    α::T
    field_size::NTuple{N, Int}
    ndim::Int
end
```

### 7.2 Constructors

```julia
# Create exponential mean filter for 2D field
exp_filter = ExponentialMean((Nx, Ny); α=0.5, dtype=Float64)

# Create Butterworth filter
but_filter = ButterworthFilter((Nx, Ny); α=0.5, dtype=Float64)

# Create Lagrangian filter (full displacement tracking)
lag_filter = LagrangianFilter((Nx, Ny);
    α=0.5,
    filter_type=:butterworth,  # or :exponential
    dtype=Float64
)
```

### 7.3 Core Functions

```julia
# Update filter with new field data
update!(filter, h, dt)           # Forward Euler
update!(filter, h, dt, Val(:RK2))  # RK2 for better accuracy

# Access filtered mean
h_mean = get_mean(filter)

# Access auxiliary field (Butterworth only)
h_aux = get_auxiliary(filter)

# Reset filter state
reset!(filter)

# Change α dynamically
set_α!(filter, new_α)

# Query filter properties
T_avg = effective_averaging_time(filter)  # Returns 1/α
response = filter_response(filter, ω)      # Returns |H(ω)|²
```

### 7.4 Lagrangian Filter Functions

```julia
# Update displacement field
update_displacement!(lag_filter, velocity, dt)

# Get mean velocity
ū = get_mean_velocity(lag_filter)

# Get displacement field
ξ = get_displacement(lag_filter)

# Compute Lagrangian mean of scalar (exponential)
lagrangian_mean!(lag_filter, gᴸ, g, dt)

# Compute Lagrangian mean of scalar (Butterworth)
lagrangian_mean!(lag_filter, gᴸ, g̃, g, dt)
```

---

## 8. Practical Examples

### 8.1 Simple Temporal Filtering

Filter a time series to extract the slowly varying mean:

```julia
using Tarang

# Setup
Nx, Ny = 64, 64
α = 0.5  # Averaging time = 2 time units
dt = 0.01
nsteps = 1000

# Create Butterworth filter
filter = ButterworthFilter((Nx, Ny); α=α)

# Generate test signal: slow trend + fast oscillations
function test_field(t)
    slow = 1.0 + 0.2 * t  # Slow linear trend
    fast = 0.5 * sin(20.0 * t)  # Fast oscillation (ω = 20 >> α = 0.5)
    return fill(slow + fast, Nx, Ny)
end

# Time integration
t = 0.0
for step in 1:nsteps
    h = test_field(t)
    update!(filter, h, dt)
    t += dt
end

# The filtered mean should capture the slow trend, not the fast oscillation
h_mean = get_mean(filter)
println("Mean value: $(h_mean[1,1])")  # Should be ≈ 1.0 + 0.2*t_final
```

### 8.2 Wave-Mean Flow Decomposition

Separate wave fluctuations from mean flow in a 2D simulation:

```julia
using Tarang

# Domain setup
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; mesh=(1,1))
basis_x = RealFourier(coords["x"]; size=64, bounds=(0.0, 2π))
basis_y = RealFourier(coords["y"]; size=64, bounds=(0.0, 2π))

# Create velocity and vorticity fields
u = VectorField(dist, "u", (basis_x, basis_y))
ζ = ScalarField(dist, "ζ", (basis_x, basis_y))  # Vorticity

# Temporal filter for vorticity (using standalone filter)
α = 0.1  # Average over ~10 time units
ζ_filter = ButterworthFilter((64, 64); α=α)

# In your simulation loop:
dt = 0.01
for step in 1:nsteps
    # ... advance u, ζ using your time stepper ...

    # Update the temporal filter with current vorticity
    ζ_data = parent(ζ.spectral_data)  # Get raw array
    update!(ζ_filter, real.(ζ_data), dt)

    # Decomposition
    ζ_mean = get_mean(ζ_filter)  # Slowly varying part
    ζ_wave = real.(ζ_data) - ζ_mean  # Wave fluctuation

    # Analysis: compute wave energy, enstrophy, etc.
end
```

### 8.3 Full Lagrangian Averaging

Track particle displacements and compute Lagrangian means:

```julia
using Tarang

Nx, Ny = 64, 64
α = 0.2  # Averaging time = 5 time units

# Create Lagrangian filter with Butterworth temporal filter
lag_filter = LagrangianFilter((Nx, Ny); α=α, filter_type=:butterworth)

# Arrays for velocity components
u = zeros(Nx, Ny, 2)  # u[:,:,1] = u-velocity, u[:,:,2] = v-velocity

# Arrays for Lagrangian mean of temperature
T = zeros(Nx, Ny)      # Temperature field
Tᴸ = zeros(Nx, Ny)     # Lagrangian mean temperature
T̃ = zeros(Nx, Ny)      # Auxiliary (for Butterworth)

dt = 0.01
for step in 1:nsteps
    # ... your physics: update u and T ...

    # Update displacement and mean velocity
    update_displacement!(lag_filter, u, dt)

    # Get Lagrangian mean velocity
    ū = get_mean_velocity(lag_filter)

    # Compute Lagrangian mean of temperature
    lagrangian_mean!(lag_filter, Tᴸ, T̃, T, dt)

    # ū is the mean velocity, Tᴸ is the Lagrangian mean temperature
end
```

### 8.4 Comparing Filter Responses

Visualize the frequency response of different filters:

```julia
using Tarang

α = 1.0

# Create both filter types
exp_filter = ExponentialMean((10,); α=α)
but_filter = ButterworthFilter((10,); α=α)

# Compute frequency response
ω_range = 0.0:0.1:20.0
exp_response = [filter_response(exp_filter, ω) for ω in ω_range]
but_response = [filter_response(but_filter, ω) for ω in ω_range]

# The Butterworth filter has much stronger attenuation at high frequencies
# At ω = 10α: exponential ≈ 0.01 (1%), butterworth ≈ 0.0001 (0.01%)
```

---

## 9. Parameter Selection Guide

### 9.1 Choosing α

The inverse averaging timescale $\alpha$ is the most important parameter.

**Step 1: Identify wave frequencies**

Determine the characteristic frequencies of waves you want to filter:
- Internal gravity waves: $\omega \sim N$ (buoyancy frequency)
- Inertia-gravity waves: $\omega \sim f$ to $N$ (Coriolis to buoyancy)
- Surface waves: $\omega = \sqrt{gk}$

**Step 2: Choose α for separation**

For good wave-mean separation:

$$\alpha \ll \omega_{wave}$$

**Rule of thumb (from Minz et al. 2025, Fig. 3):**
- Butterworth effectively filters waves with $\omega \gtrsim 20\alpha$
- Exponential filters waves with $\omega \gtrsim 50\alpha$

**Examples:**

| Application | Wave period | Suggested α |
|-------------|-------------|-------------|
| Ocean internal waves | 1 hour | 0.01 per hour |
| Atmospheric gravity waves | 10 min | 0.01 per minute |
| Near-inertial oscillations | 1 day | 0.05 per day |

### 9.2 Timestep Constraints

For numerical stability with forward Euler:

$$\Delta t < \frac{2}{\alpha}$$

For accuracy with oscillating signals:

$$\Delta t < \frac{2\pi}{10 \omega_{max}}$$

**Recommendation:** Use RK2 (`Val(:RK2)`) if $\alpha \cdot \Delta t > 0.1$.

### 9.3 Butterworth vs Exponential

**Use Butterworth when:**
- You need sharp wave-mean separation
- Waves have a clear spectral gap from mean flow
- Memory is not a constraint (2× storage)

**Use Exponential when:**
- Memory is limited
- You just need a running mean
- Simplicity is preferred

### 9.4 Spinup Time

The filters need time to "spin up" from zero initial conditions.

**Spinup time:** $T_{spinup} \approx 5/\alpha$

During spinup, the filtered mean will underestimate the true mean. Options:
1. Initialize the filter with an estimate of the mean
2. Discard data from the spinup period
3. Use longer averaging time (smaller $\alpha$)

---

## 10. Mathematical Derivations

### 10.1 Derivation of Butterworth Evolution Equations

Starting from the transfer function:

$$K(s) = \frac{\alpha^2}{s^2 + \sqrt{2}\alpha s + \alpha^2}$$

**Step 1: Partial fraction decomposition**

Factor the denominator:
$$s^2 + \sqrt{2}\alpha s + \alpha^2 = (s - p_+)(s - p_-)$$

where the poles are:
$$p_{\pm} = \frac{-\sqrt{2}\alpha \pm i\sqrt{2}\alpha}{2} = -\frac{\alpha}{\sqrt{2}}(1 \mp i)$$

**Step 2: Partial fractions**

$$K(s) = \frac{A}{s - p_+} + \frac{A^*}{s - p_-}$$

where $A = \alpha^2/(p_+ - p_-) = \alpha/(i\sqrt{2})$.

**Step 3: State-space realization**

Define auxiliary variables:
$$\tilde{h} = \mathcal{L}^{-1}\left[\frac{A}{s-p_+} + \frac{A^*}{s-p_-}\right] \cdot \hat{h}$$

After algebra (details in Minz et al. Appendix), we obtain the coupled system:

$$\frac{d\tilde{h}}{dt} = -(\sqrt{2}-1)\alpha\tilde{h} - (2-\sqrt{2})\alpha\bar{h} + \alpha h$$
$$\frac{d\bar{h}}{dt} = \alpha\tilde{h} - \alpha\bar{h}$$

### 10.2 Verification of Filter Matrix Eigenvalues

The filter matrix is:
$$\mathbf{A} = \begin{pmatrix} \sqrt{2}-1 & 2-\sqrt{2} \\ -1 & 1 \end{pmatrix}$$

Characteristic polynomial:
$$\det(\mathbf{A} - \lambda\mathbf{I}) = (\sqrt{2}-1-\lambda)(1-\lambda) + (2-\sqrt{2})$$
$$= \lambda^2 - \sqrt{2}\lambda + 1$$

Eigenvalues:
$$\lambda = \frac{\sqrt{2} \pm \sqrt{2-4}}{2} = \frac{\sqrt{2} \pm i\sqrt{2}}{2}$$

These match the poles of $K(s)/\alpha$ as expected.

### 10.3 Lagrangian Mean Velocity Derivation

From the lifting map definition:
$$\frac{\partial \boldsymbol{\Xi}}{\partial t} = \mathbf{u}(\boldsymbol{\Xi}, t)$$

Since $\boldsymbol{\Xi} = \mathbf{x} + \boldsymbol{\xi}$:
$$\frac{\partial \boldsymbol{\xi}}{\partial t} = \mathbf{u}(\mathbf{x} + \boldsymbol{\xi}, t)$$

Taking the exponential mean (with the specific property that $\bar{\mathbf{x}} = \mathbf{x}$):
$$\overline{\frac{\partial \boldsymbol{\xi}}{\partial t}} = \overline{\mathbf{u}(\mathbf{x} + \boldsymbol{\xi}, t)} = \bar{\mathbf{u}}^L$$

The exponential filter satisfies:
$$\bar{f} = \alpha \int_0^\infty e^{-\alpha\tau} f(t-\tau) d\tau$$

For displacement starting from zero:
$$\bar{\boldsymbol{\xi}} = \alpha \int_0^\infty e^{-\alpha\tau} \boldsymbol{\xi}(t-\tau) d\tau$$

Differentiating and using $\boldsymbol{\xi}(-\infty) = 0$:
$$\frac{d\bar{\boldsymbol{\xi}}}{dt} = \alpha(\boldsymbol{\xi} - \bar{\boldsymbol{\xi}})$$

In the wave-averaged limit where $\bar{\boldsymbol{\xi}} = 0$ (zero mean displacement), this gives:
$$\bar{\mathbf{u}}^L = \frac{d\bar{\boldsymbol{\xi}}}{dt} + \mathcal{O}(\epsilon^2) = \alpha\boldsymbol{\xi} + \mathcal{O}(\epsilon^2)$$

---

## 11. References

### Primary Reference

1. **Minz, C., Baker, L. E., Kafiabad, H. A., & Vanneste, J.** (2025). Efficient Lagrangian averaging with exponential filters. *Physical Review Fluids*, 10, 074902.
   - [DOI: 10.1103/PhysRevFluids.10.074902](https://doi.org/10.1103/PhysRevFluids.10.074902)
   - [arXiv: 2410.00936](https://arxiv.org/abs/2410.00936)

### Background: Lagrangian Averaging Theory

2. **Andrews, D. G., & McIntyre, M. E.** (1978). An exact theory of nonlinear waves on a Lagrangian-mean flow. *Journal of Fluid Mechanics*, 89(4), 609-646.
   - The foundational paper on Generalized Lagrangian Mean (GLM) theory

3. **Bühler, O.** (2014). *Waves and Mean Flows* (2nd ed.). Cambridge University Press.
   - Excellent textbook covering wave-mean flow interaction; Chapter 4 on GLM theory

4. **Gilbert, A. D., & Vanneste, J.** (2018). Geometric generalised Lagrangian-mean theories. *Journal of Fluid Mechanics*, 839, 95-134.
   - Modern geometric perspective on GLM

### Signal Processing Background

5. **Oppenheim, A. V., & Schafer, R. W.** (2010). *Discrete-Time Signal Processing* (3rd ed.). Pearson.
   - Standard reference for filter theory

6. **Butterworth, S.** (1930). On the theory of filter amplifiers. *Experimental Wireless and the Wireless Engineer*, 7, 536-541.
   - Original paper introducing the Butterworth filter

### Applications

7. **Wagner, G. L., & Young, W. R.** (2015). Available potential vorticity and wave-averaged quasi-geostrophic flow. *Journal of Fluid Mechanics*, 785, 401-424.
   - Application of GLM to quasi-geostrophic dynamics

8. **Kafiabad, H. A., Savva, M. A. C., & Vanneste, J.** (2019). Diffusion of inertia-gravity waves by geostrophic turbulence. *Journal of Fluid Mechanics*, 869, R7.
   - Wave-mean flow interaction in geostrophic turbulence

---

## Appendix A: Quick Reference Card

### Filter Equations

| Filter | Evolution Equation | Mean Velocity |
|--------|-------------------|---------------|
| Exponential | $\dot{\bar{h}} = \alpha(h - \bar{h})$ | $\bar{\mathbf{u}} = \alpha\boldsymbol{\xi}$ |
| Butterworth | $\dot{\tilde{h}} = \alpha[h - A_{11}\tilde{h} - A_{12}\bar{h}]$ | $\bar{\mathbf{u}} = \alpha\tilde{\boldsymbol{\xi}}$ |
| | $\dot{\bar{h}} = \alpha(\tilde{h} - \bar{h})$ | |

where $A_{11} = \sqrt{2}-1 \approx 0.414$ and $A_{12} = 2-\sqrt{2} \approx 0.586$.

### Transfer Functions

| Filter | $H(s)$ | $\|H(i\omega)\|^2$ |
|--------|--------|-------------------|
| Exponential | $\frac{\alpha}{s+\alpha}$ | $\frac{\alpha^2}{\alpha^2 + \omega^2}$ |
| Butterworth | $\frac{\alpha^2}{s^2 + \sqrt{2}\alpha s + \alpha^2}$ | $\frac{\alpha^4}{(\alpha^2-\omega^2)^2 + 2\alpha^2\omega^2}$ |

### Key Numbers

| Quantity | Exponential | Butterworth |
|----------|-------------|-------------|
| Averaging time | $1/\alpha$ | $1/\alpha$ |
| Half-power frequency | $\omega = \alpha$ | $\omega = \alpha$ |
| Rolloff rate | -20 dB/decade | -40 dB/decade |
| Memory (per field) | 1 array | 2 arrays |
| Response at $\omega=10\alpha$ | 0.0099 | 0.0001 |

---

*Document version 1.0 | December 2024 | Tarang.jl*
