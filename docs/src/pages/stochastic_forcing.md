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
    injection_metric = :vorticity_kinetic,
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

This keeps one white-noise draw fixed across the stages of a supported one-step method.

---

## Why Stochastic Forcing Needs Special Handling

White-in-time noise requires careful handling for correct statistics:

1. **Stratonovich vs Itô**: Choose the diagnostic interpretation appropriate to the model
2. **Constant within timestep**: For Stratonovich correctness, forcing must be constant across all RK substeps
3. **√dt scaling**: Discrete forcing needs `F = √(Q̂/dt) · noise` to give correct variance

If forcing were evaluated independently at every RK stage, its discrete covariance would be wrong.

---

## Maintained Example: Forced 2D Turbulence

The runnable CPU/GPU example is
[`examples/ivp/forced_2d_turbulence.jl`](https://github.com/subhk/Tarang.jl/blob/main/examples/ivp/forced_2d_turbulence.jl).
It evolves vorticity, solves for streamfunction, derives velocity, and registers ring
forcing on vorticity. Its essential forcing setup is:

```julia
forcing = StochasticForcing(
    field_size = (Nx, Ny),
    domain_size = (Lx, Ly),
    energy_injection_rate = ε,
    injection_metric = :vorticity_kinetic,
    k_forcing = k_f,
    dk_forcing = dk_f,
    dt = max_dt,
    spectrum_type = :ring,
    architecture = device,
)

problem = IVP([ζ, ψ, u, tau_ψ])
add_equation!(problem, "∂t(ζ) = -u⋅∇(ζ) - drag*ζ - nu*Δ⁴(ζ)")
add_equation!(problem, "Δ(ψ) + tau_ψ - ζ = 0")
add_equation!(problem, "u - skew(grad(ψ)) = 0")
add_bc!(problem, "integ(ψ) = 0")
add_stochastic_forcing!(problem, :ζ, forcing)

solver = InitialValueSolver(problem, RK222(); dt=max_dt)
step!(solver)
```

The full file includes initialization, CFL control, output, and CPU/GPU selection.
Run it on CUDA with:

```bash
TARANG_FORCED_2D_DEVICE=gpu julia --project=. examples/ivp/forced_2d_turbulence.jl
```

The prognostic terms remain on the explicit RHS because the pure-Fourier GPU
runtime uses a device-native explicit RK path while refreshing the Poisson and
velocity constraints spectrally at every stage.

---

## Fourier--Chebyshev domains

`StochasticForcing` is the right choice when every spatial direction is
Fourier. A bounded Chebyshev direction is not a wavenumber axis, so applying a
Fourier ring formula to its coefficient index is not physically meaningful.
Use `SeparableStochasticForcing` instead:

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64, device=GPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx))
zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
domain = Domain(dist, (xbasis, zbasis))

b = ScalarField(domain, "b")
forcing = SeparableStochasticForcing(
    fourier_size=(Nx,),
    chebyshev_basis=zbasis,
    chebyshev_profile=z -> z * (Lz - z),
    domain_size=(Lx,),
    energy_injection_rate=0.05,
    injection_metric=:direct,
    k_forcing=4 * 2pi / Lx,
    dk_forcing=2pi / Lx,
    dt=dt,
    architecture=GPU(),
    rng=MersenneTwister(1234),
)
add_stochastic_forcing!(problem, :b, forcing)
solver = InitialValueSolver(problem, RK222(); dt)
```

The function profile is evaluated on the physical Chebyshev grid, transformed
once, and normalized to unit quadrature mean square. You may instead pass an
`Nz`-element vector of Chebyshev coefficients. The forcing then draws only the
Fourier modes and forms their outer product with that fixed profile. It stays
constant across RK stages and changes at the next timestep.

Mixed forcing supports `injection_metric=:direct`. The
`:vorticity_kinetic` metric is intentionally rejected because its Fourier
`1/|k|²` definition does not extend to a bounded direction without specifying
the model's inverse elliptic operator.

On GPU, forcing arrays and phase generation remain device-resident. Coupled
solver selection and distributed-layout constraints are covered in
[GPU Computing](gpu_computing.md).

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
    injection_metric = :vorticity_kinetic, # forcing is applied to vorticity
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
    # 1. Store qⁿ for the optional Stratonovich work diagnostic
    store_prevsol!(forcing, get_coeff_data(ω))

    # 2. Advance; registered forcing is generated and applied automatically
    step!(solver)

    # 3. Compute work done
    W = work_stratonovich(forcing, get_coeff_data(ω))
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

### Injection metric and normalization

```math
\varepsilon = \frac{1}{2M^2}\sum_k \hat{Q}(k)w_k,
\qquad
w_k = \begin{cases}
1 & \texttt{:direct},\\
|k|^{-2} & \texttt{:vorticity\_kinetic}.
\end{cases}
```

Here `M = prod(field_size)` because Tarang uses unnormalized FFT coefficients. The
default `injection_metric = :direct` normalizes the quadratic invariant of the forced
variable itself. When forcing is applied to 2-D vorticity and ε should mean kinetic-energy
injection, set `injection_metric = :vorticity_kinetic` explicitly. The zero mode has zero
weight for that metric.

The constructor stores the full Fourier spectrum. Diagnostics also accept real-FFT
half-spectra: omitted conjugate modes then have multiplicity 2, except the DC mode and an
even-length Nyquist endpoint, which have multiplicity 1. A positive ε with no representable
nonzero forced mode raises `ArgumentError` instead of silently creating zero forcing.

### Stratonovich vs Itô

**Why Stratonovich?**
- Chain rule works normally: d(f(X)) = f'(X) dX
- Physical limit of colored noise with τ→0
- Same formulas for stochastic and deterministic forcing

**Work calculation:**

| Calculus | Formula |
|----------|---------|
| Stratonovich | W = dt Re⟨(qⁿ + qⁿ⁺¹)/2, F̂⟩_w |
| Itô | W = dt Re⟨qⁿ, F̂⟩_w + ε dt |

The weighted pairing `⟨⋅,⋅⟩_w` includes `M⁻²`, the selected injection metric,
and real-FFT multiplicities. For vorticity-kinetic forcing, pass vorticity coefficients;
the pairing supplies `1/|k|²`.

---

## Supported timesteppers

For Runge--Kutta and other supported one-step methods, forcing stays constant within a timestep:

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

First-order `CNAB1` and `SBDF1` are also supported. Methods that reuse or combine
stochastic RHS/state values across time levels would color white noise or produce the
wrong variance, so Tarang fails before drawing forcing for `CNAB2`, `MCNAB2`, `CNLF2`,
`SBDF2`, `SBDF3`, `SBDF4`, `DiagonalIMEX_SBDF2`, `ETD_CNAB2`, and `ETD_SBDF2`.
Use a supported one-step Runge--Kutta/ETD method or a first-order IMEX method instead.

With `enforce_hermitian=true`, self-conjugate Fourier modes are projected to
`sqrt(2) * real(z)`. The factor preserves their variance while making the inverse transform real.

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
    injection_metric = :direct,          # or :vorticity_kinetic
    k_forcing = 4.0,                     # k_f
    dk_forcing = 1.0,                    # δ_f
    dt = 0.01,
    spectrum_type = :ring,               # :ring, :band, :lowk, :kolmogorov
    rng = Random.MersenneTwister(),       # fresh RNG per forcing instance
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
