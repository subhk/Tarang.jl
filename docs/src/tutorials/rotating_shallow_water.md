# Rotating Shallow Water with Lagrangian Averaging

This tutorial demonstrates how to use **temporal filters** to separate fast inertia-gravity waves from slow geostrophic flow in a rotating shallow water model.

---

## Physical Problem

The rotating shallow water equations on an f-plane:

$$
\frac{\partial u}{\partial t} - fv = -g\frac{\partial \eta}{\partial x}
$$

$$
\frac{\partial v}{\partial t} + fu = -g\frac{\partial \eta}{\partial y}
$$

$$
\frac{\partial \eta}{\partial t} + H\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right) = 0
$$

where:
- $u, v$ = velocity components
- $\eta$ = surface elevation
- $f$ = Coriolis parameter
- $g$ = gravity
- $H$ = mean depth

This system supports two types of motion:
1. **Slow geostrophic flow** - balanced flow satisfying $fv = g\partial\eta/\partial x$
2. **Fast inertia-gravity waves** - oscillations with frequency $\omega^2 = f^2 + c^2(k_x^2 + k_y^2)$

The challenge: **How do we separate these efficiently?**

---

## Solution: Temporal Filtering

Instead of explicit trajectory tracking, we use **exponential temporal filters** that run alongside the simulation. The Butterworth filter provides sharp frequency separation with only two state arrays (the mean $\bar{h}$ and the auxiliary $\tilde{h}$) per filtered field.

### Wave-Mean Decomposition

After filtering:
```
u = ū + u'     (mean + wave fluctuation)
```

The mean $\bar{u}$ captures the slow geostrophic motion, while $u'$ contains the fast wave oscillations.

---

## Complete Example Code

The filters below are the only Tarang machinery this tutorial needs; the shallow water
dynamics are a plain FFTW pseudo-spectral RK4 solver, so the example is self-contained.
The main loop is 10,000 RK4 steps at 64², which takes roughly half a minute — shorten
`T_final` if you only want to see the filter machinery work.

```julia
using Tarang
using Statistics: mean
using Printf

# =============================================================================
# ROTATING SHALLOW WATER MODEL WITH WAVE-MEAN DECOMPOSITION
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Physical Parameters
# -----------------------------------------------------------------------------

# Domain
Lx, Ly = 2π, 2π        # Domain size
Nx, Ny = 64, 64        # Resolution

# Physical constants
f = 1.0                # Coriolis parameter
g = 10.0               # Gravity
H = 1.0                # Mean depth
ν = 0.001              # Viscosity (for numerical stability)

# Derived quantities
c = sqrt(g * H)        # Gravity wave phase speed
T_inertial = 2π / f    # Inertial period

println("Physical parameters:")
println("  Gravity wave speed c = ", c)
println("  Inertial period T = ", round(T_inertial, digits=3))

# -----------------------------------------------------------------------------
# 2. Grid and Operators
# -----------------------------------------------------------------------------

dx, dy = Lx/Nx, Ly/Ny
x = [(i-1)*dx for i in 1:Nx]
y = [(j-1)*dy for j in 1:Ny]

# Spectral wavenumbers
kx = [i <= Nx÷2+1 ? i-1 : i-Nx-1 for i in 1:Nx] .* (2π/Lx)
ky = [j <= Ny÷2+1 ? j-1 : j-Ny-1 for j in 1:Ny] .* (2π/Ly)

# Preallocate FFT arrays
using FFTW
FFTW.set_num_threads(4)
plan_fft = plan_fft!(zeros(ComplexF64, Nx, Ny))
plan_ifft = plan_ifft!(zeros(ComplexF64, Nx, Ny))

# Derivative operators in spectral space
function spectral_gradient!(dfdx, dfdy, f_hat, kx, ky)
    @inbounds for j in 1:size(f_hat, 2), i in 1:size(f_hat, 1)
        dfdx[i,j] = im * kx[i] * f_hat[i,j]
        dfdy[i,j] = im * ky[j] * f_hat[i,j]
    end
end

# -----------------------------------------------------------------------------
# 3. Initialize Fields
# -----------------------------------------------------------------------------

# Velocity and elevation
u = zeros(Nx, Ny)
v = zeros(Nx, Ny)
η = zeros(Nx, Ny)

# Work arrays
u_hat = zeros(ComplexF64, Nx, Ny)
v_hat = zeros(ComplexF64, Nx, Ny)
η_hat = zeros(ComplexF64, Nx, Ny)

# Temporary arrays for derivatives
dudx = zeros(ComplexF64, Nx, Ny)
dudy = zeros(ComplexF64, Nx, Ny)
dvdx = zeros(ComplexF64, Nx, Ny)
dvdy = zeros(ComplexF64, Nx, Ny)
dηdx = zeros(ComplexF64, Nx, Ny)
dηdy = zeros(ComplexF64, Nx, Ny)

# Wave perturbation (k=2, l=2 mode)
k_wave, l_wave = 2.0, 2.0
ω_wave = sqrt(f^2 + c^2 * (k_wave^2 + l_wave^2))
amplitude = 0.1

# Initial condition: geostrophic jet + wave perturbation
for i in 1:Nx, j in 1:Ny
    # Background geostrophic jet.
    # Balance is f*u = -g ∂η/∂y, so u = 0.5 sin(y) ⇒ η = +(f/g) * 0.5 * cos(y).
    u[i,j] = 0.5 * sin(y[j])
    v[i,j] = 0.0
    η[i,j] = (f/g) * 0.5 * cos(y[j])

    η[i,j] += amplitude * cos(k_wave * x[i] + l_wave * y[j])
end

println("\nInitial condition:")
println("  Geostrophic jet: U = 0.5 sin(y)")
println("  Wave perturbation: amplitude = ", amplitude)
println("  Wave frequency ω = ", round(ω_wave, digits=3))

# -----------------------------------------------------------------------------
# 4. Setup Temporal Filters
# -----------------------------------------------------------------------------

T_wave = 2π / ω_wave    # Wave period for k = l = 2
α = 1.0 / (20 * T_wave) # Average over ~20 wave periods

println("\nFilter setup:")
println("  Wave period T_wave = ", round(T_wave, digits=3))
println("  Filter timescale 1/α = ", round(1/α, digits=1))

# Create filters for each prognostic variable
u_filter = ButterworthFilter((Nx, Ny); α=α)
v_filter = ButterworthFilter((Nx, Ny); α=α)
η_filter = ButterworthFilter((Nx, Ny); α=α)

# Precompute ETD coefficients for unconditional stability
dt = 0.01  # Time step
etd_coeffs_u = precompute_etd_coefficients(u_filter, dt)
etd_coeffs_v = precompute_etd_coefficients(v_filter, dt)
etd_coeffs_η = precompute_etd_coefficients(η_filter, dt)

println("  Using ETD integration (unconditionally stable)")
println("  α·dt = ", round(α * dt, digits=4))

# -----------------------------------------------------------------------------
# 5. Time Integration (RK4 for dynamics, ETD for filters)
# -----------------------------------------------------------------------------

function rhs!(du, dv, dη, u, v, η, u_hat, v_hat, η_hat,
              dudx, dudy, dvdx, dvdy, dηdx, dηdy,
              plan_fft, plan_ifft, kx, ky, f, g, H, ν)

    # Transform to spectral space
    u_hat .= u
    v_hat .= v
    η_hat .= η
    plan_fft * u_hat
    plan_fft * v_hat
    plan_fft * η_hat

    # Compute gradients
    spectral_gradient!(dudx, dudy, u_hat, kx, ky)
    spectral_gradient!(dvdx, dvdy, v_hat, kx, ky)
    spectral_gradient!(dηdx, dηdy, η_hat, kx, ky)

    # Transform derivatives back
    plan_ifft * dudx
    plan_ifft * dudy
    plan_ifft * dvdx
    plan_ifft * dvdy
    plan_ifft * dηdx
    plan_ifft * dηdy

    # RHS of shallow water equations
    @inbounds for j in 1:size(u, 2), i in 1:size(u, 1)
        # Nonlinear advection
        adv_u = u[i,j] * real(dudx[i,j]) + v[i,j] * real(dudy[i,j])
        adv_v = u[i,j] * real(dvdx[i,j]) + v[i,j] * real(dvdy[i,j])

        # Momentum equations
        du[i,j] = f * v[i,j] - g * real(dηdx[i,j]) - adv_u
        dv[i,j] = -f * u[i,j] - g * real(dηdy[i,j]) - adv_v

        # Continuity equation
        dη[i,j] = -H * (real(dudx[i,j]) + real(dvdy[i,j]))
    end

    # Add viscous damping in spectral space for stability
    if ν > 0
        u_hat .= u
        v_hat .= v
        plan_fft * u_hat
        plan_fft * v_hat
        @inbounds for j in 1:size(u_hat, 2), i in 1:size(u_hat, 1)
            k2 = kx[i]^2 + ky[j]^2
            visc = -ν * k2
            u_hat[i,j] *= visc
            v_hat[i,j] *= visc
        end
        plan_ifft * u_hat
        plan_ifft * v_hat
        du .+= real.(u_hat)
        dv .+= real.(v_hat)
    end
end

# RK4 time stepper for dynamics
function step_rk4!(u, v, η, dt, args...)
    # Allocate RK stages
    k1_u, k1_v, k1_η = similar(u), similar(v), similar(η)
    k2_u, k2_v, k2_η = similar(u), similar(v), similar(η)
    k3_u, k3_v, k3_η = similar(u), similar(v), similar(η)
    k4_u, k4_v, k4_η = similar(u), similar(v), similar(η)
    u_tmp, v_tmp, η_tmp = similar(u), similar(v), similar(η)

    # k1
    rhs!(k1_u, k1_v, k1_η, u, v, η, args...)

    # k2
    @. u_tmp = u + 0.5 * dt * k1_u
    @. v_tmp = v + 0.5 * dt * k1_v
    @. η_tmp = η + 0.5 * dt * k1_η
    rhs!(k2_u, k2_v, k2_η, u_tmp, v_tmp, η_tmp, args...)

    # k3
    @. u_tmp = u + 0.5 * dt * k2_u
    @. v_tmp = v + 0.5 * dt * k2_v
    @. η_tmp = η + 0.5 * dt * k2_η
    rhs!(k3_u, k3_v, k3_η, u_tmp, v_tmp, η_tmp, args...)

    # k4
    @. u_tmp = u + dt * k3_u
    @. v_tmp = v + dt * k3_v
    @. η_tmp = η + dt * k3_η
    rhs!(k4_u, k4_v, k4_η, u_tmp, v_tmp, η_tmp, args...)

    # Update
    @. u += dt/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u)
    @. v += dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    @. η += dt/6 * (k1_η + 2*k2_η + 2*k3_η + k4_η)
end

# -----------------------------------------------------------------------------
# 6. Main Time Loop
# -----------------------------------------------------------------------------

T_final = 100.0                   # Total simulation time (≈7 filter timescales)
nsteps = Int(T_final / dt)
output_interval = Int(10.0 / dt)  # Output every 10 time units

# Diagnostics storage
times = Float64[]
KE_total = Float64[]
KE_mean = Float64[]
KE_wave = Float64[]
PE_total = Float64[]

println("\n" * "="^60)
println("Starting simulation...")
println("="^60)

for step in 1:nsteps
    t = step * dt

    # Step dynamics (RK4)
    step_rk4!(u, v, η, dt,
              u_hat, v_hat, η_hat,
              dudx, dudy, dvdx, dvdy, dηdx, dηdy,
              plan_fft, plan_ifft, kx, ky, f, g, H, ν)

    # Update filters (ETD - unconditionally stable)
    update_etd!(u_filter, u, etd_coeffs_u)
    update_etd!(v_filter, v, etd_coeffs_v)
    update_etd!(η_filter, η, etd_coeffs_η)

    # Output diagnostics
    if step % output_interval == 0
        # Get filtered means
        ū = get_mean(u_filter)
        v̄ = get_mean(v_filter)
        η̄ = get_mean(η_filter)

        # Wave components
        u_wave = u .- ū
        v_wave = v .- v̄
        η_wave = η .- η̄

        # Energy diagnostics
        ke_total = 0.5 * mean(u.^2 .+ v.^2)
        ke_mean = 0.5 * mean(ū.^2 .+ v̄.^2)
        ke_wave = 0.5 * mean(u_wave.^2 .+ v_wave.^2)
        pe_total = 0.5 * g/H * mean(η.^2)

        push!(times, t)
        push!(KE_total, ke_total)
        push!(KE_mean, ke_mean)
        push!(KE_wave, ke_wave)
        push!(PE_total, pe_total)

        @printf("t = %6.2f | KE_total = %.4e | KE_mean = %.4e | KE_wave = %.4e\n",
                t, ke_total, ke_mean, ke_wave)
    end
end

# -----------------------------------------------------------------------------
# 7. Final Analysis
# -----------------------------------------------------------------------------

println("\n" * "="^60)
println("Simulation complete!")
println("="^60)

# Final wave-mean decomposition
ū = get_mean(u_filter)
v̄ = get_mean(v_filter)
η̄ = get_mean(η_filter)

u_wave = u .- ū
v_wave = v .- v̄
η_wave = η .- η̄

println("\nFinal state statistics:")
println("  Mean flow:")
println("    max|ū| = ", round(maximum(abs.(ū)), digits=4))
println("    max|v̄| = ", round(maximum(abs.(v̄)), digits=4))
println("    max|η̄| = ", round(maximum(abs.(η̄)), digits=4))

println("  Wave fluctuations:")
println("    max|u'| = ", round(maximum(abs.(u_wave)), digits=4))
println("    max|v'| = ", round(maximum(abs.(v_wave)), digits=4))
println("    max|η'| = ", round(maximum(abs.(η_wave)), digits=4))

println("\nEnergy partition:")
ke_final_total = 0.5 * mean(u.^2 .+ v.^2)
ke_final_mean = 0.5 * mean(ū.^2 .+ v̄.^2)
ke_final_wave = 0.5 * mean(u_wave.^2 .+ v_wave.^2)
println("  Total KE:     ", round(ke_final_total, digits=6))
println("  Mean flow KE: ", round(ke_final_mean, digits=6),
        " (", round(100*ke_final_mean/ke_final_total, digits=1), "%)")
println("  Wave KE:      ", round(ke_final_wave, digits=6),
        " (", round(100*ke_final_wave/ke_final_total, digits=1), "%)")
```

---

## Understanding the Output

### Filter spin-up

The filters start from zero, so the mean grows in over roughly one filter timescale
$1/\alpha \approx 14$. What you see in the diagnostic stream is the wave energy being
handed over to the mean:

```
t =  10.00 | KE_total = 6.8060e-02 | KE_mean = 2.0311e-03 | KE_wave = 4.7654e-02
t =  20.00 | KE_total = 6.4902e-02 | KE_mean = 1.5411e-02 | KE_wave = 1.9322e-02
t =  30.00 | KE_total = 6.0933e-02 | KE_mean = 3.6276e-02 | KE_wave = 4.3692e-03
t =  40.00 | KE_total = 5.8691e-02 | KE_mean = 5.3346e-02 | KE_wave = 4.2893e-04
t =  50.00 | KE_total = 5.8350e-02 | KE_mean = 6.1947e-02 | KE_wave = 1.1300e-03
t =  60.00 | KE_total = 5.7332e-02 | KE_mean = 6.3704e-02 | KE_wave = 1.3469e-03
t =  70.00 | KE_total = 5.5658e-02 | KE_mean = 6.1958e-02 | KE_wave = 6.0873e-04
t =  80.00 | KE_total = 5.4461e-02 | KE_mean = 5.9197e-02 | KE_wave = 2.6110e-04
t =  90.00 | KE_total = 5.4004e-02 | KE_mean = 5.6644e-02 | KE_wave = 7.8581e-04
t = 100.00 | KE_total = 5.2820e-02 | KE_mean = 5.4660e-02 | KE_wave = 5.8412e-04
```

Early on, "KE_wave" is large simply because $\bar{u}$ is still near zero, so
$u' = u - \bar{u}$ is the *whole* flow. Only after $t \gtrsim 3/\alpha \approx 40$ is the
decomposition meaningful.

### Energy partition

The final block prints:

```
Energy partition:
  Total KE:     0.05282
  Mean flow KE: 0.05466 (103.5%)
  Wave KE:      0.000584 (1.1%)
```

Two things are worth reading carefully:

- The mean flow holds essentially all the kinetic energy and the residual wave energy is
  ~1% of it — the filter has done its job.
- The percentages do **not** add to 100%, and the mean flow can even exceed the
  instantaneous total. That is expected, not a bug: the filtered mean is a weighted average
  over the *past*, and viscosity is slowly draining the jet, so $\bar{u}$ lags a slightly
  stronger flow than the one at time $t$. There is also no orthogonality that would force
  $\overline{|u|^2} = |\bar{u}|^2 + |u'|^2$; the cross term $\bar{u}\cdot u'$ is not zero.

### Wave-mean decomposition

The filtered mean recovers the initial geostrophic jet:

```
Final state statistics:
  Mean flow:
    max|ū| = 0.4691      # the jet is 0.5 sin(y), viscously damped and filter-lagged
    max|v̄| = 0.0001      # no mean meridional flow, as expected
    max|η̄| = 0.0468
  Wave fluctuations:
    max|u'| = 0.0592
    max|v'| = 0.0458
    max|η'| = 0.0109
```

Comparing $\bar{u}$ with the analytic jet gives `max|ū − 0.5 sin(y)| ≈ 0.031`, i.e. the mean
profile is the jet to ~6%, the deficit being the viscous decay accumulated over 100 time
units.

---

## Time Integration Methods

### Method 1: Explicit Euler (not recommended)

```julia
update!(filter, field, dt)  # Forward Euler
```
**Stability limit**: $\Delta t \leq \sqrt{2}/\alpha$. `update!` *throws* an `ArgumentError`
above it rather than silently producing garbage.

### Method 2: RK2 (moderate stability)

```julia
update!(filter, field, dt, Val(:RK2))
```
**Stability limit**: $\Delta t \approx 2.183/\alpha$ for the Butterworth filter. Query it
with `max_stable_timestep(filter; method=:RK2)` (and `method=:euler` for Method 1).

### Method 3: ETD (recommended - unconditionally stable)

```julia
coeffs = precompute_etd_coefficients(filter, dt)
update_etd!(filter, field, coeffs)
```
**No stability limit!** Can use any timestep.

### Method 4: IMEX/SBDF (for implicit PDE solvers)

```julia
coeffs = precompute_imex_coefficients(filter, dt; scheme=:SBDF2)
update_imex!(filter, (h_n, h_nm1), coeffs)
```
**No stability limit!** Integrates naturally with SBDF timestepping. Pass a 1-tuple
`(h_n,)` for `:SBDF1` and a 2-tuple `(h_n, h_nm1)` for `:SBDF2`.

---

## Choosing Filter Parameters

### The key parameter: α

$$\alpha = \frac{1}{T_{\text{avg}}}$$

**Rule of thumb**: $T_{\text{avg}} \approx 10$-$100 \times T_{\text{wave}}$

| Waves to filter | Wave period | Recommended α |
|-----------------|-------------|---------------|
| Inertia-gravity waves | $2\pi/\sqrt{f^2 + c^2 k^2}$ | $\alpha = 0.05$-$0.1 \times$ wave frequency |
| Near-inertial waves | $2\pi/f$ | $\alpha \approx f/50$ |
| Internal gravity waves | $2\pi/N$ | $\alpha \approx N/100$ |

Remember that the mean needs $\sim 3/\alpha$ of simulated time to spin up from zero, so a
long averaging window costs you a long transient.

### Filter comparison

| Filter | High-freq rolloff | State | Use case |
|--------|------------------|-------|----------|
| `ExponentialMean` | -20 dB/decade | 1 array (`h̄`) | Simple averaging |
| `ButterworthFilter` | -40 dB/decade | 2 arrays (`h̄`, `h̃`) | Sharp wave-mean separation |

---

## Extensions

### Adding Lagrangian Averaging

For a true Lagrangian mean (following particle motion), `LagrangianFilter` carries a
displacement field $\xi$ alongside the flow. Two API details matter:

- `update_displacement!` takes the velocity as **one array of shape `(field_size..., ndim)`**,
  not a tuple of components. Pack `u`, `v` into a `(Nx, Ny, 2)` array.
- With `filter_type=:butterworth`, `lagrangian_mean!` needs **five** positional arguments —
  the auxiliary state `g̃` is part of the two-stage Butterworth scheme:
  `lagrangian_mean!(filter, gᴸ, g̃, g, dt)`. (Only the `:exponential` filter takes the
  four-argument form.)

Continuing from the state reached above, run the dynamics for another `nsteps` while
advancing the Lagrangian filter:

```julia
lag_filter = LagrangianFilter((Nx, Ny); α=α, filter_type=:butterworth)

uv  = zeros(Nx, Ny, 2)   # velocity packed as (field_size..., ndim)
η_L = zeros(Nx, Ny)      # Lagrangian mean of η (output, modified in place)
η̃   = zeros(Nx, Ny)      # auxiliary Butterworth state for the tracer

for step in 1:nsteps
    step_rk4!(u, v, η, dt,
              u_hat, v_hat, η_hat,
              dudx, dudy, dvdx, dvdy, dηdx, dηdy,
              plan_fft, plan_ifft, kx, ky, f, g, H, ν)

    uv[:, :, 1] .= u
    uv[:, :, 2] .= v
    update_displacement!(lag_filter, uv, dt)
    lagrangian_mean!(lag_filter, η_L, η̃, η, dt)

    # keep the Eulerian means in step, for the Stokes drift below
    update_etd!(u_filter, u, etd_coeffs_u)
    update_etd!(v_filter, v, etd_coeffs_v)
end

# Lagrangian mean velocity, shape (Nx, Ny, 2)
ū_L = get_mean_velocity(lag_filter)
println("max|ū_L| = ", round(maximum(abs.(ū_L[:, :, 1])), digits=4))   # 0.4274
println("max|η_L| = ", round(maximum(abs.(η_L)), digits=4))            # 0.0427
```

`get_displacement(lag_filter)` returns $\xi$ itself, also shaped `(Nx, Ny, 2)`.

Note that `get_mean`, `get_mean_velocity` and `get_displacement` all hand back the filter's
*live* state array rather than a copy. The loop above keeps stepping `u_filter`, so the `ū`
bound back in the final-analysis block changes underneath you as it runs. `copy` the array if
you want a snapshot to keep.

### Computing Stokes Drift

The Stokes drift is the difference between the Lagrangian and Eulerian means, taken
component by component:

```julia
u_stokes = ū_L[:, :, 1] .- get_mean(u_filter)
v_stokes = ū_L[:, :, 2] .- get_mean(v_filter)

println("max|u_stokes| = ", round(maximum(abs.(u_stokes)), digits=5))  # 0.00166
println("max|v_stokes| = ", round(maximum(abs.(v_stokes)), digits=5))  # 0.0
```

Here the drift is small (~0.3% of the jet) because the wave amplitude is small; it is the
wave-induced transport that matters for:
- Pollutant dispersion
- Larval transport in oceans
- Sea ice drift

Note that `update_displacement!` and `lagrangian_mean!` use the small-displacement
approximation unless you supply an `interpolate_fn` keyword that evaluates a field at the
displaced positions $x + \xi$.

---

## References

1. **Minz, C., Baker, L. E., Kafiabad, H. A., & Vanneste, J.** (2025). Efficient Lagrangian averaging with exponential filters. *Phys. Rev. Fluids*, 10, 074902. [DOI](https://doi.org/10.1103/PhysRevFluids.10.074902)

2. **Vallis, G. K.** (2017). *Atmospheric and Oceanic Fluid Dynamics* (2nd ed.). Cambridge University Press. Chapter 3: Shallow water systems.

3. **Bühler, O.** (2014). *Waves and Mean Flows* (2nd ed.). Cambridge University Press.

---

## Next Steps

- [Temporal Filters Reference](../pages/temporal_filters.md) - Complete API documentation
- [LES Models](../pages/les_models.md) - Subgrid-scale modeling for turbulence
- [Surface Dynamics](surface_dynamics.md) - SQG and other surface-confined flows

---

*Tutorial version 1.0 | December 2024 | Tarang.jl*
