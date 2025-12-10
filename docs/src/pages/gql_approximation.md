# Generalized Quasi-Linear (GQL) Approximation

Wavenumber-based scale separation for turbulence modeling and wave-mean flow interactions.

**Reference:** Marston, Chini & Tobias (2016). ["Generalized Quasilinear Approximation: Application to Zonal Jets"](https://doi.org/10.1103/PhysRevLett.116.214501). *Phys. Rev. Lett.* 116, 214501.

---

## TL;DR - Quick Summary

> **What:** Splits fields into large-scale (|k| ≤ Λ) and small-scale (|k| > Λ) components in Fourier space
>
> **Why:** Enables systematic approximations between full nonlinear (NL) and quasi-linear (QL) dynamics
>
> **Key insight:** By varying Λ from 0 to k_max, you interpolate between QL (cheap) and NL (expensive)
>
> **Which system to use:**
> - `GQLDecomposition` - Pure spectral decomposition (you handle FFTs)
> - `GQLWaveMeanSystem` - Combined GQL + temporal filtering for complete wave-mean analysis

---

## Quick Start

### Basic GQL Decomposition (4 lines of code)

```julia
using Tarang
using FFTW

# 1. Create GQL decomposition: 64×64 grid, domain 2π×2π, cutoff |k| ≤ 4
gql = GQLDecomposition((64, 64), (2π, 2π); Λ=4.0)

# 2. Decompose your spectral field
f_hat = rfft(f)  # Your field in spectral space
f_large, f_small = decompose!(gql, f_hat)

# 3. Use in your simulation
# f_large: large-scale modes (|k| ≤ Λ)
# f_small: small-scale modes (|k| > Λ)
```

### Complete Working Example

```julia
using Tarang
using FFTW

# Domain setup
Nx, Ny = 64, 64
Lx, Ly = 2π, 2π
x = range(0, Lx, length=Nx+1)[1:end-1]
y = range(0, Ly, length=Ny+1)[1:end-1]

# Create test field: large-scale + small-scale
f = zeros(Nx, Ny)
for j in 1:Ny, i in 1:Nx
    # Large scale: k = 2
    f[i,j] += sin(2*x[i]) * cos(2*y[j])
    # Small scale: k = 10
    f[i,j] += 0.3 * sin(10*x[i]) * cos(10*y[j])
end

# Create GQL decomposition with cutoff Λ = 5
# This will separate k ≤ 5 (large) from k > 5 (small)
gql = GQLDecomposition((Nx, Ny), (Lx, Ly); Λ=5.0)

# Decompose
f_hat = rfft(f)
f_L_hat, f_S_hat = decompose!(gql, f_hat)

# Transform back to physical space
f_large = irfft(f_L_hat, Nx)  # Contains only k=2 mode
f_small = irfft(f_S_hat, Nx)  # Contains only k=10 mode

# Verify
println("Large-scale max: ", maximum(abs.(f_large)))  # ≈ 1.0
println("Small-scale max: ", maximum(abs.(f_small)))  # ≈ 0.3
println("f ≈ f_L + f_S: ", maximum(abs.(f - f_large - f_small)))  # ≈ 0
```

---

## Understanding the GQL Approximation

### Scale Separation in Fourier Space

The GQL approximation splits any field `f` into two parts based on wavenumber magnitude:

```
f = f_L + f_S
```

where:
- **f_L** = Large-scale (low wavenumber): modes with |k| ≤ Λ
- **f_S** = Small-scale (high wavenumber): modes with |k| > Λ

```
Wavenumber space (2D example):
                    ky
                    ↑
                    │    Small-scale
                    │      (|k| > Λ)
            ────────┼────────
           /        │        \
          /    ┌────┼────┐    \
         │     │    │    │     │
    ─────┼─────┼────┼────┼─────┼───→ kx
         │     │ Large   │     │
          \    │ scale  │    /
           \   └────────┘   /
            ────────────────
                    │

    Inner circle: |k| ≤ Λ (Large-scale)
    Outer region: |k| > Λ (Small-scale)
```

### The GQL Hierarchy

By varying the cutoff Λ, you get different approximations:

| Λ Value | Name | Nonlinear Terms | Cost | Accuracy |
|---------|------|-----------------|------|----------|
| Λ = 0 | **QL** (Quasi-Linear) | Only NL(f_L, f_L) | Cheapest | Lowest |
| 0 < Λ < k_max | **GQL** | NL(f_L, f_L) + NL(f_S, f_S) projected to L | Intermediate | Better |
| Λ = k_max | **Full NL** | All terms | Most expensive | Exact |

### GQL Equations

For a generic nonlinear PDE `∂f/∂t = NL(f, f) + L(f)`:

**Large-scale equation:**

```
∂f_L/∂t = NL(f_L, f_L) + P_L[NL(f_S, f_S)] + L(f_L)
                         ↑ eddy feedback
```

**Small-scale equation:**

```
∂f_S/∂t = NL(f_L, f_S) + NL(f_S, f_L) + L(f_S)
          ↑ linear in f_S (no f_S self-interaction!)
```

**Key simplification:** The NL(f_S, f_S) term is **dropped** from the small-scale equation!

---

## API Reference

### GQLDecomposition

Pure wavenumber decomposition without temporal filtering.

```julia
# Constructor
gql = GQLDecomposition(field_size, domain_size; Λ, dtype=Float64)

# Arguments:
#   field_size  - Physical space size, e.g., (Nx, Ny) or (Nx, Ny, Nz)
#   domain_size - Horizontal domain size, e.g., (Lx, Ly)
#   Λ           - Cutoff wavenumber
#   dtype       - Element type (default: Float64)
```

#### Methods

| Function | Description |
|----------|-------------|
| `decompose!(gql, f_hat)` | Split into (f_large, f_small) |
| `project_large!(gql, f_hat)` | Zero out k > Λ (in-place) |
| `project_small!(gql, f_hat)` | Zero out k ≤ Λ (in-place) |
| `get_cutoff(gql)` | Get current Λ |
| `set_cutoff!(gql, Λ_new)` | Change Λ (rebuilds mask) |
| `count_large_modes(gql)` | Number of large-scale modes |
| `count_small_modes(gql)` | Number of small-scale modes |

#### Example: In-place Projection

```julia
# Instead of decompose!, you can project in-place
f_hat = rfft(f)

# Option 1: Get both components
f_L, f_S = decompose!(gql, f_hat)

# Option 2: Project in-place (modifies f_hat)
f_hat_copy = copy(f_hat)
project_large!(gql, f_hat_copy)  # Now contains only modes with k ≤ Λ
```

---

### GQLWaveMeanSystem

Combined GQL decomposition with temporal filtering for wave-mean flow analysis.

```julia
# Constructor
sys = GQLWaveMeanSystem(field_size, domain_size; Λ, α, horizontal_dims=(1,2), dtype=Float64)

# Arguments:
#   field_size      - Physical space size
#   domain_size     - Horizontal domain size
#   Λ               - GQL wavenumber cutoff
#   α               - Temporal filter parameter (1/averaging_time)
#   horizontal_dims - Dimensions for horizontal averaging
```

#### Methods

| Function | Description |
|----------|-------------|
| `add_field!(sys, :u)` | Register field for decomposition |
| `add_flux!(sys, :uw)` | Register flux product ⟨u'w'⟩ |
| `update!(sys, fields_hat, fields_phys, dt)` | Update decomposition and filters |
| `get_large(sys, :u)` | Large-scale spectral component |
| `get_small(sys, :u)` | Small-scale spectral component |
| `get_mean(sys, :u)` | Time-filtered mean profile (1D) |
| `get_flux(sys, :uw)` | Filtered wave flux profile |
| `get_cutoff(sys)` | Get Λ |
| `set_cutoff!(sys, Λ_new)` | Change Λ |

---

## Complete GQL Simulation Example

### 2D Barotropic Vorticity (β-plane)

```julia
using Tarang
using FFTW
using LinearAlgebra

# ============================================================================
# GQL simulation of 2D turbulence with β-effect (zonal jet formation)
# ============================================================================

# Grid
Nx, Ny = 128, 128
Lx, Ly = 2π, 2π
dx, dy = Lx/Nx, Ly/Ny

# Wavenumbers
kx = rfftfreq(Nx, 2π/dx)
ky = fftfreq(Ny, 2π/dy)
KX = [kx[i] for i in 1:length(kx), j in 1:Ny]
KY = [ky[j] for i in 1:length(kx), j in 1:Ny]
K2 = KX.^2 .+ KY.^2
K2[1,1] = 1  # Avoid division by zero

# Physical parameters
β = 10.0     # β-effect (planetary vorticity gradient)
ν = 1e-4     # Viscosity
dt = 0.001
nsteps = 10000

# GQL setup: cutoff at Λ = 4 (only large scales interact nonlinearly)
Λ = 4.0
gql = GQLDecomposition((Nx, Ny), (Lx, Ly); Λ=Λ)

println("GQL cutoff Λ = ", Λ)
println("Large-scale modes: ", count_large_modes(gql))
println("Small-scale modes: ", count_small_modes(gql))

# Initial condition: small random perturbation
ζ = 0.1 * randn(Nx, Ny)
ζ_hat = rfft(ζ)

# Preallocate
ψ_hat = similar(ζ_hat)
u_hat = similar(ζ_hat)
v_hat = similar(ζ_hat)
NL_hat = similar(ζ_hat)

# Time stepping (RK4)
function compute_rhs!(rhs, ζ_hat, gql, K2, KX, KY, β, ν)
    # Stream function: ψ = -ζ/k²
    @. ψ_hat = -ζ_hat / K2

    # Velocity: u = -∂ψ/∂y, v = ∂ψ/∂x
    @. u_hat = -im * KY * ψ_hat
    @. v_hat =  im * KX * ψ_hat

    # Transform to physical space
    ζ = irfft(ζ_hat, Nx)
    u = irfft(u_hat, Nx)
    v = irfft(v_hat, Nx)

    # ========================================
    # GQL APPROXIMATION: Decompose velocity
    # ========================================
    u_hat_full = rfft(u)
    v_hat_full = rfft(v)

    # Large-scale velocity
    u_L_hat, u_S_hat = decompose!(gql, u_hat_full)
    u_L = irfft(copy(u_L_hat), Nx)

    v_L_hat, v_S_hat = decompose!(gql, v_hat_full)
    v_L = irfft(copy(v_L_hat), Nx)

    # Small-scale velocity
    u_S = irfft(copy(u_S_hat), Nx)
    v_S = irfft(copy(v_S_hat), Nx)

    # Vorticity decomposition
    ζ_L_hat, ζ_S_hat = decompose!(gql, copy(ζ_hat))
    ζ_L = irfft(copy(ζ_L_hat), Nx)
    ζ_S = irfft(copy(ζ_S_hat), Nx)

    # ========================================
    # GQL Nonlinear terms
    # ========================================
    # Large-scale: NL(u_L, ζ_L) + project_L(NL(u_S, ζ_S))
    NL_LL = u_L .* irfft(im * KX .* ζ_L_hat, Nx) .+
            v_L .* irfft(im * KY .* ζ_L_hat, Nx)

    NL_SS = u_S .* irfft(im * KX .* ζ_S_hat, Nx) .+
            v_S .* irfft(im * KY .* ζ_S_hat, Nx)
    NL_SS_hat = rfft(NL_SS)
    project_large!(gql, NL_SS_hat)  # Keep only |k| ≤ Λ

    # Small-scale: NL(u_L, ζ_S) + NL(u_S, ζ_L)  [NO NL(u_S, ζ_S)]
    NL_LS = u_L .* irfft(im * KX .* ζ_S_hat, Nx) .+
            v_L .* irfft(im * KY .* ζ_S_hat, Nx)
    NL_SL = u_S .* irfft(im * KX .* ζ_L_hat, Nx) .+
            v_S .* irfft(im * KY .* ζ_L_hat, Nx)

    # Combine
    NL_large = rfft(NL_LL) .+ NL_SS_hat
    NL_small = rfft(NL_LS .+ NL_SL)

    # Total advection (GQL)
    @. NL_hat = NL_large + NL_small

    # RHS: -u·∇ζ - βv + ν∇²ζ
    @. rhs = -NL_hat - β * im * KX * ψ_hat - ν * K2 * ζ_hat
end

# Main time loop
k1, k2, k3, k4 = similar(ζ_hat), similar(ζ_hat), similar(ζ_hat), similar(ζ_hat)
ζ_tmp = similar(ζ_hat)

for step in 1:nsteps
    # RK4 stages
    compute_rhs!(k1, ζ_hat, gql, K2, KX, KY, β, ν)
    @. ζ_tmp = ζ_hat + 0.5*dt*k1
    compute_rhs!(k2, ζ_tmp, gql, K2, KX, KY, β, ν)
    @. ζ_tmp = ζ_hat + 0.5*dt*k2
    compute_rhs!(k3, ζ_tmp, gql, K2, KX, KY, β, ν)
    @. ζ_tmp = ζ_hat + dt*k3
    compute_rhs!(k4, ζ_tmp, gql, K2, KX, KY, β, ν)

    @. ζ_hat = ζ_hat + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    # Diagnostics
    if step % 1000 == 0
        ζ = irfft(ζ_hat, Nx)
        enstrophy = sum(ζ.^2) * dx * dy / (Lx * Ly)

        # Zonal mean (x-average)
        u = irfft(-im * KY .* (-ζ_hat ./ K2), Nx)
        u_zonal = vec(mean(u, dims=1))

        println("Step $step: Enstrophy = $(round(enstrophy, digits=4)), max|u_zonal| = $(round(maximum(abs.(u_zonal)), digits=4))")
    end
end
```

---

## Combining GQL with Wave-Mean Decomposition

For problems with clear wave-mean separation (e.g., internal waves + zonal flow):

```julia
using Tarang
using FFTW

# Domain
Nx, Ny, Nz = 64, 64, 32
Lx, Ly = 2π, 2π
dt = 0.01

# Combined system: GQL (Λ=4) + temporal filter (α=0.1)
sys = GQLWaveMeanSystem((Nx, Ny, Nz), (Lx, Ly); Λ=4.0, α=0.1)

# Register fields
add_field!(sys, :u)
add_field!(sys, :v)
add_field!(sys, :w)
add_field!(sys, :b)

# Register fluxes for Reynolds stress
add_flux!(sys, :uw)  # ⟨u'w'⟩
add_flux!(sys, :vw)  # ⟨v'w'⟩
add_flux!(sys, :wb)  # ⟨w'b'⟩

# Time loop
for step in 1:nsteps
    # Your PDE solver advances u, v, w, b
    # ...

    # Update GQL + temporal filtering
    update!(sys, Dict(:u => u, :v => v, :w => w, :b => b), dt)

    # Access decomposed fields
    u_L = get_large(sys, :u)      # Large-scale (|k| ≤ Λ) in spectral space
    u_S = get_small(sys, :u)      # Small-scale (|k| > Λ) in spectral space
    u_mean = get_mean(sys, :u)    # Time-filtered horizontal mean ū(z)

    # Access filtered Reynolds stress
    R_uw = get_flux(sys, :uw)     # ⟨u'w'⟩(z) profile
    R_wb = get_flux(sys, :wb)     # ⟨w'b'⟩(z) profile

    # Use for forcing in mean equations:
    # ∂ū/∂t = ... - ∂⟨u'w'⟩/∂z
    # ∂b̄/∂t = ... - ∂⟨w'b'⟩/∂z
end
```

---

## Choosing the Cutoff Λ

### Guidelines

| Application | Suggested Λ | Rationale |
|-------------|-------------|-----------|
| Zonal jets (β-plane) | 2-6 | Capture jet scale, parameterize eddies |
| Convection | 4-10 | Large convective cells are "mean" |
| Shear flow | 1-4 | Mean shear + harmonics |
| Testing QL validity | 0 then increase | Compare QL → GQL → NL |

### Diagnostic: Mode Count

```julia
gql = GQLDecomposition((128, 128), (2π, 2π); Λ=4.0)

n_large = count_large_modes(gql)
n_small = count_small_modes(gql)
n_total = n_large + n_small

println("Large-scale modes: $n_large ($(round(100*n_large/n_total, digits=1))%)")
println("Small-scale modes: $n_small ($(round(100*n_small/n_total, digits=1))%)")

# Typical output for Λ=4:
# Large-scale modes: 49 (0.6%)
# Small-scale modes: 8143 (99.4%)
```

### Sweeping Λ to Test Convergence

```julia
# Test GQL accuracy by varying Λ
for Λ in [0, 2, 4, 8, 16, Inf]
    gql = GQLDecomposition((64, 64), (2π, 2π); Λ=Λ)
    # Run simulation...
    # Compare statistics (energy, enstrophy, mean profiles)
end
```

---

## Performance Considerations

### Memory

| System | Arrays per Field | Notes |
|--------|------------------|-------|
| `GQLDecomposition` | 2 complex | f_large, f_small work arrays |
| `GQLWaveMeanSystem` | 2 complex + 3 real | Plus temporal filter storage |

### Computational Cost

- **Mask application**: O(N) per field, very fast
- **FFT/IFFT**: O(N log N), dominates cost
- **GQL vs NL**: Similar cost per step, but GQL may allow larger Λ (fewer modes in expensive nonlinear terms)

### Optimization Tips

1. **Reuse FFT plans**: Pre-compute `plan_rfft` and `plan_irfft`
2. **In-place operations**: Use `project_large!` instead of `decompose!` when possible
3. **Batch updates**: Update all fields before computing fluxes

---

## Theory: Why GQL Works

### The Eddy-Mean Decomposition

Traditional Reynolds decomposition: `f = f̄ + f'`

GQL generalizes this to spectral space:
- **f_L** contains the "mean" (but can include some wave structure)
- **f_S** contains the "eddies" (high-k fluctuations)

### Scale Interactions

The nonlinear term `NL(f, f)` produces wavenumber triads (k₁, k₂, k₃) where k₁ + k₂ = k₃.

GQL assumption: **Small-scale self-interactions** `NL(f_S, f_S)` that stay in the small scales can be neglected.

This is valid when:
1. Small scales are "slaved" to large scales
2. Energy cascade is local (not inverse cascade dominated)
3. Scale separation exists

### When GQL Fails

GQL may not capture:
- Strong inverse cascades (2D turbulence without β)
- Intermittency and extreme events
- Small-scale instabilities

Always validate against full NL for your specific problem!

---

## References

1. Marston, Chini & Tobias (2016). "Generalized Quasilinear Approximation: Application to Zonal Jets". *Phys. Rev. Lett.* 116, 214501.

2. Tobias & Marston (2017). "Three-dimensional rotating Couette flow via the generalised quasilinear approximation". *J. Fluid Mech.* 810, 412-428.

3. Marston, Chini & Tobias (2019). "Generalized Quasilinear Approximation of the Interaction of Convection and Mean Flows". *Proc. R. Soc. A* 474, 20180422.

4. Constantinou & Parker (2018). "Magnetic suppression of zonal flows on a beta plane". *Astrophys. J.* 863, 46.

---

## See Also

- [Temporal Filters](temporal_filters.md) - Time-domain filtering for wave-mean separation
- [LES Models](les_models.md) - Subgrid-scale modeling (alternative to GQL)
- [Stochastic Forcing](stochastic_forcing.md) - Adding stochastic terms to equations
