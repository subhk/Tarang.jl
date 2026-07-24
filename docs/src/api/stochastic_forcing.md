# Stochastic Forcing API

## Types

### Abstract Types

```julia
abstract type Forcing end
abstract type StochasticForcingType <: Forcing end
abstract type DeterministicForcingType <: Forcing end
```

### StochasticForcing

```@docs
StochasticForcing
```

Stochastic forcing in Fourier space with white-noise temporal correlation.

**Type signature:**
```julia
mutable struct StochasticForcing{
    T<:AbstractFloat, N,
    A<:AbstractArray{T, N},
    CA<:AbstractArray{Complex{T}, N}
} <: StochasticForcingType
```

`A` and `CA` are the concrete real and complex array types on the chosen architecture, so
the same struct holds `Array`s on the CPU and `CuArray`s on the GPU.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `forcing_spectrum` | `A` | √Q̂(k) - amplitude spectrum |
| `energy_injection_rate` | `T` | Target ε |
| `injection_metric` | `Symbol` | `:direct` or `:vorticity_kinetic` |
| `k_forcing` | `T` | Central forcing wavenumber |
| `dk_forcing` | `T` | Forcing bandwidth |
| `dt` | `T` | Current timestep |
| `domain_size` | `NTuple{N, T}` | Domain extent (Lx, Ly, ...) |
| `field_size` | `NTuple{N, Int}` | Grid size (Nx, Ny, ...) |
| `wavenumbers` | `NTuple{N, Vector{T}}` | Wavenumber arrays (kx, ky, ...), always on the CPU |
| `cached_forcing` | `CA` | Cached F̂ (constant within timestep) |
| `prevsol` | `Union{Nothing, CA}` | Previous solution for Stratonovich work |
| `rng` | `AbstractRNG` | Random number generator |
| `random_phases` | `A` | Pre-allocated random phase buffer |
| `diagnostic_weights` | `Union{Nothing,A}` | Cached backend diagnostic weights |
| `diagnostic_global_shape` | `NTuple{N,Int}` | Shape associated with cached weights |
| `diagnostic_local_ranges` | `NTuple{N,UnitRange{Int}}` | Local ranges associated with cached weights |
| `diagnostic_metric` | `Symbol` | Metric associated with cached weights |
| `last_update_time` | `T` | Time of last forcing update |
| `spectrum_type` | `Symbol` | Type of forcing spectrum |
| `enforce_hermitian` | `Bool` | Enforce F̂(-k) = F̂(k)\* for real-valued fields |
| `architecture` | `AbstractArchitecture` | `CPU()` or `GPU()` backend |

**Property aliases** (via `getproperty`, not stored fields):

| Property | Returns |
|----------|---------|
| `forcing.forcing_rate` | `forcing.energy_injection_rate` |
| `forcing.spectrum` | `forcing.forcing_spectrum` |
| `forcing.is_stochastic` | `true` |
| `forcing.is_gpu` | whether `architecture` is a GPU |

**Constructor:**

```julia
StochasticForcing(;
    field_size::NTuple{N, Int},                              # Required
    domain_size::Union{Nothing, NTuple{N, Real}} = nothing,  # nothing => 2π in each direction
    energy_injection_rate::Real = 1.0,
    forcing_rate::Union{Nothing, Real} = nothing,            # alias for energy_injection_rate
    injection_metric::Symbol = :direct,
    k_forcing::Real = 4.0,
    dk_forcing::Real = 1.0,
    dt::Real = 0.01,
    spectrum_type::Symbol = :ring,
    rng::AbstractRNG = Random.MersenneTwister(),
    dtype::Type{T} = Float64,
    enforce_hermitian::Bool = true,
    architecture::AbstractArchitecture = CPU()
) where {T<:AbstractFloat, N}
```

`forcing_rate` and `energy_injection_rate` set the same target ε. `forcing_rate` wins when
provided. If a non-default `energy_injection_rate` is also supplied and the values disagree,
a warning is emitted.

`injection_metric = :direct` makes ε the injection rate of the forced variable's direct
quadratic invariant. Use `:vorticity_kinetic` when forcing 2-D vorticity and ε should be
kinetic-energy injection; it weights each nonzero Fourier mode by `1/|k|²`.

The default `rng` is a **fresh `MersenneTwister` per instance** (not
`Random.GLOBAL_RNG`), which avoids accidental RNG sharing between forcing instances.
Concurrent mutation of the same forcing object is not supported. Under MPI with more than
one rank the constructor broadcasts a seed from rank 0 on `MPI.COMM_WORLD`, so every rank
draws the same phases and the forcing stays coherent across the decomposition.

Set `enforce_hermitian = false` when the forced field is genuinely complex; with the default
`true`, F̂(-k) = F̂(k)\* is imposed so that the inverse transform of the forcing is real.
Self-conjugate modes are projected as `sqrt(2) * real(z)`; the factor preserves the
variance that would otherwise be halved by discarding the imaginary part.

`dk_forcing` must be positive for `:ring`, `:band` and `:kolmogorov` (an `ArgumentError` is thrown
otherwise); `:lowk` ignores it. `:kolmogorov` also requires `k_forcing > 0`.

**Spectrum types:**

| Symbol | Description | Unnormalized amplitude √Q̂(k) |
|--------|-------------|------------------------------|
| `:ring` | Gaussian ring around k_f | exp(-(\|k\| - k_f)² / 4δ_f²) |
| `:band` | Sharp band \|k\| ∈ (k_f - δ_f, k_f + δ_f) | 1 if \|\|k\| - k_f\| < δ_f, else 0 |
| `:lowk` | Low wavenumber forcing | 1 if \|k\| < k_f, else 0 |
| `:kolmogorov` | Large scales with a smooth cutoff at k_f + δ_f | √(\|k\|/k_f) · exp(-(\|k\| - k_f)² / 4δ_f²) if \|k\| < k_f + δ_f, else 0 |

In every case the k = 0 mode is zero, and the full stored spectrum is rescaled so
`sum(Q̂ .* w) / (2M^2) == ε`, where `M = prod(field_size)`, `w = 1` for `:direct`,
and `w = 1/|k|²` for `:vorticity_kinetic`. A positive ε whose selected spectrum contains
no representable nonzero mode raises `ArgumentError`. `:isotropic` aliases `:ring`, and
`:bandlimited` aliases `:band`.

### SeparableStochasticForcing

```@docs
SeparableStochasticForcing
```

Use `SeparableStochasticForcing` for a Fourier--Chebyshev product domain. The
random spectrum is defined only over the Fourier axes and is multiplied by a
fixed Chebyshev profile:

```math
\hat F(k_1,\ldots,k_d,n,t) = \hat\xi(k_1,\ldots,k_d,t)\,p_n.
```

```julia
forcing = SeparableStochasticForcing(
    fourier_size=(Nx,),
    chebyshev_basis=zbasis,
    chebyshev_profile=z -> z * (1 - z),
    domain_size=(Lx,),
    energy_injection_rate=0.1,
    k_forcing=4.0,
    dk_forcing=1.0,
    dt=dt,
    architecture=GPU(),
)
add_stochastic_forcing!(problem, :b, forcing)
```

`chebyshev_profile` can be either a function of the physical Chebyshev
coordinate or a vector of Chebyshev coefficients whose length equals
`chebyshev_basis.meta.size`. A function is sampled on the basis grid and
transformed once during construction. In either form the profile is normalized
to unit physical mean square with the basis quadrature weights. Non-finite,
zero-norm, and mis-sized profiles are rejected.

The mixed normalization currently supports only `injection_metric=:direct`.
Passing `:vorticity_kinetic` raises `ArgumentError`, because a kinetic-energy
metric in a bounded direction requires model-specific inverse operators rather
than the Fourier `1/|k|²` weight.

All forcing arrays and outer-product views are persistent. With a supplied RNG,
the Fourier phase draw is reproducible between CPU and GPU: one scalar seed is
drawn from the RNG and a counter-based kernel generates the phase array on the
field architecture, without host staging.
Only Fourier axes may be shortened to real-FFT half spectra when the forcing is
added to a field. The Chebyshev coefficient count must match exactly.

For Fourier--Fourier domains, continue to use `StochasticForcing`; for
Fourier--Chebyshev domains, use `SeparableStochasticForcing`.

### DeterministicForcing

```@docs
DeterministicForcing
```

Deterministic (non-random) forcing.

**Type signature:**
```julia
mutable struct DeterministicForcing{T<:AbstractFloat, N, A<:AbstractArray{T, N}} <: DeterministicForcingType
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `forcing_function` | `Function` | f(x, y, ..., t, params) → forcing array |
| `field_size` | `NTuple{N, Int}` | Grid size |
| `cached_forcing` | `A` | Cached forcing values on the chosen architecture |
| `parameters` | `Dict{Symbol, Any}` | Parameters for forcing function |
| `architecture` | `AbstractArchitecture` | CPU() or GPU() backend |

`forcing.is_gpu` is available as a property alias.

**Constructor:**

```julia
DeterministicForcing(
    forcing_function::Function,
    field_size::NTuple{N, Int};
    parameters::AbstractDict{Symbol} = Dict{Symbol, Any}(),
    dtype::Type{T} = Float64,
    architecture::AbstractArchitecture = CPU()
) where {T<:AbstractFloat, N}
```

`parameters` may be any `Dict` keyed by `Symbol` (e.g. `Dict(:A => 1.0, :k => 4.0)`); it is
copied into a `Dict{Symbol, Any}`.

The forcing function is called as `forcing_function(grid..., t, parameters)` and **must return an
array whose size is exactly `field_size`** — a scalar-valued function raises
`ArgumentError: Deterministic forcing function must return an array`, and a mis-shaped one raises
`ArgumentError: Deterministic forcing output size ... does not match field size ...`. Write it
with broadcasting over the grid arrays:

```julia
forcing = DeterministicForcing(
    (x, y, t, p) -> p[:A] .* sin.(p[:k] .* x) .* cos.(y) .* cos(p[:omega] * t),
    (16, 16);
    parameters = Dict(:A => 1.0, :k => 4.0, :omega => 1.0),
)

x = reshape(range(0, 2π, length=17)[1:16], 16, 1)   # column vector
y = reshape(range(0, 2π, length=17)[1:16], 1, 16)   # row vector
F = generate_forcing!(forcing, (x, y), 0.0)         # 16×16 Matrix{Float64}
```

---

## Forcing Generation

### generate_forcing!

```@docs
generate_forcing!
```

Generate forcing realization. Returns cached value for substeps > 1.

**Signatures:**

```julia
# Stochastic forcing
generate_forcing!(forcing::StochasticForcing, t::Real, substep::Int=1)
generate_forcing!(forcing::StochasticForcing, t::Real)
generate_forcing!(forcing::SeparableStochasticForcing, t::Real, substep::Int=1)

# Deterministic forcing
generate_forcing!(forcing::DeterministicForcing, grid, t::Real)
```

**Key behavior:**
- `substep == 1`: Generates new random forcing
- `substep > 1`: Returns cached forcing (same as substep 1)
- Time check: Won't regenerate if `t` is exactly `forcing.last_update_time`
- The k = 0 mode is always zeroed, and Hermitian symmetry is imposed when
  `enforce_hermitian = true`
- Requires `forcing.dt > 0`; otherwise an error is raised

**Returns:** The cached forcing array `forcing.cached_forcing`

### Automatic registration and timestepper compatibility

`add_stochastic_forcing!(problem, variable, forcing)` registers automatic RHS forcing.
`variable` names a scalar field or a flattened vector/tensor component (for example
`:u_x`). A container name such as `:u` is ambiguous and raises `ArgumentError`.

One forcing realization is reused across the stages of supported one-step RK/ETD methods;
`CNAB1` and `SBDF1` are supported first-order methods. Tarang rejects `CNAB2`, `MCNAB2`,
`CNLF2`, `SBDF2`, `SBDF3`, `SBDF4`, `DiagonalIMEX_SBDF2`, `ETD_CNAB2`, and
`ETD_SBDF2` before drawing forcing. Those schemes reuse or combine values across time
levels, which colors white noise or gives it the wrong variance.

### apply_forcing!

```@docs
apply_forcing!
```

Add forcing to a field in spectral space.

**Signature:**

```julia
apply_forcing!(
    rhs::AbstractArray{Complex{T}, N},
    forcing::StochasticForcing{T, N, A, CA},
    t::Real,
    substep::Int=1
) where {T, N, A, CA}
```

Equivalent to `rhs .+= generate_forcing!(forcing, t, substep)` when the shapes match exactly. When
they do not, the forcing is first sliced to the region `rhs` covers — the leading half-spectrum for
a real-FFT `rhs`, and this rank's local wavenumber block for a distributed `PencilArray` `rhs`. If
no such slice exists, an `ArgumentError` is thrown.

---

## Configuration

### set_dt!

```@docs
set_dt!
```

Update the timestep used for Stratonovich scaling.

```julia
set_dt!(forcing::StochasticForcing{T, N, A, CA}, dt::Real) where {T, N, A, CA}
```

Call this when dt changes (e.g., adaptive timestepping). If the value actually changes, the cache
is zeroed and the next `generate_forcing!` call regenerates with the new √dt scaling.

### reset_forcing!

```@docs
reset_forcing!
```

Reset forcing cache. Forces regeneration on next `generate_forcing!` call.

```julia
reset_forcing!(forcing::StochasticForcing{T, N, A, CA}) where {T, N, A, CA}
```

---

## Work Calculation

### store_prevsol!

```@docs
store_prevsol!
```

Store the current solution for Stratonovich work calculation.

```julia
store_prevsol!(forcing::StochasticForcing{T, N, A, CA}, sol::AbstractArray{Complex{T}, N})
```

Call at the **beginning** of each timestep, before advancing.

### work_stratonovich

```@docs
work_stratonovich
```

Compute work done using Stratonovich interpretation.

```julia
work_stratonovich(forcing::StochasticForcing{T, N, A, CA},
                  sol::AbstractArray{Complex{T}, N}) -> T
```

**Formula** (with `M = prod(field_size)`):
```math
W = \frac{dt}{M^2}\,\text{Re}\sum_k m_k w_k
    \frac{q^n_k + q^{n+1}_k}{2}\, \hat{F}_k^*
```

Uses midpoint evaluation and returns `zero(T)` unless `store_prevsol!` was called first.
Here `w_k` is the selected injection-metric weight and `m_k` is the real-FFT
multiplicity: 2 for an omitted conjugate partner, 1 for DC and an even-length Nyquist
endpoint. For a full complex spectrum, every `m_k = 1`.

### work_ito

```@docs
work_ito
```

Compute work done using Itô interpretation.

```julia
work_ito(forcing::StochasticForcing{T, N, A, CA},
         sol_prev::AbstractArray{Complex{T}, N}) -> T
```

**Formula:**
```math
W_{\text{Itô}} = \frac{dt}{M^2}\,\text{Re}\sum_k m_k w_k q^n_k\,
\hat{F}_k^* \;+\; \varepsilon\, dt
```

Uses the solution *before* the step (independent of the current forcing) plus the drift correction
ε·dt, which makes ⟨W_Itô⟩ = ⟨W_Stratonovich⟩ = ε·dt.

Under MPI both work functions reduce over this rank's local slab and then `Allreduce` the partial
sums, so every rank returns the same global value.

---

## Diagnostics

### mean_energy_injection_rate

```@docs
mean_energy_injection_rate
```

Return the target (mean) energy injection rate ε.

```julia
mean_energy_injection_rate(forcing::StochasticForcing) -> T
energy_injection_rate(forcing::StochasticForcing) -> T      # same value
```

### instantaneous_power

```@docs
instantaneous_power
```

Compute the instantaneous power input.

```julia
instantaneous_power(forcing::StochasticForcing{T, N, A, CA},
                    sol::AbstractArray{Complex{T}, N}) -> T
```

**Formula:**
```math
P = \frac{1}{M^2}\,\text{Re}\sum_k m_k w_k q_k\, \hat{F}_k^*
```

This is the instantaneous correlation between solution and forcing, and it fluctuates. Its mean
depends on *which* solution you pass: 0 for the solution before the forcing was applied, ε for the
midpoint (ψⁿ + ψⁿ⁺¹)/2, and 2ε for the solution after. For the Stratonovich-consistent
time-averaged power over a step, use `work_stratonovich(forcing, sol) / forcing.dt` instead.

### forcing_enstrophy_injection_rate

```@docs
forcing_enstrophy_injection_rate
```

Compute mean enstrophy injection rate (for 2D turbulence).

```julia
forcing_enstrophy_injection_rate(forcing::StochasticForcing{T, N, A, CA}) -> T
```

**Formula:**
```math
\eta = \frac{1}{2M^2} \sum_k \hat{Q}(k)
```

This assumes the forcing is applied directly to 2-D vorticity. With
`injection_metric=:direct`, η equals the configured ε. With
`:vorticity_kinetic`, ε is kinetic-energy injection and η is generally of order
`k_forcing^2 * ε`. For `N != 2` the function warns and returns `zero(T)`.

### get_forcing_spectrum

```@docs
get_forcing_spectrum
```

Return the forcing amplitude spectrum √Q̂(k), as the array type `A` of the chosen architecture.

```julia
get_forcing_spectrum(forcing::StochasticForcing) -> A
```

### get_cached_forcing

```@docs
get_cached_forcing
```

Return the current cached forcing F̂(k), as the complex array type `CA` of the chosen architecture.

```julia
get_cached_forcing(forcing::StochasticForcing) -> CA
get_forcing_real(forcing::StochasticForcing)   -> real part of F̂(k)
```

---

## Spectrum Building Utilities

These are exported so that you can inspect or build custom forcing spectra.

### build_wavenumbers

```@docs
build_wavenumbers
```

Build wavenumber arrays for each dimension, in standard FFT ordering
(`0, dk, ..., k_nyq, -(k_nyq - dk), ..., -dk` with `dk = 2π/L`).

```julia
build_wavenumbers(
    field_size::NTuple{N, Int},
    domain_size::NTuple{N, Real},
    dtype::Type{T}
) -> NTuple{N, Vector{T}}
```

For example, `build_wavenumbers((8,), (2π,), Float64)[1]` is
`[0.0, 1.0, 2.0, 3.0, 4.0, -3.0, -2.0, -1.0]`.

### compute_forcing_spectrum

```@docs
compute_forcing_spectrum
```

Compute the forcing amplitude spectrum √Q̂(k).

```julia
compute_forcing_spectrum(
    wavenumbers::NTuple{N, Vector{T}},
    k_f::Real,
    dk_f::Real,
    ε::Real,
    domain_size::NTuple{N, Real},
    spectrum_type::Symbol,
    dtype::Type{T};
    injection_metric::Symbol = :direct
) -> Array{T, N}
```

The full spectrum is normalized so that
`sum(abs2(spectrum[k]) * w[k]) / (2 * prod(field_size)^2) == ε`, with the
metric weights defined above. `compute_forcing_spectrum` uses the supplied `wavenumbers`;
the constructor obtains those wavenumbers from `domain_size` before calling it.

---

## Worked Example

A complete stochastically forced integration in Fourier space, with the work diagnostics.

```julia
using Tarang, Random

forcing = StochasticForcing(
    field_size = (32, 32),
    domain_size = (2π, 2π),
    energy_injection_rate = 0.1,
    k_forcing = 6.0,
    dk_forcing = 1.5,
    dt = 1e-3,
    spectrum_type = :ring,
    rng = MersenneTwister(7),
)

# Euler-Maruyama in spectral space: ψⁿ⁺¹ = ψⁿ + F̂ dt
function integrate!(forcing, nsteps)
    ψ = zeros(ComplexF64, forcing.field_size...)
    W = 0.0
    for n in 1:nsteps
        t = n * forcing.dt
        store_prevsol!(forcing, ψ)              # ψⁿ, BEFORE advancing
        F = generate_forcing!(forcing, t, 1)
        ψ .+= F .* forcing.dt
        W += work_stratonovich(forcing, ψ)      # work injected over this step
    end
    return ψ, W
end

ψ, W = integrate!(forcing, 1000)

@show mean_energy_injection_rate(forcing)       # 0.1, the target ε
@show W / (1000 * forcing.dt)                   # the realized injection rate
@show forcing_enstrophy_injection_rate(forcing)
```

With this seed the run prints

```
mean_energy_injection_rate(forcing) = 0.1
W / (1000 * forcing.dt) = 0.11453581797243557
forcing_enstrophy_injection_rate(forcing) = 3.937500006838822
```

The realized rate scatters around the target ε: a single 1000-step realization is a random
variable, not the ensemble mean, so expect it to land within a few tens of percent of ε and to
tighten as the run gets longer.

Inside a multi-stage timestepper, pass the substep index so the realization is held fixed across
the stages, and call `set_dt!` whenever the timestep changes:

```julia
rhs = zeros(ComplexF64, 32, 32)
for substep in 1:3
    apply_forcing!(rhs, forcing, 2.0, substep)   # substeps 2 and 3 reuse the cached F̂
end

set_dt!(forcing, 5e-4)                           # zeroes the cache; next draw rescales by 1/√dt
```

---

## Exports

```julia
export Forcing, StochasticForcingType, DeterministicForcingType
export StochasticForcing, DeterministicForcing
export generate_forcing!, apply_forcing!
export reset_forcing!, set_dt!
export store_prevsol!, work_stratonovich, work_ito
export mean_energy_injection_rate, energy_injection_rate, instantaneous_power
export forcing_enstrophy_injection_rate
export get_forcing_spectrum, get_cached_forcing, get_forcing_real
export build_wavenumbers, compute_forcing_spectrum
```

---

## Index

```@index
Pages = ["stochastic_forcing.md"]
```
