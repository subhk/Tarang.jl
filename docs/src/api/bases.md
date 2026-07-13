# Bases API

Spectral bases define how functions are represented in each coordinate direction. Tarang.jl provides Fourier, Chebyshev, and Legendre bases for different boundary conditions.

## Docstrings

```@docs
RealFourier
ComplexFourier
ChebyshevT
ChebyshevU
Legendre
```

## Overview

Each coordinate dimension requires a spectral basis that determines:
- **Representation**: How functions are expanded (Fourier series, polynomial series, etc.)
- **Boundary conditions**: Periodic, non-periodic, or special constraints
- **Grid points**: Where functions are evaluated (collocation points)
- **Derivatives**: How differentiation is computed in spectral space

## Basis Selection Guide

| Basis Type | Use Case | Boundary Conditions | Grid Points |
|------------|----------|---------------------|-------------|
| `RealFourier` | Periodic, real-valued data | Periodic | Uniform |
| `ComplexFourier` | Periodic, complex data | Periodic | Uniform |
| `ChebyshevT` | Bounded, smooth | Dirichlet/Neumann | Gauss-Lobatto (includes endpoints) |
| `ChebyshevU` | Bounded, special BCs | Custom | Gauss (interior only) |
| `Legendre` | Bounded, alternative | Dirichlet/Neumann | Gauss-Legendre (interior only) |

All examples on this page assume a coordinate system and a distributor:

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
x, z   = coords["x"], coords["z"]
```

---

## Fourier Bases

### RealFourier

Real-valued Fourier basis for periodic coordinates with real data.

**Constructor**:
```julia
RealFourier(
    coord::Coordinate;
    size::Int=32,
    bounds::Tuple{Float64,Float64}=(0.0, 2π),
    dealias::Float64=3/2,
    dtype=Float64
)
```

**Arguments**:
- `coord`: Coordinate object for this direction
- `size`: Number of grid points / modes along the axis
- `bounds`: Domain boundaries (min, max), as `Float64`. Note `2π` is a `Float64` but a bare
  `π` is an `Irrational`, so `bounds=(0.0, π)` raises a `TypeError` — write `(0.0, Float64(π))`.
- `dealias`: Dealiasing **padding factor** (default `3/2` = the 3/2 rule; `1.0` disables dealiasing). See [Dealiasing](#Dealiasing) below — this is *not* a cutoff fraction.

**Examples**:

```julia
# Basic real Fourier basis (dealiased by default)
basis = RealFourier(x; size=16, bounds=(0.0, 2π))

# Dealiasing explicitly disabled
basis = RealFourier(x; size=16, bounds=(0.0, 2π), dealias=1.0)

# Custom domain length
basis = RealFourier(x; size=32, bounds=(0.0, 4.0))
```

Even `size` is recommended (fastest FFTs, symmetric spectrum), but odd sizes are supported
and round-trip exactly.

**Properties**: basis metadata lives in `basis.meta`; the basis object itself has no `size`,
`bounds` or `dealias` field.

```julia
basis = RealFourier(x; size=32, bounds=(0.0, 4.0))

basis.meta.element_label   # "x"          — the coordinate this basis belongs to
basis.meta.size            # 32           — number of modes / grid points
basis.meta.bounds          # (0.0, 4.0)   — (min, max)
basis.meta.dealias         # 1.5          — dealiasing padding factor
basis.meta.dtype           # Float64
```

**Grid Points**:
```julia
# This rank's grid points for the axis (uniform spacing, excludes the right endpoint).
# The third argument is the SCALE factor, not an axis index: 1 = the basis's own
# resolution, 1.5 would return the 3/2-padded grid. (The axis is looked up from the basis.)
xg = local_grid(basis, dist, 1)
# 32-element Vector{Float64}: [0.0, 0.125, 0.25, ..., 3.875]  for bounds=(0.0, 4.0)

# For several bases at once — this is the idiom to use, it is correct under MPI too
z_basis = ChebyshevT(z; size=12, bounds=(0.0, 1.0))
xg, zg  = local_grids(dist, basis, z_basis)
```

`local_grid`/`local_grids` return each rank's *local* slice, so the same code works serially
and distributed. Grid spacing for a whole domain is `grid_spacing(domain)`.

**Wavenumbers**:
```julia
k = wavenumbers(basis)
# RealFourier native storage is cos/sin interleaved, so k repeats:
# [0, 1, 1, 2, 2, ..., N/2] * (2π/L)   (length N)
```

Coefficient arrays produced by the transforms use the real-to-complex (rfft) half spectrum
of length `N÷2+1`; the matching wavenumbers are `Tarang.wavenumbers_rfft(basis)`
(`[0, 1, 2, ..., N/2] * 2π/L`).

**Memory Savings**: RealFourier uses real-to-complex FFT, saving ~50% memory compared to ComplexFourier.

---

### ComplexFourier

Complex-valued Fourier basis for periodic coordinates.

**Constructor**:
```julia
ComplexFourier(
    coord::Coordinate;
    size::Int=32,
    bounds::Tuple{Float64,Float64}=(0.0, 2π),
    dealias::Float64=3/2,
    dtype=ComplexF64
)
```

**Arguments**: Same as RealFourier, except `dtype` defaults to `ComplexF64`.

**Examples**:

```julia
# Complex Fourier basis
basis = ComplexFourier(x; size=16, bounds=(0.0, 2π))
```

**Use Cases**:
- Complex-valued fields (e.g., quantum mechanics)
- Incompatible with RealFourier optimizations
- More general but uses more memory

**Wavenumbers**:
```julia
k = wavenumbers(basis)
# FFT ordering: [0, 1, ..., N/2-1, -N/2, ..., -1] * (2π/L)   (length N)
```

---

## Chebyshev Bases

### ChebyshevT

Chebyshev polynomials of the first kind for bounded, non-periodic coordinates.

**Constructor**:
```julia
ChebyshevT(
    coord::Coordinate;
    size::Int=32,
    bounds::Tuple{Float64,Float64}=(-1.0, 1.0),
    dealias::Float64=1.0,
    dtype=Float64
)
```

**Arguments**:
- `coord`: Coordinate object
- `size`: Number of modes (polynomial degree + 1)
- `bounds`: Domain boundaries (min, max)
- `dealias`: accepted, but **ignored on non-Fourier axes** (only Fourier axes are truncated)

**Examples**:

```julia
# Chebyshev basis for vertical direction
basis = ChebyshevT(z; size=12, bounds=(0.0, 1.0))

# Canonical domain
basis = ChebyshevT(z; size=32, bounds=(-1.0, 1.0))
```

**Properties**: as for the Fourier bases, everything lives in `basis.meta`:

```julia
basis.meta.element_label   # "z"
basis.meta.size            # number of modes
basis.meta.bounds          # (min, max)
basis.meta.dtype           # Float64
```

**Grid Points (Gauss-Lobatto)**:
```julia
basis = ChebyshevT(z; size=12, bounds=(0.0, 1.0))
zg = local_grid(basis, dist, 1)
# Points clustered near boundaries, ascending, endpoints INCLUDED:
# zg[1] == 0.0 and zg[end] == 1.0

# In canonical [-1, 1]:
# ξ_k = -cos(π*k / (N-1)) for k = 0, 1, ..., N-1
#
# Mapped to physical domain [a, b]:
# z_k = (a + b)/2 + (b - a)/2 * ξ_k
```

**Polynomial Expansion**:

Functions are expanded as:
```math
f(x) = \sum_{n=0}^{N-1} a_n T_n(x)
```

where T_n(x) is the nth Chebyshev polynomial.

**Boundary Conditions**:
- Values at boundaries are grid points (includes endpoints)
- Well-suited for Dirichlet boundary conditions
- Can handle Neumann conditions with tau method

---

### ChebyshevU

Chebyshev polynomials of the second kind.

**Constructor**:
```julia
ChebyshevU(
    coord::Coordinate;
    size::Int=32,
    bounds::Tuple{Float64,Float64}=(-1.0, 1.0)
)
```

**Use Cases**:
- Specific boundary condition types
- Alternative to ChebyshevT for some problems
- Less commonly used than ChebyshevT

**Grid Points (Gauss)**:
```julia
basis = ChebyshevU(z; size=12, bounds=(0.0, 1.0))
zg = local_grid(basis, dist, 1)
# Chebyshev-Gauss points, endpoints EXCLUDED (zg[1] ≈ 0.0043, zg[end] ≈ 0.9957)
# ξ_k = -cos(π*(k + 1/2) / N) for k = 0, 1, ..., N-1
```

---

## Legendre Basis

### Legendre

Legendre polynomials for bounded coordinates.

**Constructor**:
```julia
Legendre(
    coord::Coordinate;
    size::Int=32,
    bounds::Tuple{Float64,Float64}=(-1.0, 1.0)
)
```

**Arguments**: Same as ChebyshevT

**Examples**:

```julia
# Legendre basis
basis = Legendre(z; size=12, bounds=(0.0, 1.0))
```

**Grid Points (Gauss-Legendre)**:
```julia
zg = local_grid(basis, dist, 1)
# Roots of P_N: interior nodes, endpoints EXCLUDED
# (zg[1] ≈ 0.0092, zg[end] ≈ 0.9908 for bounds=(0.0, 1.0), size=12)
```

**Comparison with Chebyshev**:
- Similar accuracy and convergence properties
- Different grid point distribution: ChebyshevT includes the endpoints, Legendre does not
- Choice often based on problem-specific considerations
- Legendre may be preferred for some weight functions
- Legendre is **serial only**: MPI runs raise an error (`MPI parallelization is not yet
  supported for Legendre bases`)

---

## Compound Bases

For multi-dimensional problems, combine bases for each direction.

**Example: 2D Rayleigh-Bénard** (using the `coords` / `dist` from the top of this page):
```julia
# Periodic horizontal direction
x_basis = RealFourier(coords["x"]; size=32, bounds=(0.0, 4.0))

# Bounded vertical direction with walls
z_basis = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))

# Create domain with both bases
domain = Domain(dist, (x_basis, z_basis))
```

**Example: 3D Channel Flow**:
```julia
coords3 = CartesianCoordinates("x", "y", "z")
dist3   = Distributor(coords3; dtype=Float64, device=CPU())

# Streamwise (periodic)
xb3 = RealFourier(coords3["x"]; size=16, bounds=(0.0, 2π))

# Spanwise (periodic)
yb3 = RealFourier(coords3["y"]; size=16, bounds=(0.0, Float64(π)))

# Wall-normal (bounded)
zb3 = ChebyshevT(coords3["z"]; size=12, bounds=(0.0, 1.0))

domain3 = Domain(dist3, (xb3, yb3, zb3))
```

!!! note "Basis order under MPI"
    Serially the Chebyshev axis may sit anywhere. Under MPI it must come **first**: the
    trailing axes are the decomposed ones, and a Chebyshev axis cannot be split (its DCT
    needs the whole axis on each rank). Putting it last raises an error telling you to
    reorder. So the distributed form of the example above is
    `CartesianCoordinates("z", "x")` with `Domain(dist, (z_basis, x_basis))`.

---

## Grid and Spectral Space

### Transforms

A field carries a layout flag; `ensure_layout!` moves it between grid (`:g`) and coefficient
(`:c`) space, transforming only when needed.

```julia
domain = Domain(dist, (x_basis, z_basis))
T = ScalarField(domain, "T")

# Move to grid space and edit in physical space
ensure_layout!(T, :g)
grid_data = get_grid_data(T)          # (32, 16) Array{Float64}
grid_data[10, 8] = 1.0

# Move back to coefficient space
ensure_layout!(T, :c)
coeff_data = get_coeff_data(T)        # (17, 16) Array{ComplexF64} — rfft half spectrum
```

The same field can also be built from the distributor and a basis tuple directly:
`ScalarField(dist, "T", (x_basis, z_basis))`.

!!! warning "One basis configuration per Distributor"
    The transform plan lives on the **Distributor** and is rebuilt for the domain most
    recently constructed on it. If a script needs two domains with *different* bases, give
    each its own `Distributor` — otherwise fields belonging to the earlier domain are
    transformed with the later domain's plan.

### Grid Point Access

```julia
# Per-axis local grid vectors
xg, zg = local_grids(dist, x_basis, z_basis)

# Initialize a field by broadcasting (xg is a column, zg' a row)
ensure_layout!(T, :g)
get_grid_data(T) .= sin.(2π/4 .* xg) .* sin.(π .* zg')   # L_x = 4, so 2π/4 is the fundamental
ensure_layout!(T, :c)
```

Broadcasting against `local_grids` is the idiom to use: it is correct serially and under MPI,
where each rank owns only a slab of the grid. (`set!(field, ::Function)` builds the *global*
meshgrid and therefore only works serially.)

---

## Differentiation

Spectral differentiation is computed by applying operators in spectral space.

### Fourier Differentiation

For Fourier bases, differentiation is multiplication by ik:

```math
\frac{d}{dx} \left(\sum a_k e^{ikx}\right) = \sum (ik) a_k e^{ikx}
```

In equations you write the derivative as `∂x(u)`. Standalone, the same derivative is the
`Differentiate` operator (`T` here is the field from the section above, holding
`sin(2πx/4)·sin(πz)`):

```julia
dTdx = evaluate(Differentiate(T, coords["x"], 1), :g)   # returns a ScalarField

exact = (2π/4) .* cos.(2π/4 .* xg) .* sin.(π .* zg')
maximum(abs.(get_grid_data(dTdx) .- exact))             # 6.3e-15 — spectrally accurate
```

### Chebyshev Differentiation

For Chebyshev bases, differentiation uses recurrence relations:

```math
\frac{dT_n}{dx} = n U_{n-1}(x)
```

Implemented as sparse matrix multiplication in spectral space.

**Example** — the same operator, along the Chebyshev axis:
```julia
dTdz = evaluate(Differentiate(T, coords["z"], 1), :g)

exact = π .* sin.(2π/4 .* xg) .* cos.(π .* zg')
maximum(abs.(get_grid_data(dTdz) .- exact))             # 2.5e-13
```

In a problem the Chebyshev derivative normally enters through `lap`/`div`/`grad` on the
implicit side, written as `"∂t(T) - kappa*lap(T) = 0"`, with the wall conditions supplied by
`add_bc!`. Because each boundary condition consumes an equation slot, such a problem also
needs tau variables — see [Tau Method](../pages/tau_method.md).

!!! warning "Keep non-Fourier derivatives off the explicit side under MPI"
    A Chebyshev/Legendre derivative in the explicit (RHS) part of an equation cannot be
    evaluated distributed — each rank owns only part of that axis. Serially it is fine; under
    MPI, move the term to the implicit (L) side.

---

## Dealiasing

Dealiasing prevents aliasing errors in nonlinear terms.

### `dealias` is a padding factor

`dealias` is the padding factor, **not** a cutoff fraction: `dealias=3/2` gives the 3/2 rule
(equivalently, 2/3-rule truncation), and any value `≤ 1` switches dealiasing **off**. The
retained band is `|k| ≤ min(floor(N / (2*dealias)), (N-1)÷3)`.

```julia
# The 3/2 rule — this is already the RealFourier/ComplexFourier default
basis = RealFourier(x; size=32, bounds=(0.0, 2π), dealias=3/2)
```

Cutoffs actually applied for `RealFourier(size=32)`:

| `dealias=` | modes retained | effect |
|---|---|---|
| `3/2` (default) | wavenumbers up to 10 | the 3/2 / 2-3 rule |
| `2.0` | wavenumbers up to 8 | stronger truncation |
| `1.0` | all | dealiasing off |
| `2/3` | all | dealiasing off (it is `≤ 1`, so **not** what you want) |

The factor is ignored on non-Fourier axes: passing `dealias=3/2` to a Chebyshev or Legendre
basis is harmless but truncates nothing.

### When to Dealias

- **Always**: For products of fields (e.g., `u*∂x(u)`)
- **Optional**: For linear problems
- **Recommend**: 3/2 rule for most problems (the default)
- **Higher**: `dealias=2.0` for very nonlinear problems

---

## Resolution Guidelines

### Fourier Bases

Rule of thumb: resolve the smallest scales in the flow. With `η = (ν³/ε)^(1/4)` the
Kolmogorov scale and `L` the domain length, you need roughly `N ≈ L/η` modes per direction.

- Low Re flow: 64-128 modes
- Moderate Re: 256-512 modes
- High Re/turbulent: 1024+ modes

### Chebyshev/Legendre Bases

Exponential convergence for smooth functions:

- Smooth flows: 32-64 modes
- Boundary layers: 64-128 modes
- Steep gradients: 128-256 modes

```julia
# Example: Rayleigh-Bénard at Ra=10^6
z_basis_hires = ChebyshevT(z; size=64, bounds=(0.0, 1.0))
```

---

## Advanced Topics

### Grid sizes and padding

A basis has one size, fixed at construction (`basis.meta.size`); there is no separate grid
size to set on it. The padding used for nonlinear products comes from the `dealias` factor.
A *field* can be evaluated on a padded grid with `set_scales!`:

```julia
Tpad = ScalarField(domain, "Tpad")
ensure_layout!(Tpad, :g)
get_grid_data(Tpad) .= sin.(2π/4 .* xg) .* sin.(π .* zg')

size(get_grid_data(Tpad))     # (32, 16)
set_scales!(Tpad, 1.5)
size(get_grid_data(Tpad))     # (48, 24) — 3/2-padded grid

# The values are the same function, spectrally resampled onto the finer nodes:
xg15, zg15 = local_grids(dist, x_basis, z_basis; scales=1.5)
maximum(abs.(get_grid_data(Tpad) .- sin.(2π/4 .* xg15) .* sin.(π .* zg15')))   # 3.9e-15
```

!!! warning "On a mixed Fourier-Chebyshev field, `set_scales!` does not reverse"
    Padding works, but scaling such a field back down — `set_scales!(Tpad, 1.0)` — throws
    `TypeError: in setfield!, expected Matrix{Float64}, got a value of type Matrix{ComplexF64}`.
    The forward transform of a padded grid stores a padded half-spectrum (here 25×16 rather
    than the base 17×16), and the backward transform cannot map that back onto the base real
    grid. Give the padded output its own field instead of rescaling one you still need.
    Pure-Fourier fields rescale in both directions.

### Domain shapes

```julia
global_shape(domain, :g)   # (32, 16) — grid shape
global_shape(domain, :c)   # (17, 16) — coefficient shape (rfft halves the first axis)
local_shape(domain, :g)    # this rank's slab
grid_spacing(domain)       # per-axis spacing
```

### Spectral Accuracy

Inspect the coefficient magnitudes to check that the resolution is sufficient — they should
decay to the noise floor well before the last mode. (Plot `decay` on a log y-axis with your
plotting package of choice.)

```julia
ensure_layout!(T, :c)
decay = vec(maximum(abs.(get_coeff_data(T)); dims=2))   # per x-mode, 17 entries
# T = sin(2πx/4)·sin(πz) -> decay[1:5] = [0.0, 7.99, 0.0, 0.0, 0.0]
# and decay[end] = 8.8e-17: all the energy sits in one mode, resolution is ample
```

---

## Performance Considerations

1. **Power-of-2 sizes**: Use N = 2^m for fastest FFTs
2. **Memory alignment**: FFTW performs best with aligned arrays
3. **FFTW planning**: Use FFTW_MEASURE or FFTW_PATIENT for optimal plans
4. **Grid size**: Larger grids = more memory, slower transforms
5. **Dealiasing**: Increases computational cost by ~50%

---

## See Also

- [Coordinates](coordinates.md): Coordinate systems for bases
- [Domains](domains.md): Combining bases into domains
- [Fields](fields.md): Creating fields on bases
- [Operators](operators.md): Differential operators
