# Spectral Bases

Spectral bases define how functions are represented in each coordinate direction.

## Basis Selection Guide

| Boundary Type | Recommended Basis | Grid Points |
|---------------|-------------------|-------------|
| Periodic | RealFourier | Uniform |
| Non-periodic (walls) | ChebyshevT | Gauss-Lobatto (endpoints included) |
| Non-periodic (alternative) | ChebyshevU / Legendre | Gauss (endpoints excluded) |

## Fourier Bases

### RealFourier

For periodic directions with real-valued data.

```julia
using Tarang

coords = CartesianCoordinates("x")
x = coords["x"]

# Basic usage (dealias defaults to 3/2)
basis = RealFourier(x; size=128, bounds=(0.0, 2π))

# Explicit dealiasing factor
basis = RealFourier(x; size=128, bounds=(0.0, 2π), dealias=3/2)
```

**Parameters:**
- `size`: Number of modes (should be even)
- `bounds`: Domain boundaries (min, max)
- `dealias`: Padding factor, `3/2` by default (see the Dealiasing section below)

**Properties** live under `basis.meta`; the basis object itself has no `size`/`bounds`
fields (`basis.size` throws a `FieldError`):

```julia
basis.meta.size           # 128         number of modes
basis.meta.bounds         # (0.0, 2π)   (min, max)
basis.meta.dealias        # 1.5         padding factor
basis.meta.element_label  # "x"         coordinate name
basis.meta.dtype          # Float64

wavenumbers(basis)        # physical wavenumbers k = 2πn/L
```

### ComplexFourier

For complex-valued data or full complex FFT.

```julia
basis = ComplexFourier(x; size=128, bounds=(0.0, 2π))
```

## Chebyshev Bases

### ChebyshevT

Chebyshev polynomials of the first kind. Best for bounded domains with walls.

```julia
coords = CartesianCoordinates("z")
z = coords["z"]

basis = ChebyshevT(z; size=64, bounds=(0.0, 1.0))
```

**Grid Points:** Chebyshev-Gauss-Lobatto points, clustered near boundaries and including
both endpoints (so wall boundary conditions sit exactly on a grid point).

```julia
# Native grid in [-1, 1], ascending:
# ξ_k = -cos(πk / (N-1)),  k = 0, 1, ..., N-1

# Mapped to [a, b]:
# z_k = (a+b)/2 + (b-a)/2 * ξ_k
```

### ChebyshevU

Chebyshev polynomials of the second kind. Grid excludes endpoints
(`ξ_k = -cos(π(k+1/2)/N)`).

```julia
basis = ChebyshevU(z; size=64, bounds=(0.0, 1.0))
```

## Legendre Basis

Alternative to Chebyshev for bounded domains. Grid points are the Gauss-Legendre nodes
(roots of `P_N`), which also exclude the endpoints.

```julia
basis = Legendre(z; size=64, bounds=(0.0, 1.0))
```

**When to use:**
- Some problems with specific weight functions
- Alternative numerical properties
- Generally similar to ChebyshevT

Note that a Legendre derivative inside an explicit (RHS) term is not compiled by the lazy
RHS — see [Solvers](solvers.md). Keep non-Fourier derivatives on the implicit side.

## Resolution Guidelines

### Fourier Resolution

```julia
# Rule of thumb: resolve the smallest scales
L = 2π                      # Domain length
ν, ε = 1e-3, 0.1            # Viscosity, dissipation rate
η = (ν^3 / ε)^(1/4)         # Kolmogorov scale (turbulence)
N = ceil(Int, L / η)        # Required modes (minimum)
```

| Flow Type | Typical Resolution |
|-----------|-------------------|
| Laminar | 64-128 |
| Transitional | 128-256 |
| Turbulent | 256-1024+ |

### Chebyshev Resolution

| Feature | Typical Modes |
|---------|---------------|
| Smooth profiles | 32-64 |
| Boundary layers | 64-128 |
| Steep gradients | 128-256 |

## Dealiasing

Prevents aliasing errors in nonlinear terms. `dealias` is a **padding factor**, not a
cutoff fraction: values `> 1` dealias, values `<= 1` switch dealiasing **off**.

### 3/2 Rule

The default for `RealFourier` and `ComplexFourier`:

```julia
basis = RealFourier(x; size=128, dealias=3/2)
```

Nonlinear products are evaluated on a grid padded to `ceil(3/2 * 128) = 192` points and
truncated back to the 128-mode representation. Under MPI the equivalent 2/3-rule
truncation is applied instead: modes with `|k| > N / (2 * dealias)` are zeroed.

### 2/1 Rule

For highly nonlinear problems (padded grid of 256 points at `size=128`):

```julia
basis = RealFourier(x; size=128, dealias=2.0)
```

### No Dealiasing

For linear problems only:

```julia
basis = RealFourier(x; size=128, dealias=1.0)
```

### Cutoffs at `size = 32`

Retained modes per axis, from `min(floor(N / (2*dealias)), (N-1) ÷ 3)`:

| `dealias` | Cutoff | Effect |
|-----------|--------|--------|
| `3/2` | 10 | 3/2 rule (default) |
| `2.0` | 8 | Stronger truncation |
| `1.0` | – | Dealiasing off |
| `2/3` | – | Dealiasing **off** — `2/3` is a factor below 1, not the 2/3 rule |

Chebyshev/Legendre axes ignore `dealias` (no cutoff is applied on a non-Fourier axis);
passing it there is harmless but does nothing.

## Multi-Dimensional Domains

Combine bases for each direction:

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

# Periodic x, bounded z
x_basis = RealFourier(coords["x"]; size=256, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))

# Create domain
domain = Domain(dist, (x_basis, z_basis))
```

Under MPI, two rules apply (see [Parallelism](parallelism.md)):

- A **Chebyshev/Legendre axis must come first**, because the trailing axes are the ones
  that get decomposed and a non-Fourier axis cannot be split across ranks. Ordering it
  last raises a clear error.
- A 2-D domain takes a **1-D** process mesh (`mesh=(nprocs,)`); a 3-D domain takes a 2-D
  mesh. Omitting `mesh=` lets the distributor pick a valid one. A mesh whose size does not
  match the process count is rejected (`mesh=(2,2)` with one process throws
  `ArgumentError: Mesh size 4 does not match number of processes 1`).

## Grid and Spectral Space

### Transforms

```julia
# A new field starts in grid layout
field = ScalarField(dist, "T", (x_basis, z_basis), Float64)

# Transform to grid space (no-op if already there)
ensure_layout!(field, :g)

# Access grid data
data = get_grid_data(field)

# Transform to spectral space
ensure_layout!(field, :c)
coeffs = get_coeff_data(field)
```

For a `RealFourier(size=32) × ChebyshevT(size=16)` domain the grid data is `32 × 16` and
the coefficient data is `17 × 16` (the real FFT halves the first axis). The grid is *not*
padded by `dealias` — padding happens inside the nonlinear-product evaluation only.

### Grid Points

`local_grids` returns each rank's **local** slice of the per-axis grids, so the same code
is correct in serial and under MPI:

```julia
x_grid, z_grid = local_grids(dist, x_basis, z_basis)

ensure_layout!(field, :g)
get_grid_data(field) .= sin.(2π .* x_grid ./ 4.0) .* z_grid'
ensure_layout!(field, :c)
```

`create_meshgrid` builds full-rank meshgrid arrays (one array per coordinate, each of the
**global** grid shape), which is convenient in serial post-processing:

```julia
mesh = create_meshgrid(domain; on_device=false)   # Dict("x" => …, "z" => …)
X, Z = mesh["x"], mesh["z"]                       # each 256 × 64 here
```

Because these are global arrays, use `local_grids` (not `create_meshgrid`) whenever the
result is broadcast into field data under MPI.

## Performance Tips

1. **Power-of-2 sizes**: Use N = 2^m for fastest FFTs
2. **Dealiasing cost**: the padded nonlinear transform costs ~(3/2)^d more work in d dealiased directions
3. **Chebyshev efficiency**: Sparse differentiation matrices
4. **Memory**: RealFourier stores N/2+1 complex coefficients where ComplexFourier stores N

## See Also

- [Coordinates](coordinates.md): Coordinate systems
- [Domains](domains.md): Creating full domains
- [API: Bases](../api/bases.md): Complete reference
