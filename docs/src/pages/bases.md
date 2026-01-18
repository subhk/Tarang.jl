# Spectral Bases

Spectral bases define how functions are represented in each coordinate direction.

## Basis Selection Guide

| Boundary Type | Recommended Basis | Grid Points |
|---------------|-------------------|-------------|
| Periodic | RealFourier | Uniform |
| Non-periodic (walls) | ChebyshevT | Gauss-Lobatto |
| Non-periodic (alternative) | Legendre | Gauss-Lobatto |

## Fourier Bases

### RealFourier

For periodic directions with real-valued data.

```julia
using Tarang

coords = CartesianCoordinates("x")
x = coords["x"]

# Basic usage
basis = RealFourier(x; size=128, bounds=(0.0, 2π))

# With dealiasing (recommended for nonlinear problems)
basis = RealFourier(x; size=128, bounds=(0.0, 2π), dealias=3/2)
```

**Parameters:**
- `size`: Number of modes (should be even)
- `bounds`: Domain boundaries (min, max)
- `dealias`: Dealiasing factor (1.0 = none, 1.5 = 3/2 rule)

**Properties:**
```julia
basis.size        # Number of modes
basis.bounds      # (min, max) tuple
basis.grid_size   # Number of grid points
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

**Grid Points:** Chebyshev-Gauss-Lobatto points, clustered near boundaries.

```julia
# Grid points in [-1, 1]:
# ξ_i = cos(πi / (N-1)),  i = 0, 1, ..., N-1

# Mapped to [a, b]:
# x_i = (a+b)/2 + (b-a)/2 * ξ_i
```

### ChebyshevU

Chebyshev polynomials of the second kind. Grid excludes endpoints.

```julia
basis = ChebyshevU(z; size=64, bounds=(0.0, 1.0))
```

## Legendre Basis

Alternative to Chebyshev for bounded domains.

```julia
basis = Legendre(z; size=64, bounds=(0.0, 1.0))
```

**When to use:**
- Some problems with specific weight functions
- Alternative numerical properties
- Generally similar to ChebyshevT

## Resolution Guidelines

### Fourier Resolution

```julia
# Rule of thumb: resolve smallest scales
L = 2π              # Domain length
η = (ν^3/ε)^(1/4)   # Kolmogorov scale (turbulence)
N = L / η           # Required modes (minimum)
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

Prevents aliasing errors in nonlinear terms.

### 3/2 Rule

Most common dealiasing strategy:

```julia
# Grid is padded by 3/2 during transforms
basis = RealFourier(x; size=128, dealias=1.5)
# Transforms use 192 grid points
# Only 128 modes retained
```

### 2/1 Rule

For highly nonlinear problems:

```julia
basis = RealFourier(x; size=128, dealias=2.0)
```

### No Dealiasing

For linear problems only:

```julia
basis = RealFourier(x; size=128, dealias=1.0)
```

## Multi-Dimensional Domains

Combine bases for each direction:

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(2, 2))

# Periodic x, bounded z
x_basis = RealFourier(coords["x"]; size=256, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))

# Create domain
domain = Domain(dist, (x_basis, z_basis))
```

## Grid and Spectral Space

### Transforms

```julia
# Field starts in spectral space
field = ScalarField(dist, "T", (x_basis, z_basis), Float64)

# Transform to grid space
Tarang.ensure_layout!(field, :g)

# Access grid data
data = field.data_g

# Transform back to spectral
Tarang.ensure_layout!(field, :c)
```

### Grid Points

```julia
# Get grid for a basis
x_grid = get_grid(x_basis)
z_grid = get_grid(z_basis)

# Create 2D meshgrid
X = repeat(x_grid, 1, length(z_grid))
Z = repeat(z_grid', length(x_grid), 1)
```

## Performance Tips

1. **Power-of-2 sizes**: Use N = 2^m for fastest FFTs
2. **Dealiasing cost**: ~50% more computation with 3/2 rule
3. **Chebyshev efficiency**: Sparse differentiation matrices
4. **Memory**: RealFourier uses ~50% memory of ComplexFourier

## See Also

- [Coordinates](coordinates.md): Coordinate systems
- [Domains](domains.md): Creating full domains
- [API: Bases](../api/bases.md): Complete reference
