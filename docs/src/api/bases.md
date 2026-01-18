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
| `ChebyshevT` | Bounded, smooth | Dirichlet/Neumann | Gauss-Lobatto |
| `ChebyshevU` | Bounded, special BCs | Custom | Gauss |
| `Legendre` | Bounded, alternative | Dirichlet/Neumann | Gauss-Lobatto |

---

## Fourier Bases

### RealFourier

Real-valued Fourier basis for periodic coordinates with real data.

**Constructor**:
```julia
RealFourier(
    coord::Coordinate;
    size::Int,
    bounds::Tuple{Float64,Float64},
    dealias::Float64=1.0
)
```

**Arguments**:
- `coord`: Coordinate object for this direction
- `size`: Number of modes (must be even for real Fourier)
- `bounds`: Domain boundaries (min, max)
- `dealias`: Dealiasing factor (default 1.0 = no dealiasing, 2/3 for 3/2 rule)

**Examples**:

```julia
coords = CartesianCoordinates("x")
x = coords["x"]

# Basic real Fourier basis
basis = RealFourier(x, size=128, bounds=(0.0, 2π))

# With dealiasing (3/2 rule)
basis = RealFourier(x, size=128, bounds=(0.0, 2π), dealias=2/3)

# Custom domain length
basis = RealFourier(x, size=256, bounds=(0.0, 4.0))
```

**Properties**:
```julia
basis.coord         # Coordinate
basis.size          # Number of modes
basis.bounds        # (min, max)
basis.length        # Domain length (max - min)
basis.dealias       # Dealiasing factor
basis.grid_size     # Number of grid points
```

**Grid Points**:
```julia
# Get grid points (uniform spacing)
grid = get_grid(basis)
# Returns: [0.0, Δx, 2Δx, ..., L-Δx]

# Grid spacing
Δx = basis.length / basis.grid_size
```

**Wavenumbers**:
```julia
# Get wavenumber array
k = get_wavenumbers(basis)
# Returns: [0, 1, 2, ..., N/2-1, -N/2, ..., -1] * (2π/L)

# Fundamental wavenumber
k0 = 2π / basis.length
```

**Memory Savings**: RealFourier uses real-to-complex FFT, saving ~50% memory compared to ComplexFourier.

---

### ComplexFourier

Complex-valued Fourier basis for periodic coordinates.

**Constructor**:
```julia
ComplexFourier(
    coord::Coordinate;
    size::Int,
    bounds::Tuple{Float64,Float64},
    dealias::Float64=1.0
)
```

**Arguments**: Same as RealFourier

**Examples**:

```julia
coords = CartesianCoordinates("x")
x = coords["x"]

# Complex Fourier basis
basis = ComplexFourier(x, size=128, bounds=(0.0, 2π))
```

**Use Cases**:
- Complex-valued fields (e.g., quantum mechanics)
- Incompatible with RealFourier optimizations
- More general but uses more memory

**Wavenumbers**:
```julia
k = get_wavenumbers(basis)
# Returns: [0, 1, 2, ..., N-1] * (2π/L)
```

---

## Chebyshev Bases

### ChebyshevT

Chebyshev polynomials of the first kind for bounded, non-periodic coordinates.

**Constructor**:
```julia
ChebyshevT(
    coord::Coordinate;
    size::Int,
    bounds::Tuple{Float64,Float64}
)
```

**Arguments**:
- `coord`: Coordinate object
- `size`: Number of modes (polynomial degree + 1)
- `bounds`: Domain boundaries (min, max)

**Examples**:

```julia
coords = CartesianCoordinates("z")
z = coords["z"]

# Chebyshev basis for vertical direction
basis = ChebyshevT(z, size=64, bounds=(0.0, 1.0))

# Larger domain
basis = ChebyshevT(z, size=128, bounds=(-1.0, 1.0))
```

**Properties**:
```julia
basis.coord         # Coordinate
basis.size          # Number of modes
basis.bounds        # (min, max)
basis.length        # Domain length
basis.grid_size     # Number of grid points
```

**Grid Points (Gauss-Lobatto)**:
```julia
# Get Chebyshev-Gauss-Lobatto points
grid = get_grid(basis)
# Points clustered near boundaries for better resolution

# In canonical [-1, 1] domain:
# ξ_i = cos(π*i / (N-1)) for i = 0, 1, ..., N-1

# Mapped to physical domain [a, b]:
# x_i = (a + b)/2 + (b - a)/2 * ξ_i
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
    size::Int,
    bounds::Tuple{Float64,Float64}
)
```

**Use Cases**:
- Specific boundary condition types
- Alternative to ChebyshevT for some problems
- Less commonly used than ChebyshevT

**Grid Points (Gauss)**:
```julia
# Chebyshev-Gauss points (excludes endpoints)
grid = get_grid(basis)
# ξ_i = cos(π*(i+1/2) / N) for i = 0, 1, ..., N-1
```

---

## Legendre Basis

### Legendre

Legendre polynomials for bounded coordinates.

**Constructor**:
```julia
Legendre(
    coord::Coordinate;
    size::Int,
    bounds::Tuple{Float64,Float64}
)
```

**Arguments**: Same as ChebyshevT

**Examples**:

```julia
coords = CartesianCoordinates("z")
z = coords["z"]

# Legendre basis
basis = Legendre(z, size=64, bounds=(0.0, 1.0))
```

**Grid Points (Gauss-Lobatto)**:
```julia
# Legendre-Gauss-Lobatto points
grid = get_grid(basis)
# Points include endpoints, clustered at boundaries
```

**Polynomial Expansion**:

Functions are expanded as:
```math
f(x) = \sum_{n=0}^{N-1} a_n P_n(x)
```

where P_n(x) is the nth Legendre polynomial.

**Comparison with Chebyshev**:
- Similar accuracy and convergence properties
- Different grid point distribution
- Choice often based on problem-specific considerations
- Legendre may be preferred for some weight functions

---

## Compound Bases

For multi-dimensional problems, combine bases for each direction.

**Example: 2D Rayleigh-Bénard**:
```julia
coords = CartesianCoordinates("x", "z")

# Periodic horizontal direction
x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 4.0))

# Bounded vertical direction with walls
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

# Create domain with both bases
domain = Domain(dist, (x_basis, z_basis))
```

**Example: 3D Channel Flow**:
```julia
coords = CartesianCoordinates("x", "y", "z")

# Streamwise (periodic)
x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 2π))

# Spanwise (periodic)
y_basis = RealFourier(coords["y"], size=128, bounds=(0.0, π))

# Wall-normal (bounded)
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

domain = Domain(dist, (x_basis, y_basis, z_basis))
```

---

## Grid and Spectral Space

### Transforms

Converting between grid and spectral space:

```julia
# Field starts in spectral space
field = ScalarField(dist, "T", (x_basis, z_basis))

# Transform to grid space
to_grid!(field)

# Now can evaluate/modify in physical space
grid_data = get_grid_data(field)
grid_data[10, 20] = 1.0

# Transform back to spectral space
to_spectral!(field)
```

### Grid Point Access

```julia
# Get grid points for each basis
x_grid = get_grid(x_basis)
z_grid = get_grid(z_basis)

# Create 2D grid
X = repeat(x_grid, 1, length(z_grid))
Z = repeat(z_grid', length(x_grid), 1)

# Initialize field on grid
T_grid = get_grid_data(T)
for i in 1:size(T_grid, 1), j in 1:size(T_grid, 2)
    x, z = X[i,j], Z[i,j]
    T_grid[i,j] = sin(2π*x) * cos(π*z)
end
```

---

## Differentiation

Spectral differentiation is computed by applying operators in spectral space.

### Fourier Differentiation

For Fourier bases, differentiation is multiplication by ik:

```math
\frac{d}{dx} \left(\sum a_k e^{ikx}\right) = \sum (ik) a_k e^{ikx}
```

**Example**:
```julia
# Derivative operator created automatically
# When you use ∂x(f) in equations

add_equation!(problem, "∂x(u) = f")
```

### Chebyshev Differentiation

For Chebyshev bases, differentiation uses recurrence relations:

```math
\frac{dT_n}{dx} = n U_{n-1}(x)
```

Implemented as sparse matrix multiplication in spectral space.

**Example**:
```julia
# Vertical derivative with Chebyshev basis
add_equation!(problem, "∂z(T) = 0")
```

---

## Dealiasing

Dealiasing prevents aliasing errors in nonlinear terms.

### The 3/2 Rule

Most common dealiasing strategy for spectral methods:

```julia
# Enable 3/2 dealiasing
basis = RealFourier(x, size=128, bounds=(0.0, 2π), dealias=2/3)

# This pads the grid by 3/2 during nonlinear term evaluation
# Grid size: 128 * 3/2 = 192 points
# Retains only 128 modes after multiplication
```

### When to Dealias

- **Always**: For products of fields (e.g., u*∂x(u))
- **Optional**: For linear problems
- **Recommend**: 3/2 rule for most problems
- **Higher**: 2/1 padding for very nonlinear problems

**Configuration**:
```toml
[transforms]
DEALIAS_BEFORE_CONVERTING = true
```

---

## Resolution Guidelines

### Fourier Bases

Rule of thumb: Resolve smallest scales in the flow

```julia
# Estimate from physical parameters
L = 2π          # Domain length
η = (ν^3/ε)^(1/4)  # Kolmogorov scale
N = L / η        # Required modes

# Practical examples
# Low Re flow: 64-128 modes
# Moderate Re: 256-512 modes
# High Re/turbulent: 1024+ modes
```

### Chebyshev/Legendre Bases

Exponential convergence for smooth functions:

```julia
# Smooth flows: 32-64 modes
# Boundary layers: 64-128 modes
# Steep gradients: 128-256 modes

# Example: Rayleigh-Bénard at Ra=10^6
z_basis = ChebyshevT(z, size=64, bounds=(0.0, 1.0))
```

---

## Advanced Topics

### Custom Grid Sizes

```julia
# Manually specify grid size (for custom padding)
basis = RealFourier(x, size=128, bounds=(0.0, 2π))
basis.grid_size = 192  # 3/2 padding
```

### Basis Algebra

```julia
# Check compatibility
is_compatible(basis1, basis2)

# Get combined grid
combined_grid = get_combined_grid([basis1, basis2])
```

### Spectral Accuracy

```julia
# Estimate spectral coefficients decay
coeffs = get_spectral_data(field)
decay = abs.(coeffs)

# Plot to check resolution
using Plots
plot(decay, yscale=:log10,
     xlabel="Mode", ylabel="Coefficient magnitude")
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
