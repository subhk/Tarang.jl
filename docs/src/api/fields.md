# Fields API

Fields represent variables in your PDE system. Tarang.jl provides scalar, vector, and tensor fields with CPU, GPU, and MPI support.

## Quick Field Creation

The simplest way to create fields is from a domain:

```julia
domain = PeriodicDomain(128, 128)
T = ScalarField(domain, "temperature")
u = VectorField(domain, "velocity")
```

## ScalarField

### Constructors

```julia
# From domain (recommended)
ScalarField(domain::Domain, name::String; dtype=domain.dist.dtype)

# Full form
ScalarField(distributor::Distributor, name::String, bases::Tuple, dtype::Type=Float64)
```

**Example**:
```julia
# Simple (using convenience API)
domain = ChannelDomain(256, 64; Lx=4.0, Lz=1.0)
T = ScalarField(domain, "T")

# Explicit (when you need full control)
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords)
xb = RealFourier(coords["x"]; size=256, bounds=(0.0, 4.0))
zb = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))
T = ScalarField(dist, "T", (xb, zb))
```

### Data Access

```julia
# Grid-space data (physical values)
ensure_layout!(field, :g)
data = get_grid_data(field)         # AbstractArray

# Coefficient-space data (spectral coefficients)
ensure_layout!(field, :c)
data = get_coeff_data(field)        # AbstractArray

# Shorthand (auto-transforms if needed)
data_g = field["g"]                  # Grid data
data_c = field["c"]                  # Coeff data

# Set data
set_grid_data!(field, array)
set_coeff_data!(field, array)
```

### Layout Management

Fields live in either grid space (`:g`) or coefficient space (`:c`):

```julia
ensure_layout!(field, :g)           # Transform to grid space if needed
ensure_layout!(field, :c)           # Transform to coefficient space if needed
forward_transform!(field)           # Grid -> coefficients
backward_transform!(field)          # Coefficients -> grid
```

### Initial Conditions

```julia
# Fill with random data
fill_random!(field, "g"; seed=42, distribution="normal", scale=1.0)

# Manual initialization
ensure_layout!(field, :g)
data = get_grid_data(field)
data .= sin.(x_grid) .* cos.(y_grid)
```

## VectorField

### Constructors

```julia
# From domain (recommended)
VectorField(domain::Domain, name::String; dtype=domain.dist.dtype)

# Full form
VectorField(dist::Distributor, coordsys::CoordinateSystem, name::String, bases::Tuple)
```

**Example**:
```julia
domain = PeriodicDomain(128, 128, 128)
u = VectorField(domain, "velocity")

# Access components
ux = u.components[1]    # x-component (ScalarField)
uy = u.components[2]    # y-component
uz = u.components[3]    # z-component

# Or use indexing
ux = u[1]
```

### Vector Operations

```julia
# Dot product: u . v
result = evaluate(dot(u, v))    # or: u . v

# Cross product (3D): u x v
result = evaluate(cross(u, v))  # or: u x v

# Gradient of scalar
grad_T = evaluate(grad(T))      # Returns VectorField

# Divergence of vector
div_u = evaluate(div(u))        # Returns ScalarField
```

## TensorField

```julia
# Create rank-2 tensor
tau = TensorField(domain, "stress")

# Access components
tau_xx = tau.components[1, 1]
tau_xy = tau.components[1, 2]
```

## GPU Fields

All field operations work on GPU arrays when the domain uses `arch=GPU()`:

```julia
using CUDA
domain = PeriodicDomain(256, 256; arch=GPU())
u = ScalarField(domain, "u")    # Data lives on GPU
forward_transform!(u)            # Uses cuFFT
```

## Type Parameters

Fields are parametric: `ScalarField{T, S}` where:
- `T`: element type (`Float64`, `Float32`, etc.)
- `S`: storage backend (`SerialFieldStorage`, `TransposableFieldStorage`)

This enables the compiler to specialize operations based on element type and storage strategy.
