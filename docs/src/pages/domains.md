# Domains

Domains combine spectral bases with the MPI distributor to define the computational space.

## Creating a Domain

```julia
using Tarang

# Setup coordinates and distributor
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(2, 2), dtype=Float64)

# Define spectral bases
x_basis = RealFourier(coords["x"]; size=128, bounds=(0.0, 2π))
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))

# Create domain
domain = Domain(dist, (x_basis, z_basis))
```

## Domain Properties

```julia
domain.distributor    # MPI distributor
domain.bases          # Tuple of spectral bases
domain.ndim           # Number of dimensions
domain.shape          # Global grid shape
domain.local_shape    # Local grid shape on this process
```

## Quick Domain Creation

Tarang provides convenience functions for common domain types:

### Periodic Box

```julia
# 2D doubly-periodic domain
domain, dist = create_2d_periodic_domain(
    Lx=2π, Ly=2π,    # Domain size
    Nx=128, Ny=128,   # Resolution
    mesh=(2, 2)       # MPI processes
)

# 3D triply-periodic domain
domain, dist = create_3d_periodic_domain(
    Lx=2π, Ly=2π, Lz=2π,
    Nx=64, Ny=64, Nz=64,
    mesh=(2, 2, 2)
)
```

### Channel Domain

```julia
# Periodic in x, bounded in z
domain, dist = create_channel_domain(
    Lx=4.0, Lz=1.0,
    Nx=256, Nz=64,
    mesh=(2, 2)
)
```

### Box Domain

```julia
# Bounded in all directions
domain, dist = create_box_domain(
    Lx=1.0, Ly=1.0, Lz=1.0,
    Nx=64, Ny=64, Nz=64,
    mesh=(2, 2, 2)
)
```

## Working with Domains

### Creating Fields

```julia
# ScalarField
T = ScalarField(dist, "T", domain.bases, Float64)

# VectorField
u = VectorField(dist, coords, "u", domain.bases, Float64)
```

### Grid Information

```julia
# Get global grid points for each axis
for (i, basis) in enumerate(domain.bases)
    grid = get_grid(basis)
    println("Axis $i: $(length(grid)) points")
end

# Get local grid portion
local_grid = get_local_grid(domain, axis)
```

### Domain Dimensions

```julia
# Total number of grid points
total_points = prod(domain.shape)

# Number of spectral modes
total_modes = prod(basis.size for basis in domain.bases)

# Domain volume
volume = prod(basis.bounds[2] - basis.bounds[1] for basis in domain.bases)
```

## Multi-Dimensional Layouts

### Grid Space (:g)

Data stored as real values at collocation points:

```julia
Tarang.ensure_layout!(field, :g)
# field.data_g contains real grid values
# Shape: (Nx, Nz) for 2D
```

### Coefficient Space (:c)

Data stored as spectral coefficients:

```julia
Tarang.ensure_layout!(field, :c)
# field.data_c contains spectral coefficients
# Shape depends on basis types
```

### Layout Transforms

```julia
# Grid to spectral
Tarang.ensure_layout!(field, :c)

# Spectral to grid
Tarang.ensure_layout!(field, :g)

# Check current layout
current_layout = field.current_layout  # :g or :c
```

## Domain Decomposition

The domain is distributed across MPI processes:

```
2D Domain (128 × 64) on 4 processes (2×2 mesh):

Process 0: x ∈ [0:64], z ∈ [0:32]
Process 1: x ∈ [64:128], z ∈ [0:32]
Process 2: x ∈ [0:64], z ∈ [32:64]
Process 3: x ∈ [64:128], z ∈ [32:64]
```

### Accessing Local Data

```julia
# Local data array
local_data = field.data_g

# Local array size
local_size = size(local_data)

# Global index range for this process (internal function)
start_idx, end_idx = Tarang.local_indices(dist, axis, global_size)
```

## Coordinate Systems

### Cartesian

```julia
coords = CartesianCoordinates("x", "y", "z")
# dx, dy, dz derivatives
# grad, div, curl, lap in standard form
```

### Future Support

Spherical and cylindrical coordinates are planned:

```julia
# Planned API (not yet implemented)
coords = SphericalCoordinates("r", "θ", "φ")
coords = CylindricalCoordinates("r", "θ", "z")
```

## Performance Considerations

### Mesh Selection

Match mesh to domain aspect ratio:

| Domain Aspect | Mesh |
|---------------|------|
| Square (Lx ≈ Lz) | (n, n) |
| Wide (Lx > Lz) | (2n, n) |
| Tall (Lx < Lz) | (n, 2n) |

### Resolution Balance

- More modes in directions with:
  - Smaller scales
  - Sharper gradients
  - More active dynamics

### Memory Usage

```julia
# Estimate memory per field
bytes_per_element = 8  # Float64
total_elements = prod(domain.shape)
memory_per_field = total_elements * bytes_per_element

# Per process
memory_per_process = memory_per_field / dist.size
```

## See Also

- [Coordinates](coordinates.md): Coordinate systems
- [Bases](bases.md): Spectral basis types
- [Fields](fields.md): Creating fields on domains
- [API: Domains](../api/domains.md): Complete reference
