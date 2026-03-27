# Domains API

A domain represents the spatial discretization of your problem, combining coordinate systems, spectral bases, and MPI distribution.

## Domain

The domain class combines multiple spectral bases and manages the spatial discretization.

**Constructor**:
```julia
Domain(
    distributor::Distributor,
    bases::Tuple{Vararg{Basis}}
)
```

**Arguments**:
- `distributor`: MPI distributor managing process distribution
- `bases`: Tuple of spectral bases, one per coordinate dimension

**Examples**:

### 1D Domain

```julia
using Tarang, MPI

MPI.Init()

# Setup coordinates and distributor
coords = CartesianCoordinates("x")
dist = Distributor(coords, mesh=(4,))

# Create basis
x_basis = RealFourier(coords["x"], size=128, bounds=(0.0, 2π))

# Create domain
domain = Domain(dist, (x_basis,))
```

### 2D Domain

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

# Periodic horizontal, bounded vertical
x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

domain = Domain(dist, (x_basis, z_basis))
```

### 3D Domain

```julia
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords, mesh=(4, 4, 2))

# All periodic (e.g., turbulence)
x_basis = RealFourier(coords["x"], size=256, bounds=(0.0, 2π))
y_basis = RealFourier(coords["y"], size=256, bounds=(0.0, 2π))
z_basis = RealFourier(coords["z"], size=256, bounds=(0.0, 2π))

domain = Domain(dist, (x_basis, y_basis, z_basis))
```

---

## Properties

### Basic Properties

```julia
domain.distributor      # Distributor: MPI distribution
domain.bases            # Tuple: Spectral bases
domain.dim              # Int: Number of dimensions
domain.coords           # CoordinateSystem: Coordinate system
```

### Size Information

```julia
# Global domain size (spectral modes)
domain.global_size      # Tuple: (Nx, Ny, Nz)

# Global grid size (grid points)
domain.global_grid_size # Tuple: Grid dimensions

# Local size on this MPI rank
domain.local_size       # Tuple: Local array size

# Domain bounds
domain.bounds           # Tuple of tuples: ((xmin,xmax), (ymin,ymax), ...)

# Domain lengths
domain.lengths          # Tuple: (Lx, Ly, Lz)
```

---

## Methods

### Grid Access

```julia
# Get grid points for each dimension
x_grid = get_grid(domain, 1)  # First dimension
y_grid = get_grid(domain, 2)  # Second dimension
z_grid = get_grid(domain, 3)  # Third dimension

# Or access via basis
x_grid = get_grid(domain.bases[1])
```

### Meshgrid Creation

```julia
# Create 2D meshgrid
X, Z = meshgrid(domain)

# For 3D
X, Y, Z = meshgrid(domain)

# Example: Initialize field on grid
function init_temperature!(T, domain)
    X, Z = meshgrid(domain)
    T_grid = get_grid_data(T)

    T_grid .= @. sin(2π * X / domain.lengths[1]) * cos(π * Z / domain.lengths[2])

    to_spectral!(T)
end
```

### Wavenumber Arrays

```julia
# Get wavenumber arrays
kx = get_wavenumbers(domain, 1)
ky = get_wavenumbers(domain, 2)
kz = get_wavenumbers(domain, 3)

# Wavenumber magnitude
function get_k_magnitude(domain)
    kx = get_wavenumbers(domain, 1)
    ky = get_wavenumbers(domain, 2)
    kz = get_wavenumbers(domain, 3)

    # 3D meshgrid of wavenumbers
    KX = reshape(kx, :, 1, 1)
    KY = reshape(ky, 1, :, 1)
    KZ = reshape(kz, 1, 1, :)

    K = @. sqrt(KX^2 + KY^2 + KZ^2)
    return K
end
```

---

## Domain Types and Examples

### Periodic Domains

All directions periodic (e.g., homogeneous turbulence):

```julia
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords, mesh=(4, 4, 4))

bases = (
    RealFourier(coords["x"], size=128, bounds=(0.0, 2π)),
    RealFourier(coords["y"], size=128, bounds=(0.0, 2π)),
    RealFourier(coords["z"], size=128, bounds=(0.0, 2π))
)

domain = Domain(dist, bases)
```

**Use cases**: Turbulence, Taylor-Green vortex, forced isotropic flow

---

### Channel Domains

Periodic in streamwise/spanwise, bounded in wall-normal:

```julia
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords, mesh=(8, 4, 2))

bases = (
    RealFourier(coords["x"], size=256, bounds=(0.0, 2π)),  # Streamwise
    RealFourier(coords["y"], size=128, bounds=(0.0, π)),   # Spanwise
    ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))    # Wall-normal
)

domain = Domain(dist, bases)
```

**Use cases**: Channel flow, pipe flow, boundary layers

---

### Convection Domains

Periodic horizontal, bounded vertical:

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(4, 2))

bases = (
    RealFourier(coords["x"], size=256, bounds=(0.0, 4.0)),  # Horizontal
    ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))     # Vertical
)

domain = Domain(dist, bases)
```

**Use cases**: Rayleigh-Bénard convection, atmospheric convection

---

### Spherical Domains

For problems in spherical geometry:

```julia
coords = SphericalCoordinates()
dist = Distributor(coords, mesh=(2, 2, 2))

bases = (
    ChebyshevT(coords["r"], size=64, bounds=(0.5, 1.0)),      # Radius
    RealFourier(coords["theta"], size=128, bounds=(0.0, π)),  # Polar
    RealFourier(coords["phi"], size=256, bounds=(0.0, 2π))    # Azimuthal
)

domain = Domain(dist, bases)
```

**Use cases**: Spherical shells, planetary cores, stellar interiors

---

## Multi-Domain Problems

Some problems require multiple domains (e.g., domain decomposition, multi-scale):

```julia
# Coarse domain for large scales
coords_coarse = CartesianCoordinates("x", "z")
dist_coarse = Distributor(coords_coarse, mesh=(2, 2))
bases_coarse = (
    RealFourier(coords_coarse["x"], size=64, bounds=(0.0, 4.0)),
    ChebyshevT(coords_coarse["z"], size=32, bounds=(0.0, 1.0))
)
domain_coarse = Domain(dist_coarse, bases_coarse)

# Fine domain for small scales
dist_fine = Distributor(coords_coarse, mesh=(4, 4))
bases_fine = (
    RealFourier(coords_coarse["x"], size=256, bounds=(0.0, 4.0)),
    ChebyshevT(coords_coarse["z"], size=128, bounds=(0.0, 1.0))
)
domain_fine = Domain(dist_fine, bases_fine)
```

---

## Domain Utilities

### Volume Element

```julia
# Get volume element (dV)
dV = get_volume_element(domain)

# Useful for integrals
function integrate(field, domain)
    to_grid!(field)
    data = get_grid_data(field)
    dV = get_volume_element(domain)
    return sum(data .* dV)
end
```

### Domain Information

```julia
# Print domain information
function print_domain_info(domain)
    println("Domain Information:")
    println("  Dimensions: $(domain.dim)")
    println("  Global size: $(domain.global_size)")
    println("  Grid size: $(domain.global_grid_size)")
    println("  Bounds: $(domain.bounds)")
    println("  Lengths: $(domain.lengths)")

    for (i, basis) in enumerate(domain.bases)
        println("  Basis $i: $(typeof(basis))")
        println("    Size: $(basis.size)")
        println("    Bounds: $(basis.bounds)")
    end
end
```

### Domain Validation

```julia
# Check domain validity
function validate_domain(domain)
    # Check dimensions match
    @assert length(domain.bases) == domain.dim
    @assert length(domain.distributor.mesh) == domain.dim

    # Check process mesh matches
    @assert prod(domain.distributor.mesh) == MPI.Comm_size(domain.distributor.comm)

    # Check bases match coordinates
    for (i, basis) in enumerate(domain.bases)
        @assert basis.coord.index == i
    end

    println("Domain validation passed!")
end
```

---

## Domain Transformations

### Coordinate Transformations

For non-Cartesian geometries, coordinate transformations are handled automatically:

```julia
# Spherical coordinates example
coords = SphericalCoordinates()
# Metric tensor and Jacobian computed automatically
# Operators (grad, div, curl) use correct forms
```

### Grid Refinement

```julia
# Create refined domain (double resolution)
function refine_domain(domain_coarse)
    bases_fine = map(domain_coarse.bases) do basis
        if typeof(basis) <: RealFourier
            RealFourier(basis.coord,
                       size=2*basis.size,
                       bounds=basis.bounds,
                       dealias=basis.dealias)
        elseif typeof(basis) <: ChebyshevT
            ChebyshevT(basis.coord,
                      size=2*basis.size,
                      bounds=basis.bounds)
        end
    end

    return Domain(domain_coarse.distributor, bases_fine)
end
```

---

## Advanced Topics

### Custom Domains

For specialized geometries:

```julia
# Define custom domain with mixed basis types
coords = CartesianCoordinates("r", "theta")
dist = Distributor(coords, mesh=(2, 2))

bases = (
    ChebyshevT(coords["r"], size=64, bounds=(0.0, 1.0)),  # Radial
    RealFourier(coords["theta"], size=128, bounds=(0.0, 2π))  # Angular
)

domain = Domain(dist, bases)
```

### Domain Decomposition

For large problems, decompose into subdomains:

```julia
# Create subdomain for specific region
function create_subdomain(domain, bounds_x, bounds_z)
    # Extract relevant portions
    # ... implementation depends on specific needs
end
```

### Adaptive Mesh Refinement

```julia
# Refine in specific regions (advanced)
# Would require custom basis implementation
# Not currently built-in
```

---

## Performance Considerations

### Memory Usage

```julia
# Estimate memory usage
function estimate_memory(domain)
    bytes_per_point = 16  # Complex128
    points = prod(domain.global_grid_size)
    total_gb = bytes_per_point * points / 1e9

    per_rank_gb = total_gb / MPI.Comm_size(domain.distributor.comm)

    println("Total memory: $(total_gb) GB")
    println("Per rank: $(per_rank_gb) GB")
end
```

### Optimal Domain Decomposition

```julia
# Choose mesh based on domain aspect ratio
function optimal_mesh(Nx, Ny, Nz, total_procs)
    # Aim for cubic subdomains
    ratio = (Nx, Ny, Nz) ./ sum([Nx, Ny, Nz])

    # Distribute processes proportionally
    # ... heuristic for choosing mesh

    return (Px, Py, Pz)
end
```

---

## See Also

- [Coordinates](coordinates.md): Coordinate system setup
- [Bases](bases.md): Spectral basis selection
- [Fields](fields.md): Creating fields on domains
- [Operators](operators.md): Differential operators
- [Parallelism Guide](../pages/parallelism.md): MPI configuration
