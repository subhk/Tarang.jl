# Coordinates API

Coordinates define the dimensional structure of your problem domain. Tarang.jl supports Cartesian, spherical, and polar coordinate systems.

## Docstrings

```@docs
CartesianCoordinates
coords(::CartesianCoordinates)
```

## Coordinate Systems

### CartesianCoordinates

Cartesian coordinate system for rectangular domains.

**Constructor**:
```julia
CartesianCoordinates(names::String...)
```

**Arguments**:
- `names`: Coordinate dimension names (e.g., "x", "y", "z")

**Examples**:

```julia
# 1D Cartesian
coords = CartesianCoordinates("x")

# 2D Cartesian
coords = CartesianCoordinates("x", "z")

# 3D Cartesian
coords = CartesianCoordinates("x", "y", "z")
```

**Properties**:
```julia
coords.names        # Tuple of coordinate names
coords.dim          # Number of dimensions
coords.coords       # Dictionary: name => Coordinate object
```

**Methods**:

#### Accessing Coordinates

```julia
# Get coordinate by name
x_coord = coords["x"]

# Get coordinate by index
x_coord = coords[1]

# Iterate over coordinates
for coord in coords
    println(coord.name)
end
```

#### Unit Vectors

```julia
# Get unit vector for coordinate direction
ex = unit_vector(coords, "x")
ey = unit_vector(coords, "y")
ez = unit_vector(coords, "z")

# Use in equations
# Example: buoyancy force in z-direction
add_equation!(problem, "∂t(u) = ... + Ra*Pr*T*ez")
```

---

### SphericalCoordinates

Spherical coordinate system (r, θ, φ) for problems with spherical geometry.

!!! note
    SphericalCoordinates is not yet implemented in Tarang.jl. This is planned for a future release.

**Constructor**:
```julia
SphericalCoordinates()
SphericalCoordinates(names::Tuple{String,String,String})
```

**Default names**: ("r", "theta", "phi")

**Examples**:

```julia
# Default spherical coordinates
coords = SphericalCoordinates()

# Custom names
coords = SphericalCoordinates(("radius", "theta", "phi"))
```

**Coordinate ranges**:
- r: [0, ∞)
- θ: [0, π]
- φ: [0, 2π]

**Metric tensor**: Available for computing gradient, divergence, curl in spherical coordinates

---

### PolarCoordinates

Polar coordinate system (r, φ) for 2D axisymmetric problems.

!!! note
    PolarCoordinates is not yet implemented in Tarang.jl. This is planned for a future release.

**Constructor**:
```julia
PolarCoordinates()
PolarCoordinates(names::Tuple{String,String})
```

**Default names**: ("r", "phi")

**Examples**:

```julia
# Default polar coordinates
coords = PolarCoordinates()

# Custom names
coords = PolarCoordinates(("radius", "angle"))
```

**Coordinate ranges**:
- r: [0, ∞)
- φ: [0, 2π]

---

## Coordinate Object

Individual coordinate dimension.

**Properties**:
```julia
coord.name          # String: Coordinate name
coord.index         # Int: Position in coordinate system
coord.system        # CoordinateSystem: Parent system
```

**Example**:
```julia
coords = CartesianCoordinates("x", "y", "z")
x = coords["x"]

println(x.name)     # "x"
println(x.index)    # 1
```

---

## Distributor

The distributor manages MPI process distribution across coordinate dimensions.

**Constructor**:
```julia
Distributor(
    coords::CoordinateSystem;
    mesh::Tuple{Int,...},
    device::String="cpu"
)
```

**Arguments**:
- `coords`: Coordinate system
- `mesh`: MPI process mesh (one value per dimension)

**Examples**:

```julia
# 2D distribution with 2×2 process mesh
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

# 3D distribution with 4×4×2 process mesh
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords, mesh=(4, 4, 2))
```

**Properties**:
```julia
dist.coords         # CoordinateSystem
dist.mesh           # Tuple: Process mesh dimensions
dist.comm           # MPI.Comm: MPI communicator
dist.rank           # Int: MPI rank
dist.size           # Int: Total number of processes
```

**Methods**:

#### Process Information

```julia
# Get process rank
rank = get_rank(dist)

# Get total number of processes
nprocs = get_size(dist)

# Check if this is the root process
is_root = (get_rank(dist) == 0)
```

#### Domain Decomposition

```julia
# Get local domain bounds for this process
local_bounds = get_local_bounds(dist, basis)

# Get global domain size
global_size = get_global_size(dist, basis)
```

---

## Mesh Configuration

### Choosing Process Mesh

The process mesh determines how data is distributed across MPI processes.

**General rules**:
1. Product of mesh dimensions must equal number of MPI processes
2. Match mesh aspect ratio to domain aspect ratio
3. Use powers of 2 when possible for optimal FFT performance
4. Balance computation and communication

**Examples**:

```julia
# Square 2D mesh (recommended for square domains)
mesh=(4, 4)  # 16 processes

# Rectangular 2D mesh (for wide domains)
mesh=(8, 2)  # 16 processes, more in x-direction

# 3D mesh (for cubic domains)
mesh=(4, 4, 4)  # 64 processes

# 3D mesh (for stratified flows, thin in vertical)
mesh=(8, 8, 2)  # 128 processes, fewer in z-direction
```

### Process Mesh Strategies

#### 2D Problems

```julia
# Balanced decomposition (default)
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(4, 4))

# More processes in horizontal direction (wide domain)
dist = Distributor(coords, mesh=(8, 2))

# More processes in vertical direction (tall domain)
dist = Distributor(coords, mesh=(2, 8))
```

#### 3D Problems

```julia
# Cubic mesh (isotropic domains)
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords, mesh=(4, 4, 4))

# Anisotropic mesh (stratified flows)
# More processes in horizontal directions
dist = Distributor(coords, mesh=(8, 8, 2))

# Channel flow mesh
# More processes in streamwise and spanwise
dist = Distributor(coords, mesh=(8, 4, 2))
```

---

## Pencil Decomposition

Tarang.jl uses pencil decomposition for 3D problems, where data is distributed in two dimensions while remaining contiguous in one dimension.

### Pencil Orientations

For 3D domains, data can be organized in different pencil configurations:

- **X-pencils**: Contiguous in x, distributed in y and z
- **Y-pencils**: Contiguous in y, distributed in x and z
- **Z-pencils**: Contiguous in z, distributed in x and y

**Example**:
```julia
# 3D domain with 4×4×2 process mesh
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords, mesh=(4, 4, 2))

# PencilArrays automatically handles pencil rotations
# during FFT operations
```

### Transpose Operations

Pencil decomposition requires transpose operations to switch between orientations:

```julia
# Configured in tarang.toml
[parallelism]
TRANSPOSE_LIBRARY = "PENCIL"  # Use PencilArrays
GROUP_TRANSPOSES = true       # Batch transposes for efficiency
```

---

## Usage Examples

### Basic Setup

```julia
using Tarang, MPI

MPI.Init()

# Define coordinate system
coords = CartesianCoordinates("x", "y", "z")

# Create distributor with process mesh
dist = Distributor(coords, mesh=(2, 2, 2))

# Verify setup
if get_rank(dist) == 0
    println("Running on $(get_size(dist)) processes")
    println("Process mesh: $(dist.mesh)")
end
```

### Multi-Resolution Studies

```julia
# Create multiple distributors for different resolutions
coords = CartesianCoordinates("x", "z")

# Coarse resolution
dist_coarse = Distributor(coords, mesh=(2, 2))

# Fine resolution
dist_fine = Distributor(coords, mesh=(4, 4))

# Run simulation at each resolution
# ... (use different dist for each case)
```

### Custom Coordinate Names

```julia
# Use physics-appropriate names
coords = CartesianCoordinates("streamwise", "spanwise", "wall_normal")
dist = Distributor(coords, mesh=(8, 4, 2))

# Access by custom names
x = coords["streamwise"]
y = coords["spanwise"]
z = coords["wall_normal"]
```

---

## Advanced Topics

### Custom MPI Communicators

```julia
# Use custom MPI communicator
custom_comm = MPI.Comm_split(MPI.COMM_WORLD, color, key)
dist = Distributor(coords, mesh=(2, 2), comm=custom_comm)
```

### Load Balancing

```julia
# Check load distribution
if get_rank(dist) == 0
    for rank in 0:(get_size(dist)-1)
        local_size = get_local_size(dist, basis, rank)
        println("Rank $rank: $local_size")
    end
end
```

---

## Performance Tips

1. **Use power-of-2 mesh dimensions** for optimal FFT performance
2. **Match mesh to domain aspect ratio** for balanced load
3. **Test different meshes** to find optimal configuration
4. **Monitor communication overhead** with profiling tools
5. **Consider memory constraints** when choosing mesh

---

## See Also

- [Bases](bases.md): Spectral basis functions for each coordinate
- [Domains](domains.md): Combining coordinates and bases into domains
- [Parallelism Guide](../pages/parallelism.md): Detailed MPI configuration
- [Configuration](../pages/configuration.md): Process mesh tuning
