# Coordinates

Coordinates define the spatial dimensions and geometry of your simulation domain.

## Cartesian Coordinates

For rectangular domains with uniform grid spacing in grid space.

```julia
using Tarang

# 1D
coords = CartesianCoordinates("x")

# 2D
coords = CartesianCoordinates("x", "z")

# 3D
coords = CartesianCoordinates("x", "y", "z")
```

### Accessing Coordinates

```julia
# Get specific coordinate
x = coords["x"]
z = coords["z"]

# Iterate over all coordinates
for coord in coords
    println(coord.name)
end

# Number of dimensions
ndim = length(coords)
```

### Coordinate Properties

```julia
coord = coords["x"]

coord.name      # String: "x"
coord.index     # Int: position in coordinate tuple
coord.system    # Symbol: :cartesian
```

## Distributor

The Distributor manages MPI process distribution across the domain.

```julia
# Create distributor
dist = Distributor(coords; mesh=(2, 2), dtype=Float64)
```

### Parameters

- `coords`: Coordinate system
- `mesh`: Tuple specifying MPI process grid
- `dtype`: Data type (Float64, Float32)

### MPI Process Mesh

The mesh determines how processes are arranged:

```julia
# 4 processes in 2×2 grid
dist = Distributor(coords; mesh=(2, 2))

# 8 processes: 4 in x, 2 in z
dist = Distributor(coords; mesh=(4, 2))

# Serial (single process)
dist = Distributor(coords; mesh=(1,))
```

**Guidelines:**
- Product of mesh dimensions = total MPI processes
- Match mesh to domain aspect ratio
- Balance communication vs. computation

### Distributor Properties

```julia
dist.coords     # Coordinate system
dist.comm       # MPI communicator
dist.rank       # This process's rank
dist.size       # Total number of processes
dist.mesh       # Process mesh tuple
```

## Domain Decomposition

### Local vs Global Indices

```julia
# Get local indices for this process (internal function)
local_idx = Tarang.local_indices(dist, axis, global_size)

# Example: axis 1 with 128 global points on 4 processes
# Process 0: 1:32
# Process 1: 33:64
# Process 2: 65:96
# Process 3: 97:128
```

### Pencil Decomposition

Tarang uses pencil (slab) decomposition for efficient parallel FFTs:

```
3D domain distributed across 4 processes (2×2 mesh):

    Process 0     Process 1
    ┌─────────┐  ┌─────────┐
    │ ▓▓▓▓▓▓▓ │  │ ░░░░░░░ │
    │ ▓▓▓▓▓▓▓ │  │ ░░░░░░░ │
    │ ▓▓▓▓▓▓▓ │  │ ░░░░░░░ │
    └─────────┘  └─────────┘
    Process 2     Process 3
    ┌─────────┐  ┌─────────┐
    │ ▒▒▒▒▒▒▒ │  │ ████████ │
    │ ▒▒▒▒▒▒▒ │  │ ████████ │
    │ ▒▒▒▒▒▒▒ │  │ ████████ │
    └─────────┘  └─────────┘
```

## Coordinate-Aware Operations

### Derivatives

Coordinate names determine derivative operators:

```julia
# ∂x, ∂y, ∂z automatically available for Cartesian
add_equation!(problem, "∂t(u) = ∂x(T)")

# The operator name matches the coordinate name
# coords["x"] → ∂x(field)
# coords["z"] → ∂z(field)
```

### Grid Access

```julia
# Get grid points for a basis
x_grid = get_grid(x_basis)  # Returns array of x values

# Get local grid for distributed field
local_grid = get_local_grid(field, axis)
```

## Best Practices

### Choosing Mesh Dimensions

| Domain Shape | Recommended Mesh |
|--------------|------------------|
| Square 2D    | (n, n)           |
| Wide 2D      | (2n, n)          |
| Tall 2D      | (n, 2n)          |
| Cubic 3D     | (n, n, n)        |
| Channel 3D   | (2n, n, n)       |

### Memory Considerations

- Each process holds `global_size / num_processes` data
- Communication overhead scales with surface area between processes
- Balance process count with per-process work

## See Also

- [Bases](bases.md): Spectral bases for each coordinate
- [Domains](domains.md): Combining coordinates and bases
- [Parallelism](parallelism.md): MPI details
