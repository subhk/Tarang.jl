# Fields API

Fields represent variables in your PDE system. Tarang.jl provides scalar, vector, and tensor fields distributed across MPI processes.

## ScalarField

### Constructor

```julia
ScalarField(
    distributor::Distributor,
    name::String,
    bases::Tuple
) -> ScalarField
```

**Arguments**:
- `distributor`: MPI distributor managing parallel decomposition
- `name`: Field name (used in equations and output)
- `bases`: Tuple of spectral bases, one per dimension

**Example**:
```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

x_basis = RealFourier(coords["x"], size=128, bounds=(0.0, 2π))
z_basis = ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))

T = ScalarField(dist, "temperature", (x_basis, z_basis))
```

### Methods

#### Data Access

```julia
# Get spectral-space data (distributed PencilArray)
spectral_data = get_spectral_data(field)

# Get grid-space data (for evaluation/initial conditions)
grid_data = get_grid_data(field)

# Get local portion of data on this MPI rank
local_data = field.data
```

#### Space Transforms

```julia
# Transform from grid to spectral space
to_spectral!(field)

# Transform from spectral to grid space
to_grid!(field)

# Check current space
is_in_spectral_space(field)  # returns Bool
is_in_grid_space(field)      # returns Bool
```

#### Field Operations

```julia
# Set all values
fill!(field, value)

# Copy between fields
copy!(dest_field, src_field)

# Scale field
scale!(field, factor)

# Add fields: field1 = field1 + alpha * field2
axpy!(alpha, field2, field1)
```

### Properties

```julia
field.name          # String: Field name
field.bases         # Tuple: Spectral bases
field.data          # PencilArray: Distributed data
field.space         # Symbol: :spectral or :grid
field.distributor   # Distributor: MPI distribution
```

## VectorField

### Constructor

```julia
VectorField(
    distributor::Distributor,
    coords::Coordinates,
    name::String,
    bases::Tuple
) -> VectorField
```

**Arguments**:
- `distributor`: MPI distributor
- `coords`: Coordinate system (determines vector dimensionality)
- `name`: Base name for vector components
- `bases`: Spectral bases tuple

**Example**:
```julia
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords, mesh=(2, 2, 2))

bases = (
    RealFourier(coords["x"], size=128, bounds=(0.0, 2π)),
    RealFourier(coords["y"], size=128, bounds=(0.0, 2π)),
    ChebyshevT(coords["z"], size=64, bounds=(0.0, 1.0))
)

# Creates velocity vector u = (u_x, u_y, u_z)
u = VectorField(dist, coords, "u", bases)

# Access components
ux = u.components[1]  # x-component
uy = u.components[2]  # y-component
uz = u.components[3]  # z-component
```

### Methods

#### Component Access

```julia
# Get specific component
ux = u.components[1]
uy = u[2]  # Alternative indexing

# Iterate over components
for component in u.components
    # ... operate on each component
end
```

#### Vector Operations

```julia
# Magnitude (in grid space)
to_grid!(u)
mag = sqrt.(sum(c.data.^2 for c in u.components))

# Dot product: a · b
function dot_product(a::VectorField, b::VectorField)
    sum(ac.data .* bc.data for (ac, bc) in zip(a.components, b.components))
end

# Cross product (3D only): a × b
c = cross(a, b)  # or: c = a × b
```

### Properties

```julia
u.name          # String: Base name
u.components    # Vector{ScalarField}: Vector components
u.coords        # Coordinates: Coordinate system
u.ndim          # Int: Number of dimensions
```

## TensorField

### Constructor

```julia
TensorField(
    distributor::Distributor,
    coords::Coordinates,
    name::String,
    bases::Tuple;
    symmetric=false
) -> TensorField
```

**Arguments**:
- `distributor`: MPI distributor
- `coords`: Coordinate system
- `name`: Base name for tensor components
- `bases`: Spectral bases tuple
- `symmetric`: If true, only store upper triangular part

**Example**:
```julia
# Stress tensor τ (symmetric)
tau = TensorField(dist, coords, "tau", bases, symmetric=true)

# Velocity gradient tensor ∇u (not symmetric)
grad_u = TensorField(dist, coords, "grad_u", bases, symmetric=false)
```

### Methods

#### Component Access

```julia
# Get component (i,j)
tau_xx = tensor[1, 1]
tau_xy = tensor[1, 2]

# For symmetric tensors, tau[i,j] == tau[j,i]
```

## Field Initialization

### From Function

```julia
# Initialize scalar field
function init_temperature!(T, Lx, Lz)
    T_grid = get_grid_data(T)

    for i in 1:size(T_grid, 1), j in 1:size(T_grid, 2)
        x = (i-1) * Lx / size(T_grid, 1)
        z = (j-1) * Lz / size(T_grid, 2)

        T_grid[i, j] = sin(2π * x / Lx) * cos(π * z / Lz)
    end

    to_spectral!(T)
end

init_temperature!(T, 2π, 1.0)
```

### Random Perturbations

```julia
using Random

function add_random_perturbation!(field, amplitude=0.01)
    data = get_grid_data(field)

    # Different seed per MPI rank
    Random.seed!(42 + MPI.Comm_rank(MPI.COMM_WORLD))

    data .+= amplitude .* (rand(size(data)...) .- 0.5)

    to_spectral!(field)
end

add_random_perturbation!(T, 0.01)
```

### From Array

```julia
# Set from existing array
T_grid = get_grid_data(T)
T_grid .= my_data_array

to_spectral!(T)
```

## Field I/O

### Saving Fields

```julia
using HDF5

function save_field(field, filename)
    # Ensure in grid space
    to_grid!(field)

    # Gather to root process
    data = get_grid_data(field)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        h5write(filename, field.name, data)
    end
end

save_field(T, "temperature.h5")
```

### Loading Fields

```julia
function load_field!(field, filename)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        data = h5read(filename, field.name)
    else
        data = nothing
    end

    # Broadcast to all ranks
    # ... (distribute according to pencil decomposition)

    to_spectral!(field)
end

load_field!(T, "temperature.h5")
```

## Memory Layout

### Spectral Space

In spectral space, fields are stored as arrays of spectral coefficients:

- **Fourier**: Complex coefficients (or real for RealFourier)
- **Chebyshev**: Coefficients in Chebyshev polynomial basis
- **Legendre**: Coefficients in Legendre polynomial basis

### Grid Space

In grid space, fields are pointwise values on the collocation grid:

- **Fourier**: Uniform grid
- **Chebyshev**: Chebyshev-Gauss-Lobatto points
- **Legendre**: Legendre-Gauss-Lobatto points

### Distributed Layout

Fields use PencilArrays for distribution:

```julia
# Local size on this MPI rank
local_size = size(field.data)

# Global size
global_size = field.data.global_size

# Rank in process mesh
rank = field.distributor.rank
```

## Type Hierarchy

```
AbstractField
├── ScalarField
├── VectorField
│   └── components::Vector{ScalarField}
└── TensorField
    └── components::Array{ScalarField, 2}
```

## Performance Tips

### Minimize Transforms

Transforms are expensive. Group operations:

```julia
# Bad: Multiple transforms
to_grid!(u)
to_grid!(v)
result = u.data .* v.data
to_spectral!(result_field)

# Good: One transform per field
to_grid!(u)
to_grid!(v)
result = u.data .* v.data
result_field.data .= result
result_field.space = :grid
to_spectral!(result_field)
```

### Reuse Work Arrays

```julia
# Create work array once
work = similar(field.data)

# Reuse in loop
for i in 1:nsteps
    work .= compute_something(field)
    field.data .+= dt .* work
end
```

### In-Place Operations

```julia
# Bad: Allocates new arrays
field.data = field.data .+ alpha .* other.data

# Good: In-place operation
field.data .+= alpha .* other.data
```

## See Also

- [Operators](operators.md): Differential operators on fields
- [Domains](domains.md): Spatial domains and bases
- [Problems](problems.md): Setting up PDE problems with fields
