# Fields

Fields represent physical quantities distributed across the computational domain.

## Field Types

### ScalarField

Single-component fields like temperature or pressure.

```julia
using Tarang

# Create scalar field
T = ScalarField(dist, "temperature", (x_basis, z_basis), Float64)
p = ScalarField(dist, "pressure", (x_basis, z_basis), Float64)
```

### VectorField

Multi-component fields like velocity.

```julia
# Create velocity field
u = VectorField(dist, coords, "u", (x_basis, z_basis), Float64)

# Access components
ux = u.components[1]  # x-component
uz = u.components[2]  # z-component
```

### TensorField

Rank-2 tensor fields like stress tensors.

```julia
# Stress tensor (symmetric)
tau = TensorField(dist, coords, "tau", bases; symmetric=true)

# Velocity gradient (not symmetric)
grad_u = TensorField(dist, coords, "grad_u", bases; symmetric=false)
```

## Field Properties

```julia
field.name          # String: field name
field.dist          # Distributor
field.bases         # Tuple of spectral bases
field.dtype         # Data type (Float64, etc.)
field.current_layout # :g (grid) or :c (coefficient)
field.data_g        # Grid space data (when in :g layout)
field.data_c        # Coefficient data (when in :c layout)
```

## Data Access

### Grid Space

```julia
# Ensure field is in grid space
Tarang.ensure_layout!(field, :g)

# Access data
data = field.data_g

# Modify values
data[10, 20] = 1.0
data .= sin.(x_grid) .* cos.(z_grid)
```

### Coefficient Space

```julia
# Ensure field is in coefficient space
Tarang.ensure_layout!(field, :c)

# Access spectral coefficients
coeffs = field.data_c

# Set specific modes
coeffs[1, 1] = 0.0  # Zero mean
```

## Initialization

### Constant Value

```julia
Tarang.ensure_layout!(field, :g)
field.data_g .= 1.0
```

### From Function

```julia
function initialize_field!(field, f)
    Tarang.ensure_layout!(field, :g)

    # Get grid coordinates
    x = get_grid(field.bases[1])
    z = get_grid(field.bases[2])

    for i in eachindex(x), j in eachindex(z)
        field.data_g[i, j] = f(x[i], z[j])
    end
end

# Usage
initialize_field!(T, (x, z) -> sin(x) * cos(π*z))
```

### Random Perturbations

```julia
using Random

function add_perturbation!(field, amplitude; seed=42)
    Tarang.ensure_layout!(field, :g)

    Random.seed!(seed + field.dist.rank)
    field.data_g .+= amplitude .* (rand(size(field.data_g)...) .- 0.5)
end

add_perturbation!(T, 0.01)
```

## Field Operations

### Copy

```julia
# Create copy
T2 = copy(T)

# Deep copy
T3 = deepcopy(T)
```

### Arithmetic

```julia
# In-place operations (preferred)
field.data_g .+= other.data_g
field.data_g .*= 2.0

# Scaling
field.data_g ./= maximum(abs.(field.data_g))
```

### Reductions

```julia
# Local operations
local_max = maximum(field.data_g)
local_sum = sum(field.data_g)

# Global MPI reductions
reducer = GlobalArrayReducer(dist.comm)
global_max = reduce_scalar(reducer, local_max, MPI.MAX)
global_sum = reduce_scalar(reducer, local_sum, MPI.SUM)
```

## Layout Management

### Transforms

```julia
# Transform to grid space
Tarang.ensure_layout!(field, :g)

# Transform to coefficient space
Tarang.ensure_layout!(field, :c)

# Check current layout
if field.current_layout == :g
    println("Field is in grid space")
end
```

### Efficient Transform Usage

```julia
# Bad: Multiple unnecessary transforms
for i in 1:100
    Tarang.ensure_layout!(field, :g)
    # ... work ...
    Tarang.ensure_layout!(field, :c)
end

# Good: Batch operations in same space
Tarang.ensure_layout!(field, :g)
for i in 1:100
    # ... all grid-space work ...
end
Tarang.ensure_layout!(field, :c)
```

## VectorField Operations

### Component Access

```julia
u = VectorField(dist, coords, "u", bases, Float64)

# Individual components
ux = u.components[1]
uy = u.components[2]
uz = u.components[3]

# Iterate
for (i, component) in enumerate(u.components)
    println("Component $i: $(component.name)")
end
```

### Vector Magnitude

```julia
function magnitude(u)
    Tarang.ensure_layout!(u.components[1], :g)

    mag = zeros(size(u.components[1].data_g))
    for c in u.components
        Tarang.ensure_layout!(c, :g)
        mag .+= c.data_g.^2
    end

    return sqrt.(mag)
end
```

### Divergence Check

```julia
function check_divergence(u)
    # In spectral space, compute div(u)
    # This should be near zero for incompressible flows
    # ... implementation ...
end
```

## Memory Management

### Pre-allocation

```julia
# Create work arrays once
work = similar(field.data_g)

# Reuse in computation
for step in 1:nsteps
    work .= compute_rhs(field)
    field.data_g .+= dt .* work
end
```

### In-place Operations

```julia
# Avoid allocations
# Bad:
field.data_g = field.data_g + other.data_g

# Good:
field.data_g .+= other.data_g
```

## Parallel Considerations

### Local Data

Each process holds only its portion of the field:

```julia
# Local array size
local_size = size(field.data_g)

# This is smaller than global size
global_size = field.bases[1].size, field.bases[2].size
```

### MPI Communication

Communication happens automatically during:
- Layout transforms
- Global reductions
- Output operations

## See Also

- [Domains](domains.md): Where fields live
- [Operators](operators.md): Operations on fields
- [API: Fields](../api/fields.md): Complete reference
