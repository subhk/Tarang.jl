# Fields

Fields represent physical quantities distributed across the computational domain.

Every snippet on this page runs against the small 2D domain set up here:

```julia
using Tarang
using MPI          # for MPI.MAX / MPI.SUM in the reduction snippets

coords  = CartesianCoordinates("x", "z")
dist    = Distributor(coords; dtype=Float64, device=CPU())
x_basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
z_basis = RealFourier(coords["z"]; size=8,  bounds=(0.0, 2π))
bases   = (x_basis, z_basis)
domain  = Domain(dist, bases)
```

## Field Types

### ScalarField

Single-component fields like temperature or pressure.

```julia
# Create scalar field
T = ScalarField(dist, "temperature", (x_basis, z_basis), Float64)
p = ScalarField(dist, "pressure", (x_basis, z_basis), Float64)
```

### VectorField

Multi-component fields like velocity. One component per coordinate of the
coordinate system.

```julia
# Create velocity field
u = VectorField(dist, coords, "u", (x_basis, z_basis), Float64)

# Access components
ux = u.components[1]  # x-component, named "u_x"
uz = u.components[2]  # z-component, named "u_z"
```

### TensorField

Rank-2 tensor fields like stress tensors. The components are a dense `dim × dim`
matrix of `ScalarField`s named after the coordinate pairs (`tau_xx`, `tau_xz`, …);
there is no symmetric-storage variant.

```julia
tau = TensorField(dist, coords, "tau", bases, Float64)

# Convenience form: uses the distributor's coordinate system
grad_u = TensorField(dist, "grad_u", bases)

tau.components[1, 2]        # the xz component (a ScalarField named "tau_xz")
```

## Field Properties

```julia
T.name           # String: field name
T.dist           # Distributor
T.bases          # Tuple of spectral bases
T.dtype          # Data type (Float64, etc.)
T.current_layout # :g (grid) or :c (coefficient)
get_grid_data(T)    # Grid space data (when in :g layout)
get_coeff_data(T)   # Coefficient data (when in :c layout)
```

Basis metadata lives under `basis.meta` (`x_basis.meta.size`, `x_basis.meta.bounds`,
`x_basis.meta.dealias`), and the domain knows both shapes:

```julia
global_shape(domain, :g)   # (16, 8)  — full grid
global_shape(domain, :c)   # (9, 8)   — coefficients (rfft halves the first axis)
local_shape(domain, :g)    # this rank's slab
```

## Data Access

### Grid Space

```julia
# Ensure field is in grid space
ensure_layout!(T, :g)

# Access data
data = get_grid_data(T)

# Modify values (indices are LOCAL to this rank under MPI)
data[3, 2] = 1.0

# Broadcast over the local grid coordinates
x, z = local_grids(dist, x_basis, z_basis)
data .= sin.(x) .* cos.(z')
```

### Coefficient Space

```julia
# Ensure field is in coefficient space
ensure_layout!(T, :c)

# Access spectral coefficients
coeffs = get_coeff_data(T)

# Set specific modes
coeffs[1, 1] = 0.0  # Zero mean
```

## Initialization

### Constant Value

```julia
ensure_layout!(p, :g)
get_grid_data(p) .= 1.0
```

`set!(field, ::Number)` does the same thing and works distributed.

### From Function

`local_grids(dist, bases...)` returns each rank's *local* slice of every axis, so
broadcasting over it is correct in serial and under MPI alike:

```julia
function initialize_field!(field, f)
    ensure_layout!(field, :g)
    x, z = local_grids(field.dist, field.bases...)
    get_grid_data(field) .= f.(x, z')
    return field
end

# Usage
initialize_field!(T, (x, z) -> sin(x) * cos(z))
```

`set!(field, ::Function)` is a shorthand for the same thing, but it is **serial
only** — it builds the global meshgrid and throws a `DimensionMismatch` when the
field is distributed. Use the `local_grids` form above in code that must run under
MPI.

### Random Perturbations

`fill_random!` fills the field in place. With `reproducible=true` (the default) the
same `seed` produces the same global field regardless of how many ranks are used.

```julia
# Small-amplitude noise on top of an existing profile
noise = copy(T)
fill_random!(noise, "g"; seed=42, distribution="normal", scale=0.01)

ensure_layout!(T, :g)
ensure_layout!(noise, :g)
get_grid_data(T) .+= get_grid_data(noise)
```

`distribution` may be `"normal"`, `"standard_normal"`, or `"uniform"`; `fill_random!`
also accepts a `VectorField`, filling each component with a different seed.

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
# In-place operations (preferred); both fields must be in the same layout
ensure_layout!(T, :g)
ensure_layout!(p, :g)

get_grid_data(T) .+= get_grid_data(p)
get_grid_data(T) .*= 2.0

# Scaling by a global norm — compute the norm with a reduction (see below),
# never with a bare maximum() over distributed data
get_grid_data(T) ./= global_max(T)
```

### Reductions

Under MPI `get_grid_data(field)` is a `PencilArray`, and `sum`/`maximum`/`minimum`
on a `PencilArray` are **already collective** — they return the global value. Wrapping
one in another reduction double-reduces (the "global sum" comes out `nprocs ×` too
large). Reduce `parent(...)` — the rank-local storage — and then reduce once:

```julia
# Local operations: parent() keeps the reduction local
local_max = maximum(abs, parent(get_grid_data(T)))
local_sum = sum(parent(get_grid_data(T)))

# Global MPI reductions
reducer = GlobalArrayReducer(dist.comm)
global_maximum = reduce_scalar(reducer, local_max, MPI.MAX)
global_total   = reduce_scalar(reducer, local_sum, MPI.SUM)
```

The ready-made field helpers do exactly this and are usually what you want:

```julia
global_max(T)    # Float64, same value on every rank
global_min(T)
global_sum(T)
global_mean(T)
integrate(T)     # quadrature-weighted domain integral
```

## Layout Management

### Transforms

```julia
# Transform to grid space
ensure_layout!(T, :g)

# Transform to coefficient space
ensure_layout!(T, :c)

# Check current layout
if T.current_layout == :g
    println("Field is in grid space")
end
```

### Efficient Transform Usage

```julia
# Bad: Multiple unnecessary transforms
for i in 1:100
    ensure_layout!(T, :g)
    # ... work ...
    ensure_layout!(T, :c)
end

# Good: Batch operations in same space
ensure_layout!(T, :g)
for i in 1:100
    # ... all grid-space work ...
end
ensure_layout!(T, :c)
```

## VectorField Operations

### Component Access

```julia
# Individual components (one per coordinate; this domain is 2D)
ux = u.components[1]
uz = u.components[2]

# Iterate
for (i, component) in enumerate(u.components)
    println("Component $i: $(component.name)")
end
```

### Vector Magnitude

```julia
function magnitude(u)
    mag = nothing
    for c in u.components
        ensure_layout!(c, :g)
        d = parent(get_grid_data(c))   # rank-local storage
        mag = mag === nothing ? d .^ 2 : mag .+ d .^ 2
    end
    return sqrt.(mag)
end

initialize_field!(u.components[1], (x, z) -> sin(x))
initialize_field!(u.components[2], (x, z) -> cos(z))

maximum(magnitude(u))   # 1.4142… (magnitude returns a plain Array of this rank's |u|)
```

### Divergence Check

`divergence(u)` builds an operator; `evaluate` turns it into a `ScalarField`. For an
incompressible flow the global maximum should be at roundoff level.

```julia
# A divergence-free velocity: u = (sin(x)cos(z), -cos(x)sin(z))
initialize_field!(u.components[1], (x, z) ->  sin(x) * cos(z))
initialize_field!(u.components[2], (x, z) -> -cos(x) * sin(z))

divu = evaluate(divergence(u))
max_div = global_max(divu)   # ~1e-15 — collective, same answer on every rank
```

In *equation strings* the same operator is written `div(...)`, but in Julia code you
must call `divergence` — a bare `div(u)` resolves to `Base.div` (integer division)
and raises a `MethodError`.

## Memory Management

### Pre-allocation

```julia
# Create work array once
ensure_layout!(T, :g)
work = similar(get_grid_data(T))

# Reuse it: explicit forward Euler diffusion step
ν, dt = 0.1, 1e-3
for step in 1:10
    lapT = evaluate(lap(T))
    ensure_layout!(lapT, :g)
    ensure_layout!(T, :g)
    work .= ν .* get_grid_data(lapT)
    get_grid_data(T) .+= dt .* work
end
```

(For real runs use a [timestepper](timesteppers.md); this is only to show the buffer
reuse pattern.)

### In-place Operations

```julia
# Bad: allocates a temporary array, then copies it back
tmp = get_grid_data(T) + get_grid_data(p)
set_grid_data!(T, tmp)

# Good: no allocation
get_grid_data(T) .+= get_grid_data(p)
```

## Parallel Considerations

### Local Data

Each process holds only its portion of the field:

```julia
# Local array size (this rank's slab)
local_size = size(get_grid_data(T))

# Global size
global_size = (x_basis.meta.size, z_basis.meta.size)   # or global_shape(domain, :g)
```

Under MPI the field data is a `PencilArray`, which knows which slice of the global
grid this rank owns (in serial it is a plain `Array` and this call does not apply):

```julia
using PencilArrays
PencilArrays.range_local(get_grid_data(T))   # (1:16, 1:4) on rank 0 at np=2
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
