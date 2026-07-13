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
coords = CartesianCoordinates("x", "z")

# Get a specific coordinate, by name or by position
x = coords["x"]
z = coords[2]

# Number of dimensions
ndim = coords.dim          # 2

# The coordinate names, in axis order
coords.names               # ["x", "z"]

# Iterate over all coordinates
for coord in get_coords(coords)
    println(coord.name)
end
```

A `CartesianCoordinates` is not itself iterable — `for coord in coords` and
`length(coords)` throw a `MethodError`. Use `get_coords(coords)` to get the
`Vector{Coordinate}`, and `coords.dim` for the dimension count.

### Coordinate Properties

```julia
coord = coords["x"]

coord.name         # String: "x"
coord.coordsys     # The parent CartesianCoordinates object, {x,z}
coord.dim          # Int: 1 — a Coordinate is always one-dimensional
coord.curvilinear  # Bool: false for Cartesian
```

## Distributor

The Distributor manages MPI process distribution across the domain.

```julia
# Create distributor — the mesh is chosen for you
dist = Distributor(coords; dtype=Float64, device=CPU())
```

### Parameters

- `coords`: Coordinate system
- `mesh`: Tuple specifying the MPI process grid. Optional; omit it and Tarang picks a
  valid mesh for the number of ranks it is launched with.
- `dtype`: Data type (Float64, Float32)
- `device`: `CPU()` or `GPU()`

### MPI Process Mesh

The product of the mesh dimensions must equal the number of MPI processes; anything else
is an `ArgumentError` ("Mesh size 4 does not match number of processes 1").

The mesh must have **one dimension fewer than the domain**: a 2D domain takes a 1D mesh,
a 3D domain takes a 2D mesh. The FFT needs at least one axis held whole on each rank, so
a mesh that decomposes every axis has nothing left to transform.

```julia
coords_2d = CartesianCoordinates("x", "y")
coords_3d = CartesianCoordinates("x", "y", "z")

# 2D domain on 4 processes: a 1D mesh
dist = Distributor(coords_2d; mesh=(4,))

# 3D domain on 8 processes: a 2D mesh
dist = Distributor(coords_3d; mesh=(2, 4))

# Serial (single process)
dist = Distributor(coords_2d; mesh=(1,))
```

(Each of those lines is only valid when the job is launched with that many ranks.)

Passing a 2D mesh for a 2D domain — `mesh=(2, 2)` on 4 ranks — is a hard error:

> `PencilFFT plan creation failed with 4 MPI processes. Local FFTW fallback would produce
> incorrect results.`

The defaults are the safe choice. Omitting `mesh` gives `(2,)`, `(4,)`, `(8,)` at 2, 4 and
8 ranks for a 2D domain, and `(1, 2)`, `(2, 2)`, `(2, 4)` for a 3D domain.

### Distributor Properties

```julia
dist.coordsys   # Coordinate system, {x,z}
dist.coords     # Tuple of Coordinate objects, (x, z)
dist.dim        # Total number of dimensions
dist.comm       # MPI communicator
dist.rank       # This process's rank
dist.size       # Total number of processes
dist.mesh       # Process mesh tuple — (1,) in serial
dist.dtype      # Float64
```

## Domain Decomposition

### Local vs Global Indices

Tarang decomposes the **trailing** axes and keeps the leading axis whole on every rank.
On a 2D domain the first axis is therefore always local, and only the second is split:

```julia
# 2D domain, 32×32 grid, 4 processes (mesh (4,))
Tarang.local_indices(dist, 1, 32)   # 1:32 on every rank — axis 1 is not decomposed
Tarang.local_indices(dist, 2, 32)   # rank 0: 1:8, rank 1: 9:16, … rank 3: 25:32
```

`local_indices` is internal. In user code, get the local extent from the field itself:

```julia
using PencilArrays

u  = ScalarField(domain, "u")
gd = get_grid_data(u)               # a PencilArray when nprocs > 1

PencilArrays.range_local(gd)        # rank 0 of 4: (1:32, 1:8)
local_shape(domain, :g)             # (32, 8)
global_shape(domain, :g)            # (32, 32)
```

### Pencil Decomposition

Tarang uses pencil (slab) decomposition for efficient parallel FFTs. A 3D 16³ domain on 4
processes gets the auto mesh `(2, 2)`, which splits the two trailing axes and leaves the
first whole, so each rank holds a 16×8×8 pencil:

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

Coordinate names determine derivative operators. A coordinate named `"x"` gives you `∂x`,
one named `"z"` gives you `∂z`, and so on:

```julia
using Tarang

coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2pi), dealias=3/2)
domain = Domain(dist, (xbasis, ybasis))

u = ScalarField(domain, "u")
T = ScalarField(domain, "T")

problem = IVP([u, T])
add_parameters!(problem, nu=0.01)
add_equation!(problem, "∂t(u) - nu*lap(u) = ∂x(T)")   # ∂x — named after coords["x"]
add_equation!(problem, "∂t(T) - nu*lap(T) = 0")

x, y = local_grids(dist, xbasis, ybasis)
ensure_layout!(T, :g)
get_grid_data(T) .= sin.(x) .* cos.(y')
ensure_layout!(T, :c)

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
run!(solver; stop_iteration=5, progress=false)
```

### Grid Access

Grids come from the basis plus the distributor, never from the basis alone — that is what
makes the same code correct in serial and under MPI, since each rank gets only its own
slice.

```julia
# All local grid vectors at once, in axis order
x, y = local_grids(dist, xbasis, ybasis)      # ((16,), (16,)) in serial

# One axis at a time
x = local_grid(xbasis, dist, 1)

# Spectral wavenumbers for a Fourier basis
k = wavenumbers(xbasis)                       # physical k = 2πn/L
```

Broadcast against these to set a field, orienting each vector along its own axis:

```julia
# 2D
ensure_layout!(u, :g)
get_grid_data(u) .= sin.(x) .* cos.(y')
ensure_layout!(u, :c)

# 3D
x, y, z = local_grids(dist, xbasis, ybasis, zbasis)
get_grid_data(v) .= sin.(x) .* cos.(y') .* sin.(reshape(z, 1, 1, :))
```

This is the distributed-safe way to initialize a field, because each rank writes only its
own slab. `set!(field, ::Function)` builds the *global* meshgrid and so is serial-only —
under MPI it throws a `DimensionMismatch`. (`set!(field, ::Number)` and `fill_random!` are
fine distributed.)

## Best Practices

### Choosing Mesh Dimensions

Omit `mesh` unless you have a specific reason to set it. When you do set it, the mesh rank
is fixed by the domain — one less than the domain dimension — and only the split is yours
to choose:

| Domain | Processes | Valid mesh | Notes |
|--------|-----------|------------|-------|
| 2D     | 4         | `(4,)`     | The only shape. `(2, 2)` is an error. |
| 3D     | 4         | `(2, 2)`   | The auto choice; `(1, 4)` and `(4, 1)` also work. |
| 3D     | 8         | `(2, 4)`   | Split the two trailing axes. |

### Memory Considerations

- Each process holds `global_size / num_processes` data
- Communication overhead scales with surface area between processes
- Balance process count with per-process work

## See Also

- [Bases](bases.md): Spectral bases for each coordinate
- [Domains](domains.md): Combining coordinates and bases
- [Parallelism](parallelism.md): MPI details
