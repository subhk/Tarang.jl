# Coordinates API

Coordinates define the dimensional structure of your problem domain. Tarang.jl currently
implements Cartesian coordinates only.

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
coords.names        # Vector{String}: coordinate names, in order
coords.dim          # Int: number of dimensions
coords.coords       # Vector{Coordinate}: the Coordinate objects, in order
```

The exported `coords(coordsys)` function returns the same `Coordinate` objects as a `Tuple`.
(Naming the coordinate system `coords`, as the examples below do, shadows that function; call
it as `Tarang.coords(coords)` if you need both.)

**Methods**:

#### Accessing Coordinates

```julia
coords = CartesianCoordinates("x", "y", "z")

# Get coordinate by name
x_coord = coords["x"]

# Get coordinate by index
x_coord = coords[1]

# Iterate over coordinates (the system itself is not iterable)
for coord in coords.coords
    println(coord.name)
end
```

#### Unit Vectors

`unit_vector_fields(coords, dist)` returns one `VectorField` per coordinate direction, in
coordinate order. Each is a constant field with a single non-zero component.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())

ex, ez = unit_vector_fields(coords, dist)

typeof(ez)                                            # VectorField{Float64, …}
[get_grid_data(comp)[1] for comp in ez.components]    # [0.0, 1.0]
```

To use a unit vector inside an equation string, register it with `add_parameters!` first.
It is used for directional forces (buoyancy) and for the tau lift in a first-order
reduction — both on the **implicit (left-hand) side** of the equation:

```julia
τ_lift(A) = lift(A, derivative_basis(zbasis, 1), -1)

add_parameters!(problem, nu=Prandtl, buoy=Rayleigh*Prandtl, ez=ez, τ_lift=τ_lift,
                grad_u=grad(u) + ez * τ_lift(tau_u1))
add_equation!(problem, "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")
```

See the [Rayleigh–Bénard tutorial](../tutorials/ivp_2d_rbc.md) for the surrounding problem.

---

### Spherical and polar coordinates

Not implemented. `CartesianCoordinates` is the only coordinate system Tarang.jl provides;
`SphericalCoordinates` and `PolarCoordinates` do not exist. Curvilinear geometries are
planned for a future release.

---

## Coordinate Object

Individual coordinate dimension, produced by a coordinate system.

**Properties**:
```julia
coord.name          # String: coordinate name
coord.dim           # Int: dimension of the coordinate itself (always 1)
coord.coordsys      # CoordinateSystem: parent system
coord.curvilinear   # Bool: false for Cartesian
```

**Example**:
```julia
coords = CartesianCoordinates("x", "y", "z")
x = coords["x"]

println(x.name)                 # "x"
println(x.coordsys === coords)  # true
```

A `Coordinate` does not know its own position in the parent system; use the system's order
(`coords.names`) if you need the axis index.

---

## Distributor

The distributor manages MPI process distribution across coordinate dimensions.

**Constructor**:
```julia
Distributor(
    coords::CoordinateSystem;
    comm::MPI.Comm=MPI.COMM_WORLD,
    mesh::Union{Nothing,Tuple{Vararg{Int}}}=nothing,   # nothing = auto
    dtype::Type=Float64,
    device::AbstractArchitecture=CPU()                 # CPU() or GPU()
)
```

**Arguments**:
- `coords`: Coordinate system
- `mesh`: MPI process mesh. Omit it and Tarang picks a valid mesh for the domain
  dimension (see Mesh Configuration below). `prod(mesh)` must equal the number of MPI
  processes.

**Examples**:

```julia
using Tarang, MPI

# Serial (or MPI with an auto-chosen mesh)
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

# Explicit mesh: a 2D domain takes a 1D mesh
dist = Distributor(coords; mesh=(MPI.Comm_size(MPI.COMM_WORLD),), device=CPU())
```

`MPI` is not re-exported by Tarang — `using MPI` yourself whenever a snippet touches
`MPI.Comm_size`, `MPI.Barrier`, or a communicator.

**Properties**:
```julia
dist.coordsys       # CoordinateSystem   (dist.coords is the Tuple of Coordinate objects)
dist.mesh           # Tuple: Process mesh dimensions
dist.comm           # MPI.Comm: MPI communicator
dist.rank           # Int: MPI rank of this process
dist.size           # Int: Total number of processes
dist.dtype          # Type: default element type
```

**Methods**:

#### Process Information

There are no `get_rank`/`get_size` accessors — read the fields directly.

```julia
rank = dist.rank        # process rank
nprocs = dist.size      # total number of processes
is_root = dist.rank == 0
```

#### Domain Decomposition

Sizes come from the domain, not from the distributor:

```julia
global_shape(domain, :g)          # global grid shape, e.g. (32, 32)
local_shape(domain, :g)           # this rank's slab, e.g. (32, 16) at np=2
get_global_size(dist, basis)      # global size of one basis, e.g. 32
```

The grid coordinates this rank owns come from `local_grids`, which is the same call in
serial and under MPI:

```julia
xg, zg = local_grids(dist, xbasis, zbasis)
```

Under MPI (`dist.size > 1`) a field's grid data is a `PencilArray`, and it can tell you the
global index range this rank owns. In serial the data is a plain `Array` and these calls do
not apply (`pencil_local_range` returns `nothing`).

```julia
using PencilArrays
gd = get_grid_data(u)
PencilArrays.range_local(gd)                       # rank 0 of 2: (1:32, 1:16)

Tarang.pencil_local_range(dist, 1, dist.size, 32)  # (mesh dim, #procs, global size) -> 1:16
```

---

## Mesh Configuration

### Choosing Process Mesh

The process mesh determines how data is distributed across MPI processes. PencilFFTs
requires at least one axis to stay local for the FFT, so the mesh must have **one fewer
dimension than the domain**:

| domain | valid mesh | auto mesh at np=4 |
|---|---|---|
| 2D | 1D, `(nprocs,)` | `(4,)` |
| 3D | 2D, `(Px, Py)` with `Px*Py == nprocs` | `(2, 2)` |

A mesh whose rank equals the domain dimension decomposes *every* axis and fails hard:
a 2D domain with `mesh=(2, 2)` at np=4 aborts with
*"PencilFFT plan creation failed with 4 MPI processes"* when the domain is built. A mesh
with a unit factor (`(1, 4)`) is normalized to the equivalent slab mesh `(4,)`.

**General rules**:
1. Omit `mesh` unless you have a reason — the auto mesh is valid by construction
2. Product of mesh dimensions must equal the number of MPI processes
3. For 3D, match the mesh aspect ratio to the domain aspect ratio
4. Use powers of 2 when possible for optimal FFT performance

The mesh entries decompose the **trailing** axes, in order: for a 3D domain, `mesh=(Px, Py)`
splits axis 2 into `Px` pieces and axis 3 into `Py` pieces; axis 1 stays local on every rank
(which is why a Chebyshev axis must come **first** in a distributed mixed Fourier–Chebyshev
domain — see [Parallelism](../pages/parallelism.md)).

### Process Mesh Strategies

#### 2D Problems

Only slab decomposition is possible; the mesh is `(nprocs,)` and it splits the last axis.

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(4,), device=CPU())     # 4 processes
# 32×32 grid -> local grid (32, 8) on every rank
```

#### 3D Problems

```julia
coords = CartesianCoordinates("x", "y", "z")

# Square pencil (isotropic domains)
dist = Distributor(coords; mesh=(2, 2), device=CPU())   # 4 processes
# 16³ grid -> local grid (16, 8, 8)

# Anisotropic pencil (stratified flows)
dist = Distributor(coords; mesh=(4, 2), device=CPU())   # 8 processes
# 16³ grid -> local grid (16, 4, 8)
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
# 3D domain, 2D process mesh (auto-selected: (2, 2) at np=4)
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

# PencilArrays automatically handles pencil rotations
# during FFT operations
```

### Transpose Operations

Pencil decomposition requires transpose operations to switch between orientations.
PencilFFTs performs them inside the distributed FFT; there is no user-facing transpose
call and nothing to configure. (`tarang.toml` accepts `parallelism.TRANSPOSE_LIBRARY` and
`parallelism.GROUP_TRANSPOSES` keys, but no code path reads them — the CPU+MPI path always
uses PencilArrays.)

---

## Usage Examples

### Basic Setup

```julia
using Tarang, MPI

MPI.Init()

# Define coordinate system
coords = CartesianCoordinates("x", "y", "z")

# Create distributor (mesh auto-selected: 2D mesh for a 3D domain)
dist = Distributor(coords; dtype=Float64, device=CPU())

# Verify setup
if dist.rank == 0
    println("Running on $(dist.size) processes")
    println("Process mesh: $(dist.mesh)")
end
```

### Multi-Resolution Studies

Resolution is set by the basis sizes, not by the process mesh — the same distributor
serves every resolution.

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

for N in (32, 64)
    xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2pi), dealias=3/2)
    zbasis = RealFourier(coords["z"]; size=N, bounds=(0.0, 2pi), dealias=3/2)
    domain = Domain(dist, (xbasis, zbasis))
    println("N=$N  global grid $(global_shape(domain, :g))  local grid $(local_shape(domain, :g))")
    # ... build and run the problem at this resolution
end
```

### Custom Coordinate Names

```julia
# Use physics-appropriate names
coords = CartesianCoordinates("streamwise", "spanwise", "wall_normal")
dist = Distributor(coords; dtype=Float64, device=CPU())

# Access by custom names
x = coords["streamwise"]
y = coords["spanwise"]
z = coords["wall_normal"]
```

---

## Advanced Topics

### Custom MPI Communicators

Pass `comm=` to run independent simulations on disjoint groups of ranks. The mesh is
derived from the size of the communicator you pass, not from `COMM_WORLD`.

```julia
using Tarang, MPI
MPI.Init()

rank  = MPI.Comm_rank(MPI.COMM_WORLD)
color = rank % 2                                       # two independent groups
sub   = MPI.Comm_split(MPI.COMM_WORLD, color, rank)

coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; comm=sub, dtype=Float64, device=CPU())

println("world rank $rank -> group $color, rank $(dist.rank) of $(dist.size), mesh $(dist.mesh)")
```

### Load Balancing

Each rank prints the slab it owns (run under MPI; in serial the data is a plain `Array`).
PencilArrays splits the decomposed axis as evenly as it can.

```julia
using PencilArrays, MPI

u  = ScalarField(domain, "u")
gd = get_grid_data(u)

for r in 0:(dist.size - 1)
    if dist.rank == r
        println("rank $r: local shape $(local_shape(domain, :g)), owns $(PencilArrays.range_local(gd))")
    end
    MPI.Barrier(dist.comm)
end
```

At np=2 on a 32×32 grid this prints `rank 0: local shape (32, 16), owns (1:32, 1:16)` and
`rank 1: local shape (32, 16), owns (1:32, 17:32)`.

---

## Performance Tips

1. **Let the mesh default** unless profiling says otherwise — it is always dimensionally valid
2. **Use power-of-2 mesh dimensions** for optimal FFT performance
3. **Match mesh to domain aspect ratio** (3D) for balanced load
4. **Monitor communication overhead** with profiling tools
5. **Consider memory constraints** when choosing mesh

---

## See Also

- [Bases](bases.md): Spectral basis functions for each coordinate
- [Domains](domains.md): Combining coordinates and bases into domains
- [Parallelism Guide](../pages/parallelism.md): Detailed MPI configuration
- [Configuration](../pages/configuration.md): Process mesh tuning
