# Domains

Domains combine spectral bases with the MPI distributor to define the computational space.

## Creating a Domain

```julia
using Tarang

# Setup coordinates and distributor
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

# Define spectral bases
x_basis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
z_basis = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))

# Create domain
domain = Domain(dist, (x_basis, z_basis))
```

Leave `mesh=` off the `Distributor` unless you have a reason to set it: the mesh is chosen
automatically from the number of MPI ranks, and the automatic choice is the one that works
(see [Mesh Selection](#Mesh-Selection) below). Under MPI a Chebyshev axis must come *first*
in the basis tuple; see [Domain Decomposition](#Domain-Decomposition).

## Domain Properties

```julia
domain.dist                 # MPI distributor
domain.bases                # Tuple of spectral bases
domain.dim                  # Number of dimensions -> 2

global_shape(domain, :g)    # Global grid shape        -> (32, 16)
global_shape(domain, :c)    # Global coefficient shape -> (17, 16)  (rfft halves axis 1)
local_shape(domain, :g)     # This process's grid slab -> (32, 16) serially

grid_spacing(domain)        # Per-axis spacing -> [0.19634954, 0.01092620]
volume(domain)              # Domain volume    -> 6.283185307179586
```

Shapes are *functions*, not fields: `Domain` itself only stores `dist`, `bases` and `dim`.

## Quick Domain Creation

Tarang provides convenience constructors for common domain types. Each returns a single
`Domain`; the distributor is available as `domain.dist`.

### Periodic Box

```julia
# 2D doubly-periodic domain on [0, 2π)²
domain = PeriodicDomain(128, 128)

# 3D triply-periodic domain with a custom box size
domain = PeriodicDomain(64, 64, 64; L=(1.0, 1.0, 1.0))

dist = domain.dist   # the distributor it built for you
```

### Channel Domain

```julia
# Periodic in x (Fourier), bounded in z (Chebyshev on [0, Lz])
domain = ChannelDomain(256, 64; Lx=4.0, Lz=1.0)

# 3D: periodic in x and y, bounded in z (Chebyshev, always on [-1, 1])
domain = ChannelDomain3D(64, 64, 32; Lx=4π, Ly=2π)
```

`ChannelDomain3D` fixes its Chebyshev axis to `[-1, 1]`; it accepts an `Lz` keyword but
ignores it. If you need a different wall separation, build the domain explicitly.

Both channel constructors put the Chebyshev axis **last**, so they are **serial-only**:
under MPI they fail at construction with the "reorder your bases" error described in
[Domain Decomposition](#Domain-Decomposition). For a distributed channel, assemble the
domain by hand with the Chebyshev axis first.

### Chebyshev Box

```julia
# Bounded in all directions (Chebyshev on every axis)
domain = ChebyshevDomain(64, 64; bounds=((0.0, 1.0), (0.0, 2.0)))
```

A pure-Chebyshev domain is serial-only as well; on more than one process it errors with
*"MPI parallelization is not supported for pure Chebyshev domains"*. Distributed runs need at
least one Fourier axis to decompose.

## Working with Domains

### Creating Fields

```julia
# Directly from the domain (recommended)
T = ScalarField(domain, "T")
u = VectorField(domain, "u")

# Explicit form, when you want a different dtype or a subset of the bases
T = ScalarField(dist, "T", domain.bases, Float64)
u = VectorField(dist, coords, "u", domain.bases, Float64)
```

### Grid Information

```julia
# Per-axis local grid points
for (i, basis) in enumerate(domain.bases)
    grid = local_grid(basis, dist, 1)   # third argument is the scale factor
    println("Axis $i ($(basis.meta.element_label)): $(length(grid)) local points")
end

# All axes at once — this is the distributed-safe way to get coordinates
x, z = local_grids(dist, x_basis, z_basis)   # sizes (32,) and (16,)

# Fourier wavenumbers of a basis
k = wavenumbers(x_basis)   # 32 entries: [0.0, 1.0, 1.0, 2.0, 2.0, …]
```

`local_grids` returns each process's *local* slice, so the same code is correct in serial
and under MPI.

`wavenumbers` returns physical wavenumbers `k = 2πn/L`, one per grid point (length `N`, in
conjugate pairs) — not the half-spectrum that a `RealFourier` coefficient array stores.

### Domain Dimensions

```julia
# Total number of grid points
total_points = prod(global_shape(domain, :g))

# Number of spectral modes per axis
total_modes = prod(basis.meta.size for basis in domain.bases)

# Domain volume
vol = volume(domain)
```

Basis metadata lives under `basis.meta`: `basis.meta.size`, `basis.meta.bounds`,
`basis.meta.dealias`, `basis.meta.element_label`, `basis.meta.dtype`.

## Multi-Dimensional Layouts

### Grid Space (:g)

Data stored as real values at collocation points:

```julia
ensure_layout!(field, :g)
# get_grid_data(field) contains real grid values
# Shape: (32, 16) for the domain above
```

### Coefficient Space (:c)

Data stored as spectral coefficients:

```julia
ensure_layout!(field, :c)
# get_coeff_data(field) contains spectral coefficients
# Shape: (17, 16) — a RealFourier axis stores only the half-spectrum
```

### Layout Transforms

```julia
# Grid to spectral
ensure_layout!(field, :c)

# Spectral to grid
ensure_layout!(field, :g)

# Check current layout
current_layout = field.current_layout  # :g or :c
```

The round trip is exact to roundoff (measured `4.4e-16` on a 32×16 Fourier–Chebyshev field).

## Domain Decomposition

PencilArrays decomposes the **trailing** dimensions of the domain, so a 2D domain gets a
**1D** mesh and a 3D domain gets a **2D** mesh:

```
2D Domain (32 × 16), pure Fourier, on 2 processes (auto mesh = (2,)):

Process 0: x ∈ 1:32, y ∈ 1:8      # local shape (32, 8)
Process 1: x ∈ 1:32, y ∈ 9:16     # local shape (32, 8)

The same domain on 4 processes (auto mesh = (4,)):

Process 0: x ∈ 1:32, y ∈  1:4     # local shape (32, 4)
Process 1: x ∈ 1:32, y ∈  5:8
Process 2: x ∈ 1:32, y ∈  9:12
Process 3: x ∈ 1:32, y ∈ 13:16
```

Because the decomposed axes are the trailing ones, and a Chebyshev axis cannot be
decomposed (its DCT needs the whole axis on each rank), **a Chebyshev axis must be the first
axis under MPI**:

```julia
# Under MPI: Chebyshev first, Fourier after
coords = CartesianCoordinates("z", "x")
dist   = Distributor(coords; dtype=Float64, device=CPU())
z_basis = ChebyshevT(coords["z"]; size=12, bounds=(0.0, 1.0))
x_basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
domain  = Domain(dist, (z_basis, x_basis))
```

A Chebyshev axis placed last raises a clear error at `Domain` construction rather than
producing a wrong answer:

> MPI mixed Fourier-Chebyshev: the decomposed (trailing) axis/axes [2] are non-Fourier …
> Reorder your bases so the Chebyshev axis comes BEFORE the Fourier axes

In serial the order does not matter, and Fourier-first is the conventional choice.

### Accessing Local Data

```julia
# Local data array (a PencilArray when running on more than one process)
local_data = get_grid_data(field)

# Local array size
local_size = size(local_data)

# Global index range owned by this process along an axis (internal function)
rng = Tarang.local_indices(dist, axis, global_size)   # a UnitRange, e.g. 9:16
start_idx, end_idx = first(rng), last(rng)
```

`local_indices` returns a *range*. Do not write `start_idx, end_idx = Tarang.local_indices(…)`:
Julia would destructure the range element-wise and silently bind `(1, 2)`.

The same information is available straight from the array under MPI:

```julia
using PencilArrays
PencilArrays.range_local(get_grid_data(field))   # e.g. (1:32, 9:16) on rank 1 of 2
parent(get_grid_data(field))                     # the raw local storage
```

## Coordinate Systems

### Cartesian

```julia
coords = CartesianCoordinates("x", "y", "z")
# ∂x, ∂y, ∂z derivatives (equivalently d(u, x), d(u, y), d(u, z))
# grad/∇, div, curl, lap/Δ in standard form
```

`dx`, `dy`, `dz` are **not** derivative operators. The parser does not know them, and it warns
`Unknown variable: dx` / `Unknown function in expression: dx` at parse time. What happens next
depends on which side of the equation the term sits on:

- on the explicit right-hand side, the first timestep aborts with
  `UnrecognizedRHSExpression: dx(u) … Aborting rather than silently dropping the term`;
- on the implicit left-hand side, the term is **silently dropped** — the solver runs and
  returns exactly the answer it would have given had you never written the term.

Take the parse-time warning seriously; write `∂x`.

### Future Support

Spherical and cylindrical coordinates are not implemented — `SphericalCoordinates` and
`CylindricalCoordinates` do not exist. `CartesianCoordinates` is currently the only
coordinate system.

## Performance Considerations

### Mesh Selection

Do not hand-pick a mesh; omit `mesh=` and let the distributor choose. The rule it follows is
forced by the decomposition:

| Domain | Valid mesh | Auto choice on 4 ranks |
|--------|------------|------------------------|
| 2D | 1D, e.g. `(4,)` | `(4,)` |
| 3D | 2D, e.g. `(2, 2)` | `(2, 2)` |

A mesh whose rank equals the domain dimension (a 2D domain with a `(2, 2)` mesh) decomposes
*every* axis, leaves no axis local for the FFT, and fails hard with
`PencilFFT plan creation failed with 4 MPI processes`.

### Resolution Balance

- More modes in directions with:
  - Smaller scales
  - Sharper gradients
  - More active dynamics

### Memory Usage

```julia
# Estimate memory per field
bytes_per_element = sizeof(dist.dtype)          # 8 for Float64
total_elements = prod(global_shape(domain, :g))
memory_per_field = total_elements * bytes_per_element

# Per process
memory_per_process = memory_per_field / dist.size
```

## See Also

- [Coordinates](coordinates.md): Coordinate systems
- [Bases](bases.md): Spectral basis types
- [Fields](fields.md): Creating fields on domains
- [API: Domains](../api/domains.md): Complete reference
