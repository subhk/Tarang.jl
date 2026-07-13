# Domains API

A domain represents the spatial discretization of your problem, combining coordinate systems, spectral bases, and MPI distribution.

## Domain

The domain combines multiple spectral bases and manages the spatial discretization.

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

The process mesh belongs to the `Distributor`, not to the `Domain`. Omit `mesh=` and it is
chosen automatically from the number of MPI ranks (in serial it is `(1,)`). If you do pass
`mesh=`, `prod(mesh)` must equal the number of ranks — `Distributor(coords; mesh=(4,))` on one
rank throws `ArgumentError: Mesh size 4 does not match number of processes 1`. See
[Domains under MPI](#Domains-under-MPI) below.

**Examples**:

### 1D Domain

```julia
using Tarang

# Setup coordinates and distributor
coords = CartesianCoordinates("x")
dist = Distributor(coords; dtype=Float64, device=CPU())

# Create basis
x_basis = RealFourier(coords["x"]; size=128, bounds=(0.0, 2π))

# Create domain
domain = Domain(dist, (x_basis,))
```

### 2D Domain

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

# Periodic horizontal, bounded vertical
x_basis = RealFourier(coords["x"]; size=64, bounds=(0.0, 4.0))
z_basis = ChebyshevT(coords["z"]; size=32, bounds=(0.0, 1.0))

domain = Domain(dist, (x_basis, z_basis))
```

### 3D Domain

```julia
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

# All periodic (e.g., turbulence)
x_basis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
y_basis = RealFourier(coords["y"]; size=32, bounds=(0.0, 2π))
z_basis = RealFourier(coords["z"]; size=32, bounds=(0.0, 2π))

domain = Domain(dist, (x_basis, y_basis, z_basis))
```

---

## Properties

### Basic Properties

A `Domain` has exactly three user-facing fields:

```julia
domain.dist             # Distributor: MPI distribution
domain.bases            # Tuple: Spectral bases, sorted by axis
domain.dim              # Int: Number of dimensions
```

The coordinate system is reached through the distributor (`domain.dist.coordsys`) or a single
coordinate by name with `get_coord(domain, "x")`.

### Size Information

Sizes are *functions*, not fields:

```julia
# Global grid shape (grid points)
global_shape(domain, :g)    # (64, 32)

# Global coefficient shape — the first Fourier axis is halved by the rfft
global_shape(domain, :c)    # (33, 32)

# Local shape on this MPI rank (equals the global shape in serial)
local_shape(domain, :g)     # (64, 32)
local_shape(domain, :c)     # (33, 32)

# Grid spacing per axis (minimum spacing for Chebyshev)
grid_spacing(domain)        # [0.0625, 0.00256...]

# Domain volume (product of interval lengths)
volume(domain)              # 4.0

# Coordinate names, in axis order
basis_names(domain)         # ["x", "z"]
```

Bounds and lengths live on the bases, in `basis.meta`:

```julia
bounds  = map(b -> b.meta.bounds, domain.bases)                       # ((0.0, 4.0), (0.0, 1.0))
lengths = map(b -> b.meta.bounds[2] - b.meta.bounds[1], domain.bases) # (4.0, 1.0)
sizes   = map(b -> b.meta.size, domain.bases)                         # (64, 32)
```

---

## Methods

### Grid Access

Grids come from the bases plus the distributor, so the same call is correct in serial and
distributed — it always returns *this rank's* slice.

```julia
# All axes at once: a tuple of local 1D grid vectors
x, z = local_grids(dist, x_basis, z_basis)

# A single axis (third argument is the scale factor, 1 = no upsampling)
x = local_grid(domain.bases[1], domain.dist, 1)
```

### Meshgrid Creation

`create_meshgrid` returns a `Dict` keyed by coordinate name, with one full N-dimensional array
per coordinate:

```julia
g = create_meshgrid(domain; on_device=false)
X, Z = g["x"], g["z"]        # each of size (64, 32)
```

!!! warning "create_meshgrid is global"
    `create_meshgrid` builds the *global* grid arrays on every rank. Under MPI they do not match
    the local slab of a field, so broadcasting them into field data throws `DimensionMismatch`.
    Initialize fields from `local_grids` instead — that is correct in serial and distributed.

That idiom looks like this — correct in serial and at any rank count:

```julia
function init_temperature!(T, domain)
    x, z = local_grids(domain.dist, domain.bases...)
    Lx = domain.bases[1].meta.bounds[2] - domain.bases[1].meta.bounds[1]
    Lz = domain.bases[2].meta.bounds[2] - domain.bases[2].meta.bounds[1]

    ensure_layout!(T, :g)
    get_grid_data(T) .= sin.(2π .* x ./ Lx) .* cos.(π .* z' ./ Lz)
    ensure_layout!(T, :c)

    return T
end
```

### Wavenumber Arrays

Wavenumbers come from a Fourier basis, not from the domain:

```julia
# Physical wavenumbers k = 2πn/L, one entry per grid point
kx = wavenumbers(x_basis)   # length 64 for the size-64 basis above: [0.0, 1.5708, 1.5708, ...]

# Wavenumber magnitude on a 3D Fourier domain
function k_magnitude(domain)
    kx, ky, kz = map(wavenumbers, domain.bases)

    KX = reshape(kx, :, 1, 1)
    KY = reshape(ky, 1, :, 1)
    KZ = reshape(kz, 1, 1, :)

    return @. sqrt(KX^2 + KY^2 + KZ^2)
end
```

`wavenumbers` is only defined for `RealFourier` and `ComplexFourier` bases.

---

## Domain Types and Examples

### Periodic Domains

All directions periodic (e.g., homogeneous turbulence):

```julia
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

bases = (
    RealFourier(coords["x"]; size=32, bounds=(0.0, 2π)),
    RealFourier(coords["y"]; size=32, bounds=(0.0, 2π)),
    RealFourier(coords["z"]; size=32, bounds=(0.0, 2π))
)

domain = Domain(dist, bases)
```

**Use cases**: Turbulence, Taylor-Green vortex, forced isotropic flow

---

### Channel Domains

Periodic in streamwise/spanwise, bounded in wall-normal:

```julia
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

bases = (
    RealFourier(coords["x"]; size=32, bounds=(0.0, 2π)),          # Streamwise
    RealFourier(coords["y"]; size=16, bounds=(0.0, Float64(π))),  # Spanwise
    ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))           # Wall-normal
)

domain = Domain(dist, bases)
```

**Use cases**: Channel flow, pipe flow, boundary layers

`bounds` must be a `Tuple{Float64, Float64}`: `bounds=(0.0, π)` throws a `TypeError` because `π`
is an `Irrational`. Write `(0.0, 2π)` (already a `Float64`) or `(0.0, Float64(π))`.

This axis order (bounded axis last) is the **serial** one. Under MPI the Chebyshev axis must come
*first* — see [Domains under MPI](#Domains-under-MPI).

---

### Convection Domains

Periodic horizontal, bounded vertical:

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

bases = (
    RealFourier(coords["x"]; size=64, bounds=(0.0, 4.0)),  # Horizontal
    ChebyshevT(coords["z"]; size=32, bounds=(0.0, 1.0))    # Vertical
)

domain = Domain(dist, bases)
```

**Use cases**: Rayleigh-Bénard convection, atmospheric convection

---

### Non-Cartesian Domains

!!! note
    `CartesianCoordinates` is the only coordinate system implemented. `SphericalCoordinates` and
    other curvilinear systems are planned but do not exist yet, so there is no spherical-shell or
    polar domain, and no metric/Jacobian machinery behind `grad`, `div`, `curl`.

You can still name Cartesian coordinates anything you like and mix bases freely — but the
operators remain Cartesian; no curvature terms are added:

```julia
# Chebyshev × Fourier, with coordinates named "r" and "theta".
# The metric is still Cartesian: ∇ is (∂r, ∂theta), NOT (∂r, (1/r)∂theta).
coords = CartesianCoordinates("r", "theta")
dist = Distributor(coords; dtype=Float64, device=CPU())

bases = (
    ChebyshevT(coords["r"]; size=32, bounds=(0.0, 1.0)),
    RealFourier(coords["theta"]; size=64, bounds=(0.0, 2π))
)

domain = Domain(dist, bases)
```

---

## Multi-Domain Problems

Some problems require multiple domains (e.g., multi-scale, coarse/fine comparison). Domains are
cheap to build and can share a distributor:

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; dtype=Float64, device=CPU())

# Coarse domain for large scales
domain_coarse = Domain(dist, (
    RealFourier(coords["x"]; size=32, bounds=(0.0, 4.0)),
    ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
))

# Fine domain for small scales
domain_fine = Domain(dist, (
    RealFourier(coords["x"]; size=128, bounds=(0.0, 4.0)),
    ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))
))
```

---

## Domain Utilities

### Integration

Quadrature weights are per-axis vectors from `integration_weights`; the domain integral of a
field is `integrate`:

```julia
w = integration_weights(domain; on_device=false)   # one weight vector per axis
V = volume(domain)                                 # product of interval lengths

# Domain integral of a field (correct in serial and under MPI)
total = integrate(T)
```

### Domain Information

```julia
function print_domain_info(domain)
    println("Domain Information:")
    println("  Dimensions: $(domain.dim)")
    println("  Grid shape: $(global_shape(domain, :g))")
    println("  Coeff shape: $(global_shape(domain, :c))")
    println("  Local grid shape: $(local_shape(domain, :g))")
    println("  Volume: $(volume(domain))")

    for (i, basis) in enumerate(domain.bases)
        println("  Basis $i: $(typeof(basis)) on '$(basis.meta.element_label)'")
        println("    Size: $(basis.meta.size)")
        println("    Bounds: $(basis.meta.bounds)")
        println("    Dealias: $(basis.meta.dealias)")
    end
end
```

### Domain Validation

```julia
function validate_domain(domain)
    dist = domain.dist

    # One basis per axis
    @assert length(domain.bases) == domain.dim

    # The process mesh must cover exactly the ranks in the communicator
    @assert prod(dist.mesh) == dist.size

    # MPI decomposes the TRAILING axes: those must be Fourier
    if dist.size > 1
        ndecomp = length(dist.mesh)
        for basis in domain.bases[(end - ndecomp + 1):end]
            @assert basis isa FourierBasis "decomposed axis '$(basis.meta.element_label)' must be Fourier"
        end
    end

    println("Domain validation passed!")
end
```

### Grid Refinement

```julia
function refine_domain(domain; factor=2)
    bases_fine = map(domain.bases) do basis
        coord = get_coord(domain, basis.meta.element_label)
        if basis isa RealFourier
            RealFourier(coord; size=factor * basis.meta.size,
                        bounds=basis.meta.bounds, dealias=basis.meta.dealias)
        elseif basis isa ChebyshevT
            ChebyshevT(coord; size=factor * basis.meta.size,
                       bounds=basis.meta.bounds, dealias=basis.meta.dealias)
        else
            error("refine_domain: unsupported basis $(typeof(basis))")
        end
    end

    return Domain(domain.dist, Tuple(bases_fine))
end
```

---

## Domains under MPI

Nothing in the domain changes under MPI — the decomposition is entirely the distributor's job —
but two rules are hard constraints.

### 1. The process mesh has one dimension fewer than the domain

PencilArrays decomposes the *trailing* axes and keeps the leading axis local for the FFT.

| domain | `mesh=` | result |
|---|---|---|
| 2D | omitted | `(2,)` at 2 ranks, `(4,)` at 4 ranks — a **1D** mesh |
| 2D | `(nprocs,)` | works — same decomposition as the automatic mesh |
| 2D | `(2, 2)` on 4 ranks | **hard error**: `PencilFFT plan creation failed with 4 MPI processes` |
| 3D | omitted | `(1, 2)` at 2 ranks, `(2, 2)` at 4 ranks — a **2D** mesh |

A mesh whose rank equals the domain dimension decomposes *every* axis and leaves nothing local
for the FFT. Just omit `mesh=` unless you have a specific reason to pin it.

A 16×32 (Chebyshev × Fourier) domain therefore has `local_shape(domain, :g) == (16, 16)` at 2
ranks and `(16, 8)` at 4 ranks: only the trailing Fourier axis is split.

### 2. A Chebyshev axis must come first

A Chebyshev transform needs the whole axis on one rank, and the trailing axes are the decomposed
ones — so the Chebyshev axis has to be the leading one. This reverses the serial convention:

```julia
# Serial:  Domain(dist, (x_fourier, z_chebyshev))
# MPI:     Domain(dist, (z_chebyshev, x_fourier))

coords = CartesianCoordinates("z", "x")          # note the coordinate order too
dist = Distributor(coords; dtype=Float64, device=CPU())
z_basis = ChebyshevT(coords["z"]; size=16, bounds=(0.0, 1.0))
x_basis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))

domain = Domain(dist, (z_basis, x_basis))
```

Getting this wrong is a loud error, not a silent wrong answer — building a domain with the
Chebyshev axis last on more than one rank reports that the decomposed trailing axes are
non-Fourier and tells you to reorder the bases.

---

## Performance Considerations

### Memory Usage

```julia
function estimate_memory(domain)
    bytes_per_point = sizeof(domain.dist.dtype)
    points = prod(global_shape(domain, :g))
    total_gb = bytes_per_point * points / 1e9

    per_rank_gb = total_gb / domain.dist.size

    println("Total memory per field: $(total_gb) GB")
    println("Per rank: $(per_rank_gb) GB")
end
```

This counts one grid-space array per field; a solver also holds coefficient-space arrays
(`global_shape(domain, :c)`) and per-stage timestepper workspace, so budget a few times this.

### Choosing a Decomposition

The mesh rank is fixed by the domain (see above), so the only free choice is how the ranks are
split across the trailing axes. The automatic mesh (`mesh=` omitted) balances that for you, and
in practice it is the right answer. Whatever you choose, check what each rank actually got with
`local_shape(domain, :g)` — a slab that is only one or two points thick along a decomposed axis
spends most of its time in communication.

---

## See Also

- [Coordinates](coordinates.md): Coordinate system setup
- [Bases](bases.md): Spectral basis selection
- [Fields](fields.md): Creating fields on domains
- [Operators](operators.md): Differential operators
- [Parallelism Guide](../pages/parallelism.md): MPI configuration
