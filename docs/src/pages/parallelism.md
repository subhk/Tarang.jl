# Parallelism

Tarang.jl uses MPI for distributed-memory parallelism.

## MPI Basics

### Initialization

```julia
using MPI
using Tarang

# Always initialize MPI first
MPI.Init()

# Get process info
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)

# Your simulation code here
# ...

# Always finalize MPI
MPI.Finalize()
```

`Distributor` calls `MPI.Init()` for you if you have not called it, so the explicit
`MPI.Init()` is only required when you touch MPI before building the distributor.

### Running with MPI

```bash
# Run with 4 processes
mpiexec -n 4 julia --project simulation.jl

# With thread control
export OMP_NUM_THREADS=1
mpiexec -n 4 julia --project simulation.jl
```

## Process Mesh

The distributor organizes MPI processes into a mesh. **The mesh has one fewer dimension
than the domain**: one axis must stay local on every rank for the FFT to be planned. A mesh
with as many dimensions as the domain decomposes every axis, leaves no local axis, and
PencilFFTs fails to plan.

```julia
# 2D domain -> 1D mesh
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; mesh=(4,))          # 4 processes

# 3D domain -> 2D mesh
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; mesh=(2, 2))        # 4 processes
```

In practice, **omit `mesh=`**. The auto-generated mesh already follows the rule: a 2D domain
gets `(nprocs,)`, a 3D domain gets a balanced 2D mesh (`(2,2)` at 4 processes, `(2,4)` at 8).

```julia
dist = Distributor(coords; dtype=Float64, device=CPU())   # mesh chosen for you
dist.mesh    # (4,) for a 2D domain on 4 ranks; (2, 2) for a 3D domain on 4 ranks
```

### Mesh Guidelines

| Processes | 2D domain (1D mesh) | 3D domain (2D mesh) |
|-----------|---------------------|---------------------|
| 4 | (4,) | (2, 2) |
| 8 | (8,) | (2, 4) |
| 16 | (16,) | (4, 4) |
| 64 | (64,) | (8, 8) |

**Rules:**
- Product of mesh = number of processes
- `length(mesh) == ndims(domain) - 1` — one axis stays local for the FFT
- A square mesh on a 2D domain (e.g. `mesh=(2,2)` with 4 ranks) is a **hard error**:
  `PencilFFT plan creation failed with 4 MPI processes`
- For 3D, keep the mesh close to square to balance transpose volume

## Domain Decomposition

### Pencil Decomposition

Tarang uses PencilArrays/PencilFFTs. The **trailing** axes are decomposed; the leading axis
stays local. A 2D domain is therefore cut into slabs:

```
2D domain (32×32) decomposed across 4 processes:

Full Domain:        Decomposed (mesh = (4,)):
┌─────────────┐     ┌──────┬──────┬──────┬──────┐
│             │     │      │      │      │      │
│    32×32    │     │  P0  │  P1  │  P2  │  P3  │
│             │     │ 32×8 │ 32×8 │ 32×8 │ 32×8 │
└─────────────┘     └──────┴──────┴──────┴──────┘
                       y is split, x stays local
```

A 3D domain gets a 2D mesh: the last two axes are split into pencils, the first stays local.

### Local vs Global

```julia
# Local data on this process
ensure_layout!(u, :g)
local_data = get_grid_data(u)     # a PencilArray when nprocs > 1
local_size = size(local_data)     # e.g. (32, 8) at np=4 on a 32×32 grid

# Global size — bases expose their metadata under `.meta`
global_size = (bx.meta.size, by.meta.size)     # (32, 32)

# This rank's global index range on an axis (a UnitRange, not a start index)
local_range = Tarang.local_indices(dist, 2, by.meta.size)   # rank 2 of 4 -> 17:24

# Same information straight from PencilArrays
using PencilArrays
PencilArrays.range_local(local_data)    # rank 2 of 4 -> (1:32, 17:24)
parent(local_data)                      # the raw local storage array
```

Grid coordinates come from `local_grids`, which returns **this rank's** slice of each axis,
so the same code is correct serially and in parallel:

```julia
xg, yg = local_grids(dist, bx, by)
ensure_layout!(u, :g)
get_grid_data(u) .= sin.(xg) .* cos.(yg')
ensure_layout!(u, :c)
```

`set!(field, ::Function)` is **serial only** — it builds the global meshgrid and broadcasts it
into the local slab, which throws `DimensionMismatch` under MPI. Use the `local_grids` form
above. `set!(field, ::Number)` and `fill_random!` are safe in parallel.

### Restrictions under MPI

Distributed runs are more constrained than serial ones:

1. **A Chebyshev axis must come first.** PencilArrays decomposes the trailing axes, and a
   Chebyshev axis cannot be split (its DCT needs the whole axis on each rank). Put the
   Chebyshev basis before the Fourier bases — and order the coordinates the same way.

   ```julia
   coords = CartesianCoordinates("z", "x")
   dist   = Distributor(coords; dtype=Float64, device=CPU())
   zb     = ChebyshevT(coords["z"]; size=12, bounds=(0.0, 1.0))
   xb     = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
   domain = Domain(dist, (zb, xb))       # Chebyshev FIRST
   ```

   Chebyshev last raises a clear error rather than silently producing a wrong answer.

2. **No Chebyshev derivative on the explicit (RHS) side.** A term that differentiates along
   the non-Fourier axis — including `-u⋅∇(u)` advection on a Chebyshev–Fourier domain —
   errors on the first step, because a rank owns only part of that axis' coefficients. Move
   such terms to the implicit (`L`) side, or run serially. RHS derivatives along **Fourier**
   axes (e.g. `-b*∂x(b)`) are fine.

3. **Per-Fourier-mode tau fields do not work**, since they require the Fourier axis first.
   Use tau fields with no bases (`ScalarField(dist, "tau1", (), Float64)`).

## Communication Patterns

### Automatic Communication

Communication happens automatically during:

1. **Layout transforms**: Grid ↔ Spectral
2. **Global reductions**: `global_sum`, `global_max`, `global_min`, `global_mean`, `integrate`
3. **File I/O**: per-rank writes, plus the manifest/merge step

### Manual Communication

Use the exported helpers — they already perform the collective correctly:

```julia
ensure_layout!(u, :g)
gmax = global_max(u)     # same value on every rank
gmin = global_min(u)
gsum = global_sum(u)
gmean = global_mean(u)
total = integrate(u)     # quadrature-weighted domain integral
```

If you must reduce by hand, reduce the **local storage** with `parent`, then do exactly one
`Allreduce`:

```julia
gd = get_grid_data(u)   # PencilArray under MPI

local_max = maximum(abs, parent(gd))
global_max_val = MPI.Allreduce(local_max, MPI.MAX, MPI.COMM_WORLD)

local_sum = sum(parent(gd))
global_sum_val = MPI.Allreduce(local_sum, MPI.SUM, MPI.COMM_WORLD)
```

!!! warning "Never Allreduce a whole-PencilArray reduction"
    `sum`/`maximum`/`minimum` applied to a `PencilArray` are **already collective** — they
    Allreduce internally and return the global value. Wrapping them in another `MPI.Allreduce`
    double-reduces (a sum comes out `nprocs`× too large). On Apple Silicon, `sum(gd)` on a
    PencilArray does not even run: *"User-defined reduction operators are currently not
    supported on non-Intel architectures"*. Always reduce `parent(gd)`, or use the helpers above.

## Output Strategies

### NetCDF File Handler

Under MPI, `add_file_handler` writes **one file per rank**, each holding that rank's local
slab — there is no gather to rank 0. Passing the handler a solver auto-registers it, so
`run!` processes it every step:

```julia
h = add_file_handler("snapshots", solver; iter=5, max_writes=10)
add_task!(h, u; name="u")
run!(solver; stop_iteration=10, progress=false)
```

At 2 ranks this writes `snapshots/snapshots_s1/snapshots_s1_p0.nc` and
`snapshots/snapshots_s1/snapshots_s1_p1.nc`. The data lives in NetCDF-4 groups, so read it
with the group helpers (on a 32×32 grid at np=2, rank 0's file holds `(nwrites, 32, 16)` —
its slab, not the global array):

```julia
f = "snapshots/snapshots_s1/snapshots_s1_p0.nc"
Tarang.group_variable_names(f, "vars")   # ["u"]
Tarang.group_ncread(f, "vars", "u")      # (nwrites, 32, 16) at np=2
```

**Pros**: scalable, no rank-0 memory limit
**Cons**: post-processing needed to stitch the ranks together

### Virtual Mode

`VirtualFileHandler` writes per-rank files *plus* a rank-0 manifest recording each slab's
global offsets, which `merge_virtual!` uses to reconstruct the global array offline. It is
driven manually (it is not accepted by `run!`'s `outputs=`):

```julia
vfh = VirtualFileHandler(outdir, "virt"; comm=MPI.COMM_WORLD, cadence=5)
add_task!(vfh, u; name="u")

wall0 = time()
solver.stop_iteration = 10
while proceed(solver)
    step!(solver)
    Tarang.process!(vfh, solver, time() - wall0, solver.sim_time, solver.iteration)
end

# Post-processing (rank 0): stitch the per-rank slabs back into the global array
if rank == 0
    merged = merge_virtual!(vfh; set_num=1)   # -> virt_s1_merged.nc, u has the global shape
end
```

**Pros**: scalable parallel I/O, reconstruction metadata included
**Cons**: an extra merge step

## Performance Optimization

### Load Balance

Ensure even distribution:

```julia
# Good: Global size divisible by mesh
N = 256  # 256 / 4 = 64 per process
mesh = (4,)

# Bad: Uneven distribution
N = 250  # Runs, but some ranks get more points than others
```

Uneven splits are legal — a 30-point axis across 4 ranks is spread 7/8/7/8 — but the fullest
rank sets the pace of every transpose.

### Communication Minimization

```julia
# Minimize layout transforms
ensure_layout!(u, :g)
# Do all grid-space operations
# Then transform once
ensure_layout!(u, :c)
```

Each `:g`↔`:c` transform on a distributed field is a global transpose, so batch grid-space
work rather than round-tripping the layout inside a loop.

### FFT Plans

Tarang builds its FFT plans once, when the domain and solver are constructed, and reuses them
for the whole run. There is no user-facing planning-rigor or FFTW-wisdom knob — the planning
cost is a fixed part of startup, not something you tune per run.

## Troubleshooting

### Common Issues

**Deadlock**: All processes must call MPI functions together

```julia
# Bad - only rank 0 calls
if rank == 0
    MPI.Barrier(MPI.COMM_WORLD)
end

# Good - all ranks call
MPI.Barrier(MPI.COMM_WORLD)
if rank == 0
    # Only rank 0 does work
end
```

The same trap applies to the collective reductions above: `global_max(u)` and friends must be
called on **every** rank, never inside an `if rank == 0` block.

**Memory errors**: Each process memory limited

```julia
# Estimate memory per process
memory = total_points * bytes_per_point / nprocs
```

### Debugging

```julia
# Print from specific rank
if rank == 0
    println("Debug info...")
end

# Print from all ranks (with ordering)
for r in 0:nprocs-1
    if rank == r
        println("Rank $r: ", size(get_grid_data(u)))
    end
    MPI.Barrier(MPI.COMM_WORLD)
end
```

## See Also

- [Coordinates](coordinates.md): Distributor setup
- [Domains](domains.md): Domain decomposition
- [Running with MPI](../getting_started/running_with_mpi.md): Getting started
