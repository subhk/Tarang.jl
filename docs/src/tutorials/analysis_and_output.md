# Tutorial: Analysis and Output

This tutorial covers saving simulation data and computing diagnostics in Tarang.jl.

## Setup

Every block on this page runs against the same small periodic problem: a scalar
temperature `T` advected by a velocity `u` on a 16×16 doubly-Fourier domain.

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi), dealias=3/2)
zbasis = RealFourier(coords["z"]; size=16, bounds=(0.0, 2pi), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

T = ScalarField(domain, "T")
u = VectorField(domain, "u")

problem = IVP([T, u])
add_parameters!(problem, kappa=0.05, nu=0.05)
add_equation!(problem, "∂t(T) - kappa*lap(T) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*lap(u) = 0")

# Initial condition: local_grids is correct both in serial and under MPI
xg, zg = local_grids(dist, xbasis, zbasis)
ensure_layout!(T, :g)
get_grid_data(T) .= sin.(xg) .* cos.(zg')
ensure_layout!(T, :c)

ux, uz = u.components
ensure_layout!(ux, :g); get_grid_data(ux) .= 0.5;             ensure_layout!(ux, :c)
ensure_layout!(uz, :g); get_grid_data(uz) .= 0.2 .* sin.(xg); ensure_layout!(uz, :c)

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
```

## File Output

Output goes through a NetCDF file handler. `add_file_handler` has two methods, and
they do **not** behave the same way in `run!`.

### Handler bound to the solver

Pass the **solver**: the handler auto-registers with it, and `run!` processes it at
its own cadence — no manual `process!` anywhere. This is the recommended form.

```julia
handler = add_file_handler("output/snapshots", solver;
                           parallel="gather",   # parallel I/O mode
                           sim_dt=0.002,        # write every 0.002 time units
                           max_writes=100)      # writes per file set

add_task!(handler, T; name="temperature")   # pass the FIELD OBJECT
add_task!(handler, u; name="velocity")      # a VectorField writes all components

run!(solver; stop_iteration=10, progress=false)
```

This writes `output/snapshots_s1/snapshots_s1.nc`: the directory part of the base
path is the output root, the file name part names the file *set*. When `max_writes`
is exceeded the handler rolls over to `output/snapshots_s2/…`. (A base path with no
directory part, `"snapshots"`, uses the name itself as the root, i.e.
`snapshots/snapshots_s1/…` — passing a directory is clearer.)

Tasks must be **field or operator objects**. A string expression
(`add_task!(handler, "u*u")`) is accepted silently and then writes zeros — build the
operator instead: `add_task!(handler, u ⋅ u; name="u2")`.

On a CUDA run the same API stages each unique field/layout to a CPU array once
per write, because NetCDF cannot write a `CuArray` directly. This is a bulk,
synchronizing device-to-host copy; it does not use scalar GPU indexing and it
does not move the solver field itself off the GPU. Grid- and coefficient-layout
tasks are both supported. A completed timestep's algebraic constraints are not
re-solved by the writer, so producing a snapshot is observational and leaves the
live solver state unchanged.

### Handler bound to the distributor

The `dist` method takes a `vars` Dict and does **not** register itself with a solver.
If you use it, hand it to `run!` via `outputs=` (or call `process!` yourself):

```julia
handler_dist = add_file_handler("output/snapshots_dist", dist, Dict("T" => T);
                                parallel="virtual", iter=5, max_writes=100)
add_task!(handler_dist, T; name="temperature")

run!(solver; stop_iteration=20, outputs=[handler_dist], progress=false)
```

Forgetting `outputs=` is a silent no-op: the run completes and no file appears.

### Output Modes

```julia
h_gather  = add_file_handler("output/snap_gather",  solver; parallel="gather")
h_virtual = add_file_handler("output/snap_virtual", solver; parallel="virtual")
```

On the CPU NetCDF path, `parallel` only decides the file layout in the serial case:
`"gather"` with one process writes a single file, `output/snap_gather_s1/snap_gather_s1.nc`.
With more than one process — in **either** mode — every rank writes its own file,
`…_s1/…_s1_p0.nc`, `…_s1_p1.nc`, …, each holding that rank's slab: on two ranks of a
16×16 grid, a field task is `(nwrites, 16, 8)` in each of the two files. The reduction
tasks (`add_mean_task!` and friends) are the exception — they reduce globally, so every
rank writes the same global result.

To reassemble the per-rank files into one dataset, use `scripts/merge_netcdf.jl`.

### Output Frequency

The handler owns its schedule; you do not gate it from the loop. The scheduling
keywords are `sim_dt` (simulation time), `iter` (iterations) and `wall_dt`
(wall-clock seconds); a write happens whenever one of them comes due.

```julia
h1 = add_file_handler("output/snaps_simdt", solver; sim_dt=0.002)  # every 0.002 sim-time
h2 = add_file_handler("output/timeseries",  solver; iter=5)        # every 5 iterations
h3 = add_file_handler("output/restarts",    solver; wall_dt=60.0)  # every 60 wall seconds
```

If you drive the loop yourself instead of using `run!`, call `process!` each step and
let the handler decide. Set a stop criterion on the solver first: a fresh solver has
`stop_iteration = typemax(Int)` and `stop_sim_time = Inf`, so `proceed` is `true`
forever and the loop below never exits. (`run!` sets these for you from its keywords.)

```julia
solver.stop_iteration = 5      # or solver.stop_sim_time = 0.01

while proceed(solver)
    step!(solver)
    process!(handler)   # no-op unless the schedule says "write"
end
```

### Reading the Output Back

The files use NetCDF-4 **groups**, so a bare `ncread(file, "temperature")` does not
find the variable. The groups are `vars`, `time` and `grids`:

```julia
f = "output/snapshots_s1/snapshots_s1.nc"

Tarang.group_variable_names(f, "vars")     # ["temperature", "velocity", ...]
group_ncread(f, "vars", "temperature")     # (nwrites, 16, 16)
group_ncread(f, "vars", "velocity")        # (nwrites, 2, 16, 16) — component axis first
group_ncread(f, "time", "sim_time")        # (nwrites,)
group_ncread(f, "grids", "x")              # (16,)
```

## Analysis Tasks

### Computing Means

`dims` are the dimensions **averaged over**:

```julia
# Average over x, leaving a profile in z
add_mean_task!(handler, T; dims=1, name="T_mean_x")

# Average over z, leaving a profile in x
add_mean_task!(handler, T; dims=2, name="T_mean_z")

# Full spatial average (scalar)
add_mean_task!(handler, T; dims=(1,2), name="T_avg")
```

### Extracting Slices and Profiles

```julia
# Slice at a specific index along a dimension
add_slice_task!(handler, T; dim=2, idx=1,  name="T_bottom")
add_slice_task!(handler, T; dim=2, idx=16, name="T_top")

# Profile: mean over every dimension EXCEPT `dim` (singular — the axis KEPT)
add_profile_task!(handler, T; dim=2, name="T_profile_z")
```

### Custom Analysis

`postprocess` is any function of the task's data array; its return value is written
in place of the field:

```julia
half_sumsq(data) = 0.5 * sum(abs2, data)

add_task!(handler, T; name="T_half_sumsq", postprocess=half_sumsq)
```

!!! warning "postprocess sees the rank-local slab"
    Under MPI a custom `postprocess` receives only this rank's data, so a reduction
    written that way is *per-rank*, not global. The built-in `add_mean_task!`,
    `add_profile_task!`, `add_rms_task!`, `add_variance_task!` and
    `add_extrema_task!` are MPI-aware and reduce globally.

## Global Diagnostics

### CFL Condition

`CFL` is constructed from the **solver** (not the problem), and velocities are
registered as `VectorField`s. Every knob is a keyword with a default:
`initial_dt=0.01`, `cadence=1`, `safety=0.4`, `threshold=0.1`, `max_change=2.0`,
`min_change=0.5`, `max_dt=Inf`.

```julia
cfl = CFL(solver; initial_dt=1e-3, cadence=5, safety=0.5,
          max_change=1.5, min_change=0.5, max_dt=0.01)

add_velocity!(cfl, u)          # VectorField only

# Hand it to run!: the loop sets solver.dt from the CFL condition each step
run!(solver; stop_iteration=30, cfl=cfl, progress=false)
```

`compute_timestep(cfl)` returns the proposed `dt` if you drive the loop yourself
(`solver.dt = compute_timestep(cfl); step!(solver)`). It recomputes only every
`cadence`-th call, and commits a change only when it exceeds `threshold` (relative),
so the implicit LHS factorization is not rebuilt on every tiny drift.

### Flow Statistics

MPI reductions have one trap: a reduction over a whole `PencilArray` (what
`get_grid_data` returns when `nprocs > 1`) is **already collective**. Reduce the
rank-local storage with `parent(...)` and do exactly one `Allreduce`:

```julia
using MPI

reducer = GlobalArrayReducer(dist.comm)

ensure_layout!(T, :g)
Td = parent(get_grid_data(T))                            # rank-local storage

T_max = reduce_scalar(reducer, maximum(abs, Td), MPI.MAX)
T_l2  = sqrt(reduce_scalar(reducer, sum(abs2, Td), MPI.SUM))
```

The bundled helpers already do exactly that, and are usually what you want:

```julia
global_max(T)        # Float64, identical on every rank
global_sum(T)
integrate(T)         # quadrature-weighted domain integral

function total_energy(field)
    ensure_layout!(field, :g)
    return 0.5 * global_sum(field.dist, abs2.(parent(get_grid_data(field))))
end
```

### Reynolds Number

`reynolds_number(u, ν, L)` is built in. It is based on the **maximum** velocity
magnitude, `Re = max|u| · L / ν`, and reduces across ranks:

```julia
Re = reynolds_number(u, 0.05, 2pi)
```

### Nusselt Number

For thermal convection there is no built-in, so reduce explicitly:

```julia
function compute_nusselt(T::ScalarField, uz::ScalarField, L, kappa, dT)
    ensure_layout!(T, :g)
    ensure_layout!(uz, :g)
    Td  = parent(get_grid_data(T))
    uzd = parent(get_grid_data(uz))

    # Convective heat flux ⟨T·u_z⟩, averaged over the global grid
    r    = GlobalArrayReducer(T.dist.comm)
    npts = reduce_scalar(r, Float64(length(Td)), MPI.SUM)
    flux = reduce_scalar(r, sum(Td .* uzd), MPI.SUM) / npts

    # Nusselt = total flux / conductive flux, with dT the temperature difference
    return 1.0 + flux * L / (kappa * dT)
end

Nu = compute_nusselt(T, u.components[2], 2pi, 0.05, 1.0)
```

## Energy Spectra

Use the built-in `power_spectrum` — no need to hand-roll wavenumber binning. It is
the radially-binned shell spectrum of a `ScalarField` whose axes are **all Fourier**;
a Chebyshev axis is not supported and throws
`Unsupported field dimensions: 2 with 1 Fourier bases`. It returns a NamedTuple
`(k, power, bin_counts, bin_edges)`, where `k` holds the bin centres in physical
wavenumber.

```julia
ps = power_spectrum(T)                  # T::ScalarField, every axis Fourier
ps.k                                    # bin centres: [0.5, 1.5, 2.5, ...]
ps.power                                # binned power
ps.bin_counts                           # modes per bin
```

There is no `power_spectrum(::VectorField)` method — for a velocity field use the
vector spectra, which return the same NamedTuple:

```julia
es = energy_spectrum(u)                 # → (k, power, bin_counts, bin_edges)
en = enstrophy_spectrum(u)              # vorticity spectrum, same shape
```

Watch the return types of the related diagnostics: `kinetic_energy` returns the
*field* ½ρ|u|², not a number.

```julia
ke_field = kinetic_energy(u)            # ScalarField
ke_total = total_kinetic_energy(u)      # Float64
```

See the [Analysis API](../api/analysis.md#Spectral-Analysis) for options
(`max_wavenumber`, `radial_average`, binning).

## Time Series

### Storing Scalar Time Series

```julia
using DelimitedFiles

# Initialize storage
times = Float64[]
energies = Float64[]
nusselts = Float64[]

# Record diagnostics via a run! callback (Int interval = every N iterations;
# use a Float interval for every-Δt of sim-time).
# total_energy / compute_nusselt are the helpers defined under Global Diagnostics.
run!(solver; stop_iteration=40, progress=false,
     callbacks=[2 => function (s)
         push!(times, s.sim_time)
         push!(energies, total_energy(T))
         push!(nusselts, compute_nusselt(T, u.components[2], 2pi, 0.05, 1.0))
     end])

# Save to file (plain vectors → any format; here a CSV via the stdlib)
if dist.rank == 0
    writedlm("diagnostics.csv", [times energies nusselts], ',')
end
```

## Visualization Integration

Tarang depends on no plotting package, so it cannot draw a figure for you. It *does*
export `plot_1d`, `plot_2d`, `plot_vector_field` and `plot_streamlines`, but these
**render nothing**: `plot_2d(T)` logs the data ranges and hands back a `PlotData`
struct (`x`, `y`, `data`, labels) for you to feed to a real backend.

For an actual plot, take the grid data — a plain array — and give it to whatever
package you already use. Convert with `Array` first: under MPI `get_grid_data` returns
this rank's `PencilArray` slab, and a plot of it is a plot of one slab only.

```julia
ensure_layout!(T, :g)
data = Array(get_grid_data(T))    # dense (16, 16) Matrix{Float64}, ready to plot
```

`data'` is then the usual (x horizontal, z vertical) orientation for a `heatmap`.

## Checkpointing

Tarang has no built-in checkpoint type — write a small helper over the evolving
state fields (`solver.state`, each a `ScalarField` with a `.name`) and plain
NetCDF. Grid space is exact and real-valued, so it round-trips losslessly.

### Saving State

```julia
using NetCDF

function save_checkpoint(solver, path)
    isfile(path) && rm(path)
    for f in solver.state
        ensure_layout!(f, :g)
        g = get_grid_data(f)
        dimspec = collect(Iterators.flatten(
            ("$(f.name)_d$i" => s for (i, s) in enumerate(size(g)))))
        nccreate(path, f.name, dimspec...; t=NC_DOUBLE)   # NC_DOUBLE: keep Float64
        ncwrite(g, path, f.name)
    end
    ncputatt(path, "Global", Dict("sim_time" => solver.sim_time,
                                  "iteration" => solver.iteration, "dt" => solver.dt))
    return path
end
```

### Loading State

```julia
function load_checkpoint!(solver, path)
    for f in solver.state
        ensure_layout!(f, :g)                 # ensure the grid buffer exists / layout is :g
        get_grid_data(f) .= ncread(path, f.name)
    end
    solver.sim_time  = ncgetatt(path, "Global", "sim_time")
    solver.iteration = Int(ncgetatt(path, "Global", "iteration"))
    solver.dt        = ncgetatt(path, "Global", "dt")
    return solver
end
```

Use `solver.state` (the integrator's live fields), **not** the problem-variable
handles. For a vector variable the state holds its components, named `u_x`, `u_z`, …
Restart then continues from `run!` as usual, and reproduces the uninterrupted
trajectory to the bit:

```julia
save_checkpoint(solver, "chk.nc")
# … later, in a fresh session with the same problem/solver built …
load_checkpoint!(solver, "chk.nc")
run!(solver; stop_iteration=solver.iteration + 20, cfl=cfl, progress=false)
```

!!! warning "The state and your field handles are the same object only until the first step"
    On a freshly built solver `solver.state[1] === T`. After the first step they are
    distinct objects holding equal data: the stepper rebinds `solver.state` to its own
    history buffer and then copies *from* the state back into the problem variables. The
    sync is **one-way**. So once a run has started, `T` is a read-only view for
    diagnostics — writing into it (to perturb, reset, or restore a field) is silently
    discarded on the next step. Write to `solver.state` instead, which is what the
    `load_checkpoint!` above does.

!!! note "MPI"
    `get_grid_data` is the rank-local slab, so this helper is per-rank/serial.
    Under MPI either write one file per rank (include the rank in `path`), or
    gather to rank 0 with `gather_array(f.dist, get_grid_data(f))` before writing
    and scatter on load.

## Complete Example

```julia
using Tarang
using MPI
using Printf

MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# Setup
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi), dealias=3/2)
zbasis = RealFourier(coords["z"]; size=16, bounds=(0.0, 2pi), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

T = ScalarField(domain, "T")
u = VectorField(domain, "u")

problem = IVP([T, u])
add_parameters!(problem, kappa=0.05, nu=0.05)
add_equation!(problem, "∂t(T) - kappa*lap(T) = -u⋅∇(T)")
add_equation!(problem, "∂t(u) - nu*lap(u) = 0")

xg, zg = local_grids(dist, xbasis, zbasis)
ensure_layout!(T, :g); get_grid_data(T) .= sin.(xg) .* cos.(zg'); ensure_layout!(T, :c)
ux, uz = u.components
ensure_layout!(ux, :g); get_grid_data(ux) .= 0.5;             ensure_layout!(ux, :c)
ensure_layout!(uz, :g); get_grid_data(uz) .= 0.2 .* sin.(xg); ensure_layout!(uz, :c)

# Create solver
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# CFL — takes the SOLVER, not the problem
cfl = CFL(solver; initial_dt=1e-3, safety=0.5, max_dt=0.01)
add_velocity!(cfl, u)

# Output handler — pass the SOLVER so the handler auto-registers and `run!`
# processes it at its `sim_dt` cadence (no manual `process!` in the loop). No
# `vars` Dict needed: T is a problem variable, already in the solver namespace.
handler = add_file_handler("output/snapshots", solver; parallel="gather", sim_dt=0.1, max_writes=100)
add_task!(handler, T; name="temperature")            # field snapshot
add_mean_task!(handler, T; dims=1, name="T_mean")    # x-averaged profile

# Diagnostics storage (time series accumulated in a callback)
times, max_T = Float64[], Float64[]

# run! drives the whole loop: CFL-adaptive dt, auto-writes the registered handler,
# runs callbacks, and closes the handler at the end. A Float callback interval
# (0.1) fires every 0.1 sim-time units; an Int interval fires every N iterations.
# global_max is MPI-reduced, so the value is identical on every rank.
run!(solver; stop_time=1.0, cfl=cfl, progress=false,
     callbacks=[0.1 => function (s)
         push!(times, s.sim_time)
         push!(max_T, global_max(T))
         rank == 0 && @printf("t = %.3f, max|T| = %.6e\n", s.sim_time, max_T[end])
     end])

MPI.Finalize()
```

## See Also

- [Analysis API](../api/analysis.md): Full API reference
- [I/O API](../api/io.md): NetCDF output documentation
- [Parallelism](../pages/parallelism.md): Parallel I/O configuration
