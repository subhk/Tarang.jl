# Analysis API

Analysis tools for computing diagnostics, managing output, and monitoring simulations in real-time.

## CFL Conditions

### CFL

Computes adaptive timesteps based on the Courant-Friedrichs-Lewy (CFL) stability criterion.

**Constructor**:
```julia
CFL(
    problem::IVP;
    safety::Float64=0.5,
    max_change::Float64=2.0,
    min_change::Float64=0.5,
    max_dt::Float64=Inf,
    min_dt::Float64=0.0
)
```

**Arguments**:
- `problem`: IVP problem
- `safety`: Safety factor (0 < safety < 1)
- `max_change`: Maximum timestep increase factor per step
- `min_change`: Maximum timestep decrease factor per step
- `max_dt`: Maximum allowed timestep
- `min_dt`: Minimum allowed timestep

**Examples**:

```julia
# Standard CFL
cfl = CFL(problem, safety=0.5)

# Conservative settings
cfl = CFL(problem, safety=0.3, max_change=1.2, min_change=0.8)

# With timestep bounds
cfl = CFL(problem, safety=0.5, max_dt=0.01, min_dt=1e-6)
```

**Methods**:

#### add_velocity!

Add velocity field for CFL calculation.

```julia
add_velocity!(cfl, velocity_field)
```

**Example**:

```julia
cfl = CFL(problem)
add_velocity!(cfl, u)  # u is a VectorField

# For multiple velocity scales
add_velocity!(cfl, u)  # Advection velocity
add_velocity!(cfl, c)  # Sound speed (for compressible flow)
```

#### compute_timestep

Calculate adaptive timestep.

```julia
dt = compute_timestep(cfl)
```

**Returns**: Adaptive timestep satisfying CFL condition

**Example**:

```julia
while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
end
```

**CFL Condition**:

For explicit methods:
```math
\Delta t \leq C \frac{\Delta x}{|u|_{max}}
```

where C is the safety factor.

**Properties**:
```julia
cfl.safety          # Safety factor
cfl.max_dt          # Maximum timestep
cfl.min_dt          # Minimum timestep
cfl.current_dt      # Last computed timestep
```

---

## Output Handlers

### NetCDF Output

#### add_file_handler (recommended)

Create a NetCDF output handler and **register it with the solver**, so `run!`
writes it automatically at its own cadence and closes it at the end — no manual
`process!`/`close!` in a step loop (easy to forget, silently drops output).

**Syntax**:
```julia
add_file_handler(
    base_path::String,
    solver::InitialValueSolver,
    vars=<problem namespace>;   # optional; only for STRING-expression tasks (see below)
    sim_dt=nothing,             # write every sim_dt time units
    iter=nothing,               # …or every iter steps
    wall_dt=nothing,            # …or every wall_dt seconds
    max_writes=nothing,         # cap number of writes
    mode="overwrite",           # "overwrite" | "append"
)
```

**Example**:

```julia
# Register handler, add tasks, run — run! handles process!/close! and GPU→CPU
snapshots = add_file_handler("output/snapshots", solver; sim_dt=0.1)
add_task!(snapshots, u; name="u")          # field object
add_task!(snapshots, lap(u); name="lap_u") # operator object (derived field)

# Multiple handlers at different cadences (each auto-registered on the solver)
checkpoints = add_file_handler("output/checkpoints", solver; sim_dt=1.0)
add_task!(checkpoints, u; name="u")

run!(solver; stop_time=20.0, cfl=cfl)      # writes both handlers + closes them
```

A task can be a **field object** (`u`) or an **operator object** built from the
exported operator functions (`lap(u)`, `Gradient(u, coords)`, `curl(u)`, …) — see
[Derived Fields](#Derived-Fields). For reductions use `add_mean_task!` /
`add_extrema_task!`.

!!! note "The `vars` namespace"
    The optional third positional `vars` is a name→field table used only by the
    (limited) string-expression task parser. It defaults to the problem namespace.
    **Prefer field/operator objects over string expressions** — the string parser
    handles only trivial cases. Pass an explicit `Dict("name" => field)` only if
    you rely on string tasks that reference a non-problem field.

**Output files**:
- Each write-set goes in its own subdirectory: `snapshots_s1/…`, `snapshots_s2/…`
- Under MPI, one file per process per set
- Combine with `merge_netcdf_files("output/snapshots")` (see below)

---

#### NetCDFFileHandler

Direct handler type, when you build it without a solver (e.g. a custom step loop)
and pass the `Distributor` explicitly.

**Constructor**:
```julia
NetCDFFileHandler(
    base_path::String,
    dist::Distributor,
    vars;                       # Dict("name" => field) or a vector of fields
    sim_dt=nothing, iter=nothing, wall_dt=nothing,
    max_writes=nothing,
    mode="overwrite",           # "overwrite" | "append"
    parallel="gather",          # "gather" (single file) | per-rank
)
```

**Methods**:
```julia
process!(handler)   # write if the handler's sim_dt/iter/wall_dt cadence is due
close!(handler)     # finalize / stamp metadata
```

**Example** (manual loop — only if not using `run!`):

```julia
handler = NetCDFFileHandler("output", dist, Dict("u" => u, "T" => T); iter=100)

process!(handler)                          # capture initial state (t = 0)
while solver.sim_time < t_end
    step!(solver)
    process!(handler)                      # no-op unless cadence is due
end
close!(handler)
```

#### merge_netcdf_files

Merge per-process / multi-set NetCDF output into a single file.

```julia
merge_netcdf_files(
    "output/snapshots";
    set_number=1,               # which write-set to merge
    output_name="",             # default: derived from base name
    merge_mode=RECONSTRUCT,     # RECONSTRUCT | SIMPLE_CONCAT | DOMAIN_DECOMP
    cleanup=false,              # delete source files after merge
)

# Several handlers at once
batch_merge_netcdf(["output/snapshots", "output/checkpoints"])
```

---

## In-Run Diagnostics

Scalar / time-series diagnostics are computed in a `run!` **callback** — an
`interval => function` pair. The function receives the solver; an `Int` interval
fires every N iterations, a `Float64` interval every N sim-time units. Inside it,
reduce a field's grid data with the MPI-aware helpers `global_max` / `global_min`
/ `global_mean` / `global_sum` (identical value on every rank), or use `integrate`
for a true domain integral.

```julia
times, ke = Float64[], Float64[]

function diagnostics(s)
    ensure_layout!(u, :g)
    g = get_grid_data(u)
    push!(times, s.sim_time)
    push!(ke, 0.5 * global_mean(dist, g .^ 2))     # ⟨½u²⟩, MPI-reduced
    rank == 0 && println("t=$(s.sim_time)  KE=$(ke[end])")
end

run!(solver; stop_time=10.0, cfl=cfl, callbacks=[10 => diagnostics])   # every 10 steps
```

`run!` also prints a built-in progress line if you pass `log_interval=100`.

---

## Derived Fields

To write a derived quantity (vorticity, Laplacian, a gradient component, …), build
it as an **operator object** from the exported operator functions and add it as a
task — `run!` evaluates and writes it like any field.

```julia
handler = add_file_handler("output/derived", solver; sim_dt=0.1)
add_task!(handler, lap(u);              name="lap_u")     # Laplacian (Δ also works)
add_task!(handler, Gradient(u, coords); name="grad_u")    # gradient (vector)
add_task!(handler, curl(velocity);      name="vorticity") # curl of a VectorField
```

Exported builders: `grad`/`Gradient`, `divergence`/`Divergence`, `curl`/`Curl`,
`lap`/`Laplacian` (and the Unicode `∇`, `Δ`), plus `Differentiate(field, coord, order)`
for a single-axis derivative.

!!! warning "Prefer operator objects over string expressions"
    `add_task!` also accepts a string (`add_task!(h, "u*u")`), but the string
    parser handles only the most trivial cases — build operator objects instead.

### Reduction tasks

Reduce a field to a profile or extrema series, written by the same handler:

```julia
add_mean_task!(handler, u; dims=1, name="u_mean_x")   # average over axis 1 → profile
add_extrema_task!(handler, u; name="u_extrema")        # min and max each write
```

---

## In-Memory Analysis

To keep results in memory (no file I/O), attach a `DictionaryHandler` to a
`UnifiedEvaluator` and evaluate it from a callback. Results are stored by name and
read with `handler[name]`.

```julia
ev = UnifiedEvaluator(solver)
diag = add_dictionary_handler(ev; sim_dt=0.5)   # cadence: sim_dt / iter / wall_dt
add_task!(diag, u; name="u")
add_task!(diag, lap(u); name="lap_u")

# Evaluate every step; the handler gates its own write cadence internally.
run!(solver; stop_time=10.0, cfl=cfl,
     callbacks=[1 => s -> evaluate_unified_handlers!(ev, time(), s.sim_time, s.iteration)])

u_now = diag["u"]     # most recent stored evaluation
keys(diag)            # task names
```

---

## Statistical Analysis

### Time averaging

There is no separate averager type — accumulate a running mean in a callback:

```julia
snaps = Vector{Array{Float64}}()
run!(solver; stop_time=t_end, cfl=cfl, callbacks=[0.1 => function (s)
    ensure_layout!(u, :g)
    push!(snaps, copy(get_grid_data(u)))
end])
u_time_mean = sum(snaps) ./ length(snaps)
```

### Spatial averaging

For output, use `add_mean_task!(handler, field; dims=…)`. For an in-line value,
reduce the grid array directly: `global_mean(dist, get_grid_data(field))` for the
whole-domain mean, or `Statistics.mean(get_grid_data(field); dims=…)` over chosen
array dimensions.

---

## Spectral Analysis

`power_spectrum` computes the radially-binned power spectrum of a scalar field that
has at least one Fourier basis. It puts the field in coefficient layout, then bins
|f̂(k)|² by wavenumber magnitude, returning a NamedTuple with `k`, `power`, and
`bin_edges`.

```julia
ps = power_spectrum(u)                  # u::ScalarField with a Fourier axis

using Plots
plot(ps.k, ps.power; xscale=:log10, yscale=:log10, xlabel="k", ylabel="E(k)")
plot!(ps.k, ps.k .^ (-5/3); linestyle=:dash, label="k^(-5/3)")   # Kolmogorov ref

# Options
ps = power_spectrum(u; max_wavenumber=64, radial_average=true)
```

For a velocity `VectorField`, `enstrophy_spectrum(velocity)` returns the enstrophy
spectrum Z(k) = |ω̂(k)|² (vorticity power) with the same NamedTuple shape.

---

## Flow Diagnostics

Flow-property scalars are computed in a callback from the field grid data — use
`get_grid_data` (not `.data`) and the MPI-aware reducers. Example: a Nusselt-like
heat flux ⟨wT⟩ and an RMS-velocity Reynolds number, accumulated over time.

```julia
Nu, Re = Float64[], Float64[]
function flow_diag(s)
    ensure_layout!(w, :g); ensure_layout!(T, :g); ensure_layout!(u, :g)
    wT   = global_mean(dist, get_grid_data(w) .* get_grid_data(T))   # ⟨wT⟩
    urms = sqrt(global_mean(dist, get_grid_data(u) .^ 2))            # RMS speed
    push!(Nu, 1 + wT)
    push!(Re, urms * L / nu)
end
run!(solver; stop_time=t_end, cfl=cfl, callbacks=[10 => flow_diag])
```

Derivative-based quantities (shear, gradient Richardson number, …) build the
derivative as an operator object — `Differentiate(T, coords["z"])`, `Gradient`,
`lap`, … — and add it as a task or evaluate it via the in-memory handler above.

### Probe points

Sample a field at a fixed grid index inside a callback and append to a time series:

```julia
idx = (8, 16)                       # grid index to sample
t_probe, u_probe = Float64[], Float64[]
run!(solver; stop_time=t_end, cfl=cfl, callbacks=[10 => function (s)
    ensure_layout!(u, :g)
    push!(t_probe, s.sim_time)
    push!(u_probe, get_grid_data(u)[idx...])
end])
```

(Under MPI `get_grid_data` is the rank-local slab — guard the sample by which rank
owns `idx`, or write the field with a handler and probe the merged output.)

### Progress monitoring

`run!` prints a built-in progress line every `log_interval` iterations; richer
monitoring goes in a callback (see [In-Run Diagnostics](#In-Run-Diagnostics)):

```julia
run!(solver; stop_time=t_end, cfl=cfl, log_interval=100)
```

---

## Data Post-Processing

### Merging Parallel Output

After simulation with MPI:

```julia
# Merge per-rank / multi-set NetCDF files (merge_netcdf_files is a Tarang export)
merge_netcdf_files(
    "output/snapshots",
    output_name="output/snapshots_merged",
    cleanup=true  # remove individual rank files after merge
)
```

### Loading Data

Tarang writes plain NetCDF, so read it with NetCDF.jl:

```julia
using NetCDF

ncinfo("output/snapshots_merged.nc")                  # list variables / dimensions

u_data = ncread("output/snapshots_merged.nc", "u")
T_data = ncread("output/snapshots_merged.nc", "T")
time   = ncread("output/snapshots_merged.nc", "time")

# Analyze or visualize
using Plots
heatmap(T_data[:, :, end]; title="Temperature at t=$(time[end])")
```

---

## Complete Example

```julia
using Tarang, MPI

MPI.Init()

# ... setup problem and solver ...

# CFL condition
cfl = CFL(problem, safety=0.5, max_dt=0.01)
add_velocity!(cfl, u)

# Output handler — created with the solver, so run! auto-writes + closes it.
# Tasks (not the namespace) are what gets written, so add one per field.
snapshots = add_file_handler("snapshots", solver; sim_dt=0.1)
add_task!(snapshots, u; name="u")
add_task!(snapshots, p; name="p")
add_task!(snapshots, T; name="T")

# Custom diagnostics live in a callback (receives the solver). global_max is
# MPI-reduced, so the printed value is identical on every rank.
function diagnostics(s)
    ensure_layout!(T, :g)
    max_T = global_max(dist, abs.(get_grid_data(T)))
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 &&
        println("t=$(s.sim_time): dt=$(s.dt), max|T|=$max_T")
end

# Time integration — run! drives the loop: CFL-adaptive dt, auto-writes
# `snapshots` every sim_dt, fires `diagnostics` every 1000 steps, closes at end.
run!(solver; stop_time=10.0, cfl=cfl, callbacks=[1000 => diagnostics])

# Merge per-process / multi-set output into a single file
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    merge_netcdf_files("snapshots", output_name="snapshots_merged", cleanup=true)
end

MPI.Finalize()
```

---

## See Also

- [Solvers](solvers.md): Integration methods
- [Fields](fields.md): Field operations for analysis
- [Tutorial: Analysis](../tutorials/analysis_and_output.md): Detailed examples
