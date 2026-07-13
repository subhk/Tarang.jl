# Analysis API

Analysis tools for computing diagnostics, managing output, and monitoring simulations in real-time.

## CFL Conditions

### CFL

Computes adaptive timesteps based on the Courant-Friedrichs-Lewy (CFL) stability criterion.

The controller wraps an **`InitialValueSolver`** — build the solver first, then the CFL, then hand
the CFL to `run!`, which calls `compute_timestep` for you before every step.

**Constructor**:
```julia
CFL(
    solver::InitialValueSolver;
    initial_dt::Float64=0.01,
    cadence::Int=1,
    safety::Float64=0.4,
    threshold::Float64=0.1,
    max_change::Float64=2.0,
    min_change::Float64=0.5,
    max_dt::Float64=Inf
)
```

**Arguments**:
- `solver`: the `InitialValueSolver` whose `dt` will be driven
- `initial_dt`: timestep used until the first CFL computation (and whenever no velocity is registered)
- `cadence`: recompute the timestep every `cadence` calls; in between, the current `dt` is reused
- `safety`: Safety factor (0 < safety < 1)
- `threshold`: Relative-change hysteresis — a proposed `dt` within `threshold` of the current one is
  ignored, so the implicit LHS factorization is not needlessly rebuilt. Set `0.0` to commit every change.
- `max_change`: Maximum timestep increase factor per step
- `min_change`: Maximum timestep decrease factor per step (a *ratio*, not a floor on `dt`)
- `max_dt`: Maximum allowed timestep

**Examples**:

```julia
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# Standard CFL
cfl = CFL(solver; initial_dt=1e-3, safety=0.4)

# Conservative settings
cfl = CFL(solver; initial_dt=1e-3, safety=0.3, max_change=1.2, min_change=0.8)

# Bounded timestep, recomputed every 5 steps
cfl = CFL(solver; initial_dt=1e-3, safety=0.4, max_dt=0.01, cadence=5)
```

**Methods**:

#### add_velocity!

Register a velocity field for the CFL calculation. The argument must be a **`VectorField`** — its
components are the advecting velocities, each divided by the grid spacing along its own axis.

```julia
add_velocity!(cfl, u)   # u is a VectorField
```

Registering several `VectorField`s is supported; the controller takes the most restrictive of them:

```julia
add_velocity!(cfl, u)        # advection velocity
add_velocity!(cfl, u_wave)   # a second velocity scale — also a VectorField
```

#### compute_timestep

Calculate the adaptive timestep. `run!` calls this automatically when you pass `cfl=`; call it
yourself only if you drive the loop by hand.

```julia
dt = compute_timestep(cfl)
```

**Returns**: Adaptive timestep satisfying the CFL condition, after `max_dt`, `max_change`/`min_change`
and the `threshold` hysteresis have been applied.

**Example** — the canonical form, letting `run!` drive the loop:

```julia
run!(solver; stop_time=t_end, cfl=cfl)   # solver.dt is updated every step
```

and the manual equivalent. Note that `proceed` reads its stop criteria off the solver, and a freshly
built solver has none (`stop_sim_time = Inf`, `stop_iteration = typemax(Int)`) — `run!` sets them from
its keywords, so a hand-written loop has to set them itself or it never terminates:

```julia
solver.stop_sim_time = t_end            # otherwise proceed(solver) is always true
while proceed(solver)
    solver.dt = compute_timestep(cfl)
    step!(solver)
end
```

**CFL Condition**:

For explicit methods:
```math
\Delta t \leq C \left( \max \sum_i \frac{|u_i|}{\Delta x_i} \right)^{-1}
```

where C is the safety factor.

**Properties**:
```julia
cfl.safety          # Safety factor
cfl.max_dt          # Maximum timestep
cfl.max_change      # Max increase factor per step
cfl.min_change      # Max decrease factor per step
cfl.cadence         # Recompute every N calls
cfl.threshold       # Relative-change hysteresis band
cfl.current_dt      # Last committed timestep
cfl.velocities      # Registered VectorFields
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
    max_writes=nothing,         # writes per file set; a new set is started when it is reached
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
- Files are written under an *output root*, which is `dirname(base_path)` — or `base_path` itself if
  you pass a bare name. So `"output/snapshots"` writes into `output/`, while `"snapshots"` creates a
  directory `snapshots/`.
- Each write-set goes in its own subdirectory of that root: `snapshots_s1/`, `snapshots_s2/`, …
  A new set is started once `max_writes` is reached.
- Serial: one file per set — `snapshots_s1/snapshots_s1.nc`
- Under MPI: one file per process per set — `snapshots_s1/snapshots_s1_p0.nc`, `…_p1.nc`, …
  Combine them with `merge_netcdf_files` (see below).

**Reading the output**: the files are NetCDF-4 and store everything in **groups** (`vars`, `time`,
`grids`), so a top-level `NetCDF.ncread(file, "u")` will not find the variable — use the group
readers (see [Loading Data](#Loading-Data)).

---

#### NetCDFFileHandler

Direct handler type, when you build it without `add_file_handler` (e.g. a custom step loop)
and pass the `Distributor` explicitly.

**Constructor**:
```julia
NetCDFFileHandler(
    base_path::String,
    dist::Distributor,
    vars::Dict{String,Any};     # name → field; the namespace for string-expression tasks
    solver=nothing,             # bind the solver — see the warning below
    sim_dt=nothing, iter=nothing, wall_dt=nothing,
    max_writes=nothing,
    mode="overwrite",           # "overwrite" | "append"
    parallel="gather",          # only "gather" is implemented — see below
)
```

`vars` must be a `Dict` — a bare vector of fields is a `MethodError` (`cannot convert
Vector{Operand} to Dict{String,Any}`). Only `"gather"` is a working `parallel` mode: in serial it
writes one file, under MPI one file per rank. Any other value is accepted silently and merely forces
the per-rank file naming (`manual_s1/manual_s1_p0.nc`) even in serial — there is no MPI-IO or
virtual-file write path behind it.

**Methods**:
```julia
process!(handler)   # write if the handler's sim_dt/iter/wall_dt cadence is due
close!(handler)     # finalize / stamp metadata
```

!!! warning "Bind the solver, or the cadence and the timestamps are meaningless"
    `process!(handler)` takes the iteration and sim-time it stamps from `handler.solver`. An
    unbound handler reports `iteration = 0` and `sim_time = 0.0` on **every** call — so an `iter=`
    cadence fires every time and every record carries `t = 0`. Either pass `solver=solver` to the
    constructor, or call `process!(handler, solver)`. `add_file_handler(path, solver; …)` does this
    for you.

**Example** (manual loop — only if not using `run!`):

```julia
handler = NetCDFFileHandler("output/manual", dist, Dict("u" => u, "T" => T);
                            solver=solver, iter=5)
add_task!(handler, T; name="T")

solver.stop_iteration = 10                 # proceed() needs a stop criterion (run! sets it for you)
process!(handler)                          # capture initial state (t = 0)
while proceed(solver)
    step!(solver)
    process!(handler)                      # no-op unless cadence is due
end
close!(handler)
```

This writes `output/manual_s1/manual_s1.nc` holding three records — the initial state and iterations
5 and 10.

#### merge_netcdf_files

Reconstruct the **per-process** files written under MPI into one global file. Serial output is
already a single file, so there is nothing to merge.

The merger resolves paths relative to the **current working directory** and matches on the bare
handler name, so run it from the handler's *output root* — the directory that directly contains
`<name>_s<N>/`. It returns `true` on success, `false` if it found no per-process files.

```julia
# Handler created as add_file_handler("output/snapshots", solver; …)
#   → files at  output/snapshots_s1/snapshots_s1_p0.nc, …_p1.nc
cd("output") do
    merge_netcdf_files(
        "snapshots";
        set_number=1,               # which write-set to merge
        merge_mode=RECONSTRUCT,     # RECONSTRUCT | SIMPLE_CONCAT | DOMAIN_DECOMP
        cleanup=false,              # delete the per-rank files after merge
    )                               # → snapshots_s1/snapshots_s1.nc

    # Several handlers at once
    batch_merge_netcdf(["snapshots", "checkpoints"])
end
```

The merged file keeps the same group layout as the per-rank files — read it with the group readers
below, not with a bare `ncread`.

---

## In-Run Diagnostics

Scalar / time-series diagnostics are computed in a `run!` **callback** — an
`interval => function` pair. The function receives the solver; an `Int` interval
fires every N iterations, a `Float64` interval every N sim-time units. Inside it,
reduce a field's grid data with the MPI-aware helpers `global_max` / `global_min`
/ `global_mean` / `global_sum` (identical value on every rank), or use `integrate`
for a true domain integral.

```julia
rank = MPI.Initialized() ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
times, ke = Float64[], Float64[]

function diagnostics(s)
    ensure_layout!(T, :g)
    g = get_grid_data(T)
    push!(times, s.sim_time)
    push!(ke, 0.5 * global_mean(dist, g .^ 2))     # ⟨½T²⟩, MPI-reduced
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
add_task!(handler, lap(T);              name="lap_T")     # Laplacian (Δ also works)
add_task!(handler, Gradient(T, coords); name="grad_T")    # gradient (vector)
add_task!(handler, curl(u);             name="vorticity") # curl of a VectorField
add_task!(handler, Differentiate(T, coords["x"], 1); name="dTdx")   # single-axis derivative
```

Exported builders: `grad`/`Gradient`, `divergence`/`Divergence`, `curl`/`Curl`,
`lap`/`Laplacian` (and the Unicode `∇`, `Δ`), plus `Differentiate(field, coord, order)`
for a single-axis derivative.

!!! warning "Prefer operator objects over string expressions"
    `add_task!` also accepts a string (`add_task!(h, "u*u")`), but the string parser handles only
    the most trivial cases and fails **silently** — a string task typically lands in the file as a
    length-1 array of zeros instead of the field. Build operator objects instead.

### Reduction tasks

Reduce a field to a profile or extrema series, written by the same handler:

```julia
add_mean_task!(handler, T; dims=1, name="T_mean_x")   # average over axis 1 → profile
add_extrema_task!(handler, T; name="T_extrema")       # writes T_extrema_min and T_extrema_max
```

`dims` accepts axis numbers (`1`, `(1, 2)`) or coordinate names (`(:x, :y)`); omit it for a
whole-domain mean.

---

## In-Memory Analysis

To keep results in memory (no file I/O), attach a `DictionaryHandler` to a
`UnifiedEvaluator` and evaluate it from a callback. Results are stored by name and
read with `handler[name]`.

```julia
ev = UnifiedEvaluator(solver)
diag = add_dictionary_handler(ev; sim_dt=0.5)   # cadence: sim_dt / cadence / wall_dt
add_task!(diag, T; name="T")
add_task!(diag, lap(T); name="lap_T")

# Evaluate every step; the handler gates its own write cadence internally.
run!(solver; stop_time=10.0, cfl=cfl,
     callbacks=[1 => s -> evaluate_unified_handlers!(ev, time(), s.sim_time, s.iteration)])

T_now = diag["T"]      # most recent stored evaluation, a plain Array
keys(diag)             # task names that have been evaluated at least once
haskey(diag, "lap_T")
```

The handler stores only the **latest** evaluation of each task, so build a time series by pushing
into your own vector from a callback (next section).

---

## Statistical Analysis

### Time averaging

There is no separate averager type — accumulate a running mean in a callback:

```julia
snaps = Vector{Array{Float64}}()
run!(solver; stop_time=t_end, cfl=cfl, callbacks=[0.1 => function (s)
    ensure_layout!(T, :g)
    push!(snaps, copy(get_grid_data(T)))
end])
T_time_mean = sum(snaps) ./ length(snaps)
```

### Spatial averaging

For output, use `add_mean_task!(handler, field; dims=…)`. For an in-line value,
reduce the grid array directly: `global_mean(dist, get_grid_data(field))` for the
whole-domain mean, or `Statistics.mean(get_grid_data(field); dims=…)` over chosen
array dimensions.

---

## Spectral Analysis

`power_spectrum` computes the radially-binned power spectrum of a scalar field. It puts the field in
coefficient layout, then bins |f̂(k)|² by wavenumber magnitude, returning a NamedTuple with `k` (bin
centres, physical wavenumber), `power`, `bin_counts` and `bin_edges`.

**Every axis of the field must be a Fourier basis.** On a mixed Fourier×Chebyshev field it throws
`Unsupported field dimensions: 2 with 1 Fourier bases`.

```julia
ps = power_spectrum(T)                  # T::ScalarField on an all-Fourier domain
ps.k                                    # bin centres
ps.power                                # binned power
ps.bin_counts                           # modes per bin

# Options
ps = power_spectrum(T; max_wavenumber=8, radial_average=true)
```

Pass `radial_average=false` to get the unbinned spectrum as a `Dict{Tuple,Float64}` keyed by mode
index, and `binning=LogBinning(bins_per_decade=8)` (or `LinearBinning()`, `CustomBinning([…])`) to
change the bin layout. Plot `ps.power` against `ps.k` on log-log axes with your plotting package of
choice to compare against a `k^(-5/3)` reference.

For a velocity `VectorField`, `energy_spectrum(velocity)` and `enstrophy_spectrum(velocity)` return
E(k) and Z(k) = |ω̂(k)|² with the same NamedTuple shape.

---

## Flow Diagnostics

Flow-property scalars are computed in a callback from the field grid data — use
`get_grid_data` (not `.data`) and the MPI-aware reducers. Example: a Nusselt-like
heat flux ⟨wT⟩ and an RMS-velocity Reynolds number, accumulated over time.

```julia
w = u.components[end]                                                # vertical velocity
Nu, Re = Float64[], Float64[]
function flow_diag(s)
    ensure_layout!(w, :g); ensure_layout!(T, :g)
    wT   = global_mean(dist, get_grid_data(w) .* get_grid_data(T))   # ⟨wT⟩
    urms = sqrt(global_mean(dist, get_grid_data(w) .^ 2))            # RMS speed
    push!(Nu, 1 + wT)
    push!(Re, urms * L / nu)
end
run!(solver; stop_time=t_end, cfl=cfl, callbacks=[10 => flow_diag])
```

Derivative-based quantities (shear, gradient Richardson number, …) build the
derivative as an operator object — `Differentiate(T, coords["z"], 1)`, `Gradient`,
`lap`, … — and add it as a task or evaluate it via the in-memory handler above.

Whole-field diagnostics are also available directly: `total_kinetic_energy(u)` and
`total_enstrophy(u)` return a `Float64`, while `kinetic_energy(u)` and `enstrophy(u)` return the
corresponding `ScalarField`s.

### Probe points

Sample a field at a fixed grid index inside a callback and append to a time series:

```julia
idx = (8, 16)                       # grid index to sample
t_probe, T_probe = Float64[], Float64[]
run!(solver; stop_time=t_end, cfl=cfl, callbacks=[10 => function (s)
    ensure_layout!(T, :g)
    push!(t_probe, s.sim_time)
    push!(T_probe, get_grid_data(T)[idx...])
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

After a simulation with MPI, reconstruct the per-rank files from the handler's output root
(see [merge_netcdf_files](#merge_netcdf_files)):

```julia
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    cd("output") do
        merge_netcdf_files("snapshots"; cleanup=true)   # → output/snapshots_s1/snapshots_s1.nc
    end
end
```

### Loading Data

Tarang writes **NetCDF-4 with groups**: field data lives in the `vars` group, timestamps in `time`,
coordinate axes in `grids`. A bare `NetCDF.ncread(file, "T")` therefore fails with *"does not have a
variable named T"* — read through the group helpers instead.

```julia
file = "output/snapshots_s1/snapshots_s1.nc"

Tarang.group_variable_names(file, "vars")     # e.g. ["T", "u", "lap_T"]

T_data = group_ncread(file, "vars", "T")      # (write, x, y)
u_data = group_ncread(file, "vars", "u")      # (write, component, x, y)
t      = group_ncread(file, "time", "sim_time")
x      = group_ncread(file, "grids", "x")

T_last = T_data[end, :, :]                    # last written snapshot…
t[end]                                        # …and the time it was written at
```

Feed `T_last` to your plotting package of choice (Tarang does not depend on one).

---

## Complete Example

```julia
using Tarang, MPI

MPI.Init()

# ... setup problem, fields, and solver ...
solver = InitialValueSolver(problem, RK222(); dt=1e-3)

# CFL condition — takes the solver, and run! applies it every step
cfl = CFL(solver; initial_dt=1e-3, safety=0.4, max_dt=0.01)
add_velocity!(cfl, u)                       # u::VectorField

# Output handler — created with the solver, so run! auto-writes + closes it.
# Tasks (not the namespace) are what gets written, so add one per field.
snapshots = add_file_handler("output/snapshots", solver; sim_dt=0.1)
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

# Reconstruct the per-rank files into one global file, from the output root
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    cd("output") do
        merge_netcdf_files("snapshots"; cleanup=true)
    end
end

MPI.Finalize()
```

---

## See Also

- [Solvers](solvers.md): Integration methods
- [Fields](fields.md): Field operations for analysis
- [Tutorial: Analysis](../tutorials/analysis_and_output.md): Detailed examples
