# I/O API

File input/output for simulation data.

## NetCDF Output

### NetCDFFileHandler

```@docs
Tarang.NetCDFFileHandler
```

### add_file_handler

```@docs
Tarang.add_file_handler
```

There are two ways to create a handler, and they behave differently.

**Solver form (recommended).** The handler is registered with the solver, so `run!`
processes it every step and closes it at the end. It also takes its time metadata
(`sim_time`, `iteration`, `dt`) from the solver, so the `sim_dt`/`iter` cadence works
and every record is stamped with the simulation time:

```julia
handler = add_file_handler("snapshots", solver; sim_dt=0.002, max_writes=100)
add_task!(handler, T; name="T")
run!(solver; stop_iteration=6, progress=false)   # writes at sim_time 0, 0.002, 0.004, 0.006
```

**Distributor form.** Builds a handler that knows nothing about a solver: it does *not*
auto-register (you must pass it in `outputs=`), and with no solver attached its
`sim_time`/`iteration` are always `0`, so a `sim_dt` or `iter` cadence never advances and
only the first write lands. Pass `solver=` if you drive it with `run!`:

```julia
handler = add_netcdf_handler(
    "snapshots",               # Output path/name
    dist,                      # Distributor
    Dict("T" => T, "u" => u);  # Dict of fields (namespace for string-expression tasks)
    solver=solver,             # so sim_time/iteration are real, not 0
    sim_dt=0.002,
    max_writes=100             # writes per file before rolling over to the next set
)
add_task!(handler, T; name="T")
run!(solver; stop_iteration=6, outputs=[handler], progress=false)   # outputs= is required
```

**Arguments**:
- `base_path`: base path for output files (a trailing `.nc` is stripped)
- `dist`: distributor
- `vars`: dictionary mapping names to fields
- `solver`: optional solver, used for time metadata
- `sim_dt` / `iter` / `wall_dt`: write cadence (simulation time, iterations, wall seconds)
- `max_writes`: maximum writes per file before starting a new file set
- `parallel`: `"gather"` or `"virtual"` — see the Parallel I/O section below
- `mode`: `"overwrite"` (default) or `"append"`

**Returns**: `NetCDFFileHandler`

---

### add_netcdf_handler

```@docs
Tarang.add_netcdf_handler
```

Same as the distributor form of `add_file_handler` above.

---

### add_task!

```@docs
Tarang.add_task!
```

Add a field output task.

---

### add_task

```@docs
Tarang.add_task
```

Add a field output task (alternative syntax).

Always pass the **field or operator object**, never a string expression: a string task
(`add_task!(handler, "T*T")`) is accepted silently but writes a scalar of zeros instead of
the field. Build the operator instead (`add_task!(handler, T*T; name="T2")`).

```julia
add_task!(handler, field; name="field_name")
```

With postprocessing — `postprocess` receives the task's data array (this rank's slab) and
its return value is what gets written, so the written shape follows from it:

```julia
using Statistics

add_task!(handler, field;
    name="field_xmean",
    postprocess = data -> dropdims(mean(data, dims=1), dims=1)
)
```

---

### add_profile_task!

```@docs
Tarang.add_profile_task!
```

Add a task that reduces the field to a 1D profile: the mean over every dimension *except*
`dim`. The keyword is `dim` (singular) and it names the dimension to **keep** — either an
axis index or a coordinate symbol.

```julia
# z-profile: mean over x, keep z
add_profile_task!(handler, field; dim=:z, name="field_profile_z")

# same thing by axis index, on an (x, z) domain
add_profile_task!(handler, field; dim=2, name="field_profile_z")
```

To average over several dimensions instead, use `add_mean_task!` with `dims=`.

---

### add_mean_task!

```@docs
Tarang.add_mean_task!
```

Add a task that computes mean values. With `dims` it averages over those dimensions;
without `dims` it writes the single whole-domain mean (MPI-aware — combined across ranks).

```julia
add_mean_task!(handler, field; name="field_mean")                 # scalar, whole domain
add_mean_task!(handler, field; dims=(:x, :z), name="field_mean_xz")
```

---

### add_slice_task!

```@docs
Tarang.add_slice_task!
```

Add a task that extracts a slice, given the axis (`dim`) and the index along it (`idx`):

```julia
add_slice_task!(handler, field; dim=1, idx=8, name="field_slice")
```

!!! warning "`slices=` is currently broken"
    The `slices=Dict(...)` form advertised in the docstring above throws at write time
    (`MethodError: Cannot convert an object of type Int64 to an object of type Colon`).
    Use the `dim`/`idx` form.

---

### add_rms_task!

```@docs
Tarang.add_rms_task!
```

Add a task that computes RMS (root-mean-square) values — whole-domain by default,
or per-remaining-axis with `dims`.

```julia
add_rms_task!(handler, field; name="field_rms")
```

---

### add_variance_task!

```@docs
Tarang.add_variance_task!
```

Add a task that computes variance (same `dims` convention as `add_rms_task!`).

```julia
add_variance_task!(handler, field; name="field_variance")
```

---

### add_extrema_task!

```@docs
Tarang.add_extrema_task!
```

Add a task that tracks minimum and maximum values. It writes **two** variables,
`<name>_min` and `<name>_max` (whole-domain, MPI-aware):

```julia
add_extrema_task!(handler, field; name="T")   # writes vars/T_min and vars/T_max
```

---

### process!

```@docs
Tarang.process!
```

Write pending data to file (respecting the handler's `sim_dt`/`iter`/`wall_dt`
cadence). This is the low-level call.

**Recommended:** create the handler with the *solver* and let `run!` drive output
— the handler auto-registers, and `run!` calls `process!` every step and `close!`
at the end, so you never write a manual loop or forget a `process!`:

```julia
handler = add_file_handler("output/snap", solver; sim_dt=0.5)
add_task!(handler, u; name="u")
run!(solver; stop_time=20.0, cfl=cfl)   # auto: dt, process!(handler), close!
```

Call `process!`/`close!` directly only when you write your own step loop:

```julia
process!(handler;
    iteration=solver.iteration,
    wall_time=time() - wall_start,
    sim_time=solver.sim_time,
    timestep=solver.dt
)
```

It returns `true` if the write happened and `false` if the schedule said "not yet".

---

## Handler Management

### check_schedule

```@docs
Tarang.check_schedule
```

Check if the handler should write based on schedule.

---

### create_current_file!

```@docs
Tarang.create_current_file!
```

Create a new output file.

---

### current_path

```@docs
Tarang.current_path
```

Get the directory of the current file set (`<name>_s<set>`), not a file:

```julia
current_path(handler)   # "snapshots/snapshots_s1"
```

---

### current_file

```@docs
Tarang.current_file
```

Get the file the handler is writing into, inside that directory:

```julia
current_file(handler)   # serial: "snapshots/snapshots_s1/snapshots_s1.nc"
                        # np=2:   "snapshots/snapshots_s1/snapshots_s1_p0.nc" (per rank)
```

---

### get_output_files

```@docs
Tarang.get_output_files
```

List the files this handler has created (all sets).

---

### get_handler_info

```@docs
Tarang.get_handler_info
```

A `Dict` describing the handler state, with keys `base_path`, `name`, `set_num`,
`file_writes`, `total_writes`, `max_writes`, `num_tasks`, `task_names`, `parallel`,
`precision`, `mpi_rank`, `mpi_size`.

---

### close!

```@docs
Tarang.close!
```

Close the handler and finalize output.

```julia
close!(handler)
```

---

### reset!

```@docs
Tarang.reset!
```

Reset the handler state.

---

## File Merging

Under MPI each rank writes its own slab into its own file (see Parallel I/O below).
Merging is the post-processing step that stitches those per-rank files back into one file
holding the global field. It is a *serial* post-processing step: run it after the
simulation, from the handler's output directory (the one that contains `<name>_s<set>/`).
In a serial run there is nothing to merge — merging reports "No processor files found"
and returns `false`.

### NetCDFMerger

```@docs
Tarang.NetCDFMerger
```

---

### merge_netcdf_files

```@docs
Tarang.merge_netcdf_files
```

Merge one file set written by a parallel run:

```julia
cd("snapshots")   # the directory containing snapshots_s1/
merge_netcdf_files("snapshots"; set_number=1, output_name="snapshots_merged.nc")
```

The per-rank slabs are reconstructed with the `start`/`count`/`global_shape` attributes
each rank stored, so `vars/T` in the merged file has the full global shape
(`(writes, Nx, Nz)`).

Whole-domain reduction tasks (`add_mean_task!` and friends without `dims`) store no such
attributes, because their value is already global on every rank. The merger has nothing to
reconstruct for them, and says so loudly — expect, for each one,

```
┌ Warning: No global_shape metadata found for 'T_mean'. Using single-processor shape (3, 1)
│ — the merged file may contain only a fraction of the full domain.
      Error placing data from snapshots_s1_p1.nc: BoundsError(...)
```

This is noise, not failure: the merger keeps rank 0's value, which *is* the correct global
one, and the merge still returns `true`.

---

### merge_files!

```@docs
Tarang.merge_files!
```

The lower-level call: build a `NetCDFMerger` yourself and run it.

---

### batch_merge_netcdf

```@docs
Tarang.batch_merge_netcdf
```

Merge several handlers in one go; returns a `Dict` of handler name to success flag. Like
`merge_netcdf_files`, it looks for `<name>_s<set>/` in the *current* directory — so every
handler you name must have its set directories there. A handler created with a bare
`base_path` roots its sets in a directory of its own (`snapshots/snapshots_s1/`), and no
two such handlers ever share a parent; give them a common parent instead:

```julia
# at setup, so both handlers write into out/
h1 = add_file_handler("out/snapshots", solver; sim_dt=0.002)
h2 = add_file_handler("out/analysis",  solver; sim_dt=0.002)
# → out/snapshots_s1/snapshots_s1_p*.nc, out/analysis_s1/analysis_s1_p*.nc

# after the run
cd("out")
find_mergeable_handlers(".")                    # Dict("snapshots" => [1], "analysis" => [1])
batch_merge_netcdf(["snapshots", "analysis"])   # Dict("snapshots" => true, "analysis" => true)
```

Called from the wrong directory it finds nothing, warns `No processor files found for
merging`, and returns `false` for every handler.

---

### find_mergeable_handlers

```@docs
Tarang.find_mergeable_handlers
```

Scan a directory for handler file sets, returning a `Dict` of handler name to the set
numbers found:

```julia
find_mergeable_handlers("snapshots")   # Dict("snapshots" => [1])
```

---

### cleanup_source_files!

```@docs
Tarang.cleanup_source_files!
```

Remove source files after merging.

---

## Equation Parsing

Internal functions for parsing equation strings.

### split_equation

```@docs
Tarang.split_equation
```

---

### split_call

```@docs
Tarang.split_call
```

---

### lambdify_functions

```@docs
Tarang.lambdify_functions
```

---

## Reading NetCDF

### Using NetCDF.jl

Output files are **NetCDF-4 with groups**, so the task data is not at the file root: a
plain `NetCDF.ncread(file, "T")` fails with *"does not have a variable named T"*, and
`ncinfo` lists no variables. Read through the group helpers instead:

```julia
using Tarang, NetCDF

file = "snapshots/snapshots_s1/snapshots_s1.nc"

Tarang.group_variable_names(file, "vars")     # ["T", "T_mean", "T_min", "T_max", ...]
T = Tarang.group_ncread(file, "vars", "T")    # size (writes, Nx, Nz)

times = Tarang.group_ncread(file, "time", "sim_time")   # size (writes,)
x     = Tarang.group_ncread(file, "grids", "x")         # size (Nx,)

NetCDF.ncgetatt(file, "global", "title")      # "Tarang.jl simulation output"
```

## File Structure

### NetCDF Layout

Three groups: `vars` (the tasks), `time` (per-write metadata), `grids` (coordinates).
The write index is the **leading** dimension of every task variable.

```
snapshots/snapshots_s1/snapshots_s1.nc
├── vars
│   ├── T          (sim_time, x, z)
│   ├── T_profile_z (sim_time, T_profile_z_dim1)
│   └── T_min, T_max, ...   (sim_time, 1)
├── time
│   ├── sim_time    [N_writes]   (unlimited dimension)
│   ├── wall_time   [N_writes]
│   ├── timestep    [N_writes]
│   ├── iteration   [N_writes]
│   └── write_number[N_writes]
├── grids
│   ├── x [Nx]
│   ├── z [Nz]
│   └── <task>_dim1 [...]    for reduced/derived tasks
└── global attributes
```

Each task variable also carries `layout` (`"g"`/`"c"`), `grid_space`, and — for a
distributed run — the `start`/`count`/`global_shape`/`local_shape` of that rank's slab,
which is what `merge_netcdf_files` uses to reconstruct the global field.

### Global Attributes

Written automatically:

| Attribute | Value |
|-----------|-------------|
| `title` | "Tarang.jl simulation output" |
| `handler_name` | Handler name from path |
| `software` | "Tarang" |
| `software_repository` | Repository URL |
| `tarang_version` | Package version |
| `institution`, `source`, `history`, `Conventions` | Provenance / CF-1.8 |
| `set_number`, `writes` | Current file set and its write count |
| `mpi_size`, `processor_rank` | Only when running on more than one rank |

Read them with `NetCDF.ncgetatt(file, "global", name)`.

## Checkpoint and restart

`save_state` writes every evolved field in `solver.state` plus `sim_time`,
`iteration` and `dt`. `load_state!` reads it back.

```julia
save_state(solver, "checkpoints/run1")
# ... later, or in a new process ...
load_state!(solver, "checkpoints/run1")
```

Serial runs produce `run1.nc`. Under MPI each rank writes its own slab to
`run1/run1_p<rank>.nc` with no gather, so rank 0 never has to hold the whole
field. `load_state!` reads a checkpoint written at **any** rank count: each rank
works out the range it needs and reads only the overlapping hyperslabs.

### GPU

NetCDF reads into host memory, so the loader stages through a host buffer and
then performs one explicit upload into the field's existing device storage.

What is actually tested: **single-device GPU staging**, via JLArray emulation on
a serial 1-D field — it proves `load_field!` uploads into the device array
rather than replacing it with a host array. **GPU + MPI checkpointing is
untested.** It takes a different geometry branch entirely (first-dims
decomposition with the TransposableField convention, not the PencilArrays one),
no test executes that branch, and this repository's GPU CI pipeline is inert.
Treat a distributed GPU checkpoint as unverified.

### Restart fidelity

One-step schemes restart exactly: RK111, RK222, RK443, RK443\_IMEX, RKSMR,
RKGFY, ETD\_RK222, DiagonalIMEX\_RK222 and DiagonalIMEX\_RK443. So do the
first-order multistep bootstraps CNAB1 and SBDF1 — they depend only on the
current state, so there is no history to lose.

Every other scheme is multistep: it stores time levels the checkpoint does not
carry, so it re-seeds on restart and warns, naming the number of reduced-order
steps. The run stays correct but is not bit-identical to an uninterrupted one.

| Scheme | Reduced-order steps after restart |
|---|---|
| CNAB2, MCNAB2, CNLF2 | 1 |
| SBDF2, ETD\_SBDF2, ETD\_CNAB2 | 1 |
| SBDF3 | 2 |
| SBDF4 | 3 |
| DiagonalIMEX\_SBDF2 | 1 |

Note `DiagonalIMEX_SBDF2`: despite the family name it is a **multi-step**
method (so says its own docstring) and does *not* restart exactly. The other two
`DiagonalIMEX_*` schemes do.

## Field-level save and load

`save_field` and `load_field!` write and read a single `ScalarField`. They are
the layer `save_state`/`load_state!` are built on, and are useful on their own
for dumping one field outside the solver's checkpoint cadence.

```julia
save_field(u, "dumps/u")            # -> "dumps/u.nc"; returns the path written
load_field!(v, "dumps/u", "u")      # v must have the same global shape
```

The signature is `save_field(field, filename, dataset_name="field")` and
`load_field!(field, filename, dataset_name="field")`. `dataset_name` is the
NetCDF variable name, so several fields can share one file — which is exactly
how a `VectorField`'s components are stored:

```julia
for (i, component) in enumerate(w.components)
    save_field(component, "dumps/w", "component_$i")
end
```

`save_field` is additive. Writing a *new* variable into an existing file appends
to it; re-writing a variable that is already there rewrites the file, which
NetCDF can only do by deleting it — so that is allowed only when the variable is
the file's sole occupant, and refused with an error naming what would be lost
otherwise.

!!! note "Under MPI these are per-rank files, not one file"
    `save_field` returns `<stem>/<stem>_p<rank>.nc` on more than one rank, and
    each rank writes only its own slab — there is no gather. Point `load_field!`
    at the **directory** (`"dumps/u"`), not at an individual `_p<rank>.nc`, and
    it reassembles this rank's region from whichever slabs overlap it. That
    works at any rank count, including a count different from the one that
    wrote. On one rank both functions use a plain `<stem>.nc`.

`load_field!` also reads a `NetCDFFileHandler` output directory, since those
variables carry the same `start`/`count`/`global_shape` attributes. Handler
variables have a leading unlimited `sim_time` axis, and the **most recent write**
is the one loaded.

## Parallel I/O

There is no gather-to-rank-0 write path. On more than one rank **every rank writes its own
file**, containing its own slab:

```
snapshots/snapshots_s1/snapshots_s1_p0.nc    # rank 0's slab, e.g. (writes, 16, 8)
snapshots/snapshots_s1/snapshots_s1_p1.nc    # rank 1's slab
```

Reassemble them afterwards with `merge_netcdf_files` (see File Merging). Whole-domain
reductions (`add_mean_task!`, `add_rms_task!`, `add_variance_task!`, `add_extrema_task!`
without `dims`) are MPI-aware: every rank's file already holds the *global* value.

The `parallel` keyword (`"gather"`, the default, or `"virtual"`) only chooses the serial
file name: with `parallel="gather"` on one rank you get a single `snapshots_s1.nc`;
otherwise the file is named `..._p<rank>.nc`. It does not move data between ranks.

## File Management

### File Naming

Each file set lives in its own directory, and sets are numbered sequentially — a new set
starts once `max_writes` writes have landed in the current one:

```
snapshots/snapshots_s1/snapshots_s1.nc   # first set
snapshots/snapshots_s2/snapshots_s2.nc   # after max_writes reached
snapshots/snapshots_s3/snapshots_s3.nc   # etc.
```

With `mode="overwrite"` (the default) the handler deletes pre-existing sets of the same
name when it is constructed.

## Performance Tips

1. **Batch writes**: use a `sim_dt`/`iter` cadence rather than writing every iteration
2. **Cap file size**: `max_writes` rolls over to a new set instead of one huge file
3. **Write what you need**: reduction tasks (profiles, means, extrema) are far cheaper to
   store than full fields
4. **Merge offline**: reassemble per-rank files after the run, not during it

## See Also

- [Analysis Tutorial](../tutorials/analysis_and_output.md): Complete examples
- [Parallelism](../pages/parallelism.md): MPI considerations
