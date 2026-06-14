# I/O API

File input/output for simulation data.

## NetCDF Output

### NetCDFFileHandler

```@docs
Tarang.NetCDFFileHandler
```

### add_netcdf_handler

```@docs
Tarang.add_netcdf_handler
```

Create a file handler for NetCDF output.

---

### add_file_handler

```@docs
Tarang.add_file_handler
```

Alternative constructor for file handlers.

```julia
handler = add_netcdf_handler(
    base_path,          # Output path/name
    dist,               # Distributor
    fields_dict;        # Dict of fields
    parallel="gather",  # I/O mode
    max_writes=100      # Files before rollover
)
```

**Arguments**:
- `base_path`: Base path for output files
- `dist`: MPI distributor
- `fields_dict`: Dictionary mapping names to fields
- `parallel`: "gather" (single file) or "virtual" (per-process)
- `max_writes`: Maximum writes per file before creating new file

**Returns**: `NetCDFFileHandler`

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

```julia
add_task!(handler, field; name="field_name")
```

With postprocessing:

```julia
add_task!(handler, field;
    name="processed",
    postprocess=data -> mean(data, dims=1)
)
```

---

### add_profile_task!

```@docs
Tarang.add_profile_task!
```

Add a task that computes mean profile over specified dimensions.

```julia
# Mean over x (dimension 1) - produces z-profile
add_profile_task!(handler, field; dims=1, name="field_profile")

# Mean over x and y
add_profile_task!(handler, field; dims=(1,2), name="field_profile_xy")
```

---

### add_mean_task!

```@docs
Tarang.add_mean_task!
```

Add a task that computes mean values.

```julia
add_mean_task!(handler, field; name="field_mean")
```

---

### add_slice_task!

```@docs
Tarang.add_slice_task!
```

Add a task that extracts a slice.

```julia
# Slice at index
add_slice_task!(handler, field; dim=1, idx=64, name="field_slice")

# Using slices dictionary
add_slice_task!(handler, field; slices=Dict(1 => 32), name="slice")
```

---

### add_rms_task!

```@docs
Tarang.add_rms_task!
```

Add a task that computes RMS (root-mean-square) values.

```julia
add_rms_task!(handler, field; name="field_rms")
```

---

### add_variance_task!

```@docs
Tarang.add_variance_task!
```

Add a task that computes variance.

```julia
add_variance_task!(handler, field; name="field_variance")
```

---

### add_extrema_task!

```@docs
Tarang.add_extrema_task!
```

Add a task that tracks minimum and maximum values.

```julia
add_extrema_task!(handler, field; name="field_extrema")
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
    wall_time=elapsed,
    sim_time=solver.sim_time,
    timestep=dt
)
```

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

Get the current output file path.

```julia
filepath = current_path(handler)
```

---

### current_file

```@docs
Tarang.current_file
```

Get the current output file (alternative to current_path).

```julia
filepath = current_file(handler)
```

---

### get_output_files

```@docs
Tarang.get_output_files
```

Get list of all output files created by handler.

---

### get_handler_info

```@docs
Tarang.get_handler_info
```

Get information about the handler state.

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

### NetCDFMerger

```@docs
Tarang.NetCDFMerger
```

---

### merge_netcdf_files

```@docs
Tarang.merge_netcdf_files
```

Merge multiple NetCDF files from parallel output.

```julia
merge_netcdf_files(base_path; output_name="merged")
```

---

### merge_files!

```@docs
Tarang.merge_files!
```

---

### batch_merge_netcdf

```@docs
Tarang.batch_merge_netcdf
```

Merge multiple sets of parallel output files.

```julia
batch_merge_netcdf(["output1", "output2", "output3"])
```

---

### find_mergeable_handlers

```@docs
Tarang.find_mergeable_handlers
```

Find handlers that can be merged.

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

```julia
using NetCDF

# Read variable
data = NetCDF.ncread(filename, "temperature")

# Read attribute
attr = NetCDF.ncgetatt(filename, "NC_GLOBAL", "title")

# Read time array
times = NetCDF.ncread(filename, "t")
```

## File Structure

### NetCDF Layout

```
output_s1.nc
├── Dimensions
│   ├── x (128)
│   ├── z (64)
│   └── t (unlimited)
├── Coordinates
│   ├── x [128]
│   ├── z [64]
│   └── t [N_writes]
├── Variables
│   ├── temperature (t, x, z)
│   ├── velocity_x (t, x, z)
│   └── ...
└── Attributes
    ├── title
    ├── handler_name
    ├── software
    └── tarang_version
```

### Global Attributes

Written automatically:

| Attribute | Description |
|-----------|-------------|
| `title` | "Tarang.jl simulation output" |
| `handler_name` | Handler name from path |
| `software` | "Tarang" |
| `tarang_version` | Package version |

## Checkpointing

### Saving State

```julia
function save_checkpoint(solver, filename)
    using JLD2

    state = Dict(
        "sim_time" => solver.sim_time,
        "iteration" => solver.iteration,
        "dt" => solver.dt,
        "fields" => Dict()
    )

    for (name, field) in solver.problem.fields
        Tarang.ensure_layout!(field, :c)
        state["fields"][name] = copy(get_coeff_data(field))
    end

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @save filename state
    end
end
```

### Loading State

```julia
function load_checkpoint!(solver, filename)
    using JLD2
    @load filename state

    solver.sim_time = state["sim_time"]
    solver.iteration = state["iteration"]
    solver.dt = state["dt"]

    for (name, data) in state["fields"]
        field = solver.problem.fields[name]
        get_coeff_data(field) .= data
        field.current_layout = :c
    end
end
```

## Parallel I/O Modes

### Gather Mode

```julia
handler = add_netcdf_handler(path, dist, fields; parallel="gather")
```

- All data gathered to rank 0
- Single output file
- Memory limited by rank 0

### Virtual Mode

```julia
handler = add_netcdf_handler(path, dist, fields; parallel="virtual")
```

- Each process writes own file
- Filenames: `output_p0.nc`, `output_p1.nc`, etc.
- Requires post-processing to merge

## File Management

### File Naming

Files are numbered sequentially:

```
output_s1.nc   # First file
output_s2.nc   # After max_writes reached
output_s3.nc   # etc.
```

### Merging Files

For post-processing virtual files:

```julia
# Merge all parallel files
merge_netcdf_files("output"; output_name="output_merged", cleanup=true)

# Batch merge multiple handlers
batch_merge_netcdf(["snapshots", "analysis", "checkpoints"])
```

## Performance Tips

1. **Batch writes**: Don't write every iteration
2. **Use gather for small outputs**: Simpler file handling
3. **Use virtual for large outputs**: Better scaling
4. **Compress if needed**: NetCDF supports compression

## See Also

- [Analysis Tutorial](../tutorials/analysis_and_output.md): Complete examples
- [Parallelism](../pages/parallelism.md): MPI considerations
