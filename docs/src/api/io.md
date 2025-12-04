# I/O API

File input/output for simulation data.

## NetCDF Output

### add_file_handler

```@docs
add_file_handler
```

Create a file handler for NetCDF output.

```julia
handler = add_file_handler(
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

### add_task

```@docs
add_task
```

Add a field output task.

```julia
add_task(handler, field; name="field_name")
```

With postprocessing:

```julia
add_task(handler, field;
    name="processed",
    postprocess=data -> mean(data, dims=1)
)
```

### add_mean_task!

```@docs
add_mean_task!
```

Add a task that computes mean over dimensions.

```julia
# Mean over x (dimension 1)
add_mean_task!(handler, field; dims=1, name="field_mean_x")

# Mean over x and y
add_mean_task!(handler, field; dims=(1,2), name="field_mean_xy")
```

### add_slice_task!

```@docs
add_slice_task!
```

Add a task that extracts a slice.

```julia
# Slice at index
add_slice_task!(handler, field; dim=1, idx=64, name="field_slice")

# Using slices dictionary
add_slice_task!(handler, field; slices=Dict(1 => 32), name="slice")
```

### process!

```@docs
process!
```

Write pending data to file.

```julia
process!(handler;
    iteration=solver.iteration,
    wall_time=elapsed,
    sim_time=solver.sim_time,
    timestep=dt
)
```

### current_file

```@docs
current_file
```

Get the current output file path.

```julia
filepath = current_file(handler)
```

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
        state["fields"][name] = copy(field.data_c)
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
        field.data_c .= data
        field.current_layout = :c
    end
end
```

## Parallel I/O Modes

### Gather Mode

```julia
handler = add_file_handler(path, dist, fields; parallel="gather")
```

- All data gathered to rank 0
- Single output file
- Memory limited by rank 0

### Virtual Mode

```julia
handler = add_file_handler(path, dist, fields; parallel="virtual")
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
# Merge script (conceptual)
function merge_virtual_files(base_path, num_procs)
    # Read all partial files
    # Combine into single array
    # Write merged output
end
```

## Performance Tips

1. **Batch writes**: Don't write every iteration
2. **Use gather for small outputs**: Simpler file handling
3. **Use virtual for large outputs**: Better scaling
4. **Compress if needed**: NetCDF supports compression

## See Also

- [Analysis Tutorial](../tutorials/analysis_and_output.md): Complete examples
- [Parallelism](../pages/parallelism.md): MPI considerations
