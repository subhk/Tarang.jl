# NetCDF File Merging for Tarang.jl

This document describes the comprehensive NetCDF file merging system for Tarang.jl, designed to combine per-processor output files from parallel simulations.

## Overview

Tarang.jl creates per-processor NetCDF files during parallel runs:
```
snapshots_s1/
├── snapshots_s1_p0.nc    # Processor 0 data
├── snapshots_s1_p1.nc    # Processor 1 data  
├── snapshots_s1_p2.nc    # Processor 2 data
└── snapshots_s1_p3.nc    # Processor 3 data
```

The merging system combines these into single files for analysis and visualization.

## Quick Start

### Command Line Usage

```bash
# Basic merge
julia scripts/merge_netcdf.jl snapshots

# Merge with cleanup (deletes original files)
julia scripts/merge_netcdf.jl --cleanup snapshots

# Auto-discover and merge all handlers
julia scripts/merge_netcdf.jl --auto

# Merge specific set with custom output
julia scripts/merge_netcdf.jl --set=2 --output=my_data.nc snapshots

# Batch merge multiple handlers
julia scripts/merge_netcdf.jl --cleanup snapshots analysis checkpoints
```

### Programmatic Usage

```julia
using Tarang

# Basic merge
merge_netcdf_files("snapshots")

# Advanced options
merge_netcdf_files("snapshots", 
                   output_name="combined_data.nc",
                   cleanup=true, 
                   merge_mode=RECONSTRUCT)

# Batch merge
batch_merge_netcdf(["snapshots", "analysis"], cleanup=true)

# Auto-discovery
handlers = find_mergeable_handlers()
for handler in keys(handlers)
    merge_netcdf_files(handler, cleanup=true)
end
```

## Merge Modes

### 1. RECONSTRUCT (Default)
- **Purpose**: Reconstruct global fields from distributed data
- **Method**: Averages overlapping data from processors  
- **Best for**: Most general use cases
- **Output**: Single merged field per variable

```julia
merge_netcdf_files("snapshots", merge_mode=RECONSTRUCT)
```

### 2. SIMPLE_CONCAT  
- **Purpose**: Concatenate all processor data
- **Method**: Adds processor dimension, preserves all data
- **Best for**: Debugging, detailed analysis of processor-specific data
- **Output**: Data with additional processor dimension

```julia
merge_netcdf_files("snapshots", merge_mode=SIMPLE_CONCAT)
```

### 3. DOMAIN_DECOMP
- **Purpose**: Advanced domain decomposition reconstruction  
- **Method**: Reconstructs based on spatial domain decomposition
- **Best for**: Complex domain decomposition scenarios
- **Status**: Framework implemented, full reconstruction pending

```julia
merge_netcdf_files("snapshots", merge_mode=DOMAIN_DECOMP)
```

## File Structure

### Input Files (Per-Processor)
```
snapshots_s1/snapshots_s1_p0.nc:
├── sim_time[unlimited]           # Time coordinate
├── wall_time[sim_time]          # Wall clock time
├── iteration[sim_time]          # Iteration numbers
├── velocity[sim_time, x, y]     # Data variables
└── temperature[sim_time, x, y]  # More data variables

snapshots_s1/snapshots_s1_p1.nc:
├── ... (same structure, different data)
```

### Output File (Merged)
```
snapshots_s1.nc:
├── sim_time[unlimited]           # Combined time coordinate  
├── wall_time[sim_time]          # Combined wall time
├── iteration[sim_time]          # Combined iteration numbers
├── velocity[sim_time, x, y]     # Reconstructed velocity field
├── temperature[sim_time, x, y]  # Reconstructed temperature field
└── Global Attributes:
    ├── processor_count = 4
    ├── merge_timestamp = "2024-..."
    ├── source_files = "p0.nc, p1.nc, ..."
    └── reconstruction_method = "averaged_from_processors"
```

## API Reference

### Core Functions

#### `merge_netcdf_files(base_name; kwargs...)`
Merge per-processor files for a single handler.

**Arguments:**
- `base_name::String`: Handler name (e.g., "snapshots")
- `set_number::Int=1`: Set number to merge  
- `output_name::String=""`: Custom output filename
- `merge_mode::MergeMode=RECONSTRUCT`: Merge strategy
- `cleanup::Bool=false`: Delete source files after merge
- `verbose::Bool=true`: Print progress information

**Returns:** `Bool` - Success status

#### `batch_merge_netcdf(handlers; kwargs...)`
Merge multiple handlers in batch mode.

**Arguments:**
- `handlers::Vector{String}`: List of handler names
- Additional kwargs same as `merge_netcdf_files`

**Returns:** `Dict{String, Bool}` - Success status per handler

#### `find_mergeable_handlers(directory=".")`
Auto-discover handlers with processor files ready for merging.

**Returns:** `Dict{String, Vector{Int}}` - Handler names → available set numbers

### Advanced Usage

#### Custom Merger Object
```julia
# Create custom merger
merger = NetCDFMerger("snapshots", 
                     set_number=1,
                     output_name="custom.nc",
                     merge_mode=SIMPLE_CONCAT,
                     cleanup=true,
                     verbose=false)

# Analyze files before merging
file_info = analyze_processor_files(merger)
println("Found $(length(merger.processor_files)) processor files")

# Perform merge
success = merge_files!(merger)
```

## Command Line Script

### Installation
The merge script is located at `scripts/merge_netcdf.jl`. Make it executable:

```bash
chmod +x scripts/merge_netcdf.jl
```

### Usage
```bash
julia scripts/merge_netcdf.jl [OPTIONS] HANDLER_NAME [HANDLER_NAME2 ...]
```

### Options
- `--set=N`: Set number to merge (default: 1)
- `--output=FILE`: Output filename (auto-generated if not specified)
- `--mode=MODE`: Merge mode (concat, reconstruct, decomp)
- `--cleanup`: Delete source files after successful merge
- `--auto`: Auto-discover all mergeable handlers
- `--quiet`: Minimal output
- `--help`: Show help message

### Examples
```bash
# Basic usage
julia scripts/merge_netcdf.jl snapshots

# Advanced options
julia scripts/merge_netcdf.jl --mode=concat --cleanup --output=final.nc snapshots

# Auto-discovery
julia scripts/merge_netcdf.jl --auto --quiet

# Batch processing  
julia scripts/merge_netcdf.jl --cleanup snapshots analysis checkpoints
```

## Best Practices

### 1. Backup Before Cleanup
```julia
# Safe approach - merge first, then cleanup manually if satisfied
merge_netcdf_files("snapshots", cleanup=false)
# ... verify merged file ...
# Manual cleanup if needed
```

### 2. Verify Merged Files
```julia
using NetCDF

# Check merged file
info = ncinfo("snapshots_s1.nc")
println("Variables: $(length(info.vars))")

# Compare with original
original_data = ncread("snapshots_s1/snapshots_s1_p0.nc", "velocity")
merged_data = ncread("snapshots_s1.nc", "velocity")
```

### 3. Workflow Integration
```julia
# In your simulation script
function postprocess_results()
    handlers = ["snapshots", "analysis", "checkpoints"]
    
    # Merge all handlers
    results = batch_merge_netcdf(handlers, cleanup=true, verbose=false)
    
    # Check results
    failed = [h for (h, success) in results if !success]
    if !isempty(failed)
        @warn "Failed to merge: $(join(failed, ", "))"
    end
    
    return all(values(results))
end
```

## Troubleshooting

### Common Issues

**1. "No processor files found"**
- Check file naming follows pattern: `handler_s1_p0.nc`, `handler_s1_p1.nc`, etc.
- Verify files are in current directory or handler set directory

**2. "Dimension values defined more than once"**
- NetCDF dimension conflict - usually resolved automatically
- Try different merge mode if persistent

**3. "Cannot read variable"**
- File corruption or incomplete write
- Check original processor files individually

**4. Memory issues with large files**
- Use `SIMPLE_CONCAT` mode for large datasets
- Process smaller subsets or individual sets

### Debug Mode
```julia
# Enable verbose output for debugging
merge_netcdf_files("snapshots", verbose=true)

# Check file structure before merging
handlers = find_mergeable_handlers()
for (handler, sets) in handlers
    println("$handler: $sets")
end
```

## Performance Notes

- **File I/O**: Merge operations are I/O bound
- **Memory**: Keeps one time slice in memory at a time
- **Parallelization**: Merging itself is serial, but can process multiple handlers in parallel externally
- **Storage**: Merged files are typically smaller than sum of processor files due to metadata consolidation

## Integration with Analysis Tools

### Python/xarray
```python
import xarray as xr

# Load merged file  
ds = xr.open_dataset('snapshots_s1.nc')
print(ds)

# Plot time series
ds.velocity.isel(x=32, y=16).plot()
```

### Julia/NCDatasets
```julia
using NCDatasets

# Load merged file
ds = NCDataset("snapshots_s1.nc")
velocity = ds["velocity"][:]
sim_time = ds["sim_time"][:]

# Analysis
mean_velocity = mean(velocity, dims=(2,3))
```

### Command line tools
```bash
# Examine file structure
ncdump -h snapshots_s1.nc

# Extract specific variable
ncks -v velocity snapshots_s1.nc velocity_only.nc

# View with ncview
ncview snapshots_s1.nc
```