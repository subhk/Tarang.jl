# Configuration

Tarang.jl provides multiple ways to configure its behavior, from environment variables to configuration files. This guide covers all configuration options.

## Configuration Files

### Location

Tarang.jl searches for configuration files in this order:

1. Current working directory: `./tarang.toml`
2. User configuration: `~/.tarang/config.toml`
3. Package default: Built-in defaults

The first file found takes precedence.

### File Format

Configuration uses TOML format:

```toml
# tarang.toml

[parallelism]
TRANSPOSE_LIBRARY = "PENCIL"
GROUP_TRANSPOSES = true
SYNC_TRANSPOSES = true

[transforms]
GROUP_TRANSFORMS = true
DEALIAS_BEFORE_CONVERTING = true

[transforms-fftw]
PLANNING_RIGOR = "FFTW_MEASURE"
USE_FFTW_WISDOM = true
WISDOM_FILE = "fftw_wisdom.dat"

[logging]
LEVEL = "INFO"
FILE = "tarang.log"
MPI_AWARE = true

[performance]
PROFILE_DIR = "profiles"
ENABLE_PROFILING = false
MEMORY_POOL_SIZE = "1GB"

[output]
DEFAULT_FORMAT = "netcdf"
COMPRESSION_LEVEL = 4
CHUNK_SIZE = "auto"
```

## Configuration Sections

### Parallelism

Controls MPI and parallel execution:

```toml
[parallelism]
# Transpose library: "PENCIL" or "CUSTOM"
TRANSPOSE_LIBRARY = "PENCIL"

# Group multiple transposes together
GROUP_TRANSPOSES = true

# Synchronize after transposes (for debugging)
SYNC_TRANSPOSES = false

# Process mesh optimization
AUTO_OPTIMIZE_MESH = true
PREFERRED_PENCIL_AXIS = "auto"  # "auto", "x", "y", "z"
```

**Options**:
- `TRANSPOSE_LIBRARY`: Backend for array transposes
  - `"PENCIL"`: Use PencilArrays (recommended)
  - `"CUSTOM"`: Custom implementation
- `GROUP_TRANSPOSES`: Batch transposes for efficiency
- `SYNC_TRANSPOSES`: Add MPI barriers (reduces performance, useful for debugging)

### Transforms

FFT and spectral transform settings:

```toml
[transforms]
# Group multiple transforms
GROUP_TRANSFORMS = true

# Dealias before converting between spaces
DEALIAS_BEFORE_CONVERTING = true

# Transform cache size (number of temporary arrays)
CACHE_SIZE = 10

# Forward transform in-place when possible
INPLACE_FORWARD = true

# Backward transform in-place when possible
INPLACE_BACKWARD = true
```

### FFTW Settings

FFTW-specific configuration:

```toml
[transforms-fftw]
# Planning rigor: determines FFTW plan optimization level
PLANNING_RIGOR = "FFTW_MEASURE"

# Save and load FFTW wisdom
USE_FFTW_WISDOM = true
WISDOM_FILE = "fftw_wisdom.dat"

# Number of threads for FFTW (usually leave at 1 with MPI)
NUM_THREADS = 1

# FFTW flags
FLAGS = ["FFTW_MEASURE", "FFTW_DESTROY_INPUT"]
```

**PLANNING_RIGOR options**:
- `"FFTW_ESTIMATE"`: Fast planning, slower execution (good for testing)
- `"FFTW_MEASURE"`: Medium planning time, good performance (recommended)
- `"FFTW_PATIENT"`: Slow planning, best performance (for production runs)
- `"FFTW_EXHAUSTIVE"`: Very slow planning, marginally better than PATIENT

!!! tip "FFTW Wisdom"
    Save FFTW wisdom on first run, then reuse it to skip planning:
    ```julia
    using FFTW
    # After first run with FFTW_MEASURE
    FFTW.export_wisdom("fftw_wisdom.dat")

    # Subsequent runs
    FFTW.import_wisdom("fftw_wisdom.dat")
    ```

### Logging

Configure logging output:

```toml
[logging]
# Log level: "TRACE", "DEBUG", "INFO", "NOTICE", "WARN", "ERROR"
LEVEL = "INFO"

# Log file (empty string for stdout only)
FILE = "tarang.log"

# Include MPI rank in log messages
MPI_AWARE = true

# Log to stdout as well as file
ALSO_STDOUT = false

# Timestamp format
TIMESTAMP_FORMAT = "yyyy-mm-dd HH:MM:SS"
```

**Programmatic logging setup**:
```julia
using Tarang.Logging

setup_tarang_logging(
    level="INFO",
    filename="simulation.log",
    mpi_aware=true,
    console=true,
    format_timestamp=true
)
```

### Performance

Performance tuning and profiling:

```toml
[performance]
# Enable performance profiling
ENABLE_PROFILING = false

# Directory for profile output
PROFILE_DIR = "profiles"

# Memory pool size for reusable arrays
MEMORY_POOL_SIZE = "1GB"

# Pre-allocate work arrays
PREALLOCATE_WORK = true

# Number of work arrays per field
WORK_ARRAY_COUNT = 5
```

### Output

Default output settings:

```toml
[output]
# Default output format: "netcdf", "hdf5", "jld2"
DEFAULT_FORMAT = "netcdf"

# Compression level (0-9, 0=none, 9=maximum)
COMPRESSION_LEVEL = 4

# Chunk size: "auto" or explicit dimensions
CHUNK_SIZE = "auto"

# Output precision: "single", "double"
PRECISION = "double"

# Include metadata
INCLUDE_METADATA = true
```

## Environment Variables

Environment variables override configuration file settings:

### Core Settings

```bash
# Log level
export TARANG_LOG_LEVEL=DEBUG

# Log file
export TARANG_LOG_FILE=my_simulation.log

# Profile directory
export TARANG_PROFILE_DIR=my_profiles

# Configuration file location
export TARANG_CONFIG=/path/to/tarang.toml
```

### FFTW Settings

```bash
# FFTW planning rigor
export FFTW_PLANNING_RIGOR=FFTW_MEASURE

# FFTW wisdom file
export FFTW_WISDOM_FILE=fftw_wisdom.dat

# FFTW threads (use 1 with MPI)
export OMP_NUM_THREADS=1
```

### MPI Settings

```bash
# OpenMPI settings
export OMPI_MCA_mpi_show_mca_params=1
export OMPI_MCA_btl=^openib  # Disable InfiniBand

# Process binding
export OMPI_MCA_hwloc_base_binding_policy=core
```

### Julia Settings

```bash
# Julia threads
export JULIA_NUM_THREADS=4

# Optimization level
export JULIA_OPT_LEVEL=3

# Disable precompilation
export JULIA_PKG_PRECOMPILE_AUTO=0
```

## Runtime Configuration

Some settings can be changed at runtime:

```julia
using Tarang.Config

# Get current configuration
config = get_config()

# Modify settings
config.logging.level = "DEBUG"
config.transforms.group_transforms = false

# Apply changes
apply_config!(config)
```

## Example Configurations

### Development (Fast Startup)

```toml
[transforms-fftw]
PLANNING_RIGOR = "FFTW_ESTIMATE"  # Fast planning

[logging]
LEVEL = "DEBUG"
FILE = "debug.log"

[performance]
ENABLE_PROFILING = true
```

### Production (Maximum Performance)

```toml
[transforms-fftw]
PLANNING_RIGOR = "FFTW_PATIENT"
USE_FFTW_WISDOM = true

[transforms]
GROUP_TRANSFORMS = true
DEALIAS_BEFORE_CONVERTING = true

[logging]
LEVEL = "INFO"

[performance]
MEMORY_POOL_SIZE = "2GB"
PREALLOCATE_WORK = true
```

### Debugging (Verbose Output)

```toml
[parallelism]
SYNC_TRANSPOSES = true  # Add barriers for debugging

[logging]
LEVEL = "DEBUG"
MPI_AWARE = true
ALSO_STDOUT = true

[performance]
ENABLE_PROFILING = true
```

### HPC Cluster

```toml
[parallelism]
TRANSPOSE_LIBRARY = "PENCIL"
AUTO_OPTIMIZE_MESH = true

[transforms-fftw]
PLANNING_RIGOR = "FFTW_MEASURE"
USE_FFTW_WISDOM = true
WISDOM_FILE = "/scratch/user/fftw_wisdom.dat"

[logging]
LEVEL = "INFO"
FILE = "/scratch/user/simulation.log"

[output]
DEFAULT_FORMAT = "netcdf"
COMPRESSION_LEVEL = 6
```

## Configuration Priority

Settings are applied in this order (later overrides earlier):

1. Built-in defaults
2. Configuration file (`tarang.toml`)
3. Environment variables
4. Runtime API calls

Example:
```bash
# File: tarang.toml has LEVEL = "INFO"
export TARANG_LOG_LEVEL=DEBUG  # Overrides file setting
```

## Best Practices

### For Development

```bash
export OMP_NUM_THREADS=1
export FFTW_PLANNING_RIGOR=FFTW_ESTIMATE
export TARANG_LOG_LEVEL=DEBUG
```

### For Production

```toml
[transforms-fftw]
PLANNING_RIGOR = "FFTW_MEASURE"
USE_FFTW_WISDOM = true

[performance]
MEMORY_POOL_SIZE = "2GB"
PREALLOCATE_WORK = true
```

### For Benchmarking

```toml
[performance]
ENABLE_PROFILING = true

[logging]
LEVEL = "INFO"
FILE = "benchmark.log"
```

## Validation

Check your configuration:

```julia
using Tarang.Config

# Load and validate configuration
config = load_config("tarang.toml")
validate_config(config)

# Print current configuration
print_config(config)
```

## Next Steps

- [Parallelism Guide](parallelism.md): Detailed MPI configuration
- [Optimization](optimization.md): Performance tuning
- [Analysis and Output](../tutorials/analysis_and_output.md): Output configuration
