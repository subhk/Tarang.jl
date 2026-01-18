# Installation

This guide walks you through installing Tarang.jl and its dependencies.

## System Requirements

### Julia
Tarang.jl requires Julia 1.6 or later. We recommend using the latest stable release of Julia.

**Download Julia:**
- Visit [julialang.org/downloads](https://julialang.org/downloads/)
- For Linux/macOS: Use [juliaup](https://github.com/JuliaLang/juliaup) for easy version management
- For Windows: Download the installer from the Julia website

**Verify installation:**
```bash
julia --version
```

### MPI Library

Tarang.jl requires an MPI implementation for parallel computing.

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

#### macOS
```bash
brew install open-mpi
```

#### Windows
Download and install [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) or use WSL with Linux MPI.

**Verify MPI installation:**
```bash
mpiexec --version
```

### Optional: HPC Clusters

On HPC systems, load the appropriate MPI module:
```bash
module load openmpi/4.1.0  # or your system's MPI module
```

!!! warning "MPI Compatibility"
    Ensure that MPI.jl is built against the same MPI library you plan to use. See the [MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/configuration/) for details on configuring MPI.

## Installing Tarang.jl

### From GitHub (Recommended)

The package is currently hosted on GitHub and can be installed directly:

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Tarang.jl")
```

### Development Installation

For development or to get the latest features:

```bash
git clone https://github.com/subhk/Tarang.jl.git
cd Tarang.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then in Julia:
```julia
using Pkg
Pkg.develop(path="/path/to/Tarang.jl")
```

## Installing Dependencies

Tarang.jl will automatically install most Julia dependencies. Key packages include:

- **MPI.jl**: MPI bindings for Julia
- **PencilArrays.jl**: Distributed array library for spectral methods
- **PencilFFTs.jl**: Parallel FFT transforms
- **FFTW.jl**: Fast Fourier Transform library
- **HDF5.jl**: HDF5 file I/O
- **LinearAlgebra**: Standard library (included with Julia)
- **SparseArrays**: Standard library (included with Julia)

### Configuring MPI.jl

If you need to use a system-provided MPI:

```julia
using Pkg
ENV["JULIA_MPI_BINARY"] = "system"
Pkg.build("MPI"; verbose=true)
```

Verify MPI configuration:
```julia
using MPI
MPI.versioninfo()
```

## Verification

Test your installation:

```julia
using Tarang
using MPI

println("Tarang.jl version: ", Tarang.__version__)
println("MPI available: ", MPI.Initialized() || MPI.Init())
```

Run the test suite:
```julia
using Pkg
Pkg.test("Tarang")
```

## Quick Test Run

Create a test file `test_tarang.jl`:

```julia
using Tarang, MPI

MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

if rank == 0
    println("Running Tarang on $size MPI processes")
end

# Create simple 2D domain
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords, mesh=(2, 2))

x = RealFourier(coords["x"]; size=64, bounds=(0.0, 2π))
z = ChebyshevT(coords["z"]; size=32, bounds=(0.0, 1.0))

domain = Domain(dist, (x, z))

if rank == 0
    println("Successfully created domain!")
end

MPI.Finalize()
```

Run with MPI:
```bash
mpiexec -n 4 julia test_tarang.jl
```

## Troubleshooting

### MPI Issues

**Problem**: `ERROR: MPI has not been built`

**Solution**: Build MPI.jl manually:
```julia
using Pkg
Pkg.build("MPI")
```

**Problem**: MPI version mismatch

**Solution**: Rebuild MPI.jl against system MPI:
```julia
ENV["JULIA_MPI_BINARY"] = "system"
Pkg.build("MPI"; verbose=true)
```

### Performance Issues

**Problem**: Warning about `OMP_NUM_THREADS`

**Solution**: Set the environment variable:
```bash
export OMP_NUM_THREADS=1
mpiexec -n 4 julia your_script.jl
```

### FFTW Issues

**Problem**: FFTW planning errors

**Solution**: Use a different FFTW planning rigor in your configuration:
```julia
ENV["FFTW_PLANNING_RIGOR"] = "FFTW_ESTIMATE"
```

### HDF5 Issues

**Problem**: HDF5 library conflicts

**Solution**: Rebuild HDF5.jl:
```julia
using Pkg
Pkg.build("HDF5")
```

## Next Steps

Now that Tarang.jl is installed, continue to:
- [First Steps](first_steps.md): Create your first simulation
- [Running with MPI](running_with_mpi.md): Learn about parallel execution
- [Tutorials](../tutorials/overview.md): Detailed examples and guides

## System-Specific Notes

### macOS Apple Silicon (M1/M2)

Julia and all dependencies work natively on Apple Silicon. Use the ARM64 Julia build for best performance.

### HPC Clusters

Most HPC systems have modules for MPI and HDF5. Load them before using Julia:

```bash
module load julia/1.9
module load openmpi/4.1
module load hdf5/1.12
```

Configure MPI.jl to use the system MPI as shown above.

### Containers

Tarang.jl works well in containers. Example Dockerfile:

```dockerfile
FROM julia:1.9

RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN julia -e 'using Pkg; \
    Pkg.add(url="https://github.com/subhk/Tarang.jl"); \
    Pkg.precompile()'

WORKDIR /work
```
