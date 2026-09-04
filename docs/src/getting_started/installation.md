# Installation

This guide walks you through installing Tarang.jl and its dependencies.

## System Requirements

### Julia
Tarang.jl requires Julia 1.10 or later. We recommend using the latest stable release of Julia.

**Download Julia:**
- Visit [julialang.org/downloads](https://julialang.org/downloads/)
- For Linux/macOS: Use [juliaup](https://github.com/JuliaLang/juliaup) for easy version management
- For Windows: Download the installer from the Julia website

**Verify installation:**
```bash
julia --version
```

### MPI Library

Tarang.jl installs MPI.jl and a portable MPI binary artifact automatically. No
system MPI installation is needed for a workstation or a single-node run.
Install a system MPI only when you need a cluster/vendor implementation.

#### Optional system MPI: Linux (Ubuntu/Debian)
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

#### Optional system MPI: macOS
```bash
brew install open-mpi
```

#### Optional system MPI: Windows
Download and install [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi), or use WSL with Linux MPI.

**Verify the MPI implementation used by Julia:**
```julia
using MPI
MPI.versioninfo()
```

### Optional: HPC Clusters

On HPC systems, load the appropriate MPI module:
```bash
module load openmpi/4.1.0  # or your system's MPI module
```

!!! warning "MPI Compatibility"
    A launcher and its MPI library must come from the same implementation. If you
    switch MPI.jl to a cluster/system MPI, use the matching `mpiexec` (or scheduler
    launcher). See the [MPI.jl configuration guide](https://juliaparallel.org/MPI.jl/stable/configuration/).

## Installing Tarang.jl

### From GitHub (Recommended)

The package is currently hosted on GitHub and can be installed directly:

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Tarang.jl")
Pkg.add(["FFTW", "KernelAbstractions", "MPI", "NetCDF", "PencilArrays"])

using MPI
!Sys.iswindows() && MPI.install_mpiexecjl()  # Run once on Unix/macOS/WSL
```

The second line makes packages imported directly by the manual's examples
direct dependencies of your active environment. Tarang already installs them
transitively, but Julia requires a direct dependency for a top-level
`using MPI`, `using FFTW`, and similar statement in your own script.
On Unix, macOS, and WSL, `MPI.install_mpiexecjl()` installs the launcher in the
first Julia depot's `bin` directory (normally `~/.julia/bin`). Add that
directory to `PATH` if `mpiexecjl --help` is not found by your shell. Native
Windows uses the `MPI.mpiexec()` launcher pattern shown below instead; the
`mpiexecjl` wrapper itself is a Unix shell script.

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
- **NetCDF.jl**: NetCDF file I/O
- **LinearAlgebra**: Standard library (included with Julia)
- **SparseArrays**: Standard library (included with Julia)

Examples using optional packages such as CUDA.jl, Plots.jl, or Coverage.jl say
so explicitly; add only the ones you need.

### Configuring MPI.jl

If you need to use a system-provided MPI:

```julia
using Pkg
Pkg.add("MPIPreferences")  # needed once in the active environment
using MPIPreferences
MPIPreferences.use_system_binary()
```

Restart Julia after changing the preference. On a scheduler, pass its launcher
when needed, for example `use_system_binary(mpiexec="srun")` on Slurm.

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

MPI.Initialized() || MPI.Init()
println("Tarang.jl version: ", pkgversion(Tarang))
println("MPI available: ", MPI.Initialized())
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

# Put the bounded Chebyshev coordinate first for distributed mixed transforms
coords = CartesianCoordinates("z", "x")
dist = Distributor(coords; mesh=(size,), dtype=Float64, device=CPU())

x = RealFourier(coords["x"]; size=64, bounds=(0.0, 2π))
z = ChebyshevT(coords["z"]; size=32, bounds=(0.0, 1.0))

domain = Domain(dist, (z, x))

if rank == 0
    println("Successfully created domain!")
end

MPI.Finalize()
```

Run with MPI on Unix, macOS, or WSL:
```bash
mpiexecjl --project=. -n 4 julia test_tarang.jl
```

On native Windows, save this as `launch_test_tarang.jl`:

```julia
using MPI
run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. test_tarang.jl`)
```

Then run `julia --project=. launch_test_tarang.jl`. This selects the same MPI
implementation that MPI.jl loaded. If you explicitly configure Microsoft MPI
as the system implementation, its matching `mpiexec.exe` is also valid.

## Troubleshooting

### MPI Issues

**Problem**: MPI.jl cannot load the selected system MPI library

**Solution**: switch back to the portable artifact, then restart Julia:
```julia
using Pkg
Pkg.add("MPIPreferences")
using MPIPreferences
MPIPreferences.use_jll_binary()
```

**Problem**: MPI version mismatch

**Solution**: select the matching system library and launcher, then restart Julia:
```julia
using MPIPreferences
MPIPreferences.use_system_binary(mpiexec="mpiexec")
```

### Performance Issues

**Problem**: Warning about `OMP_NUM_THREADS`

**Solution**: Set the environment variable (Unix/macOS/WSL command shown):
```bash
export OMP_NUM_THREADS=1
mpiexecjl --project=. -n 4 julia your_script.jl
```

### FFTW Issues

**Problem**: FFTW planning errors

**Solution**: Use a different FFTW planning rigor in your configuration:
```julia
ENV["FFTW_PLANNING_RIGOR"] = "FFTW_ESTIMATE"
```

### NetCDF Issues

**Problem**: NetCDF artifacts are missing or incomplete

**Solution**: instantiate and precompile the active environment again:
```julia
using Pkg
Pkg.instantiate()
Pkg.precompile()
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

Load Julia and your system MPI before using Tarang. NetCDF is provided by the
bundled `NetCDF_jll` artifact, so no system NetCDF/HDF5 module is required:

```bash
module load julia/1.10
module load openmpi/4.1
```

Configure MPI.jl with `MPIPreferences.use_system_binary()` as shown above, then
restart Julia before running Tarang.

### Containers

Tarang.jl works well in containers. Example Dockerfile:

```dockerfile
FROM julia:1.10

RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN julia -e 'using Pkg; \
    Pkg.add("MPIPreferences"); \
    using MPIPreferences; \
    MPIPreferences.use_system_binary(); \
    Pkg.add(url="https://github.com/subhk/Tarang.jl"); \
    Pkg.add(["FFTW", "KernelAbstractions", "MPI", "NetCDF", "PencilArrays"]); \
    Pkg.precompile()'

WORKDIR /work
```
