# Tarang.jl Documentation (Getting Started)

This page is a concise newcomer guide. It walks you from a clean machine to a first Tarang run, then shows where to plug in boundary conditions, analysis, and output. Publish this page with GitHub Pages by pointing Pages at the `docs/` folder.

## Quick Navigation
- [1. Prerequisites](#1-prerequisites)
- [2. Install Tarang](#2-install-tarang)
- [3. The Tarang Workflow](#3-the-tarang-workflow)
- [4. Minimal IVP Example](#4-minimal-ivp-example)
- [5. Running with MPI](#5-running-with-mpi)
- [6. Logging & Configuration](#6-logging--configuration)
- [7. NetCDF Output & Postprocessing](#7-netcdf-output--postprocessing)
- [8. Hosting on GitHub Pages](#8-hosting-on-github-pages)
- [Style Guide](style_guide.md)

## 1. Prerequisites
- **Julia** >= 1.6
- **MPI**: OpenMPI or MPICH on your system path (`mpiexec` available).
- **HDF5**: Provided via HDF5.jl (no manual install needed on most systems).

## 2. Install Tarang
```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Tarang.jl")
```
Quick check:
```bash
julia --project -e 'using Tarang; println("Tarang loaded: ", Tarang.__version__)'
```

## 3. The Tarang Workflow
1. **Initialize MPI** (`MPI.Init()`).
2. **Set coordinates & distributor**: choose names (e.g., `"x"`, `"z"`) and an MPI process mesh.
3. **Choose bases** per coordinate (Fourier, ChebyshevT/U, Legendre) and build a `Domain`.
4. **Create fields**: scalar/vector/tensor fields that live on the domain.
5. **Define a problem** (`IVP`, `LBVP`, `NLBVP`, `EVP`) and add equations/parameters.
6. **Add boundary conditions** (Dirichlet/Neumann/Robin/stress-free/custom).
7. **Pick a timestepper** (e.g., `RK222`, `CNAB2`, `SBDF2`) and build a solver.
8. **Loop**: compute dt (often via `CFL`) and `step!` the solver.
9. **Output/analysis**: add NetCDF handlers, CFL/flow diagnostics, and logging.

## 4. Minimal IVP Example
This is a basic 2D Rayleigh-Benard-style scaffold you can extend.

```julia
using MPI, Tarang

MPI.Init()

# 1) Coordinates and distribution (2x2 MPI mesh)
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; mesh=(2, 2), device="cpu")

# 2) Bases and domain
x = RealFourier(coords["x"]; size=128, bounds=(0.0, 4.0))
z = ChebyshevT(coords["z"]; size=64,  bounds=(0.0, 1.0))
domain = Domain(dist, (x, z))

# 3) Fields
u = VectorField(dist, coords, "u", (x, z))  # velocity
p = ScalarField(dist, "p", (x, z))          # pressure
T = ScalarField(dist, "T", (x, z))          # temperature

# 4) Problem definition (IVP)
problem = IVP([u.components[1], u.components[2], p, T])
add_equation!(problem, "∂t(u) - Pr*Δ(u) + ∇(p) = -u⋅∇(u) + Ra*Pr*T*ez")
add_equation!(problem, "div(u) = 0")
add_equation!(problem, "∂t(T) - Δ(T) = -u⋅∇(T)")
problem.parameters["Pr"] = 1.0
problem.parameters["Ra"] = 1e5

# 5) Boundary conditions (Dedalus-style syntax auto-detected by add_equation!)
add_equation!(problem, "u(z=0) = 0")   # bottom
add_equation!(problem, "u(z=1) = 0")   # top
add_equation!(problem, "T(z=0) = 1")   # hot bottom
add_equation!(problem, "T(z=1) = 0")   # cold top

# 6) Solver and timestepper
solver = InitialValueSolver(problem, RK222(); dt=1e-3, device="cpu")

# 7) CFL-based adaptive timestep
cfl = CFL(problem; safety=0.5, max_change=1.2, min_change=0.5)
add_velocity!(cfl, u)

t_end = 1.0
while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
end

MPI.Finalize()
```

Key edits for your problem:
- Swap `CartesianCoordinates` for `SphericalCoordinates`/`PolarCoordinates` if needed.
- Pick bases per coordinate (Fourier for periodic; Chebyshev/Legendre for bounded).
- Add/remove fields in the `IVP` list.
- Modify equations and boundary conditions to match your PDEs.
- Choose a timestepper: `RK222`/`RK443` (IMEX Runge-Kutta), `CNAB2`/`SBDFk` (IMEX multistep).

## 5. Running with MPI
Save your script (e.g., `rbc.jl`) and launch:
```bash
mpiexec -n 4 julia --project rbc.jl
```
Tips:
- Match `Distributor(...; mesh=(Px, Pz))` to your process count (`Px*Pz = -n`).
- Use `OMP_NUM_THREADS=1` (Tarang warns if not set) for predictable performance.
- On clusters, load the MPI module that matches the MPI.jl build.

## 6. Logging & Configuration
- Logging helper: `setup_tarang_logging(level="INFO", filename="tarang.log", mpi_aware=true)`.
- Optional project config file `tarang.toml` (picked up from current dir, `~/.tarang/`, or package root):
```toml
[parallelism]
TRANSPOSE_LIBRARY = "PENCIL"

[logging]
LEVEL = "INFO"
FILE  = "tarang.log"
```
- Environment overrides (examples): `VARUNA_LOG_LEVEL=DEBUG`, `VARUNA_PROFILE_DIR=profiles`, `OMP_NUM_THREADS=1`.

## 7. NetCDF Output & Postprocessing
- Add handlers to your problem/solver (exports in `Tarang`): `NetCDFFileHandler`, `NetCDFEvaluator`, `UnifiedEvaluator`, `add_netcdf_handler`, `merge_processor_files`, `get_netcdf_info`.
- Merge per-rank files with the provided script:
```bash
mpiexec -n 4 julia scripts/merge_netcdf.jl --auto --cleanup
```
- For manual merging inside Julia, call `merge_netcdf_files` or `batch_merge_netcdf` on handler names.

## 8. Hosting on GitHub Pages
1. Commit and push this `docs/` directory to your repository's default branch.
2. In GitHub -> Settings -> Pages, set **Source** to `Deploy from branch`, branch `main` (or default), folder `/docs`.
3. (Optional) add a `docs/.nojekyll` file if you want Pages to bypass Jekyll entirely.
4. Your site will publish at `https://<username>.github.io/<repo>/` with this page as the landing index.

### Add More Pages
- Keep authoring Markdown in `docs/` (e.g., `docs/problems/navier_stokes.md`) and link to them from this index.
- Link out to deeper docs already in this repo: [nonlinear terms](nonlinear.md), [optimized linear algebra](optimized_linear_algebra.md), [NetCDF merging](netcdf_merging.md), [temporal filters for Lagrangian averaging](temporal_filters.md).
