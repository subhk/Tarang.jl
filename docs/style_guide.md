# Tarang User Guide

This guide provides an overview of the Tarang framework. Each section describes the key concepts and entry points.

## Domains & Bases
- Coordinates: `CartesianCoordinates`, `PolarCoordinates`, `SphericalCoordinates` (or `Libraries.SphericalCoordinates` for advanced spherical utilities).
- Bases: `RealFourier`, `ComplexFourier`, `ChebyshevT`, `ChebyshevU`, `Legendre`, `Ultraspherical`, `Jacobi`, `DiskBasis`, `AnnulusBasis`.
- Domain: `Domain(dist, (basis1, basis2, ...))`.
- Distribution: `Distributor(coords; mesh=(Px, Py, ...), dtype, device)` for processor meshes.

## Problems & Boundary Conditions
- Problems: `IVP`, `LBVP`, `NLBVP`, `EVP`.
- Equations & BCs: Use `add_equation!(problem, "...")` for both PDEs and boundary conditions. Dedalus-style BC syntax `field(coord=value)` is auto-detected and converted to `Interpolate` operators.
- Time/space-dependent values via string expressions (t, x, y, z) or `TimeDependentValue`/`SpaceDependentValue`.
- Tau/lift: automatic through `BoundaryConditionManager`; `register_tau_field!` for custom tau fields.

## Operators & Nonlinear Terms
- Differential operators: `grad`, `div`, `curl`, `lap`, `trace`, `skew`, `transpose_components`.
- Nonlinear operators: `advection`, `nonlinear_momentum`, `convection`, with helpers `evaluate_nonlinear_term`, `evaluate_transform_multiply`.
- Parsing: equation strings follow LHS (linear) / RHS (nonlinear) conventions.

## Solvers & Timesteppers
- Solvers: `InitialValueSolver(problem, timestepper)`, `BoundaryValueSolver(problem)`, `EigenvalueSolver(problem)`.
- Timesteppers: `RK111`, `RK222`, `RK443` (IMEX Runge-Kutta), `CNAB1`, `CNAB2`, `SBDF1`-`SBDF4` (IMEX multistep). All methods treat linear terms implicitly and nonlinear terms explicitly.
- CFL helper: `CFL(problem; ...)` then `add_velocity!(cfl, u)` and `compute_timestep(cfl)`.

## Analysis & Output
- File handlers: `NetCDFFileHandler` or  `add_file_handler(base_path, dist, vars; sim_dt=..., max_writes=..., parallel="gather")`.
- Tasks: `add_task!(handler, expr_or_field; name, layout, scales)` or alias `add_task(...)`.
- Analysis helpers: `add_mean_task!`, `add_slice_task!`, `add_profile_task!` for common reductions/slices.
- Merge outputs: `scripts/merge_netcdf.jl --auto --cleanup` (naming: `handler_s1/handler_s1_p0.nc`).
- Plot/flow helpers: `extras/plot_tools.jl`, `extras/flow_tools.jl`, `extras/quick_domains.jl`.

## Configuration & Logging
- Config file: `tarang.toml` (picked from cwd, `~/.tarang/`, or package root). Set transpose library, logging, profiling for configuration.
- Logging: `setup_tarang_logging(level="INFO", filename="tarang.log", mpi_aware=true)`. Env overrides: `VARUNA_LOG_LEVEL`, `VARUNA_PROFILE_DIR`, `OMP_NUM_THREADS`.

## Cluster & Parallel How-To
- MPI parallelism: use `Distributor(...; mesh=(Px, Py, Pz))` matching `mpiexec -n Px*Py*Pz`.
- Best practices: set `OMP_NUM_THREADS=1`, ensure MPI build matches cluster modules, and use `parallel="gather"` or per-rank NetCDF writes depending on I/O needs.
