# Tarang.jl Documentation

The manual is built from [`docs/src/index.md`](src/index.md) with Documenter.jl.
Use the [development manual](https://subhk.github.io/Tarang.jl/dev/) for `main`
and the [stable manual](https://subhk.github.io/Tarang.jl/stable/) for tagged releases.
Julia 1.10 or later is required.

## Start here

- [Installation](src/getting_started/installation.md)
- [First steps](src/getting_started/first_steps.md)
- [Problem types and solvers](src/pages/problems.md)
- [Boundary conditions](src/tutorials/boundary_conditions.md)
- [Time-stepper support by backend](src/pages/timesteppers.md#where-each-scheme-runs)
- [CPU threads and MPI](src/pages/parallelism.md)
- [GPU computing](src/pages/gpu_computing.md)
- [Testing](src/pages/testing.md)

The public problem names are `InitialValueProblem`, `LinearBoundaryValueProblem`,
`NonlinearBoundaryValueProblem`, and `EigenvalueProblem`. The abbreviated aliases
have been removed. The guides cover spatial and moving boundary values,
registered parameters, stress-free component conditions, and the supported
CPU/GPU solver paths.

## Build the website locally

From the repository root:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html` after the build. GitHub Actions builds PR previews;
merging to `main` updates the development manual, while tags update release
versions. GitHub Pages serves the Documenter deployment, rather than this source
folder directly.
