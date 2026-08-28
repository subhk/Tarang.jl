# Architecture and Codebase Structure

This page is a contributor map of Tarang.jl. It describes ownership and the
runtime path without duplicating type definitions that are easier to read in
the source.

## Package layout

```text
src/
├── Tarang.jl                 root module; declarative bootstrap only
├── dependencies.jl           package imports
├── load_order.jl             ordered implementation manifests
├── public_api.jl             checked supported-API registry
├── runtime_init.jl            MPI, FFTW, logging, and extension startup
├── api/
│   ├── public/               supported root exports by capability
│   └── *.jl                  Fields/Problems/Solvers/... facades
├── core/
│   ├── basis/                basis contracts and spectral matrices
│   ├── boundary_conditions/  BC construction and types
│   ├── cartesian_operators/  Cartesian differential operator core, dispatch, and eval
│   ├── distributor/          MPI layouts and communication
│   ├── field/                field storage and layout transitions
│   ├── forcing/              stochastic forcing generation and application
│   ├── operators/            symbolic and evaluated operators
│   ├── problems/             parsing, EquationIR, and matrix assembly
│   ├── solvers/              solver construction and compiled RHS
│   ├── subsystems/           per-mode systems and runtime buffers
│   ├── timesteppers/         RK, multistep, IMEX, and ETD schemes
│   ├── transforms/           serial and distributed transforms
│   ├── transpose/            MPI pencil transpose (pack/unpack, async, buffers)
│   └── nonlinear/            nonlinear evaluation and dealiasing
├── tools/                    matrix solvers, output, configuration, utilities
└── extras/                   flow diagnostics and convenience features

ext/
└── TarangCUDAExt.jl
    └── cuda/                  CUDA allocation, kernels, transforms, and bindings
```

`src/load_order.jl` loads stable subsystem manifests. Add implementation files
to the owning manifest; do not add one-off includes to `src/Tarang.jl`.

## Dependency direction

```text
contracts and utilities
        ↓
fields, bases, distributors
        ↓
operators and problems
        ↓
compiled artifacts and subproblems
        ↓
solvers and timesteppers
        ↓
models, output, and extras

CUDA extension ──implements──► backend hooks declared by core
public API     ──exposes───► selected bindings from all layers
```

Core must load without CUDA. CUDA-specific module bindings for CUSOLVER and
CUSPARSE are methods supplied by `TarangCUDAExt`; core owns only the solver
contracts and backend-neutral orchestration.

## Public API boundary

Supported root exports are declared with `@public_api` under
`src/api/public/`. The macro both exports each name and registers it in the
checked manifest returned by:

```julia
Tarang.public_api_names()
Tarang.is_public_api(:InitialValueSolver)  # true
```

Implementation files still contain compatibility exports from releases before
the boundary existed. Treat those as legacy, not as permission to grow the
root API. New supported names belong in one public capability file and, when
appropriate, in a facade such as `Tarang.Fields` or `Tarang.Solvers`.

## Problem compilation lifecycle

Problem construction has three distinct kinds of state:

| State | Owner | Purpose |
|---|---|---|
| User configuration | `problem.parameters` | coefficients and user-supplied objects |
| Parsed equations | `problem.equation_data::Vector{EquationIR}` | named `mass`, `linear`, `forcing`, `lhs`, and equation size slots plus metadata |
| Solver artifacts | `problem.compiled::CompiledProblem` | assembled matrices, subproblems, coefficient systems, and runtime caches |

`EquationIR` temporarily implements `AbstractDict{String,Any}` so downstream
code using keys such as `"M"` continues to work. Internal code should prefer
the named fields. Likewise, matrix and subproblem entries are mirrored into
`problem.parameters` for compatibility, but runtime code reads
`problem.compiled` as the canonical owner.

`reset_compiled_problem!` clears matrices, subproblems, and its
`RuntimeCacheContext` before rebuilding. Per-problem caches therefore cannot
leak through user parameters or be reused by an unrelated solver run.

## Solver build and step path

For an IVP, trace these files:

1. `core/solvers/solver_types.jl` resets compiled state, parses equations,
   assembles global compatibility matrices, builds subproblems, and compiles
   the RHS plan.
2. `core/problems/problem_matrices/` converts each `EquationIR` into sparse
   mass and linear blocks.
3. `core/subsystems/` groups Fourier modes, builds small coupled systems,
   applies valid-mode filtering, and owns per-mode runtime buffers.
4. `core/solvers/lazy_rhs.jl` translates explicit expressions into a
   type-specialized evaluation tree.
5. `core/solvers/solver_stepping.jl` refreshes dynamic boundary conditions and
   calls the timestepper dispatcher.
6. `core/timesteppers/step_subproblem_rk.jl` or
   `step_subproblem_multistep.jl` gathers, solves, and scatters each mode.

The resulting flow is:

```text
equation strings
    ↓ parse
EquationIR
    ↓ compile
CompiledProblem {global matrices, subproblems, caches}
    ↓ construct
InitialValueSolver {RHS policy, lazy plan, timestep state}
    ↓ step!
refresh BCs → evaluate RHS → per-mode solve → update fields
```

## RHS execution policy

`rhs_fallback=:auto` resolves per solver:

| Execution | Effective policy |
|---|---|
| Serial CPU | `:interpreted` compatibility is allowed |
| GPU | `:strict`; an uncompiled RHS is an error |
| MPI | `:strict`; an uncompiled RHS is an error |

Use `rhs_fallback=:strict` to require compilation on serial CPU too. Use
`:interpreted` only for a verified CPU or supported MPI compatibility case.
GPU state rejects `:interpreted` explicitly, and distributed all-Fourier
interpreted execution is rejected unconditionally because it is not correct.

This rule is broader than matrix-solver selection: a GPU field cannot select a
CPU-only coupled solver, and `:gpu` never silently degrades to a CPU solver.
NetCDF output is an explicit host I/O boundary, not a computational fallback.

## GPU ownership

The core/extension split is:

| Concern | Core | CUDA extension |
|---|---|---|
| Architecture contract | `AbstractArchitecture`, `GPU`, dispatch hooks | CUDA device and array methods |
| Fourier transforms | field/layout contract | cuFFT plans and execution |
| Mixed transforms | basis/operator selection | cached Fourier–Chebyshev plans and DCT kernels |
| Matrix solves | solver types, selection policy, reusable buffers | CUDA allocation plus CUSOLVER/CUSPARSE bindings |
| Output | scheduling and NetCDF staging contract | device-to-host bulk copy methods |

Supported single-GPU IVPs are 2D/3D pure Fourier and mixed
Fourier–Chebyshev layouts. Their transforms, RHS evaluation, and coupled
subproblem solves remain device-resident after warm-up. Unsupported layouts
raise an error.

## MPI data movement

Per-mode linear solves are rank-local. Communication surrounds them:

- pure Fourier problems communicate inside distributed FFTs;
- mixed Fourier–Chebyshev problems additionally transpose between the FFT
  pencil and solve layout once per stage or step;
- diagnostics use collective reductions;
- output may gather or write rank-local files according to its handler.

Collectives must remain outside the per-subproblem loop and every rank must
issue them in the same order.

### Which axes are decomposed

One function answers this for the whole codebase:

```julia
decomposed_axes(dist, ndim)   # global axis indices that are split, ascending
mesh_axis_for(dist, ndim, axis)   # which mesh dimension splits `axis`, or nothing
```

The two conventions it encodes differ: with PencilArrays the **last**
`length(mesh)` axes are decomposed, and with `TransposableField` (GPU+MPI) the
**first** ones are. Both live in `src/core/distributor/distributor_core.jl` and
nowhere else.

Do not re-derive the rule at a call site. It was previously written out by hand
in seventeen places, and two of those copies drifted apart — the array allocator
and the index math disagreed about which axes were split, so a field's shape and
the meaning of its indices no longer matched, with no error raised.
`test_decomposition_convention.jl` scans `src/` for hand-rolled copies, checking
the arithmetic as well as the comments, and fails if one reappears.

`ndim` is the *field's* dimensionality, which is not always `dist.dim`; pass the
one you mean.

## Extension checklist

When adding a feature:

1. Put implementation in the owning core/tool/extension directory.
2. Keep dependency direction downward; do not make core depend on an API
   facade or on CUDA.
3. Store compiled or temporary state in `CompiledProblem`,
   `RuntimeCacheContext`, or a typed subsystem cache, not in user parameters.
4. Add a lazy-RHS translation or make unsupported execution fail explicitly.
5. Declare supported user-facing names with `@public_api` and update the
   relevant facade.
6. Register tests in `test/file_lists.jl` when adding a test file.
7. Update this page only when ownership or the runtime path changes.

## See also

- [Solvers](solvers.md)
- [Time Steppers](timesteppers.md)
- [GPU Computing](gpu_computing.md)
- [Tau Method](tau_method.md)
- [Testing](testing.md)
