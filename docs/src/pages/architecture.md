# Architecture and Codebase Structure

This page is a contributor map of Tarang.jl. It describes ownership and the
runtime path without duplicating type definitions that are easier to read in
the source.

## Package layout

```text
src/
├── Tarang.jl                 root module; declarative bootstrap only
├── dependencies.jl           package imports
├── load_order.jl             ordered implementation manifests (see below)
├── public_api.jl             checked supported-API registry (@public_api)
├── runtime_init.jl           MPI, FFTW, logging, and extension startup
├── api/
│   ├── public/               supported root exports, one file per capability
│   ├── namespaces.jl         Tarang.Fields / .Problems / .Solvers / ... facades
│   └── *.jl                  the facade bodies
├── core/
│   ├── architectures.jl, module_contracts.jl   CPU/GPU contract, ownership rules
│   ├── basis/                basis contracts, wavenumbers, product matrices
│   ├── boundary_conditions/  BC construction and types
│   ├── cartesian_operators/  Cartesian differential operator core, dispatch, eval
│   ├── distributor/          MPI layouts, decomposition convention, transposes
│   ├── field/                ScalarField/VectorField/TensorField
│   │   ├── field_data/       storage, copies, scales (dealiasing) — per-field data
│   │   └── field_layout/     :g/:c layout transitions and field arithmetic
│   ├── forcing/              stochastic + deterministic forcing (types, generation, application)
│   ├── nonlinear/            nonlinear products, 3/2 padding, dealiasing
│   ├── operators/            symbolic operator tree
│   │   ├── derivatives/      Fourier / polynomial derivatives, matrix apply
│   │   ├── matrices/         operator → sparse matrix builders
│   │   ├── operations/       integrate, interpolate, lift/convert
│   │   └── tensor/           grad/div/curl/Laplacian/fractional Laplacian
│   ├── problems/             equation parsing and EquationIR
│   │   └── problem_matrices/ EquationIR → global mass/linear blocks
│   ├── solvers/              solver construction, ExecutionPlan, lazy (compiled) RHS, stepping loop
│   ├── subsystems/           per-Fourier-mode subproblems, mode batching (+ KA kernels)
│   ├── timesteppers/         RK, multistep, diagonal-IMEX, ETD schemes and path selection
│   ├── transforms/           serial transforms, layout rules, GPU dispatch hooks
│   └── transpose/            TransposableField: MPI pencil transposes (pack/unpack, async)
├── tools/                    matrix solvers (sparse, GPU, batched), NetCDF I/O, checkpoints,
│   └── temporal_filters/     config, logging, parallel helpers, temporal filters
└── extras/
    └── flow_tools/           CFL, spectra, QG/streamfunction diagnostics, quick domains, plotting

ext/
└── TarangCUDAExt.jl
    └── cuda/                 device architecture, cuFFT/DCT-I transforms, Chebyshev derivative
                              kernels, batched matsolvers, NCCL transposes, memory
```

`src/load_order.jl` is the whole include order, as twelve manifests. Each one
is a flat list of `include`s owning one slice; add implementation files to the
owning manifest and never as a one-off include in `src/Tarang.jl`:

| Order | Manifest | Owns |
|---|---|---|
| 1 | `core/load_contracts.jl` | architectures, module contracts |
| 2 | `tools/load_bootstrap.jl` | general utilities, exceptions, caches, dispatch, parsing |
| 3 | `core/load_fields.jl` | coords, bases, distributor, domain, fields, field pool, arithmetic |
| 4 | `core/load_problem_stack.jl` | operators, Cartesian operators, transforms, BCs, problems, subsystems, pencil system, linalg |
| 5 | `tools/load_matsolvers.jl` | sparse, GPU, and batched matrix solvers |
| 6 | `core/load_solver_stack.jl` | solvers, stochastic forcing, timesteppers, distributed GPU, TransposableField |
| 7 | `tools/load_output.jl` | NetCDF group API and output handlers |
| 8 | `core/load_evaluation.jl` | evaluator, nonlinear products |
| 9 | `tools/load_runtime.jl` | config, arrays, parallel, logging, progress, NetCDF merge/slab I/O, checkpoints, temporal filters |
| 10 | `core/load_models.jl` | LES models |
| 11 | `extras/load_extras.jl` | flow tools, plot tools, quick domains, analysis tasks |
| 12 | `tools/load_pretty_printing.jl` | `show` methods |

Most `src/core/*.jl` files at the top level (`field.jl`, `operators/operators.jl`,
`transforms.jl`, ...) are aggregators that include the directory of the same
name; the implementation lives in the directory.

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

For an InitialValueProblem, trace these files:

1. `core/solvers/solver_types.jl` resets compiled state, parses equations,
   assembles global compatibility matrices, builds subproblems, and compiles
   the RHS plan.
2. `core/problems/problem_matrices/` converts each `EquationIR` into sparse
   mass and linear blocks.
3. `core/subsystems/` groups Fourier modes, builds small coupled systems,
   applies valid-mode filtering, and owns per-mode runtime buffers.
4. `core/solvers/lazy_rhs.jl` translates explicit expressions into a
   type-specialized evaluation tree.
5. `core/solvers/solver_execution_plan.jl` records, once, the facts every later
   decision reads: architecture (`:cpu`/`:gpu`), distribution, spectral
   structure, and whether global matrices and subproblems were assembled.
6. `core/solvers/solver_stepping.jl` refreshes dynamic boundary conditions and
   calls the timestepper dispatcher (`core/timesteppers/dispatch.jl`), which
   first runs the loud guards (stochastic-forcing compatibility, the single-GPU
   implicit-operator refusal).
7. `core/timesteppers/step_selection.jl` chooses the runtime path; the
   per-scheme `step_*!` functions then run one of the paths below.

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

### Timestepper runtime paths

Every scheme picks one of these paths from the same facts. There is never a
silent fourth option: a configuration with no correct path raises and names
the working alternative.

| Path | File | When |
|---|---|---|
| per-mode subproblem RK / multistep | `step_subproblem_rk.jl`, `step_subproblem_multistep.jl` | any coupled (Chebyshev/Jacobi) axis; CPU, MPI, and single GPU |
| batched per-mode RK | `step_subproblem_rk_batched.jl` | as above, 2D, one Fourier axis; default on GPU, `batched_modes=true` on CPU |
| global-matrix IMEX | `step_rk.jl`, `step_multistep.jl`, `step_global_matrix.jl`, `step_etd.jl` | serial CPU with no subproblems (pure Fourier) |
| explicit field path | `step_rk.jl` (`_step_explicit_rk_gpu!`), `step_multistep_field.jl` | GPU or MPI pure-Fourier problem with no implicit operator |
| serial diagonal IMEX | `step_diagonal_imex.jl` | `DiagonalIMEX_*` on a pure-Fourier problem (CPU or GPU): per-mode division by `1 + a·dt·L̂(k)` |
| distributed diagonal IMEX / ETD | `step_diagonal_imex.jl` | MPI pure-Fourier with an implicit operator: RK family, ETD family, SBDF2 |

The user-facing consequences (which scheme runs where, and what refuses) are
tabulated in [Time Steppers](timesteppers.md#Where-each-scheme-runs).

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
6. Register tests in `test/file_lists.jl` when adding a test file, and `git add`
   it: the inventory test fails on a registered file that is not tracked.
7. Update this page only when ownership or the runtime path changes.

## See also

- [Solvers](solvers.md)
- [Time Steppers](timesteppers.md)
- [GPU Computing](gpu_computing.md)
- [Tau Method](tau_method.md)
- [Testing](testing.md)
