# Architecture

Overview of Tarang.jl's internal architecture. This page describes how the pieces fit together — from coordinate/basis setup through problem assembly, subproblem decomposition, RHS evaluation, and time stepping. It's aimed at contributors and power users who want to understand (or extend) what happens under the hood.

## Package Structure

```
Tarang.jl/
├── src/
│   ├── Tarang.jl                     # Root module and five-stage bootstrap
│   ├── dependencies.jl               # External package imports
│   ├── load_order.jl                 # Ordered implementation loaders
│   ├── public_api.jl                 # Root-module public exports
│   ├── runtime_init.jl               # MPI, FFTW, logging, and GPU initialization
│   ├── core/                         # Numerical model and solver implementation
│   │   ├── load_*.jl                 # Stable domain-level load manifests
│   │   ├── basis/                    # Basis contracts, matrices, and wavenumbers
│   │   ├── distributor/              # MPI layouts and transposes
│   │   ├── field/                    # Field types, storage, and layouts
│   │   ├── operators/                # Symbolic, evaluated, and matrix operators
│   │   ├── problems/                 # Problem parsing and matrix assembly
│   │   ├── solvers/                  # Solver construction and RHS execution
│   │   ├── subsystems/               # Per-mode subproblem build and runtime
│   │   ├── timesteppers/             # RK, multistep, IMEX, and ETD schemes
│   │   ├── transforms/               # Serial, MPI, and backend transforms
│   │   ├── transpose/                # Distributed transpose implementation
│   │   └── nonlinear/                # Dealiasing and nonlinear evaluation
│   ├── api/
│   │   ├── public/                   # Public symbols grouped by capability
│   │   └── namespaces.jl             # Fields/Problems/Solvers facade modules
│   ├── tools/                        # Runtime, output, solver, and utility support
│   │   ├── load_*.jl                 # Tool-layer load manifests
│   │   └── temporal_filters/         # Temporal-filter implementations
│   └── extras/                       # Convenience domains and analysis helpers
│       ├── load_extras.jl            # Extras load manifest
│       └── flow_tools/               # CFL, spectra, and flow diagnostics
├── ext/
│   ├── TarangCUDAExt.jl              # CUDA weak-dependency extension
│   └── cuda/                         # GPU kernels and transform backends
├── examples/                         # Runnable simulations
├── test/                             # Serial, MPI, and GPU test suites
└── docs/                             # Documenter source and build configuration
```

This map intentionally stops at stable subsystem boundaries. For an implementation detail, open the subsystem's entry file first. Files such as `basis.jl`, `field.jl`, `problems.jl`, `solvers.jl`, and `transforms.jl` act as manifests for focused files below the same-named directory.

## Package Bootstrap

`src/Tarang.jl` keeps package startup declarative. It includes five files in order:

1. `src/dependencies.jl` imports external dependencies.
2. `src/load_order.jl` loads the internal implementation in dependency order.
3. `src/public_api.jl` defines the root-module public surface from files under `src/api/public/`.
4. `src/api/namespaces.jl` creates domain facades such as `Tarang.Fields`, `Tarang.Problems`, and `Tarang.Solvers`.
5. `src/runtime_init.jl` defines `__init__` for MPI, FFTW threads, configuration, logging, and GPU solver discovery.

The implementation order in `src/load_order.jl` is grouped by responsibility rather than by individual file:

| Loader | Responsibility |
|---|---|
| `src/core/load_contracts.jl` | Architecture and module contracts needed by every later layer |
| `src/tools/load_bootstrap.jl` | General utilities, exceptions, caches, dispatch, and parsing |
| `src/core/load_fields.jl` | Coordinates, bases, distributors, domains, fields, futures, and arithmetic |
| `src/core/load_problem_stack.jl` | Operators, transforms, BCs, problems, subproblems, and system assembly |
| `src/tools/load_matsolvers.jl` | CPU and GPU matrix-solver backends |
| `src/core/load_solver_stack.jl` | Solvers, forcing, timesteppers, and distributed field runtime |
| `src/tools/load_output.jl` | Output handlers required by evaluators |
| `src/core/load_evaluation.jl` | Evaluator and nonlinear execution |
| `src/tools/load_runtime.jl` | Configuration, parallel utilities, logging, progress, I/O, and filters |
| `src/core/load_models.jl` | Physics/model layers built on the solver stack |
| `src/extras/load_extras.jl` | Optional flow tools, plotting, quick domains, and analysis tasks |
| `src/tools/load_pretty_printing.jl` | Display methods loaded after public types exist |

When load-order bugs occur, modify the owning loader rather than adding another direct include to `src/Tarang.jl`.

## Runtime Map

If you are reading the codebase for the first time, trace one IVP step in this order:

1. `src/core/solvers/solver_stepping.jl`
   Entry point for `step!(solver, dt)`. Refreshes dynamic BCs, owns `TimestepperState`, and hands off to the timestepper layer.
2. `src/core/timesteppers/dispatch.jl`
   Thin dispatch table from timestepper type to implementation file.
3. `src/core/timesteppers/step_rk.jl` or `step_multistep.jl`
   Top-level scheme logic. For IMEX RK, this is where the code chooses between subproblem stepping, explicit fallback, and legacy global-matrix solves.
4. `src/core/timesteppers/step_subproblem_rk.jl` or `step_subproblem_multistep.jl`
   Performance-critical per-subproblem path for mixed Fourier / Chebyshev problems.
5. `src/core/subsystems/subproblem_runtime.jl`
   Manifest for the per-mode gather/scatter and BC/RHS plumbing. The concrete helpers live in `subproblem_io.jl`, `subproblem_rhs.jl`, and `subproblem_modes.jl`.
6. `src/core/solvers/lazy_rhs.jl`
   Optional RHS acceleration layer used by `evaluate_rhs`.
7. `src/core/boundary_conditions.jl`
   BC manager, dynamic BC evaluation, and projection of refreshed BC values back into equation data.

## Contributor File Map

Start from the layer that owns the behavior, then follow its manifest into a focused directory:

- `src/api/public/`: root public exports grouped by user capability; implementations remain in `core/`, `tools/`, or `extras/`
- `src/core/basis/`: basis types, wavenumbers, NCC product matrices, and derivative/conversion operators
- `src/core/distributor/`: MPI layout setup, collectives, transpose helpers, and communication-buffer management
- `src/core/field/field_data/`: raw field storage, copying/allocation, local/global sizing, and scaling
- `src/core/field/field_layout/`: layout transitions, arithmetic, filtering, and vectorized field helpers
- `src/core/operators/derivatives/`, `operations/`, `matrices/`, `tensor/`: evaluation operators separated from sparse matrix assembly
- `src/core/problems/problem_matrices/`: solver-facing matrix construction, expression analysis, spectral block builders, and legacy forcing-vector helpers
- `src/core/subsystems/`: subsystem grouping, per-mode subproblem construction, runtime I/O/BC helpers, and NCC assembly
- `src/core/solvers/`: solver state, stepping entry points, interpreted/compiled RHS paths, and lazy RHS evaluation
- `src/core/timesteppers/`: timestepper types, dispatch, state management, and scheme implementations
- `src/core/transforms/` and `src/core/transpose/`: spectral transforms and distributed data movement, respectively
- `src/core/nonlinear/`: padding, transform setup, dealiasing, pencil compatibility helpers, and nonlinear evaluation
- `src/tools/`: support code used across core layers; read the appropriate `load_*.jl` file to find its load boundary
- `src/extras/flow_tools/`: CFL control, diagnostics, spectra, streamfunction/SQG helpers, QG tools, and boundary-advection helpers
- `src/tools/temporal_filters/`: temporal-filter core types, IMEX/ETD updates, wave-mean helpers, and GQL utilities
- `ext/cuda/`: CUDA-only implementations activated through `ext/TarangCUDAExt.jl`; core code must remain loadable without CUDA

Tests mirror these ownership boundaries where practical. Use `test/file_lists.jl` to see the serial, MPI, optional, and GPU inventories used by CI.

## How the Solver Actually Works

This section walks through the life of a step! call end-to-end, so you can see where each file fits.

### 1. Setup phase

```
CartesianCoordinates ─► Distributor ─► Bases ─► Domain ─► Fields ─► Problem
```

- **`CartesianCoordinates("x", "z")`** names the axes.
- **`Distributor(coords; dtype, device)`** owns the MPI communicator, architecture (CPU / GPU), and cached layout plans for different basis tuples.
- **`RealFourier(...)`**, **`ChebyshevT(...)`** define the spectral bases.
- **`Domain(dist, (xbasis, zbasis))`** ties a distributor to an ordered tuple of bases and allocates per-rank grids.
- **`ScalarField`**, **`VectorField`**, **`TensorField`** hold data in either grid space (real values on Gauss-Lobatto / uniform grids) or coefficient space (Fourier + Chebyshev coefficients). The active layout is tracked by `current_layout::Symbol` and switched via `ensure_layout!(field, :c)` / `(:g)`.
- **`IVP([vars...])`**, **`LBVP`**, **`NLBVP`**, **`EVP`** group state variables into a `Problem` and build a parser namespace from the variables' names.

### 2. Equation and BC assembly

Users add equations via string syntax:

```julia
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_bc!(problem, "T(z=0) = 1")
```

`add_equation!` pushes the string to `problem.equations`. `add_bc!(problem, bc::String)` pushes the string to `problem.boundary_conditions` **and** auto-parses it into a `DirichletBC` / `NeumannBC` object registered in `problem.bc_manager.conditions`, with `is_time_dependent` / `is_space_dependent` flags auto-detected from the value expression. This is what enables the per-step refresh path for `T(z=0) = sin(t)` or `T(z=0) = sin(2·π·x/Lx)`.

At solver build time, `_merge_boundary_conditions!` pushes each BC string into `problem.equations` (so the equation parser sees it) and links the parsed `DirichletBC` back to its equation index via `bc_manager.bc_equation_indices`.

### 3. Global matrix build

`build_solver_matrices!` → `build_matrices(problem)` → `build_matrix_expressions!(problem)`.

This logic now lives under `src/core/problems/problem_matrices/`. The entry file `problem_matrices.jl` is only a manifest; the actual work is split across:

- `problem_matrices_build.jl` for top-level assembly
- `problem_matrices_expr_analysis.jl` for equation-variable analysis and operator splitting
- `problem_matrices_spectral.jl` for spectral block construction
- `problem_matrices_support.jl` and `problem_matrices_legacy.jl` for helpers and compatibility shims

For each equation (including merged BCs), the equation-string parser splits LHS and RHS:

- LHS is parsed into an operator tree with `M` (time-derivative) and `L` (spatial) parts
- RHS is parsed into an `F` expression (handled specially per equation)

The result is a `Vector{Dict}` — `problem.equation_data` — where each entry holds `"M"`, `"L"`, `"F"`, `"lhs"`, and `"equation_size"` keys.

For the global-matrix path (used by legacy steppers and BVPs), `build_matrices` walks `equation_data` and assembles sparse `L_matrix`, `M_matrix`, and a dense `F_vector`. The result is stored in `problem.parameters["L_matrix"]`, `["M_matrix"]`, `["F_vector"]`.

### 4. Subproblem decomposition

This is the core of the modern solver path. `_try_build_subproblems!(solver)` calls `build_subproblems` in `subsystems.jl`. The implementation is now split by responsibility:

- `subproblem_build_orchestration.jl` constructs subproblems and orchestrates per-group assembly
- `subproblem_expr_helpers.jl` handles expression helpers, DOF sizing, and valid-mode checks used during build
- `subproblem_matrix_build.jl` builds the small sparse `L` / `M` blocks
- `subproblem_runtime.jl` delegates per-step I/O, BC projection, and mode checks to `subproblem_io.jl`, `subproblem_rhs.jl`, and `subproblem_modes.jl`

At runtime, the build path does the following:

1. **Enumerates Fourier-mode groups.** For each separable (Fourier) axis, each mode is a "subproblem group". In a 2D problem with `Nx = 256` RealFourier modes and one Chebyshev direction, there are `Nx/2 + 1 = 129` subproblems.
2. **Builds per-subproblem matrices.** For each subproblem, `build_matrices!(sp, ...)` walks the problem's equation list and calls `expression_matrices(expr, sp, vars)` to construct small sparse `L` and `M` matrices — typically of shape `(n_eqs_rows, n_var_cols)` where the sizes are the per-Fourier-mode DOF counts (e.g. `263 × 263` for 2D RBC at `Nz = 64`).
3. **Applies left / right permutations** (`left_permutation`, `right_permutation`) to group equations and variables by their domain dimension, matching the Dedalus subsystem ordering convention.
4. **Applies valid-mode filtering.** Rows that are identically zero in both `L` and `M` — typically trivially-satisfied gauge constraints like `integ(p) = 0` at non-DC Fourier modes — are paired with the "least-used" 1-DOF tau column (by smallest `|L| + |M|` norm) and both are dropped from the filtered system. This yields a smaller square sparse system (`L_min`, `M_min`) that a sparse LU can factor cleanly.
5. **Classifies rows by equation size.** `sp.bulk_rows` holds row indices where `eq_size ≥ Nz` (PDE rows and Nz-sized algebraic rows like continuity); `sp.bc_rows` holds row indices where `eq_size < Nz` (BC rows and gauge rows). The IMEX stepper uses `sp.bc_rows` to target algebraic-constraint rows for the per-stage `apply_bc_override!` path.

The output — a `Tuple{Vararg{Subproblem}}` stored in `problem.parameters["subproblems"]` — is what the stepper consumes.

### 5. Lazy RHS plan

`build_lazy_rhs_plan!(solver)` walks each equation's `F` expression and translates it into a type-parameterized `LazyFuture` tree (see `src/core/solvers/lazy_rhs.jl`). Each node (`LazyAdd`, `LazyMul`, `LazyDiff`, `LazyStateField`, `LazyParamField`, `LazyConst`) has a specialized `evaluate_lazy!` method. At first call, Julia's JIT specializes the entire `evaluate_lazy!` chain — eliminating dynamic dispatch and enabling broadcast fusion across arithmetic combinators.

If translation fails for any equation (unsupported operator type), the plan's `is_compiled` stays false and `evaluate_rhs` falls back to the interpreted expression path (slower but universally functional).

The plan is stored as `solver.rhs_plan::LazyRHSPlan` and used by `evaluate_rhs(solver, state, time)` during every stage of every step.

### 6. Time step

A single `step!(solver, dt)` call does:

1. **Refresh dynamic BCs** (`solver_stepping.jl:23`). If `has_time_dependent_bcs(bcm)` is true, call `update_time_dependent_bcs!(bcm, t+dt)` and `_apply_bc_values_to_equations!(solver, t+dt)`, which rewrites `equation_data[eq_idx]["F"]` with a fresh `ConstantOperator` or `ArrayOperator` for every time/space-dependent BC.
2. **Dispatch by timestepper type** (`timesteppers/dispatch.jl`). `RK222()` → `step_rk_imex!` → if `problem.parameters["subproblems"]` exists, call `step_subproblem_rk!(state, solver, sps)`. For `CNAB2`/`SBDF2` the dispatch is similar via `step_subproblem_multistep!`.
3. **Per-stage subproblem solve** (`step_subproblem_rk.jl`). For each RK stage:
   - Refresh `ALG_F` per stage (re-run `gather_alg_F!` after re-calling `update_time_dependent_bcs!` at `t + c[i]·dt`). This recovers full stage-order accuracy for rapidly-varying BCs.
   - Per subproblem: gather inputs, build stage RHS via `dt·Σ(Aᴱ·F − Aⁱ·L·X)`, then `apply_bc_override!` to overwrite the `sp.bc_rows` entries with `dt·a_ii · F_BC` so the BC row enforces `L_row·X = F_BC` instead of the 1/γ-scaled value that the raw accumulated formula would give.
   - Solve `(M_min + dt·a_ii·L_min) · X_stage = rhs` using the cached LHS solver (sparse LU for CPU, `CuSparseLU` for GPU).
   - `scatter_inputs` writes `X_stage` back to the state fields.
   - Evaluate `F_stage` and `L·X_stage` for use in the next stage.
4. **Final update**. For singular `M` (DAE system), the last stage value is kept. For non-singular `M`, the standard weighted-sum update solves `M · X_{n+1} = M · X_n + dt·Σ(bᴱ·F − bⁱ·L·X)` via the cached mass-matrix LU.

The same pattern applies to the multistep stepper (`step_subproblem_multistep.jl`), except the per-stage loop is replaced by per-subproblem history accumulation and a single `(a[0]·M + b[0]·L) · X = rhs` solve per step.

## Core Types

### Coordinates

```julia
abstract type Coordinates end

struct CartesianCoordinates <: Coordinates
    names::Vector{String}
    coords::Vector{Coordinate}
end
```

### Bases

```julia
abstract type Basis end
abstract type FourierBasis <: Basis end
abstract type JacobiBasis <: Basis end

struct RealFourier   <: FourierBasis  end
struct ComplexFourier <: FourierBasis end
struct ChebyshevT    <: JacobiBasis   end
struct ChebyshevU    <: JacobiBasis   end
struct Legendre      <: JacobiBasis   end
```

Every basis carries a `meta::BasisMeta` with `size`, `bounds`, `element_label`, `dealias`, and cached conversion matrices.

### Fields

```julia
mutable struct ScalarField{T, S<:AbstractFieldStorage} <: Operand
    dist::Distributor
    name::String
    bases::Tuple{Vararg{Basis}}
    domain::Union{Nothing, Domain}
    dtype::Type{T}
    storage::S
    layout::Union{Nothing, Layout}
    current_layout::Symbol           # :g (grid) or :c (coefficient)
    scales::Union{Nothing, Tuple}
    # ...
end
```

`VectorField` and `TensorField` hold `components::Vector{ScalarField}` / `Matrix{ScalarField}`.

### Subproblem

```julia
mutable struct Subproblem
    solver::Any
    problem::Problem
    subsystems::Tuple{Vararg{Subsystem}}
    group::Tuple
    dist::Any
    domain::Any
    dtype::DataType
    group_dict::Dict{String, Any}

    variable_range::UnitRange{Int}
    equation_range::UnitRange{Int}

    matrices::Dict{String, Any}  # build-time matrices only

    pre_left::Union{Nothing, SparseMatrixCSC}
    pre_left_pinv::Union{Nothing, SparseMatrixCSC}
    pre_right::Union{Nothing, SparseMatrixCSC}
    pre_right_pinv::Union{Nothing, SparseMatrixCSC}

    L_min::Union{Nothing, SparseMatrixCSC}
    M_min::Union{Nothing, SparseMatrixCSC}
    L_exp::Union{Nothing, SparseMatrixCSC}
    M_exp::Union{Nothing, SparseMatrixCSC}
    LHS::Union{Nothing, SparseMatrixCSC}

    update_rank::Int
    _input_buffer::Union{Nothing, Matrix{ComplexF64}}
    _output_buffer::Union{Nothing, Matrix{ComplexF64}}

    runtime::SubproblemRuntimeCache
    LHS_solvers::Dict{Float64, Any}

    bulk_rows::Vector{Int}
    bc_rows::Vector{Int}
    bulk_cols::Vector{Int}
    bc_cols::Vector{Int}
end
```

`SubproblemRuntimeCache` holds the per-mode scratch vectors, backend-adapted matrices, BC gather buffers, and RK stage buffers that used to be mixed into generic dictionaries.

### Timestepper

```julia
abstract type TimeStepper end

struct RK222 <: TimeStepper
    stages::Int
    A_explicit::Matrix{Float64}
    b_explicit::Vector{Float64}
    c_explicit::Vector{Float64}
    A_implicit::Matrix{Float64}
    b_implicit::Vector{Float64}
    c_implicit::Vector{Float64}
end
```

All RK variants carry their full Butcher tableau. Multistep variants (`CNAB1/2`, `SBDF1..4`) expose coefficient builders (`_cnab1_coefs(dt)`, `_sbdf2_coefs(dt)`, etc.) in `step_subproblem_multistep.jl`.

### Solver

```julia
mutable struct InitialValueSolver
    base::SolverBaseData
    problem::IVP
    timestepper::TimeStepper
    sim_time::Float64
    iteration::Int
    stop_sim_time::Float64
    stop_wall_time::Float64
    stop_iteration::Int
    state::Vector{<:ScalarField}
    dt::Float64
    timestepper_state::Union{Nothing, AbstractTimestepperState}
    evaluator::Any              # Evaluator or nothing
    wall_time_start::Float64
    performance_stats::SolverPerformanceStats
    rhs_plan::Any                # Lazy RHS plan or nothing
end
```

## Data Flow Diagrams

### Simulation lifecycle

```
Setup       : Coordinates → Distributor → Bases → Domain → Fields
Problem     : Fields → Problem → add_equation!, add_bc!, add_parameters!
Build       : build_solver_matrices! → _try_build_subproblems! → build_lazy_rhs_plan!
Time step   : step! → refresh BCs → dispatch → step_subproblem_rk! / multistep
Output      : evaluator → add_task! → NetCDF / callback
```

### Layout transitions

```
Grid space (physical)
    ↓ forward_transform! (FFTW.rfft + DCT-I, or CUFFT on GPU)
Coefficient space (spectral)
    ↓ apply operators (matrix-free in subproblem path)
Coefficient space
    ↓ backward_transform!
Grid space
```

### MPI communication

- **All-to-all** — layout transposes between pencils via `PencilFFTPlan`.
- **Allreduce** — global reductions for energy, CFL, diagnostics.
- **Allgatherv** — assembling subproblem-local data when a global view is needed (rare; most of the step is rank-local because subproblems are per-Fourier-mode).

Under the subproblem architecture, each MPI rank owns a subset of Fourier modes. Every rank iterates its own subproblems independently — no inter-rank communication inside the stepper loop. The only rank-to-rank traffic is during the forward/backward FFTs and the optional global reductions in user callbacks.

## Extension Points

### Custom basis

Subtype `Basis` (typically `JacobiBasis` for bounded-interval bases). Implement:

- `basis.meta::BasisMeta` with `size`, `bounds`, `element_label`, `dealias`
- `_native_grid(basis, scale)` — Gauss-Lobatto or uniform collocation nodes
- `derivative_basis(basis)` — the basis the derivative lives in
- `differentiation_matrix(basis)`, `conversion_matrix(a, b)`, `_apply_forward` / `_apply_backward`

Register with `register_operator_alias!` / `register_operator_parseable!` if it introduces new operators.

### Custom timestepper

Subtype `TimeStepper`. Implement:

- Constructor that populates Butcher tableau fields (for RK variants) or coefficient builders (for multistep)
- A new dispatch entry in `src/core/timesteppers/dispatch.jl` mapping the type to a stepping function
- A stepping function — either reuse `step_subproblem_rk!` with a new tableau, or write a dedicated `step_<name>!` that takes a `TimestepperState` and `InitialValueSolver`

### Custom operator

Subtype `Operator`. Implement:

- `subproblem_matrix(op::NewOp, sp; kwargs...)` — returns a sparse matrix column/block for per-subproblem assembly
- `evaluate_lazy!` methods if the operator should participate in the lazy RHS plan
- `expression_matrices(op::NewOp, sp, vars; kwargs...)` for arithmetic combinators (`Add`/`Subtract`/`Multiply` wrappers)

## Design Principles

1. **Per-subproblem decomposition**: the solver works one Fourier mode at a time, producing small sparse systems that an LU can crack efficiently. Gauge and BC rows are handled by valid-mode filtering and the `apply_bc_override!` DAE override.
2. **Lazy / JIT-specialized RHS**: the RHS plan is a type-parametric tree; the JIT specializes it to avoid dynamic dispatch.
3. **MPI transparency**: users never touch the communicator — `Distributor` does.
4. **Device transparency**: `CPU()` vs `GPU()` is a constructor argument. The same code path runs on both.
5. **Dedalus-style tau method**: BCs are enforced via explicit tau fields and `lift()` operators added to equations, not by row replacement.

## See Also

- [Tau Method](tau_method.md): full explanation of how BCs are enforced
- [Time Steppers](timesteppers.md): available schemes and when to use them
- [Solvers](solvers.md): the `InitialValueSolver` / BVP / EVP API
- [Contributing](contributing.md): development guidelines
- [Testing](testing.md): test architecture
