# Architecture

Overview of Tarang.jl's internal architecture. This page describes how the pieces fit together — from coordinate/basis setup through problem assembly, subproblem decomposition, RHS evaluation, and time stepping. It's aimed at contributors and power users who want to understand (or extend) what happens under the hood.

## Package Structure

```
Tarang.jl/
├── src/
│   ├── Tarang.jl                     # Main module
│   ├── core/
│   │   ├── architectures.jl            # CPU / GPU architecture abstraction
│   │   ├── coords.jl                   # Coordinate systems
│   │   ├── basis.jl                    # Manifest for basis/*
│   │   ├── basis/
│   │   │   ├── basis_core.jl
│   │   │   ├── basis_wavenumbers.jl
│   │   │   ├── basis_product_matrices.jl
│   │   │   ├── basis_operators.jl
│   │   │   └── basis_interface.jl
│   │   ├── distributor.jl              # Manifest for distributor/*
│   │   ├── distributor/
│   │   │   ├── distributor_core.jl
│   │   │   ├── distributor_mpi.jl
│   │   │   ├── distributor_transpose.jl
│   │   │   ├── distributor_exchange.jl
│   │   │   └── distributor_grouped_transpose.jl
│   │   ├── domain.jl
│   │   ├── field/
│   │   │   ├── field_types.jl
│   │   │   ├── field_data.jl           # Manifest for field_data/*
│   │   │   ├── field_data/
│   │   │   ├── field_layout.jl         # Manifest for field_layout/*
│   │   │   ├── field_layout/
│   │   │   └── field_exports.jl
│   │   ├── operators/
│   │   │   ├── derivatives.jl          # Manifest for derivatives/*
│   │   │   ├── operations.jl           # Manifest for operations/*
│   │   │   ├── matrices.jl             # Manifest for matrices/*
│   │   │   ├── tensor.jl               # Manifest for tensor/*
│   │   │   └── ...
│   │   ├── transforms/
│   │   ├── problems/
│   │   │   ├── problem_types.jl
│   │   │   ├── problem_parsing.jl
│   │   │   ├── problem_matrices.jl     # Manifest for problem_matrices/*
│   │   │   ├── problem_matrices/
│   │   │   └── problem_utils.jl
│   │   ├── boundary_conditions.jl
│   │   ├── subsystems.jl               # Manifest for subsystem/subproblem build + runtime
│   │   ├── subsystems/
│   │   │   ├── subsystem_types.jl
│   │   │   ├── subsystem_methods.jl
│   │   │   ├── subproblem_types.jl
│   │   │   ├── subproblem_runtime.jl   # Manifest for I/O, BC/RHS gather, mode checks
│   │   │   ├── subproblem_io.jl
│   │   │   ├── subproblem_rhs.jl
│   │   │   ├── subproblem_modes.jl
│   │   │   ├── subproblem_build.jl     # Manifest for construction + matrix assembly
│   │   │   ├── subproblem_build_orchestration.jl
│   │   │   ├── subproblem_expr_helpers.jl
│   │   │   ├── subproblem_matrix_build.jl
│   │   │   ├── subproblem_permutations.jl
│   │   │   ├── subproblem_matrix_utils.jl
│   │   │   ├── subproblem_ncc.jl
│   │   │   └── subsystem_exports.jl
│   │   ├── solvers/
│   │   ├── timesteppers/
│   │   ├── evaluator.jl
│   │   ├── nonlinear.jl               # Manifest for nonlinear/*
│   │   └── nonlinear/
│   ├── tools/
│   │   ├── matsolvers.jl
│   │   ├── gpu_matsolvers.jl
│   │   ├── netcdf_output.jl
│   │   ├── temporal_filters.jl        # Manifest for temporal_filters/*
│   │   └── temporal_filters/
│   └── extras/
│       ├── quick_domains.jl
│       ├── flow_tools.jl              # Manifest for flow_tools/*
│       ├── flow_tools/
│       └── plot_tools.jl
├── ext/
│   ├── TarangCUDAExt.jl
│   └── cuda/
├── examples/
├── test/
└── docs/
```

Several top-level files are now thin manifest entry points. When a file and a same-named directory both exist, read the top-level file first; it tells you which focused implementation files to open next.

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

The current tree is easier to navigate if you think in terms of focused directories rather than historic monolithic files:

- `src/core/basis/`: basis types, wavenumbers, NCC product matrices, and derivative/conversion operators
- `src/core/distributor/`: MPI layout setup, collectives, transpose helpers, and communication-buffer management
- `src/core/field/field_data/`: raw field storage, copying/allocation, local/global sizing, and scaling
- `src/core/field/field_layout/`: layout transitions, arithmetic, filtering, and vectorized field helpers
- `src/core/operators/derivatives/`, `operations/`, `matrices/`, `tensor/`: evaluation operators separated from sparse matrix assembly
- `src/core/problems/problem_matrices/`: solver-facing matrix construction, expression analysis, spectral block builders, and legacy forcing-vector helpers
- `src/core/subsystems/`: subsystem grouping, per-mode subproblem construction, runtime I/O/BC helpers, and NCC assembly
- `src/core/nonlinear/`: padding, transform setup, dealiasing, pencil compatibility helpers, and nonlinear evaluation
- `src/extras/flow_tools/`: CFL control, diagnostics, spectra, streamfunction/SQG helpers, QG tools, and boundary-advection helpers
- `src/tools/temporal_filters/`: temporal-filter core types, IMEX/ETD updates, wave-mean helpers, and GQL utilities

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
Output      : evaluator → add_task! → NetCDF / HDF5 / callback
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
