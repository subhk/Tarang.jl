"""
Solver classes for different problem types.

## CPU/GPU Architecture Strategy

This module supports both CPU and GPU architectures with a hybrid approach:

1. **Field data**: Stored on the target architecture (CPU or GPU)
   - ScalarField, VectorField, TensorField can have GPU arrays
   - All field operations respect the architecture

2. **Linear algebra / Matrix solves**: Configurable CPU or GPU
   - Default: CPU sparse solvers (UMFPACK, SuperLU) - most efficient for small/medium
   - Optional: GPU solvers via `matsolver=:gpu` or specific GPU solver types
   - GPU solvers beneficial for large problems (>50K unknowns) or iterative methods

3. **Time-stepping**: Works on target architecture
   - Field updates use broadcasting which works on GPU
   - Nonlinear term evaluation uses architecture-aware operations
   - FFTs use CUFFT on GPU via architecture abstraction

4. **Data transfers**:
   - `fields_to_vector`: Extracts field data to CPU for linear solves
   - `copy_solution_to_fields!`: Copies solution back to field's architecture
   - For GPU linear solvers, data stays on GPU (no transfer overhead)

## Linear Solver Options

| Solver | Type | Best For |
|--------|------|----------|
| `:sparse` (default) | CPU sparse LU | Small-medium sparse systems |
| `:dense` | CPU dense LU | Small dense systems |
| `:cuda_cg` | GPU iterative CG | Large SPD systems |
| `:cuda_gmres` | GPU iterative GMRES | Large non-symmetric systems |
| `:cuda_dense` | GPU dense LU | Medium dense systems |
| `:gpu` / `:hybrid` | Auto-select | Automatic CPU/GPU selection |

## Usage

```julia
# CPU solver (default) - best for most spectral method problems
solver = InitialValueSolver(problem, RK443(); dt=0.001)

# GPU fields with CPU linear solve (recommended for small-medium problems)
domain = Domain(bases..., architecture=GPU())
problem = IVP([eq1, eq2, eq3], namespace=namespace)
solver = InitialValueSolver(problem, RK443(); dt=0.001)

# GPU fields with GPU linear solve (for large problems)
solver = BoundaryValueSolver(problem; matsolver=:cuda_cg)

# Hybrid auto-selection (uses GPU if problem is large enough)
solver = BoundaryValueSolver(problem; matsolver=:hybrid)
```

## Performance Guidelines

For typical spectral method problems (N < 100K modes):
- CPU sparse solvers are usually fastest
- GPU transfer overhead can exceed GPU solve speedup
- Use GPU solvers only if benchmarking shows improvement

For large problems (N > 100K modes):
- GPU iterative solvers (CG, GMRES) can be faster
- Requires good preconditioner for efficiency
- Memory bandwidth often the bottleneck
"""

# LinearAlgebra, SparseArrays, MPI already in Tarang.jl
using Arpack
using .MatSolvers

abstract type Solver end

mutable struct SolverPerformanceStats
    total_time::Float64
    total_steps::Int
    total_solves::Int
    avg_step_time::Float64

    function SolverPerformanceStats()
        new(0.0, 0, 0, 0.0)
    end
end

mutable struct SolverBaseData
    problem::Problem
    matrix_coupling::Vector{Bool}
    entry_cutoff::Float64
    matsolver::Any         # Solver choice (Symbol, Tuple, or concrete solver)
    evaluator::Union{Nothing, AbstractEvaluator}
end

function _normalize_matsolver(choice)
    if choice isa Tuple
        key, opts... = choice
        normalized = _normalize_matsolver(key)
        if normalized == :hybrid || normalized == :gpu
            return (normalized, opts...)
        end
        return normalized
    elseif choice isa AbstractString || choice isa Symbol
        key = lowercase(String(choice))
        if key in ("direct", "lu", "sparse")
            return :sparse
        elseif key == "dense"
            return :dense
        elseif key == "iterative"
            @warn "Iterative solver requested; using sparse direct solver instead"
            return :sparse
        elseif key in ("gpu", "hybrid")
            return Symbol(key)
        end
    end
    return choice
end

function SolverBaseData(problem::Problem; matrix_coupling=nothing, entry_cutoff::Real=1e-12, matsolver=:sparse)
    dim = (problem.domain !== nothing && hasproperty(problem.domain, :dist)) ? problem.domain.dist.dim : 0
    coupling = if matrix_coupling === nothing
        fill(true, dim)
    else
        collect(Bool.(matrix_coupling))
    end
    if length(coupling) != dim
        coupling = fill(true, dim)
    end
    solver_choice = _normalize_matsolver(matsolver)
    if solver_choice isa Tuple
        key, rest... = solver_choice
        solver_type = MatSolvers.get_solver(key)
        SolverBaseData(problem, coupling, Float64(entry_cutoff), (solver_type, rest...), nothing)
    else
        solver_type = MatSolvers.get_solver(solver_choice)
        SolverBaseData(problem, coupling, Float64(entry_cutoff), solver_type, nothing)
    end
end

function solver_comm(problem::Problem)
    if problem.domain !== nothing && hasproperty(problem.domain, :dist)
        return problem.domain.dist.comm
    else
        return MPI.COMM_WORLD
    end
end

function collect_state_fields(variables::Vector{Operand})
    state = ScalarField[]
    for var in variables
        if isa(var, ScalarField)
            push!(state, var)
        elseif isa(var, VectorField)
            append!(state, var.components)
        elseif isa(var, TensorField)
            append!(state, vec(var.components))
        end
    end
    return _concretize_state_fields(state)
end

function _concretize_state_fields(state::Vector{ScalarField})
    isempty(state) && return state

    T = typeof(state[1])
    for field in state
        if typeof(field) !== T
            return state
        end
    end

    concrete_state = Vector{T}(undef, length(state))
    copyto!(concrete_state, state)
    return concrete_state
end

function sync_state_to_problem!(problem::Problem, state::Vector{<:ScalarField})
    idx = 1
    for var in problem.variables
        if isa(var, ScalarField)
            if idx <= length(state)
                copy_field_data!(var, state[idx])
            end
            idx += 1
        elseif isa(var, VectorField)
            for comp in var.components
                if idx <= length(state)
                    copy_field_data!(comp, state[idx])
                end
                idx += 1
            end
        elseif isa(var, TensorField)
            for comp in vec(var.components)
                if idx <= length(state)
                    copy_field_data!(comp, state[idx])
                end
                idx += 1
            end
        end
    end
end

# LazyRHSPlan is defined later in lazy_rhs.jl (included after this file).
# Forward-declare as Any for the solver field below.

mutable struct InitialValueSolver <: Solver
    base::SolverBaseData
    problem::IVP
    timestepper::TimeStepper

    # State variables
    sim_time::Float64
    iteration::Int
    stop_sim_time::Float64
    stop_wall_time::Float64
    stop_iteration::Int

    # Solution state
    state::Vector{<:ScalarField}
    dt::Float64

    # Timestepper state for existing timesteppers.jl infrastructure
    timestepper_state::Union{Nothing, AbstractTimestepperState}

    # Evaluator for analysis
    evaluator::Union{Nothing, AbstractEvaluator}

    # Performance tracking
    wall_time_start::Float64
    performance_stats::SolverPerformanceStats

    # Type-specialized lazy RHS evaluation plan (LazyRHSPlan from lazy_rhs.jl)
    rhs_plan::Any
end

function attach_evaluator!(solver::InitialValueSolver)
    if solver.base.evaluator === nothing
        solver.base.evaluator = Evaluator(solver)
    else
        solver.base.evaluator.solver = solver
    end
    solver.evaluator = solver.base.evaluator
end

"""
Auto-detect mixed Fourier+Chebyshev domains and build subproblem matrices.
Stores the subproblems tuple in problem.parameters["subproblems"].
"""
function _try_build_subproblems!(solver::InitialValueSolver)
    problem = solver.problem
    state = solver.state

    if isempty(state)
        return
    end

    # Find a field with bases
    field = nothing
    for f in state
        if !isempty(f.bases)
            field = f
            break
        end
    end
    field === nothing && return

    # Find one Fourier and one Chebyshev/Jacobi basis
    fourier_basis = nothing
    cheb_basis = nothing
    for basis in field.bases
        basis === nothing && continue
        if isa(basis, FourierBasis) && fourier_basis === nothing
            fourier_basis = basis
        elseif isa(basis, JacobiBasis) && cheb_basis === nothing
            cheb_basis = basis
        end
    end

    # Build whenever there is a coupled (Chebyshev/Jacobi) direction — its implicit
    # operator is NOT diagonal, so it needs a per-mode tau subproblem. This covers
    # both mixed Fourier+Chebyshev AND pure-Chebyshev (no Fourier) problems. A
    # pure-Fourier problem (cheb_basis === nothing) has a diagonal per-mode implicit
    # operator handled by a separate path, so it is intentionally skipped here.
    cheb_basis === nothing && return

    # Set matrix coupling: Fourier dimensions are separable (each mode independent),
    # Chebyshev/Jacobi dimensions are coupled (modes interact via differentiation).
    # This determines how build_subsystems creates per-mode subproblems.
    coupling = Bool[]
    for basis in field.bases
        if basis === nothing
            push!(coupling, true)
        elseif isa(basis, FourierBasis)
            push!(coupling, false)  # Separable
        else
            push!(coupling, true)   # Coupled (Chebyshev, Jacobi, etc.)
        end
    end
    solver.base.matrix_coupling = coupling

    @info "Building subproblem matrix system"
    subsystems = build_subsystems(solver)
    subproblems = build_subproblems(solver, subsystems; build_matrices=["M", "L"])
    problem.parameters["subproblems"] = subproblems

    n_total = length(subproblems)
    n_with_mats = count(sp -> sp.M_min !== nothing && sp.L_min !== nothing, subproblems)
    @info "  $n_total subproblems built ($n_with_mats with matrices)"
end

"""True for a GPU-resident IVP whose spatial axes are all Fourier.

Such problems advance through the device-native field path and refresh their
algebraic constraints spectrally.  They neither use the global CPU matrices nor
build coupled (Jacobi/Chebyshev) subproblems.
"""
function _gpu_pure_fourier_state(state::Vector{<:ScalarField})
    found_spatial = false
    for field in state
        isempty(field.bases) && continue
        found_spatial = true
        is_gpu(field_architecture(field)) || return false
        all(b -> b !== nothing && isa(b, FourierBasis), field.bases) || return false
    end
    return found_spatial
end

function _build_initial_value_solver(problem::IVP, timestepper;
                                     dt::Real=1e-3,
                                     device::String="cpu",
                                     matsolver::Union{String,Symbol,Type,Tuple}=:sparse)
    setup_domain!(problem)

    # Merge boundary conditions into equation system
    _merge_boundary_conditions!(problem)

    validate_problem(problem)

    base = SolverBaseData(problem; matsolver=matsolver)

    state = collect_state_fields(problem.variables)

    perf_stats = SolverPerformanceStats()

    solver = InitialValueSolver(base, problem, timestepper, 0.0, 0, Inf, Inf, typemax(Int),
                                state, Float64(dt), nothing, nothing, time(),
                                perf_stats, nothing)
    attach_evaluator!(solver)

    if has_time_dependent_bcs(problem.bc_manager)
        @info "Solver: Time-dependent BCs detected - enabling BC updates"
        set_time_variable!(problem.bc_manager, "t")
    end

    if _gpu_pure_fourier_state(state)
        # RK/IMEX dispatch deliberately uses the field-wise explicit path for
        # pure-Fourier GPU states.  Building global host matrices here is both
        # unused and prohibitive at production sizes (e.g. the 512² turbulence
        # example), while the algebraic Poisson/velocity constraints are handled
        # spectrally by evaluate_rhs at each stage.
        @info "Pure-Fourier GPU IVP: skipping unused global CPU matrix assembly"
    else
        build_solver_matrices!(solver)
        _try_build_subproblems!(solver)
    end

    # Build the type-specialized lazy RHS plan. If translation fails for any
    # equation (e.g., unsupported operator type), `is_compiled` stays false
    # and `evaluate_rhs` falls back to interpreted evaluation.
    try
        solver.rhs_plan = build_lazy_rhs_plan!(solver)
    catch e
        # When the caller demanded a compiled RHS, propagate instead of silently
        # degrading to the interpreted evaluator.
        REQUIRE_LAZY_RHS[] && rethrow()
        @debug "LazyRHS build skipped: $e — using interpreted evaluation"
        solver.rhs_plan = nothing
    end

    # Populate equation_data F slots for any space-dependent BCs (including
    # those that are space-only, not in `time_dependent_bcs`). Without this,
    # the first call to `gather_alg_F!` wouldn't see the evaluated array
    # values because `_apply_bc_values_to_equations!` is only otherwise
    # invoked from the step loop when `has_time_dependent_bcs` is true.
    if has_space_dependent_bcs(problem.bc_manager) ||
       has_time_dependent_bcs(problem.bc_manager)
        # Auto-populate coordinate fields from the problem's bases so the
        # user doesn't have to call `add_coordinate_field!` manually for
        # every separable axis. This is a no-op for axes that the user
        # already registered (we never overwrite existing entries).
        _auto_register_coordinate_fields!(problem)
        _apply_bc_values_to_equations!(solver, 0.0)
    end

    return solver
end

"""
    _auto_register_coordinate_fields!(problem)

Walk the problem's variables, collect the unique bases, and register their
GLOBAL grid-coordinate arrays on `problem.bc_manager.coordinate_fields`
under the basis's element label (typically `"x"`, `"y"`, `"z"`, etc.).

Using the global grid (not the per-rank local grid) means every MPI rank
sees the same full coordinate array. BC expressions like `sin(2*pi*x/Lx)`
therefore evaluate to the same full-size grid array on every rank, and
the per-subproblem Fourier projection (which needs the full array to
produce correct global coefficients) works without any inter-rank
communication. The small per-rank FFT overhead is negligible compared to
the volumetric work of the stepper.

Skips axes already registered by the user and skips bases with no grid
(0-D tau variables). Registered entries use CPU `Vector{Float64}` so the
string-expression evaluator in `boundary_conditions.jl` (which uses
scalar broadcasts) works uniformly.
"""
function _auto_register_coordinate_fields!(problem::Problem)
    bc_manager = problem.bc_manager

    seen = Set{String}()
    for var in problem.variables
        for comp in scalar_components(var)
            isempty(comp.bases) && continue
            scales = comp.scales === nothing ?
                     ntuple(_ -> 1.0, max(length(comp.bases), 1)) :
                     comp.scales
            # A BC lives on the boundary PLANE, which is spanned by the Fourier axes. When there
            # are two or more of them, a coordinate registered as a bare 1-D vector loses its axis
            # identity: `cos(2*pi*y/Ly)` evaluates to a length-Ny vector that is indistinguishable
            # from a length-Nx one, and the projection assumed the FIRST Fourier axis — so the
            # y-profile was silently applied along x. Give each coordinate its axis identity in its
            # SHAPE ((Nx,1) and (1,Ny)), so broadcasting reconstructs the correct boundary plane and
            # the projection sees the true 2-D dependence.
            fourier_axes = [i for (i, b) in enumerate(comp.bases)
                            if b !== nothing && isa(b, FourierBasis)]
            n_fourier = length(fourier_axes)
            for (axis_idx, basis) in enumerate(comp.bases)
                basis === nothing && continue
                label = String(basis.meta.element_label)
                (label in seen) && continue
                haskey(bc_manager.coordinate_fields, label) && (push!(seen, label); continue)
                try
                    scale = axis_idx <= length(scales) ? scales[axis_idx] : 1.0
                    grid = _global_bc_grid(basis, scale)
                    fpos = findfirst(==(axis_idx), fourier_axes)
                    if fpos !== nothing && n_fourier >= 2
                        # Orient this Fourier coordinate along its own axis of the boundary plane.
                        shape = ntuple(d -> d == fpos ? length(grid) : 1, n_fourier)
                        grid = reshape(grid, shape)
                    end
                    bc_manager.coordinate_fields[label] = grid
                    push!(seen, label)
                catch err
                    @debug "auto-register coordinate field for $label failed" err
                end
            end
        end
    end
    return
end

"""
    _global_bc_grid(basis, scale) -> Vector{Float64}

Return the full (non-distributed) grid-coordinate array for a basis in
problem coordinates (applies change-of-variables for bounded intervals).
Used by `_auto_register_coordinate_fields!` so that BC expressions see
the global grid regardless of MPI layout.
"""
function _global_bc_grid(basis, scale::Real)
    native = _native_grid(basis, scale)
    # Fourier bases: `_native_grid` already returns problem coordinates.
    # Other bases (e.g. Chebyshev) return the canonical [-1, 1] grid, so
    # apply the COV to map back to problem space.
    mapped = if isa(basis, FourierBasis)
        native
    elseif basis.meta.COV !== nothing
        problem_coord(basis.meta.COV, native)
    else
        _problem_coord_fallback(basis, native)
    end
    return collect(Float64, mapped)
end

"""Merge boundary_conditions strings into equations and track indices.

For each raw BC string in `problem.boundary_conditions`, push it into
`problem.equations` (so the equation parser sees it) and try to associate
it with a concrete `AbstractBoundaryCondition` object in `bc_manager.conditions`
via `bc_to_equation(manager, bc)`.

**Matching is lenient**: we normalize whitespace and small number-formatting
differences (e.g. `0` vs `0.0`) before comparing, because
`add_bc!(problem::Problem, ::String)` auto-registers a `DirichletBC`/`NeumannBC`
whose `bc_to_equation` output isn't byte-identical to the raw user string.
Without lenient matching, space- and time-dependent BCs wouldn't get their
`bc_equation_indices` populated, so `_apply_bc_values_to_equations!` would
fail to refresh them and silently fall back to zero-F BC behavior.
"""
function _merge_boundary_conditions!(problem::Problem)
    if isempty(problem.boundary_conditions)
        return
    end

    bc_manager = problem.bc_manager
    base_eq_count = length(problem.equations)
    used_bc_idxs = Set{Int}()

    for (i, bc_str) in enumerate(problem.boundary_conditions)
        push!(problem.equations, bc_str)
        eq_idx = base_eq_count + i

        # First pass: exact or lenient string match against every BC object
        # the user (or our auto-registration) has already added.
        matched_bc_idx = 0
        for (bc_idx, bc) in enumerate(bc_manager.conditions)
            if isa(bc, PeriodicBC)
                continue
            end
            bc_idx in used_bc_idxs && continue
            eq = bc_to_equation(bc_manager, bc)
            if isa(eq, Vector)
                for eq_str in eq
                    if _bc_strings_equivalent(eq_str, bc_str)
                        matched_bc_idx = bc_idx
                        break
                    end
                end
                matched_bc_idx > 0 && break
            elseif _bc_strings_equivalent(eq, bc_str)
                matched_bc_idx = bc_idx
                break
            end
        end

        if matched_bc_idx > 0
            bc_manager.bc_equation_indices[matched_bc_idx] = eq_idx
            push!(used_bc_idxs, matched_bc_idx)
        end
    end

    @debug "Merged $(length(problem.boundary_conditions)) BCs into equations (total: $(length(problem.equations)))"
end

"""
    _bc_strings_equivalent(a, b) -> Bool

Compare two BC equation strings for functional equivalence, allowing:
- whitespace differences
- number-formatting differences (`0` vs `0.0`, `1` vs `1.0`)

Returns `true` if the strings represent the same BC. Used by
`_merge_boundary_conditions!` to link an auto-registered `DirichletBC`
(whose `bc_to_equation` re-stringifies positions via `string(0.0) = "0.0"`)
back to the raw user string (`"T(z=0) = ..."`).

Strategy:
1. Fast path — exact equality.
2. Fast path — whitespace-stripped equality.
3. Try parsing both via `parse_bc_string` / `parse_neumann_bc_string` and
   compare the structured `(field, coord, position, value)` tuples,
   normalizing numeric positions via `Float64` equality. This handles the
   `0` ↔ `0.0` round-trip issue transparently.
"""
function _bc_strings_equivalent(a::AbstractString, b::AbstractString)
    a == b && return true
    sa = replace(a, r"\s+" => "")
    sb = replace(b, r"\s+" => "")
    sa == sb && return true
    # Structured comparison via the BC string parser.
    pa = try
        parse_bc_string(a)
    catch
        try; parse_neumann_bc_string(a); catch; nothing; end
    end
    pb = try
        parse_bc_string(b)
    catch
        try; parse_neumann_bc_string(b); catch; nothing; end
    end
    (pa === nothing || pb === nothing) && return false
    # Tuple layout: (field_name, coordinate, position, value)
    pa[1] == pb[1] || return false
    pa[2] == pb[2] || return false
    _bc_positions_equivalent(pa[3], pb[3]) || return false
    # Value: Numbers via ==, Strings via whitespace-stripped compare
    va, vb = pa[4], pb[4]
    if isa(va, Number) && isa(vb, Number)
        return Float64(va) == Float64(vb)
    elseif isa(va, AbstractString) && isa(vb, AbstractString)
        return replace(va, r"\s+" => "") == replace(vb, r"\s+" => "")
    else
        return va == vb
    end
end
_bc_strings_equivalent(a, b) = a == b

function _bc_positions_equivalent(a, b)
    anum = a isa Real ? Float64(a) : tryparse(Float64, string(a))
    bnum = b isa Real ? Float64(b) : tryparse(Float64, string(b))
    if anum !== nothing && bnum !== nothing
        return anum == bnum
    end
    return replace(string(a), r"\s+" => "") == replace(string(b), r"\s+" => "")
end

"""
    _apply_bc_values_to_equations!(solver, current_time)

Inject the latest BC values into `equation_data["F"]` / `["F_expr"]` for
every registered BC that is time-dependent, space-dependent, or both.

Scalar/constant results are wrapped in `ConstantOperator`; array-valued
results are wrapped in `ArrayOperator` — the subproblem-level
`gather_alg_F!` then dispatches on the operator type (`_evaluate_alg_F`)
to project the value onto each Fourier mode.

Because BC arrays are swapped in fresh here, we also invalidate the
per-problem BC-array FFT cache so the next gather call will re-transform.
"""
function _apply_bc_values_to_equations!(solver::InitialValueSolver, current_time)
    bc_manager = solver.problem.bc_manager
    equation_data = solver.problem.equation_data

    isempty(equation_data) && return

    # If this problem has space-dependent BCs, refresh their cached values
    # against the current coordinate fields first. Pure time-dependent BCs
    # are refreshed in `update_time_dependent_bcs!` which is called by the
    # stepper before this function.
    if !isempty(bc_manager.space_dependent_bcs) &&
       !isempty(bc_manager.coordinate_fields)
        # Space-dependent BC values are cached by `(bc_idx, time, hash)` and
        # recomputed against the current coordinate fields below. Drop stale
        # entries first so the cache stays bounded across steps.
        _drop_spatial_cache_entries!(bc_manager)
        try
            evaluate_space_dependent_bcs!(bc_manager, bc_manager.coordinate_fields, current_time)
        catch err
            @warn "evaluate_space_dependent_bcs! failed: $err" maxlog=3
        end
    end

    # Iterate over the union of time- and space-dependent BC indices so
    # every non-constant BC gets its equation_data refreshed.
    # Use the pre-computed cache on bc_manager; rebuild only when empty
    # (invalidated by add_bc! at setup time, stable throughout the solve).
    bc_indices = bc_manager.nonconstant_bc_indices
    if isempty(bc_indices)
        append!(bc_indices, bc_manager.time_dependent_bcs)
        append!(bc_indices, bc_manager.space_dependent_bcs)
        unique!(bc_indices)
    end

    for bc_idx in bc_indices
        eq_idx = get(bc_manager.bc_equation_indices, bc_idx, 0)
        eq_idx <= 0 || eq_idx > length(equation_data) || _write_bc_value_to_eq!(
            equation_data[eq_idx], bc_manager, bc_idx, current_time,
        )
        # Note: the `||` short-circuit above reads "if invalid eq_idx, skip".
    end

    # Any ArrayOperator we may have written is a new object: invalidate the
    # RFFT cache so the next gather will recompute transforms on demand.
    invalidate_bc_array_cache!(solver.problem)
end

"""
    _write_bc_value_to_eq!(eq_data, bc_manager, bc_idx, current_time)

Read the most recent cached BC value and store it as a `ConstantOperator`
(scalar) or `ArrayOperator` (array) in the equation's `F` / `F_expr` slots.

Handles:
- Scalar Dirichlet/Neumann: `ConstantOperator(value)`
- Scalar Robin: uses the value component; skips if it isn't numeric
- Array Dirichlet/Neumann: `ArrayOperator(value)` (for space-dependent BCs)
"""
function _write_bc_value_to_eq!(eq_data::Dict, bc_manager, bc_idx::Int, current_time)
    bc = bc_manager.conditions[bc_idx]

    # Scalar time-dependent path (cache key `(bc_idx, current_time)`).
    value = get_current_bc_value(bc_manager, bc_idx, current_time)

    # Fall back to the space-dependent cache if the time cache didn't yield
    # a value. Space-dependent entries are keyed by `(bc_idx, time, hash)`.
    if value === nothing && bc_idx in bc_manager.space_dependent_bcs
        value = _get_spatial_bc_value(
            bc_manager.bc_cache, bc, bc_idx, current_time, bc_manager.coordinate_fields,
        )
    end

    value === nothing && return

    if isa(value, Number)
        eq_data["F"] = ConstantOperator(Float64(value))
        eq_data["F_expr"] = ConstantOperator(Float64(value))
    elseif isa(value, Tuple)
        # Robin BC tuple (alpha, beta, value) — we only lift the value
        # component into the RHS; alpha/beta are already baked into the
        # LHS operator structure at problem build time.
        robin_val = length(value) >= 3 ? value[3] : nothing
        if robin_val isa Number
            eq_data["F"] = ConstantOperator(Float64(robin_val))
            eq_data["F_expr"] = ConstantOperator(Float64(robin_val))
        elseif robin_val isa AbstractArray
            eq_data["F"] = ArrayOperator(robin_val)
            eq_data["F_expr"] = ArrayOperator(robin_val)
        end
    elseif isa(value, AbstractArray)
        eq_data["F"] = ArrayOperator(value)
        eq_data["F_expr"] = ArrayOperator(value)
    end
    return
end

"""Clear all cached space-dependent BC values before recomputing them."""
function _drop_spatial_cache_entries!(bc_manager)
    _clear_spatial_bc_cache!(bc_manager.bc_cache)
    return
end

const _InitialValueSolver_constructor = _build_initial_value_solver

# Extract the concrete matsolver choice. A `(type, kwargs)` Tuple carries the
# solver type first (matching `_subproblem_solver_type` in the timestepper path);
# otherwise the choice (Symbol/String/Type) is passed through to `get_solver`.
_solver_type(choice) = choice isa Tuple ? choice[1] : choice

function _build_boundary_value_solver(problem::Union{LBVP, NLBVP};
                                      device::String="cpu",
                                      matsolver::Union{String,Symbol,Type}=:sparse,
                                      solver_type::Union{Nothing, String, Symbol}=nothing,
                                      tolerance::Real=1e-10,
                                      max_iterations::Int=100)
    setup_domain!(problem)
    # Merge add_bc! boundary conditions into the equation system (tau rows),
    # mirroring the IVP build (_build_initial_value_solver). Without this the BVP
    # system is under-determined and validation fails.
    _merge_boundary_conditions!(problem)
    validate_problem(problem)

    solver_choice = solver_type === nothing ? matsolver : solver_type
    base = SolverBaseData(problem; matsolver=solver_choice)

    state = collect_state_fields(problem.variables)

    L, M, F = build_matrices(problem)
    apply_entry_cutoff!(L, base.entry_cutoff)
    apply_entry_cutoff!(M, base.entry_cutoff)
    apply_entry_cutoff!(F, base.entry_cutoff)

    L_sparse = sparse(L)
    M_sparse = sparse(M)
    F_vec = F

    perf_stats = SolverPerformanceStats()

    problem.parameters["L_matrix"] = L_sparse
    problem.parameters["M_matrix"] = M_sparse
    problem.parameters["F_vector"] = F_vec

    # The global L can be rank-deficient for multi-variable tau systems, which
    # would make a direct (LU) factorization throw. Build the global solver
    # tolerantly: on failure `global_solver` is left `nothing` and the solve
    # falls back to the robust per-subproblem path.
    global_solver = try
        MatSolvers.solver_instance(_solver_type(base.matsolver), L_sparse)
    catch err
        @debug "BVP/EVP: global solver factorization failed; using per-subproblem solve" exception=err
        nothing
    end

    solver = BoundaryValueSolver(base, problem, state, L_sparse, M_sparse, F_vec, Float64(tolerance), max_iterations,
                                 nothing, perf_stats, global_solver, (), (), nothing)

    # Configure matrix coupling so build_subsystems creates PER-FOURIER-MODE
    # subproblems (Fourier separable, Chebyshev/Jacobi coupled), exactly like the
    # IVP path (_try_build_subproblems!). Without this the BVP builds one global
    # multi-mode subsystem, incompatible with the single-mode per-mode operator
    # matrices (lift/derivative) → DimensionMismatch. Pick a full-domain field
    # (one carrying a coupled, i.e. non-Fourier, basis) to define the coupling.
    let coupling_field = nothing
        for f in state
            if any(b -> b !== nothing && !isa(b, FourierBasis), f.bases)
                coupling_field = f
                break
            end
        end
        if coupling_field !== nothing
            solver.base.matrix_coupling =
                Bool[b === nothing ? true : !isa(b, FourierBasis) for b in coupling_field.bases]
        end
    end

    subsystems = build_subsystems(solver)
    subproblems = build_subproblems(solver, subsystems; build_matrices=["L", "M"])
    coeff_system = CoeffSystem(subproblems, eltype(L_sparse))
    setfield!(solver, :subsystems, subsystems)
    setfield!(solver, :subproblems, subproblems)
    setfield!(solver, :coeff_system, coeff_system)

    return solver
end

const _BoundaryValueSolver_constructor = _build_boundary_value_solver

function InitialValueSolver(problem::IVP, timestepper; kwargs...)
    return multiclass_new(InitialValueSolver, problem, timestepper; kwargs...)
end

# Note: BoundaryValueSolver and EigenvalueSolver convenience constructors are defined after the struct definitions below

mutable struct BoundaryValueSolver <: Solver
    base::SolverBaseData
    problem::Union{LBVP, NLBVP}

    # Solution state
    state::Vector{<:ScalarField}

    # Linear algebra objects
    L_matrix::SparseMatrixCSC{ComplexF64, Int}
    M_matrix::SparseMatrixCSC{ComplexF64, Int}
    F_vector::Vector{ComplexF64}

    # Solver parameters
    tolerance::Float64
    max_iterations::Int

    factorization::Union{Nothing, Factorization}
    performance_stats::SolverPerformanceStats
    global_solver::Any
    subsystems::Tuple{Vararg{Subsystem}}
    subproblems::Tuple{Vararg{Subproblem}}
    coeff_system::Union{Nothing, CoeffSystem}
end

mutable struct EigenvalueSolver <: Solver
    base::SolverBaseData
    problem::EVP

    # Solution state
    eigenvalues::Vector{ComplexF64}
    eigenvectors::Matrix{ComplexF64}

    # Linear algebra objects
    L_matrix::SparseMatrixCSC{ComplexF64, Int}
    M_matrix::SparseMatrixCSC{ComplexF64, Int}

    # Solver parameters
    nev::Int
    which::Symbol
    target::Union{Nothing, ComplexF64}

    performance_stats::SolverPerformanceStats
    global_solver::Any
    subsystems::Tuple{Vararg{Subsystem}}
    subproblems::Tuple{Vararg{Subproblem}}
    coeff_system::Union{Nothing, CoeffSystem}
end

# Convenience constructors (must be after struct definitions)
function BoundaryValueSolver(problem::Union{LBVP, NLBVP}; kwargs...)
    return multiclass_new(BoundaryValueSolver, problem; kwargs...)
end

function EigenvalueSolver(problem::EVP; kwargs...)
    return multiclass_new(EigenvalueSolver, problem; kwargs...)
end

function _build_eigenvalue_solver(problem::EVP;
                                  nev::Int=10,
                                  which::Union{String,Symbol}=:LM,
                                  target::Union{Nothing, ComplexF64}=nothing,
                                  matsolver::Union{String,Symbol,Type}=:sparse)
    setup_domain!(problem)
    # Merge add_bc! boundary conditions into the equation set before validating
    # (same fix as the BVP build): without this the BC rows are missing and the
    # equation/variable counts mismatch for multi-variable tau systems.
    _merge_boundary_conditions!(problem)
    validate_problem(problem)

    base = SolverBaseData(problem; matsolver=matsolver)
    L, M, _ = build_matrices(problem)
    apply_entry_cutoff!(L, base.entry_cutoff)
    # For EVP, applying entry_cutoff to M can zero out intentionally small
    # entries from tau-method boundary conditions, corrupting eigenvalues.
    M_rank_before = rank(sparse(M))
    apply_entry_cutoff!(M, base.entry_cutoff)
    L_sparse = sparse(L)
    M_sparse = sparse(M)
    M_rank_after = rank(M_sparse)
    if M_rank_after < M_rank_before
        @warn "entry_cutoff=$(base.entry_cutoff) reduced M matrix rank from $M_rank_before to $M_rank_after. " *
              "This may corrupt eigenvalues from tau-method BCs. Consider setting entry_cutoff=0." maxlog=1
    end

    problem.parameters["L_matrix"] = L_sparse
    problem.parameters["M_matrix"] = M_sparse

    perf_stats = SolverPerformanceStats()
    # The global L can be rank-deficient for multi-variable tau systems, which
    # would make a direct (LU) factorization throw. Build the global solver
    # tolerantly: on failure `global_solver` is left `nothing` and the solve
    # falls back to the robust per-subproblem path.
    global_solver = try
        MatSolvers.solver_instance(_solver_type(base.matsolver), L_sparse)
    catch err
        @debug "BVP/EVP: global solver factorization failed; using per-subproblem solve" exception=err
        nothing
    end
    which_symbol = Symbol(uppercase(String(which)))
    solver = EigenvalueSolver(base, problem, ComplexF64[], zeros(ComplexF64, 0, 0),
                              L_sparse, M_sparse, nev, which_symbol, target,
                              perf_stats, global_solver, (), (), nothing)

    # Configure matrix coupling so build_subsystems creates PER-FOURIER-MODE
    # subproblems (Fourier separable, Chebyshev/Jacobi coupled), exactly like the
    # BVP/IVP path. The per-subproblem L/M matrices are the SQUARE, full-rank tau
    # systems; the global L is rank-deficient for multi-variable tau systems and
    # makes Arpack's shift-invert factorization throw SingularException.
    let coupling_field = nothing
        for f in problem.variables
            bs = hasproperty(f, :bases) ? f.bases : ()
            if any(b -> b !== nothing && !isa(b, FourierBasis), bs)
                coupling_field = f
                break
            end
        end
        if coupling_field !== nothing
            solver.base.matrix_coupling =
                Bool[b === nothing ? true : !isa(b, FourierBasis) for b in coupling_field.bases]
        end
    end

    subsystems = build_subsystems(solver)
    subproblems = build_subproblems(solver, subsystems; build_matrices=["L", "M"])
    coeff_system = CoeffSystem(subproblems, eltype(L_sparse))

    setfield!(solver, :subsystems, subsystems)
    setfield!(solver, :subproblems, subproblems)
    setfield!(solver, :coeff_system, coeff_system)

    return solver
end

const _EigenvalueSolver_constructor = _build_eigenvalue_solver

function dispatch_check(::Type{InitialValueSolver}, args::Tuple, kwargs::NamedTuple)
    if length(args) < 2
        throw(ArgumentError("InitialValueSolver requires (problem, timestepper) arguments"))
    end
    problem = args[1]
    if !(problem isa IVP)
        throw(ArgumentError("InitialValueSolver requires an IVP problem"))
    end
    return true
end

function dispatch_check(::Type{BoundaryValueSolver}, args::Tuple, kwargs::NamedTuple)
    if isempty(args)
        throw(ArgumentError("BoundaryValueSolver requires a problem argument"))
    end
    problem = args[1]
    if !(problem isa LBVP || problem isa NLBVP)
        throw(ArgumentError("BoundaryValueSolver requires an LBVP or NLBVP problem"))
    end
    return true
end

function dispatch_check(::Type{EigenvalueSolver}, args::Tuple, kwargs::NamedTuple)
    if isempty(args)
        throw(ArgumentError("EigenvalueSolver requires a problem argument"))
    end
    problem = args[1]
    if !(problem isa EVP)
        throw(ArgumentError("EigenvalueSolver requires an EVP problem"))
    end
    return true
end

function invoke_constructor(::Type{InitialValueSolver}, args::Tuple, kwargs::NamedTuple)
    problem, timestepper = args
    return _InitialValueSolver_constructor(problem, timestepper; kwargs...)
end

function invoke_constructor(::Type{BoundaryValueSolver}, args::Tuple, kwargs::NamedTuple)
    problem = args[1]
    return _BoundaryValueSolver_constructor(problem; kwargs...)
end

function invoke_constructor(::Type{EigenvalueSolver}, args::Tuple, kwargs::NamedTuple)
    problem = args[1]
    return _EigenvalueSolver_constructor(problem; kwargs...)
end

# Solver building functions
"""Build matrices for initial value solver"""
function build_solver_matrices!(solver::InitialValueSolver)
    L, M, F = build_matrices(solver.problem)
    cutoff = solver.base.entry_cutoff
    
    apply_entry_cutoff!(L, cutoff)
    apply_entry_cutoff!(M, cutoff)
    apply_entry_cutoff!(F, cutoff)
    
    # Store matrices for timestepping
    solver.problem.parameters["L_matrix"] = sparse(L)
    solver.problem.parameters["M_matrix"] = sparse(M)
    solver.problem.parameters["F_vector"] = F
end

function apply_entry_cutoff!(A::AbstractMatrix, cutoff::Real)
    cutoff <= 0 && return A
    if A isa SparseMatrixCSC
        mask = abs.(A.nzval) .< cutoff
        A.nzval[mask] .= 0
        dropzeros!(A)
    else
        A[abs.(A) .< cutoff] .= 0
    end
    return A
end

function apply_entry_cutoff!(v::AbstractVector, cutoff::Real)
    cutoff <= 0 && return v
    v[abs.(v) .< cutoff] .= 0
    return v
end
