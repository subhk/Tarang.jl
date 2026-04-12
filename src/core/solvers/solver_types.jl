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
    evaluator::Any  # Union{Nothing, Evaluator} — Evaluator loaded after solvers.jl
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
    return state
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
    evaluator::Any  # Union{Nothing, Evaluator} — Evaluator loaded after solvers.jl

    # Performance tracking
    wall_time_start::Float64
    workspace::Dict{String, AbstractArray}
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

    # Only build for mixed Fourier + non-periodic (Chebyshev/Jacobi)
    (fourier_basis === nothing || cheb_basis === nothing) && return

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

    workspace = Dict{String, AbstractArray}()
    perf_stats = SolverPerformanceStats()

    solver = InitialValueSolver(base, problem, timestepper, 0.0, 0, Inf, Inf, typemax(Int),
                                state, Float64(dt), nothing, nothing, time(),
                                workspace, perf_stats, nothing)
    attach_evaluator!(solver)

    if has_time_dependent_bcs(problem.bc_manager)
        @info "Solver: Time-dependent BCs detected - enabling BC updates"
        set_time_variable!(problem.bc_manager, "t")
    end

    build_solver_matrices!(solver)
    _try_build_subproblems!(solver)

    # Build the type-specialized lazy RHS plan. If translation fails for any
    # equation (e.g., unsupported operator type), `is_compiled` stays false
    # and `evaluate_rhs` falls back to interpreted evaluation.
    try
        solver.rhs_plan = build_lazy_rhs_plan!(solver)
    catch e
        @debug "LazyRHS build skipped: $e — using interpreted evaluation"
        solver.rhs_plan = nothing
    end

    return solver
end

"""Merge boundary_conditions strings into equations and track indices."""
function _merge_boundary_conditions!(problem::Problem)
    if isempty(problem.boundary_conditions)
        return
    end

    bc_manager = problem.bc_manager
    base_eq_count = length(problem.equations)

    for (i, bc_str) in enumerate(problem.boundary_conditions)
        push!(problem.equations, bc_str)
        eq_idx = base_eq_count + i

        # Find which BC condition (in bc_manager.conditions) this string corresponds to
        for (bc_idx, bc) in enumerate(bc_manager.conditions)
            if isa(bc, PeriodicBC)
                continue
            end
            eq = bc_to_equation(bc_manager, bc)
            if isa(eq, Vector)
                for eq_str in eq
                    if eq_str == bc_str
                        bc_manager.bc_equation_indices[bc_idx] = eq_idx
                        break
                    end
                end
            elseif eq == bc_str
                bc_manager.bc_equation_indices[bc_idx] = eq_idx
                break
            end
        end
    end

    @debug "Merged $(length(problem.boundary_conditions)) BCs into equations (total: $(length(problem.equations)))"
end

"""Inject cached BC values into equation_data F expressions."""
function _apply_bc_values_to_equations!(solver::InitialValueSolver, current_time)
    bc_manager = solver.problem.bc_manager
    equation_data = solver.problem.equation_data

    isempty(equation_data) && return

    for bc_idx in bc_manager.time_dependent_bcs
        eq_idx = get(bc_manager.bc_equation_indices, bc_idx, 0)
        if eq_idx > 0 && eq_idx <= length(equation_data)
            value = get_current_bc_value(bc_manager, bc_idx, current_time)
            if value !== nothing
                if isa(value, Number)
                    equation_data[eq_idx]["F"] = ConstantOperator(Float64(value))
                    equation_data[eq_idx]["F_expr"] = ConstantOperator(Float64(value))
                elseif isa(value, Tuple)
                    # Robin BC tuple (alpha, beta, value) - use the value component
                    robin_val = value[3]
                    if robin_val !== nothing && isa(robin_val, Number)
                        equation_data[eq_idx]["F"] = ConstantOperator(Float64(robin_val))
                        equation_data[eq_idx]["F_expr"] = ConstantOperator(Float64(robin_val))
                    end
                elseif isa(value, AbstractArray)
                    # For array-valued BCs, wrap the array in an ArrayOperator
                    equation_data[eq_idx]["F"] = ArrayOperator(value)
                    equation_data[eq_idx]["F_expr"] = ArrayOperator(value)
                end
            end
        end
    end
end

const _InitialValueSolver_constructor = _build_initial_value_solver

function _build_boundary_value_solver(problem::Union{LBVP, NLBVP};
                                      device::String="cpu",
                                      matsolver::Union{String,Symbol,Type}=:sparse,
                                      solver_type::Union{Nothing, String, Symbol}=nothing,
                                      tolerance::Real=1e-10,
                                      max_iterations::Int=100)
    setup_domain!(problem)
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

    global_solver = MatSolvers.solver_instance(_solver_type(base.matsolver), L_sparse)

    solver = BoundaryValueSolver(base, problem, state, L_sparse, M_sparse, F_vec, Float64(tolerance), max_iterations,
                                 Dict{String, AbstractArray}(), nothing,
                                 perf_stats, global_solver, (), (), nothing)

    subsystems = build_subsystems(solver)
    subproblems = build_subproblems(solver, subsystems; build_matrices=["L", "M", "F"])
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

    workspace::Dict{String, AbstractArray}
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

    workspace::Dict{String, AbstractArray}
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

    workspace = Dict{String, AbstractArray}()
    perf_stats = SolverPerformanceStats()
    global_solver = MatSolvers.solver_instance(_solver_type(base.matsolver), L_sparse)
    which_symbol = Symbol(uppercase(String(which)))
    solver = EigenvalueSolver(base, problem, ComplexF64[], zeros(ComplexF64, 0, 0),
                              L_sparse, M_sparse, nev, which_symbol, target,
                              workspace, perf_stats, global_solver, (), (), nothing)

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
