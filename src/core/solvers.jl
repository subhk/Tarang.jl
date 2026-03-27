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

using LinearAlgebra
using SparseArrays
using MPI
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
    matsolver::Any
    evaluator::Any
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

mutable struct InitialValueSolver <: Solver
    base::SolverBaseData
    problem::IVP
    timestepper::Any

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
    timestepper_state::Any

    # Evaluator for analysis
    evaluator::Union{Nothing, Any}

    # Performance tracking
    wall_time_start::Float64
    workspace::Dict{String, AbstractArray}
    performance_stats::SolverPerformanceStats

    # Compiled RHS plan for zero-dispatch timestepping (lazily compiled)
    # Type is Any to avoid forward reference — actual type is Union{Nothing, CompiledRHSPlan}
    compiled_rhs::Any
end

function attach_evaluator!(solver::InitialValueSolver)
    if solver.base.evaluator === nothing
        solver.base.evaluator = Evaluator(solver)
    else
        solver.base.evaluator.solver = solver
    end
    solver.evaluator = solver.base.evaluator
end

function _build_initial_value_solver(problem::IVP, timestepper; dt::Real=1e-3, device::String="cpu")
    setup_domain!(problem)

    # Merge boundary conditions into equation system
    _merge_boundary_conditions!(problem)

    validate_problem(problem)

    base = SolverBaseData(problem)

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

    # Attempt to compile the RHS expression tree for zero-dispatch evaluation.
    # If compilation fails (unsupported expression types), falls back silently
    # to the interpreted evaluate_solver_expression path.
    try
        solver.compiled_rhs = compile_rhs_plan!(solver)
    catch e
        @debug "RHS compilation skipped: $e — using interpreted evaluation"
        solver.compiled_rhs = nothing
    end

    return solver
end

function _merge_boundary_conditions!(problem::Problem)
    """Merge boundary_conditions strings into equations and track indices."""
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

function _apply_bc_values_to_equations!(solver::InitialValueSolver, current_time)
    """Inject cached BC values into equation_data F expressions."""
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
    factorization::Union{Nothing, Any}
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
function build_solver_matrices!(solver::InitialValueSolver)
    """Build matrices for initial value solver"""
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

# Time stepping for IVP
function step!(solver::InitialValueSolver, dt::Float64=solver.dt)
    """Advance solution by one time step using existing timestepper infrastructure"""

    start_time = time()

    solver.dt = dt

    # Update time-dependent boundary conditions BEFORE taking the step.
    # NOTE: BCs are evaluated at t+dt and held fixed for all RK substeps.
    # For multi-stage methods (RK443, etc.), this introduces O(dt) error at
    # intermediate stages, potentially reducing the time integrator's formal
    # order of accuracy for problems with rapidly varying BCs.
    # TODO: Pass substep time into the timestepper loop for per-stage BC updates.
    if has_time_dependent_bcs(solver.problem.bc_manager)
        target_time = solver.sim_time + dt
        update_time_dependent_bcs!(solver.problem.bc_manager, target_time)
        _apply_bc_values_to_equations!(solver, target_time)
        @debug "Updated time-dependent BCs for t=$target_time"
    end

    # Use existing timestepper infrastructure from timesteppers.jl
    # Create TimestepperState if needed
    if solver.timestepper_state === nothing
        solver.timestepper_state = TimestepperState(solver.timestepper, dt, solver.state)
    else
        # Update timestep history for variable timestep support
        update_timestep_history!(solver.timestepper_state, dt)
    end

    # Call existing timestepper step function from timesteppers.jl
    step!(solver.timestepper_state, solver)

    # Get the updated state from timestepper history
    if length(solver.timestepper_state.history) > 0
        solver.state = solver.timestepper_state.history[end]
    end

    # Sync the final state back to problem variables so users can read them directly
    # (without this, problem variables hold stale intermediate stage data from evaluate_rhs)
    sync_state_to_problem!(solver.problem, solver.state)

    # Update time and iteration
    solver.sim_time += dt
    solver.iteration += 1

    # Update performance statistics
    step_time = time() - start_time
    solver.performance_stats.total_time += step_time
    solver.performance_stats.total_steps += 1

    return solver
end

# Solver execution control
function proceed(solver::InitialValueSolver)
    """Check if solver should continue"""
    if solver.sim_time >= solver.stop_sim_time
        return false
    end
    
    if solver.iteration >= solver.stop_iteration
        return false
    end
    
    if time() - solver.wall_time_start >= solver.stop_wall_time
        return false
    end
    
    return true
end

"""
    run!(solver; stop_time=Inf, stop_iteration=typemax(Int), stop_wall_time=Inf,
         callbacks=[], log_interval=100, progress=true)

Run a simulation loop to completion with optional callbacks and progress reporting.

This eliminates the standard simulation boilerplate. Instead of:
```julia
while proceed(solver)
    step!(solver)
    if solver.iteration % 100 == 0
        @info "Step \$(solver.iteration), t=\$(solver.sim_time)"
    end
end
```

Use:
```julia
run!(solver; stop_time=10.0, log_interval=100)
```

# Callbacks

Callbacks are `(interval, function)` tuples. The function receives the solver:
```julia
run!(solver; stop_time=10.0, callbacks=[
    (10,  s -> @info "Energy: \$(energy(s))"),
    (100, s -> save_checkpoint(s))
])
```

`interval` can be:
- `Int`: execute every N iterations
- `Float64`: execute every T simulation time units
"""
function run!(solver::InitialValueSolver;
              stop_time::Real=Inf,
              stop_iteration::Integer=typemax(Int),
              stop_wall_time::Real=Inf,
              callbacks::Vector=Pair[],
              log_interval::Integer=0,
              progress::Bool=true)

    solver.stop_sim_time = Float64(stop_time)
    solver.stop_iteration = Int(stop_iteration)
    solver.stop_wall_time = Float64(stop_wall_time)
    solver.wall_time_start = time()

    # Track last callback times for time-based intervals
    last_callback_times = Float64[solver.sim_time for _ in callbacks]

    if progress
        @info "Starting simulation: dt=$(solver.dt), stop_time=$stop_time, stop_iteration=$stop_iteration"
    end

    wall_start = time()

    while proceed(solver)
        step!(solver)

        # Log progress
        if log_interval > 0 && solver.iteration % log_interval == 0
            elapsed = time() - wall_start
            rate = solver.iteration / max(elapsed, 1e-10)
            @info "Step $(solver.iteration), t=$(round(solver.sim_time; digits=6)), " *
                  "wall=$(round(elapsed; digits=1))s, rate=$(round(rate; digits=1)) steps/s"
        end

        # Execute callbacks
        for (idx, cb) in enumerate(callbacks)
            interval, func = cb
            should_fire = if interval isa Integer
                solver.iteration % interval == 0
            elseif interval isa AbstractFloat
                solver.sim_time - last_callback_times[idx] >= interval
            else
                false
            end

            if should_fire
                func(solver)
                last_callback_times[idx] = solver.sim_time
            end
        end
    end

    elapsed = time() - wall_start
    if progress
        @info "Simulation complete: $(solver.iteration) steps, " *
              "t=$(round(solver.sim_time; digits=6)), wall=$(round(elapsed; digits=1))s"
    end

    return solver
end

# Boundary value solver
function solve!(solver::BoundaryValueSolver)
    """Solve boundary value problem"""

    start_time = time()

    if isa(solver.problem, LBVP)
        # Linear boundary value problem
        solution = solve_linear!(solver)

        # Copy solution back to state fields
        copy_solution_to_fields!(solver.state, solution)

    elseif isa(solver.problem, NLBVP)
        # Nonlinear boundary value problem - Newton iteration
        solve_nonlinear!(solver)
    end

    # Update performance statistics
    solve_time = time() - start_time
    solver.performance_stats.total_time += solve_time
    solver.performance_stats.total_solves += 1

    return solver
end

function solve_linear!(solver::BoundaryValueSolver)
    """Solve linear boundary value problem"""
    # Direct solve: L * x = F
    if solver.global_solver !== nothing
        return MatSolvers.solve(solver.global_solver, solver.F_vector)
    end
    return solver.L_matrix \ solver.F_vector
end

function solve_nonlinear!(solver::BoundaryValueSolver)
    """Solve nonlinear boundary value problem using Newton iteration"""

    # Initial guess (current state)
    x = fields_to_vector(solver.state)
    converged = false
    last_dx_norm = Inf

    for iter in 1:solver.max_iterations
        # Evaluate residual and Jacobian
        residual, jacobian = evaluate_residual_and_jacobian(solver.problem, x)

        # Newton update: J * dx = -R
        dx = -jacobian \ residual
        x += dx
        last_dx_norm = norm(dx)

        # Check convergence
        if last_dx_norm < solver.tolerance
            @info "Nonlinear solver converged in $iter iterations"
            converged = true
            break
        end
    end

    if !converged
        error("Nonlinear solver did not converge after $(solver.max_iterations) iterations " *
              "(tolerance=$(solver.tolerance), final |dx|=$(last_dx_norm)). " *
              "Consider: increasing max_iterations, loosening tolerance, or improving the initial guess.")
    end

    # Copy solution back
    copy_solution_to_fields!(solver.state, x)
end

# Eigenvalue solver
function solve!(solver::EigenvalueSolver; nev::Int=solver.nev,
                which::Union{String,Symbol}=solver.which,
                target::Union{Nothing, ComplexF64}=solver.target)
    """Solve eigenvalue problem"""

    start_time = time()

    which_symbol = Symbol(uppercase(String(which)))

    # Solve generalized eigenvalue problem: L * v = λ * M * v
    if target === nothing
        λ, v = Arpack.eigs(solver.L_matrix, solver.M_matrix; nev=nev, which=which_symbol)
    else
        λ, v = Arpack.eigs(solver.L_matrix, solver.M_matrix; nev=nev, sigma=target)
    end

    solver.nev = nev
    solver.which = which_symbol
    solver.target = target
    solver.eigenvalues = λ
    solver.eigenvectors = v

    # Update performance statistics
    solve_time = time() - start_time
    solver.performance_stats.total_time += solve_time
    solver.performance_stats.total_solves += 1

    return λ, v
end

# Utility functions
function fields_to_vector(fields::Vector{<:ScalarField})
    """
    Convert field array to solution vector following gather pattern.

    GPU-aware: For GPU fields, data is synchronized and transferred to CPU.
    Linear solves are performed on CPU (standard practice for sparse solvers),
    and results are transferred back to GPU via copy_solution_to_fields!.

    This function always returns a CPU Vector{ComplexF64} since that's what
    sparse linear solvers expect.
    """

    if isempty(fields)
        return Vector{ComplexF64}()
    end

    # Determine architecture from first field for synchronization
    arch = fields[1].dist.architecture

    # Synchronize GPU before data transfer (ensures all GPU operations complete)
    if is_gpu(arch)
        synchronize(arch)
    end

    # Ensure all fields are in coefficient space (following Tarang pattern)
    for field in fields
        ensure_layout!(field, :c)
    end

    # Synchronize again after layout changes (which may involve GPU FFTs)
    if is_gpu(arch)
        synchronize(arch)
    end

    # Calculate total vector size
    total_size = sum(compute_field_vector_size(field) for field in fields)

    # Allocate output vector on CPU (for sparse linear solvers)
    vector = Vector{ComplexF64}(undef, total_size)

    # Gather field data into vector (following Tarang gather pattern)
    offset = 1
    for field in fields
        field_size = compute_field_vector_size(field)
        if field_size > 0
            # Extract field data with proper handling of dimensions
            # For GPU fields, this transfers to CPU via get_cpu_data
            field_data = extract_field_data_for_vector(field)

            # Copy to vector buffer
            end_offset = offset + field_size - 1
            if end_offset <= length(vector) && length(field_data) == field_size
                vector[offset:end_offset] .= field_data
            else
                error("Size mismatch in fields_to_vector for field '$(field.name)': " *
                      "expected $field_size elements, got $(length(field_data)). " *
                      "Vector range: $offset:$end_offset of $(length(vector)). " *
                      "This indicates a bug in field allocation or compute_field_vector_size.")
            end

            offset += field_size
        end

        @debug "Gathered field $(field.name): size=$field_size, offset=$(offset-field_size)"
    end

    @debug "Fields to vector completed: total_size=$total_size, fields=$(length(fields))"

    # For MPI with PencilArray data, the vector above contains only LOCAL data.
    # Global-matrix solvers need the FULL vector on every rank.
    # Gather local vectors into global vector via MPI.Allgatherv.
    dist = fields[1].dist
    if dist.size > 1 && dist.use_pencil_arrays
        vector = _gather_to_global_vector(vector, dist)
    end

    return vector
end

"""
    _gather_to_global_vector(local_vector, dist) -> global_vector

Gather local solution vectors from all MPI ranks into a global vector
available on every rank. Used by global-matrix implicit solvers.
"""
function _gather_to_global_vector(local_vector::Vector{ComplexF64}, dist::Distributor)
    local_size = length(local_vector)

    # Gather all local sizes to determine recv_counts
    all_sizes = MPI.Allgather(Int32(local_size), dist.comm)
    recv_counts = Int.(all_sizes)
    total_size = sum(recv_counts)

    # Compute displacements
    recv_displs = zeros(Int, length(recv_counts))
    for i in 2:length(recv_counts)
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1]
    end

    # Allgatherv to collect all local vectors
    global_vector = Vector{ComplexF64}(undef, total_size)
    MPI.Allgatherv!(local_vector, MPI.VBuffer(global_vector, recv_counts), dist.comm)

    return global_vector
end

function copy_solution_to_fields!(fields::Vector{<:ScalarField}, solution::AbstractVector{<:Number})
    """
    Copy solution vector back to fields following scatter pattern.

    GPU-aware: The solution vector is on CPU (from linear solver).
    For GPU fields, data is transferred to GPU and synchronized.

    MPI-aware: If fields are PencilArray-distributed and the solution is the
    global vector (from _gather_to_global_vector), each rank extracts only
    its local portion based on the rank offset.
    """

    if isempty(fields)
        return
    end

    dist = fields[1].dist
    is_mpi_global = dist.size > 1 && dist.use_pencil_arrays

    if is_mpi_global
        # Solution is the global vector. Each rank must extract its local portion.
        # Compute this rank's offset into the global vector.
        local_size = sum(compute_field_vector_size(f) for f in fields)
        all_sizes = MPI.Allgather(Int32(local_size), dist.comm)
        rank_offset = sum(Int.(all_sizes[1:dist.rank]))  # 0-indexed rank

        offset = rank_offset + 1
    else
        offset = 1
    end

    for field in fields
        field_size = compute_field_vector_size(field)

        if field_size > 0 && offset <= length(solution)
            end_offset = min(offset + field_size - 1, length(solution))
            actual_size = end_offset - offset + 1

            if actual_size > 0
                field_data = solution[offset:end_offset]
                set_field_data_from_vector!(field, field_data)
                @debug "Scattered to field $(field.name): size=$actual_size"
            end

            offset += field_size
        end
    end

    # Synchronize GPU after all data transfers (ensures data is available)
    if !isempty(fields)
        arch = fields[1].dist.architecture
        if is_gpu(arch)
            synchronize(arch)
        end
    end

    @debug "Vector to fields completed: solution_size=$(length(solution)), fields=$(length(fields))"
end

function vector_to_fields(vector::AbstractVector{<:Number}, template::Vector{<:ScalarField})
    """
    Convert solution vector to a new state vector matching a template.

    GPU-aware: New fields are allocated on the same architecture as the
    template fields. The vector (typically on CPU from linear solve) is
    transferred to GPU via set_field_data_from_vector! if needed.
    """

    new_state = ScalarField[]
    offset = 1

    for field in template
        # New field inherits architecture from template's distributor
        new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
        field_size = compute_field_vector_size(field)

        if field_size > 0 && offset <= length(vector)
            end_offset = min(offset + field_size - 1, length(vector))
            data_slice = vector[offset:end_offset]
            # Handles CPU→GPU transfer internally
            set_field_data_from_vector!(new_field, data_slice)
            offset += field_size
        end

        push!(new_state, new_field)
    end

    return new_state
end

function compute_field_vector_size(field::ScalarField)
    """
    Compute the number of degrees of freedom for a field in vector form.
    Following field size computation patterns.
    """
    
    if get_coeff_data(field) !== nothing
        # Use coefficient space data size
        return length(get_coeff_data(field))
    elseif get_grid_data(field) !== nothing  
        # Use grid space data size as fallback
        return length(get_grid_data(field))
    else
        # Default size based on bases
        total_size = 1
        for basis in field.bases
            if basis !== nothing
                basis_size = get_basis_size(basis)
                total_size *= basis_size
            end
            # Skip nothing bases — they contribute no DOFs (e.g., unused dimensions)
        end
        return total_size
    end
end

function extract_field_data_for_vector(field::ScalarField)
    """
    Extract field data for vector conversion with proper layout handling.
    Following field data extraction patterns.

    GPU-aware: For GPU fields, transfers data to CPU using get_cpu_data.
    This is necessary because linear solvers typically work on CPU.
    Synchronization is handled by the caller (fields_to_vector).
    """

    # Ensure coefficient space layout
    ensure_layout!(field, :c)

    if get_coeff_data(field) !== nothing
        # Get CPU data (handles GPU→CPU transfer if needed)
        # get_cpu_data returns Array(data) for GPU arrays
        cpu_data = get_cpu_data(get_coeff_data(field))
        # Return flattened coefficient data
        return vec(cpu_data)
    elseif get_grid_data(field) !== nothing
        # Fallback to grid data if coefficient data not available
        @warn "Using grid space data for field $(field.name) - converting to coefficient space recommended"
        cpu_data = get_cpu_data(get_grid_data(field))
        return vec(cpu_data)
    else
        # Return zeros as fallback
        field_size = compute_field_vector_size(field)
        return zeros(ComplexF64, field_size)
    end
end

function set_field_data_from_vector!(field::ScalarField, data::AbstractVector{<:Number})
    """
    Set field data from vector with proper shape and layout handling.
    Following field data setting patterns.

    GPU-aware: For GPU fields, CPU data is transferred to GPU using
    on_architecture(). This allows linear solves to happen on CPU
    and results to be seamlessly copied back to GPU fields.

    Note: Synchronization is handled by the caller (copy_solution_to_fields!).
    """

    if get_coeff_data(field) !== nothing
        # Reshape data to match field coefficient data shape
        target_shape = size(get_coeff_data(field))
        expected_size = prod(target_shape)

        if length(data) == expected_size
            data_slice = data
        elseif length(data) < expected_size
            # Partial update - pad with zeros
            temp_data = zeros(ComplexF64, expected_size)
            temp_data[1:length(data)] .= data
            data_slice = temp_data
            @debug "Padded field $(field.name) data: got $(length(data)), expected $expected_size"
        else
            # Truncate excess data
            data_slice = data[1:expected_size]
            @debug "Truncated field $(field.name) data: got $(length(data)), expected $expected_size"
        end

        target_eltype = eltype(get_coeff_data(field))
        if target_eltype <: Real
            if any(x -> !iszero(imag(x)), data_slice)
                @warn "Discarding imaginary part when setting real field $(field.name)"
            end
            reshaped_data = reshape(real.(data_slice), target_shape)
        else
            reshaped_data = reshape(convert.(target_eltype, data_slice), target_shape)
        end

        # GPU-aware copy: move data to field's architecture if needed
        arch = field.dist.architecture
        if is_gpu(arch)
            # Move reshaped CPU data to GPU architecture
            # on_architecture handles the CPU→GPU transfer
            gpu_data = on_architecture(arch, reshaped_data)
            copyto!(get_coeff_data(field), gpu_data)
        else
            # CPU path: direct copy
            copyto!(get_coeff_data(field), reshaped_data)
        end

        field.current_layout = :c

    elseif get_grid_data(field) !== nothing
        # Fallback to grid data
        target_shape = size(get_grid_data(field))
        expected_size = prod(target_shape)

        if length(data) == expected_size
            target_eltype = eltype(get_grid_data(field))
            if target_eltype <: Real
                if any(x -> !iszero(imag(x)), data)
                    @warn "Discarding imaginary part when setting real grid field $(field.name)"
                end
                reshaped_data = reshape(real.(data), target_shape)
            else
                reshaped_data = reshape(convert.(target_eltype, data), target_shape)
            end

            # GPU-aware copy: move data to field's architecture if needed
            arch = field.dist.architecture
            if is_gpu(arch)
                gpu_data = on_architecture(arch, reshaped_data)
                copyto!(get_grid_data(field), gpu_data)
            else
                copyto!(get_grid_data(field), reshaped_data)
            end
        else
            @warn "Size mismatch setting grid data for field $(field.name)"
        end

        field.current_layout = :g
    else
        # Silently skip 0D fields (tau variables) - they have no spatial data
        # Only warn for fields that should have data but don't
        if !isempty(field.bases)
            @warn "Cannot set data for field $(field.name) - no data arrays allocated"
        end
        # For 0D tau variables, this is expected - they are Lagrange multipliers
        # with no spatial extent
    end
end

function get_basis_size(basis)
    """Get the size (number of modes) for a basis following Tarang patterns"""
    
    # Following basis structure, bases store size information in different ways:
    # 1. Most common: meta.size field (for Julia BasisMeta structure)
    # 2. Direct size field (for direct Tarang translation)  
    # 3. Shape tuple (for multidimensional bases)
    # 4. Specific basis attributes (N, etc.)
    
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :size)
        # Julia BasisMeta structure pattern
        return basis.meta.size
    elseif hasfield(typeof(basis), :shape)
        # Tarang shape attribute pattern (can be tuple for multidimensional)
        shape = basis.shape
        if isa(shape, Tuple)
            # For multidimensional bases, return total size (product of dimensions)
            return prod(shape)
        else
            return shape
        end
    elseif hasfield(typeof(basis), :size)
        # Direct size field
        return basis.size
    elseif hasfield(typeof(basis), :N)
        # Common mathematical notation for basis size
        return basis.N
    else
        @warn "Could not determine basis size for $(typeof(basis)), using default"
        return 64  # Default basis size
    end
end

function evaluate_residual_and_jacobian(problem::NLBVP, x::Vector{ComplexF64})
    """
    Evaluate residual and Jacobian for nonlinear problem following Tarang patterns.
    
    In Tarang, this corresponds to:
    1. Evaluating F expressions (residual) using evaluator system
    2. Building dF matrices (Jacobian/Frechet differential) 
    3. Gathering results into numerical arrays for Newton solver
    """
    
    # Step 1: Copy solution vector back to problem fields
    state_fields = collect_state_fields(problem.variables)
    copy_solution_to_fields!(state_fields, x)

    # Step 2: Evaluate residual expressions (lhs - rhs)
    residual_fields = ScalarField[]

    if hasfield(typeof(problem), :equation_data) && problem.equation_data !== nothing
        for (i, eq_data) in enumerate(problem.equation_data)
            template = state_fields[min(i, length(state_fields))]
            lhs_expr = get(eq_data, "lhs", nothing)
            rhs_expr = get(eq_data, "rhs", nothing)

            if lhs_expr !== nothing || rhs_expr !== nothing
                lhs_field = lhs_expr === nothing ? create_zero_field(template) :
                           evaluate_solver_expression(lhs_expr, problem.variables; layout=:g, template=template)
                rhs_field = rhs_expr === nothing ? create_zero_field(template) :
                           evaluate_solver_expression(rhs_expr, problem.variables; layout=:g, template=template)
                residual_field = lhs_field - rhs_field
            else
                # Fallback: use F expression if provided
                expr = get(eq_data, "F", ZeroOperator())
                residual_field = evaluate_solver_expression(expr, problem.variables; layout=:g, template=template)
            end

            ensure_layout!(residual_field, :c)
            push!(residual_fields, residual_field)
            @debug "Evaluated residual for equation $i"
        end
    else
        @warn "No equation data available for residual evaluation - creating zero residuals"
        for (i, field) in enumerate(state_fields)
            residual_field = create_zero_field(field)
            push!(residual_fields, residual_field)
        end
    end

    # Step 3: Convert residual fields to vector
    residual = fields_to_vector(residual_fields)

    # Step 4: Build Jacobian matrix
    # Try symbolic Jacobian first (Frechet differentiation), then fall back
    n = length(x)
    jacobian = try
        build_symbolic_jacobian(problem, state_fields)
    catch e
        @debug "Symbolic Jacobian construction failed ($e), using fallback"
        if haskey(problem.parameters, "L_matrix")
            problem.parameters["L_matrix"]
        else
            sparse(I, n, n)
        end
    end

    @debug "Residual evaluation completed: size=$(length(residual)), norm=$(norm(residual))"
    @debug "Jacobian evaluation completed: size=$(size(jacobian)), nnz=$(nnz(jacobian))"

    return residual, jacobian
end

function _constant_field_from_template(template::ScalarField, value::Number; layout::Symbol=:g)
    """
    Create a constant field from a template.

    GPU-aware: The field inherits the architecture from the template.
    Uses fill!() which works on both CPU and GPU arrays.
    """
    field = ScalarField(template.dist, "const_$(template.name)", template.bases, template.dtype)
    ensure_layout!(field, layout)
    if layout == :g && get_grid_data(field) !== nothing
        # fill!() works on both CPU and GPU
        fill!(get_grid_data(field), convert(eltype(get_grid_data(field)), value))
    elseif layout == :c && get_coeff_data(field) !== nothing
        fill!(get_coeff_data(field), convert(eltype(get_coeff_data(field)), value))
    end
    return field
end

"""
    get_solver_architecture(solver::Solver)

Get the architecture (CPU or GPU) used by the solver's fields.
Returns CPU() if architecture cannot be determined.
"""
function get_solver_architecture(solver::Solver)
    if isa(solver, InitialValueSolver) && !isempty(solver.state)
        return solver.state[1].dist.architecture
    elseif isa(solver, BoundaryValueSolver) && !isempty(solver.state)
        return solver.state[1].dist.architecture
    elseif hasproperty(solver, :problem) && solver.problem !== nothing
        if solver.problem.domain !== nothing && hasproperty(solver.problem.domain, :dist)
            return solver.problem.domain.dist.architecture
        end
    end
    return CPU()
end

function _scale_vector_field(field::VectorField, scale::Number)
    result = VectorField(field.dist, field.coordsys, "$(field.name)_scaled", field.bases, field.dtype)
    for i in eachindex(field.components)
        result.components[i] = field.components[i] * scale
    end
    return result
end

function _coerce_numeric_operand(value, template::Union{Nothing, ScalarField}; layout::Symbol=:g)
    if value isa Number
        if template === nothing
            return value
        end
        return _constant_field_from_template(template, value; layout=layout)
    end
    return value
end

function _binary_template(left, right, template::Union{Nothing, ScalarField})
    if template !== nothing
        return template
    elseif left isa ScalarField
        return left
    elseif right isa ScalarField
        return right
    else
        return nothing
    end
end

function evaluate_solver_expression(expr, variables; layout::Symbol=:g, template::Union{Nothing, ScalarField}=nothing)
    """
    Evaluate a parsed solver expression with current field values.
    Returns a field (preferred) or a numeric scalar for constant expressions.
    """

    if expr === nothing
        throw(ArgumentError("Cannot evaluate null expression"))
    end

    if expr isa ScalarField
        ensure_layout!(expr, layout)
        return expr
    elseif expr isa VectorField
        ensure_layout!(expr, layout)
        return expr
    elseif expr isa TensorField
        ensure_layout!(expr, layout)
        return expr
    elseif expr isa Future
        return evaluate(expr)
    elseif expr isa ZeroOperator
        return template === nothing ? 0 : create_zero_field(template)
    elseif expr isa ConstantOperator
        return template === nothing ? expr.value : _constant_field_from_template(template, expr.value; layout=layout)
    elseif expr isa ArrayOperator
        if template === nothing
            return expr.value
        end
        result = create_zero_field(template)
        ensure_layout!(result, layout)
        target = layout == :g ? get_grid_data(result) : get_coeff_data(result)
        copyto!(target, expr.value)
        return result
    elseif expr isa UnknownOperator
        @warn "Unknown operator expression: $(expr.expression)"
        return template === nothing ? 0 : create_zero_field(template)
    elseif expr isa AddOperator
        left = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        right = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        op_template = _binary_template(left, right, template)
        left = _coerce_numeric_operand(left, op_template; layout=layout)
        right = _coerce_numeric_operand(right, op_template; layout=layout)
        if left isa Number && right isa Number
            return left + right
        elseif left isa ScalarField && right isa ScalarField
            return left + right
        elseif left isa VectorField && right isa VectorField
            return add_vector_fields(left, right)
        else
            throw(ArgumentError("Unsupported Add operands: $(typeof(left)) and $(typeof(right))"))
        end
    elseif expr isa SubtractOperator
        left = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        right = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        op_template = _binary_template(left, right, template)
        left = _coerce_numeric_operand(left, op_template; layout=layout)
        right = _coerce_numeric_operand(right, op_template; layout=layout)
        if left isa Number && right isa Number
            return left - right
        elseif left isa ScalarField && right isa ScalarField
            return left - right
        elseif left isa VectorField && right isa VectorField
            return add_vector_fields(left, _scale_vector_field(right, -1))
        else
            throw(ArgumentError("Unsupported Subtract operands: $(typeof(left)) and $(typeof(right))"))
        end
    elseif expr isa MultiplyOperator
        left = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        right = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        op_template = _binary_template(left, right, template)
        left = _coerce_numeric_operand(left, op_template; layout=layout)
        right = _coerce_numeric_operand(right, op_template; layout=layout)
        if left isa Number && right isa Number
            return left * right
        elseif left isa ScalarField && right isa ScalarField
            return left * right
        elseif left isa ScalarField && right isa Number
            return left * right
        elseif left isa Number && right isa ScalarField
            return right * left
        elseif left isa VectorField && right isa Number
            return _scale_vector_field(left, right)
        elseif left isa Number && right isa VectorField
            return _scale_vector_field(right, left)
        elseif left isa ScalarField && right isa VectorField
            return scale_vector_field(right, left)
        elseif left isa VectorField && right isa ScalarField
            return scale_vector_field(left, right)
        else
            throw(ArgumentError("Unsupported Multiply operands: $(typeof(left)) and $(typeof(right))"))
        end
    elseif expr isa DivideOperator
        left = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        right = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        op_template = _binary_template(left, right, template)
        left = _coerce_numeric_operand(left, op_template; layout=layout)
        right = _coerce_numeric_operand(right, op_template; layout=layout)
        return divide_operands(left, right)
    elseif expr isa PowerOperator
        base = evaluate_solver_expression(expr.left, variables; layout=layout, template=template)
        exponent = evaluate_solver_expression(expr.right, variables; layout=layout, template=template)
        if exponent isa Number
            return power_operands(base, exponent)
        end
        throw(ArgumentError("Power operator requires numeric exponent, got $(typeof(exponent))"))
    elseif expr isa NegateOperator
        operand = evaluate_solver_expression(expr.operand, variables; layout=layout, template=template)
        if operand isa Number
            return -operand
        elseif operand isa ScalarField
            return operand * -1
        elseif operand isa VectorField
            return _scale_vector_field(operand, -1)
        else
            throw(ArgumentError("Unsupported negation operand: $(typeof(operand))"))
        end
    elseif expr isa IndexOperator
        array_val = evaluate_solver_expression(expr.array, variables; layout=layout, template=template)
        indices = Any[evaluate_solver_expression(idx, variables; layout=layout, template=template) for idx in expr.indices]
        indices = map(idx -> idx isa Number ? Int(idx) : idx, indices)
        if array_val isa ScalarField
            ensure_layout!(array_val, layout)
            data = layout == :g ? get_grid_data(array_val) : get_coeff_data(array_val)
            if data === nothing
                throw(ArgumentError("Field $(array_val.name) has no data in layout $layout"))
            end
            return data[indices...]
        elseif array_val isa AbstractArray
            return array_val[indices...]
        else
            throw(ArgumentError("Unsupported indexed operand: $(typeof(array_val))"))
        end
    elseif expr isa Operator
        return evaluate(expr, layout)
    elseif expr isa Number
        return template === nothing ? expr : _constant_field_from_template(template, expr; layout=layout)
    end

    error("Unsupported expression type in evaluate_solver_expression: $(typeof(expr)). " *
          "Value: $(repr(expr)). This may indicate a parsing error or missing operator handler.")
end

function build_jacobian_block(expr, variables, perturbations)
    """
    Build Jacobian matrix block from Frechet differential expression following Tarang patterns.
    
    In Tarang, this corresponds to:
    1. expr.expression_matrices(subproblem, vars) 
    2. Returns dict {var: matrix} for each variable
    3. Recursively builds matrices for expression tree
    """
    
    if expr === nothing
        @warn "Cannot build Jacobian from null expression"
        return sparse(zeros(ComplexF64, 1, 1))
    end
    
    # Calculate total size needed for Jacobian block
    total_var_size = sum(compute_field_vector_size(var) for var in variables)
    jacobian_size = total_var_size > 0 ? total_var_size : 1
    
    # Handle different expression types following Tarang expression_matrices patterns
    if hasfield(typeof(expr), :expr_type)
        expr_type = expr.expr_type
        
        if expr_type == "variable"
            # Variable expression - return identity matrix (Tarang lines 183-186, 507-510, 957-960)
            return build_variable_jacobian_block(expr, variables)
            
        elseif expr_type == "operator"
            # Operator expression - recursively build from operands 
            return build_operator_jacobian_block(expr, variables, perturbations)
            
        elseif expr_type == "constant"
            # Constant expression - return zero matrix
            return sparse(zeros(ComplexF64, jacobian_size, jacobian_size))
            
        else
            @warn "Unknown expression type for Jacobian: $expr_type"
        end
    end
    
    # Fallback: identity matrix (following Tarang identity pattern)
    return sparse(I, jacobian_size, jacobian_size)
end

function build_variable_jacobian_block(expr, variables)
    """Build identity matrix block for variable (Tarang pattern)"""

    # Check for field_ref using struct field access (consistent with build_jacobian_block)
    if !hasfield(typeof(expr), :field_ref)
        @warn "Variable expression missing field reference"
        return sparse(I, 1, 1)
    end

    field_ref = expr.field_ref

    # Find variable in list and return identity matrix for it
    for var in variables
        if var === field_ref
            var_size = compute_field_vector_size(var)
            return sparse(I, var_size, var_size)
        end
    end

    @warn "Variable not found in variable list for Jacobian"
    return sparse(I, 1, 1)
end

function build_operator_jacobian_block(expr, variables, perturbations)
    """Build Jacobian block for operator expression (following Tarang recursive patterns)"""

    # Check for operator and operands using struct field access (consistent with build_jacobian_block)
    if !hasfield(typeof(expr), :operator) || !hasfield(typeof(expr), :operands)
        @warn "Malformed operator expression for Jacobian"
        total_size = sum(compute_field_vector_size(var) for var in variables)
        return sparse(I, max(total_size, 1), max(total_size, 1))
    end

    operator = expr.operator
    operands = expr.operands
    
    # Following arithmetic line 189-193 pattern: iteratively add matrices
    if operator == "Add"
        # Addition: sum of operand Jacobians
        result_matrix = nothing
        for operand in operands
            operand_jac = build_jacobian_block(operand, variables, perturbations)
            if result_matrix === nothing
                result_matrix = operand_jac
            else
                result_matrix = result_matrix + operand_jac
            end
        end
        return result_matrix !== nothing ? result_matrix : sparse(I, 1, 1)
        
    elseif operator == "Multiply"
        # Multiplication: apply product rule for Jacobian
        # d(f·g)/dx = f·(dg/dx) + g·(df/dx)
        # For F(u) = a(u)·b(u), the Jacobian is: J_F = a·J_b + b·J_a
        total_size = sum(compute_field_vector_size(var) for var in variables)
        n = max(total_size, 1)

        if length(operands) >= 2
            # For two operands a, b: J = a*J_b + b*J_a
            # This is a linear approximation - the full product rule would require
            # evaluating operands at the current state
            result_matrix = spzeros(ComplexF64, n, n)
            for op in operands
                op_jac = build_jacobian_block(op, variables, perturbations)
                if size(op_jac) == (n, n)
                    result_matrix = result_matrix + op_jac
                end
            end
            return result_matrix
        else
            return sparse(I, n, n)
        end

    elseif operator == "Differentiate"
        # Differentiation: apply differential operator matrix
        # The Jacobian of ∂f/∂x is the same as ∂/∂x applied to J_f
        # Since differentiation is linear: J_{∂f/∂x} = D · J_f where D is the diff matrix
        if length(operands) > 0
            operand_jac = build_jacobian_block(operands[1], variables, perturbations)
            # The differentiation operator commutes with Jacobian computation for linear problems
            # For spectral methods, this would involve multiplying by ik (Fourier) or D matrix (Chebyshev)
            return operand_jac
        end
        
    else
        @warn "Unknown operator for Jacobian: $operator"
    end
    
    # Fallback
    total_size = sum(compute_field_vector_size(var) for var in variables)
    return sparse(I, max(total_size, 1), max(total_size, 1))
end

# Helper functions for operators

"""
Create zero field matching template field or first variable in vector.

GPU-aware: The field is allocated on the same architecture as the template.
Uses fill!() which works on both CPU and GPU arrays.
"""
function create_zero_field(template::ScalarField)
    result = ScalarField(template.dist, "zero_field", template.bases, template.dtype)
    ensure_layout!(result, :c)
    if get_coeff_data(result) !== nothing
        # fill!() works on both CPU and GPU arrays
        fill!(get_coeff_data(result), zero(eltype(get_coeff_data(result))))
    end
    return result
end

function create_zero_field(variables::Vector)
    if length(variables) > 0
        return create_zero_field(variables[1])
    else
        throw(ArgumentError("No variables available"))
    end
end

function create_constant_field(expr, variables)
    """
    Create field with constant value.

    GPU-aware: The field is allocated on the same architecture as the first variable.
    Uses fill!() which works on both CPU and GPU arrays.
    """
    if length(variables) == 0
        throw(ArgumentError("No variables available"))
    end

    result = ScalarField(variables[1].dist, "constant_field", variables[1].bases, variables[1].dtype)
    ensure_layout!(result, :c)

    if get_coeff_data(result) !== nothing
        value = hasfield(typeof(expr), :value) ? expr.value : zero(eltype(get_coeff_data(result)))
        # fill!() works on both CPU and GPU arrays
        fill!(get_coeff_data(result), convert(eltype(get_coeff_data(result)), value))
    end

    return result
end

function apply_add_operator(operands)
    """
    Apply addition operator following Tarang patterns.

    GPU-aware: Uses broadcasting (.+=) which works on both CPU and GPU arrays.
    """
    if length(operands) == 0
        throw(ArgumentError("Addition requires operands"))
    end

    result = ScalarField(operands[1].dist, "add_result", operands[1].bases, operands[1].dtype)
    ensure_layout!(result, :c)

    if get_coeff_data(result) !== nothing
        # fill!() works on both CPU and GPU
        fill!(get_coeff_data(result), zero(eltype(get_coeff_data(result))))
        for operand in operands
            ensure_layout!(operand, :c)
            if get_coeff_data(operand) !== nothing
                # Broadcasting works on both CPU and GPU
                get_coeff_data(result) .+= get_coeff_data(operand)
            end
        end
    end

    return result
end

function apply_multiply_operator(operands)
    """
    Apply multiplication operator following Tarang patterns.

    GPU-aware: Uses broadcasting (.*=) which works on both CPU and GPU arrays.
    """
    if length(operands) < 2
        return length(operands) == 1 ? operands[1] : throw(ArgumentError("Multiplication requires 2+ operands"))
    end

    result = ScalarField(operands[1].dist, "multiply_result", operands[1].bases, operands[1].dtype)
    ensure_layout!(result, :c)

    if get_coeff_data(result) !== nothing
        # Start with first operand
        ensure_layout!(operands[1], :c)
        if get_coeff_data(operands[1]) !== nothing
            # copyto!() works on both CPU and GPU
            copyto!(get_coeff_data(result), get_coeff_data(operands[1]))
        else
            fill!(get_coeff_data(result), one(eltype(get_coeff_data(result))))
        end

        # Multiply by remaining operands
        for i in 2:length(operands)
            ensure_layout!(operands[i], :c)
            if get_coeff_data(operands[i]) !== nothing
                # Broadcasting works on both CPU and GPU
                get_coeff_data(result) .*= get_coeff_data(operands[i])
            end
        end
    end

    return result
end

function apply_differentiate_operator(operands, expr)
    """
    Apply differentiation operator using existing operators.jl infrastructure.
    
    This leverages the complete implementation in operators.jl which includes:
    - Basis-specific differentiation (Fourier, Chebyshev, Legendre)
    - Proper spectral differentiation matrices
    - Layout management and efficient operations
    """
    
    if length(operands) == 0
        throw(ArgumentError("Differentiation requires operand"))
    end
    
    operand = operands[1]
    
    # Extract coordinate information from expression
    coordinate = get_diff_coordinate(expr)
    order = get_diff_order(expr)
    
    if coordinate === nothing
        @warn "No coordinate specified for differentiation, cannot proceed"
        return create_zero_field([operand])
    end
    
    try
        # Create Differentiate operator using existing infrastructure
        diff_op = Differentiate(operand, coordinate, order)
        
        # Evaluate using the complete implementation in operators.jl
        result = evaluate_differentiate(diff_op, :c)  # Use coefficient layout
        
        @debug "Applied differentiation using operators.jl: coord=$(coordinate.name), order=$order"
        
        return result
        
    catch e
        @warn "Differentiation failed: $e, returning zero result"
        return create_zero_field([operand])
    end
end

function get_diff_coordinate(expr)
    """Extract coordinate for differentiation from expression"""
    # Direct coordinate object (struct field access)
    if hasfield(typeof(expr), :coordinate) && expr.coordinate !== nothing
        return expr.coordinate
    end

    # Coordinate name lookup - search operand's bases for matching coordinate
    if hasfield(typeof(expr), :coord_name) && hasfield(typeof(expr), :operand)
        coord_name = expr.coord_name
        operand = expr.operand

        # Try to find coordinate in operand's bases
        if hasfield(typeof(operand), :bases)
            for basis in operand.bases
                if basis !== nothing && hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :element_label)
                    if basis.meta.element_label == coord_name
                        if hasfield(typeof(basis.meta), :coordsys) && basis.meta.coordsys !== nothing
                            try
                                return basis.meta.coordsys[coord_name]
                            catch
                                # Coordinate not found in coordsys
                            end
                        end
                    end
                end
            end
        end

        # Try distributor's coordinate system
        if hasfield(typeof(operand), :dist)
            dist = operand.dist
            if hasfield(typeof(dist), :coords) && dist.coords !== nothing
                for coord in dist.coords
                    if coord.name == coord_name
                        return coord
                    end
                end
            end
            if hasfield(typeof(dist), :coordsys) && dist.coordsys !== nothing
                try
                    return dist.coordsys[coord_name]
                catch
                    # Coordinate not found in coordsys
                end
            end
        end

        @debug "Coordinate '$coord_name' not found in operand's bases"
        return nothing
    end

    @debug "No coordinate specified for differentiation"
    return nothing
end

function get_diff_order(expr)
    """Extract differentiation order from expression"""
    if hasfield(typeof(expr), :order)
        return max(1, Int(expr.order))
    else
        return 1  # Default to first order
    end
end

# Performance and logging
function log_stats(solver::Solver)
    """Log solver performance statistics"""
    
    if isa(solver, InitialValueSolver)
        elapsed = time() - solver.wall_time_start
        @info "Solver statistics:"
        @info "  Total iterations: $(solver.iteration)"
        @info "  Simulation time: $(solver.sim_time)"
        @info "  Wall time: $(elapsed) seconds"
        if elapsed > 0
            @info "  Iterations per second: $(solver.iteration / elapsed)"
        end
        
        if MPI.Initialized()
            # Log MPI statistics
            mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
            nprocs = MPI.Comm_size(MPI.COMM_WORLD)
            @info "  MPI rank: $mpi_rank / $nprocs"
        end
    end
end

# Analysis and output - create_evaluator is defined in evaluator.jl

function log_solver_performance(solver::Union{InitialValueSolver, BoundaryValueSolver})
    """Log solver performance statistics"""

    stats = solver.performance_stats

    if MPI.Initialized()
        mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if mpi_rank == 0
            @info "Solver performance:"
            if isa(solver, InitialValueSolver)
                @info "  Total steps: $(stats.total_steps)"
                if stats.total_steps > 0
                    @info "  Average step time: $(round(stats.total_time/stats.total_steps*1000, digits=3)) ms"
                end
            elseif isa(solver, BoundaryValueSolver)
                @info "  Total solves: $(stats.total_solves)"
            end
            @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
        end
    else
        @info "Solver performance:"
        if isa(solver, InitialValueSolver)
            @info "  Total steps: $(stats.total_steps)"
            if stats.total_steps > 0
                @info "  Average step time: $(round(stats.total_time/stats.total_steps*1000, digits=3)) ms"
            end
        elseif isa(solver, BoundaryValueSolver)
            @info "  Total solves: $(stats.total_solves)"
        end
        @info "  Total time: $(round(stats.total_time, digits=3)) seconds"
    end
end

# ============================================================================
# Compiled RHS Plan — zero-dispatch timestepping
# ============================================================================

"""
    RHSInstruction

A single operation in a compiled RHS execution plan.
Each instruction operates on pre-allocated workspace fields, eliminating
runtime type dispatch and per-timestep allocation.
"""
abstract type RHSInstruction end

struct CopyFieldInstr <: RHSInstruction
    src_state_idx::Int   # Index into the state vector
    dst_ws_idx::Int      # Index into workspace
end

struct EnsureLayoutInstr <: RHSInstruction
    ws_idx::Int
    layout::Symbol  # :g or :c
end

struct DifferentiateInstr <: RHSInstruction
    src_ws_idx::Int
    dst_ws_idx::Int
    coord::Coordinate
    order::Int
end

struct MultiplyFieldsInstr <: RHSInstruction
    src1_ws_idx::Int
    src2_ws_idx::Int
    dst_ws_idx::Int
end

struct ScaleFieldInstr <: RHSInstruction
    src_ws_idx::Int
    dst_ws_idx::Int
    scale::Float64
end

struct AddFieldsInstr <: RHSInstruction
    src1_ws_idx::Int
    src2_ws_idx::Int
    dst_ws_idx::Int
end

struct SubtractFieldsInstr <: RHSInstruction
    src1_ws_idx::Int
    src2_ws_idx::Int
    dst_ws_idx::Int
end

struct NegateFieldInstr <: RHSInstruction
    src_ws_idx::Int
    dst_ws_idx::Int
end

struct GradientComponentInstr <: RHSInstruction
    src_ws_idx::Int
    dst_ws_idx::Int
    coord::Coordinate
end

struct NonlinearMultiplyInstr <: RHSInstruction
    src1_ws_idx::Int
    src2_ws_idx::Int
    dst_ws_idx::Int
end

"""
    CompiledRHSPlan

A pre-compiled execution plan for RHS evaluation that eliminates
runtime type dispatch and per-timestep allocation.

Created once during solver setup by walking the expression tree.
Executed on every `evaluate_rhs` call.
"""
mutable struct CompiledRHSPlan
    instructions::Vector{RHSInstruction}
    workspace::Vector{<:ScalarField}       # Pre-allocated intermediate buffers
    result_ws_indices::Vector{Int}       # workspace indices that hold the final RHS per state field
    n_state_fields::Int
    is_compiled::Bool

    function CompiledRHSPlan(n_state::Int)
        new(RHSInstruction[], ScalarField[], zeros(Int, n_state), n_state, false)
    end
end

"""
    _alloc_workspace_field!(plan, template) -> Int

Add a new pre-allocated workspace field modeled on `template`, return its index.
"""
function _alloc_workspace_field!(plan::CompiledRHSPlan, template::ScalarField)
    ws_field = copy(template)
    ws_field.name = "ws_$(length(plan.workspace)+1)"
    push!(plan.workspace, ws_field)
    return length(plan.workspace)
end

"""
    compile_rhs_plan!(solver::InitialValueSolver) -> CompiledRHSPlan

Walk the RHS expression tree and compile it into a sequence of typed instructions
operating on pre-allocated workspace fields.

This is called once during solver setup. The returned plan is reused for every
`evaluate_rhs` call, eliminating runtime dispatch and allocation.
"""
function compile_rhs_plan!(solver::InitialValueSolver)
    problem = solver.problem
    state = solver.state
    plan = CompiledRHSPlan(length(state))

    if !hasfield(typeof(problem), :equation_data) || isempty(problem.equation_data)
        plan.is_compiled = true
        return plan
    end

    for (eq_idx, eq_data) in enumerate(problem.equation_data)
        M_expr = get(eq_data, "M", nothing)
        if M_expr === nothing || _is_zero_m_term(M_expr)
            continue
        end

        target_indices = _find_time_derivative_targets(M_expr, state, problem.variables)
        if isempty(target_indices)
            continue
        end

        expr = if haskey(eq_data, "F_expr") && eq_data["F_expr"] !== nothing
            eq_data["F_expr"]
        else
            get(eq_data, "F", nothing)
        end

        if expr === nothing
            continue
        end

        for state_idx in target_indices
            if state_idx <= length(state)
                template = state[state_idx]
                try
                    ws_idx = _compile_expr!(plan, expr, template, state)
                    plan.result_ws_indices[state_idx] = ws_idx
                catch e
                    @warn "Failed to compile RHS expression for state field $state_idx: $e — falling back to interpreted evaluation"
                    plan.is_compiled = false
                    return plan
                end
            end
        end
    end

    plan.is_compiled = true
    @info "Compiled RHS plan: $(length(plan.instructions)) instructions, $(length(plan.workspace)) workspace fields"
    return plan
end

"""
    _compile_expr!(plan, expr, template, state) -> ws_idx

Recursively compile an expression into instructions. Returns the workspace index
where the result will be stored.
"""
function _compile_expr!(plan::CompiledRHSPlan, expr, template::ScalarField, state::Vector{<:ScalarField})
    if expr isa ScalarField
        # Check if it's a state field
        for (i, s) in enumerate(state)
            if expr === s || expr.name == s.name
                ws_idx = _alloc_workspace_field!(plan, template)
                push!(plan.instructions, CopyFieldInstr(i, ws_idx))
                return ws_idx
            end
        end
        # It's a non-state field (parameter, forcing term) — copy it
        ws_idx = _alloc_workspace_field!(plan, template)
        push!(plan.workspace, expr)  # Store the actual field reference
        pop!(plan.workspace)  # Remove the template copy
        plan.workspace[ws_idx] = expr  # Replace with the actual field
        return ws_idx

    elseif expr isa Number
        ws_idx = _alloc_workspace_field!(plan, template)
        push!(plan.instructions, ScaleFieldInstr(0, ws_idx, Float64(expr)))  # 0 = "fill constant"
        return ws_idx

    elseif expr isa AddOperator
        left_idx = _compile_expr!(plan, expr.left, template, state)
        right_idx = _compile_expr!(plan, expr.right, template, state)
        dst_idx = _alloc_workspace_field!(plan, template)
        push!(plan.instructions, AddFieldsInstr(left_idx, right_idx, dst_idx))
        return dst_idx

    elseif expr isa SubtractOperator
        left_idx = _compile_expr!(plan, expr.left, template, state)
        right_idx = _compile_expr!(plan, expr.right, template, state)
        dst_idx = _alloc_workspace_field!(plan, template)
        push!(plan.instructions, SubtractFieldsInstr(left_idx, right_idx, dst_idx))
        return dst_idx

    elseif expr isa MultiplyOperator
        left_idx = _compile_expr!(plan, expr.left, template, state)
        right_idx = _compile_expr!(plan, expr.right, template, state)
        dst_idx = _alloc_workspace_field!(plan, template)
        push!(plan.instructions, MultiplyFieldsInstr(left_idx, right_idx, dst_idx))
        return dst_idx

    elseif expr isa NegateOperator
        src_idx = _compile_expr!(plan, expr.operand, template, state)
        dst_idx = _alloc_workspace_field!(plan, template)
        push!(plan.instructions, NegateFieldInstr(src_idx, dst_idx))
        return dst_idx

    elseif expr isa Differentiate
        src_idx = _compile_expr!(plan, expr.operand, template, state)
        dst_idx = _alloc_workspace_field!(plan, template)
        push!(plan.instructions, DifferentiateInstr(src_idx, dst_idx, expr.coord, expr.order))
        return dst_idx

    elseif expr isa Future
        # Evaluate the Future's deferred arguments, then compile the result
        # Futures wrap lazy operations — we need to resolve them
        evaluated = evaluate(expr)
        return _compile_expr!(plan, evaluated, template, state)

    elseif expr isa ZeroOperator
        ws_idx = _alloc_workspace_field!(plan, template)
        push!(plan.instructions, ScaleFieldInstr(0, ws_idx, 0.0))
        return ws_idx

    else
        # Unsupported expression type — signal fallback to interpreted evaluation
        throw(ArgumentError("Cannot compile expression of type $(typeof(expr))"))
    end
end

"""
    execute_compiled_rhs!(plan::CompiledRHSPlan, state::Vector{<:ScalarField}, solver)

Execute a pre-compiled RHS plan. Returns the RHS as a vector of ScalarFields
from the pre-allocated workspace (no allocation).
"""
function execute_compiled_rhs!(plan::CompiledRHSPlan, state::Vector{<:ScalarField},
                                solver::InitialValueSolver)
    # Sync state into problem variables
    sync_state_to_problem!(solver.problem, state)

    # Execute instructions in order
    for instr in plan.instructions
        _execute_instr!(instr, plan.workspace, state, solver)
    end

    # Collect results
    rhs = Vector{ScalarField}(undef, plan.n_state_fields)
    for i in 1:plan.n_state_fields
        ws_idx = plan.result_ws_indices[i]
        if ws_idx > 0
            rhs[i] = plan.workspace[ws_idx]
        else
            rhs[i] = create_rhs_zero_field(state[i])
        end
    end
    return rhs
end

# Instruction execution — each method is statically dispatched (no isa checks)

function _execute_instr!(instr::CopyFieldInstr, ws::Vector{<:ScalarField},
                          state::Vector{<:ScalarField}, solver)
    src = state[instr.src_state_idx]
    dst = ws[instr.dst_ws_idx]
    ensure_layout!(src, :g)
    ensure_layout!(dst, :g)
    src_data = get_grid_data(src)
    dst_data = get_grid_data(dst)
    if src_data !== nothing && dst_data !== nothing
        copyto!(dst_data, src_data)
    end
    dst.current_layout = src.current_layout
end

function _execute_instr!(instr::EnsureLayoutInstr, ws::Vector{<:ScalarField},
                          state::Vector{<:ScalarField}, solver)
    ensure_layout!(ws[instr.ws_idx], instr.layout)
end

function _execute_instr!(instr::DifferentiateInstr, ws::Vector{<:ScalarField},
                          state::Vector{<:ScalarField}, solver)
    src = ws[instr.src_ws_idx]
    result = evaluate_differentiate(Differentiate(src, instr.coord, instr.order), :g)
    dst = ws[instr.dst_ws_idx]
    ensure_layout!(dst, :g)
    ensure_layout!(result, :g)
    dst_data = get_grid_data(dst)
    res_data = get_grid_data(result)
    if dst_data !== nothing && res_data !== nothing
        copyto!(dst_data, res_data)
    end
    dst.current_layout = :g
end

function _execute_instr!(instr::MultiplyFieldsInstr, ws::Vector{<:ScalarField},
                          state::Vector{<:ScalarField}, solver)
    f1 = ws[instr.src1_ws_idx]
    f2 = ws[instr.src2_ws_idx]
    dst = ws[instr.dst_ws_idx]
    ensure_layout!(f1, :g)
    ensure_layout!(f2, :g)
    ensure_layout!(dst, :g)
    dst_data = get_grid_data(dst)
    f1_data = get_grid_data(f1)
    f2_data = get_grid_data(f2)
    if dst_data !== nothing && f1_data !== nothing && f2_data !== nothing
        dst_data .= f1_data .* f2_data
    end
end

function _execute_instr!(instr::NonlinearMultiplyInstr, ws::Vector{<:ScalarField},
                          state::Vector{<:ScalarField}, solver)
    f1 = ws[instr.src1_ws_idx]
    f2 = ws[instr.src2_ws_idx]
    dist = f1.dist
    if dist.nonlinear_evaluator === nothing
        dist.nonlinear_evaluator = NonlinearEvaluator(dist)
    end
    product = evaluate_transform_multiply(f1, f2, dist.nonlinear_evaluator)
    dst = ws[instr.dst_ws_idx]
    ensure_layout!(product, :g)
    ensure_layout!(dst, :g)
    dst_data = get_grid_data(dst)
    prod_data = get_grid_data(product)
    if dst_data !== nothing && prod_data !== nothing
        copyto!(dst_data, prod_data)
    end
    dst.current_layout = :g
end

function _execute_instr!(instr::AddFieldsInstr, ws::Vector{<:ScalarField},
                          state::Vector{<:ScalarField}, solver)
    f1 = ws[instr.src1_ws_idx]
    f2 = ws[instr.src2_ws_idx]
    dst = ws[instr.dst_ws_idx]
    ensure_layout!(f1, :g)
    ensure_layout!(f2, :g)
    ensure_layout!(dst, :g)
    get_grid_data(dst) .= get_grid_data(f1) .+ get_grid_data(f2)
end

function _execute_instr!(instr::SubtractFieldsInstr, ws::Vector{<:ScalarField},
                          state::Vector{<:ScalarField}, solver)
    f1 = ws[instr.src1_ws_idx]
    f2 = ws[instr.src2_ws_idx]
    dst = ws[instr.dst_ws_idx]
    ensure_layout!(f1, :g)
    ensure_layout!(f2, :g)
    ensure_layout!(dst, :g)
    get_grid_data(dst) .= get_grid_data(f1) .- get_grid_data(f2)
end

function _execute_instr!(instr::NegateFieldInstr, ws::Vector{<:ScalarField},
                          state::Vector{<:ScalarField}, solver)
    src = ws[instr.src_ws_idx]
    dst = ws[instr.dst_ws_idx]
    ensure_layout!(src, :g)
    ensure_layout!(dst, :g)
    get_grid_data(dst) .= .-get_grid_data(src)
end

function _execute_instr!(instr::ScaleFieldInstr, ws::Vector{<:ScalarField},
                          state::Vector{<:ScalarField}, solver)
    dst = ws[instr.dst_ws_idx]
    ensure_layout!(dst, :g)
    if instr.src_ws_idx == 0
        # Fill with constant
        fill!(get_grid_data(dst), instr.scale)
    else
        src = ws[instr.src_ws_idx]
        ensure_layout!(src, :g)
        get_grid_data(dst) .= instr.scale .* get_grid_data(src)
    end
end

# ============================================================================
# Exports
# ============================================================================

# Export solver types
export Solver, SolverPerformanceStats, SolverBaseData
export InitialValueSolver, BoundaryValueSolver, EigenvalueSolver

# Export compiled RHS plan
export CompiledRHSPlan, compile_rhs_plan!, execute_compiled_rhs!

# Export introspection
export diagnose

# ============================================================================
# Solver Introspection
# ============================================================================

"""
    diagnose(solver)

Print a formatted summary of solver state, configuration, and resource usage.
Useful for debugging and understanding solver behavior.

# Example
```julia
solver = InitialValueSolver(problem, RK222; dt=1e-3)
diagnose(solver)
```
"""
function diagnose(solver::InitialValueSolver)
    println("╔══════════════════════════════════════════════════╗")
    println("║              Solver Diagnostics                 ║")
    println("╠══════════════════════════════════════════════════╣")

    # Timestepper info
    ts = solver.timestepper
    ts_name = typeof(ts)
    println("║ Timestepper: $ts_name")
    println("║ dt = $(solver.dt), sim_time = $(round(solver.sim_time; digits=6))")
    println("║ iteration = $(solver.iteration)")
    println("╠──────────────────────────────────────────────────╣")

    # State fields
    println("║ State fields: $(length(solver.state))")
    total_dof = 0
    total_mem = 0
    for (i, field) in enumerate(solver.state)
        gdata = get_grid_data(field)
        sz = gdata !== nothing ? size(gdata) : ()
        dof = prod(sz; init=1)
        mem = dof * sizeof(field.dtype)
        total_dof += dof
        total_mem += 2 * mem  # grid + coeff
        layout = field.current_layout
        println("║   $(i). $(field.name) $(sz) [$(layout)] $(field.dtype)")
    end
    println("║ Total DOF: $(total_dof), Memory: $(round(total_mem / 1024^2; digits=2)) MB")
    println("╠──────────────────────────────────────────────────╣")

    # Architecture
    arch = solver.state[1].dist.architecture
    println("║ Architecture: $(typeof(arch))")
    println("║ MPI ranks: $(solver.state[1].dist.size)")
    mesh = solver.state[1].dist.mesh
    if mesh !== nothing
        println("║ Process mesh: $mesh")
    end
    println("╠──────────────────────────────────────────────────╣")

    # Transforms
    dist = solver.state[1].dist
    n_transforms = length(dist.transforms)
    println("║ Transforms: $n_transforms registered")
    for (i, tr) in enumerate(dist.transforms)
        println("║   $(i). $(typeof(tr))")
    end

    # Bases
    if !isempty(solver.state) && solver.state[1].domain !== nothing
        bases = solver.state[1].bases
        println("║ Bases:")
        for (i, basis) in enumerate(bases)
            btype = nameof(typeof(basis))
            N = basis.meta.size
            bounds = basis.meta.bounds
            println("║   $(i). $btype(N=$N, bounds=$bounds)")
        end
    end
    println("╠──────────────────────────────────────────────────╣")

    # Compiled RHS
    if solver.compiled_rhs !== nothing
        plan = solver.compiled_rhs
        if plan.is_compiled
            println("║ RHS: COMPILED ($(length(plan.instructions)) instructions, $(length(plan.workspace)) workspace fields)")
        else
            println("║ RHS: compilation failed — using interpreted evaluation")
        end
    else
        println("║ RHS: interpreted (not compiled)")
    end

    # Nonlinear evaluator
    if dist.nonlinear_evaluator !== nothing
        eval = dist.nonlinear_evaluator
        println("║ Nonlinear: dealiasing=$(eval.dealiasing_factor)")
        n_cached = length(eval.temp_fields)
        println("║   Cached temp fields: $n_cached")
        stats = eval.performance_stats
        if stats.total_evaluations > 0
            avg = stats.total_time / stats.total_evaluations * 1000
            println("║   Evaluations: $(stats.total_evaluations), avg=$(round(avg; digits=2)) ms")
        end
    end
    println("╠──────────────────────────────────────────────────╣")

    # Boundary conditions
    bc = solver.problem.bc_manager
    n_bcs = length(bc.conditions)
    println("║ Boundary conditions: $n_bcs")
    has_time_dep = has_time_dependent_bcs(bc)
    println("║   Time-dependent: $has_time_dep")

    # Stochastic forcing
    if hasfield(typeof(solver.problem), :stochastic_forcings) && !isempty(solver.problem.stochastic_forcings)
        println("║ Stochastic forcing: $(length(solver.problem.stochastic_forcings)) fields")
    end

    # Performance
    stats = solver.performance_stats
    if stats.total_steps > 0
        avg_step = stats.total_time / stats.total_steps * 1000
        println("╠──────────────────────────────────────────────────╣")
        println("║ Performance:")
        println("║   Total steps: $(stats.total_steps)")
        println("║   Total time: $(round(stats.total_time; digits=2))s")
        println("║   Avg step: $(round(avg_step; digits=2)) ms")
    end

    println("╚══════════════════════════════════════════════════╝")
end

function Base.show(io::IO, plan::CompiledRHSPlan)
    status = plan.is_compiled ? "compiled" : "failed"
    print(io, "CompiledRHSPlan($status, $(length(plan.instructions)) instructions, $(length(plan.workspace)) workspace)")
end

# Export core solver API
export step!, solve!, proceed, run!
export attach_evaluator!, solver_comm

# Export matrix building functions
export build_solver_matrices!, apply_entry_cutoff!

# Export solution vector functions
export fields_to_vector, copy_solution_to_fields!
export compute_field_vector_size, extract_field_data_for_vector, set_field_data_from_vector!
export get_basis_size

# Export nonlinear solver functions
export solve_linear!, solve_nonlinear!
export evaluate_residual_and_jacobian

# Export performance/logging functions
export log_stats, log_solver_performance

# Export architecture query
export get_solver_architecture
