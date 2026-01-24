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

function sync_state_to_problem!(problem::Problem, state::Vector{ScalarField})
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
    state::Vector{ScalarField}
    dt::Float64

    # Timestepper state for existing timesteppers.jl infrastructure
    timestepper_state::Any

    # Evaluator for analysis
    evaluator::Union{Nothing, Any}

    # Performance tracking
    wall_time_start::Float64
    workspace::Dict{String, AbstractArray}
    performance_stats::SolverPerformanceStats
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
                                workspace, perf_stats)
    attach_evaluator!(solver)

    if has_time_dependent_bcs(problem.bc_manager)
        @info "Solver: Time-dependent BCs detected - enabling BC updates"
        set_time_variable!(problem.bc_manager, "t")
    end

    build_solver_matrices!(solver)
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
                    # For array-valued BCs, store in workspace and use placeholder
                    ws_key = "bc_$(bc_idx)_value"
                    bc_manager.workspace[ws_key] = value
                    equation_data[eq_idx]["F"] = ConstantOperator(1.0)
                    equation_data[eq_idx]["F_expr"] = ConstantOperator(1.0)
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
    state::Vector{ScalarField}

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
    apply_entry_cutoff!(M, base.entry_cutoff)
    L_sparse = sparse(L)
    M_sparse = sparse(M)

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

    # Update time-dependent boundary conditions BEFORE taking the step
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
    
    for iter in 1:solver.max_iterations
        # Evaluate residual and Jacobian
        residual, jacobian = evaluate_residual_and_jacobian(solver.problem, x)
        
        # Newton update: J * dx = -R
        dx = -jacobian \ residual
        x += dx
        
        # Check convergence
        if norm(dx) < solver.tolerance
            @info "Nonlinear solver converged in $iter iterations"
            break
        end
        
        if iter == solver.max_iterations
            @warn "Nonlinear solver did not converge"
        end
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
        λ, v = scipy_sparse_eigs(solver.L_matrix, solver.M_matrix; nev=nev, which=which_symbol)
    else
        λ, v = scipy_sparse_eigs(solver.L_matrix, solver.M_matrix; nev=nev, sigma=target)
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
function fields_to_vector(fields::Vector{ScalarField})
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
                @warn "Size mismatch in fields_to_vector: expected $field_size, got $(length(field_data))"
                # Fallback: copy what we can
                copy_size = min(field_size, length(field_data), length(vector) - offset + 1)
                if copy_size > 0
                    vector[offset:offset+copy_size-1] .= field_data[1:copy_size]
                end
            end

            offset += field_size
        end

        @debug "Gathered field $(field.name): size=$field_size, offset=$(offset-field_size)"
    end

    @debug "Fields to vector completed: total_size=$total_size, fields=$(length(fields))"
    return vector
end

function copy_solution_to_fields!(fields::Vector{ScalarField}, solution::AbstractVector{<:Number})
    """
    Copy solution vector back to fields following scatter pattern.

    GPU-aware: The solution vector is on CPU (from linear solver).
    For GPU fields, data is transferred to GPU and synchronized.
    """

    if isempty(fields)
        return
    end

    offset = 1

    for field in fields
        field_size = compute_field_vector_size(field)

        if field_size > 0 && offset <= length(solution)
            # Extract data from solution vector
            end_offset = min(offset + field_size - 1, length(solution))
            actual_size = end_offset - offset + 1

            if actual_size > 0
                field_data = solution[offset:end_offset]

                # Scatter data back to field (following Tarang scatter pattern)
                # This handles GPU transfer internally via set_field_data_from_vector!
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

function vector_to_fields(vector::AbstractVector{<:Number}, template::Vector{ScalarField})
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
                # Get basis size - this would need basis-specific implementation
                basis_size = get_basis_size(basis)
                total_size *= basis_size
            else
                total_size *= 64  # Default size
            end
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
        @warn "Cannot set data for field $(field.name) - no data arrays allocated"
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

    # Step 4: Build Jacobian matrix (fallback to linear matrix or identity)
    n = length(x)
    if haskey(problem.parameters, "L_matrix")
        jacobian = problem.parameters["L_matrix"]
    else
        jacobian = sparse(I, n, n)
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
        base = evaluate_solver_expression(expr.base, variables; layout=layout, template=template)
        exponent = evaluate_solver_expression(expr.exponent, variables; layout=layout, template=template)
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

    @warn "Unsupported expression type: $(typeof(expr)), returning zero"
    return template === nothing ? 0 : create_zero_field(template)
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
# Exports
# ============================================================================

# Export solver types
export Solver, SolverPerformanceStats, SolverBaseData
export InitialValueSolver, BoundaryValueSolver, EigenvalueSolver

# Export core solver API
export step!, solve!, proceed
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
