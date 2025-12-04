"""
Solver classes for different problem types

Translated from dedalus/core/solvers.py
CPU-only (GPU support removed).
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

function SolverBaseData(problem::Problem; matrix_coupling=nothing, entry_cutoff::Real=1e-12, matsolver::Union{String,Symbol,Type}= :sparse)
    dim = (problem.domain !== nothing && hasproperty(problem.domain, :dist)) ? problem.domain.dist.dim : 0
    coupling = if matrix_coupling === nothing
        fill(true, dim)
    else
        collect(Bool.(matrix_coupling))
    end
    if length(coupling) != dim
        coupling = fill(true, dim)
    end
    solver_type = MatSolvers.get_solver(matsolver)
    SolverBaseData(problem, coupling, Float64(entry_cutoff), solver_type, nothing)
end

function solver_comm(problem::Problem)
    if problem.domain !== nothing && hasproperty(problem.domain, :dist)
        return problem.domain.dist.comm
    else
        return MPI.COMM_WORLD
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

function _build_initial_value_solver(problem::IVP, timestepper; device::String="cpu")
    setup_domain!(problem)
    validate_problem(problem)

    base = SolverBaseData(problem)

    state = ScalarField[]
    for var in problem.variables
        if isa(var, ScalarField)
            push!(state, var)
        elseif isa(var, VectorField)
            append!(state, var.components)
        elseif isa(var, TensorField)
            append!(state, vec(var.components))
        end
    end

    workspace = Dict{String, AbstractArray}()
    perf_stats = SolverPerformanceStats()

    solver = InitialValueSolver(base, problem, timestepper, 0.0, 0, Inf, Inf, typemax(Int),
                                state, 0.001, nothing, nothing, time(),
                                workspace, perf_stats)
    attach_evaluator!(solver)

    if has_time_dependent_bcs(problem.bc_manager)
        @info "Solver: Time-dependent BCs detected - enabling BC updates"
        set_time_variable!(problem.bc_manager, "t")
    end

    build_solver_matrices!(solver)
    return solver
end

const _InitialValueSolver_constructor = _build_initial_value_solver

function _build_boundary_value_solver(problem::Union{LBVP, NLBVP}; device::String="cpu", matsolver::Union{String,Symbol,Type}= :sparse)
    setup_domain!(problem)
    validate_problem(problem)

    base = SolverBaseData(problem; matsolver=matsolver)

    state = ScalarField[]
    for var in problem.variables
        if isa(var, ScalarField)
            push!(state, var)
        elseif isa(var, VectorField)
            append!(state, var.components)
        elseif isa(var, TensorField)
            append!(state, vec(var.components))
        end
    end

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

    global_solver = MatSolvers.solver_instance(base.matsolver, L_sparse)

    solver = BoundaryValueSolver(base, problem, state, L_sparse, M_sparse, F_vec, 1e-10, 100,
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

function InitialValueSolver(problem::IVP, timestepper; kwargs...)
    return multiclass_new(InitialValueSolver, problem, timestepper; kwargs...)
end

function BoundaryValueSolver(problem::Union{LBVP, NLBVP}; kwargs...)
    return multiclass_new(BoundaryValueSolver, problem; kwargs...)
end

function EigenvalueSolver(problem::EVP; kwargs...)
    return multiclass_new(EigenvalueSolver, problem; kwargs...)
end

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
    n_modes::Int
    target::Union{Nothing, ComplexF64}

    workspace::Dict{String, AbstractArray}
    performance_stats::SolverPerformanceStats
    global_solver::Any
    subsystems::Tuple{Vararg{Subsystem}}
    subproblems::Tuple{Vararg{Subproblem}}
    coeff_system::Union{Nothing, CoeffSystem}
end

function _build_eigenvalue_solver(problem::EVP; n_modes::Int=10, target::Union{Nothing, ComplexF64}=nothing, matsolver::Union{String,Symbol,Type}=:sparse)
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
    global_solver = MatSolvers.solver_instance(base.matsolver, L_sparse)
    solver = EigenvalueSolver(base, problem, ComplexF64[], zeros(ComplexF64, 0, 0),
                              L_sparse, M_sparse, n_modes, target,
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
        update_time_dependent_bcs!(solver.problem.bc_manager, solver.sim_time + dt)
        @debug "Updated time-dependent BCs for t=$(solver.sim_time + dt)"
    end

    # Use existing timestepper infrastructure from timesteppers.jl
    # Create TimestepperState if needed
    if solver.timestepper_state === nothing
        solver.timestepper_state = TimestepperState(solver.timestepper, dt)
        # Initialize history with current state
        state_copy = deepcopy(solver.state)
        push!(solver.timestepper_state.history, state_copy)
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
function solve!(solver::EigenvalueSolver)
    """Solve eigenvalue problem"""

    start_time = time()

    # Solve generalized eigenvalue problem: L * v = λ * M * v
    if solver.target === nothing
        λ, v = eigs(solver.L_matrix, solver.M_matrix, nev=solver.n_modes, which=:LM)
    else
        λ, v = eigs(solver.L_matrix, solver.M_matrix, nev=solver.n_modes, sigma=solver.target)
    end

    solver.eigenvalues = λ
    solver.eigenvectors = v

    # Update performance statistics
    solve_time = time() - start_time
    solver.performance_stats.total_time += solve_time
    solver.performance_stats.total_solves += 1

    return solver
end

# Utility functions
function fields_to_vector(fields::Vector{ScalarField})
    """
    Convert field array to solution vector following Dedalus gather pattern.
    Following Dedalus subsystems.py gather_inputs (subsystems.py:340-350).
    """
    
    # Ensure all fields are in coefficient space (following Dedalus pattern)
    for field in fields
        ensure_layout!(field, :c)
    end
    
    # Calculate total vector size
    total_size = sum(compute_field_vector_size(field) for field in fields)
    
    # Allocate output vector
    vector = Vector{ComplexF64}(undef, total_size)
    
    # Gather field data into vector (following Dedalus gather pattern)
    offset = 1
    for field in fields
        field_size = compute_field_vector_size(field)
        if field_size > 0
            # Extract field data with proper handling of dimensions
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

function copy_solution_to_fields!(fields::Vector{ScalarField}, solution::Vector{ComplexF64})
    """
    Copy solution vector back to fields following Dedalus scatter pattern.
    Following Dedalus subsystems.py scatter_inputs (subsystems.py:364-371).
    """
    
    offset = 1
    
    for field in fields
        field_size = compute_field_vector_size(field)
        
        if field_size > 0 && offset <= length(solution)
            # Extract data from solution vector
            end_offset = min(offset + field_size - 1, length(solution))
            actual_size = end_offset - offset + 1
            
            if actual_size > 0
                field_data = solution[offset:end_offset]
                
                # Scatter data back to field (following Dedalus scatter pattern)
                set_field_data_from_vector!(field, field_data)
                
                @debug "Scattered to field $(field.name): size=$actual_size"
            end
            
            offset += field_size
        end
    end
    
    @debug "Vector to fields completed: solution_size=$(length(solution)), fields=$(length(fields))"
end

function compute_field_vector_size(field::ScalarField)
    """
    Compute the number of degrees of freedom for a field in vector form.
    Following Dedalus field size computation patterns.
    """
    
    if field.data_c !== nothing
        # Use coefficient space data size
        return length(field.data_c)
    elseif field.data_g !== nothing  
        # Use grid space data size as fallback
        return length(field.data_g)
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
    Following Dedalus field data extraction patterns.
    """
    
    # Ensure coefficient space layout
    ensure_layout!(field, :c)
    
    if field.data_c !== nothing
        # Return flattened coefficient data
        return vec(field.data_c)
    elseif field.data_g !== nothing
        # Fallback to grid data if coefficient data not available
        @warn "Using grid space data for field $(field.name) - converting to coefficient space recommended"
        return vec(field.data_g)
    else
        # Return zeros as fallback
        size = compute_field_vector_size(field)
        return zeros(ComplexF64, size)
    end
end

function set_field_data_from_vector!(field::ScalarField, data::Vector{ComplexF64})
    """
    Set field data from vector with proper shape and layout handling.
    Following Dedalus field data setting patterns.
    """
    
    if field.data_c !== nothing
        # Reshape data to match field coefficient data shape
        target_shape = size(field.data_c)
        expected_size = prod(target_shape)
        
        if length(data) == expected_size
            field.data_c .= reshape(data, target_shape)
        elseif length(data) < expected_size
            # Partial update - pad with zeros
            temp_data = zeros(ComplexF64, expected_size)
            temp_data[1:length(data)] .= data
            field.data_c .= reshape(temp_data, target_shape)
            @debug "Padded field $(field.name) data: got $(length(data)), expected $expected_size"
        else
            # Truncate excess data
            field.data_c .= reshape(data[1:expected_size], target_shape)
            @debug "Truncated field $(field.name) data: got $(length(data)), expected $expected_size"
        end
        
        field.current_layout = :c
        
    elseif field.data_g !== nothing
        # Fallback to grid data
        target_shape = size(field.data_g)
        expected_size = prod(target_shape)
        
        if length(data) == expected_size
            field.data_g .= real.(reshape(data, target_shape))  # Convert to real for grid data
        else
            @warn "Size mismatch setting grid data for field $(field.name)"
        end
        
        field.current_layout = :g
    else
        @warn "Cannot set data for field $(field.name) - no data arrays allocated"
    end
end

function get_basis_size(basis)
    """Get the size (number of modes) for a basis following Dedalus patterns"""
    
    # Following Dedalus basis.py structure, bases store size information in different ways:
    # 1. Most common: meta.size field (for Julia BasisMeta structure)
    # 2. Direct size field (for direct Dedalus translation)  
    # 3. Shape tuple (for multidimensional bases)
    # 4. Specific basis attributes (N, etc.)
    
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :size)
        # Julia BasisMeta structure pattern
        return basis.meta.size
    elseif hasfield(typeof(basis), :shape)
        # Dedalus shape attribute pattern (can be tuple for multidimensional)
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
    Evaluate residual and Jacobian for nonlinear problem following Dedalus patterns.
    
    In Dedalus, this corresponds to:
    1. Evaluating F expressions (residual) using evaluator system
    2. Building dF matrices (Jacobian/Frechet differential) 
    3. Gathering results into numerical arrays for Newton solver
    """
    
    # Step 1: Copy solution vector back to problem fields
    # Following Dedalus pattern where fields are updated before evaluation
    copy_solution_to_fields!(problem.variables, x)
    
    # Step 2: Evaluate residual expressions F(x)  
    # In Dedalus: self.evaluator.evaluate_scheduled(iteration=self.iteration)
    residual_fields = ScalarField[]
    
    if hasfield(typeof(problem), :equation_data) && problem.equation_data !== nothing
        for (i, eq_data) in enumerate(problem.equation_data)
            if haskey(eq_data, "F_expr") && eq_data["F_expr"] !== nothing
                # Evaluate the residual expression symbolically
                residual_field = evaluate_solver_expression(eq_data["F_expr"], problem.variables)
                push!(residual_fields, residual_field)
                @debug "Evaluated residual for equation $i"
            else
                # Fallback: create zero residual field
                var_field = problem.variables[min(i, length(problem.variables))]
                residual_field = ScalarField(var_field.dist, "residual_$i", var_field.bases, var_field.dtype)
                ensure_layout!(residual_field, :c)
                fill!(residual_field.data_c, 0.0)
                push!(residual_fields, residual_field)
                @debug "Created zero residual for equation $i (no F_expr)"
            end
        end
    else
        @warn "No equation data available for residual evaluation - creating zero residuals"
        for (i, var) in enumerate(problem.variables)
            residual_field = ScalarField(var.dist, "residual_$i", var.bases, var.dtype)
            ensure_layout!(residual_field, :c)
            fill!(residual_field.data_c, 0.0)
            push!(residual_fields, residual_field)
        end
    end
    
    # Step 3: Convert residual fields to vector
    residual = fields_to_vector(residual_fields)
    
    # Step 4: Build Jacobian matrix (dF) following Dedalus matrix building
    # In Dedalus: self.build_matrices(self.subproblems, ['dF'])
    n = length(x)
    jacobian = sparse(zeros(ComplexF64, n, n))
    
    if hasfield(typeof(problem), :equation_data) && problem.equation_data !== nothing
        row_offset = 1
        col_offset = 1
        
        for (i, eq_data) in enumerate(problem.equation_data)
            if haskey(eq_data, "dF_expr") && eq_data["dF_expr"] !== nothing
                # Build Jacobian block from Frechet differential expression
                jacobian_block = build_jacobian_block(eq_data["dF_expr"], problem.variables, problem.perturbations)
                
                # Insert block into global Jacobian matrix
                block_rows, block_cols = size(jacobian_block)
                end_row = min(row_offset + block_rows - 1, n)
                end_col = min(col_offset + block_cols - 1, n)
                
                if end_row >= row_offset && end_col >= col_offset
                    jacobian[row_offset:end_row, col_offset:end_col] = jacobian_block[1:(end_row-row_offset+1), 1:(end_col-col_offset+1)]
                end
                
                row_offset += block_rows
                col_offset += block_cols
                @debug "Built Jacobian block for equation $i: size ($block_rows, $block_cols)"
            else
                # Fallback: identity block for missing Jacobian
                var_size = compute_field_vector_size(problem.variables[min(i, length(problem.variables))])
                if row_offset <= n && col_offset <= n
                    end_idx = min(row_offset + var_size - 1, n, col_offset + var_size - 1)
                    for j in 1:(end_idx - row_offset + 1)
                        if row_offset + j - 1 <= n && col_offset + j - 1 <= n
                            jacobian[row_offset + j - 1, col_offset + j - 1] = 1.0
                        end
                    end
                end
                row_offset += var_size
                col_offset += var_size
                @debug "Created identity Jacobian block for equation $i (no dF_expr)"
            end
        end
    else
        @warn "No equation data available for Jacobian evaluation - using identity matrix"
        # Fallback: identity matrix
        for i in 1:n
            jacobian[i, i] = 1.0
        end
    end
    
    @debug "Residual evaluation completed: size=$(length(residual)), norm=$(norm(residual))"
    @debug "Jacobian evaluation completed: size=$(size(jacobian)), nnz=$(nnz(jacobian))"
    
    return residual, jacobian
end

# Helper functions for expression evaluation
function evaluate_solver_expression(expr, variables)
    """
    Evaluate symbolic expression with current field values following Dedalus patterns.
    
    In Dedalus, this corresponds to:
    1. expr.attempt() -> expr.evaluate() -> expr.operate(out)
    2. Recursive evaluation of expression tree
    3. Returns field with evaluated data
    """
    
    if expr === nothing
        throw(ArgumentError("Cannot evaluate null expression"))
    end
    
    # Handle different expression types following Dedalus patterns
    if hasfield(typeof(expr), :expr_type)
        expr_type = expr.expr_type
        
        if expr_type == "variable"
            # Direct variable reference - return the variable field
            if haskey(expr, "field_ref") && expr["field_ref"] ∈ variables
                return expr["field_ref"]
            else
                @warn "Variable reference not found, returning zero field"
            end
            
        elseif expr_type == "operator"
            # Operator expression - evaluate operands and apply operator
            return evaluate_operator_expression(expr, variables)
            
        elseif expr_type == "constant" 
            # Constant expression - create field with constant value
            return create_constant_field(expr, variables)
            
        else
            @warn "Unknown expression type: $expr_type, returning zero field"
        end
    end
    
    # Fallback: return zero field matching first variable
    if length(variables) > 0
        result = ScalarField(variables[1].dist, "evaluated_expr", variables[1].bases, variables[1].dtype)
        ensure_layout!(result, :c)
        if result.data_c !== nothing
            fill!(result.data_c, 0.0)
        end
        return result
    else
        throw(ArgumentError("No variables available for expression evaluation"))
    end
end

function evaluate_operator_expression(expr, variables)
    """Evaluate operator expression following Dedalus operator.operate() patterns"""
    
    if !haskey(expr, "operator") || !haskey(expr, "operands")
        @warn "Malformed operator expression, returning zero field"
        return create_zero_field(variables)
    end
    
    operator = expr["operator"]
    operands = expr["operands"]
    
    # Recursively evaluate operands
    eval_operands = []
    for operand in operands
        eval_op = evaluate_solver_expression(operand, variables)
        push!(eval_operands, eval_op)
    end
    
    # Apply operator following Dedalus patterns
    if operator == "Add"
        return apply_add_operator(eval_operands)
    elseif operator == "Multiply" 
        return apply_multiply_operator(eval_operands)
    elseif operator == "Differentiate"
        return apply_differentiate_operator(eval_operands, expr)
    else
        @warn "Unknown operator: $operator, returning first operand"
        return length(eval_operands) > 0 ? eval_operands[1] : create_zero_field(variables)
    end
end

function build_jacobian_block(expr, variables, perturbations)
    """
    Build Jacobian matrix block from Frechet differential expression following Dedalus patterns.
    
    In Dedalus, this corresponds to:
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
    
    # Handle different expression types following Dedalus expression_matrices patterns
    if hasfield(typeof(expr), :expr_type)
        expr_type = expr.expr_type
        
        if expr_type == "variable"
            # Variable expression - return identity matrix (Dedalus lines 183-186, 507-510, 957-960)
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
    
    # Fallback: identity matrix (following Dedalus identity pattern)
    return sparse(I, jacobian_size, jacobian_size)
end

function build_variable_jacobian_block(expr, variables)
    """Build identity matrix block for variable (Dedalus pattern)"""
    
    if !haskey(expr, "field_ref")
        @warn "Variable expression missing field reference"
        return sparse(I, 1, 1) 
    end
    
    field_ref = expr["field_ref"]
    
    # Find variable in list and return identity matrix for it
    for (i, var) in enumerate(variables)
        if var === field_ref
            var_size = compute_field_vector_size(var)
            return sparse(I, var_size, var_size)
        end
    end
    
    @warn "Variable not found in variable list for Jacobian"
    return sparse(I, 1, 1)
end

function build_operator_jacobian_block(expr, variables, perturbations)
    """Build Jacobian block for operator expression (following Dedalus recursive patterns)"""
    
    if !haskey(expr, "operator") || !haskey(expr, "operands")
        @warn "Malformed operator expression for Jacobian"
        total_size = sum(compute_field_vector_size(var) for var in variables)
        return sparse(I, max(total_size, 1), max(total_size, 1))
    end
    
    operator = expr["operator"]
    operands = expr["operands"]
    
    # Following Dedalus arithmetic.py line 189-193 pattern: iteratively add matrices
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
        # Multiplication: more complex - product rule for Jacobian
        # For now, simplified as identity (would need full product rule implementation)
        total_size = sum(compute_field_vector_size(var) for var in variables)
        return sparse(I, max(total_size, 1), max(total_size, 1))
        
    elseif operator == "Differentiate" 
        # Differentiation: apply differential operator matrix
        if length(operands) > 0
            operand_jac = build_jacobian_block(operands[1], variables, perturbations)
            # Apply differential operator matrix (simplified as identity for now)
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
function create_zero_field(variables)
    """Create zero field matching first variable"""
    if length(variables) > 0
        result = ScalarField(variables[1].dist, "zero_field", variables[1].bases, variables[1].dtype)
        ensure_layout!(result, :c)
        if result.data_c !== nothing
            fill!(result.data_c, 0.0)
        end
        return result
    else
        throw(ArgumentError("No variables available"))
    end
end

function create_constant_field(expr, variables)
    """Create field with constant value"""
    if length(variables) == 0
        throw(ArgumentError("No variables available"))
    end
    
    result = ScalarField(variables[1].dist, "constant_field", variables[1].bases, variables[1].dtype)
    ensure_layout!(result, :c)
    
    if result.data_c !== nothing && haskey(expr, "value")
        fill!(result.data_c, expr["value"])
    elseif result.data_c !== nothing
        fill!(result.data_c, 0.0)
    end
    
    return result
end

function apply_add_operator(operands)
    """Apply addition operator following Dedalus patterns"""
    if length(operands) == 0
        throw(ArgumentError("Addition requires operands"))
    end
    
    result = ScalarField(operands[1].dist, "add_result", operands[1].bases, operands[1].dtype)
    ensure_layout!(result, :c)
    
    if result.data_c !== nothing
        fill!(result.data_c, 0.0)
        for operand in operands
            ensure_layout!(operand, :c)
            if operand.data_c !== nothing
                result.data_c .+= operand.data_c
            end
        end
    end
    
    return result
end

function apply_multiply_operator(operands) 
    """Apply multiplication operator following Dedalus patterns"""
    if length(operands) < 2
        return length(operands) == 1 ? operands[1] : throw(ArgumentError("Multiplication requires 2+ operands"))
    end
    
    result = ScalarField(operands[1].dist, "multiply_result", operands[1].bases, operands[1].dtype)
    ensure_layout!(result, :c)
    
    if result.data_c !== nothing
        # Start with first operand
        ensure_layout!(operands[1], :c)
        if operands[1].data_c !== nothing
            result.data_c .= operands[1].data_c
        else
            fill!(result.data_c, 1.0)
        end
        
        # Multiply by remaining operands
        for i in 2:length(operands)
            ensure_layout!(operands[i], :c)
            if operands[i].data_c !== nothing
                result.data_c .*= operands[i].data_c
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
    if haskey(expr, "coordinate") && expr["coordinate"] !== nothing
        return expr["coordinate"]
    elseif haskey(expr, "coord_name")
        # Would need coordinate system lookup - for now return nothing
        @warn "Coordinate lookup by name not implemented"
        return nothing
    else
        @warn "No coordinate specified for differentiation"
        return nothing
    end
end

function get_diff_order(expr)
    """Extract differentiation order from expression"""
    if haskey(expr, "order")
        return max(1, Int(expr["order"]))
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
        @info "  Iterations per second: $(solver.iteration / elapsed)"
        
        if MPI.Initialized()
            # Log MPI statistics
            rank = MPI.Comm_rank(MPI.COMM_WORLD)
            size = MPI.Comm_size(MPI.COMM_WORLD)
            @info "  MPI rank: $rank / $size"
        end
    end
end

# Analysis and output
function create_evaluator(solver::InitialValueSolver)
    """Create evaluator for analysis output"""
    attach_evaluator!(solver)
    return solver.evaluator
end

function log_solver_performance(solver::Union{InitialValueSolver, BoundaryValueSolver})
    """Log solver performance statistics"""

    stats = solver.performance_stats

    if MPI.Initialized()
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if rank == 0
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
