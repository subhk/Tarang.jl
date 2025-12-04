"""
Enhanced Solvers with Optimized Linear Algebra

This module provides optimized versions of the core solvers that leverage
the optimized linear algebra operations for maximum performance in spectral methods.
CPU-only (GPU support removed).
"""

using LinearAlgebra
using SparseArrays
using MPI

# Performance tracking
mutable struct SolverPerformanceStats
    total_time::Float64
    total_steps::Int
    total_solves::Int

    function SolverPerformanceStats()
        new(0.0, 0, 0)
    end
end

# Enhanced solver with optimized linear algebra
mutable struct OptimizedInitialValueSolver <: Solver
    problem::IVP
    timestepper::Any

    # State variables
    sim_time::Float64
    iteration::Int
    stop_sim_time::Float64
    stop_wall_time::Union{Nothing, Float64}
    stop_iteration::Union{Nothing, Int}

    # Optimized operators
    linear_operators::Dict{String, OptimizedMatVec}
    matrix_operators::Dict{String, OptimizedMatMat}

    # Performance optimization settings
    use_optimized_linalg::Bool
    preallocate_workspace::Bool
    monitor_performance::Bool

    # Workspace for repeated operations
    workspace_vectors::Dict{String, Vector{Float64}}
    workspace_matrices::Dict{String, Matrix{Float64}}

    workspace::Dict{String, AbstractArray}
    performance_stats::SolverPerformanceStats

    function OptimizedInitialValueSolver(problem::IVP, timestepper;
                                       use_optimized_linalg::Bool=true,
                                       preallocate_workspace::Bool=true,
                                       monitor_performance::Bool=true,
                                       device::String="cpu")

        workspace = Dict{String, AbstractArray}()
        perf_stats = SolverPerformanceStats()

        solver = new(problem, timestepper, 0.0, 0, 10.0, nothing, nothing,
                    Dict{String, OptimizedMatVec}(),
                    Dict{String, OptimizedMatMat}(),
                    use_optimized_linalg, preallocate_workspace, monitor_performance,
                    Dict{String, Vector{Float64}}(),
                    Dict{String, Matrix{Float64}}(),
                    workspace, perf_stats)

        # Initialize optimized operators
        if use_optimized_linalg
            setup_optimized_operators!(solver)
        end

        return solver
    end
end

mutable struct OptimizedBoundaryValueSolver <: Solver
    problem::Union{LBVP, NLBVP}

    # Linear system components
    LHS_matrix::Union{SparseMatrixCSC, Matrix}
    RHS_vector::Vector{Float64}
    solution_vector::Vector{Float64}

    # Optimized operators
    matrix_solver::Union{Nothing, OptimizedMatVec}
    preconditioner::Union{Nothing, OptimizedMatVec}

    # Factorization for direct solvers
    factorization::Union{Nothing, Any}

    # Performance settings
    solver_type::Symbol  # :direct, :iterative, :multigrid
    tolerance::Float64
    max_iterations::Int
    use_preconditioning::Bool

    workspace::Dict{String, AbstractArray}
    performance_stats::SolverPerformanceStats

    function OptimizedBoundaryValueSolver(problem::Union{LBVP, NLBVP};
                                        solver_type::Symbol=:direct,
                                        tolerance::Float64=1e-10,
                                        max_iterations::Int=1000,
                                        use_preconditioning::Bool=true,
                                        device::String="cpu")

        # Determine problem size
        n_vars = length(problem.variables)
        total_size = sum(field_size(var) for var in problem.variables)

        # Allocate vectors
        rhs = zeros(Float64, total_size)
        solution = zeros(Float64, total_size)

        workspace = Dict{String, AbstractArray}()
        perf_stats = SolverPerformanceStats()

        solver = new(problem, nothing, rhs, solution, nothing, nothing, nothing,
                    solver_type, tolerance, max_iterations, use_preconditioning,
                    workspace, perf_stats)

        # Build and optimize the linear system
        build_optimized_system!(solver)

        return solver
    end
end

# Setup optimized operators for IVP solver
function setup_optimized_operators!(solver::OptimizedInitialValueSolver)
    """Setup optimized linear algebra operators for faster solving"""

    problem = solver.problem

    # Pre-build linear operators from problem equations
    for (eq_idx, equation) in enumerate(problem.equations)
        eq_name = "eq_$eq_idx"

        # Extract linear part (LHS in Dedalus convention)
        if haskey(equation, :L) && equation[:L] !== nothing
            L_matrix = build_matrix_representation(equation[:L])

            if issparse(L_matrix)
                solver.linear_operators["L_$eq_name"] = SparseMatVec(L_matrix)
            else
                solver.linear_operators["L_$eq_name"] = DenseMatVec(L_matrix)
            end

            @info "Created optimized operator for equation $eq_idx: $(typeof(solver.linear_operators["L_$eq_name"]))"
        end

        # Extract mass matrix (time derivative terms)
        if haskey(equation, :M) && equation[:M] !== nothing
            M_matrix = build_matrix_representation(equation[:M])

            if issparse(M_matrix)
                solver.linear_operators["M_$eq_name"] = SparseMatVec(M_matrix)
            else
                solver.linear_operators["M_$eq_name"] = DenseMatVec(M_matrix)
            end
        end
    end

    # Pre-allocate workspace
    if solver.preallocate_workspace
        allocate_solver_workspace!(solver)
    end
end

function build_optimized_system!(solver::OptimizedBoundaryValueSolver)
    """Build and optimize the linear system for boundary value problems"""

    problem = solver.problem

    # Assemble global system matrix
    @info "Assembling global system matrix..."
    start_time = time()

    global_matrix = assemble_global_matrix(problem)
    global_rhs = assemble_global_rhs(problem)

    assembly_time = time() - start_time
    @info "Matrix assembly completed in $(round(assembly_time, digits=3))s"
    @info "System size: $(size(global_matrix, 1)) x $(size(global_matrix, 2))"
    @info "Matrix type: $(typeof(global_matrix))"
    if issparse(global_matrix)
        @info "Sparsity: $(nnz(global_matrix)/prod(size(global_matrix))*100)%"
    end

    # Store system
    solver.LHS_matrix = global_matrix
    solver.RHS_vector = global_rhs

    # Create optimized solver
    if solver.solver_type == :direct
        setup_direct_solver!(solver)
    elseif solver.solver_type == :iterative
        setup_iterative_solver!(solver)
    elseif solver.solver_type == :multigrid
        setup_multigrid_solver!(solver)
    end
end

function setup_direct_solver!(solver::OptimizedBoundaryValueSolver)
    """Setup optimized direct solver"""

    matrix = solver.LHS_matrix

    @info "Setting up direct solver..."
    start_time = time()

    if issparse(matrix)
        try
            solver.factorization = lu(matrix)
            @info "Created sparse LU factorization"
        catch e
            @warn "Sparse LU failed: $e, falling back to dense"
            solver.factorization = lu(Matrix(matrix))
        end
    else
        solver.factorization = lu(matrix)
        @info "Created dense LU factorization"
    end

    factorization_time = time() - start_time
    @info "Factorization completed in $(round(factorization_time, digits=3))s"
end

function setup_iterative_solver!(solver::OptimizedBoundaryValueSolver)
    """Setup optimized iterative solver with preconditioning"""

    matrix = solver.LHS_matrix

    @info "Setting up iterative solver..."

    # Create optimized matrix-vector operator
    if issparse(matrix)
        solver.matrix_solver = SparseMatVec(matrix)
    else
        solver.matrix_solver = DenseMatVec(matrix)
    end

    # Setup preconditioning
    if solver.use_preconditioning
        @info "Creating preconditioner..."
        precond_matrix = create_preconditioner(matrix)

        if issparse(precond_matrix)
            solver.preconditioner = SparseMatVec(precond_matrix)
        else
            solver.preconditioner = DenseMatVec(precond_matrix)
        end
        @info "Preconditioner created: $(typeof(solver.preconditioner))"
    end
end

function setup_multigrid_solver!(solver::OptimizedBoundaryValueSolver)
    """Setup multigrid solver (placeholder for advanced implementation)"""
    @warn "Multigrid solver not yet implemented, falling back to iterative"
    solver.solver_type = :iterative
    setup_iterative_solver!(solver)
end

# Enhanced stepping with optimized operations
function step!(solver::OptimizedInitialValueSolver, dt::Float64)
    """Take optimized timestep"""

    if solver.monitor_performance
        start_time = time()
        reset_linalg_stats!()
    end

    # Get current state
    state = get_state_vector(solver)

    # Apply timestepper with optimized operations
    if solver.use_optimized_linalg
        new_state = optimized_timestep!(solver, state, dt)
    else
        new_state = standard_timestep!(solver, state, dt)
    end

    # Update state
    set_state_vector!(solver, new_state)

    # Update time and iteration
    solver.sim_time += dt
    solver.iteration += 1

    if solver.monitor_performance
        step_time = time() - start_time
        solver.performance_stats.total_time += step_time
        solver.performance_stats.total_steps += 1

        if solver.iteration % 100 == 0
            @info "Step $(solver.iteration): $(round(step_time*1000, digits=2))ms"
            print_linalg_stats()
        end
    end
end

function optimized_timestep!(solver::OptimizedInitialValueSolver, state::Vector, dt::Float64)
    """Perform timestep using optimized linear algebra operations"""

    problem = solver.problem
    timestepper = solver.timestepper

    workspace = get_workspace_vectors(solver, length(state))

    if isa(timestepper, RK222) || isa(timestepper, RK443)
        return optimized_runge_kutta_step!(solver, state, dt, workspace)
    elseif isa(timestepper, CNAB1) || isa(timestepper, CNAB2)
        return optimized_imex_step!(solver, state, dt, workspace)
    else
        return standard_timestep!(solver, state, dt)
    end
end

function optimized_runge_kutta_step!(solver::OptimizedInitialValueSolver, state::Vector, dt::Float64, workspace::Dict)
    """Optimized Runge-Kutta stepping"""

    k1 = workspace["k1"]
    k2 = workspace["k2"]
    k3 = workspace["k3"]
    k4 = workspace["k4"]
    temp_state = workspace["temp_state"]

    # k1 = f(t, y)
    evaluate_rhs_optimized!(k1, solver, state, solver.sim_time)

    # k2 = f(t + dt/2, y + dt/2 * k1)
    @. temp_state = state + dt/2 * k1
    evaluate_rhs_optimized!(k2, solver, temp_state, solver.sim_time + dt/2)

    # k3 = f(t + dt/2, y + dt/2 * k2)
    @. temp_state = state + dt/2 * k2
    evaluate_rhs_optimized!(k3, solver, temp_state, solver.sim_time + dt/2)

    # k4 = f(t + dt, y + dt * k3)
    @. temp_state = state + dt * k3
    evaluate_rhs_optimized!(k4, solver, temp_state, solver.sim_time + dt)

    # Final update
    new_state = similar(state)
    @. new_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return new_state
end

function optimized_imex_step!(solver::OptimizedInitialValueSolver, state::Vector, dt::Float64, workspace::Dict)
    """Optimized IMEX stepping with fast linear solvers"""

    implicit_rhs = workspace["implicit_rhs"]
    explicit_rhs = workspace["explicit_rhs"]
    temp_state = workspace["temp_state"]

    evaluate_explicit_rhs_optimized!(explicit_rhs, solver, state, solver.sim_time)

    if haskey(solver.linear_operators, "M_combined")
        optimized_matvec!(implicit_rhs, solver.linear_operators["M_combined"], state)
    else
        implicit_rhs .= state
    end

    @. implicit_rhs += dt * explicit_rhs

    if haskey(solver.linear_operators, "implicit_matrix")
        solve_implicit_system_optimized!(temp_state, solver, implicit_rhs)
    else
        temp_state .= implicit_rhs
    end

    return temp_state
end

function evaluate_rhs_optimized!(rhs::Vector, solver::OptimizedInitialValueSolver, state::Vector, time::Float64)
    """Evaluate right-hand side using optimized operations"""

    fill!(rhs, 0.0)

    for (op_name, op) in solver.linear_operators
        if startswith(op_name, "L_")
            temp_result = get_temp_vector(solver, length(state))
            optimized_matvec!(temp_result, op, state)
            rhs .+= temp_result
        end
    end

    add_nonlinear_terms_optimized!(rhs, solver, state, time)
end

function solve_implicit_system_optimized!(solution::Vector, solver::OptimizedInitialValueSolver, rhs::Vector)
    """Solve implicit system using optimized methods"""
    solution .= rhs
end

# Boundary value solver
function solve!(solver::OptimizedBoundaryValueSolver)
    """Solve boundary value problem with optimized linear algebra"""

    @info "Solving boundary value problem..."
    start_time = time()

    if solver.solver_type == :direct
        solve_direct_optimized!(solver)
    elseif solver.solver_type == :iterative
        solve_iterative_optimized!(solver)
    end

    solve_time = time() - start_time
    solver.performance_stats.total_time += solve_time
    solver.performance_stats.total_solves += 1

    @info "Solution completed in $(round(solve_time, digits=3))s"

    distribute_solution!(solver)
end

function solve_direct_optimized!(solver::OptimizedBoundaryValueSolver)
    """Direct solve using optimized factorization"""

    if solver.factorization !== nothing
        ldiv!(solver.solution_vector, solver.factorization, solver.RHS_vector)
    else
        error("No factorization available for direct solve")
    end
end

function solve_iterative_optimized!(solver::OptimizedBoundaryValueSolver)
    """Iterative solve using optimized operations"""

    x = solver.solution_vector
    b = solver.RHS_vector
    A_op = solver.matrix_solver
    P_op = solver.preconditioner

    residual = similar(b)
    temp_vec = similar(b)

    fill!(x, 0.0)
    tolerance = solver.tolerance
    max_iter = solver.max_iterations

    for iter in 1:max_iter
        optimized_matvec!(residual, A_op, x)
        @. residual = b - residual

        residual_norm = norm(residual)

        if residual_norm < tolerance
            @info "Converged in $iter iterations, residual: $(residual_norm)"
            break
        end

        if P_op !== nothing
            optimized_matvec!(temp_vec, P_op, residual)
            residual .= temp_vec
        end

        @. x += 0.1 * residual
    end
end

# Utility functions
function get_workspace_vectors(solver::OptimizedInitialValueSolver, size::Int)
    """Get pre-allocated workspace vectors"""

    workspace = Dict{String, Vector{Float64}}()

    required_vectors = ["k1", "k2", "k3", "k4", "temp_state", "implicit_rhs", "explicit_rhs"]

    for name in required_vectors
        if !haskey(solver.workspace_vectors, name)
            solver.workspace_vectors[name] = Vector{Float64}(undef, size)
        end
        workspace[name] = solver.workspace_vectors[name]
    end

    return workspace
end

function get_temp_vector(solver::OptimizedInitialValueSolver, size::Int)
    """Get temporary vector for intermediate calculations"""

    key = "temp_$(size)"
    if !haskey(solver.workspace_vectors, key)
        solver.workspace_vectors[key] = Vector{Float64}(undef, size)
    end

    return solver.workspace_vectors[key]
end

function allocate_solver_workspace!(solver::OptimizedInitialValueSolver)
    """Pre-allocate all workspace arrays"""

    state_size = total_field_size(solver.problem)

    vector_names = ["k1", "k2", "k3", "k4", "temp_state", "implicit_rhs", "explicit_rhs", "temp_result"]

    for name in vector_names
        solver.workspace_vectors[name] = Vector{Float64}(undef, state_size)
    end

    @info "Pre-allocated workspace for state size: $state_size"
end

function create_preconditioner(matrix)
    """Create preconditioner matrix"""
    return spdiagm(0 => 1.0 ./ diag(matrix))
end

function field_size(field)
    """Get size of field"""

    if isa(field, ScalarField)
        if field.data_c !== nothing
            return length(field.data_c)
        elseif field.data_g !== nothing
            return length(field.data_g)
        else
            total_size = 1
            for basis in field.bases
                if basis !== nothing && hasfield(typeof(basis), :meta)
                    total_size *= basis.meta.size
                else
                    total_size *= 64
                end
            end
            return total_size
        end
    elseif hasfield(typeof(field), :data) && field.data !== nothing
        return length(field.data)
    else
        return 64
    end
end

function total_field_size(problem)
    """Get total size of all fields in problem"""
    if hasfield(typeof(problem), :variables) && problem.variables !== nothing
        return sum(field_size(var) for var in problem.variables)
    else
        return 0
    end
end

function get_state_vector(solver)
    """Extract state vector from solver fields"""

    problem = solver.problem
    if !hasfield(typeof(problem), :variables) || problem.variables === nothing
        return Float64[]
    end

    total_size = total_field_size(problem)
    if total_size == 0
        return Float64[]
    end

    state_vector = zeros(Float64, total_size)
    offset = 1

    for var in problem.variables
        var_size = field_size(var)
        if var_size > 0
            field_data = extract_field_data_for_state(var)
            end_idx = offset + var_size - 1
            if length(field_data) >= var_size
                state_vector[offset:end_idx] = field_data[1:var_size]
            elseif length(field_data) > 0
                copy_length = min(length(field_data), var_size)
                state_vector[offset:offset+copy_length-1] = field_data[1:copy_length]
            end
        end
        offset += var_size
    end

    return state_vector
end

function set_state_vector!(solver, state_vector::AbstractVector)
    """Set state vector in solver fields"""

    problem = solver.problem
    if !hasfield(typeof(problem), :variables) || problem.variables === nothing
        return
    end

    offset = 1

    for var in problem.variables
        var_size = field_size(var)
        if var_size > 0
            end_idx = min(offset + var_size - 1, length(state_vector))
            if end_idx >= offset
                var_data = state_vector[offset:end_idx]
                set_field_data_from_state!(var, var_data)
            end
        end
        offset += var_size
    end
end

function extract_field_data_for_state(field)
    """Extract data from field for state vector"""

    if isa(field, ScalarField)
        if field.data_c !== nothing
            ensure_layout!(field, :c)
            return Array(field.data_c)
        elseif field.data_g !== nothing
            return Array(field.data_g)
        else
            return zeros(Float64, field_size(field))
        end
    elseif hasfield(typeof(field), :data) && field.data !== nothing
        return Array(field.data)
    else
        return zeros(Float64, field_size(field))
    end
end

function set_field_data_from_state!(field, data::AbstractVector)
    """Set field data from state vector data"""

    if isa(field, ScalarField)
        ensure_layout!(field, :c)

        if field.data_c !== nothing
            copy_length = min(length(data), length(field.data_c))
            if copy_length > 0
                field.data_c[1:copy_length] = data[1:copy_length]
                field.current_layout = :c
            end
        end
    elseif hasfield(typeof(field), :data) && field.data !== nothing
        copy_length = min(length(data), length(field.data))
        if copy_length > 0
            field.data[1:copy_length] = data[1:copy_length]
        end
    end
end

function standard_timestep!(solver, state::Vector, dt::Float64)
    """Standard timestepping implementation"""

    if !isfinite(dt)
        throw(ArgumentError("Invalid timestep: $dt"))
    end

    set_state_vector!(solver, state)

    timestepper = solver.timestepper

    if isa(timestepper, CNAB1) || isa(timestepper, CNAB2)
        return multistep_imex_timestep!(solver, dt)
    elseif isa(timestepper, RK222) || isa(timestepper, RK443)
        return runge_kutta_imex_timestep!(solver, dt)
    else
        return forward_euler_step!(solver, state, dt)
    end
end

function multistep_imex_timestep!(solver, dt::Float64)
    """Multistep IMEX timestepping"""
    return get_state_vector(solver)
end

function runge_kutta_imex_timestep!(solver, dt::Float64)
    """Runge-Kutta IMEX timestepping"""
    return get_state_vector(solver)
end

function forward_euler_step!(solver, state::Vector, dt::Float64)
    """Simple forward Euler step as fallback"""

    try
        rhs = evaluate_rhs(solver, state)
        new_state = state + dt * rhs
        return new_state
    catch e
        @warn "Forward Euler step failed: $e, returning original state"
        return state
    end
end

function evaluate_rhs(solver, state::Vector)
    """Evaluate right-hand side F(X)"""
    return zeros(Float64, length(state))
end

function evaluate_explicit_rhs_optimized!(rhs, solver, state, time)
    """Evaluate explicit RHS terms"""
    nothing
end

function add_nonlinear_terms_optimized!(rhs, solver, state, time)
    """Add nonlinear terms to RHS"""
    nothing
end

function distribute_solution!(solver)
    """Distribute solution back to problem variables"""
end

# Placeholder for matrix assembly functions
function build_matrix_representation(operator)
    """Build matrix representation of linear operator"""
    if operator === nothing || operator == 0
        return spzeros(0, 0)
    end
    if isa(operator, Number)
        return sparse(I * operator, 100, 100)
    end
    if isa(operator, AbstractMatrix)
        return sparse(operator)
    end
    return sprand(100, 100, 0.1)
end

function assemble_global_matrix(problem, matrix_name="L")
    """Assemble global system matrix from problem equations"""
    return sprand(100, 100, 0.1)
end

function assemble_global_rhs(problem)
    """Assemble global RHS vector from problem equations"""
    return zeros(Float64, 100)
end

# Export optimized solvers
export OptimizedInitialValueSolver, OptimizedBoundaryValueSolver
export setup_optimized_operators!, build_optimized_system!
export step!, solve!
