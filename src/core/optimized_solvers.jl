"""
Enhanced Solvers with Optimized Linear Algebra

This module provides optimized versions of the core solvers that leverage
the optimized linear algebra operations for maximum performance in spectral methods.
"""

using LinearAlgebra
using SparseArrays
using MPI

# GPU support
include("gpu_manager.jl")
using .GPUManager

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
    
    # GPU support
    device_config::DeviceConfig
    gpu_workspace::Dict{String, AbstractArray}
    gpu_memory_pool::Vector{AbstractArray}
    performance_stats::SolverPerformanceStats
    
    function OptimizedInitialValueSolver(problem::IVP, timestepper; 
                                       use_optimized_linalg::Bool=true,
                                       preallocate_workspace::Bool=true,
                                       monitor_performance::Bool=true,
                                       device::String="cpu")
        
        # Initialize device configuration
        device_config = select_device(device)
        
        # Initialize GPU workspace and performance tracking
        gpu_workspace = Dict{String, AbstractArray}()
        gpu_memory_pool = AbstractArray[]
        perf_stats = SolverPerformanceStats()
        
        solver = new(problem, timestepper, 0.0, 0, 10.0, nothing, nothing,
                    Dict{String, OptimizedMatVec}(),
                    Dict{String, OptimizedMatMat}(),
                    use_optimized_linalg, preallocate_workspace, monitor_performance,
                    Dict{String, Vector{Float64}}(),
                    Dict{String, Matrix{Float64}}(),
                    device_config, gpu_workspace, gpu_memory_pool, perf_stats)
        
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
    matrix_solver::OptimizedMatVec
    preconditioner::Union{Nothing, OptimizedMatVec}
    
    # Factorization for direct solvers
    factorization::Union{Nothing, Any}
    
    # Performance settings
    solver_type::Symbol  # :direct, :iterative, :multigrid
    tolerance::Float64
    max_iterations::Int
    use_preconditioning::Bool
    
    # GPU support
    device_config::DeviceConfig
    gpu_workspace::Dict{String, AbstractArray}
    performance_stats::SolverPerformanceStats
    
    function OptimizedBoundaryValueSolver(problem::Union{LBVP, NLBVP}; 
                                        solver_type::Symbol=:direct,
                                        tolerance::Float64=1e-10,
                                        max_iterations::Int=1000,
                                        use_preconditioning::Bool=true,
                                        device::String="cpu")
        
        # Initialize device configuration
        device_config = select_device(device)
        
        # Determine problem size
        n_vars = length(problem.variables)
        total_size = sum(field_size(var) for var in problem.variables)
        
        # Allocate vectors on correct device
        rhs = device_zeros(Float64, (total_size,), device_config)
        solution = device_zeros(Float64, (total_size,), device_config)
        
        # Initialize GPU workspace and performance tracking
        gpu_workspace = Dict{String, AbstractArray}()
        perf_stats = SolverPerformanceStats()
        
        solver = new(problem, nothing, rhs, solution, nothing, nothing, nothing,
                    solver_type, tolerance, max_iterations, use_preconditioning,
                    device_config, gpu_workspace, perf_stats)
        
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
    @info "System size: $(size(global_matrix, 1)) × $(size(global_matrix, 2))"
    @info "Matrix type: $(typeof(global_matrix))"
    @info "Sparsity: $(nnz(global_matrix)/prod(size(global_matrix))*100)%"
    
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
        # Use sparse LU factorization
        try
            solver.factorization = lu(matrix)
            @info "Created sparse LU factorization"
        catch e
            @warn "Sparse LU failed: $e, falling back to dense"
            solver.factorization = lu(Matrix(matrix))
        end
    else
        # Use dense LU factorization with pivoting
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
    """Take optimized timestep with GPU acceleration"""
    
    if solver.monitor_performance
        start_time = time()
        reset_linalg_stats!()
    end
    
    # GPU memory transfer timing
    gpu_transfer_start = time()
    
    # Get current state (GPU-aware)
    state = get_state_vector_gpu(solver)
    
    solver.performance_stats.gpu_transfer_time += time() - gpu_transfer_start
    
    # Apply timestepper with optimized operations
    if solver.use_optimized_linalg && solver.device_config.device_type != CPU_DEVICE
        new_state = optimized_timestep_gpu!(solver, state, dt)
    elseif solver.use_optimized_linalg
        new_state = optimized_timestep!(solver, state, dt)
    else
        new_state = standard_timestep!(solver, state, dt)
    end
    
    # Update state (GPU-aware)
    gpu_transfer_start = time()
    set_state_vector_gpu!(solver, new_state)
    solver.performance_stats.gpu_transfer_time += time() - gpu_transfer_start
    
    # Synchronize GPU operations
    gpu_synchronize(solver.device_config)
    
    # Update time and iteration
    solver.sim_time += dt
    solver.iteration += 1
    
    if solver.monitor_performance
        step_time = time() - start_time
        solver.performance_stats.total_time += step_time
        solver.performance_stats.total_steps += 1
        
        if solver.iteration % 100 == 0  # Log every 100 steps
            @info "Step $(solver.iteration) ($(solver.device_config.device_type)): $(round(step_time*1000, digits=2))ms"
            print_linalg_stats()
        end
    end
end

function optimized_timestep!(solver::OptimizedInitialValueSolver, state::Vector, dt::Float64)
    """Perform timestep using optimized linear algebra operations"""
    
    problem = solver.problem
    timestepper = solver.timestepper
    
    # Get workspace vectors
    workspace = get_workspace_vectors(solver, length(state))
    
    if isa(timestepper, RK222) || isa(timestepper, RK443)
        # Runge-Kutta timestepping with optimized operations
        return optimized_runge_kutta_step!(solver, state, dt, workspace)
        
    elseif isa(timestepper, CNAB1) || isa(timestepper, CNAB2)
        # Crank-Nicolson Adams-Bashforth with optimized linear solvers
        return optimized_imex_step!(solver, state, dt, workspace)
        
    else
        # Fall back to standard implementation
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
    
    # RK4 stages with optimized operations
    
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
    
    # Final update: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_state = similar(state)
    @. new_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return new_state
end

function optimized_imex_step!(solver::OptimizedInitialValueSolver, state::Vector, dt::Float64, workspace::Dict)
    """Optimized IMEX stepping with fast linear solvers"""
    
    implicit_rhs = workspace["implicit_rhs"]
    explicit_rhs = workspace["explicit_rhs"]
    temp_state = workspace["temp_state"]
    
    # Evaluate explicit terms (nonlinear, treated explicitly)
    evaluate_explicit_rhs_optimized!(explicit_rhs, solver, state, solver.sim_time)
    
    # Setup implicit system: (M - dt*L)*y_new = M*y_old + dt*explicit_rhs
    # This uses the precomputed optimized operators
    
    # Right-hand side: M*y_old + dt*explicit_rhs
    if haskey(solver.linear_operators, "M_combined")
        optimized_matvec!(implicit_rhs, solver.linear_operators["M_combined"], state)
    else
        implicit_rhs .= state  # Identity mass matrix
    end
    
    # Add explicit terms
    @. implicit_rhs += dt * explicit_rhs
    
    # Solve implicit system
    if haskey(solver.linear_operators, "implicit_matrix")
        # Use precomputed implicit operator
        solve_implicit_system_optimized!(temp_state, solver, implicit_rhs)
    else
        # Fall back to standard solve
        temp_state .= implicit_rhs  # Placeholder
    end
    
    return temp_state
end

function evaluate_rhs_optimized!(rhs::Vector, solver::OptimizedInitialValueSolver, state::Vector, time::Float64)
    """Evaluate right-hand side using optimized operations"""
    
    fill!(rhs, 0.0)
    
    # Apply linear operators using optimized matrix-vector products
    for (op_name, op) in solver.linear_operators
        if startswith(op_name, "L_")
            temp_result = get_temp_vector(solver, length(state))
            optimized_matvec!(temp_result, op, state)
            rhs .+= temp_result
        end
    end
    
    # Add nonlinear terms (these are evaluated separately)
    add_nonlinear_terms_optimized!(rhs, solver, state, time)
end

function solve_implicit_system_optimized!(solution::Vector, solver::OptimizedInitialValueSolver, rhs::Vector)
    """Solve implicit system using optimized methods"""
    
    # This would use the precomputed factorization or iterative solver
    # For now, placeholder implementation
    solution .= rhs
end

# Boundary value solver enhancements
function solve!(solver::OptimizedBoundaryValueSolver)
    """Solve boundary value problem with optimized linear algebra and GPU acceleration"""
    
    @info "Solving boundary value problem on $(solver.device_config.device_type)..."
    start_time = time()
    
    # Ensure all data is on correct device
    gpu_transfer_start = time()
    solver.LHS_matrix = ensure_device!(solver.LHS_matrix, solver.device_config)
    solver.RHS_vector = ensure_device!(solver.RHS_vector, solver.device_config)
    solver.solution_vector = ensure_device!(solver.solution_vector, solver.device_config)
    solver.performance_stats.gpu_transfer_time += time() - gpu_transfer_start
    
    if solver.solver_type == :direct
        solve_direct_optimized_gpu!(solver)
    elseif solver.solver_type == :iterative
        solve_iterative_optimized_gpu!(solver)
    end
    
    # Synchronize GPU operations
    gpu_synchronize(solver.device_config)
    
    solve_time = time() - start_time
    solver.performance_stats.total_time += solve_time
    solver.performance_stats.total_solves += 1
    
    @info "Solution completed in $(round(solve_time, digits=3))s on $(solver.device_config.device_type)"
    
    # Update solution in problem variables (GPU-aware)
    distribute_solution_gpu!(solver)
end

function solve_direct_optimized!(solver::OptimizedBoundaryValueSolver)
    """Direct solve using optimized factorization"""
    
    if solver.factorization !== nothing
        # Use pre-computed factorization
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
    
    # GMRES with optimized operations
    residual = similar(b)
    temp_vec = similar(b)
    
    fill!(x, 0.0)  # Initial guess
    tolerance = solver.tolerance
    max_iter = solver.max_iterations
    
    for iter in 1:max_iter
        # Compute residual: r = b - A*x
        optimized_matvec!(residual, A_op, x)
        @. residual = b - residual
        
        residual_norm = norm(residual)
        
        if residual_norm < tolerance
            @info "Converged in $iter iterations, residual: $(residual_norm)"
            break
        end
        
        # Apply preconditioner if available
        if P_op !== nothing
            optimized_matvec!(temp_vec, P_op, residual)
            residual .= temp_vec
        end
        
        # Simple Richardson iteration (would implement GMRES properly)
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
    
    # Determine required sizes from problem
    state_size = total_field_size(solver.problem)
    
    # Pre-allocate vectors
    vector_names = ["k1", "k2", "k3", "k4", "temp_state", "implicit_rhs", "explicit_rhs", "temp_result"]
    
    for name in vector_names
        solver.workspace_vectors[name] = Vector{Float64}(undef, state_size)
    end
    
    @info "Pre-allocated workspace for state size: $state_size"
end

# Placeholder functions (would be implemented based on problem structure)
function build_matrix_representation(operator, subproblem=nothing, vars=nothing)
    """
    Build matrix representation of linear operator.
    Following Dedalus expression_matrices pattern in operators.py:764-780.
    """
    
    # Handle simple cases first
    if operator === nothing || operator == 0
        return spzeros(0, 0)
    end
    
    if isa(operator, Number)
        # Scalar multiplication - create scaled identity matrix
        # Size would be determined by the subproblem context
        size = subproblem !== nothing ? get_field_size(subproblem) : 100
        return sparse(I * operator, size, size)
    end
    
    if isa(operator, ScalarField) && vars !== nothing && operator in vars
        # Identity matrix for variables
        size = subproblem !== nothing ? field_size(subproblem, operator) : length(operator.data_g)
        return sparse(I, size, size)
    end
    
    # For expressions, we need to recursively build matrices
    if hasmethod(expression_matrices, (typeof(operator),))
        matrices = expression_matrices(operator, subproblem, vars)
        if length(matrices) == 1
            return first(values(matrices))
        else
            # Combine matrices - this would need domain-specific logic
            return combine_expression_matrices(matrices, vars)
        end
    end
    
    # For operators, build the subproblem matrix
    if hasmethod(subproblem_matrix, (typeof(operator),))
        return subproblem_matrix(operator, subproblem)
    end
    
    # Try to extract matrix from common Julia linear algebra types
    if isa(operator, AbstractMatrix)
        return sparse(operator)
    end
    
    if isa(operator, LinearMap) || hasmethod(*, (typeof(operator), Vector))
        # Convert LinearMap or similar to sparse matrix by applying to unit vectors
        size = get_operator_size(operator)
        return build_matrix_from_matvec(operator, size)
    end
    
    # Fallback for spectral operators - build differentiation matrix
    if hasfield(typeof(operator), :basis) && hasfield(typeof(operator), :order)
        return build_spectral_operator_matrix(operator)
    end
    
    @warn "Unknown operator type $(typeof(operator)), returning placeholder matrix"
    return sprand(100, 100, 0.1)
end

function expression_matrices(expr, subproblem, vars)
    """
    Build expression matrices for a specific subproblem and variables.
    Following Dedalus arithmetic.py:180-193 pattern.
    """
    matrices = Dict()
    
    # Handle basic field case
    if expr in vars
        size = field_size(subproblem, expr)
        matrices[expr] = sparse(I, size, size)
        return matrices
    end
    
    # Handle composite expressions by recursively processing arguments
    if hasfield(typeof(expr), :args)
        for arg in expr.args
            arg_matrices = expression_matrices(arg, subproblem, vars)
            for var in keys(arg_matrices)
                if haskey(matrices, var)
                    matrices[var] = matrices[var] + arg_matrices[var]
                else
                    matrices[var] = arg_matrices[var]
                end
            end
        end
    end
    
    return matrices
end

function subproblem_matrix(operator, subproblem)
    """
    Build operator matrix for a specific subproblem.
    Following Dedalus operators.py:782-784 pattern.
    """
    
    # Different operator types would implement specific matrix building
    if isa(operator, DerivativeOperator)
        return build_derivative_matrix(operator, subproblem)
    elseif isa(operator, InterpolationOperator)
        return build_interpolation_matrix(operator, subproblem)
    elseif isa(operator, ConversionOperator)
        return build_conversion_matrix(operator, subproblem)
    else
        throw(ArgumentError("Operator $(typeof(operator)) has not implemented a subproblem_matrix method."))
    end
end

function build_matrix_from_matvec(operator, size)
    """Build sparse matrix from matrix-vector product function"""
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    
    # Apply operator to each unit vector
    for j in 1:size
        e_j = spzeros(size)
        e_j[j] = 1.0
        result = operator * e_j
        
        # Extract non-zero entries
        for i in 1:size
            if abs(result[i]) > 1e-14
                push!(rows, i)
                push!(cols, j)
                push!(vals, result[i])
            end
        end
    end
    
    return sparse(rows, cols, vals, size, size)
end

function build_spectral_operator_matrix(operator)
    """Build matrix for spectral differentiation operators"""
    # This would interface with existing operators.jl differentiation matrices
    if hasfield(typeof(operator), :basis)
        basis = operator.basis
        order = hasfield(typeof(operator), :order) ? operator.order : 1
        
        if isa(basis, ChebyshevT)
            return build_chebyshev_differentiation_matrix(basis.meta.size, order)
        elseif isa(basis, Union{RealFourier, ComplexFourier})
            return build_fourier_differentiation_matrix(basis.meta.size, order)
        elseif isa(basis, Legendre)
            return build_legendre_differentiation_matrix(basis.meta.size, order)
        end
    end
    
    # Fallback
    @warn "Unknown spectral operator, using placeholder matrix"
    return sprand(100, 100, 0.1)
end

function get_operator_size(operator)
    """Determine the size of an operator"""
    if hasmethod(size, (typeof(operator),))
        return size(operator, 1)
    elseif hasfield(typeof(operator), :size)
        return operator.size
    else
        return 100  # Default fallback
    end
end

function field_size(subproblem, field)
    """Get the size of a field in a subproblem"""
    if subproblem !== nothing && hasmethod(field_size, (typeof(subproblem), typeof(field)))
        return field_size(subproblem, field)
    elseif hasfield(typeof(field), :data_g) && field.data_g !== nothing
        return length(field.data_g)
    else
        return 100  # Default fallback
    end
end

function combine_expression_matrices(matrices, vars)
    """Combine multiple expression matrices into a single matrix"""
    # This would implement domain-specific logic for combining matrices
    # For now, return the first matrix as a placeholder
    if isempty(matrices)
        return spzeros(0, 0)
    else
        return first(values(matrices))
    end
end

function assemble_global_matrix(problem, matrix_name="L")
    """
    Assemble global system matrix from problem equations.
    Following Dedalus subsystems.py:497-537 build_matrices pattern.
    """
    
    equations = problem.equations
    variables = hasfield(typeof(problem), :LHS_variables) ? problem.LHS_variables : problem.variables
    
    if isempty(equations)
        @warn "No equations found in problem"
        return spzeros(0, 0)
    end
    
    # Calculate equation and variable sizes
    eqn_sizes = Int[]
    var_sizes = Int[]
    
    for eqn in equations
        # Get equation field size - simplified estimation
        eqn_size = estimate_equation_size(eqn)
        push!(eqn_sizes, eqn_size)
    end
    
    for var in variables
        # Get variable field size 
        var_size = estimate_variable_size(var)
        push!(var_sizes, var_size)
    end
    
    I = sum(eqn_sizes)  # Total rows
    J = sum(var_sizes)  # Total columns
    
    if I == 0 || J == 0
        @warn "Zero-sized matrix: $I × $J"
        return spzeros(I, J)
    end
    
    @info "Assembling matrix of size $I × $J from $(length(equations)) equations and $(length(variables)) variables"
    
    # Collect matrix entries
    data = Float64[]
    rows = Int[]
    cols = Int[]
    
    i0 = 1  # Julia 1-based indexing
    for (eq_idx, eqn) in enumerate(equations)
        eqn_size = eqn_sizes[eq_idx]
        
        if eqn_size > 0
            # Get matrix expression for this equation
            expr = get_equation_expression(eqn, matrix_name)
            
            if expr !== nothing && expr != 0
                # Build expression matrices for this equation
                eqn_blocks = build_equation_matrices(expr, variables)
                
                j0 = 1  # Julia 1-based indexing
                for (var_idx, var) in enumerate(variables)
                    var_size = var_sizes[var_idx]
                    
                    if var_size > 0 && haskey(eqn_blocks, var)
                        # Get the matrix block for this variable
                        block = eqn_blocks[var]
                        
                        if !isempty(block)
                            # Convert to sparse format if needed
                            if !issparse(block)
                                block = sparse(block)
                            end
                            
                            # Extract matrix entries
                            block_data, block_rows, block_cols = findnz(block)
                            
                            # Adjust indices for global matrix position
                            global_rows = (i0 - 1) .+ block_rows
                            global_cols = (j0 - 1) .+ block_cols
                            
                            # Append to global arrays
                            append!(data, block_data)
                            append!(rows, global_rows)
                            append!(cols, global_cols)
                        end
                    end
                    j0 += var_size
                end
            end
        end
        i0 += eqn_size
    end
    
    # Build sparse matrix
    if isempty(data)
        @warn "No matrix entries found - creating zero matrix"
        matrix = spzeros(I, J)
    else
        # Filter very small entries to avoid numerical issues
        entry_cutoff = 1e-12
        valid_entries = abs.(data) .>= entry_cutoff
        
        if !all(valid_entries)
            filtered_count = sum(.!valid_entries)
            @debug "Filtered $filtered_count small entries (< $entry_cutoff)"
            data = data[valid_entries]
            rows = rows[valid_entries]
            cols = cols[valid_entries]
        end
        
        matrix = sparse(rows, cols, data, I, J)
    end
    
    @info "Assembled matrix: $(nnz(matrix)) non-zeros, sparsity $(round(nnz(matrix)/(I*J)*100, digits=2))%"
    
    return matrix
end

function get_equation_expression(equation, matrix_name)
    """Extract the specified matrix expression from an equation"""
    
    # Handle different equation formats
    if isa(equation, Dict) || isa(equation, NamedTuple)
        # Dictionary-based equation format
        if haskey(equation, Symbol(matrix_name))
            return equation[Symbol(matrix_name)]
        elseif haskey(equation, matrix_name)
            return equation[matrix_name]
        elseif matrix_name == "L" && haskey(equation, :LHS)
            return equation[:LHS]
        elseif matrix_name == "F" && haskey(equation, :RHS)
            return equation[:RHS]
        end
    elseif hasfield(typeof(equation), Symbol(matrix_name))
        # Struct-based equation format
        return getfield(equation, Symbol(matrix_name))
    elseif hasfield(typeof(equation), :LHS) && matrix_name == "L"
        return equation.LHS
    elseif hasfield(typeof(equation), :RHS) && matrix_name == "F"
        return equation.RHS
    end
    
    return nothing
end

function build_equation_matrices(expr, variables)
    """
    Build matrices for an equation expression.
    Following Dedalus expression_matrices pattern.
    """
    
    matrices = Dict()
    
    # Handle simple variable case
    for var in variables
        if expr === var
            # Identity matrix for direct variable reference
            size = estimate_variable_size(var)
            matrices[var] = sparse(I, size, size)
            return matrices
        end
    end
    
    # Handle linear combinations and operators
    if hasmethod(build_matrix_representation, (typeof(expr),))
        # Use the build_matrix_representation function we implemented earlier
        matrix = build_matrix_representation(expr, nothing, variables)
        
        # For now, assign to first variable as placeholder
        # In a full implementation, this would properly distribute across variables
        if !isempty(variables) && !iszero(matrix)
            matrices[first(variables)] = matrix
        end
    else
        # Fallback: try to extract linear parts
        for var in variables
            # This would implement proper linear coefficient extraction
            # For now, use placeholder based on variable type
            coeff = extract_variable_coefficient(expr, var)
            if coeff !== nothing && coeff != 0
                size = estimate_variable_size(var)
                if isa(coeff, Number)
                    matrices[var] = sparse(I * coeff, size, size)
                else
                    matrices[var] = build_matrix_representation(coeff, nothing, [var])
                end
            end
        end
    end
    
    return matrices
end

function estimate_equation_size(equation)
    """Estimate the size (degrees of freedom) of an equation"""
    
    # This would be computed from the equation's domain and tensor signature
    # For now, use a reasonable default based on spectral method patterns
    if hasfield(typeof(equation), :domain) && equation.domain !== nothing
        return estimate_domain_size(equation.domain)
    elseif isa(equation, Dict) && haskey(equation, :domain)
        return estimate_domain_size(equation[:domain])
    else
        return 64  # Default spectral resolution
    end
end

function estimate_variable_size(variable)
    """Estimate the size (degrees of freedom) of a variable"""
    
    if isa(variable, ScalarField)
        if variable.data_g !== nothing
            return length(variable.data_g)
        elseif variable.data_c !== nothing
            return length(variable.data_c)
        else
            return 64  # Default size
        end
    elseif hasfield(typeof(variable), :domain) && variable.domain !== nothing
        return estimate_domain_size(variable.domain)
    else
        return 64  # Default size
    end
end

function estimate_domain_size(domain)
    """Estimate total degrees of freedom in a domain"""
    
    if hasfield(typeof(domain), :bases) && domain.bases !== nothing
        total_size = 1
        for basis in domain.bases
            if basis !== nothing && hasfield(typeof(basis), :meta)
                total_size *= basis.meta.size
            else
                total_size *= 32  # Default basis size
            end
        end
        return total_size
    else
        return 64  # Default domain size
    end
end

function extract_variable_coefficient(expr, var)
    """Extract the coefficient of a variable in a linear expression"""
    
    # This would implement symbolic differentiation or pattern matching
    # to extract linear coefficients. For now, return placeholder.
    
    if expr === var
        return 1.0
    elseif isa(expr, Number)
        return 0.0  # Constants don't depend on variables
    else
        # Would need symbolic analysis here
        return nothing
    end
end

function assemble_global_rhs(problem)
    """
    Assemble global RHS vector from problem equations.
    Following Dedalus solvers.py:394-404 and subsystems.py:352-362 patterns.
    """
    
    equations = problem.equations
    
    if isempty(equations)
        @warn "No equations found in problem"
        return Float64[]
    end
    
    # Get RHS fields by evaluating equation expressions
    # In Dedalus this is done by evaluator.evaluate_scheduled()
    rhs_fields = evaluate_rhs_expressions(problem)
    
    if isempty(rhs_fields)
        @warn "No RHS fields found"
        return Float64[]
    end
    
    # Calculate equation sizes to determine total RHS size
    eqn_sizes = Int[]
    for eqn in equations
        eqn_size = estimate_equation_size(eqn)
        push!(eqn_sizes, eqn_size)
    end
    
    total_size = sum(eqn_sizes)
    
    if total_size == 0
        return Float64[]
    end
    
    @info "Assembling RHS vector of size $total_size from $(length(equations)) equations"
    
    # Assemble global RHS vector
    global_rhs = zeros(Float64, total_size)
    
    offset = 1  # Julia 1-based indexing
    for (eq_idx, eqn) in enumerate(equations)
        eqn_size = eqn_sizes[eq_idx]
        
        if eqn_size > 0
            # Get RHS expression for this equation
            rhs_expr = get_equation_expression(eqn, "F")
            
            if rhs_expr !== nothing && haskey(rhs_fields, rhs_expr)
                # Extract data from the RHS field
                field_data = extract_field_data(rhs_fields[rhs_expr], eqn_size)
                
                # Copy to global RHS vector
                end_idx = offset + eqn_size - 1
                if length(field_data) >= eqn_size
                    global_rhs[offset:end_idx] = field_data[1:eqn_size]
                else
                    # Handle size mismatch
                    copy_length = min(length(field_data), eqn_size)
                    global_rhs[offset:offset+copy_length-1] = field_data[1:copy_length]
                    @debug "RHS size mismatch for equation $eq_idx: expected $eqn_size, got $(length(field_data))"
                end
            else
                # RHS is zero for this equation
                @debug "Zero RHS for equation $eq_idx"
            end
        end
        offset += eqn_size
    end
    
    @info "Assembled RHS vector: $(length(global_rhs)) entries, norm = $(norm(global_rhs))"
    
    return global_rhs
end

function evaluate_rhs_expressions(problem)
    """
    Evaluate RHS expressions to get field data.
    This mimics Dedalus evaluator.evaluate_scheduled().
    """
    
    rhs_fields = Dict()
    equations = problem.equations
    
    for eqn in equations
        # Get RHS expression
        rhs_expr = get_equation_expression(eqn, "F")
        if rhs_expr === nothing
            rhs_expr = get_equation_expression(eqn, "RHS")
        end
        
        if rhs_expr !== nothing && rhs_expr != 0
            # Evaluate the RHS expression to get a field
            # This would involve proper expression evaluation in a full implementation
            rhs_field = evaluate_expression_to_field(rhs_expr, problem)
            rhs_fields[rhs_expr] = rhs_field
        end
    end
    
    return rhs_fields
end

function evaluate_expression_to_field(expr, problem)
    """
    Evaluate an expression to produce a field.
    This would be a complex expression evaluator in a full implementation.
    """
    
    # Handle different expression types
    if isa(expr, Number)
        # Constant RHS
        size = 64  # Default size
        if hasfield(typeof(problem), :domain)
            size = estimate_domain_size(problem.domain)
        end
        return ConstantField(expr, size)
    elseif isa(expr, ScalarField)
        # Direct field reference
        return expr
    else
        # Complex expression - would need proper evaluation
        # For now, return a zero field
        size = 64
        if hasfield(typeof(problem), :equations) && !isempty(problem.equations)
            first_eq = first(problem.equations)
            size = estimate_equation_size(first_eq)
        end
        return ZeroField(size)
    end
end

struct ConstantField
    value::Float64
    size::Int
end

struct ZeroField
    size::Int
end

function extract_field_data(field, requested_size)
    """Extract numerical data from a field object"""
    
    if isa(field, ScalarField)
        # Extract from ScalarField
        if field.current_layout == :g && field.data_g !== nothing
            return Array(field.data_g)
        elseif field.current_layout == :c && field.data_c !== nothing
            return Array(field.data_c)
        else
            # Return zeros if no data
            return zeros(Float64, requested_size)
        end
    elseif isa(field, ConstantField)
        # Constant field
        return fill(field.value, min(field.size, requested_size))
    elseif isa(field, ZeroField)
        # Zero field
        return zeros(Float64, min(field.size, requested_size))
    elseif isa(field, AbstractArray)
        # Direct array
        return Array(field)
    else
        # Unknown field type
        @debug "Unknown field type: $(typeof(field))"
        return zeros(Float64, requested_size)
    end
end

function gather_outputs_from_fields(fields, subproblem=nothing)
    """
    Gather outputs from fields following Dedalus gather_outputs pattern.
    This would apply proper preconditioning in a full implementation.
    """
    
    if isempty(fields)
        return Float64[]
    end
    
    # Simple concatenation for now
    # In full implementation, this would apply left preconditioner
    all_data = Float64[]
    
    for field in fields
        field_data = extract_field_data(field, 0)  # Get all data
        append!(all_data, field_data)
    end
    
    return all_data
end

function create_preconditioner(matrix)
    """Create preconditioner matrix"""
    # Simple diagonal preconditioning as example
    return spdiagm(0 => 1.0 ./ diag(matrix))
end

function field_size(field)
    """
    Get size of field (total number of coefficients/grid points).
    Following Dedalus subsystems.py:165-166: prod(self.field_shape(field)).
    """
    
    if isa(field, ScalarField)
        # Get size from actual field data if available
        if field.data_c !== nothing
            return length(field.data_c)
        elseif field.data_g !== nothing
            return length(field.data_g)
        elseif field.domain !== nothing
            # Compute from domain
            return compute_domain_size(field.domain)
        else
            # Use basis sizes
            total_size = 1
            for basis in field.bases
                if basis !== nothing && hasfield(typeof(basis), :meta)
                    total_size *= basis.meta.size
                else
                    total_size *= 64  # Default
                end
            end
            return total_size
        end
    elseif hasfield(typeof(field), :domain) && field.domain !== nothing
        return compute_domain_size(field.domain)
    elseif hasfield(typeof(field), :data) && field.data !== nothing
        return length(field.data)
    else
        # Fallback estimation
        return estimate_variable_size(field)
    end
end

function compute_domain_size(domain)
    """Compute total degrees of freedom in a domain"""
    if hasfield(typeof(domain), :bases) && domain.bases !== nothing
        total_size = 1
        for basis in domain.bases
            if basis !== nothing && hasfield(typeof(basis), :meta)
                total_size *= basis.meta.size
            else
                total_size *= 32  # Default basis size
            end
        end
        return total_size
    else
        return 64  # Default domain size
    end
end

function total_field_size(problem)
    """
    Get total size of all fields in problem.
    Following Dedalus pattern where state = problem.variables (solvers.py:65).
    """
    if hasfield(typeof(problem), :variables) && problem.variables !== nothing
        return sum(field_size(var) for var in problem.variables)
    else
        return 0
    end
end

function get_state_vector(solver)
    """
    Extract state vector from solver fields.
    Following Dedalus gather pattern (subsystems.py:213-220).
    """
    
    problem = solver.problem
    if !hasfield(typeof(problem), :variables) || problem.variables === nothing
        return Float64[]
    end
    
    # Calculate total size
    total_size = total_field_size(problem)
    if total_size == 0
        return Float64[]
    end
    
    # Gather data from all variables
    state_vector = zeros(Float64, total_size)
    offset = 1  # Julia 1-based indexing
    
    for var in problem.variables
        var_size = field_size(var)
        if var_size > 0
            # Extract data from field
            field_data = extract_field_data_for_state(var)
            
            # Copy to state vector
            end_idx = offset + var_size - 1
            if length(field_data) >= var_size
                state_vector[offset:end_idx] = field_data[1:var_size]
            else
                # Handle size mismatch
                copy_length = min(length(field_data), var_size)
                if copy_length > 0
                    state_vector[offset:offset+copy_length-1] = field_data[1:copy_length]
                end
                @debug "State vector size mismatch for variable: expected $var_size, got $(length(field_data))"
            end
        end
        offset += var_size
    end
    
    return state_vector
end

function set_state_vector!(solver, state_vector::AbstractVector)
    """
    Set state vector in solver fields.
    Following Dedalus scatter pattern (subsystems.py:222-231).
    """
    
    problem = solver.problem
    if !hasfield(typeof(problem), :variables) || problem.variables === nothing
        @warn "No variables found in problem"
        return
    end
    
    if isempty(state_vector)
        @warn "Empty state vector provided"
        return
    end
    
    # Scatter data to all variables
    offset = 1  # Julia 1-based indexing
    
    for var in problem.variables
        var_size = field_size(var)
        if var_size > 0
            # Extract data for this variable
            end_idx = min(offset + var_size - 1, length(state_vector))
            if end_idx >= offset
                var_data = state_vector[offset:end_idx]
                
                # Set data in field
                set_field_data_from_state!(var, var_data)
            else
                @debug "State vector too short for variable at offset $offset"
            end
        end
        offset += var_size
    end
end

function extract_field_data_for_state(field)
    """
    Extract data from field for state vector.
    Prefers coefficient space following Dedalus convention.
    """
    
    if isa(field, ScalarField)
        # Prefer coefficient space (following Dedalus solvers.py:399-400)
        if field.data_c !== nothing
            ensure_layout!(field, :c)  # Ensure coefficient space
            return Array(field.data_c)
        elseif field.data_g !== nothing
            # Convert to coefficient space if possible
            if field.data_c === nothing
                require_coeff_space!(field)
            end
            if field.data_c !== nothing
                return Array(field.data_c)
            else
                # Fallback to grid space
                return Array(field.data_g)
            end
        else
            # No data allocated - return zeros
            return zeros(Float64, field_size(field))
        end
    elseif hasfield(typeof(field), :data) && field.data !== nothing
        return Array(field.data)
    else
        # Unknown field type - return zeros
        return zeros(Float64, field_size(field))
    end
end

function set_field_data_from_state!(field, data::AbstractVector)
    """
    Set field data from state vector data.
    Sets coefficient space following Dedalus convention.
    """
    
    if isa(field, ScalarField)
        # Ensure coefficient space layout
        ensure_layout!(field, :c)
        
        if field.data_c !== nothing
            # Copy data to coefficient space
            copy_length = min(length(data), length(field.data_c))
            if copy_length > 0
                field.data_c[1:copy_length] = data[1:copy_length]
                field.current_layout = :c
            end
        else
            @debug "No coefficient space data allocated for field $(field.name)"
        end
    elseif hasfield(typeof(field), :data)
        # Generic field with data attribute
        if field.data !== nothing
            copy_length = min(length(data), length(field.data))
            if copy_length > 0
                field.data[1:copy_length] = data[1:copy_length]
            end
        end
    else
        @debug "Unknown field type for state setting: $(typeof(field))"
    end
end

function standard_timestep!(solver, state::Vector, dt::Float64)
    """
    Standard timestepping implementation following Dedalus pattern.
    Following Dedalus solver step() pattern (solvers.py:683-712) and 
    timestepper step() patterns (timesteppers.py:95-188, 552-644).
    """
    
    # Assert finite timestep (following solvers.py:686-687)
    if !isfinite(dt)
        throw(ArgumentError("Invalid timestep: $dt"))
    end
    
    problem = solver.problem
    timestepper = solver.timestepper
    
    if !hasfield(typeof(problem), :variables) || problem.variables === nothing
        @warn "No variables found in problem"
        return state
    end
    
    # Set current state in problem variables (scatter from state vector)
    set_state_vector!(solver, state)
    
    # Dispatch to appropriate timestepper following Dedalus pattern
    if isa(timestepper, CNAB1) || isa(timestepper, CNAB2)
        # Multistep IMEX timestepping (following timesteppers.py:95-188)
        return multistep_imex_timestep!(solver, dt)
        
    elseif isa(timestepper, RK222) || isa(timestepper, RK443)
        # Runge-Kutta IMEX timestepping (following timesteppers.py:552-644)
        return runge_kutta_imex_timestep!(solver, dt)
        
    elseif isa(timestepper, SBDF1) || isa(timestepper, SBDF2)
        # Backwards differentiation formulas
        return bdf_timestep!(solver, dt)
        
    else
        # Fallback: simple forward Euler step
        @debug "Using fallback forward Euler step for timestepper: $(typeof(timestepper))"
        return forward_euler_step!(solver, state, dt)
    end
end

function multistep_imex_timestep!(solver, dt::Float64)
    """
    Multistep IMEX timestepping following Dedalus pattern.
    Following timesteppers.py:95-188 for CNAB1/CNAB2.
    """
    
    problem = solver.problem
    timestepper = solver.timestepper
    
    # Get timestepper coefficients (simplified for now)
    if isa(timestepper, CNAB1)
        # CNAB1: (M + 0.5*dt*L).X = M.X0 + dt*F0
        a0, b0, c1 = 1.0, 0.5*dt, dt
    elseif isa(timestepper, CNAB2)
        # CNAB2: (M + 0.5*dt*L).X = M.X0 + dt*(1.5*F0 - 0.5*F1)
        a0, b0, c1 = 1.0, 0.5*dt, 1.5*dt
    else
        # Fallback coefficients
        a0, b0, c1 = 1.0, dt, dt
    end
    
    # Build and solve linear system: (a0*M + b0*L).X = RHS
    # This would require accessing M and L matrices from problem
    # For now, return current state (placeholder)
    @debug "Multistep IMEX step with coefficients: a0=$a0, b0=$b0, c1=$c1"
    
    return get_state_vector(solver)
end

function runge_kutta_imex_timestep!(solver, dt::Float64)
    """
    Runge-Kutta IMEX timestepping following Dedalus pattern.
    Following timesteppers.py:552-644 for RK222/RK443.
    """
    
    problem = solver.problem
    timestepper = solver.timestepper
    
    stages = 1  # Default
    if isa(timestepper, RK222)
        stages = 2
    elseif isa(timestepper, RK443) 
        stages = 4
    end
    
    @debug "Runge-Kutta IMEX step with $stages stages"
    
    # For each stage: (M + k*Hii*L).X = M.X0 + k*Aij*F - k*Hij*L.X
    # This would require stage loop and matrix operations
    # For now, return current state (placeholder)
    
    return get_state_vector(solver)
end

function bdf_timestep!(solver, dt::Float64)
    """
    Backwards differentiation formula timestepping.
    Placeholder implementation.
    """
    
    @debug "BDF timestep with dt=$dt"
    return get_state_vector(solver)
end

function forward_euler_step!(solver, state::Vector, dt::Float64)
    """
    Simple forward Euler step as fallback.
    X_new = X + dt * F(X)
    """
    
    try
        # Evaluate RHS: F(X) 
        rhs = evaluate_rhs(solver, state)
        
        # Forward Euler: X_new = X + dt * F(X)
        new_state = state + dt * rhs
        
        return new_state
        
    catch e
        @warn "Forward Euler step failed: $e, returning original state"
        return state
    end
end

function evaluate_rhs(solver, state::Vector)
    """
    Evaluate right-hand side F(X) for explicit timestepping.
    This would evaluate the nonlinear terms and forcing.
    """
    
    # Placeholder: return zero RHS
    return zeros(Float64, length(state))
end

function evaluate_explicit_rhs_optimized!(rhs, solver, state, time)
    """Evaluate explicit RHS terms"""
    # Would evaluate nonlinear terms efficiently
    # Placeholder implementation
    nothing
end

function add_nonlinear_terms_optimized!(rhs, solver, state, time)
    """Add nonlinear terms to RHS"""
    # Would use the nonlinear evaluation system
    # Placeholder implementation  
    nothing
end

function distribute_solution!(solver)
    """Distribute solution back to problem variables"""
    # Would copy solution vector components back to individual fields
end

# GPU-specific optimized solver functions
function optimized_timestep_gpu!(solver::OptimizedInitialValueSolver, state::AbstractArray, dt::Float64)
    """Perform GPU-accelerated optimized timestep"""
    
    # Ensure state is on correct device
    state = ensure_device!(state, solver.device_config)
    
    problem = solver.problem
    timestepper = solver.timestepper
    
    # Get workspace arrays on GPU
    workspace = get_workspace_arrays_gpu(solver, size(state))
    
    if isa(timestepper, RK222) || isa(timestepper, RK443)
        # GPU-accelerated Runge-Kutta timestepping
        return optimized_runge_kutta_step_gpu!(solver, state, dt, workspace)
        
    elseif isa(timestepper, CNAB1) || isa(timestepper, CNAB2)
        # GPU-accelerated IMEX with optimized linear solvers
        return optimized_imex_step_gpu!(solver, state, dt, workspace)
        
    else
        # Fall back to CPU optimization
        state_cpu = Array(state)
        result_cpu = optimized_timestep!(solver, state_cpu, dt)
        return device_array(result_cpu, solver.device_config)
    end
end

function optimized_runge_kutta_step_gpu!(solver::OptimizedInitialValueSolver, state::AbstractArray, dt::Float64, workspace::Dict)
    """GPU-accelerated optimized Runge-Kutta stepping"""
    
    k1 = workspace["k1"]
    k2 = workspace["k2"]  
    k3 = workspace["k3"]
    k4 = workspace["k4"]
    temp_state = workspace["temp_state"]
    
    # RK4 stages with GPU-optimized operations
    
    # k1 = f(t, y)
    evaluate_rhs_optimized_gpu!(k1, solver, state, solver.sim_time)
    
    # k2 = f(t + dt/2, y + dt/2 * k1)
    @. temp_state = state + dt/2 * k1
    evaluate_rhs_optimized_gpu!(k2, solver, temp_state, solver.sim_time + dt/2)
    
    # k3 = f(t + dt/2, y + dt/2 * k2)
    @. temp_state = state + dt/2 * k2
    evaluate_rhs_optimized_gpu!(k3, solver, temp_state, solver.sim_time + dt/2)
    
    # k4 = f(t + dt, y + dt * k3)
    @. temp_state = state + dt * k3
    evaluate_rhs_optimized_gpu!(k4, solver, temp_state, solver.sim_time + dt)
    
    # Final update: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    new_state = device_similar(state, solver.device_config)
    @. new_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return new_state
end

function optimized_imex_step_gpu!(solver::OptimizedInitialValueSolver, state::AbstractArray, dt::Float64, workspace::Dict)
    """GPU-accelerated optimized IMEX stepping"""
    
    implicit_rhs = workspace["implicit_rhs"]
    explicit_rhs = workspace["explicit_rhs"]
    temp_state = workspace["temp_state"]
    
    # Evaluate explicit terms on GPU
    evaluate_explicit_rhs_optimized_gpu!(explicit_rhs, solver, state, solver.sim_time)
    
    # Setup implicit system on GPU: (M - dt*L)*y_new = M*y_old + dt*explicit_rhs
    
    # Right-hand side: M*y_old + dt*explicit_rhs
    if haskey(solver.linear_operators, "M_combined")
        optimized_matvec_gpu!(implicit_rhs, solver.linear_operators["M_combined"], state, solver.device_config)
    else
        implicit_rhs .= state  # Identity mass matrix
    end
    
    # Add explicit terms
    @. implicit_rhs += dt * explicit_rhs
    
    # Solve implicit system on GPU
    if haskey(solver.linear_operators, "implicit_matrix")
        solve_implicit_system_optimized_gpu!(temp_state, solver, implicit_rhs)
    else
        temp_state .= implicit_rhs  # Placeholder
    end
    
    return temp_state
end

function evaluate_rhs_optimized_gpu!(rhs::AbstractArray, solver::OptimizedInitialValueSolver, state::AbstractArray, time::Float64)
    """Evaluate right-hand side using GPU-optimized operations"""
    
    fill!(rhs, 0.0)
    
    # Apply linear operators using GPU-optimized matrix-vector products
    for (op_name, op) in solver.linear_operators
        if startswith(op_name, "L_")
            temp_result = get_temp_array_gpu(solver, size(state))
            optimized_matvec_gpu!(temp_result, op, state, solver.device_config)
            rhs .+= temp_result
            return_temp_array_gpu!(solver, temp_result)
        end
    end
    
    # Add nonlinear terms (GPU-accelerated)
    add_nonlinear_terms_optimized_gpu!(rhs, solver, state, time)
end

function evaluate_explicit_rhs_optimized_gpu!(rhs::AbstractArray, solver::OptimizedInitialValueSolver, state::AbstractArray, time::Float64)
    """Evaluate explicit RHS terms on GPU"""
    # GPU-accelerated nonlinear term evaluation
    fill!(rhs, 0.0)
    add_nonlinear_terms_optimized_gpu!(rhs, solver, state, time)
end

function add_nonlinear_terms_optimized_gpu!(rhs::AbstractArray, solver::OptimizedInitialValueSolver, state::AbstractArray, time::Float64)
    """Add nonlinear terms using GPU-optimized evaluation"""
    # Placeholder for GPU-accelerated nonlinear evaluation
    # Would interface with GPU-enabled nonlinear_terms.jl
    nothing
end

function optimized_matvec_gpu!(result::AbstractArray, op, input::AbstractArray, device_config::DeviceConfig)
    """GPU-optimized matrix-vector product"""
    
    # Ensure arrays are on correct device
    input = ensure_device!(input, device_config)
    result = ensure_device!(result, device_config)
    
    if device_config.device_type != CPU_DEVICE
        # Use GPU-accelerated linear algebra
        if hasfield(typeof(op), :matrix)
            op_matrix = ensure_device!(op.matrix, device_config)
            result .= op_matrix * input
        else
            @warn "GPU matrix-vector product not available for $(typeof(op)), using CPU fallback"
            input_cpu = Array(input)
            result_cpu = Array(result)
            optimized_matvec!(result_cpu, op, input_cpu)
            result .= device_array(result_cpu, device_config)
        end
    else
        # CPU optimized operation
        optimized_matvec!(Array(result), op, Array(input))
    end
end

function solve_implicit_system_optimized_gpu!(solution::AbstractArray, solver::OptimizedInitialValueSolver, rhs::AbstractArray)
    """Solve implicit system using GPU-optimized methods"""
    
    # Ensure arrays are on correct device
    rhs = ensure_device!(rhs, solver.device_config)
    solution = ensure_device!(solution, solver.device_config)
    
    if solver.device_config.device_type != CPU_DEVICE
        # GPU-accelerated solve
        try
            # Would use GPU-optimized solver here
            solution .= rhs  # Placeholder
        catch e
            @warn "GPU implicit solve failed: $e, using CPU fallback"
            rhs_cpu = Array(rhs)
            solution_cpu = Array(solution)
            solve_implicit_system_optimized!(solution_cpu, solver, rhs_cpu)
            solution .= device_array(solution_cpu, solver.device_config)
        end
    else
        # CPU optimized solve
        solve_implicit_system_optimized!(Array(solution), solver, Array(rhs))
    end
end

function solve_direct_optimized_gpu!(solver::OptimizedBoundaryValueSolver)
    """Direct solve using GPU-optimized factorization"""
    
    if solver.device_config.device_type == CPU_DEVICE
        # CPU optimized solve
        solve_direct_optimized!(solver)
        return
    end
    
    # GPU factorization and solve
    if solver.factorization !== nothing && check_device_compatibility(solver.factorization, solver.device_config)
        # Use pre-computed GPU factorization
        try
            ldiv!(solver.solution_vector, solver.factorization, solver.RHS_vector)
        catch e
            @warn "GPU factorization solve failed: $e, rebuilding factorization"
            solver.factorization = nothing
        end
    end
    
    if solver.factorization === nothing
        # Create new GPU factorization
        try
            if issparse(solver.LHS_matrix)
                solver.factorization = lu(solver.LHS_matrix)
            else
                solver.factorization = lu(solver.LHS_matrix)
            end
            ldiv!(solver.solution_vector, solver.factorization, solver.RHS_vector)
        catch e
            @warn "GPU direct solve failed: $e, falling back to CPU"
            LHS_cpu = Array(solver.LHS_matrix)
            RHS_cpu = Array(solver.RHS_vector)
            solution_cpu = LHS_cpu \ RHS_cpu
            solver.solution_vector .= device_array(solution_cpu, solver.device_config)
        end
    end
end

function solve_iterative_optimized_gpu!(solver::OptimizedBoundaryValueSolver)
    """Iterative solve using GPU-optimized operations"""
    
    if solver.device_config.device_type == CPU_DEVICE
        # CPU optimized solve
        solve_iterative_optimized!(solver)
        return
    end
    
    x = solver.solution_vector
    b = solver.RHS_vector
    tolerance = solver.tolerance
    max_iter = solver.max_iterations
    
    # GPU-optimized iterative solver
    residual = device_similar(b, solver.device_config)
    temp_vec = device_similar(b, solver.device_config)
    
    fill!(x, 0.0)  # Initial guess
    
    for iter in 1:max_iter
        # Compute residual: r = b - A*x (GPU operations)
        residual .= solver.LHS_matrix * x
        @. residual = b - residual
        
        residual_norm = norm(residual)
        
        if residual_norm < tolerance
            @info "GPU iterative solver converged in $iter iterations, residual: $(residual_norm)"
            break
        end
        
        # Apply preconditioning if available (GPU)
        if solver.preconditioner !== nothing
            temp_vec .= solver.preconditioner * residual
            residual .= temp_vec
        end
        
        # Simple Richardson iteration (would implement proper GPU GMRES)
        @. x += 0.1 * residual
        
        if iter == max_iter
            @warn "GPU iterative solver did not converge, residual: $(residual_norm)"
        end
    end
end

# GPU workspace management
function get_workspace_arrays_gpu(solver::OptimizedInitialValueSolver, state_size::Tuple)
    """Get pre-allocated GPU workspace arrays"""
    
    workspace = Dict{String, AbstractArray}()
    
    required_arrays = ["k1", "k2", "k3", "k4", "temp_state", "implicit_rhs", "explicit_rhs"]
    
    for name in required_arrays
        if !haskey(solver.gpu_workspace, name)
            solver.gpu_workspace[name] = device_zeros(Float64, state_size, solver.device_config)
        end
        workspace[name] = solver.gpu_workspace[name]
    end
    
    return workspace
end

function get_temp_array_gpu(solver::OptimizedInitialValueSolver, array_size::Tuple)
    """Get temporary GPU array for intermediate calculations"""
    
    # Try to reuse existing array of same size
    for (i, arr) in enumerate(solver.gpu_memory_pool)
        if size(arr) == array_size && check_device_compatibility(arr, solver.device_config)
            # Remove from pool and return
            temp_arr = splice!(solver.gpu_memory_pool, i)
            fill!(temp_arr, 0)
            return temp_arr
        end
    end
    
    # Create new array on device if no suitable one found
    return device_zeros(Float64, array_size, solver.device_config)
end

function return_temp_array_gpu!(solver::OptimizedInitialValueSolver, arr::AbstractArray)
    """Return temporary GPU array to memory pool for reuse"""
    
    # Only pool arrays up to a reasonable size to avoid memory bloat
    max_elements = 10^6
    if length(arr) <= max_elements && check_device_compatibility(arr, solver.device_config)
        push!(solver.gpu_memory_pool, arr)
    end
    
    # Keep pool size reasonable
    max_pool_size = 20
    if length(solver.gpu_memory_pool) > max_pool_size
        popfirst!(solver.gpu_memory_pool)
    end
end

function get_state_vector_gpu(solver::OptimizedInitialValueSolver)
    """Extract state vector with GPU support"""
    
    problem = solver.problem
    if !hasfield(typeof(problem), :variables) || problem.variables === nothing
        return device_zeros(Float64, (0,), solver.device_config)
    end
    
    # Calculate total size and create GPU array
    total_size = total_field_size(problem)
    if total_size == 0
        return device_zeros(Float64, (0,), solver.device_config)
    end
    
    state_vector = device_zeros(Float64, (total_size,), solver.device_config)
    
    # Gather data from all variables (ensure on correct device)
    offset = 1
    for var in problem.variables
        var_size = field_size(var)
        if var_size > 0
            # Ensure variable data is on correct device
            var.data_c = ensure_device!(var.data_c, solver.device_config)
            field_data = extract_field_data_for_state(var)
            field_data = ensure_device!(field_data, solver.device_config)
            
            # Copy to state vector
            end_idx = offset + var_size - 1
            if length(field_data) >= var_size
                state_vector[offset:end_idx] .= field_data[1:var_size]
            else
                copy_length = min(length(field_data), var_size)
                if copy_length > 0
                    state_vector[offset:offset+copy_length-1] .= field_data[1:copy_length]
                end
            end
        end
        offset += var_size
    end
    
    return state_vector
end

function set_state_vector_gpu!(solver::OptimizedInitialValueSolver, state_vector::AbstractArray)
    """Set state vector with GPU support"""
    
    # Ensure state vector is on correct device
    state_vector = ensure_device!(state_vector, solver.device_config)
    
    problem = solver.problem
    if !hasfield(typeof(problem), :variables) || problem.variables === nothing
        return
    end
    
    # Scatter data to all variables
    offset = 1
    for var in problem.variables
        var_size = field_size(var)
        if var_size > 0
            end_idx = min(offset + var_size - 1, length(state_vector))
            if end_idx >= offset
                var_data = state_vector[offset:end_idx]
                
                # Ensure variable coefficient space is on correct device
                var.data_c = ensure_device!(var.data_c, solver.device_config)
                
                # Set data in field
                set_field_data_from_state!(var, var_data)
            end
        end
        offset += var_size
    end
end

function distribute_solution_gpu!(solver::OptimizedBoundaryValueSolver)
    """Distribute solution back to problem variables with GPU support"""
    
    # Ensure solution vector is on correct device
    solver.solution_vector = ensure_device!(solver.solution_vector, solver.device_config)
    
    problem = solver.problem
    if !hasfield(typeof(problem), :variables) || problem.variables === nothing
        return
    end
    
    # Distribute solution data to variables
    offset = 1
    for var in problem.variables
        var_size = field_size(var)
        if var_size > 0
            end_idx = min(offset + var_size - 1, length(solver.solution_vector))
            if end_idx >= offset
                var_data = solver.solution_vector[offset:end_idx]
                
                # Ensure variable data is on correct device
                var.data_c = ensure_device!(var.data_c, solver.device_config)
                
                # Set field data
                if isa(var, ScalarField) && var.data_c !== nothing
                    copy_length = min(length(var_data), length(var.data_c))
                    if copy_length > 0
                        var.data_c[1:copy_length] .= var_data[1:copy_length]
                        var.current_layout = :c
                    end
                end
            end
        end
        offset += var_size
    end
end

# GPU utility functions
function device_similar(arr::AbstractArray, device_config::DeviceConfig)
    """Create similar array on specified device"""
    return device_zeros(eltype(arr), size(arr), device_config)
end

# Export optimized solvers
export OptimizedInitialValueSolver, OptimizedBoundaryValueSolver
export setup_optimized_operators!, build_optimized_system!
export step!, solve!
export optimized_timestep_gpu!, get_state_vector_gpu, set_state_vector_gpu!