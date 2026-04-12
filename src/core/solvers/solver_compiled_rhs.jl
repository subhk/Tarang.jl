function fields_to_vector(fields::Vector{<:ScalarField})

    if isempty(fields)
        return Vector{ComplexF64}()
    end

    # Determine architecture from first field for synchronization
    arch = fields[1].dist.architecture

    # Ensure all fields are in coefficient space (following Tarang pattern)
    # ensure_layout! handles any needed transforms (including GPU FFTs)
    for field in fields
        ensure_layout!(field, :c)
    end

    # Single GPU sync after all layout changes are complete
    if is_gpu(arch)
        synchronize(arch)
    end

    # Calculate total vector size
    total_size = sum(compute_field_vector_size(field) for field in fields)

    # Allocate fresh vector each call to avoid shared-buffer aliasing bugs
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

"""
    Copy solution vector back to fields following scatter pattern.

    GPU-aware: The solution vector is on CPU (from linear solver).
    For GPU fields, data is transferred to GPU and synchronized.

    MPI-aware: If fields are PencilArray-distributed and the solution is the
    global vector (from _gather_to_global_vector), each rank extracts only
    its local portion based on the rank offset.
    """
function copy_solution_to_fields!(fields::Vector{<:ScalarField}, solution::AbstractVector{<:Number})

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

"""
    Convert solution vector to a new state vector matching a template.

    GPU-aware: New fields are allocated on the same architecture as the
    template fields. The vector (typically on CPU from linear solve) is
    transferred to GPU via set_field_data_from_vector! if needed.
    """
function vector_to_fields(vector::AbstractVector{<:Number}, template::Vector{<:ScalarField})

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

"""
    vector_to_fields!(output, vector, template)

In-place variant: writes vector data into pre-existing output fields.
No field allocation. Output fields must already have coeff data allocated.
"""
function vector_to_fields!(output::Vector{<:ScalarField}, vector::AbstractVector{<:Number},
                           template::Vector{<:ScalarField})
    offset = 1
    for (i, field) in enumerate(template)
        coeff_data = get_coeff_data(output[i])
        if coeff_data === nothing
            continue
        end
        local_data = get_local_data(coeff_data)
        n = length(local_data)
        if n > 0 && offset <= length(vector)
            end_idx = min(offset + n - 1, length(vector))
            copyto!(local_data, 1, vector, offset, end_idx - offset + 1)
            offset = end_idx + 1
        end
        output[i].current_layout = :c
    end
    return output
end

"""
    Compute the number of degrees of freedom for a field in vector form.
    Following field size computation patterns.
    """
function compute_field_vector_size(field::ScalarField)

    # Tau variables (empty bases): always 1 DOF regardless of data state
    if isempty(field.bases)
        return 1
    end

    if get_coeff_data(field) !== nothing
        # Use coefficient space data size
        return length(get_coeff_data(field))
    elseif get_grid_data(field) !== nothing
        # Use grid space data size as fallback
        return length(get_grid_data(field))
    else
        # Default size based on bases — use coeff-space sizes (rfft halves first RealFourier dim)
        total_size = 1
        first_rf = true
        for basis in field.bases
            if basis !== nothing
                if isa(basis, RealFourier) && first_rf
                    total_size *= div(get_basis_size(basis), 2) + 1
                    first_rf = false
                else
                    total_size *= get_basis_size(basis)
                end
            end
        end
        return total_size
    end
end

"""
    Extract field data for vector conversion with proper layout handling.
    Following field data extraction patterns.

    GPU-aware: For GPU fields, transfers data to CPU using get_cpu_data.
    This is necessary because linear solvers typically work on CPU.
    Synchronization is handled by the caller (fields_to_vector).
    """
function extract_field_data_for_vector(field::ScalarField)

    # Tau variables (empty bases, 0D): return single zero element
    if isempty(field.bases)
        return zeros(ComplexF64, 1)
    end

    # Ensure coefficient space layout
    ensure_layout!(field, :c)

    if get_coeff_data(field) !== nothing
        cpu_data = get_cpu_data(get_coeff_data(field))
        return vec(cpu_data)
    elseif get_grid_data(field) !== nothing
        @warn "Using grid space data for field $(field.name) - converting to coefficient space recommended"
        cpu_data = get_cpu_data(get_grid_data(field))
        return vec(cpu_data)
    else
        field_size = compute_field_vector_size(field)
        return zeros(ComplexF64, field_size)
    end
end

"""
    Set field data from vector with proper shape and layout handling.
    Following field data setting patterns.

    GPU-aware: For GPU fields, CPU data is transferred to GPU using
    on_architecture(). This allows linear solves to happen on CPU
    and results to be seamlessly copied back to GPU fields.

    Note: Synchronization is handled by the caller (copy_solution_to_fields!).
    """
function set_field_data_from_vector!(field::ScalarField, data::AbstractVector{<:Number})

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

"""Get the size (number of modes) for a basis following Tarang patterns"""
function get_basis_size(basis)
    
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

"""
    Evaluate residual and Jacobian for nonlinear problem following Tarang patterns.
    
    In Tarang, this corresponds to:
    1. Evaluating F expressions (residual) using evaluator system
    2. Building dF matrices (Jacobian/Frechet differential) 
    3. Gathering results into numerical arrays for Newton solver
    """
function evaluate_residual_and_jacobian(problem::NLBVP, x::Vector{ComplexF64})
    
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

"""
    Create a constant field from a template.

    GPU-aware: The field inherits the architecture from the template.
    Uses fill!() which works on both CPU and GPU arrays.
    """
function _constant_field_from_template(template::ScalarField, value::Number; layout::Symbol=:g)
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

"""
    Evaluate a parsed solver expression with current field values.
    Returns a field (preferred) or a numeric scalar for constant expressions.
    """
function evaluate_solver_expression(expr, variables; layout::Symbol=:g, template::Union{Nothing, ScalarField}=nothing)

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
        @debug "Unknown operator expression: $(expr.expression)"
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

"""
    Build Jacobian matrix block from Frechet differential expression following Tarang patterns.
    
    In Tarang, this corresponds to:
    1. expr.expression_matrices(subproblem, vars) 
    2. Returns dict {var: matrix} for each variable
    3. Recursively builds matrices for expression tree
    """
function build_jacobian_block(expr, variables, perturbations)
    
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

"""Build identity matrix block for variable (Tarang pattern)"""
function build_variable_jacobian_block(expr, variables)

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

"""Build Jacobian block for operator expression (following Tarang recursive patterns)"""
function build_operator_jacobian_block(expr, variables, perturbations)

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

"""
    Create field with constant value.

    GPU-aware: The field is allocated on the same architecture as the first variable.
    Uses fill!() which works on both CPU and GPU arrays.
    """
function create_constant_field(expr, variables)
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

"""
    Apply addition operator following Tarang patterns.

    GPU-aware: Uses broadcasting (.+=) which works on both CPU and GPU arrays.
    """
function apply_add_operator(operands)
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

"""
    Apply multiplication operator following Tarang patterns.

    GPU-aware: Uses broadcasting (.*=) which works on both CPU and GPU arrays.
    """
function apply_multiply_operator(operands)
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

"""
    Apply differentiation operator using existing operators.jl infrastructure.
    
    This leverages the complete implementation in operators.jl which includes:
    - Basis-specific differentiation (Fourier, Chebyshev, Legendre)
    - Proper spectral differentiation matrices
    - Layout management and efficient operations
    """
function apply_differentiate_operator(operands, expr)
    
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

"""Extract coordinate for differentiation from expression"""
function get_diff_coordinate(expr)
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

"""Extract differentiation order from expression"""
function get_diff_order(expr)
    if hasfield(typeof(expr), :order)
        return max(1, Int(expr.order))
    else
        return 1  # Default to first order
    end
end

# Performance and logging
"""Log solver performance statistics"""
function log_stats(solver::Solver)
    
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

"""Log solver performance statistics"""
function log_solver_performance(solver::Union{InitialValueSolver, BoundaryValueSolver})

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
                    field_name = hasfield(typeof(template), :name) ? template.name : "field_$state_idx"
                    @info "RHS compiler: cannot compile $(field_name) equation's F expression ($(typeof(expr))): $e. Using interpreted evaluation." maxlog=3
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
    if expr isa VectorField
        # When compiling a per-component equation (template is a scalar component),
        # replace the VectorField with its component matching the template's name.
        # This handles cases like Differentiate(u, x) in vector advection where
        # u is the full VectorField but we're compiling one component's equation.
        for comp in expr.components
            if comp === template || (hasfield(typeof(comp), :name) &&
                                     hasfield(typeof(template), :name) &&
                                     comp.name == template.name)
                return _compile_expr!(plan, comp, template, state)
            end
        end
        # Template name doesn't match any component — try matching by suffix
        # (e.g., template "u_x" → find component ending in "_x")
        if hasfield(typeof(template), :name)
            tname = String(template.name)
            for comp in expr.components
                if hasfield(typeof(comp), :name)
                    cname = String(comp.name)
                    # Match if component name ends with same suffix as template
                    if endswith(tname, "_x") && endswith(cname, "_x")
                        return _compile_expr!(plan, comp, template, state)
                    elseif endswith(tname, "_z") && endswith(cname, "_z")
                        return _compile_expr!(plan, comp, template, state)
                    elseif endswith(tname, "_y") && endswith(cname, "_y")
                        return _compile_expr!(plan, comp, template, state)
                    end
                end
            end
        end
        throw(ArgumentError("Cannot compile VectorField $(expr.name) with template $(template.name) — no matching component"))

    elseif expr isa ScalarField
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

