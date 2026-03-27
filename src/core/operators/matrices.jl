"""
    Expression matrices

This file contains the expression_matrices() functions for building
sparse matrices for implicit solvers, along with helper functions
for differentiation matrices and lift matrices.
"""

using LinearAlgebra
using SparseArrays

# ============================================================================
# Expression Matrices for Matrix Assembly
# ============================================================================

"""
    expression_matrices(op::Operator, sp, vars; kwargs...)

Build expression matrices for operator applied to each variable.
Following operators expression_matrices method.

Returns Dict mapping variables to sparse matrices.
"""
function expression_matrices(op::Operator, sp, vars; kwargs...)
    # Default: return empty dict (override for specific operators)
    return Dict{Any, SparseMatrixCSC}()
end

"""
    expression_matrices(op::TimeDerivative, sp, vars; kwargs...)

Time derivative matrices: returns M matrix contribution.
Following operators TimeDerivative.expression_matrices.
"""
function expression_matrices(op::TimeDerivative, sp, vars; kwargs...)
    operand = op.operand
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Identity matrix for time derivative term
            n = field_dofs(var)
            # Mass matrix for time derivative is always identity regardless of order.
            # The derivative order is handled by the timestepping scheme, not the mass matrix.
            result[var] = sparse(I, n, n) * 1.0
        end
    end

    return result
end

"""
    expression_matrices(op::Differentiate, sp, vars; kwargs...)

Spatial differentiation matrices.
Following operators Differentiate.expression_matrices.
"""
function expression_matrices(op::Differentiate, sp, vars; kwargs...)
    operand = op.operand
    coord = op.coord
    order = op.order
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Build differentiation matrix for this variable
            D = build_operator_differentiation_matrix(var, coord, order; kwargs...)
            if D !== nothing
                result[var] = D
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Laplacian, sp, vars; kwargs...)

Laplacian matrices: sum of second derivatives.
Following operators Laplacian.expression_matrices.
"""
function expression_matrices(op::Laplacian, sp, vars; kwargs...)
    operand = op.operand
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Build Laplacian matrix = sum of D_i^2 for each coordinate
            lap_mat = nothing

            if hasfield(typeof(var), :bases)
                for basis in var.bases
                    if basis !== nothing
                        coord = get_coord_for_basis(basis)
                        D2 = build_operator_differentiation_matrix(var, coord, 2; kwargs...)
                        if D2 !== nothing
                            if lap_mat === nothing
                                lap_mat = D2
                            else
                                lap_mat = lap_mat + D2
                            end
                        end
                    end
                end
            end

            if lap_mat !== nothing
                result[var] = lap_mat
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Gradient, sp, vars; kwargs...)

Gradient matrices for scalar -> vector.
Following operators Gradient.expression_matrices.
"""
function expression_matrices(op::Gradient, sp, vars; kwargs...)
    operand = op.operand
    coordsys = op.coordsys
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # For Cartesian: gradient is vector of partial derivatives
            # Build block diagonal matrix with D_i for each component
            n = field_dofs(var)
            ndim = coordsys.dim

            blocks = SparseMatrixCSC[]
            for coord in coordsys.coords
                D = build_operator_differentiation_matrix(var, coord, 1; kwargs...)
                if D !== nothing
                    push!(blocks, D)
                else
                    push!(blocks, spzeros(Float64, n, n))
                end
            end

            # Stack blocks vertically for vector output
            if !isempty(blocks)
                result[var] = vcat(blocks...)
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Divergence, sp, vars; kwargs...)

Divergence matrices for vector -> scalar.
Following operators Divergence.expression_matrices.
"""
function expression_matrices(op::Divergence, sp, vars; kwargs...)
    operand = op.operand
    result = Dict{Any, SparseMatrixCSC}()

    if !isa(operand, VectorField)
        return result
    end

    coordsys = operand.coordsys

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && var.name == operand.name)
            # For Cartesian: divergence is sum of partial derivatives of components
            # Build row of blocks [D_x, D_y, D_z]
            n_comp = length(operand.components)
            n_per_comp = n_comp > 0 ? field_dofs(operand.components[1]) : 0

            blocks = SparseMatrixCSC[]
            for (i, coord) in enumerate(coordsys.coords)
                comp = operand.components[i]
                D = build_operator_differentiation_matrix(comp, coord, 1; kwargs...)
                if D !== nothing
                    push!(blocks, D)
                else
                    push!(blocks, spzeros(Float64, n_per_comp, n_per_comp))
                end
            end

            # Concatenate blocks horizontally for scalar output
            if !isempty(blocks)
                result[var] = hcat(blocks...)
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Lift, sp, vars; kwargs...)

Lift matrices for boundary conditions (tau method).
Following operators Lift.expression_matrices.
"""
function expression_matrices(op::Lift, sp, vars; kwargs...)
    operand = op.operand
    basis = op.basis
    n = op.n
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Lift matrix places tau values at specific spectral modes
            lift_mat = build_lift_matrix(var, basis, n; kwargs...)
            if lift_mat !== nothing
                result[var] = lift_mat
            end
        end
    end

    return result
end

"""
    expression_matrices(op::Convert, sp, vars; kwargs...)

Basis conversion matrices.
Following operators Convert.expression_matrices.
"""
function expression_matrices(op::Convert, sp, vars; kwargs...)
    operand = op.operand
    out_basis = op.basis
    result = Dict{Any, SparseMatrixCSC}()

    for var in vars
        if var === operand || (hasfield(typeof(var), :name) && hasfield(typeof(operand), :name) && var.name == operand.name)
            # Get input basis and build conversion matrix
            if hasfield(typeof(var), :bases) && !isempty(var.bases)
                in_basis = var.bases[1]
                if in_basis !== nothing && isa(in_basis, JacobiBasis) && isa(out_basis, JacobiBasis)
                    conv_mat = conversion_matrix(in_basis, out_basis)
                    result[var] = conv_mat
                end
            end
        end
    end

    return result
end

# ============================================================================
# Helper Functions for Building Operator Matrices
# ============================================================================

"""
    build_operator_differentiation_matrix(var, coord, order; kwargs...)

Build differentiation matrix for variable with respect to coordinate.
"""
function build_operator_differentiation_matrix(var, coord::Coordinate, order::Int; kwargs...)
    if !hasfield(typeof(var), :bases)
        return nothing
    end

    # Find the basis corresponding to this coordinate
    basis_idx = nothing
    target_basis = nothing

    for (i, basis) in enumerate(var.bases)
        if basis !== nothing && basis.meta.element_label == coord.name
            basis_idx = i
            target_basis = basis
            break
        end
    end

    if target_basis === nothing
        return nothing
    end

    n_total = field_dofs(var)
    n_basis = target_basis.meta.size

    # Build 1D differentiation matrix based on basis type
    D1d = nothing

    if isa(target_basis, JacobiBasis)
        D1d = differentiation_matrix(target_basis, order)
    elseif isa(target_basis, FourierBasis)
        D1d = fourier_differentiation_matrix(target_basis, order)
    end

    if D1d === nothing
        return nothing
    end

    # For multi-dimensional fields, apply Kronecker product
    if length(var.bases) == 1
        return D1d
    else
        # Build identity matrices for other dimensions
        matrices = AbstractMatrix[]
        for (i, basis) in enumerate(var.bases)
            if basis === nothing
                continue
            end
            if i == basis_idx
                push!(matrices, D1d)
            else
                push!(matrices, sparse(I, basis.meta.size, basis.meta.size))
            end
        end

        # Kronecker product in reverse order for Julia's column-major (Fortran) layout:
        # For bases (B1, B2, ..., Bn), the multi-dimensional operator is
        #   M_n ⊗ ... ⊗ M_2 ⊗ M_1
        # which acts on vectorized data stored in column-major order where
        # the first axis varies fastest.
        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end

        return result
    end
end

"""
    fourier_differentiation_matrix(basis::FourierBasis, order::Int)

Build Fourier differentiation matrix.
"""
function fourier_differentiation_matrix(basis::RealFourier, order::Int)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # RealFourier differentiation follows Dedalus convention:
    # Modes: [cos(0x), cos(1x), -sin(1x), cos(2x), -sin(2x), ...]
    # Note: Using -sin (msin) convention
    #
    # For differentiation with (ik)^order factor:
    # d/dx cos(kx) = -k sin(kx) = k * (-sin(kx))  -> k * msin
    # d/dx (-sin(kx)) = -k cos(kx)                -> -k * cos
    #
    # The 2x2 block for each wavenumber k is:
    # | 0  -k |   (maps: cos <- -k*msin, msin <- k*cos)
    # | k   0 |
    #
    # For order n, we apply this matrix n times, or equivalently compute (ik)^n
    # and extract the real/imaginary parts for the rotation.

    I_list = Int[]
    J_list = Int[]
    V_list = Float64[]

    # DC mode (k=0) derivative is zero — implicitly zero in sparse matrix

    k_max = (N - 1) ÷ 2

    for k in 1:k_max
        cos_idx = 2*k      # 1-indexed: mode 2 is cos(1x), mode 4 is cos(2x), etc.
        sin_idx = 2*k + 1  # 1-indexed: mode 3 is -sin(1x), mode 5 is -sin(2x), etc.
        k_phys = k0 * k

        if cos_idx <= N && sin_idx <= N
            # Compute (ik)^order = k^order * i^order
            # i^0 = 1, i^1 = i, i^2 = -1, i^3 = -i, i^4 = 1, ...
            # For real representation: (ik)^n = k^n * (cos(npi/2) + i*sin(npi/2))
            #
            # The 2x2 derivative block D^n for the (cos, msin) pair:
            # D^1 = | 0  -k |    D^2 = |-k^2  0  |    D^3 = | 0   k^3 |   D^4 = |k^4  0 |
            #       | k   0 |          | 0  -k^2 |          |-k^3  0  |         | 0  k^4|
            #
            # Pattern: D^n has form k^n * |cos(npi/2)  -sin(npi/2)|
            #                             |sin(npi/2)   cos(npi/2)|

            kn = k_phys^order
            phase = order * pi / 2
            c = cos(phase)
            s = sin(phase)

            # Matrix entries for the 2x2 block:
            # cos_out = c * k^n * cos_in - s * k^n * msin_in
            # msin_out = s * k^n * cos_in + c * k^n * msin_in

            # (cos_idx, cos_idx): c * k^n
            if abs(c * kn) > 1e-15
                push!(I_list, cos_idx); push!(J_list, cos_idx); push!(V_list, c * kn)
            end
            # (cos_idx, sin_idx): -s * k^n
            if abs(s * kn) > 1e-15
                push!(I_list, cos_idx); push!(J_list, sin_idx); push!(V_list, -s * kn)
            end
            # (sin_idx, cos_idx): s * k^n
            if abs(s * kn) > 1e-15
                push!(I_list, sin_idx); push!(J_list, cos_idx); push!(V_list, s * kn)
            end
            # (sin_idx, sin_idx): c * k^n
            if abs(c * kn) > 1e-15
                push!(I_list, sin_idx); push!(J_list, sin_idx); push!(V_list, c * kn)
            end
        end
    end

    if isempty(I_list)
        return spzeros(Float64, N, N)
    end

    return sparse(I_list, J_list, V_list, N, N)
end

function fourier_differentiation_matrix(basis::ComplexFourier, order::Int)
    N = basis.meta.size
    L = basis.meta.bounds[2] - basis.meta.bounds[1]
    k0 = 2π / L

    # ComplexFourier: diagonal matrix with (ik)^order (works for both even and odd N)
    k_native = [k <= N÷2 ? k : k - N for k in 0:N-1]
    k_phys = k0 .* k_native

    diag_vals = (im .* k_phys).^order

    return spdiagm(0 => diag_vals)
end

"""
    build_lift_matrix(var, basis, n; kwargs...)

Build lifting matrix for tau method boundary conditions.
Following the standard basis LiftJacobi implementation (lines 790-814).

the standard convention:
- n >= 0: sets mode n directly (0-indexed convention, 1-indexed in Julia)
- n < 0: wraps around (n = -1 means last mode, n = -2 means second-to-last, etc.)

Example: For N=10 modes
- Lift(tau, basis, 0) -> sets mode 1 (Julia 1-indexed)
- Lift(tau, basis, -1) -> sets mode N (last mode)
- Lift(tau, basis, -2) -> sets mode N-1 (second-to-last mode)

Following Dedalus LiftJacobi pattern (basis.py:790-814):
The matrix places the tau variable's coefficient at mode n in the solution.
For LBVP solvers, this creates the "tau polynomial" that adds boundary
condition enforcement terms to the highest modes.
"""
function build_lift_matrix(var, basis, n::Int; kwargs...)
    N = basis.meta.size

    # Resolve mode index: negative wrap-around (Dedalus convention)
    # n < 0: index from end (e.g., -1 -> N-1 in 0-indexed -> N in 1-indexed)
    lift_mode = n
    if lift_mode < 0
        lift_mode = N + lift_mode
    end
    lift_mode += 1  # Convert 0-indexed to 1-indexed Julia convention

    if lift_mode < 1 || lift_mode > N
        @warn "Lift mode $n (resolved to $lift_mode) out of range [1, $N] for basis $(basis.meta.element_label)"
        tau_dofs = max(1, field_dofs(var))
        full_dofs = N * tau_dofs  # Full field DOFs = lift dimension × tau DOFs
        return spzeros(Float64, full_dofs, tau_dofs)
    end

    # Build the 1D lift column vector: e_{lift_mode} of size (N, 1)
    e_lift = sparse([lift_mode], [1], [1.0], N, 1)

    # If var has no bases or only the lift basis, return 1D lift vector
    if !hasfield(typeof(var), :bases) || isempty(var.bases) || all(b -> b === nothing, var.bases)
        return e_lift
    end

    # Find which var basis (if any) matches the lift basis coordinate
    lift_coord = basis.meta.element_label
    basis_idx = nothing
    for (i, b) in enumerate(var.bases)
        if b !== nothing && b.meta.element_label == lift_coord
            basis_idx = i
            break
        end
    end

    # Multi-dimensional case: build Kronecker product
    if basis_idx !== nothing
        # Tau variable already has the lift basis - rare case
        var_basis_size = var.bases[basis_idx].meta.size
        lift_1d = sparse([lift_mode], [lift_mode], [1.0], N, var_basis_size)

        if length(var.bases) == 1
            return lift_1d
        end

        # Kronecker product for multi-dimensional lift operator.
        # Reverse iteration matches Julia's column-major layout: M_n ⊗ ... ⊗ M_1
        # where axis 1 varies fastest in the vectorized representation.
        matrices = AbstractMatrix[]
        for (i, b) in enumerate(var.bases)
            if b === nothing
                continue
            end
            if i == basis_idx
                push!(matrices, lift_1d)
            else
                push!(matrices, sparse(LinearAlgebra.I, b.meta.size, b.meta.size))
            end
        end

        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end
        return result
    else
        # Tau variable does NOT have the lift basis - standard tau method case.
        # Build Kronecker factors in coordinate order, placing e_lift at the
        # correct position (not always last) based on the coordinate system.
        coordsys = basis.meta.coordsys

        # Map tau variable bases by coordinate name for lookup
        var_basis_map = Dict{String, Basis}()
        for b in var.bases
            if b !== nothing
                var_basis_map[b.meta.element_label] = b
            end
        end

        if isempty(var_basis_map)
            return e_lift
        end

        # Build matrices in coordinate order: e_lift at the lift dimension,
        # identity matrices for the tau variable's tangential dimensions
        matrices = AbstractMatrix[]
        for coord in coordsys.coords
            if coord.name == lift_coord
                push!(matrices, e_lift)
            elseif haskey(var_basis_map, coord.name)
                b = var_basis_map[coord.name]
                push!(matrices, sparse(LinearAlgebra.I, b.meta.size, b.meta.size))
            end
        end

        if isempty(matrices)
            return e_lift
        end

        if length(matrices) == 1
            return matrices[1]
        end

        # Reverse Kronecker product for column-major layout (see note above)
        result = matrices[end]
        for i in (length(matrices)-1):-1:1
            result = kron(result, matrices[i])
        end
        return result
    end
end

"""
    get_coord_for_basis(basis::Basis)

Get coordinate associated with a basis.
"""
function get_coord_for_basis(basis::Basis)
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :coordsys)
        coordsys = basis.meta.coordsys
        coord_name = basis.meta.element_label
        if hasfield(typeof(coordsys), :coords)
            for coord in coordsys.coords
                if coord.name == coord_name
                    return coord
                end
            end
        end
    end
    return nothing
end

"""
    field_dofs(field)

Get total degrees of freedom for a field.
"""
function field_dofs(field)
    if hasfield(typeof(field), :buffers) && get_coeff_data(field) !== nothing
        return length(get_coeff_data(field))
    elseif hasfield(typeof(field), :buffers) && get_grid_data(field) !== nothing
        return length(get_grid_data(field))
    elseif hasfield(typeof(field), :bases)
        total = 1
        for basis in field.bases
            if basis !== nothing
                total *= basis.meta.size
            end
        end
        return total
    elseif hasfield(typeof(field), :components)
        # VectorField or TensorField
        return sum(field_dofs(comp) for comp in field.components)
    end
    return 0
end
