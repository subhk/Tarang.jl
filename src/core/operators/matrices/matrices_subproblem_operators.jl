"""
    Subproblem operator matrices

This file contains `subproblem_matrix` implementations for the linear operator
path used during subsystem matrix assembly, along with the remaining
operator-specific `expression_matrices` fallbacks.
"""

# ============================================================================
# subproblem_matrix implementations for linear operators
# ============================================================================

"""
    subproblem_matrix(op::TimeDerivative, sp; kwargs...)

Time derivative: identity matrix. The derivative order is handled by the
timestepping scheme, not the mass matrix.
"""
function subproblem_matrix(op::TimeDerivative, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end
    n = subproblem_field_size(sp, field)
    return sparse(ComplexF64(1) * I, n, n)
end

"""
    subproblem_matrix(op::Differentiate, sp; kwargs...)

Spatial differentiation: returns the per-subproblem differentiation matrix
for the specified coordinate and order.
"""
function subproblem_matrix(op::Differentiate, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end
    n = subproblem_field_size(sp, field)
    coord_name = isa(op.coord.name, Symbol) ? String(op.coord.name) : op.coord.name
    return _subproblem_diff_matrix(sp, coord_name, op.order, n)
end

"""
    subproblem_matrix(op::Laplacian, sp; kwargs...)

Laplacian: sum of second derivatives across all coordinates.
For 2D Fourier-Chebyshev: -kx² * I_Nz + D_z²
"""
function subproblem_matrix(op::Laplacian, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end
    n = subproblem_field_size(sp, field)
    lap_mat = spzeros(ComplexF64, n, n)

    # Sum second derivatives over all coordinates in the distributor
    if hasfield(typeof(sp), :dist) && sp.dist !== nothing && hasfield(typeof(sp.dist), :coords)
        for coord in sp.dist.coords
            coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
            D2 = _subproblem_diff_matrix(sp, coord_name, 2, n)
            lap_mat = lap_mat + D2
        end
    end
    return lap_mat
end

"""
    subproblem_matrix(op::Gradient, sp; kwargs...)

Gradient: stacks per-coordinate differentiation matrices vertically.
For scalar operand: (ndim*Nz × Nz).
For vector operand: (ndim*n_vec × n_vec).
"""
function subproblem_matrix(op::Gradient, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end
    n = subproblem_field_size(sp, field)
    coordsys = op.coordsys

    blocks = SparseMatrixCSC{ComplexF64, Int64}[]
    for coord in coordsys.coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        D = _subproblem_diff_matrix(sp, coord_name, 1, n)
        push!(blocks, D)
    end

    if isempty(blocks)
        return nothing
    end
    return vcat(blocks...)
end

"""
    subproblem_matrix(op::Divergence, sp; kwargs...)

Divergence: concatenates per-coordinate differentiation matrices horizontally.
Size: (Nz × ndim*Nz) for vector operand.
"""
function subproblem_matrix(op::Divergence, sp; kwargs...)
    # Get coordinate system from operand or from dist
    coordsys = _get_operand_coordsys(op.operand)
    if coordsys === nothing && hasfield(typeof(sp), :dist) && sp.dist !== nothing
        coordsys = sp.dist.coordsys
    end
    if coordsys === nothing
        return nothing
    end

    # For divergence, we need the scalar component size (Nz), not the full vector size
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end

    ndim = coordsys.dim
    field_size = subproblem_field_size(sp, field)
    input_size = try
        _subproblem_expr_dofs(sp, op.operand)
    catch
        field_size
    end

    # Vector divergence: div(u) maps (dim * Nz) -> Nz.
    # Tensor divergence: div(grad(u)) maps (dim * Nu) -> Nu where Nu = dim * Nz.
    if isa(field, VectorField) && input_size == field_size
        block_size = isempty(field.components) ? field_size : subproblem_field_size(sp, field.components[1])
    elseif input_size == ndim * field_size
        block_size = field_size
    elseif isa(field, VectorField) && !isempty(field.components)
        block_size = subproblem_field_size(sp, field.components[1])
    else
        block_size = field_size
    end

    blocks = SparseMatrixCSC{ComplexF64, Int64}[]
    for coord in coordsys.coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        D = _subproblem_diff_matrix(sp, coord_name, 1, block_size)
        push!(blocks, D)
    end

    @debug "Divergence subproblem_matrix: field_size=$field_size, input_size=$input_size, block_size=$block_size, ndim=$ndim, result=$(size(hcat(blocks...)))"

    if isempty(blocks)
        return nothing
    end
    return hcat(blocks...)
end

"""
    subproblem_matrix(op::Trace, sp; kwargs...)

Trace of a tensor: contracts diagonal components.
For 2D: trace_vec = [1, 0, 0, 1] (from ravel(eye(dim))).
Size: (Nz × dim²*Nz).
"""
function subproblem_matrix(op::Trace, sp; kwargs...)
    # Determine dimensionality from the operand or from dist
    coordsys = _get_operand_coordsys(op.operand)
    if coordsys === nothing && hasfield(typeof(sp), :dist) && sp.dist !== nothing
        coordsys = sp.dist.coordsys
    end
    if coordsys === nothing
        return nothing
    end
    dim = coordsys.dim

    # Get scalar block size from the operand
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return nothing
    end

    # For Trace, the operand is typically a tensor or a composed expression
    # that produces dim²*Nz rows. We need the scalar Nz.
    if isa(field, VectorField) && !isempty(field.components)
        Nz = subproblem_field_size(sp, field.components[1])
    elseif isa(field, ScalarField)
        Nz = subproblem_field_size(sp, field)
    else
        # TensorField or other
        Nz_total = subproblem_field_size(sp, field)
        Nz = div(Nz_total, dim * dim)
    end

    # Build trace vector: ravel(eye(dim)) — e.g., [1,0,0,1] for dim=2
    eye_flat = zeros(ComplexF64, dim * dim)
    for i in 1:dim
        eye_flat[(i-1)*dim + i] = ComplexF64(1.0)
    end

    # Trace matrix: kron(transpose(trace_vec), I_Nz)
    # This selects and sums diagonal blocks: (Nz × dim²*Nz)
    trace_vec = sparse(reshape(eye_flat, dim*dim, 1))
    return kron(sparse(transpose(trace_vec)), sparse(ComplexF64(1) * I, Nz, Nz))
end

"""
    subproblem_matrix(op::Lift, sp; kwargs...)

Per-subproblem lift matrix for tau method boundary conditions.

For a tau variable with `n_tau` DOFs per subproblem, the lift places each
tau DOF at a specific Chebyshev mode. The result is a `(n_comp * Nz) × n_tau`
matrix (block-diagonal of Nz×1 lift columns for each component).

The lift mode index follows the convention:
- n >= 0: sets mode n (0-indexed → Julia 1-indexed)
- n < 0: wraps around (n = -1 → last mode, n = -2 → second-to-last)
"""
function subproblem_matrix(op::Lift, sp; kwargs...)
    cheb_basis = _subproblem_cheb_basis(sp)
    if cheb_basis === nothing
        return nothing
    end
    Nz = cheb_basis.meta.size

    # Resolve lift mode index
    lift_mode = op.n
    if lift_mode < 0
        lift_mode = Nz + lift_mode  # -1 → Nz-1 (0-indexed)
    end
    lift_mode += 1  # 0-indexed → 1-indexed

    if lift_mode < 1 || lift_mode > Nz
        @warn "Lift mode $(op.n) resolved to $lift_mode, out of range [1, $Nz]" maxlog=1
        return nothing
    end

    # Build Nz×1 lift column: e_{lift_mode}
    e_lift = spzeros(ComplexF64, Nz, 1)
    e_lift[lift_mode, 1] = ComplexF64(1)

    # Determine number of components in the tau operand
    field = _resolve_operand_field(op.operand)
    if field === nothing
        return e_lift
    end

    n_comp = 1
    if isa(field, VectorField)
        n_comp = length(field.components)
    end

    if n_comp == 1
        return e_lift  # (Nz × 1)
    else
        # Block-diagonal: one lift column per component
        # Result: (n_comp * Nz) × n_comp
        blocks = [e_lift for _ in 1:n_comp]
        return blockdiag(blocks...)
    end
end

# ── Interpolate: evaluation at a point (BC constraints) ──
function subproblem_matrix(op::Interpolate, sp; kwargs...)
    cheb_basis = _subproblem_cheb_basis(sp)
    if cheb_basis === nothing
        return nothing
    end
    coord_name = isa(op.coord.name, Symbol) ? String(op.coord.name) : op.coord.name
    cheb_label = String(cheb_basis.meta.element_label)

    if coord_name != cheb_label
        # Interpolation in a Fourier (periodic) direction is mathematically
        # ill-posed: any point `x=x0` in a periodic domain is indistinguishable
        # from `x0 + n*L`, so `T(x=x0) = f(z)` does not define a unique BC.
        # Returning `nothing` here skips the matrix row entirely; without
        # this warning the BC would be silently dropped. Users should use
        # `integ()` for spatial-average constraints or set BCs only on
        # coupled (e.g. Chebyshev) directions.
        @warn "BC at a fixed point in Fourier direction `$coord_name` " *
              "is ill-posed for periodic domains and will be ignored. " *
              "Use `integ($coord_name, ...)` for spatial-average constraints, " *
              "or set BCs only on coupled directions (e.g. Chebyshev)." maxlog=2
        return nothing
    end

    Nz = cheb_basis.meta.size
    z0 = Float64(op.position)
    z_min, z_max = cheb_basis.meta.bounds[1], cheb_basis.meta.bounds[2]

    # Map physical position to canonical [-1, 1]
    xi = 2.0 * (z0 - z_min) / (z_max - z_min) - 1.0

    # Chebyshev evaluation: T_n(xi) row vector
    T = zeros(ComplexF64, 1, Nz)
    if Nz >= 1; T[1, 1] = 1.0; end
    if Nz >= 2; T[1, 2] = xi; end
    for n in 3:Nz
        T[1, n] = 2.0 * xi * T[1, n-1] - T[1, n-2]
    end

    # For vector operands, apply to each component: kron(I_ncomp, T_row)
    field = _resolve_operand_field(op.operand)
    n_comp = 1
    if field !== nothing && isa(field, VectorField)
        n_comp = length(field.components)
    end
    if n_comp > 1
        return kron(sparse(ComplexF64(1)*I, n_comp, n_comp), sparse(T))
    end
    return sparse(T)  # (1 × Nz) for scalar
end

# ── Integrate: integration over a coordinate (constraints like integ(p)=0) ──
function subproblem_matrix(op::Integrate, sp; kwargs...)
    field = _resolve_operand_field(op.operand)
    field === nothing && return nothing
    n = subproblem_field_size(sp, field)
    coords = isa(op.coord, Tuple) ? collect(op.coord) : [op.coord]

    # For multi-coordinate integration (integ over full domain):
    # Check if any separable (Fourier) coordinate gives zero for this mode.
    # If so, the entire integral is zero — return a zero row.
    for coord in coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        basis = _operand_basis_for_coord(op.operand, coord_name)
        if basis !== nothing && basis isa FourierBasis
            group_entry = _subproblem_group_index(sp, coord_name)
            if group_entry isa Integer && group_entry != 0
                # Non-DC Fourier mode: integral over x is zero
                # Return a zero row to keep the system square; valid mode
                # filtering will detect and remove it.
                return spzeros(ComplexF64, 1, n)
            end
        end
    end

    # All Fourier coordinates are DC (or none are Fourier) — compute integration weights.
    # For Chebyshev direction, build the integration weight row vector.
    cheb_basis = nothing
    for coord in coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        basis = _operand_basis_for_coord(op.operand, coord_name)
        if basis !== nothing && basis isa JacobiBasis
            cheb_basis = basis
            break
        end
    end

    if cheb_basis === nothing
        return sparse(ComplexF64(1) * I, 1, n)  # No Chebyshev: scalar identity
    end

    Nz = cheb_basis.meta.size
    z_min, z_max = cheb_basis.meta.bounds[1], cheb_basis.meta.bounds[2]
    L = z_max - z_min

    # Chebyshev integration weights: ∫_{-1}^{1} T_n(x) dx = 2/(1-n²) for even n
    w = zeros(ComplexF64, 1, Nz)
    for k in 0:(Nz-1)
        if k % 2 == 0
            w[1, k+1] = ComplexF64(L / 2.0 * 2.0 / (1.0 - k^2))
        end
    end

    # Scale by Fourier domain length for DC mode (x-integration gives Lx)
    for coord in coords
        coord_name = isa(coord.name, Symbol) ? String(coord.name) : coord.name
        basis = _operand_basis_for_coord(op.operand, coord_name)
        if basis !== nothing && basis isa FourierBasis
            Lx = basis.meta.bounds[2] - basis.meta.bounds[1]
            w .*= Lx
        end
    end

    # For vector operands
    n_comp = 1
    if field !== nothing && isa(field, VectorField)
        n_comp = length(field.components)
    end
    if n_comp > 1
        return kron(sparse(ComplexF64(1)*I, n_comp, n_comp), sparse(w))
    end
    return sparse(w)  # (1 × Nz)
end

# ============================================================================
# expression_matrices for operators not yet migrated to subproblem_matrix
# ============================================================================

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
