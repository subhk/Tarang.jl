"""
    Subproblem matrix helpers

This file contains helper functions for resolving bases, operands, and
mode-dependent metadata while building per-subproblem operator matrices.
"""

# ============================================================================
# Subproblem helper functions for compositional operator matrices
# ============================================================================

"""
    _subproblem_kx(sp, coord_name::AbstractString) -> Float64

Get the Fourier wavenumber for the coordinate `coord_name` in a subproblem.
Looks up that coordinate's mode index `sp.group_dict["n"*coord_name]` and the
matching FourierBasis, then returns k = n·2π/L. Returns 0 if `coord_name` is not
a Fourier axis of this subproblem.

NOTE: `coord_name` is REQUIRED. A problem with two Fourier directions (e.g. x and
y) builds a separate derivative matrix per axis; the previous no-argument version
always returned the FIRST Fourier wavenumber, so the y-derivative was wrongly
built with kx — yielding an implicit operator like −kx²−kx²+Dz² instead of
−(kx²+ky²)+Dz² for a 3D x,y-periodic + z-Chebyshev Laplacian.
"""
function _subproblem_kx(sp, coord_name::AbstractString)
    key = "n" * coord_name
    haskey(sp.group_dict, key) || return 0.0
    n = sp.group_dict[key]

    # Find the FourierBasis whose label matches coord_name, and its axis position
    # in the field's basis list (needed to decide rfft vs fft wavenumber layout).
    fourier_basis = nothing
    fb_axis = 0
    fb_bases = ()
    for var in sp.problem.variables
        hasfield(typeof(var), :bases) || continue
        for (axis, basis) in enumerate(var.bases)
            if basis !== nothing && isa(basis, FourierBasis) &&
               basis.meta.element_label == coord_name
                fourier_basis = basis
                fb_axis = axis
                fb_bases = var.bases
                break
            end
        end
        fourier_basis !== nothing && break
    end

    fourier_basis === nothing && return 0.0
    # `n` is the 0-based COEFFICIENT index along this axis; its physical wavenumber
    # depends on the storage layout. Only the first RealFourier axis is rfft (modes
    # 0..N/2, all non-negative); every other Fourier axis is full fft (index n>N/2
    # wraps to the NEGATIVE wavenumber n-N). The old `n·2π/L` was correct only for
    # the first axis — on a 2nd Fourier axis it gave e.g. ky=6 instead of -2 for
    # index 6 (N=8), so that mode's implicit operator used k²=36 instead of 4.
    # Reuse the same wavenumber arrays the spectral derivatives use, so the matrix
    # and the coefficient layout agree exactly.
    karr = (isa(fourier_basis, RealFourier) && _is_first_real_fourier_axis(fb_bases, fb_axis)) ?
           wavenumbers_rfft(fourier_basis) : wavenumbers_fft(fourier_basis)
    return (n + 1) <= length(karr) ? Float64(real(karr[n + 1])) : 0.0
end

"""
    _subproblem_cheb_basis(sp) -> Union{JacobiBasis, Nothing}

Get the ChebyshevT (or JacobiBasis) from a subproblem's problem variables.
Returns the first JacobiBasis found.
"""
function _subproblem_cheb_basis(sp)
    for var in sp.problem.variables
        if !hasfield(typeof(var), :bases)
            continue
        end
        for basis in var.bases
            if basis !== nothing && isa(basis, JacobiBasis)
                return basis
            end
        end
        # Also check components for VectorField
        if isa(var, VectorField)
            for comp in var.components
                for basis in comp.bases
                    if basis !== nothing && isa(basis, JacobiBasis)
                        return basis
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _subproblem_diff_matrix(sp, coord_name::String, order::Int, Nz::Int) -> SparseMatrixCSC{ComplexF64, Int64}

Get per-subproblem differentiation matrix for a coordinate.
- If coord is Chebyshev (JacobiBasis): returns ComplexF64.(differentiation_matrix(cheb_basis, order))
- If coord is Fourier: returns (im*kx)^order * I_Nz
"""
function _subproblem_diff_matrix(sp, coord_name::String, order::Int, Nz::Int)
    # Check if this coordinate corresponds to a Fourier basis
    for var in sp.problem.variables
        if !hasfield(typeof(var), :bases)
            continue
        end
        for basis in var.bases
            if basis === nothing
                continue
            end
            if basis.meta.element_label == coord_name
                if isa(basis, FourierBasis)
                    # Fourier coordinate: use (im*kx)^order * I
                    kx = _subproblem_kx(sp, coord_name)
                    coeff = (im * kx)^order
                    return sparse(ComplexF64(coeff) * I, Nz, Nz)
                elseif isa(basis, JacobiBasis)
                    # Chebyshev/Jacobi coordinate: use differentiation matrix
                    D = sparse(ComplexF64.(differentiation_matrix(basis, order)))
                    n_basis = size(D, 1)
                    if n_basis == Nz
                        return D
                    else
                        # Multi-component: block-diagonal kron(I_ncomp, D)
                        n_comp = div(Nz, n_basis)
                        return kron(sparse(ComplexF64(1) * I, n_comp, n_comp), D)
                    end
                end
            end
        end
        # Also check VectorField components
        if isa(var, VectorField)
            for comp in var.components
                for basis in comp.bases
                    if basis === nothing
                        continue
                    end
                    if basis.meta.element_label == coord_name
                        if isa(basis, FourierBasis)
                            kx = _subproblem_kx(sp, coord_name)
                            coeff = (im * kx)^order
                            return sparse(ComplexF64(coeff) * I, Nz, Nz)
                        elseif isa(basis, JacobiBasis)
                            D = sparse(ComplexF64.(differentiation_matrix(basis, order)))
                            n_basis = size(D, 1)
                            if n_basis == Nz
                                return D
                            else
                                n_comp = div(Nz, n_basis)
                                return kron(sparse(ComplexF64(1) * I, n_comp, n_comp), D)
                            end
                        end
                    end
                end
            end
        end
    end
    # Fallback: zero matrix
    return spzeros(ComplexF64, Nz, Nz)
end

"""
    _get_operand_coordsys(operand) -> Union{CoordinateSystem, Nothing}

Extract coordinate system from an operand (field or operator).
"""
function _get_operand_coordsys(operand)
    # VectorField has coordsys directly
    if isa(operand, VectorField)
        return operand.coordsys
    end
    # Gradient has coordsys
    if hasfield(typeof(operand), :coordsys)
        return operand.coordsys
    end
    # ScalarField: get from dist
    if hasfield(typeof(operand), :dist)
        dist = operand.dist
        if hasfield(typeof(dist), :coordsys)
            return dist.coordsys
        end
    end
    return nothing
end

"""
    _resolve_operand_field(operand) -> Union{ScalarField, VectorField, Nothing}

Walk the operator tree to find the leaf field.
"""
function _resolve_operand_field(operand)
    if isa(operand, ScalarField) || isa(operand, VectorField) || isa(operand, TensorField)
        return operand
    end
    if hasfield(typeof(operand), :operand)
        field = _resolve_operand_field(operand.operand)
        field !== nothing && return field
    end
    for field_name in (:left, :right, :base, :exponent)
        hasfield(typeof(operand), field_name) || continue
        field = _resolve_operand_field(getfield(operand, field_name))
        field !== nothing && return field
    end
    if hasfield(typeof(operand), :operands)
        ops = getfield(operand, :operands)
        if ops !== nothing
            for op in ops
                field = _resolve_operand_field(op)
                field !== nothing && return field
            end
        end
    end
    if isa(operand, Future)
        for op in future_args(operand)
            field = _resolve_operand_field(op)
            field !== nothing && return field
        end
    end
    return nothing
end

function _operand_basis_for_coord(operand, coord_name::String)
    field = _resolve_operand_field(operand)
    field === nothing && return nothing

    if hasfield(typeof(field), :bases)
        for basis in field.bases
            basis === nothing && continue
            label = isa(basis.meta.element_label, Symbol) ? String(basis.meta.element_label) : String(basis.meta.element_label)
            label == coord_name && return basis
        end
    end

    if isa(field, VectorField)
        for comp in field.components
            for basis in comp.bases
                basis === nothing && continue
                label = isa(basis.meta.element_label, Symbol) ? String(basis.meta.element_label) : String(basis.meta.element_label)
                label == coord_name && return basis
            end
        end
    end

    return nothing
end

function _subproblem_group_index(sp, coord_name::String)
    if !(hasfield(typeof(sp), :dist) && sp.dist !== nothing && hasfield(typeof(sp.dist), :coords))
        return nothing
    end

    for (axis, group_entry) in enumerate(sp.group)
        axis > length(sp.dist.coords) && continue
        dist_coord = sp.dist.coords[axis]
        label = isa(dist_coord.name, Symbol) ? String(dist_coord.name) : String(dist_coord.name)
        if label == coord_name
            return group_entry
        end
    end
    return nothing
end

function _integration_step_matrix(basis, coord_name::String, sp, nrows::Int)
    if basis isa FourierBasis
        group_entry = _subproblem_group_index(sp, coord_name)
        if !(group_entry isa Integer)
            return nothing
        elseif group_entry != 0
            # Non-DC mode: Fourier integral is zero. Return zero row(s) to keep
            # the matrix square — valid mode filtering will remove these rows
            # along with the corresponding tau variable columns.
            return spzeros(ComplexF64, 1, nrows)
        end

        L = basis.meta.bounds[2] - basis.meta.bounds[1]
        return sparse(ComplexF64(L) * I, nrows, nrows)
    elseif basis isa JacobiBasis
        Nz = basis.meta.size
        z_min, z_max = basis.meta.bounds[1], basis.meta.bounds[2]
        L = z_max - z_min

        w = zeros(ComplexF64, 1, Nz)
        for n in 0:(Nz-1)
            if n % 2 == 0
                w[1, n+1] = ComplexF64(L / 2.0 * 2.0 / (1.0 - n^2))
            end
        end

        if nrows == 0
            return spzeros(ComplexF64, 0, 0)
        elseif nrows == Nz
            return sparse(w)
        elseif nrows % Nz == 0
            n_comp = div(nrows, Nz)
            return kron(sparse(ComplexF64(1) * I, n_comp, n_comp), sparse(w))
        else
            return nothing
        end
    end

    return nothing
end
