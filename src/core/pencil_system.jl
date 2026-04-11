"""
Per-pencil matrix system (variable-centric) for IMEX time-stepping on mixed
Fourier-Chebyshev domains.

This module provides a per-pencil matrix system with a design that indexes by
`problem.variables` — which includes VectorFields as single entities — rather
than by a flattened list of scalar state fields.

For a 2D domain Fourier(x) × Chebyshev(z), each Fourier mode kx is decoupled.
This module stores one dense L/M matrix per Fourier mode (pencil).

The pencil vector for a given kx is laid out as:

    [ var1_comp1_cheb0 … var1_comp1_chebN-1,   # n_comp*n_cheb entries for var1
      var1_comp2_cheb0 … var1_comp2_chebN-1,   # (if VectorField with 2 comps)
      var2_cheb0       … var2_chebN-1,          # scalar var2: n_cheb entries
      tau_var_entry,                            # 1D/0D tau var: 1 entry
      … ]

See also: the original design was scalar-centric;
for backwards compatibility but not modified).
"""

using LinearAlgebra

export PencilSystem, vars_to_pencils, pencils_to_vars!

# ── Struct ────────────────────────────────────────────────────────────────────

"""
    PencilSystem

Stores per-pencil (per-Fourier-mode) dense L and M matrices for IMEX
time-stepping on a mixed Fourier × Chebyshev domain.

Indexed by `problem.variables` (VectorFields are single entities), unlike the
the new PencilSystem works with problem.variables directly.

At each implicit stage the IMEX solver forms and factorizes
    A[kx] = M[kx] + dt * a_ii * L[kx]
independently for each pencil index kx. Factorizations are cached so
repeated solves with the same (dt, a_ii) pair reuse the LU decomposition.

# Fields
- `L_pencils`    : `L_pencils[i]` is the (pencil_size × pencil_size) linear
                   operator matrix for Fourier mode `i`.
- `M_pencils`    : `M_pencils[i]` is the corresponding mass matrix.
- `lhs_cache`    : Cached LU factorizations keyed by `(pencil_idx, dt, a_ii)`.
- `n_pencils`    : Number of independent Fourier modes (N÷2+1 for RealFourier,
                   N for ComplexFourier).
- `pencil_size`  : Total degrees of freedom per pencil = `sum(var_dofs)`.
- `n_cheb`       : Number of Chebyshev modes.
- `n_vars`       : Number of problem variables (length of `problem.variables`).
- `var_names`    : Ordered variable names (for debugging / assembly).
- `var_dofs`     : Total DOFs per pencil per variable = `n_comp * comp_size`.
- `var_offsets`  : Cumulative 0-based offsets into the pencil vector.
- `var_n_comp`   : Number of components: 1 for ScalarField, 2+ for VectorField.
- `var_comp_size`: DOFs per component: `n_cheb` for 2D fields, 1 for tau/1D.
- `kx_values`    : Physical wavenumber `kx[i]` for pencil `i`.
- `cheb_label`   : The `element_label` string of the Chebyshev basis (e.g. "z").
"""
struct PencilSystem
    L_pencils :: Vector{Matrix{ComplexF64}}
    M_pencils :: Vector{Matrix{ComplexF64}}
    lhs_cache :: Dict{Tuple{Int,Float64,Float64},
                      LinearAlgebra.LU{ComplexF64,
                                       Matrix{ComplexF64},
                                       Vector{Int64}}}

    n_pencils    :: Int
    pencil_size  :: Int
    n_cheb       :: Int

    # Variable layout (indexed by problem.variables)
    n_vars       :: Int
    var_names    :: Vector{String}
    var_dofs     :: Vector{Int}   # total DOFs per pencil per variable
    var_offsets  :: Vector{Int}   # cumulative offset (0-based)
    var_n_comp   :: Vector{Int}   # components: 1 for scalar, 2+ for vector
    var_comp_size:: Vector{Int}   # DOFs per component: Nz for 2D, 1 for tau

    kx_values    :: Vector{Float64}
    cheb_label   :: String
end

# ── Helper: determine per-component pencil size for a variable ───────────────

"""
    _pencil_comp_size(var, cheb_label, n_cheb) -> Int

Return the number of Chebyshev-direction DOFs contributed per component by
`var` in the pencil vector.

- If the variable (or its first component) has no bases (`isempty`): returns 1.
- If any basis in the variable's bases tuple has `element_label == cheb_label`:
  returns `n_cheb` (full Chebyshev column).
- Otherwise (Fourier-only / tau variable): returns 1.
"""
function _pencil_comp_size(var, cheb_label::String, n_cheb::Int)
    field = isa(var, VectorField) ? var.components[1] :
            isa(var, TensorField) ? var.components[1, 1] : var
    isempty(field.bases) && return 1
    for basis in field.bases
        basis === nothing && continue
        String(basis.meta.element_label) == cheb_label && return n_cheb
    end
    return 1
end

# ── Component accessor ────────────────────────────────────────────────────────

"""
    _get_component(var::ScalarField, c::Int) -> ScalarField

Return the field itself (ScalarField has only one component).
"""
_get_component(var::ScalarField, c::Int) = var

"""
    _get_component(var::VectorField, c::Int) -> ScalarField

Return the `c`-th component of the VectorField.
"""
_get_component(var::VectorField, c::Int) = var.components[c]

# ── Constructor ───────────────────────────────────────────────────────────────

"""
    PencilSystem(problem, fourier_basis, cheb_basis)

Construct an empty (zero-filled) `PencilSystem` from a `Problem` and the two
spectral bases of the domain.

# Arguments
- `problem`       : Any `Problem` subtype whose `.variables` field is a
                    `Vector{Operand}` (IVP, LBVP, etc.).
- `fourier_basis` : A `FourierBasis` (`RealFourier` or `ComplexFourier`).
- `cheb_basis`    : A `JacobiBasis` (typically `ChebyshevT`).

# Pencil count
- `RealFourier` with N grid points → `N÷2 + 1` pencils (RFFT convention).
- `ComplexFourier` with N grid points → `N` pencils.

# Wavenumbers
`kx_values[i] = (i-1) * 2π / L` where `L` is the domain length in the Fourier
direction.

# Variable layout
Each entry in `problem.variables` is processed as:
- `ScalarField` with 2D bases (has Chebyshev): n_comp=1, comp_size=Nz
- `ScalarField` with 1D bases (Fourier only, tau): n_comp=1, comp_size=1
- `ScalarField` with empty bases (0D tau): n_comp=1, comp_size=1
- `VectorField` with 2D component bases: n_comp=length(components), comp_size=Nz
- `VectorField` with 1D component bases (tau): n_comp=length(components), comp_size=1
"""
function PencilSystem(problem, fourier_basis, cheb_basis)

    # ── Fourier direction ────────────────────────────────────────────────────
    N_fourier = fourier_basis.meta.size
    n_pencils = isa(fourier_basis, RealFourier) ? (N_fourier ÷ 2 + 1) : N_fourier

    L_domain  = fourier_basis.meta.bounds[2] - fourier_basis.meta.bounds[1]
    k0        = 2π / L_domain
    kx_values = [(i - 1) * k0 for i in 1:n_pencils]

    # ── Chebyshev direction ──────────────────────────────────────────────────
    n_cheb     = cheb_basis.meta.size
    cheb_label = String(cheb_basis.meta.element_label)

    # ── Variable layout ──────────────────────────────────────────────────────
    variables = problem.variables
    n_vars    = length(variables)

    var_names    = String[]
    var_n_comp   = Int[]
    var_comp_size= Int[]
    var_dofs     = Int[]
    var_offsets  = Int[]

    offset = 0
    for var in variables
        # Variable name
        name = hasfield(typeof(var), :name) ? var.name : "?"
        push!(var_names, name)

        # Number of components
        n_comp = isa(var, VectorField) ? length(var.components) : 1
        push!(var_n_comp, n_comp)

        # Chebyshev DOFs per component
        comp_size = _pencil_comp_size(var, cheb_label, n_cheb)
        push!(var_comp_size, comp_size)

        # Total DOFs for this variable
        dofs = n_comp * comp_size
        push!(var_dofs, dofs)

        # Cumulative offset (0-based)
        push!(var_offsets, offset)
        offset += dofs
    end

    pencil_size = offset

    # ── Allocate zero matrices ───────────────────────────────────────────────
    L_pencils = [zeros(ComplexF64, pencil_size, pencil_size) for _ in 1:n_pencils]
    M_pencils = [zeros(ComplexF64, pencil_size, pencil_size) for _ in 1:n_pencils]

    # ── LHS cache (empty at construction) ───────────────────────────────────
    lhs_cache = Dict{Tuple{Int,Float64,Float64},
                     LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}}()

    return PencilSystem(
        L_pencils, M_pencils, lhs_cache,
        n_pencils, pencil_size, n_cheb,
        n_vars, var_names, var_dofs, var_offsets, var_n_comp, var_comp_size,
        kx_values, cheb_label
    )
end

# ── Per-pencil LHS factorization ─────────────────────────────────────────────

"""
    get_pencil_lhs!(ps, kx_idx, dt, a_ii) -> LU factorization

Return (and cache) the LU factorization of the pencil LHS matrix

    A[kx] = M[kx] + dt * a_ii * L[kx]

for pencil index `kx_idx`. Repeated calls with the same `(kx_idx, dt, a_ii)`
triple reuse the cached factorization.

# Arguments
- `ps`     : `PencilSystem` holding the pencil matrices and LU cache.
- `kx_idx` : 1-based pencil (Fourier mode) index.
- `dt`     : Time-step size.
- `a_ii`   : Diagonal implicit coefficient from the IMEX Butcher tableau.
"""
function get_pencil_lhs!(ps::PencilSystem, kx_idx::Int,
                         dt::Float64, a_ii::Float64)
    key = (kx_idx, dt, a_ii)
    get!(ps.lhs_cache, key) do
        M = ps.M_pencils[kx_idx]
        L = ps.L_pencils[kx_idx]
        lu(M + dt * a_ii * L; check=false)
    end
end

# ── All-pencil solve ──────────────────────────────────────────────────────────

"""
    solve_pencil_system!(ps, rhs_pencils, dt, a_ii) -> Vector{Vector{ComplexF64}}

Solve the implicit pencil system for every Fourier mode and return the solution
pencils in-place (the same `rhs_pencils` vector is overwritten and returned).

For each pencil `k`:
- If `abs(a_ii) < 1e-14` (explicit stage, no implicit part): the solution is
  simply the RHS itself — the pencil vector is left unchanged.
- Otherwise: factorise `M[k] + dt * a_ii * L[k]` (cached) and solve by
  backslash.  Singular pencils (issuccess == false) are left unchanged.

# Arguments
- `ps`          : `PencilSystem`.
- `rhs_pencils` : `Vector{Vector{ComplexF64}}` of length `n_pencils`, each
                  entry being the pencil-sized RHS vector.  Modified in place.
- `dt`          : Time-step size.
- `a_ii`        : Diagonal implicit coefficient.

# Returns
The mutated `rhs_pencils` (now containing the solution).
"""
function solve_pencil_system!(ps          :: PencilSystem,
                              rhs_pencils :: Vector{Vector{ComplexF64}},
                              dt          :: Float64,
                              a_ii        :: Float64)
    if abs(a_ii) < 1e-14
        # Purely explicit stage — solution equals the RHS (no solve needed)
        return rhs_pencils
    end

    n_solved = 0
    n_singular = 0
    for kx_idx in 1:ps.n_pencils
        A_lu = get_pencil_lhs!(ps, kx_idx, dt, a_ii)
        if issuccess(A_lu)
            rhs_pencils[kx_idx] = A_lu \ rhs_pencils[kx_idx]
            n_solved += 1
        else
            n_singular += 1
        end
    end
    if n_singular > 0
        @warn "Pencil solve: $n_solved/$ps.n_pencils succeeded, $n_singular singular" maxlog=1
    end

    return rhs_pencils
end

# ── Cached Chebyshev differentiation matrix ───────────────────────────────────

"""
Cached Chebyshev differentiation matrices for the variable-centric system,
keyed by `(objectid(basis), order)`. Separate from `_cheb_diff_cache` used by
the previous scalar-based approach.
"""
const _ps_cheb_cache = Dict{Tuple{UInt64,Int}, Matrix{ComplexF64}}()

"""
    _ps_cheb_diff(basis::JacobiBasis, order::Int) -> Matrix{ComplexF64}

Return (and cache) the order-th Chebyshev differentiation matrix for `basis`,
promoted to `ComplexF64`. Uses `objectid(basis)` as the cache key so each
distinct basis object gets its own entry.
"""
function _ps_cheb_diff(basis::JacobiBasis, order::Int)
    key = (objectid(basis), order)
    get!(_ps_cheb_cache, key) do
        ComplexF64.(differentiation_matrix(basis, order))
    end
end

# ── Block builders ────────────────────────────────────────────────────────────

"""
    ps_laplacian_block(kx, cheb_basis, Nz) -> Matrix{ComplexF64}

Build the Nz×Nz Laplacian block for a single Fourier wavenumber `kx`:

    Δ_kx = -kx² * I + D²_z

where `D²_z` is the Chebyshev second-derivative matrix for `cheb_basis`.
"""
function ps_laplacian_block(kx::Float64, cheb_basis::JacobiBasis, Nz::Int)
    D2 = _ps_cheb_diff(cheb_basis, 2)
    nD = size(D2, 1)
    blk = -kx^2 * Matrix{ComplexF64}(I, Nz, Nz)
    n = min(nD, Nz)
    blk[1:n, 1:n] .+= D2[1:n, 1:n]
    return blk
end

"""
    ps_identity_block(n) -> Matrix{ComplexF64}

Return the n×n complex identity matrix.
"""
function ps_identity_block(n::Int)
    return Matrix{ComplexF64}(I, n, n)
end

"""
    ps_zero_block(n) -> Matrix{ComplexF64}

Return the n×n zero matrix.
"""
function ps_zero_block(n::Int)
    return zeros(ComplexF64, n, n)
end

"""
    ps_zero_block(m, n) -> Matrix{ComplexF64}

Return the m×n zero matrix (non-square variant).
"""
function ps_zero_block(m::Int, n::Int)
    return zeros(ComplexF64, m, n)
end

# ── Helper: get Chebyshev basis from a ScalarField ────────────────────────────

"""
    _get_cheb_basis(field::ScalarField, cheb_label::String) -> JacobiBasis or nothing

Search through `field.bases` for the basis whose `element_label` matches
`cheb_label` and return it. Returns `nothing` if not found.
"""
function _get_cheb_basis(field::ScalarField, cheb_label::String)
    for basis in field.bases
        basis === nothing && continue
        String(basis.meta.element_label) == cheb_label && return basis
    end
    return nothing
end

# ── Helper: matches_var_or_component ─────────────────────────────────────────

"""
    _ps_matches_var_or_component(operand, var_field) -> Bool

Check if `operand` matches `var_field` directly, OR if `operand` is a VectorField
whose components include `var_field`.

This handles the VectorField→ScalarField decomposition: an equation may have
`TimeDerivative(u)` where `u` is a VectorField, but pencil state uses the
individual scalar components `u_x`, `u_z`.
"""
function _ps_matches_var_or_component(operand, var_field)
    _operand_matches_variable(operand, var_field) && return true
    if isa(operand, VectorField)
        for comp in operand.components
            (comp === var_field || _operand_matches_variable(comp, var_field)) && return true
        end
    end
    return false
end

# ── Helper: ps_depends_on_var ────────────────────────────────────────────────

"""
    _ps_depends_on_var(expr, var_field) -> Bool

Recursively check whether `expr` depends on `var_field` anywhere in its tree.
Handles both Operator (`:operand`, `:left`, `:right`) and Future (via
`future_args`) hierarchies, as well as VectorField component expansion.

Used for require_linearity in Multiply nodes: when one factor does not
depend on `var_field`, it acts as a non-constant coefficient (NCC) and can
be folded into a scalar; when both factors depend on `var_field` the product
is nonlinear and returns `nothing`.
"""
function _ps_depends_on_var(expr, var_field)
    _operand_matches_variable(expr, var_field) && return true
    _ps_matches_var_or_component(expr, var_field) && return true
    for f in (:operand, :left, :right)
        hasfield(typeof(expr), f) || continue
        child = getfield(expr, f)
        child === nothing && continue
        _ps_depends_on_var(child, var_field) && return true
    end
    if isa(expr, Future)
        for arg in future_args(expr)
            _ps_depends_on_var(arg, var_field) && return true
        end
    end
    return false
end

# ── Core: scalar pencil block evaluator ──────────────────────────────────────

"""
    _scalar_pencil_block(expr, var_field, kx, ps, block_size)
        -> Union{Matrix{ComplexF64}, Nothing}

Return a `block_size × block_size` matrix representing the linear action of
`expr` on the scalar field `var_field` for Fourier wavenumber `kx`, or
`nothing` if `expr` has no linear dependence on `var_field`.

`var_field` must be a `ScalarField` (a single component of the variable being
assembled). `ps` is the `PencilSystem` whose `cheb_label` identifies the
Chebyshev coordinate.

The function handles both the Operator hierarchy (`AddOperator`, `Laplacian`,
etc.) and the Future hierarchy (`Add`, `Multiply`, etc.) that first-order
substitutions produce.

## Handled cases

1. **Direct match** — `_operand_matches_variable` or `_ps_matches_var_or_component`
   → identity block.
2. **Operator arithmetic** — `AddOperator`, `SubtractOperator`, `NegateOperator`,
   `MultiplyOperator` with const/param factor.
3. **Future arithmetic** — `Add`, `Subtract`, `Negate`, `Multiply` with
   separated scalar/field factors.
4. **Constants** — `Number`, `ZeroOperator`, `ConstantOperator` → `nothing`.
5. **TimeDerivative** — if operand matches `var_field` → identity.
6. **Laplacian** — if operand matches `var_field` → Laplacian block.
7. **Divergence(Gradient(var))** — div–grad pattern → Laplacian block.
8. **Differentiate** — Chebyshev direction → `_ps_cheb_diff`; Fourier direction
   → `(im*kx)^order * I`.
9. **Single-operand operators** (Lift, Skew, Curl, etc.) — recurse into operand.
10. **Fallback** → `nothing`.
"""
function _scalar_pencil_block(expr, var_field::ScalarField, kx::Float64,
                               ps::PencilSystem, block_size::Int)

    # ── 1. Direct variable match ─────────────────────────────────────────────
    if _operand_matches_variable(expr, var_field) ||
       _ps_matches_var_or_component(expr, var_field)
        return ps_identity_block(block_size)
    end

    # ── 2. Operator hierarchy arithmetic ────────────────────────────────────
    if isa(expr, AddOperator)
        l = _scalar_pencil_block(expr.left,  var_field, kx, ps, block_size)
        r = _scalar_pencil_block(expr.right, var_field, kx, ps, block_size)
        l === nothing && r === nothing && return nothing
        L = l !== nothing ? l : ps_zero_block(block_size)
        R = r !== nothing ? r : ps_zero_block(block_size)
        return L + R
    end

    if isa(expr, SubtractOperator)
        l = _scalar_pencil_block(expr.left,  var_field, kx, ps, block_size)
        r = _scalar_pencil_block(expr.right, var_field, kx, ps, block_size)
        l === nothing && r === nothing && return nothing
        L = l !== nothing ? l : ps_zero_block(block_size)
        R = r !== nothing ? r : ps_zero_block(block_size)
        return L - R
    end

    if isa(expr, NegateOperator)
        inner = _scalar_pencil_block(expr.operand, var_field, kx, ps, block_size)
        inner === nothing && return nothing
        return -inner
    end

    if isa(expr, MultiplyOperator)
        if _is_const_or_param(expr.left)
            coeff = ComplexF64(_extract_scalar(expr.left))
            inner = _scalar_pencil_block(expr.right, var_field, kx, ps, block_size)
            inner === nothing && return nothing
            return coeff * inner
        elseif _is_const_or_param(expr.right)
            coeff = ComplexF64(_extract_scalar(expr.right))
            inner = _scalar_pencil_block(expr.left, var_field, kx, ps, block_size)
            inner === nothing && return nothing
            return coeff * inner
        else
            # require_linearity: only one factor may depend on var_field
            left_dep  = _ps_depends_on_var(expr.left,  var_field)
            right_dep = _ps_depends_on_var(expr.right, var_field)
            if left_dep && !right_dep
                return _scalar_pencil_block(expr.left, var_field, kx, ps, block_size)
            elseif right_dep && !left_dep
                return _scalar_pencil_block(expr.right, var_field, kx, ps, block_size)
            else
                return nothing  # both depend (nonlinear) or neither
            end
        end
    end

    # ── 3. Future hierarchy arithmetic ──────────────────────────────────────
    if isa(expr, Future)
        args = future_args(expr)
        if isa(expr, Add)
            blks = [_scalar_pencil_block(a, var_field, kx, ps, block_size) for a in args]
            nonnothing = [b for b in blks if b !== nothing]
            isempty(nonnothing) && return nothing
            result = ps_zero_block(block_size)
            for b in nonnothing
                result = result + b
            end
            return result
        elseif isa(expr, Subtract) && length(args) >= 2
            l = _scalar_pencil_block(args[1], var_field, kx, ps, block_size)
            r = _scalar_pencil_block(args[2], var_field, kx, ps, block_size)
            l === nothing && r === nothing && return nothing
            L = l !== nothing ? l : ps_zero_block(block_size)
            R = r !== nothing ? r : ps_zero_block(block_size)
            return L - R
        elseif isa(expr, Negate) && length(args) >= 1
            inner = _scalar_pencil_block(args[1], var_field, kx, ps, block_size)
            inner === nothing && return nothing
            return -inner
        elseif isa(expr, Multiply) && length(args) >= 2
            scalars = filter(a -> _is_const_or_param(a) || isa(a, Number), args)
            fields  = filter(a -> !(_is_const_or_param(a) || isa(a, Number)), args)
            if length(fields) > 1
                # Multiple non-const factors: require_linearity
                dep   = filter(a -> _ps_depends_on_var(a, var_field), fields)
                indep = filter(a -> !_ps_depends_on_var(a, var_field), fields)
                length(dep) != 1 && return nothing  # nonlinear or no match
                inner = _scalar_pencil_block(dep[1], var_field, kx, ps, block_size)
                inner === nothing && return nothing
                scalar_val = isempty(scalars) ? ComplexF64(1) :
                             prod(ComplexF64(_extract_scalar(s)) for s in scalars)
                return scalar_val * inner
            end
            scalar_val = isempty(scalars) ? ComplexF64(1) :
                         prod(ComplexF64(_extract_scalar(s)) for s in scalars)
            if isempty(fields)
                return nothing  # pure scalar constant
            end
            inner = _scalar_pencil_block(fields[1], var_field, kx, ps, block_size)
            inner === nothing && return nothing
            return scalar_val * inner
        else
            return nothing  # unknown Future type (nonlinear/DotProduct/etc.)
        end
    end

    # ── 4. Constants and zero operators ─────────────────────────────────────
    if isa(expr, Number) || isa(expr, ZeroOperator) || isa(expr, ConstantOperator)
        return nothing
    end

    # ── 5. TimeDerivative: contributes to M matrix ──────────────────────────
    if isa(expr, TimeDerivative)
        if _ps_matches_var_or_component(expr.operand, var_field)
            return ps_identity_block(block_size)
        end
        return nothing
    end

    # ── 6. Laplacian ─────────────────────────────────────────────────────────
    if isa(expr, Laplacian)
        if _ps_matches_var_or_component(expr.operand, var_field)
            cheb_basis = _get_cheb_basis(var_field, ps.cheb_label)
            cheb_basis === nothing && return nothing
            return ps_laplacian_block(kx, cheb_basis, block_size)
        end
        return nothing
    end

    # ── 7. Divergence: check for div(grad(var)) → Laplacian ─────────────────
    if isa(expr, Divergence)
        # Collect all addends in the operand to look for grad(var)
        addends = _collect_all_addends(expr.operand)
        for a in addends
            # Strip any enclosing NegateOperator to inspect the inner term
            inner = isa(a, NegateOperator) ? a.operand : a
            sign  = isa(a, NegateOperator) ? -1 : 1
            if isa(inner, Gradient) && _operand_matches_variable(inner.operand, var_field)
                cheb_basis = _get_cheb_basis(var_field, ps.cheb_label)
                cheb_basis === nothing && return nothing
                blk = ps_laplacian_block(kx, cheb_basis, block_size)
                return sign == 1 ? blk : -blk
            end
        end
        return nothing
    end

    # ── 8. Gradient: scalar → vector — handled at assembly level ────────────
    if isa(expr, Gradient)
        return nothing
    end

    # ── 9. Differentiate ─────────────────────────────────────────────────────
    if isa(expr, Differentiate)
        _ps_matches_var_or_component(expr.operand, var_field) || return nothing
        coord_name = String(expr.coord.name)
        if coord_name == ps.cheb_label
            # d/dz in Chebyshev direction
            cheb_basis = _get_cheb_basis(var_field, ps.cheb_label)
            cheb_basis === nothing && return nothing
            return _ps_cheb_diff(cheb_basis, expr.order)
        else
            # Any non-Chebyshev coordinate is treated as the Fourier direction
            # d/dx in Fourier direction: (im*kx)^order * I
            return (im * kx)^expr.order * ps_identity_block(block_size)
        end
    end

    # ── 10. Single-operand operators: recurse into operand ──────────────────
    if hasfield(typeof(expr), :operand)
        return _scalar_pencil_block(expr.operand, var_field, kx, ps, block_size)
    end

    # ── 11. Fallback ─────────────────────────────────────────────────────────
    return nothing
end

# ── Matrix assembly ───────────────────────────────────────────────────────────

"""
    build_pencil_system_matrices!(ps, problem, cheb_basis)

Assemble L_pencils and M_pencils from problem.equation_data.

Equations map 1:1 to variables by index (equation i ↔ variable i).
Each equation produces var_dofs[i] rows; each variable produces var_dofs[j] columns.
For vector equations/variables, the block is split into component sub-blocks.
"""
function build_pencil_system_matrices!(ps::PencilSystem, problem::Problem, cheb_basis)
    vars = problem.variables
    eq_data = problem.equation_data

    for kx_idx in 1:ps.n_pencils
        kx = ps.kx_values[kx_idx]
        L = ps.L_pencils[kx_idx]
        M = ps.M_pencils[kx_idx]
        fill!(L, 0)
        fill!(M, 0)

        for eq_idx in 1:min(length(eq_data), ps.n_vars)
            eqd = eq_data[eq_idx]
            eq_var = vars[eq_idx]
            L_expr = get(eqd, "L", nothing)
            M_expr = get(eqd, "M", nothing)

            # One-time diagnostic for first pencil
            if kx_idx == 1 && eq_idx <= 3
                for vi in 1:min(ps.n_vars, 3)
                    vf = _get_component(vars[vi], 1)
                    if L_expr !== nothing
                        blk = _scalar_pencil_block(L_expr, vf, kx, ps, ps.var_comp_size[vi])
                        @info "  eq$eq_idx × var$(vi)($(vf.name)): L_block=$(blk !== nothing ? "$(size(blk)) nnz=$(count(!=(0), blk))" : "nothing")" maxlog=24
                    end
                    if M_expr !== nothing
                        blk = _scalar_pencil_block(M_expr, vf, kx, ps, ps.var_comp_size[vi])
                        @info "  eq$eq_idx × var$(vi)($(vf.name)): M_block=$(blk !== nothing ? "$(size(blk)) nnz=$(count(!=(0), blk))" : "nothing")" maxlog=24
                    end
                end
            end

            eq_offset = ps.var_offsets[eq_idx]
            eq_n_comp = ps.var_n_comp[eq_idx]
            eq_comp_sz = ps.var_comp_size[eq_idx]

            for var_idx in 1:ps.n_vars
                var = vars[var_idx]
                var_offset = ps.var_offsets[var_idx]
                var_n_comp = ps.var_n_comp[var_idx]
                var_comp_sz = ps.var_comp_size[var_idx]

                # Component-level sub-blocks
                for eq_c in 1:eq_n_comp
                    eq_field = _get_component(eq_var, eq_c)
                    row_start = eq_offset + (eq_c - 1) * eq_comp_sz + 1
                    row_end = row_start + eq_comp_sz - 1
                    row_end > ps.pencil_size && continue

                    for var_c in 1:var_n_comp
                        var_field = _get_component(var, var_c)
                        col_start = var_offset + (var_c - 1) * var_comp_sz + 1
                        col_end = col_start + var_comp_sz - 1
                        col_end > ps.pencil_size && continue

                        # Block size: use the LARGER of eq/var component sizes.
                        # _scalar_pencil_block builds a square block; we clip to
                        # the actual (eq_comp_sz × var_comp_sz) sub-block.
                        blk_size = max(eq_comp_sz, var_comp_sz)

                        # L block
                        if L_expr !== nothing
                            blk = _scalar_pencil_block(L_expr, var_field, kx, ps, blk_size)
                            if blk !== nothing
                                nr = min(size(blk, 1), eq_comp_sz)
                                nc = min(size(blk, 2), var_comp_sz)
                                L[row_start:row_start+nr-1, col_start:col_start+nc-1] .+= blk[1:nr, 1:nc]
                            end
                        end

                        # M block
                        if M_expr !== nothing
                            blk = _scalar_pencil_block(M_expr, var_field, kx, ps, blk_size)
                            if blk !== nothing
                                nr = min(size(blk, 1), eq_comp_sz)
                                nc = min(size(blk, 2), var_comp_sz)
                                M[row_start:row_start+nr-1, col_start:col_start+nc-1] .+= blk[1:nr, 1:nc]
                            end
                        end
                    end
                end
            end
        end
    end

    empty!(ps.lhs_cache)
end

# ── Variable ↔ pencil gather / scatter ───────────────────────────────────────

"""Scatter one ScalarField into pencil vectors at the given offset."""
function _ps_scatter_field!(pencils, field::ScalarField, comp_offset::Int,
                             comp_size::Int, ps::PencilSystem)
    if isempty(field.bases)
        cdata = get_coeff_data(field)
        gdata = get_grid_data(field)
        val = if cdata !== nothing && length(cdata) >= 1
            ComplexF64(cdata[1])
        elseif gdata !== nothing && length(gdata) >= 1
            ComplexF64(gdata[1])
        else
            ComplexF64(0)
        end
        pencils[1][comp_offset + 1] = val
    elseif comp_size == 1
        ensure_layout!(field, :c)
        cd = get_coeff_data(field)
        cd === nothing && return
        for kx in 1:min(ps.n_pencils, length(cd))
            pencils[kx][comp_offset + 1] = ComplexF64(cd[kx])
        end
    else
        ensure_layout!(field, :c)
        cd = get_coeff_data(field)
        cd === nothing && return
        c2d = reshape(vec(cd), ps.n_pencils, ps.n_cheb)
        for kx in 1:ps.n_pencils
            pencils[kx][comp_offset+1 : comp_offset+ps.n_cheb] .= c2d[kx, :]
        end
    end
end

function vars_to_pencils(variables, ps::PencilSystem)
    pencils = [zeros(ComplexF64, ps.pencil_size) for _ in 1:ps.n_pencils]
    for var_idx in 1:ps.n_vars
        var = variables[var_idx]
        for c in 1:ps.var_n_comp[var_idx]
            field = _get_component(var, c)
            comp_offset = ps.var_offsets[var_idx] + (c - 1) * ps.var_comp_size[var_idx]
            _ps_scatter_field!(pencils, field, comp_offset, ps.var_comp_size[var_idx], ps)
        end
    end
    return pencils
end

"""
    pencils_to_vars!(variables, pencils, ps)

Gather per-pencil solution vectors back into the spectral coefficient arrays
of each variable component. This is the inverse of `vars_to_pencils`.

For each variable `var_idx` and each component `c`:
- **0D tau**: the value at `pencils[1][comp_offset+1]` is written back to the
  field's coefficient or grid buffer.
- **1D Fourier-only**: `pencils[kx][comp_offset+1]` is written to `cd[kx]` for
  every `kx`.
- **2D Fourier×Chebyshev**: the `n_cheb`-element slice from each pencil is
  reassembled into the `(n_pencils × n_cheb)` coefficient array.

After writing, `field.current_layout` is set to `:c`.

# Arguments
- `variables` : `problem.variables` — same vector used in `vars_to_pencils`.
- `pencils`   : `Vector{Vector{ComplexF64}}` of length `ps.n_pencils`.
- `ps`        : `PencilSystem`.
"""
function pencils_to_vars!(variables, pencils::Vector{Vector{ComplexF64}},
                          ps::PencilSystem)

    for var_idx in 1:ps.n_vars
        var         = variables[var_idx]
        n_comp      = ps.var_n_comp[var_idx]
        comp_size   = ps.var_comp_size[var_idx]
        var_offset  = ps.var_offsets[var_idx]

        for c in 1:n_comp
            field       = _get_component(var, c)
            comp_offset = var_offset + (c - 1) * comp_size  # 0-based

            if isempty(field.bases)
                # ── 0D tau: write scalar back from pencil 1 ───────────────────
                val   = pencils[1][comp_offset + 1]
                cdata = get_coeff_data(field)
                gdata = get_grid_data(field)
                if cdata !== nothing && length(cdata) >= 1
                    cdata[1] = val
                elseif gdata !== nothing && length(gdata) >= 1
                    gdata[1] = real(val)
                end
                # If no buffer allocated, skip (tau value is ephemeral)

            elseif comp_size == 1
                # ── 1D Fourier-only: write one value per mode ─────────────────
                ensure_layout!(field, :c)
                cd = get_coeff_data(field)
                cd === nothing && continue
                for kx_idx in 1:ps.n_pencils
                    if kx_idx <= length(cd)
                        cd[kx_idx] = pencils[kx_idx][comp_offset + 1]
                    end
                end
                field.current_layout = :c

            else
                # ── 2D Fourier×Chebyshev: reassemble n_cheb columns ───────────
                ensure_layout!(field, :c)
                cd = get_coeff_data(field)
                cd === nothing && continue
                c2d = reshape(vec(cd), ps.n_pencils, ps.n_cheb)
                for kx_idx in 1:ps.n_pencils
                    c2d[kx_idx, :] .= pencils[kx_idx][comp_offset+1 : comp_offset+ps.n_cheb]
                end
                field.current_layout = :c
            end
        end
    end

    return nothing
end

# ── Flat scalar state → pencils (for evaluate_rhs output) ────────────────────

"""
    scalar_state_to_pencils(state_fields, variables, ps)

Scatter flat scalar state fields (from `collect_state_fields` / `evaluate_rhs`)
to pencil vectors, using the variable layout of `ps`.

`state_fields` is ordered: for each variable in `variables`, its scalar
components appear contiguously (ScalarField→1, VectorField→n_comp).
"""
function scalar_state_to_pencils(state_fields::Vector{<:ScalarField},
                                  variables, ps::PencilSystem)
    pencils = [zeros(ComplexF64, ps.pencil_size) for _ in 1:ps.n_pencils]
    sf_idx = 1  # index into state_fields

    for (var_idx, var) in enumerate(variables)
        n_comp = ps.var_n_comp[var_idx]
        comp_sz = ps.var_comp_size[var_idx]
        offset = ps.var_offsets[var_idx]

        for c in 1:n_comp
            sf_idx > length(state_fields) && break
            field = state_fields[sf_idx]
            comp_offset = offset + (c - 1) * comp_sz
            _ps_scatter_field!(pencils, field, comp_offset, comp_sz, ps)
            sf_idx += 1
        end
    end

    return pencils
end
