# Per-Pencil Subproblem Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single global sparse L/M matrix with per-pencil (per-Fourier-mode) dense matrices, enabling correct implicit treatment of diffusion in mixed Fourier-Chebyshev domains (Rayleigh-Benard, channel flows).

**Architecture:** For a 2D domain with Fourier(x) × Chebyshev(z), each Fourier mode kx has an independent dense matrix of size (n_vars × N_cheb) × (n_vars × N_cheb). The IMEX solver factorizes these small matrices independently — 129 pencils of 512×512 instead of one 33799×33799 sparse matrix. This is how Dedalus handles mixed-basis problems.

**Tech Stack:** Julia, SparseArrays, LinearAlgebra (dense LU), existing Tarang subsystem/distributor infrastructure.

---

## Background

### Why per-pencil?

For a Fourier(Nx) × Chebyshev(Nz) domain, the Laplacian Δ = ∂²/∂x² + ∂²/∂z² is block-diagonal in Fourier space:

```
For each kx mode:  Δ_kx = -kx² I_Nz + D²_z    (Nz × Nz dense matrix)
```

The global problem decouples into independent "pencils" — one per Fourier mode. Each pencil is a small dense system coupling all variables at that wavenumber. The IMEX solve `(M + dt·a·L) X = RHS` becomes many independent small dense solves, which are:
- **Exact** — real spectral operators, not markers
- **Fast** — O(Nz³) per pencil, O(Nkx · Nz³) total
- **Cacheable** — LHS factorization reused across timesteps

### What exists in Tarang

| Component | Status | Location |
|-----------|--------|----------|
| Subsystem grouping | Ready | `src/core/subsystems.jl` — groups by separable dimensions |
| Subproblem struct | Ready | `src/core/subsystems.jl:604-642` — holds matrices, preconditioners |
| Gather/scatter | Ready | `src/core/subsystems.jl:558-589` — field ↔ vector conversion |
| Expression matrices | Ready | `src/core/operators/matrices.jl` — per-operator matrix builders |
| Fourier wavenumbers | Ready | `src/core/basis.jl:673+` — wavenumber arrays |
| Chebyshev D matrix | Ready | `src/core/basis.jl:2040+` — differentiation matrices |
| PencilLinearOperator | Partial | Diagonal-only (Fourier), not full coupled system |
| Per-pencil matrix assembly | Missing | Need loop over kx building dense coupled matrices |
| Per-pencil IMEX solve | Missing | Need pencil-aware step function |

### Scope

This plan covers 2D domains (1 Fourier + 1 Chebyshev). Extension to 3D (2 Fourier + 1 Chebyshev) follows the same pattern with pencils indexed by (kx, ky).

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/core/pencil_matrices.jl` | **Create** | PencilMatrixSystem struct, per-pencil assembly, factorization cache |
| `src/core/timesteppers/step_pencil_imex.jl` | **Modify** | Per-pencil IMEX RK step function |
| `src/core/solvers/solver_types.jl` | **Modify** | Build pencil matrices during solver construction |
| `src/core/operators/matrices.jl` | **Modify** | Add `pencil_operator_matrix` for per-kx operator blocks |
| `test/test_pencil_matrices.jl` | **Create** | Unit tests for pencil matrix assembly and solve |
| `test/test_pencil_imex.jl` | **Create** | Integration test: 1D heat equation with Chebyshev + tau |

---

## Task 1: PencilMatrixSystem Data Structure

**Files:**
- Create: `src/core/pencil_matrices.jl`
- Modify: `src/Tarang.jl` (include the new file)

The core data structure: a collection of dense matrices indexed by Fourier mode.

- [ ] **Step 1: Define PencilMatrixSystem struct**

```julia
# src/core/pencil_matrices.jl

"""
Per-pencil (per-Fourier-mode) matrix system for implicit IMEX solves.

For a domain with Fourier(x) × Chebyshev(z), each Fourier mode kx
has independent dense matrices L_kx and M_kx of size
(n_scalar_vars × Nz) × (n_scalar_vars × Nz).

The IMEX solve (M + dt·a·L) X = RHS becomes independent dense
solves per pencil — exact spectral operators, O(Nz³) per pencil.
"""
mutable struct PencilMatrixSystem
    # Per-pencil dense matrices: pencil_index → Matrix
    L_pencils::Vector{Matrix{ComplexF64}}   # L[kx] for each Fourier mode
    M_pencils::Vector{Matrix{ComplexF64}}   # M[kx] for each Fourier mode

    # Cached LHS factorizations: (pencil_idx, dt, a_ii) → Factorization
    lhs_cache::Dict{Tuple{Int, Float64, Float64}, LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}}

    # Metadata
    n_pencils::Int              # Number of Fourier modes (Nkx)
    pencil_size::Int            # DOFs per pencil (n_scalar_vars × Nz)
    fourier_dim::Int            # Which dimension is Fourier (1-based)
    chebyshev_dim::Int          # Which dimension is Chebyshev
    n_cheb::Int                 # Chebyshev modes (Nz)
    n_scalar_fields::Int        # Number of scalar state fields
    field_names::Vector{String} # Ordered field names for debugging

    # Wavenumber array
    kx_values::Vector{Float64}  # Physical wavenumber for each pencil

    # Field-to-pencil index mapping
    # Maps (field_index, cheb_mode) → row/col offset in pencil matrix
    field_offsets::Vector{Int}   # field_offsets[i] = start of field i in pencil
end
```

- [ ] **Step 2: Add constructor**

```julia
function PencilMatrixSystem(state::Vector{<:ScalarField}, fourier_basis, chebyshev_basis)
    # Determine pencil count from Fourier basis
    N_fourier = fourier_basis.meta.size
    n_pencils = isa(fourier_basis, RealFourier) ? div(N_fourier, 2) + 1 : N_fourier
    n_cheb = chebyshev_basis.meta.size
    n_fields = length(state)
    pencil_size = n_fields * n_cheb

    # Build wavenumber array
    L = fourier_basis.meta.bounds[2] - fourier_basis.meta.bounds[1]
    k0 = 2π / L
    kx_values = [(i-1) * k0 for i in 1:n_pencils]

    # Field offsets: field i starts at (i-1)*n_cheb in the pencil vector
    field_offsets = [(i-1) * n_cheb for i in 1:n_fields]
    field_names = [f.name for f in state]

    # Allocate empty matrices
    L_pencils = [zeros(ComplexF64, pencil_size, pencil_size) for _ in 1:n_pencils]
    M_pencils = [zeros(ComplexF64, pencil_size, pencil_size) for _ in 1:n_pencils]

    return PencilMatrixSystem(
        L_pencils, M_pencils,
        Dict{Tuple{Int,Float64,Float64}, LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64},Vector{Int64}}}(),
        n_pencils, pencil_size, 1, 2, n_cheb, n_fields, field_names,
        kx_values, field_offsets
    )
end
```

- [ ] **Step 3: Include in Tarang.jl**

Add `include("core/pencil_matrices.jl")` after `include("core/subsystems.jl")` in `src/Tarang.jl`.

- [ ] **Step 4: Commit**

```bash
git add src/core/pencil_matrices.jl src/Tarang.jl
git commit -m "feat: add PencilMatrixSystem struct for per-pencil IMEX"
```

---

## Task 2: Per-Pencil Operator Matrix Builders

**Files:**
- Modify: `src/core/pencil_matrices.jl`

Build the actual spectral operator matrix for a single operator at a single wavenumber kx. These are small dense blocks (Nz × Nz for scalar-scalar coupling).

- [ ] **Step 1: Chebyshev differentiation block**

```julia
"""
Build Nz×Nz Chebyshev differentiation matrix D^order for the z-basis.
Cached per (basis, order) to avoid recomputation.
"""
const _cheb_diff_cache = Dict{Tuple{UInt64, Int}, Matrix{ComplexF64}}()

function _chebyshev_diff_block(basis::JacobiBasis, order::Int)
    key = (objectid(basis), order)
    get!(_cheb_diff_cache, key) do
        ComplexF64.(differentiation_matrix(basis, order))
    end
end
```

- [ ] **Step 2: Per-pencil Laplacian block (scalar field)**

```julia
"""
Build Nz×Nz Laplacian block for pencil kx_idx:
    Δ_kx = -kx² I + D²_z
"""
function pencil_laplacian_block(kx::Float64, cheb_basis::JacobiBasis)
    Nz = cheb_basis.meta.size
    D2 = _chebyshev_diff_block(cheb_basis, 2)
    return -kx^2 * I(Nz) + D2
end
```

- [ ] **Step 3: Per-pencil differentiation blocks**

```julia
"""d/dx block: diagonal ik_x for Fourier dimension."""
function pencil_dx_block(kx::Float64, Nz::Int)
    return (im * kx) * Matrix{ComplexF64}(I, Nz, Nz)
end

"""d/dz block: Chebyshev D matrix."""
function pencil_dz_block(cheb_basis::JacobiBasis)
    return ComplexF64.(_chebyshev_diff_block(cheb_basis, 1))
end

"""d²/dz² block: Chebyshev D² matrix."""
function pencil_d2z_block(cheb_basis::JacobiBasis)
    return ComplexF64.(_chebyshev_diff_block(cheb_basis, 2))
end
```

- [ ] **Step 4: Identity and zero blocks**

```julia
pencil_identity_block(Nz::Int) = Matrix{ComplexF64}(I, Nz, Nz)
pencil_zero_block(Nz::Int) = zeros(ComplexF64, Nz, Nz)
```

- [ ] **Step 5: Commit**

```bash
git add src/core/pencil_matrices.jl
git commit -m "feat: add per-pencil spectral operator block builders"
```

---

## Task 3: Per-Pencil Matrix Assembly from Equations

**Files:**
- Modify: `src/core/pencil_matrices.jl`

The key function: iterate over equations and variables, building the coupled L_kx and M_kx matrices for each pencil.

- [ ] **Step 1: Expression-to-pencil-block dispatcher**

```julia
"""
Build the Nz×Nz pencil block for operator `expr` acting on variable `var`
at wavenumber kx. Returns a dense matrix or nothing.

This replaces build_expression_matrix_block for per-pencil assembly —
uses actual spectral operators instead of identity markers.
"""
function pencil_expression_block(expr, var, kx::Float64, cheb_basis, Nz::Int)
    # Direct variable match → identity
    _operand_matches_variable(expr, var) && return pencil_identity_block(Nz)

    # Arithmetic
    if isa(expr, AddOperator)
        l = pencil_expression_block(expr.left, var, kx, cheb_basis, Nz)
        r = pencil_expression_block(expr.right, var, kx, cheb_basis, Nz)
        return (l === nothing ? pencil_zero_block(Nz) : l) +
               (r === nothing ? pencil_zero_block(Nz) : r)
    end
    if isa(expr, SubtractOperator)
        l = pencil_expression_block(expr.left, var, kx, cheb_basis, Nz)
        r = pencil_expression_block(expr.right, var, kx, cheb_basis, Nz)
        return (l === nothing ? pencil_zero_block(Nz) : l) -
               (r === nothing ? pencil_zero_block(Nz) : r)
    end
    isa(expr, NegateOperator) && begin
        inner = pencil_expression_block(expr.operand, var, kx, cheb_basis, Nz)
        return inner === nothing ? nothing : -inner
    end
    if isa(expr, MultiplyOperator)
        if _is_const_or_param(expr.left)
            coeff = ComplexF64(_extract_scalar(expr.left))
            inner = pencil_expression_block(expr.right, var, kx, cheb_basis, Nz)
            return inner === nothing ? nothing : coeff * inner
        elseif _is_const_or_param(expr.right)
            coeff = ComplexF64(_extract_scalar(expr.right))
            inner = pencil_expression_block(expr.left, var, kx, cheb_basis, Nz)
            return inner === nothing ? nothing : coeff * inner
        end
        return nothing  # nonlinear
    end

    # Future types (from Julia-level expression building)
    if isa(expr, Future)
        args = future_args(expr)
        if isa(expr, Add)
            result = pencil_zero_block(Nz)
            for arg in args
                blk = pencil_expression_block(arg, var, kx, cheb_basis, Nz)
                blk !== nothing && (result += blk)
            end
            return result
        elseif isa(expr, Multiply) && length(args) >= 2
            scalars = filter(a -> _is_const_or_param(a) || isa(a, Number), args)
            fields = filter(a -> !(_is_const_or_param(a) || isa(a, Number)), args)
            length(fields) > 1 && return nothing
            coeff = isempty(scalars) ? ComplexF64(1) :
                    prod(ComplexF64(_extract_scalar(s)) for s in scalars)
            isempty(fields) && return nothing
            inner = pencil_expression_block(fields[1], var, kx, cheb_basis, Nz)
            return inner === nothing ? nothing : coeff * inner
        elseif isa(expr, Negate) && !isempty(args)
            inner = pencil_expression_block(args[1], var, kx, cheb_basis, Nz)
            return inner === nothing ? nothing : -inner
        elseif isa(expr, Subtract) && length(args) >= 2
            l = pencil_expression_block(args[1], var, kx, cheb_basis, Nz)
            r = pencil_expression_block(args[2], var, kx, cheb_basis, Nz)
            return (l === nothing ? pencil_zero_block(Nz) : l) -
                   (r === nothing ? pencil_zero_block(Nz) : r)
        end
        return nothing
    end

    # Constants
    (isa(expr, Number) || isa(expr, ZeroOperator) || isa(expr, ConstantOperator)) && return nothing

    # Spectral operators
    if isa(expr, Laplacian)
        if _operand_matches_variable(expr.operand, var)
            return pencil_laplacian_block(kx, cheb_basis)
        end
        return nothing
    end
    if isa(expr, TimeDerivative)
        if _operand_matches_variable(expr.operand, var)
            return pencil_identity_block(Nz)
        end
        return nothing
    end
    if isa(expr, Gradient)
        # grad(scalar) at pencil kx: [ik_x, D_z] — but this produces a VECTOR
        # For the pencil block mapping scalar→vector, return nothing
        # (handled at the equation assembly level where grad produces 2 blocks)
        return nothing
    end
    if isa(expr, Divergence)
        # div(grad(f)) → Laplacian — check for first-order pattern
        inner = expr.operand
        addends = _collect_all_addends(inner)
        for a in addends
            actual = isa(a, NegateOperator) ? a.operand : a
            if isa(actual, Gradient) && _operand_matches_variable(actual.operand, var)
                return pencil_laplacian_block(kx, cheb_basis)
            end
        end
        # Plain div(vector_var): handled at assembly level (maps vector→scalar)
        return nothing
    end

    # Single-operand operators: recurse
    if hasfield(typeof(expr), :operand)
        return pencil_expression_block(expr.operand, var, kx, cheb_basis, Nz)
    end

    return nothing
end
```

- [ ] **Step 2: Full pencil matrix assembly**

```julia
"""
Assemble L_kx and M_kx for all pencils from the problem's equation data.

For each pencil kx, the matrix is (n_fields × Nz) × (n_fields × Nz).
Block (i, j) is the Nz×Nz coupling from variable j to equation i
at wavenumber kx.
"""
function build_pencil_matrices!(pms::PencilMatrixSystem, problem::Problem,
                                 state::Vector{<:ScalarField}, cheb_basis)
    Nz = pms.n_cheb
    n_fields = pms.n_scalar_fields
    eq_data = problem.equation_data

    # Map variables to state field indices
    var_to_idx = Dict{String, Int}()
    for (i, f) in enumerate(state)
        var_to_idx[f.name] = i
    end

    for kx_idx in 1:pms.n_pencils
        kx = pms.kx_values[kx_idx]
        L = pms.L_pencils[kx_idx]
        M = pms.M_pencils[kx_idx]

        fill!(L, 0)
        fill!(M, 0)

        # Iterate over equations (rows) and variables (columns)
        eq_offset = 0
        for (eq_idx, eqd) in enumerate(eq_data)
            # Determine how many rows this equation contributes
            # For scalar equations: Nz rows. For vector: 2*Nz. For tau/constraint: 1.
            eq_rows = _pencil_equation_rows(eqd, Nz, n_fields)
            eq_rows == 0 && continue

            L_expr = get(eqd, "L", nothing)
            M_expr = get(eqd, "M", nothing)

            for (var_idx, var) in enumerate(problem.variables)
                # For each scalar component of the variable
                comp_fields = isa(var, ScalarField) ? [var] :
                              isa(var, VectorField) ? collect(var.components) : ScalarField[]

                for (comp_idx, comp) in enumerate(comp_fields)
                    state_idx = get(var_to_idx, comp.name, 0)
                    state_idx == 0 && continue

                    col_offset = pms.field_offsets[state_idx]
                    col_range = (col_offset+1):(col_offset+Nz)

                    # Skip if column range exceeds matrix
                    maximum(col_range) > pms.pencil_size && continue

                    # L block
                    if L_expr !== nothing
                        blk = pencil_expression_block(L_expr, comp, kx, cheb_basis, Nz)
                        if blk !== nothing
                            row_range = (eq_offset+1):(eq_offset+min(eq_rows, Nz))
                            nr = length(row_range)
                            nc = min(Nz, size(blk, 2))
                            L[row_range[1:nr], col_range[1:nc]] .+= blk[1:nr, 1:nc]
                        end
                    end

                    # M block
                    if M_expr !== nothing
                        blk = pencil_expression_block(M_expr, comp, kx, cheb_basis, Nz)
                        if blk !== nothing
                            row_range = (eq_offset+1):(eq_offset+min(eq_rows, Nz))
                            nr = length(row_range)
                            nc = min(Nz, size(blk, 2))
                            M[row_range[1:nr], col_range[1:nc]] .+= blk[1:nr, 1:nc]
                        end
                    end
                end
            end

            eq_offset += eq_rows
        end
    end

    # Clear LHS cache (matrices changed)
    empty!(pms.lhs_cache)
end
```

- [ ] **Step 3: Commit**

```bash
git add src/core/pencil_matrices.jl
git commit -m "feat: per-pencil matrix assembly from equation data"
```

---

## Task 4: Per-Pencil LHS Factorization and Solve

**Files:**
- Modify: `src/core/pencil_matrices.jl`

Cache LU factorizations per pencil and provide a solve interface.

- [ ] **Step 1: Factorize per pencil**

```julia
"""
Get or compute the LHS factorization for pencil kx_idx:
    LHS = M_kx + dt * a_ii * L_kx
"""
function get_pencil_lhs!(pms::PencilMatrixSystem, kx_idx::Int, dt::Float64, a_ii::Float64)
    key = (kx_idx, dt, a_ii)
    get!(pms.lhs_cache, key) do
        M = pms.M_pencils[kx_idx]
        L = pms.L_pencils[kx_idx]
        lu(M + dt * a_ii * L)
    end
end

"""
Solve the pencil system for all pencils simultaneously.

Input:  rhs_pencils[kx_idx] = Vector{ComplexF64} of length pencil_size
Output: sol_pencils[kx_idx] = solution vector
"""
function solve_pencil_system!(pms::PencilMatrixSystem, rhs_pencils::Vector{Vector{ComplexF64}},
                               dt::Float64, a_ii::Float64)
    sol_pencils = similar(rhs_pencils)
    for kx_idx in 1:pms.n_pencils
        if abs(a_ii) < 1e-14
            # No implicit part — identity solve
            sol_pencils[kx_idx] = copy(rhs_pencils[kx_idx])
        else
            lhs = get_pencil_lhs!(pms, kx_idx, dt, a_ii)
            sol_pencils[kx_idx] = lhs \ rhs_pencils[kx_idx]
        end
    end
    return sol_pencils
end
```

- [ ] **Step 2: Gather/scatter between state vector and pencil vectors**

```julia
"""
Scatter state fields into per-pencil vectors.
State vector layout: field1[kx1_z1..kx1_zN, kx2_z1..], field2[...], ...
Pencil vector layout: field1_z1..field1_zN, field2_z1..field2_zN, ...
"""
function state_to_pencils(state::Vector{<:ScalarField}, pms::PencilMatrixSystem)
    pencils = [zeros(ComplexF64, pms.pencil_size) for _ in 1:pms.n_pencils]

    for (fi, field) in enumerate(state)
        ensure_layout!(field, :c)
        cdata = get_coeff_data(field)
        cdata === nothing && continue

        offset = pms.field_offsets[fi]
        if isempty(field.bases)
            # Tau variable: single value goes to all pencils at offset+1
            val = length(cdata) >= 1 ? cdata[1] : ComplexF64(0)
            for kx_idx in 1:pms.n_pencils
                pencils[kx_idx][offset+1] = val
            end
        else
            # Regular field: coeff data is (Nkx, Nz) — distribute kx slices
            data_2d = reshape(vec(cdata), pms.n_pencils, pms.n_cheb)
            for kx_idx in 1:pms.n_pencils
                pencils[kx_idx][(offset+1):(offset+pms.n_cheb)] .= data_2d[kx_idx, :]
            end
        end
    end

    return pencils
end

"""
Gather per-pencil vectors back into state fields.
"""
function pencils_to_state!(state::Vector{<:ScalarField}, pencils::Vector{Vector{ComplexF64}},
                            pms::PencilMatrixSystem)
    for (fi, field) in enumerate(state)
        offset = pms.field_offsets[fi]

        if isempty(field.bases)
            # Tau: take value from pencil 1 (k=0)
            cdata = get_coeff_data(field)
            if cdata !== nothing && length(cdata) >= 1
                cdata[1] = pencils[1][offset+1]
            end
        else
            ensure_layout!(field, :c)
            cdata = get_coeff_data(field)
            cdata === nothing && continue
            data_2d = reshape(vec(cdata), pms.n_pencils, pms.n_cheb)
            for kx_idx in 1:pms.n_pencils
                data_2d[kx_idx, :] .= pencils[kx_idx][(offset+1):(offset+pms.n_cheb)]
            end
        end

        field.current_layout = :c
    end
end
```

- [ ] **Step 3: Commit**

```bash
git add src/core/pencil_matrices.jl
git commit -m "feat: per-pencil LHS factorization and gather/scatter"
```

---

## Task 5: Per-Pencil IMEX RK Step Function

**Files:**
- Modify: `src/core/timesteppers/step_rk.jl` or create `src/core/timesteppers/step_pencil_rk.jl`
- Modify: `src/core/timesteppers/dispatch.jl`

Replace the global sparse solve with per-pencil dense solves.

- [ ] **Step 1: Per-pencil IMEX RK step**

```julia
"""
IMEX RK step using per-pencil dense solves.

Same algorithm as step_rk_imex! but operates on pencil-decomposed matrices:
- Implicit: (M_kx + dt*a*L_kx) per Fourier mode
- Explicit: evaluate_rhs (full nonlinear evaluation)
"""
function step_pencil_rk_imex!(state::TimestepperState, solver::InitialValueSolver,
                               pms::PencilMatrixSystem)
    ts = state.timestepper
    current_state = state.history[end]
    dt = state.dt
    t = solver.sim_time
    stages = ts.stages
    A_exp = ts.A_explicit
    A_imp = ts.A_implicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit
    c = ts.c_explicit

    # Scatter state to pencils
    X_n_pencils = state_to_pencils(current_state, pms)

    # Storage for stage results
    F_exp_pencils = Vector{Vector{Vector{ComplexF64}}}(undef, stages)
    X_s_pencils = X_n_pencils  # Will be updated per stage

    for s in 1:stages
        state.current_substep = s

        # Build RHS pencils: M_kx * X_n + dt * Σ contributions
        rhs_pencils = [pms.M_pencils[kx] * X_n_pencils[kx] for kx in 1:pms.n_pencils]

        for j in 1:(s-1)
            a_exp_sj = dt * A_exp[s, j]
            a_imp_sj = dt * A_imp[s, j]
            for kx in 1:pms.n_pencils
                if abs(a_exp_sj) > 1e-14
                    rhs_pencils[kx] .+= a_exp_sj .* F_exp_pencils[j][kx]
                end
                if abs(a_imp_sj) > 1e-14
                    rhs_pencils[kx] .-= a_imp_sj .* (pms.L_pencils[kx] * X_s_pencils[kx])
                end
            end
        end

        # Implicit solve per pencil
        a_ii = A_imp[s, s]
        X_s_pencils = solve_pencil_system!(pms, rhs_pencils, dt, a_ii)

        # Scatter solution to fields for nonlinear evaluation
        temp_state = copy_state(current_state)
        pencils_to_state!(temp_state, X_s_pencils, pms)

        # Evaluate explicit RHS
        F_exp_fields = evaluate_rhs(solver, temp_state, t + c[s] * dt)
        F_exp_pencils[s] = state_to_pencils(F_exp_fields, pms)
    end

    # Final update using b weights
    X_new_pencils = [pms.M_pencils[kx] * X_n_pencils[kx] for kx in 1:pms.n_pencils]
    for s in 1:stages
        be = dt * b_exp[s]
        bi = dt * b_imp[s]
        for kx in 1:pms.n_pencils
            if abs(be) > 1e-14
                X_new_pencils[kx] .+= be .* F_exp_pencils[s][kx]
            end
        end
    end
    # Apply M^{-1} per pencil (or use stiffly-accurate property)
    # For stiffly accurate methods (b_imp = A_imp[end,:]), use last stage value
    X_new_pencils = X_s_pencils  # RK222 is stiffly accurate

    new_state = copy_state(current_state)
    pencils_to_state!(new_state, X_new_pencils, pms)
    _push_trim!(state.history, new_state, 1)
end
```

- [ ] **Step 2: Dispatch integration**

In `src/core/timesteppers/dispatch.jl` or `step_rk.jl`, add detection:

```julia
# In step_rk_imex!, before the sparse matrix path:
# Check if pencil matrices are available
if haskey(solver.problem.parameters, "pencil_system")
    pms = solver.problem.parameters["pencil_system"]::PencilMatrixSystem
    step_pencil_rk_imex!(state, solver, pms)
    return
end
# ... existing sparse matrix path ...
```

- [ ] **Step 3: Commit**

```bash
git add src/core/timesteppers/step_rk.jl src/core/pencil_matrices.jl
git commit -m "feat: per-pencil IMEX RK step function"
```

---

## Task 6: Solver Integration

**Files:**
- Modify: `src/core/solvers/solver_types.jl`

Build pencil matrices during solver construction for mixed Fourier-Chebyshev domains.

- [ ] **Step 1: Detect mixed domain and build pencil system**

```julia
# In build_solver_matrices!(solver), after existing matrix building:

function _try_build_pencil_system!(solver::InitialValueSolver)
    state = solver.state
    isempty(state) && return

    # Find Fourier and Chebyshev bases
    field = state[1]
    isempty(field.bases) && return

    fourier_basis = nothing
    cheb_basis = nothing
    for basis in field.bases
        basis === nothing && continue
        if isa(basis, FourierBasis) && fourier_basis === nothing
            fourier_basis = basis
        elseif isa(basis, JacobiBasis) && cheb_basis === nothing
            cheb_basis = basis
        end
    end

    # Only build pencil system for mixed Fourier+Chebyshev domains
    (fourier_basis === nothing || cheb_basis === nothing) && return

    @info "Building per-pencil matrix system for implicit IMEX"
    pms = PencilMatrixSystem(state, fourier_basis, cheb_basis)
    build_pencil_matrices!(pms, solver.problem, state, cheb_basis)
    solver.problem.parameters["pencil_system"] = pms

    @info "Pencil system: $(pms.n_pencils) pencils × $(pms.pencil_size) DOFs each"
end
```

- [ ] **Step 2: Call from solver constructor**

Add `_try_build_pencil_system!(solver)` at the end of `_build_initial_value_solver`.

- [ ] **Step 3: Commit**

```bash
git add src/core/solvers/solver_types.jl
git commit -m "feat: auto-detect mixed domain and build pencil system"
```

---

## Task 7: Testing

**Files:**
- Create: `test/test_pencil_matrices.jl`

- [ ] **Step 1: Unit test — pencil Laplacian block**

```julia
@testset "Pencil Laplacian block" begin
    # 1D Chebyshev(N=8) at kx=2π
    basis = ChebyshevT(Coordinate("z"); size=8, bounds=(0.0, 1.0))
    kx = 2π
    lap = Tarang.pencil_laplacian_block(kx, basis)
    @test size(lap) == (8, 8)
    # Eigenvalues should all be negative (diffusion operator)
    evals = eigvals(lap)
    @test all(real.(evals) .< 0)
end
```

- [ ] **Step 2: Unit test — gather/scatter roundtrip**

```julia
@testset "Pencil gather/scatter roundtrip" begin
    domain = ChannelDomain(32, 8; Lx=2π, Lz=1.0)
    u = ScalarField(domain, "u")
    fill_random!(u, "g"; seed=42)
    ensure_layout!(u, :c)

    state = [u]
    xb, zb = domain.bases
    pms = Tarang.PencilMatrixSystem(state, xb, zb)

    # Scatter to pencils
    pencils = Tarang.state_to_pencils(state, pms)
    @test length(pencils) == pms.n_pencils

    # Gather back
    state2 = [copy(u)]
    Tarang.pencils_to_state!(state2, pencils, pms)

    # Data should match
    @test get_coeff_data(state2[1]) ≈ get_coeff_data(u)
end
```

- [ ] **Step 3: Integration test — 1D diffusion**

```julia
@testset "Pencil IMEX diffusion" begin
    # ∂t(u) - κΔ(u) = 0 on Fourier(x)×Chebyshev(z) with u(z=0)=0, u(z=1)=0
    # Initial condition: u = sin(πz) → decays as exp(-κπ²t)
    domain = ChannelDomain(4, 16; Lx=2π, Lz=1.0)
    u = ScalarField(domain, "u")
    problem = IVP([u])
    add_parameters!(problem, kappa=0.1)
    add_equation!(problem, "∂t(u) - kappa*Δ(u) = 0")

    solver = InitialValueSolver(problem, RK222(); dt=0.01)

    # Set IC
    ensure_layout!(u, :g)
    z = local_grids(domain.dist, domain.bases[2])[1]
    get_grid_data(u)[1,:] .= sin.(π .* z)

    # Run 10 steps
    for _ in 1:10
        step!(solver)
    end

    # Check decay: should be ≈ exp(-0.1*π²*0.1) ≈ 0.906 of initial
    ensure_layout!(u, :g)
    @test maximum(abs.(get_grid_data(u))) < 1.0  # decayed
    @test maximum(abs.(get_grid_data(u))) > 0.5  # not too much
end
```

- [ ] **Step 4: Commit**

```bash
git add test/test_pencil_matrices.jl
git commit -m "test: pencil matrix assembly, gather/scatter, diffusion"
```

---

## Summary

| Task | What it builds | Size |
|------|---------------|------|
| 1 | PencilMatrixSystem struct | ~80 lines |
| 2 | Spectral operator block builders | ~40 lines |
| 3 | Per-pencil matrix assembly from equations | ~120 lines |
| 4 | LHS factorization + gather/scatter | ~80 lines |
| 5 | Per-pencil IMEX RK step | ~60 lines |
| 6 | Solver integration (auto-detect) | ~30 lines |
| 7 | Tests | ~60 lines |

**Total: ~470 lines of new code** replacing 33799×33799 sparse matrix with 129 × 512×512 dense pencil matrices.

After implementation, the RB example should show:
```
Pencil system: 129 pencils × 512 DOFs each
```
And the IMEX warning should disappear — implicit diffusion works correctly.
