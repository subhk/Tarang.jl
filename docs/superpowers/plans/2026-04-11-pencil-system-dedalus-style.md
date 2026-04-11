# Per-Pencil Subproblem System (Dedalus-Style) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the pencil matrix system to follow the Dedalus subproblem architecture: variables as multi-component entities (VectorField = n_comp × Nz DOFs), expression_matrices returning component-aware blocks, and proper gather/scatter with tensor indices.

**Architecture:** Each pencil (Fourier mode kx) has a dense matrix of size `(total_var_dofs × total_var_dofs)` where `total_var_dofs = Σ field_size(var)` and `field_size(VectorField) = n_comp × Nz`, `field_size(ScalarField) = Nz`, `field_size(tau_0D) = 1`, `field_size(tau_1D) = 1`. The matrix assembly iterates over `problem.variables` (not flattened state), and `expression_matrices` returns blocks that include component structure.

**Tech Stack:** Julia, LinearAlgebra, SparseArrays (for permutations), existing Tarang basis/field infrastructure.

---

## Why rewrite (not patch)

The current pencil system has three fundamental bugs:

1. **VectorField→ScalarField mismatch**: State uses flattened scalars `[u_x, u_z]` but equations reference VectorField `u`. `pencil_expression_block(TimeDerivative(u), u_x)` fails because names don't match.

2. **Inconsistent sizing**: Column widths hardcoded to `Nz` for all fields including 1-DOF tau fields → out-of-bounds indexing.

3. **Row-column mismatch**: Equation row count (262) ≠ variable column count (263) → non-square matrix.

All three stem from the same root: the system was designed around flattened scalar state fields instead of the natural variable hierarchy. The Dedalus approach works because it keeps VectorFields as single multi-component entities throughout.

---

## Design: Variable-Centric Pencil System

### Pencil vector layout

For `problem.variables = [p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2]`:

```
Variable      Type          Components    Pencil DOFs
─────────     ────────      ──────────    ──────────
p             ScalarField   1             Nz = 64
b             ScalarField   1             Nz = 64
u             VectorField   2             2×Nz = 128
tau_p         ScalarField   1 (0D)        1
tau_b1        ScalarField   1 (1D)        1
tau_b2        ScalarField   1 (1D)        1
tau_u1        VectorField   2 (1D)        2×1 = 2
tau_u2        VectorField   2 (1D)        2×1 = 2
                                          ─────
                                Total:    263
```

Each variable occupies a contiguous block. VectorField components are interleaved within their block:
- `u` block: `[u_x[z=0], u_x[z=1], ..., u_x[z=Nz-1], u_z[z=0], ..., u_z[z=Nz-1]]`

### Equation block structure

Equations map 1:1 to variables (by index). Each equation produces the same number of rows as its variable's pencil DOFs:

```
Equation                            Variable    Rows
──────────────────────────           ────────    ────
trace(grad_u) + tau_p = 0           p           64
∂t(b) - κ div(grad_b) + ...        b           64
∂t(u) - ν div(grad_u) + ...        u           128
b(z=0) = Lz                        tau_p       1
u(z=0) = 0                         tau_b1      1
b(z=Lz) = 0                        tau_b2      1
u(z=Lz) = 0                        tau_u1      2
integ(p) = 0                       tau_u2      2
                                                ─────
                                    Total:      263
```

The matrix is square (263×263).

### Expression matrices

For a vector equation like `∂t(u) - ν Δ(u) + ∇(p) - b·ez = ...`:
- Block `(u_eq, u_var)`: `[[I, 0], [0, I]]` for M, `[[-ν Δ_kx, 0], [0, -ν Δ_kx]]` for L (block-diagonal Laplacian)
- Block `(u_eq, p_var)`: `[[ik_x I], [D_z]]` (gradient: ∂p/∂x, ∂p/∂z)
- Block `(u_eq, b_var)`: `[[0], [-I]]` (buoyancy: -b·ez = [0, -b], coefficient for b is [0, -1])

Each block has size `(eq_dofs × var_dofs)`, e.g., `(128 × 64)` for `(u_eq, p_var)`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/core/pencil_system.jl` | **Create** (replaces `pencil_matrices.jl`) | PencilSystem struct, variable-centric assembly, solve, gather/scatter |
| `src/core/timesteppers/step_pencil_rk.jl` | **Modify** | Updated to use PencilSystem |
| `src/core/solvers/solver_types.jl` | **Modify** | Build PencilSystem from problem |
| `src/core/pencil_matrices.jl` | **Delete** | Replaced by `pencil_system.jl` |
| `test/test_pencil_system.jl` | **Create** | Variable-centric tests |

---

## Task 1: `PencilSystem` struct with variable-centric sizing

**Files:**
- Create: `src/core/pencil_system.jl`

- [ ] **Step 1: Define the struct**

```julia
struct PencilSystem
    # Per-pencil dense matrices
    L_pencils::Vector{Matrix{ComplexF64}}
    M_pencils::Vector{Matrix{ComplexF64}}
    lhs_cache::Dict{Tuple{Int,Float64,Float64}, LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64},Vector{Int64}}}

    # Sizing
    n_pencils::Int           # Fourier modes (N/2+1 for rFFT)
    pencil_size::Int         # Total DOFs per pencil
    n_cheb::Int              # Chebyshev modes

    # Variable layout (indexed by problem.variables order)
    n_vars::Int              # length(problem.variables)
    var_names::Vector{String}
    var_dofs::Vector{Int}    # DOFs per pencil for each variable
    var_offsets::Vector{Int}  # cumulative offset in pencil vector
    var_n_comp::Vector{Int}  # components per variable (1 for scalar, 2+ for vector)
    var_comp_size::Vector{Int} # DOFs per component (Nz for 2D, 1 for tau)

    # Wavenumbers
    kx_values::Vector{Float64}

    # Basis references
    cheb_label::String       # Chebyshev coordinate label
end
```

- [ ] **Step 2: Constructor**

```julia
function PencilSystem(problem::Problem, fourier_basis, cheb_basis)
    variables = problem.variables
    n_vars = length(variables)
    n_cheb = cheb_basis.meta.size
    cheb_label = String(cheb_basis.meta.element_label)

    # Fourier mode count
    N_f = fourier_basis.meta.size
    n_pencils = isa(fourier_basis, RealFourier) ? (N_f ÷ 2 + 1) : N_f
    L = fourier_basis.meta.bounds[2] - fourier_basis.meta.bounds[1]
    kx_values = [(i-1) * 2π / L for i in 1:n_pencils]

    # Compute per-variable DOFs
    var_names = String[]
    var_dofs = Int[]
    var_n_comp = Int[]
    var_comp_size = Int[]

    for var in variables
        push!(var_names, hasfield(typeof(var), :name) ? var.name : "?")
        n_comp = isa(var, VectorField) ? length(var.components) :
                 isa(var, TensorField) ? length(vec(var.components)) : 1
        # Component size: Nz for 2D fields, 1 for tau (empty or 1D bases)
        comp_sz = _pencil_comp_size(var, cheb_label, n_cheb)
        push!(var_n_comp, n_comp)
        push!(var_comp_size, comp_sz)
        push!(var_dofs, n_comp * comp_sz)
    end

    # Offsets
    var_offsets = Int[]
    offset = 0
    for d in var_dofs
        push!(var_offsets, offset)
        offset += d
    end
    pencil_size = offset

    # Allocate
    L_pencils = [zeros(ComplexF64, pencil_size, pencil_size) for _ in 1:n_pencils]
    M_pencils = [zeros(ComplexF64, pencil_size, pencil_size) for _ in 1:n_pencils]
    lhs_cache = Dict{Tuple{Int,Float64,Float64}, LU{ComplexF64,Matrix{ComplexF64},Vector{Int64}}}()

    return PencilSystem(L_pencils, M_pencils, lhs_cache,
                        n_pencils, pencil_size, n_cheb, n_vars,
                        var_names, var_dofs, var_offsets, var_n_comp, var_comp_size,
                        kx_values, cheb_label)
end

function _pencil_comp_size(var, cheb_label, n_cheb)
    field = isa(var, VectorField) ? var.components[1] :
            isa(var, TensorField) ? var.components[1,1] : var
    if isempty(field.bases)
        return 1  # 0D tau
    end
    for basis in field.bases
        basis === nothing && continue
        if String(basis.meta.element_label) == cheb_label
            return n_cheb  # has Chebyshev → full pencil
        end
    end
    return 1  # 1D Fourier-only tau
end
```

- [ ] **Step 3: Include in Tarang.jl** (replace pencil_matrices.jl)
- [ ] **Step 4: Commit**

---

## Task 2: Variable-centric `pencil_expression_block`

**Files:**
- Modify: `src/core/pencil_system.jl`

The key change: the function now returns a `(eq_dofs × var_dofs)` block, not `(Nz × Nz)`. For vector equations/variables, the block includes component structure.

- [ ] **Step 1: Component-aware expression block**

```julia
"""
Build the (eq_dofs × var_dofs) pencil block for `expr` acting on `var`
at wavenumber kx. For vector expressions, the block includes component
sub-blocks: [[comp11, comp12], [comp21, comp22]].
"""
function pencil_expression_block(expr, var, kx::Float64, pms::PencilSystem, var_idx::Int)
    n_comp_var = pms.var_n_comp[var_idx]
    comp_sz = pms.var_comp_size[var_idx]
    var_dofs = pms.var_dofs[var_idx]

    # For scalar variables: delegate to scalar block builder
    if n_comp_var == 1
        return _scalar_pencil_block(expr, var, kx, pms, comp_sz)
    end

    # For VectorField: build per-component blocks
    # Need to determine which component of the equation this block represents
    # This is handled by the caller (build_pencil_matrices!) which
    # splits vector equations into component rows
    return _scalar_pencil_block(expr, var, kx, pms, comp_sz)
end
```

- [ ] **Step 2: Scalar block builder** (similar to existing but uses `pms.cheb_label`)
- [ ] **Step 3: Commit**

---

## Task 3: Variable-centric `build_pencil_matrices!`

**Files:**
- Modify: `src/core/pencil_system.jl`

- [ ] **Step 1: Assembly loop over problem.variables**

```julia
function build_pencil_matrices!(ps::PencilSystem, problem::Problem, cheb_basis)
    for kx_idx in 1:ps.n_pencils
        kx = ps.kx_values[kx_idx]
        L = ps.L_pencils[kx_idx]
        M = ps.M_pencils[kx_idx]
        fill!(L, 0); fill!(M, 0)

        for (eq_idx, eq_data) in enumerate(problem.equation_data)
            eq_idx > ps.n_vars && break  # skip BC equations beyond variable count
            eq_var = problem.variables[eq_idx]

            L_expr = get(eq_data, "L", nothing)
            M_expr = get(eq_data, "M", nothing)

            eq_offset = ps.var_offsets[eq_idx]
            eq_n_comp = ps.var_n_comp[eq_idx]
            eq_comp_sz = ps.var_comp_size[eq_idx]

            for (var_idx, var) in enumerate(problem.variables)
                var_offset = ps.var_offsets[var_idx]
                var_n_comp = ps.var_n_comp[var_idx]
                var_comp_sz = ps.var_comp_size[var_idx]

                # Build per-component sub-blocks
                for eq_c in 1:eq_n_comp
                    eq_field = _get_component(eq_var, eq_c)
                    row_start = eq_offset + (eq_c-1)*eq_comp_sz + 1
                    row_end = row_start + eq_comp_sz - 1

                    for var_c in 1:var_n_comp
                        var_field = _get_component(var, var_c)
                        col_start = var_offset + (var_c-1)*var_comp_sz + 1
                        col_end = col_start + var_comp_sz - 1

                        # Build scalar block
                        if L_expr !== nothing
                            blk = _scalar_pencil_block(L_expr, var_field, kx, ps, min(eq_comp_sz, var_comp_sz))
                            if blk !== nothing
                                nr = min(size(blk,1), eq_comp_sz)
                                nc = min(size(blk,2), var_comp_sz)
                                L[row_start:row_start+nr-1, col_start:col_start+nc-1] .+= blk[1:nr, 1:nc]
                            end
                        end
                        if M_expr !== nothing
                            blk = _scalar_pencil_block(M_expr, var_field, kx, ps, min(eq_comp_sz, var_comp_sz))
                            if blk !== nothing
                                nr = min(size(blk,1), eq_comp_sz)
                                nc = min(size(blk,2), var_comp_sz)
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

_get_component(var::ScalarField, c::Int) = var
_get_component(var::VectorField, c::Int) = var.components[c]
_get_component(var::TensorField, c::Int) = vec(var.components)[c]
```

- [ ] **Step 2: Commit**

---

## Task 4: Variable-centric gather/scatter

**Files:**
- Modify: `src/core/pencil_system.jl`

- [ ] **Step 1: `state_to_pencils` working with problem.variables**

```julia
function state_to_pencils(problem::Problem, ps::PencilSystem)
    pencils = [zeros(ComplexF64, ps.pencil_size) for _ in 1:ps.n_pencils]

    for (var_idx, var) in enumerate(problem.variables)
        offset = ps.var_offsets[var_idx]
        n_comp = ps.var_n_comp[var_idx]
        comp_sz = ps.var_comp_size[var_idx]

        for c in 1:n_comp
            field = _get_component(var, c)
            comp_offset = offset + (c-1) * comp_sz
            _scatter_field_to_pencils!(pencils, field, comp_offset, comp_sz, ps)
        end
    end
    return pencils
end

function _scatter_field_to_pencils!(pencils, field, offset, comp_sz, ps)
    if isempty(field.bases)
        # 0D tau: single value in pencil 1
        gd = get_grid_data(field)
        val = (gd !== nothing && length(gd) >= 1) ? ComplexF64(gd[1]) : ComplexF64(0)
        pencils[1][offset+1] = val
    elseif comp_sz == 1
        # 1D Fourier-only tau: one value per pencil
        ensure_layout!(field, :c)
        cd = get_coeff_data(field)
        cd === nothing && return
        for kx in 1:min(ps.n_pencils, length(cd))
            pencils[kx][offset+1] = ComplexF64(cd[kx])
        end
    else
        # 2D field: reshape (n_pencils, n_cheb)
        ensure_layout!(field, :c)
        cd = get_coeff_data(field)
        cd === nothing && return
        c2d = reshape(vec(cd), ps.n_pencils, ps.n_cheb)
        for kx in 1:ps.n_pencils
            pencils[kx][offset+1 : offset+comp_sz] .= c2d[kx, :]
        end
    end
end
```

- [ ] **Step 2: `pencils_to_problem!` inverse**
- [ ] **Step 3: Commit**

---

## Task 5: Updated step function and solver integration

**Files:**
- Modify: `src/core/timesteppers/step_pencil_rk.jl`
- Modify: `src/core/solvers/solver_types.jl`

- [ ] **Step 1: `step_pencil_rk_imex!` uses `problem.variables` for gather/scatter**

The step function now calls `state_to_pencils(problem, ps)` and `pencils_to_problem!(problem, pencils, ps)` instead of working with the flattened state. After the pencil solve, it syncs problem variables back to the solver state via `collect_state_fields`.

- [ ] **Step 2: `_try_build_pencil_system!` uses `PencilSystem(problem, ...)`**
- [ ] **Step 3: Re-enable pencil dispatch in `step_rk.jl`**
- [ ] **Step 4: Commit**

---

## Task 6: Tests

**Files:**
- Create: `test/test_pencil_system.jl`

- [ ] **Step 1: Constructor sizing test**
- [ ] **Step 2: Gather/scatter roundtrip for ScalarField + VectorField**
- [ ] **Step 3: M_pencils identity block for vector equation**
- [ ] **Step 4: 1D diffusion convergence test**
- [ ] **Step 5: Commit**

---

## Task 7: Cleanup

- [ ] **Step 1: Delete `src/core/pencil_matrices.jl`**
- [ ] **Step 2: Update `src/Tarang.jl` includes**
- [ ] **Step 3: Commit**

---

## Key Differences from Current Implementation

| Aspect | Current (broken) | New (Dedalus-style) |
|--------|-----------------|---------------------|
| Pencil indexed by | Flattened scalar state (11 fields) | problem.variables (8 vars) |
| VectorField `u` DOFs | 2 separate entries (u_x: Nz, u_z: Nz) | 1 block (2×Nz = 128) |
| Equation-variable mapping | No consistent mapping | 1:1 by index |
| Component iteration | None (breaks on VectorField) | Inner loop over components |
| Column widths | Hardcoded Nz | Per-variable from `var_dofs` |
| Gather/scatter | Uses `solver.state` (scalar) | Uses `problem.variables` (typed) |
| Matrix squareness | 262 ≠ 263 | 263 = 263 (by construction) |
