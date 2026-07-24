# ---------------------------------------------------------------------------
# Per-equation F gather (equation space, matches L_min rows)
#
# The subproblem stepper needs F in equation space — the same row ordering
# as `L_min`. Earlier `gather_outputs!` packed F in *variable* space, which
# (a) misaligned PDE rows and (b) silently dropped BC F values because BC
# equations have no time derivative.
#
# `gather_eqn_F!` walks the equations in the original order and packs:
#   - PDE rows: from `pde_F_fields[tidx]` (one entry per state field target)
#   - BC rows:  from the equation's own F expression (constants projected
#               onto the current Fourier mode)
# then applies `pre_left` to match L_min's filtered row space.
# ---------------------------------------------------------------------------

_is_zero_F_expr(::Nothing) = true
_is_zero_F_expr(::ZeroOperator) = true
_is_zero_F_expr(x::Number) = x == 0
_is_zero_F_expr(c::ConstantOperator) = c.value == 0
_is_zero_F_expr(::Any) = false

_extract_F_constant(c::ConstantOperator) = Float64(c.value)
_extract_F_constant(x::Number) = Float64(x)
_extract_F_constant(::Any) = nothing

"""
    _evaluate_alg_F(F_expr, sp) -> ComplexF64

Dispatch-based evaluation of an algebraic-equation F expression for a given
subproblem. Returns the complex value to write into the BC row of the raw
equation-space vector.

Currently supports:
- `ZeroOperator` / `nothing` → 0
- `ConstantOperator` / `Number` → `v * Nx` at DC, 0 elsewhere
- `ArrayOperator` → unnormalized RFFT of the grid-space array, picked at the
  subproblem's Fourier mode (for space-dependent BCs)
"""
_evaluate_alg_F(::Nothing, ::Subproblem) = ComplexF64(0)
_evaluate_alg_F(::ZeroOperator, ::Subproblem) = ComplexF64(0)
_evaluate_alg_F(c::ConstantOperator, sp::Subproblem) = _bc_constant_projection(Float64(c.value), sp)
_evaluate_alg_F(x::Number, sp::Subproblem) = _bc_constant_projection(Float64(x), sp)
_evaluate_alg_F(a::ArrayOperator, sp::Subproblem) = _bc_array_projection(a.value, sp)
function _evaluate_alg_F(expr, sp::Subproblem)
    # A COMPOUND CONSTANT — `T(z=0) = 10*25`, `= h*T_amb`, `= 1/Re` — arrives here as a
    # Multiply/Add/Divide operator tree, not as a ConstantOperator. It used to fall through to
    # the silent zero below, so the boundary condition was enforced as 0 with no warning and the
    # solve reported success. `_is_const_or_param` / `_extract_scalar` already fold exactly these
    # node types for the L/M matrices; use them here too.
    if _is_const_or_param(expr)
        return _bc_constant_projection(Float64(_extract_scalar(expr)), sp)
    end
    # Anything else genuinely is not supported as a BC right-hand side. Enforcing it as zero is a
    # silently wrong answer, which is worse than a slow or absent one — say so.
    @warn "Boundary condition right-hand side of type $(typeof(expr)) is not supported and is " *
          "being enforced as ZERO. Supported: a constant, a compound constant (`10*25`, `h*T_amb`), " *
          "or a grid array (space-dependent BC). Rewrite the BC, or the solve will silently " *
          "satisfy the wrong condition." maxlog=5
    return ComplexF64(0)
end

"""
Project a constant value onto the current subproblem's Fourier mode.

Tarang uses unnormalized `FFTW.rfft` / `FFTW.fft` along each separable
axis. For a constant `v` in grid space, the only nonzero Fourier
coefficient is at the full-DC mode `(kx=0, ky=0, ...)`, with value
`v * Nx * Ny * ...` (product of all Fourier-axis grid sizes). All
non-DC Fourier modes project to zero.

For problems without any Fourier axis (e.g. a pure-coupled BVP), the
DC-mode value is `v` itself — no scaling applied.
"""
function _bc_constant_projection(v::Float64, sp::Subproblem)
    v == 0 && return ComplexF64(0)
    # Every separable (Fourier) axis of this subproblem must be at its DC
    # mode (global index 1) for a constant to have any contribution.
    fourier_idx = _subproblem_fourier_group_indices(sp)
    for k in fourier_idx
        if k != 1
            return ComplexF64(0)
        end
    end
    # DC on every Fourier axis → `v * ∏ N_k` via unnormalized FFTs.
    scale = 1.0
    for N in _bc_fourier_axis_sizes(sp)
        scale *= Float64(N)
    end
    return ComplexF64(v * scale)
end

function _find_bc_fourier_basis(sp::Subproblem)
    for var in sp.problem.variables
        for comp in scalar_components(var)
            for basis in comp.bases
                if basis !== nothing && isa(basis, FourierBasis)
                    return basis
                end
            end
        end
    end
    return nothing
end

"""
    _bc_fourier_axis_sizes(sp) -> Vector{Int}

Ordered list of grid sizes for the problem's separable (Fourier) axes. Used
to determine the expected output shape of a BC array so that lower-rank
user inputs (e.g. `sin(x)` in a 3D problem) can be broadcast to the full
output before being transformed.
"""
function _bc_fourier_axis_sizes(sp::Subproblem)
    sizes = Int[]
    seen = Set{String}()
    for var in sp.problem.variables
        for comp in scalar_components(var)
            for basis in comp.bases
                basis === nothing && continue
                isa(basis, FourierBasis) || continue
                label = String(basis.meta.element_label)
                label in seen && continue
                push!(seen, label)
                push!(sizes, basis.meta.size)
            end
        end
    end
    return sizes
end

"""
    _subproblem_fourier_group_indices(sp) -> Vector{Int}

Return the 1-based Fourier mode indices for every separable axis of this
subproblem. For a 2D problem with one Fourier axis this is `[kx_global]`;
for a 3D problem it's `[kx_global, ky_global]`.
"""
function _subproblem_fourier_group_indices(sp::Subproblem)
    idx = Int[]
    for g in sp.group
        g isa Integer || continue
        push!(idx, g + 1)
    end
    return idx
end

"""Expand a BC array onto the full boundary plane spanned by the Fourier axes.

A BC that depends on a subset of the Fourier axes evaluates to an array with singleton (or
missing) dimensions — `(1, Ny)` for `cos(2πy/Ly)`, `(Nx,)` for `sin(2πx/Lx)` in a 1-Fourier-axis
problem. Broadcast it up to `fourier_sizes` so the transform is taken over the right axes.

A 1-D array in a MULTI-Fourier-axis problem is ambiguous — it carries no axis identity — and used
to be silently treated as the first axis. It can only arise now if a caller registered a
coordinate array by hand, so accept it when its length pins the axis unambiguously and refuse
otherwise, rather than guessing."""
function _expand_bc_array_to_plane(arr::AbstractArray, fourier_sizes)
    dims = Tuple(fourier_sizes)
    size(arr) == dims && return arr
    n = length(dims)

    if ndims(arr) == n
        # Singleton dims → broadcast. (Every non-singleton dim must already match.)
        all(d -> size(arr, d) == dims[d] || size(arr, d) == 1, 1:n) || return arr
        out = Array{Float64}(undef, dims)
        out .= arr
        return out
    end

    if ndims(arr) == 1 && n >= 2
        matches = findall(==(length(arr)), collect(dims))
        if length(matches) == 1
            shape = ntuple(d -> d == matches[1] ? length(arr) : 1, n)
            out = Array{Float64}(undef, dims)
            out .= reshape(arr, shape)
            return out
        end
        @warn "BC array of length $(length(arr)) is ambiguous on a boundary plane of size " *
              "$dims — it does not identify which axis it varies along. Register the " *
              "coordinate with its axis shape (e.g. reshape to (1, N)) so the BC is applied to " *
              "the intended direction." maxlog=3
    end
    return arr
end

"""
    _bc_array_projection(arr, sp)

Project a grid-space array `arr` (from a space-dependent BC) onto the current
subproblem's Fourier mode. Returns a `ComplexF64` value suitable for writing
into the BC row of the raw equation-space vector.

`arr` is first expanded onto the full boundary plane (see
`_expand_bc_array_to_plane`), then transformed. The FFT result is cached by
identity on `sp.problem.parameters` via an `IdDict`, so all subproblems sharing
the same `ArrayOperator` reuse a single FFT per refresh.
"""
function _bc_array_projection(arr::AbstractArray, sp::Subproblem)
    (arr === nothing || length(arr) == 0) && return ComplexF64(0)

    fourier_sizes = _bc_fourier_axis_sizes(sp)
    if isempty(fourier_sizes)
        # No Fourier axes at all (pure-coupled / BVP-like). Use arr[1] as
        # the DC-mode value.
        return ComplexF64(first(arr))
    end

    # The BC expression is evaluated against coordinate arrays that carry their axis identity in
    # their shape (see `_auto_register_coordinate_fields!`), so a BC depending on only some of the
    # Fourier axes comes back with singleton dims — `(1, Ny)` for `cos(2πy/Ly)`. Expand it onto the
    # full boundary plane before transforming; otherwise the FFT is taken over the wrong axis and
    # the profile is silently applied along the wrong direction.
    arr = _expand_bc_array_to_plane(arr, fourier_sizes)

    coeffs = _get_or_compute_bc_array_coeffs!(arr, sp, fourier_sizes)
    coeffs === nothing && return ComplexF64(0)

    fourier_idx = _subproblem_fourier_group_indices(sp)
    if ndims(coeffs) == 1
        kx = isempty(fourier_idx) ? 1 : first(fourier_idx)
        return (kx >= 1 && kx <= length(coeffs)) ?
               ComplexF64(coeffs[kx]) : ComplexF64(0)
    elseif ndims(coeffs) == 2
        length(fourier_idx) >= 2 || return ComplexF64(0)
        kx, ky = fourier_idx[1], fourier_idx[2]
        return (1 <= kx <= size(coeffs, 1) && 1 <= ky <= size(coeffs, 2)) ?
               ComplexF64(coeffs[kx, ky]) : ComplexF64(0)
    elseif ndims(coeffs) == 3
        length(fourier_idx) >= 3 || return ComplexF64(0)
        kx, ky, kz = fourier_idx[1], fourier_idx[2], fourier_idx[3]
        return (1 <= kx <= size(coeffs, 1) &&
                1 <= ky <= size(coeffs, 2) &&
                1 <= kz <= size(coeffs, 3)) ?
               ComplexF64(coeffs[kx, ky, kz]) : ComplexF64(0)
    else
        @warn "BC array projection: unsupported coefficient rank $(ndims(coeffs))" maxlog=3
        return ComplexF64(0)
    end
end

"""
    _get_or_compute_bc_array_coeffs!(arr, sp, fourier_sizes) -> coefficients

Cache-backed helper that:
1. Reshapes/broadcasts `arr` to the full Fourier-output shape (derived from
   `fourier_sizes`) so lower-rank inputs work in higher-dim problems.
2. Takes an unnormalized forward FFT along the first axis (via `rfft` for
   real input) and complex FFT along remaining axes.
3. Returns the complex coefficient array for downstream indexing.

The result is cached on `sp.problem.parameters["_bc_rfft_cache"]` by array
object identity (`IdDict`) so all subproblems reusing the same `arr` share
a single FFT per refresh.
"""
function _get_or_compute_bc_array_coeffs!(arr::AbstractArray,
                                          sp::Subproblem,
                                          fourier_sizes::Vector{Int})
    cache = get(sp.problem.parameters, "_bc_rfft_cache", nothing)
    if cache === nothing
        cache = IdDict{Any, Any}()
        sp.problem.parameters["_bc_rfft_cache"] = cache
    end
    cached = get(cache, arr, nothing)
    cached !== nothing && return cached

    coeffs = try
        broadcast_arr = _broadcast_bc_array_to_output(arr, fourier_sizes)
        _forward_fft_bc(broadcast_arr)
    catch err
        @warn "BC array FFT failed: $err" maxlog=1
        return nothing
    end

    cache[arr] = coeffs
    return coeffs
end

"""
    _broadcast_bc_array_to_output(arr, fourier_sizes) -> Array

Broadcast a grid-space BC array to the full Fourier-output shape.

- If `arr` already matches `fourier_sizes`, returns it (as a concrete array).
- If `arr` has fewer dimensions, reshape it into a singleton-padded shape
  and broadcast to the full shape. We assume the user supplies the array
  in axis order matching the problem's Fourier axes, so a 1D `arr` of
  length `fourier_sizes[k]` is broadcast along the matching dimension and
  replicated along the rest.
- A single-element `arr` becomes a uniform constant over the full shape.
"""
function _broadcast_bc_array_to_output(arr::AbstractArray, fourier_sizes::Vector{Int})
    output_shape = Tuple(fourier_sizes)
    ndout = length(fourier_sizes)

    # Scalar-ish input (length 1) → constant over the output.
    if length(arr) == 1
        result = Array{Float64}(undef, output_shape...)
        fill!(result, Float64(first(arr)))
        return result
    end

    # Exact match (shape and rank) → collect to a concrete array to keep
    # downstream FFT predictable.
    if size(arr) == output_shape
        return collect(Float64.(arr))
    end

    # Same rank, same total length but different dim ordering (unusual).
    if length(arr) == prod(output_shape)
        return reshape(collect(Float64.(arr)), output_shape...)
    end

    # Lower-dimensional input: pad its shape with trailing singletons so
    # that Julia's `broadcast` can expand it along the missing axes.
    nd_in = ndims(arr)
    if nd_in < ndout
        # Try to match each input dimension to an output dimension of the
        # same size; fall back to leading-dim match.
        dims_padded = ntuple(i -> i <= nd_in ? size(arr, i) : 1, ndout)
        if dims_padded[1] == output_shape[1] || any(i -> dims_padded[i] == output_shape[i], 1:nd_in)
            reshaped = reshape(collect(Float64.(arr)), dims_padded)
            target = Array{Float64}(undef, output_shape...)
            target .= reshaped
            return target
        end
    end

    # Length matches a single Fourier axis but rank is 1 — broadcast along
    # the first axis with that size. (Covers 1D `sin(x)` in 3D problems.)
    if nd_in == 1
        for (axis, sz) in enumerate(output_shape)
            if length(arr) == sz
                # Reshape to have the Fourier length on `axis`, singletons elsewhere.
                rshape = ntuple(i -> i == axis ? sz : 1, ndout)
                reshaped = reshape(collect(Float64.(arr)), rshape)
                target = Array{Float64}(undef, output_shape...)
                target .= reshaped
                return target
            end
        end
    end

    throw(ArgumentError(
        "BC array shape $(size(arr)) incompatible with Fourier output " *
        "shape $(output_shape); provide the array in either the full " *
        "output shape or a 1-D/1-element form that can be broadcast."
    ))
end

"""
    _forward_fft_bc(arr) -> complex coefficient array

Unnormalized forward Fourier transform matching Tarang's `RealFourier`
convention: `FFTW.rfft` along the first dimension for real input, full
`FFTW.fft` for complex input. Multi-dim real input transforms the first
dim with `rfft` and remaining dims with `fft`, matching the shape that
`_bc_fourier_axis_sizes` + `_subproblem_fourier_group_indices` expect
when looking up per-mode coefficients.
"""
function _forward_fft_bc(arr::AbstractArray)
    if eltype(arr) <: Complex
        return FFTW.fft(ComplexF64.(arr))
    else
        return FFTW.rfft(Float64.(arr))
    end
end

"""
    invalidate_bc_array_cache!(problem)

Clear the per-problem BC-array FFT cache. Call when BC arrays change (e.g.
at the start of each step, or after evaluating new time-dependent array
values via `_apply_bc_values_to_equations!`).
"""
function invalidate_bc_array_cache!(problem)
    cache = get(problem.parameters, "_bc_rfft_cache", nothing)
    cache === nothing && return
    if cache isa IdDict
        empty!(cache)
    end
    return
end

"""
    gather_eqn_F!(dest, sp, solver, pde_F_fields, state_fields)

Pack PDE-equation F values into equation-space. For each equation with a time
derivative (`M` term), pull F from `pde_F_fields` at the equation's target
state-field indices. Algebraic/BC equations (no `M` term) contribute ZERO to
this vector — they are handled separately via `gather_alg_F!` + a direct RHS
override in the stepper, because the IMEX-RK accumulated formula gives the
wrong `1/γ` scaling for inhomogeneous algebraic constraints.
"""
function gather_eqn_F!(dest::AbstractVector{ComplexF64}, sp::Subproblem, solver,
                       pde_F_fields::Vector, state_fields::Vector)
    problem = sp.problem
    eqns = problem.equation_data

    eqn_sizes = _subproblem_eqn_sizes(sp)
    eqn_targets = _subproblem_eqn_targets(sp, state_fields)
    I_raw = _subproblem_raw_eqn_size(sp)

    raw = _subproblem_cached_vector!(sp, :gather_eqn_F_raw, I_raw; like=dest)
    fill!(raw, zero(eltype(raw)))

    kx_global = _kx_index_global(sp)

    i0 = 0
    for (eq_idx, eq_data) in enumerate(eqns)
        eq_size = eqn_sizes[eq_idx]
        if eq_size == 0
            continue
        end

        target_indices = eqn_targets[eq_idx]
        if !isempty(target_indices)
            offset = i0
            for tidx in target_indices
                if tidx >= 1 && tidx <= length(pde_F_fields)
                    fld = pde_F_fields[tidx]
                    if fld !== nothing
                        offset = _gather_field_raw!(raw, offset, fld, kx_global, sp)
                        continue
                    end
                end
                if tidx >= 1 && tidx <= length(state_fields)
                    offset += subproblem_field_size(sp, state_fields[tidx])
                end
            end
        end
        # Algebraic rows intentionally left zero — see gather_alg_F!.

        i0 += eq_size
    end

    compress_equation_space!(dest, sp, raw)
    return dest
end

"""
    gather_alg_F!(dest, sp)

Pack algebraic-constraint F values (from BC / constraint equations that have
no time derivative) into equation-space, with zeros at PDE rows.

For each equation without an `M` term, evaluate its stored `F` expression
(typically a `ConstantOperator`) and project onto the current Fourier mode.
The result is used by the stepper to OVERRIDE the BC rows of the RHS with
`dt * a_ii * F_alg`, yielding `L_row * X = F_alg` at each stage — the correct
enforcement of the algebraic constraint.

This override is necessary because the standard IMEX-RK accumulated RHS
formula gives `L_row * X = (A^E[i,j]/a_ii) * F_BC` which is wrong by a factor
of `1/γ` for inhomogeneous algebraic constraints.
"""
function gather_alg_F!(dest::AbstractVector{ComplexF64}, sp::Subproblem)
    problem = sp.problem
    eqns = problem.equation_data

    eqn_sizes = _subproblem_eqn_sizes(sp)
    I_raw = _subproblem_raw_eqn_size(sp)

    # Build the sparse BC F vector on the HOST via scalar writes (a few nonzero
    # entries at BC row offsets, the rest zero). Then upload once into the
    # device-resident `raw` buffer via `_assign_to_buffer!` — this keeps
    # scalar indexing off of GPU arrays so the helper is safe under
    # `CUDA.allowscalar(false)`.
    raw_cpu = sp.runtime.gather_alg_F_raw_cpu
    if raw_cpu === nothing || length(raw_cpu) != I_raw
        raw_cpu = zeros(ComplexF64, I_raw)
        sp.runtime.gather_alg_F_raw_cpu = raw_cpu
    else
        fill!(raw_cpu, zero(ComplexF64))
    end

    i0 = 0
    for (eq_idx, eq_data) in enumerate(eqns)
        eq_size = eqn_sizes[eq_idx]
        if eq_size == 0
            continue
        end

        M_expr = get(eq_data, "M", nothing)
        is_alg = M_expr === nothing || _is_zero_m_term(M_expr)

        if is_alg
            F_expr = get(eq_data, "F_expr", nothing)
            if F_expr === nothing
                F_expr = get(eq_data, "F", nothing)
            end
            if !_is_zero_F_expr(F_expr)
                coeff = _evaluate_alg_F(F_expr, sp)
                if coeff != 0
                    # Replicate the value across all rows of the BC
                    # equation's block. For scalar BCs `eq_size == 1` and
                    # this writes a single entry; for vector BCs (e.g.
                    # `u(z=0) = c` with `u` a 2-component vector, `eq_size
                    # == 2`) the same coefficient is written to every
                    # component row. The Interpolate LHS for a vector
                    # operand is `kron(I_ncomp, row)`, so replicating the
                    # scalar F across rows enforces the same value on each
                    # component — which is what "u = c" means.
                    @inbounds for r in 1:eq_size
                        raw_cpu[i0 + r] = coeff
                    end
                end
            end
        end

        i0 += eq_size
    end

    # Upload the CPU-built raw vector into the device-resident raw buffer.
    # This is an INTENTIONAL one-shot H2D upload of a freshly host-built staging
    # vector (the scalar writes above must stay off device arrays), NOT field
    # staging — so use a direct copyto! rather than `_assign_to_buffer!`, whose
    # same-architecture guard exists to catch *accidental* CPU/GPU mixing and
    # would (correctly, for its purpose) refuse this transfer. Routing through
    # it killed every GPU coupled subproblem step at the pre-stage ALG_F gather.
    raw = _subproblem_cached_vector!(sp, :gather_alg_F_raw, I_raw; like=dest)
    copyto!(raw, raw_cpu)

    compress_equation_space!(dest, sp, raw)
    return dest
end
