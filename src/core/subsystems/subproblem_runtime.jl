# Per-subproblem gather/scatter
# ---------------------------------------------------------------------------

"""Return the global 1-based Fourier mode index for this subproblem."""
function _kx_index_global(sp::Subproblem)
    for g in sp.group
        if g isa Integer
            return g + 1  # 0-based → 1-based
        end
    end
    return 1
end

"""
Convert a global 1-based Fourier mode index to a local index within the
coefficient data array. For serial runs, local == global. For MPI with
PencilArrays, the local buffer only holds this rank's modes.
"""
function _global_to_local_kx(kx_global::Int, field::ScalarField, sp::Subproblem)
    dist = sp.dist
    if dist === nothing || dist.size <= 1
        return kx_global
    end
    # Find the separable (Fourier) axis and its global size
    for (axis, g) in enumerate(sp.group)
        if g isa Integer && axis <= length(field.bases) && field.bases[axis] !== nothing
            basis = field.bases[axis]
            if isa(basis, FourierBasis)
                global_size = isa(basis, RealFourier) ? div(basis.meta.size, 2) + 1 : basis.meta.size
                local_range = local_indices(dist, axis, global_size)
                return kx_global - first(local_range) + 1
            end
        end
    end
    return kx_global
end

"""
Get the local coefficient data array. For PencilArrays (MPI), returns the
underlying local buffer via `parent()`. For regular arrays, returns as-is.
"""
_local_coeff_data(cd::AbstractArray) = get_local_data(cd)
_local_coeff_data(::Nothing) = nothing

@inline function _subproblem_backend_field(which::Symbol)
    if which === :M
        return :M_backend
    elseif which === :L
        return :L_backend
    elseif which === :pre_right_pinv
        return :pre_right_pinv_backend
    elseif which === :pre_right
        return :pre_right_backend
    elseif which === :pre_left
        return :pre_left_backend
    end
    throw(ArgumentError("Unknown backend cache key: $which"))
end

@inline function _subproblem_selection_field(which::Symbol)
    if which === :pre_right_pinv
        return :pre_right_pinv_indices
    elseif which === :pre_left
        return :pre_left_indices
    end
    throw(ArgumentError("Unknown selection-index cache key: $which"))
end

@inline function _subproblem_vector_field(which::Symbol)
    if which === :compress_variable_space
        return :compress_variable_space
    elseif which === :expand_variable_space
        return :expand_variable_space
    elseif which === :compress_equation_space
        return :compress_equation_space
    elseif which === :gather_inputs_raw
        return :gather_inputs_raw
    elseif which === :scatter_inputs_expanded
        return :scatter_inputs_expanded
    elseif which === :gather_outputs_raw
        return :gather_outputs_raw
    elseif which === :gather_eqn_F_raw
        return :gather_eqn_F_raw
    elseif which === :gather_alg_F_raw
        return :gather_alg_F_raw
    end
    throw(ArgumentError("Unknown vector cache key: $which"))
end

function _subproblem_backend_matrix!(sp::Subproblem, matrix, which::Symbol, data::AbstractVector)
    matrix === nothing && return nothing
    if !is_gpu_array(data)
        return matrix
    end
    field = _subproblem_backend_field(which)
    cached = getfield(sp.runtime, field)
    cached !== nothing && return cached
    backend = _gpu_sparse_csr(matrix, eltype(matrix))
    setfield!(sp.runtime, field, backend)
    return backend
end

"""
    _bc_rows_device(sp, reference)

Return `sp.bc_rows` as an index vector on the same device as `reference`.

For CPU `reference`, returns the CPU `Vector{Int}` as-is. For a GPU
`reference`, returns (and caches) a device-side `Int` array so that vectorized
`rhs[bc_rows] .= coeff .* alg_f[bc_rows]` stays on-device and does not trigger
scalar indexing when `CUDA.allowscalar(false)` is set.
"""
function _bc_rows_device(sp::Subproblem, reference::AbstractVector)
    if isempty(sp.bc_rows) || !is_gpu_array(reference)
        return sp.bc_rows
    end
    cached = sp.runtime.bc_rows_device
    if cached !== nothing && length(cached) == length(sp.bc_rows)
        return cached
    end
    device_rows = similar(reference, Int, length(sp.bc_rows))
    copyto!(device_rows, sp.bc_rows)
    sp.runtime.bc_rows_device = device_rows
    return device_rows
end

"""
    apply_bc_override!(rhs, alg_f, sp, coeff)

Override the algebraic-constraint rows of `rhs` with `coeff * alg_f[bc_rows]`.

Pipeline:

    tmp .= alg_f[bc_rows]     # gather into cached scratch buffer
    tmp .*= coeff             # in-place scale
    rhs[bc_rows] = tmp        # scatter

All three operations are vectorized and stay on the same device as `rhs`, so
this is safe under `CUDA.allowscalar(false)`. The `bc_rows` index vector is
materialized on the correct device by `_bc_rows_device` and cached on the
subproblem to avoid per-step host→device transfers. The gather target
`tmp` is also a cached scratch buffer (`_bc_override_scratch!`), so this
function performs zero per-call allocation after the first call.
"""
function apply_bc_override!(rhs::AbstractVector, alg_f::AbstractVector,
                            sp::Subproblem, coeff)
    isempty(sp.bc_rows) && return rhs
    c = ComplexF64(coeff)
    if is_gpu_array(rhs)
        # GPU path — vectorized scatter via on-device index array. The
        # `alg_f[bc_rows]` gather allocates a small CuArray (length =
        # length(sp.bc_rows), typically ≤ 16), but the allocation is cheap
        # on modern device runtimes and cannot be avoided without a custom
        # kernel. The cached `_bc_rows_device` avoids per-call H2D.
        bc_rows = _bc_rows_device(sp, rhs)
        rhs[bc_rows] = c .* alg_f[bc_rows]
    else
        # CPU path — scalar loop, zero allocation. For small bc_rows counts
        # (typical: 5–20) this is faster than broadcast-gather anyway.
        bc_rows = sp.bc_rows  # CPU Vector{Int}
        @inbounds @simd for i in eachindex(bc_rows)
            r = bc_rows[i]
            rhs[r] = c * alg_f[r]
        end
    end
    return rhs
end

function _subproblem_selection_indices!(sp::Subproblem, matrix::Union{Nothing, SparseMatrixCSC},
                                        which::Symbol)
    matrix === nothing && return nothing
    field = _subproblem_selection_field(which)
    cached = getfield(sp.runtime, field)
    cached !== nothing && return cached

    m, n = size(matrix)
    if nnz(matrix) == 0
        indices = Int[]
        setfield!(sp.runtime, field, indices)
        return indices
    end

    rows, cols, vals = findnz(matrix)
    length(rows) == m || return nothing

    indices = Vector{Int}(undef, m)
    seen = falses(m)
    oneval = one(eltype(matrix))
    @inbounds for k in eachindex(rows)
        r = rows[k]
        c = cols[k]
        if r < 1 || r > m || c < 1 || c > n || seen[r] || vals[k] != oneval
            return nothing
        end
        seen[r] = true
        indices[r] = c
    end
    all(seen) || return nothing

    setfield!(sp.runtime, field, indices)
    return indices
end

function _assign_to_buffer!(dest::AbstractVector{ComplexF64}, src)
    if is_gpu_array(dest)
        src_dev = is_gpu_array(src) ? src : on_architecture(architecture(dest), Array(src))
        if eltype(src_dev) <: Complex
            dest .= src_dev
        else
            dest .= ComplexF64.(src_dev)
        end
    else
        src_cpu = is_gpu_array(src) ? Array(src) : src
        if eltype(src_cpu) <: Complex
            copyto!(dest, src_cpu)
        else
            copyto!(dest, ComplexF64.(src_cpu))
        end
    end
    return dest
end

function _assign_from_buffer!(dest, src)
    if is_gpu_array(dest)
        src_dev = is_gpu_array(src) ? src : on_architecture(architecture(dest), Array(src))
        if eltype(dest) <: Real
            dest .= real.(src_dev)
        else
            dest .= src_dev
        end
    else
        src_cpu = is_gpu_array(src) ? Array(src) : src
        if eltype(dest) <: Real
            dest .= real.(src_cpu)
        else
            dest .= src_cpu
        end
    end
    return dest
end

function _subproblem_cached_vector!(sp::Subproblem, which::Symbol, n::Int;
                                    like::Union{Nothing, AbstractVector}=nothing)
    field = _subproblem_vector_field(which)
    cached = getfield(sp.runtime, field)
    if cached !== nothing && length(cached) == n
        if like === nothing || is_gpu_array(cached) == is_gpu_array(like)
            return cached
        end
    end

    buffer = if like === nothing
        zeros(sp.dist.architecture, ComplexF64, n)
    else
        similar_zeros(like, ComplexF64, n)
    end
    setfield!(sp.runtime, field, buffer)
    return buffer
end

"""Extract per-mode coefficients from all fields into a flat vector (no permutation)."""
function _gather_subproblem_raw!(buffer::AbstractVector{ComplexF64}, sp::Subproblem, fields::Vector)
    kx_global = _kx_index_global(sp)
    offset = 0
    for field in fields
        offset = _gather_field_raw!(buffer, offset, field, kx_global, sp)
    end
    return buffer
end

function _gather_subproblem_raw(sp::Subproblem, fields::Vector)
    total = sum(subproblem_field_size(sp, field) for field in fields)
    buffer = zeros(sp.dist.architecture, ComplexF64, total)
    return _gather_subproblem_raw!(buffer, sp, fields)
end

function _gather_field_raw!(buffer::AbstractVector{ComplexF64}, offset::Int, field::ScalarField, kx_global::Int, sp::Subproblem)
    ensure_layout!(field, :c)
    cd_raw = get_coeff_data(field)
    if cd_raw === nothing
        n = subproblem_field_size(sp, field)
        # Vectorized zero-fill: works on CPU and GPU without scalar indexing.
        if n > 0
            fill!(view(buffer, offset + 1 : offset + n), ComplexF64(0))
        end
        return offset + n
    end
    cd = _local_coeff_data(cd_raw)

    if isempty(field.bases) || all(b -> b === nothing, field.bases)
        dest = view(buffer, offset + 1:offset + 1)
        _assign_to_buffer!(dest, view(cd, 1:1))
        return offset + 1
    elseif ndims(cd) == 1
        # 1D field (tau with Fourier basis): convert to local index
        kx_local = _global_to_local_kx(kx_global, field, sp)
        dest = view(buffer, offset + 1:offset + 1)
        if kx_local >= 1 && kx_local <= length(cd)
            _assign_to_buffer!(dest, view(cd, kx_local:kx_local))
        else
            fill!(dest, ComplexF64(0))
        end
        return offset + 1
    else
        # 2D field: extract local row across all Chebyshev modes
        kx_local = _global_to_local_kx(kx_global, field, sp)
        Nz = size(cd, 2)
        dest = view(buffer, offset + 1:offset + Nz)
        if kx_local >= 1 && kx_local <= size(cd, 1)
            _assign_to_buffer!(dest, selectdim(cd, 1, kx_local))
        else
            fill!(dest, ComplexF64(0))
        end
        return offset + Nz
    end
end

function _gather_field_raw!(buffer::AbstractVector{ComplexF64}, offset::Int, field::VectorField, kx_global::Int, sp::Subproblem)
    for comp in field.components
        offset = _gather_field_raw!(buffer, offset, comp, kx_global, sp)
    end
    return offset
end

"""Write per-mode coefficients back to fields from a flat vector (no permutation)."""
function _scatter_subproblem_raw(sp::Subproblem, data::AbstractVector, fields::Vector)
    kx_global = _kx_index_global(sp)
    offset = 0
    for field in fields
        offset = _scatter_field_raw!(field, data, offset, kx_global, sp)
    end
end

function _scatter_field_raw!(field::ScalarField, data::AbstractVector, offset::Int, kx_global::Int, sp::Subproblem)
    ensure_layout!(field, :c)
    cd_raw = get_coeff_data(field)
    if cd_raw === nothing
        return offset + subproblem_field_size(sp, field)
    end
    cd = _local_coeff_data(cd_raw)

    if isempty(field.bases) || all(b -> b === nothing, field.bases)
        _assign_from_buffer!(view(cd, 1:1), view(data, offset + 1:offset + 1))
        return offset + 1
    elseif ndims(cd) == 1
        kx_local = _global_to_local_kx(kx_global, field, sp)
        if kx_local >= 1 && kx_local <= length(cd)
            _assign_from_buffer!(view(cd, kx_local:kx_local), view(data, offset + 1:offset + 1))
        end
        return offset + 1
    else
        kx_local = _global_to_local_kx(kx_global, field, sp)
        Nz = size(cd, 2)
        if kx_local >= 1 && kx_local <= size(cd, 1)
            _assign_from_buffer!(selectdim(cd, 1, kx_local), view(data, offset + 1:offset + Nz))
        end
        return offset + Nz
    end
end

function _scatter_field_raw!(field::VectorField, data::AbstractVector, offset::Int, kx_global::Int, sp::Subproblem)
    for comp in field.components
        offset = _scatter_field_raw!(comp, data, offset, kx_global, sp)
    end
    return offset
end

function compress_variable_space!(dest::AbstractVector, sp::Subproblem, raw::AbstractVector)
    if sp.pre_right_pinv !== nothing
        indices = (!is_gpu_array(dest) && !is_gpu_array(raw)) ?
                  _subproblem_selection_indices!(sp, sp.pre_right_pinv, :pre_right_pinv) :
                  nothing
        if indices !== nothing
            @inbounds for i in eachindex(indices)
                dest[i] = raw[indices[i]]
            end
        else
            pre = _subproblem_backend_matrix!(sp, sp.pre_right_pinv, :pre_right_pinv, raw)
            if !is_gpu_array(dest) && !is_gpu_array(raw) && pre isa AbstractMatrix
                mul!(dest, pre, raw)
            else
                _assign_to_buffer!(dest, pre * raw)
            end
        end
    else
        _assign_to_buffer!(dest, raw)
    end
    return dest
end

function compress_variable_space(sp::Subproblem, raw::AbstractVector)
    n = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 1) : length(raw)
    dest = _subproblem_cached_vector!(sp, :compress_variable_space, n; like=raw)
    return compress_variable_space!(dest, sp, raw)
end

function expand_variable_space!(dest::AbstractVector, sp::Subproblem, data::AbstractVector)
    if sp.pre_right !== nothing
        indices = (!is_gpu_array(dest) && !is_gpu_array(data)) ?
                  _subproblem_selection_indices!(sp, sp.pre_right_pinv, :pre_right_pinv) :
                  nothing
        if indices !== nothing
            fill!(dest, zero(eltype(dest)))
            @inbounds for i in eachindex(indices)
                dest[indices[i]] = data[i]
            end
        else
            pre = _subproblem_backend_matrix!(sp, sp.pre_right, :pre_right, data)
            if !is_gpu_array(dest) && !is_gpu_array(data) && pre isa AbstractMatrix
                mul!(dest, pre, data)
            else
                _assign_to_buffer!(dest, pre * data)
            end
        end
    else
        _assign_to_buffer!(dest, data)
    end
    return dest
end

function expand_variable_space(sp::Subproblem, data::AbstractVector)
    n = sp.pre_right !== nothing ? size(sp.pre_right, 1) : length(data)
    dest = _subproblem_cached_vector!(sp, :expand_variable_space, n; like=data)
    return expand_variable_space!(dest, sp, data)
end

function compress_equation_space!(dest::AbstractVector, sp::Subproblem, raw::AbstractVector)
    if sp.pre_left !== nothing
        indices = (!is_gpu_array(dest) && !is_gpu_array(raw)) ?
                  _subproblem_selection_indices!(sp, sp.pre_left, :pre_left) :
                  nothing
        if indices !== nothing
            @inbounds for i in eachindex(indices)
                dest[i] = raw[indices[i]]
            end
        else
            pre = _subproblem_backend_matrix!(sp, sp.pre_left, :pre_left, raw)
            if !is_gpu_array(dest) && !is_gpu_array(raw) && pre isa AbstractMatrix
                mul!(dest, pre, raw)
            else
                _assign_to_buffer!(dest, pre * raw)
            end
        end
    else
        _assign_to_buffer!(dest, raw)
    end
    return dest
end

function compress_equation_space(sp::Subproblem, raw::AbstractVector)
    n = sp.pre_left !== nothing ? size(sp.pre_left, 1) : length(raw)
    dest = _subproblem_cached_vector!(sp, :compress_equation_space, n; like=raw)
    return compress_equation_space!(dest, sp, raw)
end

"""Gather per-mode coefficients and compress them into variable space."""
function gather_inputs(sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, :gather_inputs_raw, raw_len)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space(sp, raw)
end

function gather_inputs!(dest::AbstractVector, sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, :gather_inputs_raw, raw_len; like=dest)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space!(dest, sp, raw)
end

"""Expand from variable space and scatter back to fields."""
function scatter_inputs(sp::Subproblem, data::AbstractVector, fields::Vector)
    expanded_len = sp.pre_right !== nothing ? size(sp.pre_right, 1) : length(data)
    expanded = _subproblem_cached_vector!(sp, :scatter_inputs_expanded, expanded_len; like=data)
    expand_variable_space!(expanded, sp, data)
    _scatter_subproblem_raw(sp, expanded, fields)
end

"""
Gather per-mode RHS coefficients in state/variable ordering.

`evaluate_rhs` returns one field per state variable, so the explicit RHS must be
compressed with the variable-side preconditioner, not the equation-side one.
"""
function gather_outputs(sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, :gather_outputs_raw, raw_len)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space(sp, raw)
end

function gather_outputs!(dest::AbstractVector, sp::Subproblem, fields::Vector)
    raw_len = sp.pre_right_pinv !== nothing ? size(sp.pre_right_pinv, 2) :
              sum(subproblem_field_size(sp, field) for field in fields)
    raw = _subproblem_cached_vector!(sp, :gather_outputs_raw, raw_len; like=dest)
    _gather_subproblem_raw!(raw, sp, fields)
    return compress_variable_space!(dest, sp, raw)
end

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
    @debug "gather_alg_F!: unsupported F expression type" expr_type=typeof(expr)
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

"""
    _bc_array_projection(arr, sp)

Project a grid-space array `arr` (from a space-dependent BC) onto the current
subproblem's Fourier mode. Returns a `ComplexF64` value suitable for writing
into the BC row of the raw equation-space vector.

Handles several shape conventions for `arr`:
- `arr` is already the full BC output shape (1D for 2D problems, 2D for
  3D-with-two-periodic-axes problems, etc.) — FFT it directly.
- `arr` is lower-dimensional than the output (e.g., a 1D `sin(x)` in a 3D
  `(x, y, z)` problem) — broadcast it to the full Fourier output shape
  under the assumption that axes beyond `ndims(arr)` are constant.
- `arr` is a scalar-like 1-element array — treat as constant (DC only).

The broadcast/FFT result is cached by identity on `sp.problem.parameters`
via an `IdDict`, so all subproblems sharing the same `ArrayOperator` reuse
a single FFT per refresh.
"""
function _bc_array_projection(arr::AbstractArray, sp::Subproblem)
    (arr === nothing || length(arr) == 0) && return ComplexF64(0)

    fourier_sizes = _bc_fourier_axis_sizes(sp)
    if isempty(fourier_sizes)
        # No Fourier axes at all (pure-coupled / BVP-like). Use arr[1] as
        # the DC-mode value.
        return ComplexF64(first(arr))
    end

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
    raw = _subproblem_cached_vector!(sp, :gather_alg_F_raw, I_raw; like=dest)
    _assign_to_buffer!(raw, raw_cpu)

    compress_equation_space!(dest, sp, raw)
    return dest
end

"""
    check_condition(sp::Subproblem, eq_data)

Check if equation condition is satisfied for this subproblem.
Following subsystems:494-495.
"""
function check_condition(sp::Subproblem, eq_data::Dict)
    condition = get(eq_data, "condition", "true")
    if condition == "true" || condition === nothing || condition == true
        return true
    end
    if condition == "false" || condition == false
        return false
    end

    # Evaluate condition expression with subproblem group dictionary
    # The condition is typically a string expression involving group indices
    # For example: "nx != 0" or "kx == 0 && ky == 0"
    group_dict = sp.group_dict

    # Simple boolean conditions
    if isa(condition, Bool)
        return condition
    end

    # String conditions - evaluate with group variables
    if isa(condition, String)
        condition_str = strip(condition)

        # Handle compound conditions with && and ||
        if occursin("&&", condition_str)
            parts = split(condition_str, "&&")
            return all(check_condition(strip(p), group_dict) for p in parts)
        elseif occursin("||", condition_str)
            parts = split(condition_str, "||")
            return any(check_condition(strip(p), group_dict) for p in parts)
        end

        # Parse simple comparisons like "nx != 0", "kx == 0"
        if occursin("!=", condition_str)
            parts = split(condition_str, "!=")
            if length(parts) == 2
                var_name = strip(parts[1])
                value = tryparse(Int, strip(parts[2]))
                if value !== nothing && haskey(group_dict, var_name)
                    return group_dict[var_name] != value
                elseif value !== nothing && haskey(group_dict, Symbol(var_name))
                    return group_dict[Symbol(var_name)] != value
                end
            end
        elseif occursin("==", condition_str)
            parts = split(condition_str, "==")
            if length(parts) == 2
                var_name = strip(parts[1])
                value = tryparse(Int, strip(parts[2]))
                if value !== nothing && haskey(group_dict, var_name)
                    return group_dict[var_name] == value
                elseif value !== nothing && haskey(group_dict, Symbol(var_name))
                    return group_dict[Symbol(var_name)] == value
                end
            end
        end

        # Unparseable condition — warn loudly rather than silently assuming true
        @warn "Could not parse condition expression: '$condition_str'. Assuming true (equation included)." maxlog=3
    end

    return true
end

"""
    valid_modes(sp::Subproblem, field, valid_modes_array)

Get valid modes for field in this subproblem.
Following subsystems:476-478.
"""
function valid_modes(sp::Subproblem, field, valid_modes_array)
    if valid_modes_array === nothing
        # All modes valid by default
        return ones(Bool, field_size(sp, field))
    end
    slices = field_slices(sp, field)
    return valid_modes_array[slices...]
end
