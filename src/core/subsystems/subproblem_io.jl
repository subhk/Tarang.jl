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
    if cd_raw === nothing || isempty(cd_raw)
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
        if any(b -> b !== nothing && !isa(b, FourierBasis), field.bases)
            # Pure 1D coupled field (e.g. a single Chebyshev/Jacobi axis, no
            # separable Fourier direction): there is no Fourier mode to select,
            # so gather the ENTIRE coefficient spectrum for this subproblem.
            n = length(cd)
            dest = view(buffer, offset + 1:offset + n)
            _assign_to_buffer!(dest, cd)
            return offset + n
        end
        # 1D Fourier field (tau on a Fourier basis): select the single kx mode.
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
    if cd_raw === nothing || isempty(cd_raw)
        return offset + subproblem_field_size(sp, field)
    end
    cd = _local_coeff_data(cd_raw)

    if isempty(field.bases) || all(b -> b === nothing, field.bases)
        _assign_from_buffer!(view(cd, 1:1), view(data, offset + 1:offset + 1))
        return offset + 1
    elseif ndims(cd) == 1
        if any(b -> b !== nothing && !isa(b, FourierBasis), field.bases)
            # Pure 1D coupled field: write back the ENTIRE coefficient spectrum
            # (mirror of the gather path above).
            n = length(cd)
            _assign_from_buffer!(view(cd, 1:n), view(data, offset + 1:offset + n))
            return offset + n
        end
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
