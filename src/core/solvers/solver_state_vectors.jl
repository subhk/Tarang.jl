# -----------------------------------------------------------------------------
# Field/vector transport for solver and timestepper runtime paths.
#
# Sparse matrix solvers operate on CPU vectors, while Tarang state lives in
# ScalarField objects that may be CPU/GPU-backed and, in MPI runs, distributed
# by PencilArrays. This file owns the conversion boundary and keeps RHS
# evaluation files focused on expression evaluation.
# -----------------------------------------------------------------------------

function _state_vector_transport_mode(fields::Vector{<:ScalarField})
    isempty(fields) && return :empty

    dist = fields[1].dist
    if dist.size > 1 && dist.use_pencil_arrays
        return :mpi_global
    end
    return :local
end

function _fields_vector_size(fields::Vector{<:ScalarField})
    total_size = 0
    for field in fields
        total_size += compute_field_vector_size(field)
    end
    return total_size
end

function _ensure_coeff_layout!(fields::Vector{<:ScalarField})
    isempty(fields) && return nothing

    arch = fields[1].dist.architecture
    for field in fields
        ensure_layout!(field, :c)
    end
    if is_gpu(arch)
        synchronize(arch)
    end
    return arch
end

function _copy_field_data_to_vector!(vector::AbstractVector{ComplexF64}, offset::Int,
                                     field::ScalarField, field_size::Int)
    field_size == 0 && return offset

    end_offset = offset + field_size - 1
    end_offset <= length(vector) || error("Vector too small for field '$(field.name)'")

    if isempty(field.bases)
        vector[offset] = 0
        return offset + field_size
    end

    if get_coeff_data(field) !== nothing
        cpu_data = get_cpu_data(get_coeff_data(field))
        data = vec(cpu_data)
        length(data) == field_size || error("Size mismatch in fields_to_vector! for field '$(field.name)': " *
                                            "expected $field_size elements, got $(length(data)).")
        copyto!(vector, offset, data, 1, field_size)
    elseif get_grid_data(field) !== nothing
        @warn "Using grid space data for field $(field.name) - converting to coefficient space recommended"
        cpu_data = get_cpu_data(get_grid_data(field))
        data = vec(cpu_data)
        length(data) == field_size || error("Size mismatch in fields_to_vector! for field '$(field.name)': " *
                                            "expected $field_size elements, got $(length(data)).")
        copyto!(vector, offset, data, 1, field_size)
    else
        @inbounds for i in offset:end_offset
            vector[i] = 0
        end
    end

    return offset + field_size
end

"""
    fields_to_vector!(vector, fields)

Write coefficient-space field data into a pre-allocated CPU vector.

This is the allocation-free serial/local variant used by timestepper hot paths.
For MPI PencilArray fields, use `fields_to_vector`, which performs the required
global gather into a freshly allocated full vector.
"""
function fields_to_vector!(vector::AbstractVector{ComplexF64}, fields::Vector{<:ScalarField})
    mode = _state_vector_transport_mode(fields)

    if mode === :empty
        isempty(vector) || throw(ArgumentError("fields_to_vector!: empty fields require an empty output vector"))
        return vector
    elseif mode === :mpi_global
        throw(ArgumentError("fields_to_vector! does not support MPI global gather; use fields_to_vector instead"))
    end

    # Ensure all fields are in coefficient space before computing sizes:
    # Real-valued grid layouts and spectral layouts can have different lengths.
    _ensure_coeff_layout!(fields)
    expected_size = _fields_vector_size(fields)
    length(vector) == expected_size ||
        throw(DimensionMismatch("fields_to_vector!: output has length $(length(vector)), expected $expected_size"))

    offset = 1
    for field in fields
        field_size = compute_field_vector_size(field)
        offset = _copy_field_data_to_vector!(vector, offset, field, field_size)
        @debug "Gathered field $(field.name): size=$field_size, offset=$(offset-field_size)"
    end

    @debug "Fields to vector completed: total_size=$expected_size, fields=$(length(fields))"
    return vector
end

function fields_to_vector(fields::Vector{<:ScalarField})
    mode = _state_vector_transport_mode(fields)
    mode === :empty && return Vector{ComplexF64}()

    # Ensure all fields are in coefficient space before computing sizes:
    # Real-valued grid layouts and spectral layouts can have different lengths.
    _ensure_coeff_layout!(fields)

    total_size = _fields_vector_size(fields)

    # Allocate fresh vector each call to avoid shared-buffer aliasing bugs in
    # callers that retain the returned vector.
    vector = Vector{ComplexF64}(undef, total_size)

    offset = 1
    for field in fields
        field_size = compute_field_vector_size(field)
        offset = _copy_field_data_to_vector!(vector, offset, field, field_size)
        @debug "Gathered field $(field.name): size=$field_size, offset=$(offset-field_size)"
    end

    @debug "Fields to vector completed: total_size=$total_size, fields=$(length(fields))"

    if mode === :mpi_global
        vector = _gather_to_global_vector(vector, fields[1].dist)
    end

    return vector
end

"""
    _gather_to_global_vector(local_vector, dist) -> global_vector

Gather local solution vectors from all MPI ranks into a global vector available
on every rank. Used by global-matrix implicit solvers.
"""
function _gather_to_global_vector(local_vector::Vector{ComplexF64}, dist::Distributor)
    local_size = length(local_vector)

    all_sizes = MPI.Allgather(Int32(local_size), dist.comm)
    recv_counts = Int.(all_sizes)
    total_size = sum(recv_counts)

    recv_displs = zeros(Int, length(recv_counts))
    for i in 2:length(recv_counts)
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1]
    end

    global_vector = Vector{ComplexF64}(undef, total_size)
    MPI.Allgatherv!(local_vector, MPI.VBuffer(global_vector, recv_counts), dist.comm)

    return global_vector
end

function _solution_vector_start_offset(fields::Vector{<:ScalarField}, mode::Symbol)
    mode === :mpi_global || return 1

    dist = fields[1].dist
    local_size = sum(compute_field_vector_size(f) for f in fields)
    all_sizes = MPI.Allgather(Int32(local_size), dist.comm)
    rank_offset = sum(Int.(all_sizes[1:dist.rank]))  # dist.rank is 0-indexed.
    return rank_offset + 1
end

"""
    copy_solution_to_fields!(fields, solution)

Copy a solution vector back into fields following the inverse transport path
used by `fields_to_vector`.
"""
function copy_solution_to_fields!(fields::Vector{<:ScalarField}, solution::AbstractVector{<:Number})
    mode = _state_vector_transport_mode(fields)
    mode === :empty && return nothing

    offset = _solution_vector_start_offset(fields, mode)

    for field in fields
        field_size = compute_field_vector_size(field)

        if field_size > 0 && offset <= length(solution)
            end_offset = min(offset + field_size - 1, length(solution))
            actual_size = end_offset - offset + 1

            if actual_size > 0
                field_data = @view solution[offset:end_offset]
                set_field_data_from_vector!(field, field_data)
                @debug "Scattered to field $(field.name): size=$actual_size"
            end

            offset += field_size
        end
    end

    arch = fields[1].dist.architecture
    if is_gpu(arch)
        synchronize(arch)
    end

    @debug "Vector to fields completed: solution_size=$(length(solution)), fields=$(length(fields))"
    return nothing
end

"""
    vector_to_fields(vector, template)

Convert a solution vector to a new state vector matching a template.
"""
function vector_to_fields(vector::AbstractVector{<:Number}, template::Vector{<:ScalarField})
    new_state = ScalarField[]
    offset = 1

    for field in template
        new_field = ScalarField(field.dist, field.name, field.bases, field.dtype)
        field_size = compute_field_vector_size(field)

        if field_size > 0 && offset <= length(vector)
            end_offset = min(offset + field_size - 1, length(vector))
            data_slice = @view vector[offset:end_offset]
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
No field allocation. Output fields must already have coefficient data allocated.
"""
function vector_to_fields!(output::Vector{<:ScalarField}, vector::AbstractVector{<:Number},
                           template::Vector{<:ScalarField})
    offset = 1
    for (i, _) in enumerate(template)
        coeff_data = get_coeff_data(output[i])
        if coeff_data === nothing
            # The output field may carry only grid data (e.g. a state produced by
            # `copy_state` of a grid-layout field — `evaluate_rhs` leaves the
            # multistep state in :g). Allocate its coefficient buffer so the
            # solution vector is actually written; skipping here silently leaves
            # the field frozen at its copied value (degrades multistep to 1st order).
            ensure_layout!(output[i], :c)
            coeff_data = get_coeff_data(output[i])
            coeff_data === nothing && continue
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
    compute_field_vector_size(field)

Compute the number of degrees of freedom for a field in vector form.
"""
function compute_field_vector_size(field::ScalarField)
    if isempty(field.bases)
        return 1
    end

    if get_coeff_data(field) !== nothing
        return length(get_coeff_data(field))
    elseif get_grid_data(field) !== nothing
        return length(get_grid_data(field))
    else
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
    extract_field_data_for_vector(field)

Extract field data for vector conversion with proper layout handling.
"""
function extract_field_data_for_vector(field::ScalarField)
    if isempty(field.bases)
        return zeros(ComplexF64, 1)
    end

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
    set_field_data_from_vector!(field, data)

Set field data from vector storage, including CPU/GPU transfer and layout
bookkeeping.
"""
function set_field_data_from_vector!(field::ScalarField, data::AbstractVector{<:Number})
    # 0-D tau fields have no spatial storage (length-0 sentinel); their DOF lives
    # in the matrix system, not field data. Skip — matches the isempty(bases) guard
    # in compute_field_vector_size / extract_field_data_for_vector, and avoids
    # spuriously flipping the tau field's layout to :c.
    isempty(field.bases) && return nothing
    if get_coeff_data(field) !== nothing
        target_shape = size(get_coeff_data(field))
        expected_size = prod(target_shape)

        if length(data) == expected_size
            data_slice = data
        elseif length(data) < expected_size
            temp_data = zeros(ComplexF64, expected_size)
            temp_data[1:length(data)] .= data
            data_slice = temp_data
            @debug "Padded field $(field.name) data: got $(length(data)), expected $expected_size"
        else
            data_slice = @view data[1:expected_size]
            @debug "Truncated field $(field.name) data: got $(length(data)), expected $expected_size"
        end

        target_eltype = eltype(get_coeff_data(field))
        if target_eltype <: Real
            if any(x -> !iszero(imag(x)), data_slice)
                @warn "Discarding imaginary part when setting real field $(field.name)"
            end
            reshaped_data = reshape(real.(data_slice), target_shape)
        elseif eltype(data_slice) <: target_eltype
            reshaped_data = reshape(data_slice, target_shape)
        else
            reshaped_data = reshape(convert.(target_eltype, data_slice), target_shape)
        end

        arch = field.dist.architecture
        if is_gpu(arch)
            gpu_data = on_architecture(arch, reshaped_data)
            copyto!(get_coeff_data(field), gpu_data)
        else
            copyto!(get_coeff_data(field), reshaped_data)
        end

        field.current_layout = :c

    elseif get_grid_data(field) !== nothing
        target_shape = size(get_grid_data(field))
        expected_size = prod(target_shape)

        if length(data) == expected_size
            target_eltype = eltype(get_grid_data(field))
            if target_eltype <: Real
                if any(x -> !iszero(imag(x)), data)
                    @warn "Discarding imaginary part when setting real grid field $(field.name)"
                end
                reshaped_data = reshape(real.(data), target_shape)
            elseif eltype(data) <: target_eltype
                reshaped_data = reshape(data, target_shape)
            else
                reshaped_data = reshape(convert.(target_eltype, data), target_shape)
            end

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
    elseif !isempty(field.bases)
        @warn "Cannot set data for field $(field.name) - no data arrays allocated"
    end

    return nothing
end

"""Get the size (number of modes) for a basis following Tarang patterns."""
function get_basis_size(basis)
    if hasfield(typeof(basis), :meta) && hasfield(typeof(basis.meta), :size)
        return basis.meta.size
    elseif hasfield(typeof(basis), :shape)
        shape = basis.shape
        if isa(shape, Tuple)
            return prod(shape)
        else
            return shape
        end
    elseif hasfield(typeof(basis), :size)
        return basis.size
    elseif hasfield(typeof(basis), :N)
        return basis.N
    else
        @warn "Could not determine basis size for $(typeof(basis)), using default"
        return 64
    end
end
