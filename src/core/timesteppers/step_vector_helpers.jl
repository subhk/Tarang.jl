# Shared vector, linear solve, and history-buffer helpers for timestepper paths.

@inline function _timestep_ldiv!(dest::AbstractVector, lhs::F,
                                rhs::AbstractVector) where {F}
    ldiv!(dest, lhs, rhs)
    return dest
end

function _timestep_fields_vector!(state::TimestepperState, key::Symbol,
                                  fields::Vector{<:ScalarField})
    _ensure_coeff_layout!(fields)
    vector = _timestep_vector_buffer!(state, key, _fields_vector_size(fields))
    return fields_to_vector!(vector, fields)
end

function _timestep_matvec!(state::TimestepperState, key::Symbol,
                           matrix::AbstractMatrix, vector::AbstractVector{ComplexF64})
    dest = _timestep_vector_buffer!(state, key, size(matrix, 1))
    mul!(dest, matrix, vector)
    return dest
end

function _prepend_history_buffer!(history::Vector{Vector{ComplexF64}},
                                  scratch::Vector{ComplexF64}, max_len::Int)
    max_len <= 0 && return history

    if length(history) >= max_len
        slot = pop!(history)
        if length(slot) != length(scratch)
            slot = Vector{ComplexF64}(undef, length(scratch))
        end
    else
        slot = Vector{ComplexF64}(undef, length(scratch))
    end

    copyto!(slot, scratch)
    pushfirst!(history, slot)
    return history
end
