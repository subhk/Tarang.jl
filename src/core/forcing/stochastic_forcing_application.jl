"""
    stochastic_forcing_application

Applying a realization to a solution vector, and the work/energy-injection
bookkeeping that goes with it.

Split out of the former 1767-line `stochastic_forcing.jl`; the sections are
unchanged, only relocated.
"""

# ============================================================================
# Forcing Application and Work Calculation
# ============================================================================

"""
    apply_forcing!(rhs::AbstractArray, forcing::StochasticForcing, t::Real, substep::Int=1)

Add stochastic forcing to the RHS in spectral space. Works on both CPU and GPU.

## Arguments

- `rhs`: Right-hand side array (modified in-place)
- `forcing`: StochasticForcing configuration
- `t`: Current simulation time
- `substep`: Current substep number
"""
function apply_forcing!(
    rhs::AbstractArray{Complex{T}, N},
    forcing::StochasticForcing{T, N, A, CA},
    t::Real,
    substep::Int=1
) where {T, N, A, CA}

    F = generate_forcing!(forcing, t, substep)
    # Pass `rhs` itself (not size(rhs)): under MPI rhs is a PencilArray, so this
    # selects the offset-aware PencilArray method (slices the rank's axes_local).
    # Passing the NTuple `size(rhs)` instead routed to the offset-BLIND method
    # (ranges = 1:local_n), injecting forcing at the wrong global wavenumbers on
    # every rank>0 (or throwing on a non-matching local slab shape).
    F_view = _matched_forcing_view(forcing, rhs)
    if F_view === nothing
        throw(ArgumentError("Forcing size $(size(F)) does not match RHS size $(size(rhs))"))
    end

    # Use broadcasting for GPU compatibility
    rhs .+= F_view
    return rhs
end

"""
    store_prevsol!(forcing::StochasticForcing, sol::AbstractArray)

Store the current solution for Stratonovich work calculation.
Works on both CPU and GPU arrays.

Call this at the **beginning** of each timestep, before advancing.
"""
function store_prevsol!(forcing::StochasticForcing{T, N, A, CA}, sol::AbstractArray{Complex{T}, N}) where {T, N, A, CA}
    if sol isa PencilArrays.PencilArray
        # Store the local slab as a plain N-D Array whose CARTESIAN layout matches the
        # PencilArray's logical getindex (pa[I] is logical, verified), so the
        # work/power reductions — which iterate CartesianIndices — pair it correctly.
        # Note: a `[sol[I] for I in CartesianIndices(sol)]` comprehension does NOT do
        # this (it collects in the PencilArray's LINEAR/storage order == collect(sol),
        # transposing a permuted pencil); an explicit `dest[I] = sol[I]` loop does.
        # Round-5 audit 2026-06-23.
        if forcing.prevsol === nothing || size(forcing.prevsol) != size(sol)
            forcing.prevsol = Array{Complex{T}, N}(undef, size(sol))
        end
        dest = forcing.prevsol
        @inbounds for I in CartesianIndices(sol)
            dest[I] = sol[I]
        end
        forcing.prevsol = dest
        return
    end
    if forcing.prevsol === nothing || size(forcing.prevsol) != size(sol)
        # Allocate matching the solution shape (not forcing shape — they can differ
        # when forcing is on the full complex grid but solution uses RFFT half-grid)
        forcing.prevsol = similar(sol)
    end
    # Use copyto! for GPU compatibility
    copyto!(forcing.prevsol, sol)
end

function _diagnostic_layout(target::AbstractArray{<:Any, N}) where N
    return size(target), ntuple(d -> 1:size(target, d), N)
end

function _diagnostic_layout(target::PencilArrays.PencilArray)
    return Tuple(PencilArrays.size_global(target)), Tuple(PencilArrays.pencil(target).axes_local)
end

@inline function _rfft_multiplicity(::Type{T}, full_n::Int, target_n::Int,
                                    global_i::Int) where T
    if target_n == full_n
        return one(T)
    elseif full_n == 2 * (target_n - 1) # even physical size: DC and Nyquist are unique
        return (global_i == 1 || global_i == target_n) ? one(T) : T(2)
    elseif full_n == 2 * target_n - 1   # odd physical size: only DC is unique
        return global_i == 1 ? one(T) : T(2)
    end
    throw(ArgumentError("Forcing size $full_n does not match spectral target size $target_n"))
end

function _diagnostic_weights(forcing::StochasticForcing{T, N},
                             global_shape::NTuple{N, Int},
                             local_ranges::NTuple{N, UnitRange{Int}}) where {T, N}
    local_shape = ntuple(d -> length(local_ranges[d]), N)
    weights = Array{T}(undef, local_shape)
    M2 = T(prod(forcing.field_size))^2

    for I in CartesianIndices(weights)
        multiplicity = one(T)
        k2 = zero(T)
        for d in 1:N
            global_i = first(local_ranges[d]) + I[d] - 1
            multiplicity *= _rfft_multiplicity(T, forcing.field_size[d],
                                                global_shape[d], global_i)
            kd = forcing.wavenumbers[d][global_i]
            k2 += kd^2
        end
        weights[I] = multiplicity *
                     _injection_metric_weight(k2, forcing.injection_metric) / M2
    end
    return weights
end

function _validate_diagnostic_layout(forcing::StochasticForcing{T, N},
                                     global_shape::NTuple{N, Int},
                                     local_ranges::NTuple{N, UnitRange{Int}}) where {T, N}
    for d in 1:N
        r = local_ranges[d]
        (first(r) < 1 || last(r) > global_shape[d]) &&
            throw(ArgumentError("Local spectral range $r exceeds target size $(global_shape[d])"))
        _rfft_multiplicity(T, forcing.field_size[d], global_shape[d], first(r))
    end
    return nothing
end

function _cached_gpu_diagnostic_weights!(forcing::StochasticForcing{T, N, A},
                                         global_shape::NTuple{N, Int},
                                         local_ranges::NTuple{N, UnitRange{Int}}) where {T, N, A}
    if forcing.diagnostic_weights === nothing ||
       forcing.diagnostic_global_shape != global_shape ||
       forcing.diagnostic_local_ranges != local_ranges ||
       forcing.diagnostic_metric !== forcing.injection_metric
        weights_cpu = _diagnostic_weights(forcing, global_shape, local_ranges)
        forcing.diagnostic_weights = on_architecture(forcing.architecture, weights_cpu)
        forcing.diagnostic_global_shape = global_shape
        forcing.diagnostic_local_ranges = local_ranges
        forcing.diagnostic_metric = forcing.injection_metric
    end
    return forcing.diagnostic_weights::A
end

@inline function _diagnostic_mode_weight(forcing::StochasticForcing{T, N},
                                         global_shape::NTuple{N, Int},
                                         G::CartesianIndex{N}, M2::T) where {T, N}
    multiplicity = one(T)
    k2 = zero(T)
    for d in 1:N
        global_i = G[d]
        multiplicity *= _rfft_multiplicity(T, forcing.field_size[d],
                                            global_shape[d], global_i)
        kd = forcing.wavenumbers[d][global_i]
        k2 += kd^2
    end
    return multiplicity * _injection_metric_weight(k2, forcing.injection_metric) / M2
end

function _diagnostic_inner_product_cpu(forcing::StochasticForcing{T, N},
                                       sol::AbstractArray{Complex{T}, N},
                                       global_shape::NTuple{N, Int},
                                       local_ranges::NTuple{N, UnitRange{Int}},
                                       other::Nothing) where {T, N}
    M2 = T(prod(forcing.field_size))^2
    offset = CartesianIndex(ntuple(d -> first(local_ranges[d]) - 1, N))
    value = zero(T)
    @inbounds for I in CartesianIndices(sol)
        G = I + offset
        weight = _diagnostic_mode_weight(forcing, global_shape, G, M2)
        value += weight * real(sol[I] * conj(forcing.cached_forcing[G]))
    end
    return T(_forcing_reduce_partial(sol, value))
end

function _diagnostic_inner_product_cpu(forcing::StochasticForcing{T, N},
                                       sol::AbstractArray{Complex{T}, N},
                                       global_shape::NTuple{N, Int},
                                       local_ranges::NTuple{N, UnitRange{Int}},
                                       other::AbstractArray{Complex{T}, N}) where {T, N}
    M2 = T(prod(forcing.field_size))^2
    offset = CartesianIndex(ntuple(d -> first(local_ranges[d]) - 1, N))
    value = zero(T)
    @inbounds for I in CartesianIndices(sol)
        G = I + offset
        weight = _diagnostic_mode_weight(forcing, global_shape, G, M2)
        midpoint = (other[I] + sol[I]) / T(2)
        value += weight * real(midpoint * conj(forcing.cached_forcing[G]))
    end
    return T(_forcing_reduce_partial(sol, value))
end

function _diagnostic_inner_product_gpu(forcing::StochasticForcing{T, N},
                                       sol::AbstractArray{Complex{T}, N},
                                       global_shape::NTuple{N, Int},
                                       local_ranges::NTuple{N, UnitRange{Int}},
                                       other::Nothing) where {T, N}
    weights = _cached_gpu_diagnostic_weights!(forcing, global_shape, local_ranges)
    cf = view(forcing.cached_forcing, local_ranges...)
    partial = mapreduce((w, s, f) -> w * real(s * conj(f)), +,
                        weights, sol, cf; init=zero(T))
    # Sum this rank's slab into the global value, like the CPU path. Without the
    # reduce a distributed-GPU diagnostic would report only this rank's fraction.
    return T(_forcing_reduce_partial(sol, partial))
end

function _diagnostic_inner_product_gpu(forcing::StochasticForcing{T, N},
                                       sol::AbstractArray{Complex{T}, N},
                                       global_shape::NTuple{N, Int},
                                       local_ranges::NTuple{N, UnitRange{Int}},
                                       other::AbstractArray{Complex{T}, N}) where {T, N}
    weights = _cached_gpu_diagnostic_weights!(forcing, global_shape, local_ranges)
    cf = view(forcing.cached_forcing, local_ranges...)
    partial = mapreduce((w, a, b, f) -> w * real((a + b) / T(2) * conj(f)), +,
                        weights, other, sol, cf; init=zero(T))
    # See the other GPU method: reduce this rank's slab into the global value.
    return T(_forcing_reduce_partial(sol, partial))
end

function _diagnostic_inner_product(forcing::StochasticForcing{T, N},
                                   sol::AbstractArray{Complex{T}, N},
                                   other::Union{Nothing, AbstractArray{Complex{T}, N}}=nothing) where {T, N}
    global_shape, local_ranges = _diagnostic_layout(sol)
    _validate_diagnostic_layout(forcing, global_shape, local_ranges)
    if is_gpu(forcing.architecture) || is_gpu_array(sol)
        return _diagnostic_inner_product_gpu(forcing, sol, global_shape, local_ranges, other)
    end
    return _diagnostic_inner_product_cpu(forcing, sol, global_shape, local_ranges, other)
end

"""
    work_stratonovich(forcing::StochasticForcing, sol::AbstractArray)

Compute work done by forcing using Stratonovich interpretation.
Works on both CPU and GPU arrays.

## Formula

    W = Re⟨(ψⁿ + ψⁿ⁺¹)/2 · ΔF̂*⟩

where ΔF̂ = √Q̂ · ξ · √dt is the forcing increment over dt.

This correctly accounts for the correlation between forcing and response.

## Arguments

- `forcing`: StochasticForcing with prevsol stored
- `sol`: Current solution ψⁿ⁺¹

## Returns

Work done during this timestep (scalar, units of energy).
"""
function work_stratonovich(forcing::StochasticForcing{T, N, A, CA}, sol::AbstractArray{Complex{T}, N}) where {T, N, A, CA}
    if forcing.prevsol === nothing
        return zero(T)
    end

    # Stratonovich work: W = Re⟨ψ_mid · ΔF̂*⟩ where ΔF̂ = F̂_stored · dt
    # Since F̂_stored = √Q̂ · ξ / √dt, we have ΔF̂ = √Q̂ · ξ · √dt
    ps = forcing.prevsol
    size(ps) == size(sol) || throw(ArgumentError("Previous solution size $(size(ps)) does not match current solution size $(size(sol))"))
    work = _diagnostic_inner_product(forcing, sol, ps)

    # The cached_forcing stores F̂ = √Q̂ · ξ / √dt
    # The forcing increment is ΔF̂ = F̂ · dt = √Q̂ · ξ · √dt
    # Parseval contributes M⁻² for Tarang's unnormalised FFT coefficients.
    return T(work * forcing.dt)
end

"""
    work_ito(forcing::StochasticForcing, sol::AbstractArray)

Compute work done by forcing using Itô interpretation.
Works on both CPU and GPU arrays.

## Formula

    W_Itô = Re⟨ψⁿ · ΔF̂*⟩ + ε · dt

where ΔF̂ = √Q̂ · ξ · √dt is the forcing increment.

The drift correction ε · dt accounts for the Itô-Stratonovich conversion.
In Itô calculus, ψⁿ is independent of Fⁿ⁺¹, so ⟨ψⁿ · ΔF̂⟩ = 0.
The drift ensures ⟨W_Itô⟩ = ⟨W_Stratonovich⟩ = ε · dt.
"""
function work_ito(forcing::StochasticForcing{T, N, A, CA}, sol_prev::AbstractArray{Complex{T}, N}) where {T, N, A, CA}
    work = _diagnostic_inner_product(forcing, sol_prev)

    # The Itô integral has zero mean, so we add drift correction
    # to match Stratonovich mean: ⟨W_Itô⟩ = 0 + ε·dt = ε·dt
    drift = forcing.energy_injection_rate * forcing.dt

    return T(work * forcing.dt + drift)
end

