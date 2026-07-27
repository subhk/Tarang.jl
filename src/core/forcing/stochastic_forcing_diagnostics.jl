"""
    stochastic_forcing_diagnostics

Diagnostics, utility helpers and internal shape/layout plumbing.

Split out of the former 1767-line `stochastic_forcing.jl`; the sections are
unchanged, only relocated.
"""

# ============================================================================
# Diagnostics
# ============================================================================

"""
    mean_energy_injection_rate(forcing::StochasticForcing)

Return the target (mean) energy injection rate ε.

This is the ensemble average of work done per unit time.
"""
mean_energy_injection_rate(forcing::StochasticForcing) = forcing.energy_injection_rate
mean_energy_injection_rate(forcing::SeparableStochasticForcing) = forcing.energy_injection_rate

energy_injection_rate(forcing::StochasticForcing) = forcing.energy_injection_rate
energy_injection_rate(forcing::SeparableStochasticForcing) = forcing.energy_injection_rate

function get_forcing_real(forcing::StochasticForcing)
    return real.(forcing.cached_forcing)
end

"""
    instantaneous_power(forcing::StochasticForcing, sol::AbstractArray)

Compute instantaneous power input from the Parseval-normalized spectral pairing.
Works on both CPU and GPU arrays.

This is the correlation between the solution and forcing at a given instant.
For white noise forcing, the expected value depends on which solution is passed:
- ⟨P⟩ = 0 if sol is the solution BEFORE the forcing was applied (independent)
- ⟨P⟩ = ε if sol is the MIDPOINT (ψⁿ + ψⁿ⁺¹)/2
- ⟨P⟩ = 2ε if sol is the solution AFTER forcing (includes full response)

Note: For the Stratonovich-consistent time-averaged power over the timestep,
use `work_stratonovich(forcing, sol) / forcing.dt` instead.

## Returns

Instantaneous power (energy per unit time).
"""
function instantaneous_power(forcing::StochasticForcing{T, N, A, CA}, sol::AbstractArray{Complex{T}, N}) where {T, N, A, CA}
    return _diagnostic_inner_product(forcing, sol)
end

"""
    forcing_enstrophy_injection_rate(forcing::StochasticForcing)

Compute mean enstrophy injection rate (for 2D turbulence).

This assumes the stochastic forcing is applied directly to vorticity. With
`M = prod(field_size)` and Tarang's unnormalized FFT convention,

    η = ∑_k Q̂(k) / (2M²)

For `injection_metric=:direct`, this equals the configured energy injection
rate. For `:vorticity_kinetic`, it is generally of order `k_forcing²` times
the configured kinetic-energy injection rate.
"""
function forcing_enstrophy_injection_rate(forcing::StochasticForcing{T, N, A, CA}) where {T, N, A, CA}
    if N != 2
        @warn "Enstrophy injection rate is only meaningful for 2D"
        return zero(T)
    end

    M = T(prod(forcing.field_size))
    return sum(abs2, forcing.forcing_spectrum) / (2 * M^2)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    set_dt!(forcing::StochasticForcing, dt::Real)

Update the timestep for Stratonovich scaling.

Call this when dt changes (e.g., adaptive timestepping).
"""
function set_dt!(forcing::StochasticForcing{T, N, A, CA}, dt::Real) where {T, N, A, CA}
    new_dt = T(dt)
    if forcing.dt != new_dt
        forcing.dt = new_dt
        forcing.last_update_time = T(-Inf)
        fill!(forcing.cached_forcing, zero(Complex{T}))
    end
    return forcing
end

function set_dt!(forcing::SeparableStochasticForcing{T}, dt::Real) where T
    new_dt = T(dt)
    if forcing.dt != new_dt
        forcing.dt = new_dt
        forcing.last_update_time = T(-Inf)
        fill!(forcing.cached_forcing, zero(eltype(forcing.cached_forcing)))
        fill!(forcing.fourier_realization, zero(eltype(forcing.fourier_realization)))
    end
    return forcing
end

"""
    reset_forcing!(forcing::StochasticForcing)

Reset the forcing cache, causing regeneration on next call.
Works on both CPU and GPU.
"""
function reset_forcing!(forcing::StochasticForcing{T, N, A, CA}) where {T, N, A, CA}
    forcing.last_update_time = T(-Inf)
    # Use fill! which works on both CPU and GPU arrays
    fill!(forcing.cached_forcing, zero(Complex{T}))
    if forcing.prevsol !== nothing
        fill!(forcing.prevsol, zero(Complex{T}))
    end
end

function reset_forcing!(forcing::SeparableStochasticForcing{T}) where T
    forcing.last_update_time = T(-Inf)
    fill!(forcing.cached_forcing, zero(eltype(forcing.cached_forcing)))
    fill!(forcing.fourier_realization, zero(eltype(forcing.fourier_realization)))
    if forcing.prevsol !== nothing
        fill!(forcing.prevsol, zero(eltype(forcing.prevsol)))
    end
    return forcing
end

"""
    get_forcing_spectrum(forcing::StochasticForcing)

Return the forcing amplitude spectrum √Q̂(k).
"""
get_forcing_spectrum(forcing::StochasticForcing) = forcing.forcing_spectrum
get_forcing_spectrum(forcing::SeparableStochasticForcing) = forcing.forcing_spectrum

"""
    get_cached_forcing(forcing::StochasticForcing)

Return the current cached forcing F̂(k).
"""
get_cached_forcing(forcing::StochasticForcing) = forcing.cached_forcing
get_cached_forcing(forcing::SeparableStochasticForcing) = forcing.cached_forcing

# ============================================================================
# Internal helpers
# ============================================================================

# Sum a per-rank partial scalar (computed over the LOCAL PencilArray slab)
# across the distributed field's MPI communicator. The work/power diagnostics
# reduce over the owned slab and use the GLOBAL transform size for Parseval
# normalization, so the slab partials must be combined first.
# Serial / single-rank / non-PencilArray inputs are returned unchanged, so
# existing serial and degenerate (single-rank) configurations are untouched.
_forcing_reduce_partial(::AbstractArray, partial) = partial

function _forcing_reduce_partial(sol::PencilArrays.PencilArray, partial)
    MPI.Initialized() || return partial
    comm = PencilArrays.get_comm(PencilArrays.pencil(sol))
    MPI.Comm_size(comm) > 1 || return partial
    return MPI.Allreduce(partial, MPI.SUM, comm)
end

function _matched_forcing_view(forcing::StochasticForcing{T, N, A, CA},
                               target_shape::NTuple{N, Int}) where {T, N, A, CA}
    forcing_shape = size(forcing.cached_forcing)
    if forcing_shape == target_shape
        return forcing.cached_forcing
    end

    ranges = Vector{UnitRange{Int}}(undef, N)
    for d in 1:N
        if forcing_shape[d] == target_shape[d]
            ranges[d] = 1:target_shape[d]
        elseif forcing_shape[d] == 2 * (target_shape[d] - 1) ||
               forcing_shape[d] == 2 * target_shape[d] - 1
            ranges[d] = 1:target_shape[d]
        else
            return nothing
        end
    end

    # view() works on both CPU and GPU arrays
    return view(forcing.cached_forcing, Tuple(ranges)...)
end

function _matched_forcing_view(forcing::StochasticForcing{T, N, A, CA},
                               target::AbstractArray) where {T, N, A, CA}
    return _matched_forcing_view(forcing, size(target))
end

function _matched_forcing_view(forcing::StochasticForcing{T, N, A, CA},
                               target::PencilArrays.PencilArray) where {T, N, A, CA}
    # `axes_local` is in LOGICAL (physical-dim) order, so index it by the
    # logical/physical dim directly to slice the logical-order global
    # cached_forcing — exactly as _integrate_full_distributed slices its global
    # weight array (operations_integrate.jl:187). The consumer broadcasts against
    # the PencilArray's PARENT (memory/storage) order, so the logical-order view
    # is finally permuted into storage order via the pencil permutation.
    #
    # The previous code indexed axes_local by the PERMUTED storage position
    # (findfirst(==(physical_dim), perm)), which both mis-sliced the spectrum
    # (forcing injected at the wrong wavenumbers under MPI, empirically ERR≈9 at
    # np=2) and could not broadcast on non-square local blocks (DimensionMismatch).
    # Tuple(NoPermutation()) is `nothing`, treated as the identity permutation.
    local_axes = PencilArrays.pencil(target).axes_local
    length(local_axes) == N || return nothing

    forcing_shape = size(forcing.cached_forcing)
    ranges = Vector{UnitRange{Int}}(undef, N)

    for physical_dim in 1:N
        local_range = local_axes[physical_dim]
        if first(local_range) < 1 || last(local_range) > forcing_shape[physical_dim]
            return nothing
        end
        ranges[physical_dim] = local_range
    end

    logical_view = view(forcing.cached_forcing, Tuple(ranges)...)

    perm_raw = Tuple(PencilArrays.permutation(target))
    perm_raw === nothing && return logical_view   # identity permutation
    return PermutedDimsArray(logical_view, perm_raw)
end

function _separable_forcing_ranges(
    forcing_shape::NTuple{N,Int},
    target_shape::NTuple{N,Int},
    fourier_dims::Int,
) where N
    ranges = Vector{UnitRange{Int}}(undef, N)
    for d in 1:fourier_dims
        nf, nt = forcing_shape[d], target_shape[d]
        if nf == nt || nf == 2 * (nt - 1) || nf == 2 * nt - 1
            ranges[d] = 1:nt
        else
            return nothing
        end
    end

    chebyshev_dim = fourier_dims + 1
    forcing_shape[chebyshev_dim] == target_shape[chebyshev_dim] || return nothing
    ranges[chebyshev_dim] = 1:target_shape[chebyshev_dim]
    return Tuple(ranges)
end

function _matched_forcing_view(
    forcing::SeparableStochasticForcing{T,NF,N},
    target_shape::NTuple{N,Int},
) where {T,NF,N}
    ranges = _separable_forcing_ranges(size(forcing.cached_forcing), target_shape, NF)
    ranges === nothing && return nothing
    return view(forcing.cached_forcing, ranges...)
end

function _matched_forcing_view(
    forcing::SeparableStochasticForcing{T,NF,N},
    target::AbstractArray,
) where {T,NF,N}
    ndims(target) == N || return nothing
    return _matched_forcing_view(forcing, size(target))
end


function _matched_forcing_view(
    forcing::SeparableStochasticForcing{T,NF,N},
    target::PencilArrays.PencilArray,
) where {T,NF,N}
    global_shape = Tuple(PencilArrays.size_global(target))
    _separable_forcing_ranges(size(forcing.cached_forcing), global_shape, NF) === nothing &&
        return nothing

    local_axes = PencilArrays.pencil(target).axes_local
    length(local_axes) == N || return nothing
    forcing_shape = size(forcing.cached_forcing)
    ranges = ntuple(N) do d
        local_range = local_axes[d]
        first(local_range) >= 1 && last(local_range) <= forcing_shape[d] ||
            return nothing
        local_range
    end
    any(isnothing, ranges) && return nothing

    logical_view = view(forcing.cached_forcing, ranges...)
    perm_raw = Tuple(PencilArrays.permutation(target))
    perm_raw === nothing && return logical_view
    return PermutedDimsArray(logical_view, perm_raw)
end

"""
    _fill_random_phases!(arch, phases, rng)

Fill an array with random phases in `[0, 2π)`. The caller RNG supplies one
scalar seed, then a counter-based kernel expands it on `arch`. Seeded CPU and
GPU runs therefore agree without moving a field-sized phase array through CPU
memory.
"""
@inline function _phase_splitmix64(x::UInt64)
    z = x + UInt64(0x9e3779b97f4a7c15)
    z = xor(z, z >> 30) * UInt64(0xbf58476d1ce4e5b9)
    z = xor(z, z >> 27) * UInt64(0x94d049bb133111eb)
    return xor(z, z >> 31)
end

@kernel function _fill_random_phase_kernel!(phases, seed::UInt64, two_pi)
    i = @index(Global, Linear)
    if i <= length(phases)
        bits = _phase_splitmix64(xor(seed, UInt64(i)))
        u = eltype(phases)(bits >> 11) * eltype(phases)(1.1102230246251565e-16)
        @inbounds phases[i] = u * two_pi
    end
end

function _fill_random_phases!(arch::AbstractArchitecture, phases::AbstractArray{T}, rng::AbstractRNG) where {T}
    seed = rand(rng, UInt64)
    launch!(arch, _fill_random_phase_kernel!, phases, seed, T(2π);
            ndrange=length(phases))
    return phases
end

"""
    _try_gpu_rand!(phases) -> Bool

Legacy extension hook retained for compatibility. New forcing code uses the
counter-based architecture kernel above.
"""
_try_gpu_rand!(phases::AbstractArray) = false

