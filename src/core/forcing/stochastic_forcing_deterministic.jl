"""
    stochastic_forcing_deterministic

Deterministic (non-stochastic) forcing, plus the module's exports.

Split out of the former 1767-line `stochastic_forcing.jl`; the sections are
unchanged, only relocated.
"""

# ============================================================================
# Deterministic Forcing
# ============================================================================

"""
    DeterministicForcing{T, N, A<:AbstractArray{T,N}}

Deterministic (non-random) forcing.

The forcing function is called as `forcing_function(grid..., t, parameters)` and must return an
array whose size is exactly `field_size`, so write it with broadcasting over the grid arrays.

## Example

```julia
# Sinusoidal forcing on a 16×16 grid
forcing = DeterministicForcing(
    (x, y, t, p) -> p[:A] .* sin.(p[:k] .* x) .* cos.(y) .* cos(p[:omega] * t),
    (16, 16);
    parameters = Dict(:A => 1.0, :k => 4.0, :omega => 1.0)
)

x = reshape(range(0, 2π, length=17)[1:16], 16, 1)   # column vector
y = reshape(range(0, 2π, length=17)[1:16], 1, 16)   # row vector
F = generate_forcing!(forcing, (x, y), 0.0)         # 16×16 Matrix{Float64}
```
"""
mutable struct DeterministicForcing{T<:AbstractFloat, N, A<:AbstractArray{T, N}} <: DeterministicForcingType
    forcing_function::Function
    field_size::NTuple{N, Int}
    cached_forcing::A
    parameters::Dict{Symbol, Any}
    architecture::AbstractArchitecture
end

function Base.getproperty(forcing::DeterministicForcing, name::Symbol)
    if name === :is_gpu
        return is_gpu(getfield(forcing, :architecture))
    end
    return getfield(forcing, name)
end

"""
    DeterministicForcing(forcing_function, field_size; parameters=Dict())

Create deterministic forcing.

## Arguments

- `forcing_function`: Function(x, y, ..., t, params) → forcing value
- `field_size`: Grid size
- `parameters`: Dictionary passed to forcing function
"""
function DeterministicForcing(
    forcing_function::Function,
    field_size::NTuple{N, Int};
    parameters::AbstractDict{Symbol} = Dict{Symbol, Any}(),
    dtype::Type{T} = Float64,
    architecture::AbstractArchitecture = CPU()
) where {T<:AbstractFloat, N}

    cached_forcing = zeros(architecture, T, field_size...)

    # Convert to Dict{Symbol, Any} for storage
    params = Dict{Symbol, Any}(k => v for (k, v) in parameters)

    DeterministicForcing{T, N, typeof(cached_forcing)}(
        forcing_function,
        field_size,
        cached_forcing,
        params,
        architecture
    )
end

"""
    generate_forcing!(forcing::DeterministicForcing, grid, t::Real)

Evaluate deterministic forcing at time t on the given grid.
"""
function generate_forcing!(forcing::DeterministicForcing{T, N, A}, grid, t::Real) where {T, N, A}
    values = forcing.forcing_function(grid..., t, forcing.parameters)

    if !(values isa AbstractArray)
        throw(ArgumentError("Deterministic forcing function must return an array"))
    end

    if size(values) != forcing.field_size
        throw(ArgumentError("Deterministic forcing output size $(size(values)) does not match field size $(forcing.field_size)"))
    end

    coerced = eltype(values) <: T ? values : T.(values)
    data_on_arch = on_architecture(forcing.architecture, coerced)
    copyto!(forcing.cached_forcing, data_on_arch)
    return forcing.cached_forcing
end

# ============================================================================
# Exports
# ============================================================================

# Export abstract types
export Forcing, StochasticForcingType, DeterministicForcingType

# Export concrete forcing types
export StochasticForcing, SeparableStochasticForcing, DeterministicForcing

# Export forcing generation and application
export generate_forcing!, apply_forcing!
export reset_forcing!, set_dt!

# Export work calculation (Stratonovich calculus)
export store_prevsol!, work_stratonovich, work_ito

# Export diagnostics
export mean_energy_injection_rate, energy_injection_rate, instantaneous_power
export forcing_enstrophy_injection_rate
export get_forcing_spectrum, get_cached_forcing, get_forcing_real

# Export spectrum building utilities (for custom forcing spectra)
export build_wavenumbers, compute_forcing_spectrum

