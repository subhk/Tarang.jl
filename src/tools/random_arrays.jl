"""
Random array utilities.

Provides chunked random number generators and helper containers that
deliver deterministic pseudo-random data without materialising entire
arrays in memory.
"""

using Random

export ChunkedRNG, chunked_rng, rng_element, rng_elements,
       IndexArray, ChunkedRandomArray
# ---------------------------------------------------------------------------
# Chunked RNG infrastructure
# ---------------------------------------------------------------------------

struct ChunkedRNG{K<:NamedTuple}
    seed::UInt64
    chunk_size::Int
    distribution::Symbol
    params::K
end

function chunked_rng(seed::Union{Integer,Nothing}, chunk_size::Integer,
                     distribution::Union{Symbol,String}=:uniform; kwargs...)
    chunk_size = max(1, Int(chunk_size))
    dist = Symbol(distribution)
    params = NamedTuple(kwargs)
    seed_val = seed === nothing ? rand(RandomDevice(), UInt64) : UInt64(seed)
    return ChunkedRNG(seed_val, chunk_size, dist, params)
end

chunked_rng(; seed=nothing, chunk_size=2^20, distribution=:uniform, kwargs...) =
    chunked_rng(seed, chunk_size, distribution; kwargs...)

function Base.iterate(r::ChunkedRNG)
    rng = Random.MersenneTwister(r.seed)
    data = draw_distribution!(rng, r.distribution, r.chunk_size, r.params)
    return ((0, data), (1, rng))
end

function Base.iterate(r::ChunkedRNG, state::Tuple{Int,Random.AbstractRNG})
    chunk_index, rng = state
    data = draw_distribution!(rng, r.distribution, r.chunk_size, r.params)
    return ((chunk_index, data), (chunk_index + 1, rng))
end

function draw_distribution!(rng::Random.AbstractRNG, distribution::Symbol,
                            n::Int, params::NamedTuple)
    n = max(n, 1)
    if distribution === :uniform
        low = get(params, :low, 0.0)
        high = get(params, :high, 1.0)
        T = get(params, :dtype, Float64)
        # Swap if low > high to ensure valid range
        if low > high
            low, high = high, low
        end
        # For non-float types, generate Float64 samples first then convert
        if T <: AbstractFloat
            return low .+ (high - low) .* rand(rng, T, n)
        else
            # Generate floats in [low, high) then convert to target type
            samples = low .+ (high - low) .* rand(rng, Float64, n)
            return T.(round.(samples))
        end
    elseif distribution === :normal || distribution === :gaussian
        μ = get(params, :mean, 0.0)
        σ = get(params, :sigma, get(params, :std, 1.0))
        T = get(params, :dtype, Float64)
        # randn only supports AbstractFloat types
        if !(T <: AbstractFloat)
            throw(ArgumentError("draw_distribution!: :normal distribution requires floating-point dtype, got $T"))
        end
        # Use absolute value of sigma (negative sigma is mathematically equivalent)
        σ = abs(σ)
        return μ .+ σ .* randn(rng, T, n)
    elseif distribution === :complex_normal
        μ = get(params, :mean, 0.0)
        σ = get(params, :sigma, get(params, :std, 1.0))
        # Use absolute value of sigma
        σ = abs(σ)
        real_part = randn(rng, Float64, n)
        imag_part = randn(rng, Float64, n)
        return Complex.(μ, 0) .+ σ .* Complex.(real_part, imag_part)
    else
        throw(ArgumentError("Unsupported distribution: $distribution"))
    end
end

# ---------------------------------------------------------------------------
# Direct element access
# ---------------------------------------------------------------------------

function rng_element(index::Integer, seed, chunk_size, distribution; kwargs...)
    index = Int(index)
    if index < 0
        throw(ArgumentError("rng_element: index must be non-negative, got $index"))
    end
    chunk_size_eff = min(max(Int(chunk_size), 1), index + 1)
    div, mod = divrem(index, chunk_size_eff)
    for (chunk, data) in chunked_rng(seed, chunk_size_eff, distribution; kwargs...)
        if chunk == div
            return data[mod + 1]
        elseif chunk > div
            break
        end
    end
    error("Failed to retrieve RNG element at index $index")
end

function rng_elements(indices, seed, chunk_size, distribution; kwargs...)
    output_shape = size(indices)
    idx = vec(Int.(indices))
    if isempty(idx)
        sample = rng_element(0, seed, chunk_size, distribution; kwargs...)
        return fill(zero(sample), output_shape)
    end
    # Validate all indices are non-negative
    min_index = minimum(idx)
    if min_index < 0
        throw(ArgumentError("rng_elements: all indices must be non-negative, got minimum $min_index"))
    end
    max_index = maximum(idx)
    chunk_size_eff = min(max(Int(chunk_size), 1), max_index + 1)
    # Use div/mod instead of ÷/% to ensure non-negative results (though indices are already validated)
    divs = div.(idx, chunk_size_eff)
    mods = mod.(idx, chunk_size_eff)
    chunk_map = Dict{Int, Vector{Int}}()
    for (pos, chunk) in enumerate(divs)
        push!(get!(chunk_map, chunk, Int[]), pos)
    end
    max_div = maximum(divs)
    values = nothing
    for (chunk, data) in chunked_rng(seed, chunk_size_eff, distribution; kwargs...)
        if values === nothing
            values = Vector{eltype(data)}(undef, length(idx))
        end
        if haskey(chunk_map, chunk)
            for pos in chunk_map[chunk]
                values[pos] = data[mods[pos] + 1]
            end
        end
        if chunk >= max_div
            break
        end
    end
    if values === nothing
        sample = rng_element(first(idx), seed, chunk_size, distribution; kwargs...)
        values = fill(sample, length(idx))
    end
    return reshape(values, output_shape)
end

# ---------------------------------------------------------------------------
# Index helper types
# ---------------------------------------------------------------------------

struct IndexArray
    shape::Tuple{Vararg{Int}}
    order::Symbol
    function IndexArray(shape::Tuple{Vararg{Int}}, order::Symbol=:C)
        # Validate all dimensions are positive
        for (i, dim) in enumerate(shape)
            if dim < 1
                throw(ArgumentError("IndexArray: dimension $i must be positive, got $dim"))
            end
        end
        order = order == :F ? :F : :C
        new(shape, order)
    end
end

IndexArray(shape::Tuple; order::Symbol=:C) = IndexArray(shape, order)

Base.size(ia::IndexArray) = ia.shape
Base.length(ia::IndexArray) = prod(ia.shape)
Base.ndims(ia::IndexArray) = length(ia.shape)

function expand_selection(sel, size::Int)
    if sel === Colon()
        return collect(1:size)
    elseif isa(sel, Integer)
        idx = Int(sel)
        if idx < 1 || idx > size
            throw(ArgumentError("Index $idx out of bounds for dimension of size $size"))
        end
        return [idx]
    elseif isa(sel, AbstractRange)
        r = collect(sel)
        if !isempty(r) && (minimum(r) < 1 || maximum(r) > size)
            throw(ArgumentError("Range $sel out of bounds for dimension of size $size"))
        end
        return r
    else
        throw(ArgumentError("Unsupported index selection: $(sel)"))
    end
end

function linear_index(coords::NTuple{N,Int}, shape::NTuple{N,Int}, order::Symbol) where {N}
    zero_based = ntuple(i -> coords[i] - 1, N)
    if order == :F
        stride = 1
        lin = 0
        for i in 1:N
            lin += zero_based[i] * stride
            stride *= shape[i]
        end
    else
        stride = 1
        lin = 0
        for i in N:-1:1
            lin += zero_based[i] * stride
            stride *= shape[i]
        end
    end
    return lin
end

function Base.getindex(ia::IndexArray, key...)
    selections = key === () ? () : key
    if length(selections) < length(ia.shape)
        selections = Tuple(vcat(collect(selections),
                                ntuple(_ -> Colon(), length(ia.shape) - length(selections))))
    elseif length(selections) > length(ia.shape)
        throw(ArgumentError("Too many indices for IndexArray"))
    end
    ranges = map(expand_selection, selections, ia.shape)
    dims = map(length, ranges)
    values = Vector{Int}(undef, prod(dims))
    it = Iterators.product(ranges...)
    i = 1
    for coords in it
        coords_tuple = ntuple(j -> coords[j], length(ia.shape))
        values[i] = linear_index(coords_tuple, ia.shape, ia.order)
        i += 1
    end
    return reshape(values, Tuple(dims))
end

# ---------------------------------------------------------------------------
# Chunked random array container
# ---------------------------------------------------------------------------

struct ChunkedRandomArray{K<:NamedTuple}
    shape::Tuple{Vararg{Int}}
    seed::UInt64
    chunk_size::Int
    distribution::Symbol
    params::K
    order::Symbol
end

function ChunkedRandomArray(shape::Tuple{Vararg{Int}}; seed=nothing,
                            chunk_size=2^20, distribution=:uniform,
                            order::Symbol=:C, kwargs...)
    # Validate all dimensions are positive (empty tuple for scalar is allowed)
    for (i, dim) in enumerate(shape)
        if dim < 1
            throw(ArgumentError("ChunkedRandomArray: dimension $i must be positive, got $dim"))
        end
    end
    seed_val = seed === nothing ? rand(RandomDevice(), UInt64) : UInt64(seed)
    dist = Symbol(distribution)
    params = NamedTuple(kwargs)
    order_val = order == :F ? :F : :C
    return ChunkedRandomArray(shape, seed_val, max(1, Int(chunk_size)), dist, params, order_val)
end

Base.size(cra::ChunkedRandomArray) = cra.shape
Base.length(cra::ChunkedRandomArray) = prod(cra.shape)
Base.ndims(cra::ChunkedRandomArray) = length(cra.shape)
Base.axes(cra::ChunkedRandomArray) = ntuple(i -> Base.OneTo(cra.shape[i]), length(cra.shape))

function Base.eltype(cra::ChunkedRandomArray)
    # Infer element type from distribution and params
    T = get(cra.params, :dtype, Float64)
    if cra.distribution === :complex_normal
        return Complex{Float64}
    elseif T <: AbstractFloat
        return T
    else
        return T
    end
end

function Base.getindex(cra::ChunkedRandomArray, key...)
    ia = IndexArray(cra.shape, cra.order)
    indices = ia[key...]  # zero-based ints
    return rng_elements(indices, cra.seed, cra.chunk_size, cra.distribution; cra.params...)
end

function Base.getindex(cra::ChunkedRandomArray)
    ia = IndexArray(cra.shape, cra.order)
    indices = ia[ntuple(_ -> Colon(), length(cra.shape))...]
    return rng_elements(indices, cra.seed, cra.chunk_size, cra.distribution; cra.params...)
end
