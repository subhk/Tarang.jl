"""Shared types and cache containers for nonlinear evaluation."""

# Note: PencilArrays, PencilFFTs, MPI, LinearAlgebra, FFTW are already imported in Tarang.jl

# Performance monitoring (defined first as it's used by NonlinearEvaluator)
mutable struct NonlinearPerformanceStats
    total_evaluations::Int
    total_time::Float64
    dealiasing_time::Float64
    transform_time::Float64

    function NonlinearPerformanceStats()
        new(0, 0.0, 0.0, 0.0)
    end
end

const _PaddedDealiasKey = Tuple{UInt, DataType, UInt, Bool}

abstract type AbstractNonlinearTransformConfig end

# Nonlinear evaluation builds several kinds of transform plans:
# ordinary grid-shape plans, tuple-keyed legacy plans, padded/dealiased
# plans, and pencil-specific plans. Keeping them in separate dictionaries
# makes cache invalidation and type assertions cheaper than a single
# `Dict{Any, Any}`.
mutable struct NonlinearTransformCache
    shape_transforms::Dict{String, AbstractNonlinearTransformConfig}
    tuple_transforms::Dict{Tuple, AbstractNonlinearTransformConfig}
    padded_dealiasing::Dict{_PaddedDealiasKey, AbstractNonlinearTransformConfig}
    padded_pencil::Dict{String, AbstractNonlinearTransformConfig}
end

NonlinearTransformCache() = NonlinearTransformCache(
    Dict{String, AbstractNonlinearTransformConfig}(),
    Dict{Tuple, AbstractNonlinearTransformConfig}(),
    Dict{_PaddedDealiasKey, AbstractNonlinearTransformConfig}(),
    Dict{String, AbstractNonlinearTransformConfig}(),
)

@inline function _is_padded_dealias_key(key)
    return key isa Tuple &&
           length(key) == 4 &&
           key[1] isa UInt &&
           key[2] isa DataType &&
           key[3] isa UInt &&
           key[4] isa Bool
end

@inline function _nonlinear_cache_store(cache::NonlinearTransformCache, key)
    # Route keys to the same backing store that created them. This preserves
    # older tuple/string call sites while giving padded dealiasing a typed key.
    if key isa String
        return startswith(key, "padded_pencil_") ? cache.padded_pencil : cache.shape_transforms
    elseif _is_padded_dealias_key(key)
        return cache.padded_dealiasing
    elseif key isa Tuple
        return cache.tuple_transforms
    end
    throw(ArgumentError("Unsupported nonlinear cache key: $(typeof(key))"))
end

Base.haskey(cache::NonlinearTransformCache, key) = haskey(_nonlinear_cache_store(cache, key), key)
Base.getindex(cache::NonlinearTransformCache, key) = getindex(_nonlinear_cache_store(cache, key), key)
Base.setindex!(cache::NonlinearTransformCache, value, key) = setindex!(_nonlinear_cache_store(cache, key), value, key)
Base.get(cache::NonlinearTransformCache, key, default) = get(_nonlinear_cache_store(cache, key), key, default)
Base.isempty(cache::NonlinearTransformCache) = isempty(cache.shape_transforms) &&
                                               isempty(cache.tuple_transforms) &&
                                               isempty(cache.padded_dealiasing) &&
                                               isempty(cache.padded_pencil)
function Base.empty!(cache::NonlinearTransformCache)
    empty!(cache.shape_transforms)
    empty!(cache.tuple_transforms)
    empty!(cache.padded_dealiasing)
    empty!(cache.padded_pencil)
    return cache
end

@inline function _cache_shape_transform!(cache::NonlinearTransformCache, shape::Tuple, shape_key::String, value)
    cache.shape_transforms[shape_key] = value
    cache.tuple_transforms[shape] = value
    return value
end

struct FFTWTransformConfig{FP, BP, RA<:AbstractArray, CA<:AbstractArray} <: AbstractNonlinearTransformConfig
    # `kind` distinguishes scalar FFT, real-to-complex, and dealiased plans
    # without requiring separate wrapper structs for each plan family.
    kind::Symbol
    shape::Tuple{Vararg{Int}}
    dealiased_shape::Tuple{Vararg{Int}}
    forward_plan::FP
    backward_plan::BP
    scratch_real::RA
    scratch_complex::CA
end

struct PencilTransformConfig{A1, A2, F1, F2, P, D, N} <: AbstractNonlinearTransformConfig
    # Pencil transforms carry both the communication layout and the local FFT
    # plans needed to move between grid/coefficient layouts under MPI.
    # Parametric so plan/array fields are concrete per instance; consumers
    # specialize through a function barrier when pulling configs from caches.
    config::PencilConfig
    forward_pencil_1::A1
    forward_pencil_2::A2
    fft_plan_1::F1
    fft_plan_2::F2
    shape::NTuple{N, Int}
    serial::Bool
    pencil::P
    decomp_dims::D
end

# Nonlinear operator types
abstract type NonlinearOperator <: Operator end

struct AdvectionOperator <: NonlinearOperator
    velocity::VectorField
    scalar::ScalarField
    name::String

    function AdvectionOperator(velocity::VectorField, scalar::ScalarField, name::String="advection")
        new(velocity, scalar, name)
    end
end

struct NonlinearAdvectionOperator <: NonlinearOperator
    velocity::VectorField
    name::String

    function NonlinearAdvectionOperator(velocity::VectorField, name::String="nonlinear_advection")
        new(velocity, name)
    end
end

struct ConvectiveOperator <: NonlinearOperator
    field1::Union{ScalarField, VectorField}
    field2::Union{ScalarField, VectorField}
    operation::Symbol  # :multiply, :dot_product, :cross_product
    name::String

    function ConvectiveOperator(field1, field2, operation::Symbol, name::String="convective")
        new(field1, field2, operation, name)
    end
end

# Nonlinear evaluation engine
mutable struct NonlinearEvaluator <: AbstractNonlinearEvaluator
    dist::Distributor
    pencil_transforms::NonlinearTransformCache
    dealiasing_factor::Float64
    # Temporary fields and scratch arrays are owned by the evaluator so repeated
    # nonlinear RHS calls can reuse memory instead of allocating every stage.
    temp_fields::Dict{String, ScalarField}
    # Rotating pool of nonlinear-product result buffers. Tuple key (pool slot,
    # bases hash, dtype) avoids the per-product string-key allocation that a
    # String-keyed dict would incur in `_checkout_nl_result!`.
    nl_result_pool::Dict{Tuple{Int, UInt, DataType}, ScalarField}
    memory_pool::Vector{PencilArrays.PencilArray}
    scratch_arrays::Vector{AbstractArray}
    performance_stats::NonlinearPerformanceStats

    function NonlinearEvaluator(dist::Distributor; dealiasing_factor::Float64=3.0/2.0)
        evaluator = new(dist, NonlinearTransformCache(), dealiasing_factor, Dict{String, ScalarField}(),
                       Dict{Tuple{Int, UInt, DataType}, ScalarField}(), PencilArrays.PencilArray[],
                       AbstractArray[], NonlinearPerformanceStats())
        setup_nonlinear_transforms!(evaluator)
        return evaluator
    end
end

# Architecture helper functions for NonlinearEvaluator
"""
    architecture(evaluator::NonlinearEvaluator)

Get the architecture (CPU or GPU) for the nonlinear evaluator.
"""
architecture(evaluator::NonlinearEvaluator) = evaluator.dist.architecture

"""
    is_gpu(evaluator::NonlinearEvaluator)

Check if the nonlinear evaluator is using GPU architecture.
"""
is_gpu(evaluator::NonlinearEvaluator) = is_gpu(evaluator.dist.architecture)
