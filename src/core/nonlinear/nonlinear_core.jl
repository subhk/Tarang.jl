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

mutable struct NonlinearTransformCache
    shape_transforms::Dict{String, Any}
    tuple_transforms::Dict{Tuple, Any}
    padded_dealiasing::Dict{_PaddedDealiasKey, Any}
    padded_pencil::Dict{String, Any}
end

NonlinearTransformCache() = NonlinearTransformCache(
    Dict{String, Any}(),
    Dict{Tuple, Any}(),
    Dict{_PaddedDealiasKey, Any}(),
    Dict{String, Any}(),
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

abstract type AbstractNonlinearTransformConfig end

struct FFTWTransformConfig{FP, BP, RA<:AbstractArray, CA<:AbstractArray} <: AbstractNonlinearTransformConfig
    kind::Symbol
    shape::Tuple{Vararg{Int}}
    dealiased_shape::Tuple{Vararg{Int}}
    forward_plan::FP
    backward_plan::BP
    scratch_real::RA
    scratch_complex::CA
end

struct PencilTransformConfig <: AbstractNonlinearTransformConfig
    config::PencilConfig
    forward_pencil_1::Any
    forward_pencil_2::Any
    fft_plan_1::Any
    fft_plan_2::Any
    shape::Tuple{Vararg{Int}}
    serial::Bool
    pencil::Any
    decomp_dims::Any
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
    temp_fields::Dict{String, ScalarField}
    memory_pool::Vector{PencilArrays.PencilArray}
    scratch_arrays::Vector{AbstractArray}
    performance_stats::NonlinearPerformanceStats

    function NonlinearEvaluator(dist::Distributor; dealiasing_factor::Float64=3.0/2.0)
        evaluator = new(dist, NonlinearTransformCache(), dealiasing_factor, Dict{String, ScalarField}(), PencilArrays.PencilArray[],
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
