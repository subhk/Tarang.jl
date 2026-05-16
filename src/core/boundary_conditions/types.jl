# Boundary condition value wrappers, concrete BCs, cache storage, and manager state.

mutable struct BCPerformanceStats
    total_time::Float64
    total_evaluations::Int
    bc_updates::Int
    cache_hits::Int
    cache_misses::Int

    function BCPerformanceStats()
        new(0.0, 0, 0, 0, 0)
    end
end

const _BCTimeKey = Tuple{Int, Float64}
const _BCSpatialKey = Tuple{Int, Float64, UInt64}

mutable struct BCRobinCacheValue
    alpha::Any
    beta::Any
    value::Any
end

mutable struct BCScratchSpace
    arrays::Vector{AbstractArray}
end

BCScratchSpace() = BCScratchSpace(AbstractArray[])

Base.empty!(scratch::BCScratchSpace) = (empty!(scratch.arrays); scratch)
Base.isempty(scratch::BCScratchSpace) = isempty(scratch.arrays)

mutable struct BCCacheStore
    time_values::Dict{_BCTimeKey, Any}
    time_robin_values::Dict{_BCTimeKey, BCRobinCacheValue}
    spatial_values::Dict{_BCSpatialKey, Any}
    spatial_robin_values::Dict{_BCSpatialKey, BCRobinCacheValue}
end

BCCacheStore() = BCCacheStore(
    Dict{_BCTimeKey, Any}(),
    Dict{_BCTimeKey, BCRobinCacheValue}(),
    Dict{_BCSpatialKey, Any}(),
    Dict{_BCSpatialKey, BCRobinCacheValue}(),
)

function Base.empty!(cache::BCCacheStore)
    empty!(cache.time_values)
    empty!(cache.time_robin_values)
    empty!(cache.spatial_values)
    empty!(cache.spatial_robin_values)
    return cache
end

function Base.isempty(cache::BCCacheStore)
    return isempty(cache.time_values) &&
           isempty(cache.time_robin_values) &&
           isempty(cache.spatial_values) &&
           isempty(cache.spatial_robin_values)
end

@inline _bc_cache_time_key(key::Tuple{Int,<:Real}) = (key[1], Float64(key[2]))
@inline _bc_cache_robin_key(key::Tuple{Int,<:Real,Symbol}) = (key[1], Float64(key[2]))

function _set_robin_component!(robin::BCRobinCacheValue, component::Symbol, value)
    if component === :alpha
        robin.alpha = value
    elseif component === :beta
        robin.beta = value
    elseif component === :value
        robin.value = value
    else
        throw(KeyError(component))
    end
    return value
end

function _get_robin_component(robin::BCRobinCacheValue, component::Symbol)
    if component === :alpha
        return robin.alpha
    elseif component === :beta
        return robin.beta
    elseif component === :value
        return robin.value
    else
        throw(KeyError(component))
    end
end

function Base.setindex!(cache::BCCacheStore, value, key::Tuple{Int,<:Real})
    cache.time_values[_bc_cache_time_key(key)] = value
    return cache
end

function Base.getindex(cache::BCCacheStore, key::Tuple{Int,<:Real})
    return cache.time_values[_bc_cache_time_key(key)]
end

function Base.haskey(cache::BCCacheStore, key::Tuple{Int,<:Real})
    return haskey(cache.time_values, _bc_cache_time_key(key))
end

function Base.setindex!(cache::BCCacheStore, value, key::Tuple{Int,<:Real,Symbol})
    robin = get!(cache.time_robin_values, _bc_cache_robin_key(key)) do
        BCRobinCacheValue(nothing, nothing, nothing)
    end
    _set_robin_component!(robin, key[3], value)
    return cache
end

function Base.getindex(cache::BCCacheStore, key::Tuple{Int,<:Real,Symbol})
    return _get_robin_component(cache.time_robin_values[_bc_cache_robin_key(key)], key[3])
end

function Base.haskey(cache::BCCacheStore, key::Tuple{Int,<:Real,Symbol})
    return haskey(cache.time_robin_values, _bc_cache_robin_key(key))
end

struct FieldReference
    name::String
    expression::Union{String, Function}
    dependencies::Vector{String}
end

struct TimeDependentValue
    expression::String
    variables::Vector{String}
    function_obj::Union{Function, Nothing}
end

struct SpaceDependentValue
    expression::String
    coordinates::Vector{String}
    function_obj::Union{Function, Nothing}
end

struct TimeSpaceDependentValue
    expression::String
    time_variables::Vector{String}
    space_coordinates::Vector{String}
    function_obj::Union{Function, Nothing}
end

abstract type AbstractBoundaryCondition end

const BCValueType = Union{Real, String, Function, FieldReference,
                          TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue}

struct DirichletBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
    position::Union{Real, String}
    value::BCValueType
    tau_field::Union{String, Nothing}
    is_time_dependent::Bool
    is_space_dependent::Bool
end

struct NeumannBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
    position::Union{Real, String}
    derivative_order::Int
    value::BCValueType
    tau_field::Union{String, Nothing}
    is_time_dependent::Bool
    is_space_dependent::Bool
end

struct RobinBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
    position::Union{Real, String}
    alpha::BCValueType
    beta::BCValueType
    value::BCValueType
    tau_field::Union{String, Nothing}
    is_time_dependent::Bool
    is_space_dependent::Bool
end

struct PeriodicBC <: AbstractBoundaryCondition
    field::String
    coordinate::String
end

struct StressFreeBC <: AbstractBoundaryCondition
    velocity_field::String
    coordinate::String
    position::Union{Real, String}
    tau_fields::Vector{String}
    component_coordinates::Vector{String}
end

struct CustomBC <: AbstractBoundaryCondition
    expression::String
    tau_fields::Vector{String}
end

mutable struct BoundaryConditionManager{Arch<:AbstractArchitecture}
    conditions::Vector{AbstractBoundaryCondition}
    tau_fields::Dict{String, Any}
    lift_operators::Dict{String, Any}
    coordinate_info::Dict{String, Any}
    time_variable::Union{String, Nothing}
    coordinate_fields::Dict{String, Any}
    time_dependent_bcs::Vector{Int}
    space_dependent_bcs::Vector{Int}
    bc_update_required::Bool
    nonconstant_bc_indices::Vector{Int}
    bc_equation_indices::Dict{Int, Int}
    workspace::BCScratchSpace
    bc_cache::BCCacheStore
    performance_stats::BCPerformanceStats
    architecture::Arch

    function BoundaryConditionManager(; architecture::Arch=CPU()) where {Arch<:AbstractArchitecture}
        workspace = BCScratchSpace()
        bc_cache = BCCacheStore()
        perf_stats = BCPerformanceStats()

        new{Arch}(AbstractBoundaryCondition[], Dict{String, Any}(),
            Dict{String, Any}(), Dict{String, Any}(), nothing,
            Dict{String, Any}(), Int[], Int[], false,
            Int[],
            Dict{Int, Int}(),
            workspace, bc_cache, perf_stats, architecture)
    end
end

"""
    is_gpu(manager::BoundaryConditionManager)

Check if the boundary condition manager is configured for GPU.
"""
is_gpu(manager::BoundaryConditionManager) = is_gpu(manager.architecture)

"""
    architecture(manager::BoundaryConditionManager)

Get the architecture (CPU or GPU) of the boundary condition manager.
"""
architecture(manager::BoundaryConditionManager) = manager.architecture
