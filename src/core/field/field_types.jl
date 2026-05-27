"""
Field classes for data fields
"""

# PencilArrays, LinearAlgebra, SparseArrays, LoopVectorization already in Tarang.jl
using LinearAlgebra: mul!, ldiv!
using NetCDF
using Random

abstract type Operand end

# ---------------------------------------------------------------------------
# Field storage traits (serial vs pencil-distributed)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Field Storage Hierarchy
#
# ScalarField{T, S} is parameterized on:
#   T — element type (Float64, Float32, etc.)
#   S — storage backend (<: AbstractFieldStorage)
#
# Storage backends:
#   SerialFieldStorage   — local arrays, FFTW/serial transforms
#   PencilFieldStorage   — PencilArrays for MPI-distributed FFT
#   TransposableFieldStorage — 2D pencil decomposition with transposes (GPU+MPI)
#
# Adding S as a type parameter enables the compiler to specialize
# forward_transform!/backward_transform! without runtime dispatch.
# ---------------------------------------------------------------------------

abstract type AbstractFieldStorage end

# Legacy trait types (kept for compatibility with existing storage_mode() dispatch)
abstract type FieldStorageMode end
struct SerialStorage <: FieldStorageMode end
struct PencilStorage <: FieldStorageMode end
struct TransposableStorage <: FieldStorageMode end

storage_mode(dist::Distributor) = dist.use_pencil_arrays ? PencilStorage() : SerialStorage()

is_pencil_storage(x) = storage_mode(x) isa PencilStorage
is_serial_storage(x) = storage_mode(x) isa SerialStorage
is_transposable_storage(x) = storage_mode(x) isa TransposableStorage

"""
    SerialFieldStorage <: AbstractFieldStorage

Storage for serial (single-process) or PencilArray-backed fields.
Wraps the existing FieldBuffers structure.
"""
mutable struct SerialFieldStorage <: AbstractFieldStorage
    architecture::AbstractArchitecture
    grid::Union{Nothing, AbstractArray}   # PencilArray <: AbstractArray — no need for separate Pencil branch
    coeff::Union{Nothing, AbstractArray}  # Reduced from 3-way to 2-way Union for better type inference

    function SerialFieldStorage(arch::AbstractArchitecture)
        new(arch, nothing, nothing)
    end
end

# TransposableFieldStorage is defined in transposable_field.jl (loaded later)
# because it depends on TransposeBuffers, Topology2D, etc. from transpose_types.jl.
# It inherits from AbstractFieldStorage defined above.

# Backward-compatible alias: FieldBuffers is now SerialFieldStorage
const FieldBuffers = SerialFieldStorage

@inline function _update_field_buffer_architecture!(buffers::FieldBuffers, value)
    if value === nothing
        return
    elseif value isa AbstractArray
        buffers.architecture = architecture(value)
    end
end

mutable struct ScalarField{T, S<:AbstractFieldStorage} <: Operand
    dist::Distributor
    name::String
    bases::Tuple{Vararg{Basis}}
    domain::Union{Nothing, Domain}
    dtype::Type{T}

    storage::S

    # Layout information
    layout::Union{Nothing, Layout}
    current_layout::Symbol  # :g for grid, :c for coefficient

    # Scale information
    scales::Union{Nothing, Tuple{Vararg{Float64}}}  # Current scales for each dimension

    # GPU FFT preference (:auto, :cpu, :gpu)
    fft_mode::Symbol

    # Pool tracking metadata (managed by FieldPool)
    _from_pool::Bool
    _pool_generation::Int

    function ScalarField(dist::Distributor, name::String="field", bases::Tuple{Vararg{Basis}}=(),
                         dtype::Type{T}=dist.dtype) where T
        domain = length(bases) > 0 ? Domain(dist, bases) : nothing
        layout = length(bases) > 0 ? get_layout(dist, bases, dtype) : nothing

        storage = SerialFieldStorage(dist.architecture)
        current_layout = :g

        # Initialize scales: (1,) * dist.dim
        initial_scales = length(bases) > 0 ? ntuple(_ -> 1.0, dist.dim) : nothing

        field = new{T, SerialFieldStorage}(dist, name, bases, domain, dtype, storage, layout, current_layout, initial_scales, :auto, false, 0)

        # Allocate data if we have a domain; otherwise install typed length-0
        # sentinels so storage is never nothing (Phase 1 type-stability).
        if domain !== nothing
            allocate_data!(field)
        else
            set_grid_data!(field, _empty_grid(T))
            set_coeff_data!(field, _empty_coeff(T))
        end

        return field
    end

    # Inner constructor for explicit storage type (e.g., TransposableFieldStorage)
    function ScalarField(dist::Distributor, name::String, bases::Tuple{Vararg{Basis}},
                         dtype::Type{T}, storage::S) where {T, S<:AbstractFieldStorage}
        domain = length(bases) > 0 ? Domain(dist, bases) : nothing
        layout = length(bases) > 0 ? get_layout(dist, bases, dtype) : nothing
        initial_scales = length(bases) > 0 ? ntuple(_ -> 1.0, dist.dim) : nothing
        field = new{T, S}(dist, name, bases, domain, dtype, storage, layout, :g, initial_scales, :auto, false, 0)
        # Install typed length-0 sentinels for 0-D fields so storage is never nothing.
        if domain === nothing
            set_grid_data!(field, _empty_grid(T))
            set_coeff_data!(field, _empty_coeff(T))
        end
        return field
    end
end

# Note: getproperty/setproperty! for ScalarField are defined later in this file.
# They include backward-compatible :buffers → :storage mapping.

mutable struct VectorField{T, S<:AbstractFieldStorage} <: Operand
    dist::Distributor
    coordsys::CoordinateSystem
    name::String
    bases::Tuple{Vararg{Basis}}
    domain::Union{Nothing, Domain}
    dtype::Type{T}

    # Component fields
    components::Vector{ScalarField{T, S}}

    # Optional stacked component buffer for SoA access
    component_buffer::Union{Nothing, AbstractArray}
    buffer_layout::Union{Nothing, Symbol}
    buffer_architecture::AbstractArchitecture

    function VectorField(dist::Distributor, coordsys::CoordinateSystem, name::String="vector",
                         bases::Tuple{Vararg{Basis}}=(), dtype::Type{T}=dist.dtype) where T
        domain = length(bases) > 0 ? Domain(dist, bases) : nothing

        # Create component fields (SerialFieldStorage by default)
        components = ScalarField{T, SerialFieldStorage}[]
        for (i, coord_name) in enumerate(coordsys.names)
            component_name = "$(name)_$coord_name"
            component = ScalarField(dist, component_name, bases, dtype)
            push!(components, component)
        end

        buffer_architecture = dist.architecture
        new{T, SerialFieldStorage}(dist, coordsys, name, bases, domain, dtype, components, nothing, nothing, buffer_architecture)
    end
end

# Convenience constructor: uses dist.coordsys as default coordinate system
"""
    VectorField(dist, name, bases, dtype=dist.dtype)

Create a VectorField using the distributor's coordinate system.

This is a convenience constructor equivalent to:
    VectorField(dist, dist.coordsys, name, bases, dtype)

# Example
```julia
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; dtype=Float64)
xb = RealFourier(coords["x"]; size=64, bounds=(0.0, 2π))
yb = RealFourier(coords["y"]; size=64, bounds=(0.0, 2π))

# Simple form (recommended):
u = VectorField(dist, "u", (xb, yb), Float64)

# Explicit form (when you need a different coordinate system):
u = VectorField(dist, coords, "u", (xb, yb), Float64)
```
"""
VectorField(dist::Distributor, name::String, bases::Tuple{Vararg{Basis}}, dtype::Type=dist.dtype) =
    VectorField(dist, dist.coordsys, name, bases, dtype)

mutable struct TensorField{T, S<:AbstractFieldStorage} <: Operand
    dist::Distributor
    coordsys::CoordinateSystem
    name::String
    bases::Tuple{Vararg{Basis}}
    domain::Union{Nothing, Domain}
    dtype::Type{T}

    # Component fields as matrix
    components::Matrix{ScalarField{T, S}}

    component_buffer::Union{Nothing, AbstractArray}
    buffer_layout::Union{Nothing, Symbol}
    buffer_architecture::AbstractArchitecture

    function TensorField(dist::Distributor, coordsys::CoordinateSystem, name::String="tensor",
                         bases::Tuple{Vararg{Basis}}=(), dtype::Type{T}=dist.dtype) where T
        domain = length(bases) > 0 ? Domain(dist, bases) : nothing

        dim = coordsys.dim
        components = Matrix{ScalarField{T, SerialFieldStorage}}(undef, dim, dim)
        for i in 1:dim, j in 1:dim
            component_name = "$(name)_$(coordsys.names[i])$(coordsys.names[j])"
            components[i,j] = ScalarField(dist, component_name, bases, dtype)
        end

        buffer_architecture = dist.architecture
        new{T, SerialFieldStorage}(dist, coordsys, name, bases, domain, dtype, components, nothing, nothing, buffer_architecture)
    end
end

# Convenience constructor: uses dist.coordsys as default coordinate system
"""
    TensorField(dist, name, bases, dtype=dist.dtype)

Create a TensorField using the distributor's coordinate system.

This is a convenience constructor equivalent to:
    TensorField(dist, dist.coordsys, name, bases, dtype)
"""
TensorField(dist::Distributor, name::String, bases::Tuple{Vararg{Basis}}, dtype::Type=dist.dtype) =
    TensorField(dist, dist.coordsys, name, bases, dtype)

# storage_mode methods — dispatch on storage type parameter when available
storage_mode(::ScalarField{T, SerialFieldStorage}) where T = SerialStorage()
# TransposableFieldStorage dispatch is defined in transposable_field.jl (loaded later)
storage_mode(field::ScalarField) = storage_mode(field.dist)  # fallback
storage_mode(vf::VectorField) = storage_mode(vf.dist)
storage_mode(tf::TensorField) = storage_mode(tf.dist)

