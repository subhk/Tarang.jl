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
#   SerialFieldStorage   — local Array/CuArray (serial/GPU), or PencilArray for
#                          MPI-distributed FFT (no separate Pencil storage type;
#                          MPI binds the storage param to abstract PencilArray)
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
    SerialFieldStorage{G,C} <: AbstractFieldStorage

Storage for serial (single-process) or PencilArray-backed fields.
Wraps the existing FieldBuffers structure.

Parametrized on the CONCRETE grid (`G`) and coefficient (`C`) array types so
that field-data access (`get_grid_data`/`get_coeff_data`) is type-stable. The
arrays are built up-front by `_build_field_arrays` (or the typed length-0
sentinels for 0-D fields) and the storage is constructed from them — fields are
architecture-fixed, so these types are stable for the field's life.
"""
mutable struct SerialFieldStorage{G<:AbstractArray, C<:AbstractArray} <: AbstractFieldStorage
    architecture::AbstractArchitecture
    grid::G
    coeff::C
end

# Julia auto-generates the inferring outer constructor
# `SerialFieldStorage(arch, grid, coeff)` (binding G=typeof(grid), C=typeof(coeff))
# from the struct definition above, so no explicit outer constructor is needed —
# adding one would collide with the auto-generated method during precompilation.

# Storage type-parameter selection. Local arrays (Array / CuArray) bind to their
# CONCRETE type so serial/GPU field-data access is type-stable. MPI fields bind to
# the abstract `PencilArray` UnionAll instead: a field's pencil decomposition and
# permutation change under transposes (e.g. group_transpose_fields!),
# producing a DIFFERENT concrete PencilArray type, so a
# frozen exact type would make set_grid_data!/set_coeff_data! throw on the
# re-decomposed array. MPI access stays abstract (as before parametrization);
# the type-stability win targets the serial/single-node hot path.
_field_storage_param(::PencilArrays.PencilArray) = PencilArrays.PencilArray
_field_storage_param(a::AbstractArray) = typeof(a)

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
        initial_scales = length(bases) > 0 ? ntuple(_ -> 1.0, dist.dim) : nothing

        # Build the concrete arrays BEFORE storage so SerialFieldStorage{G,C} is
        # parametrized on their real types. 0-D fields get typed length-0
        # sentinels so storage is never nothing (Phase 1 type-stability).
        g, c = domain !== nothing ? _build_field_arrays(dist, domain, T) : (_empty_grid(T), _empty_coeff(T))
        storage = SerialFieldStorage{_field_storage_param(g), _field_storage_param(c)}(dist.architecture, g, c)
        return new{T, typeof(storage)}(dist, name, bases, domain, dtype, storage, layout, :g, initial_scales, :auto, false, 0)
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

        # Create component fields. SerialFieldStorage is now parametric, so build
        # into an abstractly-typed vector then narrow with identity.() — all
        # components share bases/dtype/dist, hence one concrete storage type.
        components = ScalarField[]
        for (i, coord_name) in enumerate(coordsys.names)
            component_name = "$(name)_$coord_name"
            component = ScalarField(dist, component_name, bases, dtype)
            push!(components, component)
        end
        comps = identity.(components)
        S = _component_storage_type(comps)

        buffer_architecture = dist.architecture
        new{T, S}(dist, coordsys, name, bases, domain, dtype, comps, nothing, nothing, buffer_architecture)
    end
end

# Resolve the concrete storage type parameter from a realized component
# container. After identity.() narrowing, every component shares one concrete
# SerialFieldStorage{G,C}; an empty/abstract container (degenerate: no
# coordinates) falls back to the abstract storage supertype.
_component_storage_type(::AbstractArray{ScalarField{T, S}}) where {T, S} = S
_component_storage_type(::AbstractArray) = SerialFieldStorage

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
        # SerialFieldStorage is now parametric: build an abstractly-typed matrix
        # then narrow with identity.() — all components share one concrete
        # storage type (same bases/dtype/dist).
        components = Matrix{ScalarField}(undef, dim, dim)
        for i in 1:dim, j in 1:dim
            component_name = "$(name)_$(coordsys.names[i])$(coordsys.names[j])"
            components[i,j] = ScalarField(dist, component_name, bases, dtype)
        end
        comps = identity.(components)
        S = _component_storage_type(comps)

        buffer_architecture = dist.architecture
        new{T, S}(dist, coordsys, name, bases, domain, dtype, comps, nothing, nothing, buffer_architecture)
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
storage_mode(::ScalarField{T, <:SerialFieldStorage}) where T = SerialStorage()
# TransposableFieldStorage dispatch is defined in transposable_field.jl (loaded later)
storage_mode(field::ScalarField) = storage_mode(field.dist)  # fallback
storage_mode(vf::VectorField) = storage_mode(vf.dist)
storage_mode(tf::TensorField) = storage_mode(tf.dist)

