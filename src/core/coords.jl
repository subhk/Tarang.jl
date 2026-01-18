"""
Coordinate system classes

This module implements coordinate systems:
- CoordinateSystem: Abstract base type for all coordinate systems
- Coordinate: Individual coordinate within a coordinate system
- CartesianCoordinates: Cartesian (x, y, z) coordinate system
- DirectProduct: Direct product of coordinate systems
- S2Coordinates: Spherical surface coordinates (azimuth, colatitude)
- PolarCoordinates: Polar coordinates (azimuth, radius)
- SphericalCoordinates: Full spherical coordinates (azimuth, colatitude, radius)
"""

using StaticArrays
using LinearAlgebra
using SparseArrays

# ============================================================================
# Abstract types
# ============================================================================

abstract type CoordinateSystem end

# ============================================================================
# Coordinate: Individual coordinate within a system
# ============================================================================

"""
    Coordinate

Represents a single coordinate within a coordinate system.

Fields:
- `name`: Name of the coordinate (e.g., "x", "y", "z")
- `coordsys`: Parent coordinate system (can be nothing for standalone coordinates)
- `dim`: Always 1 for a single coordinate
- `curvilinear`: Whether this coordinate is curvilinear (false for Cartesian)
- `default_nonconst_groups`: Default groups for non-constant modes
"""
mutable struct Coordinate
    name::String
    coordsys::Union{Nothing, CoordinateSystem}
    dim::Int
    curvilinear::Bool
    default_nonconst_groups::Tuple{Vararg{Int}}

    function Coordinate(name::String; cs::Union{Nothing, CoordinateSystem}=nothing)
        new(name, cs, 1, false, (1,))
    end
end

function Base.:(==)(a::Coordinate, b::Coordinate)
    return typeof(a) == typeof(b) && a.name == b.name
end

function Base.hash(c::Coordinate, h::UInt)
    return hash(c.name, hash(:Coordinate, h))
end

function Base.show(io::IO, c::Coordinate)
    print(io, c.name)
end

"""
    check_bounds(coord::Coordinate, bounds)

Check if bounds are valid for this coordinate.
"""
function check_bounds(coord::Coordinate, bounds)
    if coord.coordsys !== nothing
        check_bounds(coord.coordsys, coord, bounds)
    end
end

"""
    forward_vector_intertwiner(coord::Coordinate, subaxis, group)

Forward intertwiner for vector components. Identity for Cartesian coordinates.
Following coords:92-93.
"""
function forward_vector_intertwiner(coord::Coordinate, subaxis, group)
    return ones(1, 1)  # 1x1 identity matrix
end

"""
    backward_vector_intertwiner(coord::Coordinate, subaxis, group)

Backward intertwiner for vector components. Identity for Cartesian coordinates.
Following coords:95-96.
"""
function backward_vector_intertwiner(coord::Coordinate, subaxis, group)
    return ones(1, 1)  # 1x1 identity matrix
end

# ============================================================================
# AzimuthalCoordinate: Special coordinate for azimuthal directions
# Following coords:192-193
# ============================================================================

"""
    AzimuthalCoordinate

Special coordinate type for azimuthal (angular) coordinates.
Used in polar, cylindrical, and spherical coordinate systems.
"""
mutable struct AzimuthalCoordinate
    name::String
    coordsys::Union{Nothing, CoordinateSystem}
    dim::Int
    curvilinear::Bool
    default_nonconst_groups::Tuple{Vararg{Int}}

    function AzimuthalCoordinate(name::String; cs::Union{Nothing, CoordinateSystem}=nothing)
        new(name, cs, 1, false, (1,))
    end
end

function Base.:(==)(a::AzimuthalCoordinate, b::AzimuthalCoordinate)
    return a.name == b.name
end

function Base.hash(c::AzimuthalCoordinate, h::UInt)
    return hash(c.name, hash(:AzimuthalCoordinate, h))
end

function Base.show(io::IO, c::AzimuthalCoordinate)
    print(io, c.name)
end

"""
    check_bounds(coord::AzimuthalCoordinate, bounds)

Check if bounds are valid for azimuthal coordinate.
Azimuthal coordinates are typically periodic over [0, 2π).
"""
function check_bounds(coord::AzimuthalCoordinate, bounds)
    if coord.coordsys !== nothing
        check_bounds(coord.coordsys, coord, bounds)
    end
end

"""
    forward_vector_intertwiner(coord::AzimuthalCoordinate, subaxis, group)

Forward intertwiner for azimuthal vector components.
"""
function forward_vector_intertwiner(coord::AzimuthalCoordinate, subaxis, group)
    return ones(1, 1)  # 1x1 identity matrix
end

"""
    backward_vector_intertwiner(coord::AzimuthalCoordinate, subaxis, group)

Backward intertwiner for azimuthal vector components.
"""
function backward_vector_intertwiner(coord::AzimuthalCoordinate, subaxis, group)
    return ones(1, 1)  # 1x1 identity matrix
end

# ============================================================================
# CartesianCoordinates: Cartesian coordinate system
# Following coords:159-189
# ============================================================================

"""
    CartesianCoordinates

Cartesian coordinate system with named coordinates.
Following implementation in coords:159-189.

# Constructor
    CartesianCoordinates(names...; right_handed=true)

# Arguments
- `names`: Names for each coordinate (e.g., "x", "y", "z")
- `right_handed`: Whether the coordinate system is right-handed (only used for 3D)

# Examples
```julia
# 1D
coords = CartesianCoordinates("x")

# 2D
coords = CartesianCoordinates("x", "z")

# 3D
coords = CartesianCoordinates("x", "y", "z")
coords["x"]  # Get x coordinate
coords[1]    # Get first coordinate
```
"""
struct CartesianCoordinates <: CoordinateSystem
    names::Vector{String}
    dim::Int
    coords::Vector{Coordinate}
    curvilinear::Bool
    right_handed::Union{Nothing, Bool}
    default_nonconst_groups::Tuple{Vararg{Int}}

    function CartesianCoordinates(names...; right_handed::Bool=true)
        names_vec = collect(String(name) for name in names)

        # Check for unique names ( coords:164-165)
        if length(Set(names_vec)) < length(names_vec)
            throw(ArgumentError("Must specify unique coordinate names."))
        end

        dim = length(names_vec)

        # Create coordinate objects
        # We need to create the CartesianCoordinates first, then set cs on coordinates
        coords_vec = Coordinate[]

        # Create the coordinate system first (with empty coords)
        cs = new(
            names_vec,
            dim,
            coords_vec,
            false,  # curvilinear = false for Cartesian
            dim == 3 ? right_handed : nothing,  # right_handed only for 3D
            ntuple(_ -> 1, dim)  # default_nonconst_groups = (1,) * dim
        )

        # Now create coordinates with reference to this coordinate system
        for name in names_vec
            coord = Coordinate(name; cs=cs)
            push!(coords_vec, coord)
        end

        return cs
    end
end

function Base.show(io::IO, cs::CartesianCoordinates)
    print(io, "{", join(cs.names, ","), "}")
end

function Base.:(==)(a::CartesianCoordinates, b::CartesianCoordinates)
    if length(a.coords) != length(b.coords)
        return false
    end
    if a.right_handed !== b.right_handed
        return false
    end
    for i in 1:length(a.coords)
        if a.coords[i] != b.coords[i]
            return false
        end
    end
    return true
end

function Base.hash(cs::CartesianCoordinates, h::UInt)
    return hash((cs.names, cs.right_handed), hash(:CartesianCoordinates, h))
end

"""
    check_bounds(cs::CartesianCoordinates, coord, bounds)

Check if bounds are valid. Cartesian coordinates have no restrictions.
"""
function check_bounds(cs::CartesianCoordinates, coord, bounds)
    # No restrictions for Cartesian coordinates
    return nothing
end

"""
    forward_vector_intertwiner(cs::CartesianCoordinates, subaxis, group)

Forward intertwiner for vector components. Identity for Cartesian coordinates.
Following coords:176-177.
"""
function forward_vector_intertwiner(cs::CartesianCoordinates, subaxis, group)
    return Matrix{Float64}(I, cs.dim, cs.dim)
end

"""
    backward_vector_intertwiner(cs::CartesianCoordinates, subaxis, group)

Backward intertwiner for vector components. Identity for Cartesian coordinates.
Following coords:179-180.
"""
function backward_vector_intertwiner(cs::CartesianCoordinates, subaxis, group)
    return Matrix{Float64}(I, cs.dim, cs.dim)
end

"""
    forward_intertwiner(cs::CartesianCoordinates, subaxis, order, group)

Forward intertwiner for tensor components. Identity for Cartesian coordinates.
"""
function forward_intertwiner(cs::CartesianCoordinates, subaxis, order, group)
    vector_int = forward_vector_intertwiner(cs, subaxis, group)
    return nkron(vector_int, order)
end

"""
    backward_intertwiner(cs::CartesianCoordinates, subaxis, order, group)

Backward intertwiner for tensor components. Identity for Cartesian coordinates.
"""
function backward_intertwiner(cs::CartesianCoordinates, subaxis, order, group)
    vector_int = backward_vector_intertwiner(cs, subaxis, group)
    return nkron(vector_int, order)
end

# ============================================================================
# DirectProduct: Direct product of coordinate systems
# Following coords:99-156
# ============================================================================

"""
    DirectProduct

Direct product of coordinate systems.
Following implementation in coords:99-156.

# Constructor
    DirectProduct(coordsystems...; right_handed=nothing)

# Examples
```julia
# Combine 1D systems
x_coord = CartesianCoordinates("x")
z_coord = CartesianCoordinates("z")
coords = DirectProduct(x_coord, z_coord)
```
"""
struct DirectProduct <: CoordinateSystem
    coordsystems::Tuple{Vararg{CoordinateSystem}}
    coords::Vector{Union{Coordinate, AzimuthalCoordinate}}
    names::Vector{String}
    dim::Int
    curvilinear::Bool
    right_handed::Union{Nothing, Bool}
    default_nonconst_groups::Tuple{Vararg{Int}}
    subaxis_by_cs::Dict{CoordinateSystem, Int}

    function DirectProduct(coordsystems...; right_handed::Union{Nothing, Bool}=nothing)
        # Collect all coordinates from component systems
        all_coords = Union{Coordinate, AzimuthalCoordinate}[]
        for cs in coordsystems
            append!(all_coords, get_coords(cs))
        end

        # Check for duplicate coordinates ( coords:107-108)
        if length(Set(all_coords)) < length(all_coords)
            throw(ArgumentError("Cannot repeat coordinates in DirectProduct."))
        end

        # Collect names
        names = String[]
        for cs in coordsystems
            append!(names, cs.names)
        end

        if length(Set(names)) < length(names)
            throw(ArgumentError("Cannot repeat coordinate names in DirectProduct."))
        end

        dim = sum(cs.dim for cs in coordsystems)

        # Determine curvilinear property
        curv = any(cs.curvilinear for cs in coordsystems)

        # Handle right_handed for 3D ( coords:110-117)
        if dim == 3
            if curv
                right_handed = right_handed === nothing ? false : right_handed
            else
                right_handed = right_handed === nothing ? true : right_handed
            end
        end

        # Build subaxis mapping
        subaxis_dict = Dict{CoordinateSystem, Int}()
        subaxis = 0
        for cs in coordsystems
            subaxis_dict[cs] = subaxis
            subaxis += cs.dim
        end

        # Collect default_nonconst_groups
        groups = Int[]
        for cs in coordsystems
            append!(groups, collect(cs.default_nonconst_groups))
        end

        new(
            coordsystems,
            all_coords,
            names,
            dim,
            curv,
            right_handed,
            Tuple(groups),
            subaxis_dict
        )
    end
end

"""
    forward_vector_intertwiner(cs::DirectProduct, subaxis, group)

Forward intertwiner for vector components in a direct product.
Following coords:132-141.
"""
function forward_vector_intertwiner(cs::DirectProduct, subaxis, group)
    factors = []
    start_axis = 0

    for sub_cs in cs.coordsystems
        if start_axis <= subaxis < start_axis + sub_cs.dim
            push!(factors, forward_vector_intertwiner(sub_cs, subaxis - start_axis, group))
        else
            push!(factors, Matrix{Float64}(I, sub_cs.dim, sub_cs.dim))
        end
        start_axis += sub_cs.dim
    end

    return sparse_block_diag(factors)
end

"""
    backward_vector_intertwiner(cs::DirectProduct, subaxis, group)

Backward intertwiner for vector components in a direct product.
Following coords:143-152.
"""
function backward_vector_intertwiner(cs::DirectProduct, subaxis, group)
    factors = []
    start_axis = 0

    for sub_cs in cs.coordsystems
        if start_axis <= subaxis < start_axis + sub_cs.dim
            push!(factors, backward_vector_intertwiner(sub_cs, subaxis - start_axis, group))
        else
            push!(factors, Matrix{Float64}(I, sub_cs.dim, sub_cs.dim))
        end
        start_axis += sub_cs.dim
    end

    return sparse_block_diag(factors)
end

"""
    forward_intertwiner(cs::DirectProduct, subaxis, order, group)

Forward intertwiner for tensor components.
"""
function forward_intertwiner(cs::DirectProduct, subaxis, order, group)
    vector_int = forward_vector_intertwiner(cs, subaxis, group)
    return nkron(vector_int, order)
end

"""
    backward_intertwiner(cs::DirectProduct, subaxis, order, group)

Backward intertwiner for tensor components.
"""
function backward_intertwiner(cs::DirectProduct, subaxis, order, group)
    vector_int = backward_vector_intertwiner(cs, subaxis, group)
    return nkron(vector_int, order)
end

# ============================================================================
# Helper functions
# ============================================================================

"""
    nkron(A, n)

N-fold Kronecker product of matrix A with itself.
"""
function nkron(A::AbstractMatrix, n::Int)
    if n < 0
        throw(ArgumentError("nkron requires non-negative n, got n=$n"))
    elseif n == 0
        return ones(eltype(A), 1, 1)
    elseif n == 1
        return A
    else
        result = A
        for _ in 2:n
            result = kron(result, A)
        end
        return result
    end
end

"""
    sparse_block_diag(matrices)

Create a sparse block diagonal matrix from a vector of matrices.
"""
function sparse_block_diag(matrices::Vector)
    if isempty(matrices)
        return spzeros(Float64, 0, 0)
    end

    # Delegate to the varargs implementation to preserve sparsity and element types.
    return sparse_block_diag(matrices...)
end

"""
    get_coords(cs::CoordinateSystem)

Get the coordinates from a coordinate system.
"""
function get_coords(cs::CartesianCoordinates)
    return cs.coords
end

function get_coords(cs::DirectProduct)
    return cs.coords
end

"""
    coords(coordsys::CoordinateSystem)

Return tuple of Coordinate objects for this coordinate system.
Following pattern.
"""
function coords(coordsys::CartesianCoordinates)
    return tuple(coordsys.coords...)
end

function coords(coordsys::DirectProduct)
    return tuple(coordsys.coords...)
end

# ============================================================================
# Indexing
# ============================================================================

function Base.getindex(coordsys::CartesianCoordinates, key::String)
    index = findfirst(==(key), coordsys.names)
    if index === nothing
        throw(KeyError("Coordinate '$key' not found"))
    end
    return coordsys.coords[index]
end

function Base.getindex(coordsys::CartesianCoordinates, key::Int)
    if key < 1 || key > coordsys.dim
        throw(BoundsError(coordsys.coords, key))
    end
    return coordsys.coords[key]
end

function Base.getindex(coordsys::DirectProduct, key::String)
    index = findfirst(==(key), coordsys.names)
    if index === nothing
        throw(KeyError("Coordinate '$key' not found"))
    end
    return coordsys.coords[index]
end

function Base.getindex(coordsys::DirectProduct, key::Int)
    if key < 1 || key > coordsys.dim
        throw(BoundsError(coordsys.coords, key))
    end
    return coordsys.coords[key]
end

# Legacy getindex for generic CoordinateSystem (for backwards compatibility)
function Base.getindex(cs::CoordinateSystem, key::Union{String, Int})
    if hasfield(typeof(cs), :coords) && hasfield(typeof(cs), :names)
        coords_vec = getfield(cs, :coords)
        names_vec = getfield(cs, :names)
        if isa(key, String)
            index = findfirst(==(key), names_vec)
            if index === nothing
                throw(KeyError("Coordinate '$key' not found"))
            end
            return coords_vec[index]
        else
            if key < 1 || key > length(names_vec)
                throw(BoundsError(names_vec, key))
            end
            return coords_vec[key]
        end
    end

    if isa(key, String)
        if !hasfield(typeof(cs), :names)
            throw(ArgumentError("Coordinate system does not define named coordinates"))
        end
        index = findfirst(==(key), cs.names)
        if index === nothing
            throw(KeyError("Coordinate '$key' not found"))
        end
        return Coordinate(key; cs=cs)
    else
        if !hasfield(typeof(cs), :names)
            throw(ArgumentError("Coordinate system does not define named coordinates"))
        end
        if key < 1 || key > length(cs.names)
            throw(BoundsError(cs.names, key))
        end
        return Coordinate(cs.names[key]; cs=cs)
    end
end

# ============================================================================
# Note: unit_vector_fields has been moved to field.jl to avoid circular dependency
# (it needs VectorField which is defined in field.jl, but coords.jl is included first)
# ============================================================================

# ============================================================================
# Note: local_grids is implemented in distributor.jl
# The distributor's local_grids method handles coordinate grid generation
# by delegating to individual basis local_grids methods
# ============================================================================

# ============================================================================
# Exports
# ============================================================================

# Export types
export CoordinateSystem, Coordinate, AzimuthalCoordinate,
       CartesianCoordinates, DirectProduct

# Export functions
export check_bounds, forward_vector_intertwiner, backward_vector_intertwiner,
       forward_intertwiner, backward_intertwiner,
       nkron, sparse_block_diag, get_coords, coords
