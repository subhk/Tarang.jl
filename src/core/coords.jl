"""
Coordinate system classes

Translated from dedalus/core/coords.py
"""

using StaticArrays
using LinearAlgebra

abstract type CoordinateSystem end

struct Coordinate
    coordsys::CoordinateSystem
    index::Int
    name::String
end

struct DirectProduct <: CoordinateSystem
    components::Tuple{Vararg{CoordinateSystem}}
    names::Vector{String}
    
    function DirectProduct(components...)
        names = vcat([cs.names for cs in components]...)
        new(components, names)
    end
end

struct CartesianCoordinates <: CoordinateSystem
    names::Vector{String}
    dim::Int
    
    function CartesianCoordinates(names...)
        names_vec = collect(String(name) for name in names)
        new(names_vec, length(names_vec))
    end
end

struct PolarCoordinates <: CoordinateSystem
    names::Vector{String}
    dim::Int
    
    function PolarCoordinates(names...)
        names_vec = collect(String(name) for name in names)
        if length(names_vec) != 2
            throw(ArgumentError("Polar coordinates must have exactly 2 dimensions"))
        end
        new(names_vec, 2)
    end
end

struct SphericalCoordinates <: CoordinateSystem
    names::Vector{String}
    dim::Int
    
    function SphericalCoordinates(names...)
        names_vec = collect(String(name) for name in names)
        if length(names_vec) != 3
            throw(ArgumentError("Spherical coordinates must have exactly 3 dimensions"))
        end
        new(names_vec, 3)
    end
end

struct S2Coordinates <: CoordinateSystem
    names::Vector{String}
    dim::Int
    
    function S2Coordinates(names...)
        names_vec = collect(String(name) for name in names)
        if length(names_vec) != 2
            throw(ArgumentError("S2 coordinates must have exactly 2 dimensions"))
        end
        new(names_vec, 2)
    end
end

struct AzimuthalCoordinate <: CoordinateSystem
    names::Vector{String}
    dim::Int
    
    function AzimuthalCoordinate(name)
        new([String(name)], 1)
    end
end

function Base.getindex(coordsys::CoordinateSystem, key::Union{String, Int})
    if isa(key, String)
        index = findfirst(==(key), coordsys.names)
        if index === nothing
            throw(KeyError("Coordinate '$key' not found"))
        end
        return Coordinate(coordsys, index, key)
    else
        if key < 1 || key > length(coordsys.names)
            throw(BoundsError(coordsys.names, key))
        end
        return Coordinate(coordsys, key, coordsys.names[key])
    end
end

function unit_vector_fields(coordsys::CoordinateSystem, dist)
    # Return unit vector fields for each coordinate
    # Following Dedalus implementation in coords.py:183
    fields = VectorField[]
    for (i, coord) in enumerate(coordsys.coords)
        # Create vector field for each coordinate direction
        ec = VectorField(dist, coordsys, name="e$(coord.name)")
        
        # Set the i-th component to 1 (unit vector in that direction)
        # In Dedalus: ec['g'][i] = 1
        # This means the i-th component of the vector field is set to 1
        for j in 1:length(ec.components)
            if j == i
                # Set the i-th component to 1 (unit vector in that direction)
                fill!(ec.components[j]["g"], 1.0)
            else
                # Set all other components to 0
                fill!(ec.components[j]["g"], 0.0)
            end
        end
        
        push!(fields, ec)
    end
    return tuple(fields...)
end

# Note: local_grids is implemented in distributor.jl
# The distributor's local_grids method handles coordinate grid generation
# by delegating to individual basis local_grids methods

# Add coords property to coordinate systems
# Following Dedalus pattern where coords is a tuple of Coordinate objects
function coords(coordsys::CoordinateSystem)
    """Return tuple of Coordinate objects for this coordinate system."""
    return tuple([Coordinate(coordsys, i, name) for (i, name) in enumerate(coordsys.names)]...)
end