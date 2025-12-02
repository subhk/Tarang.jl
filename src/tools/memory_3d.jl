"""
3D Memory optimization utilities

Specialized memory management for large-scale 3D simulations
"""

using MPI

# Memory pool for 3D arrays
mutable struct Memory3DPool
    free_arrays::Dict{Tuple{Int,Int,Int}, Vector{Array{Float64,3}}}
    allocated_arrays::Set{Array{Float64,3}}
    total_allocated::Int
    max_allocation::Int
    
    function Memory3DPool()
        new(Dict{Tuple{Int,Int,Int}, Vector{Array{Float64,3}}}(), 
            Set{Array{Float64,3}}(), 0, 0)
    end
end

const global_3d_pool = Memory3DPool()

function allocate_3d_array(shape::Tuple{Int,Int,Int}, dtype::Type=Float64)
    """Allocate 3D array from memory pool"""
    
    pool = global_3d_pool
    
    # Check if we have a free array of the right size
    if haskey(pool.free_arrays, shape) && !isempty(pool.free_arrays[shape])
        arr = pop!(pool.free_arrays[shape])
        fill!(arr, 0.0)  # Clear the array
        push!(pool.allocated_arrays, arr)
        return arr
    end
    
    # Allocate new array
    arr = zeros(dtype, shape...)
    push!(pool.allocated_arrays, arr)
    
    # Track memory usage
    bytes = prod(shape) * sizeof(dtype)
    pool.total_allocated += bytes
    pool.max_allocation = max(pool.max_allocation, pool.total_allocated)
    
    return arr
end

function deallocate_3d_array!(arr::Array{T,3}) where T
    """Return 3D array to memory pool"""
    
    pool = global_3d_pool
    
    if arr in pool.allocated_arrays
        delete!(pool.allocated_arrays, arr)
        
        shape = size(arr)
        if !haskey(pool.free_arrays, shape)
            pool.free_arrays[shape] = Array{T,3}[]
        end
        
        push!(pool.free_arrays[shape], arr)
        
        # Update memory tracking
        bytes = prod(shape) * sizeof(T)
        pool.total_allocated -= bytes
    end
end

function clear_3d_memory_pool!()
    """Clear the entire 3D memory pool"""
    
    pool = global_3d_pool
    empty!(pool.free_arrays)
    empty!(pool.allocated_arrays)
    pool.total_allocated = 0
    pool.max_allocation = 0
    
    # Force garbage collection
    GC.gc()
end

function get_3d_memory_stats()
    """Get 3D memory pool statistics"""
    
    pool = global_3d_pool
    
    total_free = 0
    free_count = 0
    
    for (shape, arrays) in pool.free_arrays
        bytes_per_array = prod(shape) * sizeof(Float64)
        total_free += length(arrays) * bytes_per_array
        free_count += length(arrays)
    end
    
    allocated_count = length(pool.allocated_arrays)
    
    return Dict(
        "allocated_arrays" => allocated_count,
        "free_arrays" => free_count,
        "allocated_bytes" => pool.total_allocated,
        "free_bytes" => total_free,
        "max_allocated" => pool.max_allocation,
        "total_bytes" => pool.total_allocated + total_free
    )
end

# In-place 3D operations to reduce memory allocation
function add_3d_inplace!(result::Array{T,3}, a::Array{T,3}, b::Array{T,3}) where T
    """In-place addition: result .= a .+ b"""
    @. result = a + b
    return result
end

function multiply_3d_inplace!(result::Array{T,3}, a::Array{T,3}, scalar::Real) where T
    """In-place scalar multiplication: result .= a .* scalar"""
    @. result = a * scalar
    return result
end

function copy_3d_inplace!(dest::Array{T,3}, src::Array{T,3}) where T
    """In-place copy: dest .= src"""
    dest .= src
    return dest
end

# Optimized 3D field operations
function compute_3d_gradient_inplace!(grad_x::Array{T,3}, grad_y::Array{T,3}, grad_z::Array{T,3},
                                     field::Array{T,3}, dx::Real, dy::Real, dz::Real) where T
    """Compute 3D gradient in-place using finite differences"""
    
    nx, ny, nz = size(field)
    
    # x-derivative
    for k in 1:nz, j in 1:ny
        grad_x[1, j, k] = (field[2, j, k] - field[nx, j, k]) / (2*dx)  # Periodic
        grad_x[nx, j, k] = (field[1, j, k] - field[nx-1, j, k]) / (2*dx)
        for i in 2:nx-1
            grad_x[i, j, k] = (field[i+1, j, k] - field[i-1, j, k]) / (2*dx)
        end
    end
    
    # y-derivative
    for k in 1:nz, i in 1:nx
        grad_y[i, 1, k] = (field[i, 2, k] - field[i, ny, k]) / (2*dy)  # Periodic
        grad_y[i, ny, k] = (field[i, 1, k] - field[i, ny-1, k]) / (2*dy)
        for j in 2:ny-1
            grad_y[i, j, k] = (field[i, j+1, k] - field[i, j-1, k]) / (2*dy)
        end
    end
    
    # z-derivative
    for j in 1:ny, i in 1:nx
        grad_z[i, j, 1] = (field[i, j, 2] - field[i, j, nz]) / (2*dz)  # Periodic
        grad_z[i, j, nz] = (field[i, j, 1] - field[i, j, nz-1]) / (2*dz)
        for k in 2:nz-1
            grad_z[i, j, k] = (field[i, j, k+1] - field[i, j, k-1]) / (2*dz)
        end
    end
end

function compute_3d_divergence_inplace!(div_result::Array{T,3}, 
                                       ux::Array{T,3}, uy::Array{T,3}, uz::Array{T,3},
                                       dx::Real, dy::Real, dz::Real) where T
    """Compute 3D divergence in-place"""
    
    nx, ny, nz = size(ux)
    
    # Allocate temporary arrays from pool
    dux_dx = allocate_3d_array(size(ux))
    duy_dy = allocate_3d_array(size(uy))
    duz_dz = allocate_3d_array(size(uz))
    
    try
        # Compute partial derivatives
        compute_3d_gradient_inplace!(dux_dx, allocate_3d_array(size(ux)), allocate_3d_array(size(ux)), 
                                   ux, dx, dy, dz)
        compute_3d_gradient_inplace!(allocate_3d_array(size(uy)), duy_dy, allocate_3d_array(size(uy)), 
                                   uy, dx, dy, dz)
        compute_3d_gradient_inplace!(allocate_3d_array(size(uz)), allocate_3d_array(size(uz)), duz_dz, 
                                   uz, dx, dy, dz)
        
        # Sum partial derivatives
        @. div_result = dux_dx + duy_dy + duz_dz
        
    finally
        # Return arrays to pool
        deallocate_3d_array!(dux_dx)
        deallocate_3d_array!(duy_dy)
        deallocate_3d_array!(duz_dz)
    end
end

function compute_3d_curl_inplace!(curl_x::Array{T,3}, curl_y::Array{T,3}, curl_z::Array{T,3},
                                 ux::Array{T,3}, uy::Array{T,3}, uz::Array{T,3},
                                 dx::Real, dy::Real, dz::Real) where T
    """Compute 3D curl in-place"""
    
    # Allocate temporary arrays for partial derivatives
    temp_arrays = [allocate_3d_array(size(ux)) for _ in 1:6]
    dux_dy, dux_dz, duy_dx, duy_dz, duz_dx, duz_dy = temp_arrays
    
    try
        # Compute all needed partial derivatives
        compute_3d_gradient_inplace!(allocate_3d_array(size(ux)), dux_dy, dux_dz, ux, dx, dy, dz)
        compute_3d_gradient_inplace!(duy_dx, allocate_3d_array(size(uy)), duy_dz, uy, dx, dy, dz)
        compute_3d_gradient_inplace!(duz_dx, duz_dy, allocate_3d_array(size(uz)), uz, dx, dy, dz)
        
        # Compute curl components
        @. curl_x = duz_dy - duy_dz
        @. curl_y = dux_dz - duz_dx  
        @. curl_z = duy_dx - dux_dy
        
    finally
        # Return arrays to pool
        for arr in temp_arrays
            deallocate_3d_array!(arr)
        end
    end
end

# Memory monitoring for 3D simulations
function monitor_3d_memory_usage(rank::Int=0)
    """Monitor and log 3D memory usage"""
    
    # Julia memory info
    julia_stats = Base.gc_num()
    julia_mem_mb = julia_stats.allocd / 1024^2
    
    # Pool memory info
    pool_stats = get_3d_memory_stats()
    pool_mem_mb = pool_stats["total_bytes"] / 1024^2
    
    # System memory (if available)
    try
        sys_mem_kb = parse(Int, readchomp(`grep MemAvailable /proc/meminfo` |> x -> split(x)[2]))
        sys_mem_gb = sys_mem_kb / 1024^2
        
        if MPI.Initialized()
            @info "[Rank $rank] Memory usage: Julia=$(round(julia_mem_mb, digits=1))MB, " *
                  "Pool=$(round(pool_mem_mb, digits=1))MB, System=$(round(sys_mem_gb, digits=1))GB available"
        else
            @info "Memory usage: Julia=$(round(julia_mem_mb, digits=1))MB, " *
                  "Pool=$(round(pool_mem_mb, digits=1))MB, System=$(round(sys_mem_gb, digits=1))GB available"
        end
    catch
        # Fallback if /proc/meminfo not available
        if MPI.Initialized()
            @info "[Rank $rank] Memory usage: Julia=$(round(julia_mem_mb, digits=1))MB, " *
                  "Pool=$(round(pool_mem_mb, digits=1))MB"
        else
            @info "Memory usage: Julia=$(round(julia_mem_mb, digits=1))MB, " *
                  "Pool=$(round(pool_mem_mb, digits=1))MB"
        end
    end
end

# Memory-efficient 3D I/O
function write_3d_field_slice(field::Array{T,3}, filename::String, slice_dim::Int, slice_index::Int) where T
    """Write a 2D slice of 3D field to reduce I/O overhead"""
    
    if slice_dim == 1
        slice_data = field[slice_index, :, :]
    elseif slice_dim == 2
        slice_data = field[:, slice_index, :]
    elseif slice_dim == 3
        slice_data = field[:, :, slice_index]
    else
        throw(ArgumentError("slice_dim must be 1, 2, or 3"))
    end
    
    # Write slice (would integrate with HDF5 for actual implementation)
    # For now, just save as binary
    open(filename, "w") do io
        write(io, size(slice_data))
        write(io, slice_data)
    end
end

function read_3d_field_slice(filename::String, dtype::Type=Float64)
    """Read a 2D slice from file"""
    
    open(filename, "r") do io
        dims = read(io, Tuple{Int,Int})
        slice_data = zeros(dtype, dims...)
        read!(io, slice_data)
        return slice_data
    end
end

# Spectral methods memory utilities focused on 3D operations
# (Halo exchanges not needed for spectral methods - removed in favor of proper spectral utilities)