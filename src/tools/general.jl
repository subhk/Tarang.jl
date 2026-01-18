"""
General utility functions
"""

using OrderedCollections
using Logging

export CPUDeviceConfig, DEFAULT_DEVICE, device_config,
       device_zeros, device_ones, device_fill, device_similar,
       move_to_device, move_to_host, check_device_compatibility,
       default_memory_info,
       OrderedSet, DeferredTuple,
       unify_attributes, unify,
       is_complex_dtype, is_real_dtype, promote_dtype,
       apply_along_axis, apply_along_axis_fast,
       axis_sum, axis_mean, axis_maximum, axis_minimum, axis_sort,
       broadcast_to_shape,
       format_time, safe_divide, clamp_to_range,
       deep_merge!, flatten_dict,
       ProgressTracker, update!,
       validate_positive, validate_nonnegative, validate_in_range,
       get_memory_usage, setup_logger, close_logger

# ---------------------------------------------------------------------------
# CPU-only device configuration (GPU support available via separate extension)
# ---------------------------------------------------------------------------

const CPU_DEVICE = :cpu

"""
Lightweight device configuration used for CPU-only execution.
"""
struct CPUDeviceConfig
    device_type::Symbol
end

CPUDeviceConfig() = CPUDeviceConfig(CPU_DEVICE)
const DEFAULT_DEVICE = CPUDeviceConfig()
const device_config = DEFAULT_DEVICE
const DEFAULT_MEMORY_INFO = (total=typemax(Int64), available=typemax(Int64), used=0)

default_memory_info() = DEFAULT_MEMORY_INFO

# Basic array helpers that ignore device placement and always use CPU arrays.
device_zeros(::Type{T}, dims, ::Any=CPUDeviceConfig()) where {T} = zeros(T, dims)
device_ones(::Type{T}, dims, ::Any=CPUDeviceConfig()) where {T} = ones(T, dims)
device_fill(val, dims, ::Any=CPUDeviceConfig()) = fill(val, dims)
device_similar(arr::AbstractArray, ::Any=CPUDeviceConfig()) = similar(arr)
move_to_device(x, ::Any=CPUDeviceConfig()) = x
move_to_host(x) = x
check_device_compatibility(::Any, ::Any=CPUDeviceConfig()) = true

# Ordered set implementation
struct OrderedSet{T}
    dict::OrderedDict{T, Nothing}
    
    function OrderedSet{T}() where T
        new{T}(OrderedDict{T, Nothing}())
    end
    
    function OrderedSet{T}(items) where T
        os = new{T}(OrderedDict{T, Nothing}())
        for item in items
            os.dict[item] = nothing
        end
        return os
    end
end

function OrderedSet(items)
    if isempty(items)
        return OrderedSet{Any}()
    else
        # Use eltype for collections, or promote_type for iterables
        T = try
            eltype(items)
        catch
            # Fallback: collect and get common type
            collected = collect(items)
            isempty(collected) ? Any : mapreduce(typeof, promote_type, collected)
        end
        # If eltype is Any or abstract, keep it; otherwise use specific type
        return OrderedSet{T}(items)
    end
end

OrderedSet() = OrderedSet{Any}()

# Convenience constructor from varargs
OrderedSet(items...) = OrderedSet(items)

Base.push!(os::OrderedSet{T}, item) where T = (os.dict[convert(T, item)] = nothing; os)
Base.in(item, os::OrderedSet) = haskey(os.dict, item)
Base.length(os::OrderedSet) = length(os.dict)
Base.isempty(os::OrderedSet) = isempty(os.dict)
Base.eltype(::Type{OrderedSet{T}}) where T = T
Base.iterate(os::OrderedSet) = iterate(keys(os.dict))
Base.iterate(os::OrderedSet, state) = iterate(keys(os.dict), state)
Base.collect(os::OrderedSet) = collect(keys(os.dict))

# Deferred tuple for lazy evaluation
struct DeferredTuple{T}
    items::Vector{T}
    
    function DeferredTuple{T}(items::Vector{T}) where T
        new{T}(items)
    end
end

function DeferredTuple(items...)
    if isempty(items)
        return DeferredTuple{Any}(Any[])
    else
        # Compute common type for all items using promote_type
        T = mapreduce(typeof, promote_type, items)
        return DeferredTuple{T}(collect(T, items))
    end
end

Base.length(dt::DeferredTuple) = length(dt.items)
Base.getindex(dt::DeferredTuple, i::Int) = dt.items[i]
Base.getindex(dt::DeferredTuple, r::AbstractRange) = dt.items[r]
Base.iterate(dt::DeferredTuple) = iterate(dt.items)
Base.iterate(dt::DeferredTuple, state) = iterate(dt.items, state)
Base.eltype(::Type{DeferredTuple{T}}) where T = T
Base.firstindex(dt::DeferredTuple) = firstindex(dt.items)
Base.lastindex(dt::DeferredTuple) = lastindex(dt.items)
Base.isempty(dt::DeferredTuple) = isempty(dt.items)

# Attribute unification functions
"""Unify attributes across objects, ensuring consistency."""
function unify_attributes(objects, attr_name::String)
    attrs = Any[]
    
    for obj in objects
        if hasfield(typeof(obj), Symbol(attr_name))
            attr = getfield(obj, Symbol(attr_name))
            push!(attrs, attr)
        end
    end
    
    if isempty(attrs)
        return nothing
    end
    
    # Check if all attributes are the same
    first_attr = attrs[1]
    if all(attr == first_attr for attr in attrs)
        return first_attr
    else
        throw(ArgumentError("Inconsistent attributes: $(attrs)"))
    end
end

"""Unify objects, returning the common object if all are the same."""
function unify(objects)
    if isempty(objects)
        return nothing
    end
    
    first_obj = objects[1]
    if all(obj == first_obj for obj in objects)
        return first_obj
    else
        throw(ArgumentError("Objects are not unified: $(objects)"))
    end
end

# Type checking utilities
"""Check if data type is complex."""
function is_complex_dtype(dtype::Type)
    return dtype <: Complex
end

"""Check if data type is real."""
function is_real_dtype(dtype::Type)
    return dtype <: Real
end

"""Promote data types to common type."""
function promote_dtype(dtypes...)
    return promote_type(dtypes...)
end

# Array utilities
"""
Apply function along specified axis following NumPy patterns.

Parameters:
- func: Function to apply to 1-D slices along the axis
- axis: Axis along which arr is sliced
- arr: Input array
- args: Additional positional arguments to func
- kwargs: Additional keyword arguments to func

Returns:
- Output array with shape modified according to func output
"""
function apply_along_axis(func::Function, axis::Int, arr::AbstractArray; args=(), kwargs=())
    # Validate axis
    if axis < 1 || axis > ndims(arr)
        throw(ArgumentError("axis $axis is out of bounds for array of dimension $(ndims(arr))"))
    end
    
    # Get array shape and compute output shape template
    arr_shape = size(arr)
    
    # Extract a sample slice along the axis to determine output shape and type
    sample_indices = [i == axis ? Colon() : 1 for i in 1:ndims(arr)]
    sample_slice = arr[sample_indices...]
    sample_1d = ndims(sample_slice) == 1 ? sample_slice : vec(sample_slice)
    
    # Apply function to sample to determine output properties
    sample_output = func(sample_1d, args...; kwargs...)
    
    # Determine output shape
    output_shape_axis = isa(sample_output, AbstractArray) ? size(sample_output) : nothing
    if output_shape_axis !== nothing
        # Replace the axis dimension with the function output dimensions
        output_shape = tuple(
            arr_shape[1:axis-1]..., 
            output_shape_axis..., 
            arr_shape[axis+1:end]...
        )
    else
        # Scalar output - remove the axis dimension
        output_shape = tuple(arr_shape[1:axis-1]..., arr_shape[axis+1:end]...)
    end
    
    # Initialize output array
    output_type = typeof(sample_output)
    if isa(sample_output, AbstractArray)
        output_elem_type = eltype(sample_output)
        output = Array{output_elem_type}(undef, output_shape)
    else
        output = Array{output_type}(undef, output_shape)
    end
    
    # Apply function along the specified axis
    # Create iteration indices for all dimensions except the specified axis
    iter_shape = tuple(arr_shape[1:axis-1]..., arr_shape[axis+1:end]...)
    
    if isempty(iter_shape)
        # Special case: 1D array
        slice_1d = vec(arr)
        result = func(slice_1d, args...; kwargs...)
        if isa(result, AbstractArray)
            if output_shape_axis === nothing || size(result) != output_shape_axis
                throw(ArgumentError("Inconsistent output shape from apply_along_axis: " *
                                    "expected $output_shape_axis, got $(size(result))"))
            end
            output[:] = result
        else
            output[] = result
        end
    else
        # General case: iterate over all combinations of indices except the axis dimension
        for cart_idx in CartesianIndices(iter_shape)
            # Convert CartesianIndex to linear indices for dimensions before and after axis
            before_len = axis - 1
            before_indices = before_len == 0 ? () : cart_idx.I[1:before_len]
            after_indices = before_len == length(cart_idx.I) ? () : cart_idx.I[(before_len + 1):end]
            
            # Extract 1-D slice along the axis
            slice_indices = (before_indices..., Colon(), after_indices...)
            slice_data = arr[slice_indices...]
            
            # Ensure we have a 1-D vector
            if ndims(slice_data) != 1
                slice_1d = vec(slice_data)
            else
                slice_1d = slice_data
            end
            
            # Apply function
            result = func(slice_1d, args...; kwargs...)
            
            # Store result in output array
            if isa(result, AbstractArray)
                if size(result) != output_shape_axis
                    throw(ArgumentError("Inconsistent output shape from apply_along_axis: " *
                                        "expected $output_shape_axis, got $(size(result))"))
                end
                output_indices = (before_indices..., ntuple(_ -> Colon(), ndims(result))..., after_indices...)
                output[output_indices...] = result
            else
                # Scalar result
                output[cart_idx] = result
            end
        end
    end
    
    return output
end

# Convenient wrapper functions for common axis operations
"""Convenience wrapper for apply_along_axis with positional args."""
function apply_along_axis(func::Function, axis::Int, arr::AbstractArray, args...)
    return apply_along_axis(func, axis, arr; args=args, kwargs=())
end

"""Sum along specified axis (NumPy-style)."""
function axis_sum(arr::AbstractArray, axis::Int)
    return apply_along_axis(sum, axis, arr)
end

"""Mean along specified axis (NumPy-style)."""
function axis_mean(arr::AbstractArray, axis::Int)
    return apply_along_axis(x -> isempty(x) ? NaN : sum(x) / length(x), axis, arr)
end

"""Maximum along specified axis (NumPy-style)."""
function axis_maximum(arr::AbstractArray, axis::Int)
    return apply_along_axis(maximum, axis, arr)
end

"""Minimum along specified axis (NumPy-style)."""
function axis_minimum(arr::AbstractArray, axis::Int)
    return apply_along_axis(minimum, axis, arr)
end

"""Sort along specified axis (NumPy-style)."""
function axis_sort(arr::AbstractArray, axis::Int)
    return apply_along_axis(sort, axis, arr)
end

# Fast version using Julia's mapslices when function output shape is predictable
"""
Fast version of apply_along_axis using Julia's mapslices.

This version is more efficient for functions that:
1. Return arrays of the same size as input
2. Return scalars
3. Have predictable output shapes

For general functions with unpredictable output shapes, use apply_along_axis.
"""
function apply_along_axis_fast(func::Function, axis::Int, arr::AbstractArray)
    # Validate axis
    if axis < 1 || axis > ndims(arr)
        throw(ArgumentError("axis $axis is out of bounds for array of dimension $(ndims(arr))"))
    end
    
    return mapslices(func, arr, dims=axis)
end

"""Broadcast array to target shape."""
function broadcast_to_shape(arr::AbstractArray, shape::Tuple)
    broadcast_shape = Base.Broadcast.broadcast_shape(size(arr), shape)
    if broadcast_shape != shape
        throw(ArgumentError("Target shape $shape is not broadcast-compatible with $(size(arr))"))
    end

    out = similar(arr, shape)
    out .= arr
    return out
end

# String utilities
# Note: split_equation, split_call, lambdify_functions are defined in tools/parsing.jl
# Note: parse_expression is implemented in core/problems.jl

"""Format time duration in human readable format."""
function format_time(seconds::Float64)
    # Guard against non-finite input
    if !isfinite(seconds)
        return "?"
    end
    if seconds < 0
        seconds = 0.0
    end
    if seconds < 60
        return "$(round(seconds, digits=2))s"
    elseif seconds < 3600
        minutes = floor(Int, seconds / 60)
        secs = seconds - minutes * 60
        return "$(minutes)m $(round(secs, digits=1))s"
    else
        hours = floor(Int, seconds / 3600)
        remaining = seconds - hours * 3600
        minutes = floor(Int, remaining / 60)
        secs = remaining - minutes * 60
        return "$(hours)h $(minutes)m $(round(secs, digits=1))s"
    end
end

# Numerical utilities
"""Safe division with default for division by zero."""
function safe_divide(a, b; default=0.0)
    return b == 0 ? default : a / b
end

"""Clamp value to range."""
function clamp_to_range(x, min_val, max_val)
    return clamp(x, min_val, max_val)
end

# Dictionary utilities
"""Deep merge dictionary dict2 into dict1."""
function deep_merge!(dict1::Dict, dict2::Dict)
    for (key, value) in dict2
        if haskey(dict1, key) && isa(dict1[key], Dict) && isa(value, Dict)
            deep_merge!(dict1[key], value)
        else
            dict1[key] = value
        end
    end
    return dict1
end

"""Flatten nested dictionary."""
function flatten_dict(d::Dict, parent_key::String="", sep::String=".")
    items = Pair{String, Any}[]
    
    for (key, value) in d
        new_key = isempty(parent_key) ? string(key) : "$parent_key$sep$key"
        
        if isa(value, Dict)
            append!(items, flatten_dict(value, new_key, sep))
        else
            push!(items, new_key => value)
        end
    end
    
    return Dict(items)
end

# Progress tracking
mutable struct ProgressTracker
    total::Int
    current::Int
    start_time::Float64
    last_update_time::Float64
    update_interval::Float64

    function ProgressTracker(total::Int; update_interval::Float64=1.0)
        if total < 0
            throw(ArgumentError("ProgressTracker: total must be non-negative, got $total"))
        end
        if update_interval <= 0
            throw(ArgumentError("ProgressTracker: update_interval must be positive, got $update_interval"))
        end
        start_time = time()
        new(total, 0, start_time, start_time, update_interval)
    end
end

"""Update progress tracker."""
function update!(tracker::ProgressTracker, current::Int=tracker.current + 1)
    tracker.current = current
    current_time = time()

    if current_time - tracker.last_update_time >= tracker.update_interval || current >= tracker.total
        elapsed = current_time - tracker.start_time

        # Handle edge cases for rate and ETA calculation
        if elapsed > 0 && current > 0
            rate = current / elapsed
            remaining = tracker.total - current
            eta = rate > 0 ? remaining / rate : Inf
            eta_str = isfinite(eta) ? format_time(eta) : "calculating..."
        else
            rate = 0.0
            eta_str = "calculating..."
        end

        percent = tracker.total > 0 ? (current / tracker.total) * 100 : 0.0

        @info "Progress: $current/$(tracker.total) ($(round(percent, digits=1))%), " *
              "Rate: $(round(rate, digits=2))/s, ETA: $eta_str"

        tracker.last_update_time = current_time
    end
end

# Validation utilities
"""Validate that value is positive."""
function validate_positive(value, name::String)
    # Check for NaN first (NaN comparisons always return false)
    if isnan(value) || value <= 0
        throw(ArgumentError("$name must be positive, got $value"))
    end
    return value
end

"""Validate that value is non-negative."""
function validate_nonnegative(value, name::String)
    # Check for NaN first (NaN comparisons always return false)
    if isnan(value) || value < 0
        throw(ArgumentError("$name must be non-negative, got $value"))
    end
    return value
end

"""Validate that value is in specified range."""
function validate_in_range(value, min_val, max_val, name::String)
    # Check for NaN first (NaN comparisons always return false)
    if isnan(value) || value < min_val || value > max_val
        throw(ArgumentError("$name must be in range [$min_val, $max_val], got $value"))
    end
    return value
end

# Memory utilities
"""Get current memory usage."""
function get_memory_usage()
    # Julia-specific memory info
    stats = Base.gc_num()
    return Dict(
        "allocated" => stats.allocd,
        "freed" => stats.freed,
        "total_time" => stats.total_time / 1e9  # Convert to seconds
    )
end

# Logging utilities
# Track the current log file handle for proper cleanup
const _LOG_FILE_HANDLE = Ref{Union{IO, Nothing}}(nothing)

"""
Setup logging configuration.

Args:
    level: Log level ("DEBUG", "INFO", "WARN", "ERROR")
    filename: Optional file path to log to. If nothing, logs to stderr.

Note: Call close_logger() to properly close file handles when done.
"""
function setup_logger(level::String="INFO", filename::Union{String, Nothing}=nothing)
    # Close any existing log file
    close_logger()

    log_level = if uppercase(level) == "DEBUG"
        Logging.Debug
    elseif uppercase(level) == "INFO"
        Logging.Info
    elseif uppercase(level) == "WARN"
        Logging.Warn
    elseif uppercase(level) == "ERROR"
        Logging.Error
    else
        Logging.Info
    end

    if filename !== nothing
        # Log to file - ensure directory exists and handle errors
        try
            dir = dirname(filename)
            if !isempty(dir) && !isdir(dir)
                mkpath(dir)
            end
            io = open(filename, "a")
            _LOG_FILE_HANDLE[] = io
            logger = SimpleLogger(io, log_level)
            global_logger(logger)
        catch e
            @warn "Failed to open log file '$filename': $e. Falling back to console."
            logger = ConsoleLogger(stderr, log_level)
            global_logger(logger)
        end
    else
        # Log to console
        logger = ConsoleLogger(stderr, log_level)
        global_logger(logger)
    end
end

"""Close the current log file handle if one is open."""
function close_logger()
    if _LOG_FILE_HANDLE[] !== nothing
        try
            flush(_LOG_FILE_HANDLE[])
            close(_LOG_FILE_HANDLE[])
        catch
            # Ignore errors during cleanup
        end
        _LOG_FILE_HANDLE[] = nothing
    end
    # Reset to default console logger
    global_logger(ConsoleLogger(stderr, Logging.Info))
end

# Register cleanup on exit
atexit(close_logger)
