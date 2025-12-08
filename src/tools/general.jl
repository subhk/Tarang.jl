"""
General utility functions
"""

using OrderedCollections
using Logging

# ---------------------------------------------------------------------------
# CPU-only device stubs (GPU support removed)
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
        T = typeof(first(items))
        return OrderedSet{T}(items)
    end
end

OrderedSet() = OrderedSet{Any}()

Base.push!(os::OrderedSet{T}, item::T) where T = (os.dict[item] = nothing; os)
Base.in(item, os::OrderedSet) = haskey(os.dict, item)
Base.length(os::OrderedSet) = length(os.dict)
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
        T = typeof(first(items))
        return DeferredTuple{T}(collect(T, items))
    end
end

Base.length(dt::DeferredTuple) = length(dt.items)
Base.getindex(dt::DeferredTuple, i::Int) = dt.items[i]
Base.iterate(dt::DeferredTuple) = iterate(dt.items)
Base.iterate(dt::DeferredTuple, state) = iterate(dt.items, state)

# Attribute unification functions
function unify_attributes(objects, attr_name::String)
    """Unify attributes across objects, ensuring consistency"""
    attrs = []
    
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

function unify(objects)
    """Unify objects, returning the common object if all are the same"""
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
function is_complex_dtype(dtype::Type)
    """Check if data type is complex"""
    return dtype <: Complex
end

function is_real_dtype(dtype::Type)
    """Check if data type is real"""
    return dtype <: Real && !(dtype <: Complex)
end

function promote_dtype(dtypes...)
    """Promote data types to common type"""
    return promote_type(dtypes...)
end

# Array utilities
function apply_along_axis(func::Function, axis::Int, arr::AbstractArray; args=(), kwargs=())
    """
    Apply function along specified axis following NumPy patterns
    
    Parameters:
    - func: Function to apply to 1-D slices along the axis
    - axis: Axis along which arr is sliced
    - arr: Input array
    - args: Additional positional arguments to func
    - kwargs: Additional keyword arguments to func
    
    Returns:
    - Output array with shape modified according to func output
    """
    
    # Validate axis
    if axis < 1 || axis > ndims(arr)
        throw(ArgumentError("axis $axis is out of bounds for array of dimension $(ndims(arr))"))
    end
    
    # Get array shape and compute output shape template
    arr_shape = size(arr)
    
    # Extract a sample slice to determine output shape and type
    sample_indices = [i == axis ? 1 : Colon() for i in 1:ndims(arr)]
    sample_slice = arr[sample_indices...]
    
    # Get 1-D slice along the axis
    if ndims(sample_slice) == 1
        sample_1d = sample_slice
    else
        # Extract 1-D vector from the slice
        axis_size = arr_shape[axis]
        sample_1d = vec(selectdim(arr, axis, 1))
    end
    
    # Apply function to sample to determine output properties
    sample_output = func(sample_1d, args...; kwargs...)
    
    # Determine output shape
    if isa(sample_output, AbstractArray)
        output_shape_axis = size(sample_output)
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
    axis_size = arr_shape[axis]
    
    # Create iteration indices for all dimensions except the specified axis
    iter_shape = tuple(arr_shape[1:axis-1]..., arr_shape[axis+1:end]...)
    
    if isempty(iter_shape)
        # Special case: 1D array
        slice_1d = vec(arr)
        result = func(slice_1d, args...; kwargs...)
        if isa(result, AbstractArray)
            output[:] = result
        else
            output[] = result
        end
    else
        # General case: iterate over all combinations of indices except the axis dimension
        for cart_idx in CartesianIndices(iter_shape)
            # Convert CartesianIndex to linear indices for dimensions before and after axis
            before_indices = cart_idx.I[1:axis-1]
            after_indices = cart_idx.I[axis:end]
            
            # Extract 1-D slice along the axis
            slice_indices = [before_indices..., Colon(), after_indices...]
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
                output_indices = [before_indices..., Colon(), after_indices...]
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
function apply_along_axis(func::Function, axis::Int, arr::AbstractArray, args...)
    """Convenience wrapper for apply_along_axis with positional args"""
    return apply_along_axis(func, axis, arr; args=args, kwargs=())
end

function axis_sum(arr::AbstractArray, axis::Int)
    """Sum along specified axis (NumPy-style)"""
    return apply_along_axis(sum, axis, arr)
end

function axis_mean(arr::AbstractArray, axis::Int)
    """Mean along specified axis (NumPy-style)"""
    return apply_along_axis(x -> sum(x) / length(x), axis, arr)
end

function axis_maximum(arr::AbstractArray, axis::Int)
    """Maximum along specified axis (NumPy-style)"""
    return apply_along_axis(maximum, axis, arr)
end

function axis_minimum(arr::AbstractArray, axis::Int)
    """Minimum along specified axis (NumPy-style)"""
    return apply_along_axis(minimum, axis, arr)
end

function axis_sort(arr::AbstractArray, axis::Int)
    """Sort along specified axis (NumPy-style)"""
    return apply_along_axis(sort, axis, arr)
end

# Fast version using Julia's mapslices when function output shape is predictable
function apply_along_axis_fast(func::Function, axis::Int, arr::AbstractArray)
    """
    Fast version of apply_along_axis using Julia's mapslices
    
    This version is more efficient for functions that:
    1. Return arrays of the same size as input
    2. Return scalars 
    3. Have predictable output shapes
    
    For general functions with unpredictable output shapes, use apply_along_axis
    """
    
    # Validate axis
    if axis < 1 || axis > ndims(arr)
        throw(ArgumentError("axis $axis is out of bounds for array of dimension $(ndims(arr))"))
    end
    
    return mapslices(func, arr, dims=axis)
end

function broadcast_to_shape(arr::AbstractArray, shape::Tuple)
    """Broadcast array to target shape"""
    # Create array with singleton dimensions where needed
    current_shape = size(arr)
    ndims_diff = length(shape) - ndims(arr)
    
    if ndims_diff > 0
        # Add singleton dimensions at the beginning
        new_shape = tuple(ones(Int, ndims_diff)..., current_shape...)
        arr = reshape(arr, new_shape)
    end
    
    return arr .* ones(eltype(arr), shape)
end

# String utilities - Note: parse_expression is implemented in core/problems.jl

function split_equation(equation::String)
    """
    Split equation string into LHS and RHS strings following Tarang patterns

    Examples:
    --------
    >>> split_equation("∂ₜ(u) = -∂x(u)")
    ("∂ₜ(u)", "-∂x(u)")
    """
    
    # Find top-level equals signs by tracking parenthetical level
    # This avoids capturing equals signs in function calls or keyword arguments
    parentheses = 0
    top_level_equals = Int[]
    
    for (i, char) in enumerate(equation)
        if char == '('
            parentheses += 1
        elseif char == ')'
            parentheses -= 1
        elseif char == '=' && parentheses == 0
            push!(top_level_equals, i)
        end
    end
    
    # Validate equation format
    if length(top_level_equals) == 0
        throw(ArgumentError("Equation contains no top-level equals signs: $equation"))
    elseif length(top_level_equals) > 1
        throw(ArgumentError("Equation contains multiple top-level equals signs: $equation"))
    end
    
    # Split at the equals sign
    eq_pos = top_level_equals[1]
    lhs = strip(equation[1:eq_pos-1])
    rhs = strip(equation[eq_pos+1:end])
    
    return (lhs, rhs)
end

function lambdify_functions(call::String, result::String)
    """
    Convert math-style function definitions into lambda expressions
    Following Tarang parsing patterns
    
    Examples:
    --------
    >>> lambdify_functions("f(x, y)", "x*y")
    ("f", "lambda x,y: x*y")  # In Julia this would be ("f", (x,y) -> x*y)
    >>> lambdify_functions("f", "a*b") 
    ("f", "a*b")
    """
    
    # Check if signature matches a function call
    func_pattern = r"(.+)\((.*)\)"
    m = match(func_pattern, call)
    
    if m !== nothing
        head = strip(m.captures[1])
        argstring = strip(m.captures[2])
        
        if !isempty(argstring)
            # Build lambda expression (Julia anonymous function syntax)
            args = [strip(arg) for arg in split(argstring, ',')]
            args_joined = join(args, ", ")
            lambda_expr = "($args_joined) -> $result"
            return (head, lambda_expr)
        else
            # No arguments - return as-is
            return (head, result)
        end
    else
        # Not a function call
        return (call, result)
    end
end

function format_time(seconds::Float64)
    """Format time duration in human readable format"""
    if seconds < 60
        return "$(round(seconds, digits=2))s"
    elseif seconds < 3600
        minutes = div(seconds, 60)
        secs = seconds % 60
        return "$(Int(minutes))m $(round(secs, digits=1))s"
    else
        hours = div(seconds, 3600)
        remaining = seconds % 3600
        minutes = div(remaining, 60)
        secs = remaining % 60
        return "$(Int(hours))h $(Int(minutes))m $(round(secs, digits=1))s"
    end
end

# Numerical utilities
function safe_divide(a, b; default=0.0)
    """Safe division with default for division by zero"""
    return b == 0 ? default : a / b
end

function clamp_to_range(x, min_val, max_val)
    """Clamp value to range"""
    return clamp(x, min_val, max_val)
end

# Dictionary utilities
function deep_merge!(dict1::Dict, dict2::Dict)
    """Deep merge dictionary dict2 into dict1"""
    for (key, value) in dict2
        if haskey(dict1, key) && isa(dict1[key], Dict) && isa(value, Dict)
            deep_merge!(dict1[key], value)
        else
            dict1[key] = value
        end
    end
    return dict1
end

function flatten_dict(d::Dict, parent_key::String="", sep::String=".")
    """Flatten nested dictionary"""
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
        start_time = time()
        new(total, 0, start_time, start_time, update_interval)
    end
end

function update!(tracker::ProgressTracker, current::Int=tracker.current + 1)
    """Update progress tracker"""
    tracker.current = current
    current_time = time()
    
    if current_time - tracker.last_update_time >= tracker.update_interval || current >= tracker.total
        elapsed = current_time - tracker.start_time
        rate = current / elapsed
        remaining = tracker.total - current
        eta = remaining / rate
        
        percent = (current / tracker.total) * 100
        
        @info "Progress: $current/$(tracker.total) ($(round(percent, digits=1))%), " *
              "Rate: $(round(rate, digits=2))/s, ETA: $(format_time(eta))"
        
        tracker.last_update_time = current_time
    end
end

# Validation utilities
function validate_positive(value, name::String)
    """Validate that value is positive"""
    if value <= 0
        throw(ArgumentError("$name must be positive, got $value"))
    end
    return value
end

function validate_nonnegative(value, name::String)
    """Validate that value is non-negative"""
    if value < 0
        throw(ArgumentError("$name must be non-negative, got $value"))
    end
    return value
end

function validate_in_range(value, min_val, max_val, name::String)
    """Validate that value is in specified range"""
    if value < min_val || value > max_val
        throw(ArgumentError("$name must be in range [$min_val, $max_val], got $value"))
    end
    return value
end

# Memory utilities
function get_memory_usage()
    """Get current memory usage"""
    # Julia-specific memory info
    stats = Base.gc_num()
    return Dict(
        "allocated" => stats.allocd,
        "freed" => stats.freed,
        "total_time" => stats.total_time / 1e9  # Convert to seconds
    )
end

# Logging utilities
function setup_logger(level::String="INFO", filename::Union{String, Nothing}=nothing)
    """Setup logging configuration"""
    # Julia uses the Logging standard library

    log_level = if level == "DEBUG"
        Logging.Debug
    elseif level == "INFO"
        Logging.Info
    elseif level == "WARN"
        Logging.Warn
    elseif level == "ERROR"
        Logging.Error
    else
        Logging.Info
    end
    
    if filename !== nothing
        # Log to file
        io = open(filename, "a")
        logger = SimpleLogger(io, log_level)
        global_logger(logger)
    else
        # Log to console
        logger = ConsoleLogger(stderr, log_level)
        global_logger(logger)
    end
end
