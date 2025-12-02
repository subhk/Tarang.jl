"""
Logging utilities for Tarang

Enhanced logging with MPI support
"""

using Logging
using Dates
using MPI

# Custom log levels
const TRACE = LogLevel(-1000)  # More verbose than Debug
const NOTICE = LogLevel(1500)  # Between Info and Warn

# MPI-aware logger
struct MPILogger <: AbstractLogger
    base_logger::AbstractLogger
    rank::Int
    size::Int
    rank_width::Int
    
    function MPILogger(base_logger::AbstractLogger)
        if MPI.Initialized()
            rank = MPI.Comm_rank(MPI.COMM_WORLD)
            size = MPI.Comm_size(MPI.COMM_WORLD)
        else
            rank = 0
            size = 1
        end
        
        rank_width = length(string(size - 1))  # Width needed for largest rank
        new(base_logger, rank, size, rank_width)
    end
end

# Logger interface implementation
function Logging.min_enabled_level(logger::MPILogger)
    return Logging.min_enabled_level(logger.base_logger)
end

function Logging.shouldlog(logger::MPILogger, level, _module, group, id)
    return Logging.shouldlog(logger.base_logger, level, _module, group, id)
end

function Logging.catch_exceptions(logger::MPILogger)
    return Logging.catch_exceptions(logger.base_logger)
end

function Logging.handle_message(logger::MPILogger, level, message, _module, group, id, 
                               file, line; kwargs...)
    
    # Format message with rank information
    rank_str = lpad(logger.rank, logger.rank_width, '0')
    
    if logger.size > 1
        prefixed_message = "[Rank $rank_str/$(logger.size-1)] $message"
    else
        prefixed_message = message
    end
    
    # Delegate to base logger
    Logging.handle_message(logger.base_logger, level, prefixed_message, _module, group, id, 
                          file, line; kwargs...)
end

# Custom formatter for file logging
struct TarangFormatter
    show_timestamp::Bool
    show_level::Bool
    show_module::Bool
    timestamp_format::String
    
    function TarangFormatter(; show_timestamp::Bool=true, show_level::Bool=true, 
                            show_module::Bool=true, timestamp_format::String="yyyy-mm-dd HH:MM:SS")
        new(show_timestamp, show_level, show_module, timestamp_format)
    end
end

function format_message(formatter::TarangFormatter, level, message, _module, group, id, file, line)
    parts = String[]
    
    if formatter.show_timestamp
        timestamp = Dates.format(now(), formatter.timestamp_format)
        push!(parts, "[$timestamp]")
    end
    
    if formatter.show_level
        level_str = string(level)
        push!(parts, "[$level_str]")
    end
    
    if formatter.show_module && _module !== nothing
        module_str = string(_module)
        push!(parts, "[$module_str]")
    end
    
    push!(parts, message)
    
    return join(parts, " ")
end

# File logger with custom formatting
struct FileLogger <: AbstractLogger
    io::IO
    min_level::LogLevel
    formatter::TarangFormatter
    own_io::Bool  # Whether we own the IO stream
    
    function FileLogger(filename::String; min_level::LogLevel=Logging.Info, 
                       formatter::TarangFormatter=TarangFormatter())
        io = open(filename, "a")
        new(io, min_level, formatter, true)
    end
    
    function FileLogger(io::IO; min_level::LogLevel=Logging.Info,
                       formatter::TarangFormatter=TarangFormatter())
        new(io, min_level, formatter, false)
    end
end

function Logging.min_enabled_level(logger::FileLogger)
    return logger.min_level
end

function Logging.shouldlog(logger::FileLogger, level, _module, group, id)
    return level >= logger.min_level
end

function Logging.catch_exceptions(logger::FileLogger)
    return true
end

function Logging.handle_message(logger::FileLogger, level, message, _module, group, id,
                               file, line; kwargs...)
    
    formatted_msg = format_message(logger.formatter, level, message, _module, group, id, file, line)
    
    # Add any keyword arguments
    if !isempty(kwargs)
        kv_pairs = ["$k=$v" for (k, v) in kwargs]
        formatted_msg *= " | " * join(kv_pairs, ", ")
    end
    
    println(logger.io, formatted_msg)
    flush(logger.io)
end

function Base.close(logger::FileLogger)
    if logger.own_io
        close(logger.io)
    end
end

# Tee logger - logs to multiple destinations
struct TeeLogger <: AbstractLogger
    loggers::Vector{AbstractLogger}
    
    function TeeLogger(loggers::AbstractLogger...)
        new(collect(loggers))
    end
end

function Logging.min_enabled_level(logger::TeeLogger)
    return minimum(Logging.min_enabled_level(l) for l in logger.loggers)
end

function Logging.shouldlog(logger::TeeLogger, level, _module, group, id)
    return any(Logging.shouldlog(l, level, _module, group, id) for l in logger.loggers)
end

function Logging.catch_exceptions(logger::TeeLogger)
    return all(Logging.catch_exceptions(l) for l in logger.loggers)
end

function Logging.handle_message(logger::TeeLogger, level, message, _module, group, id,
                               file, line; kwargs...)
    
    for l in logger.loggers
        if Logging.shouldlog(l, level, _module, group, id)
            Logging.handle_message(l, level, message, _module, group, id, file, line; kwargs...)
        end
    end
end

# Setup functions
function setup_tarang_logging(; 
    level::Union{String, LogLevel}=Logging.Info,
    console::Bool=true,
    filename::Union{String, Nothing}=nothing,
    mpi_aware::Bool=true,
    format_timestamp::Bool=true)
    """Setup Tarang logging system"""
    
    # Parse level
    if isa(level, String)
        level_map = Dict(
            "TRACE" => TRACE,
            "DEBUG" => Logging.Debug,
            "INFO" => Logging.Info,
            "NOTICE" => NOTICE, 
            "WARN" => Logging.Warn,
            "ERROR" => Logging.Error
        )
        log_level = get(level_map, uppercase(level), Logging.Info)
    else
        log_level = level
    end
    
    loggers = AbstractLogger[]
    
    # Console logger
    if console
        console_logger = ConsoleLogger(stderr, log_level)
        push!(loggers, console_logger)
    end
    
    # File logger
    if filename !== nothing
        formatter = TarangFormatter(show_timestamp=format_timestamp)
        file_logger = FileLogger(filename, min_level=log_level, formatter=formatter)
        push!(loggers, file_logger)
    end
    
    # Combine loggers
    if length(loggers) == 1
        base_logger = loggers[1]
    elseif length(loggers) > 1
        base_logger = TeeLogger(loggers...)
    else
        # Default console logger
        base_logger = ConsoleLogger(stderr, log_level)
    end
    
    # Wrap with MPI logger if requested
    if mpi_aware
        final_logger = MPILogger(base_logger)
    else
        final_logger = base_logger
    end
    
    global_logger(final_logger)
    
    return final_logger
end

function get_logger_info()
    """Get information about current logger"""
    current_logger = global_logger()
    
    info = Dict{String, Any}(
        "type" => string(typeof(current_logger)),
        "min_level" => string(Logging.min_enabled_level(current_logger))
    )
    
    if isa(current_logger, MPILogger)
        info["mpi_rank"] = current_logger.rank
        info["mpi_size"] = current_logger.size
        info["base_logger"] = string(typeof(current_logger.base_logger))
    end
    
    return info
end

# Convenience logging macros
macro trace(ex)
    quote
        @logmsg TRACE $(esc(ex))
    end
end

macro notice(ex)
    quote
        @logmsg NOTICE $(esc(ex))
    end
end

# Performance logging
macro log_timing(level, name, expr)
    quote
        start_time = time()
        result = $(esc(expr))
        elapsed = time() - start_time
        @logmsg $(esc(level)) "Timing: $($(esc(name))) took $(round(elapsed, digits=4)) seconds"
        result
    end
end

# Context manager for temporary log level changes
struct LogLevelContext
    original_logger::AbstractLogger
    temp_logger::AbstractLogger
    
    function LogLevelContext(temp_level::LogLevel)
        original = global_logger()
        temp = ConsoleLogger(stderr, temp_level)
        global_logger(temp)
        new(original, temp)
    end
end

function Base.close(ctx::LogLevelContext)
    global_logger(ctx.original_logger)
end

macro with_log_level(level, expr)
    quote
        ctx = LogLevelContext($(esc(level)))
        try
            $(esc(expr))
        finally
            close(ctx)
        end
    end
end

# Log rotation utilities
function rotate_log_file(filename::String, max_size::Int=10*1024*1024, max_files::Int=5)
    """Rotate log file if it exceeds max_size"""
    
    if !isfile(filename)
        return false
    end
    
    if filesize(filename) <= max_size
        return false
    end
    
    # Rotate existing files
    for i in max_files-1:-1:1
        old_name = "$filename.$i"
        new_name = "$filename.$(i+1)"
        
        if isfile(old_name)
            if i == max_files-1
                # Delete oldest file
                rm(old_name)
            else
                mv(old_name, new_name)
            end
        end
    end
    
    # Move current file to .1
    mv(filename, "$filename.1")
    
    @info "Rotated log file $filename"
    return true
end

# Initialize logging when module is loaded
function __init_logging__()
    # Check for environment variables
    log_level = get(ENV, "VARUNA_LOG_LEVEL", "INFO")
    log_file = get(ENV, "VARUNA_LOG_FILE", nothing)
    
    setup_tarang_logging(level=log_level, filename=log_file)
end