"""
Logging utilities for Tarang

Enhanced logging with MPI support
"""

using Logging
using Dates
using MPI

export TRACE, NOTICE,
       MPILogger, TarangFormatter, FileLogger, TeeLogger, LevelFilterLogger,
       setup_tarang_logging, get_logger_info,
       rotate_log_file, init_logging!,
       @trace, @notice, @log_timing, @with_log_level,
       LogLevelContext

# Custom log levels
const TRACE = LogLevel(-2000)  # More verbose than Debug (-1000)
const NOTICE = LogLevel(500)   # Between Info (0) and Warn (1000)

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
    # Use format [Rank X/N] where X is current rank and N is total process count
    rank_str = lpad(logger.rank, logger.rank_width, '0')

    if logger.size > 1
        prefixed_message = "[Rank $rank_str/$(logger.size)] $message"
    else
        prefixed_message = message
    end

    # Delegate to base logger
    Logging.handle_message(logger.base_logger, level, prefixed_message, _module, group, id,
                          file, line; kwargs...)
end

"""Close the base logger if it supports closing."""
function Base.close(logger::MPILogger)
    try
        close(logger.base_logger)
    catch
        # Base logger may not support close
    end
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

"""Close all loggers in the TeeLogger that support closing."""
function Base.close(logger::TeeLogger)
    for l in logger.loggers
        try
            close(l)
        catch
            # Not all loggers may support close, ignore errors
        end
    end
end

# Setup functions
"""
    setup_tarang_logging(; level, console, filename, mpi_aware, format_timestamp)

Setup Tarang logging system with configurable options.

# Keyword Arguments
- `level`: Log level (String like "INFO" or LogLevel). Default: `Logging.Info`
- `console`: Whether to log to console. Default: `true`
- `filename`: Optional file path to write logs to. Default: `nothing`
- `mpi_aware`: Whether to prefix messages with MPI rank. Default: `true`
- `format_timestamp`: Whether to include timestamps in file logs. Default: `true`

# Returns
The configured logger instance.
"""
function setup_tarang_logging(;
    level::Union{String, LogLevel}=Logging.Info,
    console::Bool=true,
    filename::Union{String, Nothing}=nothing,
    mpi_aware::Bool=true,
    format_timestamp::Bool=true)
    
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

"""Get information about the current logger configuration."""
function get_logger_info()
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

# Level filter logger - wraps an existing logger with a different minimum level
# This preserves all configuration (file logging, MPI awareness, formatters) while
# only changing the minimum log level
struct LevelFilterLogger <: AbstractLogger
    base_logger::AbstractLogger
    min_level::LogLevel
end

function Logging.min_enabled_level(logger::LevelFilterLogger)
    return logger.min_level
end

function Logging.shouldlog(logger::LevelFilterLogger, level, _module, group, id)
    return level >= logger.min_level && Logging.shouldlog(logger.base_logger, level, _module, group, id)
end

function Logging.catch_exceptions(logger::LevelFilterLogger)
    return Logging.catch_exceptions(logger.base_logger)
end

function Logging.handle_message(logger::LevelFilterLogger, level, message, _module, group, id,
                               file, line; kwargs...)
    Logging.handle_message(logger.base_logger, level, message, _module, group, id,
                          file, line; kwargs...)
end

"""Close the base logger if it supports closing."""
function Base.close(logger::LevelFilterLogger)
    try
        close(logger.base_logger)
    catch
        # Base logger may not support close
    end
end

# Context manager for temporary log level changes
# Uses LevelFilterLogger to preserve existing logger configuration
"""
    LogLevelContext(temp_level::LogLevel)

Context manager for temporary log level changes.
Wraps the existing logger with a level filter, preserving all configuration
(file logging, MPI awareness, formatters) while changing the minimum level.

Use with `@with_log_level` macro for scoped level changes.
"""
struct LogLevelContext
    original_logger::AbstractLogger
    temp_logger::AbstractLogger

    function LogLevelContext(temp_level::LogLevel)
        original = global_logger()
        # Wrap existing logger with level filter instead of replacing it
        # This preserves all configuration (file logging, MPI, formatters)
        temp = LevelFilterLogger(original, temp_level)
        global_logger(temp)
        new(original, temp)
    end
end

"""Restore the original logger when the context is closed."""
function Base.close(ctx::LogLevelContext)
    global_logger(ctx.original_logger)
end

"""
    @with_log_level level expr

Execute `expr` with a temporary log level, then restore the original level.
Preserves all existing logger configuration (file logging, MPI awareness, formatters).

# Example
```julia
# Temporarily enable debug logging for a specific section
@with_log_level Logging.Debug begin
    @debug "This will be logged"
    some_function()
end
# Debug logging is now disabled again
```
"""
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
"""
    rotate_log_file(filename, max_size=10*1024*1024, max_files=5)

Rotate log file if it exceeds max_size.

# Arguments
- `filename`: Path to the log file
- `max_size`: Maximum file size in bytes before rotation. Default: 10 MB
- `max_files`: Maximum number of rotated files to keep. Default: 5

# Returns
`true` if rotation occurred, `false` otherwise.
"""
function rotate_log_file(filename::String, max_size::Int=10*1024*1024, max_files::Int=5)
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

"""
    init_logging!()

Initialize the Tarang logging system based on environment variables.

Environment variables:
- `TARANG_LOG_LEVEL`: Log level (TRACE, DEBUG, INFO, NOTICE, WARN, ERROR). Default: INFO
- `TARANG_LOG_FILE`: Optional file path to write logs to

This function is called automatically from `Tarang.__init__()` but can also
be called manually to reinitialize logging.
"""
function init_logging!()
    # Check for environment variables
    log_level = get(ENV, "TARANG_LOG_LEVEL", "INFO")
    log_file = get(ENV, "TARANG_LOG_FILE", nothing)

    # Treat empty string as nothing (no log file)
    if log_file !== nothing && isempty(strip(log_file))
        log_file = nothing
    end

    setup_tarang_logging(level=log_level, filename=log_file)
end