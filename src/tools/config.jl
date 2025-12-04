"""
Configuration management for Tarang.jl

Provides a comprehensive configuration system with:
- Hierarchical configuration with sections
- TOML file loading/saving
- Environment variable overrides
- Type-safe accessors
- Validation and defaults
- Thread-safe access
- Runtime configuration updates

Based on dedalus/tools/config.py patterns.
"""

import TOML
using Base.Threads: ReentrantLock, @lock

# ============================================================================
# Configuration Constants
# ============================================================================

const CONFIG_LOCK = ReentrantLock()

# Valid FFTW planning rigor levels
const VALID_FFTW_RIGOR = ["FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE"]

# Valid log levels
const VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "NONE"]

# Valid transpose libraries
const VALID_TRANSPOSE_LIBS = ["PENCIL", "ALLTOALL", "MANUAL"]

# ============================================================================
# Configuration Section
# ============================================================================

struct ConfigSection
    data::Dict{String, Any}
    name::String
end

function Base.show(io::IO, section::ConfigSection)
    print(io, "ConfigSection($(section.name), $(length(section.data)) keys)")
end

Base.get(section::ConfigSection, key::String) = section.data[key]
Base.get(section::ConfigSection, key::String, default) = get(section.data, key, default)
Base.haskey(section::ConfigSection, key::String) = haskey(section.data, key)
Base.keys(section::ConfigSection) = keys(section.data)
Base.values(section::ConfigSection) = values(section.data)
Base.iterate(section::ConfigSection) = iterate(section.data)
Base.iterate(section::ConfigSection, state) = iterate(section.data, state)
Base.length(section::ConfigSection) = length(section.data)

function getboolean(section::ConfigSection, key::String; default::Union{Bool, Nothing}=nothing)
    """Get boolean value from config section."""
    if !haskey(section.data, key)
        if default !== nothing
            return default
        end
        throw(KeyError("Key '$key' not found in section '$(section.name)'"))
    end

    value = section.data[key]
    if isa(value, Bool)
        return value
    elseif isa(value, String)
        lv = lowercase(strip(value))
        if lv in ["true", "yes", "1", "on", "enabled"]
            return true
        elseif lv in ["false", "no", "0", "off", "disabled"]
            return false
        else
            throw(ArgumentError("Cannot parse '$value' as boolean in $(section.name).$key"))
        end
    elseif isa(value, Number)
        return value != 0
    else
        throw(ArgumentError("Cannot convert $(typeof(value)) to boolean in $(section.name).$key"))
    end
end

function getint(section::ConfigSection, key::String; default::Union{Int, Nothing}=nothing)
    """Get integer value from config section."""
    if !haskey(section.data, key)
        if default !== nothing
            return default
        end
        throw(KeyError("Key '$key' not found in section '$(section.name)'"))
    end

    value = section.data[key]
    if isa(value, Integer)
        return Int(value)
    elseif isa(value, AbstractFloat)
        if isinteger(value)
            return Int(value)
        else
            throw(ArgumentError("Float value $value is not an integer in $(section.name).$key"))
        end
    elseif isa(value, String)
        try
            return parse(Int, strip(value))
        catch
            throw(ArgumentError("Cannot parse '$value' as integer in $(section.name).$key"))
        end
    else
        throw(ArgumentError("Cannot convert $(typeof(value)) to integer in $(section.name).$key"))
    end
end

function getfloat(section::ConfigSection, key::String; default::Union{Float64, Nothing}=nothing)
    """Get float value from config section."""
    if !haskey(section.data, key)
        if default !== nothing
            return default
        end
        throw(KeyError("Key '$key' not found in section '$(section.name)'"))
    end

    value = section.data[key]
    if isa(value, Number)
        return Float64(value)
    elseif isa(value, String)
        try
            return parse(Float64, strip(value))
        catch
            throw(ArgumentError("Cannot parse '$value' as float in $(section.name).$key"))
        end
    else
        throw(ArgumentError("Cannot convert $(typeof(value)) to float in $(section.name).$key"))
    end
end

function getstring(section::ConfigSection, key::String; default::Union{String, Nothing}=nothing)
    """Get string value from config section."""
    if !haskey(section.data, key)
        if default !== nothing
            return default
        end
        throw(KeyError("Key '$key' not found in section '$(section.name)'"))
    end

    value = section.data[key]
    if value === nothing
        return default !== nothing ? default : ""
    end
    return string(value)
end

function getlist(section::ConfigSection, key::String; default::Union{Vector, Nothing}=nothing)
    """Get list/array value from config section."""
    if !haskey(section.data, key)
        if default !== nothing
            return default
        end
        throw(KeyError("Key '$key' not found in section '$(section.name)'"))
    end

    value = section.data[key]
    if isa(value, Vector)
        return value
    elseif isa(value, String)
        # Parse comma-separated string
        return [strip(s) for s in split(value, ",")]
    else
        return [value]
    end
end

# ============================================================================
# Main Configuration Structure
# ============================================================================

mutable struct Config
    data::Dict{String, Any}
    _loaded_files::Vector{String}
    _modified::Bool

    function Config()
        default_config = _create_default_config()
        new(default_config, String[], false)
    end
end

function _create_default_config()
    """Create default configuration dictionary."""
    return Dict{String, Any}(
        # Parallelism settings
        "parallelism" => Dict{String, Any}(
            "TRANSPOSE_LIBRARY" => "PENCIL",
            "GROUP_TRANSPOSES" => true,
            "SYNC_TRANSPOSES" => true,
            "MPI_ENABLED" => true,
            "MESH" => nothing,  # Auto-detect
            "PROCESS_MESH" => nothing,  # (Px, Py) for 2D decomposition
            "ALLTOALL_SPLITGATHER" => true
        ),

        # Threading settings
        "threading" => Dict{String, Any}(
            "OMP_NUM_THREADS" => nothing,  # Use system default
            "BLAS_THREADS" => nothing,
            "FFTW_THREADS" => nothing,
            "JULIA_THREADS" => Threads.nthreads()
        ),

        # Transform settings
        "transforms" => Dict{String, Any}(
            "GROUP_TRANSFORMS" => true,
            "DEALIAS_BEFORE_CONVERTING" => true,
            "DEFAULT_DEALIASING" => 1.5,  # 3/2 rule
            "CACHE_TRANSFORMS" => true,
            "TRANSFORM_BATCH_SIZE" => nothing  # Auto
        ),

        # FFTW settings
        "transforms-fftw" => Dict{String, Any}(
            "PLANNING_RIGOR" => "FFTW_MEASURE",
            "WISDOM_FILE" => nothing,  # Path to wisdom file
            "LOAD_WISDOM" => true,
            "SAVE_WISDOM" => true,
            "TIMELIMIT" => -1.0  # No limit
        ),

        # Memory settings
        "memory" => Dict{String, Any}(
            "PREALLOCATION" => true,
            "FIELD_POOL_SIZE" => 10,
            "MATRIX_CACHE_SIZE" => 100,
            "GC_AFTER_SOLVE" => false,
            "SPARSE_THRESHOLD" => 0.1  # Use sparse if density < 10%
        ),

        # Solver settings
        "solvers" => Dict{String, Any}(
            "DEFAULT_TOLERANCE" => 1e-10,
            "MAX_ITERATIONS" => 1000,
            "MATRIX_COUPLING" => true,
            "STORE_LU" => true,
            "LU_REFACTOR_PERIOD" => 0,  # Never refactor
            "CONDITION_WARNING_THRESHOLD" => 1e12,
            "USE_ITERATIVE" => false,
            "ITERATIVE_SOLVER" => "GMRES"
        ),

        # Timestepping settings
        "timestepping" => Dict{String, Any}(
            "DEFAULT_SCHEME" => "SBDF2",
            "SAFETY_FACTOR" => 0.8,
            "MAX_DT_FACTOR" => 2.0,
            "MIN_DT_FACTOR" => 0.5,
            "CFL_CADENCE" => 1,
            "CFL_THRESHOLD" => 0.5
        ),

        # Profiling settings
        "profiling" => Dict{String, Any}(
            "PROFILE_DEFAULT" => false,
            "PARALLEL_PROFILE_DEFAULT" => false,
            "PROFILE_DIRECTORY" => "profiles",
            "PROFILE_FORMAT" => "json",  # json, csv, or flamegraph
            "TRACE_ALLOCATIONS" => false,
            "TIMING_PRECISION" => "ns"  # ns, us, ms, s
        ),

        # Logging settings
        "logging" => Dict{String, Any}(
            "LEVEL" => "INFO",
            "FILE" => nothing,
            "FORMAT" => "[%(levelname)s] %(name)s: %(message)s",
            "DATE_FORMAT" => "yyyy-mm-dd HH:MM:SS",
            "COLORIZE" => true,
            "SHOW_RANK" => true  # Show MPI rank in log messages
        ),

        # I/O settings
        "io" => Dict{String, Any}(
            "DEFAULT_FILE_HANDLER" => "HDF5",
            "COMPRESSION" => "gzip",
            "COMPRESSION_LEVEL" => 4,
            "CHUNK_SIZE" => nothing,  # Auto
            "PARALLEL_IO" => true,
            "FLUSH_CADENCE" => 100  # Timesteps between flushes
        ),

        # Analysis settings
        "analysis" => Dict{String, Any}(
            "BUFFER_SIZE" => 10,
            "PARALLEL_ANALYSIS" => true,
            "CHECKPOINT_CADENCE" => 1000,
            "CHECKPOINT_WALL_TIME" => 3600  # seconds
        ),

        # Debug settings
        "debug" => Dict{String, Any}(
            "ENABLED" => false,
            "CHECK_FINITE" => false,
            "VERBOSE_MATRICES" => false,
            "DUMP_OPERATORS" => false,
            "TRACE_ALLOCATIONS" => false
        )
    )
end

function Base.show(io::IO, config::Config)
    n_sections = length(config.data)
    n_keys = sum(length(v) for v in values(config.data))
    modified_str = config._modified ? " (modified)" : ""
    print(io, "Config($n_sections sections, $n_keys total keys$modified_str)")
end

# ============================================================================
# Configuration Access Methods
# ============================================================================

function Base.getindex(config::Config, section::String)
    @lock CONFIG_LOCK begin
        if !haskey(config.data, section)
            throw(KeyError("Configuration section '$section' not found. Available: $(join(keys(config.data), ", "))"))
        end
        return ConfigSection(config.data[section], section)
    end
end

function Base.haskey(config::Config, section::String)
    @lock CONFIG_LOCK begin
        return haskey(config.data, section)
    end
end

function Base.keys(config::Config)
    @lock CONFIG_LOCK begin
        return keys(config.data)
    end
end

function Base.get(config::Config, section::String, default=nothing)
    @lock CONFIG_LOCK begin
        if haskey(config.data, section)
            return ConfigSection(config.data[section], section)
        end
        return default
    end
end

function get_value(config::Config, section::String, key::String; default=nothing)
    """Get a specific value from config with optional default."""
    @lock CONFIG_LOCK begin
        if haskey(config.data, section) && haskey(config.data[section], key)
            return config.data[section][key]
        end
        return default
    end
end

function set_value!(config::Config, section::String, key::String, value)
    """Set a specific value in config."""
    @lock CONFIG_LOCK begin
        if !haskey(config.data, section)
            config.data[section] = Dict{String, Any}()
        end
        config.data[section][key] = value
        config._modified = true
    end
end

function add_section!(config::Config, section::String, data::Dict{String, Any}=Dict{String, Any}())
    """Add a new configuration section."""
    @lock CONFIG_LOCK begin
        if haskey(config.data, section)
            @warn "Section '$section' already exists, merging..."
            merge_config!(config.data[section], data)
        else
            config.data[section] = data
        end
        config._modified = true
    end
end

# ============================================================================
# Configuration Loading and Saving
# ============================================================================

function load_config!(config::Config, filename::String; merge::Bool=true)
    """
    Load configuration from TOML file.

    Args:
        config: Config instance to load into
        filename: Path to TOML file
        merge: If true, merge with existing config; if false, replace
    """
    if !isfile(filename)
        @warn "Configuration file $filename not found"
        return false
    end

    @lock CONFIG_LOCK begin
        try
            file_config = TOML.parsefile(filename)

            # Validate loaded config
            validate_config(file_config)

            if merge
                merge_config!(config.data, file_config)
            else
                config.data = file_config
            end

            push!(config._loaded_files, abspath(filename))
            config._modified = false

            @info "Loaded configuration from $filename"
            return true
        catch e
            @error "Failed to load configuration from $filename" exception=e
            return false
        end
    end
end

function load_config!(filename::String; merge::Bool=true)
    """Load configuration into global config instance."""
    return load_config!(config, filename; merge=merge)
end

function save_config(config::Config, filename::String; sections::Union{Nothing, Vector{String}}=nothing)
    """
    Save configuration to TOML file.

    Args:
        config: Config instance to save
        filename: Path to TOML file
        sections: Optional list of sections to save (default: all)
    """
    @lock CONFIG_LOCK begin
        try
            data_to_save = if sections !== nothing
                Dict(s => config.data[s] for s in sections if haskey(config.data, s))
            else
                config.data
            end

            # Ensure directory exists
            dir = dirname(filename)
            if !isempty(dir) && !isdir(dir)
                mkpath(dir)
            end

            open(filename, "w") do io
                TOML.print(io, data_to_save)
            end

            config._modified = false
            @info "Saved configuration to $filename"
            return true
        catch e
            @error "Failed to save configuration to $filename" exception=e
            return false
        end
    end
end

function save_config(filename::String; kwargs...)
    """Save global config instance to file."""
    return save_config(config, filename; kwargs...)
end

function merge_config!(base::Dict, overlay::Dict)
    """Recursively merge configuration dictionaries."""
    for (key, value) in overlay
        if haskey(base, key) && isa(base[key], Dict) && isa(value, Dict)
            merge_config!(base[key], value)
        else
            base[key] = value
        end
    end
end

function reset_config!(config::Config)
    """Reset configuration to defaults."""
    @lock CONFIG_LOCK begin
        config.data = _create_default_config()
        empty!(config._loaded_files)
        config._modified = false
        @info "Configuration reset to defaults"
    end
end

function reset_config!()
    """Reset global config to defaults."""
    reset_config!(config)
end

# ============================================================================
# Configuration Validation
# ============================================================================

function validate_config(data::Dict)
    """Validate configuration values."""

    # Validate FFTW rigor
    if haskey(data, "transforms-fftw") && haskey(data["transforms-fftw"], "PLANNING_RIGOR")
        rigor = data["transforms-fftw"]["PLANNING_RIGOR"]
        if !(rigor in VALID_FFTW_RIGOR)
            throw(ArgumentError("Invalid FFTW planning rigor: $rigor. Valid values: $(join(VALID_FFTW_RIGOR, ", "))"))
        end
    end

    # Validate log level
    if haskey(data, "logging") && haskey(data["logging"], "LEVEL")
        level = uppercase(data["logging"]["LEVEL"])
        if !(level in VALID_LOG_LEVELS)
            throw(ArgumentError("Invalid log level: $level. Valid values: $(join(VALID_LOG_LEVELS, ", "))"))
        end
    end

    # Validate transpose library
    if haskey(data, "parallelism") && haskey(data["parallelism"], "TRANSPOSE_LIBRARY")
        lib = uppercase(data["parallelism"]["TRANSPOSE_LIBRARY"])
        if !(lib in VALID_TRANSPOSE_LIBS)
            throw(ArgumentError("Invalid transpose library: $lib. Valid values: $(join(VALID_TRANSPOSE_LIBS, ", "))"))
        end
    end

    # Validate dealiasing factor
    if haskey(data, "transforms") && haskey(data["transforms"], "DEFAULT_DEALIASING")
        dealiasing = data["transforms"]["DEFAULT_DEALIASING"]
        if dealiasing < 1.0
            throw(ArgumentError("Dealiasing factor must be >= 1.0, got $dealiasing"))
        end
    end

    # Validate tolerance
    if haskey(data, "solvers") && haskey(data["solvers"], "DEFAULT_TOLERANCE")
        tol = data["solvers"]["DEFAULT_TOLERANCE"]
        if tol <= 0
            throw(ArgumentError("Solver tolerance must be positive, got $tol"))
        end
    end

    return true
end

# ============================================================================
# Environment Variable Overrides
# ============================================================================

function apply_env_overrides!(config::Config)
    """Apply environment variable overrides to configuration."""

    @lock CONFIG_LOCK begin
        # Threading overrides
        if haskey(ENV, "OMP_NUM_THREADS")
            config.data["threading"]["OMP_NUM_THREADS"] = parse(Int, ENV["OMP_NUM_THREADS"])
        end

        if haskey(ENV, "JULIA_NUM_THREADS")
            config.data["threading"]["JULIA_THREADS"] = parse(Int, ENV["JULIA_NUM_THREADS"])
        end

        # Logging level override
        if haskey(ENV, "TARANG_LOG_LEVEL")
            level = uppercase(ENV["TARANG_LOG_LEVEL"])
            if level in VALID_LOG_LEVELS
                config.data["logging"]["LEVEL"] = level
            else
                @warn "Invalid TARANG_LOG_LEVEL: $level, ignoring"
            end
        end

        # Profile directory override
        if haskey(ENV, "TARANG_PROFILE_DIR")
            config.data["profiling"]["PROFILE_DIRECTORY"] = ENV["TARANG_PROFILE_DIR"]
        end

        # Debug mode
        if haskey(ENV, "TARANG_DEBUG") && lowercase(ENV["TARANG_DEBUG"]) in ["1", "true", "yes"]
            config.data["debug"]["ENABLED"] = true
            config.data["logging"]["LEVEL"] = "DEBUG"
        end

        # MPI settings
        if haskey(ENV, "TARANG_MPI_ENABLED")
            config.data["parallelism"]["MPI_ENABLED"] = lowercase(ENV["TARANG_MPI_ENABLED"]) in ["1", "true", "yes"]
        end

        # FFTW wisdom file
        if haskey(ENV, "TARANG_FFTW_WISDOM")
            config.data["transforms-fftw"]["WISDOM_FILE"] = ENV["TARANG_FFTW_WISDOM"]
        end

        # FFTW planning rigor
        if haskey(ENV, "TARANG_FFTW_RIGOR")
            rigor = uppercase(ENV["TARANG_FFTW_RIGOR"])
            if rigor in VALID_FFTW_RIGOR
                config.data["transforms-fftw"]["PLANNING_RIGOR"] = rigor
            end
        end
    end
end

function apply_env_overrides!()
    """Apply environment overrides to global config."""
    apply_env_overrides!(config)
end

# ============================================================================
# Convenience Functions
# ============================================================================

function is_debug_enabled()
    """Check if debug mode is enabled."""
    return get_value(config, "debug", "ENABLED"; default=false)
end

function is_mpi_enabled()
    """Check if MPI is enabled."""
    return get_value(config, "parallelism", "MPI_ENABLED"; default=true)
end

function get_log_level()
    """Get current log level."""
    return get_value(config, "logging", "LEVEL"; default="INFO")
end

function get_fftw_rigor()
    """Get FFTW planning rigor setting."""
    return get_value(config, "transforms-fftw", "PLANNING_RIGOR"; default="FFTW_MEASURE")
end

function get_default_tolerance()
    """Get default solver tolerance."""
    return get_value(config, "solvers", "DEFAULT_TOLERANCE"; default=1e-10)
end

function get_dealiasing_factor()
    """Get default dealiasing factor."""
    return get_value(config, "transforms", "DEFAULT_DEALIASING"; default=1.5)
end

function get_thread_count()
    """Get configured thread count."""
    return get_value(config, "threading", "JULIA_THREADS"; default=Threads.nthreads())
end

function print_config(io::IO=stdout)
    """Print current configuration in a readable format."""
    @lock CONFIG_LOCK begin
        println(io, "Tarang Configuration")
        println(io, "=" ^ 50)

        for section_name in sort(collect(keys(config.data)))
            println(io, "\n[$section_name]")
            section = config.data[section_name]

            for key in sort(collect(keys(section)))
                value = section[key]
                value_str = value === nothing ? "<not set>" : string(value)
                println(io, "  $key = $value_str")
            end
        end

        if !isempty(config._loaded_files)
            println(io, "\nLoaded from:")
            for f in config._loaded_files
                println(io, "  - $f")
            end
        end
    end
end

# ============================================================================
# Global Config Instance
# ============================================================================

const config = Config()

# ============================================================================
# Initialization
# ============================================================================

function __init_config__()
    """Initialize configuration system."""

    # Apply environment overrides first
    apply_env_overrides!()

    # Look for config file in standard locations
    config_files = [
        get(ENV, "TARANG_CONFIG", ""),  # Environment-specified config
        joinpath(homedir(), ".tarang", "tarang.toml"),
        joinpath(homedir(), ".config", "tarang", "tarang.toml"),
        "tarang.toml",
        joinpath(dirname(@__FILE__), "..", "..", "tarang.toml")
    ]

    for config_file in config_files
        if !isempty(config_file) && isfile(config_file)
            load_config!(config_file)
            break
        end
    end
end

# Initialize on module load
__init_config__()
