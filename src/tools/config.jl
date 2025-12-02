"""
Configuration management

Simplified version of dedalus/tools/config.py
"""

using Pkg.TOML

mutable struct Config
    data::Dict{String, Any}
    
    function Config()
        # Default configuration
        default_config = Dict(
            "parallelism" => Dict(
                "TRANSPOSE_LIBRARY" => "PENCIL",
                "GROUP_TRANSPOSES" => true,
                "SYNC_TRANSPOSES" => true
            ),
            "transforms" => Dict(
                "GROUP_TRANSFORMS" => true,
                "DEALIAS_BEFORE_CONVERTING" => true
            ),
            "transforms-fftw" => Dict(
                "PLANNING_RIGOR" => "FFTW_ESTIMATE"
            ),
            "profiling" => Dict(
                "PROFILE_DEFAULT" => false,
                "PARALLEL_PROFILE_DEFAULT" => false,
                "PROFILE_DIRECTORY" => "profiles"
            ),
            "logging" => Dict(
                "LEVEL" => "INFO",
                "FILE" => nothing,
                "FORMAT" => "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        
        new(default_config)
    end
end

# Global config instance
const config = Config()

# Configuration access methods
Base.getindex(config::Config, section::String) = ConfigSection(config.data[section])
Base.haskey(config::Config, section::String) = haskey(config.data, section)

struct ConfigSection
    data::Dict{String, Any}
end

Base.get(section::ConfigSection, key::String) = section.data[key]
Base.get(section::ConfigSection, key::String, default) = get(section.data, key, default)
Base.haskey(section::ConfigSection, key::String) = haskey(section.data, key)

function getboolean(section::ConfigSection, key::String)
    """Get boolean value from config"""
    value = section.data[key]
    if isa(value, Bool)
        return value
    elseif isa(value, String)
        return lowercase(value) in ["true", "yes", "1", "on"]
    else
        return Bool(value)
    end
end

function getint(section::ConfigSection, key::String)
    """Get integer value from config"""
    return Int(section.data[key])
end

function getfloat(section::ConfigSection, key::String)
    """Get float value from config"""
    return Float64(section.data[key])
end

# Configuration loading and saving
function load_config!(filename::String)
    """Load configuration from TOML file"""
    if isfile(filename)
        file_config = TOML.parsefile(filename)
        merge_config!(config.data, file_config)
        @info "Loaded configuration from $filename"
    else
        @warn "Configuration file $filename not found, using defaults"
    end
end

function save_config(filename::String)
    """Save configuration to TOML file"""
    open(filename, "w") do io
        TOML.print(io, config.data)
    end
    @info "Saved configuration to $filename"
end

function merge_config!(base::Dict, overlay::Dict)
    """Recursively merge configuration dictionaries"""
    for (key, value) in overlay
        if haskey(base, key) && isa(base[key], Dict) && isa(value, Dict)
            merge_config!(base[key], value)
        else
            base[key] = value
        end
    end
end

# Environment variable overrides
function apply_env_overrides!()
    """Apply environment variable overrides to configuration"""
    
    # Threading override
    omp_threads = get(ENV, "OMP_NUM_THREADS", nothing)
    if omp_threads !== nothing
        config.data["parallelism"]["OMP_NUM_THREADS"] = omp_threads
    end
    
    # Logging level override
    log_level = get(ENV, "VARUNA_LOG_LEVEL", nothing)
    if log_level !== nothing
        config.data["logging"]["LEVEL"] = log_level
    end
    
    # Profile directory override
    profile_dir = get(ENV, "VARUNA_PROFILE_DIR", nothing)
    if profile_dir !== nothing
        config.data["profiling"]["PROFILE_DIRECTORY"] = profile_dir
    end
end

# Initialize configuration
function __init_config__()
    """Initialize configuration system"""
    
    # Apply environment overrides
    apply_env_overrides!()
    
    # Look for config file in standard locations
    config_files = [
        joinpath(homedir(), ".tarang", "tarang.toml"),
        "tarang.toml",
        joinpath(dirname(@__FILE__), "..", "..", "tarang.toml")
    ]
    
    for config_file in config_files
        if isfile(config_file)
            load_config!(config_file)
            break
        end
    end
end

# Call initialization
__init_config__()