#!/usr/bin/env julia

"""
NetCDF Merge Script for Tarang.jl

Command-line utility for merging per-processor NetCDF files created by Tarang.jl
Provides easy workflow integration for post-processing.

USAGE:
    julia merge_netcdf.jl [OPTIONS] HANDLER_NAME [HANDLER_NAME2 ...]

EXAMPLES:
    # Basic merge
    julia merge_netcdf.jl snapshots
    
    # Merge with cleanup 
    julia merge_netcdf.jl --cleanup snapshots analysis
    
    # Merge all available sets for a handler
    julia merge_netcdf.jl snapshots

    # Merge specific set with custom output
    julia merge_netcdf.jl --set=2 --output=my_output.nc snapshots
    
    # Different merge modes
    julia merge_netcdf.jl --mode=concat snapshots
    julia merge_netcdf.jl --mode=reconstruct analysis
    
    # Auto-discover and merge all handlers
    julia merge_netcdf.jl --auto --cleanup
    
    # Quiet mode
    julia merge_netcdf.jl --quiet snapshots

OPTIONS:
    --set=N              Set number to merge (default: all discovered sets)
    --output=FILE        Output filename (auto-generated if not specified)  
    --mode=MODE          Merge mode: concat, reconstruct, decomp (default: reconstruct)
    --cleanup            Delete source files after successful merge
    --auto               Auto-discover all mergeable handlers
    --quiet              Minimal output
    --help, -h           Show this help message
    --version, -v        Show version information

MERGE MODES:
    concat      - Concatenate data with processor dimension
    reconstruct - Reconstruct global fields (averaging method) 
    decomp      - Domain decomposition reconstruction (advanced)

FILE STRUCTURE:
Input files should follow Tarang.jl naming:
    snapshots_s1/snapshots_s1_p0.nc    # Processor 0
    snapshots_s1/snapshots_s1_p1.nc    # Processor 1  
    snapshots_s1/snapshots_s1_p2.nc    # Processor 2
    ...

Output:
    snapshots_s1/snapshots_s1.nc  # Merged file for direct set directories
    snapshots/snapshots_s1.nc     # Merged file for nested handler directories
    
With --cleanup, original processor files are deleted after successful merge.
"""

using ArgParse
using Tarang

# Version info from package metadata; avoids depending on private module globals.
const VERSION = string(pkgversion(Tarang))
const SCRIPT_NAME = "merge_file.jl"

function parse_arguments()
    # Keep the CLI surface here and the NetCDF merge mechanics in
    # src/tools/netcdf_merge.jl so scripts stay thin wrappers around library
    # functionality.
    s = ArgParseSettings(description="Merge per-processor NetCDF files from Tarang.jl",
                        version="$SCRIPT_NAME v$VERSION",
                        add_version=true)

    @add_arg_table! s begin
        "handlers"
            nargs = '*'
            help = "Handler names to merge (e.g., snapshots, analysis)"
            
        "--set", "-s"
            arg_type = Int
            default = 0
            help = "Set number to merge (default: all discovered sets)"
            
        "--output", "-o"
            arg_type = String
            default = ""
            help = "Output filename (auto-generated if not specified)"
            
        "--mode", "-m"
            arg_type = String
            default = "reconstruct"
            help = "Merge mode: concat, reconstruct, decomp"
            
        "--cleanup", "-c"
            action = :store_true
            help = "Delete source files after successful merge"
            
        "--auto", "-a"
            action = :store_true
            help = "Auto-discover all mergeable handlers"
            
        "--quiet", "-q"
            action = :store_true
            help = "Minimal output"
    end

    return parse_args(s)
end

function str_to_merge_mode(mode_str::String)
    mode_lower = lowercase(mode_str)
    # Convert user-facing strings into the internal merge-mode constants used by
    # the library implementation.
    if mode_lower == "concat"
        return SIMPLE_CONCAT
    elseif mode_lower == "reconstruct"
        return RECONSTRUCT
    elseif mode_lower == "decomp" || mode_lower == "domain_decomp"
        return DOMAIN_DECOMP
    else
        error("Unknown merge mode: $mode_str. Use: concat, reconstruct, decomp")
    end
end

function print_banner(verbose::Bool)
    if verbose
        println("=" * "="^58)
        println("  Tarang.jl NetCDF File Merger v$VERSION")
        println("     Merge per-processor NetCDF files")
        println("=" * "="^58)
        println()
    end
end

function normalize_handler_arg(handler::String)
    normalized = normpath(strip(handler))
    if isempty(normalized) || normalized == "."
        throw(ArgumentError("Handler name cannot be empty"))
    end
    return last(splitpath(normalized))
end

function _has_processor_files(search_dir::String, set_pattern::String)
    isdir(search_dir) || return false

    for file in readdir(search_dir, join=true)
        name = basename(file)
        if isfile(file) && startswith(name, "$(set_pattern)_p") && match(r"_p\d+\.nc$", name) !== nothing
            return true
        end
    end

    return false
end

function _set_number_from_name(name::String, handler::String; processor_file::Bool=false)
    prefix = "$(handler)_s"
    startswith(name, prefix) || return nothing

    suffix = name[(lastindex(prefix) + 1):end]
    if processor_file
        m = match(r"^(\d+)_p\d+\.nc$", suffix)
        return m === nothing ? nothing : parse(Int, m.captures[1])
    end

    !isempty(suffix) && all(isdigit, suffix) || return nothing
    return parse(Int, suffix)
end

function discover_handler_sets(handler::String, requested_set::Int; fallback::Bool=true)
    handler = normalize_handler_arg(handler)

    if requested_set > 0
        return [(requested_set, handler_workdir(handler, requested_set))]
    elseif requested_set < 0
        throw(ArgumentError("--set must be positive, or omitted to merge all discovered sets"))
    end

    sets = Dict{Int, String}()

    for file in readdir(".", join=true)
        name = basename(file)
        set_number = _set_number_from_name(name, handler; processor_file=true)
        if set_number !== nothing
            get!(sets, set_number, ".")
        end
    end

    for entry in readdir(".", join=true)
        name = basename(entry)
        set_number = _set_number_from_name(name, handler)
        if isdir(entry) && set_number !== nothing
            set_pattern = "$(handler)_s$(set_number)"
            if _has_processor_files(entry, set_pattern)
                get!(sets, set_number, ".")
            end
        end
    end

    if isdir(handler)
        for entry in readdir(handler, join=true)
            name = basename(entry)
            set_number = _set_number_from_name(name, handler)
            if isdir(entry) && set_number !== nothing
                set_pattern = "$(handler)_s$(set_number)"
                if _has_processor_files(entry, set_pattern) && !haskey(sets, set_number)
                    sets[set_number] = handler
                end
            end
        end
    end

    discovered = [(set_number, workdir) for (set_number, workdir) in sets]
    sort!(discovered, by=first)

    if isempty(discovered) && fallback
        return [(1, handler_workdir(handler, 1))]
    end

    return discovered
end

function auto_discover_and_merge(args)
    verbose = !args["quiet"]
    
    verbose && println("Auto-discovering mergeable handlers...")
    
    requested_set = args["set"]
    handler_names = Set{String}(keys(find_mergeable_handlers()))
    for entry in readdir(".", join=true)
        isdir(entry) || continue
        handler = basename(entry)
        if !isempty(discover_handler_sets(handler, requested_set; fallback=false))
            push!(handler_names, handler)
        end
    end
    
    if isempty(handler_names)
        println("No mergeable handlers found in current directory")
        return false
    end

    handlers = Dict{String, Vector{Int}}()
    for handler in handler_names
        handlers[handler] = [set_number for (set_number, _) in discover_handler_sets(handler, requested_set; fallback=false)]
    end
    
    verbose && println("Found $(length(handlers)) mergeable handlers:")
    for (handler, sets) in handlers
        verbose && println("   - $handler: sets $(join(sets, ", "))")
    end
    println()
    
    merge_args = copy(args)
    merge_args["handlers"] = sort(collect(keys(handlers)))
    successful = merge_specific_handlers(merge_args)
    return successful
end

function handler_workdir(handler::String, set_number::Int)
    handler = normalize_handler_arg(handler)

    set_pattern = "$(handler)_s$(set_number)"
    direct_set_dir = set_pattern
    nested_set_dir = joinpath(handler, set_pattern)

    if !isdir(direct_set_dir) && isdir(nested_set_dir)
        return handler
    end

    return "."
end

function merge_output_name(handler::String, set_number::Int, workdir::String, output_name::String)
    handler = normalize_handler_arg(handler)

    if !isempty(output_name)
        return abspath(output_name)
    elseif workdir != "."
        return abspath(joinpath(workdir, "$(handler)_s$(set_number).nc"))
    else
        return ""
    end
end

function merge_specific_handlers(args)
    verbose = !args["quiet"]
    handlers = unique(normalize_handler_arg.(args["handlers"]))
    
    if isempty(handlers)
        println("No handler names provided. Use --auto for auto-discovery or specify handler names.")
        return false
    end
    
    merge_mode = str_to_merge_mode(args["mode"])
    requested_set = args["set"]
    cleanup = args["cleanup"]
    output_name = args["output"]
    
    success_count = 0
    total_count = 0
    
    for handler in handlers
        # Each requested handler is independent; continue through the list so a
        # single failed merge does not hide results for later handlers.
        verbose && println("\n" * "="^60)
        verbose && println("Merging handler: $handler")
        verbose && println("="^60)

        set_specs = discover_handler_sets(handler, requested_set)
        if requested_set == 0 && length(set_specs) > 1 && !isempty(output_name)
            println("Cannot use --output when merging multiple sets for handler $handler. Use --set=N with --output.")
            total_count += length(set_specs)
            continue
        end

        requested_set == 0 && verbose && println("Discovered sets: $(join(first.(set_specs), ", "))")

        for (set_number, workdir) in discover_handler_sets(handler, requested_set)
            total_count += 1
            verbose && length(set_specs) > 1 && println("\nMerging set $set_number")

            effective_output_name = merge_output_name(handler, set_number, workdir, output_name)
            if workdir != "."
                verbose && println("Using handler directory: $workdir")
            end

            success = cd(workdir) do
                merge_netcdf_files(handler,
                                   set_number=set_number,
                                   output_name=effective_output_name,
                                   merge_mode=merge_mode,
                                   cleanup=cleanup,
                                   verbose=verbose)
            end

            if success
                success_count += 1
                if verbose
                    println("Successfully merged $(handler)_s$(set_number)")
                else
                    println("Success: $(handler)_s$(set_number)")
                end
            else
                if verbose
                    println("Failed to merge $(handler)_s$(set_number)")
                else
                    println("Failed: $(handler)_s$(set_number)")
                end
            end
        end
    end
    
    # Final summary
    if verbose
        println("\n" * "="^60)
        if success_count == total_count
            println("All $total_count merge operations completed successfully!")
        else
            println("Summary: $success_count/$total_count merge operations completed successfully")
        end
        println("="^60)
    else
        println("$success_count/$total_count successful")
    end
    
    return success_count > 0
end

function show_usage_examples()
    println("""
USAGE EXAMPLES:

Basic Operations:
  julia merge_netcdf.jl snapshots                    # Merge all snapshots_s* sets
  julia merge_netcdf.jl snapshots analysis           # Merge multiple handlers
  julia merge_netcdf.jl --set=2 snapshots            # Merge specific set

Advanced Options:
  julia merge_netcdf.jl --cleanup snapshots          # Delete source files after merge
  julia merge_netcdf.jl --mode=concat analysis       # Use concatenation merge mode
  julia merge_netcdf.jl --output=final.nc snapshots  # Custom output filename

Auto-Discovery:
  julia merge_netcdf.jl --auto                       # Find and merge all handlers
  julia merge_netcdf.jl --auto --cleanup             # Auto-merge with cleanup

Quiet Mode:
  julia merge_netcdf.jl --quiet snapshots            # Minimal output for scripts

File Structure Expected:
  snapshots/
  ├── snapshots_s1/
  │   ├── snapshots_s1_p0.nc    # Processor 0 data
  │   └── snapshots_s1_p1.nc    # Processor 1 data
  └── snapshots_s2/
      ├── snapshots_s2_p0.nc
      └── snapshots_s2_p1.nc

After Merge:
  snapshots/
  ├── snapshots_s1.nc
  └── snapshots_s2.nc
""")
end

function main()
    args = try
        parse_arguments()
    catch e
        if isa(e, ArgParseError) && occursin("help", string(e))
            show_usage_examples()
            return 0
        else
            println("Error parsing arguments: $e")
            return 1
        end
    end

    verbose = !args["quiet"]
    
    print_banner(verbose)
    
    try
        if args["auto"]
            success = auto_discover_and_merge(args)
        else
            success = merge_specific_handlers(args)
        end
        
        if success
            verbose && println()
            verbose && println("Merge operation completed successfully!")
            return 0
        else
            verbose && println()
            verbose && println("Some merge operations failed.")
            return 1
        end
        
    catch e
        println("Error during merge: $e")
        if verbose
            println("Stack trace:")
            showerror(stdout, e, catch_backtrace())
        end
        return 1
    end
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
