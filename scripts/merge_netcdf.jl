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
    --set=N              Set number to merge (default: 1)
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
    snapshots_s1/snapshots_s1.nc  # Merged file
    
With --cleanup, original processor files are deleted after successful merge.
"""

using Pkg
# Ensure we can find the Tarang tools
if !("." in LOAD_PATH)
    push!(LOAD_PATH, dirname(@__DIR__))
end

# Load merging functionality
include(joinpath(dirname(@__DIR__), "src", "tools", "netcdf_merge.jl"))

using ArgParse

# Version info (sourced from main module)
const VERSION = Tarang.__version__
const SCRIPT_NAME = "merge_netcdf.jl"

function parse_arguments()
    s = ArgParseSettings(description="Merge per-processor NetCDF files from Tarang.jl",
                        version="$SCRIPT_NAME v$VERSION",
                        add_version=true)

    @add_arg_table! s begin
        "handlers"
            nargs = '*'
            help = "Handler names to merge (e.g., snapshots, analysis)"
            
        "--set", "-s"
            arg_type = Int
            default = 1
            help = "Set number to merge"
            
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

function auto_discover_and_merge(args)
    verbose = !args["quiet"]
    
    verbose && println("Auto-discovering mergeable handlers...")
    
    handlers = find_mergeable_handlers()
    
    if isempty(handlers)
        println("No mergeable handlers found in current directory")
        return false
    end
    
    verbose && println("Found $(length(handlers)) mergeable handlers:")
    for (handler, sets) in handlers
        verbose && println("   - $handler: sets $(join(sets, ", "))")
    end
    println()
    
    # Merge each handler
    merge_mode = str_to_merge_mode(args["mode"])
    set_number = args["set"]
    cleanup = args["cleanup"]
    
    results = batch_merge_netcdf(collect(keys(handlers)),
                               set_number=set_number,
                               merge_mode=merge_mode,
                               cleanup=cleanup,
                               verbose=verbose)
    
    # Summary
    successful = sum(values(results))
    total = length(results)
    
    if verbose
        println()
        if successful == total
            println("All $total handlers merged successfully!")
        else
            println("Warning: $successful/$total handlers merged successfully")
            for (handler, success) in results
                if !success
                    println("   Failed: $handler")
                end
            end
        end
    else
        println("Merged $successful/$total handlers")
    end
    
    return successful > 0
end

function merge_specific_handlers(args)
    verbose = !args["quiet"]
    handlers = args["handlers"]
    
    if isempty(handlers)
        println("No handler names provided. Use --auto for auto-discovery or specify handler names.")
        return false
    end
    
    merge_mode = str_to_merge_mode(args["mode"])
    set_number = args["set"]
    cleanup = args["cleanup"]
    output_name = args["output"]
    
    success_count = 0
    
    for handler in handlers
        verbose && println("\n" * "="^60)
        verbose && println("Merging handler: $handler")
        verbose && println("="^60)
        
        success = merge_netcdf_files(handler,
                                   set_number=set_number,
                                   output_name=isempty(output_name) ? "" : output_name,
                                   merge_mode=merge_mode,
                                   cleanup=cleanup,
                                   verbose=verbose)
        
        if success
            success_count += 1
            if verbose
                println("Successfully merged $handler")
            else
                println("Success: $handler")
            end
        else
            if verbose
                println("Failed to merge $handler")
            else
                println("Failed: $handler")
            end
        end
    end
    
    # Final summary
    total = length(handlers)
    if verbose
        println("\n" * "="^60)
        if success_count == total
            println("All $total handlers merged successfully!")
        else
            println("Summary: $success_count/$total handlers merged successfully")
        end
        println("="^60)
    else
        println("$success_count/$total successful")
    end
    
    return success_count > 0
end

function show_usage_examples()
    println("""
USAGE EXAMPLES:

Basic Operations:
  julia merge_netcdf.jl snapshots                    # Merge snapshots_s1 files
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
  snapshots_s1/
  ├── snapshots_s1_p0.nc    # Processor 0 data
  ├── snapshots_s1_p1.nc    # Processor 1 data  
  └── snapshots_s1_p2.nc    # Processor 2 data

After Merge:
  snapshots_s1/
  └── snapshots_s1.nc    # Combined data (with --cleanup, p*.nc files removed)
""")
end

function main()
    try
        args = parse_arguments()
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