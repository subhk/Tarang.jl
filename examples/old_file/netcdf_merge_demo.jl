"""
NetCDF Merge Demonstration for Tarang.jl

This example demonstrates the comprehensive NetCDF file merging capabilities:
- Creating sample per-processor files (simulating MPI run)
- Merging files using different modes 
- File cleanup options
- Batch processing of multiple handlers

Shows the complete workflow from creation → merging → analysis.

Run with:
    julia --project=.. netcdf_merge_demo.jl
"""

# Include required modules
include("../src/tools/netcdf_output.jl")
include("../src/tools/netcdf_merge.jl")

using MPI

function create_sample_processor_files()
    """Create sample per-processor NetCDF files to demonstrate merging"""
    
    println("Creating sample per-processor files for demonstration...")
    
    # Initialize MPI if needed
    if !MPI.Initialized()
        MPI.Init()
    end
    
    # Clean up any existing demo files
    for entry in readdir(".", join=true)
        if occursin("demo_", basename(entry)) && (isdir(entry) || endswith(entry, ".nc"))
            if isdir(entry)
                rm(entry, recursive=true)
            else
                rm(entry)
            end
        end
    end
    
    # Simulate creating files from 4 different "processors"
    original_rank = 0
    handlers_created = String[]
    
    for sim_rank in 0:3
        println("  Simulating processor $sim_rank...")
        
        # Temporarily override MPI rank for demonstration
        dist = nothing
        vars = Dict{String, Any}()
        
        # Create different handlers as if from different processors
        for (handler_name, config) in [
            ("demo_snapshots", (sim_dt=0.1, max_writes=5)),
            ("demo_analysis", (iter=2, max_writes=3))
        ]
            
            # Override rank in the handler creation
            handler = NetCDFFileHandler(handler_name, dist, vars; config...)
            handler.rank = sim_rank  # Override for demonstration
            handler.size = 4
            
            # Add some tasks
            add_task!(handler, "velocity_field", name="velocity")  
            add_task!(handler, "pressure_field", name="pressure")
            if handler_name == "demo_snapshots"
                add_task!(handler, "temperature_field", name="temperature")
            end
            
            # Generate some sample data at different "time steps"
            time_steps = handler_name == "demo_snapshots" ? [0.1, 0.2, 0.3] : [0.2, 0.4]
            
            for (i, sim_time) in enumerate(time_steps)
                if check_schedule(handler, iteration=i, sim_time=sim_time, timestep=0.1)
                    process!(handler, iteration=i, sim_time=sim_time, 
                           wall_time=sim_time*0.8, timestep=0.1)
                end
            end
            
            if sim_rank == 0 && !(handler_name in handlers_created)
                push!(handlers_created, handler_name)
            end
        end
    end
    
    println("  Created sample files for $(length(handlers_created)) handlers across 4 simulated processors")
    
    return handlers_created
end

function demonstrate_merge_modes(handler_names::Vector{String})
    """Demonstrate different merge modes"""
    
    println("\nDemonstrating different merge modes...")
    
    for (mode, mode_enum) in [("concatenation", SIMPLE_CONCAT), ("reconstruction", RECONSTRUCT)]
        println("\n" * "─"^50)
        println("Testing $mode merge mode")
        println("─"^50)
        
        handler_name = handler_names[1]  # Use first handler for demo
        
        # Create merger with specific mode
        merger = NetCDFMerger(handler_name, 
                             merge_mode=mode_enum,
                             output_name="$(handler_name)_$(mode).nc",
                             cleanup=false,
                             verbose=true)
        
        success = merge_files!(merger)
        
        if success
            println("$mode merge completed: $(merger.output_file)")
            
            # Show some info about the merged file
            if isfile(merger.output_file)
                file_size = round(filesize(merger.output_file)/1024, digits=2)
                println("   File size: $file_size KB")
                
                # Try to show some basic info
                try
                    info = ncinfo(merger.output_file)
                    println("   Variables: $(length(info.vars)) | Dimensions: $(length(info.dim))")
                catch e
                    println("   Warning: Could not read file info: $e")
                end
            end
        else
            println("$mode merge failed")
        end
    end
end

function demonstrate_cleanup_options(handler_names::Vector{String})
    """Demonstrate file cleanup functionality"""
    
    println("\nDemonstrating cleanup options...")
    
    if length(handler_names) < 2
        println("Warning: Need at least 2 handlers to demonstrate cleanup")
        return
    end
    
    handler_name = handler_names[2]  # Use second handler
    
    println("\nFiles before cleanup:")
    list_handler_files(handler_name)
    
    # Merge with cleanup enabled
    println("\nMerging with cleanup enabled...")
    
    success = merge_netcdf_files(handler_name,
                               cleanup=true,  # This will delete source files
                               verbose=true)
    
    if success
        println("\nFiles after cleanup:")
        list_handler_files(handler_name)
        
        println("Cleanup demonstration completed")
        println("   Note: Original per-processor files have been deleted")
        println("   Only the merged file remains")
    else
        println("Cleanup demonstration failed")
    end
end

function list_handler_files(handler_name::String)
    """List files related to a handler"""
    
    files_found = 0
    total_size = 0
    
    for entry in readdir(".", join=true)
        entry_name = basename(entry)
        
        if occursin(handler_name, entry_name)
            if isfile(entry) && endswith(entry, ".nc")
                size_kb = round(filesize(entry)/1024, digits=2)
                total_size += size_kb
                files_found += 1
                println("   $(entry_name) ($(size_kb) KB)")
            elseif isdir(entry)
                println("   $(entry_name)/")
                for subfile in readdir(entry, join=true)
                    if endswith(subfile, ".nc")
                        size_kb = round(filesize(subfile)/1024, digits=2)
                        total_size += size_kb
                        files_found += 1
                        println("      $(basename(subfile)) ($(size_kb) KB)")
                    end
                end
            end
        end
    end
    
    if files_found > 0
        println("   Total: $files_found files, $(round(total_size, digits=2)) KB")
    else
        println("   No files found for $handler_name")
    end
end

function demonstrate_batch_merge()
    """Demonstrate batch merging of multiple handlers"""
    
    println("\nDemonstrating batch merge functionality...")
    
    # Find all available handlers
    available_handlers = find_mergeable_handlers()
    
    if isempty(available_handlers)
        println("Warning: No handlers found for batch merge demonstration")
        return
    end
    
    println("Available handlers for batch merge:")
    handler_list = String[]
    for (handler, sets) in available_handlers
        println("   - $handler (sets: $(join(sets, ", ")))")
        push!(handler_list, handler)
    end
    
    # Batch merge all handlers
    println("\nStarting batch merge...")
    
    results = batch_merge_netcdf(handler_list,
                               merge_mode=RECONSTRUCT,
                               cleanup=false,  # Don't cleanup for demo
                               verbose=true)
    
    # Show results  
    successful = sum(values(results))
    total = length(results)
    
    println("\nBatch merge results:")
    for (handler, success) in results
        status = success ? "Success" : "Failed"
        println("   $status: $handler")
    end
    
    println("\nBatch merge summary: $successful/$total handlers successfully merged")
end

function demonstrate_auto_discovery()
    """Demonstrate automatic handler discovery"""
    
    println("\nDemonstrating automatic handler discovery...")
    
    handlers = find_mergeable_handlers()
    
    println("Auto-discovered handlers:")
    if isempty(handlers)
        println("   No mergeable handlers found")
    else
        for (handler, sets) in handlers
            println("   $handler:")
            for set_num in sets
                # Count processor files
                pattern = "$(handler)_s$(set_num)_p"
                proc_files = filter(f -> occursin(pattern, f) && endswith(f, ".nc"), readdir("."))
                dir_path = "$(handler)_s$(set_num)"
                if isdir(dir_path)
                    proc_files_in_dir = filter(f -> occursin(pattern, f) && endswith(f, ".nc"), readdir(dir_path))
                    println("      Set $set_num: $(length(proc_files_in_dir)) processor files in $dir_path/")
                else
                    println("      Set $set_num: $(length(proc_files)) processor files")
                end
            end
        end
        
        total_handlers = length(handlers)
        total_sets = sum(length(sets) for sets in values(handlers))
        println("\nDiscovery summary: $total_handlers handlers, $total_sets total sets")
    end
end

function main()
    """Main demonstration function"""
    
    println("NetCDF Merge Demonstration for Tarang.jl")
    println("="^60)
    
    # Step 1: Create sample files
    handler_names = create_sample_processor_files()
    
    if isempty(handler_names)
        println("Failed to create sample files")
        return
    end
    
    # Step 2: Demonstrate auto-discovery
    demonstrate_auto_discovery()
    
    # Step 3: Demonstrate different merge modes
    demonstrate_merge_modes(handler_names)
    
    # Step 4: Demonstrate batch processing
    demonstrate_batch_merge()
    
    # Step 5: Demonstrate cleanup (this will delete files!)
    demonstrate_cleanup_options(handler_names)
    
    println("\n" * "="^60)
    println("NetCDF merge demonstration completed!")
    println()
    println("Key takeaways:")
    println("   - Multiple merge modes available (concat, reconstruct)")
    println("   - Batch processing for multiple handlers")
    println("   - Optional cleanup of source files")
    println("   - Auto-discovery of mergeable files")
    println("   - Preserves metadata and coordinate information")
    println()
    println("Usage in your workflow:")
    println("   julia merge_netcdf.jl snapshots                    # Basic merge")
    println("   julia merge_netcdf.jl --auto --cleanup             # Auto-merge all with cleanup")
    println("   julia merge_netcdf.jl --mode=concat analysis       # Specific merge mode")
    
    return true
end

# Run demonstration if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
        
        # Clean shutdown
        if MPI.Initialized()
            MPI.Finalize()
        end
    catch e
        println("Error during demonstration: $e")
        if MPI.Initialized()
            MPI.Finalize()
        end
        rethrow(e)
    end
end