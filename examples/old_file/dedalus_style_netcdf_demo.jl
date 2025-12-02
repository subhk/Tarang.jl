"""
Dedalus-Style NetCDF Output Demonstration for Tarang.jl

This example demonstrates NetCDF output that exactly matches Dedalus file handling patterns:
- File structure: snapshots_s1/snapshots_s1_p0.nc (following Dedalus naming)
- API usage: snapshots = evaluator.add_file_handler('snapshots', sim_dt=0.25)
- Task system: snapshots.add_task(field, name='buoyancy')
- Automatic processing with proper time coordinates
- Per-processor files for MPI parallelism
- User-selectable Float32/Float64 precision

This matches the exact Dedalus pattern from examples like:
```python
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
```

Run with: 
- Serial: julia --project=. dedalus_style_netcdf_demo.jl
- Parallel: mpiexec -n 4 julia --project=. dedalus_style_netcdf_demo.jl
"""

# Include NetCDF output functionality directly
include("../src/tools/netcdf_output.jl")

using MPI

function dedalus_style_demo()
    """Demonstrate exact Dedalus-style NetCDF output API"""
    
    println("=== Dedalus-Style NetCDF Output Demo ===")
    
    # Initialize MPI if needed
    if !MPI.Initialized()
        MPI.Init()
    end
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    if rank == 0
        println("Running Dedalus-style NetCDF demo with $size MPI processes")
        println("This creates files following Dedalus patterns:")
        println("  snapshots_s1/snapshots_s1_p0.nc, snapshots_s1_p1.nc, ...")
    end
    
    # Simulate basic domain/vars (in real Tarang, these would be from problem setup)
    dist = nothing  # Placeholder for distributor
    vars = Dict("x" => "coordinate x", "z" => "coordinate z")
    
    # Create file handler with Dedalus-style API
    # Matches: snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
    snapshots = NetCDFFileHandler("snapshots", dist, vars, 
                                 sim_dt=0.25, max_writes=10, precision=Float64)
    
    println("Process $rank: Created snapshots handler following Dedalus pattern")
    println("Process $rank: Output directory: $(current_path(snapshots))")
    println("Process $rank: Output file: $(current_file(snapshots))")
    
    # Add tasks with Dedalus-style API
    # Matches: snapshots.add_task(b, name='buoyancy')
    # Matches: snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
    add_task!(snapshots, "buoyancy_field", name="buoyancy")
    add_task!(snapshots, "vorticity_field", name="vorticity") 
    add_task!(snapshots, "temperature_field", name="temperature")
    
    println("Process $rank: Added $(length(snapshots.tasks)) tasks to handler")
    
    # Simulate time stepping loop (matching Dedalus solver.step() calls)
    dt = 0.1
    sim_time = 0.0
    wall_time_start = time()
    
    println("Process $rank: Starting time stepping simulation...")
    
    for iteration in 1:25
        sim_time += dt
        wall_time = time() - wall_time_start
        
        # Check if handler should output (matching Dedalus evaluation logic)
        if check_schedule(snapshots, iteration=iteration, sim_time=sim_time, 
                         wall_time=wall_time, timestep=dt)
            
            println("Process $rank: Writing output at iteration $iteration, sim_time=$sim_time")
            
            # Process handler (matches Dedalus solver.evaluate_scheduled())
            process!(snapshots, iteration=iteration, sim_time=sim_time,
                    wall_time=wall_time, timestep=dt)
            
            println("Process $rank: Completed output write $(snapshots.total_write_num)")
        end
        
        # Simulate some work
        sleep(0.01)
    end
    
    return snapshots
end

function demonstrate_precision_options()
    """Show different precision options matching Dedalus flexibility"""
    
    println("=== Precision Options Demo ===")
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    dist = nothing
    vars = Dict{String, Any}()
    
    # Float32 for reduced file size (good for large 3D simulations)
    analysis_f32 = NetCDFFileHandler("analysis_f32", dist, vars,
                                    iter=5, max_writes=5, precision=Float32)
    add_task!(analysis_f32, "velocity_field", name="velocity")
    
    # Float64 for high precision (good for detailed analysis)  
    analysis_f64 = NetCDFFileHandler("analysis_f64", dist, vars,
                                    iter=5, max_writes=5, precision=Float64)
    add_task!(analysis_f64, "velocity_field", name="velocity")
    
    if rank == 0
        println("Created handlers with different precisions:")
        println("  Float32: $(current_file(analysis_f32))")
        println("  Float64: $(current_file(analysis_f64))")
    end
    
    # Write some data to show size difference
    for i in 1:3
        process!(analysis_f32, iteration=i*5, sim_time=i*0.5, timestep=0.1)
        process!(analysis_f64, iteration=i*5, sim_time=i*0.5, timestep=0.1)
    end
    
    if rank == 0
        println("Written 3 outputs to each precision handler")
    end
end

function demonstrate_file_structure()
    """Show the Dedalus-style file structure that gets created"""
    
    comm = MPI.COMM_WORLD 
    rank = MPI.Comm_rank(comm)
    
    if rank == 0
        println("=== File Structure Created ===")
        println("Following Dedalus pattern of handler_sN/handler_sN_pR.nc:")
        println()
        
        # List created directories and files
        for entry in readdir(".", join=false)
            if occursin("_s", entry)
                if isdir(entry)
                    println("$entry/")
                    for file in readdir(entry, join=false)
                        if endswith(file, ".nc")
                            file_path = joinpath(entry, file)
                            file_size = round(filesize(file_path) / 1024, digits=2)
                            println("  $file ($file_size KB)")
                        end
                    end
                    println()
                else
                    file_size = round(filesize(entry) / 1024, digits=2)
                    println("$entry ($file_size KB)")
                end
            end
        end
        
        println("Use tools like ncdump, ncview, or Python xarray to inspect:")
        println("  ncdump -h snapshots_s1/snapshots_s1_p0.nc")
        println("  # Shows Dedalus-style metadata and coordinate structure")
    end
end

function main()
    """Main demonstration function"""
    
    # Initialize MPI first
    if !MPI.Initialized()
        MPI.Init()
    end
    
    # Clean up from previous runs
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    if rank == 0
        # Clean previous outputs
        for entry in readdir(".", join=true)
            if occursin("_s", basename(entry)) && (isdir(entry) || endswith(entry, ".nc"))
                if isdir(entry)
                    rm(entry, recursive=true)
                else
                    rm(entry)
                end
            end
        end
    end
    
    MPI.Barrier(comm)
    
    # Run demonstrations
    snapshots = dedalus_style_demo()
    
    MPI.Barrier(comm)
    demonstrate_precision_options()
    
    MPI.Barrier(comm) 
    demonstrate_file_structure()
    
    if rank == 0
        println("\n=== Dedalus-Style NetCDF Demo Complete ===")
        println("Created per-processor NetCDF files following Dedalus naming")
        println("Demonstrated task-based output system") 
        println("Showed precision options (Float32/Float64)")
        println("Used exact Dedalus API patterns")
        println("\nFile structure matches Dedalus HDF5 layout but in NetCDF format.")
        println("Per-processor files ready for post-processing or merging.")
    end
end

# Run the demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    main()
    
    # Clean shutdown
    if MPI.Initialized()
        MPI.Finalize()
    end
end