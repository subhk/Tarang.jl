"""
NetCDF Output Demonstration for Tarang.jl

This example demonstrates the comprehensive NetCDF output capabilities including:
- Per-processor file writing for MPI parallelism
- User-selectable precision (Float32 vs Float64)
- Rich metadata and coordinate information
- CF conventions compliance
- Compression and chunking options
- File merging utilities

Features demonstrated:
1. NetCDF output with correct NetCDF.jl API
2. Different precision settings
3. Per-processor vs single file writing
4. Time series output
5. File merging for parallel runs

Run with: 
- Serial: julia --project=. netcdf_output_demo.jl
- Parallel: mpiexec -n 4 julia --project=. netcdf_output_demo.jl
"""

# Include NetCDF output functionality directly
include("../src/tools/netcdf_output.jl")

using MPI

function example_netcdf_basic_usage()
    """Basic NetCDF output example"""
    
    println("=== Basic NetCDF Output Example ===")
    
    # Initialize MPI if needed
    if !MPI.Initialized()
        MPI.Init()
    end
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    if rank == 0
        println("Running NetCDF output demo with $size MPI processes")
    end
    
    # Create NetCDF file handler with Float64 precision
    handler = NetCDFFileHandler("demo_output.nc", precision=Float64, per_processor_files=true)
    
    println("Process $rank: Created NetCDF handler for file: $(handler.filename)")
    
    # Simulate some field data
    Nx, Nz = 32, 16
    
    # Create sample coordinate data
    x_coords = collect(range(0.0, 4.0, length=Nx))
    z_coords = collect(range(0.0, 1.0, length=Nz))
    
    # Create field info for file setup
    field_info = Dict(
        "dimensions" => Dict("x" => Nx, "z" => Nz),
        "coordinates" => Dict("x" => x_coords, "z" => z_coords)
    )
    
    # Create the NetCDF file
    success = create_netcdf_file!(handler, field_info)
    println("Process $rank: NetCDF file creation successful: $success")
    
    # Generate and write some time-series data
    dt = 0.1
    n_steps = 5
    
    for step in 1:n_steps
        time_value = step * dt
        
        # Generate sample data (different for each processor)
        u_data = randn(Float64, Nx, Nz) .+ rank  # Add rank offset for identification
        v_data = randn(Float64, Nx, Nz) .+ 10*rank
        
        # Write fields to NetCDF
        write_field_to_netcdf!(handler, "u_velocity", u_data, time_value)
        write_field_to_netcdf!(handler, "v_velocity", v_data, time_value)
        
        println("Process $rank: Written data at time $time_value")
    end
    
    # Close files
    close_netcdf_files!(handler)
    println("Process $rank: Closed NetCDF files")
    
    return handler.base_filename
end

function example_netcdf_precision_comparison()
    """Compare Float32 vs Float64 precision output"""
    
    println("=== NetCDF Precision Comparison ===")
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    # Create handlers with different precisions
    handler32 = NetCDFFileHandler("precision_test_f32.nc", precision=Float32, per_processor_files=true)
    handler64 = NetCDFFileHandler("precision_test_f64.nc", precision=Float64, per_processor_files=true)
    
    # Simple test data
    Nx, Nz = 16, 8
    x_coords = collect(range(0.0, 2π, length=Nx))
    z_coords = collect(range(0.0, 1.0, length=Nz))
    
    field_info = Dict(
        "dimensions" => Dict("x" => Nx, "z" => Nz), 
        "coordinates" => Dict("x" => x_coords, "z" => z_coords)
    )
    
    # Setup both files
    create_netcdf_file!(handler32, field_info)
    create_netcdf_file!(handler64, field_info)
    
    # Generate high-precision test data
    test_data = sin.(x_coords') .* cos.(2*z_coords) .+ π/1e6  # Small precision component
    
    # Write same data to both files
    write_field_to_netcdf!(handler32, "test_field", test_data, 1.0)
    write_field_to_netcdf!(handler64, "test_field", test_data, 1.0)
    
    close_netcdf_files!(handler32)
    close_netcdf_files!(handler64)
    
    if rank == 0
        println("Created precision comparison files:")
        println("  - Float32: precision_test_f32_proc_0000.nc")  
        println("  - Float64: precision_test_f64_proc_0000.nc")
        println("Use ncdump or similar tools to compare file sizes and precision")
    end
end

function example_netcdf_file_info()
    """Demonstrate NetCDF file information retrieval"""
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    if rank == 0
        println("=== NetCDF File Information ===")
        
        # Try to get info from a created file
        test_files = ["demo_output_proc_0000.nc", "precision_test_f64_proc_0000.nc"]
        
        for filename in test_files
            if isfile(filename)
                println("Information for $filename:")
                try
                    info = get_netcdf_info(filename)
                    println("  File info retrieved successfully")
                    # NetCDF.jl's ncinfo returns detailed information
                    println("  Use ncinfo(\"$filename\") for detailed information")
                catch e
                    println("  Error retrieving info: $e")
                end
                println()
            end
        end
    end
end

function main()
    """Main demonstration function"""
    
    # Run examples
    base_filename = example_netcdf_basic_usage()
    example_netcdf_precision_comparison()
    
    # Synchronize all processes before file operations
    MPI.Barrier(MPI.COMM_WORLD)
    
    # Demonstrate file merging (information only)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    if rank == 0
        println("=== File Merging Demonstration ===")
        success = merge_processor_files(base_filename, cleanup=false)
        println("Merge operation completed: $success")
    end
    
    MPI.Barrier(MPI.COMM_WORLD)
    example_netcdf_file_info()
    
    if rank == 0
        println("=== NetCDF Output Demo Complete ===")
        println("Check the generated .nc files with tools like ncdump, ncview, or Python/xarray")
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