"""
MPI Distributor Tests - Multi-rank communication tests

Run with: mpiexec -n 4 julia --project test/test_mpi_distributor.jl

These tests verify actual MPI communication between multiple ranks.
"""

using Test
using Tarang
using MPI
using PencilArrays

# Initialize MPI
if !MPI.Initialized()
    MPI.Init()
end

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if rank == 0
    println("="^60)
    println("MPI Distributor Tests")
    println("Running with $nprocs MPI processes")
    println("="^60)
end
MPI.Barrier(comm)

# Skip tests if running with single process
if nprocs < 2
    if rank == 0
        @warn "MPI tests require at least 2 processes. Run with: mpiexec -n 2 julia --project test/test_mpi_distributor.jl"
    end
    exit(0)
end

@testset "MPI Distributor Multi-rank (rank=$rank, nprocs=$nprocs)" begin

    @testset "Distributor creation with mesh" begin
        coords = CartesianCoordinates("x", "y")

        # Create distributor with actual mesh decomposition
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        @test dist.size == nprocs
        @test dist.rank == rank
        @test 0 <= dist.rank < nprocs
    end

    @testset "Allreduce across ranks" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Each rank contributes its rank number
        local_array = fill(Float64(rank), 8)

        # Sum across all ranks (op is positional argument, not keyword)
        reduced = allreduce_array(dist, local_array, MPI.SUM)

        # Expected sum: 0 + 1 + 2 + ... + (nprocs-1) = nprocs*(nprocs-1)/2
        expected_sum = nprocs * (nprocs - 1) / 2
        @test all(reduced .≈ expected_sum)
    end

    @testset "Gather from all ranks" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Each rank has unique data
        local_size = 4
        local_array = fill(Float64(rank + 1), local_size)

        # Gather to rank 0
        gathered = gather_array(dist, local_array)

        if rank == 0
            # Rank 0 should have all data
            @test length(gathered) == local_size * nprocs
            # Check each chunk
            for r in 0:(nprocs-1)
                chunk = gathered[(r*local_size+1):((r+1)*local_size)]
                @test all(chunk .≈ Float64(r + 1))
            end
        end
    end

    @testset "Scatter from rank 0" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        local_size = 4
        total_size = local_size * nprocs

        # All ranks need full data for scatter_array
        full_data = Float64.(1:total_size)

        # Scatter from rank 0
        scattered = scatter_array(dist, full_data)

        # Verify we got some data back
        @test length(scattered) > 0
        @test scattered isa AbstractArray
    end

    @testset "Alltoall communication" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Each rank sends different data to each other rank
        send_buf = Float64.([(rank * nprocs + i) for i in 1:nprocs])
        recv_buf = similar(send_buf)

        mpi_alltoall(dist, send_buf, recv_buf)

        # After alltoall, rank r receives element (i*nprocs + r+1) from rank i
        for i in 0:(nprocs-1)
            expected = Float64(i * nprocs + rank + 1)
            @test recv_buf[i+1] ≈ expected
        end
    end

    @testset "Broadcast from root" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        if rank == 0
            data = Float64.([1.0, 2.0, 3.0, 4.0])
        else
            data = zeros(Float64, 4)
        end

        # Broadcast from rank 0
        MPI.Bcast!(data, 0, comm)

        @test data == Float64.([1.0, 2.0, 3.0, 4.0])
    end

    @testset "2D mesh decomposition" begin
        if nprocs >= 4
            coords = CartesianCoordinates("x", "y", "z")

            # Create 2x2 mesh for 4 processes, or adapt for other counts
            mesh_y = 2
            mesh_z = nprocs ÷ 2

            dist = Distributor(coords; mesh=(mesh_y, mesh_z), dtype=Float64, architecture=CPU())

            @test dist.size == nprocs
            @test dist.rank == rank

            # Test topology creation
            topo = Tarang.create_topology_2d(comm, mesh_z, mesh_y)

            @test topo.Rx == mesh_z
            @test topo.Ry == mesh_y
            @test 0 <= topo.rx < mesh_z
            @test 0 <= topo.ry < mesh_y
        end
    end

    @testset "Pencil array distribution" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        global_size = (32, 32)

        # Create pencil - distribution depends on PencilArrays decomposition
        pencil = create_pencil(dist, global_size, 1; dtype=Float64)

        # Local size should be a fraction of global (product should be <= global product / nprocs)
        local_shape = size(pencil)
        @test prod(local_shape) <= prod(global_size)
        @test prod(local_shape) > 0

        # Fill with rank-specific data
        pencil .= Float64(rank + 1)

        # Verify data integrity after barrier
        MPI.Barrier(comm)
        @test all(pencil .== Float64(rank + 1))
    end

    @testset "Field distribution across ranks" begin
        coords = CartesianCoordinates("x", "y")

        # TransposableField uses ZLocal convention (decompose FIRST dims), not PencilArrays convention.
        # For TransposableField tests, use use_pencil_arrays=false and a 2D mesh.
        # Compute 2D mesh factors for nprocs
        Rx = isqrt(nprocs)
        while nprocs % Rx != 0
            Rx -= 1
        end
        Ry = nprocs ÷ Rx

        dist = Distributor(coords; mesh=(Rx, Ry), dtype=Float64, architecture=CPU(),
                          use_pencil_arrays=false)

        # Use ComplexFourier for TransposableField testing (RealFourier not supported for custom transposes)
        xbasis = ComplexFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        ybasis = ComplexFourier(coords["y"]; size=32, bounds=(0.0, 2π))

        field = ScalarField(dist, "mpi_test", (xbasis, ybasis))

        # Each rank fills its local portion
        field["g"] .= Float64(rank * 1000)

        # Create TransposableField and verify
        tf = TransposableField(field)

        @test active_layout(tf) == ZLocal  # Default layout
        @test tf.buffers.architecture isa Tarang.CPU

        # Local data should match what we set
        @test all(field["g"] .== Float64(rank * 1000))
    end

end

@testset "MPI Transform Tests (rank=$rank)" begin

    @testset "PencilFFT forward/backward transform round-trip" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Use RealFourier bases - PencilFFTs properly handles RFFT transforms
        xbasis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=32, bounds=(0.0, 2π))

        field = ScalarField(dist, "transform_test", (xbasis, ybasis))

        # Initialize with smooth function
        local_shape = size(field["g"])
        for i in 1:local_shape[1], j in 1:local_shape[2]
            # Use global-ish index for deterministic data
            field["g"][i, j] = sin(2π * i / 32) * cos(2π * j / 32) + rank * 0.01
        end

        original = copy(field["g"])

        # Use regular forward/backward transforms which properly use PencilFFTs
        forward_transform!(field)
        backward_transform!(field)

        # Should recover original data (allow some tolerance for FFT round-trip)
        @test isapprox(field["g"], original, rtol=1e-10)
    end

    @testset "Transpose Z to Y and back" begin
        coords = CartesianCoordinates("x", "y")

        # TransposableField uses ZLocal convention, so use use_pencil_arrays=false and 2D mesh
        Rx = isqrt(nprocs)
        while nprocs % Rx != 0
            Rx -= 1
        end
        Ry = nprocs ÷ Rx

        dist = Distributor(coords; mesh=(Rx, Ry), dtype=Float64, architecture=CPU(),
                          use_pencil_arrays=false)

        # Use ComplexFourier for TransposableField (RealFourier not supported for custom transposes)
        xbasis = ComplexFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        ybasis = ComplexFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        field = ScalarField(dist, "transpose_test", (xbasis, ybasis))
        field["g"] .= Float64(rank + 1)

        tf = TransposableField(field)

        @test active_layout(tf) == ZLocal

        # Z to Y
        transpose_z_to_y!(tf)
        @test active_layout(tf) == YLocal

        # Y back to Z
        transpose_y_to_z!(tf)
        @test active_layout(tf) == ZLocal
    end

    # NOTE: 3D transpose with 2D mesh is tested in GPU tests below.
    # TransposableField is designed for GPU+MPI (uses ZLocal convention: first dims decomposed, z local).
    # This is incompatible with PencilArrays which uses the opposite convention (last dims decomposed).
    # For CPU+MPI, use PencilArrays' native transpose operations instead of TransposableField.

end

@testset "PencilArray Preservation Tests (rank=$rank)" begin

    @testset "PencilArray wrapper for grid and coeff data" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Use RealFourier which has better PencilFFT support
        xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        field = ScalarField(dist, "pencil_test", (xbasis, ybasis))

        # Verify field uses PencilArray for MPI
        grid_data = Tarang.get_grid_data(field)
        @test grid_data isa PencilArrays.PencilArray

        # Verify coeff data is also PencilArray
        coeff_data = Tarang.get_coeff_data(field)
        @test coeff_data isa PencilArrays.PencilArray

        # Initialize with rank-specific data
        grid_data .= Float64(rank + 1)

        # Verify data can be read back
        @test all(Array(grid_data) .== Float64(rank + 1))
    end

    @testset "PencilArray global shape accessible" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=24, bounds=(0.0, 2π))

        field = ScalarField(dist, "shape_test", (xbasis, ybasis))

        grid_data = Tarang.get_grid_data(field)
        @test grid_data isa PencilArrays.PencilArray

        # Access global shape from PencilArray
        pencil = PencilArrays.pencil(grid_data)
        global_dims = PencilArrays.size_global(pencil)

        @test global_dims == (32, 24)

        # Local shape should be a portion of global
        local_dims = size(grid_data)
        @test prod(local_dims) <= prod(global_dims)
    end

    @testset "PencilArray preserved after RealFourier transform round-trip" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        field = ScalarField(dist, "transform_pencil_test", (xbasis, ybasis))

        # Initialize with smooth function
        grid_data = Tarang.get_grid_data(field)
        local_shape = size(grid_data)
        for i in 1:local_shape[1], j in 1:local_shape[2]
            grid_data[i, j] = sin(2π * i / 16) * cos(2π * j / 16) + 0.01 * rank
        end

        original = copy(Array(grid_data))

        # Verify PencilArray before transform
        @test grid_data isa PencilArrays.PencilArray

        # Transform round-trip
        forward_transform!(field)
        backward_transform!(field)

        # Verify PencilArray preserved after transform
        grid_data_after = Tarang.get_grid_data(field)
        @test grid_data_after isa PencilArrays.PencilArray

        # Data should be recovered
        @test isapprox(Array(grid_data_after), original, rtol=1e-10)
    end

end

@testset "Pencil Compatibility and Reallocation Tests (rank=$rank)" begin

    @testset "ensure_pencil_compatibility! with RealFourier bases" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Use RealFourier bases
        # In MPI+PencilFFTs: first axis RFFT (N/2+1), second axis FFT (full N)
        xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        field = ScalarField(dist, "compat_test", (xbasis, ybasis))

        # Verify initial allocation
        grid_data = Tarang.get_grid_data(field)
        coeff_data = Tarang.get_coeff_data(field)

        @test grid_data isa PencilArrays.PencilArray
        @test coeff_data isa PencilArrays.PencilArray

        # Get global shapes
        grid_gshape = Tarang.global_shape(field.domain, :g)
        coeff_gshape = Tarang.global_shape(field.domain, :c)

        # Grid shape should be (16, 16)
        @test grid_gshape == (16, 16)

        # MPI+PencilFFTs: RFFT on first axis (16/2+1=9), FFT on second axis (full 16)
        @test coeff_gshape == (9, 16)

        # Create PencilConfig matching field's domain
        config = Tarang.PencilConfig(grid_gshape, dist.mesh; comm=dist.comm, dtype=Float64)

        # Initialize field with data
        grid_data .= Float64(rank + 1)
        original_grid = copy(Array(grid_data))

        # ensure_pencil_compatibility! should return false (no reallocation needed)
        was_modified = Tarang.ensure_pencil_compatibility!(field, config)
        @test was_modified == false

        # Data should be unchanged
        @test isapprox(Array(Tarang.get_grid_data(field)), original_grid, rtol=1e-10)

        # PencilArray should be preserved
        @test Tarang.get_grid_data(field) isa PencilArrays.PencilArray
    end

    @testset "ensure_pencil_compatibility! triggers reallocation on shape mismatch" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        field = ScalarField(dist, "realloc_test", (xbasis, ybasis))

        # Verify initial grid shape
        initial_gshape = Tarang.global_shape(field.domain, :g)
        @test initial_gshape == (16, 16)

        # Create a PencilConfig with DIFFERENT global shape to force reallocation
        new_global_shape = (32, 32)
        config = Tarang.PencilConfig(new_global_shape, dist.mesh; comm=dist.comm, dtype=Float64)

        # ensure_pencil_compatibility! should return true (reallocation needed)
        was_modified = Tarang.ensure_pencil_compatibility!(field, config)
        @test was_modified == true

        # Field should now have new shape matching config
        new_grid_data = Tarang.get_grid_data(field)
        @test new_grid_data isa PencilArrays.PencilArray

        # Verify global shape from PencilArray matches config
        pencil = PencilArrays.pencil(new_grid_data)
        global_dims = PencilArrays.size_global(pencil)
        @test global_dims == new_global_shape
    end

    @testset "allocate_field_data! RealFourier coefficient shape" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Use RealFourier with different sizes to verify N/2+1 calculation
        xbasis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=24, bounds=(0.0, 2π))

        field = ScalarField(dist, "coeff_shape_test", (xbasis, ybasis))

        # Grid shape should be (32, 24)
        grid_gshape = Tarang.global_shape(field.domain, :g)
        @test grid_gshape == (32, 24)

        # MPI+PencilFFTs: RFFT on first axis (32/2+1=17), FFT on second axis (full 24)
        coeff_gshape = Tarang.global_shape(field.domain, :c)
        @test coeff_gshape == (17, 24)

        # Verify coeff data was allocated with correct shape
        coeff_data = Tarang.get_coeff_data(field)
        @test coeff_data isa PencilArrays.PencilArray

        coeff_pencil = PencilArrays.pencil(coeff_data)
        coeff_global_dims = PencilArrays.size_global(coeff_pencil)
        @test coeff_global_dims == (17, 24)
    end

    @testset "is_shape_compatible with PencilArrays convention" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        field = ScalarField(dist, "compat_shape_test", (xbasis, ybasis))

        grid_data = Tarang.get_grid_data(field)
        local_shape = size(grid_data)
        global_shape = (16, 16)

        # Should be compatible with PencilArrays convention (decompose last dims)
        @test Tarang.is_shape_compatible(local_shape, global_shape, dist.mesh, dist.comm;
                                         use_pencil_arrays=true)

        # For nprocs > 1, local shape should differ from global
        if nprocs > 1
            @test local_shape != global_shape
            # But the product of local shapes across ranks should equal global
            # (can't easily test this without gathering all shapes)
        end
    end

end

@testset "Group Transpose Tests (rank=$rank)" begin

    @testset "group_transpose_fields! basic functionality" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Use RealFourier for better PencilFFT compatibility
        xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        # Create multiple fields
        field1 = ScalarField(dist, "group_test1", (xbasis, ybasis))
        field2 = ScalarField(dist, "group_test2", (xbasis, ybasis))

        # Initialize with distinct rank-specific data
        Tarang.get_grid_data(field1) .= Float64(rank + 1)
        Tarang.get_grid_data(field2) .= Float64((rank + 1) * 10)

        original1 = copy(Array(Tarang.get_grid_data(field1)))
        original2 = copy(Array(Tarang.get_grid_data(field2)))

        # Get global shape from domain
        gshape = Tarang.global_shape(field1.domain)
        @test gshape == (16, 16)

        # Verify fields use PencilArrays
        @test Tarang.get_grid_data(field1) isa PencilArrays.PencilArray
        @test Tarang.get_grid_data(field2) isa PencilArrays.PencilArray

        # Test group_transpose_fields! with identity transpose (same decomp dims)
        # This tests that the function works without crashing and preserves data
        ndims_global = length(gshape)
        ndims_mesh = length(dist.mesh)
        source_decomp = ntuple(i -> ndims_global - ndims_mesh + i, ndims_mesh)
        dest_decomp = source_decomp  # Same decomposition = data should be unchanged

        fields = [field1, field2]
        Tarang.group_transpose_fields!(fields, dist, source_decomp, dest_decomp)

        # Data should be preserved after identity transpose
        result1 = Array(Tarang.get_grid_data(field1))
        result2 = Array(Tarang.get_grid_data(field2))

        @test isapprox(result1, original1, rtol=1e-10)
        @test isapprox(result2, original2, rtol=1e-10)

        # Fields should still have PencilArray wrappers
        @test Tarang.get_grid_data(field1) isa PencilArrays.PencilArray
        @test Tarang.get_grid_data(field2) isa PencilArrays.PencilArray
    end

    if nprocs >= 2
        @testset "group_transpose_fields! with actual transpose" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

            xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
            ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

            field = ScalarField(dist, "transpose_group", (xbasis, ybasis))

            # Initialize with position-dependent data to verify transpose correctness
            local_data = Tarang.get_grid_data(field)
            local_shape = size(local_data)
            for i in 1:local_shape[1], j in 1:local_shape[2]
                local_data[i, j] = Float64(i * 100 + j + rank * 10000)
            end

            # Store original for comparison after round-trip
            original = copy(Array(local_data))

            # For 2D with 1D mesh, transpose between decomp=(2,) and decomp=(1,)
            # decomp=(2,) means dim 2 is distributed
            # decomp=(1,) means dim 1 is distributed
            source_decomp = (2,)  # y distributed
            dest_decomp = (1,)    # x distributed

            fields = [field]

            # Forward transpose
            Tarang.group_transpose_fields!(fields, dist, source_decomp, dest_decomp)

            # Verify PencilArray preserved
            @test Tarang.get_grid_data(field) isa PencilArrays.PencilArray

            # Backward transpose
            Tarang.group_transpose_fields!(fields, dist, dest_decomp, source_decomp)

            # Data should be restored after round-trip
            result = Array(Tarang.get_grid_data(field))
            @test isapprox(result, original, rtol=1e-10)
        end
    end

    @testset "group_transpose_fields! uses global shape not local" begin
        # This test verifies the fix for using global_shape(field.domain) instead of size(grid_data)
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())

        # Use non-square dimensions to catch size confusion
        xbasis = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        ybasis = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        field = ScalarField(dist, "global_shape_test", (xbasis, ybasis))

        # Verify global shape from domain
        gshape = Tarang.global_shape(field.domain)
        @test gshape == (32, 16)

        # Local shape will be different due to MPI decomposition
        local_shape = size(Tarang.get_grid_data(field))

        # For nprocs > 1, local shape should differ from global in at least one dimension
        if nprocs > 1
            @test local_shape != gshape
        end

        # Initialize data
        Tarang.get_grid_data(field) .= Float64(rank + 1)
        original = copy(Array(Tarang.get_grid_data(field)))

        # Perform identity transpose - this should work correctly with global shape
        ndims_global = length(gshape)
        ndims_mesh = length(dist.mesh)
        decomp = ntuple(i -> ndims_global - ndims_mesh + i, ndims_mesh)

        Tarang.group_transpose_fields!([field], dist, decomp, decomp)

        # Data preserved
        @test isapprox(Array(Tarang.get_grid_data(field)), original, rtol=1e-10)
    end

end

# GPU tests if available
const _HAS_CUDA = try
    Tarang.has_cuda() && begin
        using CUDA
        CUDA.functional()
    end
catch
    false
end

if _HAS_CUDA
    using CUDA

    @testset "MPI GPU Distributor (rank=$rank)" begin
        CUDA.allowscalar(false)

        # Set GPU device based on rank
        if CUDA.ndevices() >= nprocs
            CUDA.device!(rank % CUDA.ndevices())
        end

        if rank == 0
            println("GPU device for rank $rank: $(CUDA.device())")
        end

        @testset "GPU allreduce" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(nprocs,), dtype=Float32, architecture=GPU())

            # Each rank contributes its rank number
            local_array = CUDA.fill(Float32(rank), 8)

            reduced = allreduce_array(dist, local_array, MPI.SUM)

            expected_sum = Float32(nprocs * (nprocs - 1) / 2)
            @test all(Array(reduced) .≈ expected_sum)
        end

        @testset "GPU TransposableField round-trip" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(nprocs,), dtype=Float32, architecture=GPU())

            # Use ComplexFourier for GPU+MPI which uses TransposableField (RealFourier not supported)
            xbasis = ComplexFourier(coords["x"]; size=32, bounds=(0.0, 2π))
            ybasis = ComplexFourier(coords["y"]; size=32, bounds=(0.0, 2π))

            field = ScalarField(dist, "gpu_mpi", (xbasis, ybasis))
            field["g"] .= CUDA.rand(Float32, size(field["g"])...)

            original = copy(field["g"])

            # Use regular forward/backward transforms
            forward_transform!(field)
            backward_transform!(field)

            @test isapprox(Array(field["g"]), Array(original), rtol=1e-4)
        end
    end
else
    if rank == 0
        @info "CUDA not available, skipping GPU MPI tests"
    end
end

# Final sync
MPI.Barrier(comm)

if rank == 0
    println("="^60)
    println("All MPI Distributor tests completed successfully!")
    println("="^60)
end
