"""
Unit tests for TransposableField

Tests construction, local shape computation, pack/unpack operations,
and round-trip accuracy for the TransposableField pattern.
"""

using Test
using Tarang
using MPI

@testset "TransposableField Basic" begin

    @testset "TransposeLayout enum" begin
        @test XLocal isa TransposeLayout
        @test YLocal isa TransposeLayout
        @test ZLocal isa TransposeLayout
        @test Int(XLocal) != Int(YLocal)
        @test Int(YLocal) != Int(ZLocal)
    end

    @testset "divide_evenly" begin
        # Even division
        @test Tarang.divide_evenly(12, 4, 0) == 3
        @test Tarang.divide_evenly(12, 4, 1) == 3
        @test Tarang.divide_evenly(12, 4, 2) == 3
        @test Tarang.divide_evenly(12, 4, 3) == 3

        # Uneven division - remainder goes to first processes
        @test Tarang.divide_evenly(10, 4, 0) == 3  # rank 0 gets extra
        @test Tarang.divide_evenly(10, 4, 1) == 3  # rank 1 gets extra
        @test Tarang.divide_evenly(10, 4, 2) == 2
        @test Tarang.divide_evenly(10, 4, 3) == 2

        # Single process
        @test Tarang.divide_evenly(10, 1, 0) == 10
    end

    @testset "local_range" begin
        # Test local_range function
        @test Tarang.local_range(12, 4, 0) == 1:3
        @test Tarang.local_range(12, 4, 1) == 4:6
        @test Tarang.local_range(12, 4, 2) == 7:9
        @test Tarang.local_range(12, 4, 3) == 10:12

        # Uneven division
        @test Tarang.local_range(10, 4, 0) == 1:3
        @test Tarang.local_range(10, 4, 1) == 4:6
        @test Tarang.local_range(10, 4, 2) == 7:8
        @test Tarang.local_range(10, 4, 3) == 9:10
    end

    @testset "TransposeCounts construction" begin
        counts = Tarang.TransposeCounts(4)
        @test length(counts.zy_send_counts) == 4
        @test length(counts.zy_recv_counts) == 4
        @test length(counts.yx_send_counts) == 4
        @test all(counts.zy_send_counts .== 0)
    end

    @testset "TransposeComms construction" begin
        comms = Tarang.TransposeComms()
        @test comms.zy_comm === nothing
        @test comms.zy_rank == 0
        @test comms.zy_size == 1
    end

    @testset "Topology2D construction" begin
        # Default topology
        topo = Tarang.Topology2D()
        @test topo.Rx == 1
        @test topo.Ry == 1
        @test topo.rx == 0
        @test topo.ry == 0
        @test topo.row_comm === nothing
        @test topo.col_comm === nothing
    end

    @testset "auto_topology" begin
        # Test automatic topology computation
        @test Tarang.auto_topology(4, 3) == (2, 2)
        @test Tarang.auto_topology(6, 3) == (2, 3)
        @test Tarang.auto_topology(8, 3) == (2, 4)
        @test Tarang.auto_topology(9, 3) == (3, 3)
        @test Tarang.auto_topology(16, 3) == (4, 4)

        # 2D case: uses 1D decomposition
        @test Tarang.auto_topology(4, 2) == (4, 1)
        @test Tarang.auto_topology(8, 2) == (8, 1)
    end

    @testset "AsyncTransposeState construction" begin
        state = Tarang.AsyncTransposeState()
        @test state.request === nothing
        @test state.in_progress == false
        @test state.from_layout == ZLocal
        @test state.to_layout == ZLocal
        @test state.pack_time == 0.0
        @test state.comm_time == 0.0
        @test state.unpack_time == 0.0
        @test state.wait_time == 0.0
    end

end

@testset "TransposableField 2D" begin

    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())

    # Create a 2D domain
    xbasis = Fourier(coords, "x", 16)
    ybasis = Fourier(coords, "y", 16)
    domain = Domain(dist, (xbasis, ybasis))

    # Create a scalar field
    field = ScalarField(dist, "test", (xbasis, ybasis))
    field["g"] .= rand(16, 16)

    @testset "Construction" begin
        tf = TransposableField(field)

        @test tf.field === field
        @test tf.global_shape == (16, 16)
        @test length(tf.local_shapes) >= 1
    end

    @testset "Local shapes" begin
        tf = TransposableField(field)

        # For serial execution, shapes should be full global shape
        @test haskey(tf.local_shapes, YLocal)
        @test haskey(tf.local_shapes, XLocal)
    end

    @testset "Active layout" begin
        tf = TransposableField(field)

        @test active_layout(tf) == ZLocal
    end

    @testset "local_shape accessor" begin
        tf = TransposableField(field)

        # For serial execution, local shapes should match global shape
        @test local_shape(tf, XLocal) == tf.global_shape
        @test local_shape(tf, YLocal) == tf.global_shape
        @test local_shape(tf, ZLocal) == tf.global_shape
    end

    @testset "current_data accessor" begin
        tf = TransposableField(field)

        # Initially in ZLocal layout
        @test active_layout(tf) == ZLocal
        data = Tarang.current_data(tf)
        @test data === tf.buffers.z_local_data
    end

    @testset "make_transposable helper" begin
        tf = make_transposable(field)
        @test tf isa TransposableField
        @test tf.field === field
    end

    @testset "get_active_buffers" begin
        tf = TransposableField(field)

        send_buf, recv_buf = Tarang.get_active_buffers(tf)
        @test send_buf !== nothing
        @test recv_buf !== nothing

        # Test buffer swapping
        Tarang.swap_buffers!(tf)
        send_buf2, recv_buf2 = Tarang.get_active_buffers(tf)
        @test send_buf2 !== send_buf || recv_buf2 !== recv_buf  # At least one should differ

        # Swap back
        Tarang.swap_buffers!(tf)
        send_buf3, recv_buf3 = Tarang.get_active_buffers(tf)
        @test send_buf3 === send_buf
        @test recv_buf3 === recv_buf
    end

end

@testset "TransposableField 3D" begin

    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())

    # Create a 3D domain
    xbasis = Fourier(coords, "x", 8)
    ybasis = Fourier(coords, "y", 8)
    zbasis = Fourier(coords, "z", 8)
    domain = Domain(dist, (xbasis, ybasis, zbasis))

    # Create a scalar field
    field = ScalarField(dist, "test3d", (xbasis, ybasis, zbasis))
    field["g"] .= rand(8, 8, 8)

    @testset "Construction" begin
        tf = TransposableField(field)

        @test tf.field === field
        @test tf.global_shape == (8, 8, 8)
    end

    @testset "Local shapes for 3D" begin
        tf = TransposableField(field)

        @test haskey(tf.local_shapes, ZLocal)
        @test haskey(tf.local_shapes, YLocal)
        @test haskey(tf.local_shapes, XLocal)

        # For serial execution with single process
        # All shapes should match global shape
        @test tf.local_shapes[ZLocal] == (8, 8, 8)
        @test tf.local_shapes[YLocal] == (8, 8, 8)
        @test tf.local_shapes[XLocal] == (8, 8, 8)
    end

    @testset "Buffers allocated" begin
        tf = TransposableField(field)

        @test tf.buffers.z_local_data !== nothing
        @test tf.buffers.y_local_data !== nothing
        @test tf.buffers.x_local_data !== nothing
        @test tf.buffers.send_buffer !== nothing
        @test tf.buffers.recv_buffer !== nothing
    end

end

@testset "TransposableField Serial Transforms" begin

    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())

    xbasis = Fourier(coords, "x", 16)
    ybasis = Fourier(coords, "y", 16)
    domain = Domain(dist, (xbasis, ybasis))

    field = ScalarField(dist, "transform_test", (xbasis, ybasis))

    # Initialize with a known function
    x = range(0, 2π, length=16)
    y = range(0, 2π, length=16)
    for i in 1:16, j in 1:16
        field["g"][i, j] = sin(x[i]) * cos(y[j])
    end

    @testset "Forward transform preserves energy" begin
        tf = TransposableField(field)

        initial_energy = sum(abs2.(field["g"]))

        # Forward transform (serial - no actual MPI)
        distributed_forward_transform!(tf)

        # Energy should be approximately preserved (Parseval's theorem)
        spectral_energy = sum(abs2.(field["c"])) / prod(size(field["c"]))

        # Allow for numerical tolerance
        @test isapprox(initial_energy, spectral_energy * prod(size(field["g"])), rtol=0.1)
    end

    @testset "Round-trip transform" begin
        tf = TransposableField(field)

        # Store original data
        original = copy(field["g"])

        # Forward transform
        distributed_forward_transform!(tf)

        # Backward transform
        distributed_backward_transform!(tf)

        # Data should be recovered
        @test isapprox(field["g"], original, rtol=1e-10)
    end

    @testset "Round-trip with overlap flag" begin
        tf = TransposableField(field)

        # Store original data
        original = copy(field["g"])

        # Forward transform with overlap=true (still serial, but tests the code path)
        distributed_forward_transform!(tf; overlap=true)

        # Backward transform with overlap=true
        distributed_backward_transform!(tf; overlap=false)

        # Data should be recovered
        @test isapprox(field["g"], original, rtol=1e-10)
    end

    @testset "Performance statistics" begin
        tf = TransposableField(field)

        # Reset stats
        reset_transpose_stats!(tf)

        # Do some transforms
        distributed_forward_transform!(tf)
        distributed_backward_transform!(tf)

        # Get stats
        stats = get_transpose_stats(tf)

        @test stats.num_transposes >= 0
        @test stats.total_fft_time >= 0.0
        @test stats.total_pack_time >= 0.0
        @test stats.total_unpack_time >= 0.0
    end

end

@testset "TransposableField Pack/Unpack CPU" begin

    # Test pack and unpack operations
    data = rand(ComplexF64, 4, 4, 4)
    buffer = zeros(ComplexF64, length(data))

    counts = [16, 16, 16, 16]  # Equal chunks
    displs = [0, 16, 32, 48]

    arch = CPU()

    @testset "Pack operation" begin
        Tarang.pack_for_transpose!(buffer, data, counts, displs, 3, 4, arch)

        # Buffer should contain data
        @test sum(abs2.(buffer)) > 0
    end

    @testset "Unpack operation" begin
        output = zeros(ComplexF64, 4, 4, 4)

        Tarang.unpack_from_transpose!(output, buffer, counts, displs, 3, 4, arch)

        # After pack → unpack, data should match
        @test isapprox(output, data, rtol=1e-10)
    end

end

@testset "compute_local_shapes" begin

    @testset "2D 1D decomposition" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(2,), dtype=Float64, architecture=CPU())

        global_shape = (16, 16)
        shapes = Tarang.compute_local_shapes(global_shape, dist)

        @test haskey(shapes, YLocal)
        @test haskey(shapes, XLocal)
    end

    @testset "3D 2D decomposition" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(2, 2), dtype=Float64, architecture=CPU())

        global_shape = (16, 16, 16)
        shapes = Tarang.compute_local_shapes(global_shape, dist)

        @test haskey(shapes, ZLocal)
        @test haskey(shapes, YLocal)
        @test haskey(shapes, XLocal)
    end

end

# GPU tests (if available)
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

    @testset "TransposableField GPU" begin
        CUDA.allowscalar(false)

        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1,), dtype=Float32, architecture=GPU())

        xbasis = Fourier(coords, "x", 16)
        ybasis = Fourier(coords, "y", 16)
        domain = Domain(dist, (xbasis, ybasis))

        field = ScalarField(dist, "gpu_test", (xbasis, ybasis))
        field["g"] .= CUDA.rand(Float32, 16, 16)

        @testset "GPU construction" begin
            tf = TransposableField(field)

            @test tf.buffers.architecture isa Tarang.GPU
            @test tf.buffers.z_local_data isa CuArray
            @test tf.buffers.send_buffer isa CuArray
        end

        @testset "GPU round-trip" begin
            tf = TransposableField(field)

            original = copy(field["g"])

            distributed_forward_transform!(tf)
            distributed_backward_transform!(tf)

            @test isapprox(Array(field["g"]), Array(original), rtol=1e-4)
        end

    end
else
    @testset "TransposableField GPU" begin
        @test_skip "CUDA not available"
    end
end
