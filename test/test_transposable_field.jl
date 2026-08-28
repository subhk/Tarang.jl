"""
Unit tests for TransposableField

Tests construction, local shape computation, pack/unpack operations,
and round-trip accuracy for the TransposableField pattern.
"""

using Test
using Tarang
using MPI

# This file mixes rank-agnostic unit tests with rank-specific distributed
# construction tests. Each distributed testset is guarded by the communicator
# size so the file runs cleanly at 1, 2, or 4 ranks (CI exercises all three via
# test/run_mpi_ci.jl):
#   * serial / mesh=(1,)          → NPROCS == 1
#   * 1D decomposition mesh=(2,)  → NPROCS == 2
#   * 2D decomposition mesh=(2,2) → NPROCS == 4
MPI.Initialized() || MPI.Init()
const NPROCS = MPI.Comm_size(MPI.COMM_WORLD)

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

if NPROCS == 1
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

        # With one rank, the wrapped field is authoritative; transpose buffers
        # are unnecessary scratch storage and may have a different spectral shape.
        @test active_layout(tf) == ZLocal
        data = Tarang.current_data(tf)
        @test data === Tarang.get_grid_data(field)
        @test data == field["g"]
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
end  # if NPROCS == 1

if NPROCS == 1
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
end  # if NPROCS == 1

if NPROCS == 1
@testset "TransposableField Serial Transforms" begin

    function serial_transform_field(name)
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())
        xbasis = Fourier(coords, "x", 16)
        ybasis = Fourier(coords, "y", 16)
        field = ScalarField(dist, name, (xbasis, ybasis))
        x = range(0, 2π; length=17)[1:16]
        y = range(0, 2π; length=17)[1:16]
        for i in 1:16, j in 1:16
            field["g"][i, j] = sin(x[i]) * cos(y[j])
        end
        return field
    end

    @testset "Serial short-circuit delegates to the field's own transform" begin
        # At one rank there is no transpose to perform and the field's regular
        # transform path owns the basis-specific shapes (a RealFourier half
        # spectrum has no representation in the fixed-shape transpose buffers).
        # This asserts the DELEGATION — that the wrapper leaves the field
        # authoritative — which is all the serial path claims. The claim that the
        # distributed path reproduces serial COEFFICIENTS is an np>1 statement and
        # lives in test_mpi_transposable_parity.jl.
        field = serial_transform_field("transform_forward")
        reference = serial_transform_field("transform_forward_reference")
        tf = TransposableField(field)
        forward_transform!(reference)
        distributed_forward_transform!(tf)
        @test field.current_layout == :c
        @test field["c"] ≈ reference["c"]
        @test Tarang.current_data(tf) === Tarang.get_coeff_data(field)
    end

    @testset "Round-trip transform" begin
        field = serial_transform_field("transform_roundtrip")
        tf = TransposableField(field)
        original = copy(field["g"])
        distributed_forward_transform!(tf)
        distributed_backward_transform!(tf)
        @test field.current_layout == :g
        @test field["g"] ≈ original rtol=1e-10
        @test Tarang.current_data(tf) === Tarang.get_grid_data(field)
        @test Tarang.current_data(tf) ≈ original rtol=1e-10
    end

    @testset "Round-trip with overlap flag" begin
        field = serial_transform_field("transform_overlap")
        tf = TransposableField(field)
        original = copy(field["g"])
        distributed_forward_transform!(tf; overlap=true)
        distributed_backward_transform!(tf; overlap=false)
        @test field["g"] ≈ original rtol=1e-10
    end

    @testset "Performance statistics" begin
        field = serial_transform_field("transform_stats")
        tf = TransposableField(field)
        reset_transpose_stats!(tf)
        distributed_forward_transform!(tf)
        distributed_backward_transform!(tf)
        stats = get_transpose_stats(tf)
        @test stats.num_transposes == 0
        @test stats.total_fft_time >= 0.0
        @test stats.total_pack_time == 0.0
        @test stats.total_unpack_time == 0.0
    end

end
end  # if NPROCS == 1

if NPROCS > 1
@testset "TransposableField distributed 2D public transform API" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; comm=MPI.COMM_WORLD, mesh=(NPROCS,), dtype=Float64,
                       architecture=CPU(), use_pencil_arrays=false)
    bases = (
        ComplexFourier(coords, "x", 17),
        ComplexFourier(coords, "y", 13),
    )
    field = ScalarField(dist, "distributed_2d_roundtrip", bases)

    grid = Tarang.get_grid_data(field)
    for j in axes(grid, 2), i in axes(grid, 1)
        grid[i, j] = NPROCS * dist.rank + 0.1i + 0.01j
    end
    original = copy(grid)

    tf = TransposableField(field)
    distributed_forward_transform!(tf)
    @test field.current_layout == :c

    distributed_backward_transform!(tf)
    @test field.current_layout == :g
    @test isapprox(Tarang.get_grid_data(field), original; rtol=1e-12, atol=1e-12)
end

@testset "transpose workspace is cached per shape, not per field" begin
    # TransposableField's constructor performs two collective MPI.Comm_splits.
    # One workspace per FIELD would burn 2*nfields communicators; two fields of
    # the same shape must share one.
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; comm=MPI.COMM_WORLD, mesh=(NPROCS,),
                       dtype=ComplexF64, architecture=CPU(), use_pencil_arrays=false)
    bases = (ComplexFourier(coords, "x", 8), ComplexFourier(coords, "y", 6))
    a = ScalarField(dist, "ws_a", bases)
    b = ScalarField(dist, "ws_b", bases)

    wa = Tarang.transpose_workspace!(dist, a)
    wb = Tarang.transpose_workspace!(dist, b)
    @test wa === wb

    # A different shape gets its own workspace.
    other = (ComplexFourier(coords, "x", 16), ComplexFourier(coords, "y", 6))
    c = ScalarField(dist, "ws_c", other)
    @test Tarang.transpose_workspace!(dist, c) !== wa
end

@testset "CPU fields do not select transposable storage, even under MPI (np=$NPROCS)" begin
    # RENAMED from "distributed GPU fields select transposable storage": this
    # testset only ever built a CPU-architecture field and checked the negative
    # case (_uses_transpose_storage(CPU(), n) == false, via is_transposable_storage
    # on the resulting field) — it never touched a GPU architecture, so the old
    # name overclaimed what was actually verified.
    #
    # The GPU-positive case — _uses_transpose_storage(GPU, n>1) == true — cannot
    # be exercised through the real ScalarField constructor on this box:
    # _build_field_arrays runs BEFORE storage selection and needs
    # array_type(::GPU, T), which errors without the CUDA extension loaded (see
    # the "synthetic device-storage field dispatches..." testset in
    # test_mpi_transposable_parity.jl for the same constraint, and the
    # explicit-storage constructor it uses to work around it). The predicate
    # itself is pinned directly — dispatch only, no array allocation — for both
    # GPU and CPU, in test_decomposition_convention.jl's "_uses_transpose_storage"
    # testset; that is where the actual GPU-positive assertion lives.
    #
    # TransposableFieldStorage has been defined, documented and dispatched since
    # the wrapper was split up — but nothing ever constructed it, so every field
    # got SerialFieldStorage and the distributed transform stayed unreachable
    # from the ordinary API.
    coords = CartesianCoordinates("x", "y")

    cpu_dist = Distributor(coords; comm=MPI.COMM_WORLD, mesh=(NPROCS,),
                           dtype=ComplexF64, architecture=CPU(),
                           use_pencil_arrays=false)
    bases = (ComplexFourier(coords, "x", 8), ComplexFourier(coords, "y", 6))
    cpu_field = ScalarField(cpu_dist, "storage_cpu", bases)
    @test !Tarang.is_transposable_storage(cpu_field)
end
end  # if NPROCS > 1

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

@testset "Generic GPU Pack/Unpack Refuses CPU Staging" begin
    data = reshape(collect(1.0:8.0), 2, 2, 2)
    buffer = similar(vec(data))
    counts = [length(data)]
    displs = [0]
    gpu = Tarang.GPU(:mock)

    @test_throws ErrorException Tarang.pack_for_transpose!(
        buffer, data, counts, displs, 3, 1, gpu)
    @test_throws ErrorException Tarang.unpack_from_transpose!(
        data, buffer, counts, displs, 3, 1, gpu)
end

@testset "compute_local_shapes" begin

    if NPROCS == 2
        @testset "2D 1D decomposition" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(2,), dtype=Float64, architecture=CPU())

            global_shape = (16, 16)
            shapes = Tarang.compute_local_shapes(global_shape, dist)

            @test haskey(shapes, YLocal)
            @test haskey(shapes, XLocal)
        end
    end

    if NPROCS == 4
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

if _HAS_CUDA && NPROCS == 1
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
        @test_skip "GPU TransposableField requires CUDA at 1 rank"
    end
end
