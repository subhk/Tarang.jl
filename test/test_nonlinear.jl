"""
Test suite for nonlinear.jl

Tests:
1. NonlinearPerformanceStats creation
2. NonlinearOperator types (AdvectionOperator, NonlinearAdvectionOperator, ConvectiveOperator)
3. NonlinearEvaluator creation and setup
4. Transform setup functions (1D, 2D, 3D FFTW plans)
5. get_nonlinear_transform function
6. Dealiasing functions
7. PencilArray compatibility utilities
8. evaluate_transform_multiply
9. Convenience constructors
10. compute_local_shape, compute_local_range
"""

using Test

@testset "Nonlinear Terms Module" begin
    using Tarang

    @testset "NonlinearPerformanceStats" begin
        stats = NonlinearPerformanceStats()
        @test isa(stats, NonlinearPerformanceStats)
        @test stats.total_evaluations == 0
        @test stats.total_time == 0.0
        @test stats.dealiasing_time == 0.0
        @test stats.transform_time == 0.0

        # Test mutability
        stats.total_evaluations = 10
        stats.total_time = 1.5
        @test stats.total_evaluations == 10
        @test stats.total_time == 1.5
    end

    @testset "NonlinearEvaluator Construction" begin
        # Create a simple distributor for testing
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)

        # Create NonlinearEvaluator
        evaluator = NonlinearEvaluator(dist)
        @test isa(evaluator, NonlinearEvaluator)
        @test evaluator.dealiasing_factor == 1.5  # Default 3/2 rule
        @test isa(evaluator.pencil_transforms, Dict{String, Any})
        @test isa(evaluator.temp_fields, Dict{String, Any})
        @test isa(evaluator.performance_stats, NonlinearPerformanceStats)

        # Test custom dealiasing factor
        evaluator2 = NonlinearEvaluator(dist; dealiasing_factor=2.0)
        @test evaluator2.dealiasing_factor == 2.0
    end

    @testset "Transform Setup Functions" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        evaluator = NonlinearEvaluator(dist)

        @testset "1D FFTW Plans" begin
            # Check that 1D plans were created
            Tarang.setup_1d_fftw_plans!(evaluator, 32)
            @test haskey(evaluator.pencil_transforms, "1d_32")

            transform = evaluator.pencil_transforms["1d_32"]
            @test transform["type"] == :fftw_1d
            @test transform["size"] == 32
            @test transform["dealiased_size"] == ceil(Int, 32 * 1.5)
            @test haskey(transform, "forward_plan")
            @test haskey(transform, "backward_plan")
            @test haskey(transform, "scratch_real")
            @test haskey(transform, "scratch_complex")
        end

        @testset "2D FFTW Plans" begin
            Tarang.setup_2d_fftw_plans!(evaluator, (64, 32))
            @test haskey(evaluator.pencil_transforms, "2d_64x32")

            transform = evaluator.pencil_transforms["2d_64x32"]
            @test transform["type"] == :fftw_2d
            @test transform["shape"] == (64, 32)
            @test transform["dealiased_shape"] == (ceil(Int, 64 * 1.5), ceil(Int, 32 * 1.5))
        end

        @testset "3D FFTW Plans" begin
            Tarang.setup_3d_fftw_plans!(evaluator, (32, 32, 16))
            @test haskey(evaluator.pencil_transforms, "3d_32x32x16")

            transform = evaluator.pencil_transforms["3d_32x32x16"]
            @test transform["type"] == :fftw_3d
            @test transform["shape"] == (32, 32, 16)
        end
    end

    @testset "get_nonlinear_transform" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        evaluator = NonlinearEvaluator(dist)

        # Test 1D lookup
        Tarang.setup_1d_fftw_plans!(evaluator, 64)
        transform = Tarang.get_nonlinear_transform(evaluator, (64,))
        @test transform !== nothing
        @test transform["size"] == 64

        # Test 2D lookup
        Tarang.setup_2d_fftw_plans!(evaluator, (128, 64))
        transform = Tarang.get_nonlinear_transform(evaluator, (128, 64))
        @test transform !== nothing
        @test transform["shape"] == (128, 64)

        # Test 3D lookup
        Tarang.setup_3d_fftw_plans!(evaluator, (64, 64, 32))
        transform = Tarang.get_nonlinear_transform(evaluator, (64, 64, 32))
        @test transform !== nothing
        @test transform["shape"] == (64, 64, 32)

        # Test on-the-fly creation for unknown shape
        transform = Tarang.get_nonlinear_transform(evaluator, (48,))
        @test transform !== nothing
        @test haskey(evaluator.pencil_transforms, "1d_48")
    end

    @testset "Dealiasing Functions" begin
        @testset "get_dealiasing_cutoffs" begin
            # Test 1D cutoff
            cutoffs = Tarang.get_dealiasing_cutoffs((64,), 1.5)
            @test cutoffs == (42,)  # floor(64/1.5) = 42

            # Test 2D cutoffs
            cutoffs = Tarang.get_dealiasing_cutoffs((128, 64), 1.5)
            @test cutoffs == (85, 42)

            # Test 3D cutoffs
            cutoffs = Tarang.get_dealiasing_cutoffs((64, 64, 32), 1.5)
            @test cutoffs == (42, 42, 21)

            # Test with different dealiasing factor
            cutoffs = Tarang.get_dealiasing_cutoffs((64,), 2.0)
            @test cutoffs == (32,)
        end

        @testset "apply_1d_spectral_cutoff! (vector)" begin
            # Create test data
            data = ones(ComplexF64, 16)
            Tarang.apply_1d_spectral_cutoff!(data, 1, 4)

            # Check that low frequencies are preserved
            @test data[1] == 1.0  # k=0
            @test data[2] == 1.0  # k=1
            @test data[3] == 1.0  # k=2
            @test data[4] == 1.0  # k=3
            @test data[5] == 1.0  # k=4

            # Check that high frequencies are zeroed
            @test data[7] == 0.0  # k=6
            @test data[8] == 0.0  # k=7
        end

        @testset "apply_2d_spectral_cutoff!" begin
            # Create test data
            data = ones(ComplexF64, 16, 16)
            Tarang.apply_2d_spectral_cutoff!(data, (4, 4))

            # Check that corner (low frequency) is preserved
            @test data[1, 1] == 1.0  # kx=0, ky=0
            @test data[2, 2] == 1.0  # kx=1, ky=1

            # Check that high frequencies are zeroed
            @test data[8, 8] == 0.0  # High frequency corner
        end

        @testset "apply_3d_spectral_cutoff!" begin
            # Create test data
            data = ones(ComplexF64, 16, 16, 8)
            Tarang.apply_3d_spectral_cutoff!(data, (4, 4, 2))

            # Check that DC component is preserved
            @test data[1, 1, 1] == 1.0

            # Check that high frequencies are zeroed
            @test data[8, 8, 4] == 0.0
        end

        @testset "apply_spherical_spectral_cutoff!" begin
            # Create test data
            data = ones(ComplexF64, 8, 8)
            Tarang.apply_spherical_spectral_cutoff!(data, 2)

            # DC component should be preserved (|k|^2 = 0)
            @test data[1, 1] == 1.0

            # kx=1, ky=1 has |k|^2 = 2, should be preserved
            @test data[2, 2] == 1.0

            # kx=3, ky=0 has |k|^2 = 9 > 4, should be zeroed
            @test data[4, 1] == 0.0
        end
    end

    @testset "compute_local_shape" begin
        using MPI
        comm = MPI.COMM_WORLD

        # Single process case
        local_shape = Tarang.compute_local_shape((100, 100), (1, 1), comm)
        @test local_shape == (100, 100)

        # With 1D mesh (serial execution)
        local_shape = Tarang.compute_local_shape((128, 64), (1,), comm)
        @test local_shape == (128, 64)
    end

    @testset "compute_local_range" begin
        using MPI
        comm = MPI.COMM_WORLD

        # Single process case
        ranges = Tarang.compute_local_range((100, 100), (1, 1), comm)
        @test ranges[1] == (1, 100)
        @test ranges[2] == (1, 100)
    end

    @testset "is_shape_compatible" begin
        using MPI
        comm = MPI.COMM_WORLD

        # Serial case: local should match global
        @test Tarang.is_shape_compatible((64, 64), (64, 64), (1, 1), comm) == true
        @test Tarang.is_shape_compatible((64, 32), (64, 64), (1, 1), comm) == false

        # Dimension mismatch
        @test Tarang.is_shape_compatible((64,), (64, 64), (1, 1), comm) == false
    end

    @testset "NonlinearOperator Types" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
        domain = Domain(dist, (xb, yb))

        # Create test fields
        u = ScalarField(dist, "u", (xb, yb), Float64)
        v = ScalarField(dist, "v", (xb, yb), Float64)

        velocity = VectorField(dist, coords, "vel", (xb, yb), Float64)

        @testset "AdvectionOperator" begin
            op = AdvectionOperator(velocity, u)
            @test isa(op, AdvectionOperator)
            @test isa(op, NonlinearOperator)
            @test op.velocity === velocity
            @test op.scalar === u
            @test op.name == "advection"

            # Test with custom name
            op2 = AdvectionOperator(velocity, u, "custom_adv")
            @test op2.name == "custom_adv"
        end

        @testset "NonlinearAdvectionOperator" begin
            op = NonlinearAdvectionOperator(velocity)
            @test isa(op, NonlinearAdvectionOperator)
            @test isa(op, NonlinearOperator)
            @test op.velocity === velocity
            @test op.name == "nonlinear_advection"
        end

        @testset "ConvectiveOperator" begin
            op = ConvectiveOperator(u, v, :multiply)
            @test isa(op, ConvectiveOperator)
            @test isa(op, NonlinearOperator)
            @test op.field1 === u
            @test op.field2 === v
            @test op.operation == :multiply

            # Test other operations
            op_dot = ConvectiveOperator(velocity, velocity, :dot_product)
            @test op_dot.operation == :dot_product

            op_cross = ConvectiveOperator(velocity, velocity, :cross_product)
            @test op_cross.operation == :cross_product
        end
    end

    @testset "Convenience Constructors" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        φ = ScalarField(dist, "phi", (xb, yb), Float64)
        vel = VectorField(dist, coords, "u", (xb, yb), Float64)

        @testset "advection" begin
            op = advection(vel, φ)
            @test isa(op, AdvectionOperator)
            @test op.velocity === vel
            @test op.scalar === φ
        end

        @testset "nonlinear_momentum" begin
            op = nonlinear_momentum(vel)
            @test isa(op, NonlinearAdvectionOperator)
            @test op.velocity === vel
        end

        @testset "convection" begin
            op = convection(φ, φ, :multiply)
            @test isa(op, ConvectiveOperator)
            @test op.operation == :multiply
        end
    end

    @testset "Memory Management" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        evaluator = NonlinearEvaluator(dist)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))

        template = ScalarField(dist, "template", (xb, yb), Float64)

        @testset "get_temp_field" begin
            temp1 = Tarang.get_temp_field(evaluator, template, "temp1")
            @test isa(temp1, ScalarField)

            # Same key should return same field
            temp1_again = Tarang.get_temp_field(evaluator, template, "temp1")
            @test temp1 === temp1_again

            # Different key should return different field
            temp2 = Tarang.get_temp_field(evaluator, template, "temp2")
            @test temp2 !== temp1
        end

        @testset "clear_temp_fields!" begin
            # Add some temp fields
            Tarang.get_temp_field(evaluator, template, "to_clear1")
            Tarang.get_temp_field(evaluator, template, "to_clear2")
            @test !isempty(evaluator.temp_fields)

            Tarang.clear_temp_fields!(evaluator)
            @test isempty(evaluator.temp_fields)
        end

        @testset "get_temp_array" begin
            arr = Tarang.get_temp_array(evaluator, (10, 10), Float64)
            @test size(arr) == (10, 10)
            @test eltype(arr) == Float64
            @test all(arr .== 0.0)
        end

        @testset "return_temp_array!" begin
            arr = zeros(Float64, 10, 10)
            result = Tarang.return_temp_array!(evaluator, arr)
            @test result === nothing  # No-op for CPU
        end
    end

    @testset "evaluate_transform_multiply" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
        domain = Domain(dist, (xb, yb))

        evaluator = NonlinearEvaluator(dist)

        # Create two test fields
        f1 = ScalarField(dist, "f1", (xb, yb), Float64)
        f2 = ScalarField(dist, "f2", (xb, yb), Float64)

        # Initialize with test data
        ensure_layout!(f1, :g)
        ensure_layout!(f2, :g)
        fill!(f1["g"], 2.0)
        fill!(f2["g"], 3.0)

        # Test multiplication
        result = Tarang.evaluate_transform_multiply(f1, f2, evaluator)

        @test isa(result, ScalarField)
        ensure_layout!(result, :g)

        # The result should be approximately 2.0 * 3.0 = 6.0
        # (may have small deviations due to transforms)
        @test all(abs.(result["g"] .- 6.0) .< 1e-10)

        # Check performance stats updated
        @test evaluator.performance_stats.total_evaluations >= 1
    end

    @testset "Field Multiplication with Different Values" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))
        domain = Domain(dist, (xb,))

        evaluator = NonlinearEvaluator(dist)

        # Create fields with sinusoidal patterns
        f1 = ScalarField(dist, "f1", (xb,), Float64)
        f2 = ScalarField(dist, "f2", (xb,), Float64)

        ensure_layout!(f1, :g)
        ensure_layout!(f2, :g)

        # Set up a simple test: f1 = sin(x), f2 = cos(x)
        x_grid = range(0, 2π, length=32+1)[1:32]
        for (i, x) in enumerate(x_grid)
            f1["g"][i] = sin(x)
            f2["g"][i] = cos(x)
        end

        # Test multiplication
        result = Tarang.evaluate_transform_multiply(f1, f2, evaluator)
        ensure_layout!(result, :g)

        # Check that result ≈ sin(x) * cos(x) = 0.5 * sin(2x)
        for (i, x) in enumerate(x_grid)
            expected = sin(x) * cos(x)
            @test abs(result["g"][i] - expected) < 1e-10
        end
    end

    @testset "Vector Operations" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(1, 1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        zb = RealFourier(coords["z"]; size=8, bounds=(0.0, 2π))
        domain = Domain(dist, (xb, yb, zb))

        # Create 3D vector fields
        v1 = VectorField(dist, coords, "v1", (xb, yb, zb), Float64)
        v2 = VectorField(dist, coords, "v2", (xb, yb, zb), Float64)

        # Initialize with constant values
        for comp in v1.components
            ensure_layout!(comp, :g)
            fill!(comp["g"], 1.0)
        end
        for (i, comp) in enumerate(v2.components)
            ensure_layout!(comp, :g)
            fill!(comp["g"], Float64(i))  # 1.0, 2.0, 3.0
        end

        @testset "evaluate_vector_dot_product" begin
            result = Tarang.evaluate_vector_dot_product(v1, v2)
            @test isa(result, ScalarField)
            ensure_layout!(result, :g)

            # v1 = (1, 1, 1), v2 = (1, 2, 3)
            # v1 · v2 = 1*1 + 1*2 + 1*3 = 6
            @test all(abs.(result["g"] .- 6.0) .< 1e-10)
        end

        @testset "evaluate_vector_cross_product" begin
            # Create simpler vectors for cross product test
            a = VectorField(dist, coords, "a", (xb, yb, zb), Float64)
            b = VectorField(dist, coords, "b", (xb, yb, zb), Float64)

            # a = (1, 0, 0), b = (0, 1, 0) => a × b = (0, 0, 1)
            for comp in a.components
                ensure_layout!(comp, :g)
                fill!(comp["g"], 0.0)
            end
            fill!(a.components[1]["g"], 1.0)  # a_x = 1

            for comp in b.components
                ensure_layout!(comp, :g)
                fill!(comp["g"], 0.0)
            end
            fill!(b.components[2]["g"], 1.0)  # b_y = 1

            result = Tarang.evaluate_vector_cross_product(a, b)
            @test isa(result, VectorField)
            @test length(result.components) == 3

            for comp in result.components
                ensure_layout!(comp, :g)
            end

            # Check: a × b = (0, 0, 1)
            @test all(abs.(result.components[1]["g"]) .< 1e-10)  # x = 0
            @test all(abs.(result.components[2]["g"]) .< 1e-10)  # y = 0
            @test all(abs.(result.components[3]["g"] .- 1.0) .< 1e-10)  # z = 1
        end
    end

    @testset "Performance Stats Logging" begin
        stats = NonlinearPerformanceStats()
        stats.total_evaluations = 100
        stats.total_time = 5.0
        stats.dealiasing_time = 1.0
        stats.transform_time = 2.0

        # Just verify the function runs without error
        @test begin
            Tarang.log_nonlinear_performance(stats)
            true
        end
    end

    @testset "Interpolate Field Data" begin
        # Test same shape (copy)
        src = [1.0, 2.0, 3.0, 4.0]
        dest = zeros(4)
        Tarang.interpolate_field_data!(dest, src)
        @test dest == src

        # Test different shapes (nearest neighbor interpolation)
        src = [1.0, 2.0, 3.0, 4.0]
        dest = zeros(8)
        Tarang.interpolate_field_data!(dest, src)
        @test dest[1] == 1.0  # First element matches
        @test dest[8] == 4.0  # Last element matches
    end

    @testset "PencilConfig Compatibility" begin
        using MPI
        comm = MPI.COMM_WORLD
        config = Tarang.PencilConfig((64, 64), (1, 1); comm=comm, dtype=Float64)

        @test config.global_shape == (64, 64)
        @test config.mesh == (1, 1)
        @test config.dtype == Float64
    end
end

println("All nonlinear terms tests passed!")
