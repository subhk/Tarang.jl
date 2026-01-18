"""
Test suite for general.jl

Tests:
1. CPU Device Configuration
2. OrderedSet implementation
3. DeferredTuple implementation
4. Attribute unification functions
5. Type checking utilities
6. Array utilities (apply_along_axis, axis operations)
7. Time/Numerical utilities
8. Dictionary utilities
9. Progress tracking
10. Validation utilities
11. Memory utilities
12. Logging utilities
"""

using Test

@testset "General Utilities Module" begin
    using Tarang
    # Don't import OrderedCollections directly to avoid name conflict with Tarang.OrderedSet

    @testset "CPU Device Configuration" begin
        @testset "CPUDeviceConfig struct" begin
            config = CPUDeviceConfig()
            @test config.device_type == :cpu
            @test isa(config, CPUDeviceConfig)

            config2 = CPUDeviceConfig(:cpu)
            @test config2.device_type == :cpu
        end

        @testset "DEFAULT_DEVICE" begin
            @test isa(DEFAULT_DEVICE, CPUDeviceConfig)
            @test DEFAULT_DEVICE.device_type == :cpu
            @test device_config === DEFAULT_DEVICE
        end

        @testset "device_zeros" begin
            arr = device_zeros(Float64, (3, 4))
            @test size(arr) == (3, 4)
            @test eltype(arr) == Float64
            @test all(arr .== 0.0)

            arr2 = device_zeros(ComplexF64, (2, 2), CPUDeviceConfig())
            @test eltype(arr2) == ComplexF64
        end

        @testset "device_ones" begin
            arr = device_ones(Float64, (2, 3))
            @test size(arr) == (2, 3)
            @test all(arr .== 1.0)

            arr2 = device_ones(Int64, (5,))
            @test all(arr2 .== 1)
        end

        @testset "device_fill" begin
            arr = device_fill(5.0, (2, 2))
            @test all(arr .== 5.0)

            arr2 = device_fill(3, (3, 3), CPUDeviceConfig())
            @test all(arr2 .== 3)
        end

        @testset "device_similar" begin
            original = ones(3, 4)
            similar_arr = device_similar(original)
            @test size(similar_arr) == size(original)
            @test eltype(similar_arr) == eltype(original)
        end

        @testset "move_to_device and move_to_host" begin
            arr = [1.0, 2.0, 3.0]
            # On CPU, these should be no-ops
            @test move_to_device(arr) === arr
            @test move_to_host(arr) === arr
        end

        @testset "check_device_compatibility" begin
            arr = ones(5, 5)
            @test check_device_compatibility(arr) == true
            @test check_device_compatibility(arr, CPUDeviceConfig()) == true
        end

        @testset "default_memory_info" begin
            info = default_memory_info()
            @test haskey(info, :total)
            @test haskey(info, :available)
            @test haskey(info, :used)
            @test info.total == typemax(Int64)
        end
    end

    @testset "OrderedSet" begin
        @testset "Empty OrderedSet" begin
            os = Tarang.OrderedSet()
            @test length(os) == 0
            @test collect(os) == []
        end

        @testset "OrderedSet from items" begin
            os = Tarang.OrderedSet([1, 2, 3])
            @test length(os) == 3
            @test 1 in os
            @test 2 in os
            @test 3 in os
            @test !(4 in os)
        end

        @testset "OrderedSet preserves order" begin
            os = Tarang.OrderedSet([3, 1, 4, 1, 5])
            collected = collect(os)
            @test collected[1] == 3
            @test collected[2] == 1
            @test collected[3] == 4
            @test collected[4] == 5
            @test length(collected) == 4  # Duplicates removed
        end

        @testset "OrderedSet with typed elements" begin
            os = Tarang.OrderedSet{Int}([1, 2, 3])
            @test length(os) == 3

            os2 = Tarang.OrderedSet{String}(["a", "b", "c"])
            @test "a" in os2
        end

        @testset "push! to OrderedSet" begin
            os = Tarang.OrderedSet{Int}()
            push!(os, 1)
            push!(os, 2)
            push!(os, 1)  # Duplicate
            @test length(os) == 2
            @test collect(os) == [1, 2]
        end

        @testset "iterate over OrderedSet" begin
            os = Tarang.OrderedSet([10, 20, 30])
            total = 0
            for item in os
                total += item
            end
            @test total == 60
        end

        @testset "OrderedSet from varargs" begin
            os = Tarang.OrderedSet(1, 2, 3)
            @test length(os) == 3
            @test collect(os) == [1, 2, 3]
        end
    end

    @testset "DeferredTuple" begin
        @testset "Empty DeferredTuple" begin
            dt = DeferredTuple()
            @test length(dt) == 0
        end

        @testset "DeferredTuple from items" begin
            dt = DeferredTuple(1, 2, 3)
            @test length(dt) == 3
            @test dt[1] == 1
            @test dt[2] == 2
            @test dt[3] == 3
        end

        @testset "iterate over DeferredTuple" begin
            dt = DeferredTuple(10, 20, 30)
            total = 0
            for item in dt
                total += item
            end
            @test total == 60
        end

        @testset "DeferredTuple with mixed types" begin
            dt = DeferredTuple(1, 2.0, 3)
            @test length(dt) == 3
            # Types should be promoted
            @test dt[1] == 1.0
            @test dt[2] == 2.0
        end
    end

    @testset "Attribute Unification" begin
        @testset "unify_attributes" begin
            # Create test structs
            struct TestObj
                value::Int
            end

            obj1 = TestObj(5)
            obj2 = TestObj(5)
            obj3 = TestObj(5)

            result = Tarang.unify_attributes([obj1, obj2, obj3], "value")
            @test result == 5
        end

        @testset "unify_attributes with inconsistent values" begin
            struct TestObj2
                value::Int
            end

            obj1 = TestObj2(5)
            obj2 = TestObj2(10)

            @test_throws ArgumentError Tarang.unify_attributes([obj1, obj2], "value")
        end

        @testset "unify_attributes with missing field" begin
            struct TestObj3
                other::Int
            end

            obj = TestObj3(5)
            result = Tarang.unify_attributes([obj], "nonexistent")
            @test result === nothing
        end

        @testset "unify" begin
            @test Tarang.unify([1, 1, 1]) == 1
            @test Tarang.unify(["a", "a"]) == "a"
            @test Tarang.unify([]) === nothing
        end

        @testset "unify with inconsistent objects" begin
            @test_throws ArgumentError Tarang.unify([1, 2, 3])
        end
    end

    @testset "Type Checking Utilities" begin
        @testset "is_complex_dtype" begin
            @test Tarang.is_complex_dtype(ComplexF64) == true
            @test Tarang.is_complex_dtype(ComplexF32) == true
            @test Tarang.is_complex_dtype(Complex{Int}) == true
            @test Tarang.is_complex_dtype(Float64) == false
            @test Tarang.is_complex_dtype(Int64) == false
        end

        @testset "is_real_dtype" begin
            @test Tarang.is_real_dtype(Float64) == true
            @test Tarang.is_real_dtype(Float32) == true
            @test Tarang.is_real_dtype(Int64) == true
            @test Tarang.is_real_dtype(ComplexF64) == false
        end

        @testset "promote_dtype" begin
            @test Tarang.promote_dtype(Float32, Float64) == Float64
            @test Tarang.promote_dtype(Int32, Int64) == Int64
            @test Tarang.promote_dtype(Float64, ComplexF64) == ComplexF64
        end
    end

    @testset "Array Utilities" begin
        @testset "apply_along_axis with sum" begin
            arr = [1 2 3; 4 5 6]  # 2x3 array

            # Sum along axis 1 (rows)
            result1 = Tarang.apply_along_axis(sum, 1, arr)
            @test result1 == [5, 7, 9]  # Column sums

            # Sum along axis 2 (columns)
            result2 = Tarang.apply_along_axis(sum, 2, arr)
            @test result2 == [6, 15]  # Row sums
        end

        @testset "apply_along_axis with sort" begin
            arr = [3 1 2; 6 4 5]

            # Sort along axis 1
            result1 = Tarang.apply_along_axis(sort, 1, arr)
            @test result1[1, :] == [3, 1, 2]  # First row unchanged
            @test result1[2, :] == [6, 4, 5]  # Second row unchanged

            # Sort along axis 2
            result2 = Tarang.apply_along_axis(sort, 2, arr)
            @test result2[1, :] == [1, 2, 3]
            @test result2[2, :] == [4, 5, 6]
        end

        @testset "apply_along_axis invalid axis" begin
            arr = ones(3, 4)
            @test_throws ArgumentError Tarang.apply_along_axis(sum, 0, arr)
            @test_throws ArgumentError Tarang.apply_along_axis(sum, 3, arr)
        end

        @testset "axis_sum" begin
            arr = [1 2; 3 4]
            @test Tarang.axis_sum(arr, 1) == [4, 6]
            @test Tarang.axis_sum(arr, 2) == [3, 7]
        end

        @testset "axis_mean" begin
            arr = [1.0 2.0; 3.0 4.0]
            @test Tarang.axis_mean(arr, 1) == [2.0, 3.0]
            @test Tarang.axis_mean(arr, 2) == [1.5, 3.5]
        end

        @testset "axis_maximum" begin
            arr = [1 5 3; 2 4 6]
            @test Tarang.axis_maximum(arr, 1) == [2, 5, 6]
            @test Tarang.axis_maximum(arr, 2) == [5, 6]
        end

        @testset "axis_minimum" begin
            arr = [1 5 3; 2 4 6]
            @test Tarang.axis_minimum(arr, 1) == [1, 4, 3]
            @test Tarang.axis_minimum(arr, 2) == [1, 2]
        end

        @testset "axis_sort" begin
            arr = [3 1 2; 6 4 5]
            result = Tarang.axis_sort(arr, 2)
            @test result[1, :] == [1, 2, 3]
            @test result[2, :] == [4, 5, 6]
        end

        @testset "apply_along_axis_fast" begin
            arr = [1 2 3; 4 5 6]
            result = Tarang.apply_along_axis_fast(sum, 1, arr)
            @test vec(result) == [5, 7, 9]

            @test_throws ArgumentError Tarang.apply_along_axis_fast(sum, 0, arr)
        end

        @testset "broadcast_to_shape" begin
            arr = [1, 2, 3]
            result = Tarang.broadcast_to_shape(arr, (3, 4))
            @test size(result) == (3, 4)
            @test result[:, 1] == [1, 2, 3]
            @test result[:, 4] == [1, 2, 3]
        end

        @testset "broadcast_to_shape incompatible" begin
            arr = ones(3, 4)
            # DimensionMismatch is thrown by Base.Broadcast.broadcast_shape
            @test_throws DimensionMismatch Tarang.broadcast_to_shape(arr, (2, 4))
        end

        @testset "apply_along_axis 1D array" begin
            arr = [1, 2, 3, 4, 5]
            result = Tarang.apply_along_axis(sum, 1, arr)
            # Result is a 0-dimensional array containing 15
            @test result[] == 15
        end

        @testset "apply_along_axis 3D array" begin
            arr = ones(2, 3, 4)
            result = Tarang.apply_along_axis(sum, 2, arr)
            @test size(result) == (2, 4)
            @test all(result .== 3.0)
        end
    end

    @testset "Time/Numerical Utilities" begin
        @testset "format_time seconds" begin
            @test Tarang.format_time(5.0) == "5.0s"
            @test Tarang.format_time(30.5) == "30.5s"
            @test Tarang.format_time(59.99) == "59.99s"
        end

        @testset "format_time minutes" begin
            @test Tarang.format_time(60.0) == "1m 0.0s"
            @test Tarang.format_time(90.0) == "1m 30.0s"
            @test Tarang.format_time(3599.0) == "59m 59.0s"
        end

        @testset "format_time hours" begin
            @test Tarang.format_time(3600.0) == "1h 0m 0.0s"
            @test Tarang.format_time(7200.0) == "2h 0m 0.0s"
            @test Tarang.format_time(3661.0) == "1h 1m 1.0s"
        end

        @testset "safe_divide" begin
            @test Tarang.safe_divide(10, 2) == 5.0
            @test Tarang.safe_divide(10, 0) == 0.0
            @test Tarang.safe_divide(10, 0; default=Inf) == Inf
            @test Tarang.safe_divide(1, 0; default=-1.0) == -1.0
        end

        @testset "clamp_to_range" begin
            @test Tarang.clamp_to_range(5, 0, 10) == 5
            @test Tarang.clamp_to_range(-5, 0, 10) == 0
            @test Tarang.clamp_to_range(15, 0, 10) == 10
            @test Tarang.clamp_to_range(0.5, 0.0, 1.0) == 0.5
        end
    end

    @testset "Dictionary Utilities" begin
        @testset "deep_merge!" begin
            dict1 = Dict("a" => 1, "b" => Dict("c" => 2))
            dict2 = Dict("b" => Dict("d" => 3), "e" => 4)

            Tarang.deep_merge!(dict1, dict2)

            @test dict1["a"] == 1
            @test dict1["b"]["c"] == 2
            @test dict1["b"]["d"] == 3
            @test dict1["e"] == 4
        end

        @testset "deep_merge! overwrite" begin
            dict1 = Dict("a" => 1)
            dict2 = Dict("a" => 2)

            Tarang.deep_merge!(dict1, dict2)
            @test dict1["a"] == 2
        end

        @testset "flatten_dict" begin
            nested = Dict("a" => 1, "b" => Dict("c" => 2, "d" => 3))
            flat = Tarang.flatten_dict(nested)

            @test flat["a"] == 1
            @test flat["b.c"] == 2
            @test flat["b.d"] == 3
        end

        @testset "flatten_dict custom separator" begin
            nested = Dict("x" => Dict("y" => 1))
            flat = Tarang.flatten_dict(nested, "", "/")

            @test flat["x/y"] == 1
        end

        @testset "flatten_dict deeply nested" begin
            nested = Dict("a" => Dict("b" => Dict("c" => Dict("d" => 1))))
            flat = Tarang.flatten_dict(nested)

            @test flat["a.b.c.d"] == 1
        end
    end

    @testset "Progress Tracking" begin
        @testset "ProgressTracker creation" begin
            tracker = ProgressTracker(100)
            @test tracker.total == 100
            @test tracker.current == 0
            @test tracker.update_interval == 1.0
        end

        @testset "ProgressTracker custom interval" begin
            tracker = ProgressTracker(50; update_interval=0.5)
            @test tracker.update_interval == 0.5
        end

        @testset "ProgressTracker update!" begin
            tracker = ProgressTracker(10; update_interval=0.0)  # Always update

            Tarang.update!(tracker, 1)
            @test tracker.current == 1

            Tarang.update!(tracker, 5)
            @test tracker.current == 5

            Tarang.update!(tracker)  # Increment by 1
            @test tracker.current == 6
        end
    end

    @testset "Validation Utilities" begin
        @testset "validate_positive" begin
            @test Tarang.validate_positive(5, "value") == 5
            @test Tarang.validate_positive(0.1, "value") == 0.1

            @test_throws ArgumentError Tarang.validate_positive(0, "value")
            @test_throws ArgumentError Tarang.validate_positive(-1, "value")
        end

        @testset "validate_nonnegative" begin
            @test Tarang.validate_nonnegative(5, "value") == 5
            @test Tarang.validate_nonnegative(0, "value") == 0

            @test_throws ArgumentError Tarang.validate_nonnegative(-1, "value")
            @test_throws ArgumentError Tarang.validate_nonnegative(-0.1, "value")
        end

        @testset "validate_in_range" begin
            @test Tarang.validate_in_range(5, 0, 10, "value") == 5
            @test Tarang.validate_in_range(0, 0, 10, "value") == 0
            @test Tarang.validate_in_range(10, 0, 10, "value") == 10

            @test_throws ArgumentError Tarang.validate_in_range(-1, 0, 10, "value")
            @test_throws ArgumentError Tarang.validate_in_range(11, 0, 10, "value")
        end
    end

    @testset "Memory Utilities" begin
        @testset "get_memory_usage" begin
            info = Tarang.get_memory_usage()
            @test isa(info, Dict)
            @test haskey(info, "allocated")
            @test haskey(info, "freed")
            @test haskey(info, "total_time")
            @test info["total_time"] >= 0
        end
    end

    @testset "Logging Utilities" begin
        @testset "setup_logger levels" begin
            # Test that setup_logger doesn't throw for valid levels
            @test begin
                Tarang.setup_logger("DEBUG")
                true
            end

            @test begin
                Tarang.setup_logger("INFO")
                true
            end

            @test begin
                Tarang.setup_logger("WARN")
                true
            end

            @test begin
                Tarang.setup_logger("ERROR")
                true
            end

            # Unknown level defaults to INFO
            @test begin
                Tarang.setup_logger("UNKNOWN")
                true
            end
        end

        @testset "setup_logger to file" begin
            tmpfile = tempname()
            try
                Tarang.setup_logger("INFO", tmpfile)
                @info "Test log message"
                Tarang.close_logger()

                # File should exist and contain the message
                @test isfile(tmpfile)
                content = read(tmpfile, String)
                @test occursin("Test log message", content)
            finally
                isfile(tmpfile) && rm(tmpfile)
            end
        end

        @testset "close_logger" begin
            # Should not throw even when no file is open
            @test begin
                Tarang.close_logger()
                true
            end
        end
    end

    @testset "Edge Cases" begin
        @testset "OrderedSet with empty iterator" begin
            os = Tarang.OrderedSet(Int[])
            @test length(os) == 0
        end

        @testset "DeferredTuple single element" begin
            dt = DeferredTuple(42)
            @test length(dt) == 1
            @test dt[1] == 42
        end

        @testset "apply_along_axis with identity" begin
            arr = [1.0, 2.0, 3.0]
            result = Tarang.apply_along_axis(identity, 1, arr)
            @test result == arr
        end

        @testset "deep_merge! with empty dicts" begin
            dict1 = Dict{String, Any}()
            dict2 = Dict("a" => 1)
            Tarang.deep_merge!(dict1, dict2)
            @test dict1["a"] == 1

            dict3 = Dict("b" => 2)
            dict4 = Dict{String, Any}()
            Tarang.deep_merge!(dict3, dict4)
            @test dict3["b"] == 2
        end

        @testset "flatten_dict empty" begin
            empty_dict = Dict{String, Any}()
            flat = Tarang.flatten_dict(empty_dict)
            @test isempty(flat)
        end
    end
end

println("All general utility tests passed!")
