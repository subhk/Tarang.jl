"""
Test suite for analysis_tasks.jl

Tests the postprocessing wrappers for NetCDF handlers:
1. add_mean_task! - Spatial mean computation
2. add_slice_task! - Data slicing along dimensions
3. add_profile_task! - 1D profile extraction (mean over all but one dimension)
"""

using Test
using Statistics: mean

@testset "Analysis Tasks Module" begin
    using Tarang

    # ============================================================================
    # Test the postprocessing functions directly (isolated from handler)
    # ============================================================================

    @testset "Mean Postprocessing Logic" begin
        @testset "Global mean (dims=:)" begin
            # Test the postprocess function that add_mean_task! creates
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3 array

            # Simulate the postprocess function from add_mean_task! with dims=:
            post = data -> begin
                result = mean(data; dims=:)
                if (:) === Colon()
                    return result
                else
                    return dropdims(result, dims=:)
                end
            end

            result = post(data)
            @test result ≈ 3.5  # (1+2+3+4+5+6)/6 = 3.5
        end

        @testset "Mean along dimension 1" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3 array
            dims = 1

            # Simulate the postprocess function
            post = data -> begin
                result = mean(data; dims=dims)
                reduced = dropdims(result, dims=dims)
                if ndims(reduced) == 0
                    return reduced[]
                end
                return reduced
            end

            result = post(data)
            @test result ≈ [2.5, 3.5, 4.5]  # mean over rows
        end

        @testset "Mean along dimension 2" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3 array
            dims = 2

            post = data -> begin
                result = mean(data; dims=dims)
                reduced = dropdims(result, dims=dims)
                if ndims(reduced) == 0
                    return reduced[]
                end
                return reduced
            end

            result = post(data)
            @test result ≈ [2.0, 5.0]  # mean over columns
        end

        @testset "Mean of 3D data along dims (1,2)" begin
            data = reshape(collect(1.0:24.0), 2, 3, 4)
            dims = (1, 2)

            post = data -> begin
                result = mean(data; dims=dims)
                reduced = dropdims(result, dims=dims)
                if ndims(reduced) == 0
                    return reduced[]
                end
                return reduced
            end

            result = post(data)
            @test length(result) == 4  # Only dimension 3 remains
            # Mean of first 6 elements: (1+2+3+4+5+6)/6 = 3.5
            @test result[1] ≈ 3.5
        end

        @testset "Mean reduces to scalar" begin
            data = [1.0, 2.0, 3.0]  # 1D array
            dims = 1

            post = data -> begin
                result = mean(data; dims=dims)
                reduced = dropdims(result, dims=dims)
                if ndims(reduced) == 0
                    return reduced[]
                end
                return reduced
            end

            result = post(data)
            @test result ≈ 2.0
            @test result isa Float64  # Should be scalar
        end
    end

    @testset "Slice Postprocessing Logic" begin
        @testset "Slice 2D along dim=1 at idx=1" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3 array
            dim = 1
            idx = 1

            # Simulate the postprocess function from add_slice_task!
            post = data -> data |> x -> @views x[ntuple(i -> i == dim ? idx : :, ndims(x))...]

            result = post(data)
            @test result == [1.0, 2.0, 3.0]  # First row
        end

        @testset "Slice 2D along dim=1 at idx=2" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]
            dim = 1
            idx = 2

            post = data -> data |> x -> @views x[ntuple(i -> i == dim ? idx : :, ndims(x))...]

            result = post(data)
            @test result == [4.0, 5.0, 6.0]  # Second row
        end

        @testset "Slice 2D along dim=2 at idx=2" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]
            dim = 2
            idx = 2

            post = data -> data |> x -> @views x[ntuple(i -> i == dim ? idx : :, ndims(x))...]

            result = post(data)
            @test result == [2.0, 5.0]  # Second column
        end

        @testset "Slice 3D along dim=3 at idx=2" begin
            data = reshape(collect(1.0:24.0), 2, 3, 4)
            dim = 3
            idx = 2

            post = data -> data |> x -> @views x[ntuple(i -> i == dim ? idx : :, ndims(x))...]

            result = post(data)
            @test size(result) == (2, 3)  # 2D slice
            @test result[1, 1] == 7.0  # data[1,1,2]
        end

        @testset "Slice preserves view semantics" begin
            data = [1.0 2.0; 3.0 4.0]
            dim = 1
            idx = 1

            post = data -> data |> x -> @views x[ntuple(i -> i == dim ? idx : :, ndims(x))...]

            result = post(data)
            @test result isa SubArray  # Should be a view
        end
    end

    @testset "Profile Postprocessing Logic" begin
        @testset "Profile along dim=1 (mean over dim 2)" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3 array
            dim = 1

            # Simulate the postprocess function from add_profile_task!
            post = data -> begin
                dims_to_mean = setdiff(collect(1:ndims(data)), [dim])
                if isempty(dims_to_mean)
                    return data
                end
                dims_tuple = Tuple(dims_to_mean)
                dropdims(mean(data; dims=dims_tuple), dims=dims_tuple)
            end

            result = post(data)
            @test result ≈ [2.0, 5.0]  # Mean over columns for each row
        end

        @testset "Profile along dim=2 (mean over dim 1)" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3 array
            dim = 2

            post = data -> begin
                dims_to_mean = setdiff(collect(1:ndims(data)), [dim])
                if isempty(dims_to_mean)
                    return data
                end
                dims_tuple = Tuple(dims_to_mean)
                dropdims(mean(data; dims=dims_tuple), dims=dims_tuple)
            end

            result = post(data)
            @test result ≈ [2.5, 3.5, 4.5]  # Mean over rows for each column
        end

        @testset "Profile of 3D data along dim=3" begin
            data = reshape(collect(1.0:24.0), 2, 3, 4)
            dim = 3

            post = data -> begin
                dims_to_mean = setdiff(collect(1:ndims(data)), [dim])
                if isempty(dims_to_mean)
                    return data
                end
                dims_tuple = Tuple(dims_to_mean)
                dropdims(mean(data; dims=dims_tuple), dims=dims_tuple)
            end

            result = post(data)
            @test length(result) == 4  # Profile along 4th dimension
            # Mean of first slice [1:2, 1:3, 1] = mean(1:6) = 3.5
            @test result[1] ≈ 3.5
        end

        @testset "Profile of 1D data along dim=1 (edge case)" begin
            data = [1.0, 2.0, 3.0, 4.0, 5.0]
            dim = 1

            post = data -> begin
                dims_to_mean = setdiff(collect(1:ndims(data)), [dim])
                if isempty(dims_to_mean)
                    return data
                end
                dims_tuple = Tuple(dims_to_mean)
                dropdims(mean(data; dims=dims_tuple), dims=dims_tuple)
            end

            result = post(data)
            @test result == data  # No dimensions to average, return as-is
        end
    end

    # ============================================================================
    # Test mathematical properties
    # ============================================================================

    @testset "Mathematical Properties" begin
        @testset "Mean preserves sum/count relationship" begin
            data = rand(10, 20, 30)

            # Global mean
            global_mean = mean(data)
            @test global_mean ≈ sum(data) / length(data)

            # Mean along axis should preserve total
            mean_along_1 = mean(data; dims=1)
            @test sum(mean_along_1) * size(data, 1) ≈ sum(data)
        end

        @testset "Slice extracts correct hyperplane" begin
            # Create data with known values at each index
            data = zeros(4, 5, 6)
            for i in 1:4, j in 1:5, k in 1:6
                data[i, j, k] = i * 100 + j * 10 + k
            end

            # Slice at k=3 should have all values ending in 3
            dim = 3
            idx = 3
            post = x -> @views x[ntuple(i -> i == dim ? idx : :, ndims(x))...]
            slice = post(data)

            @test all(slice .% 10 .== 3)  # All values end in 3
        end

        @testset "Profile is idempotent" begin
            data = rand(8, 8)
            dim = 2

            post = data -> begin
                dims_to_mean = setdiff(collect(1:ndims(data)), [dim])
                if isempty(dims_to_mean)
                    return data
                end
                dims_tuple = Tuple(dims_to_mean)
                dropdims(mean(data; dims=dims_tuple), dims=dims_tuple)
            end

            profile1 = post(data)
            # Expand profile back to 2D by repeating
            profile_expanded = repeat(profile1', size(data, 1), 1)
            # Profile of profile_expanded should give same result
            profile2 = post(profile_expanded)

            @test profile1 ≈ profile2
        end

        @testset "Slice + Mean = Profile" begin
            data = rand(5, 6)
            dim_keep = 2

            # Method 1: Slice all rows and average
            slices = [data[i, :] for i in 1:size(data, 1)]
            avg_slices = mean(slices)

            # Method 2: Profile directly
            profile_post = data -> begin
                dims_to_mean = setdiff(collect(1:ndims(data)), [dim_keep])
                if isempty(dims_to_mean)
                    return data
                end
                dims_tuple = Tuple(dims_to_mean)
                dropdims(mean(data; dims=dims_tuple), dims=dims_tuple)
            end
            profile = profile_post(data)

            @test avg_slices ≈ profile
        end
    end

    # ============================================================================
    # Test edge cases
    # ============================================================================

    @testset "Edge Cases" begin
        @testset "Single element array" begin
            data = [42.0]

            # Mean should return scalar
            @test mean(data) == 42.0

            # Slice should return value
            post = x -> @views x[ntuple(i -> i == 1 ? 1 : :, ndims(x))...]
            @test post(data) isa Real
        end

        @testset "Zero array" begin
            data = zeros(3, 4, 5)

            @test mean(data) == 0.0
            @test mean(data; dims=1) == zeros(1, 4, 5)
        end

        @testset "Large values (numerical stability)" begin
            data = fill(1e15, 100, 100)
            @test isfinite(mean(data))
            @test mean(data) ≈ 1e15
        end

        @testset "Mixed positive/negative values" begin
            data = [-3.0 -2.0 -1.0; 1.0 2.0 3.0]
            @test mean(data) ≈ 0.0
        end

        @testset "NaN handling" begin
            data = [1.0, 2.0, NaN, 4.0]
            @test isnan(mean(data))  # Standard mean propagates NaN
        end

        @testset "Inf handling" begin
            data = [1.0, 2.0, Inf, 4.0]
            @test mean(data) == Inf

            data2 = [1.0, 2.0, -Inf, Inf]
            @test isnan(mean(data2))  # Inf - Inf = NaN
        end
    end

    # ============================================================================
    # Test ntuple indexing correctness
    # ============================================================================

    @testset "ntuple Indexing" begin
        @testset "ntuple generates correct indices" begin
            # For 3D array, ntuple(i -> i == 2 ? 5 : :, 3) should give (:, 5, :)
            indices = ntuple(i -> i == 2 ? 5 : Colon(), 3)
            @test indices == (Colon(), 5, Colon())
        end

        @testset "Slice indexing matches manual indexing" begin
            data = reshape(1:60, 3, 4, 5)

            # Manual slice at dim=2, idx=3
            manual_slice = data[:, 3, :]

            # Using ntuple
            dim = 2
            idx = 3
            ntuple_slice = data[ntuple(i -> i == dim ? idx : Colon(), 3)...]

            @test manual_slice == ntuple_slice
        end
    end

    # ============================================================================
    # Integration with handler (mock test)
    # ============================================================================

    @testset "Handler Integration Pattern" begin
        @testset "Postprocess function composition" begin
            # Test that postprocess functions can be composed
            data = rand(4, 5, 6)

            # Slice then mean
            slice_post = x -> @views x[ntuple(i -> i == 3 ? 2 : :, ndims(x))...]
            sliced = slice_post(data)
            @test size(sliced) == (4, 5)

            mean_result = mean(sliced)
            @test isfinite(mean_result)
        end

        @testset "Layout preservation" begin
            # The postprocess should work regardless of data layout
            # Test with different array types that might come from fields

            # Column-major (Julia default)
            data_col = reshape(1:12, 3, 4)

            # The postprocessing should still work
            dims = 1
            post = data -> dropdims(mean(data; dims=dims), dims=dims)
            result = post(Array(data_col))

            @test length(result) == 4
        end
    end

    # ============================================================================
    # Test specific analysis_tasks.jl function signatures
    # ============================================================================

    @testset "Function Signatures" begin
        @testset "add_mean_task! exists with correct signature" begin
            # Check that the function exists and accepts the right parameters
            @test hasmethod(Tarang.add_mean_task!, Tuple{Any, Any})
        end

        @testset "add_slice_task! exists with correct signature" begin
            @test hasmethod(Tarang.add_slice_task!, Tuple{Any, Any})
        end

        @testset "add_profile_task! exists with correct signature" begin
            @test hasmethod(Tarang.add_profile_task!, Tuple{Any, Any})
        end

        @testset "Functions are exported" begin
            # These should be accessible from the Tarang module
            @test isdefined(Tarang, :add_mean_task!)
            @test isdefined(Tarang, :add_slice_task!)
            @test isdefined(Tarang, :add_profile_task!)
        end
    end

    # ============================================================================
    # Dimension reduction verification
    # ============================================================================

    @testset "Dimension Reduction" begin
        @testset "Mean reduces ndims correctly" begin
            data = rand(2, 3, 4, 5)

            @test ndims(mean(data; dims=1)) == 4  # Still 4D (size 1 in dim 1)
            @test ndims(dropdims(mean(data; dims=1), dims=1)) == 3
            @test ndims(dropdims(mean(data; dims=(1,2)), dims=(1,2))) == 2
        end

        @testset "Profile produces 1D output" begin
            data = rand(3, 4, 5)
            dim = 2

            post = data -> begin
                dims_to_mean = setdiff(collect(1:ndims(data)), [dim])
                dims_tuple = Tuple(dims_to_mean)
                dropdims(mean(data; dims=dims_tuple), dims=dims_tuple)
            end

            result = post(data)
            @test ndims(result) == 1
            @test length(result) == size(data, dim)
        end

        @testset "Slice reduces ndims by 1" begin
            data = rand(3, 4, 5)

            post = x -> @views x[ntuple(i -> i == 2 ? 2 : :, ndims(x))...]
            result = post(data)

            @test ndims(result) == 2
            @test size(result) == (3, 5)
        end
    end

    # ============================================================================
    # Statistical verification
    # ============================================================================

    @testset "Statistical Properties" begin
        @testset "Mean of uniform data" begin
            val = 7.5
            data = fill(val, 10, 20, 30)

            @test mean(data) ≈ val
            @test all(mean(data; dims=1) .≈ val)
            @test all(mean(data; dims=(1,2)) .≈ val)
        end

        @testset "Mean bounds" begin
            data = rand(100, 100)
            m = mean(data)

            @test m >= minimum(data)
            @test m <= maximum(data)
        end

        @testset "Profile mean equals global mean" begin
            data = rand(5, 6, 7)

            # Mean of profile should equal global mean
            profile_post = data -> begin
                dims_to_mean = setdiff(collect(1:ndims(data)), [2])
                dims_tuple = Tuple(dims_to_mean)
                dropdims(mean(data; dims=dims_tuple), dims=dims_tuple)
            end
            profile = profile_post(data)

            # Weighted average of profile (uniform weights = regular mean)
            @test mean(profile) ≈ mean(data)
        end
    end
end

println("All analysis tasks tests passed!")
