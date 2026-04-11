using Test
using Tarang
using Tarang: CPU, launch!, KernelOperation, workgroup_size, architecture, ensure_device!
using KernelAbstractions

@testset "KernelOperation" begin
    # ========================================================================
    # Constructor tests
    # ========================================================================

    @testset "keyword constructor" begin
        op = KernelOperation(identity; ndrange_fn=(args...) -> 10)
        @test op.kernel === identity
        @test op.ndrange_fn((nothing,)) == 10
    end

    @testset "default ndrange_fn" begin
        op = KernelOperation(identity)
        # default_ndrange_fn returns length of first argument
        @test op.ndrange_fn([1, 2, 3]) == 3
        @test op.ndrange_fn(zeros(5, 5)) == 25
    end

    @testset "do-block constructor" begin
        # This is the critical bug we fixed: do-block desugars to
        # KernelOperation(lambda, kernel) — two positional args
        op = KernelOperation(identity) do args...
            42
        end
        @test op.kernel === identity
        @test op.ndrange_fn() == 42
    end

    @testset "do-block with typed args" begin
        op = KernelOperation(identity) do c, a, b
            length(c)
        end
        @test op.ndrange_fn([1, 2, 3], nothing, nothing) == 3
    end

    @testset "struct fields preserved" begin
        my_fn(x) = x * 2
        ndrange_fn(args...) = 100
        op = KernelOperation(ndrange_fn, my_fn)
        @test op.kernel === my_fn
        @test op.ndrange_fn === ndrange_fn
    end

    # ========================================================================
    # CPU launch! integration
    # ========================================================================

    @kernel function test_add_kernel!(c, a, b)
        i = @index(Global)
        @inbounds c[i] = a[i] + b[i]
    end

    @kernel function test_scale_kernel!(c, a, alpha)
        i = @index(Global)
        @inbounds c[i] = a[i] * alpha
    end

    @testset "launch! on CPU with explicit ndrange" begin
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        c = zeros(3)
        launch!(CPU(), test_add_kernel!, c, a, b; ndrange=3)
        @test c == [5.0, 7.0, 9.0]
    end

    @testset "KernelOperation callable on CPU" begin
        op = KernelOperation(test_add_kernel!) do c, a, b
            length(c)
        end

        a = [1.0, 2.0, 3.0, 4.0]
        b = [10.0, 20.0, 30.0, 40.0]
        c = zeros(4)
        op(CPU(), c, a, b)
        @test c == [11.0, 22.0, 33.0, 44.0]
    end

    @testset "KernelOperation with scalar arg" begin
        op = KernelOperation(test_scale_kernel!) do c, a, _
            length(c)
        end

        a = [1.0, 2.0, 3.0]
        c = zeros(3)
        op(CPU(), c, a, 3.0)
        @test c == [3.0, 6.0, 9.0]
    end

    @testset "KernelOperation ndrange override" begin
        op = KernelOperation(test_add_kernel!) do c, a, b
            length(c)
        end

        a = [1.0, 2.0, 3.0, 4.0]
        b = [10.0, 20.0, 30.0, 40.0]
        c = zeros(4)
        # Only process first 2 elements
        op(CPU(), c, a, b; ndrange=2)
        @test c[1] == 11.0
        @test c[2] == 22.0
        @test c[3] == 0.0  # untouched
        @test c[4] == 0.0  # untouched
    end

    @testset "launch! via array dispatch" begin
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        c = zeros(3)
        # launch! can infer architecture from array
        launch!(c, test_add_kernel!, c, a, b; ndrange=3)
        @test c == [5.0, 7.0, 9.0]
    end
end
