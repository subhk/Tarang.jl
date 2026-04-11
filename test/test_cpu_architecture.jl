using Test
using Tarang
using Tarang: CPU, GPU, launch!, KernelOperation, workgroup_size, architecture,
             ensure_device!, on_architecture, is_gpu, is_gpu_array, array_type

@testset "CPU Architecture" begin
    # ========================================================================
    # Type hierarchy
    # ========================================================================

    @testset "type hierarchy" begin
        arch = CPU()
        @test arch isa Tarang.AbstractArchitecture
        @test arch isa Tarang.AbstractSerialArchitecture
        @test arch isa CPU
    end

    @testset "GPU type without CUDA" begin
        @test GPU <: Tarang.AbstractSerialArchitecture
        # GPU() should error without CUDA loaded
        @test_throws ErrorException GPU()
    end

    # ========================================================================
    # device / array_type
    # ========================================================================

    @testset "device returns KA CPU backend" begin
        arch = CPU()
        d = Tarang.device(arch)
        @test d isa KernelAbstractions.CPU
    end

    @testset "array_type" begin
        arch = CPU()
        @test Tarang.array_type(arch) === Array
        @test Tarang.array_type(arch, Float64) === Array{Float64}
        @test Tarang.array_type(arch, ComplexF64) === Array{ComplexF64}
    end

    # ========================================================================
    # Architecture inference
    # ========================================================================

    @testset "architecture inference" begin
        @test architecture(zeros(3)) isa CPU
        @test architecture(ones(2, 2)) isa CPU
        @test architecture(CPU()) isa CPU
    end

    # ========================================================================
    # ensure_device! (no-op on CPU)
    # ========================================================================

    @testset "ensure_device! is no-op on CPU" begin
        @test ensure_device!(CPU()) === nothing
    end

    # ========================================================================
    # is_gpu / has_cuda
    # ========================================================================

    @testset "GPU detection" begin
        @test is_gpu(CPU()) == false
        @test is_gpu_array(zeros(3)) == false
        @test is_gpu_array(ones(2, 2)) == false
    end

    # ========================================================================
    # Array allocation
    # ========================================================================

    @testset "zeros on CPU" begin
        a = zeros(CPU(), Float64, 3, 4)
        @test size(a) == (3, 4)
        @test eltype(a) == Float64
        @test all(a .== 0.0)
        @test a isa Array{Float64}
    end

    @testset "ones on CPU" begin
        a = ones(CPU(), Float32, 5)
        @test size(a) == (5,)
        @test eltype(a) == Float32
        @test all(a .== 1.0f0)
        @test a isa Array{Float32}
    end

    @testset "similar on CPU" begin
        src = rand(3, 4)
        a = similar(CPU(), src)
        @test size(a) == (3, 4)
        @test eltype(a) == Float64
        @test a isa Array{Float64}

        b = similar(CPU(), src, Float32)
        @test size(b) == (3, 4)
        @test eltype(b) == Float32
        @test b isa Array{Float32}
    end

    @testset "zeros with tuple dims" begin
        a = zeros(CPU(), Float64, (2, 3))
        @test size(a) == (2, 3)
        @test a isa Array{Float64}
    end

    # ========================================================================
    # on_architecture (CPU data movement)
    # ========================================================================

    @testset "on_architecture CPU to CPU" begin
        a = [1.0, 2.0, 3.0]
        b = on_architecture(CPU(), a)
        @test b === a  # same object, no copy
    end

    @testset "on_architecture scalars pass through" begin
        @test on_architecture(CPU(), 42) === 42
        @test on_architecture(CPU(), 3.14) === 3.14
        @test on_architecture(CPU(), "hello") === "hello"
        @test on_architecture(CPU(), :sym) === :sym
        @test on_architecture(CPU(), nothing) === nothing
    end

    @testset "on_architecture tuples" begin
        t = ([1.0, 2.0], [3.0, 4.0])
        result = on_architecture(CPU(), t)
        @test result[1] == [1.0, 2.0]
        @test result[2] == [3.0, 4.0]
    end

    @testset "on_architecture named tuples" begin
        nt = (a=[1.0, 2.0], b=[3.0, 4.0])
        result = on_architecture(CPU(), nt)
        @test result.a == [1.0, 2.0]
        @test result.b == [3.0, 4.0]
    end

    # ========================================================================
    # Allocation helpers
    # ========================================================================

    @testset "allocate_like" begin
        a = rand(3, 4)
        b = Tarang.allocate_like(a, Float64, 5, 6)
        @test size(b) == (5, 6)
        @test eltype(b) == Float64
        @test all(b .== 0.0)
    end

    @testset "similar_zeros" begin
        a = rand(3, 4)
        b = Tarang.similar_zeros(a)
        @test size(b) == (3, 4)
        @test eltype(b) == Float64
        @test all(b .== 0.0)
    end

    # ========================================================================
    # workgroup_size
    # ========================================================================

    @testset "workgroup_size for CPU" begin
        arch = CPU()
        # CPU workgroup_size should return a reasonable value
        ws = workgroup_size(arch, 100)
        @test ws isa Integer || ws isa Tuple
        @test ws > 0 || all(x -> x > 0, ws)
    end

    # ========================================================================
    # Printing
    # ========================================================================

    @testset "show/summary" begin
        @test summary(CPU()) == "CPU"
        @test sprint(show, CPU()) == "CPU()"
    end
end
