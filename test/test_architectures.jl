"""
Test suite for core/architectures.jl

Tests:
1. CPU/GPU type hierarchy
2. CPU construction and properties
3. architecture() inference
4. array_type for CPU
5. on_architecture data movement
6. Array allocation (zeros, ones, similar, create_array)
7. Utility functions (is_gpu, launch_config, workgroup_size)
"""

using Test

@testset "Architectures Module" begin
    using Tarang

    # ------------------------------------------------------------------
    # Type hierarchy
    # ------------------------------------------------------------------
    @testset "Type hierarchy" begin
        @test CPU <: Tarang.AbstractSerialArchitecture
        @test Tarang.AbstractSerialArchitecture <: Tarang.AbstractArchitecture
        @test GPU <: Tarang.AbstractSerialArchitecture
    end

    # ------------------------------------------------------------------
    # CPU construction
    # ------------------------------------------------------------------
    @testset "CPU construction" begin
        arch = CPU()
        @test isa(arch, CPU)
        @test isa(arch, Tarang.AbstractArchitecture)
        # Two instances should be equal (singleton-like struct)
        @test CPU() === CPU()
    end

    # ------------------------------------------------------------------
    # GPU construction (no CUDA loaded -- should error)
    # ------------------------------------------------------------------
    @testset "GPU without CUDA errors" begin
        @test_throws ErrorException GPU()
    end

    # ------------------------------------------------------------------
    # is_gpu
    # ------------------------------------------------------------------
    @testset "is_gpu" begin
        @test is_gpu(CPU()) == false
    end

    # ------------------------------------------------------------------
    # has_cuda (default stub returns false)
    # ------------------------------------------------------------------
    @testset "has_cuda" begin
        @test has_cuda() == false
    end

    # ------------------------------------------------------------------
    # array_type
    # ------------------------------------------------------------------
    @testset "array_type for CPU" begin
        @test array_type(CPU()) === Array
        @test array_type(CPU(), Float64) === Array{Float64}
        @test array_type(CPU(), ComplexF64) === Array{ComplexF64}
    end

    # ------------------------------------------------------------------
    # architecture() inference
    # ------------------------------------------------------------------
    @testset "architecture inference" begin
        # From a plain Array
        a = rand(4, 4)
        @test architecture(a) isa CPU

        # Identity: architecture of an architecture
        @test architecture(CPU()) === CPU()

        # From an AbstractArray subtype backed by Array (e.g., a view)
        v = @view a[1:2, :]
        @test architecture(v) isa CPU
    end

    # ------------------------------------------------------------------
    # on_architecture -- CPU paths
    # ------------------------------------------------------------------
    @testset "on_architecture CPU" begin
        arch = CPU()

        # Array -> CPU is a no-op (same object)
        a = rand(3, 3)
        @test on_architecture(arch, a) === a

        # SubArray -> CPU is a no-op
        v = @view a[1:2, :]
        @test on_architecture(arch, v) === v

        # Scalars pass through
        @test on_architecture(arch, 42) === 42
        @test on_architecture(arch, 3.14) === 3.14
        @test on_architecture(arch, "hello") === "hello"
        @test on_architecture(arch, :sym) === :sym

        # Nothing passes through
        @test on_architecture(arch, nothing) === nothing

        # Tuples are handled recursively
        t = ([1, 2], [3, 4])
        result = on_architecture(arch, t)
        @test result[1] === t[1]  # same object on CPU
        @test result[2] === t[2]

        # NamedTuples are handled recursively
        nt = (x=[1.0, 2.0], y=[3.0, 4.0])
        result_nt = on_architecture(arch, nt)
        @test result_nt.x === nt.x
        @test result_nt.y === nt.y
    end

    # ------------------------------------------------------------------
    # Array allocation helpers
    # ------------------------------------------------------------------
    @testset "zeros on CPU" begin
        a = zeros(CPU(), Float64, 3, 4)
        @test size(a) == (3, 4)
        @test eltype(a) == Float64
        @test all(a .== 0.0)
        @test isa(a, Array{Float64})

        # Tuple form
        b = zeros(CPU(), Float32, (2, 5))
        @test size(b) == (2, 5)
        @test eltype(b) == Float32
    end

    @testset "ones on CPU" begin
        a = ones(CPU(), Float64, 2, 3)
        @test size(a) == (2, 3)
        @test all(a .== 1.0)
        @test isa(a, Array{Float64})
    end

    @testset "similar on CPU" begin
        src = rand(ComplexF64, 4, 5)
        s = similar(CPU(), src)
        @test size(s) == (4, 5)
        @test eltype(s) == ComplexF64

        # With explicit element type
        s2 = similar(CPU(), src, Float32)
        @test size(s2) == (4, 5)
        @test eltype(s2) == Float32
    end

    @testset "create_array on CPU" begin
        a = create_array(CPU(), Float64, 6, 7)
        @test size(a) == (6, 7)
        @test eltype(a) == Float64
        @test all(a .== 0.0)

        # Tuple form
        b = create_array(CPU(), Int, (3, 3))
        @test size(b) == (3, 3)
    end

    # ------------------------------------------------------------------
    # similar_zeros and allocate_like
    # ------------------------------------------------------------------
    @testset "similar_zeros" begin
        a = ones(Float64, 3, 3)
        z = Tarang.similar_zeros(a)
        @test size(z) == (3, 3)
        @test all(z .== 0.0)

        # With different dims
        z2 = Tarang.similar_zeros(a, 2, 4)
        @test size(z2) == (2, 4)
        @test all(z2 .== 0.0)

        # With different type and dims
        z3 = Tarang.similar_zeros(a, Float32, 5)
        @test eltype(z3) == Float32
        @test all(z3 .== 0.0f0)
    end

    @testset "allocate_like" begin
        a = ones(Float64, 2, 2)
        b = Tarang.allocate_like(a, Float64, 3, 3)
        @test size(b) == (3, 3)
        @test all(b .== 0.0)
    end

    # ------------------------------------------------------------------
    # launch_config and workgroup_size
    # ------------------------------------------------------------------
    @testset "launch_config CPU" begin
        cfg = Tarang.launch_config(CPU(), 1024)
        @test cfg == (1024,)
    end

    @testset "workgroup_size CPU" begin
        @test Tarang.workgroup_size(CPU(), 128) == min(128, 64)
        @test Tarang.workgroup_size(CPU(), 32) == 32
    end

    # ------------------------------------------------------------------
    # synchronize and unsafe_free! on CPU (no-ops)
    # ------------------------------------------------------------------
    @testset "CPU no-ops" begin
        @test synchronize(CPU()) === nothing
        @test unsafe_free!(CPU(), ones(2)) === nothing
    end

    # ------------------------------------------------------------------
    # is_gpu_array
    # ------------------------------------------------------------------
    @testset "is_gpu_array" begin
        @test is_gpu_array(rand(3)) == false
        @test is_gpu_array(zeros(2, 2)) == false
    end

    # ------------------------------------------------------------------
    # show / summary
    # ------------------------------------------------------------------
    @testset "show and summary" begin
        @test sprint(show, CPU()) == "CPU()"
        @test summary(CPU()) == "CPU"
    end
end

println("All architecture tests passed!")
