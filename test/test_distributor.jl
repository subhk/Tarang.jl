using Test
using Tarang
using MPI

@testset "Distributor CPU" begin
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())

    cpu_array = create_pencil(dist, (8,), 1; dtype=Float64)
    @test cpu_array isa Array{Float64, 1}
    cpu_array .= collect(1.0:8.0)

    gathered = gather_array(dist, cpu_array)
    @test gathered == collect(1.0:8.0)

    scattered = scatter_array(dist, gathered)
    @test scattered == gathered

    reduced = allreduce_array(dist, cpu_array)
    @test reduced == cpu_array

    recv = similar(cpu_array)
    mpi_alltoall(dist, cpu_array, recv)
    @test recv == cpu_array
end

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
    @testset "Distributor GPU" begin
        CUDA.allowscalar(false)
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float32, architecture=GPU())

        gpu_array = create_pencil(dist, (16,), 1; dtype=Float32)
        @test gpu_array isa CuArray{Float32, 1}
        gpu_array .= CuArray(Float32.(1:16))

        # scatter converts CPU input to GPU output
        scattered = scatter_array(dist, Float32.(1:16))
        @test scattered isa CuArray
        @test Array(scattered) == Float32.(1:16)

        # allreduce returns GPU array
        reduced = allreduce_array(dist, scattered)
        @test reduced isa CuArray
        @test Array(reduced) == Float32.(1:16)

        # mpi_alltoall handles GPU buffers (single-rank)
        recv = similar(gpu_array)
        mpi_alltoall(dist, gpu_array, recv)
        @test Array(recv) == Float32.(1:16)
    end
else
    @testset "Distributor GPU" begin
        @test_skip "CUDA not available"
    end
end
