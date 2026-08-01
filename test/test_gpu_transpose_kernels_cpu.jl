"""
Value tests for the GPU transpose pack/unpack index math — WITHOUT a GPU.

`pack_for_transpose_kernel_{2,3}d!` and their unpack partners are
KernelAbstractions kernels, so the very same kernel objects the CUDA path
launches on a `CUDABackend()` also run on `KernelAbstractions.CPU()` over plain
`Array`s. CUDA.jl itself loads on machines with no NVIDIA GPU, so the extension
(and therefore the kernels) is reachable in ordinary CI.

This matters because these kernels decide which element of a pencil goes to which
rank. That is exactly the math the deleted `nccl_pack_for_transpose!(packed, data,
dim)` stub could not express — its signature omitted `counts`/`displs`/`nranks`,
so it flat-copied, which is correct only at nranks == 1. Its round-trip test
passed for that reason: `unpack(pack(x)) == x` holds for any no-op. The
assertions below compare the packed buffer against an independently constructed
rank-contiguous layout, so a flat copy fails them.
"""

using Test
using Tarang
using KernelAbstractions

const _CUDA_LOADED = try
    @eval using CUDA
    true
catch err
    @info "CUDA.jl unavailable; skipping GPU transpose kernel value tests" err
    false
end

# The specification, written independently of the kernels: rank r owns the slab
# of `dim`-indices it was assigned, stored column-major, placed at displs[r].
function _reference_packed(data::AbstractArray{T,3}, dim::Int, chunks::Vector{Int}) where T
    out = T[]
    offset = 0
    for chunk in chunks
        range = (offset + 1):(offset + chunk)
        slab = dim == 3 ? data[:, :, range] :
               dim == 2 ? data[:, range, :] :
                          data[range, :, :]
        append!(out, vec(slab))
        offset += chunk
    end
    return out
end

function _reference_packed(data::AbstractArray{T,2}, dim::Int, chunks::Vector{Int}) where T
    out = T[]
    offset = 0
    for chunk in chunks
        range = (offset + 1):(offset + chunk)
        slab = dim == 2 ? data[:, range] : data[range, :]
        append!(out, vec(slab))
        offset += chunk
    end
    return out
end

# counts[r] is the ELEMENT count rank r receives; the launchers divide it back out
# to a chunk size, so build it the same way the launchers expect.
_counts_from_chunks(chunks, total, split_extent) =
    [c * (total ÷ split_extent) for c in chunks]

@testset "GPU transpose pack/unpack index math (CPU backend)" begin
    if !_CUDA_LOADED
        @test_skip "CUDA.jl not loadable in this environment"
    else
        ext = Base.get_extension(Tarang, :TarangCUDAExt)
        @test ext !== nothing

        backend = KernelAbstractions.CPU()

        @testset "3D pack matches rank-contiguous layout (dim=$dim)" for dim in 1:3
            Nx, Ny, Nz = 4, 6, 8
            data = reshape(collect(1.0:(Nx * Ny * Nz)), Nx, Ny, Nz)
            extent = (Nx, Ny, Nz)[dim]
            chunks = [extent ÷ 2, extent - extent ÷ 2]
            counts = _counts_from_chunks(chunks, Nx * Ny * Nz, extent)
            displs = [0, counts[1]]
            prefix = cumsum(chunks)

            buffer = zeros(Float64, Nx * Ny * Nz)
            kernel = ext.pack_for_transpose_kernel_3d!(backend)
            kernel(buffer, data, Nx, Ny, Nz, length(chunks), dim,
                   chunks, displs, prefix; ndrange=Nx * Ny * Nz)
            KernelAbstractions.synchronize(backend)

            @test buffer == _reference_packed(data, dim, chunks)

            # A flat copy would satisfy any round-trip test; for dims 1 and 2 the
            # pack is a genuine permutation, so it must NOT equal vec(data).
            dim == 3 || @test buffer != vec(data)

            # Unpack is the inverse mapping and must restore the array exactly.
            restored = zeros(Float64, Nx, Ny, Nz)
            unpack = ext.unpack_from_transpose_kernel_3d!(backend)
            unpack(restored, buffer, Nx, Ny, Nz, length(chunks), dim,
                   chunks, displs, prefix; ndrange=Nx * Ny * Nz)
            KernelAbstractions.synchronize(backend)
            @test restored == data
        end

        @testset "2D pack matches rank-contiguous layout (dim=$dim)" for dim in 1:2
            Nx, Ny = 6, 4
            data = reshape(collect(1.0:(Nx * Ny)), Nx, Ny)
            extent = (Nx, Ny)[dim]
            chunks = [extent ÷ 2, extent - extent ÷ 2]
            counts = _counts_from_chunks(chunks, Nx * Ny, extent)
            displs = [0, counts[1]]
            prefix = cumsum(chunks)

            buffer = zeros(Float64, Nx * Ny)
            kernel = ext.pack_for_transpose_kernel_2d!(backend)
            kernel(buffer, data, Nx, Ny, length(chunks), dim,
                   chunks, displs, prefix; ndrange=Nx * Ny)
            KernelAbstractions.synchronize(backend)

            @test buffer == _reference_packed(data, dim, chunks)
            dim == 2 || @test buffer != vec(data)

            restored = zeros(Float64, Nx, Ny)
            unpack = ext.unpack_from_transpose_kernel_2d!(backend)
            unpack(restored, buffer, Nx, Ny, length(chunks), dim,
                   chunks, displs, prefix; ndrange=Nx * Ny)
            KernelAbstractions.synchronize(backend)
            @test restored == data
        end

        @testset "Uneven rank splits" begin
            # Remainder-carrying chunks are where an off-by-one in the prefix-sum
            # rank lookup shows up.
            Nx, Ny, Nz = 3, 7, 5
            data = reshape(collect(1.0:(Nx * Ny * Nz)), Nx, Ny, Nz)
            chunks = [3, 1, 3]          # Ny = 7 across three ranks
            counts = _counts_from_chunks(chunks, Nx * Ny * Nz, Ny)
            displs = cumsum([0; counts[1:end-1]])
            prefix = cumsum(chunks)

            buffer = zeros(Float64, Nx * Ny * Nz)
            kernel = ext.pack_for_transpose_kernel_3d!(backend)
            kernel(buffer, data, Nx, Ny, Nz, length(chunks), 2,
                   chunks, displs, prefix; ndrange=Nx * Ny * Nz)
            KernelAbstractions.synchronize(backend)
            @test buffer == _reference_packed(data, 2, chunks)

            restored = zeros(Float64, Nx, Ny, Nz)
            unpack = ext.unpack_from_transpose_kernel_3d!(backend)
            unpack(restored, buffer, Nx, Ny, Nz, length(chunks), 2,
                   chunks, displs, prefix; ndrange=Nx * Ny * Nz)
            KernelAbstractions.synchronize(backend)
            @test restored == data
        end

        @testset "The removed stubs stay removed" begin
            # They were exported, flat-copied, and were correct only at nranks==1.
            @test !isdefined(ext, :nccl_pack_for_transpose!)
            @test !isdefined(ext, :nccl_unpack_from_transpose!)
        end
    end
end
