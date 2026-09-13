"""
Value tests for the eight DIRECTIONAL NCCL pack/unpack kernels — WITHOUT a GPU.

`ext/cuda/nccl_transpose.jl` defines `pack_z_to_y_kernel!` and its seven
partners. They decide which element of a pencil goes to which rank in
`Tarang.transpose_{z_to_y,y_to_z,y_to_x,x_to_y}!`, the live NCCL multi-GPU
transpose drivers. Every one of their sixteen launch sites hardcodes
`CUDABackend()`, and the multi-GPU NCCL CI job is disabled, so on every machine
that currently runs tests these kernels are dead text: nothing notices if the
index math is wrong.

That is not hypothetical here. The closing comment of `nccl_transpose.jl`
records that `nccl_pack_for_transpose!` shipped as a flat copy — "correct only
at nranks == 1 and silently wrong for every real decomposition" — because
`unpack(pack(x)) == x` holds for a no-op. The sibling
`test_gpu_transpose_kernels_cpu.jl` closed that hole for the GENERIC
`pack_for_transpose_kernel_{2,3}d!` pair by comparing against an independently
written specification. These eight never got the same treatment.

They are KernelAbstractions kernels, so the same kernel objects the CUDA path
launches also run on `KernelAbstractions.CPU()` over plain Arrays.
"""

using Test
using Tarang
using KernelAbstractions

const _CUDA_LOADED_NCCL = try
    @eval using CUDA
    true
catch err
    @info "CUDA.jl unavailable; skipping NCCL transpose kernel value tests" err
    false
end

# The specification, written from the pencil layout rather than from the kernels:
# rank r owns the slab of `dim`-indices assigned to it; the packed buffer is
# rank-contiguous and each rank's segment holds that slab in column-major order.
function _nccl_reference_packed(data::AbstractArray{T,3}, dim::Int,
                                chunks::Vector{Int}) where T
    out = T[]
    offset = 0
    for chunk in chunks
        r = (offset + 1):(offset + chunk)
        append!(out, vec(selectdim(data, dim, r)))
        offset += chunk
    end
    return out
end

# Inverse spec: segment r reshapes back into rank r's slab of the output.
function _nccl_reference_unpacked(packed::Vector{T}, shape::NTuple{3,Int},
                                  dim::Int, chunks::Vector{Int}) where T
    out = zeros(T, shape)
    offset = 0
    pos = 1
    for chunk in chunks
        slab_shape = ntuple(d -> d == dim ? chunk : shape[d], 3)
        n = prod(slab_shape)
        r = (offset + 1):(offset + chunk)
        selectdim(out, dim, r) .= reshape(packed[pos:(pos + n - 1)], slab_shape)
        offset += chunk
        pos += n
    end
    return out
end

# name, kernel-extent tuple (E1,E2,E3), dimension the kernel partitions
const _NCCL_PACK_CASES = [
    (:pack_z_to_y_kernel!, (4, 6, 8), 3),
    (:pack_y_to_z_kernel!, (4, 6, 8), 2),
    (:pack_y_to_x_kernel!, (4, 6, 8), 2),
    (:pack_x_to_y_kernel!, (4, 6, 8), 1),
]
const _NCCL_UNPACK_CASES = [
    (:unpack_z_to_y_kernel!, (4, 6, 8), 2),
    (:unpack_y_to_z_kernel!, (4, 6, 8), 3),
    (:unpack_y_to_x_kernel!, (4, 6, 8), 1),
    (:unpack_x_to_y_kernel!, (4, 6, 8), 2),
]

_split(extent) = [extent ÷ 2, extent - extent ÷ 2]

# displs are ELEMENT offsets: each rank's segment is (total ÷ extent) * chunk long.
function _displs_for(shape::NTuple{3,Int}, dim::Int, chunks::Vector{Int})
    per = prod(shape) ÷ shape[dim]
    d = Int[]; acc = 0
    for c in chunks; push!(d, acc); acc += per * c; end
    return d
end

@testset "NCCL directional pack/unpack index math (CPU backend)" begin
    if !_CUDA_LOADED_NCCL
        @test_skip "CUDA.jl not loadable in this environment"
    else
        ext = Base.get_extension(Tarang, :TarangCUDAExt)
        @test ext !== nothing
        backend = KernelAbstractions.CPU()

        @testset "$name packs rank-contiguous slabs of dim=$dim" for (name, shape, dim) in _NCCL_PACK_CASES
            data = reshape(collect(1.0:prod(shape)), shape)
            chunks = _split(shape[dim])
            displs = _displs_for(shape, dim, chunks)
            prefix = cumsum(chunks)

            buffer = zeros(Float64, prod(shape))
            kernel = getproperty(ext, name)(backend)
            kernel(buffer, data, shape[1], shape[2], shape[3],
                   chunks, displs, length(chunks), prefix; ndrange=prod(shape))
            KernelAbstractions.synchronize(backend)

            @test buffer == _nccl_reference_packed(data, dim, chunks)
            # A flat copy satisfies any round-trip test. For a partitioned dim
            # other than the last, the pack is a real permutation.
            dim == 3 || @test buffer != vec(data)
        end

        @testset "$name scatters rank segments along dim=$dim" for (name, shape, dim) in _NCCL_UNPACK_CASES
            chunks = _split(shape[dim])
            displs = _displs_for(shape, dim, chunks)
            prefix = cumsum(chunks)
            packed = collect(1.0:prod(shape))

            out = zeros(Float64, shape)
            kernel = getproperty(ext, name)(backend)
            kernel(out, packed, shape[1], shape[2], shape[3],
                   chunks, displs, length(chunks), prefix; ndrange=prod(shape))
            KernelAbstractions.synchronize(backend)

            @test out == _nccl_reference_unpacked(packed, shape, dim, chunks)
            dim == 3 || @test vec(out) != packed
        end

        @testset "pack/unpack partners invert each other" begin
            shape = (4, 6, 8)
            for (pk, pdim, uk, udim) in ((:pack_z_to_y_kernel!, 3, :unpack_y_to_z_kernel!, 3),
                                         (:pack_x_to_y_kernel!, 1, :unpack_y_to_x_kernel!, 1))
                data = reshape(collect(1.0:prod(shape)), shape)
                chunks = _split(shape[pdim]); displs = _displs_for(shape, pdim, chunks)
                prefix = cumsum(chunks)
                buf = zeros(Float64, prod(shape))
                getproperty(ext, pk)(backend)(buf, data, shape[1], shape[2], shape[3],
                    chunks, displs, length(chunks), prefix; ndrange=prod(shape))
                KernelAbstractions.synchronize(backend)
                back = zeros(Float64, shape)
                getproperty(ext, uk)(backend)(back, buf, shape[1], shape[2], shape[3],
                    chunks, displs, length(chunks), prefix; ndrange=prod(shape))
                KernelAbstractions.synchronize(backend)
                @test back == data
            end
        end
    end
end
