"""
    Transpose Pack/Unpack - Data packing operations for TransposableField

This file contains the CPU implementations for packing and unpacking data
for MPI.Alltoallv operations during transposes.
"""

# ============================================================================
# Pack/Unpack Operations (CPU implementations)
# ============================================================================

"""
    pack_for_transpose!(buffer, data, counts, displs, dim, nranks, arch::CPU)

Pack data into contiguous buffer for MPI.Alltoallv (CPU version).
Reorders data so each destination rank receives a contiguous chunk.

For dim=3 (Z→Y transpose): packs z-slices for each rank
For dim=2 (Y→X transpose): packs y-slices for each rank
For dim=1 (X redistribution): packs x-slices for each rank
"""
function pack_for_transpose!(buffer, data, counts, displs, dim::Int,
                             nranks::Int, arch::CPU)
    # Guard: Empty arrays should not be packed
    if length(data) == 0
        @warn "pack_for_transpose! called with empty data array. No data to pack." maxlog=1
        return buffer
    end

    # Guard: Check for zero-sized dimensions
    for i in 1:ndims(data)
        if size(data, i) == 0
            error("pack_for_transpose!: data has zero-sized dimension $i (shape=$(size(data))). " *
                  "This indicates incorrect buffer allocation or decomposition.")
        end
    end

    if nranks == 1
        # Single process - simple copy
        copyto!(view(buffer, 1:length(data)), vec(data))
        return buffer
    end

    ndims_data = ndims(data)

    if ndims_data == 3
        Nx, Ny, Nz = size(data)
        _pack_3d_cpu!(buffer, data, counts, displs, dim, nranks, Nx, Ny, Nz)
    elseif ndims_data == 2
        Nx, Ny = size(data)
        _pack_2d_cpu!(buffer, data, counts, displs, dim, nranks, Nx, Ny)
    else
        # 1D - simple copy
        copyto!(view(buffer, 1:length(data)), vec(data))
    end

    return buffer
end

function _pack_3d_cpu!(buffer, data, counts, displs, dim::Int, nranks::Int,
                       Nx::Int, Ny::Int, Nz::Int)
    # Compute chunk sizes from counts
    chunk_sizes = zeros(Int, nranks)
    expected_dim_size = dim == 3 ? Nz : (dim == 2 ? Ny : Nx)
    divisor = dim == 3 ? (Nx * Ny) : (dim == 2 ? (Nx * Nz) : (Ny * Nz))

    for r in 1:nranks
        chunk_sizes[r] = counts[r] ÷ divisor
        # Validate division is exact (no remainder lost)
        if counts[r] % divisor != 0
            error("Pack chunk size computation error: counts[$r]=$(counts[r]) is not evenly divisible by $divisor. " *
                  "This indicates a mismatch between counts array and data dimensions.")
        end
    end

    # Validate chunk sizes sum to expected dimension size
    chunk_sum = sum(chunk_sizes)
    if chunk_sum != expected_dim_size
        error("Pack chunk size sum ($chunk_sum) does not match expected dimension size ($expected_dim_size) for dim=$dim. " *
              "This indicates corrupted counts array or incorrect decomposition.")
    end

    buf_len = length(buffer)

    # Precompute rank lookup table: O(1) per element instead of O(nranks) linear scan
    rank_for_idx = Vector{Int}(undef, expected_dim_size)
    offset_for_idx = Vector{Int}(undef, expected_dim_size)
    cumulative = 0
    for r in 1:nranks
        for j in 1:chunk_sizes[r]
            idx = cumulative + j
            rank_for_idx[idx] = r
            offset_for_idx[idx] = cumulative
        end
        cumulative += chunk_sizes[r]
    end

    # Pack data using precomputed rank table
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        if dim == 3
            rank = rank_for_idx[iz]
            local_iz = iz - offset_for_idx[iz]
            local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx
        elseif dim == 2
            rank = rank_for_idx[iy]
            local_iy = iy - offset_for_idx[iy]
            local_idx = (iz - 1) * Nx * chunk_sizes[rank] + (local_iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx
        else  # dim == 1
            rank = rank_for_idx[ix]
            local_ix = ix - offset_for_idx[ix]
            local_idx = (iz - 1) * chunk_sizes[rank] * Ny + (iy - 1) * chunk_sizes[rank] + local_ix
            buf_idx = displs[rank] + local_idx
        end

        if buf_idx < 1 || buf_idx > buf_len
            error("Pack buffer index out of bounds: buf_idx=$buf_idx, buffer length=$buf_len. " *
                  "Position: ix=$ix, iy=$iy, iz=$iz, rank=$rank, displs[rank]=$(displs[rank])")
        end
        @inbounds buffer[buf_idx] = data[ix, iy, iz]
    end
end

function _pack_2d_cpu!(buffer, data, counts, displs, dim::Int, nranks::Int,
                       Nx::Int, Ny::Int)
    # Compute chunk sizes with validation
    chunk_sizes = zeros(Int, nranks)
    expected_dim_size = dim == 2 ? Ny : Nx
    divisor = dim == 2 ? Nx : Ny

    for r in 1:nranks
        chunk_sizes[r] = counts[r] ÷ divisor
        if counts[r] % divisor != 0
            error("Pack 2D chunk size error: counts[$r]=$(counts[r]) not divisible by $divisor.")
        end
    end

    chunk_sum = sum(chunk_sizes)
    if chunk_sum != expected_dim_size
        error("Pack 2D chunk sum ($chunk_sum) != expected ($expected_dim_size) for dim=$dim.")
    end

    buf_len = length(buffer)

    # Precompute rank lookup table
    rank_for_idx = Vector{Int}(undef, expected_dim_size)
    offset_for_idx = Vector{Int}(undef, expected_dim_size)
    cumulative = 0
    for r in 1:nranks
        for j in 1:chunk_sizes[r]
            idx = cumulative + j
            rank_for_idx[idx] = r
            offset_for_idx[idx] = cumulative
        end
        cumulative += chunk_sizes[r]
    end

    for iy in 1:Ny, ix in 1:Nx
        if dim == 2
            rank = rank_for_idx[iy]
            local_iy = iy - offset_for_idx[iy]
            local_idx = (local_iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx
        else  # dim == 1
            rank = rank_for_idx[ix]
            local_ix = ix - offset_for_idx[ix]
            local_idx = (iy - 1) * chunk_sizes[rank] + local_ix
            buf_idx = displs[rank] + local_idx
        end

        if buf_idx < 1 || buf_idx > buf_len
            error("Pack 2D buffer out of bounds: buf_idx=$buf_idx, len=$buf_len, ix=$ix, iy=$iy")
        end
        @inbounds buffer[buf_idx] = data[ix, iy]
    end
end

function pack_for_transpose!(buffer, data, counts, displs, dim::Int,
                             nranks::Int, arch::AbstractArchitecture)
    # Default implementation for GPU - use CPU version via staging
    # GPU version overrides this in TarangCUDAExt
    data_cpu = on_architecture(CPU(), data)
    buffer_cpu = on_architecture(CPU(), buffer)

    pack_for_transpose!(buffer_cpu, data_cpu, counts, displs, dim, nranks, CPU())
    copyto!(buffer, on_architecture(arch, buffer_cpu))

    return buffer
end

"""
    unpack_from_transpose!(data, buffer, counts, displs, dim, nranks, arch::CPU)

Unpack data from buffer after MPI.Alltoallv (CPU version).
Reconstructs the array from chunks received from different ranks.
"""
function unpack_from_transpose!(data, buffer, counts, displs, dim::Int,
                                nranks::Int, arch::CPU)
    # Guard: Empty arrays should not be unpacked
    if length(data) == 0
        @warn "unpack_from_transpose! called with empty data array. No data to unpack." maxlog=1
        return data
    end

    # Guard: Check for zero-sized dimensions
    for i in 1:ndims(data)
        if size(data, i) == 0
            error("unpack_from_transpose!: data has zero-sized dimension $i (shape=$(size(data))). " *
                  "This indicates incorrect buffer allocation or decomposition.")
        end
    end

    if nranks == 1
        # Single process - simple copy
        copyto!(vec(data), view(buffer, 1:length(data)))
        return data
    end

    ndims_data = ndims(data)

    if ndims_data == 3
        Nx, Ny, Nz = size(data)
        _unpack_3d_cpu!(data, buffer, counts, displs, dim, nranks, Nx, Ny, Nz)
    elseif ndims_data == 2
        Nx, Ny = size(data)
        _unpack_2d_cpu!(data, buffer, counts, displs, dim, nranks, Nx, Ny)
    else
        copyto!(vec(data), view(buffer, 1:length(data)))
    end

    return data
end

function _unpack_3d_cpu!(data, buffer, counts, displs, dim::Int, nranks::Int,
                         Nx::Int, Ny::Int, Nz::Int)
    # Compute chunk sizes with validation
    chunk_sizes = zeros(Int, nranks)
    expected_dim_size = dim == 2 ? Ny : (dim == 1 ? Nx : Nz)
    divisor = dim == 2 ? (Nx * Nz) : (dim == 1 ? (Ny * Nz) : (Nx * Ny))

    for r in 1:nranks
        chunk_sizes[r] = counts[r] ÷ divisor
        if counts[r] % divisor != 0
            error("Unpack 3D chunk size error: counts[$r]=$(counts[r]) not divisible by $divisor for dim=$dim.")
        end
    end

    chunk_sum = sum(chunk_sizes)
    if chunk_sum != expected_dim_size
        error("Unpack 3D chunk sum ($chunk_sum) != expected ($expected_dim_size) for dim=$dim.")
    end

    buf_len = length(buffer)

    # Precompute rank lookup table
    rank_for_idx = Vector{Int}(undef, expected_dim_size)
    offset_for_idx = Vector{Int}(undef, expected_dim_size)
    cumulative = 0
    for r in 1:nranks
        for j in 1:chunk_sizes[r]
            idx = cumulative + j
            rank_for_idx[idx] = r
            offset_for_idx[idx] = cumulative
        end
        cumulative += chunk_sizes[r]
    end

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        if dim == 2
            rank = rank_for_idx[iy]
            local_iy = iy - offset_for_idx[iy]
            local_idx = (iz - 1) * Nx * chunk_sizes[rank] + (local_iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx
        elseif dim == 1
            rank = rank_for_idx[ix]
            local_ix = ix - offset_for_idx[ix]
            local_idx = (iz - 1) * chunk_sizes[rank] * Ny + (iy - 1) * chunk_sizes[rank] + local_ix
            buf_idx = displs[rank] + local_idx
        else  # dim == 3
            rank = rank_for_idx[iz]
            local_iz = iz - offset_for_idx[iz]
            local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx
        end

        if buf_idx < 1 || buf_idx > buf_len
            error("Unpack 3D buffer out of bounds: buf_idx=$buf_idx, len=$buf_len")
        end
        @inbounds data[ix, iy, iz] = buffer[buf_idx]
    end
end

function _unpack_2d_cpu!(data, buffer, counts, displs, dim::Int, nranks::Int,
                         Nx::Int, Ny::Int)
    # Compute chunk sizes with validation
    chunk_sizes = zeros(Int, nranks)
    expected_dim_size = dim == 2 ? Ny : Nx
    divisor = dim == 2 ? Nx : Ny

    for r in 1:nranks
        chunk_sizes[r] = counts[r] ÷ divisor
        if counts[r] % divisor != 0
            error("Unpack 2D chunk size error: counts[$r]=$(counts[r]) not divisible by $divisor.")
        end
    end

    chunk_sum = sum(chunk_sizes)
    if chunk_sum != expected_dim_size
        error("Unpack 2D chunk sum ($chunk_sum) != expected ($expected_dim_size) for dim=$dim.")
    end

    buf_len = length(buffer)

    # Precompute rank lookup table
    rank_for_idx = Vector{Int}(undef, expected_dim_size)
    offset_for_idx = Vector{Int}(undef, expected_dim_size)
    cumulative = 0
    for r in 1:nranks
        for j in 1:chunk_sizes[r]
            idx = cumulative + j
            rank_for_idx[idx] = r
            offset_for_idx[idx] = cumulative
        end
        cumulative += chunk_sizes[r]
    end

    for iy in 1:Ny, ix in 1:Nx
        if dim == 2
            rank = rank_for_idx[iy]
            local_iy = iy - offset_for_idx[iy]
            local_idx = (local_iy - 1) * Nx + ix
            buf_idx = displs[rank] + local_idx
        else  # Receiving x-chunks
            rank = rank_for_idx[ix]
            local_ix = ix - offset_for_idx[ix]
            local_idx = (iy - 1) * chunk_sizes[rank] + local_ix
            buf_idx = displs[rank] + local_idx
        end

        if buf_idx < 1 || buf_idx > buf_len
            error("Unpack 2D buffer out of bounds: buf_idx=$buf_idx, len=$buf_len, ix=$ix, iy=$iy")
        end
        @inbounds data[ix, iy] = buffer[buf_idx]
    end
end

function unpack_from_transpose!(data, buffer, counts, displs, dim::Int,
                                nranks::Int, arch::AbstractArchitecture)
    # Default implementation
    data_cpu = on_architecture(CPU(), data)
    buffer_cpu = on_architecture(CPU(), buffer)

    unpack_from_transpose!(data_cpu, buffer_cpu, counts, displs, dim, nranks, CPU())
    copyto!(data, on_architecture(arch, data_cpu))

    return data
end

