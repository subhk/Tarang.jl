# ============================================================================
# GPU Kernels for Distributed Transpose Operations
# ============================================================================

"""
GPU kernels for efficient pack/unpack operations during distributed transposes.
These kernels enable high-performance MPI communication by preparing data
in contiguous buffers suitable for MPI.Alltoallv.

The pack operation reorganizes data from a multidimensional array into a
flat buffer with chunks destined for different MPI ranks.

The unpack operation reverses this, taking received data and placing it
into the appropriate positions in the output array.
"""

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const

# ============================================================================
# Pack Kernels for Transpose
# ============================================================================

"""
    pack_for_transpose_kernel_3d!(buffer, data, Nx, Ny, Nz, nranks, dim, chunk_sizes)

Pack 3D array data into contiguous buffer for MPI.Alltoallv.

Arguments:
- buffer: Output flat buffer
- data: Input 3D array
- Nx, Ny, Nz: Array dimensions
- nranks: Number of MPI ranks in the transpose communicator
- dim: Dimension being redistributed (1=x, 2=y, 3=z)
- chunk_sizes: Array of chunk sizes for each rank
"""
@kernel function pack_for_transpose_kernel_3d!(buffer, @Const(data),
                                               Nx::Int, Ny::Int, Nz::Int,
                                               nranks::Int, dim::Int,
                                               @Const(chunk_sizes), @Const(displs))
    i = @index(Global)

    # Total elements
    total = Nx * Ny * Nz
    if i > total
        return
    end

    # Convert linear index to 3D indices (column-major order)
    ix = ((i - 1) % Nx) + 1
    iy = (((i - 1) ÷ Nx) % Ny) + 1
    iz = ((i - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank this element goes to based on redistributed dimension
    if dim == 3  # Z→Y transpose: z-dimension being redistributed
        # Find which rank owns this z-index after transpose
        rank = 0
        z_offset = 0
        for r in 1:nranks
            if iz <= z_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            z_offset += chunk_sizes[r]
        end

        # Compute position within rank's chunk
        local_iz = iz - z_offset
        local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix

        # Compute buffer position
        buf_idx = displs[rank + 1] + local_idx

    elseif dim == 2  # Y→X transpose: y-dimension being redistributed
        rank = 0
        y_offset = 0
        for r in 1:nranks
            if iy <= y_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            y_offset += chunk_sizes[r]
        end

        local_iy = iy - y_offset
        local_idx = (iz - 1) * Nx * chunk_sizes[rank + 1] + (local_iy - 1) * Nx + ix

        buf_idx = displs[rank + 1] + local_idx

    else  # dim == 1: X being redistributed
        rank = 0
        x_offset = 0
        for r in 1:nranks
            if ix <= x_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            x_offset += chunk_sizes[r]
        end

        local_ix = ix - x_offset
        local_idx = (iz - 1) * chunk_sizes[rank + 1] * Ny + (iy - 1) * chunk_sizes[rank + 1] + local_ix

        buf_idx = displs[rank + 1] + local_idx
    end

    @inbounds buffer[buf_idx] = data[ix, iy, iz]
end

"""
    pack_for_transpose_kernel_2d!(buffer, data, Nx, Ny, nranks, dim, chunk_sizes, displs)

Pack 2D array data into contiguous buffer for MPI.Alltoallv.
"""
@kernel function pack_for_transpose_kernel_2d!(buffer, @Const(data),
                                               Nx::Int, Ny::Int,
                                               nranks::Int, dim::Int,
                                               @Const(chunk_sizes), @Const(displs))
    i = @index(Global)

    total = Nx * Ny
    if i > total
        return
    end

    # Convert linear index to 2D indices
    ix = ((i - 1) % Nx) + 1
    iy = ((i - 1) ÷ Nx) + 1

    if dim == 2  # Y being redistributed
        rank = 0
        y_offset = 0
        for r in 1:nranks
            if iy <= y_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            y_offset += chunk_sizes[r]
        end

        local_iy = iy - y_offset
        local_idx = (local_iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx

    else  # dim == 1: X being redistributed
        rank = 0
        x_offset = 0
        for r in 1:nranks
            if ix <= x_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            x_offset += chunk_sizes[r]
        end

        local_ix = ix - x_offset
        local_idx = (iy - 1) * chunk_sizes[rank + 1] + local_ix
        buf_idx = displs[rank + 1] + local_idx
    end

    @inbounds buffer[buf_idx] = data[ix, iy]
end

# ============================================================================
# Unpack Kernels for Transpose
# ============================================================================

"""
    unpack_from_transpose_kernel_3d!(data, buffer, Nx, Ny, Nz, nranks, dim, chunk_sizes, displs)

Unpack data from flat buffer after MPI.Alltoallv into 3D array.
"""
@kernel function unpack_from_transpose_kernel_3d!(data, @Const(buffer),
                                                  Nx::Int, Ny::Int, Nz::Int,
                                                  nranks::Int, dim::Int,
                                                  @Const(chunk_sizes), @Const(displs))
    i = @index(Global)

    total = Nx * Ny * Nz
    if i > total
        return
    end

    # Convert linear index to 3D indices
    ix = ((i - 1) % Nx) + 1
    iy = (((i - 1) ÷ Nx) % Ny) + 1
    iz = ((i - 1) ÷ (Nx * Ny)) + 1

    # Determine which rank this element came from
    if dim == 2  # After Z→Y transpose: receiving y-chunks from different ranks
        rank = 0
        y_offset = 0
        for r in 1:nranks
            if iy <= y_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            y_offset += chunk_sizes[r]
        end

        local_iy = iy - y_offset
        local_idx = (iz - 1) * Nx * chunk_sizes[rank + 1] + (local_iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx

    elseif dim == 1  # After Y→X transpose: receiving x-chunks
        rank = 0
        x_offset = 0
        for r in 1:nranks
            if ix <= x_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            x_offset += chunk_sizes[r]
        end

        local_ix = ix - x_offset
        local_idx = (iz - 1) * chunk_sizes[rank + 1] * Ny + (iy - 1) * chunk_sizes[rank + 1] + local_ix
        buf_idx = displs[rank + 1] + local_idx

    else  # dim == 3: receiving z-chunks
        rank = 0
        z_offset = 0
        for r in 1:nranks
            if iz <= z_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            z_offset += chunk_sizes[r]
        end

        local_iz = iz - z_offset
        local_idx = (local_iz - 1) * Nx * Ny + (iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx
    end

    @inbounds data[ix, iy, iz] = buffer[buf_idx]
end

"""
    unpack_from_transpose_kernel_2d!(data, buffer, Nx, Ny, nranks, dim, chunk_sizes, displs)

Unpack data from flat buffer after MPI.Alltoallv into 2D array.
"""
@kernel function unpack_from_transpose_kernel_2d!(data, @Const(buffer),
                                                  Nx::Int, Ny::Int,
                                                  nranks::Int, dim::Int,
                                                  @Const(chunk_sizes), @Const(displs))
    i = @index(Global)

    total = Nx * Ny
    if i > total
        return
    end

    ix = ((i - 1) % Nx) + 1
    iy = ((i - 1) ÷ Nx) + 1

    if dim == 1  # Receiving x-chunks
        rank = 0
        x_offset = 0
        for r in 1:nranks
            if ix <= x_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            x_offset += chunk_sizes[r]
        end

        local_ix = ix - x_offset
        local_idx = (iy - 1) * chunk_sizes[rank + 1] + local_ix
        buf_idx = displs[rank + 1] + local_idx

    else  # dim == 2: receiving y-chunks
        rank = 0
        y_offset = 0
        for r in 1:nranks
            if iy <= y_offset + chunk_sizes[r]
                rank = r - 1
                break
            end
            y_offset += chunk_sizes[r]
        end

        local_iy = iy - y_offset
        local_idx = (local_iy - 1) * Nx + ix
        buf_idx = displs[rank + 1] + local_idx
    end

    @inbounds data[ix, iy] = buffer[buf_idx]
end

# ============================================================================
# Simple Copy Kernel (for serial or single-rank cases)
# ============================================================================

"""
    copy_to_buffer_kernel!(buffer, data)

Simple kernel to copy data to flat buffer.
"""
@kernel function copy_to_buffer_kernel!(buffer, @Const(data))
    i = @index(Global)
    if i <= length(data)
        @inbounds buffer[i] = data[i]
    end
end

"""
    copy_from_buffer_kernel!(data, buffer)

Simple kernel to copy from flat buffer to data array.
"""
@kernel function copy_from_buffer_kernel!(data, @Const(buffer))
    i = @index(Global)
    if i <= length(data)
        @inbounds data[i] = buffer[i]
    end
end

# ============================================================================
# GPU FFT in Dimension Kernel Helpers
# ============================================================================

"""
    apply_fft_normalization_kernel!(data, scale)

Apply FFT normalization factor.
"""
@kernel function apply_fft_normalization_kernel!(data, scale)
    i = @index(Global)
    if i <= length(data)
        @inbounds data[i] = data[i] * scale
    end
end

# ============================================================================
# Launcher Functions
# ============================================================================

# Constants for kernel operations
const GPU_PACK_3D_OP = KernelOperation(pack_for_transpose_kernel_3d!) do buffer, data, Nx, Ny, Nz, _, _, _, _
    Nx * Ny * Nz
end

const GPU_UNPACK_3D_OP = KernelOperation(unpack_from_transpose_kernel_3d!) do data, buffer, Nx, Ny, Nz, _, _, _, _
    Nx * Ny * Nz
end

const GPU_PACK_2D_OP = KernelOperation(pack_for_transpose_kernel_2d!) do buffer, data, Nx, Ny, _, _, _, _
    Nx * Ny
end

const GPU_UNPACK_2D_OP = KernelOperation(unpack_from_transpose_kernel_2d!) do data, buffer, Nx, Ny, _, _, _, _
    Nx * Ny
end

"""
    gpu_pack_for_transpose!(buffer, data, counts, displs, dim, nranks)

Launch GPU kernel to pack data for transpose operation.
"""
function gpu_pack_for_transpose!(buffer::CuArray, data::CuArray,
                                 counts::Vector{Int}, displs::Vector{Int},
                                 dim::Int, nranks::Int)
    ndims_data = ndims(data)

    # Ensure we're on the device where data lives for allocations, kernel launch, and sync
    data_device = CUDA.device(data)
    prev_device = CUDA.device()
    CUDA.device!(data_device)

    if ndims_data == 3
        Nx, Ny, Nz = size(data)
        n_elements = Nx * Ny * Nz

        # Compute chunk_sizes from counts (kernel expects chunk size, not total count)
        chunk_sizes = zeros(Int, nranks)
        if dim == 3
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ (Nx * Ny)
            end
        elseif dim == 2
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ (Nx * Nz)
            end
        else  # dim == 1
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ (Ny * Nz)
            end
        end

        chunk_sizes_gpu = CuArray(chunk_sizes)
        displs_gpu = CuArray(displs)

        kernel = pack_for_transpose_kernel_3d!(CUDABackend())
        kernel(buffer, data, Nx, Ny, Nz, nranks, dim, chunk_sizes_gpu, displs_gpu;
               ndrange=n_elements)

    elseif ndims_data == 2
        Nx, Ny = size(data)
        n_elements = Nx * Ny

        # Compute chunk_sizes from counts
        chunk_sizes = zeros(Int, nranks)
        if dim == 2
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ Nx
            end
        else  # dim == 1
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ Ny
            end
        end

        chunk_sizes_gpu = CuArray(chunk_sizes)
        displs_gpu = CuArray(displs)

        kernel = pack_for_transpose_kernel_2d!(CUDABackend())
        kernel(buffer, data, Nx, Ny, nranks, dim, chunk_sizes_gpu, displs_gpu;
               ndrange=n_elements)
    else
        # Fallback: simple copy
        copyto!(view(buffer, 1:length(data)), vec(data))
    end

    CUDA.synchronize()
    CUDA.device!(prev_device)
    return buffer
end

"""
    gpu_unpack_from_transpose!(data, buffer, counts, displs, dim, nranks)

Launch GPU kernel to unpack data after transpose operation.
"""
function gpu_unpack_from_transpose!(data::CuArray, buffer::CuArray,
                                    counts::Vector{Int}, displs::Vector{Int},
                                    dim::Int, nranks::Int)
    ndims_data = ndims(data)

    # Ensure we're on the device where data lives for allocations, kernel launch, and sync
    data_device = CUDA.device(data)
    prev_device = CUDA.device()
    CUDA.device!(data_device)

    if ndims_data == 3
        Nx, Ny, Nz = size(data)
        n_elements = Nx * Ny * Nz

        # Compute chunk_sizes from counts (kernel expects chunk size, not total count)
        chunk_sizes = zeros(Int, nranks)
        if dim == 2  # After Z→Y: receiving y-chunks
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ (Nx * Nz)
            end
        elseif dim == 1  # After Y→X: receiving x-chunks
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ (Ny * Nz)
            end
        else  # dim == 3: receiving z-chunks
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ (Nx * Ny)
            end
        end

        chunk_sizes_gpu = CuArray(chunk_sizes)
        displs_gpu = CuArray(displs)

        kernel = unpack_from_transpose_kernel_3d!(CUDABackend())
        kernel(data, buffer, Nx, Ny, Nz, nranks, dim, chunk_sizes_gpu, displs_gpu;
               ndrange=n_elements)

    elseif ndims_data == 2
        Nx, Ny = size(data)
        n_elements = Nx * Ny

        # Compute chunk_sizes from counts
        chunk_sizes = zeros(Int, nranks)
        if dim == 2  # Receiving y-chunks
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ Nx
            end
        else  # dim == 1: Receiving x-chunks
            for r in 1:nranks
                chunk_sizes[r] = counts[r] ÷ Ny
            end
        end

        chunk_sizes_gpu = CuArray(chunk_sizes)
        displs_gpu = CuArray(displs)

        kernel = unpack_from_transpose_kernel_2d!(CUDABackend())
        kernel(data, buffer, Nx, Ny, nranks, dim, chunk_sizes_gpu, displs_gpu;
               ndrange=n_elements)
    else
        copyto!(vec(data), view(buffer, 1:length(data)))
    end

    CUDA.synchronize()
    CUDA.device!(prev_device)
    return data
end

# ============================================================================
# Override pack/unpack for GPU architecture
# ============================================================================

"""
Override pack_for_transpose! for GPU architecture.
"""
function Tarang.pack_for_transpose!(buffer::CuArray, data::CuArray,
                                    counts::Vector{Int}, displs::Vector{Int},
                                    dim::Int, nranks::Int, arch::Tarang.GPU)
    return gpu_pack_for_transpose!(buffer, data, counts, displs, dim, nranks)
end

"""
Override unpack_from_transpose! for GPU architecture.
"""
function Tarang.unpack_from_transpose!(data::CuArray, buffer::CuArray,
                                       counts::Vector{Int}, displs::Vector{Int},
                                       dim::Int, nranks::Int, arch::Tarang.GPU)
    return gpu_unpack_from_transpose!(data, buffer, counts, displs, dim, nranks)
end

# ============================================================================
# GPU FFT in dimension
# ============================================================================

"""
Override fft_in_dim! for GPU arrays using CUFFT.
"""
function Tarang.fft_in_dim!(data::CuArray, dim::Int, direction::Symbol, arch::Tarang.GPU)
    if direction == :forward
        # Use CUFFT for forward FFT along specified dimension
        plan = CUFFT.plan_fft(data, (dim,))
        data .= plan * data
    else
        # Use CUFFT for inverse FFT along specified dimension
        plan = CUFFT.plan_ifft(data, (dim,))
        data .= plan * data
    end
    CUDA.synchronize()
    return data
end

# ============================================================================
# GPU DCT in dimension (for Chebyshev transforms)
# ============================================================================

"""
Override dct_in_dim! for GPU arrays using GPU DCT kernels.

Uses the GPUDCTPlanDim from dct.jl for efficient GPU-based DCT.
"""
function Tarang.dct_in_dim!(data::CuArray{T,N}, dim::Int, direction::Symbol, arch::Tarang.GPU) where {T,N}
    # Handle complex data by transforming real and imaginary parts separately
    if T <: Complex
        # Use GPU-native real/imag extraction (no host round-trip)
        real_part = real.(data)
        imag_part = imag.(data)

        Tarang.dct_in_dim!(real_part, dim, direction, arch)
        Tarang.dct_in_dim!(imag_part, dim, direction, arch)

        data .= complex.(real_part, imag_part)
        CUDA.synchronize()
        return data
    end

    # Real data path - create DCT plan
    real_T = T <: Complex ? real(T) : T
    full_size = size(data)
    plan = plan_gpu_dct_dim(arch, full_size, real_T, dim)

    # Create output array (can be same as input for in-place)
    output = similar(data)

    if direction == :forward
        gpu_dct_dim!(output, data, plan, Val(:forward))
    else
        gpu_dct_dim!(output, data, plan, Val(:backward))
    end

    # Copy result back to data
    copyto!(data, output)
    CUDA.synchronize()
    return data
end

# ============================================================================
# Exports
# ============================================================================

export pack_for_transpose_kernel_3d!, pack_for_transpose_kernel_2d!
export unpack_from_transpose_kernel_3d!, unpack_from_transpose_kernel_2d!
export copy_to_buffer_kernel!, copy_from_buffer_kernel!
export apply_fft_normalization_kernel!
export gpu_pack_for_transpose!, gpu_unpack_from_transpose!
export GPU_PACK_3D_OP, GPU_UNPACK_3D_OP, GPU_PACK_2D_OP, GPU_UNPACK_2D_OP
