"""
    Transpose Transforms - Distributed FFT operations for TransposableField

This file contains the distributed transform operations that combine
transposes with FFTs for spectral transforms:
- distributed_forward_transform!
- distributed_backward_transform!
- get_fft_plan
- basis-aware transform helpers
"""

# ============================================================================
# Distributed Transform Operations (with overlap support)
# ============================================================================

"""
    distributed_forward_transform!(tf::TransposableField; overlap=false)

Perform complete forward transform with transposes for distributed GPU+MPI.

Automatically selects FFT for Fourier bases and DCT for Chebyshev/Jacobi bases.

# Arguments
- `tf`: The TransposableField
- `overlap`: If true, use async transposes with computation overlap

Algorithm (3D Fourier-Fourier-Fourier):
1. Start in ZLocal layout
2. FFT/DCT in z (local)
3. Transpose Z→Y (with optional overlap)
4. FFT/DCT in y (local)
5. Transpose Y→X (with optional overlap)
6. FFT/DCT in x (local)

Supports mixed basis types (e.g., Chebyshev-Fourier).
"""
function distributed_forward_transform!(tf::TransposableField{F,T,N};
                                        overlap::Bool=false,
                                        plans=nothing) where {F,T,N}
    arch = tf.buffers.architecture

    # Validate field size matches TransposableField expected shape
    expected_size = size(tf.buffers.z_local_data)
    actual_size = size(tf.field["g"])
    if expected_size != actual_size
        error("Field size mismatch in distributed_forward_transform!: " *
              "TransposableField expects $expected_size but field['g'] has $actual_size. " *
              "Ensure the Field was allocated with the same Distributor and grid dimensions.")
    end

    # Get bases for each dimension
    xbasis = get_basis_for_dim(tf, 1)
    ybasis = get_basis_for_dim(tf, 2)
    zbasis = N >= 3 ? get_basis_for_dim(tf, 3) : nothing

    # Ensure we start in ZLocal layout with data from field
    tf.buffers.active_layout[] = ZLocal
    copyto!(vec(tf.buffers.z_local_data), vec(tf.field["g"]))

    if N >= 3
        # 3D case with full transpose sequence
        fft_start = time()

        # Step 1: Transform in z (local in ZLocal layout)
        z_plan = get(tf.fft_plans, ZLocal, nothing)
        transform_in_dim!(tf.buffers.z_local_data, 3, :forward, zbasis, arch; plan=z_plan)

        if overlap && tf.topology.row_size > 1
            # Async transpose with overlap pattern
            async_transpose_z_to_y!(tf)
            # Could do other work here while transpose is in progress
            wait_transpose!(tf)
        else
            transpose_z_to_y!(tf)
        end

        # Step 3: Transform in y (local in YLocal layout)
        y_plan = get(tf.fft_plans, YLocal, nothing)
        transform_in_dim!(tf.buffers.y_local_data, 2, :forward, ybasis, arch; plan=y_plan)

        if overlap && tf.topology.col_size > 1
            async_transpose_y_to_x!(tf)
            wait_transpose!(tf)
        else
            transpose_y_to_x!(tf)
        end

        # Step 5: Transform in x (local in XLocal layout)
        x_plan = get(tf.fft_plans, XLocal, nothing)
        transform_in_dim!(tf.buffers.x_local_data, 1, :forward, xbasis, arch; plan=x_plan)

        tf.total_fft_time += time() - fft_start

        # Step 6: Transpose back to ZLocal layout to match field["c"] allocation
        # (field["c"] is allocated with ZLocal decomposition via get_local_array_size)
        transpose_x_to_y!(tf)
        transpose_y_to_z!(tf)

        # Copy result to field - now both are in ZLocal layout
        copyto!(vec(tf.field["c"]), vec(tf.buffers.z_local_data))

    elseif N == 2
        topo = tf.topology
        Rx, Ry = topo.Rx, topo.Ry
        fft_start = time()

        # Retrieve cached plans for 2D
        y_plan = get(tf.fft_plans, YLocal, nothing)
        x_plan = get(tf.fft_plans, XLocal, nothing)

        if Rx > 1 && Ry > 1
            # True 2D mesh on 2D domain
            if overlap
                @warn "Async overlap is not supported for 2D mesh on 2D domain (Rx=$Rx, Ry=$Ry). Using blocking transposes." maxlog=1
            end

            transpose_z_to_y!(tf)
            transform_in_dim!(tf.buffers.y_local_data, 2, :forward, ybasis, arch; plan=y_plan)
            transpose_y_to_x!(tf)
            transform_in_dim!(tf.buffers.x_local_data, 1, :forward, xbasis, arch; plan=x_plan)

            tf.total_fft_time += time() - fft_start

            transpose_x_to_y!(tf)
            transpose_y_to_z!(tf)

            copyto!(vec(tf.field["c"]), vec(tf.buffers.z_local_data))
        elseif Ry > 1
            # 1D decomposition with Ry>1
            transform_in_dim!(tf.buffers.z_local_data, 1, :forward, xbasis, arch; plan=x_plan)
            transpose_z_to_y!(tf)
            transform_in_dim!(tf.buffers.y_local_data, 2, :forward, ybasis, arch; plan=y_plan)

            tf.total_fft_time += time() - fft_start

            transpose_y_to_z!(tf)
            copyto!(vec(tf.field["c"]), vec(tf.buffers.z_local_data))
        else
            # 1D decomposition with Rx>1
            transform_in_dim!(tf.buffers.z_local_data, 2, :forward, ybasis, arch; plan=y_plan)
            transpose_z_to_y!(tf)
            transform_in_dim!(tf.buffers.y_local_data, 1, :forward, xbasis, arch; plan=x_plan)

            tf.total_fft_time += time() - fft_start

            transpose_y_to_z!(tf)
            copyto!(vec(tf.field["c"]), vec(tf.buffers.z_local_data))
        end

    else
        # 1D case - no transpose needed
        fft_start = time()
        x_plan = get(tf.fft_plans, XLocal, nothing)
        transform_in_dim!(tf.buffers.z_local_data, 1, :forward, xbasis, arch; plan=x_plan)
        tf.total_fft_time += time() - fft_start
        copyto!(vec(tf.field["c"]), vec(tf.buffers.z_local_data))
    end

    return tf
end

"""
    distributed_backward_transform!(tf::TransposableField; overlap=false)

Perform complete backward transform with transposes for distributed GPU+MPI.

Automatically selects IFFT for Fourier bases and inverse DCT for Chebyshev/Jacobi bases.
"""
function distributed_backward_transform!(tf::TransposableField{F,T,N};
                                         overlap::Bool=false,
                                         plans=nothing) where {F,T,N}
    arch = tf.buffers.architecture

    # Validate field size matches TransposableField expected shape
    # For spectral data, use coefficient-space buffer for validation
    expected_size = size(tf.buffers.z_local_data)
    actual_size = size(tf.field["c"])
    if expected_size != actual_size
        error("Field size mismatch in distributed_backward_transform!: " *
              "TransposableField expects $expected_size but field['c'] has $actual_size. " *
              "Ensure the Field was allocated with the same Distributor and grid dimensions.")
    end

    # Get bases for each dimension
    xbasis = get_basis_for_dim(tf, 1)
    ybasis = get_basis_for_dim(tf, 2)
    zbasis = N >= 3 ? get_basis_for_dim(tf, 3) : nothing

    if N >= 3
        # Start in ZLocal layout with spectral data (field["c"] is stored in ZLocal)
        tf.buffers.active_layout[] = ZLocal
        copyto!(vec(tf.buffers.z_local_data), vec(tf.field["c"]))

        fft_start = time()

        # Step 0: Transpose Z→Y→X to get to XLocal for inverse transforms
        transpose_z_to_y!(tf)
        transpose_y_to_x!(tf)

        # Step 1: Inverse transform in x (local in XLocal layout)
        x_plan = get(tf.fft_plans, XLocal, nothing)
        transform_in_dim!(tf.buffers.x_local_data, 1, :backward, xbasis, arch; plan=x_plan)

        # Step 2: Transpose X→Y
        transpose_x_to_y!(tf)

        # Step 3: Inverse transform in y (local in YLocal layout)
        y_plan = get(tf.fft_plans, YLocal, nothing)
        transform_in_dim!(tf.buffers.y_local_data, 2, :backward, ybasis, arch; plan=y_plan)

        # Step 4: Transpose Y→Z
        transpose_y_to_z!(tf)

        # Step 5: Inverse transform in z (local in ZLocal layout)
        z_plan = get(tf.fft_plans, ZLocal, nothing)
        transform_in_dim!(tf.buffers.z_local_data, 3, :backward, zbasis, arch; plan=z_plan)

        tf.total_fft_time += time() - fft_start

        # Copy result to field - preserve complex values if field dtype is complex
        if tf.field.dtype <: Complex
            copyto!(vec(tf.field["g"]), vec(tf.buffers.z_local_data))
        else
            copyto!(vec(tf.field["g"]), real.(vec(tf.buffers.z_local_data)))
        end

    elseif N == 2
        topo = tf.topology
        Rx, Ry = topo.Rx, topo.Ry
        fft_start = time()

        if Rx > 1 && Ry > 1
            # True 2D mesh on 2D domain
            # XLocal=(Nx, Ny/Rx) → YLocal=(Nx/Rx, Ny) → ZLocal=(Nx/Rx, Ny/Ry)
            #
            # NOTE: Async overlap is NOT supported for 2D mesh on 2D domain because
            # Y→Z uses Allgatherv (reverse direction), and async Allgatherv is not implemented.
            if overlap
                @warn "Async overlap is not supported for 2D mesh on 2D domain (Rx=$Rx, Ry=$Ry). Using blocking transposes." maxlog=1
            end

            # Start in ZLocal with spectral data (field["c"] is stored in ZLocal)
            tf.buffers.active_layout[] = ZLocal
            copyto!(vec(tf.buffers.z_local_data), vec(tf.field["c"]))

            # Step 0: Transpose Z→Y→X to get to XLocal for inverse transforms
            transpose_z_to_y!(tf)
            transpose_y_to_x!(tf)

            # Retrieve cached plans for 2D backward
            y_plan = get(tf.fft_plans, YLocal, nothing)
            x_plan = get(tf.fft_plans, XLocal, nothing)

            # Step 1: Inverse transform in x (dim 1, local in XLocal)
            transform_in_dim!(tf.buffers.x_local_data, 1, :backward, xbasis, arch; plan=x_plan)

            # Step 2: Transpose X→Y (Alltoallv - blocking)
            transpose_x_to_y!(tf)

            # Step 3: Inverse transform in y (dim 2, local in YLocal)
            transform_in_dim!(tf.buffers.y_local_data, 2, :backward, ybasis, arch; plan=y_plan)

            # Step 4: Transpose Y→Z (local extraction)
            transpose_y_to_z!(tf)

            tf.total_fft_time += time() - fft_start
            if tf.field.dtype <: Complex
                copyto!(vec(tf.field["g"]), vec(tf.buffers.z_local_data))
            else
                copyto!(vec(tf.field["g"]), real.(vec(tf.buffers.z_local_data)))
            end
        elseif Ry > 1
            # 1D decomposition with Ry>1
            tf.buffers.active_layout[] = ZLocal
            copyto!(vec(tf.buffers.z_local_data), vec(tf.field["c"]))

            y_plan = get(tf.fft_plans, YLocal, nothing)
            x_plan = get(tf.fft_plans, XLocal, nothing)

            transpose_z_to_y!(tf)
            transform_in_dim!(tf.buffers.y_local_data, 2, :backward, ybasis, arch; plan=y_plan)
            transpose_y_to_z!(tf)
            transform_in_dim!(tf.buffers.z_local_data, 1, :backward, xbasis, arch; plan=x_plan)

            tf.total_fft_time += time() - fft_start
            if tf.field.dtype <: Complex
                copyto!(vec(tf.field["g"]), vec(tf.buffers.z_local_data))
            else
                copyto!(vec(tf.field["g"]), real.(vec(tf.buffers.z_local_data)))
            end
        else
            # 1D decomposition with Rx>1
            tf.buffers.active_layout[] = ZLocal
            copyto!(vec(tf.buffers.z_local_data), vec(tf.field["c"]))

            y_plan = get(tf.fft_plans, YLocal, nothing)
            x_plan = get(tf.fft_plans, XLocal, nothing)

            transpose_z_to_y!(tf)
            transform_in_dim!(tf.buffers.y_local_data, 1, :backward, xbasis, arch; plan=x_plan)
            transpose_y_to_z!(tf)
            transform_in_dim!(tf.buffers.z_local_data, 2, :backward, ybasis, arch; plan=y_plan)

            tf.total_fft_time += time() - fft_start
            if tf.field.dtype <: Complex
                copyto!(vec(tf.field["g"]), vec(tf.buffers.z_local_data))
            else
                copyto!(vec(tf.field["g"]), real.(vec(tf.buffers.z_local_data)))
            end
        end

    else
        tf.buffers.active_layout[] = ZLocal
        copyto!(vec(tf.buffers.z_local_data), vec(tf.field["c"]))

        fft_start = time()
        x_plan = get(tf.fft_plans, XLocal, nothing)
        transform_in_dim!(tf.buffers.z_local_data, 1, :backward, xbasis, arch; plan=x_plan)
        tf.total_fft_time += time() - fft_start

        # Preserve complex values if field dtype is complex
        if tf.field.dtype <: Complex
            copyto!(vec(tf.field["g"]), vec(tf.buffers.z_local_data))
        else
            copyto!(vec(tf.field["g"]), real.(vec(tf.buffers.z_local_data)))
        end
    end

    return tf
end

# ============================================================================
# Basis-Aware Transform Helpers
# ============================================================================

"""
    transform_in_dim!(data, dim, direction, basis, arch)

Perform spectral transform along specified dimension, choosing FFT or DCT
based on the basis type.

# Arguments
- `data`: Array to transform (modified in-place)
- `dim`: Dimension along which to transform (1, 2, or 3)
- `direction`: `:forward` (physical→spectral) or `:backward` (spectral→physical)
- `basis`: The Basis object for this dimension (Fourier, ChebyshevT, etc.)
- `arch`: Architecture (CPU() or GPU())
"""
function transform_in_dim!(data, dim::Int, direction::Symbol, basis::Basis, arch::AbstractArchitecture;
                          plan=nothing)
    if basis isa FourierBasis
        fft_in_dim!(data, dim, direction, arch; plan=plan)
    elseif basis isa JacobiBasis  # Includes ChebyshevT, ChebyshevU, Legendre, etc.
        dct_in_dim!(data, dim, direction, arch)
    else
        # Fallback to FFT for unknown basis types
        @warn "Unknown basis type $(typeof(basis)), falling back to FFT"
        fft_in_dim!(data, dim, direction, arch)
    end
    return data
end

# Convenience method when no basis is specified (defaults to FFT)
function transform_in_dim!(data, dim::Int, direction::Symbol, ::Nothing, arch::AbstractArchitecture;
                          plan=nothing)
    fft_in_dim!(data, dim, direction, arch; plan=plan)
end

"""
    fft_in_dim!(data, dim, direction, arch::CPU)

Perform FFT along specified dimension (CPU version using FFTW).
"""
function fft_in_dim!(data, dim::Int, direction::Symbol, arch::CPU; plan=nothing)
    if plan !== nothing && plan !== :gpu_pending
        # Use precomputed FFTW plan for better performance (MEASURE vs ESTIMATE)
        if direction == :forward
            data .= plan * data
        else
            data .= plan \ data
        end
    elseif direction == :forward
        data .= FFTW.fft(data, dim)
    else
        data .= FFTW.ifft(data, dim)
    end
    return data
end

function fft_in_dim!(data, dim::Int, direction::Symbol, arch::AbstractArchitecture; plan=nothing)
    # Default: fall back to CPU (GPU plans are handled by CUDA extension)
    data_cpu = on_architecture(CPU(), data)
    fft_in_dim!(data_cpu, dim, direction, CPU(); plan=plan)
    copyto!(data, on_architecture(arch, data_cpu))
    return data
end

"""
    dct_in_dim!(data, dim, direction, arch::CPU)

Perform DCT along specified dimension for Chebyshev transforms (CPU version).

Uses DCT-I (FFTW.REDFT00) to match the Gauss-Lobatto grid used by
ChebyshevTransform in transform_chebyshev.jl:
- Forward: DCT-I with 1/(N-1) normalization and endpoint halving
- Backward: DCT-I (self-inverse up to normalization) with endpoint doubling

This ensures serial and distributed transform results agree exactly.
"""
function dct_in_dim!(data, dim::Int, direction::Symbol, arch::CPU; grid_size::Union{Nothing,Int}=nothing)
    n = size(data, dim)

    # Handle complex data by transforming real and imaginary parts separately
    # (same approach as _chebyshev_forward in transforms.jl)
    if eltype(data) <: Complex
        real_part = real.(data)
        imag_part = imag.(data)

        dct_in_dim!(real_part, dim, direction, arch; grid_size=grid_size)
        dct_in_dim!(imag_part, dim, direction, arch; grid_size=grid_size)

        data .= complex.(real_part, imag_part)
        return data
    end

    # Real data path - matches ChebyshevTransform DCT-I scaling
    if direction == :forward
        # DCT-I: grid → coefficients (matches _chebyshev_forward)
        result = FFTW.r2r(data, FFTW.REDFT00, (dim,))

        # DCT-I normalization: divide by (N-1), half-weight at endpoints
        norm_factor = n > 1 ? 1.0 / (n - 1) : 1.0
        result .*= norm_factor
        _apply_dct_scaling!(result, dim, 0.5, 1.0)  # half DC
        # Half the last coefficient (DCT-I endpoint convention) only if it IS
        # the physical DCT-I endpoint. When data is truncated (dealiasing),
        # the last stored coefficient is NOT the endpoint and should not be halved.
        n_dim = size(result, dim)
        if n_dim > 1 && (grid_size === nothing || n_dim == grid_size)
            last_idx = ntuple(i -> i == dim ? (n_dim:n_dim) : Colon(), ndims(result))
            result[last_idx...] .*= 0.5
        end
        data .= result
    else
        # DCT-I backward: undo endpoint halving, then apply DCT-I and divide by 2
        scaled_data = copy(data)
        # Double DC (first coefficient)
        first_idx = ntuple(i -> i == dim ? (1:1) : Colon(), ndims(scaled_data))
        scaled_data[first_idx...] .*= 2.0
        # Only double last coefficient if it IS the physical DCT-I endpoint
        # (i.e., data has not been truncated by dealiasing: n == grid_size)
        # When grid_size is unknown (nothing), assume no truncation for backward compat.
        if n > 1 && (grid_size === nothing || n == grid_size)
            last_idx = ntuple(i -> i == dim ? (n:n) : Colon(), ndims(scaled_data))
            scaled_data[last_idx...] .*= 2.0
        end

        result = FFTW.r2r(scaled_data, FFTW.REDFT00, (dim,))
        # DCT-I(DCT-I(x)) = 2(N-1)*x, forward divided by (N-1), so divide by 2
        result ./= 2.0
        data .= result
    end

    return data
end

"""
Apply scaling along a dimension (used for DCT normalization).
Matches the _scale_along_axis! pattern in transforms.jl.
"""
function _apply_dct_scaling!(data, dim::Int, scale_zero::Float64, scale_pos::Float64)
    n = size(data, dim)
    # Build scale vector: [scale_zero, scale_pos, scale_pos, ...]
    scale_vec = fill(scale_pos, n)
    scale_vec[1] = scale_zero

    # Reshape for broadcasting along the specified dimension
    shape = ntuple(i -> i == dim ? n : 1, ndims(data))
    data .*= reshape(scale_vec, shape...)
end

function dct_in_dim!(data, dim::Int, direction::Symbol, arch::AbstractArchitecture; grid_size::Union{Nothing,Int}=nothing)
    # Default: fall back to CPU
    data_cpu = on_architecture(CPU(), data)
    dct_in_dim!(data_cpu, dim, direction, CPU(); grid_size=grid_size)
    copyto!(data, on_architecture(arch, data_cpu))
    return data
end

"""
    get_basis_for_dim(tf::TransposableField, dim::Int)

Get the basis for a specific dimension from the field.
Returns nothing if bases are not available.
"""
function get_basis_for_dim(tf::TransposableField, dim::Int)
    if isempty(tf.field.bases)
        return nothing
    end
    if dim > length(tf.field.bases)
        return nothing
    end
    return tf.field.bases[dim]
end

