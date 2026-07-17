# ============================================================================
# Distributed DCT for Multi-GPU Chebyshev Transforms
# ============================================================================

"""
    DistributedDCTPlan{T}

Plan for a distributed 3D **basis-aware** spectral transform across multiple GPUs
using pencil decomposition. `T` is the real element type (e.g. `Float64`); all
buffers are `Complex{T}` (complex-everywhere — design decision #1).

Per-axis dispatch (design decision #2): each axis is transformed by a primitive
chosen from `axis_kind`:
- `:chebyshev`       → `local_dct1_along_dim!` (complex DCT-I)
- `:complex_fourier` / `:real_fourier` → `local_fft_along_dim!` (C2C FFT)

Chebyshev axes need NO stored local plan — `gpu_dct1_along_dim!` builds/caches its
`GPUChebyshevDerivPlan` internally (keyed by (n, batch, T)). Fourier axes use
inline CUFFT plans inside `local_fft_along_dim!`. So this struct carries no
per-axis local-plan store, only `axis_kind`.

Transform sequence (forward, from Z-pencil grid values):
1. transform in Z (local) → transpose Z→Y → transform in Y (local)
   → transpose Y→X → transform in X (local)
2. if `axis_kind[1] == :real_fourier`: truncate dim 1 to the half-spectrum
   `div(Nx,2)+1` (dim 1 is LOCAL here, in X-pencil)
3. OUTPUT ADAPTER (design decision #4): transpose X→Y→Z so coeffs land **Z-local**,
   matching the field's distributed coeff buffer (dims 1,2 decomposed, dim 3 local,
   complex, half-spectrum on dim 1). This transpose-back runs on `coeff_pencil`,
   a coeff-sized pencil whose dim-1 global length is the (possibly truncated)
   half-spectrum length — see `build_coeff_pencil`.

Backward mirrors this: Z-local coeffs → transpose Z→Y→X (coeff_pencil) →
Hermitian-expand dim 1 if RealFourier → inverse X → transpose X→Y → inverse Y →
transpose Y→Z → inverse Z → `real.()` for a real grid.

NOTE: this DOUBLES the number of transposes vs. a one-way pipeline (correctness
over performance for v1 — design decision #4).
"""
struct DistributedDCTPlan{T}
    # Full-spectrum pencil decomposition (dim 1 length = Nx)
    pencil::PencilDecomposition

    # Coeff-sized pencil for the transpose-back / transpose-in (dim 1 length =
    # div(Nx,2)+1 for a RealFourier dim-1 axis, else Nx). Shares pencil's comms.
    coeff_pencil::PencilDecomposition

    # Per-axis basis classification (drives local_transform_along_dim!)
    axis_kind::NTuple{3, Symbol}

    # NCCL transpose buffer (complex-everywhere)
    transpose_buffer::NCCLTransposeBuffer{Complex{T}}

    # Work arrays for local transforms, one per pencil orientation (complex)
    work_arrays::Vector{CuArray{Complex{T}, 3}}
end

"""
    DistributedDCTPlan(pencil, bases::Tuple, ::Type{T})
    DistributedDCTPlan(pencil, ::Type{T})            # legacy: all-Chebyshev

Create a distributed transform plan for `pencil`.

The 3-arg form classifies the axes via `Tarang.axis_kinds(bases)`. The legacy
2-arg form assumes all axes are Chebyshev (the original all-axes DCT behaviour);
it is kept for the existing draft GPU tests.

# Arguments
- `pencil`: full-spectrum pencil decomposition describing the domain layout
- `bases`: tuple of basis objects (RealFourier / ComplexFourier / ChebyshevT)
- `T`: real element type of the transform (e.g. Float64, Float32)
"""
function DistributedDCTPlan(pencil::PencilDecomposition, bases::Tuple, ::Type{T}) where T
    return _build_distributed_dct_plan(pencil, Tarang.axis_kinds(bases), T)
end

function DistributedDCTPlan(pencil::PencilDecomposition, ::Type{T}) where T
    return _build_distributed_dct_plan(pencil, (:chebyshev, :chebyshev, :chebyshev), T)
end

function _build_distributed_dct_plan(pencil::PencilDecomposition,
                                     axis_kind::NTuple{3, Symbol}, ::Type{T}) where T
    # Verify NCCL is available — required for multi-GPU transposes
    if !Tarang.nccl_available() && !Tarang._try_load_nccl()
        error("DistributedDCTPlan requires NCCL.jl for multi-GPU Chebyshev transposes. " *
              "Install NCCL.jl: `using Pkg; Pkg.add(\"NCCL\")`, or use " *
              "device=CPU() with MPI (PencilArrays) for distributed Chebyshev transforms.")
    end

    Nx, Ny, Nz = pencil.global_shape

    # Coeff-sized pencil: dim 1 is the half-spectrum length for a RealFourier
    # dim-1 axis (the framework's coeff convention), otherwise the full Nx.
    coeff_Nx = axis_kind[1] === :real_fourier ? (div(Nx, 2) + 1) : Nx
    coeff_pencil = build_coeff_pencil(pencil, (coeff_Nx, Ny, Nz))

    # Complex-everywhere transpose buffer (sized for the full pencil — the coeff
    # pencil's shapes are never larger, so the same buffer serves both).
    transpose_buffer = NCCLTransposeBuffer(pencil, Complex{T})

    # Complex work arrays, one per (full-spectrum) pencil orientation.
    work_arrays = CuArray{Complex{T}, 3}[
        CUDA.zeros(Complex{T}, pencil.x_pencil_shape...),
        CUDA.zeros(Complex{T}, pencil.y_pencil_shape...),
        CUDA.zeros(Complex{T}, pencil.z_pencil_shape...)
    ]

    return DistributedDCTPlan{T}(pencil, coeff_pencil, axis_kind, transpose_buffer, work_arrays)
end

"""
    local_transform_along_dim!(output, input, plan::DistributedDCTPlan, dim, direction)

Per-axis dispatch (design decision #2): apply the basis-appropriate local 1D
transform along `dim` of a 3D complex array, selected by `plan.axis_kind[dim]`.
"""
function local_transform_along_dim!(output, input, plan::DistributedDCTPlan, dim::Int, direction::Symbol)
    k = plan.axis_kind[dim]
    if k === :chebyshev
        return local_dct1_along_dim!(output, input, dim, direction)
    elseif k === :complex_fourier || k === :real_fourier
        return local_fft_along_dim!(output, input, dim, direction)
    else
        error("unknown axis_kind $k at dim $dim")
    end
end

# ============================================================================
# FFT-based Batched DCT for 3D Arrays
# ============================================================================

# Cache for batched R2C / C2R plans (keyed by device, shape, dim)
const _BATCHED_RFFT_CACHE = Dict{Tuple{Int, Tuple, Int}, Any}()
const _BATCHED_IRFFT_CACHE = Dict{Tuple{Int, Tuple, Int, Int}, Any}()
const _BATCHED_DCT_CACHE_LOCK = ReentrantLock()

function _get_batched_rfft_plan(shape::Tuple, dim::Int, ::Type{T}) where T
    device_id = _current_device_id()
    key = (device_id, shape, dim)
    lock(_BATCHED_DCT_CACHE_LOCK)
    try
        if !haskey(_BATCHED_RFFT_CACHE, key)
            dummy = CUDA.zeros(T, shape...)
            _BATCHED_RFFT_CACHE[key] = CUFFT.plan_rfft(dummy, (dim,))
        end
        return _BATCHED_RFFT_CACHE[key]
    finally
        unlock(_BATCHED_DCT_CACHE_LOCK)
    end
end

function _get_batched_irfft_plan(shape::Tuple, dim::Int, n_out::Int, ::Type{T}) where T
    device_id = _current_device_id()
    key = (device_id, shape, dim, n_out)
    lock(_BATCHED_DCT_CACHE_LOCK)
    try
        if !haskey(_BATCHED_IRFFT_CACHE, key)
            dummy = CUDA.zeros(Complex{T}, shape...)
            _BATCHED_IRFFT_CACHE[key] = CUFFT.plan_irfft(dummy, n_out, (dim,))
        end
        return _BATCHED_IRFFT_CACHE[key]
    finally
        unlock(_BATCHED_DCT_CACHE_LOCK)
    end
end

"""
3D forward twiddle kernel along dimension 1.
Applies twiddle factors to R2C FFT output along dim 1 to produce DCT coefficients.
Each thread handles one (k_freq, j, k_z) element of the FFT output.
"""
@kernel function twiddle_3d_dim1_forward_kernel!(output, @Const(fft_out), @Const(twiddle),
                                                  scale_zero, scale_pos, N, Ny, Nz)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    # Map to (freq, j, k)
    freq = ((idx - 1) % half_N_plus1) + 1  # 1-indexed frequency
    j = (((idx - 1) ÷ half_N_plus1) % Ny) + 1
    k = ((idx - 1) ÷ (half_N_plus1 * Ny)) + 1

    if freq <= half_N_plus1 && j <= Ny && k <= Nz
        @inbounds begin
            twiddled = twiddle[freq] * fft_out[freq, j, k]
            if freq == 1
                output[1, j, k] = real(twiddled) * scale_zero
            elseif freq == half_N_plus1
                output[freq, j, k] = real(twiddled) * scale_pos
            else
                output[freq, j, k] = real(twiddled) * scale_pos
                output[N - freq + 2, j, k] = -imag(twiddled) * scale_pos
            end
        end
    end
end

"""
3D forward twiddle kernel along dimension 2.
"""
@kernel function twiddle_3d_dim2_forward_kernel!(output, @Const(fft_out), @Const(twiddle),
                                                  scale_zero, scale_pos, N, Nx, Nz)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    i = ((idx - 1) % Nx) + 1
    freq = (((idx - 1) ÷ Nx) % half_N_plus1) + 1
    k = ((idx - 1) ÷ (Nx * half_N_plus1)) + 1

    if i <= Nx && freq <= half_N_plus1 && k <= Nz
        @inbounds begin
            twiddled = twiddle[freq] * fft_out[i, freq, k]
            if freq == 1
                output[i, 1, k] = real(twiddled) * scale_zero
            elseif freq == half_N_plus1
                output[i, freq, k] = real(twiddled) * scale_pos
            else
                output[i, freq, k] = real(twiddled) * scale_pos
                output[i, N - freq + 2, k] = -imag(twiddled) * scale_pos
            end
        end
    end
end

"""
3D forward twiddle kernel along dimension 3.
"""
@kernel function twiddle_3d_dim3_forward_kernel!(output, @Const(fft_out), @Const(twiddle),
                                                  scale_zero, scale_pos, N, Nx, Ny)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    freq = ((idx - 1) ÷ (Nx * Ny)) + 1

    if i <= Nx && j <= Ny && freq <= half_N_plus1
        @inbounds begin
            twiddled = twiddle[freq] * fft_out[i, j, freq]
            if freq == 1
                output[i, j, 1] = real(twiddled) * scale_zero
            elseif freq == half_N_plus1
                output[i, j, freq] = real(twiddled) * scale_pos
            else
                output[i, j, freq] = real(twiddled) * scale_pos
                output[i, j, N - freq + 2] = -imag(twiddled) * scale_pos
            end
        end
    end
end

"""
3D backward (inverse) twiddle kernel along dimension 1.
Prepares complex array for C2R IFFT from DCT coefficients.
"""
@kernel function twiddle_3d_dim1_backward_kernel!(complex_out, @Const(coeffs), @Const(twiddle_inv),
                                                    scale_zero, scale_pos, N, Ny, Nz)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    freq = ((idx - 1) % half_N_plus1) + 1
    j = (((idx - 1) ÷ half_N_plus1) % Ny) + 1
    k = ((idx - 1) ÷ (half_N_plus1 * Ny)) + 1

    if freq <= half_N_plus1 && j <= Ny && k <= Nz
        @inbounds begin
            if freq == 1
                complex_out[1, j, k] = coeffs[1, j, k] * scale_zero * twiddle_inv[1]
            elseif freq == half_N_plus1
                complex_out[freq, j, k] = coeffs[freq, j, k] * scale_pos * twiddle_inv[freq]
            else
                sr = coeffs[freq, j, k] * scale_pos
                si = -coeffs[N - freq + 2, j, k] * scale_pos
                complex_out[freq, j, k] = Complex(sr, si) * twiddle_inv[freq]
            end
        end
    end
end

"""
3D backward (inverse) twiddle kernel along dimension 2.
"""
@kernel function twiddle_3d_dim2_backward_kernel!(complex_out, @Const(coeffs), @Const(twiddle_inv),
                                                    scale_zero, scale_pos, N, Nx, Nz)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    i = ((idx - 1) % Nx) + 1
    freq = (((idx - 1) ÷ Nx) % half_N_plus1) + 1
    k = ((idx - 1) ÷ (Nx * half_N_plus1)) + 1

    if i <= Nx && freq <= half_N_plus1 && k <= Nz
        @inbounds begin
            if freq == 1
                complex_out[i, 1, k] = coeffs[i, 1, k] * scale_zero * twiddle_inv[1]
            elseif freq == half_N_plus1
                complex_out[i, freq, k] = coeffs[i, freq, k] * scale_pos * twiddle_inv[freq]
            else
                sr = coeffs[i, freq, k] * scale_pos
                si = -coeffs[i, N - freq + 2, k] * scale_pos
                complex_out[i, freq, k] = Complex(sr, si) * twiddle_inv[freq]
            end
        end
    end
end

"""
3D backward (inverse) twiddle kernel along dimension 3.
"""
@kernel function twiddle_3d_dim3_backward_kernel!(complex_out, @Const(coeffs), @Const(twiddle_inv),
                                                    scale_zero, scale_pos, N, Nx, Ny)
    idx = @index(Global)
    half_N_plus1 = N ÷ 2 + 1
    i = ((idx - 1) % Nx) + 1
    j = (((idx - 1) ÷ Nx) % Ny) + 1
    freq = ((idx - 1) ÷ (Nx * Ny)) + 1

    if i <= Nx && j <= Ny && freq <= half_N_plus1
        @inbounds begin
            if freq == 1
                complex_out[i, j, 1] = coeffs[i, j, 1] * scale_zero * twiddle_inv[1]
            elseif freq == half_N_plus1
                complex_out[i, j, freq] = coeffs[i, j, freq] * scale_pos * twiddle_inv[freq]
            else
                sr = coeffs[i, j, freq] * scale_pos
                si = -coeffs[i, j, N - freq + 2] * scale_pos
                complex_out[i, j, freq] = Complex(sr, si) * twiddle_inv[freq]
            end
        end
    end
end

"""
    local_dct_along_dim!(output, input, dct_plan, dim, direction)

Apply local 1D DCT/IDCT along specified dimension of 3D array.
Uses FFT-based O(N log N) algorithm via:
  Forward: even-odd reorder → batched R2C FFT → twiddle factors
  Backward: inverse twiddle → batched C2R IFFT → inverse reorder

# Arguments
- `output`: Output 3D array for DCT coefficients or grid values
- `input`: Input 3D array
- `dct_plan`: OptimizedGPUDCTPlan for the transform dimension
- `dim`: Dimension along which to transform (1, 2, or 3)
- `direction`: Transform direction (:forward or :backward)

# Returns
- Output array with transform applied along specified dimension
"""
function local_dct_along_dim!(output::CuArray{T, 3}, input::CuArray{T, 3},
                               dct_plan::OptimizedGPUDCTPlan{T}, dim::Int,
                               direction::Symbol) where T
    Nx, Ny, Nz = size(input)
    N = size(input, dim)
    @assert size(output) == size(input) "Output and input must have same size"

    arch = Tarang.architecture(input)
    half_N_plus1 = N ÷ 2 + 1

    forward_scale_zero = dct_plan.forward_scale_zero
    forward_scale_pos = dct_plan.forward_scale_pos
    backward_scale_zero = dct_plan.backward_scale_zero
    backward_scale_pos = dct_plan.backward_scale_pos

    if direction == :forward
        # Step 1: Even-odd reorder
        work = similar(input)
        reorder_for_dct_dim!(work, input, dim)

        # Step 2: Batched R2C FFT along dim
        rfft_plan = _get_batched_rfft_plan(size(work), dim, T)
        fft_out = rfft_plan * work

        # Step 3: Apply twiddle factors to extract DCT coefficients
        fft_total = prod(size(fft_out))
        if dim == 1
            launch!(arch, twiddle_3d_dim1_forward_kernel!, output, fft_out, dct_plan.twiddle,
                    forward_scale_zero, forward_scale_pos, N, Ny, Nz;
                    ndrange=fft_total)
        elseif dim == 2
            launch!(arch, twiddle_3d_dim2_forward_kernel!, output, fft_out, dct_plan.twiddle,
                    forward_scale_zero, forward_scale_pos, N, Nx, Nz;
                    ndrange=fft_total)
        else
            launch!(arch, twiddle_3d_dim3_forward_kernel!, output, fft_out, dct_plan.twiddle,
                    forward_scale_zero, forward_scale_pos, N, Nx, Ny;
                    ndrange=fft_total)
        end

    else  # :backward
        # Step 1: Allocate complex array for C2R input
        complex_shape = ntuple(i -> i == dim ? half_N_plus1 : size(input, i), 3)
        complex_buf = CUDA.zeros(Complex{T}, complex_shape...)
        complex_total = prod(complex_shape)

        # Step 2: Apply inverse twiddle factors
        if dim == 1
            launch!(arch, twiddle_3d_dim1_backward_kernel!, complex_buf, input, dct_plan.twiddle_inv,
                    backward_scale_zero, backward_scale_pos, N, Ny, Nz;
                    ndrange=complex_total)
        elseif dim == 2
            launch!(arch, twiddle_3d_dim2_backward_kernel!, complex_buf, input, dct_plan.twiddle_inv,
                    backward_scale_zero, backward_scale_pos, N, Nx, Nz;
                    ndrange=complex_total)
        else
            launch!(arch, twiddle_3d_dim3_backward_kernel!, complex_buf, input, dct_plan.twiddle_inv,
                    backward_scale_zero, backward_scale_pos, N, Nx, Ny;
                    ndrange=complex_total)
        end

        # Step 3: Batched C2R IFFT along dim
        irfft_plan = _get_batched_irfft_plan(complex_shape, dim, N, T)
        work = irfft_plan * complex_buf

        # Step 4: Inverse reorder
        inverse_reorder_for_dct_dim!(output, work, dim)
    end

    CUDA.synchronize()
    return output
end

# ============================================================================
# Distributed DCT Operations
# ============================================================================

"""
    distributed_forward_dct!(coeffs, data, plan::DistributedDCTPlan)

Perform the forward distributed 3D basis-aware spectral transform.

Transform order (starting from Z-pencil grid values), basis-aware per axis:
1. transform in Z (local) → transpose Z→Y → transform in Y (local)
   → transpose Y→X → transform in X (local)
2. if `axis_kind[1] == :real_fourier`: truncate dim 1 to `div(Nx,2)+1` (local in X-pencil)
3. OUTPUT ADAPTER: transpose X→Y→Z on `plan.coeff_pencil` so coeffs land Z-local.

Input: real-or-complex grid values in Z-pencil layout (real is promoted to complex).
Output: complex spectral coefficients **Z-local on the coeff pencil**
(`plan.coeff_pencil.z_pencil_shape`); dim 1 is the half-spectrum length for a
RealFourier dim-1 axis, otherwise Nx.

# Arguments
- `coeffs`: Output array, `Complex{T}`, shape `plan.coeff_pencil.z_pencil_shape`
- `data`: Input grid values (Z-pencil shape), real or complex
- `plan`: DistributedDCTPlan created for this decomposition

# Returns
- `coeffs` array filled with spectral coefficients

# Example
```julia
pencil = PencilDecomposition(global_shape, proc_grid, rank, comm)
plan = DistributedDCTPlan(pencil, bases, Float64)

data   = CUDA.rand(Float64, pencil.z_pencil_shape...)
coeffs = CUDA.zeros(ComplexF64, plan.coeff_pencil.z_pencil_shape...)

distributed_forward_dct!(coeffs, data, plan)
```
"""
function distributed_forward_dct!(coeffs::CuArray{Complex{T}, 3}, data::CuArray,
                                   plan::DistributedDCTPlan{T}) where {T}
    pencil  = plan.pencil
    cpencil = plan.coeff_pencil

    # Ensure starting in Z-pencil grid layout.
    @assert current_orientation(pencil) == :z_pencil "Must start in Z-pencil layout"
    @assert size(data) == pencil.z_pencil_shape "Data shape must match Z-pencil shape"
    # Output lands Z-local on the coeff pencil (half-spectrum on dim 1 if RealFourier).
    @assert size(coeffs) == cpencil.z_pencil_shape "Coeffs must match coeff Z-pencil (half-spectrum) shape"

    # Promote real grid input → Complex{T} (design decision #1).
    cdata = eltype(data) <: Complex ? data : Complex{T}.(data)

    # --- forward local transforms with Z→Y→X transposes (full-spectrum pencil) ---
    z_work = plan.work_arrays[3]
    local_transform_along_dim!(z_work, cdata, plan, 3, :forward)

    y_data = transpose_z_to_y!(plan.transpose_buffer, z_work, pencil)
    y_work = plan.work_arrays[2]
    local_transform_along_dim!(y_work, y_data, plan, 2, :forward)

    x_data = transpose_y_to_x!(plan.transpose_buffer, y_work, pencil)
    x_work = plan.work_arrays[1]
    local_transform_along_dim!(x_work, x_data, plan, 1, :forward)

    # RealFourier dim-1 half-spectrum truncation. dim 1 is LOCAL here (X-pencil),
    # so truncating it is correct; the result has shape == cpencil.x_pencil_shape.
    spectral_x = plan.axis_kind[1] === :real_fourier ? _realfourier_truncate(x_work) : x_work

    # --- OUTPUT ADAPTER: transpose X→Y→Z on the COEFF-sized pencil so the dim-1
    # length the transposes split/gather is the (possibly truncated) half-spectrum,
    # not Nx. This lands the coeffs Z-local, matching the field coeff buffer. ---
    set_orientation!(cpencil, :x_pencil)
    y_back = transpose_x_to_y!(plan.transpose_buffer, spectral_x, cpencil)
    z_back = transpose_y_to_z!(plan.transpose_buffer, y_back, cpencil)
    copyto!(coeffs, z_back)

    # Reset orientations for re-entrancy (both pencils return to :z_pencil).
    set_orientation!(pencil, :z_pencil)
    set_orientation!(cpencil, :z_pencil)
    return coeffs
end

"""
    distributed_backward_dct!(data, coeffs, plan::DistributedDCTPlan)

Perform the backward distributed 3D basis-aware spectral transform (inverse).

Transform order (starting from Z-local coeffs), basis-aware per axis:
1. INPUT ADAPTER: transpose Z→Y→X on `plan.coeff_pencil`.
2. if `axis_kind[1] == :real_fourier`: Hermitian-expand dim 1 `div(Nx,2)+1` → Nx (local in X-pencil).
3. inverse X (local) → transpose X→Y → inverse Y (local)
   → transpose Y→Z → inverse Z (local).
4. `real.()` into `data` for a real grid.

Input: complex spectral coefficients **Z-local on the coeff pencil**
(`plan.coeff_pencil.z_pencil_shape`).
Output: grid values in Z-pencil layout (real `data` receives `real.(...)`).

# Arguments
- `data`: Output grid values (Z-pencil shape), real or complex
- `coeffs`: Input coefficients, `Complex{T}`, shape `plan.coeff_pencil.z_pencil_shape`
- `plan`: DistributedDCTPlan created for this decomposition

# Returns
- `data` array filled with grid values

# Example
```julia
pencil = PencilDecomposition(global_shape, proc_grid, rank, comm)
plan = DistributedDCTPlan(pencil, bases, Float64)

data = CUDA.zeros(Float64, pencil.z_pencil_shape...)
distributed_backward_dct!(data, coeffs, plan)
```
"""
function distributed_backward_dct!(data::CuArray, coeffs::CuArray{Complex{T}, 3},
                                    plan::DistributedDCTPlan{T}) where {T}
    pencil  = plan.pencil
    cpencil = plan.coeff_pencil

    # Coeffs are stored Z-local on the coeff pencil (half-spectrum on dim 1 if RealFourier).
    @assert size(coeffs) == cpencil.z_pencil_shape "Coeffs must match coeff Z-pencil (half-spectrum) shape"
    @assert size(data) == pencil.z_pencil_shape "Data shape must match Z-pencil shape"

    # --- INPUT ADAPTER: bring Z-local coeffs to X-pencil via Z→Y→X on the COEFF
    # pencil (mirror of the forward output adapter). dim 1 stays the half-spectrum
    # length until it is LOCAL again in X-pencil. ---
    set_orientation!(cpencil, :z_pencil)
    y_c = transpose_z_to_y!(plan.transpose_buffer, coeffs, cpencil)
    x_c = transpose_y_to_x!(plan.transpose_buffer, y_c, cpencil)
    # x_c now has shape == cpencil.x_pencil_shape (dim 1 = half-spectrum length).

    # RealFourier dim-1 Hermitian expansion: half-spectrum → full Nx. dim 1 is
    # LOCAL here (X-pencil), so expanding it is correct; result dim 1 == Nx.
    Nx = pencil.global_shape[1]
    # CORRECTNESS GUARD: _realfourier_hermitian_expand runs here while dims 2/3 are
    # STILL spectral, doing a per-(k2,k3)-column conjugate (full[N1-k1+2,k2,k3] =
    # conj(half[k1,k2,k3]), NO transverse-wavenumber flip). That is correct only
    # when every transverse axis is real-kernel (Chebyshev DCT-I) or physical. A
    # ComplexFourier (or RealFourier) transverse axis needs the conjugate partner
    # at the flipped wavenumber ((N2-k2)%N2, …) — equivalently the dim-1 inverse
    # must be the LAST distributed-backward stage (as on CPU; transform_fourier.jl
    # walks transforms in reverse). Fail loudly instead of silently corrupting the
    # round-trip. The proper reorder fix needs multi-GPU verification — see
    # memory/project_gpu_ff_ffc_audit_2026_06_22.md.
    if plan.axis_kind[1] === :real_fourier &&
       any(k -> k === :complex_fourier || k === :real_fourier, plan.axis_kind[2:end])
        error("Distributed-GPU backward DCT: RealFourier on dim 1 together with a " *
              "Fourier transverse axis (dims 2/3) is not yet supported — the per-column " *
              "Hermitian expand would place conjugate partners at the wrong transverse " *
              "wavenumber. Reorder so the dim-1 inverse is the final stage, or use the " *
              "CPU path for this layout.")
    end
    spectral_x_full = plan.axis_kind[1] === :real_fourier ?
        _realfourier_hermitian_expand(x_c, Nx) : x_c

    # --- inverse local transforms with X→Y→Z transposes (full-spectrum pencil) ---
    x_work = plan.work_arrays[1]
    local_transform_along_dim!(x_work, spectral_x_full, plan, 1, :backward)

    set_orientation!(pencil, :x_pencil)
    y_data = transpose_x_to_y!(plan.transpose_buffer, x_work, pencil)
    y_work = plan.work_arrays[2]
    local_transform_along_dim!(y_work, y_data, plan, 2, :backward)

    z_data = transpose_y_to_z!(plan.transpose_buffer, y_work, pencil)
    z_work = plan.work_arrays[3]
    local_transform_along_dim!(z_work, z_data, plan, 3, :backward)

    # Grid output: drop the (numerically ~0) imaginary part for a real grid.
    if eltype(data) <: Complex
        copyto!(data, z_work)
    else
        data .= real.(z_work)
    end

    # Reset orientations for re-entrancy.
    set_orientation!(pencil, :z_pencil)
    set_orientation!(cpencil, :z_pencil)
    return data
end

"""
    finalize_distributed_dct_plan!(plan::DistributedDCTPlan)

Clean up resources used by the distributed DCT plan.

This function releases NCCL sub-communicators and allows
work arrays to be garbage collected.

# Arguments
- `plan`: DistributedDCTPlan to finalize

# Example
```julia
plan = DistributedDCTPlan(pencil, Float64)
# ... use plan for transforms ...
finalize_distributed_dct_plan!(plan)
```
"""
function finalize_distributed_dct_plan!(plan::DistributedDCTPlan)
    finalize_nccl_transpose!(plan.transpose_buffer)
    # Work arrays will be garbage collected
    return nothing
end

# ============================================================================
# Local primitives for the distributed RealFourier × Chebyshev (DCT-I) path
# ============================================================================
#
# These are the per-rank, single-GPU building blocks the distributed
# Fourier-Chebyshev transform composes between pencil transposes:
#   - local_fft_along_dim!    : C2C FFT along one dim (Fourier axes)
#   - local_dct1_along_dim!   : complex DCT-I along one dim (Chebyshev axes)
#   - _realfourier_truncate / _realfourier_hermitian_expand : dim-1 RealFourier
#     half-spectrum <-> full-spectrum (Hermitian symmetry)
# Plan caching for the FFTs is a later task; the inline plan here is fine.

"""
    local_fft_along_dim!(output, input, dim, direction) -> output

Batched complex-to-complex FFT along `dim` of a 3D complex array.
`:forward` is unnormalized; `:backward` carries the 1/N (CUFFT `plan_ifft`),
matching the FFTW `plan_fft`/`plan_ifft` convention of the CPU Fourier path.
"""
function local_fft_along_dim!(output::CuArray{Complex{T},3}, input::CuArray{Complex{T},3},
                              dim::Int, direction::Symbol) where {T}
    if direction === :forward
        plan = CUFFT.plan_fft(input, (dim,))
        mul!(output, plan, input)
    elseif direction === :backward
        plan = CUFFT.plan_ifft(input, (dim,))   # carries 1/N
        mul!(output, plan, input)
    else
        error("direction must be :forward or :backward, got $direction")
    end
    return output
end

"""
    local_dct1_along_dim!(output, input, dim, direction) -> output

Complex DCT-I (REDFT00) along `dim` of a 3D complex array. The real DCT-I only
accepts real input, so the real and imaginary parts are transformed
independently (via `gpu_dct1_along_dim!`) and recombined — exactly mirroring how
the CPU Chebyshev transform splits complex fields (see transform_chebyshev.jl).
"""
function local_dct1_along_dim!(output::CuArray{Complex{T},3}, input::CuArray{Complex{T},3},
                               dim::Int, direction::Symbol) where {T}
    re = real.(input); im = imag.(input)
    ore = similar(re); oim = similar(im)
    gpu_dct1_along_dim!(ore, re, dim, direction)
    gpu_dct1_along_dim!(oim, im, dim, direction)
    output .= Complex.(ore, oim)
    return output
end

"""
    _realfourier_truncate(full) -> half

Forward RealFourier dim-1 truncation: keep the first `div(N,2)+1` modes along
dim 1 (the non-redundant half-spectrum of a real-valued signal).
"""
_realfourier_truncate(full::CuArray{Complex{T},3}) where {T} =
    full[1:div(size(full, 1), 2) + 1, :, :]

"""
    _realfourier_hermitian_expand(half, N) -> full

Backward RealFourier dim-1 expansion: rebuild the full length-`N` spectrum along
dim 1 from the half-spectrum via Hermitian symmetry `X[N-k+2] = conj(X[k])`.
Batched over dims 2 and 3. The index map matches the tested CPU reference
`Tarang._hermitian_full_from_half` in src/core/transforms/transform_gpu.jl
(`full[N-k+2] = conj(half[k])` for `k = 2 … (N - M + 1)`, `M = div(N,2)+1`).
"""
function _realfourier_hermitian_expand(half::CuArray{Complex{T},3}, N::Int) where {T}
    M = div(N, 2) + 1
    @assert size(half, 1) == M "half dim-1 length must be div(N,2)+1 = $M, got $(size(half,1))"
    full = CUDA.zeros(Complex{T}, N, size(half, 2), size(half, 3))
    full[1:M, :, :] .= half
    krange = 2:(N - M + 1)
    full[N .- krange .+ 2, :, :] .= conj.(half[krange, :, :])
    return full
end
