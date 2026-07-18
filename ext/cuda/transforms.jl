# ============================================================================
# GPU Transform Implementations
# ============================================================================

# Field data accessors: imported from Tarang (get_grid_data, get_coeff_data,
# set_grid_data!, set_coeff_data!) — use field.buffers.grid/coeff internally.

# ============================================================================
# Distributed GPU Detection
# ============================================================================

"""
    is_distributed_gpu(arch, nprocs::Int)

Check if we should use distributed GPU transforms.
Returns true if GPU architecture + multiple processes + NCCL available.
"""
function is_distributed_gpu(arch::GPU, nprocs::Int)
    return nprocs > 1 && Tarang.nccl_available()
end

is_distributed_gpu(arch::CPU, nprocs::Int) = false
is_distributed_gpu(arch, nprocs::Int) = false

"""
    needs_distributed_dct(field)

Check if the field requires distributed DCT (has Chebyshev basis in any dimension).
Returns true only for 3D fields with at least one Chebyshev basis.
"""
function needs_distributed_dct(field)
    bases = field.bases
    if length(bases) != 3
        return false  # Only support 3D distributed DCT for now
    end
    for b in bases
        if b isa ChebyshevT
            return true
        end
    end
    return false
end

# ============================================================================
# Distributed DCT Plan Cache
# ============================================================================

# Global cache for distributed DCT plans (thread-safe).
# Value type is Any: DistributedDCTPlan is defined in dct_distributed.jl, which
# is included AFTER this file — naming it here is an UndefVarError at load time.
const DISTRIBUTED_DCT_PLAN_CACHE = Dict{Tuple, Any}()
const DISTRIBUTED_DCT_PLAN_LOCK = ReentrantLock()

"""
    get_or_create_distributed_dct_plan(field)

Get or create a cached distributed DCT plan for the given field.
Thread-safe caching by field properties (global shape, proc grid, element type).
"""
function get_or_create_distributed_dct_plan(field)
    dist = field.dist

    # Create cache key from field properties. axis_kind is part of the key because
    # the plan's per-axis dispatch (and the coeff-pencil half-spectrum sizing)
    # depend on the basis kinds, not just the shape/proc-grid/eltype.
    global_shape_tuple = Tuple(Tarang.global_shape(field.domain))
    proc_grid = _compute_proc_grid(dist.size)
    T = real(eltype(get_grid_data(field)))
    bases = field.bases
    axis_kind = Tarang.axis_kinds(bases)

    key = (global_shape_tuple, proc_grid, T, axis_kind)

    lock(DISTRIBUTED_DCT_PLAN_LOCK) do
        if haskey(DISTRIBUTED_DCT_PLAN_CACHE, key)
            return DISTRIBUTED_DCT_PLAN_CACHE[key]
        end

        # Create pencil decomposition from field's distributor
        pencil = get_or_create_pencil(dist, global_shape_tuple)

        # Create basis-aware plan
        plan = DistributedDCTPlan(pencil, bases, T)

        DISTRIBUTED_DCT_PLAN_CACHE[key] = plan
        return plan
    end
end

"""
    _compute_proc_grid(nprocs::Int)

Compute a 2D process grid (P1, P2) for the given number of processes.
Prefers a square-ish distribution.
"""
function _compute_proc_grid(nprocs::Int)
    P1 = isqrt(nprocs)
    while nprocs % P1 != 0
        P1 -= 1
    end
    P2 = nprocs ÷ P1
    return (P1, P2)
end

"""
    get_or_create_pencil(dist, global_shape::NTuple{3, Int})

Get or create PencilDecomposition from a Distributor.
"""
function get_or_create_pencil(dist, global_shape::NTuple{3, Int})
    nprocs = dist.size
    rank = dist.rank
    comm = dist.comm

    # Honor a user-supplied process mesh: Distributor stores it as `dist.mesh`
    # (a 2D tuple for 3D domains, e.g. (4, 2)). Deriving the grid from
    # _compute_proc_grid alone ignored it — e.g. mesh (4, 2) silently became
    # pencil grid (2, 4), so the pencil's local shapes disagreed with every
    # mesh-derived buffer and the shape asserts rejected the run. Fall back to
    # the square-ish heuristic only when no usable 2D mesh is present.
    mesh = dist.mesh
    proc_grid = if mesh isa Tuple && length(mesh) == 2 && prod(mesh) == nprocs
        (mesh[1], mesh[2])
    else
        _compute_proc_grid(nprocs)
    end

    # Align the pencil's block ownership with the distributor's column-major
    # field-buffer convention (design decision #5). Only matters when P1>1 AND
    # P2>1 (np≥4); coincides with row-major for 1×P / P×1 grids. Needs np≥4 GPU
    # validation.
    gc = column_major_grid_coords(rank, proc_grid)
    return PencilDecomposition(global_shape, proc_grid, rank, comm; grid_coords=gc)
end

"""
    clear_distributed_dct_plan_cache!()

Clear all cached distributed DCT plans (thread-safe).
"""
function clear_distributed_dct_plan_cache!()
    lock(DISTRIBUTED_DCT_PLAN_LOCK) do
        # Finalize all plans before clearing
        for plan in values(DISTRIBUTED_DCT_PLAN_CACHE)
            finalize_distributed_dct_plan!(plan)
        end
        empty!(DISTRIBUTED_DCT_PLAN_CACHE)
    end
end

# ============================================================================
# Distributed GPU Transform Functions
# ============================================================================

"""
    distributed_gpu_forward_transform!(field::ScalarField)

Forward transform using distributed GPU DCT for Chebyshev fields.
Transforms from grid space (Z-pencil layout) to spectral space (X-pencil layout).
"""
function distributed_gpu_forward_transform!(field::ScalarField)
    # Get or create distributed DCT plan
    plan = get_or_create_distributed_dct_plan(field)

    # Get local grid data
    data = get_grid_data(field)

    # Ensure we're in Z-pencil orientation (grid space)
    set_orientation!(plan.pencil, :z_pencil)

    # Allocate output. The rewritten driver lands coeffs **Z-local on the coeff
    # pencil** (complex, half-spectrum on dim 1 for a RealFourier dim-1 axis) —
    # NOT X-pencil. See distributed_forward_dct!. (This dispatch fires from
    # _gpu_forward_transform_impl! when is_distributed_gpu + needs_distributed_dct
    # + Tarang.distributed_gpu_supported all hold.)
    T = real(eltype(data))
    coeffs = similar(data, Complex{T}, plan.coeff_pencil.z_pencil_shape...)

    # Perform distributed transform (resets pencil orientations internally).
    distributed_forward_dct!(coeffs, data, plan)

    # Store coefficients (Z-local layout).
    set_coeff_data!(field, coeffs)

    return true
end

"""
    distributed_gpu_backward_transform!(field::ScalarField)

Backward transform using distributed GPU DCT for Chebyshev fields.
Transforms from spectral space (X-pencil layout) to grid space (Z-pencil layout).
"""
function distributed_gpu_backward_transform!(field::ScalarField)
    plan = get_or_create_distributed_dct_plan(field)

    # Get local coefficient data (Z-local on the coeff pencil, complex).
    coeffs = get_coeff_data(field)

    # Allocate real grid output (Z-pencil shape). The rewritten driver starts from
    # Z-local coeffs and drives the pencils' orientations internally.
    T = real(eltype(coeffs))
    data = similar(coeffs, T, plan.pencil.z_pencil_shape...)

    # Perform distributed inverse transform.
    distributed_backward_dct!(data, coeffs, plan)

    # Store grid data
    set_grid_data!(field, data)

    return true
end

"""
    _gpu_forward_transform_impl!(field::ScalarField)

GPU-specific forward transform using CUFFT (extension-local implementation).
Registered into `Tarang._GPU_FORWARD_TRANSFORM_HOOK` by `TarangCUDAExt.__init__`;
`Tarang.gpu_forward_transform!` calls it through the hook (a same-signature
`Tarang.gpu_forward_transform!` method here would overwrite the src method).
Returns `true` if the GPU transform was applied, `false` to fall back to CPU.
Supports:
- Pure Fourier (RealFourier, ComplexFourier) - uses cuFFT
- Chebyshev-containing fields route to CPU (GPU DCT machinery is DCT-II,
  Tarang's Chebyshev convention is DCT-I), except the distributed DCT-I path.
"""
function _gpu_forward_transform_impl!(field::ScalarField)
    arch = field.dist.architecture
    if !Tarang.is_gpu(arch)
        return false
    end

    data_g = get_grid_data(field)
    if !isa(data_g, CuArray)
        return false
    end

    gpu_arch = arch::GPU

    # Ensure correct device is active for multi-GPU support
    ensure_device!(gpu_arch)

    # Determine transform type based on bases
    bases = field.bases
    if isempty(bases)
        return false
    end

    nprocs = field.dist.size
    # Distributed multi-GPU spectral transform (basis-aware DCT-I + Fourier), gated
    # behind the support predicate. Fires only for nprocs>1 + NCCL (is_distributed_gpu)
    # AND a Chebyshev-containing 3D field (needs_distributed_dct). distributed_forward_dct!
    # now implements the framework DCT-I convention (complex-everywhere, Z-local coeffs).
    if is_distributed_gpu(arch, nprocs) && needs_distributed_dct(field)
        if Tarang.distributed_gpu_supported(field)
            return distributed_gpu_forward_transform!(field)
        else
            return false   # unsupported layout (e.g. RealFourier not on dim 1) -> CPU DCT-I fallback
        end
    end

    # CRITICAL: GPU+MPI for Fourier transforms requires TransposableField/distributed paths
    # A local-only cuFFT in MPI mode produces INCORRECT results (each rank FFTs its local slab)
    # If we reach here with nprocs > 1 and have Fourier bases, error out explicitly
    if nprocs > 1
        has_fourier = any(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
        if has_fourier
            error("GPU+MPI Fourier transforms require using TransposableField or the distributed " *
                  "transform path (distributed_forward_transform!). Direct forward_transform! on " *
                  "GPU with $(nprocs) MPI processes would produce incorrect results (local FFT only). " *
                  "Either: (1) use TransposableField for distributed transforms, " *
                  "(2) use distributed_forward_transform!, or (3) run with a single GPU.")
        end
    end

    # Use LOCAL array size from actual data (not global domain size)
    # This is critical for distributed computing where each rank owns a portion
    local_grid_shape = size(data_g)

    if !Tarang.should_use_gpu_fft(field, local_grid_shape)
        return false
    end

    # Classify bases
    all_fourier = all(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    all_chebyshev = all(b -> isa(b, ChebyshevT), bases)
    has_fourier = any(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    has_chebyshev = any(b -> isa(b, ChebyshevT), bases)

    # CORRECTNESS: the GPU DCT path below implements DCT-II (Makhoul), but Tarang's
    # Chebyshev convention is DCT-I on the Gauss-Lobatto grid — see
    # transform_chebyshev.jl (REDFT00 + 1/(N-1) norm + endpoint half-weight + odd-
    # index sign flip). DCT-II and DCT-I are different transforms on different nodes,
    # so the GPU Chebyshev/mixed coefficients disagree with the CPU path and with
    # every downstream operator (differentiation matrix, BC rows, interpolation).
    # Until a correct GPU DCT-I is implemented AND validated on a GPU, route any
    # Chebyshev-containing field through the (correct) CPU transform chain: returning
    # false makes `forward_transform!` walk `dist.transforms`, whose Fourier and
    # Chebyshev `_apply_forward!` methods copy GPU arrays to the host, transform with
    # FFTW, and copy back. Pure-Fourier fields stay on the GPU (cuFFT, correct).
    # NOTE: the multi-GPU distributed Chebyshev path (distributed_gpu_forward_transform!)
    # is taken above when supported; this `return false` now only handles single-GPU /
    # non-distributed / unsupported-layout Chebyshev fields, routing them to the CPU DCT-I chain.
    if has_chebyshev
        return false
    end

    input_T = eltype(data_g)
    coeff_T = Tarang.coefficient_eltype(field.dtype)

    if all_fourier
        # Pure Fourier case - use optimized multi-dimensional FFT.
        #
        # CANONICAL LAYOUT (src/core/domain.jl `_fourier_output_size`): only the
        # FIRST Fourier axis is halved, and only if it is RealFourier (rfft of real
        # data). Every other axis — including a RealFourier axis that is NOT first —
        # stays full size (the CPU chain sees complex input there and runs a full
        # C2C fft, transform_fourier.jl `_fourier_forward`). So:
        #   - R2C (multi-dim rfft: R2C on dim 1, C2C on rest) ONLY when bases[1] is
        #     RealFourier AND the grid data is real;
        #   - otherwise full multi-dim C2C, coeff shape == grid shape.
        # A rfft along a non-first RealFourier axis would produce a coeff buffer of
        # a NON-canonical shape (halved along the wrong axis), silently misaligning
        # every consumer sized from `coefficient_shape`.
        first_is_real = isa(bases[1], RealFourier)
        use_r2c = first_is_real && !(input_T <: Complex)

        if use_r2c
            # dim 1 is RealFourier with real data: multi-dim rfft (R2C on dim 1,
            # C2C on the rest) — matches the canonical halved-first-axis layout.
            plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, input_T; real_input=true)
            local_coeff_shape = (div(local_grid_shape[1], 2) + 1, local_grid_shape[2:end]...)

            existing_coeff = get_coeff_data(field)
            needs_alloc = !(existing_coeff isa CuArray) ||
                          eltype(existing_coeff) != coeff_T ||
                          size(existing_coeff) != local_coeff_shape
            if needs_alloc
                set_coeff_data!(field, CUDA.zeros(coeff_T, local_coeff_shape...))
            end

            gpu_forward_fft!(get_coeff_data(field), data_g, plan)
        else
            # Full C2C on all axes: covers (a) all-ComplexFourier, (b) RealFourier
            # NOT on dim 1 (canonical layout keeps it full size), and (c) complex
            # input with RealFourier on dim 1 (CPU falls back to fft too).
            # C2C requires complex input; promote real data if needed.
            if input_T <: Real
                fft_input = Complex{input_T}.(data_g)
                plan_T = Complex{input_T}
            else
                fft_input = data_g
                plan_T = input_T
            end
            plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, plan_T; real_input=false)

            existing_coeff = get_coeff_data(field)
            needs_alloc = !(existing_coeff isa CuArray) ||
                          eltype(existing_coeff) != plan_T ||
                          size(existing_coeff) != local_grid_shape
            if needs_alloc
                set_coeff_data!(field, CUDA.zeros(plan_T, local_grid_shape...))
            end

            gpu_forward_fft!(get_coeff_data(field), fft_input, plan)
        end

        return true

    elseif all_chebyshev && length(bases) == 1
        # ██ DEAD CODE — DO NOT RE-ENABLE WITHOUT A DCT-I REIMPLEMENTATION ██
        # This branch implements DCT-II (Makhoul), but Tarang's Chebyshev
        # convention is DCT-I on the Gauss-Lobatto grid. It is UNREACHABLE behind
        # the `has_chebyshev → return false` guard above and would produce wrong
        # coefficients if reached.
        # Pure Chebyshev 1D case - use GPU DCT
        n = local_grid_shape[1]
        local_coeff_shape = local_grid_shape  # Chebyshev: same shape

        existing_coeff = get_coeff_data(field)
        needs_alloc = !(existing_coeff isa CuArray) ||
                      eltype(existing_coeff) != input_T ||
                      size(existing_coeff) != local_coeff_shape
        if needs_alloc
            set_coeff_data!(field, CUDA.zeros(input_T, local_coeff_shape...))
        end

        if input_T <: Complex
            # Complex Chebyshev: DCT real and imag parts separately (kernels need
            # real input). Reuse cached scratch instead of allocating per call.
            real_T = real(input_T)
            dct_plan = get_gpu_dct_plan(gpu_arch, n, real_T, 1)
            sc = get_gpu_dct_scratch(gpu_arch, size(data_g), real_T, 4)
            real_in, imag_in, real_out, imag_out = sc[1], sc[2], sc[3], sc[4]
            real_in .= real.(data_g)
            imag_in .= imag.(data_g)
            gpu_forward_dct_1d!(real_out, real_in, dct_plan)
            gpu_forward_dct_1d!(imag_out, imag_in, dct_plan)
            get_coeff_data(field) .= complex.(real_out, imag_out)
        else
            dct_plan = get_gpu_dct_plan(gpu_arch, n, input_T, 1)
            gpu_forward_dct_1d!(get_coeff_data(field), data_g, dct_plan)
        end
        return true

    elseif all_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # ██ DEAD CODE — DO NOT RE-ENABLE WITHOUT A DCT-I REIMPLEMENTATION ██
        # This branch implements DCT-II (Makhoul), but Tarang's Chebyshev
        # convention is DCT-I. Unreachable behind the `has_chebyshev → return
        # false` guard above.
        # Pure Chebyshev 2D/3D case - use GPU DCT on all dimensions
        # Coefficient shape equals grid shape for Chebyshev
        local_coeff_shape = local_grid_shape

        existing_coeff = get_coeff_data(field)
        needs_alloc = !(existing_coeff isa CuArray) ||
                      eltype(existing_coeff) != input_T ||
                      size(existing_coeff) != local_coeff_shape
        if needs_alloc
            set_coeff_data!(field, CUDA.zeros(input_T, local_coeff_shape...))
        end

        if input_T <: Complex
            # Complex Chebyshev: apply DCT to real and imaginary parts separately
            # (DCT kernels use cos() which requires real-valued inputs on GPU)
            real_T = real(input_T)
            # Ping-pong between cached buffers (shape is invariant under DCT), so
            # no per-dimension device allocation.
            sc = get_gpu_dct_scratch(gpu_arch, size(data_g), real_T, 4)
            cur_real, buf_real, cur_imag, buf_imag = sc[1], sc[2], sc[3], sc[4]
            cur_real .= real.(data_g)
            cur_imag .= imag.(data_g)
            for dim in 1:length(bases)
                dct_plan = get_gpu_dct_dim_plan(gpu_arch, size(cur_real), real_T, dim)
                gpu_dct_dim!(buf_real, cur_real, dct_plan, Val(:forward))
                gpu_dct_dim!(buf_imag, cur_imag, dct_plan, Val(:forward))
                cur_real, buf_real = buf_real, cur_real
                cur_imag, buf_imag = buf_imag, cur_imag
            end
            get_coeff_data(field) .= complex.(cur_real, cur_imag)
        else
            # Apply DCT along each dimension (ping-pong between cached buffers)
            sc = get_gpu_dct_scratch(gpu_arch, size(data_g), input_T, 2)
            cur, buf = sc[1], sc[2]
            copyto!(cur, data_g)
            for dim in 1:length(bases)
                dct_plan = get_gpu_dct_dim_plan(gpu_arch, size(cur), input_T, dim)
                gpu_dct_dim!(buf, cur, dct_plan, Val(:forward))
                cur, buf = buf, cur
            end
            copyto!(get_coeff_data(field), cur)
        end
        return true

    elseif has_fourier && has_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # ██ DEAD CODE — DO NOT RE-ENABLE WITHOUT A DCT-I REIMPLEMENTATION ██
        # The Chebyshev stages of the mixed plan implement DCT-II (Makhoul), but
        # Tarang's Chebyshev convention is DCT-I. Unreachable behind the
        # `has_chebyshev → return false` guard above.
        # Mixed Fourier-Chebyshev 2D/3D case
        # Use the mixed transform plan for dimension-by-dimension transforms
        # Supports: Fourier-Chebyshev, Fourier-Fourier-Chebyshev, Fourier-Chebyshev-Chebyshev, etc.

        # Get or create mixed transform plan (determines correct coeff_shape)
        plan = get_gpu_mixed_transform_plan(gpu_arch, bases, local_grid_shape, input_T)
        local_coeff_shape = plan.coeff_shape

        existing_coeff = get_coeff_data(field)
        needs_alloc = !(existing_coeff isa CuArray) ||
                      eltype(existing_coeff) != coeff_T ||
                      size(existing_coeff) != local_coeff_shape
        if needs_alloc
            set_coeff_data!(field, CUDA.zeros(coeff_T, local_coeff_shape...))
        end

        # Execute mixed transform
        gpu_mixed_forward_transform!(get_coeff_data(field), data_g, plan)

        return true
    end

    # For unsupported combinations (e.g., Legendre), fall back to CPU
    return false
end

"""
    _gpu_backward_c2c_fft!(field, gpu_arch, data_c, local_grid_shape)

Shared full multi-dimensional C2C inverse FFT for the pure-Fourier backward
path: promotes real coefficient data if needed, sizes/allocates the grid buffer
to `local_grid_shape` (== coeff shape; C2C preserves shape), and stores the
complex inverse-FFT result — mirroring the CPU chain, which keeps the complex
ifft output for C2C axes.
"""
function _gpu_backward_c2c_fft!(field::ScalarField, gpu_arch::GPU, data_c::CuArray,
                                local_grid_shape::Tuple)
    coeff_T = eltype(data_c)
    # C2C inverse requires complex input; promote if needed (shouldn't normally happen)
    if coeff_T <: Real
        fft_input = Complex{coeff_T}.(data_c)
        plan_T = Complex{coeff_T}
    else
        fft_input = data_c
        plan_T = coeff_T
    end
    plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, plan_T; real_input=false)

    existing_grid = get_grid_data(field)
    needs_alloc = existing_grid === nothing ||
                  !(existing_grid isa CuArray) ||
                  eltype(existing_grid) != plan_T ||
                  size(existing_grid) != local_grid_shape
    if needs_alloc
        set_grid_data!(field, CUDA.zeros(plan_T, local_grid_shape...))
    end

    gpu_backward_fft!(get_grid_data(field), fft_input, plan)
    return nothing
end

"""
    _gpu_backward_transform_impl!(field::ScalarField)

GPU-specific backward transform using CUFFT (extension-local implementation).
Registered into `Tarang._GPU_BACKWARD_TRANSFORM_HOOK` by `TarangCUDAExt.__init__`;
`Tarang.gpu_backward_transform!` calls it through the hook (a same-signature
`Tarang.gpu_backward_transform!` method here would overwrite the src method).
Returns `true` if the GPU transform was applied, `false` to fall back to CPU.
Supports:
- Pure Fourier (RealFourier, ComplexFourier) - uses cuFFT (including the
  upsampled-rfft scaled/dealias backward path)
- Chebyshev-containing fields route to CPU (GPU DCT machinery is DCT-II,
  Tarang's Chebyshev convention is DCT-I), except the distributed DCT-I path.
"""
function _gpu_backward_transform_impl!(field::ScalarField)
    arch = field.dist.architecture
    if !Tarang.is_gpu(arch)
        return false
    end

    data_c = get_coeff_data(field)
    if !isa(data_c, CuArray)
        return false
    end

    gpu_arch = arch::GPU

    # Ensure correct device is active for multi-GPU support
    ensure_device!(gpu_arch)

    bases = field.bases
    if isempty(bases)
        return false
    end

    nprocs = field.dist.size
    # Distributed multi-GPU inverse transform — mirror of gpu_forward_transform!.
    # Gated behind the support predicate; distributed_backward_dct! now implements DCT-I.
    if is_distributed_gpu(arch, nprocs) && needs_distributed_dct(field)
        if Tarang.distributed_gpu_supported(field)
            return distributed_gpu_backward_transform!(field)
        else
            return false   # unsupported layout (e.g. RealFourier not on dim 1) -> CPU DCT-I fallback
        end
    end

    # Use LOCAL coefficient array size to determine grid shape
    # This is critical for distributed computing where each rank owns a portion
    local_coeff_shape = size(data_c)

    # Classify bases
    all_fourier = all(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    all_chebyshev = all(b -> isa(b, ChebyshevT), bases)
    has_fourier = any(b -> isa(b, RealFourier) || isa(b, ComplexFourier), bases)
    has_chebyshev = any(b -> isa(b, ChebyshevT), bases)

    # CORRECTNESS: GPU DCT path is DCT-II but Tarang Chebyshev is DCT-I (see the
    # matching comment in gpu_forward_transform! and transform_chebyshev.jl). Route
    # any Chebyshev-containing field through the correct CPU transform chain. The
    # multi-GPU distributed Chebyshev path (DCT-I) is taken above when supported; this
    # only handles single-GPU / non-distributed / unsupported-layout Chebyshev fields.
    if has_chebyshev
        return false
    end

    if all_fourier
        # Pure Fourier case — exact mirror of _gpu_forward_transform_impl!:
        # the forward uses R2C (multi-dim rfft) ONLY when bases[1] is RealFourier
        # and the data is real; every other combination — all-ComplexFourier,
        # RealFourier NOT on dim 1 (canonical layout keeps such axes FULL size,
        # src/core/domain.jl `_fourier_output_size`), complex input — is a full
        # multi-dim C2C with coeff shape == grid shape.
        first_is_real = isa(bases[1], RealFourier)

        if first_is_real && !(field.dtype <: Complex)
            # dim 1 stored as an rfft half-spectrum (forward used R2C). Classify
            # the backward path using basis metadata, mirroring the CPU detection
            # ORDER (transform_fourier.jl `_apply_backward!`): test the R2C
            # interpretation FIRST — pure shape heuristics misclassify N=1/N=2,
            # where div(N,2)+1 == N — then the upsampled (scaled) half-spectrum,
            # then C2C. Anything else is ambiguous: return false (CPU fallback),
            # NEVER run a guessed same-shape ifft and store junk.
            scaled_shape = Tarang.get_scaled_shape(field)
            coeff_T_bk = eltype(data_c)
            if length(scaled_shape) != length(local_coeff_shape) || !(coeff_T_bk <: Complex)
                return false  # missing/mismatched shape info or non-complex coeffs → CPU
            end
            grid_n1 = scaled_shape[1]
            base_n1 = bases[1].meta.size
            axis_len = local_coeff_shape[1]

            if axis_len == div(grid_n1, 2) + 1
                # Direct irfft: half-spectrum already at the (scaled) grid length.
                upsampled = false
                local_grid_shape = (grid_n1, local_coeff_shape[2:end]...)
            elseif grid_n1 > base_n1 && axis_len == div(base_n1, 2) + 1
                # UPSAMPLED rfft axis (scaled/dealiased field): the stored
                # half-spectrum is the BASE length div(base_N,2)+1 but the target
                # grid is finer (grid_n1 = ceil(scale*base_N) > base_N). Mirror
                # the CPU rule (transform_fourier.jl `_apply_backward!` upsampled
                # branch): zero-pad to div(grid_n,2)+1, rescale by grid_n/base_n,
                # zero the base Nyquist bin, then irfft at grid_n.
                upsampled = true
                local_grid_shape = (grid_n1, local_coeff_shape[2:end]...)
            elseif axis_len == grid_n1
                # Full-length dim 1: a C2C spectrum despite the real dtype (e.g.
                # coefficients written directly). CPU stores the complex ifft.
                if !Tarang.should_use_gpu_fft(field, local_coeff_shape)
                    return false
                end
                _gpu_backward_c2c_fft!(field, gpu_arch, data_c, local_coeff_shape)
                return true
            else
                # Shape matches no recognized half/full spectrum layout.
                return false
            end

            if !Tarang.should_use_gpu_fft(field, local_grid_shape)
                return false
            end

            real_T = field.dtype
            existing_grid = get_grid_data(field)
            needs_alloc = existing_grid === nothing ||
                          !(existing_grid isa CuArray) ||
                          eltype(existing_grid) != real_T ||
                          size(existing_grid) != local_grid_shape
            if needs_alloc
                set_grid_data!(field, CUDA.zeros(real_T, local_grid_shape...))
            end

            plan = get_gpu_fft_plan(gpu_arch, local_grid_shape, real_T; real_input=true)
            if upsampled
                # Zero-pad the base half-spectrum to div(grid_n,2)+1 along dim 1
                # and rescale by grid_n/base_n: the irfft divides by the (finer)
                # grid_n while the stored coeffs were formed on base_n points.
                padded_len = div(grid_n1, 2) + 1
                padded = CUDA.zeros(coeff_T_bk, padded_len, local_coeff_shape[2:end]...)
                nd = length(local_coeff_shape)
                front = ntuple(i -> i == 1 ? (1:axis_len) : Colon(), nd)
                @views padded[front...] .= data_c .* (grid_n1 / base_n1)
                if iseven(base_n1)
                    # The base Nyquist mode is ambiguous on the finer grid; drop it
                    # so this path agrees with the CPU spectral upsample.
                    nyq_i = div(base_n1, 2) + 1
                    nyq = ntuple(i -> i == 1 ? (nyq_i:nyq_i) : Colon(), nd)
                    @views padded[nyq...] .= 0
                end
                gpu_backward_fft!(get_grid_data(field), padded, plan)
            else
                gpu_backward_fft!(get_grid_data(field), data_c, plan)
            end
        else
            # Full multi-dim C2C inverse (grid shape == coeff shape): covers
            # all-ComplexFourier, RealFourier on a non-first dim (those axes were
            # transformed C2C at full size — canonical layout), and complex-dtype
            # fields with RealFourier on dim 1 (forward used C2C). Mirrors the CPU
            # chain, which stores the complex ifft result.
            local_grid_shape = local_coeff_shape

            if !Tarang.should_use_gpu_fft(field, local_grid_shape)
                return false
            end

            _gpu_backward_c2c_fft!(field, gpu_arch, data_c, local_grid_shape)
        end

        return true

    elseif all_chebyshev && length(bases) == 1
        # ██ DEAD CODE — DO NOT RE-ENABLE WITHOUT A DCT-I REIMPLEMENTATION ██
        # This branch implements the inverse of DCT-II (Makhoul), but Tarang's
        # Chebyshev convention is DCT-I on the Gauss-Lobatto grid. It is
        # UNREACHABLE behind the `has_chebyshev → return false` guard above and
        # would produce wrong grid data if reached.
        # Pure Chebyshev 1D case - use GPU inverse DCT
        local_grid_shape = local_coeff_shape  # Chebyshev: same shape

        if !Tarang.should_use_gpu_fft(field, local_grid_shape)
            return false
        end

        input_T = eltype(data_c)
        n = local_coeff_shape[1]

        existing_grid = get_grid_data(field)
        needs_alloc = existing_grid === nothing ||
                      !(existing_grid isa CuArray) ||
                      eltype(existing_grid) != input_T ||
                      size(existing_grid) != local_grid_shape
        if needs_alloc
            set_grid_data!(field, CUDA.zeros(input_T, local_grid_shape...))
        end

        if input_T <: Complex
            # Complex Chebyshev: apply inverse DCT to real and imaginary parts separately
            real_T = real(input_T)
            dct_plan = get_gpu_dct_plan(gpu_arch, n, real_T, 1)
            sc = get_gpu_dct_scratch(gpu_arch, size(data_c), real_T, 4)
            real_in, imag_in, real_out, imag_out = sc[1], sc[2], sc[3], sc[4]
            real_in .= real.(data_c)
            imag_in .= imag.(data_c)
            gpu_backward_dct_1d!(real_out, real_in, dct_plan)
            gpu_backward_dct_1d!(imag_out, imag_in, dct_plan)
            get_grid_data(field) .= complex.(real_out, imag_out)
        else
            dct_plan = get_gpu_dct_plan(gpu_arch, n, input_T, 1)
            gpu_backward_dct_1d!(get_grid_data(field), data_c, dct_plan)
        end
        return true

    elseif all_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # ██ DEAD CODE — DO NOT RE-ENABLE WITHOUT A DCT-I REIMPLEMENTATION ██
        # This branch implements the inverse of DCT-II (Makhoul), but Tarang's
        # Chebyshev convention is DCT-I. Unreachable behind the `has_chebyshev →
        # return false` guard above.
        # Pure Chebyshev 2D/3D case - use GPU DCT on all dimensions (in reverse order)
        local_grid_shape = local_coeff_shape  # For Chebyshev, shapes are equal

        if !Tarang.should_use_gpu_fft(field, local_grid_shape)
            return false
        end

        input_T = eltype(data_c)

        existing_grid = get_grid_data(field)
        needs_alloc = existing_grid === nothing ||
                      !(existing_grid isa CuArray) ||
                      eltype(existing_grid) != input_T ||
                      size(existing_grid) != local_grid_shape
        if needs_alloc
            set_grid_data!(field, CUDA.zeros(input_T, local_grid_shape...))
        end

        if input_T <: Complex
            # Complex Chebyshev: apply inverse DCT to real and imaginary parts separately
            # (DCT kernels use cos() which requires real-valued inputs on GPU)
            real_T = real(input_T)
            # Ping-pong between cached buffers (shape invariant), no per-dim alloc.
            sc = get_gpu_dct_scratch(gpu_arch, size(data_c), real_T, 4)
            cur_real, buf_real, cur_imag, buf_imag = sc[1], sc[2], sc[3], sc[4]
            cur_real .= real.(data_c)
            cur_imag .= imag.(data_c)
            for dim in reverse(1:length(bases))
                dct_plan = get_gpu_dct_dim_plan(gpu_arch, size(cur_real), real_T, dim)
                gpu_dct_dim!(buf_real, cur_real, dct_plan, Val(:backward))
                gpu_dct_dim!(buf_imag, cur_imag, dct_plan, Val(:backward))
                cur_real, buf_real = buf_real, cur_real
                cur_imag, buf_imag = buf_imag, cur_imag
            end
            get_grid_data(field) .= complex.(cur_real, cur_imag)
        else
            # Inverse DCT along each dimension (ping-pong between cached buffers)
            sc = get_gpu_dct_scratch(gpu_arch, size(data_c), input_T, 2)
            cur, buf = sc[1], sc[2]
            copyto!(cur, data_c)
            for dim in reverse(1:length(bases))
                dct_plan = get_gpu_dct_dim_plan(gpu_arch, size(cur), input_T, dim)
                gpu_dct_dim!(buf, cur, dct_plan, Val(:backward))
                cur, buf = buf, cur
            end
            copyto!(get_grid_data(field), cur)
        end
        return true

    elseif has_fourier && has_chebyshev && (length(bases) == 2 || length(bases) == 3)
        # ██ DEAD CODE — DO NOT RE-ENABLE WITHOUT A DCT-I REIMPLEMENTATION ██
        # The Chebyshev stages of the mixed plan implement (inverse) DCT-II
        # (Makhoul), but Tarang's Chebyshev convention is DCT-I. Unreachable
        # behind the `has_chebyshev → return false` guard above. (Its R2C/C2C
        # shape-reconstruction heuristics below also predate the canonical-layout
        # fixes — do not reuse them.)
        # Mixed Fourier-Chebyshev 2D/3D case
        # Supports: Fourier-Chebyshev, Fourier-Fourier-Chebyshev, Fourier-Chebyshev-Chebyshev, etc.

        # Determine grid shape: use existing grid data if available,
        # otherwise use basis metadata for robust detection of R2C vs C2C.
        existing_grid = get_grid_data(field)
        if existing_grid isa CuArray
            local_grid_shape = size(existing_grid)
        else
            first_real_found = false
            local_grid_shape = ntuple(length(bases)) do dim
                basis = bases[dim]
                if isa(basis, RealFourier) && !first_real_found
                    grid_n = basis.meta.size
                    if local_coeff_shape[dim] == div(grid_n, 2) + 1
                        # R2C was used for this dim
                        first_real_found = true
                        return grid_n
                    else
                        # C2C was used (complex input)
                        return local_coeff_shape[dim]
                    end
                else
                    return local_coeff_shape[dim]
                end
            end
        end

        if !Tarang.should_use_gpu_fft(field, local_grid_shape)
            return false
        end

        real_T = field.dtype
        needs_alloc = existing_grid === nothing ||
                      !(existing_grid isa CuArray) ||
                      eltype(existing_grid) != real_T ||
                      size(existing_grid) != local_grid_shape
        if needs_alloc
            set_grid_data!(field, CUDA.zeros(real_T, local_grid_shape...))
        end

        # Get or create mixed transform plan (uses grid_shape as canonical reference)
        input_T = real_T <: Complex ? real_T : real_T
        plan = get_gpu_mixed_transform_plan(gpu_arch, bases, local_grid_shape, input_T)

        # Execute mixed backward transform
        gpu_mixed_backward_transform!(get_grid_data(field), data_c, plan)

        return true
    end

    # For unsupported combinations (e.g., Legendre), fall back to CPU
    return false
end

# ============================================================================
# GPU FFT Transforms using CUFFT
# ============================================================================

"""
    GPUFFTPlan

Wrapper for CUFFT plans that work with CuArrays.
"""
struct GPUFFTPlan{P, IP}
    plan::P
    iplan::IP
    size::Tuple{Vararg{Int}}
    is_real::Bool
end

"""
    plan_gpu_fft(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)

Create a GPU FFT plan using CUFFT for element type `T`.

**Important:** `local_size` should be the LOCAL array shape (what this process owns),
not the global domain size. In distributed computing, each rank creates plans
for its local data portion.

For multi-GPU: ensures the plan is created on the correct device.
"""
function plan_gpu_fft(arch::GPU{CuDevice}, local_size::Tuple, T::Type; real_input::Bool=false)
    # Ensure we're on the correct device for plan creation
    ensure_device!(arch)

    complex_T = T <: Complex ? T : Complex{T}

    if real_input
        # Real-to-complex FFT (like rfft)
        # Plan is created based on local array dimensions
        dummy_in = CUDA.zeros(T, local_size...)
        plan = CUFFT.plan_rfft(dummy_in)

        # Complex-to-real inverse FFT (like irfft)
        out_size = (div(local_size[1], 2) + 1, local_size[2:end]...)
        dummy_out = CUDA.zeros(complex_T, out_size...)
        iplan = CUFFT.plan_irfft(dummy_out, local_size[1])

        return GPUFFTPlan(plan, iplan, local_size, true)
    else
        # Complex-to-complex FFT
        dummy = CUDA.zeros(complex_T, local_size...)
        plan = CUFFT.plan_fft(dummy)
        iplan = CUFFT.plan_ifft(dummy)

        return GPUFFTPlan(plan, iplan, local_size, false)
    end
end

# Fallback for generic GPU: delegates to device-specific version using current device,
# ensuring proper device context via ensure_device!
plan_gpu_fft(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false) =
    plan_gpu_fft(GPU{CuDevice}(CUDA.device()), local_size, T; real_input=real_input)

"""
    gpu_forward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)

Execute forward FFT on GPU.
"""
function gpu_forward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)
    mul!(output, plan.plan, input)
    return output
end

"""
    gpu_backward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)

Execute backward (inverse) FFT on GPU.

!!! warning "Scratch-cache single-task contract"
    The C2R branch borrows a scratch buffer from `get_gpu_dct_scratch`, which
    caches buffers per (device, shape, eltype, count) and hands the SAME buffers
    to every caller with matching keys. Concurrent same-shape transforms from
    multiple Julia tasks would collide on that shared scratch; serial use (one
    transform at a time per device) is the supported pattern.
"""
function gpu_backward_fft!(output::CuArray, input::CuArray, plan::GPUFFTPlan)
    if plan.is_real
        # cuFFT C2R (irfft) OVERWRITES its input buffer (same as FFTW irfft). Here `input`
        # is the field's coefficient buffer, so transforming from it directly corrupts the
        # caller's coefficients. Copy into a cached scratch first and transform from that —
        # mirrors the CPU path (transform_fourier.jl), which copies into a cached scratch for
        # exactly this reason. (C2C inverse, below, is non-destructive — no copy needed.)
        arch = Tarang.architecture(input)
        scratch = get_gpu_dct_scratch(arch, size(input), eltype(input), 1)[1]
        copyto!(scratch, input)
        mul!(output, plan.iplan, scratch)
    else
        mul!(output, plan.iplan, input)
    end
    return output
end

# ============================================================================
# GPU Transform Plan Cache (Thread-Safe)
# ============================================================================

"""
    GPUTransformCache

Thread-safe cache for GPU FFT plans to avoid recreation overhead.
Keys include device ID for multi-GPU safety.
Uses a ReentrantLock to protect concurrent access from multiple Julia threads.
"""
struct GPUTransformCache
    plans::Dict{Tuple, GPUFFTPlan}
    lock::ReentrantLock
end

const GPU_TRANSFORM_CACHE = GPUTransformCache(Dict{Tuple, GPUFFTPlan}(), ReentrantLock())

"""
    get_gpu_fft_plan(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)

Get or create a cached GPU FFT plan (thread-safe).
Plans are cached by (device_id, size, element_type, real_input).
"""
function get_gpu_fft_plan(arch::GPU{CuDevice}, local_size::Tuple, T::Type; real_input::Bool=false)
    device_id = CUDA.deviceid(arch.device)
    key = (device_id, local_size, T, real_input)

    lock(GPU_TRANSFORM_CACHE.lock) do
        if !haskey(GPU_TRANSFORM_CACHE.plans, key)
            GPU_TRANSFORM_CACHE.plans[key] = plan_gpu_fft(arch, local_size, T; real_input=real_input)
        end
        return GPU_TRANSFORM_CACHE.plans[key]
    end
end

# Fallback for generic GPU
function get_gpu_fft_plan(arch::GPU, local_size::Tuple, T::Type; real_input::Bool=false)
    device_id = _current_device_id()
    key = (device_id, local_size, T, real_input)

    lock(GPU_TRANSFORM_CACHE.lock) do
        if !haskey(GPU_TRANSFORM_CACHE.plans, key)
            GPU_TRANSFORM_CACHE.plans[key] = plan_gpu_fft(arch, local_size, T; real_input=real_input)
        end
        return GPU_TRANSFORM_CACHE.plans[key]
    end
end

"""
    clear_gpu_transform_cache!()

Clear all cached GPU transform plans (thread-safe).
"""
function clear_gpu_transform_cache!()
    lock(GPU_TRANSFORM_CACHE.lock) do
        empty!(GPU_TRANSFORM_CACHE.plans)
    end
end
