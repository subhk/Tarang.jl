"""
    Transform GPU - GPU transform support and heuristics

This file contains GPU-specific transform support including
FFT heuristics and CPU fallback execution.
"""

# ============================================================================
# GPU Transform Support
# ============================================================================

# Note: is_gpu_array is defined in architectures.jl

# ---------------------------------------------------------------------------
# GPU transform heuristics
# ---------------------------------------------------------------------------

const GPU_FFT_MIN_ELEMENTS = Ref(32_768)

"""
    set_gpu_fft_min_elements!(n::Integer)

Set the minimum number of elements required before GPU FFTs are attempted.
"""
function set_gpu_fft_min_elements!(n::Integer)
    GPU_FFT_MIN_ELEMENTS[] = max(1, Int(n))
    return GPU_FFT_MIN_ELEMENTS[]
end

gpu_fft_min_elements() = GPU_FFT_MIN_ELEMENTS[]

function should_use_gpu_fft(field::ScalarField, data_shape::Tuple)
    mode = gpu_fft_mode(field)
    if mode === :gpu
        return true
    elseif mode === :cpu
        return false
    end
    use_gpu = prod(data_shape) >= GPU_FFT_MIN_ELEMENTS[]
    if !use_gpu
        @debug "GPU FFT bypassed: $(prod(data_shape)) elements < threshold $(GPU_FFT_MIN_ELEMENTS[]). " *
               "Using CPU FFTW with GPU↔CPU transfer. Set set_gpu_fft_min_elements!(1) to force GPU." maxlog=1
    end
    return use_gpu
end

should_use_gpu_fft(field::ScalarField) = (get_grid_data(field) !== nothing) && should_use_gpu_fft(field, size(get_grid_data(field)))

"""
    gpu_forward_transform!(field::ScalarField)

GPU-specific forward transform using CUFFT.
Returns true if GPU transform was applied, false otherwise.
"""
function gpu_forward_transform!(field::ScalarField)
    # Check if we're on GPU architecture
    arch = field.dist.architecture
    if !is_gpu(arch)
        return false
    end

    # Check if data is on GPU
    data_g = get_grid_data(field)
    if !is_gpu_array(data_g)
        return false
    end

    # GPU transform will be dispatched via extension
    # The extension overrides this function when CUDA is loaded
    @warn "GPU architecture specified but CUDA extension not loaded. Falling back to CPU." maxlog=1
    return false
end

"""
    gpu_backward_transform!(field::ScalarField)

GPU-specific backward transform using CUFFT.
Returns true if GPU transform was applied, false otherwise.
"""
function gpu_backward_transform!(field::ScalarField)
    # Check if we're on GPU architecture
    arch = field.dist.architecture
    if !is_gpu(arch)
        return false
    end

    # Check if data is on GPU
    data_c = get_coeff_data(field)
    if !is_gpu_array(data_c)
        return false
    end

    # GPU transform will be dispatched via extension
    @warn "GPU architecture specified but CUDA extension not loaded. Falling back to CPU." maxlog=1
    return false
end

# -----------------------------------------------------------------------------
# Helper utilities for GPU fallbacks
# -----------------------------------------------------------------------------

"""
    _execute_on_cpu(f, data)

Ensure `f` runs on CPU memory even when `data` lives on a GPU.
Returns the result on the original device (GPU or CPU).

Note: f is the first argument to support do-block syntax:
    _execute_on_cpu(data) do host_data
        ...
    end
"""
# Fast path for CPU arrays — avoids is_gpu_array check and enables inlining
@inline _execute_on_cpu(f, data::Array) = f(data)

# Fallback for other array types (GPU arrays, wrapped arrays)
function _execute_on_cpu(f, data::AbstractArray)
    if is_gpu_array(data)
        host_data = Array(data)
        host_result = f(host_data)
        return copy_to_device(host_result, data)
    end
    return f(data)
end

# Transform execution functions
"""Apply forward transform to field"""
function forward_transform!(field::ScalarField, target_layout::Symbol=:c)

    if field.domain === nothing
        return
    end

    ensure_layout!(field, :g)  # Start in grid space

    # Try GPU transform first if on GPU architecture
    if gpu_forward_transform!(field)
        field.current_layout = :c
        return
    end

    # Find appropriate transform
    pencil_plan = _find_pencil_plan(field.dist)
    if pencil_plan !== nothing
        # PencilFFTs is CPU-only; if data is on GPU, move to CPU first
        grid_data = get_grid_data(field)
        if is_gpu_array(grid_data)
            host_data = Array(grid_data)
            host_result = pencil_plan * host_data
            set_coeff_data!(field, copy_to_device(host_result, grid_data))
        else
            # Use in-place mul! if coeff data is already allocated. The
            # fallback catch path allocates a fresh PencilArray each call,
            # so a single `@warn` the first time it fires makes it obvious
            # whether the fast path is working under MPI.
            coeff_data = get_coeff_data(field)
            if coeff_data !== nothing && isa(coeff_data, PencilArrays.PencilArray)
                try
                    mul!(coeff_data, pencil_plan, grid_data)
                catch err
                    @warn "forward_transform!: PencilFFTs mul! fast path failed, falling back to allocating `*`. This costs one PencilArray per transform — consider upgrading PencilFFTs or checking the field buffer layout." exception=(err, catch_backtrace()) maxlog=1
                    set_coeff_data!(field, pencil_plan * grid_data)
                end
            else
                @warn "forward_transform!: coeff buffer is not a PencilArray; using allocating `*` fallback. Expected `allocate_data!` to pre-allocate via PencilFFTs.allocate_output." maxlog=1
                set_coeff_data!(field, pencil_plan * grid_data)
            end
        end
        field.current_layout = :c
        return
    end

    # CRITICAL: Guard against running local transforms on distributed data
    # If we reach here with dist.size > 1, no PencilFFTPlan was found above,
    # so local transforms would produce incorrect results on distributed data.
    if field.dist.size > 1
        if is_gpu_array(get_grid_data(field))
            error("Cannot run local GPU transforms on distributed data without TransposableField. " *
                  "GPU+MPI requires explicit transposes for correct distributed FFT. " *
                  "Use TransposableField with distributed_forward_transform!/distributed_backward_transform! " *
                  "for correct GPU+MPI spectral transforms.")
        else
            error("Cannot run local FFTW transforms on distributed CPU data without PencilFFTs. " *
                  "No PencilFFTPlan found for this domain. " *
                  "For MPI+CPU Fourier, set use_pencil_arrays=true in Distributor. " *
                  "For MPI+GPU, use TransposableField with distributed transforms.")
        end
    end

    # ── Zero-allocation in-place transform chain ───────────────────────────
    #
    # Walk `field.dist.transforms` in order. Each transform has an in-place
    # dispatch method `_apply_forward!(out, in, transform)` that uses cached
    # plans and scratch buffers (see transform_fourier.jl / transform_chebyshev.jl).
    #
    # For intermediate transforms, the output is a cached scratch buffer on
    # the transform object. For the FINAL transform, we write directly into
    # `field.coeff_data`, which was already pre-allocated in `allocate_data!`
    # with the correct shape and eltype. If the pre-allocated buffer has the
    # wrong shape or eltype (shouldn't happen — shape is derived from the
    # same basis metadata — but be defensive), reallocate on the field.
    current = get_grid_data(field)
    transforms = field.dist.transforms
    n_transforms = length(transforms)
    if n_transforms == 0
        # No transforms registered: copy grid into coeff buffer (also a fallback
        # for fields with no spectral bases).
        coeff = get_coeff_data(field)
        if coeff !== nothing && size(coeff) == size(current) && eltype(coeff) == eltype(current)
            copyto!(coeff, current)
        else
            set_coeff_data!(field, copy(current))
        end
        field.current_layout = :c
        return
    end

    for (idx, transform) in enumerate(transforms)
        out_shape, out_eltype = _forward_output_spec(current, transform)
        if idx == n_transforms
            # Final stage: target is the field's coeff buffer. Reuse when
            # shape/eltype match (the common case); otherwise allocate once.
            coeff = get_coeff_data(field)
            if coeff === nothing || size(coeff) != out_shape || eltype(coeff) != out_eltype
                coeff = zeros(out_eltype, out_shape...)
                set_coeff_data!(field, coeff)
            end
            _apply_forward!(coeff, current, transform)
            current = coeff
        else
            # Intermediate stage: write into this transform's cached scratch.
            # Key by (out_shape, out_eltype, :fwd_inter) so it doesn't collide
            # with other scratch entries (e.g., the DCT real/imag buffers on
            # ChebyshevTransform, which live in a differently-typed Dict).
            # FourierTransform.fwd_scratch is Dict{Tuple, AbstractArray}; use
            # the generic _get_or_alloc_scratch! helper.
            out = _get_scratch_for_transform!(transform, :fwd_inter, out_shape, out_eltype)
            _apply_forward!(out, current, transform)
            current = out
        end
    end
    field.current_layout = :c
end

# ---------------------------------------------------------------------------
# Helper: fetch an intermediate-output scratch buffer from a transform.
# ---------------------------------------------------------------------------
# ChebyshevTransform's `fwd_scratch` / `bwd_scratch` fields are typed
# `Dict{Tuple, ChebScratch}` (hold real/imag/plan scratch records, not raw
# arrays), so we can't reuse them for the intermediate-output buffer.
# For ChebyshevTransform we therefore fall back to a small per-transform
# `matrices`-style Dict stored in a free-form field (`forward_matrix` etc.
# are sparse matrices; we don't touch those). Instead, cache intermediate
# scratch in a lazily-created Dict attached via `objectid`-keyed module-
# level storage.
#
# Since Tarang is single-threaded in this hot path, a module-level
# IdDict keyed on the transform object is both safe and cheap. The entries
# live as long as the transform itself.
const _TRANSFORM_INTER_SCRATCH = IdDict{Any, Dict{Tuple, AbstractArray}}()

@inline function _get_inter_cache(transform)
    cache = get(_TRANSFORM_INTER_SCRATCH, transform, nothing)
    if cache === nothing
        cache = Dict{Tuple, AbstractArray}()
        _TRANSFORM_INTER_SCRATCH[transform] = cache
    end
    return cache
end

@inline function _get_scratch_for_transform!(transform::FourierTransform, tag::Symbol,
                                             shape::Tuple, ::Type{T}) where {T}
    # FourierTransform has a native fwd_scratch / bwd_scratch dict of the
    # right type; use it directly.
    dict = tag === :fwd_inter ? transform.fwd_scratch : transform.bwd_scratch
    return _get_or_alloc_scratch!(dict, (shape, T, tag), shape, T)
end

@inline function _get_scratch_for_transform!(transform::Transform, tag::Symbol,
                                             shape::Tuple, ::Type{T}) where {T}
    # Generic fallback for other transform types (Chebyshev, Legendre, etc.):
    # use the module-level IdDict cache keyed on the transform object.
    dict = _get_inter_cache(transform)
    return _get_or_alloc_scratch!(dict, (shape, T, tag), shape, T)
end

