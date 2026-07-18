"""
    Transform GPU - GPU transform support and heuristics

This file contains GPU-specific transform support including
FFT heuristics and CPU fallback execution.
"""

# ============================================================================
# Axis-kind classification and distributed-GPU eligibility predicate
# ============================================================================

"""
    axis_kinds(bases::Tuple) → Tuple{Symbol...}

Return a tuple of symbols classifying each basis in `bases`:
- `:real_fourier`    — `RealFourier`
- `:complex_fourier` — `ComplexFourier`
- `:chebyshev`       — `ChebyshevT`

Errors on any unrecognised basis type.
"""
function axis_kinds(bases::Tuple)
    map(bases) do b
        if isa(b, RealFourier)
            :real_fourier
        elseif isa(b, ComplexFourier)
            :complex_fourier
        elseif isa(b, ChebyshevT)
            :chebyshev
        else
            error("Unsupported basis for distributed GPU transform: $(typeof(b))")
        end
    end
end

"""
    distributed_gpu_supported(bases::Tuple) → Bool

Return `true` iff the basis tuple is eligible for the distributed GPU
Chebyshev DCT-I transform path. The conditions are:
1. Exactly 3 dimensions.
2. At least one `ChebyshevT` axis.
3. Every `RealFourier` axis is on dim 1 (the framework's `bases[1]` convention).

A `RealFourier` axis on dim 2 or dim 3 cannot be handled by the distributed
GPU path and must fall back to CPU.
"""
function distributed_gpu_supported(bases::Tuple)
    length(bases) == 3 || return false
    kinds = axis_kinds(bases)
    any(==(:chebyshev), kinds) || return false
    for (dim, k) in enumerate(kinds)
        if k === :real_fourier && dim != 1
            return false
        end
    end
    # RealFourier on dim 1 combined with a Fourier transverse axis: the forward
    # pipeline completes, but the backward Hermitian expansion needs the
    # conjugate partner at the FLIPPED transverse wavenumber and hard-errors
    # (see the guard in distributed_backward_dct!). Reject at plan level so the
    # layout falls back to CPU instead of dying on the first backward transform.
    if kinds[1] === :real_fourier &&
       any(k -> k === :complex_fourier || k === :real_fourier, kinds[2:end])
        return false
    end
    return true
end

"""
    distributed_gpu_supported(field) → Bool

Convenience overload: delegates to `distributed_gpu_supported(field.bases)`.
"""
distributed_gpu_supported(field) = distributed_gpu_supported(field.bases)

# ============================================================================
# Hermitian half-spectrum → full-spectrum expansion (1-D CPU reference)
# ============================================================================

"""
    _hermitian_full_from_half(half::AbstractVector{<:Complex}, N::Int) → Vector

Expand a half-spectrum of length `div(N,2)+1` (the non-redundant coefficients
of a real-valued signal stored by RFFT convention) to the full complex spectrum
of length `N` using Hermitian symmetry:

    full[N - k + 2] = conj(full[k])   for k = 2 … (N - div(N,2))

`full[1]` (DC) and — for even `N` — `full[div(N,2)+1]` (Nyquist) are real by
construction; this function does not enforce that (it copies them as-is from
`half`).

Works for both even and odd `N`.  A later GPU kernel reproduces this exact index
map along dim 1 of a 3-D CuArray.
"""
function _hermitian_full_from_half(half::AbstractVector{<:Complex}, N::Int)
    M = div(N, 2) + 1
    @assert length(half) == M "half length must be div(N,2)+1 = $M, got $(length(half))"
    full = similar(half, N)
    @inbounds for k in 1:M
        full[k] = half[k]
    end
    @inbounds for k in 2:(N - M + 1)        # fill the mirror; covers even and odd N
        full[N - k + 2] = conj(half[k])
    end
    return full
end

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
const _GPU_FORWARD_TRANSFORM_HOOK = Ref{Any}(nothing)
const _GPU_BACKWARD_TRANSFORM_HOOK = Ref{Any}(nothing)

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

    # The CUDA extension registers the implementation from its __init__
    # (a same-signature method here would be illegal method overwriting).
    h = _GPU_FORWARD_TRANSFORM_HOOK[]
    if h === nothing
        @warn "GPU architecture specified but CUDA extension not loaded. Falling back to CPU." maxlog=1
        return false
    end
    return h(field)::Bool
end

"""
    gpu_backward_transform!(field::ScalarField)

GPU-specific backward transform using CUFFT.
Returns true if GPU transform was applied, false otherwise.
"""
function gpu_backward_transform!(field)
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

    # See gpu_forward_transform! — implementation is hook-registered.
    h = _GPU_BACKWARD_TRANSFORM_HOOK[]
    if h === nothing
        @warn "GPU architecture specified but CUDA extension not loaded. Falling back to CPU." maxlog=1
        return false
    end
    return h(field)::Bool
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
function forward_transform!(field::ScalarField, target_layout::Symbol=:c; apply_coupled_dct::Bool=true)

    if field.domain === nothing
        return
    end
    _count_transform!(:forward)

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
            if isa(grid_data, PencilArrays.PencilArray)
                if coeff_data === nothing || !isa(coeff_data, PencilArrays.PencilArray)
                    coeff_data = PencilFFTs.allocate_output(pencil_plan)
                    set_coeff_data!(field, coeff_data)
                end
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
        # The PencilFFT plan transforms ONLY the Fourier axes; a coupled
        # (Chebyshev/Jacobi) axis is left in GRID space. Apply its local DCT now so
        # distributed `:c` holds true spectral coefficients (no-op unless mixed+MPI).
        # `apply_coupled_dct=false` (used ONLY by `to_solve_layout!`'s fused
        # grid→solve path) SKIPS it: the caller applies the coupled DCT directly in
        # the solve pencil after a single fft→solve transpose, avoiding the redundant
        # solve→fft back-transpose here that `to_solve_layout!` would immediately undo.
        # The field is then left Fourier-spectral but coupled-axis GRID; the caller
        # resolves it to fully-spectral before the `:c` flag is observed.
        apply_coupled_dct && _apply_distributed_coupled_dct!(field, true)
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

    # Index-based loop (not `enumerate`) avoids a `Tuple{Int, Any}` heap
    # allocation per step when `transforms isa Vector{Any}`. Each stage runs
    # behind `_forward_transform_stage!`, a function barrier — see its docstring.
    for idx in 1:n_transforms
        transform = transforms[idx]
        current = _forward_transform_stage!(field, transform, current, idx == n_transforms)
    end
    field.current_layout = :c
end

"""
    _forward_transform_stage!(field, transform, in_arr, is_final) → out_arr

Function barrier for one forward transform stage. `transform` arrives as
`Any` (element of `dist.transforms::Vector{Any}`) and `in_arr` as an abstract
`AbstractArray`; calling through this boundary makes Julia dispatch on the
concrete transform type AND specialize on `typeof(in_arr)` exactly once. Inside,
`_forward_output_spec`, scratch lookup, plan lookup and `mul!` all resolve
statically, so a warm cache runs allocation-free.

Without the barrier this work runs inline in the type-unstable `forward_transform!`
body, where per-stage dynamic dispatch + `Any`-tuple destructuring + shape
splatting box heavily on Julia 1.10 (≈6.6 KiB/round-trip vs the 2 KiB test budget).
"""
function _forward_transform_stage!(field::ScalarField, transform, in_arr::AbstractArray,
                                   is_final::Bool)
    out_shape, out_eltype = _forward_output_spec(in_arr, transform)
    if is_final
        # Final stage: target is the field's coeff buffer. Reuse when
        # shape/eltype match (the common case); otherwise allocate once.
        coeff = get_coeff_data(field)
        if !_buffer_matches(coeff, out_shape, out_eltype)
            coeff = zeros(out_eltype, out_shape...)
            set_coeff_data!(field, coeff)
        end
        _apply_forward!(coeff, in_arr, transform)
        return coeff
    end
    # Intermediate stage: write into this transform's cached scratch, keyed by
    # (out_shape, eltype_tag, SLOT_FWD_INTER) to avoid colliding with other scratch.
    out = _get_scratch_for_transform!(transform, SLOT_FWD_INTER, out_shape, out_eltype)
    _apply_forward!(out, in_arr, transform)
    return out
end

# ---------------------------------------------------------------------------
# Helper: fetch an intermediate-output scratch buffer from a transform.
# ---------------------------------------------------------------------------
# ChebyshevTransform's `fwd_scratch` / `bwd_scratch` fields are typed
# `Dict{Tuple, ChebScratch}` (hold real/imag/plan scratch records, not raw
# arrays), so we can't reuse them for the intermediate-output buffer.
# For ChebyshevTransform we therefore fall back to a small per-transform
# scratch Dict stored in module-level weak-key storage.
#
# Since transform objects are mutable, a WeakKeyDict lets their scratch
# buffers disappear automatically once the owning transform becomes
# unreachable. This avoids retaining old transform plans and workspaces
# across solver rebuilds.
const _TRANSFORM_INTER_SCRATCH = WeakKeyDict{Any, Dict{Tuple, AbstractArray}}()

@inline function _get_inter_cache(transform)
    cache = get(_TRANSFORM_INTER_SCRATCH, transform, nothing)
    if cache === nothing
        cache = Dict{Tuple, AbstractArray}()
        _TRANSFORM_INTER_SCRATCH[transform] = cache
    end
    return cache
end

# `slot` is one of SLOT_FWD_INTER / SLOT_BWD_INTER (UInt8 literals) so the cache
# key `(shape, eltype_tag, slot)` is isbits — no per-call key allocation.
@inline function _get_scratch_for_transform!(transform::FourierTransform, slot::UInt8,
                                             shape::NTuple{N,Int}, ::Type{T}) where {N,T}
    # FourierTransform has a native fwd_scratch / bwd_scratch dict of the
    # right type; use it directly.
    dict = slot === SLOT_FWD_INTER ? transform.fwd_scratch : transform.bwd_scratch
    return _get_or_alloc_scratch!(dict, (shape, _fft_eltype_tag(T), slot), shape, T)
end

@inline function _get_scratch_for_transform!(transform::Transform, slot::UInt8,
                                             shape::NTuple{N,Int}, ::Type{T}) where {N,T}
    # Generic fallback for other transform types (Chebyshev, Legendre, etc.):
    # use the module-level weak-key cache keyed on the transform object.
    dict = _get_inter_cache(transform)
    return _get_or_alloc_scratch!(dict, (shape, _fft_eltype_tag(T), slot), shape, T)
end
