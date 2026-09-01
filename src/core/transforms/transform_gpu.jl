"""
    Transform GPU - GPU transform support and heuristics

This file contains GPU-specific transform support. A field configured for a
GPU is never transformed by staging its data through CPU memory: unsupported
GPU transforms fail explicitly.
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
3. At least one Fourier axis (pure-Chebyshev real coefficient storage is not
   supported by the complex distributed driver yet).
4. Every `RealFourier` axis is on dim 1 (the framework's `bases[1]` convention).

A `RealFourier` axis on dim 2 or dim 3 cannot be handled by the distributed
GPU path and is rejected explicitly.
"""
function distributed_gpu_supported(bases::Tuple)
    return _distributed_gpu_dct_bases_supported(bases)
end

"""
    distributed_gpu_supported(field) → Bool

Convenience overload: delegates to `distributed_gpu_supported(field.bases)`.
"""
distributed_gpu_supported(field) = distributed_gpu_supported(field.bases)

"""
    distributed_gpu_unsupported_reason(bases_or_field) → Union{Nothing, String}

The specific reason the multi-GPU DCT-I path declines these bases (`nothing` if
it accepts them). Refusal messages quote this instead of restating the rule, so
they cannot describe a rule the predicate does not apply.
"""
distributed_gpu_unsupported_reason(bases::Tuple) =
    _distributed_gpu_dct_unsupported_reason(bases)
distributed_gpu_unsupported_reason(field) =
    _distributed_gpu_dct_unsupported_reason(field.bases)

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

Set the legacy minimum-size heuristic used by non-GPU transform selection.
GPU-resident fields always use their device transform, regardless of size.
"""
function set_gpu_fft_min_elements!(n::Integer)
    GPU_FFT_MIN_ELEMENTS[] = max(1, Int(n))
    return GPU_FFT_MIN_ELEMENTS[]
end

gpu_fft_min_elements() = GPU_FFT_MIN_ELEMENTS[]

function should_use_gpu_fft(field::ScalarField, data_shape::Tuple)
    mode = gpu_fft_mode(field)
    if is_gpu(field.dist.architecture)
        mode === :cpu && throw(ArgumentError(
            "GPU fields cannot use fft_mode=:cpu because GPU transforms may not " *
            "fall back to CPU. Use :auto or :gpu."))
        return true
    end
    if mode === :gpu
        return true
    elseif mode === :cpu
        return false
    end
    use_gpu = prod(data_shape) >= GPU_FFT_MIN_ELEMENTS[]
    if !use_gpu
        @debug "GPU FFT bypassed by the legacy size heuristic: $(prod(data_shape)) " *
               "elements < threshold $(GPU_FFT_MIN_ELEMENTS[])." maxlog=1
    end
    return use_gpu
end

should_use_gpu_fft(field::ScalarField) = (get_grid_data(field) !== nothing) && should_use_gpu_fft(field, size(get_grid_data(field)))

"""
    GPUTransformUnsupported <: Exception

Raised when no on-device transform covers a field. Carries the REASON, which a
plain `false` return could not: a refusal reads
"scaled Chebyshev axis 2 (26 grid points, basis size 17)" instead of the
shrug that every unsupported configuration used to share.
"""
struct GPUTransformUnsupported <: Exception
    field::String
    bases::String
    direction::Symbol
    reason::String
end

function Base.showerror(io::IO, e::GPUTransformUnsupported)
    print(io, "No on-device $(e.direction) transform supports field $(e.field) ",
          "with bases $(e.bases)")
    isempty(e.reason) || print(io, ": ", e.reason)
    print(io, ". CPU fallback is disabled.")
end

"""
    gpu_transform_unsupported(field, direction, reason)

Refuse an on-device transform with an actionable reason. Backends call this
instead of `return false`; it throws, so it type-checks anywhere a `Bool` is
expected. `false` remains valid for a backend with nothing to say.
"""
@noinline function gpu_transform_unsupported(field, direction::Symbol, reason::AbstractString)
    throw(GPUTransformUnsupported(repr(field.name), string(map(typeof, field.bases)),
                                  direction, String(reason)))
end

_raise_unsupported_gpu_transform(field, direction::Symbol) =
    gpu_transform_unsupported(field, direction, "")

"""
    _gpu_forward_transform_backend!(arch, field) -> Bool
    _gpu_backward_transform_backend!(arch, field) -> Bool

Device backend entry points. A device package (currently `TarangCUDAExt`) adds a
method on the concrete architecture — `::GPU` is strictly more specific than the
`::AbstractArchitecture` fallback below, so the extension EXTENDS these rather
than overwriting them. That specificity gap is the whole reason these exist as
plain functions: an extension method with the same signature as a `Tarang` method
would be method overwriting, which is what the previous `Ref{Any}` hook
indirection was working around. Dispatch gives the same wiring with inference
through the call and no "hook not registered" state to check at runtime.

Return `true` when the transform ran on-device; `false` means unsupported, and
the caller raises (never a CPU fallback).
"""
function _gpu_forward_transform_backend! end
function _gpu_backward_transform_backend! end

_gpu_forward_transform_backend!(::AbstractArchitecture, ::ScalarField) =
    error("GPU forward transform backend is unavailable. Load CUDA.jl before " *
          "constructing GPU fields; CPU fallback is disabled.")
_gpu_backward_transform_backend!(::AbstractArchitecture, ::Any) =
    error("GPU backward transform backend is unavailable. Load CUDA.jl before " *
          "constructing GPU fields; CPU fallback is disabled.")

"""
    gpu_forward_transform!(field::ScalarField)

GPU-specific forward transform, dispatched to the device backend.
Returns `false` only for a CPU field. A GPU field either completes on-device or
throws; it is never handed to the CPU transform chain.
"""
gpu_forward_transform!(field::ScalarField) =
    _gpu_forward_transform!(field.dist.architecture, field)

# "Not a GPU field" is an architecture method, not an `is_gpu` test, so the
# device implementation below is only ever entered with a GPU architecture.
_gpu_forward_transform!(::CPU, ::ScalarField) = false

function _gpu_forward_transform!(arch::GPU, field::ScalarField)
    # Check if data is on GPU
    data_g = get_grid_data(field)
    if !is_gpu_array(data_g)
        error("GPU forward transform requires GPU-resident grid data, but field " *
              "$(repr(field.name)) stores $(typeof(data_g)). Refusing CPU fallback.")
    end

    handled = _gpu_forward_transform_backend!(arch, field)::Bool
    handled || _raise_unsupported_gpu_transform(field, :forward)
    return true
end

"""
    gpu_backward_transform!(field::ScalarField)

GPU-specific backward transform using the registered device backend.
Returns `false` only for a CPU field. A GPU field either completes on-device or
throws; it is never handed to the CPU transform chain.
"""
gpu_backward_transform!(field) =
    _gpu_backward_transform!(field.dist.architecture, field)

# See gpu_forward_transform! — the CPU case is a method, not a branch.
_gpu_backward_transform!(::CPU, _) = false

function _gpu_backward_transform!(arch::GPU, field)
    # Check if data is on GPU
    data_c = get_coeff_data(field)
    if !is_gpu_array(data_c)
        error("GPU backward transform requires GPU-resident coefficient data, but field " *
              "$(repr(field.name)) stores $(typeof(data_c)). Refusing CPU fallback.")
    end

    # See gpu_forward_transform! — the backend is a dispatched method.
    handled = _gpu_backward_transform_backend!(arch, field)::Bool
    handled || _raise_unsupported_gpu_transform(field, :backward)
    return true
end

# -----------------------------------------------------------------------------
# Helper utilities for CPU-only local transforms
# -----------------------------------------------------------------------------

"""
    _execute_on_cpu(f, data)

Run `f` on CPU memory. GPU input is rejected so internal transform helpers
cannot silently download device data.

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
        error("A CPU-only transform was called with GPU data $(typeof(data)); " *
              "CPU fallback is disabled.")
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

    # A distributed GPU field transforms by explicit transposes. PencilFFTs is
    # CPU-only and the local transform chain would compute a per-rank FFT of a
    # slab, which is silently wrong rather than an error.
    if is_transposable_storage(field)
        # Dropping target_layout/apply_coupled_dct here is safe: is_transposable_storage
        # always implies is_gpu(dist.architecture) || !dist.use_pencil_arrays, so
        # plan_transforms! (transform_planning.jl:135) returns before dist.pencil_solve is
        # ever built, and every apply_coupled_dct=false call site (lazy_rhs.jl,
        # subproblem_io.jl) is gated on dist.pencil_solve !== nothing.
        ws = transpose_workspace!(field.dist, field)
        distributed_forward_transform!(ws)
        return
    end

    # Try GPU transform first if on GPU architecture
    if gpu_forward_transform!(field)
        field.current_layout = :c
        return
    end

    # Resolve the plan owned by this field's domain and dtype. Another field on
    # the same Distributor may have a different shape, precision, or transform
    # domain, so the distributor's legacy active plan is not an execution source.
    bundle = _field_transform_bundle(field)
    pencil_plan = _find_pencil_plan(bundle)
    if pencil_plan !== nothing
        # PencilFFTs is CPU-only; GPU data must have been handled by its backend.
        grid_data = get_grid_data(field)
        if is_gpu_array(grid_data)
            error("PencilFFTs cannot transform GPU data; CPU fallback is disabled.")
        else
            plan_input = grid_data
            complex_grid = get(bundle.pencil_work_cache, :complex_grid, nothing)
            if complex_grid !== nothing
                grid_data isa PencilArrays.PencilArray || error(
                    "a PencilFFT complex-grid promotion requires PencilArray field storage")
                parent(complex_grid) .= parent(grid_data)
                plan_input = complex_grid
            end
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
                    mul!(coeff_data, pencil_plan, plan_input)
                catch err
                    @warn "forward_transform!: PencilFFTs mul! fast path failed, falling back to allocating `*`. This costs one PencilArray per transform — consider upgrading PencilFFTs or checking the field buffer layout." exception=(err, catch_backtrace()) maxlog=1
                    set_coeff_data!(field, pencil_plan * plan_input)
                end
            else
                @warn "forward_transform!: coeff buffer is not a PencilArray; using allocating `*` fallback. Expected `allocate_data!` to pre-allocate via PencilFFTs.allocate_output." maxlog=1
                set_coeff_data!(field, pencil_plan * plan_input)
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
        error("Cannot run local FFTW transforms on distributed CPU data without PencilFFTs. " *
              "No PencilFFTPlan found for this domain. " *
              "For MPI+CPU Fourier, set use_pencil_arrays=true in Distributor. " *
              "For MPI+GPU, use TransposableField with distributed transforms.")
    end

    # ── Zero-allocation in-place transform chain ───────────────────────────
    #
    # Walk the field-owned transform vector in order. Each transform has an in-place
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
    transforms = bundle.transforms
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
