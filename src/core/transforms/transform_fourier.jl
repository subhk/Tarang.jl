"""
    Transform Fourier - Fourier transform execution

This file contains the forward and backward Fourier transform implementations.
"""

function _fourier_forward(data::AbstractArray, transform::FourierTransform)
    return _execute_on_cpu(data) do host_data
        # Use precomputed plan only if data size and element type match the plan
        # (FFTW plans are type-specific: a Float64 plan cannot be applied to Float32 data)
        if ndims(host_data) == 1 && transform.plan_forward !== nothing &&
           length(host_data) == transform.basis.meta.size &&
           eltype(host_data) === transform.plan_dtype
            return transform.plan_forward * host_data
        end

        dims = (transform.axis,)
        if isa(transform.basis, RealFourier)
            # rfft requires real input; if data is already complex (from a prior
            # rfft or fft along another axis in a multi-dimensional transform),
            # use fft instead. This is standard for multi-dimensional real FFTs:
            # rfft is applied along one axis, fft along the remaining axes.
            if eltype(host_data) <: Complex
                return FFTW.fft(host_data, dims)
            end
            return FFTW.rfft(host_data, dims)
        end

        return FFTW.fft(host_data, dims)
    end
end

"""Opt-in transform counter (OFF by default; one `Ref` load when off).

Spectral transforms are the dominant MPI cost, and a redundant one is INVISIBLE to an
allocation guard — `ldiv!`/`mul!` are in-place, so a wasted distributed FFT allocates
nothing. (A real regression of exactly this kind shipped and was caught only by hand-counting.)
This lets a test assert the transform BUDGET of a step directly:

    Tarang.reset_transform_counts!()
    step!(solver, dt)
    Tarang.transform_counts()   # (forward = n, backward = m, coupled_dct = k)

`coupled_dct` counts round-trips through `_apply_distributed_coupled_dct!`, each of which is
TWO collective `PencilArrays.transpose!` calls. It is tracked separately because it is the
dominant collective cost of a distributed mixed Fourier–Chebyshev step, and — like a wasted
FFT — it allocates nothing, so no allocation guard can see it.
"""
const _TRANSFORM_COUNT_ON = Ref(false)
const _TRANSFORM_COUNTS = Ref((0, 0, 0))   # (forward, backward, coupled_dct)

enable_transform_counts!(on::Bool=true) = (_TRANSFORM_COUNT_ON[] = on)
reset_transform_counts!() = (_TRANSFORM_COUNTS[] = (0, 0, 0); nothing)
transform_counts() = (forward     = _TRANSFORM_COUNTS[][1],
                      backward    = _TRANSFORM_COUNTS[][2],
                      coupled_dct = _TRANSFORM_COUNTS[][3])

@inline function _count_transform!(kind::Symbol)
    _TRANSFORM_COUNT_ON[] || return nothing
    f, b, c = _TRANSFORM_COUNTS[]
    _TRANSFORM_COUNTS[] = kind === :forward     ? (f + 1, b, c) :
                          kind === :backward    ? (f, b + 1, c) :
                                                  (f, b, c + 1)
    return nothing
end

"""Apply backward transform to field """
function backward_transform!(field::ScalarField, target_layout::Symbol=:g; apply_coupled_dct::Bool=true)

    if field.domain === nothing
        return
    end
    _count_transform!(:backward)

    ensure_layout!(field, :c)  # Start in coefficient space

    # Try GPU transform first if on GPU architecture
    if gpu_backward_transform!(field)
        field.current_layout = :g
        return
    end

    # Find appropriate transform
    pencil_plan = _find_pencil_plan(field.dist)
    if pencil_plan !== nothing
        # Invert the coupled (Chebyshev/Jacobi) DCT BEFORE the PencilFFT ldiv!:
        # distributed `:c` is Chebyshev-SPECTRAL, but the PencilFFT plan handles only
        # the Fourier axes and needs the coupled axis in GRID space. No-op unless mixed+MPI.
        # `apply_coupled_dct=false` (used ONLY by `from_solve_layout!`'s fused solve→grid
        # path) SKIPS it: the caller already inverted the coupled DCT locally in the solve
        # pencil, leaving the coupled axis in GRID space, so only the Fourier `ldiv!`
        # remains here. Mirrors the `forward_transform!` `apply_coupled_dct` kwarg.
        apply_coupled_dct && _apply_distributed_coupled_dct!(field, false)
        # PencilFFTs is CPU-only; GPU data must have been handled by its backend.
        coeff_data = get_coeff_data(field)
        if is_gpu_array(coeff_data)
            error("PencilFFTs cannot transform GPU data; CPU fallback is disabled.")
        else
            # In-place backward via `ldiv!` when the grid buffer is a
            # pre-allocated PencilArray (the common case — `allocate_data!`
            # uses PencilFFTs.allocate_input which returns one). Mirrors
            # the forward branch's `mul!(coeff_data, plan, grid_data)`
            # fast path. Falls back to the allocating `\` if ldiv! raises
            # (e.g. on PencilFFTs versions that don't support it on this
            # plan type). The fallback is logged once so an MPI run will
            # make it obvious if the fast path isn't firing.
            grid_data = get_grid_data(field)
            if isa(coeff_data, PencilArrays.PencilArray)
                if grid_data === nothing || !isa(grid_data, PencilArrays.PencilArray)
                    grid_data = PencilFFTs.allocate_input(pencil_plan)
                    set_grid_data!(field, grid_data)
                end
                try
                    ldiv!(grid_data, pencil_plan, coeff_data)
                catch err
                    @warn "backward_transform!: PencilFFTs ldiv! fast path failed, falling back to allocating `\\`. This costs one PencilArray per transform — consider upgrading PencilFFTs or checking the field buffer layout." exception=(err, catch_backtrace()) maxlog=1
                    set_grid_data!(field, pencil_plan \ coeff_data)
                end
            else
                @warn "backward_transform!: grid buffer is not a PencilArray; using allocating `\\` fallback. Expected `allocate_data!` to pre-allocate via PencilFFTs.allocate_input." maxlog=1
                set_grid_data!(field, pencil_plan \ coeff_data)
            end
        end
        field.current_layout = :g
        return
    end

    # CRITICAL: Guard against running local transforms on distributed data
    # If we reach here with dist.size > 1, no PencilFFTPlan was found above,
    # so local transforms would produce incorrect results on distributed data.
    if field.dist.size > 1
        if is_gpu_array(get_coeff_data(field))
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

    # ── Zero-allocation in-place backward transform chain ─────────────────
    # Mirror of the forward path in transform_gpu.jl. Walk transforms in
    # reverse order. For intermediate stages, write into a cached scratch;
    # for the FINAL stage, write directly into the field's pre-allocated
    # grid buffer.
    #
    # We iterate via index arithmetic (`n_transforms:-1:1`) rather than
    # `collect(reverse(...))` to avoid allocating a fresh reversed vector
    # each call.
    current = get_coeff_data(field)
    transforms = field.dist.transforms
    n_transforms = length(transforms)
    if n_transforms == 0
        grid = get_grid_data(field)
        if grid !== nothing && size(grid) == size(current) && eltype(grid) == eltype(current)
            copyto!(grid, current)
        else
            set_grid_data!(field, copy(current))
        end
        field.current_layout = :g
        return
    end

    # Walk transforms in reverse order. `step` counts 1..n_transforms so
    # we can detect the final step (step == n_transforms) to write directly
    # into the field's pre-allocated grid buffer. Each stage runs behind
    # `_backward_transform_stage!`, a function barrier — see its docstring.
    for step in 1:n_transforms
        transform = transforms[n_transforms - step + 1]
        current = _backward_transform_stage!(field, transform, current, step == n_transforms)
    end
    field.current_layout = :g
end

"""
    _backward_transform_stage!(field, transform, in_arr, is_final) → out_arr

Function barrier for one backward transform stage — the mirror of
`_forward_transform_stage!`. `transform` arrives as `Any` (element of
`dist.transforms::Vector{Any}`) and `in_arr` as an abstract `AbstractArray`;
dispatching through this boundary makes Julia specialize the body on both
concrete types so `_backward_output_spec`, scratch/plan lookup and the in-place
`mul!` resolve statically and run allocation-free on a warm cache.
"""
# Scaled grid size along a Fourier transform's axis (scale × base size). Threaded
# into the backward path so the irfft targets the scaled grid, not the base. 0 for
# non-Fourier / non-RealFourier axes (no scale context needed → base-size fallback).
_bwd_rfft_target(field::ScalarField, transform) = 0
function _bwd_rfft_target(field::ScalarField, transform::FourierTransform)
    isa(transform.basis, RealFourier) || return 0
    ax = transform.axis
    scale = field.scales === nothing ? 1.0 : field.scales[ax]
    # MUST match the scaled-grid allocation in get_scaled_shape (field_data_scales.jl),
    # which uses ceil(N*scale). Using round here disagreed for non-integer N*scale
    # (e.g. N=7, scale=1.5 → ceil 11 vs round 10), so the backward irfft targeted the
    # wrong grid length and the forward∘backward round-trip was broken.
    return ceil(Int, scale * transform.basis.meta.size)
end

function _backward_transform_stage!(field::ScalarField, transform, in_arr::AbstractArray,
                                    is_final::Bool)
    out_n = _bwd_rfft_target(field, transform)
    out_shape, out_eltype = _backward_output_spec(in_arr, transform, out_n)
    # `out_eltype` is decided at runtime — real for the final irfft stage,
    # complex for the ifft stages — so it is a type-UNSTABLE `DataType` value.
    # Forward output is always complex (stable), but backward must union-split
    # here: matching `out_eltype` against the concrete FFT element types pins it
    # to a compile-time `T` inside `_backward_stage_typed!`, so the buffer match,
    # scratch fetch and `mul!` specialize and run allocation-free. Without this,
    # the unstable eltype boxes a `Tuple{Int,Int}` on every backward stage.
    # The branches must cover every FFT element type and the fallback must NOT
    # forward the abstract `out_eltype` — an `error` returns `Union{}` so it drops
    # out of the inferred return type, leaving a small concrete union
    # (`Union{Matrix{Float64}, Matrix{ComplexF64}, …}`). Forwarding `out_eltype`
    # instead widened the return to `Union{Nothing, AbstractArray}`, which x86_64
    # codegen boxes on every backward stage.
    if out_eltype === ComplexF64
        return _backward_stage_typed!(field, transform, in_arr, is_final, out_shape, ComplexF64, out_n)
    elseif out_eltype === Float64
        return _backward_stage_typed!(field, transform, in_arr, is_final, out_shape, Float64, out_n)
    elseif out_eltype === ComplexF32
        return _backward_stage_typed!(field, transform, in_arr, is_final, out_shape, ComplexF32, out_n)
    elseif out_eltype === Float32
        return _backward_stage_typed!(field, transform, in_arr, is_final, out_shape, Float32, out_n)
    else
        error("Tarang backward transform: unsupported output element type $out_eltype")
    end
end

@inline function _backward_stage_typed!(field::ScalarField, transform, in_arr::AbstractArray,
                                        is_final::Bool, out_shape::NTuple{N,Int},
                                        ::Type{T}, out_n::Int) where {N,T}
    if is_final
        # Bind `grid` to a concrete `Array{T,N}` via the ternary: the `isa` test
        # narrows the existing buffer in the true branch, and `_alloc_grid_buffer!`
        # returns `Array{T,N}` in the false branch — so both arms are concrete and
        # the function's return type is `Array{T,N}`, not `Union{Nothing,AbstractArray}`.
        # The previous `if !_buffer_matches(...)` form left the return abstract,
        # which Julia 1.10/1.11 boxes on x86_64 (≈5.5 KiB/round-trip; arm64 elides it).
        existing = get_grid_data(field)
        grid::Array{T,N} = (existing isa Array{T,N} && size(existing) == out_shape) ?
            existing : _alloc_grid_buffer!(field, T, out_shape)
        _apply_backward!(grid, in_arr, transform, out_n)
        return grid
    end
    out::Array{T,N} = _get_scratch_for_transform!(transform, SLOT_BWD_INTER, out_shape, T)
    _apply_backward!(out, in_arr, transform, out_n)
    return out
end

@inline function _alloc_grid_buffer!(field::ScalarField, ::Type{T}, shape::NTuple{N,Int}) where {N,T}
    g = zeros(T, shape...)
    set_grid_data!(field, g)
    return g
end

function _fourier_backward(data::AbstractArray, transform::FourierTransform, out_n::Int=0)
    grid_n = _bwd_grid_size(transform, out_n)
    return _execute_on_cpu(data) do host_data
        # Use the precomputed BASE 1D plan only if data matches it (it was built at
        # setup for the base size; a scaled field's data won't match and falls through).
        base_rfft_size = isa(transform.basis, RealFourier) ? div(transform.basis.meta.size, 2) + 1 : transform.basis.meta.size
        if ndims(host_data) == 1 && transform.plan_backward !== nothing &&
           length(host_data) == base_rfft_size &&
           eltype(host_data) === Complex{transform.plan_dtype}
            return transform.plan_backward * host_data
        end

        dims = (transform.axis,)
        if isa(transform.basis, RealFourier)
            # Detect rfft vs fft against the SCALED grid size (irfft target = grid_n).
            actual_size = size(host_data, transform.axis)
            if actual_size == div(grid_n, 2) + 1
                return FFTW.irfft(host_data, grid_n, dims)
            else
                # fft was used (complex input case), use ifft
                return FFTW.ifft(host_data, dims)
            end
        end

        return FFTW.ifft(host_data, dims)
    end
end

# Dispatch methods for transform loop (replaces isa() chains)
_apply_forward(current, t::FourierTransform) = _fourier_forward(current, t)
_apply_backward(current, t::FourierTransform) = _fourier_backward(current, t)

# ============================================================================
# In-place Fourier transforms — zero-alloc once plan+scratch caches warm
# ============================================================================
#
# The original out-of-place paths above (`_fourier_forward` /
# `_fourier_backward`) are kept for rare callers and as a fallback. The
# in-place entry points below are used by `forward_transform!` /
# `backward_transform!` on the hot path.
#
# Per-call work for a warm cache:
#   1. Dict lookup on `transform.fwd_plan_cache[(size, eltype)]` → plan
#   2. `mul!(out, plan, in)` — pure FFTW call, no allocations
#
# Separate plan caches are needed because FFTW plans are parameterized
# by input size, element type, AND strides. A plan built for a (192, 48)
# Float64 array cannot be applied to a (192, 32) array or to a Float32
# array. The `(size, eltype)` key captures the relevant dimensions.
#
# For RealFourier, the forward transform uses `plan_rfft` when the input
# is real, but a full complex `plan_fft` when the input is already
# complex (which happens when Fourier is NOT the first transform in the
# chain and a previous transform produced complex output). These two
# plan types have different output shapes and are cached under separate
# keys (`(size, Float64)` vs `(size, ComplexF64)`).

@inline _replace_axis_shape(shape::Tuple{Int}, axis::Int, value::Int) =
    axis == 1 ? (value,) : throw(BoundsError(shape, axis))

@inline function _replace_axis_shape(shape::Tuple{Int, Int}, axis::Int, value::Int)
    axis == 1 && return (value, shape[2])
    axis == 2 && return (shape[1], value)
    throw(BoundsError(shape, axis))
end

@inline function _replace_axis_shape(shape::Tuple{Int, Int, Int}, axis::Int, value::Int)
    axis == 1 && return (value, shape[2], shape[3])
    axis == 2 && return (shape[1], value, shape[3])
    axis == 3 && return (shape[1], shape[2], value)
    throw(BoundsError(shape, axis))
end

@inline _replace_axis_shape(shape::Tuple, axis::Int, value::Int) =
    ntuple(i -> i == axis ? value : shape[i], length(shape))

function _forward_output_spec(in::AbstractArray, transform::FourierTransform)
    in_shape = size(in)
    in_eltype = eltype(in)
    real_T = in_eltype <: Complex ? real(in_eltype) : in_eltype
    complex_T = Complex{real_T}
    ax = transform.axis

    if isa(transform.basis, RealFourier)
        if in_eltype <: Complex
            # Already-complex input: full fft, shape unchanged
            return in_shape, complex_T
        end
        # Real input: rfft halves the transform axis from N to div(N, 2) + 1
        out_shape = _replace_axis_shape(in_shape, ax, div(in_shape[ax], 2) + 1)
        return out_shape, complex_T
    else  # ComplexFourier
        return in_shape, complex_T
    end
end

# `out_n` is the SCALED grid size along this transform's axis (scale × base),
# threaded from the backward stage which has the field. Using it instead of the
# base `basis.meta.size` makes rfft-axis detection and the irfft target size
# correct for scaled fields. `out_n <= 0` means "no scale info" → fall back to the
# base size (callers without a field, e.g. the out-of-place legacy path).
_bwd_grid_size(transform::FourierTransform, out_n::Int) =
    out_n > 0 ? out_n : transform.basis.meta.size

# 3-arg dispatch: non-Fourier transforms ignore out_n (delegate to their 2-arg spec).
_backward_output_spec(in::AbstractArray, transform, out_n::Int) = _backward_output_spec(in, transform)

function _backward_output_spec(in::AbstractArray, transform::FourierTransform, out_n::Int)
    in_shape = size(in)
    in_eltype = eltype(in)
    real_T = in_eltype <: Complex ? real(in_eltype) : in_eltype
    ax = transform.axis

    if isa(transform.basis, RealFourier)
        grid_n = _bwd_grid_size(transform, out_n)
        expected_rfft_size = div(grid_n, 2) + 1
        axis_len = in_shape[ax]
        base_rfft_size = div(transform.basis.meta.size, 2) + 1
        if axis_len == expected_rfft_size
            # irfft: axis expands back from N/2+1 to the (scaled) grid size
            out_shape = _replace_axis_shape(in_shape, ax, grid_n)
            return out_shape, real_T
        elseif grid_n > transform.basis.meta.size && axis_len == base_rfft_size
            # UPSAMPLED rfft axis: the stored half-spectrum is the BASE length
            # div(base_N,2)+1, but the target grid is finer (grid_n > base_N). The
            # backward zero-pads the half-spectrum to div(grid_n,2)+1 and irfft-s
            # to the real grid of length grid_n. Gating only on grid_n (the old
            # `axis_len == expected_rfft_size`) missed this case and fell through
            # to a same-shape complex ifft, returning wrong-length complex output.
            out_shape = _replace_axis_shape(in_shape, ax, grid_n)
            return out_shape, real_T
        else
            # ifft fallback (complex input from a non-first-axis fft): same shape, still complex
            return in_shape, Complex{real_T}
        end
    else  # ComplexFourier
        return in_shape, Complex{real_T}
    end
end

# Preserve the 2-arg form (base size) for any caller without scale context.
function _backward_output_spec(in::AbstractArray, transform::FourierTransform)
    return _backward_output_spec(in, transform, 0)
end

"""Get or create a cached forward plan for this (input_size, input_eltype)."""
function _get_or_plan_forward!(transform::FourierTransform, in::AbstractArray)
    in_shape = size(in)
    in_eltype = eltype(in)
    key = (in_shape, _fft_eltype_tag(in_eltype))
    cached = get(transform.fwd_plan_cache, key, nothing)
    if cached !== nothing
        return cached
    end

    dims = (transform.axis,)
    plan = if isa(transform.basis, RealFourier) && !(in_eltype <: Complex)
        FFTW.plan_rfft(in, dims)
    else
        FFTW.plan_fft(in, dims)
    end
    transform.fwd_plan_cache[key] = plan
    return plan
end

"""Get or create a cached backward plan for this (input_size, input_eltype).
`out_n` is the scaled grid size along the axis (0 ⇒ use the base basis size)."""
function _get_or_plan_backward!(transform::FourierTransform, in::AbstractArray, out_n::Int=0)
    in_shape = size(in)
    in_eltype = eltype(in)
    grid_n = _bwd_grid_size(transform, out_n)
    # Key includes grid_n: a scaled vs unscaled field of the same coeff shape needs
    # distinct irfft plans (different output length).
    key = (in_shape, _fft_eltype_tag(in_eltype), grid_n)
    cached = get(transform.bwd_plan_cache, key, nothing)
    if cached !== nothing
        return cached
    end

    dims = (transform.axis,)
    plan = if isa(transform.basis, RealFourier)
        expected_rfft_size = div(grid_n, 2) + 1
        axis_len = in_shape[transform.axis]
        base_n = transform.basis.meta.size
        if axis_len == expected_rfft_size
            # irfft path: plan_irfft needs a dummy input of the same shape
            FFTW.plan_irfft(in, grid_n, dims)
        elseif grid_n > base_n && axis_len == div(base_n, 2) + 1
            # UPSAMPLED rfft axis: the half-spectrum is zero-padded along the axis
            # to div(grid_n,2)+1 before the irfft, so the plan must be built on a
            # dummy of that padded shape (not the base-length `in`).
            padded_shape = _replace_axis_shape(in_shape, transform.axis, expected_rfft_size)
            dummy = zeros(eltype(in), padded_shape...)
            FFTW.plan_irfft(dummy, grid_n, dims)
        else
            FFTW.plan_ifft(in, dims)
        end
    else  # ComplexFourier
        FFTW.plan_ifft(in, dims)
    end
    transform.bwd_plan_cache[key] = plan
    return plan
end

"""
    _apply_forward!(out, in, transform::FourierTransform) → out

Zero-allocation forward Fourier transform into pre-allocated `out`.
Caller must ensure `out` has the shape/eltype returned by
`_forward_output_spec(in, transform)`.
"""
function _apply_forward!(out::AbstractArray, in::AbstractArray, transform::FourierTransform)
    # Only FFTW (CPU) arrays are handled here; GPU-arch fields are dispatched
    # on-device (or error) before the CPU chain. Reaching this with device data
    # means a dispatch bug — refuse loudly rather than host-compute. (The old
    # defensive branch silently computed on host and silently SKIPPED the
    # copyto! on any shape/eltype mismatch, returning stale `out`.)
    if is_gpu_array(in) || is_gpu_array(out)
        error("_apply_forward!(::FourierTransform): GPU data reached the CPU transform " *
              "chain (in=$(typeof(in)), out=$(typeof(out))); CPU fallback is disabled.")
    end

    plan = _get_or_plan_forward!(transform, in)
    return _fourier_fwd_kernel!(out, in, plan)
end

@inline function _fourier_fwd_kernel!(out::AbstractArray, in::AbstractArray, plan::P) where {P}
    mul!(out, plan, in)
    return out
end

"""
    _apply_backward!(out, in, transform::FourierTransform) → out

Zero-allocation backward Fourier transform into pre-allocated `out`.
"""
# 4-arg dispatch: non-Fourier transforms ignore out_n (delegate to their 3-arg form).
_apply_backward!(out::AbstractArray, in::AbstractArray, transform, out_n::Int) =
    _apply_backward!(out, in, transform)
# 3-arg Fourier form: no scale context → base size (out_n = 0).
_apply_backward!(out::AbstractArray, in::AbstractArray, transform::FourierTransform) =
    _apply_backward!(out, in, transform, 0)

function _apply_backward!(out::AbstractArray, in::AbstractArray, transform::FourierTransform, out_n::Int)
    # Mirror of _apply_forward!: device data here is a dispatch bug — refuse
    # loudly, never host-compute (and never silently skip the copyto!).
    if is_gpu_array(in) || is_gpu_array(out)
        error("_apply_backward!(::FourierTransform): GPU data reached the CPU transform " *
              "chain (in=$(typeof(in)), out=$(typeof(out))); CPU fallback is disabled.")
    end

    plan = _get_or_plan_backward!(transform, in, out_n)

    # FFTW irfft plans are destructive on the input array. The high-level
    # `FFTW.irfft(x, n, dims)` protects callers by internally copying `x`
    # into a workspace — exactly the allocation we're trying to avoid.
    # Use a cached scratch that's reused across calls. Only irfft needs
    # this; ifft (complex → complex) is non-destructive.
    is_irfft_path = false
    is_upsampled_irfft = false
    if isa(transform.basis, RealFourier)
        # Detect against the SCALED grid size (grid_n), mirroring _get_or_plan_backward!.
        grid_n = _bwd_grid_size(transform, out_n)
        base_n = transform.basis.meta.size
        axis_len = size(in, transform.axis)
        if axis_len == div(grid_n, 2) + 1
            is_irfft_path = true
        elseif grid_n > base_n && axis_len == div(base_n, 2) + 1
            # UPSAMPLED rfft axis: stored half-spectrum is the base length but the
            # target grid is finer — zero-pad the half-spectrum then irfft.
            is_upsampled_irfft = true
        end
    end

    if is_upsampled_irfft
        grid_n = _bwd_grid_size(transform, out_n)
        base_n = transform.basis.meta.size
        ax = transform.axis
        padded_shape = _replace_axis_shape(size(in), ax, div(grid_n, 2) + 1)
        scratch_key = (padded_shape, _fft_eltype_tag(eltype(in)), SLOT_IRFFT)
        scratch = _get_or_alloc_scratch!(transform.bwd_scratch, scratch_key,
                                         padded_shape, eltype(in))
        fill!(scratch, zero(eltype(scratch)))
        # Copy the base half-spectrum into the low modes and rescale by
        # grid_n/base_n: irfft divides by the (finer) grid_n while the stored
        # coeffs were formed on base_n points, so without this factor the
        # interpolated grid amplitude would be scaled by base_n/grid_n.
        low = ntuple(i -> i == ax ? (1:size(in, ax)) : Colon(), ndims(in))
        @views scratch[low...] .= in .* (grid_n / base_n)
        if iseven(base_n)
            # The base Nyquist mode (C_{N/2}) is ambiguous on the finer grid;
            # drop it so this path agrees with the grid-space resample_1d! upsample.
            nyq_i = div(base_n, 2) + 1
            nyq = ntuple(i -> i == ax ? (nyq_i:nyq_i) : Colon(), ndims(in))
            @views scratch[nyq...] .= 0
        end
        return _fourier_bwd_kernel!(out, scratch, plan)
    elseif is_irfft_path
        scratch_key = (size(in), _fft_eltype_tag(eltype(in)), SLOT_IRFFT)
        scratch = _get_or_alloc_scratch!(transform.bwd_scratch, scratch_key,
                                         size(in), eltype(in))
        copyto!(scratch, in)
        return _fourier_bwd_kernel!(out, scratch, plan)
    else
        return _fourier_bwd_kernel!(out, in, plan)
    end
end

@inline function _fourier_bwd_kernel!(out::AbstractArray, in::AbstractArray, plan::P) where {P}
    mul!(out, plan, in)
    return out
end

# Axis-aware Chebyshev helpers
function _scale_along_axis!(data::AbstractArray, axis::Int, scale::AbstractVector{<:Real})
    if axis > ndims(data)
        return
    end
    shape = ntuple(i -> i == axis ? length(scale) : 1, ndims(data))
    data .*= reshape(scale, shape...)
end
