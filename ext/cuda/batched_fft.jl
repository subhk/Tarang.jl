# ============================================================================
# Batched FFT Support
# ============================================================================

"""
    BatchedGPUFFTPlan

FFT plan for batched transforms on multiple arrays simultaneously.
More efficient than individual FFTs for multi-field operations.
"""
struct BatchedGPUFFTPlan{P, IP}
    plan::P
    iplan::IP
    field_size::Tuple{Vararg{Int}}
    batch_size::Int
    is_real::Bool
end

"""
    plan_batched_gpu_fft(arch::GPU, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false)

Create a batched GPU FFT plan for multiple fields.

# Arguments
- `arch`: GPU architecture
- `field_size`: Size of each individual field
- `T`: Element type
- `batch_size`: Number of fields to transform simultaneously
- `real_input`: If true, create real-to-complex plan

# Example
```julia
plan = plan_batched_gpu_fft(GPU(), (64, 64), Float64, 4; real_input=true)
```

**Important:** `field_size` should be the LOCAL field shape (what this process owns),
not the global domain size.
"""
function plan_batched_gpu_fft(arch::GPU{CuDevice}, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false)
    # Ensure correct device for multi-GPU support
    ensure_device!(arch)
    _create_batched_fft_plan(field_size, T, batch_size, real_input)
end

# Fallback for generic GPU: delegates to device-specific version using current device,
# ensuring proper device context via ensure_device!
plan_batched_gpu_fft(arch::GPU, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false) =
    plan_batched_gpu_fft(GPU{CuDevice}(CUDA.device()), field_size, T, batch_size; real_input=real_input)

# Internal helper to create batched FFT plan
function _create_batched_fft_plan(field_size::Tuple, T::Type, batch_size::Int, real_input::Bool)
    complex_T = T <: Complex ? T : Complex{T}

    # Create batched shape (field dims..., batch)
    batched_size = (field_size..., batch_size)

    if real_input
        # Real-to-complex batched FFT
        dummy_in = CUDA.zeros(T, batched_size...)
        # FFT over all dimensions except the last (batch dimension)
        fft_dims = ntuple(i -> i, length(field_size))
        plan = CUFFT.plan_rfft(dummy_in, fft_dims)

        # Inverse plan
        out_size = (div(field_size[1], 2) + 1, field_size[2:end]..., batch_size)
        dummy_out = CUDA.zeros(complex_T, out_size...)
        iplan = CUFFT.plan_irfft(dummy_out, field_size[1], fft_dims)

        return BatchedGPUFFTPlan(plan, iplan, field_size, batch_size, true)
    else
        # Complex-to-complex batched FFT
        dummy = CUDA.zeros(complex_T, batched_size...)
        fft_dims = ntuple(i -> i, length(field_size))
        plan = CUFFT.plan_fft(dummy, fft_dims)
        iplan = CUFFT.plan_ifft(dummy, fft_dims)

        return BatchedGPUFFTPlan(plan, iplan, field_size, batch_size, false)
    end
end

# Batched FFT plan cache (Thread-Safe)
"""
Thread-safe cache for batched GPU FFT plans.
Uses a ReentrantLock to protect concurrent access from multiple Julia threads.
"""
struct BatchedFFTCache
    plans::Dict{Tuple, BatchedGPUFFTPlan}
    lock::ReentrantLock
end

const BATCHED_FFT_CACHE = BatchedFFTCache(Dict{Tuple, BatchedGPUFFTPlan}(), ReentrantLock())

"""
    _batched_plan_key(arch, field_size, T, batch_size, real_input)

Generate a cache key for batched FFT plans.
Includes device ID for multi-GPU support.
"""
_batched_plan_key(arch::GPU{CuDevice}, field_size::Tuple, T::Type, batch_size::Int, real_input::Bool) =
    (CUDA.deviceid(arch.device), field_size, T, batch_size, real_input)

_batched_plan_key(arch::GPU, field_size::Tuple, T::Type, batch_size::Int, real_input::Bool) =
    (CUDA.deviceid(), field_size, T, batch_size, real_input)

"""
    get_batched_fft_plan(arch::GPU, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false)

Get or create a cached batched FFT plan (thread-safe).

**Important:** `field_size` should be the LOCAL field shape (what this process owns),
not the global domain size. Plans are cached per (device, size, type, batch_size, real_input).
"""
function get_batched_fft_plan(arch::GPU, field_size::Tuple, T::Type, batch_size::Int; real_input::Bool=false)
    key = _batched_plan_key(arch, field_size, T, batch_size, real_input)
    lock(BATCHED_FFT_CACHE.lock) do
        if !haskey(BATCHED_FFT_CACHE.plans, key)
            BATCHED_FFT_CACHE.plans[key] = plan_batched_gpu_fft(arch, field_size, T, batch_size; real_input=real_input)
        end
        return BATCHED_FFT_CACHE.plans[key]
    end
end

"""
    batched_fft!(outputs::Vector{<:CuArray}, inputs::Vector{<:CuArray}, plan::BatchedGPUFFTPlan)

Execute batched FFT on multiple fields simultaneously.
"""
function batched_fft!(outputs::Vector{<:CuArray}, inputs::Vector{<:CuArray}, plan::BatchedGPUFFTPlan)
    @assert length(inputs) == plan.batch_size "Input count must match batch size"
    @assert length(outputs) == plan.batch_size "Output count must match batch size"

    # Stack inputs into batched array
    batched_in = cat(inputs...; dims=ndims(inputs[1])+1)

    # Execute FFT
    batched_out = plan.plan * batched_in

    # Split results back to individual arrays
    for i in 1:plan.batch_size
        selectdim(outputs[i], ndims(outputs[i]), 1:size(outputs[i], ndims(outputs[i]))) .=
            selectdim(batched_out, ndims(batched_out), i)
    end

    return outputs
end

"""
    batched_ifft!(outputs::Vector{<:CuArray}, inputs::Vector{<:CuArray}, plan::BatchedGPUFFTPlan)

Execute batched inverse FFT on multiple fields simultaneously.
"""
function batched_ifft!(outputs::Vector{<:CuArray}, inputs::Vector{<:CuArray}, plan::BatchedGPUFFTPlan)
    @assert length(inputs) == plan.batch_size "Input count must match batch size"
    @assert length(outputs) == plan.batch_size "Output count must match batch size"

    # Stack inputs into batched array
    batched_in = cat(inputs...; dims=ndims(inputs[1])+1)

    # Execute inverse FFT
    batched_out = plan.iplan * batched_in

    # Split results back to individual arrays
    for i in 1:plan.batch_size
        selectdim(outputs[i], ndims(outputs[i]), 1:size(outputs[i], ndims(outputs[i]))) .=
            selectdim(batched_out, ndims(batched_out), i)
    end

    return outputs
end

"""
    clear_batched_fft_cache!()

Clear all cached batched FFT plans (thread-safe).
"""
function clear_batched_fft_cache!()
    lock(BATCHED_FFT_CACHE.lock) do
        empty!(BATCHED_FFT_CACHE.plans)
    end
end

# ============================================================================
# Stream-aware FFT Execution
# ============================================================================

"""
    gpu_fft_async!(output::CuArray, input::CuArray, plan; stream=nothing, synchronize=false)

Execute FFT asynchronously on specified stream.
The FFT is truly asynchronous - call CUDA.synchronize(stream) to wait for completion.

Uses CUDA.stream! context manager which properly integrates with CUDA.jl's
internal stream management (update_stream) so the plan correctly tracks
which stream it's executing on.
"""
function gpu_fft_async!(output::CuArray, input::CuArray, plan::GPUFFTPlan; stream=nothing, synchronize::Bool=false)
    # Derive device from input array to ensure stream matches plan/data device
    input_device = CUDA.device(input)
    device_id = CUDA.deviceid(input_device)
    s = stream !== nothing ? stream : get_compute_stream(; device_id=device_id)

    if s !== nothing
        # Ensure we're on the correct device for the FFT plan
        prev_device = CUDA.device()
        CUDA.device!(input_device)
        # Use stream context - this properly sets the task-local stream
        # so CUFFT's update_stream() will detect it and set the plan's stream
        CUDA.stream!(s) do
            mul!(output, plan.plan, input)
        end
        if synchronize
            CUDA.synchronize(s)
        end
        CUDA.device!(prev_device)
    else
        # No stream specified, use default stream
        mul!(output, plan.plan, input)
    end

    return output
end

"""
    gpu_ifft_async!(output::CuArray, input::CuArray, plan; stream=nothing, synchronize=false)

Execute inverse FFT asynchronously on specified stream.
The FFT is truly asynchronous - call CUDA.synchronize(stream) to wait for completion.

Uses CUDA.stream! context manager which properly integrates with CUDA.jl's
internal stream management (update_stream) so the plan correctly tracks
which stream it's executing on.
"""
function gpu_ifft_async!(output::CuArray, input::CuArray, plan::GPUFFTPlan; stream=nothing, synchronize::Bool=false)
    # Derive device from input array to ensure stream matches plan/data device
    input_device = CUDA.device(input)
    device_id = CUDA.deviceid(input_device)
    s = stream !== nothing ? stream : get_compute_stream(; device_id=device_id)

    if s !== nothing
        # Ensure we're on the correct device for the FFT plan
        prev_device = CUDA.device()
        CUDA.device!(input_device)
        # Use stream context - this properly sets the task-local stream
        # so CUFFT's update_stream() will detect it and set the plan's stream
        CUDA.stream!(s) do
            mul!(output, plan.iplan, input)
        end
        if synchronize
            CUDA.synchronize(s)
        end
        CUDA.device!(prev_device)
    else
        # No stream specified, use default stream
        mul!(output, plan.iplan, input)
    end

    return output
end

