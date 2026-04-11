# GPU Implementation Bugfixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 13 confirmed bugs in Tarang.jl's GPU implementation — 5 critical (crash/wrong results), 8 important (concurrency/correctness under specific conditions).

**Architecture:** All fixes are isolated to `ext/cuda/` (CUDA extension) and `src/tools/gpu_matsolvers.jl` (GPU solvers), plus one fix in `src/core/architectures.jl`. Each task targets one bug or a closely related group. No new files created.

**Tech Stack:** Julia, CUDA.jl, KernelAbstractions.jl, NCCL.jl, MPI.jl

---

### Task 1: Fix `KernelOperation` do-block constructor mismatch

**Files:**
- Modify: `src/core/architectures.jl:532`

The do-block syntax `KernelOperation(kernel) do args... end` desugars to `KernelOperation(lambda, kernel)` — two positional args. But the only constructor is `KernelOperation(kernel; ndrange_fn=...)` — one positional + one keyword. This causes `MethodError` when the CUDA extension loads.

- [ ] **Step 1: Add two-positional-arg constructor**

In `src/core/architectures.jl`, after line 532, add:

```julia
KernelOperation(ndrange_fn::F, kernel::K) where {K,F} = KernelOperation{K, typeof(ndrange_fn)}(kernel, ndrange_fn)
```

This makes the do-block syntax work: `KernelOperation(add_kernel!) do c, a, b; length(c); end` desugars to `KernelOperation((c,a,b)->length(c), add_kernel!)`, which now matches.

- [ ] **Step 2: Verify the extension file parses**

Run:
```bash
cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'using Tarang; println("KernelOperation constructors: ", methods(Tarang.KernelOperation))'
```

Expected: No `MethodError`, prints both constructor methods.

---

### Task 2: Replace all `CUDA.deviceid()` calls with valid CUDA.jl API

**Files:**
- Modify: `ext/cuda/config.jl:61,77,84,99,106`
- Modify: `ext/cuda/dct_distributed.jl:55,92,107`
- Modify: `ext/cuda/memory.jl:38,43`
- Modify: `ext/cuda/transforms.jl:872`
- Modify: `ext/cuda/batched_fft.jl:103`
- Modify: `ext/cuda/mixed_transforms.jl:135`
- Modify: `ext/cuda/utils.jl:16`

`CUDA.deviceid()` is not a public CUDA.jl API function. The correct way to get an integer device ID is `CUDA.device().handle` (returns an `Int32`).

- [ ] **Step 1: Add a helper function in config.jl**

At the top of `ext/cuda/config.jl` (after line 4), add a local helper:

```julia
# CUDA.jl does not export deviceid() — use device().handle for the integer ordinal
_current_device_id() = Int(CUDA.device().handle)
```

- [ ] **Step 2: Replace all occurrences in config.jl**

Replace every `CUDA.deviceid()` in `ext/cuda/config.jl` with `_current_device_id()`:

- Line 61: `device_id = _current_device_id()`
- Line 77 (docstring): `get_compute_stream(; device_id::Int=_current_device_id())`
- Line 84: `function get_compute_stream(; device_id::Int=_current_device_id())`
- Line 99 (docstring): `get_transfer_stream(; device_id::Int=_current_device_id())`
- Line 106: `function get_transfer_stream(; device_id::Int=_current_device_id())`

- [ ] **Step 3: Replace all occurrences in other ext/cuda/ files**

Each file uses `CUDA.deviceid()` in default kwarg expressions or local variables. Replace with `_current_device_id()`. Since `_current_device_id` is defined in `config.jl` which is included first by `TarangCUDAExt.jl`, it will be available in all subsequent includes.

- `ext/cuda/memory.jl:38,43` — both in `pool_allocate` signature
- `ext/cuda/utils.jl:16` — in `get_fft_1d_plan` signature
- `ext/cuda/transforms.jl:872` — local variable assignment
- `ext/cuda/batched_fft.jl:103` — cache key tuple element
- `ext/cuda/mixed_transforms.jl:135` — cache key tuple element
- `ext/cuda/dct_distributed.jl:55` — `GPU(device_id = _current_device_id())`
- `ext/cuda/dct_distributed.jl:92,107` — local variable assignments

- [ ] **Step 4: Verify no remaining occurrences**

Run:
```bash
cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && grep -rn "CUDA\.deviceid()" ext/ src/
```

Expected: No output (zero matches).

---

### Task 3: Fix GMRES Hessenberg matrix — allocate on CPU

**Files:**
- Modify: `src/tools/gpu_matsolvers.jl:602`

The Hessenberg matrix `H` is allocated as a GPU array but then scalar-indexed in the Arnoldi loop (lines 615-624). With `allowscalar(false)`, this hard-fails.

- [ ] **Step 1: Change `_gpu_zeros` to `zeros` for the Hessenberg matrix**

In `src/tools/gpu_matsolvers.jl`, line 602, change:

```julia
H = _gpu_zeros(T, m + 1, m)  # Hessenberg matrix
```

to:

```julia
H = zeros(T, m + 1, m)  # Hessenberg matrix — CPU to avoid scalar indexing in Arnoldi
```

- [ ] **Step 2: Remove the now-redundant `Array(H)` conversion**

Line 632 does `H_cpu = Array(H)` which is now a no-op (H is already a CPU array). Change to:

```julia
H_cpu = H  # Already on CPU
```

- [ ] **Step 3: Verify the file parses**

Run:
```bash
cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && julia --project -e 'include("src/tools/gpu_matsolvers.jl"); println("OK")'
```

Expected: Prints "OK" with no errors.

---

### Task 4: Fix batched FFT R2C output shape — reduce last dimension, not first

**Files:**
- Modify: `ext/cuda/batched_fft.jl:65-67`

`plan_rfft` over all dims reduces the **last** transformed dimension (FFTW convention), but the code halves dimension 1 in `out_size`. This produces a shape mismatch.

- [ ] **Step 1: Fix out_size and irfft length argument**

In `ext/cuda/batched_fft.jl`, replace lines 65-67:

```julia
        out_size = (div(field_size[1], 2) + 1, field_size[2:end]..., batch_size)
        dummy_out = CUDA.zeros(complex_T, out_size...)
        iplan = CUFFT.plan_irfft(dummy_out, field_size[1], fft_dims)
```

with:

```julia
        last_field_dim = length(field_size)
        out_size = (field_size[1:end-1]..., div(field_size[end], 2) + 1, batch_size)
        dummy_out = CUDA.zeros(complex_T, out_size...)
        iplan = CUFFT.plan_irfft(dummy_out, field_size[last_field_dim], fft_dims)
```

---

### Task 5: Fix `compute_transpose_counts!` recv formula for `:y_to_x`

**Files:**
- Modify: `ext/cuda/nccl_transpose.jl:974-975`

The recv formula uses full `Nx` instead of per-rank `chunk_x`, inflating recv counts by `comm_size`x.

- [ ] **Step 1: Fix the recv_counts formula**

In `ext/cuda/nccl_transpose.jl`, replace lines 974-975:

```julia
            buffer.send_counts[i] = chunk_x * local_shape[2] * local_shape[3]
            buffer.recv_counts[i] = Nx * chunk_y * local_shape[3]
```

with:

```julia
            buffer.send_counts[i] = chunk_x * local_shape[2] * local_shape[3]
            buffer.recv_counts[i] = local_shape[1] * chunk_y * local_shape[3]
```

After Y→X transpose, each rank has `local_shape[1]` (= full Nx) in X... wait — in Y-pencil, `local_shape[1]` is `Nx/P1`, not full `Nx`. The recv side is X-pencil where this rank owns full Nx. The correct recv formula: for each peer `i`, we receive `chunk_x_i * Ny_me * Nz_local` elements, where `Ny_me` is this rank's Y-chunk in X-pencil layout.

Actually, looking at the Y→X transpose semantics: we're going from Y-pencil `(Nx/P1, Ny, Nz/P2)` to X-pencil `(Nx, Ny/P1, Nz/P2)`. Each peer `i` sends us their portion of X (= `chunk_x_i`), and our Y is split as `Ny/P1`. So recv from peer `i` = `chunk_x * (Ny/comm_size for this rank) * local_shape[3]`.

The simplest correct fix using the same variable names already computed:

```julia
            my_chunk_y = div(Ny, comm_size) + ((my_rank_in_comm) < mod(Ny, comm_size) ? 1 : 0)
            buffer.recv_counts[i] = chunk_x * my_chunk_y * local_shape[3]
```

But we don't have `my_rank_in_comm` in this convenience function. The safest fix that mirrors the Z→Y case pattern (which uses `local_shape` dimensions consistently):

Replace lines 965-976 with:

```julia
    elseif direction == :y_to_x
        # Y->X: distribute X, gather Y
        Nx = pencil.global_shape[1]
        Ny = pencil.global_shape[2]
        local_shape = pencil.y_pencil_shape
        x_pencil_shape = pencil.x_pencil_shape

        for i in 1:comm_size
            chunk_x = div(Nx, comm_size) + ((i-1) < mod(Nx, comm_size) ? 1 : 0)
            chunk_y = div(Ny, comm_size) + ((i-1) < mod(Ny, comm_size) ? 1 : 0)
            buffer.send_counts[i] = chunk_x * local_shape[2] * local_shape[3]
            buffer.recv_counts[i] = x_pencil_shape[1] * chunk_y * x_pencil_shape[3]
        end
```

This uses the X-pencil shape for recv (which has full Nx and local Nz), and per-peer `chunk_y` for the Y dimension that was distributed.

- [ ] **Step 2: Verify the recv counts are symmetric**

For a balanced grid (N divisible by comm_size), `sum(recv_counts)` should equal `prod(x_pencil_shape)`. Verify mentally: `sum_i(Nx * (Ny/P1) * Nz_local) = Nx * Ny * Nz_local / P1 * P1 = Nx * Ny * Nz_local`. And `prod(x_pencil_shape) = Nx * (Ny/P1) * (Nz/P2)`. These match when `Nz_local = Nz/P2` — correct.

---

### Task 6: Fix `device()` side-effect — move `device!` to `launch!`

**Files:**
- Modify: `ext/cuda/architecture.jl:56-61`
- Modify: `src/core/architectures.jl` (the `launch!` function, around line 480)

`device(::GPU{CuDevice})` calls `CUDA.device!(gpu.device)` as a side effect, which mutates thread-local CUDA context. This is unsafe in concurrent scenarios. The device switch should happen explicitly in `launch!`.

- [ ] **Step 1: Read the launch! function to find exact lines**

Read `src/core/architectures.jl` around the `launch!` definition.

- [ ] **Step 2: Remove device! from device()**

In `ext/cuda/architecture.jl`, change lines 56-61 from:

```julia
function Tarang.device(gpu::GPU{CuDevice})
    CUDA.device!(gpu.device)
    return CUDABackend()
end
```

to:

```julia
function Tarang.device(gpu::GPU{CuDevice})
    return CUDABackend()
end
```

- [ ] **Step 3: Add ensure_device! call in launch!**

In `src/core/architectures.jl`, at the top of the `launch!` function body (before `backend = device(arch)`), add:

```julia
ensure_device!(arch)
```

Where `ensure_device!` is a no-op for CPU and calls `CUDA.device!(gpu.device)` for GPU (already defined in `ext/cuda/architecture.jl`). Check if `ensure_device!` already exists; if not, define it.

---

### Task 7: Add lock to `PinnedBufferPool`

**Files:**
- Modify: `ext/cuda/memory.jl:199-215,228-246,258+`

`GPUMemoryPool` has a `ReentrantLock` but `PinnedBufferPool` does not. Concurrent `get_pinned_buffer`/`release_pinned_buffer!` will corrupt internal dicts.

- [ ] **Step 1: Add lock field to PinnedBufferPool**

In `ext/cuda/memory.jl`, add a `lock` field to the struct (line 199-215):

```julia
mutable struct PinnedBufferPool
    available::Dict{Tuple{DataType, Tuple}, Vector{Vector{UInt8}}}
    max_buffers_per_key::Int
    total_pooled_bytes::Int
    max_total_bytes::Int
    in_use::Dict{UInt, Tuple{Vector{UInt8}, Int}}
    lock::ReentrantLock  # NEW

    function PinnedBufferPool(; max_buffers::Int=5, max_total_mb::Int=1024)
        new(Dict{Tuple{DataType, Tuple}, Vector{Vector{UInt8}}}(),
            max_buffers, 0, max_total_mb * 1_000_000,
            Dict{UInt, Tuple{Vector{UInt8}, Int}}(),
            ReentrantLock())
    end
end
```

- [ ] **Step 2: Wrap `get_pinned_buffer` body in lock**

Wrap the body of `get_pinned_buffer` (lines 228-247) with `lock(PINNED_BUFFER_POOL.lock) do ... end`.

- [ ] **Step 3: Wrap `release_pinned_buffer!` body in lock**

Wrap the body of `release_pinned_buffer!` (starting at line 258) with `lock(PINNED_BUFFER_POOL.lock) do ... end`.

---

### Task 8: Add lock to `clear_memory_pool!`

**Files:**
- Modify: `ext/cuda/memory.jl:159-167`

All other `GPUMemoryPool` operations acquire `GPU_MEMORY_POOL.lock`, but `clear_memory_pool!` does not.

- [ ] **Step 1: Wrap clear_memory_pool! body in lock**

Change lines 159-167 from:

```julia
function clear_memory_pool!()
    for (key, arrays) in GPU_MEMORY_POOL.pools
        empty!(arrays)
    end
    empty!(GPU_MEMORY_POOL.pools)
    GPU_MEMORY_POOL.total_pooled_bytes = 0
    CUDA.reclaim()
    @info "GPU memory pool cleared"
end
```

to:

```julia
function clear_memory_pool!()
    lock(GPU_MEMORY_POOL.lock) do
        for (key, arrays) in GPU_MEMORY_POOL.pools
            empty!(arrays)
        end
        empty!(GPU_MEMORY_POOL.pools)
        GPU_MEMORY_POOL.total_pooled_bytes = 0
    end
    CUDA.reclaim()
    @info "GPU memory pool cleared"
end
```

Note: `CUDA.reclaim()` is outside the lock — it's a CUDA runtime call that shouldn't hold our lock.

---

### Task 9: Fix `DISTRIBUTED_DCT_PLAN_CACHE` key type — use Tuple instead of hash

**Files:**
- Modify: `ext/cuda/transforms.jl:49,66`

Using `UInt64` hash as dict key risks silent collisions. Every other cache in the codebase uses the raw tuple.

- [ ] **Step 1: Change cache type and key computation**

In `ext/cuda/transforms.jl`, line 49, change:

```julia
const DISTRIBUTED_DCT_PLAN_CACHE = Dict{UInt64, DistributedDCTPlan}()
```

to:

```julia
const DISTRIBUTED_DCT_PLAN_CACHE = Dict{Tuple, DistributedDCTPlan}()
```

And line 66, change:

```julia
    key = hash((global_shape_tuple, proc_grid, T))
```

to:

```julia
    key = (global_shape_tuple, proc_grid, T)
```

---

### Task 10: Fix NCCL self-copy guard condition

**Files:**
- Modify: `ext/cuda/nccl_transpose.jl:506`

The guard `send_counts[self] > 0 && recv_counts[self] > 0` skips the self-copy when recv count is zero but send count is non-zero, silently dropping data.

- [ ] **Step 1: Remove recv_counts check from self-copy guard**

In `ext/cuda/nccl_transpose.jl`, line 506, change:

```julia
    if my_rank >= 0 && send_counts[my_rank+1] > 0 && recv_counts[my_rank+1] > 0
```

to:

```julia
    if my_rank >= 0 && send_counts[my_rank+1] > 0
```

The recv side is sized by send (self-send count == self-recv count for correctly constructed transposes), so checking only send is sufficient and safe.

---

### Task 11: Remove dangerous `my_rank=-1` default in `nccl_alltoall!`

**Files:**
- Modify: `ext/cuda/nccl_transpose.jl:478`

If a caller omits `my_rank`, the self-copy is skipped and NCCL tries self-send, which hangs.

- [ ] **Step 1: Make my_rank a required keyword argument**

In `ext/cuda/nccl_transpose.jl`, line 478, change:

```julia
                         nccl_comm; my_rank::Int=-1)
```

to:

```julia
                         nccl_comm; my_rank::Int)
```

- [ ] **Step 2: Check all call sites pass my_rank**

Run:
```bash
cd /Users/subhajitkar/Documents/GitHub/Tarang.jl && grep -rn "nccl_alltoall!" ext/ src/
```

Verify every call site provides `my_rank=...`. If any omit it, add the parameter.

---

### Task 12: Fix dealiasing kernels for negative frequencies

**Files:**
- Modify: `ext/cuda/utils.jl:220-241`

The kernels only check `i > cutoff_x` which misses high-|k| negative frequency modes in the upper half of complex FFT arrays.

- [ ] **Step 1: Fix 2D dealiasing kernel**

In `ext/cuda/utils.jl`, replace lines 220-227:

```julia
@kernel function dealiasing_2d_kernel!(data, cutoff_x::Int, cutoff_y::Int, nx::Int, ny::Int)
    idx = @index(Global)
    j = ((idx - 1) ÷ nx) + 1
    i = ((idx - 1) % nx) + 1
    @inbounds if i > cutoff_x || j > cutoff_y
        data[i, j] = zero(eltype(data))
    end
end
```

with:

```julia
@kernel function dealiasing_2d_kernel!(data, cutoff_x::Int, cutoff_y::Int, nx::Int, ny::Int)
    idx = @index(Global)
    j = ((idx - 1) ÷ nx) + 1
    i = ((idx - 1) % nx) + 1
    # Zero modes where |k| exceeds cutoff in any dimension
    # For complex FFT: indices 1..N map to wavenumbers 0..N/2, -(N/2-1)..-1
    # Positive freqs beyond cutoff: i > cutoff_x
    # Negative freqs beyond cutoff: i > nx - cutoff_x + 1 (when cutoff_x < nx)
    dealias_x = i > cutoff_x && i <= nx - cutoff_x + 1 ? false : (i > cutoff_x)
    dealias_neg_x = cutoff_x < nx && i > 1 && i > nx - cutoff_x + 1
    dealias_y = j > cutoff_y && j <= ny - cutoff_y + 1 ? false : (j > cutoff_y)
    dealias_neg_y = cutoff_y < ny && j > 1 && j > ny - cutoff_y + 1
    @inbounds if dealias_x || dealias_neg_x || dealias_y || dealias_neg_y
        data[i, j] = zero(eltype(data))
    end
end
```

Simplification — the condition reduces to: zero if `i > cutoff_x || (i > 1 && i > nx - cutoff_x + 1)`, and similarly for j:

```julia
@kernel function dealiasing_2d_kernel!(data, cutoff_x::Int, cutoff_y::Int, nx::Int, ny::Int)
    idx = @index(Global)
    j = ((idx - 1) ÷ nx) + 1
    i = ((idx - 1) % nx) + 1
    # Zero modes beyond cutoff in positive OR negative frequency range
    kill_x = i > cutoff_x || (i > 1 && nx - i + 1 > cutoff_x)
    kill_y = j > cutoff_y || (j > 1 && ny - j + 1 > cutoff_y)
    @inbounds if kill_x || kill_y
        data[i, j] = zero(eltype(data))
    end
end
```

- [ ] **Step 2: Fix 3D dealiasing kernel**

Apply the same pattern to the 3D kernel at lines 232-241:

```julia
@kernel function dealiasing_3d_kernel!(data, cutoff_x::Int, cutoff_y::Int, cutoff_z::Int,
                                        nx::Int, ny::Int, nz::Int)
    idx = @index(Global)
    i = ((idx - 1) % nx) + 1
    j = (((idx - 1) ÷ nx) % ny) + 1
    k = ((idx - 1) ÷ (nx * ny)) + 1
    kill_x = i > cutoff_x || (i > 1 && nx - i + 1 > cutoff_x)
    kill_y = j > cutoff_y || (j > 1 && ny - j + 1 > cutoff_y)
    kill_z = k > cutoff_z || (k > 1 && nz - k + 1 > cutoff_z)
    @inbounds if kill_x || kill_y || kill_z
        data[i, j, k] = zero(eltype(data))
    end
end
```

---

### Task 13: Fix `GPUDCTPlan` per-call buffer allocation

**Files:**
- Modify: `ext/cuda/dct.jl:698-730,738-754`

`gpu_forward_dct_1d!` and `gpu_backward_dct_1d!` allocate `CUDA.zeros(Complex{T}, 2*n)` on every call. In a time-stepping loop this causes severe allocation pressure.

- [ ] **Step 1: Use plan.work_complex as work_in and add work_out to GPUDCTPlan**

First, check the `GPUDCTPlan` struct definition to see what fields exist. The plan already has `work_complex` (allocated at creation). Add a second work buffer `work_complex_out`:

Find the `GPUDCTPlan` struct and add:
```julia
work_complex_out::CuVector{Complex{T}}  # Output buffer for out-of-place FFT
```

And in the constructor, allocate it alongside `work_complex`:
```julia
work_complex_out = CuVector{Complex{T}}(undef, 2 * n)
```

- [ ] **Step 2: Replace local allocations in gpu_forward_dct_1d!**

In `ext/cuda/dct.jl`, lines 711-712, change:

```julia
    work_in = CUDA.zeros(Complex{T}, 2 * n)
    work_out = CUDA.zeros(Complex{T}, 2 * n)
```

to:

```julia
    work_in = plan.work_complex
    work_out = plan.work_complex_out
    fill!(work_in, zero(Complex{T}))
```

- [ ] **Step 3: Replace local allocations in gpu_backward_dct_1d!**

In `ext/cuda/dct.jl`, lines 752-753, apply the same change:

```julia
    work_in = plan.work_complex
    work_out = plan.work_complex_out
    fill!(work_in, zero(Complex{T}))
```

Note: These functions are NOT thread-safe after this change — concurrent calls on the same plan will race on the shared buffers. This matches the `OptimizedGPUDCTPlan` pattern which uses a lock. If concurrent use is needed, callers should use `OptimizedGPUDCTPlan` instead, or a lock should be added around these calls.

---

### Task 14: Add GPUConfig lock for thread safety

**Files:**
- Modify: `ext/cuda/config.jl:11-34,84-96,106-118,126-144`

`GPU_CONFIG` dicts are accessed without locking. Add a `ReentrantLock` to `GPUConfig`.

- [ ] **Step 1: Add lock field to GPUConfig struct**

In `ext/cuda/config.jl`, add `lock::ReentrantLock` to the struct and initialize in constructor:

```julia
mutable struct GPUConfig
    compute_streams::Dict{Int, CuStream}
    transfer_streams::Dict{Int, CuStream}
    streams_enabled::Bool
    use_memory_pool::Bool
    use_tensor_cores::Bool
    tensor_math_mode::Any
    default_workgroup_1d::Int
    default_workgroup_2d::Tuple{Int, Int}
    default_workgroup_3d::Tuple{Int, Int, Int}
    lock::ReentrantLock  # NEW

    function GPUConfig()
        new(Dict{Int, CuStream}(), Dict{Int, CuStream}(), false,
            true, false, nothing,
            256, (16, 16), (8, 8, 4),
            ReentrantLock())
    end
end
```

- [ ] **Step 2: Add lock to get_compute_stream on-demand creation**

Wrap the on-demand stream creation in `get_compute_stream` (lines 88-94) with `lock(GPU_CONFIG.lock) do ... end`:

```julia
function get_compute_stream(; device_id::Int=_current_device_id())
    if !GPU_CONFIG.streams_enabled
        return nothing
    end
    lock(GPU_CONFIG.lock) do
        if !haskey(GPU_CONFIG.compute_streams, device_id)
            prev_device = CUDA.device()
            CUDA.device!(CuDevice(device_id))
            GPU_CONFIG.compute_streams[device_id] = CuStream()
            CUDA.device!(prev_device)
        end
        return GPU_CONFIG.compute_streams[device_id]
    end
end
```

- [ ] **Step 3: Add lock to get_transfer_stream**

Same pattern for `get_transfer_stream` (lines 106-118).

- [ ] **Step 4: Add lock to sync_streams!**

Wrap the dict iteration in `sync_streams!` (lines 126-144) with `lock(GPU_CONFIG.lock) do ... end`.
