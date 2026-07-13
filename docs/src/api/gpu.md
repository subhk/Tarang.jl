# GPU API Reference

This page documents the GPU-related functions and types that Tarang.jl exports.

GPU support ships as a **package extension** (`TarangCUDAExt`), which Julia loads
automatically once CUDA.jl is available. You therefore need both packages in scope:

```julia
using CUDA     # loads the extension
using Tarang
```

Without CUDA loaded, `GPU()` raises an actionable error
(`"GPU architecture requires CUDA.jl to be loaded"`), so everything runs on the CPU.

The API below is *architecture-generic*: the same functions take `CPU()` or `GPU()`, and you
select the device once, on the `Distributor`:

```julia
dist = Distributor(coords; dtype=Float64, device=CPU())   # or device=GPU()
```

Fields, transforms, and timesteppers all inherit the architecture from the distributor. The
examples on this page are written with `CPU()` so that they run anywhere; substituting
`device=GPU()` is the only change needed to run them on a GPU.

## Architecture Types

```@docs
GPU
CPU
AbstractArchitecture
```

## Array Operations

### Data Movement

```julia
a = zeros(CPU(), Float64, 4, 4)

on_architecture(CPU(), a)   # move/keep an array on the CPU (a no-op for a plain Array)

is_gpu_array(a)             # false
architecture(a)             # CPU()  — architecture inferred from the array type
is_gpu(CPU())               # false
has_cuda()                  # false when CUDA.jl is not loaded
```

With the extension loaded, `on_architecture(GPU(), a)` returns a `CuArray` and
`on_architecture(CPU(), cu_a)` copies it back to a host `Array`. `on_architecture` also
accepts tuples, named tuples, numbers, and `nothing`, so it can be mapped over heterogeneous
state without special-casing.

### Allocation

`Base.zeros`, `Base.ones`, and `Base.similar` take an architecture as their first argument;
with `GPU()` they allocate `CuArray`s instead of `Array`s.

```julia
zeros(CPU(), Float64, 128, 128)     # Matrix{Float64}
ones(CPU(), Float32, 256)           # Vector{Float32}
similar(CPU(), a)                   # same shape/eltype, same architecture
create_array(CPU(), ComplexF64, 2, 2)
```

`allocate_like(a, dims...)`, `similar_zeros(a)`, and `copy_to_device(a, target)` allocate
from an *existing* array instead of an architecture, which keeps the device implicit.

### Device Context and Cleanup

| Function | Description |
|----------|-------------|
| `ensure_device!(arch)` | Activate the correct CUDA device. No-op on `CPU()`. Called by `launch!` before every kernel. |
| `synchronize(arch)` | Wait for outstanding device work. No-op on `CPU()`. |
| `unsafe_free!(arch, a)` | Eagerly release a device buffer. No-op on `CPU()`. |
| `device(arch)` | The KernelAbstractions backend (`KernelAbstractions.CPU(false)` for `CPU()`). |
| `array_type(arch)` | `Array` for `CPU()`, `CuArray` for `GPU()`. |

For multi-GPU runs (one GPU per MPI rank) build the architecture with an explicit device id —
`GPU(device_id = MPI.Comm_rank(comm) % CUDA.ndevices())` — and let `ensure_device!` handle the
context switching; `launch!` calls it on every launch.

## Transform Functions

### FFT Backend Control

Transforms choose between CPU FFTW and GPU CUFFT per field. The choice is a per-field mode
plus a global size threshold.

| Function | Description |
|----------|-------------|
| `gpu_fft_mode(field)` | Current mode: `:auto` (default), `:cpu`, or `:gpu` |
| `set_gpu_fft_mode!(field, mode)` | Set the mode. Any other symbol throws `ArgumentError` |
| `set_gpu_fft_min_elements!(n)` | Element count below which `:auto` stays on the CPU (default `32_768`) |
| `gpu_fft_min_elements()` | Read that threshold |
| `should_use_gpu_fft(field)` | Whether the GPU FFT path would be taken for this field |

```julia
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
bx = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi))
by = RealFourier(coords["y"]; size=16, bounds=(0.0, 2pi))
s  = ScalarField(Domain(dist, (bx, by)), "s")

gpu_fft_mode(s)                    # :auto
should_use_gpu_fft(s)              # false — 256 elements < the 32_768 threshold

set_gpu_fft_mode!(s, :gpu)         # force GPU regardless of size
should_use_gpu_fft(s)              # true

set_gpu_fft_mode!(s, :auto)
set_gpu_fft_min_elements!(1)       # lower the threshold instead
should_use_gpu_fft(s)              # true
```

Note that `should_use_gpu_fft` reports the *preference*: on a `CPU()` distributor the GPU path
is never actually taken, whatever the mode says.

### Transform Execution

| Function | Description |
|----------|-------------|
| `forward_transform!(field)` | Grid → coefficient transform; dispatches to GPU when the field is on a GPU |
| `backward_transform!(field)` | Coefficient → grid transform |

```julia
forward_transform!(s)      # s.current_layout == :c
backward_transform!(s)     # s.current_layout == :g, round-trip error ~3e-16
```

`Tarang.gpu_forward_transform!` / `Tarang.gpu_backward_transform!` are the internal dispatch
hooks these call. They are not exported, and they are not a way to "force" a GPU transform:
each returns a `Bool` saying whether it *did* run on the GPU (always `false` on a CPU
architecture). Use `set_gpu_fft_mode!` to steer the backend.

## Writing Kernels: `KernelOperation`

`KernelOperation` wraps a KernelAbstractions `@kernel` together with a default `ndrange`, and
launches it through the architecture abstraction so the same code runs on CPU and GPU.

```julia
using KernelAbstractions: @kernel, @index, @Const

@kernel function scale_kernel!(y, @Const(x), alpha)
    i = @index(Global)
    @inbounds y[i] = alpha * x[i]
end

# ndrange from a do-block over the call arguments
my_op = KernelOperation(scale_kernel!) do y, x, alpha
    length(y)
end

x = ones(CPU(), Float64, 8)
y = zeros(CPU(), Float64, 8)
my_op(CPU(), y, x, 2.5)            # y == fill(2.5, 8)
```

Import KernelAbstractions **selectively**, as above. A bare `using KernelAbstractions`
alongside `using Tarang` makes `CPU` ambiguous — both packages export that name — and every
later `CPU()` fails with ``UndefVarError: `CPU` not defined in `Main` ``.

Without a do-block the default `ndrange` is `length(args[1])`, and `launch!` is the underlying
one-shot form:

```julia
op = KernelOperation(scale_kernel!)   # ndrange = length(y)
op(CPU(), y, x, 3.0)

launch!(CPU(), scale_kernel!, y, x, 4.0; ndrange=length(y))
workgroup_size(CPU(), 8)              # 8
```

`launch!` accepts an architecture *or* an array (whose architecture it infers), and calls
`ensure_device!` before launching.

## Distributed Transforms: `TransposableField`

`TransposableField` wraps a `ScalarField` with the buffers and sub-communicators needed for
pencil transposes, and drives the transpose-FFT-transpose sequence. It is the *custom*
transpose path, written for **multi-GPU** runs: `create_distributed_gpu_transform` builds one
internally, and its MPI validation errors name the GPU+MPI configuration.

It is **not** the CPU+MPI path. On a `CPU()` distributor with more than one rank, fields are
allocated by PencilArrays as z-decomposed slabs, and `TransposableField` — which expects its
own ZLocal decomposition (y decomposed, z local) — rejects them:

> `TransposableField layout mismatch: field storage shape (8, 8, 4) does not match expected
> ZLocal shape (8, 4, 8) for topology (Rx=1, Ry=2)`

For CPU + MPI, just call `forward_transform!` / `backward_transform!`; PencilFFTs does the
transposes for you. `TransposableField` on a CPU distributor is useful at `nprocs == 1`, which
is what the example below runs.

Two constraints come from the implementation:

* the field's bases must all be `ComplexFourier` under MPI — `RealFourier`'s half-spectrum
  layout is incompatible with the custom transposes. This one *is* enforced: the constructor
  errors at `nprocs > 1`.
* the distributor's `dtype` must be complex, since the transpose buffers are complex. This one
  is **not** checked. With a real `dtype` the `TransposableField` constructs (the buffers are
  promoted to `Complex{dtype}`), and the failure only surfaces inside the transform, as
  `MethodError: no method matching mul!(::Array{ComplexF64, 3}, ::FFTW.cFFTWPlan{…}, ::Array{Float64, 3}, …)`.
  Pass `dtype=ComplexF64`.

```julia
coords = CartesianCoordinates("x", "y", "z")
dist   = Distributor(coords; dtype=ComplexF64, device=CPU())
bx = ComplexFourier(coords["x"]; size=8, bounds=(0.0, 2pi))
by = ComplexFourier(coords["y"]; size=8, bounds=(0.0, 2pi))
bz = ComplexFourier(coords["z"]; size=8, bounds=(0.0, 2pi))
f  = ScalarField(Domain(dist, (bx, by, bz)), "f")

xg, yg, zg = local_grids(dist, bx, by, bz)
ensure_layout!(f, :g)
get_grid_data(f) .= sin.(xg) .* cos.(yg') .* ones(1, 1, 8)

tf = TransposableField(f)          # or make_transposable(f)

active_layout(tf)                  # ZLocal  (also XLocal, YLocal)
current_data(tf)                   # the buffer for the active layout
local_shape(tf, XLocal)            # (8, 8, 8)

distributed_forward_transform!(tf)   # writes the result back into f["c"]
distributed_backward_transform!(tf)  # round-trip error ~2e-16
```

Both transforms accept `overlap=true` to use async transposes that overlap communication with
computation (not supported for a 2D domain on a true 2D mesh, which warns and falls back).

### CUDA-Aware MPI

```julia
check_cuda_aware_mpi()      # false unless the MPI build supports device pointers
```

When it returns `true`, distributed transforms pass `CuArray`s straight to MPI; otherwise the
buffers are staged through the host.

## Memory Management

Tarang uses CUDA.jl's built-in memory pool — there is no custom pooling layer. Use CUDA.jl's
own tools for inspection and reclamation (`CUDA.memory_status()`, `CUDA.reclaim()`), and
`unsafe_free!(arch, a)` when you want to drop a large buffer without waiting for the GC.

## The CUDA Extension is Not Public API

The GPU kernels, FFT/DCT plans, and transpose helpers live in the extension module
`TarangCUDAExt` (`ext/cuda/*.jl`). A Julia package extension **cannot** add names to its
parent module, so these are *not* reachable as `Tarang.x` and are not in scope after
`using Tarang, CUDA` — `isdefined(Tarang, :gpu_add!)` is `false`. They are implementation
details that Tarang's own transform and timestepping code calls; they are reachable only via
`Base.get_extension(Tarang, :TarangCUDAExt)` and may change without notice.

What the extension implements, for orientation when reading the source:

| Area | Source | Contents |
|------|--------|----------|
| Element-wise kernels | `ext/cuda/kernels.jl` | `gpu_add!`, `gpu_sub!`, `gpu_mul!`, `gpu_scale!`, `gpu_axpy!`, `gpu_linear_combination!` and their `GPU_*_OP` `KernelOperation` wrappers |
| Fused kernels | `ext/cuda/kernels.jl` | `gpu_rk_stage!`, `gpu_axpby!`, `gpu_fma!`, `gpu_dealias_multiply!`, `gpu_triple_product!`, `gpu_conj_multiply!`, `gpu_squared_magnitude!`, `gpu_kinetic_energy_2d!/3d!`, `gpu_grad_mag_sq_2d!`, `gpu_viscous_damping!` |
| FFT plans | `ext/cuda/transforms.jl`, `ext/cuda/batched_fft.jl` | `plan_gpu_fft`, `gpu_forward_fft!`, `gpu_backward_fft!`, batched plans, plan caches |
| Chebyshev DCT | `ext/cuda/dct.jl`, `ext/cuda/dct_distributed.jl` | DCT plans, multi-GPU distributed DCT |
| Mixed transforms | `ext/cuda/mixed_transforms.jl` | Fourier-Chebyshev plans |
| Transposes | `ext/cuda/transpose_kernels.jl`, `ext/cuda/nccl_transpose.jl` | pack/unpack kernels, NCCL all-to-all |
| Config / memory | `ext/cuda/config.jl`, `ext/cuda/memory.jl` | tensor-core toggles, async host↔device copies |

To write your own device code, use the public `KernelOperation` / `launch!` path above rather
than reaching into the extension.
