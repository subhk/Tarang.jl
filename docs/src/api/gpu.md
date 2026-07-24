# GPU API Reference

GPU support is provided by the CUDA package extension. Load both packages before
constructing `GPU()`:

```julia
using CUDA
using Tarang
```

The architecture is selected on the `Distributor` and inherited by its fields:

```julia
dist = Distributor(coords; dtype=Float64, device=GPU())
```

## Architectures

```@docs
GPU
CPU
AbstractArchitecture
```

Useful public helpers include:

| Function | Purpose |
|----------|---------|
| `architecture(array)` | Return the array architecture |
| `is_gpu(arch)` | Test an architecture |
| `is_gpu_array(array)` | Test whether storage is device-resident |
| `has_cuda()` | Report whether the CUDA extension is loaded |
| `ensure_device!(arch)` | Activate the architecture device |
| `synchronize(arch)` | Wait for queued work |
| `array_type(arch)` | Return the architecture array type |

## Allocation and explicit transfers

Architecture-aware allocation uses the architecture as the first argument:

```julia
a = zeros(GPU(), Float64, 128, 128)
b = ones(GPU(), Float32, 256)
c = create_array(GPU(), ComplexF64, 16, 16)
```

Use `on_architecture` for an explicit transfer:

```julia
a_host = on_architecture(CPU(), a)
a_device = on_architecture(GPU(), a_host)
```

`allocate_like`, `similar_zeros`, and `copy_to_device` infer the target from an
existing array. Tarang uses CUDA.jl's memory pool; `unsafe_free!` can release a
large device buffer eagerly.

Explicit output, checkpoint, gather, and architecture-conversion APIs may copy
data to the host. GPU computation itself does not silently switch to CPU.

## Transform dispatch

| Function | Purpose |
|----------|---------|
| `forward_transform!(field)` | Grid to coefficient layout |
| `backward_transform!(field)` | Coefficient to grid layout |
| `gpu_fft_mode(field)` | Return `:auto`, `:gpu`, or `:cpu` |
| `set_gpu_fft_mode!(field, mode)` | Set the field transform mode |
| `should_use_gpu_fft(field)` | Report whether the device path is selected |

For a GPU field, `:auto` and `:gpu` both remain on-device for every array size.
`:cpu` is rejected. An unsupported transform raises an error instead of
downloading the field and calling FFTW.

`set_gpu_fft_min_elements!` and `gpu_fft_min_elements` are legacy CPU-field
preference controls; they do not affect GPU fields.

See [GPU Computing](../pages/gpu_computing.md#Transforms) for the supported
Fourier and Chebyshev combinations.

## Portable kernels

`KernelOperation` wraps a KernelAbstractions kernel and launches it through the
selected architecture.

```julia
using KernelAbstractions: @kernel, @index

@kernel function add_kernel!(c, a, b)
    i = @index(Global)
    @inbounds c[i] = a[i] + b[i]
end

add = KernelOperation(add_kernel!) do c, a, b
    length(c)
end

arch = GPU()
a = ones(arch, Float64, 1024)
b = ones(arch, Float64, 1024)
c = similar(a)
add(arch, c, a, b)
```

The lower-level `launch!` accepts an architecture or an array and calls
`ensure_device!` before launching. Import KernelAbstractions names selectively;
both packages export a type named `CPU`.

## Distributed GPU transforms

The public distributed interface includes:

| Function or type | Purpose |
|------------------|---------|
| `check_cuda_aware_mpi()` | Check whether MPI accepts device buffers |
| `TransposableField(field)` | Create distributed transpose storage |
| `distributed_forward_transform!(tf)` | Distributed forward transform |
| `distributed_backward_transform!(tf)` | Distributed inverse transform |
| `active_layout(tf)` | Return the active transpose layout |
| `current_data(tf)` | Return the active local buffer |
| `local_shape(tf, layout)` | Return a local layout shape |

Multi-GPU runs must select a device per MPI rank:

```julia
using MPI

gpu_id = MPI.Comm_rank(comm) % CUDA.ndevices()
CUDA.device!(gpu_id)
arch = GPU(device_id=gpu_id)
```

GPU MPI communication requires CUDA-aware MPI. Missing support raises an error;
host staging is disabled. Pure complex-Fourier domains use `TransposableField`.
The limited distributed Fourier–Chebyshev DCT-I path is described in
[GPU Computing](../pages/gpu_computing.md#Multi-GPU-execution).

## Solver behavior

GPU runtime state and solver vectors stay on the device. Coupled GPU IVPs map
`:auto`, `:gpu`, and `:hybrid` to `:cuda_sparse`; CPU-only solver choices are
rejected. GPU LBVPs require an explicit CUDA solver. GPU NLBVP and EVP solves
are currently unsupported.

Unsupported solver operations and factorization failures raise errors rather
than retrying with a CPU solver.

## Extension boundary

CUDA kernels and transform plans under `ext/cuda/` are implementation details.
Application code should use the public architecture, transform, solver, and
`KernelOperation` interfaces documented above.
