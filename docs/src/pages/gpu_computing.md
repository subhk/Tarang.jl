# GPU Computing

Tarang uses CUDA.jl through a Julia package extension. Load CUDA before creating
GPU fields:

```julia
using CUDA
using Tarang

@assert CUDA.functional()
```

## Quick start

Choose the architecture on the `Distributor`; fields and solvers inherit it.

```julia
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; dtype=Float64, device=GPU())

xb = RealFourier(coords["x"]; size=256, bounds=(0.0, 2π))
yb = RealFourier(coords["y"]; size=256, bounds=(0.0, 2π))
domain = Domain(dist, (xb, yb))
u = ScalarField(domain, "u")

u["g"] .= CUDA.rand(Float64, size(u["g"])...)
forward_transform!(u)
backward_transform!(u)

@assert get_grid_data(u) isa CuArray
```

Use `CPU()` instead of `GPU()` to construct a CPU simulation.

## Strict GPU execution

A GPU field is never downloaded to run an unavailable CPU implementation.
Supported operations remain on the selected device; unsupported operations
raise an error. This applies to transforms, resampling, derivatives, solver
vectors, MPI communication, and stochastic phase generation.

Host transfers still occur when explicitly requested, for example:

```julia
host_data = on_architecture(CPU(), u["g"])
```

Output, checkpoints, gathers, and explicit architecture conversions are data
movement APIs, not computational fallbacks.

## Transforms

GPU transforms run on the device for every array size.

```julia
set_gpu_fft_mode!(u, :auto)  # default; device-resident
set_gpu_fft_mode!(u, :gpu)   # explicitly require the GPU backend
```

`set_gpu_fft_mode!(u, :cpu)` is rejected for GPU fields. The legacy global FFT
size threshold applies only to CPU-field preferences and cannot move a GPU field
to FFTW.

Current single-GPU transform support is:

| Bases | Support |
|-------|---------|
| Real or complex Fourier | Device FFT |
| 2D Fourier × Chebyshev | Device FFT plus DCT-I, including scaled grids |
| 3D Fourier × Chebyshev | Device FFT plus DCT-I for supported layouts |
| Same-size pure Chebyshev, up to 3D | Device DCT-I |
| Legendre transforms or scaled pure-Chebyshev transforms | Explicitly unsupported |

Unsupported basis and layout combinations fail before entering the CPU
transform chain.

### Complete 2D Fourier--Chebyshev path

The single-GPU 2D path supports `RealFourier` or `ComplexFourier` paired with
`ChebyshevT`, with either basis first. Scaled Chebyshev grids are transformed at
their nodal length and truncated or zero-padded on the device; Fourier scaling
used by 3/2-rule nonlinear products also remains device-resident. The focused
validation covers transforms, derivatives along both axes, dealiased products,
and a nonlinear wall-bounded RK222 IVP with tau boundary conditions and CUDA
sparse subproblem solves.

Run the strict validation on an NVIDIA node from the repository root:

```bash
julia --project=. test/run_gpu_fc_2d.jl
```

This command prints CUDA device information and fails if CUDA is unavailable,
scalar indexing is attempted, a CPU/GPU value comparison fails, or the warmed
IVP step performs a fresh device allocation.

The current boundary of this validation is deliberate:

- scaled pure-Chebyshev fields remain unsupported on the device;
- multi-rank 2D FC execution is not part of the single-GPU path (use the
  separately documented distributed interfaces for supported 3D layouts); and
- transform scratch is cached per device and shape for serial use. Run one
  transforming Julia task at a time on each CUDA device; concurrent same-shape
  transforms can share scratch and are not supported yet.

## 2D time stepping and solves

Pure-Fourier GPU IVPs use field-native stepping. When the left-hand side has an
implicit diagonal Fourier operator, select a diagonal IMEX scheme such as
`DiagonalIMEX_RK222()` or `DiagonalIMEX_SBDF2()` so the operator is applied in
spectral space on the device.

Fourier–Chebyshev IVPs use per-mode coupled subproblems. With GPU fields,
`matsolver=:auto`, `:gpu`, and `:hybrid` resolve to the concrete CUDA sparse
solver; CPU-only solvers are rejected and solver failures are not retried on
CPU.

GPU LBVPs require an explicit CUDA solver, for example:

```julia
solver = BoundaryValueSolver(problem; matsolver=:cuda_sparse)
```

GPU NLBVP and EVP solves are not device-native yet and raise an unsupported
operation error.

## Multi-GPU execution

Assign one device to each MPI rank before allocating fields:

```julia
using MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
gpu_id = rank % CUDA.ndevices()
CUDA.device!(gpu_id)

arch = GPU(device_id=gpu_id)
coords = CartesianCoordinates("x", "y", "z")
dist = Distributor(coords; comm=comm, dtype=ComplexF64, device=arch)
```

Distributed GPU communication requires CUDA-aware MPI. Tarang raises an error
when device buffers cannot be passed directly; it does not stage them through
host memory.

```julia
@assert check_cuda_aware_mpi()
```

Pure complex-Fourier domains use `TransposableField`:

```julia
bases = (
    ComplexFourier(coords["x"]; size=128, bounds=(0.0, 2π)),
    ComplexFourier(coords["y"]; size=128, bounds=(0.0, 2π)),
    ComplexFourier(coords["z"]; size=128, bounds=(0.0, 2π)),
)
field = ScalarField(Domain(dist, bases), "u")
tf = TransposableField(field)
distributed_forward_transform!(tf)
distributed_backward_transform!(tf)
```

The distributed DCT-I path supports selected three-dimensional
Fourier–Chebyshev layouts. It requires at least one Fourier and one Chebyshev
axis. `RealFourier` is allowed only on the first axis and cannot be combined
with another Fourier axis. Other layouts, including pure Chebyshev, are rejected.

## Custom kernels

Use the public architecture layer instead of importing implementation details
from `TarangCUDAExt`:

```julia
using KernelAbstractions: @kernel, @index

@kernel function scale_kernel!(y, x, α)
    i = @index(Global)
    @inbounds y[i] = α * x[i]
end

scale = KernelOperation(scale_kernel!) do y, x, α
    length(y)
end

arch = GPU()
x = ones(arch, Float64, 1024)
y = similar(x)
scale(arch, y, x, 2.0)
```

`KernelOperation` accepts an architecture or an array as its first argument;
passing `arch` makes the launch target explicit.

## Memory and profiling

Tarang uses CUDA.jl's memory pool. Prefer preallocated work arrays and use
CUDA.jl directly for inspection and profiling:

```julia
CUDA.memory_status()
CUDA.@profile begin
    forward_transform!(u)
    backward_transform!(u)
end
CUDA.reclaim()
```

See the [GPU API reference](../api/gpu.md) for the public architecture,
transform, and distributed interfaces.
