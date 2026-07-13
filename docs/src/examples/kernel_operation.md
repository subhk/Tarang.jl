# KernelOperation: Portable CPU/GPU Kernels

Use `KernelOperation` when you want a single KernelAbstractions kernel to run
on both CPU and GPU backends through Tarang's architecture abstraction.

Import the kernel macros by name rather than with `using KernelAbstractions`:
KernelAbstractions also exports `CPU`, so a bare `using Tarang, KernelAbstractions`
makes `CPU()` ambiguous and `device = CPU()` fails with an `UndefVarError`.

```julia
using Tarang
using KernelAbstractions: @kernel, @index, @Const

# Define a simple kernel (here: y .= α * x)
@kernel function scale_kernel!(y, α, @Const(x))
    i = @index(Global)
    @inbounds y[i] = α * x[i]
end

# Wrap the kernel; the do-block computes the default ndrange from the call arguments
scale_op = KernelOperation(scale_kernel!) do y, α, x
    length(y)  # can return a tuple for multi-dimensional arrays
end

# Example data
device = CPU()  # or GPU(), which requires CUDA.jl to be loaded
x = ones(device, Float64, 1024)
y = zeros(device, Float64, 1024)

# Launch via the operation (auto-selects backend, workgroup, etc.)
scale_op(device, y, 2.0, x)   # y == fill(2.0, 1024)
```

The same wrapper works on multi-dimensional arrays as long as the kernel indexes
them that way and the do-block returns the matching `ndrange` tuple:

```julia
@kernel function scale2d!(y, α, @Const(x))
    i, j = @index(Global, NTuple)
    @inbounds y[i, j] = α * x[i, j]
end

op2 = KernelOperation(scale2d!) do y, α, x
    size(y)
end

X = ones(device, Float64, 8, 4)
Y = zeros(device, Float64, 8, 4)
op2(device, Y, 2.5, X)        # Y == fill(2.5, 8, 4)
```

Key points:

- Backend selection is driven by `device`: `CPU()`, or `GPU()` once CUDA.jl is
  loaded (`GPU()` carries the CUDA device, so `GPU(device_id=1)` picks the second
  one). Without CUDA.jl loaded, `GPU()` throws a clear error telling you to load it.
- The workgroup size is chosen for you by `workgroup_size(device, ndrange)`
  (`workgroup_size(CPU(), 1024) == 64`) inside `launch!`.
- Pass `ndrange` to override the do-block default for a single call:
  `scale_op(device, y, 3.0, x; ndrange=512)` touches only `y[1:512]`.
- On the GPU, call `synchronize(device)` when you need to wait for the launch to
  finish before reading the result.

If you do not want the wrapper, `launch!` takes the raw kernel and an explicit
`ndrange`, and accepts either an architecture or an array (whose architecture it
infers):

```julia
launch!(device, scale_kernel!, y, 4.0, x; ndrange=length(y))
launch!(y, scale_kernel!, y, 5.0, x; ndrange=length(y))
```
