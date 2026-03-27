# KernelOperation: Portable CPU/GPU Kernels

Use `KernelOperation` when you want a single KernelAbstractions kernel to run
on both CPU and GPU backends through Tarang's architecture abstraction.

```julia
using Tarang, KernelAbstractions

# Define a simple kernel (here: y .= α * x)
@kernel function scale_kernel!(y, α, @Const(x))
    i = @index(Global)
    @inbounds y[i] = α * x[i]
end

# Wrap the kernel
scale_op = KernelOperation(scale_kernel!) do y, α, x
    length(y)  # default ndrange (can be tuple for multi-d arrays)
end

# Example data
arch = GPU()  # or CPU()
x = ones(arch, Float64, 1024)
y = zeros(arch, Float64, 1024)

# Launch via the operation (auto-selects backend, workgroup, etc.)
scale_op(arch, y, 2.0, x)
```

Key benefits:

- Backend selection is driven by `arch` (a `CPU()`/`GPU()` or array).
- Workgroup size and synchronization are handled via `launch!`.
- You can pass `ndrange`/`dependencies` keywords to `scale_op` when needed.
```
