# GPU Transform Dispatch

Tarang executes every transform for a `GPU()` field on its device. Transform
size does not select a CPU fallback. Unsupported GPU layouts raise an error.

```julia
using Tarang

field = ScalarField(dist, "θ", bases)
set_gpu_fft_mode!(field, :gpu)   # require the GPU backend explicitly
set_gpu_fft_mode!(field, :auto)  # default; still always on-device for GPU fields
```

`set_gpu_fft_mode!(field, :cpu)` is rejected for GPU fields. To run transforms
on CPU, construct the field with a `CPU()` distributor and transfer inputs
explicitly.

The legacy global FFT size threshold does not affect GPU fields.
