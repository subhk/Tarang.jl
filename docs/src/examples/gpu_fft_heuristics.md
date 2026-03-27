# GPU FFT Heuristics

Tarang lets you control when a field uses GPU FFTs via `set_gpu_fft_min_elements!`
and per-field `set_gpu_fft_mode!`.

```julia
using Tarang

field = ScalarField(dist, "θ", bases)
set_gpu_fft_mode!(field, :cpu)   # force CPU
set_gpu_fft_mode!(field, :gpu)   # force GPU
set_gpu_fft_mode!(field, :auto)  # default heuristic

set_gpu_fft_min_elements!(64_000)  # global cutoff for :auto fields
```
