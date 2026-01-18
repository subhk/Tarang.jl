# Tarang.jl

[![CI](https://github.com/subhk/Tarang.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/subhk/Tarang.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/subhk/Tarang.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/subhk/Tarang.jl/actions/workflows/Documentation.yml)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/Tarang.jl/dev/)

A Julia implementation of the spectral PDE framework, designed for high-performance parallel computing.

## Examples
- [KernelOperation: Portable CPU/GPU Kernels](docs/src/examples/kernel_operation.md)
- [GPU Solvers: Custom Preconditioners](docs/src/examples/gpu_solvers_preconditioning.md)
- [GPU FFT Heuristics](docs/src/examples/gpu_fft_heuristics.md)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Tarang.jl")
```
