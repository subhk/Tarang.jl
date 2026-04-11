# Distributed DCT for Multi-GPU Chebyshev Transforms

**Date:** 2026-01-31
**Status:** Design Complete
**Estimated Effort:** 5-6 weeks

## Overview

This document describes the design for true multi-GPU DCT (Discrete Cosine Transform) scaling in Tarang.jl, enabling distributed Chebyshev transforms across multiple GPUs using NCCL for communication.

### Goals

- Support general 3D Chebyshev domains with 512+ points per dimension
- Scale across multiple GPUs (NVLink, PCIe, and multi-node)
- Transparent API - existing user code works unchanged
- Memory-efficient algorithm (no data doubling)

### Key Decisions

| Decision | Choice |
|----------|--------|
| Implementation | Pure Julia + NCCL (no cuFFTMp C interop) |
| Algorithm | FFT-based DCT with even-odd reordering |
| Decomposition | Pencil (2D) for better scaling |
| API | Transparent dispatch based on architecture |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Code (unchanged)                     │
│         forward_transform!(field)  /  backward_transform!    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Transform Dispatch (transform_gpu.jl)           │
│  Detects: single-GPU vs distributed, Fourier vs Chebyshev   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────┐         ┌──────────────────────────┐
│  Single-GPU DCT      │         │  Distributed DCT (NEW)   │
│  (ext/cuda/dct.jl)   │         │  (ext/cuda/dct_distributed.jl)
└──────────────────────┘         └──────────────────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │  NCCL Transpose Layer    │
                                 │  (gpu_distributed.jl)    │
                                 │  - Pencil decomposition  │
                                 │  - All-to-all comm       │
                                 └──────────────────────────┘
```

### New Files

- `ext/cuda/pencil.jl` - Pencil decomposition and transpose (~300 lines)
- `ext/cuda/dct_distributed.jl` - Distributed DCT plans and execution (~400 lines)
- `test/test_distributed_dct.jl` - Comprehensive tests (~300 lines)

### Modified Files

- `src/core/gpu_distributed.jl` - Add pencil support, enhance NCCL (+200 lines)
- `ext/cuda/transforms.jl` - Add distributed DCT dispatch (+50 lines)

## Pencil Decomposition

For a 3D domain (Nx × Ny × Nz) distributed across a P₁ × P₂ process grid:

```
Global domain: Nx × Ny × Nz = 512 × 512 × 512
Process grid:  P₁ × P₂ = 4 × 4 (16 GPUs)

Three data layouts (pencils aligned along each axis):

X-pencil: (Nx, Ny/P₁, Nz/P₂) = (512, 128, 128)
          └─ Full X dimension, local FFT/DCT in X

Y-pencil: (Nx/P₁, Ny, Nz/P₂) = (128, 512, 128)
          └─ Full Y dimension, local FFT/DCT in Y

Z-pencil: (Nx/P₁, Ny/P₂, Nz) = (128, 128, 512)
          └─ Full Z dimension, local FFT/DCT in Z
```

### Data Structure

```julia
struct PencilDecomposition
    # Process grid
    proc_grid::Tuple{Int, Int}      # (P₁, P₂)
    row_comm::MPI.Comm              # Communicator for P₁ ranks (same row)
    col_comm::MPI.Comm              # Communicator for P₂ ranks (same column)

    # Global and local shapes for each layout
    global_shape::NTuple{3, Int}
    x_pencil_shape::NTuple{3, Int}  # Local shape when X is full
    y_pencil_shape::NTuple{3, Int}  # Local shape when Y is full
    z_pencil_shape::NTuple{3, Int}  # Local shape when Z is full

    # Current layout
    current_layout::Symbol          # :x_pencil, :y_pencil, :z_pencil
end
```

### Transpose Operations

- `X→Y`: All-to-all on `col_comm` (P₂ ranks)
- `Y→Z`: All-to-all on `row_comm` (P₁ ranks)
- Reverse for backward transforms

## NCCL Communication Layer

### Data Structure

```julia
struct NCCLTranspose{T}
    # Communicators (subset of GPUs for each transpose direction)
    row_nccl_comm::Any              # NCCL comm for row (Y↔Z transpose)
    col_nccl_comm::Any              # NCCL comm for column (X↔Y transpose)

    # Pre-allocated GPU buffers for all-to-all
    send_buffer::CuArray{T, 3}      # Packed data to send
    recv_buffer::CuArray{T, 3}      # Received data before unpack

    # Send/recv counts and displacements (for variable-size slabs)
    send_counts::Vector{Int}
    recv_counts::Vector{Int}
    send_displs::Vector{Int}
    recv_displs::Vector{Int}
end
```

### All-to-All Implementation

```julia
function nccl_alltoall!(send_buf::CuArray, recv_buf::CuArray,
                        counts::Vector{Int}, comm)
    nranks = length(counts)

    # NCCL doesn't have native Alltoallv, so we use grouped send/recv
    NCCL.group_start()
    for peer in 0:(nranks-1)
        send_slice = view(send_buf, get_slice(peer, send_displs, counts)...)
        recv_slice = view(recv_buf, get_slice(peer, recv_displs, counts)...)
        NCCL.send!(send_slice, peer, comm)
        NCCL.recv!(recv_slice, peer, comm)
    end
    NCCL.group_end()

    CUDA.synchronize()  # Ensure completion
end
```

### Fallback Strategy

- Detect if NCCL unavailable → use CUDA-aware MPI
- Detect if CUDA-aware MPI unavailable → CPU staging (existing code)

## Optimized DCT Algorithm

Memory-efficient FFT-based DCT using even-odd reordering instead of symmetric extension:

```
Standard FFT-based DCT (original):
  x[N] → symmetric_extend → y[2N] → FFT(2N) → twiddle → coeffs[N]
  Memory: 2N complex (4N floats for Float64)

Optimized DCT (new):
  x[N] → reorder → z[N] → R2C FFT(N) → twiddle → coeffs[N]
  Memory: N real + N/2+1 complex (~2N floats)

Memory reduction: ~60%
```

### Forward DCT

```julia
function dct_forward_optimized!(coeffs, x, plan)
    N = length(x)

    # Step 1: Reorder - interleave even and odd indices
    # z[k] = x[2k] for k = 0..N/2-1  (even indices)
    # z[N-1-k] = x[2k+1] for k = 0..N/2-1  (odd indices, reversed)
    reorder_for_dct!(plan.work_real, x)

    # Step 2: Real-to-Complex FFT of length N
    # Output: N/2+1 complex values
    plan.rfft_plan * plan.work_real  # In-place R2C

    # Step 3: Apply twiddle factors: exp(-i*π*k/(2N))
    # and extract real part for DCT coefficients
    apply_twiddle_rfft!(coeffs, plan.work_complex, plan.twiddle)
end
```

### Backward DCT

```julia
function dct_backward_optimized!(x, coeffs, plan)
    N = length(coeffs)

    # Step 1: Apply inverse twiddle: exp(+iπk/2N)
    apply_inverse_twiddle!(plan.work_complex, coeffs, plan.twiddle_inv)

    # Step 2: Inverse Real FFT (C2R)
    irfft!(plan.work_real, plan.work_complex)

    # Step 3: Inverse reorder (undo even-odd interleaving)
    x[1:2:N] = plan.work_real[1:N÷2]         # even indices
    x[2:2:N] = plan.work_real[N:-1:N÷2+1]    # odd indices
end
```

### Distributed Transform Sequence

For 3D DCT starting in Z-pencil layout:

```
Forward DCT (grid → coefficients):

Step 1: DCT in Z (local - Z is full in Z-pencil)
Step 2: Transpose Z→Y (NCCL all-to-all on row_comm)
Step 3: DCT in Y (local - Y is now full)
Step 4: Transpose Y→X (NCCL all-to-all on col_comm)
Step 5: DCT in X (local - X is now full)

Result: Spectral coefficients in X-pencil layout
```

### Memory Usage

For 512³ on 16 GPUs (4×4 grid):

| Metric | Value |
|--------|-------|
| Per-GPU local size | 128 × 128 × 512 |
| Per-GPU memory (data) | ~64 MB (Float64) |
| Per-GPU memory (work) | ~96 MB |
| Per-GPU peak | ~160 MB |
| Total across 16 GPUs | ~2.5 GB |

## Integration

### Dispatch Logic

```julia
function Tarang.gpu_forward_transform!(field::ScalarField)
    arch = field.dist.architecture

    if !is_gpu(arch)
        return false
    end

    # Check if distributed GPU setup
    if is_distributed_gpu(field.dist)
        return distributed_gpu_forward_transform!(field)
    end

    # Existing single-GPU path
    return single_gpu_forward_transform!(field)
end

function is_distributed_gpu(dist::Distributor)
    return is_gpu(dist.architecture) &&
           dist.nprocs > 1 &&
           nccl_available()
end
```

### Plan Structure

```julia
struct DistributedDCTPlan{T}
    # Local transform plans (one per dimension)
    local_rfft_plans::NTuple{3, CUFFT.rCuFFTPlan}
    local_irfft_plans::NTuple{3, CUFFT.rCuFFTPlan}

    # Twiddle factors: exp(-iπk/2N) for each dimension
    twiddles::NTuple{3, CuVector{Complex{T}}}
    twiddles_inv::NTuple{3, CuVector{Complex{T}}}

    # Work arrays (real and complex, per dimension)
    work_real::NTuple{3, CuArray{T, 3}}
    work_complex::NTuple{3, CuArray{Complex{T}, 3}}

    # Pencil decomposition
    pencil::PencilDecomposition

    # NCCL transpose
    transpose::NCCLTranspose{T}
end
```

## Testing Strategy

### Level 1: Unit Tests

```julia
@testset "DCT Reorder" begin
    x = CuArray(rand(512))
    work = similar(x)
    reorder_for_dct!(work, x)
    inverse_reorder!(result, work)
    @test result ≈ x atol=1e-14
end

@testset "Optimized DCT vs Reference" begin
    x = CuArray(rand(512))
    ref = gpu_forward_dct_1d!(similar(x), x, original_plan)
    opt = dct_forward_optimized!(similar(x), x, optimized_plan)
    @test ref ≈ opt rtol=1e-12
end
```

### Level 2: Distributed Transpose Tests

```julia
@testset "NCCL Transpose Correctness" begin
    # Round-trip: Z→Y→X→Y→Z
    transpose_to_pencil!(local_data, pencil, :Y)
    transpose_to_pencil!(local_data, pencil, :X)
    transpose_to_pencil!(local_data, pencil, :Y)
    transpose_to_pencil!(local_data, pencil, :Z)

    @test gathered_result ≈ original_data rtol=1e-14
end
```

### Level 3: End-to-End Validation

```julia
@testset "Distributed DCT vs Single-GPU Reference" begin
    ref_coeffs = single_gpu_dct_3d(CuArray(global_data))
    dist_coeffs = distributed_dct_3d(global_data, pencil)
    @test gathered ≈ Array(ref_coeffs) rtol=1e-11
end
```

### Level 4: Physics Validation

```julia
@testset "Parseval's Theorem" begin
    physical_energy = global_sum(x.^2)
    spectral_energy = global_sum(coeffs.^2) * normalization_factor
    @test physical_energy ≈ spectral_energy rtol=1e-10
end
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- `ext/cuda/pencil.jl`: PencilDecomposition struct, process grid setup
- Tests: Process grid creation, shape calculations

### Phase 2: NCCL Transpose (Week 2-3)

- Extend `gpu_distributed.jl`: NCCLTranspose, sub-communicators, all-to-all
- `ext/cuda/pencil.jl`: pack/unpack kernels, transpose_to_pencil!
- Tests: Round-trip transpose, data integrity

### Phase 3: Optimized Local DCT (Week 3-4)

- `ext/cuda/dct_distributed.jl`: reorder kernels, twiddle kernels, optimized DCT
- Tests: Optimized vs reference, round-trip correctness

### Phase 4: Integration (Week 4-5)

- DistributedDCTPlan, plan caching
- Transform dispatch modifications
- Tests: End-to-end, physics validation

### Phase 5: Optimization & Hardening (Week 5-6)

- Overlap transpose with local DCT (pipelining)
- Tune NCCL chunk sizes
- Handle edge cases (non-power-of-2, uneven decomposition)
- Documentation and examples

## Summary

| Metric | Value |
|--------|-------|
| New code | ~1000 lines |
| Modified code | ~250 lines |
| Test code | ~300 lines |
| Estimated effort | 5-6 weeks |
| Memory efficiency | 60% reduction vs naive |
| Scaling target | 16+ GPUs with >80% efficiency |
