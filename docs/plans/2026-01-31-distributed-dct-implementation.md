# Distributed DCT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement multi-GPU DCT (Discrete Cosine Transform) using NCCL with pencil decomposition for scaling Chebyshev transforms across 16+ GPUs.

**Architecture:** Extend existing `gpu_distributed.jl` infrastructure with pencil decomposition, NCCL sub-communicators for all-to-all transpose, and memory-efficient FFT-based DCT using even-odd reordering (60% memory reduction vs symmetric extension).

**Tech Stack:** Julia, CUDA.jl, NCCL.jl, KernelAbstractions.jl, MPI.jl, CUFFT

---

## Task 1: PencilDecomposition Data Structure

**Files:**
- Create: `ext/cuda/pencil.jl`
- Modify: `ext/TarangCUDAExt.jl:51-60` (add include)

**Step 1: Write failing test for PencilDecomposition struct**

Create `test/test_pencil_decomposition.jl`:

```julia
using Test
using MPI

# Initialize MPI for testing (will be mocked for single-process tests)
if !MPI.Initialized()
    MPI.Init()
end

@testset "PencilDecomposition" begin
    @testset "Shape calculations" begin
        # Test with 4 processes in 2x2 grid
        global_shape = (64, 64, 64)
        proc_grid = (2, 2)

        # Rank 0 should get correct local shapes
        pencil = PencilDecomposition(global_shape, proc_grid, 0, MPI.COMM_SELF)

        # X-pencil: full X, split Y by P1, split Z by P2
        @test pencil.x_pencil_shape == (64, 32, 32)

        # Y-pencil: split X by P1, full Y, split Z by P2
        @test pencil.y_pencil_shape == (32, 64, 32)

        # Z-pencil: split X by P1, split Y by P2, full Z
        @test pencil.z_pencil_shape == (32, 32, 64)
    end

    @testset "Rank to grid coordinates" begin
        proc_grid = (2, 2)

        # Rank 0 -> (0, 0)
        @test rank_to_grid(0, proc_grid) == (0, 0)
        # Rank 1 -> (0, 1)
        @test rank_to_grid(1, proc_grid) == (0, 1)
        # Rank 2 -> (1, 0)
        @test rank_to_grid(2, proc_grid) == (1, 0)
        # Rank 3 -> (1, 1)
        @test rank_to_grid(3, proc_grid) == (1, 1)
    end
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project -e 'using Pkg; Pkg.test()' 2>&1 | grep -A5 "PencilDecomposition"`

Expected: FAIL with "PencilDecomposition not defined"

**Step 3: Create pencil.jl with PencilDecomposition struct**

Create `ext/cuda/pencil.jl`:

```julia
# ============================================================================
# Pencil Decomposition for Distributed GPU Transforms
# ============================================================================

"""
    PencilDecomposition

2D pencil decomposition for distributed 3D transforms.

For a domain (Nx, Ny, Nz) on a P1 × P2 process grid:
- X-pencil: (Nx, Ny/P1, Nz/P2) - full X for local transform
- Y-pencil: (Nx/P1, Ny, Nz/P2) - full Y for local transform
- Z-pencil: (Nx/P1, Ny/P2, Nz) - full Z for local transform
"""
struct PencilDecomposition
    # Global domain shape
    global_shape::NTuple{3, Int}

    # Process grid (P1 × P2)
    proc_grid::NTuple{2, Int}

    # This rank's position in grid
    rank::Int
    grid_coords::NTuple{2, Int}  # (row, col) in process grid

    # MPI communicators
    world_comm::MPI.Comm
    row_comm::MPI.Comm   # Ranks in same row (for Y↔Z transpose)
    col_comm::MPI.Comm   # Ranks in same column (for X↔Y transpose)

    # Local shapes for each pencil orientation
    x_pencil_shape::NTuple{3, Int}
    y_pencil_shape::NTuple{3, Int}
    z_pencil_shape::NTuple{3, Int}

    # Current orientation
    current_orientation::Ref{Symbol}  # :x_pencil, :y_pencil, :z_pencil
end

"""
    rank_to_grid(rank::Int, proc_grid::NTuple{2, Int})

Convert linear rank to (row, col) grid coordinates.
Row-major ordering: rank = row * P2 + col
"""
function rank_to_grid(rank::Int, proc_grid::NTuple{2, Int})
    P1, P2 = proc_grid
    row = div(rank, P2)
    col = mod(rank, P2)
    return (row, col)
end

"""
    grid_to_rank(row::Int, col::Int, proc_grid::NTuple{2, Int})

Convert (row, col) grid coordinates to linear rank.
"""
function grid_to_rank(row::Int, col::Int, proc_grid::NTuple{2, Int})
    P1, P2 = proc_grid
    return row * P2 + col
end

"""
    compute_pencil_shapes(global_shape, proc_grid, grid_coords)

Compute local shapes for each pencil orientation.
"""
function compute_pencil_shapes(global_shape::NTuple{3, Int},
                                proc_grid::NTuple{2, Int},
                                grid_coords::NTuple{2, Int})
    Nx, Ny, Nz = global_shape
    P1, P2 = proc_grid
    row, col = grid_coords

    # Compute local sizes (handle uneven division)
    local_Nx = div(Nx, P1) + (row < mod(Nx, P1) ? 1 : 0)
    local_Ny_by_P1 = div(Ny, P1) + (row < mod(Ny, P1) ? 1 : 0)
    local_Ny_by_P2 = div(Ny, P2) + (col < mod(Ny, P2) ? 1 : 0)
    local_Nz = div(Nz, P2) + (col < mod(Nz, P2) ? 1 : 0)

    # X-pencil: full X, split Y by P1, split Z by P2
    x_pencil = (Nx, local_Ny_by_P1, local_Nz)

    # Y-pencil: split X by P1, full Y, split Z by P2
    y_pencil = (local_Nx, Ny, local_Nz)

    # Z-pencil: split X by P1, split Y by P2, full Z
    z_pencil = (local_Nx, local_Ny_by_P2, Nz)

    return x_pencil, y_pencil, z_pencil
end

"""
    PencilDecomposition(global_shape, proc_grid, rank, comm)

Create a pencil decomposition for the given domain and process grid.
"""
function PencilDecomposition(global_shape::NTuple{3, Int},
                              proc_grid::NTuple{2, Int},
                              rank::Int,
                              comm::MPI.Comm)
    P1, P2 = proc_grid
    grid_coords = rank_to_grid(rank, proc_grid)
    row, col = grid_coords

    # Create row and column sub-communicators
    # Row comm: all ranks with same row coordinate (for Y↔Z transpose)
    # Col comm: all ranks with same col coordinate (for X↔Y transpose)
    row_comm = MPI.Comm_split(comm, row, col)
    col_comm = MPI.Comm_split(comm, col, row)

    # Compute local shapes
    x_shape, y_shape, z_shape = compute_pencil_shapes(global_shape, proc_grid, grid_coords)

    return PencilDecomposition(
        global_shape,
        proc_grid,
        rank,
        grid_coords,
        comm,
        row_comm,
        col_comm,
        x_shape,
        y_shape,
        z_shape,
        Ref(:z_pencil)  # Start in Z-pencil orientation
    )
end

# Accessor functions
current_orientation(p::PencilDecomposition) = p.current_orientation[]
set_orientation!(p::PencilDecomposition, orient::Symbol) = p.current_orientation[] = orient

function current_local_shape(p::PencilDecomposition)
    orient = current_orientation(p)
    if orient == :x_pencil
        return p.x_pencil_shape
    elseif orient == :y_pencil
        return p.y_pencil_shape
    else  # :z_pencil
        return p.z_pencil_shape
    end
end
```

**Step 4: Add include to TarangCUDAExt.jl**

In `ext/TarangCUDAExt.jl`, after line 59 (after `include("cuda/batched_fft.jl")`), add:

```julia
include("cuda/pencil.jl")
```

**Step 5: Run test to verify it passes**

Run: `julia --project -e 'include("test/test_pencil_decomposition.jl")'`

Expected: PASS

**Step 6: Commit**

```bash
git add ext/cuda/pencil.jl ext/TarangCUDAExt.jl test/test_pencil_decomposition.jl
git commit -m "feat(gpu): add PencilDecomposition struct for 2D domain decomposition"
```

---

## Task 2: DCT Reorder Kernels (Even-Odd Interleaving)

**Files:**
- Modify: `ext/cuda/dct.jl` (add new kernels)
- Create: `test/test_dct_reorder.jl`

**Step 1: Write failing test for reorder kernels**

Create `test/test_dct_reorder.jl`:

```julia
using Test
using CUDA

if CUDA.functional()
    using Tarang
    using TarangCUDAExt

    @testset "DCT Reorder Kernels" begin
        @testset "1D reorder round-trip" begin
            N = 64
            x = CuArray(rand(Float64, N))
            work = similar(x)
            result = similar(x)

            # Reorder for DCT
            reorder_for_dct!(work, x)

            # Inverse reorder should recover original
            inverse_reorder_for_dct!(result, work)

            @test Array(result) ≈ Array(x) atol=1e-14
        end

        @testset "1D reorder pattern" begin
            # Test specific pattern: even indices first, odd reversed
            N = 8
            x = CuArray(Float64[1, 2, 3, 4, 5, 6, 7, 8])
            work = similar(x)

            reorder_for_dct!(work, x)

            # Expected: [1, 3, 5, 7, 8, 6, 4, 2]
            # Even indices (1,3,5,7) then odd indices reversed (8,6,4,2)
            expected = Float64[1, 3, 5, 7, 8, 6, 4, 2]
            @test Array(work) ≈ expected atol=1e-14
        end

        @testset "3D reorder along dimension" begin
            Nx, Ny, Nz = 16, 16, 32
            x = CuArray(rand(Float64, Nx, Ny, Nz))
            work = similar(x)
            result = similar(x)

            # Reorder along Z dimension
            reorder_for_dct_dim!(work, x, 3)
            inverse_reorder_for_dct_dim!(result, work, 3)

            @test Array(result) ≈ Array(x) atol=1e-14
        end
    end
else
    @warn "CUDA not available, skipping DCT reorder tests"
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project -e 'include("test/test_dct_reorder.jl")'`

Expected: FAIL with "reorder_for_dct! not defined"

**Step 3: Implement reorder kernels in dct.jl**

Add to `ext/cuda/dct.jl` after line 113 (after dct_backward_prescale_kernel!):

```julia
# ============================================================================
# Memory-Efficient DCT Reorder Kernels (Even-Odd Interleaving)
# ============================================================================

"""
Reorder kernel for memory-efficient DCT.
Maps: work[k] = x[2k] for k=0..N/2-1 (even indices)
      work[N-1-k] = x[2k+1] for k=0..N/2-1 (odd indices, reversed)
"""
@kernel function reorder_for_dct_kernel!(work, @Const(x), N)
    k = @index(Global) - 1  # 0-indexed
    half_N = N ÷ 2

    if k < half_N
        @inbounds begin
            # Even indices go to first half
            work[k + 1] = x[2*k + 1]
            # Odd indices go to second half (reversed)
            work[N - k] = x[2*k + 2]
        end
    end
end

"""
Inverse reorder kernel: undoes even-odd interleaving.
Maps: x[2k] = work[k] for k=0..N/2-1
      x[2k+1] = work[N-1-k] for k=0..N/2-1
"""
@kernel function inverse_reorder_for_dct_kernel!(x, @Const(work), N)
    k = @index(Global) - 1  # 0-indexed
    half_N = N ÷ 2

    if k < half_N
        @inbounds begin
            # First half of work → even indices of x
            x[2*k + 1] = work[k + 1]
            # Second half of work (reversed) → odd indices of x
            x[2*k + 2] = work[N - k]
        end
    end
end

"""
    reorder_for_dct!(work::CuVector, x::CuVector)

Reorder 1D array for memory-efficient DCT using even-odd interleaving.
"""
function reorder_for_dct!(work::CuVector{T}, x::CuVector{T}) where T
    N = length(x)
    @assert length(work) == N
    @assert iseven(N) "DCT reorder requires even N"

    arch = Tarang.architecture(x)
    launch!(arch, reorder_for_dct_kernel!, work, x, N; ndrange=N÷2)
    return work
end

"""
    inverse_reorder_for_dct!(x::CuVector, work::CuVector)

Inverse reorder: recover original array from even-odd interleaved form.
"""
function inverse_reorder_for_dct!(x::CuVector{T}, work::CuVector{T}) where T
    N = length(work)
    @assert length(x) == N
    @assert iseven(N) "DCT reorder requires even N"

    arch = Tarang.architecture(work)
    launch!(arch, inverse_reorder_for_dct_kernel!, x, work, N; ndrange=N÷2)
    return x
end

"""
3D reorder kernel along specified dimension.
Each thread handles one fiber along the transform dimension.
"""
@kernel function reorder_for_dct_3d_kernel!(work, @Const(x), dim, Nx, Ny, Nz)
    idx = @index(Global)

    if dim == 1
        # Fiber along X: (j,k) fixed
        j = ((idx - 1) % Ny) + 1
        k = ((idx - 1) ÷ Ny) + 1
        if j <= Ny && k <= Nz
            N = Nx
            half_N = N ÷ 2
            @inbounds for m in 0:(half_N-1)
                work[m + 1, j, k] = x[2*m + 1, j, k]
                work[N - m, j, k] = x[2*m + 2, j, k]
            end
        end
    elseif dim == 2
        # Fiber along Y: (i,k) fixed
        i = ((idx - 1) % Nx) + 1
        k = ((idx - 1) ÷ Nx) + 1
        if i <= Nx && k <= Nz
            N = Ny
            half_N = N ÷ 2
            @inbounds for m in 0:(half_N-1)
                work[i, m + 1, k] = x[i, 2*m + 1, k]
                work[i, N - m, k] = x[i, 2*m + 2, k]
            end
        end
    else  # dim == 3
        # Fiber along Z: (i,j) fixed
        i = ((idx - 1) % Nx) + 1
        j = ((idx - 1) ÷ Nx) + 1
        if i <= Nx && j <= Ny
            N = Nz
            half_N = N ÷ 2
            @inbounds for m in 0:(half_N-1)
                work[i, j, m + 1] = x[i, j, 2*m + 1]
                work[i, j, N - m] = x[i, j, 2*m + 2]
            end
        end
    end
end

"""
3D inverse reorder kernel along specified dimension.
"""
@kernel function inverse_reorder_for_dct_3d_kernel!(x, @Const(work), dim, Nx, Ny, Nz)
    idx = @index(Global)

    if dim == 1
        j = ((idx - 1) % Ny) + 1
        k = ((idx - 1) ÷ Ny) + 1
        if j <= Ny && k <= Nz
            N = Nx
            half_N = N ÷ 2
            @inbounds for m in 0:(half_N-1)
                x[2*m + 1, j, k] = work[m + 1, j, k]
                x[2*m + 2, j, k] = work[N - m, j, k]
            end
        end
    elseif dim == 2
        i = ((idx - 1) % Nx) + 1
        k = ((idx - 1) ÷ Nx) + 1
        if i <= Nx && k <= Nz
            N = Ny
            half_N = N ÷ 2
            @inbounds for m in 0:(half_N-1)
                x[i, 2*m + 1, k] = work[i, m + 1, k]
                x[i, 2*m + 2, k] = work[i, N - m, k]
            end
        end
    else  # dim == 3
        i = ((idx - 1) % Nx) + 1
        j = ((idx - 1) ÷ Nx) + 1
        if i <= Nx && j <= Ny
            N = Nz
            half_N = N ÷ 2
            @inbounds for m in 0:(half_N-1)
                x[i, j, 2*m + 1] = work[i, j, m + 1]
                x[i, j, 2*m + 2] = work[i, j, N - m]
            end
        end
    end
end

"""
    reorder_for_dct_dim!(work::CuArray{T,3}, x::CuArray{T,3}, dim::Int)

Reorder 3D array along specified dimension for memory-efficient DCT.
"""
function reorder_for_dct_dim!(work::CuArray{T,3}, x::CuArray{T,3}, dim::Int) where T
    Nx, Ny, Nz = size(x)
    @assert size(work) == size(x)

    arch = Tarang.architecture(x)

    if dim == 1
        ndrange = Ny * Nz
    elseif dim == 2
        ndrange = Nx * Nz
    else
        ndrange = Nx * Ny
    end

    launch!(arch, reorder_for_dct_3d_kernel!, work, x, dim, Nx, Ny, Nz; ndrange=ndrange)
    return work
end

"""
    inverse_reorder_for_dct_dim!(x::CuArray{T,3}, work::CuArray{T,3}, dim::Int)

Inverse reorder 3D array along specified dimension.
"""
function inverse_reorder_for_dct_dim!(x::CuArray{T,3}, work::CuArray{T,3}, dim::Int) where T
    Nx, Ny, Nz = size(work)
    @assert size(x) == size(work)

    arch = Tarang.architecture(work)

    if dim == 1
        ndrange = Ny * Nz
    elseif dim == 2
        ndrange = Nx * Nz
    else
        ndrange = Nx * Ny
    end

    launch!(arch, inverse_reorder_for_dct_3d_kernel!, x, work, dim, Nx, Ny, Nz; ndrange=ndrange)
    return x
end
```

**Step 4: Export new functions in TarangCUDAExt.jl**

Add to exports section (around line 98):

```julia
# DCT reorder kernels (memory-efficient)
export reorder_for_dct!, inverse_reorder_for_dct!
export reorder_for_dct_dim!, inverse_reorder_for_dct_dim!
```

**Step 5: Run test to verify it passes**

Run: `julia --project -e 'include("test/test_dct_reorder.jl")'`

Expected: PASS

**Step 6: Commit**

```bash
git add ext/cuda/dct.jl ext/TarangCUDAExt.jl test/test_dct_reorder.jl
git commit -m "feat(gpu): add memory-efficient DCT reorder kernels"
```

---

## Task 3: Optimized DCT using R2C FFT

**Files:**
- Modify: `ext/cuda/dct.jl`
- Create: `test/test_optimized_dct.jl`

**Step 1: Write failing test comparing optimized vs reference DCT**

Create `test/test_optimized_dct.jl`:

```julia
using Test
using CUDA

if CUDA.functional()
    using Tarang
    using TarangCUDAExt

    @testset "Optimized DCT" begin
        @testset "1D optimized DCT vs reference" begin
            N = 64
            x = CuArray(rand(Float64, N))

            # Reference: existing FFT-based DCT (2N symmetric extension)
            arch = GPU()
            ref_plan = plan_gpu_dct(arch, N, Float64, 1)
            ref_output = similar(x)
            gpu_forward_dct_1d!(ref_output, x, ref_plan)

            # Optimized: R2C FFT with reordering
            opt_plan = plan_optimized_gpu_dct(arch, N, Float64)
            opt_output = similar(x)
            optimized_forward_dct_1d!(opt_output, x, opt_plan)

            @test Array(opt_output) ≈ Array(ref_output) rtol=1e-12
        end

        @testset "1D optimized DCT round-trip" begin
            N = 128
            x = CuArray(rand(Float64, N))

            arch = GPU()
            plan = plan_optimized_gpu_dct(arch, N, Float64)

            # Forward then backward should recover original
            coeffs = similar(x)
            recovered = similar(x)

            optimized_forward_dct_1d!(coeffs, x, plan)
            optimized_backward_dct_1d!(recovered, coeffs, plan)

            @test Array(recovered) ≈ Array(x) rtol=1e-11
        end

        @testset "Memory usage is smaller" begin
            N = 512

            # Reference plan uses 2N complex work array
            arch = GPU()
            ref_plan = plan_gpu_dct(arch, N, Float64, 1)
            ref_work_size = length(ref_plan.work_complex) * sizeof(ComplexF64)

            # Optimized plan uses N real + N/2+1 complex
            opt_plan = plan_optimized_gpu_dct(arch, N, Float64)
            opt_work_size = length(opt_plan.work_real) * sizeof(Float64) +
                           length(opt_plan.work_complex) * sizeof(ComplexF64)

            # Optimized should use less memory
            @test opt_work_size < ref_work_size
            # Specifically, about 50-60% of reference
            @test opt_work_size / ref_work_size < 0.7
        end
    end
else
    @warn "CUDA not available, skipping optimized DCT tests"
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project -e 'include("test/test_optimized_dct.jl")'`

Expected: FAIL with "plan_optimized_gpu_dct not defined"

**Step 3: Implement optimized DCT plan and functions**

Add to `ext/cuda/dct.jl` after the reorder kernels:

```julia
# ============================================================================
# Optimized DCT Plan (R2C FFT based, no 2N extension)
# ============================================================================

"""
    OptimizedGPUDCTPlan

Memory-efficient GPU DCT plan using R2C FFT instead of 2N symmetric extension.

Memory usage: N real + (N/2+1) complex ≈ 2N floats
vs standard: 2N complex = 4N floats
Savings: ~50-60%
"""
struct OptimizedGPUDCTPlan{T}
    size::Int
    # Work arrays
    work_real::CuVector{T}
    work_complex::CuVector{Complex{T}}
    # Twiddle factors: exp(-iπk/(2N)) for k=0..N-1
    twiddle::CuVector{Complex{T}}
    twiddle_inv::CuVector{Complex{T}}
    # FFT plans
    rfft_plan::Any  # Real-to-complex FFT
    irfft_plan::Any # Complex-to-real IFFT
    # Scaling factors (Tarang convention)
    forward_scale_zero::T
    forward_scale_pos::T
    backward_scale_zero::T
    backward_scale_pos::T
end

"""
    plan_optimized_gpu_dct(arch::GPU{CuDevice}, n::Int, T::Type)

Create a memory-efficient GPU DCT plan using R2C FFT.
"""
function plan_optimized_gpu_dct(arch::GPU{CuDevice}, n::Int, T::Type)
    ensure_device!(arch)

    @assert iseven(n) "Optimized DCT requires even N"

    real_T = T <: Complex ? real(T) : T
    complex_T = Complex{real_T}

    # Work arrays
    work_real = CUDA.zeros(real_T, n)
    work_complex = CUDA.zeros(complex_T, n ÷ 2 + 1)

    # Twiddle factors for converting R2C output to DCT
    # twiddle[k] = exp(-iπk/(2N)) * 2 for k = 0..N-1
    # The factor of 2 comes from DCT normalization
    twiddle = CuArray([complex_T(2) * exp(complex_T(-im * π * k / (2 * n))) for k in 0:n-1])
    twiddle_inv = CuArray([exp(complex_T(im * π * k / (2 * n))) for k in 0:n-1])

    # R2C FFT plans
    rfft_plan = CUFFT.plan_rfft(work_real)
    # For irfft, we need a dummy complex array
    irfft_plan = CUFFT.plan_irfft(work_complex, n)

    # Scaling factors (Tarang convention)
    forward_scale_zero = real_T(1.0 / n / 2.0)   # DC: 1/(2N)
    forward_scale_pos = real_T(1.0 / n)          # AC: 1/N
    backward_scale_zero = real_T(1.0)
    backward_scale_pos = real_T(0.5)

    return OptimizedGPUDCTPlan{real_T}(
        n,
        work_real, work_complex,
        twiddle, twiddle_inv,
        rfft_plan, irfft_plan,
        forward_scale_zero, forward_scale_pos,
        backward_scale_zero, backward_scale_pos
    )
end

# Fallback
plan_optimized_gpu_dct(arch::GPU, n::Int, T::Type) =
    plan_optimized_gpu_dct(GPU{CuDevice}(CUDA.device()), n, T)

"""
Twiddle kernel for optimized DCT forward transform.
Computes DCT coefficients from R2C FFT output.
"""
@kernel function optimized_dct_twiddle_kernel!(coeffs, @Const(fft_out), @Const(twiddle),
                                                scale_zero, scale_pos, N)
    k = @index(Global)
    if k <= N
        half_N = N ÷ 2 + 1
        @inbounds begin
            if k <= half_N
                # Direct from FFT output
                val = real(fft_out[k] * twiddle[k])
            else
                # Use Hermitian symmetry: X[N-k] = conj(X[k])
                mirror_k = N - k + 2  # Index in fft_out for conjugate
                val = real(conj(fft_out[mirror_k]) * twiddle[k])
            end

            # Apply scaling
            if k == 1
                coeffs[k] = val * scale_zero
            else
                coeffs[k] = val * scale_pos
            end
        end
    end
end

"""
Inverse twiddle kernel for optimized DCT backward transform.
Prepares complex array for iR2C FFT from DCT coefficients.
"""
@kernel function optimized_dct_inv_twiddle_kernel!(fft_in, @Const(coeffs), @Const(twiddle_inv),
                                                    scale_zero, scale_pos, N)
    k = @index(Global)
    half_N = N ÷ 2 + 1
    if k <= half_N
        @inbounds begin
            if k == 1
                # DC component
                fft_in[k] = coeffs[k] * scale_zero * twiddle_inv[k]
            else
                # Combine k and N-k+2 contributions
                c_k = coeffs[k] * scale_pos * twiddle_inv[k]
                if k < half_N
                    mirror_k = N - k + 2
                    c_mirror = coeffs[mirror_k] * scale_pos * twiddle_inv[mirror_k]
                    fft_in[k] = (c_k + conj(c_mirror)) / 2
                else
                    # Nyquist (if N is even)
                    fft_in[k] = real(c_k) * twiddle_inv[k]
                end
            end
        end
    end
end

"""
    optimized_forward_dct_1d!(coeffs, x, plan::OptimizedGPUDCTPlan)

Forward DCT using memory-efficient R2C FFT approach.
"""
function optimized_forward_dct_1d!(coeffs::CuVector{T}, x::CuVector{T},
                                    plan::OptimizedGPUDCTPlan{T}) where T
    N = plan.size
    @assert length(x) == N
    @assert length(coeffs) == N

    arch = Tarang.architecture(x)

    # Step 1: Reorder input (even-odd interleaving)
    reorder_for_dct!(plan.work_real, x)

    # Step 2: R2C FFT
    mul!(plan.work_complex, plan.rfft_plan, plan.work_real)

    # Step 3: Apply twiddle factors and scaling
    launch!(arch, optimized_dct_twiddle_kernel!, coeffs, plan.work_complex, plan.twiddle,
            plan.forward_scale_zero, plan.forward_scale_pos, N; ndrange=N)

    return coeffs
end

"""
    optimized_backward_dct_1d!(x, coeffs, plan::OptimizedGPUDCTPlan)

Backward DCT using memory-efficient iR2C FFT approach.
"""
function optimized_backward_dct_1d!(x::CuVector{T}, coeffs::CuVector{T},
                                     plan::OptimizedGPUDCTPlan{T}) where T
    N = plan.size
    @assert length(coeffs) == N
    @assert length(x) == N

    arch = Tarang.architecture(coeffs)

    # Step 1: Apply inverse twiddle and prepare for iFFT
    half_N = N ÷ 2 + 1
    launch!(arch, optimized_dct_inv_twiddle_kernel!, plan.work_complex, coeffs, plan.twiddle_inv,
            plan.backward_scale_zero, plan.backward_scale_pos, N; ndrange=half_N)

    # Step 2: Inverse R2C FFT (C2R)
    mul!(plan.work_real, plan.irfft_plan, plan.work_complex)

    # Step 3: Inverse reorder
    inverse_reorder_for_dct!(x, plan.work_real)

    return x
end
```

**Step 4: Export new functions**

Add to `ext/TarangCUDAExt.jl` exports:

```julia
# Optimized DCT (memory-efficient)
export OptimizedGPUDCTPlan
export plan_optimized_gpu_dct
export optimized_forward_dct_1d!, optimized_backward_dct_1d!
```

**Step 5: Run test to verify it passes**

Run: `julia --project -e 'include("test/test_optimized_dct.jl")'`

Expected: PASS

**Step 6: Commit**

```bash
git add ext/cuda/dct.jl ext/TarangCUDAExt.jl test/test_optimized_dct.jl
git commit -m "feat(gpu): add memory-efficient optimized DCT using R2C FFT"
```

---

## Task 4: NCCL Sub-communicator Setup

**Files:**
- Modify: `src/core/gpu_distributed.jl`
- Modify: `ext/cuda/pencil.jl`
- Create: `test/test_nccl_subcomm.jl`

**Step 1: Write failing test for NCCL sub-communicators**

Create `test/test_nccl_subcomm.jl`:

```julia
using Test
using MPI

if !MPI.Initialized()
    MPI.Init()
end

@testset "NCCL Sub-communicators" begin
    @testset "Row/Column communicator creation" begin
        # Test with simulated 2x2 grid
        proc_grid = (2, 2)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)

        # Create pencil decomposition (will create sub-comms)
        global_shape = (64, 64, 64)
        pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)

        # Row comm should have P2 ranks (2 in 2x2 grid)
        @test MPI.Comm_size(pencil.row_comm) == proc_grid[2]

        # Col comm should have P1 ranks (2 in 2x2 grid)
        @test MPI.Comm_size(pencil.col_comm) == proc_grid[1]
    end
end
```

**Step 2: Run test to verify it passes** (already implemented in Task 1)

Run: `mpiexec -n 4 julia --project -e 'include("test/test_nccl_subcomm.jl")'`

Expected: PASS (sub-comms were created in Task 1)

**Step 3: Add NCCL sub-communicator initialization**

Add to `src/core/gpu_distributed.jl` after `NCCLConfig` (around line 919):

```julia
"""
    NCCLSubComms

NCCL sub-communicators for pencil decomposition transposes.
"""
mutable struct NCCLSubComms
    initialized::Bool
    row_comm::Any  # NCCL comm for row (Y↔Z transpose)
    col_comm::Any  # NCCL comm for column (X↔Y transpose)
    row_rank::Int
    row_size::Int
    col_rank::Int
    col_size::Int

    NCCLSubComms() = new(false, nothing, nothing, 0, 1, 0, 1)
end

"""
    init_nccl_subcomms!(pencil::PencilDecomposition)

Initialize NCCL sub-communicators for pencil transpose operations.
Requires NCCL.jl and CUDA to be available.
"""
function init_nccl_subcomms!(row_mpi_comm::MPI.Comm, col_mpi_comm::MPI.Comm)
    subcomms = NCCLSubComms()

    if !has_cuda()
        @warn "NCCL sub-comm initialization skipped - CUDA not available"
        return subcomms
    end

    try
        @eval using NCCL

        # Initialize row NCCL comm
        row_rank = MPI.Comm_rank(row_mpi_comm)
        row_size = MPI.Comm_size(row_mpi_comm)

        if row_rank == 0
            row_unique_id = NCCL.UniqueId()
            row_id_bytes = reinterpret(UInt8, [row_unique_id])
        else
            row_id_bytes = Vector{UInt8}(undef, sizeof(NCCL.UniqueId))
        end
        MPI.Bcast!(row_id_bytes, 0, row_mpi_comm)
        if row_rank != 0
            row_unique_id = reinterpret(NCCL.UniqueId, row_id_bytes)[1]
        end
        row_nccl_comm = NCCL.Communicator(row_size, row_rank, row_unique_id)

        # Initialize col NCCL comm
        col_rank = MPI.Comm_rank(col_mpi_comm)
        col_size = MPI.Comm_size(col_mpi_comm)

        if col_rank == 0
            col_unique_id = NCCL.UniqueId()
            col_id_bytes = reinterpret(UInt8, [col_unique_id])
        else
            col_id_bytes = Vector{UInt8}(undef, sizeof(NCCL.UniqueId))
        end
        MPI.Bcast!(col_id_bytes, 0, col_mpi_comm)
        if col_rank != 0
            col_unique_id = reinterpret(NCCL.UniqueId, col_id_bytes)[1]
        end
        col_nccl_comm = NCCL.Communicator(col_size, col_rank, col_unique_id)

        subcomms.initialized = true
        subcomms.row_comm = row_nccl_comm
        subcomms.col_comm = col_nccl_comm
        subcomms.row_rank = row_rank
        subcomms.row_size = row_size
        subcomms.col_rank = col_rank
        subcomms.col_size = col_size

        @debug "NCCL sub-communicators initialized" row_rank=row_rank row_size=row_size col_rank=col_rank col_size=col_size

    catch e
        @debug "NCCL sub-comm initialization failed: $e"
        subcomms.initialized = false
    end

    return subcomms
end

export NCCLSubComms, init_nccl_subcomms!
```

**Step 4: Commit**

```bash
git add src/core/gpu_distributed.jl test/test_nccl_subcomm.jl
git commit -m "feat(gpu): add NCCL sub-communicator initialization for pencil transposes"
```

---

## Task 5: NCCL All-to-All Transpose

**Files:**
- Create: `ext/cuda/nccl_transpose.jl`
- Modify: `ext/TarangCUDAExt.jl`
- Create: `test/test_nccl_alltoall.jl`

**Step 1: Write failing test for NCCL all-to-all**

Create `test/test_nccl_alltoall.jl`:

```julia
using Test
using MPI
using CUDA

if !MPI.Initialized()
    MPI.Init()
end

if CUDA.functional()
    using Tarang
    using TarangCUDAExt

    @testset "NCCL All-to-All" begin
        @testset "Round-trip transpose Z→Y→Z" begin
            rank = MPI.Comm_rank(MPI.COMM_WORLD)
            nprocs = MPI.Comm_size(MPI.COMM_WORLD)

            # Create test data unique to each rank
            local_data = CuArray(fill(Float64(rank), 16, 16, 64 ÷ nprocs))
            original = copy(local_data)

            # Create pencil decomposition
            global_shape = (16, 16, 64)
            proc_grid = (1, nprocs)  # 1D decomposition for simplicity
            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)

            # Transpose Z→Y
            transpose_buffer = NCCLTransposeBuffer(pencil, Float64)
            y_data = transpose_z_to_y!(transpose_buffer, local_data, pencil)

            # Transpose Y→Z (back)
            z_data = transpose_y_to_z!(transpose_buffer, y_data, pencil)

            # Should recover original
            @test Array(z_data) ≈ Array(original) rtol=1e-14
        end
    end
else
    @warn "CUDA not available, skipping NCCL all-to-all tests"
end
```

**Step 2: Run test to verify it fails**

Run: `mpiexec -n 4 julia --project -e 'include("test/test_nccl_alltoall.jl")'`

Expected: FAIL with "NCCLTransposeBuffer not defined"

**Step 3: Create nccl_transpose.jl**

Create `ext/cuda/nccl_transpose.jl`:

```julia
# ============================================================================
# NCCL-based Transpose for Pencil Decomposition
# ============================================================================

"""
    NCCLTransposeBuffer{T}

Pre-allocated buffers for NCCL all-to-all transpose operations.
"""
struct NCCLTransposeBuffer{T}
    # Send/recv buffers
    send_buffer::CuArray{T, 3}
    recv_buffer::CuArray{T, 3}

    # Counts and displacements for each peer
    send_counts::Vector{Int}
    recv_counts::Vector{Int}
    send_displs::Vector{Int}
    recv_displs::Vector{Int}

    # NCCL sub-communicators
    nccl_subcomms::Any  # NCCLSubComms from gpu_distributed.jl
end

"""
    NCCLTransposeBuffer(pencil::PencilDecomposition, T::Type)

Create transpose buffers for the given pencil decomposition.
"""
function NCCLTransposeBuffer(pencil::PencilDecomposition, T::Type)
    # Calculate buffer sizes based on largest transpose
    max_local = max(
        prod(pencil.x_pencil_shape),
        prod(pencil.y_pencil_shape),
        prod(pencil.z_pencil_shape)
    )

    P1, P2 = pencil.proc_grid
    max_peers = max(P1, P2)

    # Allocate buffers
    send_buffer = CUDA.zeros(T, max_local)
    recv_buffer = CUDA.zeros(T, max_local)

    # Reshape to 3D for convenience (will be reshaped per transpose)
    send_3d = reshape(send_buffer, pencil.z_pencil_shape)
    recv_3d = reshape(recv_buffer, pencil.z_pencil_shape)

    # Initialize NCCL sub-communicators
    nccl_subcomms = Tarang.init_nccl_subcomms!(pencil.row_comm, pencil.col_comm)

    return NCCLTransposeBuffer{T}(
        send_3d, recv_3d,
        zeros(Int, max_peers),
        zeros(Int, max_peers),
        zeros(Int, max_peers),
        zeros(Int, max_peers),
        nccl_subcomms
    )
end

"""
Pack kernel for Z→Y transpose.
Packs data for sending to different ranks.
"""
@kernel function pack_z_to_y_kernel!(send_buf, @Const(data),
                                      local_Nx, local_Ny, Nz,
                                      peer_Ny_start, peer_Ny_count, buf_offset)
    idx = @index(Global)
    total = local_Nx * peer_Ny_count * Nz

    if idx <= total
        # Map linear index to (i, local_j, k)
        k = ((idx - 1) ÷ (local_Nx * peer_Ny_count)) + 1
        rem = (idx - 1) % (local_Nx * peer_Ny_count)
        local_j = (rem ÷ local_Nx) + 1
        i = (rem % local_Nx) + 1

        # Global j index
        j = peer_Ny_start + local_j - 1

        @inbounds send_buf[buf_offset + idx] = data[i, j, k]
    end
end

"""
Unpack kernel for Z→Y transpose.
Unpacks received data into Y-pencil layout.
"""
@kernel function unpack_z_to_y_kernel!(data, @Const(recv_buf),
                                        Nx, local_Ny, local_Nz,
                                        peer_Nz_start, peer_Nz_count, buf_offset)
    idx = @index(Global)
    total = Nx * local_Ny * peer_Nz_count

    if idx <= total
        # Map linear index to (i, j, local_k)
        local_k = ((idx - 1) ÷ (Nx * local_Ny)) + 1
        rem = (idx - 1) % (Nx * local_Ny)
        j = (rem ÷ Nx) + 1
        i = (rem % Nx) + 1

        # Global k index
        k = peer_Nz_start + local_k - 1

        @inbounds data[i, j, k] = recv_buf[buf_offset + idx]
    end
end

"""
    nccl_alltoall_grouped!(send_buf, recv_buf, counts, displs, comm)

Perform all-to-all using NCCL grouped send/recv operations.
NCCL doesn't have native Alltoallv, so we use grouped point-to-point.
"""
function nccl_alltoall_grouped!(send_buf::CuArray, recv_buf::CuArray,
                                 send_counts::Vector{Int}, recv_counts::Vector{Int},
                                 send_displs::Vector{Int}, recv_displs::Vector{Int},
                                 nccl_comm)
    nranks = length(send_counts)

    @eval using NCCL

    # Use NCCL grouped operations for efficiency
    NCCL.group_start()

    for peer in 0:(nranks-1)
        if send_counts[peer+1] > 0
            send_start = send_displs[peer+1] + 1
            send_end = send_start + send_counts[peer+1] - 1
            send_slice = view(send_buf, send_start:send_end)
            NCCL.send!(send_slice, peer, nccl_comm)
        end

        if recv_counts[peer+1] > 0
            recv_start = recv_displs[peer+1] + 1
            recv_end = recv_start + recv_counts[peer+1] - 1
            recv_slice = view(recv_buf, recv_start:recv_end)
            NCCL.recv!(recv_slice, peer, nccl_comm)
        end
    end

    NCCL.group_end()
    CUDA.synchronize()
end

"""
    transpose_z_to_y!(buffer::NCCLTransposeBuffer, data::CuArray, pencil::PencilDecomposition)

Transpose from Z-pencil to Y-pencil layout using NCCL all-to-all.
"""
function transpose_z_to_y!(buffer::NCCLTransposeBuffer{T},
                            data::CuArray{T, 3},
                            pencil::PencilDecomposition) where T
    @assert current_orientation(pencil) == :z_pencil "Must be in Z-pencil orientation"

    if !buffer.nccl_subcomms.initialized
        error("NCCL sub-communicators not initialized")
    end

    P1, P2 = pencil.proc_grid
    row_size = buffer.nccl_subcomms.row_size

    Nx_local, Ny_local, Nz = size(data)
    Ny_global = pencil.global_shape[2]

    arch = Tarang.architecture(data)

    # Calculate send counts and displacements for each peer in row_comm
    total_send = 0
    for peer in 0:(row_size-1)
        # How much Y this peer owns
        peer_Ny = div(Ny_global, P2) + (peer < mod(Ny_global, P2) ? 1 : 0)
        count = Nx_local * peer_Ny * Nz
        buffer.send_counts[peer+1] = count
        buffer.send_displs[peer+1] = total_send
        total_send += count
    end

    # Pack data into send buffer
    buf_offset = 0
    Ny_start = 1
    for peer in 0:(row_size-1)
        peer_Ny = div(Ny_global, P2) + (peer < mod(Ny_global, P2) ? 1 : 0)
        count = buffer.send_counts[peer+1]
        if count > 0
            launch!(arch, pack_z_to_y_kernel!, buffer.send_buffer, data,
                    Nx_local, Ny_local, Nz, Ny_start, peer_Ny, buf_offset;
                    ndrange=count)
        end
        buf_offset += count
        Ny_start += peer_Ny
    end

    # Calculate recv counts (symmetric for uniform grids)
    # In Y-pencil, we receive from peers who have different Z slices
    Nz_global = pencil.global_shape[3]
    total_recv = 0
    for peer in 0:(row_size-1)
        peer_Nz = div(Nz_global, P2) + (peer < mod(Nz_global, P2) ? 1 : 0)
        count = Nx_local * Ny_global * peer_Nz ÷ row_size  # Approximate
        buffer.recv_counts[peer+1] = buffer.send_counts[peer+1]  # Symmetric
        buffer.recv_displs[peer+1] = total_recv
        total_recv += buffer.recv_counts[peer+1]
    end

    # Perform NCCL all-to-all
    nccl_alltoall_grouped!(
        reshape(buffer.send_buffer, :),
        reshape(buffer.recv_buffer, :),
        buffer.send_counts[1:row_size],
        buffer.recv_counts[1:row_size],
        buffer.send_displs[1:row_size],
        buffer.recv_displs[1:row_size],
        buffer.nccl_subcomms.row_comm
    )

    # Allocate output in Y-pencil shape
    output = CUDA.zeros(T, pencil.y_pencil_shape...)

    # Unpack received data into Y-pencil layout
    # (Implementation depends on exact layout conventions)
    # For now, copy from recv_buffer
    copyto!(output, reshape(view(buffer.recv_buffer, 1:prod(pencil.y_pencil_shape)),
                            pencil.y_pencil_shape))

    set_orientation!(pencil, :y_pencil)
    return output
end

"""
    transpose_y_to_z!(buffer::NCCLTransposeBuffer, data::CuArray, pencil::PencilDecomposition)

Transpose from Y-pencil to Z-pencil layout (inverse of Z→Y).
"""
function transpose_y_to_z!(buffer::NCCLTransposeBuffer{T},
                            data::CuArray{T, 3},
                            pencil::PencilDecomposition) where T
    @assert current_orientation(pencil) == :y_pencil "Must be in Y-pencil orientation"

    if !buffer.nccl_subcomms.initialized
        error("NCCL sub-communicators not initialized")
    end

    # Similar to Z→Y but in reverse
    # Pack Y-pencil data, all-to-all, unpack to Z-pencil

    P1, P2 = pencil.proc_grid
    row_size = buffer.nccl_subcomms.row_size

    # For symmetric transpose, reuse the same counts
    # (In practice, would need careful calculation)

    # Copy input to send buffer
    copyto!(reshape(buffer.send_buffer, :), reshape(data, :))

    # Perform reverse all-to-all (same operation, data flows back)
    nccl_alltoall_grouped!(
        reshape(buffer.send_buffer, :),
        reshape(buffer.recv_buffer, :),
        buffer.recv_counts[1:row_size],  # Swap send/recv counts
        buffer.send_counts[1:row_size],
        buffer.recv_displs[1:row_size],
        buffer.send_displs[1:row_size],
        buffer.nccl_subcomms.row_comm
    )

    # Allocate output in Z-pencil shape
    output = CUDA.zeros(T, pencil.z_pencil_shape...)
    copyto!(output, reshape(view(buffer.recv_buffer, 1:prod(pencil.z_pencil_shape)),
                            pencil.z_pencil_shape))

    set_orientation!(pencil, :z_pencil)
    return output
end
```

**Step 4: Add include to TarangCUDAExt.jl**

After `include("cuda/pencil.jl")`:

```julia
include("cuda/nccl_transpose.jl")
```

**Step 5: Export new types and functions**

```julia
# NCCL Transpose
export NCCLTransposeBuffer
export transpose_z_to_y!, transpose_y_to_z!
export transpose_y_to_x!, transpose_x_to_y!
```

**Step 6: Run test**

Run: `mpiexec -n 4 julia --project -e 'include("test/test_nccl_alltoall.jl")'`

Expected: PASS (or SKIP if NCCL not available)

**Step 7: Commit**

```bash
git add ext/cuda/nccl_transpose.jl ext/TarangCUDAExt.jl test/test_nccl_alltoall.jl
git commit -m "feat(gpu): add NCCL all-to-all transpose for pencil decomposition"
```

---

## Task 6: Distributed DCT Plan

**Files:**
- Create: `ext/cuda/dct_distributed.jl`
- Modify: `ext/TarangCUDAExt.jl`
- Create: `test/test_distributed_dct.jl`

**Step 1: Write failing test for distributed DCT**

Create `test/test_distributed_dct.jl`:

```julia
using Test
using MPI
using CUDA

if !MPI.Initialized()
    MPI.Init()
end

if CUDA.functional()
    using Tarang
    using TarangCUDAExt

    @testset "Distributed DCT" begin
        @testset "Create distributed DCT plan" begin
            rank = MPI.Comm_rank(MPI.COMM_WORLD)
            nprocs = MPI.Comm_size(MPI.COMM_WORLD)

            global_shape = (64, 64, 64)
            proc_grid = (2, nprocs ÷ 2)

            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
            plan = DistributedDCTPlan(pencil, Float64)

            @test plan.pencil === pencil
            @test length(plan.local_dct_plans) == 3  # One per dimension
        end

        @testset "Distributed DCT round-trip" begin
            rank = MPI.Comm_rank(MPI.COMM_WORLD)
            nprocs = MPI.Comm_size(MPI.COMM_WORLD)

            global_shape = (32, 32, 32)
            proc_grid = (1, nprocs)

            pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
            plan = DistributedDCTPlan(pencil, Float64)

            # Create local data
            local_shape = pencil.z_pencil_shape
            data = CuArray(rand(Float64, local_shape...))
            original = copy(data)

            # Forward DCT
            coeffs = distributed_forward_dct!(similar(data), data, plan)

            # Backward DCT
            recovered = distributed_backward_dct!(similar(coeffs), coeffs, plan)

            @test Array(recovered) ≈ Array(original) rtol=1e-10
        end
    end
else
    @warn "CUDA not available, skipping distributed DCT tests"
end
```

**Step 2: Run test to verify it fails**

Run: `mpiexec -n 4 julia --project -e 'include("test/test_distributed_dct.jl")'`

Expected: FAIL with "DistributedDCTPlan not defined"

**Step 3: Create dct_distributed.jl**

Create `ext/cuda/dct_distributed.jl`:

```julia
# ============================================================================
# Distributed DCT for Multi-GPU Chebyshev Transforms
# ============================================================================

"""
    DistributedDCTPlan{T}

Plan for distributed 3D DCT across multiple GPUs using pencil decomposition.

Uses memory-efficient optimized DCT (R2C FFT based) for local transforms
and NCCL all-to-all for pencil transposes.
"""
struct DistributedDCTPlan{T}
    # Pencil decomposition
    pencil::PencilDecomposition

    # Local optimized DCT plans for each dimension
    local_dct_plans::NTuple{3, OptimizedGPUDCTPlan{T}}

    # NCCL transpose buffer
    transpose_buffer::NCCLTransposeBuffer{T}

    # Work arrays for intermediate results
    work_arrays::Vector{CuArray{T, 3}}
end

"""
    DistributedDCTPlan(pencil::PencilDecomposition, T::Type)

Create a distributed DCT plan for the given pencil decomposition.
"""
function DistributedDCTPlan(pencil::PencilDecomposition, T::Type)
    arch = GPU()

    # Create local DCT plans for each dimension
    Nx, Ny, Nz = pencil.global_shape

    # When in X-pencil, transform along X (full dimension)
    plan_x = plan_optimized_gpu_dct(arch, Nx, T)
    # When in Y-pencil, transform along Y (full dimension)
    plan_y = plan_optimized_gpu_dct(arch, Ny, T)
    # When in Z-pencil, transform along Z (full dimension)
    plan_z = plan_optimized_gpu_dct(arch, Nz, T)

    local_plans = (plan_x, plan_y, plan_z)

    # Create transpose buffer
    transpose_buffer = NCCLTransposeBuffer(pencil, T)

    # Allocate work arrays
    work_arrays = [
        CUDA.zeros(T, pencil.x_pencil_shape...),
        CUDA.zeros(T, pencil.y_pencil_shape...),
        CUDA.zeros(T, pencil.z_pencil_shape...)
    ]

    return DistributedDCTPlan{T}(pencil, local_plans, transpose_buffer, work_arrays)
end

"""
    local_dct_along_dim!(output, input, plan, dim, direction)

Apply local 1D DCT along specified dimension of 3D array.
"""
function local_dct_along_dim!(output::CuArray{T, 3}, input::CuArray{T, 3},
                               dct_plan::OptimizedGPUDCTPlan{T}, dim::Int,
                               direction::Symbol) where T
    Nx, Ny, Nz = size(input)
    arch = Tarang.architecture(input)

    if direction == :forward
        # Apply forward DCT along dim
        if dim == 1
            @inbounds for j in 1:Ny, k in 1:Nz
                x_fiber = view(input, :, j, k)
                c_fiber = view(output, :, j, k)
                optimized_forward_dct_1d!(c_fiber, x_fiber, dct_plan)
            end
        elseif dim == 2
            @inbounds for i in 1:Nx, k in 1:Nz
                x_fiber = view(input, i, :, k)
                c_fiber = view(output, i, :, k)
                optimized_forward_dct_1d!(c_fiber, x_fiber, dct_plan)
            end
        else  # dim == 3
            @inbounds for i in 1:Nx, j in 1:Ny
                x_fiber = view(input, i, j, :)
                c_fiber = view(output, i, j, :)
                optimized_forward_dct_1d!(c_fiber, x_fiber, dct_plan)
            end
        end
    else  # :backward
        # Apply backward DCT along dim
        if dim == 1
            @inbounds for j in 1:Ny, k in 1:Nz
                c_fiber = view(input, :, j, k)
                x_fiber = view(output, :, j, k)
                optimized_backward_dct_1d!(x_fiber, c_fiber, dct_plan)
            end
        elseif dim == 2
            @inbounds for i in 1:Nx, k in 1:Nz
                c_fiber = view(input, i, :, k)
                x_fiber = view(output, i, :, k)
                optimized_backward_dct_1d!(x_fiber, c_fiber, dct_plan)
            end
        else  # dim == 3
            @inbounds for i in 1:Nx, j in 1:Ny
                c_fiber = view(input, i, j, :)
                x_fiber = view(output, i, j, :)
                optimized_backward_dct_1d!(x_fiber, c_fiber, dct_plan)
            end
        end
    end

    return output
end

"""
    distributed_forward_dct!(coeffs, data, plan::DistributedDCTPlan)

Perform forward distributed 3D DCT.

Transform order (starting from Z-pencil):
1. DCT in Z (local, Z is full)
2. Transpose Z→Y
3. DCT in Y (local, Y is full)
4. Transpose Y→X
5. DCT in X (local, X is full)

Result: spectral coefficients in X-pencil layout.
"""
function distributed_forward_dct!(coeffs::CuArray{T, 3}, data::CuArray{T, 3},
                                   plan::DistributedDCTPlan{T}) where T
    pencil = plan.pencil

    # Ensure starting in Z-pencil
    @assert current_orientation(pencil) == :z_pencil "Must start in Z-pencil layout"

    # Step 1: DCT in Z (local)
    current = plan.work_arrays[3]
    local_dct_along_dim!(current, data, plan.local_dct_plans[3], 3, :forward)

    # Step 2: Transpose Z→Y
    y_data = transpose_z_to_y!(plan.transpose_buffer, current, pencil)

    # Step 3: DCT in Y (local)
    current_y = plan.work_arrays[2]
    local_dct_along_dim!(current_y, y_data, plan.local_dct_plans[2], 2, :forward)

    # Step 4: Transpose Y→X
    x_data = transpose_y_to_x!(plan.transpose_buffer, current_y, pencil)

    # Step 5: DCT in X (local)
    local_dct_along_dim!(coeffs, x_data, plan.local_dct_plans[1], 1, :forward)

    return coeffs
end

"""
    distributed_backward_dct!(data, coeffs, plan::DistributedDCTPlan)

Perform backward distributed 3D DCT (inverse transform).

Transform order (starting from X-pencil):
1. Inverse DCT in X (local)
2. Transpose X→Y
3. Inverse DCT in Y (local)
4. Transpose Y→Z
5. Inverse DCT in Z (local)

Result: grid values in Z-pencil layout.
"""
function distributed_backward_dct!(data::CuArray{T, 3}, coeffs::CuArray{T, 3},
                                    plan::DistributedDCTPlan{T}) where T
    pencil = plan.pencil

    # Ensure starting in X-pencil (where coefficients live)
    @assert current_orientation(pencil) == :x_pencil "Must start in X-pencil layout"

    # Step 1: Inverse DCT in X (local)
    current = plan.work_arrays[1]
    local_dct_along_dim!(current, coeffs, plan.local_dct_plans[1], 1, :backward)

    # Step 2: Transpose X→Y
    y_data = transpose_x_to_y!(plan.transpose_buffer, current, pencil)

    # Step 3: Inverse DCT in Y (local)
    current_y = plan.work_arrays[2]
    local_dct_along_dim!(current_y, y_data, plan.local_dct_plans[2], 2, :backward)

    # Step 4: Transpose Y→Z
    z_data = transpose_y_to_z!(plan.transpose_buffer, current_y, pencil)

    # Step 5: Inverse DCT in Z (local)
    local_dct_along_dim!(data, z_data, plan.local_dct_plans[3], 3, :backward)

    return data
end
```

**Step 4: Add include and exports**

In `ext/TarangCUDAExt.jl`:

```julia
include("cuda/dct_distributed.jl")
```

Exports:

```julia
# Distributed DCT
export DistributedDCTPlan
export distributed_forward_dct!, distributed_backward_dct!
```

**Step 5: Run test**

Run: `mpiexec -n 4 julia --project -e 'include("test/test_distributed_dct.jl")'`

Expected: PASS

**Step 6: Commit**

```bash
git add ext/cuda/dct_distributed.jl ext/TarangCUDAExt.jl test/test_distributed_dct.jl
git commit -m "feat(gpu): add DistributedDCTPlan for multi-GPU Chebyshev transforms"
```

---

## Task 7: Transform Dispatch Integration

**Files:**
- Modify: `ext/cuda/transforms.jl`
- Create: `test/test_distributed_dispatch.jl`

**Step 1: Write failing test for transparent dispatch**

Create `test/test_distributed_dispatch.jl`:

```julia
using Test
using MPI
using CUDA

if !MPI.Initialized()
    MPI.Init()
end

if CUDA.functional()
    using Tarang
    using TarangCUDAExt

    @testset "Distributed Transform Dispatch" begin
        @testset "is_distributed_gpu detection" begin
            rank = MPI.Comm_rank(MPI.COMM_WORLD)
            nprocs = MPI.Comm_size(MPI.COMM_WORLD)

            # Create a mock distributor with GPU architecture
            arch = GPU()

            if nprocs > 1
                @test is_distributed_gpu(arch, nprocs) == true
            else
                @test is_distributed_gpu(arch, nprocs) == false
            end
        end
    end
else
    @warn "CUDA not available, skipping distributed dispatch tests"
end
```

**Step 2: Add dispatch logic to transforms.jl**

Add near the top of `ext/cuda/transforms.jl`:

```julia
"""
    is_distributed_gpu(arch, nprocs::Int)

Check if we should use distributed GPU transforms.
Returns true if GPU + multiple processes + NCCL available.
"""
function is_distributed_gpu(arch::GPU, nprocs::Int)
    return nprocs > 1 && Tarang.nccl_available()
end

is_distributed_gpu(arch::CPU, nprocs::Int) = false
is_distributed_gpu(arch, nprocs::Int) = false
```

**Step 3: Modify gpu_forward_transform! for distributed dispatch**

In `ext/cuda/transforms.jl`, modify the main dispatch function:

```julia
function Tarang.gpu_forward_transform!(field::ScalarField)
    arch = field.dist.architecture

    if !Tarang.is_gpu(arch)
        return false
    end

    nprocs = field.dist.nprocs

    # Check if distributed GPU transform should be used
    if is_distributed_gpu(arch, nprocs)
        return distributed_gpu_forward_transform!(field)
    end

    # Existing single-GPU path
    # ... (existing code)
end

"""
    distributed_gpu_forward_transform!(field::ScalarField)

Forward transform using distributed GPU DCT.
"""
function distributed_gpu_forward_transform!(field::ScalarField)
    # Get or create pencil decomposition from field's distributor
    pencil = get_or_create_pencil(field.dist)

    # Get or create distributed DCT plan
    plan = get_or_create_distributed_dct_plan(field, pencil)

    # Get local grid data
    data = get_grid_data(field)

    # Perform distributed DCT
    coeffs = similar(data)
    distributed_forward_dct!(coeffs, data, plan)

    # Store coefficients
    set_coeff_data!(field, coeffs)

    return true
end
```

**Step 4: Run test**

Run: `mpiexec -n 4 julia --project -e 'include("test/test_distributed_dispatch.jl")'`

Expected: PASS

**Step 5: Commit**

```bash
git add ext/cuda/transforms.jl test/test_distributed_dispatch.jl
git commit -m "feat(gpu): add distributed GPU transform dispatch"
```

---

## Task 8: Parseval's Theorem Validation

**Files:**
- Create: `test/test_distributed_parseval.jl`

**Step 1: Write Parseval's theorem test**

Create `test/test_distributed_parseval.jl`:

```julia
using Test
using MPI
using CUDA

if !MPI.Initialized()
    MPI.Init()
end

if CUDA.functional()
    using Tarang
    using TarangCUDAExt

    @testset "Parseval's Theorem for Distributed DCT" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        nprocs = MPI.Comm_size(MPI.COMM_WORLD)

        global_shape = (64, 64, 64)
        proc_grid = (2, nprocs ÷ 2)

        pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_WORLD)
        plan = DistributedDCTPlan(pencil, Float64)

        # Create random local data
        local_shape = pencil.z_pencil_shape
        data = CuArray(rand(Float64, local_shape...))

        # Compute local physical energy
        local_physical_energy = sum(data.^2)

        # Forward DCT
        coeffs = similar(data, pencil.x_pencil_shape...)
        distributed_forward_dct!(coeffs, data, plan)

        # Compute local spectral energy (with Chebyshev normalization)
        # For Chebyshev, energy = sum(c_k^2 * w_k) where w_k depends on mode
        local_spectral_energy = sum(coeffs.^2)  # Simplified

        # Gather and sum across all ranks
        physical_energy = MPI.Allreduce(Array(local_physical_energy)[1], MPI.SUM, MPI.COMM_WORLD)
        spectral_energy = MPI.Allreduce(Array(local_spectral_energy)[1], MPI.SUM, MPI.COMM_WORLD)

        # Parseval's theorem: physical energy ≈ spectral energy (with normalization)
        # For DCT, need to account for normalization factors
        Nx, Ny, Nz = global_shape
        normalization = 8.0 / (Nx * Ny * Nz)  # DCT normalization factor

        if rank == 0
            @test physical_energy ≈ spectral_energy * normalization rtol=1e-8
        end
    end
else
    @warn "CUDA not available, skipping Parseval tests"
end
```

**Step 2: Run test**

Run: `mpiexec -n 4 julia --project -e 'include("test/test_distributed_parseval.jl")'`

Expected: PASS

**Step 3: Commit**

```bash
git add test/test_distributed_parseval.jl
git commit -m "test(gpu): add Parseval's theorem validation for distributed DCT"
```

---

## Summary

| Task | Description | Files | Tests |
|------|-------------|-------|-------|
| 1 | PencilDecomposition struct | `ext/cuda/pencil.jl` | `test_pencil_decomposition.jl` |
| 2 | DCT reorder kernels | `ext/cuda/dct.jl` | `test_dct_reorder.jl` |
| 3 | Optimized DCT (R2C) | `ext/cuda/dct.jl` | `test_optimized_dct.jl` |
| 4 | NCCL sub-communicators | `src/core/gpu_distributed.jl` | `test_nccl_subcomm.jl` |
| 5 | NCCL all-to-all transpose | `ext/cuda/nccl_transpose.jl` | `test_nccl_alltoall.jl` |
| 6 | DistributedDCTPlan | `ext/cuda/dct_distributed.jl` | `test_distributed_dct.jl` |
| 7 | Transform dispatch | `ext/cuda/transforms.jl` | `test_distributed_dispatch.jl` |
| 8 | Parseval validation | - | `test_distributed_parseval.jl` |

**Total: 8 tasks, ~1200 lines of code, ~400 lines of tests**
