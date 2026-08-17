# Strict single-GPU validation entry point for the complete 2D
# Fourier--Chebyshev path. Unlike the test file itself, this runner must fail
# when CUDA is unavailable so a cluster validation job cannot pass by skipping.

try
    @eval using CUDA
catch err
    error("2D Fourier--Chebyshev validation requires CUDA.jl: " *
          sprint(showerror, err))
end

CUDA.functional() || error(
    "2D Fourier--Chebyshev validation requires a functional CUDA device")
CUDA.allowscalar(false)
CUDA.versioninfo()

ENV["TARANG_REQUIRE_CUDA"] = "true"
include(joinpath(@__DIR__, "test_gpu_fc_2d_complete.jl"))

# ── Batched Fourier-mode solve on real hardware ──────────────────────────────
#
# This is the ONLY place `CUBLAS.getrf_strided_batched!` /
# `getrs_strided_batched!` are ever executed. Every local test runs the
# device-generic CPU path: `test_batched_dense_lu.jl` builds plain `Array`s and
# `test_mode_batch_parity.jl` defaults to `device=CPU()`, so merely including
# them on a GPU machine would touch zero CUBLAS code and still report a pass —
# coverage that reads as real and is not. The blocks below reuse the parity
# helper's `device` keyword instead of duplicating the problem setup, so the
# assertions are literally the same ones, driven on device.
using Test
using LinearAlgebra: I
include(joinpath(@__DIR__, "test_mode_batch_parity.jl"))

# `_seed_parity_ic!` writes element-by-element into `get_grid_data(b)`, which
# throws under `CUDA.allowscalar(false)`. Build the same field on the host and
# copy it across in one shot. Seeding is not cosmetic here: with a zero IC and
# an x-independent BC every kx != 0 mode stays exactly zero, so a batch parity
# check would compare zeros across all but one mode and pass trivially.
function _seed_parity_ic_device!(b)
    ensure_layout!(b, :g)
    gd = get_grid_data(b)
    host = Array{eltype(gd)}(undef, size(gd))
    for idx in CartesianIndices(host)
        host[idx] = sin(0.3 * sum(Tuple(idx))) + 0.1 * prod(Tuple(idx)) % 1
    end
    copyto!(gd, host)
    return b
end

@testset "batched mode solve on GPU" begin
    ref_solver, ref_b = _parity_channel_solver(; device=Tarang.GPU(), bc_low="1",
                                                 batched_modes=false)
    bat_solver, bat_b = _parity_channel_solver(; device=Tarang.GPU(), bc_low="1",
                                                 batched_modes=true)

    _seed_parity_ic_device!(ref_b)
    _seed_parity_ic_device!(bat_b)

    step!(ref_solver)
    step!(bat_solver)

    # Load-bearing: without these a regression that silently disables batching
    # on GPU leaves this testset green while exercising nothing new.
    @test !isempty(Tarang.active_mode_batches(bat_solver))   # it really engaged
    @test isempty(Tarang.active_mode_batches(ref_solver))    # and really did not

    for _ in 1:5
        step!(ref_solver)
        step!(bat_solver)
    end
    ensure_layout!(ref_b, :g)
    ensure_layout!(bat_b, :g)
    r = Array(get_grid_data(ref_b))
    b = Array(get_grid_data(bat_b))
    scale = maximum(abs, r)
    @test scale > 1e-8
    @test maximum(abs, b .- r) / scale < 1e-12
end

# Singular-mode detection on device: the `info` array is a DEVICE array and its
# check has never run against real hardware. Assert it raises rather than
# returning buffer contents that read as a plausible solution.
@testset "batched LU singular mode raises on GPU" begin
    A = CUDA.zeros(ComplexF64, 4, 4, 3)
    A[:, :, 1] .= CuArray(Matrix{ComplexF64}(I, 4, 4))
    A[:, :, 3] .= CuArray(Matrix{ComplexF64}(I, 4, 4))
    s = Tarang.BatchedDenseLU(A)                  # mode 2 is exactly singular
    @test_throws Exception Tarang.batched_factor!(s)
end
