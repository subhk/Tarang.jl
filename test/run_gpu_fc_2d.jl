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
