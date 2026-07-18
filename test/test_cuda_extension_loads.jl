"""
Extension-load smoke test for TarangCUDAExt.

CUDA.jl loads fine on machines WITHOUT an NVIDIA GPU (`CUDA.functional()` is
simply false), so this test runs everywhere and catches the whole class of
load-fatal extension bugs without hardware:
  - same-signature method overwriting between ext and src (precompile error)
  - unqualified definitions of imported functions (load error)
  - include-order UndefVarErrors inside the extension
  - broken hook registration in TarangCUDAExt.__init__

A GPU audit found the extension had NEVER loaded successfully — every GPU test
was gated on `CUDA.functional()` and silently skipped, so nothing caught it.
"""

using Test
using Tarang

const _CUDA_LOADED = try
    @eval using CUDA
    true
catch err
    @info "CUDA.jl unavailable; skipping extension-load smoke test" err
    false
end

@testset "TarangCUDAExt extension loads" begin
    if !_CUDA_LOADED
        @test_skip "CUDA.jl not installable/loadable in this environment"
    else
        ext = Base.get_extension(Tarang, :TarangCUDAExt)
        @test ext !== nothing

        # Dispatch/hook wiring: the ext adds a CuDevice _gpu_device method and a
        # Val{:cuda} _cuda_functional method; transforms are hook-registered.
        @test hasmethod(Tarang._gpu_device, Tuple{Int})
        @test Tarang._GPU_FORWARD_TRANSFORM_HOOK[] !== nothing
        @test Tarang._GPU_BACKWARD_TRANSFORM_HOOK[] !== nothing

        # GPU solver activation (was dead in every load order before the fix)
        @test Tarang.CUDA_AVAILABLE[]
        @test Tarang._CUDA_MOD[] === CUDA
        for name in ("cuda_lu", "cuda_dense", "cuda_cg", "cuda_gmres", "cuda_sparse")
            @test haskey(Tarang.MatSolvers.SOLVER_REGISTRY, name)
        end

        # has_cuda() now reflects CUDA.functional() rather than a stale false
        @test Tarang.has_cuda() == CUDA.functional()
    end
end
