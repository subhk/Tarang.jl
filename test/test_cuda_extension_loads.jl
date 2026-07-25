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

        # Dispatch wiring: the ext adds a CuDevice _gpu_device method, a
        # Val{:cuda} _cuda_functional method, and the two transform backends.
        # The backends are dispatched (::GPU beats Tarang's
        # ::AbstractArchitecture fallback), so "is the backend wired?" is a
        # method-table question, not a `Ref` that may still hold `nothing`.
        @test hasmethod(Tarang._gpu_device, Tuple{Int})
        for f in (Tarang._gpu_forward_transform_backend!, Tarang._gpu_backward_transform_backend!)
            m = which(f, Tuple{Tarang.GPU, Tarang.ScalarField})
            @test m.module === ext          # the ext's method wins, not the error fallback
        end

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
