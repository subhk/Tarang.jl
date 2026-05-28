# Single-process GPU test driver for CI (JuliaGPU Buildkite).
#
# Runs each file in GPU_TEST_FILES in its own `julia` process (the files are
# standalone scripts that `using CUDA` and self-skip / exit(0) when CUDA is not
# functional, so they must not share a process). Runs them all even if some
# fail, and exits non-zero if any failed.
#
# CUDA must be available in the active project (the Buildkite GPU job adds it
# before invoking this driver, since CUDA is a weak dependency of Tarang).
#
# Usage:
#   julia --project test/run_gpu_ci.jl

include("file_lists.jl")

const PROJECT_DIR = dirname(@__DIR__)
const TEST_DIR = @__DIR__
const JULIA = Base.julia_cmd()

println("="^60)
println("  Tarang.jl single-GPU CI driver")
println("  files : $(length(GPU_TEST_FILES))")
println("="^60)

failed = String[]
for file in GPU_TEST_FILES
    path = joinpath(TEST_DIR, file)
    println("\n>>> $file")
    cmd = `$JULIA --project=$PROJECT_DIR $path`
    try
        run(cmd)
        println("    ✓ $file")
    catch err
        push!(failed, file)
        @error "GPU test failed" file = file exception = err
    end
end

println("\n" * "="^60)
println("  GPU summary: $(length(GPU_TEST_FILES) - length(failed)) passed, $(length(failed)) failed")
println("="^60)

if !isempty(failed)
    error("GPU tests failed: " * join(failed, ", "))
end
