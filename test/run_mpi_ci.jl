# Multi-rank MPI test driver for CI.
#
# Launches every file in the selected fileset in its own `mpiexec -n <ranks>`
# world (each test calls MPI.Init itself, so they must not share a process),
# runs them all even if some fail, and exits non-zero if any failed.
#
# Filesets (env TARANG_MPI_FILESET):
#   "mpi"             -> MPI_TEST_FILES             (CPU MPI; GitHub Actions)   [default]
#   "distributed_gpu" -> DISTRIBUTED_GPU_TEST_FILES (CUDA/NCCL; JuliaGPU Buildkite)
#
# Usage:
#   julia --project test/run_mpi_ci.jl [nprocs]
#   TARANG_MPI_NPROCS=2 TARANG_MPI_FILESET=distributed_gpu julia --project test/run_mpi_ci.jl
#
# `nprocs` resolution order: CLI arg > TARANG_MPI_NPROCS env > 4.

using MPI

include("file_lists.jl")

const NPROCS = let
    if !isempty(ARGS)
        parse(Int, ARGS[1])
    else
        parse(Int, get(ENV, "TARANG_MPI_NPROCS", "4"))
    end
end

const FILESET = get(ENV, "TARANG_MPI_FILESET", "mpi")
const FILES = if FILESET == "distributed_gpu"
    DISTRIBUTED_GPU_TEST_FILES
elseif FILESET == "mpi"
    MPI_TEST_FILES
else
    error("Unknown TARANG_MPI_FILESET=$FILESET (expected \"mpi\" or \"distributed_gpu\")")
end

const PROJECT_DIR = dirname(@__DIR__)
const TEST_DIR = @__DIR__
const MPIEXEC = MPI.mpiexec()
const JULIA = Base.julia_cmd()

println("="^60)
println("  Tarang.jl MPI CI driver")
println("  fileset    : $FILESET ($(length(FILES)) files)")
println("  ranks      : $NPROCS")
println("  mpiexec    : $MPIEXEC")
println("="^60)

failed = String[]
for file in FILES
    path = joinpath(TEST_DIR, file)
    println("\n>>> [$NPROCS ranks] $file")
    cmd = `$MPIEXEC -n $NPROCS $JULIA --project=$PROJECT_DIR $path`
    try
        run(cmd)
        println("    ✓ $file")
    catch err
        push!(failed, file)
        @error "MPI test failed" file = file exception = err
    end
end

println("\n" * "="^60)
println("  MPI summary ($FILESET @ $NPROCS ranks): $(length(FILES) - length(failed)) passed, $(length(failed)) failed")
println("="^60)

if !isempty(failed)
    error("MPI tests failed ($FILESET @ $NPROCS ranks): " * join(failed, ", "))
end
