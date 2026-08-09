"""
Subproblem runtime helpers split into focused sub-files:
- subproblem_io.jl: gather/scatter helpers, cached vectors, and space compression
- mode_batch.jl: structural bucketing and the batched per-mode working set
- mode_batch_kernels.jl: KernelAbstractions kernels for the batched gather/scatter/spmv/BC/LHS-assembly ops
- subproblem_rhs.jl: equation-space F gather, BC projections, and BC-array caches
- subproblem_modes.jl: condition parsing and valid-mode selection helpers
"""

include("subproblem_io.jl")
include("mode_batch.jl")
include("mode_batch_kernels.jl")
include("subproblem_rhs.jl")
include("subproblem_modes.jl")
