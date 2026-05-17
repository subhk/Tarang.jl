# Compatibility shim for older load-order references.
#
# The implementation moved to `step_global_matrix.jl`; keep this include target
# so stale precompile caches or intermediate branches that still reference the
# old filename fail gracefully.

if !isdefined(@__MODULE__, :_timestep_fields_vector!)
    include("step_vector_helpers.jl")
end

if !isdefined(@__MODULE__, :step_mcnab2!)
    include("step_global_matrix.jl")
end
