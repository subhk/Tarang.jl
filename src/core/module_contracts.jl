# Abstract contracts used to break circular dependencies between implementation
# files. Concrete types live in their owning files; these shared names let early
# loaded code reference later subsystems without forcing a larger module split.
#
# ON THE SUBMODULE SPLIT THIS FILE USED TO ADVERTISE. The listing here named six
# target submodules (Core / Operators / TransformOps / Solvers / Output / Extras)
# as though the split were pending work. It is not pending — it is blocked, and
# had been for as long as the comment stood, which made the comment worse than
# nothing: it described an architecture the dependency graph forbids.
#
# `src/load_order.jl` alternates owners (core, tools, core, tools, core, tools).
# That alternation is the dependency graph, not a formatting accident: code in
# one owner calls names first defined by another owner several stages later.
# Julia resolves those calls at run time, so the package loads and runs fine and
# the inversion is invisible — until someone actually tries to draw a module
# boundary, at which point `core` turns out to require the output layer.
#
# `test/test_layering.jl` measures this. It derives the stages from
# `load_order.jl` and enumerates every backwards cross-owner call, with the
# reason each survivor is still there. Splitting into submodules means driving
# that list to empty first; the test is the checklist, and the count is a ratchet
# so the list cannot quietly grow again.

abstract type AbstractNonlinearEvaluator end
abstract type AbstractEvaluator end
abstract type AbstractDistributedGPUConfig end
abstract type AbstractTransposeComms end
abstract type AbstractTransposeCounts end
abstract type TimeStepper end
abstract type AbstractTimestepperState end
# Compiled RHS evaluation plan. `LazyRHSPlan` (solvers/lazy_rhs.jl) is the only
# implementation, and it loads long after `InitialValueSolver`, which is why that
# field was typed `::Any` — the odd one out among its siblings, all of which name
# a contract declared here. Every consumer already had to narrow the field with
# `::LazyRHSPlan` by hand; this at least makes assigning something unrelated a
# type error at the assignment rather than a puzzle at the next read.
abstract type AbstractRHSPlan end

# Custom PencilConfig struct for pencil array configuration.
struct PencilConfig{N, M}
    global_shape::NTuple{N, Int}
    mesh::NTuple{M, Int}
    comm::MPI.Comm
    decomp_dims::NTuple{M, Bool}
    dtype::Type

    function PencilConfig(global_shape::NTuple{N, Int}, mesh::NTuple{M, Int};
                         comm::MPI.Comm=MPI.COMM_WORLD,
                         decomp_dims::NTuple{M, Bool}=ntuple(i -> true, M),
                         dtype::Type=Float64) where {N, M}
        new{N, M}(global_shape, mesh, comm, decomp_dims, dtype)
    end
end
