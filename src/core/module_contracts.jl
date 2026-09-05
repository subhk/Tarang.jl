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

# Root of the problem hierarchy (`IVP`, `LBVP`, `NLBVP`, `EVP`, all in
# problems/problem_types.jl). Declared here rather than next to those structs
# because the operator layer loads two stages earlier and needs the name to
# annotate signatures — `symbolic_diff.jl` takes `::Problem`, which is what
# stops a duck-typed stand-in from reaching code that assumes the equation IR.
abstract type Problem end

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

# ---------------------------------------------------------------------------
# BUFFER OWNERSHIP — which returned fields the caller may keep
# ---------------------------------------------------------------------------
#
# Two buffer-recycling designs exist in this package and they have opposite
# safety properties. Know which one you are holding.
#
# 1. `FieldPool` (`core/field_pool.jl`) tracks ownership: a buffer is reissued
#    only after an explicit `return!`, cross-pool returns throw, and
#    `with_pool_field` gives RAII. It is NOT installed by default — see the
#    docstring on `step!` for why — so `checkout_or_alloc` always allocates.
#
# 2. The rotating result pools track nothing. They hand out slot `idx % N` and
#    reissue it after N further checkouts no matter who is still holding it:
#
#      `_DERIV_RESULT_POOL`   (16) — `operators/derivatives/derivatives_eval.jl`
#      `_NL_RESULT_POOL`      ( 8) — `nonlinear/nonlinear_padding.jl`
#      `_POISSON_RESULT_POOL` ( 4) — `timesteppers/state_utils.jl`
#
# THE CONTRACT. A rotating-pool result may be borrowed only by a caller that
# consumes it before returning. Any caller that stores it in a container, returns
# it across an API boundary, or hands it to user code MUST take ownership with
# `_own_borrowed_field`. Growing N is not a fix: nothing bounds how many results
# a caller may hold, so every N is wrong for some caller.
#
# This has been a live wrong-answer bug twice — `grad()` retaining derivative
# slots, and `Base.:*` returning a nonlinear slot to user code — both silent, both
# found only by asserting values. `evaluate_transform_multiply` therefore defaults
# to `own=true` and makes borrowing the explicit opt-in, so a caller that forgets
# gets a slow answer rather than a wrong one. Prefer that direction for any new
# pooled producer. `_spectral_poisson_solve` is the exception that needs no
# ownership call: it has a single consumer that `copy_field_data!`s the result on
# the next line.
#
# `test/test_deriv_pool_ownership.jl`, `test/test_nl_product_ownership.jl` and
# `test/test_buffer_ownership_ratchet.jl` pin all of the above.

# ---------------------------------------------------------------------------
# COLLECTIVE ENTRY POINTS — which calls every rank must make together
# ---------------------------------------------------------------------------
#
# Nothing in a signature says "collective". These are, and nothing else in the
# public API may become one without being added here:
#
#   Distributor(...)                        MPI topology + Comm_split
#   Domain(dist, bases) / plan_transforms!  PencilFFT plan construction
#   ScalarField/VectorField/TensorField constructors on a distributed dist
#   TransposableField(field), transpose_workspace!(dist, field)
#   InitialValueSolver / BoundaryValueSolver construction
#   forward_transform!/backward_transform!, evaluate_rhs, step!, solve!
#   NetCDFFileHandler(...), process!(handler), save_state/load_state!
#   close(dist)
#
# Three rules keep the count of collectives identical on every rank:
#   1. An accessor never plans. `_field_transform_bundle` refuses when a field
#      has no bundle; `local_shape(domain, :c)` derives geometry locally.
#   2. A per-rank decision gates the BODY of a collective, never the call
#      (`create_current_file!(handler; create=needs_create)`).
#   3. Per-rank-fallible I/O is settled with `_collectively`, so a failure on
#      one rank aborts all ranks before the next collective.

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
