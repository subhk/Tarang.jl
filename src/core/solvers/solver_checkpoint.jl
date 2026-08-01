# ============================================================================
# Solver checkpoint / restart.
#
# Writes every evolved field in `solver.state` plus the simulation clock, using
# the slab layer, so a checkpoint written on N ranks can be read on M.
#
# Writes `solver.state`, NOT the problem-variable handles: those are separate
# objects and writing to them does not restore the integrator.
# ============================================================================

"""Steps a multistep scheme spends re-seeding after a restart with no stored history.

Enumerated from the timestepper roster in `src/core/timesteppers/types.jl`, not
from the subset that happened to be tested: a scheme missing from this table
restarts with a silently re-seeded history and NO warning, which is exactly the
quiet-wrong-answer failure this repo keeps finding. `_ONE_STEP_SCHEMES` below
records the deliberate exclusions, and `test_checkpoint_restart.jl` asserts that
every concrete `TimeStepper` appears in one table or the other, so a new scheme
cannot be silently omitted.

The count is the number of steps the scheme runs below its nominal order while
it rebuilds the history the checkpoint does not carry.

NOTE ON THE WARNING DEDUP: `_warn_multistep_restart` tags its `@warn` with
`_id=Symbol(:multistep_restart_, scheme)`, and that dedup is PROCESS-global. The
serial suite runs every test file in one process, so a scheme whose warning has
already been asserted once (`test_checkpoint_restart.jl` spends CNAB2/SBDF2/
SBDF3/SBDF4) will emit nothing the second time. A later test re-asserting an
already-spent scheme's warning would observe silence and fail for a reason that
has nothing to do with the code under test — pick an unspent scheme instead."""
const _MULTISTEP_RESEED_STEPS = Dict(
    "CNAB2"              => 1,   # AB2 explicit extrapolation needs F^{n-1}
    "MCNAB2"             => 1,   # theta-method + AB2 explicit: same F^{n-1}
    "CNLF2"              => 1,   # leapfrog: needs X^{n-1}
    "SBDF2"              => 1,
    "SBDF3"              => 2,
    "SBDF4"              => 3,
    "ETD_CNAB2"          => 1,   # AB2 coefficients under an exponential integrator
    "ETD_SBDF2"          => 1,   # BDF2 coefficients under an exponential integrator
    "DiagonalIMEX_SBDF2" => 1,   # its own docstring: "Multi-step method"
)

"""Schemes deliberately absent from `_MULTISTEP_RESEED_STEPS`: they restart exactly.

Two groups. The one-step schemes (every RK variant, including the diagonal-IMEX
and exponential RK ones) carry no history at all. `CNAB1` and `SBDF1` are the
first-order bootstrap formulas of their families: both depend only on the
current state, so there is nothing to re-seed."""
const _ONE_STEP_SCHEMES = Set([
    "RK111", "RK222", "RK443", "RK443_IMEX", "RKSMR", "RKGFY",
    "ETD_RK222", "DiagonalIMEX_RK222", "DiagonalIMEX_RK443",
    "CNAB1", "SBDF1",
])

function _warn_multistep_restart(solver::InitialValueSolver)
    scheme = string(nameof(typeof(solver.timestepper)))
    steps = get(_MULTISTEP_RESEED_STEPS, scheme, 0)
    steps == 0 && return nothing
    # `maxlog` dedups by macro-expansion callsite, not by message content, and
    # there is exactly one @warn callsite here serving every scheme. Without a
    # per-scheme `_id`, the first multistep restart in a process would be the
    # only one that ever warns -- a different scheme, or the same scheme on a
    # different solver, would restart silently for the rest of the session.
    @warn "$scheme restart re-seeds its multistep history: the checkpoint carries the " *
          "state but not the stored time levels, so the first $steps step(s) run at the " *
          "seeding order. The run is correct but not bit-identical to an uninterrupted " *
          "one. One-step schemes (RK111/RK222/RK443, RK443_IMEX, RKSMR, RKGFY, ETD_RK222, " *
          "DiagonalIMEX_RK222/RK443) and the first-order bootstraps CNAB1/SBDF1 restart " *
          "exactly; DiagonalIMEX_SBDF2 does NOT -- it is multistep." maxlog=1 _id=Symbol(:multistep_restart_, scheme)
    return nothing
end

"""
    save_state(solver, path) -> String

Write `solver.state` and the simulation clock to NetCDF. Serial runs produce
`<path>.nc`; under MPI each rank writes `<path>/<name>_p<rank>.nc` with no gather.

Zero-dimensional tau variables are skipped: they carry no spatial data and are
re-solved from the state on the next step.

The whole write path — the stale-file removal, the per-field `write_local_slab`
loop, and the final `ncputatt` — runs inside one `_collectively` region. Every
one of those steps is per-rank fallible (ENOSPC on one node, a corrupt leftover
file), and an unguarded throw would leave that rank unwinding while its peers
walked into the solver's next collective.
"""
function save_state(solver::InitialValueSolver, path::AbstractString)
    isempty(solver.state) && error("save_state: solver.state is empty; nothing to write.")
    dist = solver.state[1].dist
    target = _slab_output_path!(dist, path, "save_state")

    # Transpose EVERY field to grid space before the write loop, not inside it.
    # `ensure_layout!` is collective under MPI, and the guard below wraps the whole
    # loop rather than each iteration: with the transposes interleaved, a per-rank
    # write failure at field k (a full filesystem on one node) would unwind that
    # rank to the guard's Allreduce while its peers entered field k+1's transpose —
    # a hang, not an error. Hoisting leaves the loop body free of collectives, so
    # whole-loop granularity is sound. Every rank runs the same transposes in the
    # same order because `solver.state` is replicated metadata.
    for field in solver.state
        isempty(field.bases) && continue
        ensure_layout!(field, :g)
    end

    return _collectively(dist, "save_state") do
        isfile(target) && rm(target)

        for field in solver.state
            isempty(field.bases) && continue
            global_shape, _, local_start = _field_slab_geometry(field)
            host = Array(get_local_data(get_grid_data(field)))
            write_local_slab(target, field.name, host, local_start, global_shape)
        end

        ncputatt(target, "global", Dict("sim_time" => Float64(solver.sim_time),
                                        "iteration" => Int(solver.iteration),
                                        "dt" => Float64(solver.dt)))
        target
    end
end

"""Clock attributes every checkpoint must carry."""
const _CHECKPOINT_CLOCK_ATTRS = ("sim_time", "iteration", "dt")

"""
    load_state!(solver, path) -> solver

Restore `solver.state` and the simulation clock from a checkpoint written at any
rank count. Discards the timestepper's cached state so the scheme restarts from
the loaded fields.

All three clock attributes are REQUIRED. Treating them as optional made the two
paths that produce a clock-less file — `save_state` dying between its field
writes and its final `ncputatt` (the field writes are the slow part, so that is
the likely crash window), and pointing `load_state!` at `save_field` output,
which writes no global attributes at all and accepts the same path form — return
successfully with the fields correctly restored and the clock silently left at
`sim_time = 0.0, iteration = 0`. The restart then integrates for the wrong
duration, stamps the wrong output timestamps, and evaluates any time-dependent
forcing at the wrong time, all with correct-looking field values.

The whole body is one `_collectively` region (see its docstring for why nesting
it around `load_field!`'s own guard is safe).
"""
function load_state!(solver::InitialValueSolver, path::AbstractString)
    isempty(solver.state) && error("load_state!: solver.state is empty; nothing to restore.")
    dist = solver.state[1].dist

    _collectively(dist, "load_state!") do
        for field in solver.state
            isempty(field.bases) && continue
            load_field!(field, path, field.name)
        end

        src = open_slab_source(path)
        attrs = netcdf_file_info(first(src.files)).gatts

        missing_attrs = filter(k -> !haskey(attrs, k), _CHECKPOINT_CLOCK_ATTRS)
        isempty(missing_attrs) || error(
            "load_state!: '$path' is missing the required clock attribute(s) " *
            "$(join(missing_attrs, ", ")). The fields loaded, but the simulation clock " *
            "cannot be restored, so continuing would integrate for the wrong duration " *
            "with correct-looking field values. Either the checkpoint is incomplete " *
            "(save_state writes the clock attributes LAST, after every field, so a run " *
            "that died mid-write leaves exactly this) or '$path' is save_field output, " *
            "which writes field data with no global attributes at all. Files scanned: " *
            "$(src.files).")

        solver.sim_time = Float64(attrs["sim_time"])
        solver.iteration = Int(attrs["iteration"])
        solver.dt = Float64(attrs["dt"])

        # A checkpoint variable with no matching solver.state field is not an
        # error -- a file may legitimately hold more than this solver needs --
        # but it is worth naming, because the usual cause is a renamed field or
        # a checkpoint from a different problem, and the silent half of that
        # mistake is the state field that then keeps its pre-load values. (The
        # reverse case, a state field with no variable on disk, already errors
        # loudly inside load_field!.)
        restored = Set(field.name for field in solver.state if !isempty(field.bases))
        unused = sort([v for v in keys(src.entries) if !(v in restored)])
        isempty(unused) || @info "load_state!: '$path' also holds variable(s) " *
            "$(join(unused, ", ")) with no matching field in solver.state; they were " *
            "not loaded."
        nothing
    end

    # Drop the cached timestepper state: its history refers to the pre-restart
    # trajectory. `_ensure_timestepper_state!` rebuilds it from the loaded fields.
    solver.timestepper_state = nothing
    _warn_multistep_restart(solver)
    return solver
end
