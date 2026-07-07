"""Adaptive CFL controller and pretty-printing helpers."""

# CFL condition calculation
mutable struct CFL
    solver::InitialValueSolver
    initial_dt::Float64
    safety::Float64
    threshold::Float64
    max_change::Float64
    min_change::Float64
    max_dt::Float64
    cadence::Int

    # State
    velocities::Vector{VectorField}
    current_dt::Float64
    iteration_count::Int
    reducer::GlobalArrayReducer

    function CFL(solver::InitialValueSolver;
                initial_dt::Float64=0.01,
                cadence::Int=1,
                safety::Float64=0.4,
                threshold::Float64=0.1,
                max_change::Float64=2.0,
                min_change::Float64=0.5,
                max_dt::Float64=Inf)

        comm = solver.base !== nothing ? solver_comm(solver.problem) : MPI.COMM_WORLD
        reducer = GlobalArrayReducer(comm)
        cfl = new(solver, initial_dt, safety, threshold, max_change, min_change, max_dt,
                 cadence, VectorField[], initial_dt, 0, reducer)
        return cfl
    end
end

"""Add velocity field for CFL calculation"""
function add_velocity!(cfl::CFL, velocity::VectorField)
    push!(cfl.velocities, velocity)
end

"""
    compute_timestep(cfl::CFL)

Compute the adaptive timestep based on the CFL condition. Returns the new
`dt` to be used by the solver for the next step.

## Sticky-dt hysteresis

Calling the implicit solver with a changed `dt` invalidates the cached LHS
factorization in `step_subproblem_rk!` / `step_subproblem_multistep!` — the
sparse LU has to be rebuilt. For adaptive timestepping, this rebuild can
dominate wall-clock time if `dt` drifts by tiny amounts every CFL
recomputation.

To amortize this cost, we only commit a new `dt` when the proposed value
differs from the current value by more than `cfl.threshold` (relative):

    |proposed_dt − current_dt| / current_dt > threshold  →  commit
    otherwise                                            →  keep current_dt

This lets the simulation ride out several steps at a stable `dt`, reusing
the same LU factorization, until the CFL signal drifts far enough to
matter. Default `threshold = 0.1` (10%) gives a good balance of safety and
throughput; set `threshold = 0.0` to disable hysteresis and commit every
change.

Note: the hysteresis only suppresses *upward* or small bidirectional drift.
A sharp CFL violation (proposed drops by 50%, say) always passes through
because it exceeds the threshold.
"""
function compute_timestep(cfl::CFL)

    cfl.iteration_count += 1

    # Only compute every cadence iterations
    if cfl.iteration_count % cfl.cadence != 0
        return cfl.current_dt
    end

    if isempty(cfl.velocities)
        return cfl.current_dt
    end

    min_dt = Inf

    # Compute each velocity's LOCAL max CFL frequency, then agree on the GLOBAL
    # maxima with ONE batched Allreduce(MAX) instead of one collective per
    # velocity. `empty=0.0` matches the prior `global_max(...; empty=0.0)`
    # semantics (an empty local slab contributes 0.0 under MAX).
    n_vel = length(cfl.velocities)
    local_maxes = fill(0.0, n_vel)

    for (k, velocity) in enumerate(cfl.velocities)
        # Get domain and grid spacing
        domain = velocity.domain
        spacings = grid_spacing(domain)

        cfl_frequency = nothing

        for (i, component) in enumerate(velocity.components)
            ensure_layout!(component, :g)

            component_frequency = abs.(get_grid_data(component)) ./ spacings[i]
            if cfl_frequency === nothing
                cfl_frequency = component_frequency
            else
                cfl_frequency .+= component_frequency
            end
        end

        if cfl_frequency !== nothing
            # LOCAL max via `parent` (matches global_max: maximum(::PencilArray)
            # is itself collective; we want only the per-rank max here).
            pf = parent(cfl_frequency)
            local_maxes[k] = isempty(pf) ? 0.0 : Float64(maximum(pf))
        end
    end

    # Single collective for ALL velocities (was K separate global_max Allreduces).
    reduce_vector!(cfl.reducer, local_maxes, MPI.MAX)

    for k in 1:n_vel
        max_frequency = local_maxes[k]
        if max_frequency > 0
            # Multidimensional advective CFL: dt < 1 / max(sum_i |u_i| / dx_i)
            min_dt = min(min_dt, inv(max_frequency))
        end
    end

    if min_dt == Inf
        proposed_dt = cfl.initial_dt
    else
        proposed_dt = cfl.safety * min_dt
    end

    # Apply max_dt constraint
    proposed_dt = min(proposed_dt, cfl.max_dt)

    # Limit change rate (only after first real CFL computation)
    if cfl.iteration_count > 1 && cfl.current_dt > 0 && cfl.current_dt != cfl.initial_dt
        max_allowed = cfl.current_dt * cfl.max_change
        min_allowed = cfl.current_dt * cfl.min_change
        proposed_dt = clamp(proposed_dt, min_allowed, max_allowed)
    end

    # Sticky-dt hysteresis: only commit changes larger than `threshold`
    # (relative) to avoid invalidating the cached LU factorization on every
    # CFL call. Default threshold = 0.1 (10%); set to 0.0 to disable.
    if cfl.current_dt > 0 && cfl.threshold > 0
        rel_change = abs(proposed_dt - cfl.current_dt) / cfl.current_dt
        if rel_change <= cfl.threshold
            # Change is inside the hysteresis band — stick with current dt.
            return cfl.current_dt
        end
    end

    cfl.current_dt = proposed_dt
    return proposed_dt
end

function Base.show(io::IO, cfl::CFL)
    nvel = length(cfl.velocities)
    print(io, "CFL(dt=$(round(cfl.current_dt; sigdigits=3)), safety=$(cfl.safety), $nvel velocities)")
end

function Base.show(io::IO, ::MIME"text/plain", cfl::CFL)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("CFL Controller"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    println(io, _box_text(@sprintf("Current dt:    %.4e", cfl.current_dt)))
    println(io, _box_text(@sprintf("Safety:        %.2f", cfl.safety)))
    println(io, _box_text(@sprintf("Max dt:        %.4e", cfl.max_dt)))
    println(io, _box_text("Cadence:       $(cfl.cadence)"))
    println(io, _box_text(@sprintf("Max change:    %.2f", cfl.max_change)))
    println(io, _box_text(@sprintf("Min change:    %.2f", cfl.min_change)))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    nvel = length(cfl.velocities)
    println(io, _box_text("Velocities:    $nvel registered"))
    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end
