"""Adaptive CFL controller and pretty-printing helpers."""

"""
    CFLDiffusivity

One diffusivity registered with [`add_diffusivity!`](@ref) for the *diffusive*
(parabolic) stability limit.

`value` is a constant `Float64`, a per-point array (this rank's slab — e.g.
whatever `get_eddy_viscosity(model)` returns), or a `ScalarField`. Arrays and
fields are stored **by reference**, so an LES model that refreshes νₑ in place
each step is picked up automatically by the next `compute_timestep` without
re-registering.

`domain` supplies the grid spacings. It is resolved lazily inside
`compute_timestep` (falling back to the first registered velocity's domain, then
to the problem domain) so a diffusivity may be registered before any velocity.
"""
struct CFLDiffusivity
    value::Union{Float64, AbstractArray, ScalarField}
    domain::Union{Nothing, Domain}
end

"""
    CFL(solver::InitialValueSolver; kwargs...)

Adaptive timestep controller. Register the fields that constrain the step with
[`add_velocity!`](@ref) / [`add_diffusivity!`](@ref), then call
[`compute_timestep`](@ref) (or hand the controller to `run!(solver; cfl=cfl)`).

!!! warning "Advection only, unless you say otherwise"
    A `CFL` with no registered diffusivity returns a **purely advective**
    timestep, `dt = safety / max(Σᵢ |uᵢ|/Δxᵢ)`. Any diffusion you integrate
    **explicitly** — most importantly an LES eddy viscosity νₑ, which cannot go
    down the implicit path because that path cannot represent a spatially
    varying coefficient — imposes its own `dt ≲ 1/(2 ν Σᵢ Δxᵢ⁻²)` limit that is
    **not** accounted for here. Register it with `add_diffusivity!(cfl, ν)`.
    Diffusion handled implicitly by the timestepper needs no registration.

Keywords: `initial_dt`, `cadence`, `safety`, `threshold`, `max_change`,
`min_change`, `max_dt` — see `compute_timestep` for the hysteresis semantics.
"""
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
    diffusivities::Vector{CFLDiffusivity}
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
                 cadence, VectorField[], CFLDiffusivity[], initial_dt, 0, reducer)
        return cfl
    end
end

"""Add velocity field for CFL calculation"""
function add_velocity!(cfl::CFL, velocity::VectorField)
    push!(cfl.velocities, velocity)
end

"""
    add_diffusivity!(cfl::CFL, ν; domain=nothing) -> cfl

Register a diffusivity `ν` so `compute_timestep` also enforces the **explicit
diffusion** stability limit. Opt-in: a `CFL` with nothing registered here keeps
returning the advection-only step it always did.

`ν` may be

* a `Real` — a constant molecular/eddy diffusivity;
* an `AbstractArray` — per-point diffusivity. Under MPI this is **this rank's
  slab**; the global maximum is taken for you through the same batched
  `Allreduce(MAX)` as the velocity terms, so an LES field can be handed over
  directly: `add_diffusivity!(cfl, get_eddy_viscosity(model))`;
* a `ScalarField` — transformed to grid space on each evaluation.

Arrays and fields are held **by reference**, so a model that overwrites νₑ in
place every step needs to be registered only once.

## Stability limit

With `ν_max` the global maximum of the registered coefficient and `Δxᵢ` the grid
spacing on axis `i` (minimum spacing on a Chebyshev axis), the entry contributes
the frequency

    f_diff = 2 ν_max Σᵢ Δxᵢ⁻²        ⟹        dt ≤ 1 / f_diff

which is the forward-Euler limit for the second-order central Laplacian: its
extreme eigenvalue is `-4ν Σᵢ Δxᵢ⁻²`, and `|1 + λ dt| ≤ 1` gives
`dt ≤ 1/(2ν Σᵢ Δxᵢ⁻²)`. The **sum** over axes (not the max) is the correct
anisotropic form, and it matches the advective term in this file, which likewise
sums `|uᵢ|/Δxᵢ` over axes. On an isotropic `d`-dimensional grid it reduces to the
familiar `dt ≤ Δx²/(2dν)`.

The result is folded into the same `min_dt` reduction as the advective limit, so
the smaller of the two wins, and the `safety` factor applies to both.

## Grid spacings

`domain` defaults to the domain of a `ScalarField` argument, else to the first
registered velocity's domain, else to the problem domain. Pass it explicitly when
a bare array belongs to a grid other than those.

```julia
cfl = CFL(solver; safety=0.4)
add_velocity!(cfl, u)
add_diffusivity!(cfl, nu)                            # constant, explicit ν
add_diffusivity!(cfl, get_eddy_viscosity(les_model)) # LES νₑ, refreshed in place
```

See also [`add_velocity!`](@ref), [`compute_timestep`](@ref).
"""
function add_diffusivity!(cfl::CFL, ν::Real; domain::Union{Nothing, Domain}=nothing)
    push!(cfl.diffusivities, CFLDiffusivity(Float64(ν), domain))
    return cfl
end

function add_diffusivity!(cfl::CFL, ν::AbstractArray; domain::Union{Nothing, Domain}=nothing)
    push!(cfl.diffusivities, CFLDiffusivity(ν, domain))
    return cfl
end

function add_diffusivity!(cfl::CFL, ν::ScalarField; domain::Union{Nothing, Domain}=nothing)
    push!(cfl.diffusivities, CFLDiffusivity(ν, domain === nothing ? ν.domain : domain))
    return cfl
end

"""
LOCAL (per-rank) maximum of a registered diffusivity. Never reduces across
ranks — the caller folds this into the one batched `Allreduce(MAX)`.
"""
_cfl_local_max_diffusivity(ν::Float64) = ν

function _cfl_local_max_diffusivity(ν::AbstractArray)
    # `parent` keeps this LOCAL: maximum(::PencilArray) is itself collective.
    p = parent(ν)
    return isempty(p) ? 0.0 : Float64(maximum(p))
end

function _cfl_local_max_diffusivity(ν::ScalarField)
    ensure_layout!(ν, :g)
    p = parent(get_grid_data(ν))
    return isempty(p) ? 0.0 : Float64(maximum(p))
end

"""
Resolve the domain whose grid spacings bound a registered diffusivity: the
explicitly supplied one, else the first registered velocity's, else the problem's.
"""
function _cfl_diffusivity_domain(cfl::CFL, entry::CFLDiffusivity)
    entry.domain === nothing || return entry.domain

    for velocity in cfl.velocities
        velocity.domain === nothing || return velocity.domain
    end

    problem = cfl.solver.problem
    if hasproperty(problem, :domain) && problem.domain !== nothing
        return problem.domain
    end

    # Fail loudly rather than silently dropping the diffusive limit.
    throw(ArgumentError(
        "add_diffusivity!: cannot determine the grid for this diffusivity — no " *
        "velocity or problem domain is available. Pass one explicitly, e.g. " *
        "`add_diffusivity!(cfl, ν; domain=field.domain)`."))
end

"""
    compute_timestep(cfl::CFL)

Compute the adaptive timestep based on the CFL condition. Returns the new
`dt` to be used by the solver for the next step.

## What is (and is not) limited

Every registered velocity contributes the advective frequency
`max(Σᵢ |uᵢ|/Δxᵢ)`, and every diffusivity registered with
[`add_diffusivity!`](@ref) contributes the diffusive frequency
`2 ν_max Σᵢ Δxᵢ⁻²`. `dt` is `safety / f` for the largest frequency `f` of all of
them, so the tightest limit wins.

!!! warning
    **With no diffusivity registered the returned `dt` accounts for advection
    ONLY** — `dt = safety / max(Σᵢ |uᵢ|/Δxᵢ)`, capped by `max_dt`. Diffusion that
    the timestepper integrates *explicitly* is then completely unconstrained
    here. The usual offender is an LES eddy viscosity νₑ: a spatially varying
    coefficient cannot go down the implicit path, so it is stepped explicitly and
    carries `dt ≲ 1/(2 ν_max Σᵢ Δxᵢ⁻²)` (i.e. `Δx²/(2dν)` on an isotropic
    `d`-dimensional grid). Nothing enforces that unless you call
    `add_diffusivity!(cfl, get_eddy_viscosity(model))`. It bites hardest on a
    Chebyshev axis, where the near-wall spacing is far below `L/N`. Diffusion
    treated implicitly needs no registration.

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

    if isempty(cfl.velocities) && isempty(cfl.diffusivities)
        return cfl.current_dt
    end

    min_dt = Inf

    # Compute each velocity's / diffusivity's LOCAL max stability frequency, then
    # agree on the GLOBAL maxima with ONE batched Allreduce(MAX) instead of one
    # collective per entry. `empty=0.0` matches the prior `global_max(...; empty=0.0)`
    # semantics (an empty local slab contributes 0.0 under MAX). Layout of the
    # buffer: velocities occupy 1:n_vel, diffusivities n_vel+1:n_vel+n_diff.
    n_vel = length(cfl.velocities)
    n_diff = length(cfl.diffusivities)
    local_maxes = fill(0.0, n_vel + n_diff)

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

    # Diffusive (parabolic) limit — opt-in via add_diffusivity!. Forward-Euler
    # stability for the second-order central Laplacian: its extreme eigenvalue is
    # -4ν Σᵢ Δxᵢ⁻², and |1 + λ dt| ≤ 1 gives dt ≤ 1 / (2ν Σᵢ Δxᵢ⁻²). We SUM the
    # inverse-square spacings over axes (the correct anisotropic form; reduces to
    # Δx²/(2dν) on an isotropic d-dimensional grid), mirroring the advective term
    # above, which likewise sums |uᵢ|/Δxᵢ over axes.
    for (k, entry) in enumerate(cfl.diffusivities)
        spacings = grid_spacing(_cfl_diffusivity_domain(cfl, entry))

        inv_dx2_sum = 0.0
        for dx in spacings
            dx > 0 && (inv_dx2_sum += inv(dx * dx))
        end

        # LOCAL max only (arrays are this rank's slab); the Allreduce below makes
        # it global. Grid spacings come from the GLOBAL basis size/bounds, so they
        # are rank-independent and may be applied before the reduction.
        # Clamp at 0: a negative coefficient is anti-diffusion, not a dt limit.
        ν_local = max(0.0, _cfl_local_max_diffusivity(entry.value))
        local_maxes[n_vel + k] = 2.0 * ν_local * inv_dx2_sum
    end

    # Single collective for ALL velocities AND diffusivities (was K separate
    # global_max Allreduces).
    reduce_vector!(cfl.reducer, local_maxes, MPI.MAX)

    for k in eachindex(local_maxes)
        max_frequency = local_maxes[k]
        if max_frequency > 0
            # Advective CFL: dt < 1 / max(sum_i |u_i| / dx_i).
            # Diffusive limit: dt < 1 / (2 ν_max sum_i dx_i^-2).
            # Both are frequencies, so the tightest (largest) one wins.
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
    ndiff = length(cfl.diffusivities)
    diff_str = ndiff == 0 ? "" : ", $ndiff diffusivities"
    print(io, "CFL(dt=$(round(cfl.current_dt; sigdigits=3)), safety=$(cfl.safety), $nvel velocities$diff_str)")
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
    ndiff = length(cfl.diffusivities)
    println(io, _box_text("Velocities:    $nvel registered"))
    println(io, _box_text("Diffusivities: $ndiff registered"))
    if ndiff == 0
        println(io, _box_text("               (advective limit only)"))
    end
    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end
