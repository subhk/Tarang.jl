# ── Batched per-mode IMEX Runge-Kutta hot path ───────────────────────────────
#
# Mathematically the same stage loop as `step_subproblem_rk.jl`; the only change
# is that every Fourier mode is a COLUMN of an `(n, nmodes)` matrix instead of
# its own vector, so each operation is one launch rather than `nmodes` of them.
#
# Sign convention, stage formula and BC-override semantics are unchanged — see
# the header of `step_subproblem_rk.jl`. What follows documents only where the
# batched path DIFFERS, because those are the places a silent numerical drift
# could hide.
#
# 1. The stage LHS is factored as a DENSE batched LU (`BatchedDenseLU`) instead
#    of a per-mode sparse LU / Woodbury block solve. Same matrix, different
#    factorization algorithm, so stage solutions agree to roundoff rather than
#    bit-for-bit. This is why the parity test compares to a relative 1e-12 and
#    not to `==`.
#
# 2. The MASS solve (`a_ii == 0` stages and the final update) has TWO paths,
#    and the per-mode one is not dead code. In a tau/BC formulation `M_min` has
#    an all-zero row for every algebraic equation, so it is structurally
#    SINGULAR — measured rank 8 of 10 on the channel problem used by
#    `test_mode_batch_parity.jl`. `_get_or_compute_mass_lu!` therefore falls
#    back to SPQR and the per-mode result is a rank-deficient LEAST-SQUARES
#    solution, which a dense batched LU cannot reproduce and `batched_factor!`
#    would raise on. But the measured `M_min` is also a scaled PARTIAL
#    PERMUTATION, and for that structure the minimum-norm least-squares solution
#    is a permuted scaled copy of the RHS — one kernel, no solver. When
#    `mass_selection_plan` verifies that structure for every mode in the batch,
#    `_batch_mass_solve!` takes `batched_mass_apply!`; when it does not, the
#    same function routes both solves through the same
#    `_get_or_compute_mass_lu!` + `_solve_cached_system!` the per-mode loop
#    uses, one column at a time, exactly as before.
#
# 3. `L*X_i` is formed from a RE-GATHER of the state after the stage scatter,
#    exactly as the per-mode loop does, not from the solve output still sitting
#    in the workspace. `evaluate_rhs_buffered` pulls the state to grid space and
#    the next `coeff_data!` transforms it back, so the two differ by an FFT
#    round trip.
#
# 4. `L` is applied through `L_exp`, never `L_min`. `L_min`'s pattern is not
#    uniform across modes (kx=0 stores fewer nonzeros because ∂xx is the literal
#    zero operator there); `L_exp` carries the same values in `LHS`'s union
#    pattern. See `ModeBatch`.

# ── Per-field gather/scatter plans ───────────────────────────────────────────

"""
    BatchFieldPlan

Where one field's coefficients live, for every mode in a batch at once.

`starts[m]` is mode `m`'s MEASURED linear start offset into the field's local
coefficient array, obtained from `_subproblem_strided_index`. It is stored, not
derived: the subproblem tuple is not ordered by Fourier mode (the channel
problem's subproblem 1 is kx=3), so `starts` is not an arithmetic progression
and `starts[1] + (m-1)*stride` would gather the wrong modes while looking
perfectly regular.

`step`/`len` are shared across modes — every mode selects the same coupled axis
— and that is asserted when the plan is built, not assumed.

`cd_size` is the SHAPE of the coefficient array the offsets were measured
against. It is re-checked on every use: the offsets are linear indices, so a
field whose storage was reshaped or re-pencilled underneath the plan would
gather plausible numbers from the wrong places, and a mismatch here is the
cheapest way to turn that into an error.

Shape and not length, because length is not enough: the offsets were computed
from `strides(cd)`, so a length-preserving reshape (8x9 to 9x8) leaves every
offset in bounds while changing what each one names — a plausible wrong answer
with no error, this repository's dominant historical bug class. The per-mode
path is immune for the same reason stated positively:
`_subproblem_strided_index` memoizes on `hash(size(cd), ...)`, so a reshape
misses the cache and re-measures.
"""
struct BatchFieldPlan
    starts::AbstractVector{Int}
    step::Int
    len::Int
    row_offset::Int
    cd_size::Dims
end

"""
    BatchWorkspace

Step-invariant working set for one `ModeBatch`. Every buffer is fully
reassigned before it is read in a step, so reuse across steps introduces no
stale reference — the same contract the per-mode slot containers in
`step_subproblem_rk.jl` rely on.
"""
mutable struct BatchWorkspace
    n_var::Int
    n_eq::Int
    X0::AbstractMatrix{ComplexF64}      # compressed state at t_n
    MX0::AbstractMatrix{ComplexF64}     # M_min * X0
    RHS::AbstractMatrix{ComplexF64}     # stage RHS
    X::AbstractMatrix{ComplexF64}       # stage solution
    Xg::AbstractMatrix{ComplexF64}      # state RE-GATHERED after the scatter
    ALG_F::AbstractMatrix{ComplexF64}   # algebraic/BC forcing, equation space
    F::Vector{AbstractMatrix{ComplexF64}}    # F_j, one per stage
    LX::Vector{AbstractMatrix{ComplexF64}}   # L*X_j, one per stage
    raw_var::AbstractMatrix{ComplexF64}      # variable space before compression
    raw_eq::AbstractMatrix{ComplexF64}       # equation space before compression
    alg_F_host::Matrix{ComplexF64}           # host staging for ALG_F
    alg_F_col::Vector{ComplexF64}            # host staging, one mode
    alg_F_valid::Bool
    state_plan::Vector{BatchFieldPlan}       # aligned with `state_fields`
    F_plan::Vector{BatchFieldPlan}           # aligned with `F_plan_targets`
    F_plan_targets::Vector{Int}              # indices into the F field list
    F_plan_fields::Vector{Any}               # identity key for the F plan
    F_plan_built::Bool
    mass_ok::Vector{Bool}                    # per mode: last mass solve succeeded
end

"""
    BatchedRKPlan

Everything `step_subproblem_rk_batched!` needs, built once per solver and cached
on `state.timestepper_data`. `leftovers` lists every subproblem index NOT
covered by a batch — including the `M_min === nothing` ones — and those stay on
the per-mode code paths mirrored below.
"""
struct BatchedRKPlan
    batches::Vector{ModeBatch}
    workspaces::Vector{BatchWorkspace}
    leftovers::Vector{Int}
end

# Instrumentation for the lifecycle invariant tested in
# `test_mode_batch_parity.jl`: `batched_factor!` must never run without an
# immediately preceding `batched_assemble_lhs!` over the same buffer, because
# the GPU `getrf_strided_batched!` overwrites `A` with its own LU factors while
# the CPU `lu(view(A,:,:,m))` copies. Code that factors twice without
# re-assembling works on CPU and factors the factors on GPU. Two counter bumps
# per factorization (i.e. per dt change), not per stage.
const BATCH_FACTOR_STATS = (assembles = Ref(0), factors = Ref(0))

"""Reset the assemble/factor counters. Test support; not called by the stepper."""
function reset_batch_factor_stats!()
    BATCH_FACTOR_STATS.assembles[] = 0
    BATCH_FACTOR_STATS.factors[] = 0
    return nothing
end

# ── Plan construction ────────────────────────────────────────────────────────

_batch_like(batch::ModeBatch) = batch.M_exp_nzval

"""
    _batch_field_plan(batch, sps, field, row_offset) -> BatchFieldPlan or nothing

Measure `(start, step, len)` for `field` at every mode in `batch` and check that
the batch really is uniform. Returns `nothing` — decline to batch — when any of
the following does not hold, rather than guessing:

- the field has coefficient storage of a complex element type (the batched
  scatter writes `ComplexF64` straight into it, where `_scatter_field_raw!`
  would take `real` for a real-typed array);
- the selection is expressible as ONE strided run for every mode (≥2 coupled
  axes is not, and `_subproblem_strided_index` says so by returning `nothing`);
- `step` and `len` agree across modes;
- `len` equals `subproblem_field_size`, i.e. the run really is the whole of this
  field's contribution to the flat vector.
"""
function _batch_field_plan(batch::ModeBatch, sps, field, row_offset::Int)
    cd_raw = coeff_data!(field)
    (cd_raw === nothing || isempty(cd_raw)) && return nothing
    cd = _local_coeff_data(cd_raw)
    (cd === nothing || isempty(cd)) && return nothing
    eltype(cd) <: Complex || return nothing

    nmodes = batch.nmodes
    starts = Vector{Int}(undef, nmodes)
    step_ = 0
    len = 0
    for (m, i) in enumerate(batch.sp_indices)
        sp = sps[i]
        sidx = _subproblem_strided_index(cd, field, sp)
        sidx === nothing && return nothing
        s, st, ln = sidx
        if m == 1
            step_ = st
            len = ln
        elseif st != step_ || ln != len
            return nothing
        end
        ln == subproblem_field_size(sp, field) || return nothing
        (s >= 1 && s + (ln - 1) * st <= length(cd)) || return nothing
        starts[m] = s
    end

    dev_starts = similar(_batch_like(batch), Int, nmodes)
    copyto!(dev_starts, starts)
    return BatchFieldPlan(dev_starts, step_, len, row_offset, size(cd))
end

"""Resolve a field's local coefficient array and check it still matches its plan."""
@inline function _plan_coeff_data(field, p::BatchFieldPlan)
    cd = _local_coeff_data(coeff_data!(field))
    size(cd) == p.cd_size || error(
        "step_subproblem_rk_batched!: coefficient storage for a batched field " *
        "changed shape ($(size(cd)) now, $(p.cd_size) when the gather plan " *
        "was measured). The plan's linear offsets are no longer valid.")
    return cd
end

"""
    _batch_state_plan(batch, sps, state_fields) -> Vector{BatchFieldPlan} or nothing

Plans for the flat state gather, in `state_fields` order, mirroring the offset
walk in `_gather_subproblem_raw!`.
"""
function _batch_state_plan(batch::ModeBatch, sps, state_fields)
    plans = BatchFieldPlan[]
    offset = 0
    for field in state_fields
        p = _batch_field_plan(batch, sps, field, offset)
        p === nothing && return nothing
        push!(plans, p)
        offset += p.len
    end
    return plans
end

"""
    _batch_workspace(batch, sps, state_fields) -> BatchWorkspace or nothing

Allocate the batch's working set, or decline (returning `nothing`) when the
batch does not satisfy what the batched loop assumes: a square `M_min` matching
the dense LHS order, projections whose shapes line up with the raw gather sizes,
and a state gather expressible as one strided run per field per mode.
"""
function _batch_workspace(batch::ModeBatch, sps, state_fields)
    n = batch.n
    nmodes = batch.nmodes
    sp1 = sps[batch.sp_indices[1]]
    size(sp1.M_min) == (n, n) || return nothing

    plans = _batch_state_plan(batch, sps, state_fields)
    plans === nothing && return nothing

    n_raw_var = isempty(plans) ? 0 : plans[end].row_offset + plans[end].len
    n_raw_eq = _subproblem_raw_eqn_size(sp1)

    # Every mode must agree on the equation-space packing too, or the shared
    # `raw_eq` layout would mean different things in different columns.
    eqn_sizes1 = _subproblem_eqn_sizes(sp1)
    for i in batch.sp_indices
        sp = sps[i]
        size(sp.M_min) == (n, n) || return nothing
        _subproblem_raw_eqn_size(sp) == n_raw_eq || return nothing
        _subproblem_eqn_sizes(sp) == eqn_sizes1 || return nothing
    end

    cv = batch.compress_var
    ev = batch.expand_var
    ce = batch.compress_eqn
    n_var = cv === nothing ? n_raw_var : cv.nrows
    n_var == n || return nothing
    cv === nothing || cv.ncols == n_raw_var || return nothing
    if ev === nothing
        n_raw_var == n || return nothing
    else
        (ev.nrows == n_raw_var && ev.ncols == n) || return nothing
    end
    if ce === nothing
        n_raw_eq == n || return nothing
    else
        (ce.nrows == n && ce.ncols == n_raw_eq) || return nothing
    end

    like = _batch_like(batch)
    mat(rows) = similar_zeros(like, ComplexF64, rows, nmodes)
    stages_placeholder = AbstractMatrix{ComplexF64}[]

    return BatchWorkspace(n, n,
                          mat(n), mat(n), mat(n), mat(n), mat(n), mat(n),
                          stages_placeholder, AbstractMatrix{ComplexF64}[],
                          mat(n_raw_var), mat(n_raw_eq),
                          Matrix{ComplexF64}(undef, n, nmodes),
                          Vector{ComplexF64}(undef, n),
                          false,
                          plans, BatchFieldPlan[], Int[], Any[], false,
                          fill(true, nmodes))
end

"""Grow the per-stage `F`/`LX` matrices to `stages` entries (once per solver)."""
function _batch_stage_buffers!(ws::BatchWorkspace, batch::ModeBatch, stages::Int)
    length(ws.F) == stages && return ws
    like = _batch_like(batch)
    ws.F = AbstractMatrix{ComplexF64}[similar_zeros(like, ComplexF64, ws.n_eq,
                                                    batch.nmodes)
                                      for _ in 1:stages]
    ws.LX = AbstractMatrix{ComplexF64}[similar_zeros(like, ComplexF64, ws.n_eq,
                                                     batch.nmodes)
                                       for _ in 1:stages]
    return ws
end

"""
    _build_batched_rk_plan(solver, subproblems, state_fields; batches=nothing)
        -> BatchedRKPlan or nothing

Bucket the subproblems, build a batch for every bucket that
`should_batch_modes` accepts, and allocate each batch's working set. Batches
whose working set cannot be built are dropped back to the per-mode path rather
than approximated.

`batches` overrides the bucketing; it exists so tests can pin the
partly-batched configuration (some modes batched, the rest leftover) that a
naturally-uniform problem never produces on its own.
"""
function _build_batched_rk_plan(solver, subproblems, state_fields;
                                batches::Union{Nothing, Vector{ModeBatch}}=nothing)
    n_sp = length(subproblems)
    n_sp == 0 && return nothing

    built = if batches === nothing
        sp1 = nothing
        for sp in subproblems
            if sp.M_min !== nothing
                sp1 = sp
                break
            end
        end
        sp1 === nothing && return nothing
        dist = sp1.dist
        nprocs = dist === nothing ? 1 : dist.size
        is_gpu = _gpu_subproblem_execution(sp1)

        # Short-circuit the three gates that do not depend on a bucket, BEFORE
        # bucketing. `should_batch_modes` remains the authority and re-checks
        # all three per bucket; this only avoids hashing every subproblem's
        # sparsity pattern on runs that provably cannot batch — which is every
        # MPI run, every default CPU run, and every problem with more than one
        # Fourier axis, the last of which in 3-D means Nx·Ny subproblems hashed
        # for nothing.
        nprocs == 1 || return nothing
        setting = solver.base.batched_modes
        (setting === nothing ? is_gpu : setting) || return nothing
        # Exactly one Fourier axis: the batched path's declared 2-D scope. See
        # condition 4 of `should_batch_modes`.
        _mode_batch_fourier_axes(sp1) == 1 || return nothing

        like = zeros(sp1.dist.architecture, ComplexF64, 0)
        build_mode_batches!(solver.base, subproblems; is_gpu=is_gpu,
                            nprocs=nprocs, like=like)
    else
        batches
    end
    isempty(built) && return nothing

    kept = ModeBatch[]
    workspaces = BatchWorkspace[]
    for batch in built
        ws = _batch_workspace(batch, subproblems, state_fields)
        if ws === nothing
            @info("Batched mode solve declined for the bucket starting at " *
                  "subproblem $(batch.sp_indices[1]): its gather layout or " *
                  "matrix shapes do not satisfy the batched contract. Those " *
                  "modes stay on the per-mode loop.", maxlog=1)
            continue
        end
        push!(kept, batch)
        push!(workspaces, ws)
    end
    isempty(kept) && return nothing

    covered = falses(n_sp)
    for batch in kept, i in batch.sp_indices
        covered[i] = true
    end
    leftovers = [i for i in 1:n_sp if !covered[i]]

    return BatchedRKPlan(kept, workspaces, leftovers)
end

"""
Fetch (or build once) the batched plan for this solver's subproblem tuple.

The plan and its cache key live under SEPARATE `timestepper_data` entries so
that (a) a cached `nothing` — the normal CPU-default outcome — is
distinguishable from "not yet computed", and (b) the per-step check on the
default path is one Dict lookup plus a pointer comparison, with no tuple
indexing off an `Any` to box. `step_subproblem_rk!` runs this on EVERY step,
including every step that will never batch.
"""
function _sp_rk_batch_plan!(state::TimestepperState, solver, subproblems,
                            state_fields)
    td = state.timestepper_data
    if get(td, :_sp_rk_mode_batches_key, nothing) === subproblems
        return get(td, :_sp_rk_mode_batches, nothing)::Union{Nothing, BatchedRKPlan}
    end
    plan = _build_batched_rk_plan(solver, subproblems, state_fields)
    td[:_sp_rk_mode_batches] = plan
    td[:_sp_rk_mode_batches_key] = subproblems
    return plan
end

"""
    active_mode_batches(solver) -> Vector{ModeBatch}

The `ModeBatch`es this solver's RK stepper is actually running on, empty when
batching declined or has not been reached yet. Lets a caller check ENGAGEMENT,
which a numerical parity test cannot: a batched solver that quietly fell back to
the per-mode loop passes parity trivially.
"""
function active_mode_batches(solver)
    state = solver.timestepper_state
    state === nothing && return ModeBatch[]
    plan = get(state.timestepper_data, :_sp_rk_mode_batches, nothing)
    plan === nothing && return ModeBatch[]
    return (plan::BatchedRKPlan).batches
end

# ── Batched primitives built on the Task 3 kernels ───────────────────────────

# Concrete-typed AXPY barrier over `(n, nmodes)` matrices — the matrix analogue
# of `_sp_axpy!`. The workspace matrices are declared `AbstractMatrix` so a
# CuArray fits the same slot, and a bare broadcast over that abstract eltype
# dynamic-dispatches getindex/setindex! per element. For a `-=` site pass `-a`:
# `dest[i] += (-a)*src[i]` is IEEE-identical to `dest[i] -= a*src[i]`.
@inline function _batch_axpy!(dest::AbstractMatrix, a::Number, src::AbstractMatrix)
    if dest isa Matrix{ComplexF64} && src isa Matrix{ComplexF64}
        @inbounds @simd for i in eachindex(dest, src)
            dest[i] += a * src[i]
        end
    else
        dest .+= a .* src
    end
    return dest
end

"""Apply one `BatchedSparseOp` to every mode, or copy when there is none."""
function _batch_apply!(dest::AbstractMatrix{ComplexF64},
                       op::Union{Nothing, BatchedSparseOp},
                       src::AbstractMatrix{ComplexF64})
    if op === nothing
        copyto!(dest, src)
    else
        batched_spmv!(dest, op.rowptr, op.colval, op.nzval, src)
    end
    return dest
end

"""Gather every state field's strided run into `raw`, then compress into `dest`."""
function _batched_gather_state!(dest::AbstractMatrix{ComplexF64},
                                ws::BatchWorkspace, batch::ModeBatch,
                                state_fields)
    length(state_fields) == length(ws.state_plan) || error(
        "step_subproblem_rk_batched!: the state-field list changed length " *
        "under a live gather plan.")
    raw = ws.raw_var
    fill!(raw, zero(ComplexF64))
    @inbounds for k in eachindex(state_fields)
        p = ws.state_plan[k]
        cd = _plan_coeff_data(state_fields[k], p)
        batched_gather!(raw, cd, p.starts, p.step, p.len, p.row_offset)
    end
    return _batch_apply!(dest, batch.compress_var, raw)
end

"""Expand `X` into raw variable space and scatter it back into the fields."""
function _batched_scatter_state!(ws::BatchWorkspace, batch::ModeBatch,
                                 state_fields, X::AbstractMatrix{ComplexF64})
    length(state_fields) == length(ws.state_plan) || error(
        "step_subproblem_rk_batched!: the state-field list changed length " *
        "under a live scatter plan.")
    raw = ws.raw_var
    _batch_apply!(raw, batch.expand_var, X)
    @inbounds for k in eachindex(state_fields)
        p = ws.state_plan[k]
        cd = _plan_coeff_data(state_fields[k], p)
        batched_scatter!(cd, raw, p.starts, p.step, p.len, p.row_offset)
    end
    return nothing
end

"""
    _batch_build_F_plan!(ws, batch, sps, F_fields, state_fields) -> Bool

Mirror of `gather_eqn_F!`'s raw walk: for each equation carrying a time
derivative, place each target field's run at the running offset; every other row
stays zero. Built once and re-validated by field identity, because the buffered
RHS reuses the same field objects across stages.
"""
function _batch_build_F_plan!(ws::BatchWorkspace, batch::ModeBatch, sps,
                              F_fields, state_fields)
    if ws.F_plan_built && length(ws.F_plan_fields) == length(ws.F_plan_targets)
        ok = true
        @inbounds for j in eachindex(ws.F_plan_targets)
            t = ws.F_plan_targets[j]
            if t > length(F_fields) || F_fields[t] !== ws.F_plan_fields[j]
                ok = false
                break
            end
        end
        ok && return true
    end

    sp1 = sps[batch.sp_indices[1]]
    eqn_sizes = _subproblem_eqn_sizes(sp1)
    targets = _subproblem_eqn_targets(sp1, state_fields)

    plans = BatchFieldPlan[]
    tidxs = Int[]
    flds = Any[]
    i0 = 0
    for eq_idx in eachindex(eqn_sizes)
        eq_size = eqn_sizes[eq_idx]
        eq_size == 0 && continue
        offset = i0
        for tidx in targets[eq_idx]
            if tidx >= 1 && tidx <= length(F_fields) && F_fields[tidx] !== nothing
                p = _batch_field_plan(batch, sps, F_fields[tidx], offset)
                p === nothing && return false
                push!(plans, p)
                push!(tidxs, tidx)
                push!(flds, F_fields[tidx])
                offset += p.len
            elseif tidx >= 1 && tidx <= length(state_fields)
                offset += subproblem_field_size(sp1, state_fields[tidx])
            end
        end
        i0 += eq_size
    end

    ws.F_plan = plans
    ws.F_plan_targets = tidxs
    ws.F_plan_fields = flds
    ws.F_plan_built = true
    return true
end

"""Batched `gather_eqn_F!`: pack PDE F values into equation space, then compress."""
function _batched_gather_eqn_F!(dest::AbstractMatrix{ComplexF64},
                                ws::BatchWorkspace, batch::ModeBatch, sps,
                                F_fields, state_fields)
    _batch_build_F_plan!(ws, batch, sps, F_fields, state_fields) || error(
        "step_subproblem_rk_batched!: the explicit-RHS fields do not admit the " *
        "strided batched gather that the state fields did. This cannot be " *
        "recovered from mid-step; rerun with `batched_modes=false`.")

    raw = ws.raw_eq
    fill!(raw, zero(ComplexF64))
    @inbounds for j in eachindex(ws.F_plan)
        p = ws.F_plan[j]
        cd = _plan_coeff_data(F_fields[ws.F_plan_targets[j]], p)
        batched_gather!(raw, cd, p.starts, p.step, p.len, p.row_offset)
    end
    return _batch_apply!(dest, batch.compress_eqn, raw)
end

"""
Batched `gather_alg_F!`. Built on the HOST, one mode at a time, then uploaded
once — the same shape `gather_alg_F!` itself uses, because the BC coefficient
writes are scalar and must stay off device arrays. Runs once per run for static
BCs; every stage when the problem has time-dependent BCs, exactly as the
per-mode loop does.
"""
function _batched_gather_alg_F!(ws::BatchWorkspace, batch::ModeBatch, sps)
    host = ws.alg_F_host
    col = ws.alg_F_col
    for (m, i) in enumerate(batch.sp_indices)
        gather_alg_F!(col, sps[i])
        @inbounds @views host[:, m] .= col
    end
    copyto!(ws.ALG_F, host)
    ws.alg_F_valid = true
    return ws.ALG_F
end

"""
    _ensure_batch_factored!(batch, dt, a_ii) -> BatchedDenseLU

Assemble every mode's dense stage LHS and factor it, reusing the existing
factorization when `(dt, a_ii)` is unchanged AND the batch is not dirty. Both
conditions, not either: a bare flag with no key silently survives a dt change,
and a bare key with no flag silently survives a buffer that was overwritten
underneath it.

**This is the only place `batched_factor!` may be called from.** The two calls
are adjacent on purpose: `getrf_strided_batched!` overwrites `A` with the LU
factors on GPU while the CPU `lu(view(A,:,:,m))` copies, so factoring a buffer
that was not just re-assembled works on CPU and factors the factors on GPU —
a plausible wrong answer with no error.
"""
function _ensure_batch_factored!(batch::ModeBatch, dt::Float64, a_ii::Float64)
    key = (dt, a_ii)
    cached = batch.lu[]
    if cached !== nothing && !batch.dirty[] && batch.factored_key[] == key
        return cached
    end

    batched_assemble_lhs!(batch.lhs_dense, batch.lhs_colptr, batch.lhs_rowval,
                          batch.M_exp_nzval, batch.L_exp_nzval,
                          ComplexF64(dt * a_ii))
    BATCH_FACTOR_STATS.assembles[] += 1

    s = cached === nothing ? BatchedDenseLU(batch.lhs_dense) : cached
    batched_factor!(s)
    BATCH_FACTOR_STATS.factors[] += 1

    batch.lu[] = s
    batch.factored_key[] = key
    batch.dirty[] = false
    return s
end

"""
    _batch_mass_solve!(X, RHS, batch, sps; skip_missing) -> Vector{Bool}

Solve `M_min * X[:, m] = RHS[:, m]` for every mode, batched when `M_min` is a
scaled partial permutation and per-mode otherwise.

**Fast path.** `batch.mass_src !== nothing` means `mass_selection_plan` verified
at build time, for EVERY mode, that `M_min` selects and scales — so the
minimum-norm least-squares solution is `X[j, m] = RHS[src[j], m] / scale[j]`
with zeros on the null columns, for any RHS. One kernel replaces `nmodes` sparse
solves. This is what makes the mass solve mode-count-independent; see
`mass_selection_plan` for why the shortcut is legal only for that structure.

**Slow path.** Otherwise, one mode at a time through the SAME cached solver the
per-mode loop uses (see note 2 in this file's header). Columns are staged
through the subproblem's own cached vectors rather than passed as
`view(X, :, m)`: the GPU sparse solvers take device vectors they own, and a
strided view into a batch matrix is not that.

Returns the per-mode success mask. `skip_missing=false` reproduces the
`a_ii == 0` stage's behaviour of falling back to `x = rhs` when no mass solver
could be built, `true` reproduces the final update's `continue`. The fast path
sets the whole mask true and ignores `skip_missing`: it cannot fail per-mode,
because its validity was established for every mode before the batch existed.
`fill!` and not "leave it alone" — callers read the mask (`all(ws.mass_ok)`, and
`ws.mass_ok[m] || continue` in `_scatter_solved_modes!`), so inheriting a stale
`false` would silently skip that mode's scatter and freeze part of the state.
"""
function _batch_mass_solve!(X::AbstractMatrix{ComplexF64},
                            RHS::AbstractMatrix{ComplexF64},
                            batch::ModeBatch, sps, ws::BatchWorkspace;
                            skip_missing::Bool)
    src = batch.mass_src
    scale = batch.mass_scale
    if src !== nothing && scale !== nothing
        # `batched_mass_apply!` is permutation-like: `X[j]` reads `B[src[j]]`,
        # a slot another workitem may already have written. Distinct buffers are
        # the caller's contract, so check rather than assume — `ws.X` and
        # `ws.RHS` are separate allocations today and this is what keeps them so.
        X === RHS && error(
            "_batch_mass_solve!: the batched mass apply cannot run in place — " *
            "X and RHS are the same buffer, and the kernel reads permuted rows " *
            "of RHS while writing X.")
        batched_mass_apply!(X, RHS, src, scale)
        fill!(ws.mass_ok, true)
        return ws.mass_ok
    end

    for (m, i) in enumerate(batch.sp_indices)
        sp = sps[i]
        n_eq = size(sp.M_min, 1)
        n_var = size(sp.M_min, 2)
        rhs_v = _sp_stage_vector!(sp, :batch_mass_rhs, n_eq)
        copyto!(rhs_v, view(RHS, :, m))
        M_lu = _get_or_compute_mass_lu!(sp)
        if M_lu !== nothing
            x_v = _sp_stage_vector!(sp, :batch_mass_sol, n_var)
            _solve_cached_system!(x_v, M_lu, rhs_v)
            copyto!(view(X, :, m), x_v)
            ws.mass_ok[m] = true
        elseif skip_missing
            ws.mass_ok[m] = false
        else
            copyto!(view(X, :, m), rhs_v)
            ws.mass_ok[m] = true
        end
    end
    return ws.mass_ok
end

# ── Per-mode mirrors, for the subproblems no batch covers ────────────────────
#
# These reproduce `step_subproblem_rk.jl`'s per-mode blocks verbatim so a bucket
# that declines to batch (one lone mode, a workspace over the byte cap, a shape
# the gather plan rejects) keeps stepping. They are only reachable when at least
# one OTHER bucket did batch; with no batches at all `step_subproblem_rk!`
# never enters this file. Kept separate from the originals so a change here can
# never alter the default path.

function _leftover_prestage!(sp, sp_idx, MX0, RHS, ALG_F, state_fields,
                             bc_dynamic::Bool)
    if sp.M_min === nothing
        MX0[sp_idx] = ComplexF64[]
        RHS[sp_idx] = ComplexF64[]
        ALG_F[sp_idx] = ComplexF64[]
        return nothing
    end
    n_eq = size(sp.M_min, 1)
    n_var = size(sp.M_min, 2)
    mx0 = _sp_stage_vector!(sp, :mx0, n_eq)
    rhs = _sp_stage_vector!(sp, :rhs, n_var)
    x0_pre = gather_inputs!(rhs, sp, state_fields)
    _apply_subproblem_operator!(mx0, _subproblem_operator(sp, :M, x0_pre), x0_pre)
    MX0[sp_idx] = mx0
    RHS[sp_idx] = rhs

    alg_f = _sp_stage_vector!(sp, :alg_f, n_eq, mx0)
    if bc_dynamic || sp.runtime.alg_F_gathered_into !== alg_f
        gather_alg_F!(alg_f, sp)
    end
    ALG_F[sp_idx] = alg_f
    return nothing
end

function _leftover_stage_solve!(sp, sp_idx, i::Int, dt::Float64, A_exp, A_imp,
                                MX0, RHS, F, LX, ALG_F, state_fields)
    rhs = RHS[sp_idx]
    _assign_to_buffer!(rhs, MX0[sp_idx])

    for j in 1:(i - 1)
        a_ej = dt * A_exp[i, j]
        a_ij = dt * A_imp[i, j]
        if abs(a_ej) > 1e-14
            _sp_axpy!(rhs, a_ej, F[j][sp_idx])
        end
        if abs(a_ij) > 1e-14
            _sp_axpy!(rhs, -a_ij, LX[j][sp_idx])
        end
    end

    a_ii = A_imp[i, i]
    if abs(a_ii) > 1e-14
        apply_bc_override!(rhs, ALG_F[sp_idx], sp, dt * a_ii)
    end

    x_sol = _sp_stage_vector!(sp, :sol_stage, i, size(sp.M_min, 2), RHS[sp_idx])
    if abs(a_ii) < 1e-14
        M_lu = _get_or_compute_mass_lu!(sp)
        if M_lu !== nothing
            _solve_cached_system!(x_sol, M_lu, rhs)
        else
            _assign_to_buffer!(x_sol, rhs)
        end
    else
        lhs_solver = _get_or_build_lhs!(sp, i, dt, a_ii)
        if lhs_solver !== nothing
            _solve_cached_system!(x_sol, lhs_solver, rhs)
        else
            @warn "step_subproblem_rk_batched!: LHS factorization failed for sp group=$(sp.group), stage=$i; using rhs as fallback" maxlog=1
            _assign_to_buffer!(x_sol, rhs)
        end
    end

    scatter_inputs(sp, x_sol, state_fields)
    return nothing
end

function _leftover_stage_post!(sp, sp_idx, i::Int, solver, F_fields, state_fields,
                               MX0, RHS, F, LX)
    f_stage = _sp_stage_vector!(sp, :F_stage, i, size(sp.M_min, 1), MX0[sp_idx])
    x_pre = RHS[sp_idx]
    gather_eqn_F!(f_stage, sp, solver, F_fields, state_fields)
    gather_inputs!(x_pre, sp, state_fields)
    L_op = _subproblem_operator(sp, :L, x_pre)
    lx_stage = _sp_stage_vector!(sp, :LX_stage, i, length(x_pre), x_pre)
    _apply_subproblem_operator!(lx_stage, L_op, x_pre)
    F[i][sp_idx] = f_stage
    LX[i][sp_idx] = lx_stage
    return nothing
end

function _leftover_final!(sp, sp_idx, dt::Float64, stages::Int, b_exp, b_imp,
                          MX0, RHS, F, LX, state_fields)
    rhs = _sp_stage_vector!(sp, :final_rhs, size(sp.M_min, 1), MX0[sp_idx])
    copyto!(rhs, MX0[sp_idx])
    for s in 1:stages
        be = dt * b_exp[s]
        bi = dt * b_imp[s]
        if abs(be) > 1e-14
            _sp_axpy!(rhs, be, F[s][sp_idx])
        end
        if abs(bi) > 1e-14
            _sp_axpy!(rhs, -bi, LX[s][sp_idx])
        end
    end

    M_lu = _get_or_compute_mass_lu!(sp)
    M_lu === nothing && return nothing
    x_sol = _sp_stage_vector!(sp, :sol_final, size(sp.M_min, 2), RHS[sp_idx])
    _solve_cached_system!(x_sol, M_lu, rhs)
    scatter_inputs(sp, x_sol, state_fields)
    return nothing
end

# ── The batched step ─────────────────────────────────────────────────────────

"""
    step_subproblem_rk_batched!(solver, state, dt, t, plan, subproblems,
                                state_fields, dist, ts)

One IMEX RK step with every batched bucket solved as `(n, nmodes)` matrices and
`plan.leftovers` on the per-mode mirrors above.

Called only from `step_subproblem_rk!`, and only after that function's preamble
(state-field layout, dt-change invalidation) has run — the solve-layout bracket
below assumes the state is in coefficient space on entry, exactly as the
per-mode loop does.
"""
function step_subproblem_rk_batched!(solver::InitialValueSolver,
                                     state::TimestepperState, dt::Float64,
                                     t::Float64, plan::BatchedRKPlan,
                                     subproblems::Tuple, state_fields, dist,
                                     ts::TimeStepper)
    stages = ts.stages
    A_exp = ts.A_explicit
    A_imp = ts.A_implicit
    c = ts.c_explicit
    b_exp = ts.b_explicit
    b_imp = ts.b_implicit

    batches = plan.batches
    workspaces = plan.workspaces
    leftovers = plan.leftovers
    problem = solver.problem

    for k in eachindex(batches)
        _batch_stage_buffers!(workspaces[k], batches[k], stages)
    end

    n_sp = length(subproblems)
    MX0 = _sp_slots!(state, :_sp_rk_MX0, n_sp)
    F = _sp_stage_slots!(state, :_sp_rk_F, stages, n_sp)
    LX = _sp_stage_slots!(state, :_sp_rk_LX, stages, n_sp)
    RHS = _sp_slots!(state, :_sp_rk_RHS, n_sp)
    ALG_F = _sp_slots!(state, :_sp_rk_ALG_F, n_sp)

    bc_dynamic = has_time_dependent_bcs(problem.bc_manager)

    # ── Pre-stage: M*X_n, batched ────────────────────────────────────────
    solve_stash = to_solve_layout!(state_fields, dist; fuse_from_grid=true)
    for k in eachindex(batches)
        batch = batches[k]
        ws = workspaces[k]
        _batched_gather_state!(ws.X0, ws, batch, state_fields)
        batched_spmv!(ws.MX0, batch.M_min_rowptr, batch.M_min_colval,
                      batch.M_min_nzval, ws.X0)
        if bc_dynamic || !ws.alg_F_valid
            _batched_gather_alg_F!(ws, batch, subproblems)
        end
    end
    for sp_idx in leftovers
        _leftover_prestage!(subproblems[sp_idx], sp_idx, MX0, RHS, ALG_F,
                            state_fields, bc_dynamic)
    end

    # ── Stage loop ───────────────────────────────────────────────────────
    for i in 1:stages
        state.current_substep = i

        if bc_dynamic
            stage_time = t + dt * c[i]
            if _refresh_bcs_for_stage!(solver, stage_time)
                for k in eachindex(batches)
                    _batched_gather_alg_F!(workspaces[k], batches[k], subproblems)
                end
                for sp_idx in leftovers
                    sp = subproblems[sp_idx]
                    sp.M_min === nothing && continue
                    gather_alg_F!(ALG_F[sp_idx], sp)
                end
            end
        end

        a_ii = A_imp[i, i]
        for k in eachindex(batches)
            batch = batches[k]
            ws = workspaces[k]

            copyto!(ws.RHS, ws.MX0)
            for j in 1:(i - 1)
                a_ej = dt * A_exp[i, j]
                a_ij = dt * A_imp[i, j]
                if abs(a_ej) > 1e-14
                    _batch_axpy!(ws.RHS, a_ej, ws.F[j])
                end
                if abs(a_ij) > 1e-14
                    _batch_axpy!(ws.RHS, -a_ij, ws.LX[j])
                end
            end

            if abs(a_ii) > 1e-14
                batched_bc_override!(ws.RHS, ws.ALG_F, batch.bc_rows, dt * a_ii)
                s = _ensure_batch_factored!(batch, dt, a_ii)
                batched_solve!(ws.X, s, ws.RHS)
            else
                _batch_mass_solve!(ws.X, ws.RHS, batch, subproblems, ws;
                                   skip_missing=false)
            end

            _batched_scatter_state!(ws, batch, state_fields, ws.X)
        end
        for sp_idx in leftovers
            sp = subproblems[sp_idx]
            sp.M_min === nothing && continue
            _leftover_stage_solve!(sp, sp_idx, i, dt, A_exp, A_imp, MX0, RHS, F,
                                   LX, ALG_F, state_fields)
        end

        from_solve_layout!(solve_stash, dist; to_grid=true)

        F_fields = evaluate_rhs_buffered(solver, state_fields, t + dt * c[i])

        solve_stash = to_solve_layout!(state_fields, dist; fuse_from_grid=true)
        _fg_F_stash = to_solve_layout!(F_fields, dist)
        for k in eachindex(batches)
            batch = batches[k]
            ws = workspaces[k]
            _batched_gather_eqn_F!(ws.F[i], ws, batch, subproblems, F_fields,
                                   state_fields)
            # RE-GATHER, not `ws.X`: see note 3 in this file's header.
            _batched_gather_state!(ws.Xg, ws, batch, state_fields)
            batched_spmv!(ws.LX[i], batch.L_rowptr, batch.L_colval,
                          batch.L_nzval, ws.Xg)
        end
        for sp_idx in leftovers
            sp = subproblems[sp_idx]
            sp.M_min === nothing && continue
            _leftover_stage_post!(sp, sp_idx, i, solver, F_fields, state_fields,
                                  MX0, RHS, F, LX)
        end

        for (f, fft_pa) in _fg_F_stash
            set_coeff_data!(f, fft_pa)
        end
        _release_rhs_buffer!(F_fields, solver)
    end

    # ── Final update: M*X_{n+1} = M*X_n + dt*Σ(b^E*F - b^I*L*X) ──────────
    for k in eachindex(batches)
        batch = batches[k]
        ws = workspaces[k]

        copyto!(ws.RHS, ws.MX0)
        for s in 1:stages
            be = dt * b_exp[s]
            bi = dt * b_imp[s]
            if abs(be) > 1e-14
                _batch_axpy!(ws.RHS, be, ws.F[s])
            end
            if abs(bi) > 1e-14
                _batch_axpy!(ws.RHS, -bi, ws.LX[s])
            end
        end

        _batch_mass_solve!(ws.X, ws.RHS, batch, subproblems, ws;
                           skip_missing=true)
        if all(ws.mass_ok)
            _batched_scatter_state!(ws, batch, state_fields, ws.X)
        else
            # Singular M with no usable solver on SOME modes: the per-mode loop
            # keeps the last stage value for exactly those, so scatter only the
            # modes that solved.
            _scatter_solved_modes!(ws, batch, subproblems, state_fields)
        end
    end
    for sp_idx in leftovers
        sp = subproblems[sp_idx]
        sp.M_min === nothing && continue
        _leftover_final!(sp, sp_idx, dt, stages, b_exp, b_imp, MX0, RHS, F, LX,
                         state_fields)
    end

    from_solve_layout!(solve_stash, dist)
    _push_trim!(state.history, state_fields, 1)
    return nothing
end

"""Scatter only the modes whose mass solve succeeded, one at a time."""
function _scatter_solved_modes!(ws::BatchWorkspace, batch::ModeBatch, sps,
                                state_fields)
    for (m, i) in enumerate(batch.sp_indices)
        ws.mass_ok[m] || continue
        sp = sps[i]
        x_v = _sp_stage_vector!(sp, :batch_mass_sol, size(sp.M_min, 2))
        copyto!(x_v, view(ws.X, :, m))
        scatter_inputs(sp, x_v, state_fields)
    end
    return nothing
end
