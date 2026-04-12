"""
    diagnose(solver::InitialValueSolver)

Print a tree-style summary of the solver configuration,
following the Oceananigans/FourierFlows display convention.
"""
function diagnose(solver::InitialValueSolver)
    problem = solver.problem
    dist = solver.state[1].dist
    arch = dist.architecture

    ts_name = nameof(typeof(solver.timestepper))
    println("Simulation of $(length(problem.equations))-equation IVP")
    println("├── timestepper: $ts_name")
    println("├── Δt: $(solver.dt)")
    println("├── sim_time: $(round(solver.sim_time; digits=6))")
    println("├── iteration: $(solver.iteration)")

    # Architecture
    println("├── architecture: $(nameof(typeof(arch)))")
    println("├── MPI ranks: $(dist.size)")

    # Domain & bases
    if !isempty(solver.state) && solver.state[1].domain !== nothing
        bases = solver.state[1].bases
        println("├── bases: $(length(bases))")
        for (i, basis) in enumerate(bases)
            btype = nameof(typeof(basis))
            N = basis.meta.size
            bounds = basis.meta.bounds
            prefix = i < length(bases) ? "│   ├── " : "│   └── "
            println("$prefix$btype(N=$N, bounds=$bounds)")
        end
    end

    # State fields
    total_dof = 0
    total_mem = 0
    println("├── state fields: $(length(solver.state))")
    for (i, field) in enumerate(solver.state)
        gdata = get_grid_data(field)
        sz = gdata !== nothing ? size(gdata) : (isempty(field.bases) ? () : tuple([b.meta.size for b in field.bases if b !== nothing]...))
        dof = max(prod(sz; init=1), 1)
        mem = dof * sizeof(field.dtype)
        total_dof += dof
        total_mem += 2 * mem
        layout = field.current_layout
        prefix = i < length(solver.state) ? "│   ├── " : "│   └── "
        println("$prefix$(field.name) $sz [$layout] $(field.dtype)")
    end
    println("│   Total DOF: $total_dof, Memory: $(round(total_mem / 1024^2; digits=2)) MB")

    # Equations
    println("├── equations: $(length(problem.equations))")
    for (i, eq) in enumerate(problem.equations)
        prefix = i < length(problem.equations) ? "│   ├── " : "│   └── "
        println("$prefix$eq")
    end

    # RHS evaluation
    if solver.rhs_plan !== nothing
        plan = solver.rhs_plan
        if plan.is_compiled
            n_eqs = count(!isnothing, plan.exprs)
            println("├── RHS: lazy (type-specialized, $n_eqs equations)")
        else
            println("├── RHS: interpreted (lazy translation skipped)")
        end
    else
        println("├── RHS: interpreted")
    end

    # Boundary conditions
    bc = problem.bc_manager
    n_bcs = length(bc.conditions)
    has_time_dep = has_time_dependent_bcs(bc)
    println("├── boundary conditions: $n_bcs (time-dependent: $has_time_dep)")

    # Subproblem decomposition summary (per-rank locality report).
    # This tells users how many per-Fourier-mode subproblems each rank
    # owns, so they can spot load imbalance before a production run.
    if haskey(problem.parameters, "subproblems")
        sps = problem.parameters["subproblems"]
        if sps isa Tuple
            n_sp_local = length(sps)
            n_sp_with_matrices = count(sp -> sp.M_min !== nothing, sps)
            if dist.size > 1
                # Collect counts from all ranks for load-balance reporting.
                # (Rank-0 prints the summary; other ranks still compute it.)
                try
                    comm = dist.comm
                    counts = MPI.Allgather(Int32(n_sp_local), comm)
                    if dist.rank == 0
                        total_sp = sum(Int.(counts))
                        min_sp = minimum(counts)
                        max_sp = maximum(counts)
                        imbalance = max_sp - min_sp
                        pct = imbalance > 0 ? round(100 * imbalance / max(max_sp, 1); digits=1) : 0.0
                        println("├── subproblems: $total_sp total across $(dist.size) ranks")
                        println("│   ├── local: $n_sp_local ($n_sp_with_matrices with matrices)")
                        println("│   ├── min/max per rank: $min_sp / $max_sp")
                        println("│   └── imbalance: $imbalance subproblems ($pct%)")
                    end
                catch
                    println("├── subproblems: $n_sp_local local ($n_sp_with_matrices with matrices)")
                end
            else
                println("├── subproblems: $n_sp_local ($n_sp_with_matrices with matrices)")
            end
        end
    end

    # Stochastic forcing
    if hasfield(typeof(problem), :stochastic_forcings) && !isempty(problem.stochastic_forcings)
        println("├── stochastic forcing: $(length(problem.stochastic_forcings)) fields")
    end

    # Performance (if steps taken)
    stats = solver.performance_stats
    if stats.total_steps > 0
        avg_step = stats.total_time / stats.total_steps * 1000
        println("├── performance:")
        println("│   ├── total steps: $(stats.total_steps)")
        println("│   ├── wall time: $(round(stats.total_time; digits=2))s")
        println("│   └── avg step: $(round(avg_step; digits=2)) ms")
    end

    println("└── stop: time=$(solver.stop_sim_time), iteration=$(solver.stop_iteration)")
end

function Base.show(io::IO, plan::LazyRHSPlan)
    status = plan.is_compiled ? "compiled" : "failed"
    n_eqs = count(!isnothing, plan.exprs)
    print(io, "LazyRHSPlan($status, $n_eqs equations)")
end

"""
    subproblem_locality_report(solver) -> NamedTuple

Compute a structured report of the per-rank subproblem distribution for an
`InitialValueSolver` under MPI. Useful for diagnosing load imbalance
before or during a production run.

Returns a `NamedTuple` with fields:

- `total::Int` — total subproblem count across all MPI ranks
- `per_rank::Vector{Int}` — subproblem count on each rank (rank 0 is index 1)
- `local_count::Int` — subproblem count on this rank
- `with_matrices::Int` — subproblems on this rank that have non-nothing `M_min`
- `min_per_rank::Int` / `max_per_rank::Int` / `mean_per_rank::Float64`
- `imbalance_pct::Float64` — `(max - min) / max * 100`, zero means perfect balance
- `rank::Int` — this rank's id

For serial runs (`dist.size == 1`) the report reflects a single rank with
all subproblems local. Safe to call on any rank; the `per_rank` vector is
filled via `MPI.Allgather` so every rank has the full view.

### Example

```julia
solver = InitialValueSolver(problem, RK222(); dt=1e-3)
report = subproblem_locality_report(solver)
if report.rank == 0 && report.imbalance_pct > 20
    @warn "Subproblem imbalance \$(report.imbalance_pct)% — consider adjusting MPI rank count to divide Nx_c evenly"
end
```

### When the imbalance is a problem

For a 2D problem with `Nx = 256` → `Nx_c = 129` subproblems, running on 8
ranks gives `floor(129/8) = 16` subproblems per rank plus a leftover of 1.
One rank gets 17, seven get 16 — imbalance `1/17 ≈ 6%`. Manageable.

Running the same problem on 16 ranks gives `129 = 16*8 + 1`, so most
ranks get 8 while one gets 9. 12% imbalance. Still OK.

Running on 7 ranks gives `129 = 18*7 + 3`, so 3 ranks get 19 and 4 get 18.
5% imbalance, fine. But if rank count doesn't divide `Nx_c` cleanly, the
overall throughput is limited by the slowest (most-loaded) rank, so
significant imbalance (>20%) is worth fixing by adjusting the rank count
to share a common factor with `Nx_c`.
"""
function subproblem_locality_report(solver::InitialValueSolver)
    problem = solver.problem
    dist = solver.state[1].dist

    n_sp_local = 0
    with_matrices = 0
    if haskey(problem.parameters, "subproblems")
        sps = problem.parameters["subproblems"]
        if sps isa Tuple
            n_sp_local = length(sps)
            with_matrices = count(sp -> sp.M_min !== nothing, sps)
        end
    end

    per_rank = if dist.size > 1
        try
            Int.(MPI.Allgather(Int32(n_sp_local), dist.comm))
        catch
            Int[n_sp_local]
        end
    else
        Int[n_sp_local]
    end

    total = sum(per_rank)
    min_pr = isempty(per_rank) ? 0 : minimum(per_rank)
    max_pr = isempty(per_rank) ? 0 : maximum(per_rank)
    mean_pr = isempty(per_rank) ? 0.0 : sum(per_rank) / length(per_rank)
    imbalance_pct = max_pr > 0 ? 100 * (max_pr - min_pr) / max_pr : 0.0

    return (
        total = total,
        per_rank = per_rank,
        local_count = n_sp_local,
        with_matrices = with_matrices,
        min_per_rank = min_pr,
        max_per_rank = max_pr,
        mean_per_rank = mean_pr,
        imbalance_pct = imbalance_pct,
        rank = dist.rank,
    )
end

# Export core solver API
export step!, solve!, proceed, run!, subproblem_locality_report
