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

# Export core solver API
export step!, solve!, proceed, run!
