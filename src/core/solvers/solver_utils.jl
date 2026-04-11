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

    # RHS compilation
    if solver.compiled_rhs !== nothing
        plan = solver.compiled_rhs
        if plan.is_compiled
            println("├── RHS: compiled ($(length(plan.instructions)) instructions)")
        else
            println("├── RHS: interpreted (compilation skipped)")
        end
    else
        println("├── RHS: interpreted")
    end

    # Boundary conditions
    bc = problem.bc_manager
    n_bcs = length(bc.conditions)
    has_time_dep = has_time_dependent_bcs(bc)
    println("├── boundary conditions: $n_bcs (time-dependent: $has_time_dep)")

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

function Base.show(io::IO, plan::CompiledRHSPlan)
    status = plan.is_compiled ? "compiled" : "failed"
    print(io, "CompiledRHSPlan($status, $(length(plan.instructions)) instructions, $(length(plan.workspace)) workspace)")
end

# Export core solver API
export step!, solve!, proceed, run!
