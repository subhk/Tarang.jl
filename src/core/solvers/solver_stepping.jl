# -----------------------------------------------------------------------------
# IVP runtime entry point.
#
# Read this file first when tracing one solver step:
# 1. refresh time-dependent boundary conditions for the new step time
# 2. create or update `TimestepperState`
# 3. hand off to `timesteppers/dispatch.jl`
# 4. sync the final state back into `problem.variables`
#
# Scheme-specific stage logic lives in `src/core/timesteppers/step_*.jl`.
# -----------------------------------------------------------------------------

"""Advance solution by one time step using existing timestepper infrastructure"""
function step!(solver::InitialValueSolver, dt::Float64=solver.dt)

    # NOTE: FieldPool is disabled until the checkout_or_alloc lifetime/aliasing
    # issues are resolved. Multiple arithmetic operations (dot product, cross product,
    # RHS evaluation) require simultaneous intermediate fields; the pool can return
    # the same buffer for different live intermediates, causing silent data corruption.
    # See: evaluate_vector_cross_product, dot_operands, evaluate_rhs.
    pool_owner = false

    try
        start_time = time()

        solver.dt = dt

        # Update time-dependent boundary conditions BEFORE taking the step.
        # Pure space-only BCs are already populated at solver-build time and
        # don't change between steps, so we skip the refresh for them to
        # avoid redundant FFT recomputation. Space-AND-time dependent BCs
        # live in `time_dependent_bcs` as well, so the guard below still
        # catches them and both axes get refreshed inside
        # `_apply_bc_values_to_equations!`.
        #
        # For RK steppers, `step_subproblem_rk!` re-runs the refresh at each
        # stage time (see `_refresh_bcs_for_stage!`) to recover full
        # stage-order accuracy on rapidly-varying BCs. For the multistep
        # stepper a single per-step refresh is sufficient.
        bcm = solver.problem.bc_manager
        if has_time_dependent_bcs(bcm)
            target_time = solver.sim_time + dt
            update_time_dependent_bcs!(bcm, target_time)
            _apply_bc_values_to_equations!(solver, target_time)
            @debug "Refreshed BCs at t=$target_time"
        end

        # Use existing timestepper infrastructure from timesteppers.jl
        # Create TimestepperState if needed
        if solver.timestepper_state === nothing
            solver.timestepper_state = TimestepperState(solver.timestepper, dt, solver.state)
        else
            # Update timestep history for variable timestep support
            update_timestep_history!(solver.timestepper_state, dt)
        end

        # Call existing timestepper step function from timesteppers.jl
        step!(solver.timestepper_state, solver)

        # Get the updated state from timestepper history
        if length(solver.timestepper_state.history) > 0
            solver.state = solver.timestepper_state.history[end]
        end

        # Sync the final state back to problem variables so users can read them directly
        # (without this, problem variables hold stale intermediate stage data from evaluate_rhs)
        sync_state_to_problem!(solver.problem, solver.state)

        # Update time and iteration
        solver.sim_time += dt
        solver.iteration += 1

        # Update performance statistics
        step_time = time() - start_time
        solver.performance_stats.total_time += step_time
        solver.performance_stats.total_steps += 1

        return solver
    finally
        if pool_owner
            set_field_pool!(nothing)
        end
    end
end

# Solver execution control
"""Check if solver should continue"""
function proceed(solver::InitialValueSolver)
    if solver.sim_time >= solver.stop_sim_time
        return false
    end
    
    if solver.iteration >= solver.stop_iteration
        return false
    end
    
    if time() - solver.wall_time_start >= solver.stop_wall_time
        return false
    end
    
    return true
end

"""
    run!(solver; stop_time=Inf, stop_iteration=typemax(Int), stop_wall_time=Inf,
         callbacks=[], log_interval=100, progress=true)

Run a simulation loop to completion with optional callbacks and progress reporting.

This eliminates the standard simulation boilerplate. Instead of:
```julia
while proceed(solver)
    step!(solver)
    if solver.iteration % 100 == 0
        @info "Step \$(solver.iteration), t=\$(solver.sim_time)"
    end
end
```

Use:
```julia
run!(solver; stop_time=10.0, log_interval=100)
```

# Callbacks

Callbacks are `(interval, function)` tuples. The function receives the solver:
```julia
run!(solver; stop_time=10.0, callbacks=[
    (10,  s -> @info "Energy: \$(energy(s))"),
    (100, s -> save_checkpoint(s))
])
```

`interval` can be:
- `Int`: execute every N iterations
- `Float64`: execute every T simulation time units
"""
function run!(solver::InitialValueSolver;
              stop_time::Real=Inf,
              stop_iteration::Integer=typemax(Int),
              stop_wall_time::Real=Inf,
              callbacks::Vector=Pair[],
              log_interval::Integer=0,
              progress::Bool=true)

    solver.stop_sim_time = Float64(stop_time)
    solver.stop_iteration = Int(stop_iteration)
    solver.stop_wall_time = Float64(stop_wall_time)
    solver.wall_time_start = time()

    # Track last callback times for time-based intervals
    last_callback_times = Float64[solver.sim_time for _ in callbacks]

    if progress
        @info "Starting simulation: dt=$(solver.dt), stop_time=$stop_time, stop_iteration=$stop_iteration"
    end

    wall_start = time()

    while proceed(solver)
        step!(solver)

        # Log progress
        if log_interval > 0 && solver.iteration % log_interval == 0
            elapsed = time() - wall_start
            rate = solver.iteration / max(elapsed, 1e-10)
            @info "Step $(solver.iteration), t=$(round(solver.sim_time; digits=6)), " *
                  "wall=$(round(elapsed; digits=1))s, rate=$(round(rate; digits=1)) steps/s"
        end

        # Execute callbacks
        for (idx, cb) in enumerate(callbacks)
            interval, func = cb
            should_fire = if interval isa Integer
                solver.iteration % interval == 0
            elseif interval isa AbstractFloat
                solver.sim_time - last_callback_times[idx] >= interval
            else
                false
            end

            if should_fire
                func(solver)
                last_callback_times[idx] = solver.sim_time
            end
        end
    end

    elapsed = time() - wall_start
    if progress
        @info "Simulation complete: $(solver.iteration) steps, " *
              "t=$(round(solver.sim_time; digits=6)), wall=$(round(elapsed; digits=1))s"
    end

    return solver
end

# Boundary value solver
"""Solve boundary value problem"""
function solve!(solver::BoundaryValueSolver)

    start_time = time()

    if isa(solver.problem, LBVP)
        # Linear boundary value problem
        solution = solve_linear!(solver)

        # Copy solution back to state fields
        copy_solution_to_fields!(solver.state, solution)

    elseif isa(solver.problem, NLBVP)
        # Nonlinear boundary value problem - Newton iteration
        solve_nonlinear!(solver)
    end

    # Update performance statistics
    solve_time = time() - start_time
    solver.performance_stats.total_time += solve_time
    solver.performance_stats.total_solves += 1

    return solver
end

"""Solve linear boundary value problem"""
function solve_linear!(solver::BoundaryValueSolver)
    # Direct solve: L * x = F
    if solver.global_solver !== nothing
        return MatSolvers.solve(solver.global_solver, solver.F_vector)
    end
    return solver.L_matrix \ solver.F_vector
end

"""Solve nonlinear boundary value problem using Newton iteration"""
function solve_nonlinear!(solver::BoundaryValueSolver)

    # Initial guess (current state)
    x = fields_to_vector(solver.state)
    converged = false
    last_dx_norm = Inf

    for iter in 1:solver.max_iterations
        # Evaluate residual and Jacobian
        residual, jacobian = evaluate_residual_and_jacobian(solver.problem, x)

        # Newton update: J * dx = -R
        dx = -jacobian \ residual
        x += dx
        last_dx_norm = norm(dx)

        # Check convergence
        if last_dx_norm < solver.tolerance
            @info "Nonlinear solver converged in $iter iterations"
            converged = true
            break
        end
    end

    if !converged
        error("Nonlinear solver did not converge after $(solver.max_iterations) iterations " *
              "(tolerance=$(solver.tolerance), final |dx|=$(last_dx_norm)). " *
              "Consider: increasing max_iterations, loosening tolerance, or improving the initial guess.")
    end

    # Copy solution back
    copy_solution_to_fields!(solver.state, x)
end

# Eigenvalue solver
function solve!(solver::EigenvalueSolver; nev::Int=solver.nev,
                which::Union{String,Symbol}=solver.which,
                target::Union{Nothing, ComplexF64}=solver.target)
    """Solve eigenvalue problem"""

    start_time = time()

    which_symbol = Symbol(uppercase(String(which)))

    # Solve generalized eigenvalue problem: L * v = λ * M * v
    if target === nothing
        λ, v = Arpack.eigs(solver.L_matrix, solver.M_matrix; nev=nev, which=which_symbol)
    else
        λ, v = Arpack.eigs(solver.L_matrix, solver.M_matrix; nev=nev, sigma=target)
    end

    solver.nev = nev
    solver.which = which_symbol
    solver.target = target
    solver.eigenvalues = λ
    solver.eigenvectors = v

    # Update performance statistics
    solve_time = time() - start_time
    solver.performance_stats.total_time += solve_time
    solver.performance_stats.total_solves += 1

    return λ, v
end


# Utility functions
"""
    Convert field array to solution vector following gather pattern.

    GPU-aware: For GPU fields, data is synchronized and transferred to CPU.
    Linear solves are performed on CPU (standard practice for sparse solvers),
    and results are transferred back to GPU via copy_solution_to_fields!.

    This function always returns a CPU Vector{ComplexF64} since that's what
    sparse linear solvers expect.
    """
