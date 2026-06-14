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

        _refresh_step_boundary_conditions!(solver, dt)
        ts_state = _ensure_timestepper_state!(solver, dt)

        # Call existing timestepper step function from timesteppers.jl
        step!(ts_state, solver)

        step_time = time() - start_time
        _sync_solver_from_timestepper!(solver)
        _advance_solver_clock!(solver, dt, step_time)

        return solver
    finally
        if pool_owner
            set_field_pool!(nothing)
        end
    end
end

_callback_should_fire(interval::Integer, solver::InitialValueSolver, _::Float64) =
    solver.iteration % interval == 0
_callback_should_fire(interval::AbstractFloat, solver::InitialValueSolver, last_time::Float64) =
    solver.sim_time - last_time >= interval

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
            should_fire = _callback_should_fire(interval, solver, last_callback_times[idx])
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
    _solve_bvp!(solver, solver.problem)
    solve_time = time() - start_time
    solver.performance_stats.total_time += solve_time
    solver.performance_stats.total_solves += 1
    return solver
end

function _solve_bvp!(solver::BoundaryValueSolver, ::LBVP)
    solution = solve_linear!(solver)
    copy_solution_to_fields!(solver.state, solution)
end

function _solve_bvp!(solver::BoundaryValueSolver, ::NLBVP)
    solve_nonlinear!(solver)
end

"""Solve linear boundary value problem.

Preferred path: solve PER-FOURIER-MODE subproblem (`L_sp x = F_sp`, scatter back),
reusing the same machinery as the IVP timestepper. Each subproblem is a small
square tau system over the coupled (Chebyshev) dimension for one separable mode,
so the single-mode operator matrices are correct. Falls back to the (legacy)
global `L \\ F` solve only when no per-mode subproblems are available.
"""
function solve_linear!(solver::BoundaryValueSolver)
    sps = solver.subproblems
    if !isempty(sps) && any(sp -> sp.L_min !== nothing, sps)
        _solve_bvp_per_subproblem!(solver)
        return fields_to_vector(solver.state)
    end
    if solver.global_solver !== nothing
        return MatSolvers.solve(solver.global_solver, solver.F_vector)
    end
    return solver.L_matrix \ solver.F_vector
end

# Build a solver for a subproblem's L_min, with SPQR → dense fallbacks for
# rank-revealing / awkward tau systems (mirrors the IVP `_get_or_build_lhs!`).
function _bvp_lhs_solver(sp::Subproblem)
    st = _subproblem_solver_type(sp.solver.base.matsolver)
    try
        return MatSolvers.solver_instance(st, sp.L_min)
    catch
        try
            return MatSolvers.solver_instance(MatSolvers.SPQRSolver, sp.L_min)
        catch
            return Matrix(sp.L_min)
        end
    end
end

# Per-Fourier-mode steady solve: assemble each subproblem RHS (PDE forcing via
# gather_eqn_F!, BC rows via gather_alg_F!/apply_bc_override!), solve, scatter.
function _solve_bvp_per_subproblem!(solver::BoundaryValueSolver)
    problem = solver.problem
    state = solver.state
    sps = solver.subproblems

    # The IVP forcing path is M-term-gated (`_subproblem_eqn_targets` returns []
    # for equations without a time derivative), so for a steady BVP it places
    # nothing. We therefore build BVP targets directly: the "bulk" PDE equations
    # are the largest equation blocks (full coupled-dimension size); the small
    # blocks are algebraic BC/constraint rows (handled by gather_alg_F! +
    # apply_bc_override! below). Each bulk equation's RHS forcing is gathered onto
    # the coupled (non-Fourier) variable's rows. The forcing is solution-
    # independent for a linear BVP, so evaluating on the current state is exact.
    # Each BULK PDE equation needs its OWN forcing container — a distinct field of
    # the coupled/bulk shape. Targeting every bulk equation at a single `udix`
    # (the first coupled field) made them overwrite one pde_F slot, so in a COUPLED
    # MULTI-FIELD system every bulk equation received the LAST equation's RHS. Map
    # the k-th bulk equation to the k-th coupled (non-Fourier) variable: distinct
    # containers of the right shape. The container's variable identity is
    # irrelevant — gather_eqn_F! places pde_F[ti] onto THIS equation's rows — only
    # distinctness + shape matter. Single-field problems map to the same one var.
    coupled_vars = findall(f -> any(b -> b !== nothing && !isa(b, FourierBasis), f.bases), state)
    eqn_sizes = _subproblem_eqn_sizes(sps[1])
    maxsz = isempty(eqn_sizes) ? 0 : maximum(eqn_sizes)

    pde_F = Vector{Any}(nothing, length(state))
    bvp_targets = Vector{Vector{Int}}(undef, length(problem.equation_data))
    bulk_seen = 0
    for (eq_idx, eq_data) in enumerate(problem.equation_data)
        es = eq_idx <= length(eqn_sizes) ? eqn_sizes[eq_idx] : 0
        if es == maxsz && es > 1 && !isempty(coupled_vars)
            bulk_seen += 1
            ti = coupled_vars[min(bulk_seen, length(coupled_vars))]
            bvp_targets[eq_idx] = [ti]
            F_expr = get(eq_data, "F_expr", nothing)
            F_expr === nothing && (F_expr = get(eq_data, "F", nothing))
            if F_expr !== nothing && !_is_zero_F_expr(F_expr)
                pde_F[ti] = evaluate_solver_expression(F_expr, problem.variables;
                                                       layout = :g, template = state[ti])
            end
        else
            bvp_targets[eq_idx] = Int[]
        end
    end

    for sp in sps
        sp.L_min === nothing && continue
        # Override the M-gated target cache with the BVP bulk-equation targets.
        sp.runtime.eqn_targets = bvp_targets
        n_eq = size(sp.L_min, 1)
        rhs = zeros(ComplexF64, n_eq)
        gather_eqn_F!(rhs, sp, solver, pde_F, state)
        alg_f = zeros(ComplexF64, n_eq)
        gather_alg_F!(alg_f, sp)
        apply_bc_override!(rhs, alg_f, sp, 1.0)

        x = Vector{ComplexF64}(undef, size(sp.L_min, 2))
        _solve_cached_system!(x, _bvp_lhs_solver(sp), rhs)
        scatter_inputs(sp, x, state)
    end
    return solver
end

"""Solve nonlinear boundary value problem using Newton iteration"""
# Solve nonlinear boundary value problem via PER-FOURIER-MODE Newton iteration,
# mirroring Dedalus' NonlinearBoundaryValueSolver (per-subproblem dF Jacobian
# rebuilt each iteration, per-subproblem residual, per-subproblem solve).
#
# Each iteration, per subproblem:
#   residual  F_sp = L_lin_sp*x_sp - RHS_sp   (linear operator + nonlinear forcing)
#   Jacobian  dF_sp = build of (orig_L - frechet(rhs))   (full-rank per mode)
#   step      x_sp <- x_sp - dF_sp_inv * F_sp ;  scatter
#
# Falls back to the legacy global Newton when no per-mode subproblems are present.
function solve_nonlinear!(solver::BoundaryValueSolver)
    sps = solver.subproblems
    if isempty(sps) || all(sp -> sp.L_min === nothing, sps)
        return _solve_nonlinear_global!(solver)
    end
    problem = solver.problem
    state = solver.state
    eqd = problem.equation_data

    # Per-equation operators: original linear LHS, and the Frechet Jacobian
    # operator dF = orig_L − frechet(rhs). dF carries the current-state NCCs
    # (e.g. ∂(u²)/∂u = 2u), so build_matrices! rebuilds it fresh each iteration.
    origL = [get(ed, "L", nothing) for ed in eqd]
    dF_expr = Vector{Any}(undef, length(eqd))
    for (i, ed) in enumerate(eqd)
        rhs_expr = get(ed, "rhs", nothing)
        fr = (rhs_expr === nothing) ? 0 :
             frechet_differential(rhs_expr, problem.variables, problem.variables)
        dF_expr[i] = (origL[i] === nothing) ? origL[i] :
                     ((fr === 0 || fr === nothing) ? origL[i] : origL[i] - fr)
    end

    # PDE forcing targets: the largest equation block is the bulk PDE (targets the
    # coupled variable); the small blocks are algebraic BC rows (left empty).
    # Distinct forcing container per bulk equation (see solve_linear! above): map
    # the k-th bulk PDE equation to the k-th coupled variable so a coupled
    # multi-field system does not collapse every equation's RHS onto one slot.
    coupled_vars = findall(f -> any(b -> b !== nothing && !isa(b, FourierBasis), f.bases), state)
    eqn_sizes = _subproblem_eqn_sizes(sps[1])
    maxsz = isempty(eqn_sizes) ? 0 : maximum(eqn_sizes)
    bvp_targets = Vector{Vector{Int}}(undef, length(eqd))
    bulk_seen = 0
    for i in eachindex(eqd)
        es = i <= length(eqn_sizes) ? eqn_sizes[i] : 0
        if es == maxsz && es > 1 && !isempty(coupled_vars)
            bulk_seen += 1
            bvp_targets[i] = [coupled_vars[min(bulk_seen, length(coupled_vars))]]
        else
            bvp_targets[i] = Int[]
        end
    end

    x = fields_to_vector(state)
    converged = false
    resnorm = Inf
    for iter in 1:solver.max_iterations
        copy_solution_to_fields!(state, x)

        # --- Residual F_sp (sp.L_min holds the LINEAR operator here) ---
        # nonlinear RHS forcing fields (e.g. u²+g) evaluated at the current state
        pde_F = Vector{Any}(nothing, length(state))
        for (i, ed) in enumerate(eqd)
            isempty(bvp_targets[i]) && continue
            rhs_expr = get(ed, "rhs", nothing)
            rhs_expr === nothing && continue
            ti = bvp_targets[i][1]
            pde_F[ti] = evaluate_solver_expression(rhs_expr, problem.variables; layout=:g, template=state[ti])
        end
        F = Vector{Vector{ComplexF64}}(undef, length(sps))
        resnorm = 0.0
        for (k, sp) in enumerate(sps)
            if sp.L_min === nothing
                F[k] = ComplexF64[]
                continue
            end
            n_eq = size(sp.L_min, 1)
            x_sp = zeros(ComplexF64, size(sp.L_min, 2))
            gather_inputs!(x_sp, sp, state)
            Lx = zeros(ComplexF64, n_eq)
            _apply_subproblem_operator!(Lx, _subproblem_operator(sp, :L, x_sp), x_sp)
            rhs = zeros(ComplexF64, n_eq)
            sp.runtime.eqn_targets = bvp_targets
            gather_eqn_F!(rhs, sp, solver, pde_F, state)
            alg = zeros(ComplexF64, n_eq)
            gather_alg_F!(alg, sp)
            apply_bc_override!(rhs, alg, sp, 1.0)
            F[k] = Lx .- rhs
            resnorm = max(resnorm, norm(F[k]))
        end
        if resnorm < solver.tolerance
            converged = true
            break
        end

        # --- Rebuild dF per subproblem (Jacobian at current state) ---
        for (i, ed) in enumerate(eqd); ed["L"] = dF_expr[i]; end
        for sp in sps; sp.L_min === nothing || build_matrices!(sp, ["L"], solver); end

        # --- Newton step: dF_sp δ = F_sp ; x_sp ← x_sp − δ ; scatter ---
        for (k, sp) in enumerate(sps)
            sp.L_min === nothing && continue
            x_sp = zeros(ComplexF64, size(sp.L_min, 2))
            gather_inputs!(x_sp, sp, state)
            δ = Vector{ComplexF64}(undef, size(sp.L_min, 2))
            _solve_cached_system!(δ, _bvp_lhs_solver(sp), F[k])
            scatter_inputs(sp, x_sp .- δ, state)
        end

        # --- Restore the linear operator for the next residual ---
        for (i, ed) in enumerate(eqd); ed["L"] = origL[i]; end
        for sp in sps; sp.L_min === nothing || build_matrices!(sp, ["L"], solver); end

        x = fields_to_vector(state)
    end

    copy_solution_to_fields!(solver.state, x)
    if !converged
        @warn "NLBVP per-mode Newton did not reach tolerance $(solver.tolerance) in \
               $(solver.max_iterations) iters (final |F|=$resnorm)"
    end
    return solver
end

# Legacy global Newton (fallback when no per-mode subproblems are available).
function _solve_nonlinear_global!(solver::BoundaryValueSolver)
    x = fields_to_vector(solver.state)
    converged = false
    last_dx_norm = Inf
    for iter in 1:solver.max_iterations
        residual, jacobian = evaluate_residual_and_jacobian(solver.problem, x)
        dx = -jacobian \ residual
        x += dx
        last_dx_norm = norm(dx)
        if last_dx_norm < solver.tolerance
            converged = true
            break
        end
    end
    if !converged
        error("Nonlinear solver did not converge after $(solver.max_iterations) iterations " *
              "(tolerance=$(solver.tolerance), final |dx|=$(last_dx_norm)).")
    end
    copy_solution_to_fields!(solver.state, x)
end

# Eigenvalue solver
# Order eigenvalues per the ARPACK-style `which` selector (and optional target,
# which selects by proximity instead of the default ordering).
function _eig_order(λ::AbstractVector, which::Symbol, target)
    if target !== nothing
        return sortperm(λ; by = x -> abs(x - target))
    end
    key = which === :LM ? (x -> -abs(x)) :
          which === :SM ? (x ->  abs(x)) :
          which === :LR ? (x -> -real(x)) :
          which === :SR ? (x ->  real(x)) :
          which === :LI ? (x -> -imag(x)) :
          which === :SI ? (x ->  imag(x)) :
          (x -> abs(x))
    return sortperm(λ; by = key)
end

"""Solve the generalized eigenvalue problem `L v = λ M v`.

Solves PER-FOURIER-MODE on the square per-subproblem tau matrices (`sp.L_min`,
`sp.M_min`), mirroring the BVP solver and Dedalus. The GLOBAL `L`/`M` are
rank-deficient for multi-variable tau systems, so the dense per-subproblem solve
is both correct and robust (Arpack on the global matrices throws
`SingularException`). Spurious eigenvalues from the singular mass matrix (the
algebraic BC/tau rows have zero `M`) come back as non-finite or astronomically
large and are filtered out. Falls back to global Arpack only when no per-mode
subproblems are available.
"""
function solve!(solver::EigenvalueSolver; nev::Int=solver.nev,
                which::Union{String,Symbol}=solver.which,
                target::Union{Nothing, ComplexF64}=solver.target)
    start_time = time()
    which_symbol = Symbol(uppercase(String(which)))
    sps = solver.subproblems
    use_persp = !isempty(sps) && any(sp -> sp.L_min !== nothing, sps)

    local λ, v
    if use_persp
        all_λ = ComplexF64[]
        single_vecs = nothing          # eigenvectors only when there is one subproblem
        n_active = count(sp -> sp.L_min !== nothing, sps)
        for sp in sps
            sp.L_min === nothing && continue
            Lm = Matrix{ComplexF64}(sp.L_min)
            Mm = Matrix{ComplexF64}(sp.M_min)
            F = eigen(Lm, Mm)
            # drop spurious eigenvalues from the singular (zero-row) mass matrix
            keep = findall(x -> isfinite(x) && abs(x) < 1e10, F.values)
            append!(all_λ, F.values[keep])
            if n_active == 1
                single_vecs = F.vectors[:, keep]
            end
        end
        order = _eig_order(all_λ, which_symbol, target)
        k = min(nev, length(order))
        sel = order[1:k]
        λ = all_λ[sel]
        v = single_vecs === nothing ? zeros(ComplexF64, 0, 0) : single_vecs[:, sel]
    else
        # Legacy global path (no per-mode subproblems available).
        if target === nothing
            λ, v = Arpack.eigs(solver.L_matrix, solver.M_matrix; nev=nev, which=which_symbol)
        else
            λ, v = Arpack.eigs(solver.L_matrix, solver.M_matrix; nev=nev, sigma=target)
        end
    end

    solver.nev = nev
    solver.which = which_symbol
    solver.target = target
    solver.eigenvalues = λ
    solver.eigenvectors = v

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
