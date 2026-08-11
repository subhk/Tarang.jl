"""
Engagement guards and end-to-end parity for the batched Fourier-mode solve.

The guards matter as much as the parity: batching must be OFF by default on CPU
and must never construct under MPI, so that every existing run is byte-for-byte
unchanged. `batched_modes=true` is what the suite uses to exercise the real
device-generic code path without a GPU.
"""

using Test
using Tarang

# `device` is threaded through to the Distributor so a GPU runner can drive
# these same assertions on device. Without it the file is hard-wired to CPU: a
# cluster runner that included it on a real GPU would run the CPU path,
# exercise zero device code, and report a pass. Every testset below uses the
# `CPU()` default; the parameter exists for the runner, not for them.
# `bc_low` is the lower Dirichlet value, as a BC-expression string. It defaults
# to the homogeneous `"0"` so the structural guard testsets below keep the exact
# problem Task 5 pinned — but a homogeneous BC makes `maxabs(ALG_F) == 0.0`,
# which leaves `bc_rows`, `batched_bc_override!` and `_batched_gather_alg_F!`
# numerically INERT: deleting the override's body still reproduces the per-mode
# answer to 4.8e-16. Every parity comparison therefore passes `bc_low="1"`
# (measured `maxabs(ALG_F) == 16.0`) or a time-dependent expression.
function _parity_channel_solver(; nx=16, nz=8, dt=1e-3, bc_low="0",
                                  device=Tarang.CPU(), kwargs...)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64, device=device)
    xbasis = RealFourier(coords["x"]; size=nx, bounds=(0.0, 2π), dealias=3 / 2)
    zbasis = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
    domain = Domain(dist, (xbasis, zbasis))

    b = ScalarField(domain, "b")
    tau1 = ScalarField(dist, "tau1", (xbasis,), Float64)
    tau2 = ScalarField(dist, "tau2", (xbasis,), Float64)
    _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    tau_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * tau_lift(tau1)

    problem = IVP([b, tau1, tau2])
    add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
    add_equation!(problem,
                  "∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = $bc_low")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt, kwargs...)
    return solver, b
end

# The smallest 3-D Fourier x Fourier x Chebyshev IVP that still builds
# subproblems: `nx=ny=4` gives 12 modes, and they all land in one bucket. Same
# equation, BCs and tau structure as the 2-D channel above, so the ONLY thing
# that differs is the number of Fourier axes — which is the point.
function _parity_channel_solver_3d(; nx=4, ny=4, nz=8, dt=1e-3,
                                     device=Tarang.CPU(), kwargs...)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype=Float64, device=device)
    xbasis = RealFourier(coords["x"]; size=nx, bounds=(0.0, 2π), dealias=3 / 2)
    ybasis = RealFourier(coords["y"]; size=ny, bounds=(0.0, 2π), dealias=3 / 2)
    zbasis = ChebyshevT(coords["z"]; size=nz, bounds=(0.0, 1.0))
    domain = Domain(dist, (xbasis, ybasis, zbasis))

    b = ScalarField(domain, "b")
    tau1 = ScalarField(dist, "tau1", (xbasis, ybasis), Float64)
    tau2 = ScalarField(dist, "tau2", (xbasis, ybasis), Float64)
    _, _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    tau_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * tau_lift(tau1)

    problem = IVP([b, tau1, tau2])
    add_parameters!(problem; kappa=0.1, grad_b, tau_lift)
    add_equation!(problem,
                  "∂t(b) - kappa*div(grad_b) + tau_lift(tau2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt, kwargs...)
    return solver, b
end

@testset "batched mode engagement guards" begin
    @testset "default on CPU is OFF" begin
        solver, _ = _parity_channel_solver()
        @test solver.base.batched_modes === nothing
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices;
                                         is_gpu=false, nprocs=1)
    end

    @testset "default resolves to ON for GPU — the reachable half of the default" begin
        # `nothing` must mean "GPU yes, CPU no", not "always off". Every other
        # testset here either passes an explicit `batched_modes` or checks
        # `is_gpu=false`, so none of them distinguish `setting === nothing ?
        # is_gpu : setting` from the simpler (wrong) `setting === true`: a
        # default-constructed solver has `batched_modes === nothing`, and on
        # CPU (`is_gpu=false`) both formulas agree (`false`). Only this
        # testset, with `is_gpu=true`, tells them apart.
        solver, _ = _parity_channel_solver()
        @test solver.base.batched_modes === nothing
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test Tarang.should_batch_modes(solver.base, sps, indices;
                                        is_gpu=true, nprocs=1)
    end

    @testset "opt-in turns it on for CPU" begin
        solver, _ = _parity_channel_solver(; batched_modes=true)
        @test solver.base.batched_modes === true
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test Tarang.should_batch_modes(solver.base, sps, indices;
                                        is_gpu=false, nprocs=1)
    end

    @testset "batched_modes=false overrides even GPU" begin
        solver, _ = _parity_channel_solver(; batched_modes=false)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices;
                                         is_gpu=true, nprocs=1)
    end

    @testset "MPI never batches, whatever the flag says" begin
        solver, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices;
                                         is_gpu=true, nprocs=2)
    end

    @testset "a 3-D problem declines, silently" begin
        # Nothing else stops it. Measured on this exact problem before the gate
        # existed: one bucket of 12 modes, `should_batch_modes` true,
        # `build_mode_batches!` returning a batch. At nx=ny=nz=64 that is 4096
        # modes at a few hundred MB — under the 1 GiB default — so a 3-D GPU run
        # would have engaged BY DEFAULT a gather over a `(kx, ky, :)` selection
        # that has never executed and that no test on this branch covers. The
        # declared scope is 2-D; decline like the one-mode bucket does, without
        # the byte cap's `@info` (an unsupported dimensionality is not a
        # surprise the way a silent performance cliff is).
        solver, _ = _parity_channel_solver_3d(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        live = [sp for sp in sps if sp.M_min !== nothing]
        @test length(live) >= 2

        # The gate's input, with the 2-D discriminator alongside it: the mode
        # group pins two Fourier axes here and exactly one in the channel.
        @test Tarang._mode_batch_fourier_axes(live[1]) == 2
        solver_2d, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver_2d)
        sps_2d = collect(solver_2d.problem.compiled.subproblems)
        @test Tarang._mode_batch_fourier_axes(sps_2d[1]) == 1

        # Every OTHER condition passes, so the decline can only be the gate.
        buckets = Tarang.bucket_subproblems(sps)
        @test length(buckets) == 1
        indices = first(values(buckets))
        @test length(indices) >= 2
        @test Tarang.mode_batch_bytes(sps[indices[1]], length(indices)) <=
              solver.base.batched_modes_max_bytes

        result = @test_logs Tarang.should_batch_modes(solver.base, sps, indices;
                                                        is_gpu=true, nprocs=1)
        @test !result
        @test isempty(Tarang.build_mode_batches!(solver.base, sps; is_gpu=true,
                                                 nprocs=1, like=ComplexF64[]))

        # And through the production entry point, which short-circuits before
        # bucketing. Layout is set first so that a REMOVED gate would reach the
        # workspace build and fail this assertion, rather than erroring earlier
        # for an unrelated reason.
        state = solver.timestepper_state
        state_fields = state.timestepper_data[:_sp_state_fields][2]
        for f in state_fields
            ensure_layout!(f, :c)
        end
        @test Tarang._build_batched_rk_plan(solver,
                                            solver.problem.compiled.subproblems,
                                            state_fields) === nothing
    end

    @testset "a one-mode bucket declines silently" begin
        # "Silently" is a testable claim, not just a title: wrap the call in a
        # zero-pattern @test_logs so an accidental @info here (or one migrated
        # in from the cap branch by a future edit) fails the test instead of
        # passing unnoticed alongside a correct-looking return value.
        solver, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        result = @test_logs Tarang.should_batch_modes(solver.base, sps, indices[1:1];
                                                       is_gpu=false, nprocs=1)
        @test !result
    end

    @testset "exceeding the byte cap declines and says so" begin
        # The title promises an @info, so assert it: @test_logs (:info,) fails
        # if the log block is ever deleted, which a bare return-value check
        # cannot detect. `should_batch_modes` is called exactly once here (the
        # call @test_logs itself makes) — `@info ... maxlog=1`'s throttle
        # counter lives on the logger @test_logs installs for the duration of
        # the macro, so a second, separately-wrapped call would not reliably
        # reproduce the log even though the first one did; verified directly
        # (see the fix-round report) that repeated @test_logs/collect_test_logs
        # blocks each get an independent view regardless of prior calls under
        # the default logger.
        solver, _ = _parity_channel_solver(; batched_modes=true,
                                             batched_modes_max_bytes=1)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        result = @test_logs (:info,) Tarang.should_batch_modes(solver.base, sps, indices;
                                                                is_gpu=false, nprocs=1)
        @test !result
    end

    @testset "the byte counter matches what is actually allocated" begin
        # Task 2's byte test restated the formula instead of measuring the
        # buffer, so the counter and the allocation could drift apart and the
        # memory cap would guard a number nothing allocates. Measure the real
        # thing: this is the gate's whole purpose.
        #
        # "The real thing" is EVERY array the batch holds, not `lhs_dense`
        # alone. Comparing against `sizeof(batch.lhs_dense)` — which this
        # testset used to do — was wrong by construction: `M_exp_nzval`,
        # `L_exp_nzval` and the CSR `L_nzval` are each `nnz(LHS) x nmodes`, so a
        # cap checked against the dense buffer alone lets a user ask for 8 GiB
        # on a 12 GB device, resident ~15 GB (measured 1.91x at nz=64), and OOM
        # with the cap's `@info` never firing. The sum below walks
        # `fieldnames(ModeBatch)` rather than naming fields, so a field added
        # later without a matching term in `mode_batch_bytes` fails here instead
        # of quietly widening the gap.
        solver, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        batch = Tarang.build_mode_batch(sps, indices; like=ComplexF64[])

        total = 0
        for name in fieldnames(Tarang.ModeBatch)
            v = getfield(batch, name)
            if v isa AbstractArray
                total += sizeof(v)
            elseif v isa Tarang.BatchedSparseOp
                total += sizeof(v.rowptr) + sizeof(v.colval) + sizeof(v.nzval)
            end
        end

        @test Tarang.mode_batch_bytes(sps[indices[1]], length(indices)) == total
        @test total > sizeof(batch.lhs_dense)
    end

    @testset "build_mode_batches! emptiness and ordering" begin
        # Neither of build_mode_batches!'s two contract points — "empty when
        # nothing qualifies" and "sorted by sp_indices[1] for deterministic
        # order" — is pinned anywhere else. The channel problem alone yields
        # exactly one bucket (see test_mode_batch_signature.jl), which cannot
        # exercise the sort at all, so the second sub-testset below perturbs
        # two modes to force a second qualifying bucket.
        solver, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        live = [sp for sp in sps if sp.M_min !== nothing]
        @test length(live) >= 4   # room for two buckets of >= 2 modes each

        @testset "nothing qualifies under MPI: empty vector, not an error" begin
            batches = Tarang.build_mode_batches!(solver.base, sps; is_gpu=false,
                                                 nprocs=2, like=ComplexF64[])
            @test isempty(batches)
        end

        @testset "two qualifying buckets come back sorted by sp_indices[1]" begin
            # Perturb the LAST TWO live modes identically (same structural
            # nonzero at the same position) so they split into their own
            # 2-mode bucket, leaving the rest (>= 2 modes) in the original
            # bucket — both then qualify under should_batch_modes. Verified by
            # direct inspection (fix-round report) that Dict's own, unsorted
            # iteration order over these two buckets is [8, 1], NOT [1, 8]:
            # this scenario only passes because build_mode_batches! sorts.
            odd1, odd2 = live[end - 1], live[end]
            original1, original2 = odd1.LHS, odd2.LHS
            target_row = findfirst(r -> original1[r, 1] == 0, 1:size(original1, 1))
            @test target_row !== nothing
            perturbed1 = copy(original1)
            perturbed1[target_row, 1] = 1.0 + 0.0im
            perturbed2 = copy(original2)
            perturbed2[target_row, 1] = 1.0 + 0.0im
            odd1.LHS = perturbed1
            odd2.LHS = perturbed2

            buckets = Tarang.bucket_subproblems(sps)
            @test length(buckets) == 2
            sizes = sort!(collect(length.(values(buckets))))
            @test sizes == [2, length(live) - 2]

            batches = Tarang.build_mode_batches!(solver.base, sps; is_gpu=false,
                                                 nprocs=1, like=ComplexF64[])
            @test length(batches) == 2
            firsts = [b.sp_indices[1] for b in batches]
            @test issorted(firsts)
            @test firsts[1] < firsts[2]

            odd1.LHS = original1
            odd2.LHS = original2
        end
    end
end

# ── Task 6: the batched stage loop ───────────────────────────────────────────

"""Identical, non-trivial initial condition, so parity is not a comparison of zeros."""
function _seed_parity_ic!(b)
    ensure_layout!(b, :g)
    gd = get_grid_data(b)
    for idx in CartesianIndices(gd)
        gd[idx] = sin(0.3 * sum(Tuple(idx))) + 0.1 * prod(Tuple(idx)) % 1
    end
    return b
end

@testset "ModeBatch cannot express a CSC matvec pattern" begin
    # `batched_spmv!` iterates ROWS. Task 2 shipped `M_min_colptr`/
    # `M_min_rowval` — the CSC pattern — with no caller; feeding those to it
    # computes `transpose(M_min)*x`, and `M_min` is not symmetric, so the
    # result would be wrong rather than merely differently ordered. The fields
    # are removed, not deprecated, so the mistake is unrepresentable.
    @test !hasfield(Tarang.ModeBatch, :M_min_colptr)
    @test !hasfield(Tarang.ModeBatch, :M_min_rowval)
    @test hasfield(Tarang.ModeBatch, :M_min_rowptr)
    @test hasfield(Tarang.ModeBatch, :M_min_colval)
    @test hasfield(Tarang.ModeBatch, :L_rowptr)
    @test hasfield(Tarang.ModeBatch, :L_colval)
end

@testset "batched M_min matvec reproduces the per-mode mul!" begin
    solver, _ = _parity_channel_solver(; batched_modes=true)
    step!(solver)
    sps = collect(solver.problem.compiled.subproblems)
    indices = first(values(Tarang.bucket_subproblems(sps)))
    batch = Tarang.build_mode_batch(sps, indices; like=ComplexF64[])

    n = batch.n
    # M_min is asymmetric here, which is what makes this test able to fail:
    # a CSC/CSR mix-up would silently apply the transpose.
    @test sps[indices[1]].M_min != permutedims(sps[indices[1]].M_min)

    X = ComplexF64[(0.3 * r - 0.11 * m) + (0.07 * r * m)im
                   for r in 1:n, m in 1:batch.nmodes]
    Y = zeros(ComplexF64, n, batch.nmodes)
    Tarang.batched_spmv!(Y, batch.M_min_rowptr, batch.M_min_colval,
                         batch.M_min_nzval, X)
    for (m, i) in enumerate(indices)
        @test Y[:, m] ≈ sps[i].M_min * X[:, m] atol=1e-12
    end

    # Same for L, which the stage loop applies through L_exp (uniform pattern)
    # rather than L_min (kx=0 stores fewer nonzeros).
    YL = zeros(ComplexF64, n, batch.nmodes)
    Tarang.batched_spmv!(YL, batch.L_rowptr, batch.L_colval, batch.L_nzval, X)
    for (m, i) in enumerate(indices)
        @test YL[:, m] ≈ sps[i].L_min * X[:, m] atol=1e-12
    end
end

@testset "gather starts are MEASURED, not extrapolated" begin
    # The subproblem tuple is not ordered by Fourier mode, so mode m's start
    # offset is not `starts[1] + (m-1)*stride`. Assert that directly: if a
    # future edit replaces the measured vector with an arithmetic guess, this
    # fails instead of silently gathering the wrong modes.
    solver, _ = _parity_channel_solver(; batched_modes=true)
    step!(solver)
    plan = Tarang.active_mode_batches(solver)
    @test !isempty(plan)

    cached = solver.timestepper_state.timestepper_data[:_sp_rk_mode_batches]
    ws = cached.workspaces[1]
    field_plan = ws.state_plan[1]                 # the 2-D Chebyshev field `b`
    starts = collect(field_plan.starts)
    @test length(starts) == cached.batches[1].nmodes
    stride1 = starts[2] - starts[1]
    @test any(m -> starts[m] != starts[1] + (m - 1) * stride1,
              3:length(starts))
    @test allunique(starts)
end

@testset "batches actually engaged during the run" begin
    solver, _ = _parity_channel_solver(; batched_modes=true)
    step!(solver)
    batches = Tarang.active_mode_batches(solver)
    @test !isempty(batches)
    @test sum(b -> b.nmodes, batches) ==
          count(sp -> sp.M_min !== nothing,
                solver.problem.compiled.subproblems)

    # ... and the default CPU solver does NOT engage them, which is the other
    # half of the claim: parity below would pass trivially if it did not.
    plain, _ = _parity_channel_solver()
    step!(plain)
    @test isempty(Tarang.active_mode_batches(plain))
end

@testset "batched stage loop reproduces the per-mode loop" begin
    nsteps = 5

    # INHOMOGENEOUS FIRST, and not optional. With `b(z=0) = 0` the algebraic
    # forcing is identically zero, so this comparison cannot see a broken BC
    # override at all — the override writes `coeff * 0` either way. `b(z=0) = 1`
    # makes `maxabs(ALG_F) == 16.0`, which is what gives the assertion its
    # teeth. The homogeneous case runs second so the zero path keeps coverage.
    for bc_low in ("1", "0")
        @testset "b(z=0) = $bc_low" begin
            ref_solver, ref_b = _parity_channel_solver(; bc_low, batched_modes=false)
            bat_solver, bat_b = _parity_channel_solver(; bc_low, batched_modes=true)

            _seed_parity_ic!(ref_b)
            _seed_parity_ic!(bat_b)

            for _ in 1:nsteps
                step!(ref_solver)
                step!(bat_solver)
            end

            @test isempty(Tarang.active_mode_batches(ref_solver))
            @test !isempty(Tarang.active_mode_batches(bat_solver))

            # The BC override path is only exercised when this is non-zero;
            # assert it directly rather than trusting the BC string.
            bat_ws = bat_solver.timestepper_state.timestepper_data[:_sp_rk_mode_batches].workspaces[1]
            alg_max = maximum(abs, Array(bat_ws.ALG_F))
            if bc_low == "1"
                @test alg_max > 1.0
            else
                @test alg_max == 0.0
            end

            ensure_layout!(ref_b, :g)
            ensure_layout!(bat_b, :g)
            ref_g = Array(get_grid_data(ref_b))
            bat_g = Array(get_grid_data(bat_b))

            scale = maximum(abs, ref_g)
            @test scale > 1e-8               # guard against comparing zeros
            @test maximum(abs, bat_g .- ref_g) / scale < 1e-12
        end
    end
end

@testset "batched stage loop reproduces the per-mode loop: time-dependent BC" begin
    # `step_subproblem_rk_batched.jl`'s `bc_dynamic` branch — the per-stage
    # `_refresh_bcs_for_stage!` plus ALG_F re-gather at `t + c[i]*dt` — never
    # ran in any other test, because every batching problem here had static
    # BCs. It is also the branch a GPU run reaches by default. dt is 1e-2 so
    # the boundary value moves appreciably WITHIN a step (the stage times are
    # c = [0, 0.293, 1]), which is what makes a dropped per-stage re-gather
    # visible rather than a rounding difference.
    nsteps = 6
    bc = "sin(6.283185307*t)"

    ref_solver, ref_b = _parity_channel_solver(; bc_low=bc, dt=1e-2,
                                                 batched_modes=false)
    bat_solver, bat_b = _parity_channel_solver(; bc_low=bc, dt=1e-2,
                                                 batched_modes=true)
    @test Tarang.has_time_dependent_bcs(bat_solver.problem.bc_manager)

    _seed_parity_ic!(ref_b)
    _seed_parity_ic!(bat_b)

    for _ in 1:nsteps
        step!(ref_solver)
        step!(bat_solver)
    end

    @test !isempty(Tarang.active_mode_batches(bat_solver))
    bat_ws = bat_solver.timestepper_state.timestepper_data[:_sp_rk_mode_batches].workspaces[1]
    @test maximum(abs, Array(bat_ws.ALG_F)) > 1e-3   # the BC is live, not 0

    ensure_layout!(ref_b, :g)
    ensure_layout!(bat_b, :g)
    ref_g = Array(get_grid_data(ref_b))
    bat_g = Array(get_grid_data(bat_b))

    scale = maximum(abs, ref_g)
    @test scale > 1e-8
    @test maximum(abs, bat_g .- ref_g) / scale < 1e-12
end

@testset "assemble and factor calls stay paired across a dt change" begin
    # `batched_factor!` must never run without an immediately preceding
    # `batched_assemble_lhs!` over the same buffer: the GPU
    # `getrf_strided_batched!` overwrites `A` with the LU factors, while the CPU
    # `lu(view(A,:,:,m))` copies. Factoring twice without re-assembling
    # therefore works on CPU and factors the factors on GPU — a plausible wrong
    # answer with no error. Counting is the only way to see the pairing at
    # runtime; the calls are adjacent in `_ensure_batch_factored!` and nowhere
    # else in `src/`.
    Tarang.reset_batch_factor_stats!()
    solver, b = _parity_channel_solver(; batched_modes=true)
    _seed_parity_ic!(b)

    for _ in 1:3
        step!(solver)
    end
    after_fixed_dt = Tarang.BATCH_FACTOR_STATS.factors[]
    @test after_fixed_dt >= 1
    @test Tarang.BATCH_FACTOR_STATS.assembles[] == after_fixed_dt
    # ESDIRK: one factorization serves every implicit stage AND every step at
    # constant dt.
    @test after_fixed_dt == 1

    for _ in 1:2
        step!(solver, 2e-3)
    end
    @test Tarang.BATCH_FACTOR_STATS.factors[] > after_fixed_dt   # dt change refactored
    @test Tarang.BATCH_FACTOR_STATS.assembles[] ==
          Tarang.BATCH_FACTOR_STATS.factors[]

    # And structurally: exactly one call site each, so the counters above
    # cannot be equal merely because a second pair exists somewhere else.
    #
    # Scans EVERY .jl file under src/ and ext/ with NO file excluded. The first
    # version of this test excluded the definition site by bare basename
    # (`f != "batched_matsolvers.jl"`), which silently skipped
    # `ext/cuda/batched_matsolvers.jl` as well — the GPU specialization, and the
    # single most likely home for a GPU-only violation of the assemble-then-
    # factor lifecycle, with no GPU CI to catch it any other way. A guard with a
    # blind spot had reproduced the blind spot one level down.
    #
    # Instead of excluding files, `_call_lines` excludes by what a line IS: a
    # `function ` definition, or any line inside a `"""` docstring block (both
    # definition files echo their own signature in a docstring). That needs no
    # maintenance when files move and leaves nothing unscanned.
    function _call_lines(path::AbstractString, needle::AbstractString)
        n = 0
        in_docstring = false
        for line in eachline(path)
            # An ODD number of `"""` on a line flips the docstring state; an
            # even number (a one-line docstring) leaves it where it was.
            if isodd(count("\"\"\"", line))
                in_docstring = !in_docstring
                continue
            end
            in_docstring && continue
            occursin(needle, line) || continue
            startswith(lstrip(line), "function ") && continue   # the definition
            n += 1
        end
        return n
    end

    repo = dirname(@__DIR__)
    factor_sites = Tuple{String, Int}[]
    assemble_sites = Tuple{String, Int}[]
    for root in (joinpath(repo, "src"), joinpath(repo, "ext"))
        isdir(root) || continue
        for (dir, _, files) in walkdir(root), f in files
            endswith(f, ".jl") || continue
            path = joinpath(dir, f)
            rel = relpath(path, repo)
            nf = _call_lines(path, "batched_factor!(")
            nf > 0 && push!(factor_sites, (rel, nf))
            na = _call_lines(path, "batched_assemble_lhs!(")
            na > 0 && push!(assemble_sites, (rel, na))
        end
    end
    target = joinpath("src", "core", "timesteppers",
                      "step_subproblem_rk_batched.jl")
    @test sort(factor_sites) == [(target, 1)]
    @test sort(assemble_sites) == [(target, 1)]
end

@testset "the dirty bit alone forces a refactor" begin
    # I3: the two-part gate is `!dirty[] && factored_key[] == (dt, a_ii)`, but
    # the only production writer of `dirty[] = true` is the dt handler, which
    # ALSO changes the key — so no end-to-end scenario separates the two, and
    # rewriting the `&&` as `||` left every other test green. Drive the missing
    # half directly: dirty with an UNCHANGED key must still re-assemble and
    # re-factor. Under `||` the gate would return the cached factorization and
    # the counters would not move, failing the first two assertions below.
    Tarang.reset_batch_factor_stats!()
    solver, b = _parity_channel_solver(; bc_low="1", batched_modes=true)
    _seed_parity_ic!(b)
    step!(solver)

    batches = Tarang.active_mode_batches(solver)
    @test length(batches) == 1
    batch = batches[1]
    key = batch.factored_key[]
    @test !batch.dirty[]                       # a completed step leaves it clean

    n_f = Tarang.BATCH_FACTOR_STATS.factors[]
    n_a = Tarang.BATCH_FACTOR_STATS.assembles[]

    # Same key, dirty set by hand.
    batch.dirty[] = true
    Tarang._ensure_batch_factored!(batch, key[1], key[2])
    @test Tarang.BATCH_FACTOR_STATS.factors[] == n_f + 1
    @test Tarang.BATCH_FACTOR_STATS.assembles[] == n_a + 1
    @test batch.factored_key[] == key
    @test !batch.dirty[]

    # Clean and same key: no work, which is the other half of the gate.
    Tarang._ensure_batch_factored!(batch, key[1], key[2])
    @test Tarang.BATCH_FACTOR_STATS.factors[] == n_f + 1
    @test Tarang.BATCH_FACTOR_STATS.assembles[] == n_a + 1

    # The batch is still usable: re-assemble+factor of the same matrix is a
    # no-op numerically, so stepping on must neither refactor nor blow up.
    step!(solver)
    @test Tarang.BATCH_FACTOR_STATS.factors[] == n_f + 1
    ensure_layout!(b, :g)
    @test all(isfinite, Array(get_grid_data(b)))
end

@testset "leftover modes step alongside a partial batch" begin
    # A uniform Fourier problem puts every mode in one bucket, so the leftover
    # path — the modes a declining bucket leaves on the per-mode loop — is
    # unreachable from a normal run and would rot untested. Install a plan that
    # batches all but the last mode and check parity anyway.
    nsteps = 4
    ref_solver, ref_b = _parity_channel_solver(; bc_low="1", batched_modes=false)
    bat_solver, bat_b = _parity_channel_solver(; bc_low="1", batched_modes=true)
    _seed_parity_ic!(ref_b)
    _seed_parity_ic!(bat_b)

    # One step first: `solver.timestepper_state` is `nothing` until then, and
    # the plan has to be installed on it.
    step!(ref_solver)
    step!(bat_solver)

    sps = bat_solver.problem.compiled.subproblems
    indices = first(values(Tarang.bucket_subproblems(collect(sps))))
    @test length(indices) >= 3

    state = bat_solver.timestepper_state
    state_fields = state.timestepper_data[:_sp_state_fields][2]
    for f in state_fields
        ensure_layout!(f, :c)
    end
    partial = Tarang.build_mode_batch(sps, indices[1:(end - 1)]; like=ComplexF64[])
    plan = Tarang._build_batched_rk_plan(bat_solver, sps, state_fields;
                                         batches=[partial])
    @test plan !== nothing
    @test plan.leftovers == [indices[end]]

    state.timestepper_data[:_sp_rk_mode_batches] = plan
    state.timestepper_data[:_sp_rk_mode_batches_key] = sps

    for _ in 1:(nsteps - 1)
        step!(ref_solver)
        step!(bat_solver)
    end

    engaged = Tarang.active_mode_batches(bat_solver)
    @test length(engaged) == 1
    @test engaged[1].nmodes == length(indices) - 1

    ensure_layout!(ref_b, :g)
    ensure_layout!(bat_b, :g)
    ref_g = Array(get_grid_data(ref_b))
    bat_g = Array(get_grid_data(bat_b))
    scale = maximum(abs, ref_g)
    @test scale > 1e-8
    @test maximum(abs, bat_g .- ref_g) / scale < 1e-12
end
