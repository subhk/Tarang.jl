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
function _parity_channel_solver(; nx=16, nz=8, dt=1e-3, device=Tarang.CPU(), kwargs...)
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
        solver, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        batch = Tarang.build_mode_batch(sps, indices; like=ComplexF64[])

        @test Tarang.mode_batch_bytes(batch.n, batch.nmodes) ==
              sizeof(batch.lhs_dense)
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

    ref_solver, ref_b = _parity_channel_solver(; batched_modes=false)
    bat_solver, bat_b = _parity_channel_solver(; batched_modes=true)

    _seed_parity_ic!(ref_b)
    _seed_parity_ic!(bat_b)

    for _ in 1:nsteps
        step!(ref_solver)
        step!(bat_solver)
    end

    @test isempty(Tarang.active_mode_batches(ref_solver))
    @test !isempty(Tarang.active_mode_batches(bat_solver))

    ensure_layout!(ref_b, :g)
    ensure_layout!(bat_b, :g)
    ref_g = Array(get_grid_data(ref_b))
    bat_g = Array(get_grid_data(bat_b))

    scale = maximum(abs, ref_g)
    @test scale > 1e-8                       # guard against comparing zeros
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

    # And structurally: exactly one call site each, in one function, so the
    # counters above cannot be equal merely because a second pair exists
    # somewhere else.
    ts_dir = joinpath(dirname(@__DIR__), "src", "core", "timesteppers")
    calls = 0
    assembles = 0
    for f in readdir(ts_dir; join=true)
        endswith(f, ".jl") || continue
        src = read(f, String)
        calls += count("batched_factor!(", src)
        assembles += count("batched_assemble_lhs!(", src)
    end
    @test calls == 1
    @test assembles == 1
end

@testset "leftover modes step alongside a partial batch" begin
    # A uniform Fourier problem puts every mode in one bucket, so the leftover
    # path — the modes a declining bucket leaves on the per-mode loop — is
    # unreachable from a normal run and would rot untested. Install a plan that
    # batches all but the last mode and check parity anyway.
    nsteps = 4
    ref_solver, ref_b = _parity_channel_solver(; batched_modes=false)
    bat_solver, bat_b = _parity_channel_solver(; batched_modes=true)
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
