"""
Engagement guards and end-to-end parity for the batched Fourier-mode solve.

The guards matter as much as the parity: batching must be OFF by default on CPU
and must never construct under MPI, so that every existing run is byte-for-byte
unchanged. `batched_modes=true` is what the suite uses to exercise the real
device-generic code path without a GPU.
"""

using Test
using Tarang

function _parity_channel_solver(; nx=16, nz=8, dt=1e-3, kwargs...)
    coords = CartesianCoordinates("x", "z")
    dist = Distributor(coords; dtype=Float64, device=CPU())
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
