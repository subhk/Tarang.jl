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
        solver, _ = _parity_channel_solver(; batched_modes=true)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices[1:1];
                                         is_gpu=false, nprocs=1)
    end

    @testset "exceeding the byte cap declines and says so" begin
        solver, _ = _parity_channel_solver(; batched_modes=true,
                                             batched_modes_max_bytes=1)
        step!(solver)
        sps = collect(solver.problem.compiled.subproblems)
        indices = first(values(Tarang.bucket_subproblems(sps)))
        @test !Tarang.should_batch_modes(solver.base, sps, indices;
                                         is_gpu=false, nprocs=1)
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
end
