"""
Bucketing tests for the batched Fourier-mode solve.

`batch_signature` must be computed from the matrices as actually built. If it
were derived from `nz`/`nvars` arithmetic instead, a problem whose kx=0 mode
carries a different BC or gauge constraint would be batched together with the
rest and silently solve the wrong system.
"""

using Test
using Tarang
using SparseArrays

function _channel_solver(; nx=16, nz=8, dt=1e-3)
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
    solver = InitialValueSolver(problem, RK222(); dt)
    step!(solver)   # forces build_matrices!
    return solver
end

@testset "mode batch signature" begin
    solver = _channel_solver()
    sps = collect(solver.problem.compiled.subproblems)
    live = [sp for sp in sps if sp.M_min !== nothing]
    @test length(live) > 1

    @testset "uniform problem yields exactly one bucket" begin
        buckets = Tarang.bucket_subproblems(sps)
        @test length(buckets) == 1
        only_bucket = first(values(buckets))
        @test length(only_bucket) == length(live)
        @test issorted(only_bucket)
    end

    @testset "signature is stable and value-independent" begin
        sig1 = Tarang.batch_signature(live[1])
        sig2 = Tarang.batch_signature(live[2])
        @test sig1 == sig2
        @test sig1 != 0x0
        # nzval differs across modes but must NOT change the signature
        @test live[1].LHS.nzval != live[2].LHS.nzval
    end

    @testset "a perturbed pattern splits into its own bucket" begin
        # Give one mode a structurally different LHS. Signature must change,
        # and bucketing must isolate it rather than batching it with the rest.
        odd = live[end]
        original = odd.LHS
        perturbed = copy(original)
        # Add a structural nonzero where there was none.
        target_row = findfirst(r -> perturbed[r, 1] == 0, 1:size(perturbed, 1))
        @test target_row !== nothing
        perturbed[target_row, 1] = 1.0 + 0.0im
        odd.LHS = perturbed

        @test Tarang.batch_signature(odd) != Tarang.batch_signature(live[1])
        buckets = Tarang.bucket_subproblems(sps)
        @test length(buckets) == 2
        sizes = sort!(collect(length.(values(buckets))))
        @test sizes == [1, length(live) - 1]

        odd.LHS = original
    end

    @testset "kx=0 batches with everyone else" begin
        # Regression pin. `L_min` at kx=0 stores FEWER nonzeros than at other
        # modes (the ∂xx term is the zero operator there), so a signature built
        # over `L_min` splits kx=0 into its own bucket on essentially every
        # problem with a second derivative. The signature uses `L_exp` — same
        # values, LHS's union pattern, uniform across all modes.
        zero_mode = findfirst(sp -> sp.group[1] == 0, live)
        @test zero_mode !== nothing
        other = findfirst(sp -> sp.group[1] != 0, live)

        @test nnz(live[zero_mode].L_min) != nnz(live[other].L_min)   # they DO differ
        @test nnz(live[zero_mode].L_exp) == nnz(live[other].L_exp)   # L_exp does not
        @test Matrix(live[zero_mode].L_exp) == Matrix(live[zero_mode].L_min)

        @test Tarang.batch_signature(live[zero_mode]) ==
              Tarang.batch_signature(live[other])
    end

    @testset "an unbuilt subproblem is not batchable" begin
        sp = live[1]
        saved = sp.M_min
        sp.M_min = nothing
        @test Tarang.batch_signature(sp) == 0x0
        sp.M_min = saved
    end
end
