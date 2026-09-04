# Guards on MPI transform planning (2026-08-20 MPI review, findings T1/T2/T3):
#
# T1: the MPI plan-reuse guard keyed only on the grid SHAPE — a mixed
#     Cheb×Fourier domain with the same gshape as an earlier pure-Fourier one
#     silently reused the pure-Fourier plan: the Chebyshev axis got FFT'd, no
#     DCT was registered, `pencil_solve` was never built. The guard now also
#     compares the (basis type, size) signature per axis.
# T2: replanning did not clear a stale `dist.pencil_solve`, so a mixed →
#     pure-Fourier replan left downstream predicates (`pencil_solve !==
#     nothing`) treating a diagonal pure-Fourier solve as a coupled mixed one.
# T3: ChebyshevU/Ultraspherical/generic Jacobi axes fell into NO classification
#     bucket under MPI: routed as "pure Fourier", coupled axis silently got
#     NoTransform (grid values consumed as spectral coefficients — serial DOES
#     register the transform). They now refuse loudly, like Legendre.
using Tarang
using MPI
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI transform-planning guard test requires >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

@testset "MPI plan reuse keys on basis signature (rank=$rank)" begin
    N = 16
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    fb1 = ComplexFourier(coords["z"]; size=N, bounds=(0.0, 2π))
    fb2 = ComplexFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    dom_fourier = Domain(dist, (fb1, fb2))
    Tarang.plan_transforms!(dist, dom_fourier)
    @test dist.pencil_fft_input !== nothing
    @test dist.pencil_solve === nothing
    @test dist.plan_basis_signature == ((:ComplexFourier, N), (:ComplexFourier, N))

    # Same gshape (N, N), different composition: must NOT reuse the plan.
    zb = ChebyshevT(coords["z"]; size=N, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    dom_mixed = Domain(dist, (zb, xb))
    Tarang.plan_transforms!(dist, dom_mixed)
    @test any(t -> t isa Tarang.ChebyshevTransform, dist.transforms)   # T1
    @test dist.pencil_solve !== nothing                                # T1
    @test dist.plan_basis_signature == ((:ChebyshevT, N), (:RealFourier, N))

    # Replan back to pure Fourier: the stale solve pencil must be dropped.
    Tarang.plan_transforms!(dist, dom_fourier)
    @test dist.pencil_solve === nothing                                # T2
    @test !any(t -> t isa Tarang.ChebyshevTransform, dist.transforms)

    # Same signature: reuse (plan object unchanged).
    plan_before = dist.pencil_fft_plan
    Tarang.plan_transforms!(dist, dom_fourier)
    @test dist.pencil_fft_plan === plan_before
end

@testset "unsupported Jacobi bases refuse loudly under MPI (rank=$rank)" begin
    N = 16
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    ub = ChebyshevU(coords["z"]; size=N, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    # Domain construction itself triggers transform planning, so the refusal
    # fires here already.
    err = try
        dom = Domain(dist, (ub, xb))
        Tarang.plan_transforms!(dist, dom)
        nothing
    catch e
        e
    end
    @test err isa ErrorException                                       # T3
    @test occursin("ChebyshevU", sprint(showerror, err))
end

@testset "1D MPI basis validation matches Domain refusal (rank=$rank)" begin
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
    validation_err = try
        Tarang.validate_mpi_fourier_only((basis,), nprocs; use_pencil_arrays=true)
        nothing
    catch e
        e
    end
    @test validation_err isa ErrorException
    @test occursin("not supported for 1D", sprint(showerror, validation_err))

    domain_err = try
        Domain(dist, (basis,))
        nothing
    catch e
        e
    end
    @test domain_err isa ErrorException
    @test occursin("not supported for 1D", sprint(showerror, domain_err))
    @test isempty(dist.transforms)
    @test dist.pencil_fft_plan === nothing
end
