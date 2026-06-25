# Guard: distributed Chebyshev-Fourier STEADY solvers (LBVP + NLBVP) at np>=2.
#
# The IVP subproblem steppers bracket every per-mode coeff gather/scatter with the
# Cheb-Fourier solve-layout transpose (to_solve_layout!/from_solve_layout!), but
# the steady BVP/NLBVP solvers in solver_stepping.jl did NOT — so the per-mode
# gather indexed the PencilFFT-output pencil (Chebyshev axis decomposed, Fourier
# axis local) with solve-pencil index logic and produced a DimensionMismatch (or
# wrong coefficients) at np>=2. The NLBVP Newton loop additionally deadlocked once
# the transpose was added because each rank's residual norm only covered its local
# Fourier modes (different break iteration ⇒ mismatched collective count); the norm
# is now Allreduced. Round-2 MPI CPU audit 2026-06-23.
#
# Manufactured Poisson Δu + lift(τ) = -2 (LBVP) / = u² + g (NLBVP), u(0)=u(Lz)=0
# ⇒ u = z(Lz - z), x-independent. Serial reference: sumsq=21.0, max≈0.950484.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "Distributed Cheb-Fourier BVP test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const Lz = 2.0
const Nz = 8
const Nx = 8
const SUMSQ_REF = 21.0
const MAX_REF = 0.9504844339512096

_gather(u) = (gd = get_grid_data(u);
              gd isa PencilArrays.PencilArray ? PencilArrays.gather(gd) :
              Array(Tarang.get_cpu_data(gd)))

@testset "Distributed Cheb-Fourier steady solvers (np=$nprocs)" begin
    coords = CartesianCoordinates("z", "x")           # Chebyshev FIRST (MPI-supported)
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    dom = Domain(dist, (zb, xb))
    lb  = derivative_basis(zb, 1)

    @testset "Linear BVP matches serial" begin
        u    = ScalarField(dom, "u")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        prob = Tarang.LBVP([u, tau1, tau2])
        add_parameters!(prob; Lz=Lz, l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2))
        Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = -2")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=Lz) = 0")
        solver = Tarang.BoundaryValueSolver(prob)
        Tarang.solve!(solver)
        ensure_layout!(u, :g)
        g = _gather(u)
        if rank == 0
            xmean = sum(g, dims=2) ./ size(g, 2)
            @test maximum(abs.(g .- xmean)) < 1e-9        # x-independent
            @test isapprox(sum(abs2, g), SUMSQ_REF; rtol=1e-8)
            @test isapprox(maximum(g), MAX_REF; rtol=1e-7)
        end
    end

    @testset "EVP eigenvalues match serial (global spectrum gather)" begin
        # Diffusion dt(u)=Δu ⇒ σ_{kx,n} = -(kx² + (nπ/Lz)²). Each rank's subproblems
        # cover only its local Fourier modes, so the smallest-|σ| selection must
        # gather every rank's eigenvalues; otherwise np>=2 returns local-subset
        # extrema (e.g. spurious -88/-89 from a high-kx rank) instead of the global
        # set. Serial reference for Nz=$Nz, Nx=$Nx, Lz=$Lz.
        SER = [-11.4674, -10.8706, -9.8706, -6.4674, -3.4674, -2.4674]
        u    = ScalarField(dom, "u")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        prob = Tarang.EVP([u, tau1, tau2]; eigenvalue=:σ)
        add_parameters!(prob; Lz=Lz, l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2))
        Tarang.add_equation!(prob, "dt(u) - Δ(u) - l1 - l2 = 0")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=Lz) = 0")
        solver = Tarang.EigenvalueSolver(prob; nev=6, which=:SM)
        λ, _ = Tarang.solve!(solver)
        if rank == 0
            got = sort(real.(λ))
            @test length(got) == 6
            @test maximum(abs.(got .- SER)) < 1e-3
        end
    end

    @testset "Nonlinear BVP matches serial (Newton)" begin
        u    = ScalarField(dom, "u")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        g    = ScalarField(dom, "g"); ensure_layout!(g, :g)
        zfull = [Lz/2*(1-cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
        gglob = [-2 - (zfull[iz]*(Lz-zfull[iz]))^2 for iz in 1:Nz, ix in 1:Nx]
        gd0 = get_grid_data(g)
        if gd0 isa PencilArrays.PencilArray
            gv = PencilArrays.global_view(gd0)
            for I in CartesianIndices(gv); gv[I] = gglob[I]; end
        else
            Tarang.get_cpu_data(gd0) .= gglob
        end
        prob = Tarang.NLBVP([u, tau1, tau2])
        add_parameters!(prob; Lz=Lz, l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2), g=g)
        Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = u*u + g")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=Lz) = 0")
        solver = Tarang.BoundaryValueSolver(prob)
        Tarang.solve!(solver)
        ensure_layout!(u, :g)
        gu = _gather(u)
        if rank == 0
            xmean = sum(gu, dims=2) ./ size(gu, 2)
            @test maximum(abs.(gu .- xmean)) < 1e-8       # x-independent
            @test isapprox(sum(abs2, gu), SUMSQ_REF; rtol=1e-7)
            @test isapprox(maximum(gu), MAX_REF; rtol=1e-6)
        end
    end
end
