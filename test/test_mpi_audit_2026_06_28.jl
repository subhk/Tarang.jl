# Regression guards for the MPI CPU correctness audit (2026-06-28).
#
# Bug A — dealias truncation low-pass cutoff was np-dependent. The distributed
#   path (_apply_spectral_cutoff_distributed!) caps the 2/3-rule cutoff at the
#   alias-safe (N-1)÷3, but the serial path (apply_spectral_cutoff!) used the
#   uncapped floor(N/2f), so for any Fourier axis with N % 3 == 0 the two kept
#   DIFFERENT bands (e.g. N=12, f=3/2: serial keeps |k|≤4, distributed keeps
#   |k|≤3). Same apply_basic_dealiasing! call on the same input → serial identity,
#   distributed wiped. Now both go through _axis_dealias_cutoff → identical band.
#
# Bug B — apply_forcing! (the exported manual-time-loop API) passed size(rhs)
#   (an NTuple) to _matched_forcing_view, routing to the offset-BLIND NTuple
#   method (ranges = 1:local_n) instead of the PencilArray method that slices the
#   rank's axes_local. So every rank injected forcing from global modes 1:local_n
#   → wrong wavenumbers on rank>0. Now it passes rhs itself so the PencilArray
#   method (offset-correct) is selected under MPI.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

@testset "MPI CPU audit 2026-06-28 (np=$nprocs)" begin

    # --- Bug A: dealias truncation cutoff is alias-safe AND np-independent ---
    @testset "dealias truncation cutoff alias-safe + np-independent" begin
        N = 12  # N % 3 == 0 → the missing (N-1)÷3 cap made serial≠distributed
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
        bx = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
        by = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
        domain = Domain(dist, (bx, by))
        xg = [2π * (i - 1) / N for i in 1:N]

        # max|grid| after low-pass dealiasing a single x-mode cos(k·x), constant in y.
        function dealias_amp(kmode)
            f = ScalarField(domain, "f")
            ensure_layout!(f, :g)
            gdata = [cos(kmode * xg[i]) for i in 1:N, _ in 1:N]
            d = get_grid_data(f)
            if d isa PencilArrays.PencilArray
                ax = PencilArrays.pencil(d).axes_local
                parent(d) .= gdata[ax...]
            else
                d .= gdata
            end
            Tarang.apply_basic_dealiasing!(f, 1.5)
            ensure_layout!(f, :g)
            d2 = get_grid_data(f)
            loc = d2 isa PencilArrays.PencilArray ? parent(d2) : d2
            MPI.Allreduce(maximum(abs.(loc)), MPI.MAX, comm)
        end

        # cutoff = min(floor(N/3)=4, (N-1)÷3=3) = 3 on EVERY np:
        @test dealias_amp(4) < 1e-9   # |k|=4 (alias-unsafe) removed — was KEPT at np=1
        @test dealias_amp(3) > 0.5    # |k|=3 (retained band) preserved on every np
    end

    # --- Bug B: apply_forcing! manual API lands forcing at correct wavenumbers ---
    @testset "apply_forcing! places forcing at correct global wavenumbers" begin
        N = 5
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = ComplexFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        f = ScalarField(dist, "u", (xb, yb), ComplexF64)
        ensure_layout!(f, :c)
        cd = get_coeff_data(f)

        # Known global spectrum in LOGICAL order encoding each wavenumber's coords.
        G = ComplexF64[i + 1000im * j for i in 1:N, j in 1:N]
        forcing = Tarang.StochasticForcing(; field_size=(N, N), k_forcing=2.0, dk_forcing=1.0)
        forcing.cached_forcing = copy(G)

        fill!(cd, 0)
        # substep=2 → generate_forcing! returns the cached G unchanged (no regen),
        # so this exercises ONLY apply_forcing!'s view dispatch + broadcast add.
        Tarang.apply_forcing!(cd, forcing, 0.0, 2)

        g = isa(cd, PencilArrays.PencilArray) ? PencilArrays.gather(cd) : copy(cd)
        if rank == 0
            @test maximum(abs.(ComplexF64.(g) .- G)) < 1e-10
        end
    end

    # --- Bug C: distributed mixed Fourier–Chebyshev direct :c applies the Cheb DCT ---
    # The PencilFFT plan transforms ONLY the Fourier axis (Chebyshev = NoTransform,
    # and is the decomposed axis), so forward_transform! used to leave the coupled
    # axis in GRID space → direct get_coeff_data returned un-DCT'd (spread) data,
    # while the serial path (and solves) were spectral. forward/backward_transform!
    # now apply the coupled DCT via the solve pencil. We assert the :c spectrum of a
    # smooth field is CONCENTRATED (Chebyshev DCT applied) on every np, and that the
    # forward→backward round-trip is the identity.
    @testset "distributed mixed Cheb-Fourier :c is Chebyshev-spectral" begin
        Nx = 8
        Nz = 16
        coords = CartesianCoordinates("z", "x")  # Cheb FIRST (required): x decomposed, z local
        dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
        bz = Chebyshev(coords["z"]; size=Nz, bounds=(-1.0, 1.0))
        bx = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
        domain = Domain(dist, (bz, bx))

        f = ScalarField(domain, "f")
        ensure_layout!(f, :g)
        zline = collect(get_grid_coordinates(domain)["z"])
        gd = get_grid_data(f)
        gloc = gd isa PencilArrays.PencilArray ? parent(gd) : gd
        g0 = exp.(reshape(zline, Nz, 1)) .* ones(1, size(gloc, 2))  # f = exp(z), const in x
        gloc .= g0

        ensure_layout!(f, :c)
        cd = get_coeff_data(f)
        cloc = cd isa PencilArrays.PencilArray ? parent(cd) : cd
        # Participation ratio PR = (Σ|c|²)² / Σ|c|⁴ — permutation/scale-invariant.
        # exp(z) Cheb spectrum is concentrated (PR ≈ 2.1); un-DCT'd nodal data PR ≈ 7.3.
        s2 = MPI.Allreduce(sum(abs2, cloc), MPI.SUM, comm)
        s4 = MPI.Allreduce(sum(x -> abs2(x)^2, cloc), MPI.SUM, comm)
        PR = s2^2 / s4
        @test PR < 3.5   # Cheb DCT applied (was ≈7.3 = nodal/spread under MPI)

        # forward (already :c) → backward → grid must recover the input.
        ensure_layout!(f, :g)
        gd2 = get_grid_data(f)
        gloc2 = gd2 isa PencilArrays.PencilArray ? parent(gd2) : gd2
        rt = MPI.Allreduce(maximum(abs.(gloc2 .- g0)), MPI.MAX, comm)
        @test rt < 1e-10
    end

    # --- Bug D (H1): over-decomposed mixed solve fails LOUD, not crash/deadlock ---
    # When nprocs > #Fourier coeff modes, some rank owns ZERO Fourier modes →
    # build_subsystems returned a global-fallback subsystem for it while mode-owning
    # ranks built per-mode subproblems. The per-mode Chebyshev tau solve issues a
    # collective solve-layout transpose the zero-mode rank cannot match → the solve
    # crashed (DimensionMismatch) / could deadlock. build_subsystems now Allreduces
    # the empty flag and errors UNIFORMLY on every rank with a clear message.
    @testset "over-decomposed mixed Cheb-Fourier solve errors uniformly" begin
        Nz = 8
        Nx = 4                       # RealFourier → Nx/2+1 = 3 Fourier coeff modes
        n_modes = Nx ÷ 2 + 1
        coords = CartesianCoordinates("z", "x")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 2.0))
        xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
        dom = Domain(dist, (zb, xb))
        lb = derivative_basis(zb, 1)
        u    = ScalarField(dom, "u")
        tau1 = ScalarField(dist, "tau1", (), Float64)
        tau2 = ScalarField(dist, "tau2", (), Float64)
        prob = Tarang.LBVP([u, tau1, tau2])
        add_parameters!(prob; Lz=2.0, l1=lift(tau1, lb, -1), l2=lift(tau2, lb, -2))
        Tarang.add_equation!(prob, "Δ(u) + l1 + l2 = -2")
        Tarang.add_bc!(prob, "u(z=0) = 0")
        Tarang.add_bc!(prob, "u(z=2.0) = 0")

        if nprocs > n_modes
            # over-decomposed: clear ErrorException on EVERY rank (no DimensionMismatch/hang).
            @test_throws ErrorException Tarang.BoundaryValueSolver(prob)
        else
            # not over-decomposed: construction + solve succeed (no false-positive guard).
            solver = Tarang.BoundaryValueSolver(prob)
            Tarang.solve!(solver)
            ensure_layout!(u, :g)
            @test true
        end
    end
end
