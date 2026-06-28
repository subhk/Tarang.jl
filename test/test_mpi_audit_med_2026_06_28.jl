# Regression guards for the MED-severity MPI CPU audit batch (2026-06-28).
#
# M4 — LES eddy-viscosity diagnostics (mean/max/sgs) reduced only the per-rank slab
#   under MPI (the model holds no communicator), so they returned per-rank values
#   instead of the global mean/max. Now reduced over COMM_WORLD (Σs/Σn for means,
#   MAX for max) — correct for decomposed OR replicated slabs.
#
# M3 — VirtualFileHandler errored on complex fields (`nccreate(t=ComplexF64)` is
#   unsupported, and the merge allocated Float64). `_local_task_data` now encodes a
#   complex field as a real array with a trailing size-2 [real, imag] axis, so the
#   write + offset merge stay real and round-trip exactly.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays
using NetCDF

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

@testset "MPI CPU audit MED 2026-06-28 (np=$nprocs)" begin

    # --- M4: LES diagnostics reduce globally under MPI ---
    @testset "LES eddy-viscosity diagnostics are global, not per-rank" begin
        model = SmagorinskyModel(C_s=0.1, filter_width=(1.0, 1.0), field_size=(4, 4))
        # Each rank holds a distinct uniform slab: rank r → all entries (r+1).
        fill!(model.eddy_viscosity, Float64(rank + 1))
        # Equal slab sizes ⇒ global mean = mean over ranks of (r+1) = avg(1..nprocs).
        expected_mean = sum(1:nprocs) / nprocs
        @test isapprox(mean_eddy_viscosity(model), expected_mean; rtol=1e-12)
        @test max_eddy_viscosity(model) == Float64(nprocs)          # global max
        # sgs mean with strain ≡ 1 ⇒ mean(νₑ·1) = same global mean.
        strain = ones(Float64, 4, 4)
        @test isapprox(mean_sgs_dissipation(model, strain), expected_mean; rtol=1e-12)
    end

    # --- M3: VirtualFileHandler round-trips a COMPLEX field (real/imag encoding) ---
    @testset "VirtualFileHandler complex field round-trips" begin
        pid0 = (r = Ref(Int(getpid())); MPI.Bcast!(r, 0, comm); r[])
        outdir = joinpath(tempdir(), "tarang_vfh_cplx_np$(nprocs)_$(pid0)")
        rank == 0 && mkpath(outdir)
        MPI.Barrier(comm)

        N = 8
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = ComplexFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb, yb), ComplexF64)

        ref = [ComplexF64(gi * 10 + gj, gi - gj) for gi in 1:N, gj in 1:N]
        ensure_layout!(u, :g)
        data = get_grid_data(u)
        if isa(data, PencilArrays.PencilArray)
            gv = PencilArrays.global_view(data)
            for I in CartesianIndices(gv); gv[I] = ref[I]; end
        else
            data .= ref
        end

        problem = IVP([u]); add_equation!(problem, "∂t(u) = 0")
        solver = InitialValueSolver(problem, RK222(); dt=0.01)
        vfh = VirtualFileHandler(outdir, "vcplx"; comm=comm, cadence=1)
        Tarang.add_task!(vfh, u, "u")
        Tarang.process!(vfh, solver, 0.0, 0.0, 1)   # used to throw on complex
        MPI.Barrier(comm)

        @test isfile(joinpath(outdir, "vcplx_s1_p$(rank).nc"))
        if rank == 0
            merged_file = Tarang.merge_virtual!(vfh; set_num=1)
            m = ncread(merged_file, "u")                 # real, trailing size-2 axis
            @test size(m) == (N, N, 2)
            recon = complex.(m[:, :, 1], m[:, :, 2])
            @test maximum(abs.(recon .- ref)) < 1e-10
        end
        MPI.Barrier(comm)
        rank == 0 && rm(outdir; recursive=true, force=true)
    end
end
