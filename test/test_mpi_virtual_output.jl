using Test
using MPI
MPI.Init()
using Tarang
using PencilArrays
using NetCDF

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

# All ranks must share one output directory (keyed by rank 0's pid).
const pid0 = (r = Ref(Int(getpid())); MPI.Bcast!(r, 0, comm); r[])
const outdir = joinpath(tempdir(), "tarang_vfh_test_np$(nprocs)_$(pid0)")
rank == 0 && mkpath(outdir)
MPI.Barrier(comm)

@testset "VirtualFileHandler per-rank output (np=$nprocs)" begin
    N = 16
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), Float64)

    # Deterministic global content so reconstruction is checkable exactly
    ref = [Float64(gi * 1000 + gj) for gi in 1:N, gj in 1:N]
    ensure_layout!(u, :g)
    data = get_grid_data(u)
    if isa(data, PencilArrays.PencilArray)
        gv = PencilArrays.global_view(data)
        for I in CartesianIndices(gv)
            gv[I] = ref[I]
        end
    else
        data .= ref
    end

    problem = IVP([u])
    add_equation!(problem, "∂t(u) = 0")
    solver = InitialValueSolver(problem, RK222(); dt=0.01)

    vfh = VirtualFileHandler(outdir, "vtest"; comm=comm, cadence=1)
    Tarang.add_task!(vfh, u, "u")
    Tarang.process!(vfh, solver, 0.0, 0.0, 1)
    MPI.Barrier(comm)

    rank_file = joinpath(outdir, "vtest_s1_p$(rank).nc")
    @test isfile(rank_file)

    @testset "per-rank file holds LOCAL slab, not global copy" begin
        block = ncread(rank_file, "u")
        if nprocs > 1
            @test length(block) < N * N   # must be a strict subset
        else
            @test size(block) == (N, N)
        end
        # start indices recorded for reconstruction
        starts = Int.(ncread(rank_file, "u_start"))
        @test length(starts) == 2
        # block content matches the reference at its global offsets
        rng = ntuple(d -> starts[d]:(starts[d] + size(block, d) - 1), 2)
        @test block == ref[rng...]
    end

    @testset "merge reconstructs the exact global field" begin
        MPI.Barrier(comm)
        if rank == 0
            merged_file = Tarang.merge_virtual!(vfh; set_num=1)
            merged = ncread(merged_file, "u")
            @test size(merged) == (N, N)
            @test merged == ref
        end
        MPI.Barrier(comm)
    end

    @testset "UnifiedEvaluator drives virtual handlers (orphan-loop guard)" begin
        # REGRESSION GUARD: evaluate_unified_handlers! previously iterated only
        # netcdf + dictionary handlers, never virtual_handlers, so VFH output was
        # silently never written through the unified path.
        evaluator = Tarang.UnifiedEvaluator(solver)
        h = Tarang.add_virtual_file_handler(evaluator, outdir, "vunified"; cadence=1)
        Tarang.add_task!(h, u, "u")
        Tarang.evaluate_unified_handlers!(evaluator, 0.0, 0.0, 1)
        MPI.Barrier(comm)
        @test isfile(joinpath(outdir, "vunified_s1_p$(rank).nc"))
    end
end

MPI.Barrier(comm)
rank == 0 && rm(outdir; recursive=true, force=true)
MPI.Finalize()
