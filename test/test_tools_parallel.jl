# Tests for the parallel/MPI helper wrappers in src/tools/parallel.jl.
#
# These helpers are designed to ALSO work in serial (single process, no real
# MPI ranks), which is exactly how they are exercised here: this file runs as a
# single process with no mpiexec, so only the serial-valid behavior is checked.
#
# The work-distribution helpers are pure index math, so they are validated
# against an INDEPENDENT oracle: the per-worker ranges must tile 1:total with no
# gaps or overlaps and be balanced (sizes differ by <= 1). The collective
# reductions are checked for their documented serial contract: with MPI
# unavailable they must return the input value unchanged.

using Test
using Tarang

@testset "tools/parallel.jl" begin

    # This file runs as a single process with no mpiexec. Depending on whether
    # MPI auto-initializes (MPI.Init succeeds with a single-rank world), the
    # helpers take EITHER the pure-serial fallback path (comm=nothing) OR a
    # single-rank MPI path. Both produce rank 0 / size 1 and make every
    # reduction a value-preserving no-op, so the contracts below are asserted
    # unconditionally. `single_rank` is true in both cases and is the real oracle.
    single_rank = Tarang.get_mpi_info().size == 1
    pure_serial = !Tarang.mpi_available()  # only true when comm is nothing

    @testset "mpi_available / get_mpi_info" begin
        @test Tarang.mpi_available() isa Bool

        info = Tarang.get_mpi_info()
        @test info.rank isa Integer
        @test info.size isa Integer
        @test info.size >= 1
        @test info.rank >= 0
        @test info.rank < info.size
        # Single process => one rank, indexed 0.
        @test single_rank
        @test info.size == 1
        @test info.rank == 0
        if pure_serial
            # Documented pure-serial fallback exposes comm=nothing; the
            # single-rank MPI path exposes a real COMM_WORLD instead.
            @test info.comm === nothing
        end
    end

    @testset "ensure_mpi_initialized / barrier / parallel_print (smoke)" begin
        # In serial these must run without error and without finalizing MPI.
        @test Tarang.ensure_mpi_initialized() === nothing || true
        @test (Tarang.barrier(); true)
        @test (Tarang.parallel_print("tools/parallel test message"); true)
        @test (Tarang.parallel_print("non-root message", 99); true)
    end

    @testset "distribute_work: tiling + balance oracle" begin
        # Oracle: over all workers, the ranges must partition 1:total (no gaps,
        # no overlaps) and the chunk sizes must differ by at most 1.
        function check_partition(total, num_workers)
            covered = Int[]
            sizes = Int[]
            for wid in 0:(num_workers - 1)
                s, e = Tarang.distribute_work(total, num_workers, wid)
                # Each range is contiguous and 1-based (empty allowed when
                # total < num_workers, signalled by e < s).
                if e >= s
                    @test s >= 1
                    @test e <= total
                    append!(covered, collect(s:e))
                    push!(sizes, e - s + 1)
                else
                    push!(sizes, 0)
                end
            end
            # Exact tiling of 1:total with no overlap or gaps.
            @test sort(covered) == collect(1:total)
            @test length(covered) == total
            # Balanced: max and min chunk differ by at most one item.
            if !isempty(sizes)
                @test maximum(sizes) - minimum(sizes) <= 1
            end
        end

        # Evenly divisible.
        check_partition(12, 4)
        check_partition(10, 1)
        check_partition(10, 10)
        # Non-divisible (remainder goes to lowest worker ids).
        check_partition(10, 3)
        check_partition(7, 4)
        check_partition(100, 7)
        check_partition(13, 5)
        # total < num_workers: some workers get an empty range.
        check_partition(3, 5)
        check_partition(1, 4)
        # Zero total work: every worker empty, partition of 1:0 is empty.
        check_partition(0, 3)
    end

    @testset "distribute_work: spot-check exact ranges" begin
        # 10 items over 3 workers -> remainder 1 to worker 0: [1:4],[5:7],[8:10].
        @test Tarang.distribute_work(10, 3, 0) == (1, 4)
        @test Tarang.distribute_work(10, 3, 1) == (5, 7)
        @test Tarang.distribute_work(10, 3, 2) == (8, 10)
        # Even split 12/4: [1:3],[4:6],[7:9],[10:12].
        @test Tarang.distribute_work(12, 4, 0) == (1, 3)
        @test Tarang.distribute_work(12, 4, 3) == (10, 12)
    end

    @testset "distribute_work: argument validation" begin
        @test_throws ArgumentError Tarang.distribute_work(-1, 2, 0)
        @test_throws ArgumentError Tarang.distribute_work(10, 0, 0)
        @test_throws ArgumentError Tarang.distribute_work(10, -2, 0)
        @test_throws ArgumentError Tarang.distribute_work(10, 3, -1)
        @test_throws ArgumentError Tarang.distribute_work(10, 3, 3)
    end

    @testset "get_local_work (single rank => whole range)" begin
        # With one rank (rank 0, size 1) the local work is the whole range.
        @test Tarang.get_local_work(8) == (1, 8)
        @test Tarang.get_local_work(1) == (1, 1)
        # And it must equal distribute_work for this rank/size in any case.
        info = Tarang.get_mpi_info()
        @test Tarang.get_local_work(20) == Tarang.distribute_work(20, info.size, info.rank)
    end

    @testset "parallel_sum/max/min (single rank returns input)" begin
        # Contract: a one-element reduction (serial fallback OR single-rank
        # Allreduce) returns its own value unchanged.
        @test Tarang.parallel_sum(3.5) == 3.5
        @test Tarang.parallel_sum(7) == 7
        @test Tarang.parallel_max(-2.0) == -2.0
        @test Tarang.parallel_min(42) == 42
        # Explicit comm=nothing must behave identically (defaults to COMM_WORLD
        # under MPI, no-op when serial).
        @test Tarang.parallel_sum(1.25; comm=nothing) == 1.25
        @test Tarang.parallel_max(9; comm=nothing) == 9
        @test Tarang.parallel_min(9; comm=nothing) == 9
        # Type is preserved.
        @test Tarang.parallel_sum(2.0) isa Float64
        @test Tarang.parallel_sum(2) isa Integer
    end

    @testset "parallel_all/any (single rank returns input)" begin
        @test Tarang.parallel_all(true) == true
        @test Tarang.parallel_all(false) == false
        @test Tarang.parallel_any(true) == true
        @test Tarang.parallel_any(false) == false
        @test Tarang.parallel_all(true; comm=nothing) == true
        @test Tarang.parallel_any(false; comm=nothing) == false
        @test Tarang.parallel_all(true) isa Bool
    end

    @testset "parallel_mkdir" begin
        parent = mktempdir()
        try
            newdir = joinpath(parent, "a", "b", "c")
            @test !isdir(newdir)
            ret = Tarang.parallel_mkdir(newdir)
            @test ret == newdir
            @test isdir(newdir)
            # Idempotent: calling again on an existing dir succeeds.
            @test Tarang.parallel_mkdir(newdir) == newdir
            @test isdir(newdir)
            # Empty path is rejected.
            @test_throws ArgumentError Tarang.parallel_mkdir("")
        finally
            rm(parent; recursive=true, force=true)
        end
    end

    @testset "PerformanceTimer lifecycle" begin
        t = Tarang.PerformanceTimer("unit")
        @test Tarang.is_running(t) == false
        @test Tarang.average_time(t) == 0.0

        # Stopping a never-started timer returns 0.0 and does not record a count.
        @test Tarang.stop_timer!(t) == 0.0
        @test Tarang.timer_stats(t).count == 0

        Tarang.start_timer!(t)
        @test Tarang.is_running(t) == true
        # start is idempotent: a second start while running keeps it running.
        Tarang.start_timer!(t)
        @test Tarang.is_running(t) == true
        elapsed = Tarang.stop_timer!(t)
        @test elapsed >= 0.0
        @test Tarang.is_running(t) == false

        stats = Tarang.timer_stats(t)
        @test stats.count == 1
        @test stats.total_time >= 0.0
        @test stats.average_time == stats.total_time / 1
        @test stats.running == false

        # reset clears all accumulated state.
        Tarang.reset_timer!(t)
        rs = Tarang.timer_stats(t)
        @test rs.count == 0
        @test rs.total_time == 0.0
        @test Tarang.is_running(t) == false

        # show prints a non-empty descriptive string without error.
        @test occursin("PerformanceTimer", sprint(show, t))
    end
end
