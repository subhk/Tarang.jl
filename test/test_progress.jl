"""
Test suite for progress.jl

Tests:
1. ProgressState struct
2. ProgressLogger struct and constructor
3. ProgressIterator struct
4. log_progress function
5. print_progress function
6. Iterator protocol (iterate, length, eltype)
7. build_message function
8. format_progress_time function
"""

using Test
using Logging

@testset "Progress Module" begin
    using Tarang

    @testset "format_progress_time" begin
        @testset "Seconds only" begin
            @test Tarang.format_progress_time(0.0) == "0s"
            @test Tarang.format_progress_time(1.0) == "1s"
            @test Tarang.format_progress_time(30.0) == "30s"
            @test Tarang.format_progress_time(59.0) == "59s"
        end

        @testset "Minutes and seconds" begin
            @test Tarang.format_progress_time(60.0) == "1m 00s"
            @test Tarang.format_progress_time(90.0) == "1m 30s"
            @test Tarang.format_progress_time(125.0) == "2m 05s"
            @test Tarang.format_progress_time(3599.0) == "59m 59s"
        end

        @testset "Hours, minutes, seconds" begin
            @test Tarang.format_progress_time(3600.0) == "1h 00m 00s"
            @test Tarang.format_progress_time(3661.0) == "1h 01m 01s"
            @test Tarang.format_progress_time(7200.0) == "2h 00m 00s"
            @test Tarang.format_progress_time(7323.0) == "2h 02m 03s"
        end

        @testset "Edge cases" begin
            @test Tarang.format_progress_time(Inf) == "?"
            @test Tarang.format_progress_time(-Inf) == "?"
            @test Tarang.format_progress_time(NaN) == "?"
            @test Tarang.format_progress_time(-1.0) == "?"
        end

        @testset "Rounding" begin
            @test Tarang.format_progress_time(0.4) == "0s"
            @test Tarang.format_progress_time(0.6) == "1s"
            @test Tarang.format_progress_time(59.5) == "1m 00s"
        end
    end

    @testset "ProgressState" begin
        @testset "Creation" begin
            state = Tarang.ProgressState(0, time(), -1, -1)
            @test state.index == 0
            @test state.start_time > 0
            @test state.last_iter_div == -1
            @test state.last_time_div == -1
        end

        @testset "Mutability" begin
            state = Tarang.ProgressState(0, 0.0, -1, -1)
            state.index = 5
            @test state.index == 5

            state.last_iter_div = 2
            @test state.last_iter_div == 2

            state.last_time_div = 3
            @test state.last_time_div == 3
        end
    end

    @testset "ProgressLogger" begin
        @testset "Basic constructor" begin
            prog = ProgressLogger("Test", 100)
            @test prog.desc == "Test"
            @test prog.total == 100
            @test prog.min_iter == 1
            @test prog.min_seconds == Inf
            @test prog.level == Logging.Info
        end

        @testset "With frac parameter" begin
            # frac provides upper bound: min_iter = max(1, min(iter, ceil(frac*total)))
            # With iter=100 (high), frac controls: min(100, ceil(0.1*100)) = 10
            prog = ProgressLogger("Test", 100; frac=0.1, iter=100)
            @test prog.min_iter == 10  # min(100, 10) = 10
        end

        @testset "With iter parameter" begin
            prog = ProgressLogger("Test", 100; iter=5)
            @test prog.min_iter == 5
        end

        @testset "With dt parameter" begin
            prog = ProgressLogger("Test", 100; dt=2.0)
            @test prog.min_seconds == 2.0
        end

        @testset "With level parameter" begin
            prog = ProgressLogger("Test", 100; level=Logging.Debug)
            @test prog.level == Logging.Debug
        end

        @testset "min_iter bounds" begin
            # frac takes precedence if smaller
            prog1 = ProgressLogger("Test", 100; frac=0.05, iter=10)
            @test prog1.min_iter == 5  # min(10, ceil(0.05 * 100))

            # iter takes precedence if smaller
            prog2 = ProgressLogger("Test", 100; frac=0.5, iter=3)
            @test prog2.min_iter == 3

            # At least 1
            prog3 = ProgressLogger("Test", 100; iter=0)
            @test prog3.min_iter == 1
        end

        @testset "Invalid dt" begin
            prog1 = ProgressLogger("Test", 100; dt=0.0)
            @test prog1.min_seconds == Inf

            prog2 = ProgressLogger("Test", 100; dt=-1.0)
            @test prog2.min_seconds == Inf
        end
    end

    @testset "ProgressIterator" begin
        @testset "Creation via log_progress" begin
            iter = log_progress(1:5; desc="Testing")
            @test isa(iter, Tarang.ProgressIterator)
            @test length(iter) == 5
        end

        @testset "Creation via print_progress" begin
            io = IOBuffer()
            iter = print_progress(1:3, io; desc="Print test")
            @test isa(iter, Tarang.ProgressIterator)
            @test length(iter) == 3
        end

        @testset "eltype" begin
            iter = log_progress([1, 2, 3])
            @test eltype(iter) == Int

            iter2 = log_progress(["a", "b"])
            @test eltype(iter2) == String
        end

        @testset "IteratorSize" begin
            iter = log_progress(1:10)
            @test Base.IteratorSize(typeof(iter)) == Base.HasLength()
        end

        @testset "IteratorEltype" begin
            iter = log_progress(1:10)
            @test Base.IteratorEltype(typeof(iter)) == Base.HasEltype()
        end
    end

    @testset "Iteration" begin
        @testset "Basic iteration" begin
            items = Int[]
            for item in log_progress(1:5; desc="Basic", iter=10)  # iter=10 > 5, so minimal logging
                push!(items, item)
            end
            @test items == [1, 2, 3, 4, 5]
        end

        @testset "Empty iteration" begin
            items = Int[]
            for item in log_progress(Int[])
                push!(items, item)
            end
            @test isempty(items)
        end

        @testset "Single item" begin
            items = Int[]
            for item in log_progress([42]; desc="Single")
                push!(items, item)
            end
            @test items == [42]
        end

        @testset "Collect works" begin
            result = collect(log_progress(1:3; iter=10))
            @test result == [1, 2, 3]
        end

        @testset "Map works" begin
            result = map(x -> x^2, log_progress([1, 2, 3]; iter=10))
            @test result == [1, 4, 9]
        end

        @testset "Sum works" begin
            total = sum(log_progress(1:10; iter=100))
            @test total == 55
        end

        @testset "With string items" begin
            items = String[]
            for item in log_progress(["a", "b", "c"]; iter=10)
                push!(items, item)
            end
            @test items == ["a", "b", "c"]
        end
    end

    @testset "print_progress output" begin
        @testset "Prints to buffer" begin
            io = IOBuffer()
            items = Int[]
            # Force output on every iteration
            for item in print_progress(1:3, io; desc="Buffer test", iter=1)
                push!(items, item)
            end
            @test items == [1, 2, 3]

            output = String(take!(io))
            @test occursin("Buffer test", output)
            @test occursin("3/3", output)  # Final progress
        end

        @testset "Shows percentage" begin
            io = IOBuffer()
            for _ in print_progress(1:10, io; desc="Percent", iter=1)
            end
            output = String(take!(io))
            @test occursin("100%", output)
        end

        @testset "Shows elapsed time" begin
            io = IOBuffer()
            for _ in print_progress(1:2, io; desc="Elapsed", iter=1)
            end
            output = String(take!(io))
            @test occursin("Elapsed:", output)
        end

        @testset "Shows remaining time" begin
            io = IOBuffer()
            for _ in print_progress(1:2, io; desc="Remaining", iter=1)
            end
            output = String(take!(io))
            @test occursin("Remaining:", output)
        end

        @testset "Shows rate" begin
            io = IOBuffer()
            for _ in print_progress(1:2, io; desc="Rate", iter=1)
            end
            output = String(take!(io))
            @test occursin("Rate:", output)
        end
    end

    @testset "build_message" begin
        @testset "Basic message" begin
            prog = ProgressLogger("Test", 10)
            msg = Tarang.build_message(prog, 5, 1.0)

            @test occursin("Test", msg)
            @test occursin("5/10", msg)
            @test occursin("50%", msg)
            @test occursin("Elapsed:", msg)
            @test occursin("Remaining:", msg)
            @test occursin("Rate:", msg)
        end

        @testset "Zero total" begin
            prog = ProgressLogger("Empty", 0)
            msg = Tarang.build_message(prog, 0, 0.0)

            @test occursin("Empty", msg)
            @test occursin("0/0", msg)
            @test occursin("100%", msg)  # Special case for 0 total
        end

        @testset "Zero elapsed" begin
            prog = ProgressLogger("Quick", 10)
            msg = Tarang.build_message(prog, 5, 0.0)

            @test occursin("Rate: 0/s", msg)
            @test occursin("Remaining: ?", msg)  # Inf remaining
        end

        @testset "Rate formatting" begin
            prog = ProgressLogger("Rate test", 100)
            msg = Tarang.build_message(prog, 50, 10.0)  # 5 items/sec

            @test occursin("5.00e+00/s", msg)
        end

        @testset "100% complete" begin
            prog = ProgressLogger("Complete", 10)
            msg = Tarang.build_message(prog, 10, 5.0)

            @test occursin("10/10", msg)
            @test occursin("100%", msg)
        end
    end

    @testset "Progress reporting intervals" begin
        @testset "iter-based reporting" begin
            io = IOBuffer()
            # iter=2 means report every 2 iterations
            for _ in print_progress(1:6, io; desc="Iter", iter=2)
            end
            output = String(take!(io))
            lines = split(strip(output), '\n')

            # Should have lines at iterations 2, 4, 6 (every 2)
            # Plus the final line at 6
            @test length(lines) >= 3
        end

        @testset "frac-based reporting" begin
            io = IOBuffer()
            # frac=0.5 means report at every 50% (every 5 items for 10 total)
            for _ in print_progress(1:10, io; desc="Frac", frac=0.5)
            end
            output = String(take!(io))

            # At minimum should have final report
            @test occursin("10/10", output)
        end

        @testset "Always reports last item" begin
            io = IOBuffer()
            # Even with high iter value, last item is always reported
            for _ in print_progress(1:5, io; desc="Last", iter=100)
            end
            output = String(take!(io))

            @test occursin("5/5", output)
            @test occursin("100%", output)
        end
    end

    @testset "Edge cases" begin
        @testset "Very large total" begin
            prog = ProgressLogger("Large", 1_000_000)
            @test prog.total == 1_000_000
        end

        @testset "Generator input" begin
            gen = (x^2 for x in 1:5)
            result = collect(log_progress(gen; iter=10))
            @test result == [1, 4, 9, 16, 25]
        end

        @testset "Range input" begin
            result = collect(log_progress(1:3; iter=10))
            @test result == [1, 2, 3]
        end

        @testset "Array input" begin
            result = collect(log_progress([10, 20, 30]; iter=10))
            @test result == [10, 20, 30]
        end

        @testset "Tuple converted to array" begin
            result = collect(log_progress((1, 2, 3); iter=10))
            @test result == [1, 2, 3]
        end

        @testset "Long description" begin
            desc = "This is a very long description for testing purposes"
            io = IOBuffer()
            for _ in print_progress(1:2, io; desc=desc, iter=1)
            end
            output = String(take!(io))
            @test occursin(desc, output)
        end
    end

    @testset "log_progress parameters" begin
        @testset "Default parameters" begin
            iter = log_progress(1:10)
            @test iter.prog.desc == "Iteration"
            @test iter.prog.total == 10
            @test iter.prog.level == Logging.Info
        end

        @testset "Custom description" begin
            iter = log_progress(1:10; desc="Custom")
            @test iter.prog.desc == "Custom"
        end

        @testset "Custom level" begin
            iter = log_progress(1:10; level=Logging.Debug)
            @test iter.prog.level == Logging.Debug
        end

        @testset "All parameters" begin
            iter = log_progress(1:100;
                desc="Full test",
                frac=0.1,
                iter=5,
                dt=1.0,
                level=Logging.Warn)

            @test iter.prog.desc == "Full test"
            @test iter.prog.total == 100
            @test iter.prog.min_iter == 5  # min(5, ceil(0.1 * 100))
            @test iter.prog.min_seconds == 1.0
            @test iter.prog.level == Logging.Warn
        end
    end

    @testset "print_progress parameters" begin
        @testset "Default IO is stdout" begin
            # Just verify it doesn't error with default IO
            iter = print_progress(1:2)
            @test iter.prog.total == 2
        end

        @testset "Custom IO buffer" begin
            io = IOBuffer()
            iter = print_progress(1:5, io; desc="Buffer")
            @test iter.prog.desc == "Buffer"
            @test iter.prog.total == 5
        end

        @testset "All parameters" begin
            io = IOBuffer()
            iter = print_progress(1:100, io;
                desc="Full print",
                frac=0.2,
                iter=10,
                dt=0.5)

            @test iter.prog.desc == "Full print"
            @test iter.prog.total == 100
            @test iter.prog.min_iter == 10  # min(10, ceil(0.2 * 100))
            @test iter.prog.min_seconds == 0.5
        end
    end

    @testset "Integration" begin
        @testset "Real iteration with timing" begin
            io = IOBuffer()
            results = Int[]

            for item in print_progress(1:5, io; desc="Timed", iter=1)
                push!(results, item * 2)
                sleep(0.01)  # Small delay
            end

            @test results == [2, 4, 6, 8, 10]
            output = String(take!(io))
            @test occursin("5/5", output)
        end

        @testset "Nested progress (outer only)" begin
            io = IOBuffer()
            results = Int[]

            for i in print_progress(1:3, io; desc="Outer", iter=1)
                for j in 1:2  # Inner loop without progress
                    push!(results, i * j)
                end
            end

            @test length(results) == 6
            output = String(take!(io))
            @test occursin("Outer", output)
        end

        @testset "Break early" begin
            io = IOBuffer()
            items = Int[]

            for item in print_progress(1:10, io; desc="Break", iter=1)
                push!(items, item)
                if item == 3
                    break
                end
            end

            @test items == [1, 2, 3]
            output = String(take!(io))
            @test occursin("3/10", output)
        end

        @testset "Continue behavior" begin
            io = IOBuffer()
            items = Int[]

            for item in print_progress(1:5, io; desc="Continue", iter=1)
                if item == 3
                    continue
                end
                push!(items, item)
            end

            @test items == [1, 2, 4, 5]
        end
    end
end

println("All progress tests passed!")
