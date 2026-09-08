using Test
using Tarang
using MPI

function _derivative_result_test_domain(N=16)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; comm=MPI.COMM_SELF, dtype=Float64, architecture=CPU())
    basis = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    return coords, dist, (basis,)
end

@testset "active tasks have exclusive derivative result buffers" begin
    _, dist, bases = _derivative_result_test_domain()
    count = 2 * Tarang._DERIV_RESULT_POOL_SIZE + 3
    ready = Channel{Any}(count)
    release = Channel{Nothing}(count)
    checkout_lock = ReentrantLock()

    tasks = map(1:count) do marker
        Threads.@spawn begin
            # Serialize checkout to isolate buffer ownership from dictionary
            # races. Every task keeps its borrowed result after dropping this
            # lock, so all leases overlap even with a single worker thread.
            borrowed = lock(checkout_lock) do
                result = Tarang._checkout_deriv_result!(bases, Float64, dist)
                fill!(get_grid_data(result), marker)
                result
            end
            put!(ready, borrowed)
            take!(release)
            (marker, copy(get_grid_data(borrowed)))
        end
    end
    held = [take!(ready) for _ in 1:count]
    @test length(unique(objectid.(held))) == count
    @test length(unique(objectid.(get_grid_data.(held)))) == count
    for _ in 1:count
        put!(release, nothing)
    end
    for task in tasks
        marker, values = fetch(task)
        @test all(==(marker), values)
    end
end

@testset "derivative result buffers retain their distributor" begin
    coords, first_dist, bases = _derivative_result_test_domain()
    second_dist = Distributor(coords; comm=MPI.COMM_SELF, dtype=Float64, architecture=CPU())
    for _ in 1:Tarang._DERIV_RESULT_POOL_SIZE
        @test Tarang._checkout_deriv_result!(bases, Float64, first_dist).dist === first_dist
    end
    for _ in 1:Tarang._DERIV_RESULT_POOL_SIZE
        @test Tarang._checkout_deriv_result!(bases, Float64, second_dist).dist === second_dist
    end
end

@testset "propagated task-local values do not share derivative buffers" begin
    _, dist, bases = _derivative_result_test_domain()
    parent_buffers = [Tarang._checkout_deriv_result!(bases, Float64, dist)
                      for _ in 1:Tarang._DERIV_RESULT_POOL_SIZE]
    inherited_storage = copy(task_local_storage())
    child_buffers = fetch(Threads.@spawn begin
        merge!(task_local_storage(), inherited_storage)
        [Tarang._checkout_deriv_result!(bases, Float64, dist)
         for _ in 1:Tarang._DERIV_RESULT_POOL_SIZE]
    end)
    @test all(parent !== child for parent in parent_buffers, child in child_buffers)
end

# Returning only a weak reference prevents the test from retaining either the
# finished task or its scratch field across the collection boundary.
@noinline function _finished_derivative_result_reference()
    _, dist, bases = _derivative_result_test_domain(18)
    fetch(Threads.@spawn WeakRef(Tarang._checkout_deriv_result!(bases, Float64, dist)))
end

@testset "finished tasks do not leave derivative results globally retained" begin
    result = _finished_derivative_result_reference()
    # An idle Julia worker can retain its last finished task while scheduling.
    # Give the workers subsequent work before testing cache reachability.
    for _ in 1:5
        fetch.([Threads.@spawn(yield()) for _ in 1:(4 * Threads.nthreads(:default))])
        GC.gc(true)
        result.value === nothing && break
    end
    @test result.value === nothing
end

@testset "public same-basis derivatives remain independent across tasks" begin
    coords, dist, bases = _derivative_result_test_domain(128)
    count = Tarang._DERIV_RESULT_POOL_SIZE + 4
    grid = [2π * (i - 1) / 128 for i in 1:128]
    fields = map(1:count) do amplitude
        field = ScalarField(dist, "concurrent_$amplitude", bases, Float64)
        set!(field, x -> amplitude * sin(3x))
        field
    end
    snapshots = [copy(get_grid_data(field)) for field in fields]
    start = Base.Event()
    tasks = map(enumerate(fields)) do (amplitude, field)
        Threads.@spawn begin
            wait(start)
            op = Tarang.Differentiate(field, coords["x"], 1)
            held = Tarang.evaluate_differentiate(op, :g)
            largest_error = 0.0
            expected = amplitude .* 3 .* cos.(3 .* grid)
            for _ in 1:(2 * Tarang._DERIV_RESULT_POOL_SIZE)
                result = evaluate(op, :g)
                largest_error = max(largest_error,
                    maximum(abs, get_grid_data(result) .- expected))
                yield()
            end
            (held, largest_error, expected)
        end
    end
    notify(start)
    for (i, task) in enumerate(tasks)
        held, largest_error, expected = fetch(task)
        @test largest_error < 1e-9
        @test maximum(abs, get_grid_data(held) .- expected) < 1e-9
        @test get_grid_data(fields[i]) == snapshots[i]
    end
end
