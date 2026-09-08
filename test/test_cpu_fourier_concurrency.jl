using Test
using Tarang
using FFTW

function _fourier_concurrency_cases(n, count; shared_basis=nothing)
    [let
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        scale = shared_basis === nothing ? Float64(j) : 1.0
        basis = shared_basis === nothing ?
                RealFourier(coords["x"]; size=n, bounds=(0.0, 2π * scale)) : shared_basis
        input = ScalarField(dist, "concurrent_fourier_$j", (basis,), Float64)
        theta = collect(0:n-1) .* (2π / n)
        values = @. j * sin((j + 1) * theta) + 0.2 * cos((j + 3) * theta)
        expected = @. j * (j + 1) / scale * cos((j + 1) * theta) -
                      0.2 * (j + 3) / scale * sin((j + 3) * theta)
        input["g"] .= values
        (; input, values, expected, op=Differentiate(input, coords["x"], 1))
    end for j in 1:count]
end

function _fourier_concurrent_errors(cases; repeats=24)
    # Waiting on an event yields: this also works with one default thread, an
    # interactive thread, and tasks that migrate between default worker threads.
    ready = Channel{Nothing}(length(cases))
    start = Base.Event()
    tasks = map(cases) do c
        Threads.@spawn begin
            put!(ready, nothing)
            wait(start)
            [maximum(abs, evaluate(c.op, :g)["g"] .- c.expected) for _ in 1:repeats]
        end
    end
    for _ in cases
        take!(ready)
    end
    notify(start)
    return reduce(vcat, fetch.(tasks))
end

@testset "CPU Fourier derivatives own concurrent FFT scratch" begin
    previous_fftw_threads = FFTW.get_num_threads()
    FFTW.set_num_threads(1)
    try
        cases = _fourier_concurrency_cases(16384, min(4, Threads.nthreads(:default)))
        # Distinct bases isolate FFT scratch from the separate derivative result
        # pool. Warm every public result slot and derivative multiplier serially.
        @test length(unique(hash(c.input.bases) for c in cases)) == length(cases)
        sequential_errors = [maximum(abs, evaluate(c.op, :g)["g"] .- c.expected)
                             for c in cases for _ in 1:64]
        @test all(e -> isfinite(e) && e < 1e-8, sequential_errors)

        # No calls are active while clearing the cache. The first group exercises
        # concurrent cold checkout/planning; the next reuses returned workspaces.
        empty!(Tarang._DERIV_FFT_WS)
        for cache_state in ("cold", "warm")
            @testset "$cache_state cache" begin
                errors = _fourier_concurrent_errors(cases)
                @test all(e -> isfinite(e) && e < 1e-8, errors)
                @test all(c -> c.input["g"] == c.values, cases)
            end
        end

        @testset "exclusive checkout and reuse after errors" begin
            data = cases[1].input["g"]
            empty!(Tarang._DERIV_FFT_WS)
            first_ws = Tarang._get_deriv_workspace!(data, 1, 1)
            second_ws = Tarang._get_deriv_workspace!(data, 1, 1)
            try
                @test all(first_ws[i] !== second_ws[i] for i in 1:3)
            finally
                Tarang._return_deriv_workspace!(data, 1, 1, first_ws)
                Tarang._return_deriv_workspace!(data, 1, 1, second_ws)
            end

            # A wrong-shaped destination fails after FFT execution. The leased
            # workspace must still return to the pool and remain usable.
            empty!(Tarang._DERIV_FFT_WS)
            workspace = Tarang._get_deriv_workspace!(data, 1, 1)
            Tarang._return_deriv_workspace!(data, 1, 1, workspace)
            bad_result = ScalarField(PeriodicDomain(32), "wrong_shape")
            @test_throws DimensionMismatch Tarang.evaluate_fourier_derivative!(
                bad_result, cases[1].input, 1, 1, :g)
            recovered = Tarang._get_deriv_workspace!(data, 1, 1)
            try
                @test recovered === workspace
            finally
                Tarang._return_deriv_workspace!(data, 1, 1, recovered)
            end
            @test evaluate(cases[1].op, :g)["g"] ≈ cases[1].expected atol=1e-8
        end

        @testset "independent fields sharing a basis" begin
            shared = _fourier_concurrency_cases(16384, length(cases);
                                                shared_basis=cases[1].input.bases[1])
            # Warm public evaluation, then remove only the derivative multiplier
            # so every caller enters the same basis cache on a cold miss.
            for c in shared
                @test evaluate(c.op, :g)["g"] ≈ c.expected atol=1e-8
            end
            delete!(shared[1].input.bases[1].transforms, (:deriv_mult, 16384, 1))
            empty!(Tarang._DERIV_FFT_WS)
            errors = _fourier_concurrent_errors(shared)
            @test all(e -> isfinite(e) && e < 1e-8, errors)
            @test all(c -> c.input["g"] == c.values, shared)
        end
    finally
        FFTW.set_num_threads(previous_fftw_threads)
    end
end
