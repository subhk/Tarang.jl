using Test
using Tarang
import FFTW

@testset "FFTW threading policy" begin
    # Default (serial, no override): uses all Julia threads, never < 1.
    n_default = Tarang._init_fftw_threads!()
    @test n_default >= 1
    @test n_default == max(1, Threads.nthreads())

    # Explicit override is honored.
    withenv("TARANG_FFTW_THREADS" => "2") do
        @test Tarang._init_fftw_threads!() == 2
    end

    # Garbage override falls back to 1 instead of throwing.
    withenv("TARANG_FFTW_THREADS" => "not_an_int") do
        @test Tarang._init_fftw_threads!() == 1
    end

    # The value actually reached FFTW (get_num_threads added in FFTW.jl >= 1.6).
    if isdefined(FFTW, :get_num_threads)
        withenv("TARANG_FFTW_THREADS" => "1") do
            Tarang._init_fftw_threads!()
            @test FFTW.get_num_threads() == 1
        end
    end
end
