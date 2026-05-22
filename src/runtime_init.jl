# Runtime initialization for the package root module.

function __init__()
    if !MPI.Initialized() && get(ENV, "TARANG_USE_MPI", "") != "0"
        try
            MPI.Init()
        catch e
            @warn "MPI initialization failed (set TARANG_USE_MPI=0 to disable): $e"
        end
    end

    _init_fftw_threads!()
    init_config!()
    init_logging!()
    _init_gpu_solvers!()
end

"""
    _init_fftw_threads!() -> Int

Configure the number of FFTW threads for shared-memory parallelism and return it.

FFTW defaults to a single thread, so without this call every CPU transform runs
serially even when Julia is started with `-t N`. Policy:

- Serial run (<= 1 MPI rank): use all Julia threads (`Threads.nthreads()`).
- Multi-rank MPI run: use 1 FFTW thread per rank — ranks already provide the
  parallelism, and threading on top would oversubscribe the cores.

Override with the `TARANG_FFTW_THREADS` environment variable. Any failure
(unparseable value, FFTW without thread support) falls back to a single thread.
"""
function _init_fftw_threads!()
    n = try
        if haskey(ENV, "TARANG_FFTW_THREADS")
            parse(Int, ENV["TARANG_FFTW_THREADS"])
        elseif MPI.Initialized() && MPI.Comm_size(MPI.COMM_WORLD) > 1
            1
        else
            Threads.nthreads()
        end
    catch
        1
    end
    n = max(1, n)
    try
        FFTW.set_num_threads(n)
    catch e
        @warn "FFTW.set_num_threads($n) failed; transforms stay single-threaded: $e"
    end
    return n
end
