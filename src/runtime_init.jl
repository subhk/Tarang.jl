# Runtime initialization for the package root module.

function __init__()
    if !MPI.Initialized() && get(ENV, "TARANG_USE_MPI", "") != "0"
        try
            MPI.Init()
        catch e
            @warn "MPI initialization failed (set TARANG_USE_MPI=0 to disable): $e"
        end
    end

    init_config!()
    init_logging!()
    _init_gpu_solvers!()
end
