"""
Parallel computing utilities (CPU-only)
"""

using MPI
using Printf

export mpi_available, ensure_mpi_initialized, get_mpi_info, parallel_print, barrier,
       parallel_mkdir, distribute_work, get_local_work,
       parallel_sum, parallel_max, parallel_min, parallel_all, parallel_any,
       PerformanceTimer, start_timer!, stop_timer!, reset_timer!, average_time, timer_stats, is_running

# Basic MPI helpers

"""
    mpi_available() -> Bool

Check if MPI is available for use (initialized and NOT finalized).
After MPI.Finalize(), Initialized() still returns true but MPI calls will fail.
Use this helper instead of just checking MPI.Initialized().
"""
function mpi_available()
    return MPI.Initialized() && !MPI.Finalized()
end

function ensure_mpi_initialized()
    if MPI.Finalized()
        error("Cannot initialize MPI: MPI has already been finalized. " *
              "MPI can only be initialized once per process.")
    end
    if !MPI.Initialized()
        MPI.Init()
    end
end

function get_mpi_info()
    if mpi_available()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        return (comm=comm, rank=rank, size=size)
    else
        return (comm=nothing, rank=0, size=1)
    end
end

function parallel_print(msg::String, root::Int=0)
    info = get_mpi_info()
    if info.rank == root
        println(msg)
    end
end

function barrier()
    if mpi_available()
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

# Filesystem helper
function parallel_mkdir(path::String; mode::Integer=0o755)
    if isempty(path)
        throw(ArgumentError("parallel_mkdir: path cannot be empty"))
    end
    info = get_mpi_info()
    local mkdir_success = true
    if info.rank == 0 && !isdir(path)
        try
            mkpath(path; mode=mode)
        catch e
            mkdir_success = false
        end
    end
    # Broadcast success/failure from rank 0 to all ranks so all throw consistently
    if mpi_available()
        mkdir_success_ref = Ref(mkdir_success)
        MPI.Bcast!(mkdir_success_ref, MPI.COMM_WORLD; root=0)
        mkdir_success = mkdir_success_ref[]
    end
    barrier()
    if !mkdir_success
        error("parallel_mkdir: failed to create directory '$path' on rank 0")
    end
    return path
end

# Work distribution helpers
function distribute_work(total_work::Int, num_workers::Int, worker_id::Int)
    if total_work < 0
        throw(ArgumentError("Total work must be non-negative"))
    end
    if num_workers <= 0
        throw(ArgumentError("Number of workers must be positive"))
    end
    if worker_id < 0 || worker_id >= num_workers
        throw(ArgumentError("Worker ID must be in range [0, $(num_workers-1)]"))
    end
    work_per_worker = div(total_work, num_workers)
    remainder = total_work % num_workers
    if worker_id < remainder
        start_idx = worker_id * (work_per_worker + 1) + 1
        end_idx = start_idx + work_per_worker
    else
        start_idx = worker_id * work_per_worker + remainder + 1
        end_idx = start_idx + work_per_worker - 1
    end
    end_idx = min(end_idx, total_work)
    return (start_idx, end_idx)
end

function get_local_work(total_work::Int)
    info = get_mpi_info()
    return distribute_work(total_work, info.size, info.rank)
end

# Simple collective reductions
# All accept an optional communicator (default: MPI.COMM_WORLD)
function parallel_sum(value::Number; comm::Union{Nothing, MPI.Comm}=nothing)
    if mpi_available()
        c = comm !== nothing ? comm : MPI.COMM_WORLD
        return MPI.Allreduce(value, +, c)
    else
        return value
    end
end

function parallel_max(value::Number; comm::Union{Nothing, MPI.Comm}=nothing)
    if mpi_available()
        c = comm !== nothing ? comm : MPI.COMM_WORLD
        return MPI.Allreduce(value, max, c)
    else
        return value
    end
end

function parallel_min(value::Number; comm::Union{Nothing, MPI.Comm}=nothing)
    if mpi_available()
        c = comm !== nothing ? comm : MPI.COMM_WORLD
        return MPI.Allreduce(value, min, c)
    else
        return value
    end
end

function parallel_all(value::Bool; comm::Union{Nothing, MPI.Comm}=nothing)
    if mpi_available()
        c = comm !== nothing ? comm : MPI.COMM_WORLD
        return MPI.Allreduce(value, &, c)
    else
        return value
    end
end

function parallel_any(value::Bool; comm::Union{Nothing, MPI.Comm}=nothing)
    if mpi_available()
        c = comm !== nothing ? comm : MPI.COMM_WORLD
        return MPI.Allreduce(value, |, c)
    else
        return value
    end
end

# Lightweight performance timer
mutable struct PerformanceTimer
    name::String
    start_time::Float64
    total_time::Float64
    count::Int
    running::Bool
    function PerformanceTimer(name::String)
        new(name, 0.0, 0.0, 0, false)
    end
end

function start_timer!(timer::PerformanceTimer)
    if timer.running
        return timer  # Already running, don't restart (idempotent)
    end
    timer.start_time = time()
    timer.running = true
    return timer
end

function stop_timer!(timer::PerformanceTimer)
    if !timer.running
        return 0.0  # Timer was not started, return 0
    end
    elapsed = max(0.0, time() - timer.start_time)  # Guard against clock drift
    timer.total_time += elapsed
    timer.count += 1
    timer.running = false
    return elapsed
end

function reset_timer!(timer::PerformanceTimer)
    timer.start_time = 0.0
    timer.total_time = 0.0
    timer.count = 0
    timer.running = false
    return timer
end

function average_time(timer::PerformanceTimer)
    return timer.count == 0 ? 0.0 : timer.total_time / timer.count
end

function timer_stats(timer::PerformanceTimer)
    return (count=timer.count, total_time=timer.total_time, average_time=average_time(timer), running=timer.running)
end

function is_running(timer::PerformanceTimer)
    return timer.running
end

function Base.show(io::IO, timer::PerformanceTimer)
    status = timer.running ? "running" : "stopped"
    avg = @sprintf("%.4g", average_time(timer))
    print(io, "PerformanceTimer(\"$(timer.name)\", $status, count=$(timer.count), total=$(timer.total_time)s, avg=$(avg)s)")
end
