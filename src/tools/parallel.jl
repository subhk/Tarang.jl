"""
Parallel computing utilities (CPU-only)
"""

using MPI
using PencilArrays

# Basic MPI helpers
function ensure_mpi_initialized()
    if !MPI.Initialized()
        MPI.Init()
    end
end

function get_mpi_info()
    if MPI.Initialized()
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
    if MPI.Initialized()
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

# Filesystem helper
function parallel_mkdir(path::String; mode::Int=0o755)
    info = get_mpi_info()
    if info.rank == 0 && !isdir(path)
        mkpath(path; mode=mode)
    end
    barrier()
    return path
end

# Work distribution helpers
function distribute_work(total_work::Int, num_workers::Int, worker_id::Int)
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
    return (start_idx, end_idx)
end

function get_local_work(total_work::Int)
    info = get_mpi_info()
    return distribute_work(total_work, info.size, info.rank)
end

# Simple collective reductions
function parallel_sum(value::Number)
    if MPI.Initialized()
        return MPI.Allreduce(value, +, MPI.COMM_WORLD)
    else
        return value
    end
end

function parallel_max(value::Number)
    if MPI.Initialized()
        return MPI.Allreduce(value, max, MPI.COMM_WORLD)
    else
        return value
    end
end

function parallel_min(value::Number)
    if MPI.Initialized()
        return MPI.Allreduce(value, min, MPI.COMM_WORLD)
    else
        return value
    end
end

function parallel_all(value::Bool)
    if MPI.Initialized()
        return MPI.Allreduce(value, &, MPI.COMM_WORLD)
    else
        return value
    end
end

function parallel_any(value::Bool)
    if MPI.Initialized()
        return MPI.Allreduce(value, |, MPI.COMM_WORLD)
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
    function PerformanceTimer(name::String)
        new(name, 0.0, 0.0, 0)
    end
end

function start_timer!(timer::PerformanceTimer)
    timer.start_time = time()
    return timer
end

function stop_timer!(timer::PerformanceTimer)
    elapsed = time() - timer.start_time
    timer.total_time += elapsed
    timer.count += 1
    return elapsed
end

function reset_timer!(timer::PerformanceTimer)
    timer.start_time = 0.0
    timer.total_time = 0.0
    timer.count = 0
    return timer
end

function average_time(timer::PerformanceTimer)
    return timer.count == 0 ? 0.0 : timer.total_time / timer.count
end

function timer_stats(timer::PerformanceTimer)
    return (count=timer.count, total_time=timer.total_time, average_time=average_time(timer))
end
