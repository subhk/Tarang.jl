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
    # MPI.jl cannot be reinitialized after finalization in the same process.
    # Fail early with a clear message instead of triggering lower-level MPI
    # errors later in setup.
    if MPI.Finalized()
        error("Cannot initialize MPI: MPI has already been finalized. " *
              "MPI can only be initialized once per process.")
    end
    if !MPI.Initialized()
        MPI.Init()
    end
end

"""
    get_mpi_info() -> NamedTuple

Return `(comm, rank, size)` for `MPI.COMM_WORLD`, or the serial fallback
`(comm=nothing, rank=0, size=1)` when MPI is unavailable. Lets serial and MPI
code paths read the same fields.
"""
function get_mpi_info()
    if mpi_available()
        # Keep the common communicator metadata in one place so serial fallback
        # paths can use the same tuple fields as MPI paths.
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
"""
    parallel_mkdir(path; mode=0o755) -> path

Create `path` (and parents) from rank 0 only, then broadcast success to all ranks so
every rank either returns `path` or throws together. Collective: all ranks must call it.
"""
function parallel_mkdir(path::String; mode::Integer=0o755)
    if isempty(path)
        throw(ArgumentError("parallel_mkdir: path cannot be empty"))
    end
    info = get_mpi_info()
    local mkdir_success = true
    # Only rank 0 touches the filesystem; all ranks learn the result below.
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
"""
    distribute_work(total_work, num_workers, worker_id) -> (start_idx, end_idx)

Split `total_work` items into near-equal contiguous 1-based ranges, one per worker.
The first `total_work % num_workers` workers each take one extra item, so the returned
`start_idx:end_idx` ranges tile `1:total_work` without gaps or overlap.
"""
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
    # Remainder work is assigned to the lowest worker ids, producing contiguous
    # 1-based ranges that are convenient for Julia array slicing.
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
"""
    parallel_sum(value; comm=nothing) -> value

Allreduce `value` across `comm` (default `MPI.COMM_WORLD`); every rank gets the result.
Returns `value` unchanged when MPI is unavailable. `parallel_max`, `parallel_min`,
`parallel_all`, and `parallel_any` follow the same contract with `max`/`min`/`&`/`|`.
"""
function parallel_sum(value::Number; comm::Union{Nothing, MPI.Comm}=nothing)
    if mpi_available()
        # Accept an explicit communicator for row/column reductions while
        # defaulting to COMM_WORLD for existing serial-style call sites.
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
"""
    PerformanceTimer(name)

Minimal accumulating wall-clock timer for lightweight utilities (the full solver
performance counters are heavier). Drive with `start_timer!` / `stop_timer!`; each
stop adds the elapsed interval to `total_time` and bumps `count`. Query with
`average_time`, `timer_stats`, `is_running`; clear with `reset_timer!`.
"""
mutable struct PerformanceTimer
    # Minimal timer used in lightweight utilities where the full solver
    # performance counters would be too heavy.
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
