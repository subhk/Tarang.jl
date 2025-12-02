"""
Progress reporting utilities adapted from Dedalus ``tools/progress.py``.

These helpers expose lightweight logging/printing wrappers so long-running
loops can emit periodic updates without duplicating timing logic.
"""

using Logging
using Printf

mutable struct ProgressState
    index::Int
    start_time::Float64
    last_iter_div::Int
    last_time_div::Int
end

struct ProgressLogger
    desc::String
    total::Int
    min_iter::Int
    min_seconds::Float64
    level::LogLevel
end

function ProgressLogger(desc::String, total::Int;
                        frac::Float64=1.0,
                        iter::Int=1,
                        dt::Float64=Inf,
                        level::LogLevel=Logging.Info)
    min_iter = max(1, min(iter, Int(ceil(frac * total))))
    return ProgressLogger(desc, total, min_iter, dt, level)
end

struct ProgressIterator{T}
    items::Vector{T}
    writer::Function
    prog::ProgressLogger
end

function ProgressIterator(items::Vector, writer::Function, prog::ProgressLogger)
    return ProgressIterator{eltype(items)}(items, writer, prog)
end

function log_progress(iterable, logger::Module=Logging, level::LogLevel=Logging.Info;
                      desc::String="Iteration", frac::Float64=1.0, iter::Int=1, dt::Float64=Inf)
    items = collect(iterable)
    prog = ProgressLogger(desc, length(items); frac=frac, iter=iter, dt=dt, level=level)
    writer = msg -> logger.logmsg(level, msg)
    return ProgressIterator(items, writer, prog)
end

function print_progress(iterable, io::IO=stdout; desc::String="Iteration", frac::Float64=1.0, iter::Int=1, dt::Float64=Inf)
    items = collect(iterable)
    prog = ProgressLogger(desc, length(items); frac=frac, iter=iter, dt=dt, level=Logging.Info)
    writer = msg -> println(io, msg)
    return ProgressIterator(items, writer, prog)
end

function Base.iterate(iter::ProgressIterator{T}) where T
    if isempty(iter.items)
        return nothing
    end
    state = ProgressState(0, time(), -1, -1)
    return iterate(iter, state)
end

function Base.iterate(iter::ProgressIterator{T}, state::ProgressState) where T
    index = state.index + 1
    if index > length(iter.items)
        return nothing
    end

    item = iter.items[index]
    state.index = index

    elapsed = time() - state.start_time
    completed = index

    time_div = iter.prog.min_seconds == Inf ? -1 : floor(Int, elapsed / iter.prog.min_seconds)
    iter_div = iter.prog.min_iter == 0 ? -1 : floor(Int, completed / iter.prog.min_iter)
    scheduled = (time_div > state.last_time_div) || (iter_div > state.last_iter_div)
    last_item = (completed == iter.prog.total)

    if scheduled || last_item
        state.last_time_div = time_div
        state.last_iter_div = iter_div
        message = build_message(iter.prog, completed, elapsed)
        iter.writer(message)
    end

    return item, state
end

function build_message(prog::ProgressLogger, completed::Int, elapsed::Float64)
    percent = prog.total == 0 ? 100 : round(Int, 100 * completed / prog.total)
    rate = elapsed == 0 ? 0.0 : completed / elapsed
    projected = rate == 0 ? Inf : prog.total / rate
    remaining = projected - elapsed
    return string(
        prog.desc, " ",
        completed, "/", prog.total, " (~", percent, "%) ",
        "Elapsed: ", format_time(elapsed),
        ", Remaining: ", format_time(remaining),
        ", Rate: ", rate > 0 ? @sprintf("%.2e/s", rate) : "0/s"
    )
end

function format_time(seconds::Float64)
    if !isfinite(seconds) || seconds < 0
        return "?"
    end
    total_seconds = round(Int, seconds)
    minutes, sec = divrem(total_seconds, 60)
    hours, min = divrem(minutes, 60)
    if hours > 0
        return @sprintf("%dh %02dm %02ds", hours, min, sec)
    elseif minutes > 0
        return @sprintf("%dm %02ds", minutes, sec)
    else
        return @sprintf("%ds", sec)
    end
end
