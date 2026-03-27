"""
Progress reporting utilities.

These helpers expose lightweight logging/printing wrappers so long-running
loops can emit periodic updates without duplicating timing logic.
"""

using Logging
using Printf

export ProgressLogger, ProgressIterator, ProgressState,
       log_progress, print_progress, format_progress_time

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
    # Validate total is non-negative
    if total < 0
        throw(ArgumentError("ProgressLogger: total must be non-negative, got $total"))
    end
    # Clamp frac to valid range (0, 1]
    frac = clamp(frac, 0.001, 1.0)
    # Clamp iter to be positive
    iter = max(1, iter)
    # Compute min_iter: at least 1, at most iter or frac*total
    min_iter = total == 0 ? 1 : max(1, min(iter, Int(ceil(frac * total))))
    min_seconds = (!isfinite(dt) || dt <= 0) ? Inf : dt
    return ProgressLogger(desc, total, min_iter, min_seconds, level)
end

struct ProgressIterator{T}
    items::Vector{T}
    writer::Function
    prog::ProgressLogger
end
# Julia automatically creates ProgressIterator(items::Vector{T}, ...) where T

function log_progress(iterable; level::LogLevel=Logging.Info,
                      desc::String="Iteration", frac::Float64=1.0, iter::Int=1, dt::Float64=Inf)
    items = iterable isa Vector ? iterable : collect(iterable)
    prog = ProgressLogger(desc, length(items); frac=frac, iter=iter, dt=dt, level=level)
    # Create a writer closure that logs at the specified level
    writer = msg -> @logmsg level msg
    return ProgressIterator(items, writer, prog)
end

function print_progress(iterable, io::IO=stdout; desc::String="Iteration", frac::Float64=1.0, iter::Int=1, dt::Float64=Inf)
    items = iterable isa Vector ? iterable : collect(iterable)
    prog = ProgressLogger(desc, length(items); frac=frac, iter=iter, dt=dt, level=Logging.Info)
    writer = msg -> println(io, msg)
    return ProgressIterator(items, writer, prog)
end

# Iterator protocol methods
Base.length(iter::ProgressIterator) = length(iter.items)
Base.size(iter::ProgressIterator) = (length(iter.items),)
Base.eltype(::Type{ProgressIterator{T}}) where T = T
Base.IteratorSize(::Type{<:ProgressIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ProgressIterator}) = Base.HasEltype()

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

    # Compute division indices for scheduling (min_iter is always >= 1, min_seconds is Inf or > 0)
    time_div = iter.prog.min_seconds == Inf ? -1 : floor(Int, elapsed / iter.prog.min_seconds)
    iter_div = floor(Int, completed / iter.prog.min_iter)
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
    percent = prog.total == 0 ? 100 : clamp(round(Int, 100 * completed / prog.total), 0, 100)
    rate = elapsed == 0 ? 0.0 : completed / elapsed
    projected = rate == 0 ? Inf : prog.total / rate
    remaining = max(0.0, projected - elapsed)  # Don't show negative remaining time
    return string(
        prog.desc, " ",
        completed, "/", prog.total, " (~", percent, "%) ",
        "Elapsed: ", format_progress_time(elapsed),
        ", Remaining: ", format_progress_time(remaining),
        ", Rate: ", rate > 0 ? @sprintf("%.2e/s", rate) : "0/s"
    )
end

# format_time is defined in general.jl
# This local version handles edge cases for progress display
function format_progress_time(seconds::Float64)
    if !isfinite(seconds) || seconds < 0
        return "?"
    end
    # Guard against overflow for very large values (> ~68 years in seconds)
    if seconds > 2.0e9
        return ">63 years"
    end
    total_seconds = round(Int, seconds)
    minutes, sec = divrem(total_seconds, 60)
    hours, min = divrem(minutes, 60)
    days, hours = divrem(hours, 24)
    if days > 0
        return @sprintf("%dd %02dh %02dm", days, hours, min)
    elseif hours > 0
        return @sprintf("%dh %02dm %02ds", hours, min, sec)
    elseif minutes > 0
        return @sprintf("%dm %02ds", minutes, sec)
    else
        return @sprintf("%ds", sec)
    end
end
