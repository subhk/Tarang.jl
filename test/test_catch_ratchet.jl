# Ratchet on BARE `catch` clauses in src/ — a `catch` that binds no exception.
#
# Tarang's dominant failure mode is the silent wrong value: a mismatch that resolves to a
# plausible number instead of raising (see test_silent_zero_terms.jl). Every instance found so
# far shares one mechanic — a `catch` used as control flow, where a genuine failure is
# indistinguishable from an expected miss and a substitute value is returned instead.
#
# Recent examples, all of which passed the full suite before they were found:
#   * the parser dropped a `lift` operator on auto-detect failure, leaving a tau-method solve
#     ~100% wrong (test_lift_not_dropped.jl);
#   * `parse_expression` turned any evaluation error into an `UnknownOperator` placeholder,
#     which then masked the fix above;
#   * BC callbacks probed their arity by exception, so an error inside a correctly-shaped
#     callback silently degraded it to a shorter call (test_bc_function_arity.jl);
#   * `MPI.Allgather` was wrapped in try/catch, diverging the communicator on failure.
#
# A bare `catch` cannot name what it swallowed and cannot distinguish the failure it expects
# from one it does not. This test does not forbid them — it stops the population growing while
# the existing ones are worked through. Follow the JET ratchet idiom in test_jet.jl.
#
# TO FIX A FAILURE HERE: bind and inspect the exception, e.g.
#     catch e
#         e isa MethodError || rethrow()
#         ...
# Do NOT raise the ratchet to make this pass. Lower it when you remove one.

using Test

const CATCH_RATCHET_SRC = normpath(joinpath(@__DIR__, "..", "src"))

# `catch` as its own clause with nothing bound: end-of-line, or `catch;` in an inline
# `try ... catch; ... end`. The trailing `[^A-Za-z0-9_!]` guard keeps `catch_backtrace()`
# from matching.
const CATCH_BARE_RE = r"(^|[^A-Za-z0-9_!])catch[ \t]*(;|$)"
const CATCH_ANY_RE = r"(^|[^A-Za-z0-9_!])catch([^A-Za-z0-9_!]|$)"

"""Drop a trailing `#` comment, ignoring `#` inside a double-quoted string."""
function _catch_strip_comment(line::AbstractString)
    in_string = false
    escaped = false
    for (i, c) in pairs(line)
        if escaped
            escaped = false
        elseif c == '\\'
            escaped = true
        elseif c == '"'
            in_string = !in_string
        elseif c == '#' && !in_string
            return rstrip(line[1:prevind(line, i)])
        end
    end
    return rstrip(line)
end

"""Return (bare_sites, total_catch_clauses) for every `.jl` file under `root`."""
function _scan_catches(root::AbstractString)
    bare = Tuple{String, Int, String}[]
    total = 0
    for (dir, _, files) in walkdir(root), f in files
        endswith(f, ".jl") || continue
        path = joinpath(dir, f)
        for (lineno, raw) in enumerate(eachline(path))
            code = _catch_strip_comment(raw)
            isempty(code) && continue
            occursin(CATCH_ANY_RE, code) || continue
            total += 1
            if occursin(CATCH_BARE_RE, code)
                push!(bare, (relpath(path, root), lineno, strip(code)))
            end
        end
    end
    return bare, total
end

@testset "bare `catch` ratchet" begin
    bare, total = _scan_catches(CATCH_RATCHET_SRC)
    n_bare = length(bare)

    @info "src/ catch clauses: $total total, $n_bare bare (unbound exception)"

    # Current count. Lower it when you remove one; never raise it.
    RATCHET = 54

    if n_bare > RATCHET
        # There is no stored baseline, so this is the full inventory, not just the new ones.
        # Diff it against the ratchet count to find what your branch added.
        sites = sort(bare; by = x -> (x[1], x[2]))
        shown = first(sites, 25)
        listing = join(("  $f:$ln  $txt" for (f, ln, txt) in shown), "\n")
        length(sites) > length(shown) && (listing *= "\n  … and $(length(sites) - length(shown)) more")
        @warn "$n_bare bare `catch` clauses in src/, ratchet is $RATCHET. Bind the exception " *
              "and rethrow what you did not expect, rather than returning a substitute " *
              "value. Full inventory (not only the new ones):\n" * listing
    elseif n_bare < RATCHET
        @info "Bare `catch` count dropped to $n_bare — lower RATCHET in $(basename(@__FILE__)) to match."
    end

    @test n_bare <= RATCHET

    # Sanity: the scanner must actually be finding things. A regex or path regression that
    # silently matched nothing would make the ratchet vacuously green.
    @test total >= 100
    @test n_bare >= 1
end
