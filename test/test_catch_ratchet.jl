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

# How many CODE lines after a `catch` may be searched for a re-raise. Long enough to
# cover a handler that logs, cleans up and then rethrows; short enough that it cannot
# wander into unrelated code. The scan also stops early at a top-level `function`/`end`.
const CATCH_LOOKAHEAD = 30

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

"""Count non-overlapping `\"\"\"` markers on a single line."""
function _catch_count_triple_quotes(line::AbstractString)
    n = 0
    idx = firstindex(line)
    while idx <= lastindex(line)
        r = findnext("\"\"\"", line, idx)
        r === nothing && break
        n += 1
        idx = nextind(line, last(r))
    end
    return n
end

"""Return (bare_sites, total_catch_clauses, unbalanced_files) for every `.jl` file under `root`.

Lines inside a `\"\"\"`-delimited block (a docstring) are skipped entirely before the
`catch` regexes ever see them: English prose that mentions a `catch` block — e.g.
explaining why some code uses `throw` instead of `rethrow` — reads to a line-oriented
regex exactly like code. State is tracked per file as `in_triple`, toggled by every
`\"\"\"` marker encountered. A line that both opens and closes a docstring (an even,
nonzero marker count) is treated as fully inside one, same as a line that only opens
one whose matching close appears many lines later.

That state machine is also the scanner's own weak point: if a file's `\"\"\"` markers
do not balance — a `\"\"\"` inside a string literal, say — `in_triple` is left stuck
`true` and EVERY REMAINING LINE of that file is skipped, silently disabling the
ratchet over the rest of it. A weakened ratchet stays green, which is the one thing
a ratchet must never do. `unbalanced_files` names any file that ends mid-docstring so
the testset can fail on it."""
function _scan_catches(root::AbstractString)
    bare = Tuple{String, Int, String}[]
    silent = Tuple{String, Int, String}[]
    total = 0
    unbalanced = String[]
    for (dir, _, files) in walkdir(root), f in files
        endswith(f, ".jl") || continue
        path = joinpath(dir, f)
        in_triple = false

        # Collapse the file to its CODE lines first — comments stripped, docstring
        # bodies dropped — so the `catch` match and the rethrow look-ahead below
        # both see the same filtered view. Doing the look-ahead over raw lines would
        # let the word "rethrow" in a comment or docstring clear a genuinely silent
        # catch, which would quietly weaken the second ratchet.
        code_lines = Tuple{Int, String}[]
        for (lineno, raw) in enumerate(eachline(path))
            n_markers = _catch_count_triple_quotes(raw)
            was_in_triple = in_triple
            isodd(n_markers) && (in_triple = !in_triple)
            (was_in_triple || n_markers > 0) && continue
            code = _catch_strip_comment(raw)
            isempty(code) && continue
            push!(code_lines, (lineno, code))
        end
        in_triple && push!(unbalanced, relpath(path, root))

        for (i, (lineno, code)) in pairs(code_lines)
            occursin(CATCH_ANY_RE, code) || continue
            total += 1
            if occursin(CATCH_BARE_RE, code)
                push!(bare, (relpath(path, root), lineno, strip(code)))
            end
            # Does this handler re-raise anything? Scan forward over code lines,
            # stopping at a top-level `function`/`end` so the window cannot borrow a
            # `rethrow` from the next definition.
            reraises = false
            for (_, ahead) in code_lines[(i + 1):min(i + CATCH_LOOKAHEAD, end)]
                if occursin(r"^(function |end$)", ahead)
                    break
                end
                if occursin("rethrow", ahead) || occursin(r"\bthrow\(", ahead) ||
                   occursin(r"\berror\(", ahead)
                    reraises = true
                    break
                end
            end
            reraises || push!(silent, (relpath(path, root), lineno, strip(code)))
        end
    end
    return bare, silent, total, unbalanced
end

@testset "bare `catch` ratchet" begin
    bare, silent, total, unbalanced = _scan_catches(CATCH_RATCHET_SRC)
    n_bare = length(bare)

    @info "src/ catch clauses: $total total, $n_bare bare (unbound exception)"

    # Current count. Lower it when you remove one; never raise it.
    #
    # The population is now ZERO: every `catch` in src/ binds its exception and
    # re-raises what it did not expect. Keep it there — a bare `catch` cannot name
    # what it swallowed, and every silent-wrong-value bug this project has found
    # came from one used as control flow.
    RATCHET = 0

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
    #
    # This used to assert `n_bare >= 1` on the same reasoning, but the population is now
    # zero, so that check would force a bare `catch` to be kept alive purely to prove the
    # scanner works. `total` counts EVERY catch clause, bound or not, through the same
    # walk and the same comment/docstring stripping — so a broken scanner still fails
    # here, and a genuinely clean tree stays green.
    @test total >= 100

    # The docstring skip is the scanner's own blind spot: a file whose `"""` markers do
    # not balance leaves `in_triple` stuck true, so every line after the offending one is
    # skipped and the ratchet quietly stops scanning the rest of that file. Balanced at
    # end-of-file is the invariant that keeps the skip from weakening the ratchet.
    if !isempty(unbalanced)
        @warn "Docstring scan ended mid-`\"\"\"` block in $(length(unbalanced)) file(s); " *
              "every line after the unbalanced marker was skipped, so the ratchet did NOT " *
              "scan them:\n" * join(("  " * f for f in unbalanced), "\n")
    end
    @test isempty(unbalanced)
end

@testset "non-re-raising `catch` ratchet" begin
    # The second population, and the one the bare-catch ratchet above cannot see.
    #
    # Binding the exception was only half the fix. A handler that binds `err`, logs
    # it, and then returns a substitute value is still control flow by exception —
    # it just has a name for what it swallowed. That is the exact shape of the bugs
    # in the header comment: the parser bound its error and returned an
    # `UnknownOperator`; the BC arity probe bound its error and called the shorter
    # signature. Both would pass the ratchet above.
    #
    # This is NOT a claim that all 87 are wrong. Most are legitimate: an in-place
    # FFT that falls back to the allocating form, a sparse factorization that falls
    # back to dense, an MPI communicator free during teardown. Those reach the same
    # answer by another route, or are cleanup where there is nothing to re-raise
    # into. The ones that matter substitute a DIFFERENT NUMERICAL ANSWER, and no
    # regex can tell the two apart.
    #
    # So this is a growth ratchet, not a correctness assertion: the population may
    # shrink freely and must not grow. Adding a handler that swallows should be a
    # deliberate act that makes you come here and justify it, rather than something
    # that lands unnoticed.
    _, silent, total, _ = _scan_catches(CATCH_RATCHET_SRC)
    n_silent = length(silent)

    @info "src/ catch clauses: $total total, $n_silent with no rethrow/throw/error in the handler"

    # Current count. Lower it when you remove one; never raise it.
    SILENT_RATCHET = 87

    if n_silent > SILENT_RATCHET
        sites = sort(silent; by = x -> (x[1], x[2]))
        shown = first(sites, 25)
        listing = join(("  $f:$ln  $txt" for (f, ln, txt) in shown), "\n")
        length(sites) > length(shown) && (listing *= "\n  … and $(length(sites) - length(shown)) more")
        @warn "$n_silent non-re-raising `catch` clauses in src/, ratchet is $SILENT_RATCHET. " *
              "A handler that logs and returns a substitute value is still control flow by " *
              "exception. Either rethrow what you did not expect, or — if the fallback " *
              "genuinely reaches the same answer — say so in a comment at the site. " *
              "Full inventory (not only the new ones):\n" * listing
    elseif n_silent < SILENT_RATCHET
        @info "Non-re-raising `catch` count dropped to $n_silent — lower SILENT_RATCHET in " *
              "$(basename(@__FILE__)) to match."
    end

    @test n_silent <= SILENT_RATCHET

    # Sanity: the look-ahead must actually be discriminating. If it matched nothing,
    # every catch would look like it re-raises and the ratchet would be vacuous; if
    # it matched everything, `n_silent` would equal `total` and carry no signal.
    @test 0 < n_silent < total
end
