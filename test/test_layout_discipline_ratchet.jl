# Ratchet on manual layout management, plus the contract for the accessors that
# replace it.
#
# A field's layout is a mutable `Symbol` (`current_layout`), and `get_grid_data` /
# `get_coeff_data` return whatever buffer the field holds without consulting it.
# Reading the wrong one is therefore not an error — it is stale or untransformed
# numbers, which is this project's dominant bug shape: a mismatch that resolves to a
# plausible value instead of raising. Nothing in the type system can catch it; an
# earlier audit (2026-07-28) established that encoding the layout as a type
# parameter is a dead end, so the only lever left is reducing how often a human has
# to get it right by hand.
#
# `grid_data!` / `coeff_data!` collapse the two-line `ensure_layout!` + `get_*_data`
# pair into one call, which removes the possibility of writing the read without the
# transform. That handles the EASY half. The hard half is code that sets a layout so
# that something else, later or elsewhere, can read it — there the transform and the
# read are not adjacent and no accessor can fuse them. Those sites are the residue
# this ratchet counts.
#
# As with test_catch_ratchet.jl and test_jet.jl: the number may fall freely and must
# not rise. New code that needs a field in a given layout should call `grid_data!` /
# `coeff_data!` and leave this count alone.

using Test
using Tarang

const LAYOUT_SRC = normpath(joinpath(@__DIR__, "..", "src"))

# `ensure_layout!(x, :g)` / `ensure_layout!(x, :c)` with a literal layout. Calls with
# a computed layout (`ensure_layout!(f, target)`) are not counted: they cannot be
# mechanically replaced, and folding them in would make the number move for reasons
# unrelated to discipline.
const LAYOUT_CALL_RE = r"ensure_layout!\(\s*[^,]+,\s*:(g|c)\s*\)"

"""Count literal-layout `ensure_layout!` call sites under `root`, skipping the file
that defines the accessors (whose own uses are the implementation, not a call site)
and skipping comment-only lines so prose about layouts does not inflate the count."""
function _count_layout_calls(root::AbstractString)
    total = 0
    per_file = Dict{String, Int}()
    for (dir, _, files) in walkdir(root), f in files
        endswith(f, ".jl") || continue
        f == "field_layout_access.jl" && continue
        path = joinpath(dir, f)
        n = 0
        for raw in eachline(path)
            code = strip(raw)
            startswith(code, "#") && continue
            occursin(LAYOUT_CALL_RE, code) && (n += 1)
        end
        n > 0 && (per_file[relpath(path, root)] = n)
        total += n
    end
    return total, per_file
end

@testset "layout-enforcing accessors" begin
    # The accessors must actually transform, not merely return a buffer. A field
    # built in grid space and then read through `coeff_data!` must come back in
    # coefficient space — if `coeff_data!` degenerated to `get_coeff_data` this
    # would return the untransformed (or empty) buffer and the whole point is lost.
    domain = PeriodicDomain(16)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x))

    ensure_layout!(u, :g)
    @test u.current_layout === :g

    c = coeff_data!(u)
    @test u.current_layout === :c          # it transformed
    @test c === get_coeff_data(u)          # and returned that layout's buffer

    g = grid_data!(u)
    @test u.current_layout === :g
    @test g === get_grid_data(u)

    # Round trip preserved the values, so the transform is real rather than a
    # relabelling of `current_layout`.
    xs = [2π * (i - 1) / 16 for i in 1:16]
    @test maximum(abs, real.(Array(g)) .- sin.(xs)) < 1e-12

    # Idempotent: calling for a layout the field is already in is a no-op read.
    @test grid_data!(u) === g
    @test u.current_layout === :g

    # Part of the supported surface, not an internal helper — the point is that new
    # code reaches for these instead of the manual pair.
    @test Tarang.is_public_api(:grid_data!)
    @test Tarang.is_public_api(:coeff_data!)
end

@testset "manual layout-management ratchet" begin
    total, per_file = _count_layout_calls(LAYOUT_SRC)

    @info "src/ literal-layout `ensure_layout!` call sites: $total"

    # Current count. Lower it when a site is folded into `grid_data!`/`coeff_data!`
    # or removed; never raise it.
    LAYOUT_RATCHET = 277

    if total > LAYOUT_RATCHET
        worst = sort(collect(per_file); by = kv -> -kv[2])
        shown = first(worst, 15)
        listing = join(("  $(lpad(n, 4))  $f" for (f, n) in shown), "\n")
        @warn "$total literal-layout `ensure_layout!` call sites in src/, ratchet is " *
              "$LAYOUT_RATCHET. If the transform is immediately followed by a read of the " *
              "same field, use `grid_data!`/`coeff_data!` instead. If it is not — if you " *
              "are setting a layout for something else to read later — that is the pattern " *
              "this ratchet exists to discourage. Heaviest files:\n" * listing
    elseif total < LAYOUT_RATCHET
        @info "Layout call-site count dropped to $total — lower LAYOUT_RATCHET in " *
              "$(basename(@__FILE__)) to match."
    end

    @test total <= LAYOUT_RATCHET

    # Sanity: the scanner must be finding things. A regex or path regression that
    # matched nothing would make this ratchet vacuously green forever.
    @test total >= 100
end
