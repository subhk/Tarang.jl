"""
Field-presence tests are not a substitute for dispatch.

WHY THIS FILE EXISTS. `hasfield(typeof(x), :f)` answers `false` for two
different reasons — `x` genuinely has no such thing, or the field was renamed —
and the second is silent. A guard written for the first reason keeps compiling,
keeps running, and quietly stops taking its branch. This package has been bitten
by that shape repeatedly:

  * `get_scalar_size` gated its coefficient-length branches on
    `hasfield(typeof(operand), :buffers)`. `ScalarField` has no `:buffers` field
    (`TransposableField` is the only `Operand` that does), so those branches were
    unreachable — which is the only reason the function was right, because the
    Cartesian operator matrices are assembled at the GRID size and those branches
    would have returned the smaller coefficient count. A dead branch that is also
    wrong is worse than either alone: its deadness is load-bearing and nothing
    says so. The same file's `CartesianGradient` vector branch DID measure with
    `field_dofs`, and its block assembly raised `DimensionMismatch`.
  * `check_conditions(::CartesianDivergence)` and `(::CartesianLaplacian)`
    returned `false` for every `ScalarField` for the same reason, under a comment
    reading "For ScalarField or similar".
  * `get_solver_domain` opened with `hasfield(typeof(solver), :domain)`.
    `InitialValueSolver` has no `domain` field; the branch documented as the
    direct lookup could never run.
  * `get_basis_size` tried `.shape`, `.size` and `.N` — no `Basis` subtype has
    any of them — and, failing all three, returned a literal `64`.

So this file pins the two things such a guard depends on: the COUNT of remaining
`hasfield` sites (a ratchet, so the population only shrinks), and the STRUCTURAL
facts the deleted guards were asserting. If someone renames `equation_data` or
moves `current_layout`, the structural block fails loudly here instead of a
branch going quiet somewhere in the solver.

WHAT THIS TEST CANNOT SEE. It is a regex scanner over source text, like
`test_catch_ratchet.jl` and `test_layering.jl`. It cannot tell which type
actually flows into a given call site, so it does not attempt to decide
reachability; that work is done by hand and recorded in the structural block.
"""

using Test
using Tarang
using InteractiveUtils: subtypes

const HF_SRC = normpath(joinpath(@__DIR__, "..", "src"))
const HF_RE = r"hasfield\(\s*typeof\("

"""Every `.jl` file under `src/`."""
function _hf_sources()
    out = String[]
    for (root, _, files) in walkdir(HF_SRC), f in files
        endswith(f, ".jl") && push!(out, joinpath(root, f))
    end
    sort!(out)
    return out
end

"""Concrete leaves of an abstract type."""
function _hf_concrete(T::Type)
    out = Type[]
    stack = Type[T]
    while !isempty(stack)
        S = pop!(stack)
        isabstracttype(S) ? append!(stack, subtypes(S)) : push!(out, S)
    end
    return out
end

# Code-only sites (comments and docstrings excluded) after the 2026-08-07 sweep,
# down from 281. Lower it when you remove more; never raise it.
#
# The largest remaining block is expression-tree traversal — 43 sites across 20
# files asking for `:operand` / `:operands` / `:left` / `:right` to walk an
# operator node. Those want an `operator_children(::Operator)` interface, which
# is its own change: the sites look uniform but differ in what they do with the
# children, so they cannot be swept mechanically.
const HASFIELD_RATCHET = 183

# Symbols whose guards were PROVED decidable and removed. A `hasfield` on any of
# these is either always true or always false at every site that can reach it,
# so re-introducing one puts back a branch that cannot do what it looks like it
# does. The value is the reason.
const HASFIELD_BANNED = Dict(
    :equation_data => "all four Problem subtypes declare it — the guard is always true",
    :buffers       => "only TransposableField has it, and it has no get_coeff_data/get_grid_data method",
)

@testset "hasfield dispatch ratchet" begin

    # Prose discusses these guards at length — that is the point of removing
    # them — so `#` comments and docstring bodies are stripped before matching,
    # or the notes explaining a deletion would count as the thing deleted.
    sites = Tuple{String, Int, String}[]
    for path in _hf_sources()
        in_docstring = false
        for (i, line) in enumerate(eachline(path))
            code = strip(line)
            ndq = length(collect(eachmatch(r"\"\"\"", code)))
            if in_docstring
                isodd(ndq) && (in_docstring = false)
                continue
            elseif isodd(ndq)
                in_docstring = true
                continue
            end
            startswith(code, "#") && continue
            h = findfirst('#', code)
            h === nothing || (code = strip(code[1:prevind(code, h)]))
            occursin(HF_RE, code) || continue
            push!(sites, (relpath(path, HF_SRC), i, code))
        end
    end

    @testset "count only shrinks" begin
        @test length(sites) <= HASFIELD_RATCHET
        # A zero here would mean the scanner broke, not that the work is done.
        @test length(sites) > 0
    end

    @testset "no guard on a structurally guaranteed field" begin
        for (sym, why) in HASFIELD_BANNED
            offenders = [(f, i, txt) for (f, i, txt) in sites
                         if occursin(Regex("hasfield\\(\\s*typeof\\([^)]*\\)\\s*,\\s*:$(sym)\\b"), txt)]
            @test isempty(offenders) || (@info "banned hasfield(:$sym) — $why" offenders; false)
        end
    end

    # ---------------------------------------------------------------------
    # The structural facts the deleted guards were asserting. Each of these
    # failing means a rename happened and some hand-removed guard would have
    # gone silently false.
    # ---------------------------------------------------------------------
    @testset "Problem: equation_data on every subtype" begin
        problems = _hf_concrete(Tarang.Problem)
        @test length(problems) == 4
        for P in problems
            @test hasfield(P, :equation_data)
            @test hasfield(P, :variables)
            @test hasfield(P, :domain)
        end
        # IVP-only fields. These are real polymorphism, so they must NOT become
        # universal either — if they do, the dispatch that replaced their guards
        # is wrong.
        @test count(P -> hasfield(P, :stochastic_forcings), problems) == 1
        @test count(P -> hasfield(P, :temporal_filters), problems) == 1
        @test count(P -> hasfield(P, :time), problems) == 1
    end

    @testset "Basis: everything lives under meta" begin
        bases = _hf_concrete(Tarang.Basis)
        @test length(bases) >= 8
        for B in bases
            @test hasfield(B, :meta)
            M = fieldtype(B, :meta)
            @test hasfield(M, :size)
            @test hasfield(M, :bounds)
            @test hasfield(M, :element_label)
            @test hasfield(M, :coordsys)
            # The spellings `get_basis_size` used to try before giving up and
            # returning 64. None of them exists; that is why it always returned 64.
            @test !hasfield(B, :size)
            @test !hasfield(B, :shape)
            @test !hasfield(B, :N)
            @test !hasfield(B, :bounds)
        end
    end

    @testset "InitialValueSolver: what it does and does not carry" begin
        S = Tarang.InitialValueSolver
        @test hasfield(S, :problem)
        @test hasfield(S, :state)
        # `get_solver_domain` opened by testing for this. It has never existed.
        @test !hasfield(S, :domain)
    end

    @testset "Operand: layout and storage belong to one type each" begin
        operands = _hf_concrete(Tarang.Operand)
        @test count(O -> hasfield(O, :current_layout), operands) == 1
        @test hasfield(Tarang.ScalarField, :current_layout)
        @test count(O -> hasfield(O, :buffers), operands) == 1
        @test hasfield(Tarang.TransposableField, :buffers)
        # ...which is exactly why `hasfield(..., :buffers)` could never stand in
        # for "this operand has data": ScalarField answers false.
        @test !hasfield(Tarang.ScalarField, :buffers)
    end

    @testset "the accessors that replaced the guards" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb,), Float64)

        @test Tarang.operand_layout(u) === u.current_layout
        @test Tarang.operand_domain(u) === u.domain

        # An operator node carries neither, and says so rather than erroring.
        d = Tarang.Differentiate(u, coords["x"], 1)
        @test Tarang.operand_layout(d) === nothing
        @test Tarang.operand_domain(d) === nothing

        # Non-Operands are not silently absorbed.
        @test_throws MethodError Tarang.operand_layout(42)
        @test_throws MethodError Tarang.operand_domain("x")
    end
end
