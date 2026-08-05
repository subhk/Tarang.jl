"""
Layer direction: a file must not call into a layer that loads after it.

WHY THIS FILE EXISTS. `src/load_order.jl` alternates between owners —
core, tools, core, tools, core, tools — and that alternation is not a style
choice, it is the shape of the dependency graph. `src/core/module_contracts.jl`
has advertised a six-submodule split since the beginning; the split was never
built, and could not be, because core code calls into layers that load after it.

Nothing catches that on its own. Julia resolves a call at run time, not at
include time, so a `core/` function body may call a `tools/` function defined
several stages later and the package loads and runs perfectly. The inversion is
invisible until someone tries to lift `core` into a module and discovers it drags
the NetCDF layer with it. That is exactly how
`field_layout_arithmetic_io.jl` — half field arithmetic, half NetCDF save/load —
sat in `core/` at load stage 3 calling the slab layer at stage 9.

So this test derives the load stages from `load_order.jl` itself (the source of
truth, not a hardcoded list), then flags every call from one owner into a name
first defined by a DIFFERENT owner at a LATER stage. The remaining violations are
enumerated below with the reason each survives; the count is a ratchet.

WHAT THIS TEST CANNOT SEE. It is a regex scanner, not a parser. It skips comments
and docstring bodies (prose like `f(x) = ...` reads as a definition otherwise)
and ignores names Base already owns (extending `Base.push!` is not a layer entry
point). Dynamic calls, names built at run time, and dispatch through an abstract
interface are invisible to it. It is a direction check, not a proof.
"""

using Test

const LAYER_SRC = normpath(joinpath(@__DIR__, "..", "src"))
const LAYER_INCLUDE_RE = r"^\s*include\(\"([^\"]+)\"\)"

"""Recursively expand a loader into every file it pulls in, in order."""
function _layer_expand(path, acc = String[])
    isfile(path) || return acc
    push!(acc, path)
    dir = dirname(path)
    for line in eachline(path)
        m = match(LAYER_INCLUDE_RE, line)
        m === nothing && continue
        _layer_expand(normpath(joinpath(dir, m.captures[1])), acc)
    end
    return acc
end

"""The load stages, in order, as `load_order.jl` declares them."""
function _layer_stages()
    stages = String[]
    for line in eachline(joinpath(LAYER_SRC, "load_order.jl"))
        m = match(LAYER_INCLUDE_RE, line)
        m === nothing && continue
        push!(stages, normpath(joinpath(LAYER_SRC, m.captures[1])))
    end
    return stages
end

_layer_owner(path) = first(split(replace(relpath(path, LAYER_SRC), '\\' => '/'), '/'))

"""Code lines only, as `(lineno, text)`. Comments and docstring bodies are
dropped: they contain prose that a regex cannot distinguish from code."""
function _layer_code_lines(path)
    out = Tuple{Int, String}[]
    in_docstring = false
    for (i, raw) in enumerate(eachline(path))
        code = strip(raw)
        fences = count("\"\"\"", code)
        if in_docstring
            in_docstring = !isodd(fences)
            continue
        elseif isodd(fences)
            in_docstring = true
            continue
        end
        (isempty(code) || startswith(code, "#")) && continue
        push!(out, (i, code))
    end
    return out
end

const _LAYER_DEF_RES = (
    r"^(?:@inline\s+|@propagate_inbounds\s+)?function\s+([a-zA-Z_][a-zA-Z_0-9!]*)\s*\(",
    # Short form. `[^=]*` and the lookahead keep `f(x) == y` and `f(x) => y` out.
    r"^(?:@inline\s+)?([a-zA-Z_][a-zA-Z_0-9!]*)\s*\([^=]*\)\s*=(?![=>])",
    r"^(?:mutable\s+)?struct\s+([A-Z][a-zA-Z_0-9]*)",
    r"^abstract\s+type\s+([A-Z][a-zA-Z_0-9]*)",
)
const _LAYER_CALL_RE = r"(?<![A-Za-z_0-9.])([a-zA-Z_][a-zA-Z_0-9!]*)\("

"""Every cross-owner call that runs backwards against the load order, as
`name => [site, ...]`."""
function _layer_violations()
    stages = _layer_stages()

    stage_of = Dict{String, Int}()
    for (i, stage) in enumerate(stages), file in _layer_expand(stage)
        haskey(stage_of, file) || (stage_of[file] = i)
    end

    first_def = Dict{String, Int}()
    for (file, stage) in stage_of, (_, code) in _layer_code_lines(file), re in _LAYER_DEF_RES
        m = match(re, code)
        m === nothing && continue
        name = m.captures[1]
        first_def[name] = min(get(first_def, name, typemax(Int)), stage)
    end
    # A name Base already owns is being extended, not introduced by a layer.
    filter!(kv -> !isdefined(Base, Symbol(kv[1])), first_def)

    violations = Dict{String, Vector{String}}()
    for (file, stage) in stage_of
        owner = _layer_owner(file)
        for (lineno, code) in _layer_code_lines(file), m in eachmatch(_LAYER_CALL_RE, code)
            name = m.captures[1]
            defined_at = get(first_def, name, 0)
            defined_at > stage || continue
            _layer_owner(stages[defined_at]) == owner && continue
            push!(get!(violations, name, String[]),
                  "$(relpath(file, LAYER_SRC)):$lineno  [$owner stage $stage -> " *
                  "$(_layer_owner(stages[defined_at])) stage $defined_at]")
        end
    end
    return violations, stages
end

# The violations that remain, each with why it is still here. Shrink this list;
# do not grow it. A NEW name appearing here fails the test even if the total
# count happens to stay flat.
const LAYER_ALLOWLIST = Dict(
    # Box-drawing helpers in tools/pretty_printing.jl, the very last stage.
    # extras/ formats CFL diagnostics with them. Cheap to fix by moving the
    # helpers down into a bootstrap file; not done here because it is cosmetic
    # and touches an unrelated file.
    "_box_text"          => 9,
    "_box_line"          => 4,
    "_box_text_centered" => 1,

    # tools/dispatch.jl is the operator-construction framework; core/arithmetic.jl
    # supplies the `dispatch_preprocess` / `dispatch_check` methods for concrete
    # operator types. This is the plugin direction (framework below,
    # implementations above) and is only flagged because the framework declares no
    # generic fallback of its own. Giving it one would make the intent explicit.
    "dispatch_preprocess" => 2,
    "dispatch_check"      => 2,

    # core/solvers/solver_stepping.jl drives output handlers directly. Removing
    # these needs the handler interface that core/evaluator.jl also wants — the
    # evaluator stores `netcdf_handlers` as a struct field, so the fix is an
    # abstract handler protocol, not a call-site edit.
    "process!" => 2,
    "close!"   => 1,

    # core reaching up into extras/ for CFL and convenience constructors.
    "compute_timestep" => 1,
    "add_parameters!"  => 1,

    # Progress reporting decorating a core build loop. One call site; needs an
    # injected reporter to remove.
    "log_progress" => 1,
)

@testset "layer direction" begin
    violations, stages = _layer_violations()

    # The scanner must actually be reading the project. A path or regex
    # regression that found nothing would make this test vacuously green.
    @test length(stages) >= 10
    @test any(s -> _layer_owner(s) == "core", stages)
    @test any(s -> _layer_owner(s) == "tools", stages)

    total = sum(length(v) for v in values(violations); init = 0)
    @info "cross-layer backward references: $(length(violations)) names, $total sites"

    # No name may appear that is not accounted for above.
    unexpected = setdiff(keys(violations), keys(LAYER_ALLOWLIST))
    if !isempty(unexpected)
        listing = join(("  $n\n" * join(("    " .* violations[n]), "\n")
                        for n in sort(collect(unexpected))), "\n")
        @warn "New cross-layer backward reference(s). A file is calling into a layer " *
              "that loads after it; the package still runs, because Julia resolves the " *
              "call at run time, but the layer boundary is now false. Move the code to " *
              "the layer that owns what it calls, or add it to LAYER_ALLOWLIST with the " *
              "reason it must stay:\n" * listing
    end
    @test isempty(unexpected)

    # And no allowlisted name may spread to more sites than it holds today.
    for (name, cap) in LAYER_ALLOWLIST
        @test length(get(violations, name, String[])) <= cap
    end

    @test total <= 24
end

@testset "NetCDF persistence is not in core" begin
    # `save_field`/`load_field!` and the checkpoint pair read and write NetCDF, so
    # they belong in the layer that owns it. They used to live under `core/` —
    # `field_layout_arithmetic_io.jl` at stage 3 and `solvers/solver_checkpoint.jl`
    # at stage 6 — calling a slab layer that loads at stage 9. Pin the move so
    # they do not drift back.
    stages = _layer_stages()
    stage_of = Dict{String, Int}()
    for (i, stage) in enumerate(stages), file in _layer_expand(stage)
        haskey(stage_of, file) || (stage_of[file] = i)
    end

    slab_stage = nothing
    for (file, stage) in stage_of
        endswith(file, "netcdf_slab_io.jl") && (slab_stage = stage)
    end
    @test slab_stage !== nothing

    for basename_ in ("field_netcdf_io.jl", "solver_checkpoint.jl")
        matches = [f for f in keys(stage_of) if endswith(f, basename_)]
        @test length(matches) == 1
        file = only(matches)
        # Above the slab layer it calls, and owned by tools/, not core/.
        @test stage_of[file] >= slab_stage
        @test _layer_owner(file) == "tools"
    end

    # The arithmetic half stayed in core and must not have taken the I/O with it.
    arith = [f for f in keys(stage_of) if endswith(f, "field_layout_arithmetic.jl")]
    @test length(arith) == 1
    @test _layer_owner(only(arith)) == "core"
    arith_src = read(only(arith), String)
    @test !occursin("write_local_slab", arith_src)
    @test !occursin("netcdf_file_info", arith_src)
end
