# ONE statement of "which axes are decomposed".
#
# This convention used to be re-derived by hand at nine call sites. Nine copies
# of one rule is how the PencilArrays convention (decompose LAST mesh dims) and
# the TransposableField convention (decompose FIRST mesh dims) drifted apart.
# This file pins the single source of truth and ratchets against new copies.
using Test
using Tarang

# A Distributor stand-in: decomposed_axes reads only these three fields, so the
# convention can be tested for mesh/ndim combinations that need no live MPI
# world (the duck-typed-fake-dist trick from test_netcdf_slab_geometry.jl).
struct FakeDist
    size::Int
    mesh::Union{Nothing, Tuple{Vararg{Int}}}
    use_pencil_arrays::Bool
end

# ---------------------------------------------------------------------------
# The re-derivation scanner. Two independent tells, checked line-by-line so a
# line that CITES decomposed_axes (rather than re-deriving alongside it) can
# be exempted without blinding the scan to the rest of the file:
#
#   * PROSE  -- the convention restated in words: "decompose(s) [the]
#     LAST/FIRST ..." in either word order. Cheap to write, cheap to omit.
#   * ARITHMETIC -- the convention re-derived in code, which a comment-less
#     copy can't dodge. The two shapes distributor_mpi.jl, netcdf_output.jl,
#     and nonlinear_transforms.jl actually had before migration:
#       - a coordinate variable bound straight off mesh[1] (`P1 = mesh[1]`,
#         then `coord1 = rank % P1`) instead of asking mesh_axis_for which
#         axis that even is;
#       - the PencilArrays decomp_dims closure `ntuple(i -> ndim - ndims_mesh
#         + i, ndims_mesh)` instead of calling decomposed_axes.
const CONVENTION_PROSE_RE =
    r"decompose[sd]?\s+(the\s+)?(LAST|FIRST)|mesh\s+decompose[sd]?\s+the\s+(LAST|FIRST)"i
const CONVENTION_ARITHMETIC_RE =
    r"\w+\s*=\s*(dist\.)?mesh\[1\]|n(dims?_)?mesh\w*\s*\+\s*i\b"

"""
    line_rederives_convention(line) -> Bool

Does this ONE line either restate or re-derive "which axes are decomposed",
without also citing `decomposed_axes` as the authority?
"""
function line_rederives_convention(line::AbstractString)
    occursin("decomposed_axes", line) && return false
    return occursin(CONVENTION_PROSE_RE, line) || occursin(CONVENTION_ARITHMETIC_RE, line)
end

"""
    file_rederives_convention(path) -> Bool

Scan one file for `line_rederives_convention`, gated on the file mentioning
`use_pencil_arrays` at all (both conventions only mean something relative to
that flag; gating avoids matching unrelated LAST/FIRST prose elsewhere).
"""
function file_rederives_convention(path::AbstractString)
    text = read(path, String)
    occursin(r"use_pencil_arrays"i, text) || return false
    for line in split(text, '\n')
        line_rederives_convention(line) && return true
    end
    return false
end
# ---------------------------------------------------------------------------

@testset "decomposed_axes" begin

    @testset "serial and unmeshed decompose nothing" begin
        @test Tarang.decomposed_axes(FakeDist(1, (4,), true), 3) == ()
        @test Tarang.decomposed_axes(FakeDist(4, nothing, true), 3) == ()
        @test Tarang.decomposed_axes(FakeDist(1, (2, 2), false), 2) == ()
    end

    @testset "PencilArrays decomposes the LAST mesh dims" begin
        @test Tarang.decomposed_axes(FakeDist(4, (4,), true), 2) == (2,)
        @test Tarang.decomposed_axes(FakeDist(4, (4,), true), 3) == (3,)
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 3) == (2, 3)
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 2) == (1, 2)
    end

    @testset "TransposableField decomposes the FIRST mesh dims, at most two" begin
        @test Tarang.decomposed_axes(FakeDist(4, (4, 1), false), 2) == (1, 2)
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), false), 3) == (1, 2)
        @test Tarang.decomposed_axes(FakeDist(2, (2,), false), 2) == (1,)
    end

    @testset "pencil path cannot decompose more dims than the field has" begin
        # get_local_array_size leaves the shape untouched when ndim < length(mesh);
        # decomposed_axes must agree or the allocator and the index math diverge.
        @test Tarang.decomposed_axes(FakeDist(4, (2, 2), true), 1) == ()
    end

    @testset "mesh_axis_for inverts decomposed_axes" begin
        d = FakeDist(4, (2, 2), true)
        @test Tarang.mesh_axis_for(d, 3, 1) === nothing
        @test Tarang.mesh_axis_for(d, 3, 2) == 1
        @test Tarang.mesh_axis_for(d, 3, 3) == 2
        @test Tarang.is_decomposed_axis(d, 3, 3)
        @test !Tarang.is_decomposed_axis(d, 3, 1)

        t = FakeDist(4, (2, 2), false)
        @test Tarang.mesh_axis_for(t, 3, 1) == 1
        @test Tarang.mesh_axis_for(t, 3, 2) == 2
        @test Tarang.mesh_axis_for(t, 3, 3) === nothing
    end

    @testset "out-of-range axes are not decomposed" begin
        d = FakeDist(4, (2, 2), true)
        @test Tarang.mesh_axis_for(d, 3, 0) === nothing
        @test Tarang.mesh_axis_for(d, 3, 4) === nothing
    end
end

@testset "get_local_range agrees with the convention" begin
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())
    # Serial: every axis is whole.
    for axis in 1:3
        @test Tarang.get_local_range(dist, 12, axis) == (1, 12)
    end
end

@testset "get_process_coordinate accepts an explicit rank" begin
    # Untyped (duck-typed like decomposed_axes/mesh_axis_for), and its `rank`
    # argument defaults to `dist.rank` but can name ANY rank -- what
    # distributor_mpi.jl's scatter loop and netcdf_output.jl's
    # get_local_shape/get_local_start (both driven by an explicit `rank`, not
    # necessarily this process's own) both need. FakeDist has no `.rank`
    # field at all, so passing rank explicitly below also confirms the
    # `rank=dist.rank` default is never evaluated unless omitted.
    d1 = FakeDist(4, (4,), false)
    @test [Tarang.get_process_coordinate(d1, 1, r) for r in 0:3] == [0, 1, 2, 3]

    # 2-D mesh, column-major: coord[1] = rank % mesh[1], coord[2] = (rank ÷
    # mesh[1]) % mesh[2] -- matches the OLD coord1/coord2/dest_coord1/
    # dest_coord2 formulas at distributor_mpi.jl and netcdf_output.jl.
    d2 = FakeDist(6, (2, 3), false)
    @test [Tarang.get_process_coordinate(d2, 1, r) for r in 0:5] == [0, 1, 0, 1, 0, 1]
    @test [Tarang.get_process_coordinate(d2, 2, r) for r in 0:5] == [0, 0, 1, 1, 2, 2]
end

@testset "allocator and index math agree on every axis" begin
    # get_local_array_size decides the ALLOCATED shape; local_indices decides
    # which global indices those slots mean. If they disagree the field is
    # silently mis-addressed — no error, wrong values.
    #
    # SERIAL SMOKE CHECK ONLY: at mesh=(1,) with size==1, every function below
    # takes its identity early-return, so this exercises none of the
    # decomposed branches. The load-bearing three-way agreement assertion —
    # under LIVE decomposition, np=2 and np=4 — lives in
    # test/test_mpi_local_indices.jl.
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, architecture=CPU())
    gshape = (8, 6, 4)
    local_shape = Tarang.get_local_array_size(dist, gshape)
    for axis in 1:3
        @test length(Tarang.local_indices(dist, axis, gshape[axis])) == local_shape[axis]
    end
    @test collect(Tarang.compute_local_shape(dist, gshape)) == collect(local_shape)
end

@testset "the convention is stated in exactly one place" begin
    # HISTORY: the convention was re-derived by hand at TEN sites in the
    # original inventory (Tasks 2-6); all ten are migrated onto
    # decomposed_axes. A PROSE-only regex then found a SECOND wave: five more
    # sites (Task 14), found only once the scan could see verb-first prose
    # with no fixed suffix after LAST/FIRST ("decompose LAST dimensions", not
    # just "...LAST ndims_mesh dims") — including two functions
    # (netcdf_output.jl's get_local_shape/get_local_start) where only the
    # PencilArrays half was migrated and the TransposableField half was left
    # hand-rolled. All five are now migrated onto decomposed_axes /
    # mesh_axis_for / get_local_range / get_process_coordinate.
    #
    # A prose-only scan is a weak tell: a comment is the easiest thing to
    # paraphrase or omit, and a re-derivation with NO comment at all is
    # invisible to it. So the scan below adds an ARITHMETIC check as the
    # primary signal — the actual re-derived CODE shapes, not just the prose
    # around them — file_rederives_convention/line_rederives_convention
    # (defined above) run both. A line that also names `decomposed_axes` is
    # citing the authority, not re-deriving independently, so it is exempt
    # even if it also matches PROSE or ARITHMETIC (see transform_planning.jl
    # and field_data_distributor_utils.jl, which explain the convention in
    # error text that now cites decomposed_axes by name).
    #
    # KNOWN_OFFENDERS below (declared after the scan) is an explicit, named,
    # already-triaged list — same idiom as test_catch_ratchet.jl's bare-
    # `catch` ratchet — not a magic count. THE TARGET IS AN EMPTY DICT, and
    # after Task 14 it IS empty: every site from both waves is migrated.
    # Every future entry is a TODO for a follow-up migration task, not a
    # permanent exemption.
    srcdir = joinpath(@__DIR__, "..", "src")
    allowed = joinpath("core", "distributor", "distributor_core.jl")

    offenders = String[]
    for (root, _, files) in walkdir(srcdir), file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        occursin(allowed, path) && continue
        file_rederives_convention(path) && push!(offenders, relpath(path, srcdir))
    end

    found = Set(offenders)
    @info "convention re-derivation scan: $(length(found)) file(s) match" sort(collect(found))

    # THE TARGET: zero known offenders. A new hand-rolled re-derivation
    # anywhere in src/ (outside distributor_core.jl, the authority file)
    # fails this test by breaking set equality below.
    KNOWN_OFFENDERS = Dict{String, String}()
    known = Set(keys(KNOWN_OFFENDERS))

    new_offenders = sort(collect(setdiff(found, known)))
    fixed_offenders = sort(collect(setdiff(known, found)))

    if !isempty(new_offenders)
        @warn "convention re-derived in file(s) NOT in KNOWN_OFFENDERS — migrate onto " *
              "decomposed_axes, or if this is a deliberate new exemption, add it there " *
              "with a one-line reason:\n" *
              join(("  " * f for f in new_offenders), "\n")
    end
    if !isempty(fixed_offenders)
        @info "file(s) listed in KNOWN_OFFENDERS no longer re-derive the convention — " *
              "delete their entries to tighten the ratchet:\n" *
              join(("  $f  (was: $(KNOWN_OFFENDERS[f]))" for f in fixed_offenders), "\n")
    end

    # SET EQUALITY, not a subset check. `offenders ⊆ KNOWN_OFFENDERS` would
    # only ever catch a NEW offender; a listed file that gets fixed would just
    # keep sitting in KNOWN_OFFENDERS with no signal to remove it, so the list
    # could only grow or hold — never shrink — and would rot into a permanent
    # excuse. Set equality makes the ratchet bite in BOTH directions: a new
    # file joining `found` fails ("the convention was re-derived somewhere
    # new"), and a listed file dropping out of `found` WITHOUT its entry being
    # deleted also fails ("the ratchet must tighten when you fix one") —
    # exactly the discipline test_catch_ratchet.jl asks a human to apply by
    # hand when lowering RATCHET, enforced here by the assertion itself.
    @test found == known
end

@testset "the scanner discriminates re-derivations from citations" begin
    # ARITHMETIC family: the two code shapes the five sites actually had
    # before Task 14 (see CONVENTION_ARITHMETIC_RE above).
    @test line_rederives_convention("        P1 = mesh[1]")
    @test line_rederives_convention("        P2 = dist.mesh[1]")
    @test line_rederives_convention(
        "            ntuple(i -> ndims_global - ndims_mesh + i, ndims_mesh)")
    @test line_rederives_convention("    ntuple(i -> ndim - ndims_mesh + i, ndims_mesh)")

    # PROSE family, both word orders.
    @test line_rederives_convention("# decompose LAST dims")
    @test line_rederives_convention("# mesh decomposes the FIRST dims")
    @test line_rederives_convention("decomposes the LAST \$mesh_dim dimension(s)")

    # Citing decomposed_axes on the SAME line exempts it, even though the
    # line still contains the arithmetic or prose shape being explained --
    # this is what lets transform_planning.jl's error message and
    # field_data_distributor_utils.jl's error strings stay readable instead
    # of being mangled just to dodge the regex.
    @test !line_rederives_convention("P1 = mesh[1]  # superseded by decomposed_axes")
    @test !line_rederives_convention(
        "decomposes the LAST \$mesh_dim dimension(s), per decomposed_axes")
    @test !line_rederives_convention("    decomp_dims = decomposed_axes(dist, ndims_global)")

    # Ordinary code/prose that merely mentions "mesh" or "last" must NOT trip
    # either family -- a generic index into mesh, or English prose with no
    # "decompose" nearby.
    @test !line_rederives_convention("    n_procs = mesh[mesh_dim_idx]")
    @test !line_rederives_convention(
        "    proc_coord = get_process_coordinate(dist, mesh_dim_idx, rank)")
    @test !line_rederives_convention("# The last argument is the mesh size.")

    # Direction 1: a hand-rolled derivation pasted into an otherwise-clean
    # file MUST be caught, and the offending FILE named -- exactly what
    # would have caught any of the five sites pre-migration. Written to a
    # scratch dir, not a real source file.
    mktempdir() do dir
        clean = joinpath(dir, "clean.jl")
        write(clean, "function f(dist)\n    check(dist.use_pencil_arrays)\nend\n")
        @test !file_rederives_convention(clean)

        dirty = joinpath(dir, "dirty.jl")
        dirty_src = "function f(dist)\n" *
                    "    if dist.use_pencil_arrays\n" *
                    "        P1 = mesh[1]\n" *
                    "        coord1 = dist.rank % P1\n" *
                    "    end\n" *
                    "end\n"
        write(dirty, dirty_src)
        @test file_rederives_convention(dirty)
    end

    # Direction 2: KNOWN_OFFENDERS must equal `found` EXACTLY, not merely
    # contain it -- a stale/bogus entry for a file that does NOT re-derive
    # the convention must break the ratchet's set-equality assertion, so a
    # fixed file can't be left in the list. Checked against a THROWAWAY
    # dict here, not the real (empty) KNOWN_OFFENDERS above, so this can't
    # itself fail CI.
    srcdir = joinpath(@__DIR__, "..", "src")
    allowed = joinpath("core", "distributor", "distributor_core.jl")
    real_found = Set{String}()
    for (root, _, files) in walkdir(srcdir), file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        occursin(allowed, path) && continue
        file_rederives_convention(path) && push!(real_found, relpath(path, srcdir))
    end
    bogus_known = Set(["tools/netcdf_output.jl"])  # migrated; does not re-derive
    @test real_found != bogus_known
end

@testset "_uses_transpose_storage" begin
    # Real GPU+MPI field construction (field_types.jl:140) asks this predicate
    # to choose between TransposableFieldStorage (explicit-transpose GPU+MPI
    # transforms) and SerialFieldStorage (PencilFFTs/local transforms). Nothing
    # tested it directly before this testset. If it ever inverts — a refactor
    # that flips the `nprocs > 1` branch, or a new architecture whose dispatch
    # falls through to the wrong method — every GPU+MPI field silently reverts
    # to SerialFieldStorage, and gpu_forward_transform! runs a LOCAL FFT on
    # just this rank's slab of a distributed field: wrong numbers, no error.
    #
    # The zero-arg Tarang.GPU() cannot be constructed with no CUDA loaded
    # (_gpu_device errors with installation guidance), but the PARAMETRIC form
    # bypasses that constructor entirely and builds fine — GPU{Int}(0) is a
    # real, dispatchable GPU-architecture value, just one that cannot allocate
    # GPU arrays. That is all dispatch needs: _uses_transpose_storage matches
    # on the architecture's TYPE, not on whether it can allocate.
    gpu = Tarang.GPU{Int}(0)

    @test Tarang._uses_transpose_storage(gpu, 2) == true
    # n == 1: a serial GPU field must NOT get transpose storage — there is no
    # second rank to transpose with, and TransposableFieldStorage would need a
    # workspace (Comm_split etc.) for a decomposition that does not exist.
    @test Tarang._uses_transpose_storage(gpu, 1) == false

    @test Tarang._uses_transpose_storage(CPU(), 4) == false
    @test Tarang._uses_transpose_storage(CPU(), 1) == false
end
