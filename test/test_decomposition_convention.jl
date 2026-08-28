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
    # decomposed_axes. The regex below was then broadened (this task) because
    # the original required a mesh-suffixed word immediately after LAST/FIRST
    # (e.g. "ndims_mesh"), which missed both subject-first prose ("mesh
    # decomposes the LAST dims") and plain verb-first prose with no such word
    # ("decompose LAST dimensions"). Broadening it surfaced a SECOND wave: five
    # more sites, found only once the regex could see them, that were never in
    # the original ten-site inventory — including two functions
    # (netcdf_output.jl's get_local_shape/get_local_start) where only the
    # PencilArrays half was migrated and the TransposableField half was left
    # hand-rolled.
    #
    # KNOWN_OFFENDERS below (declared after the scan) is that second wave's
    # baseline population — same idiom as test_catch_ratchet.jl's bare-`catch`
    # ratchet: an explicit, named, already-triaged list, not a magic count.
    # THE TARGET IS AN EMPTY DICT. Every entry is a TODO for a follow-up
    # migration task, not a permanent exemption.
    srcdir = joinpath(@__DIR__, "..", "src")
    allowed = joinpath("core", "distributor", "distributor_core.jl")

    # The tell is a use_pencil_arrays branch that decides axis indices, restated
    # in prose as "decompose(s) [the] LAST/FIRST ..." — in EITHER word order.
    # A single verb-first alternative with no fixed suffix after LAST/FIRST
    # already subsumes the subject-first case as a substring match (regex
    # `occursin` doesn't require the whole sentence, just some contiguous
    # span), but both orders are spelled out explicitly so the check does not
    # depend on that being true forever:
    #   - verb-first:    "decompose(s)/(d) [the] LAST/FIRST ..."
    #   - subject-first: "mesh decompose(s)/(d) the LAST/FIRST ..."
    convention_re = r"decompose[sd]?\s+(the\s+)?(LAST|FIRST)|mesh\s+decompose[sd]?\s+the\s+(LAST|FIRST)"i

    offenders = String[]
    for (root, _, files) in walkdir(srcdir), file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        occursin(allowed, path) && continue
        text = read(path, String)
        if occursin(r"use_pencil_arrays"i, text) && occursin(convention_re, text)
            push!(offenders, relpath(path, srcdir))
        end
    end

    found = Set(offenders)
    @info "convention re-derivation scan: $(length(found)) file(s) match" sort(collect(found))

    # The known, already-triaged population. Each reason names the function(s)
    # and which half of the convention (PencilArrays/LAST vs
    # TransposableField/FIRST) is hand-rolled instead of calling
    # decomposed_axes / mesh_axis_for.
    KNOWN_OFFENDERS = Dict(
        "core/distributor/distributor_mpi.jl" =>
            "_scatter_array_from_root hand-rolls BOTH the PencilArrays " *
            "(LAST dims) and TransposableField (FIRST dims) decompositions, " *
            "each computed twice (once for the local rank, again per " *
            "dest_rank in the scatter loop).",
        "core/field/field_data/field_data_distributor_utils.jl" =>
            "validate_decomposition_convention restates the convention in " *
            "prose inside its error messages; it computes no axis indices " *
            "itself, so only its documentation — not a numerical result — " *
            "can drift.",
        "core/nonlinear/nonlinear_transforms.jl" =>
            "setup_pencil_transforms_for_shape! hand-rolls the PencilArrays " *
            "(LAST dims) decomp_dims formula.",
        "core/transforms/transform_planning.jl" =>
            "setup_pencil_fft_transforms_2d! hand-rolls the PencilArrays " *
            "(LAST dims) trailing-axes formula.",
        "tools/netcdf_output.jl" =>
            "get_local_shape and get_local_start each migrated their " *
            "PencilArrays branch onto mesh_axis_for, but left their " *
            "TransposableField (non-pencil) branch hand-rolling FIRST-dims " *
            "decomposition.",
    )
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
