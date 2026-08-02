# Ratchet on the LEGACY export surface — names exported from `Tarang` that are not
# declared in the `@public_api` manifest under `src/api/public/`.
#
# `src/public_api.jl` says the manifest is the supported surface and that "core
# implementation files may retain legacy exports during the compatibility window".
# That window has no end and no pressure: 1205 names are exported, 352 are declared,
# and nothing stops the gap widening with every new file that writes `export`.
#
# WHY IT MATTERS HERE SPECIFICALLY. This is not tidiness. An exported name is a
# promise, and this project has already been bitten by promises nobody was keeping:
# `save_field`/`load_field!` were exported, broken and untested at the same time,
# which is only possible because exporting is free and unexamined. `scripts/
# classify_exports.jl` currently finds 49 legacy exports with no reference anywhere
# in src/, test/, docs/ or examples/ — 49 more of exactly that. Everything exported
# is also effectively unrefactorable, because you cannot tell a name someone depends
# on from a name that leaked.
#
# WHAT THIS FILE DOES. Stops the bleeding, nothing more. Un-exporting is a breaking
# change and is deliberately NOT done here; the inventory in
# `docs/legacy_export_inventory.md` is the input for that decision when it is made.
# Follows the ratchet idiom of test_jet.jl and test_catch_ratchet.jl: the count may
# fall freely and must not rise.

using Test
using Tarang

@testset "public API manifest is coherent" begin
    manifest = Set(Tarang.public_api_names())
    exported = setdiff(Set(names(Tarang)), Set([:Tarang]))

    # Every declared name must actually be exported. `@public_api` emits the export
    # itself, so a mismatch means a declaration was removed without its export, or a
    # file under src/api/public/ stopped being included — either way the manifest
    # would be describing a surface that does not exist.
    declared_but_missing = setdiff(manifest, exported)
    @test isempty(declared_but_missing)

    # The manifest must be non-trivial: an empty one would make every assertion
    # below vacuous.
    @test length(manifest) >= 300
end

@testset "legacy export ratchet" begin
    manifest = Set(Tarang.public_api_names())
    exported = setdiff(Set(names(Tarang)), Set([:Tarang]))
    legacy = setdiff(exported, manifest)
    n_legacy = length(legacy)

    @info "Tarang export surface: $(length(exported)) exported, $(length(manifest)) in the " *
          "@public_api manifest, $n_legacy legacy"

    # Current count. Lower it as names are promoted to `@public_api` or un-exported;
    # never raise it. Adding a new supported name means declaring it with
    # `@public_api` in `src/api/public/`, which leaves this number unchanged.
    LEGACY_RATCHET = 853

    if n_legacy > LEGACY_RATCHET
        added = sort(collect(legacy); by = string)
        shown = first(added, 30)
        listing = join(("  " * string(n) for n in shown), "\n")
        length(added) > length(shown) && (listing *= "\n  … and $(length(added) - length(shown)) more")
        @warn "$n_legacy legacy exports, ratchet is $LEGACY_RATCHET. A new supported name " *
              "belongs in src/api/public/ declared with `@public_api`, which does not move " *
              "this count. A bare `export` in an implementation file does. Full inventory " *
              "(not only the new ones) — see docs/legacy_export_inventory.md:\n" * listing
    elseif n_legacy < LEGACY_RATCHET
        @info "Legacy export count dropped to $n_legacy — lower LEGACY_RATCHET in " *
              "$(basename(@__FILE__)) to match."
    end

    @test n_legacy <= LEGACY_RATCHET

    # Sanity: the comparison must be discriminating. If `public_api_names()` ever
    # returned everything, `legacy` would be empty and the ratchet vacuously green;
    # if it returned nothing, `legacy` would be the whole surface and carry no signal.
    @test 0 < n_legacy < length(exported)
end
