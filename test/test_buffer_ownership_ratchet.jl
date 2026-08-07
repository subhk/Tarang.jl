# Ratchet on rotating buffer pools, plus the contract that makes them safe.
#
# A rotating pool hands out slot `idx % N` and reissues it after N further
# checkouts, tracking nothing about who is still holding the previous tenant. Any
# caller that RETAINS such a result therefore holds memory that changes value
# later, with no error — this project's dominant failure shape. It has been a live
# wrong-answer bug twice: `grad()` storing derivative slots into the tensor it
# returned (max err 3.0), and `Base.:*(field, field)` handing a nonlinear slot to
# user code (max err 99). Both were found only by asserting values.
#
# `FieldPool` in the same tree is the opposite design — explicit `checkout!` /
# `return!`, ownership recorded in `_FIELD_POOL_OWNERS`, cross-pool returns throw,
# `with_pool_field` for RAII — and it is deliberately NOT installed, because
# `Future.evaluate` returns every intermediate to the pool after `operate` and the
# RHS keeps several live at once.
#
# So the safe mechanism is dormant and the unsafe one is in the hot path. That is
# a defensible trade (the rotating pools are what keep the RHS allocation-free)
# only while the ownership boundary is respected at every escape site. This file
# ratchets the two facts that make it checkable: the set of rotating pools does
# not grow silently, and no pool is installed behind a caller's back.
#
# Value-level ownership assertions live in `test_deriv_pool_ownership.jl` and
# `test_nl_product_ownership.jl`. The contract is written up in
# `src/core/module_contracts.jl`.

using Test
using Tarang

const OWNERSHIP_SRC = normpath(joinpath(@__DIR__, "..", "src"))

# A rotating pool is identifiable by its size constant: `const _X_POOL_SIZE = N`.
const POOL_SIZE_RE = r"^\s*const\s+(_[A-Z0-9_]*POOL_SIZE)\s*=\s*(\d+)"

"""Every `const _*_POOL_SIZE = N` declaration under `root`, as name => size."""
function _rotating_pool_sizes(root::AbstractString)
    found = Dict{String, Int}()
    for (dir, _, files) in walkdir(root), f in files
        endswith(f, ".jl") || continue
        for line in eachline(joinpath(dir, f))
            m = match(POOL_SIZE_RE, line)
            m === nothing && continue
            found[m.captures[1]] = parse(Int, m.captures[2])
        end
    end
    return found
end

@testset "the set of rotating pools is known" begin
    pools = _rotating_pool_sizes(OWNERSHIP_SRC)

    # Each entry here is a pool whose escape sites have been audited and either
    # take ownership via `_own_borrowed_field` or provably consume the result
    # before returning. Adding a pool without making that decision is the bug;
    # adding one to this list is how you record that you made it.
    expected = Dict(
        # grad/div retain components -> `_own_borrowed_field` at 5 sites.
        "_DERIV_RESULT_POOL_SIZE"   => 16,
        # `evaluate_transform_multiply(...; own=true)` copies out by default.
        "_NL_RESULT_POOL_SIZE"      => 8,
        # Single consumer, `copy_field_data!`s the result on the next line
        # (`_solve_poisson_constraint!` in timesteppers/state_utils.jl).
        "_POISSON_RESULT_POOL_SIZE" => 4,
    )

    @test sort(collect(keys(pools))) == sort(collect(keys(expected)))
    for (name, size) in expected
        @test get(pools, name, -1) == size
    end
end

@testset "no FieldPool is installed by default" begin
    # `checkout_or_alloc` is used at ~17 sites and must allocate at every one of
    # them. If a pool ever gets installed globally, those sites start handing out
    # recycled buffers to callers written against fresh ones.
    @test get_field_pool() === nothing

    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    bases = (RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π)),)

    a = Tarang.checkout_or_alloc(bases, Float64, dist)
    b = Tarang.checkout_or_alloc(bases, Float64, dist)
    @test a !== b
    @test a._from_pool == false
    @test b._from_pool == false

    # This also catches a leak from `test_field_pool.jl`, which installs a pool in
    # three testsets and resets it in each; a missed reset would corrupt every
    # `checkout_or_alloc` caller that runs after it in the same process.
end

@testset "FieldPool still enforces ownership when installed" begin
    # The dormant design is the reference for what "owned" means. Keep it working:
    # if the rotating pools are ever replaced, this is what replaces them.
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype = Float64, architecture = CPU())
    bases = (RealFourier(coords["x"]; size = 8, bounds = (0.0, 2π)),)

    pool = FieldPool(dist)
    f = checkout!(pool, bases, Float64)
    @test f._from_pool == true

    # A field never checked out cannot be returned.
    stranger = ScalarField(dist, "stranger", bases, Float64)
    @test_throws ArgumentError return!(pool, stranger)

    # Nor can it be returned to a different pool.
    other = FieldPool(dist)
    @test_throws ArgumentError return!(other, f)

    # Only after an explicit return! is the buffer reissued — the property the
    # rotating pools lack.
    return!(pool, f)
    @test checkout!(pool, bases, Float64) === f
end
