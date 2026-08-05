# Ratchet on architecture branching, plus the contract for the primitive that
# replaces it.
#
# `CPU` and `GPU{D}` are real types under `AbstractArchitecture`, but most code
# asked `is_gpu(arch)` and branched on the resulting `Bool`. That is the shape
# behind nearly every GPU defect in this project's history: the device branch is
# unreachable text on a CPU-only CI run, so when it disagrees with the host
# branch — a guard that reads state the device path never built, an assembly step
# the device path skips — nothing fails. The run just produces host answers, or
# zeros.
#
# Dispatching on the architecture type instead changes the failure mode. A branch
# that silently fell through to the host path becomes a `MethodError` naming the
# missing method, and each side is separately callable in a test by passing a
# `GPU()` without any device present.
#
# Not every site can or should move. `is_gpu` is legitimate when the architecture
# is one input among several to a decision (CUDA-aware MPI probing, matsolver
# selection, cache keys) rather than a choice of implementation. This ratchet
# therefore counts, and only requires that the number not rise.
#
# `is_gpu_array(x)` is deliberately NOT counted: it interrogates an array's
# storage, which is a separate question from which architecture a component was
# configured for, and the two are exactly what the checked-both-ways guards in
# les_models.jl and transform_gpu.jl exist to reconcile.

using Test
using Tarang

const BACKEND_SRC = normpath(joinpath(@__DIR__, "..", "src"))

# `is_gpu(` but not `is_gpu_array(` / `is_gpu_field(`.
const BACKEND_CALL_RE = r"(?<![_a-zA-Z])is_gpu\("

"""Count `is_gpu(...)` sites under `root`, skipping `architectures.jl` (which
defines the predicate) and comment-only lines so prose does not inflate the
count."""
function _count_backend_branches(root::AbstractString)
    total = 0
    per_file = Dict{String, Int}()
    for (dir, _, files) in walkdir(root), f in files
        endswith(f, ".jl") || continue
        f == "architectures.jl" && continue
        path = joinpath(dir, f)
        n = 0
        for raw in eachline(path)
            code = strip(raw)
            startswith(code, "#") && continue
            occursin(BACKEND_CALL_RE, code) && (n += 1)
        end
        n > 0 && (per_file[relpath(path, root)] = n)
        total += n
    end
    return total, per_file
end

@testset "architecture dispatch primitives" begin
    # `to_architecture` is the host-identity counterpart to `on_architecture`.
    # The distinction is the whole reason the `is_gpu ? on_architecture : x`
    # idiom existed, so pin it. `on_architecture(CPU(), _)` has identity methods
    # for `Array` and for views of one, but its `AbstractArray` fallback
    # materializes a dense copy — so a host-resident wrapper outside those two
    # cases is copied, while `to_architecture(CPU(), _)` hands back the very same
    # object whatever it is.
    p = PermutedDimsArray(reshape(collect(1.0:12.0), 3, 4), (2, 1))
    @test Tarang.to_architecture(Tarang.CPU(), p) === p
    @test Tarang.on_architecture(Tarang.CPU(), p) isa Array
    @test Tarang.on_architecture(Tarang.CPU(), p) !== p

    # The cases `on_architecture` already treats as identity — which is what made
    # the guards removable rather than merely redundant-looking.
    a = collect(1.0:4.0)
    v = view(reshape(collect(1.0:12.0), 3, 4), 1:2, 1:2)
    for x in (a, v)
        @test Tarang.to_architecture(Tarang.CPU(), x) === x
        @test Tarang.on_architecture(Tarang.CPU(), x) === x
    end

    # `synchronize` is a no-op on the host, which is what lets call sites drop
    # the `is_gpu` guard around it entirely.
    @test Tarang.synchronize(Tarang.CPU()) === nothing

    # The transform entry points answer "not a GPU field" by method, not by
    # testing a Bool. Both must therefore accept a CPU architecture and decline.
    domain = PeriodicDomain(16)
    u = ScalarField(domain, "u")
    set!(u, (x,) -> sin(x))
    @test Tarang.gpu_forward_transform!(u) === false
    @test Tarang.gpu_backward_transform!(u) === false

    # And the declining path must be dispatch on the architecture, so the device
    # method is reachable for inspection without a device attached.
    @test hasmethod(Tarang._gpu_forward_transform!, Tuple{Tarang.CPU, ScalarField})
    @test hasmethod(Tarang._gpu_forward_transform!, Tuple{Tarang.GPU, ScalarField})
    @test hasmethod(Tarang._ensure_array_on_architecture, Tuple{Tarang.CPU, AbstractArray})
    @test hasmethod(Tarang._ensure_array_on_architecture, Tuple{Tarang.GPU, AbstractArray})
    @test hasmethod(Tarang._plan_local_axis_fft, Tuple{Tarang.CPU, Type, Tuple, Int})
    @test hasmethod(Tarang._plan_local_axis_fft, Tuple{Tarang.GPU, Type, Tuple, Int})
end

@testset "architecture branch ratchet" begin
    total, per_file = _count_backend_branches(BACKEND_SRC)

    @info "src/ `is_gpu(...)` sites: $total"

    # Current count. Lower it when a branch becomes a method; never raise it.
    BACKEND_RATCHET = 50

    if total > BACKEND_RATCHET
        worst = sort(collect(per_file); by = kv -> -kv[2])
        listing = join(("  $(lpad(n, 4))  $f" for (f, n) in first(worst, 15)), "\n")
        @warn "$total `is_gpu(...)` sites in src/, ratchet is $BACKEND_RATCHET. If the " *
              "branch selects between a host and a device implementation, write one " *
              "method per architecture instead — that is what makes the device side " *
              "fail loudly rather than fall through to the host path on a CPU run. " *
              "Heaviest files:\n" * listing
    elseif total < BACKEND_RATCHET
        @info "Architecture branch count dropped to $total — lower BACKEND_RATCHET in " *
              "$(basename(@__FILE__)) to match."
    end

    @test total <= BACKEND_RATCHET

    # Sanity: a regex or path regression that matched nothing would make this
    # ratchet vacuously green forever.
    @test total >= 20
end
