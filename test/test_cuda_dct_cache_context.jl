using Test
using MPI

const TRANSFORMS_SOURCE = joinpath(@__DIR__, "..", "ext", "cuda", "transforms.jl")
const DCT_SOURCE = joinpath(@__DIR__, "..", "ext", "cuda", "dct_distributed.jl")
const PENCIL_SOURCE = joinpath(@__DIR__, "..", "ext", "cuda", "pencil.jl")
const GPU_DISTRIBUTED_SOURCE = joinpath(@__DIR__, "..", "src", "core", "gpu_distributed.jl")

function _definition_name(expr)
    expr isa Expr || return nothing
    signature = if expr.head == :function
        expr.args[1]
    elseif expr.head == :(=)
        expr.args[1]
    else
        return nothing
    end
    signature isa Expr && signature.head == :where && (signature = signature.args[1])
    signature isa Expr && signature.head == :call || return nothing
    callee = signature.args[1]
    callee isa Symbol && return callee
    if callee isa Expr && callee.head == :. && callee.args[end] isa QuoteNode
        return callee.args[end].value
    end
    return nothing
end

function _find_definition(expr, name::Symbol)
    _definition_name(expr) == name && return expr
    expr isa Expr || return nothing
    for arg in expr.args
        found = _find_definition(arg, name)
        found === nothing || return found
    end
    return nothing
end

function _calls(expr, name::Symbol)
    expr isa Expr || return false
    expr.head == :call && expr.args[1] == name && return true
    return any(arg -> _calls(arg, name), expr.args)
end

@testset "distributed CUDA DCT cache context" begin
    transforms_ast = Meta.parseall(read(TRANSFORMS_SOURCE, String))

    # Evaluate the actual pure production helpers in a tiny harness. This avoids
    # loading CUDA/NCCL and therefore works on CPU-only CI workers.
    harness = Module(:DistributedDCTCacheContextHarness)
    Core.eval(harness, :(using MPI))
    for name in (:_compute_proc_grid, :_distributed_dct_proc_grid,
                 :_distributed_dct_plan_cache_key,
                 :_distributed_dct_comm_token)
        definition = _find_definition(transforms_ast, name)
        @test definition !== nothing
        Core.eval(harness, definition)
    end

    proc_grid = getfield(harness, :_distributed_dct_proc_grid)
    @test proc_grid((; size=8, mesh=(4, 2))) == (4, 2)
    @test proc_grid((; size=8, mesh=nothing)) == (2, 4)

    key = getfield(harness, :_distributed_dct_plan_cache_key)
    shape = (32, 48, 64)
    axes = (:chebyshev, :chebyshev, :chebyshev)
    baseline = key(shape, (2, 4), Float64, axes, :comm_a, 0)

    @test key(shape, (4, 2), Float64, axes, :comm_a, 0) != baseline
    @test key(shape, (2, 4), Float64, axes, :comm_b, 0) != baseline
    @test key(shape, (2, 4), Float64, axes, :comm_a, 1) != baseline
    @test key(shape, (2, 4), Float32, axes, :comm_a, 0) != baseline

    comm_token = getfield(harness, :_distributed_dct_comm_token)
    @test comm_token(MPI.COMM_SELF) != comm_token(MPI.COMM_WORLD)

    # Pin the call-site wiring as well as helper behavior: all three execution
    # contexts must participate when the production cache key is constructed.
    get_plan = _find_definition(transforms_ast, :get_or_create_distributed_dct_plan)
    @test get_plan !== nothing
    @test _calls(get_plan, :_distributed_dct_proc_grid)
    @test _calls(get_plan, :_distributed_dct_comm_token)
    @test _calls(get_plan, :_dct_cache_device_id)
    @test _calls(get_plan, :_distributed_dct_plan_cache_key)

    clear_cache = _find_definition(transforms_ast, :clear_distributed_dct_plan_cache!)
    @test clear_cache !== nothing
    @test _calls(clear_cache, :finalize_distributed_dct_plan!)
    @test _calls(clear_cache, :empty!)

    # Explicit cache teardown must release the MPI row/column communicators held
    # by each pencil; relying on a later GC finalizer makes communicator lifetime
    # nondeterministic and can retain obsolete contexts after a cache clear.
    dct_ast = Meta.parseall(read(DCT_SOURCE, String))
    finalize_plan = _find_definition(dct_ast, :finalize_distributed_dct_plan!)
    @test finalize_plan !== nothing
    @test _calls(finalize_plan, :free_pencil_decomposition!)

    pencil_ast = Meta.parseall(read(PENCIL_SOURCE, String))
    constructor = _find_definition(pencil_ast, :PencilDecomposition)
    @test constructor !== nothing
    @test !_calls(constructor, :finalizer)
    @test _calls(constructor, :_split_pencil_subcommunicators)

    close_pencil = _find_definition(pencil_ast, :close)
    @test close_pencil !== nothing
    @test _calls(close_pencil, :free_pencil_decomposition!)

    # TransposableField has no GC finalizer either, so the one in-tree owner of
    # a TransposableField workspace must close the wrapper it replaces and
    # expose a close of its own; otherwise every field switch leaks the
    # workspace's row/column communicators.
    gpu_distributed_ast = Meta.parseall(read(GPU_DISTRIBUTED_SOURCE, String))
    setup_workspace = _find_definition(gpu_distributed_ast, :setup_transposable_workspace!)
    @test setup_workspace !== nothing
    @test _calls(setup_workspace, :close)
    @test _calls(setup_workspace, :TransposableField)
    close_transform = _find_definition(gpu_distributed_ast, :close)
    @test close_transform !== nothing
    @test _calls(close_transform, :close)

    # The split helper is dependency-injected so a failure after the first
    # communicator can be tested without MPI ranks or a GPU.
    split_definition = _find_definition(pencil_ast, :_split_pencil_subcommunicators)
    @test split_definition !== nothing
    if split_definition !== nothing
        Core.eval(harness, split_definition)
        split_calls = Ref(0)
        freed = Any[]
        splitter = (_comm, _color, _key) -> begin
            split_calls[] += 1
            split_calls[] == 1 ? :row_comm : error("injected second split failure")
        end
        @test_throws ErrorException getfield(
            harness, :_split_pencil_subcommunicators)(
                :world, 0, 0; splitter=splitter, freer=comm -> push!(freed, comm))
        @test freed == [:row_comm]
    end
end
