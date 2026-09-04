"""
Static ratchet on the GPU test files, which no GPU-less CI run executes.

`GPU_TEST_FILES` reaches a real GPU only when the Buildkite pipeline has a
CUDA-tagged single-GPU agent connected. `DISTRIBUTED_GPU_TEST_FILES` need the
disabled multi-GPU NCCL job and are intentionally not active in the current
single-GPU pipeline. On GitHub Actions, and on any machine without the hardware,
those distributed files are dead text as far as execution is concerned: they
can stop parsing, or call a function that was deleted months ago, and nothing
notices until somebody runs them on a suitable GPU node.

This test cannot run those files (no NVIDIA GPU on CI, and several of them call
`MPI.Init`). It performs the checks that need no hardware:

1. **They parse.** A syntax error in a GPU test file is caught here, not on the
   GPU node.
2. **The extension API they call still exists.** Every identifier that looks like
   `TarangCUDAExt` API (the `gpu_*` / `plan_*` / `distributed_*` / ... prefixes
   and the `GPU*` / `Pencil*` / `NCCL*` CamelCase names) is checked against the
   loaded extension. This is the check that catches a rename or a deletion:
   the GPU tests reference those names UNQUALIFIED inside `if CUDA.functional()`
   blocks, so on a CPU-only machine the reference is never resolved and a stale
   call is invisible.
3. **The manual GPU reporter is safe.** Its shell regression suite runs in a
   temporary Git repository, including source-mutation and status-error cases.
4. **Capability-probe errors propagate.** An invalid MPI CUDA override is tested
   in a clean subprocess so MPI state cached by other test files cannot mask it.
5. **CUDA installation guidance preserves the checkout.** Documentation and
   reporter errors must install CUDA in Julia's stacked default environment.
"""

using Test
using Tarang

include("file_lists.jl")

const _CUDA_LOADED_R = try
    @eval using CUDA
    true
catch err
    @info "CUDA.jl unavailable; skipping GPU test-file API reachability check" err
    false
end

const _TESTDIR = @__DIR__

@testset "GPU CI report shell regressions" begin
    script = joinpath(_TESTDIR, "..", "scripts", "test_gpu_ci_report.sh")
    @test isfile(script)

    output = IOBuffer()
    process = run(
        pipeline(ignorestatus(`bash $script`), stdout=output, stderr=output),
    )
    shell_output = String(take!(output))
    success(process) || @error "gpu_ci_report.sh regressions failed" output=shell_output
    @test success(process)
end

@testset "invalid MPI CUDA override propagates" begin
    repository_root = normpath(joinpath(_TESTDIR, ".."))
    code = raw"""
        using Tarang

        caught = withenv(
            "TARANG_CUDA_AWARE_MPI" => nothing,
            "JULIA_MPI_HAS_CUDA" => "not-a-bool",
            "OMPI_MCA_opal_cuda_support" => nothing,
            "MV2_USE_CUDA" => nothing,
            "MPIR_CVAR_ENABLE_GPU" => nothing,
            "MPICH_GPU_SUPPORT_ENABLED" => nothing,
        ) do
            try
                check_cuda_aware_mpi()
                nothing
            catch err
                err
            end
        end

        if !(caught isa ArgumentError)
            println(stderr, "expected ArgumentError, got ", repr(caught))
            exit(1)
        end
    """
    output = IOBuffer()
    command = `$(Base.julia_cmd()) --startup-file=no --project=$repository_root -e $code`
    process = run(
        pipeline(ignorestatus(command), stdout=output, stderr=output),
    )
    subprocess_output = String(take!(output))
    success(process) || @error "invalid JULIA_MPI_HAS_CUDA was swallowed" output=subprocess_output
    @test success(process)
end

@testset "CUDA installation guidance preserves the checkout" begin
    repository_root = normpath(joinpath(_TESTDIR, ".."))
    guidance_paths = (
        joinpath(repository_root, "README.md"),
        joinpath(repository_root, "docs", "src", "index.md"),
        joinpath(repository_root, "docs", "src", "pages", "testing.md"),
        joinpath(repository_root, "scripts", "gpu_ci_report.sh"),
    )

    unsafe_lines = Tuple{String, Int, String}[]
    for path in guidance_paths
        @test isfile(path)
        for (line_number, line) in enumerate(eachline(path))
            normalized_line = replace(line, '\\' => "")
            mentions_add = occursin("Pkg.add(\"CUDA\")", normalized_line) ||
                           occursin("Pkg.add([\"CUDA\"", normalized_line)
            mentions_add || continue
            occursin("--project=@v#.#", line) && continue
            push!(unsafe_lines, (relpath(path, repository_root), line_number, strip(line)))
        end
    end

    isempty(unsafe_lines) || @error "CUDA guidance can dirty the Tarang checkout" unsafe_lines
    @test isempty(unsafe_lines)
end

# Names that look like extension API rather than a local helper or a Base call.
const _EXT_PREFIXES = ("gpu_", "plan_gpu", "plan_optimized", "plan_batched",
                       "distributed_", "local_dct", "local_fft", "nccl_",
                       "reorder_for_dct", "inverse_reorder_for_dct",
                       "get_or_create_", "clear_gpu_", "clear_distributed_",
                       "clear_batched_", "finalize_nccl_", "finalize_distributed_",
                       "compute_pencil_", "compute_transpose_", "rank_to_grid",
                       "grid_to_rank", "current_orientation", "set_orientation!",
                       "current_local_shape", "is_distributed_gpu",
                       "needs_distributed_dct", "allocate_gpu_data",
                       "create_dealiasing_mask_gpu", "apply_dealiasing_gpu!",
                       "enable_tensor_cores!", "disable_tensor_cores!")
const _EXT_TYPE_PREFIXES = ("GPU", "Pencil", "NCCL", "DistributedDCT",
                            "BatchedGPU", "OptimizedGPU")

_looks_like_ext_api(name::Symbol) = begin
    s = String(name)
    any(p -> startswith(s, p), _EXT_PREFIXES) ||
        (any(p -> startswith(s, p), _EXT_TYPE_PREFIXES) && isuppercase(s[1]))
end

"""Collect `Expr(:error, ...)` / `Expr(:incomplete, ...)` nodes that `Meta.parseall`
embeds instead of throwing."""
function _collect_errors!(expr, out::Vector{Expr})
    expr isa Expr || return
    (expr.head === :error || expr.head === :incomplete) && push!(out, expr)
    for a in expr.args
        _collect_errors!(a, out)
    end
    return
end

"""Collect symbols appearing in call position, and separately the names the file
defines itself (which must not be flagged)."""
function _scan(expr, called::Set{Symbol}, defined::Set{Symbol})
    expr isa Expr || return
    if expr.head === :call && !isempty(expr.args) && expr.args[1] isa Symbol
        push!(called, expr.args[1])
    elseif expr.head === :. && length(expr.args) == 2 && expr.args[2] isa QuoteNode &&
           expr.args[1] in (:TarangCUDAExt, :ext, :Tarang)
        # ONLY module-qualified access. A plain field access like `plan.gpu_id`
        # also parses as `Expr(:., …)`, and treating that as an API reference
        # would flag every struct field whose name happens to start with `gpu_`.
        expr.args[2].value isa Symbol && push!(called, expr.args[2].value)
    end
    if expr.head === :function || (expr.head === :(=) && expr.args[1] isa Expr &&
                                   expr.args[1].head === :call)
        sig = expr.args[1]
        if sig isa Expr && sig.head === :call && sig.args[1] isa Symbol
            push!(defined, sig.args[1])
        end
    elseif expr.head === :struct
        nm = expr.args[2]
        nm isa Symbol && push!(defined, nm)
        nm isa Expr && nm.head === :curly && nm.args[1] isa Symbol && push!(defined, nm.args[1])
    elseif expr.head === :const && expr.args[1] isa Expr && expr.args[1].head === :(=)
        lhs = expr.args[1].args[1]
        lhs isa Symbol && push!(defined, lhs)
    end
    for a in expr.args
        _scan(a, called, defined)
    end
    return
end

@testset "GPU test files parse and reference live API" begin
    @test "test_distributed_gpu_transpose.jl" in DISTRIBUTED_GPU_TEST_FILES

    gpu_files = unique(vcat(GPU_TEST_FILES, DISTRIBUTED_GPU_TEST_FILES))
    @test !isempty(gpu_files)

    # The current hardware policy is one GPU per Buildkite box. Keep the
    # single-GPU suite active and prevent the two-rank distributed transpose
    # test from being reintroduced accidentally. Ignore comment-only YAML lines
    # so the disabled multi-GPU example does not affect these assertions.
    pipeline_path = joinpath(_TESTDIR, "..", ".buildkite", "pipeline.yml")
    @test isfile(pipeline_path)
    pipeline = read(pipeline_path, String)
    active_pipeline = join(
        filter(line -> !startswith(strip(line), "#"), split(pipeline, '\n')),
        '\n',
    )
    @test occursin("test/run_gpu_ci.jl", active_pipeline)
    @test occursin("single-GPU", active_pipeline)
    @test !occursin("test/test_distributed_gpu_transpose.jl", active_pipeline)

    trigger_path = joinpath(_TESTDIR, "..", ".github", "workflows",
                            "gpu-buildkite.yml")
    @test isfile(trigger_path)
    trigger = read(trigger_path, String)
    @test occursin(r"workflow_dispatch\s*:", trigger)
    @test occursin("vars.BUILDKITE_ORG || 'subhajit-kar'", trigger)

    parsed = Dict{String, Any}()
    @testset "parse" begin
        for f in gpu_files
            path = joinpath(_TESTDIR, f)
            @test isfile(path)
            isfile(path) || continue
            ast = try
                Meta.parseall(read(path, String); filename = f)
            catch err
                @error "GPU test file does not parse" file = f exception = err
                nothing
            end
            @test ast !== nothing
            # `parseall` wraps errors in the AST rather than throwing.
            bad = Expr[]
            ast === nothing || _collect_errors!(ast, bad)
            @test isempty(bad)
            ast === nothing || (parsed[f] = ast)
        end
    end

    if !_CUDA_LOADED_R
        @test_skip "CUDA.jl not loadable; extension API reachability unchecked"
    else
        ext = Base.get_extension(Tarang, :TarangCUDAExt)
        @test ext !== nothing
        if ext !== nothing
            missing_api = Tuple{String, Symbol}[]
            for (f, ast) in parsed
                called, defined = Set{Symbol}(), Set{Symbol}()
                _scan(ast, called, defined)
                for name in called
                    name in defined && continue
                    _looks_like_ext_api(name) || continue
                    (isdefined(ext, name) || isdefined(Tarang, name)) && continue
                    push!(missing_api, (f, name))
                end
            end
            if !isempty(missing_api)
                @error "GPU test files call extension API that no longer exists" missing_api
            end
            @test isempty(missing_api)
        end
    end
end
