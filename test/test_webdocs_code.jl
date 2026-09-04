using Test
using TOML
import Tarang
import NetCDF

# Default mode validates every Julia, Bash, TOML, and Dockerfile fence under docs/src.
# Optional runtime modes execute self-contained examples in fresh temp dirs:
#   TARANG_RUN_WEBDOCS_EXAMPLES=true      (CPU examples)
#   TARANG_RUN_WEBDOCS_MPI_EXAMPLES=true  (MPI examples; defaults to two ranks)
# Use TARANG_WEBDOCS_FILTER, TARANG_WEBDOCS_TIMEOUT, TARANG_WEBDOCS_JOBS,
# TARANG_WEBDOCS_RANKS, and TARANG_WEBDOCS_REPORT to tune focused/manual runs.

struct WebdocsCodeBlock
    path::String
    line::Int
    language::String
    code::String
end

const WEBDOCS_ROOT = normpath(joinpath(@__DIR__, "..", "docs", "src"))
const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))

function webdocs_markdown_files()
    files = String[]
    for (root, _, names) in walkdir(WEBDOCS_ROOT)
        for name in names
            endswith(name, ".md") || continue
            push!(files, joinpath(root, name))
        end
    end
    return sort!(files)
end

function webdocs_code_blocks(path::AbstractString)
    blocks = WebdocsCodeBlock[]
    lines = readlines(path)
    opening_line = 0
    opening_indent = ""
    fence_marker = ""
    language = ""
    body = String[]

    for (line_number, line) in enumerate(lines)
        if opening_line == 0
            match_result = match(r"^([ \t]*)(`{3,}|~{3,})[ \t]*([^ \t`]*)", line)
            match_result === nothing && continue
            opening_line = line_number
            opening_indent = match_result.captures[1]
            fence_marker = match_result.captures[2]
            language = match_result.captures[3]
            empty!(body)
        else
            closing_match = match(r"^[ \t]*(`{3,}|~{3,})[ \t]*$", line)
            closes_fence = closing_match !== nothing &&
                first(closing_match.captures[1]) == first(fence_marker) &&
                length(closing_match.captures[1]) >= length(fence_marker)
            if !closes_fence
                deindented = isempty(opening_indent) || !startswith(line, opening_indent) ?
                    line : chop(line; head=length(opening_indent), tail=0)
                push!(body, deindented)
                continue
            end

            relative_path = relpath(path, REPOSITORY_ROOT)
            push!(blocks, WebdocsCodeBlock(
                relative_path,
                opening_line + 1,
                language,
                join(body, '\n'),
            ))
            opening_line = 0
            opening_indent = ""
            fence_marker = ""
            language = ""
            empty!(body)
        end
    end

    opening_line == 0 || error("unterminated code fence in $(relpath(path, REPOSITORY_ROOT)):$opening_line")
    return blocks
end

function self_contained_tarang_example(block::WebdocsCodeBlock)
    block.language == "julia" || return false
    return occursin(r"(?m)^\s*(?:using|import)\s+[^\n#]*\bTarang(?:\b|\.)", block.code)
end

function contains_parse_error(node)
    node isa Expr || return false
    node.head in (:error, :incomplete) && return true
    return any(contains_parse_error, node.args)
end

function bash_syntax_error(block::WebdocsCodeBlock)
    bash = Sys.which("bash")
    bash === nothing && error("bash is unavailable")
    output = IOBuffer()
    process = run(
        pipeline(ignorestatus(`$bash -n -c $(block.code)`), stdout=output, stderr=output),
    )
    success(process) && return nothing
    return strip(String(take!(output)))
end

function runtime_exclusion_reason(block::WebdocsCodeBlock)
    code = block.code
    executable_code = replace(code, r"(?m)#.*$" => "")

    annotation = match(r"(?m)^\s*#\s*webdocs-audit:\s*skip\s*-\s*(.+)$", code)
    annotation === nothing || return strip(annotation.captures[1])

    occursin(r"\bPkg\.(?:add|develop|instantiate|rm|update|resolve)\b", code) &&
        return "changes the active package environment"
    occursin(
        r"\b(?:CUDA|CuArray|CuDense|CuIterative\w*|NCCL)\b|\bGPU\s*\(",
        executable_code,
    ) &&
        return "requires a CUDA-capable environment"
    occursin(r"(?m)^\s*(?:using|import)\s+[^\n#]*\bMPI\b|\bMPI\.", code) &&
        return "requires an MPI process context"
    occursin(r"(?m)^\s*(?:using|import)\s+[^\n#]*\b(?:Plots|CairoMakie|GLMakie|NCDatasets)\b", code) &&
        return "requires an optional plotting or data package"

    return nothing
end

function julia_audit_category(block::WebdocsCodeBlock)
    self_contained_tarang_example(block) || return "syntax-only fragment or continuation"
    reason = runtime_exclusion_reason(block)
    return reason === nothing ? "self-contained CPU runtime candidate" : reason
end

const MAX_WEBDOCS_OUTPUT_BYTES = 1024 * 1024

monotonic_seconds() = time_ns() / 1.0e9

function signal_process_tree(process_group::Int32, signal::Integer)
    if Sys.iswindows()
        taskkill = Sys.which("taskkill")
        taskkill === nothing && return
        run(ignorestatus(`$taskkill /PID $process_group /T /F`))
        return
    end

    result = ccall(:kill, Cint, (Cint, Cint), -process_group, signal)
    result == 0 && return
    Libc.errno() == Libc.ESRCH && return
    Base.systemerror("kill process group", Libc.errno())
end

function captured_webdocs_output(output::IO, output_path::AbstractString)
    flush(output)
    output_size = filesize(output_path)
    omitted_bytes = max(0, output_size - MAX_WEBDOCS_OUTPUT_BYTES)
    seek(output, omitted_bytes)
    captured = String(read(output))
    omitted_bytes == 0 && return captured
    return "[... $omitted_bytes earlier output bytes omitted ...]\n$captured"
end

function run_bounded_command(
    command::Cmd;
    run_directory::AbstractString,
    timeout_seconds::Real,
)
    started_at = monotonic_seconds()
    output_path = joinpath(run_directory, ".webdocs-audit-output.log")

    return open(output_path, "w+") do output
        detached_command = Cmd(
            command;
            dir=run_directory,
            detach=true,
            ignorestatus=true,
        )
        process = run(pipeline(detached_command, stdout=output, stderr=output); wait=false)
        process_group = Libc.getpid(process)
        deadline = monotonic_seconds() + timeout_seconds

        while process_running(process) && monotonic_seconds() < deadline
            sleep(0.05)
        end

        timed_out = process_running(process)
        if timed_out
            signal_process_tree(process_group, Base.SIGTERM)
            grace_deadline = monotonic_seconds() + 1
            while process_running(process) && monotonic_seconds() < grace_deadline
                sleep(0.05)
            end
            # The launcher may exit before its descendants, so always signal the
            # entire detached group after the grace period.
            signal_process_tree(process_group, Base.SIGKILL)
        end
        wait(process)

        return (
            passed=!timed_out && success(process),
            timed_out=timed_out,
            elapsed_seconds=monotonic_seconds() - started_at,
            output=captured_webdocs_output(output, output_path),
        )
    end
end

function run_webdocs_example(
    block::WebdocsCodeBlock;
    timeout_seconds::Real=180,
    launcher=nothing,
    ranks::Int=1,
)
    return mktempdir() do run_directory
        project_argument = "--project=$(REPOSITORY_ROOT)"
        command = `$(Base.julia_cmd()) --startup-file=no --threads=1 $project_argument -e $(block.code)`
        launcher === nothing || (command = `$launcher -n $ranks $command`)
        command = addenv(command, "OMP_NUM_THREADS" => "1")
        return run_bounded_command(
            command;
            run_directory=run_directory,
            timeout_seconds=timeout_seconds,
        )
    end
end

@testset "webdocs fence extraction" begin
    mktemp() do path, io
        write(io, join((
            "```julia",
            "outer()",
            "```",
            "!!! note \"Nested fence\"",
            "    ```julia",
            "    nested()",
            "    ```",
        ), '\n'))
        close(io)

        blocks = webdocs_code_blocks(path)
        @test length(blocks) == 2
        @test blocks[2].language == "julia"
        @test blocks[2].line == 6
        @test blocks[2].code == "nested()"
    end
end

@testset "webdocs timeout kills descendant processes" begin
    if Sys.iswindows()
        @test_skip false  # The Unix process-group regression uses /bin/sh.
    else
        mktempdir() do directory
            ready_path = joinpath(directory, "ready")
            release_path = joinpath(directory, "release")
            marker_path = joinpath(directory, "orphan-wrote-after-timeout")
            script = raw"""
                (
                    while [ ! -e "$RELEASE_PATH" ]; do sleep 0.05; done
                    printf orphaned > "$MARKER_PATH"
                ) &
                printf ready > "$READY_PATH"
                sleep 60
            """
            command = addenv(
                `/bin/sh -c $script`,
                "READY_PATH" => ready_path,
                "RELEASE_PATH" => release_path,
                "MARKER_PATH" => marker_path,
            )

            result = run_bounded_command(
                command;
                run_directory=directory,
                timeout_seconds=1,
            )
            @test result.timed_out
            @test isfile(ready_path)

            write(release_path, "release")
            sleep(0.5)
            @test !isfile(marker_path)
        end
    end
end

if get(ENV, "TARANG_RUN_WEBDOCS_MPI_EXAMPLES", "false") == "true"
    @eval using MPI

    @testset "self-contained webdocs MPI examples run" begin
        examples = WebdocsCodeBlock[]
        mpi_reason = "requires an MPI process context"

        for path in webdocs_markdown_files(), block in webdocs_code_blocks(path)
            self_contained_tarang_example(block) || continue
            runtime_exclusion_reason(block) == mpi_reason || continue
            push!(examples, block)
        end

        source_filter = get(ENV, "TARANG_WEBDOCS_FILTER", "")
        if !isempty(source_filter)
            pattern = Regex(source_filter)
            filter!(block -> occursin(pattern, "$(block.path):$(block.line)"), examples)
        end

        @info "Webdocs MPI runtime audit" runnable=length(examples)
        @test !isempty(examples)

        timeout_seconds = parse(Float64, get(ENV, "TARANG_WEBDOCS_TIMEOUT", "120"))
        ranks = parse(Int, get(ENV, "TARANG_WEBDOCS_RANKS", "2"))
        launcher = MPI.mpiexec()
        failures = String[]

        for (index, block) in enumerate(examples)
            @info "Running webdocs MPI example" index total=length(examples) source="$(block.path):$(block.line)" ranks
            result = run_webdocs_example(
                block;
                timeout_seconds=timeout_seconds,
                launcher=launcher,
                ranks=ranks,
            )
            @info "Finished webdocs MPI example" index passed=result.passed elapsed_seconds=round(result.elapsed_seconds; digits=2)
            result.passed && continue
            status = result.timed_out ? "timed out after $(timeout_seconds) seconds" : "failed"
            push!(failures, "$(block.path):$(block.line) $status\n$(result.output)")
        end

        report_path = get(ENV, "TARANG_WEBDOCS_REPORT", "")
        if !isempty(report_path)
            open(report_path, "w") do report
                for failure in failures
                    println(report, failure)
                    println(report, "\n", "="^80, "\n")
                end
            end
        end

        concise_failures = map(failures) do failure
            lines = split(failure, '\n')
            error_line = findfirst(line -> startswith(line, "ERROR:") || occursin("LoadError:", line), lines)
            error_line === nothing ? first(lines) : "$(first(lines)) — $(lines[error_line])"
        end
        isempty(failures) || @error "Webdocs MPI runtime failures" failures=concise_failures full_report=report_path
        @test isempty(concise_failures)
    end
end

@testset "webdocs Julia fences parse" begin
    markdown_files = webdocs_markdown_files()
    @test !isempty(markdown_files)

    julia_blocks = WebdocsCodeBlock[]
    audit_categories = Dict{String, Int}()
    parse_failures = String[]
    for path in markdown_files
        for block in webdocs_code_blocks(path)
            block.language == "julia" || continue
            push!(julia_blocks, block)
            category = julia_audit_category(block)
            audit_categories[category] = get(audit_categories, category, 0) + 1
            try
                parsed = Meta.parseall(block.code; filename=block.path)
                if contains_parse_error(parsed)
                    push!(parse_failures, "$(block.path):$(block.line): parser returned an error expression")
                end
            catch error
                message = sprint(showerror, error)
                push!(parse_failures, "$(block.path):$(block.line): $message")
            end
        end
    end

    @test !isempty(julia_blocks)
    @test sum(values(audit_categories)) == length(julia_blocks)
    @info "Webdocs Julia audit inventory" total=length(julia_blocks) categories=audit_categories
    isempty(parse_failures) || @error "Webdocs syntax failures" failures=parse_failures
    @test isempty(parse_failures)
end

@testset "webdocs non-Julia executable fences parse" begin
    bash_blocks = WebdocsCodeBlock[]
    toml_blocks = WebdocsCodeBlock[]
    dockerfile_blocks = WebdocsCodeBlock[]
    parse_failures = String[]
    bare_mpi_launchers = String[]
    bash_available = Sys.which("bash") !== nothing

    for path in webdocs_markdown_files(), block in webdocs_code_blocks(path)
        if block.language == "bash"
            push!(bash_blocks, block)
            executable_code = replace(block.code, r"(?m)^\s*#.*$" => "")
            occursin(r"\bmpiexec(?:\s|$)", executable_code) && push!(
                bare_mpi_launchers,
                "$(block.path):$(block.line)",
            )
            if bash_available
                message = bash_syntax_error(block)
                message === nothing || push!(
                    parse_failures,
                    "$(block.path):$(block.line): Bash syntax error: $message",
                )
            end
        elseif block.language == "toml"
            push!(toml_blocks, block)
            try
                TOML.parse(block.code)
            catch error
                message = sprint(showerror, error)
                push!(parse_failures, "$(block.path):$(block.line): TOML syntax error: $message")
            end
        elseif block.language == "dockerfile"
            push!(dockerfile_blocks, block)
            lines = filter(line -> !isempty(line) && !startswith(line, "#"), strip.(split(block.code, '\n')))
            any(line -> occursin(r"(?i)^FROM\s+\S+", line), lines) || push!(
                parse_failures,
                "$(block.path):$(block.line): Dockerfile has no FROM instruction",
            )
            !isempty(lines) && endswith(last(lines), '\\') && push!(
                parse_failures,
                "$(block.path):$(block.line): Dockerfile ends with an unfinished continuation",
            )
        end
    end

    @test !isempty(bash_blocks)
    if !bash_available
        @test_skip false  # Bash is not guaranteed to be installed on Windows.
    end
    @test !isempty(toml_blocks)
    @test !isempty(dockerfile_blocks)
    isempty(bare_mpi_launchers) || @error "Use MPI.jl's project-aware mpiexecjl in webdocs commands" sources=bare_mpi_launchers
    @test isempty(bare_mpi_launchers)
    isempty(parse_failures) || @error "Webdocs non-Julia syntax failures" failures=parse_failures
    @test isempty(parse_failures)
end

@testset "CUDA-only webdocs public API exists" begin
    documented_gpu_names = (
        :CUDA_AVAILABLE,
        :CuIterativeCG,
        :CuIterativeGMRES,
        :GPU,
        :PeriodicDomain,
        :ScalarField,
        :SmagorinskyModel,
        :compute_eddy_viscosity!,
        :forward_transform!,
        :set_gpu_fft_mode!,
        :solve,
    )
    for name in documented_gpu_names
        @test isdefined(Tarang, name)
    end
    for name in (
        :CuIterativeCG,
        :CuIterativeGMRES,
        :GPU,
        :PeriodicDomain,
        :ScalarField,
        :SmagorinskyModel,
        :compute_eddy_viscosity!,
        :forward_transform!,
        :set_gpu_fft_mode!,
    )
        @test name in names(Tarang)
    end
end

@testset "contextual NetCDF webdocs API exists" begin
    @test isdefined(Tarang, :group_variable_names)
    @test isdefined(Tarang, :group_ncread)
    @test :group_ncread in names(Tarang)
    @test isdefined(NetCDF, :ncgetatt)
end

if get(ENV, "TARANG_RUN_WEBDOCS_EXAMPLES", "false") == "true"
    @testset "self-contained webdocs CPU examples run" begin
        examples = WebdocsCodeBlock[]
        exclusions = Dict{String, Int}()

        for path in webdocs_markdown_files(), block in webdocs_code_blocks(path)
            self_contained_tarang_example(block) || continue
            reason = runtime_exclusion_reason(block)
            if reason === nothing
                push!(examples, block)
            else
                exclusions[reason] = get(exclusions, reason, 0) + 1
            end
        end

        source_filter = get(ENV, "TARANG_WEBDOCS_FILTER", "")
        if !isempty(source_filter)
            pattern = Regex(source_filter)
            filter!(block -> occursin(pattern, "$(block.path):$(block.line)"), examples)
        end

        @info "Webdocs runtime audit" runnable=length(examples) excluded=exclusions
        @test !isempty(examples)

        timeout_seconds = parse(Float64, get(ENV, "TARANG_WEBDOCS_TIMEOUT", "180"))
        requested_jobs = parse(Int, get(ENV, "TARANG_WEBDOCS_JOBS", "4"))
        jobs = max(1, min(requested_jobs, length(examples)))
        work = Channel{Int}(length(examples))
        foreach(index -> put!(work, index), eachindex(examples))
        close(work)
        results = Vector{Any}(undef, length(examples))

        @sync for _ in 1:jobs
            @async for index in work
                block = examples[index]
                @info "Running webdocs example" index total=length(examples) source="$(block.path):$(block.line)"
                results[index] = run_webdocs_example(block; timeout_seconds=timeout_seconds)
                @info "Finished webdocs example" index passed=results[index].passed elapsed_seconds=round(results[index].elapsed_seconds; digits=2)
            end
        end

        failures = String[]
        for (block, result) in zip(examples, results)
            result.passed && continue
            status = result.timed_out ? "timed out after $(timeout_seconds) seconds" : "failed"
            push!(failures, "$(block.path):$(block.line) $status\n$(result.output)")
        end

        report_path = get(ENV, "TARANG_WEBDOCS_REPORT", "")
        if !isempty(report_path)
            open(report_path, "w") do report
                for failure in failures
                    println(report, failure)
                    println(report, "\n", "="^80, "\n")
                end
            end
        end

        concise_failures = map(failures) do failure
            lines = split(failure, '\n')
            error_line = findfirst(line -> startswith(line, "ERROR:") || occursin("LoadError:", line), lines)
            error_line === nothing ? first(lines) : "$(first(lines)) — $(lines[error_line])"
        end
        isempty(failures) || @error "Webdocs runtime failures" failures=concise_failures full_report=report_path
        @test isempty(concise_failures)
    end
end
