# Testing

Guide to running and writing tests for Tarang.jl.

## Running Tests

### Full Test Suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Specific Test File

Every file under `test/` is self-contained (it starts with `using Test, Tarang`),
so it can be included on its own:

```bash
julia --project=. -e 'using Test, Tarang; include("test/test_solvers.jl")'
```

Files that emulate a GPU with `JLArrays` (`test_gpu_*_jlarray.jl`,
`test_2d_gpu_domain_compat.jl`, `test_gpu_2d_device_stack.jl`, ...) need the
test dependencies, which resolve only under `Pkg.test()`; run those through the
full suite, or from a scratch environment that `Pkg.develop`s this checkout and
adds `JLArrays`. Do not run with `--project=test`: that creates a
`test/Project.toml` and breaks the suite.

The optional and GPU groups are switched on with environment variables read by
`test/runtests.jl`:

| Variable | Effect |
|---|---|
| `TARANG_RUN_OPTIONAL_TESTS=true` | also run `OPTIONAL_TEST_FILES` |
| `TARANG_ONLY_OPTIONAL_TESTS=true` | run only `OPTIONAL_TEST_FILES` |
| `TARANG_RUN_GPU_TESTS=true` | also run `GPU_TEST_FILES` (they skip without CUDA) |

### With MPI

The multi-rank MPI tests each run in their own MPI world, via a driver
(CI exercises 1, 2, and 4 ranks):

```bash
julia --project=. test/run_mpi_ci.jl 4      # all MPI tests at 4 ranks
./test/run_mpi_tests.sh 4                    # convenience wrapper
```

### GPU

GPU tests need an NVIDIA GPU and run on JuliaGPU Buildkite CI (see
[Continuous Integration](#continuous-integration)). To run them locally on a
CUDA host:

```bash
julia --project=@v#.# -e 'using Pkg; Pkg.add("CUDA")' # keep this checkout clean
julia --project=. test/run_gpu_ci.jl                # single-process GPU tests
julia --project=. test/run_gpu_fc_2d.jl             # strict focused 2D FC validation
# distributed (NCCL) tests across, e.g., 2 GPUs:
TARANG_MPI_FILESET=distributed_gpu julia --project=. test/run_mpi_ci.jl 2
```

`test/run_gpu_fc_2d.jl` is intended for a single NVIDIA device on a cluster. It
requires functional CUDA, disables scalar indexing, prints CUDA device
information, and runs the CPU/GPU value and allocation checks for the complete
2D Fourier--Chebyshev path. It exits nonzero rather than skipping when CUDA is
missing. For ordinary CPU development, running
`test/test_gpu_fc_2d_complete.jl` directly is safe and reports one skipped
testset when no functional device is present.

## Test Structure

There is no hand-maintained list of test files in the runner. `test/file_lists.jl`
is the single registry, and every driver reads it:

| List | Files | Run by |
|---|---|---|
| `TEST_FILES` | ~170 | `Pkg.test()` on every CI job |
| `OPTIONAL_TEST_FILES` | 7 | `Pkg.test()` with `TARANG_RUN_OPTIONAL_TESTS=true` |
| `GPU_TEST_FILES` | 5 | `test/run_gpu_ci.jl` on a CUDA host (Buildkite) |
| `MPI_TEST_FILES` | 58 | `test/run_mpi_ci.jl [nprocs]`, one `mpiexec` world per file |
| `DISTRIBUTED_GPU_TEST_FILES` | 8 | `TARANG_MPI_FILESET=distributed_gpu test/run_mpi_ci.jl 2` (CUDA + NCCL) |

`test_test_inventory.jl` runs first and fails if a `test_*.jl` file on disk is
missing from the registry, a registered file is missing on disk, or a registered
file is not tracked by git (a file that was never `git add`ed passes locally and
fails on every clean clone). Register new files in exactly one list.

Beyond feature tests, several files are *ratchets* that pin a population the
codebase must not grow — `test_layout_discipline_ratchet.jl`,
`test_backend_dispatch_ratchet.jl`, `test_hasfield_ratchet.jl`,
`test_buffer_ownership_ratchet.jl`, `test_decomposition_convention.jl`, and the
JET/Aqua files. Their header comments explain what each count guards.

## Testing GPU paths without a GPU

GPU CI is a self-hosted Buildkite job that rarely runs, so most device coverage
is hardware-free and lives in the default CPU suite. Three methods are in use;
reach for the one that matches what you are testing.

**Device arrays via `JLArrays`.** `JLArray` is GPUArrays' CPU-backed reference
device array. A test file declares `Tarang.is_gpu_array(::JLArray) = true`,
`architecture`, `on_architecture`, `copy_to_device`, and `array_type` for
`GPU{JLBackend}`, and then every field built on
`Distributor(coords; device=GPU(JLBackend()))` is device-resident: the
execution plan says `:gpu`, the field path and diagonal-IMEX broadcasts run on
device arrays, and `GPUArrays.allowscalar(false)` turns any scalar indexing
into an error, exactly as on a `CuArray`. What `JLArray` lacks is a device FFT.
`test_gpu_timesteppers_jlarray.jl` supplies a cuFFT stand-in — the transform
backend copies the device buffer into a CPU twin field of the same domain, runs
the CPU transform, and copies back — which makes device-vs-CPU comparisons of
whole time steps exact. That file runs all twenty timesteppers on the real
single-GPU dispatch path.

`test_gpu_boundary_regressions_jlarray.jl` additionally exercises boundary-value
solves, unit-vector fields, and moving wall constraints with scalar indexing
disabled. FFTs and sparse LU use explicit host stand-ins; field storage,
boundary gather/override, stage arithmetic, and solution scatter use device
arrays. Native CUDA coverage lives in `test_gpu_fc_2d_complete.jl` and requires
CUDA hardware.

**KernelAbstractions kernels on the CPU backend.** The kernels in
`ext/cuda/` and `src/core/subsystems/mode_batch_kernels.jl` are
`KernelAbstractions.@kernel`s, so the same kernel object launches on
`KernelAbstractions.CPU()` over plain arrays
(`test_gpu_dct1_kernels_cpu.jl`, `test_gpu_transpose_kernels_cpu.jl`,
`test_gpu_kernels_cpu.jl`). Index math and normalization are verified against
the CPU chain; only the CUDA launch itself is not.

**GPU decision branches on CPU arrays.** Some GPU-only code is selected by a
predicate rather than by array type — the coupled subproblem steppers choose
the batched solve and the device-style constrained final update through
`_gpu_subproblem_execution(sp)`. Forcing that predicate on CPU arrays runs the
GPU branch with CPU linear algebra, which is how
`test_timestepper_boundaries.jl` pins that the GPU and CPU final updates agree
to roundoff on well-posed problems.

When you write a hardware-free GPU test, break the thing it guards and confirm
that exactly the intended assertion fails (mutation testing). Most of the GPU
bugs found in this repository were behind checks that looked like verification
and were not: a parity test whose homogeneous boundary conditions made the
whole boundary path inert, a runner that would have run the CPU path and
reported a pass, a signature that hashed a field that differs across modes.

## Documentation code is tested

`test_webdocs_code.jl` parses every Julia, Bash, TOML, and Dockerfile fence under
`docs/src` on every run. A fence that does not parse fails the suite, so
pseudo-code belongs in a `text` fence. With `TARANG_RUN_WEBDOCS_EXAMPLES=true`
the self-contained Julia examples (fences containing `using Tarang`) are also
executed in fresh temporary directories; `TARANG_WEBDOCS_FILTER` narrows the run
to matching files.

## Writing Tests

### Basic Test

```julia
using Test
using Tarang

@testset "My Feature" begin
    # Setup
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)

    # Test
    @test dist.size == 1
    @test dist.rank == 0
end
```

### Testing Fields

```julia
@testset "ScalarField" begin
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))

    field = ScalarField(dist, "T", (basis,), Float64)

    # Test creation
    @test field.name == "T"
    @test field.dtype == Float64

    # Test data
    Tarang.ensure_layout!(field, :g)
    get_grid_data(field) .= 1.0
    @test all(get_grid_data(field) .== 1.0)
end
```

### Testing Transforms

```julia
@testset "Transforms" begin
    # Setup
    field = ScalarField(dist, "f", (basis,), Float64)

    # Initialize in grid space
    Tarang.ensure_layout!(field, :g)
    get_grid_data(field) .= sin.(x_grid)

    # Transform to spectral
    Tarang.ensure_layout!(field, :c)

    # Transform back
    Tarang.ensure_layout!(field, :g)

    # Check roundtrip
    @test get_grid_data(field) ≈ sin.(x_grid) atol=1e-10
end
```

### Testing Solvers

```julia
@testset "InitialValueProblem Solver" begin
    # Setup problem
    problem = InitialValueProblem([field])
    Tarang.add_equation!(problem, "∂t(f) = -f")

    # Create solver
    solver = InitialValueSolver(problem, RK222(); dt=0.01)

    # Run
    step!(solver)

    # Check
    @test solver.sim_time ≈ 0.01
    @test solver.iteration == 1
end
```

## Test Patterns

### Analytical Comparison

```julia
@testset "Analytical Solution" begin
    # Solve diffusion equation
    # Compare with exact solution
    exact = exp.(-kappa * k^2 * t) .* initial

    @test maximum(abs.(numerical .- exact)) < 1e-6
end
```

### Convergence Test

```julia
@testset "Convergence" begin
    errors = Float64[]

    for N in [16, 32, 64, 128]
        # Solve at resolution N
        error = compute_error(N)
        push!(errors, error)
    end

    # Check spectral convergence
    for i in 2:length(errors)
        @test errors[i] < errors[i-1] / 2
    end
end
```

### MPI Test

```julia
@testset "MPI Parallelism" begin
    MPI.Init()

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    size = MPI.Comm_size(MPI.COMM_WORLD)

    # Test distributed computation
    local_sum = compute_local()
    global_sum = MPI.Allreduce(local_sum, MPI.SUM, MPI.COMM_WORLD)

    @test global_sum ≈ expected_total

    MPI.Finalize()
end
```

## Test Coverage

### Generate Coverage Report

```julia
using Coverage

# Run tests with coverage
coverage = process_folder()

# Print summary
println(coverage)
```

## Continuous Integration

CPU tests run on **GitHub Actions** for every push and pull request:

- the default suite on Julia 1.10/1.11/1.12 across Linux, macOS, and Windows;
- the optional CPU feature tests (`TARANG_ONLY_OPTIONAL_TESTS=true`);
- the MPI suite via `test/run_mpi_ci.jl` at 1, 2, and 4 ranks.

GPU tests cannot run on GitHub-hosted runners (no NVIDIA GPU), so they run on
**Buildkite**, defined in `.buildkite/pipeline.yml`:

- a single-GPU job (`test/run_gpu_ci.jl`) on Julia 1.10/1.11/1.12.

Because CUDA is a *weak* dependency — which keeps CPU installs lean — that job
`Pkg.add`s CUDA before running instead of using the standard package test target.

!!! note "The multi-GPU NCCL job is currently disabled"
    A second step exercising `DISTRIBUTED_GPU_TEST_FILES` (the CUDA + NCCL
    distributed transpose, `test/run_mpi_ci.jl` with the `distributed_gpu`
    fileset at 2 ranks) is commented out at the bottom of
    `.buildkite/pipeline.yml`. It needs two physical GPUs on one agent host.
    Re-enabling is an uncomment plus a `multigpu=true` agent tag; the file spells
    out both. The test files themselves are untouched — they stay in
    `test/file_lists.jl`, `run_mpi_ci.jl` still accepts the fileset, and
    `test/test_gpu_test_files_reachable.jl` still parse-checks them on CPU CI.

### Which agent runs it

Buildkite provides orchestration, not compute: no Buildkite-hosted agent shape
offers a GPU, so the job needs a self-hosted agent on a machine with an NVIDIA
card. Buildkite's free plan supports self-hosted agents, and this pipeline is
four jobs at most, so no paid plan is required.

The queue is parameterized, so the same file works on a personal Buildkite
organization and on JuliaGPU's shared agents with no edit:

| `TARANG_GPU_QUEUE` | Queue used |
|---|---|
| unset | `default` — a self-hosted agent that sets no queue |
| `juliagpu` | JuliaGPU's shared GPU pool |

Set it under Pipeline Settings > Environment Variables.

The agent must also carry a `cuda` tag, since the step requires `cuda: "*"`:

```bash
buildkite-agent start --tags "queue=default,cuda=true"
```

!!! warning "An untagged agent looks like a hang, not an error"
    A step whose agent tags match no connected agent neither fails nor skips —
    the job sits queued until `timeout_in_minutes` expires. Forgetting the `cuda`
    tag therefore presents as a 120-minute stall rather than a configuration
    error.

### When it runs

The Buildkite GitHub App creates a build for every push and every pull request,
but the GPU job does not run on all of them. GPU agent time is scarce, and on a
shared organization the pipeline settings that would narrow the trigger are not
necessarily ours to change, so `.buildkite/pipeline.yml` guards the step with an
`if:` expression instead:

| Build source | Runs the GPU job on |
|---|---|
| push / pull request (Buildkite GitHub App) | `main` only |
| Buildkite UI, "New Build" button | any branch |
| GitHub Actions, the **GPU (Buildkite)** workflow | any branch |

A build whose steps are all filtered out finishes green with no jobs, so pull
request builds stay cheap instead of erroring.

To run the GPU suite on a branch before merging it, dispatch the **GPU
(Buildkite)** workflow (`.github/workflows/gpu-buildkite.yml`) from the GitHub
Actions tab and give it the branch name. It calls the Buildkite REST API, which
produces an `api`-source build that the guard lets through on any branch. That
workflow needs a one-time `BUILDKITE_API_TOKEN` repository secret (a Buildkite
token with the `write_builds` scope) and, if the pipeline slug differs from the
default, the `BUILDKITE_ORG` / `BUILDKITE_PIPELINE` repository variables — its
header comment spells out the setup. Pressing "New Build" in the Buildkite UI
does the same thing without any GitHub-side configuration.

Add `[skip tests]` to a commit message to suppress the GPU job regardless of how
the build was created.

### Reporting a GPU run by hand

If the GPU machine is not running an agent — or you just want a one-off result on
a branch — `scripts/gpu_ci_report.sh` runs the suite locally and posts the outcome
to GitHub as a commit status, which shows up as a check next to the commit and on
any pull request containing it:

```bash
./scripts/gpu_ci_report.sh                 # test HEAD, post a gpu/cuda status
./scripts/gpu_ci_report.sh --sha 6a4da42   # require the checkout to be this commit
./scripts/gpu_ci_report.sh --no-status     # run only, post nothing
./scripts/gpu_ci_report.sh --gist          # also upload the log as a secret gist
```

The reporter only runs from a clean working tree (including no untracked files),
and `--sha` must resolve to the checkout's current `HEAD`. To report another
commit, check out that commit first. These preconditions ensure the status can
only describe the exact source tree the GPU suite executed.

It needs `gh` authenticated with a token carrying `repo:status` (the plain `repo`
scope covers it), and CUDA available in Julia's stacked default environment.
Install it with `julia --project=@v#.# -e 'using Pkg; Pkg.add("CUDA")'`; installing
there keeps this checkout clean, which the reporter requires.

!!! warning "It refuses to run without a working GPU, by design"
    Every file in `GPU_TEST_FILES` self-guards with `CUDA.functional()` and exits
    0 when no device is present. That is right for CI, but it means running the
    suite on a CUDA-less machine yields a *vacuous pass*: every test skipped, exit
    code 0, and a green status for a GPU suite that never touched a GPU. The
    script therefore hard-gates on `CUDA.functional()` and reports a missing
    device as an `error` status, never as `success`.

## See Also

- [Contributing](contributing.md): Development guidelines
- [Architecture](architecture.md): Code structure
