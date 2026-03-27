# Testing

Guide to running and writing tests for Tarang.jl.

## Running Tests

### Full Test Suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Specific Test File

```bash
julia --project=. test/test_specific.jl
```

### With MPI

```bash
mpiexec -n 4 julia --project=. test/runtests.jl
```

## Test Structure

```
test/
├── runtests.jl              # Main test runner
├── test_cfl.jl              # CFL condition tests
├── test_domain_metadata.jl  # Domain tests
├── test_solvers.jl          # Solver tests
├── test_flow_tools.jl       # Analysis tools
├── test_quick_domains.jl    # Domain helpers
├── test_plot_tools.jl       # Visualization
└── test_compatibility.jl    # Compatibility tests
```

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
    field.data_g .= 1.0
    @test all(field.data_g .== 1.0)
end
```

### Testing Transforms

```julia
@testset "Transforms" begin
    # Setup
    field = ScalarField(dist, "f", (basis,), Float64)

    # Initialize in grid space
    Tarang.ensure_layout!(field, :g)
    field.data_g .= sin.(x_grid)

    # Transform to spectral
    Tarang.ensure_layout!(field, :c)

    # Transform back
    Tarang.ensure_layout!(field, :g)

    # Check roundtrip
    @test field.data_g ≈ sin.(x_grid) atol=1e-10
end
```

### Testing Solvers

```julia
@testset "IVP Solver" begin
    # Setup problem
    problem = IVP([field])
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

Tests run automatically on:
- Pull requests
- Pushes to main
- Scheduled (nightly)

## See Also

- [Contributing](contributing.md): Development guidelines
- [Architecture](architecture.md): Code structure
