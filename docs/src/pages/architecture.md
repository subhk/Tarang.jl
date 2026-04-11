# Architecture

Overview of Tarang.jl's internal architecture.

## Package Structure

```
Tarang.jl/
├── src/
│   ├── Tarang.jl              # Main module
│   ├── core/                  # Core functionality
│   │   ├── architectures.jl     # CPU/GPU abstraction
│   │   ├── coords.jl           # Coordinate systems
│   │   ├── basis.jl            # Spectral bases (Fourier, Chebyshev, Legendre)
│   │   ├── distributor.jl      # MPI distribution
│   │   ├── domain.jl           # Domain construction
│   │   ├── field.jl            # Hub: includes field/ sub-files
│   │   ├── field/
│   │   │   ├── field_types.jl     # ScalarField, VectorField, TensorField
│   │   │   ├── field_data.jl      # Data access, allocation, components
│   │   │   ├── field_layout.jl    # Layout transitions, transforms
│   │   │   └── field_exports.jl   # Export declarations
│   │   ├── operators/           # Differential operators (hub + sub-files)
│   │   ├── transforms/          # Spectral transforms (hub + sub-files)
│   │   ├── problems.jl         # Hub: includes problems/ sub-files
│   │   ├── problems/
│   │   │   ├── problem_types.jl   # IVP, LBVP, NLBVP, EVP
│   │   │   ├── problem_parsing.jl # Expression parsing
│   │   │   ├── problem_matrices.jl# Matrix building for solvers
│   │   │   └── problem_utils.jl   # Validation, introspection
│   │   ├── solvers.jl           # Hub: includes solvers/ sub-files
│   │   ├── solvers/
│   │   │   ├── solver_types.jl       # Solver definitions
│   │   │   ├── solver_stepping.jl    # Time stepping, BVP/EVP solve
│   │   │   ├── solver_compiled_rhs.jl# RHS compilation & execution
│   │   │   └── solver_utils.jl       # Diagnostics, exports
│   │   ├── timesteppers/        # Time integration (hub + sub-files)
│   │   ├── boundary_conditions.jl
│   │   ├── evaluator.jl
│   │   └── nonlinear.jl
│   ├── tools/                 # Utilities and I/O
│   │   ├── config.jl            # Configuration management
│   │   ├── netcdf_output.jl     # NetCDF output handlers
│   │   ├── netcdf_merge.jl      # NetCDF file merging
│   │   ├── matsolvers.jl        # CPU matrix solvers
│   │   ├── gpu_matsolvers.jl    # GPU matrix solvers
│   │   └── ...                  # logging, parsing, progress, etc.
│   └── extras/                # Convenience functions
│       ├── quick_domains.jl     # PeriodicDomain, ChannelDomain, etc.
│       ├── flow_tools.jl        # Energy, enstrophy, CFL diagnostics
│       └── plot_tools.jl        # Plotting utilities (experimental)
├── ext/                       # CUDA extension (loaded when CUDA.jl available)
│   ├── TarangCUDAExt.jl         # Extension entry point
│   └── cuda/                    # GPU implementations
│       ├── config.jl              # Device ID, tensor cores
│       ├── memory.jl              # CPU↔GPU data transfer
│       ├── architecture.jl        # GPU type methods
│       ├── transforms.jl          # CUFFT plans
│       ├── kernels.jl             # KernelAbstractions GPU kernels
│       ├── dct.jl                 # DCT for Chebyshev basis
│       └── ...                    # batched FFT, NCCL, pencil, etc.
├── test/                      # Tests
└── docs/                      # Documentation
```

## Core Components

### Coordinates

```julia
abstract type Coordinates end

struct CartesianCoordinates <: Coordinates
    names::Vector{String}
    coords::Vector{Coordinate}
end
```

### Bases

```julia
abstract type SpectralBasis end

struct RealFourier <: SpectralBasis
    coord::Coordinate
    size::Int
    bounds::Tuple{Float64, Float64}
    dealias::Float64
end

struct ChebyshevT <: SpectralBasis
    coord::Coordinate
    size::Int
    bounds::Tuple{Float64, Float64}
end
```

### Fields

```julia
abstract type AbstractField end

mutable struct ScalarField <: AbstractField
    dist::Distributor
    name::String
    bases::Tuple
    dtype::Type
    data_g::Union{Array, Nothing}
    data_c::Union{Array, Nothing}
    current_layout::Symbol
end
```

### Distributor

```julia
struct Distributor
    coords::Coordinates
    comm::MPI.Comm
    rank::Int
    size::Int
    mesh::Tuple
end
```

## Data Flow

### Simulation Lifecycle

```
1. Setup
   Coordinates → Distributor → Bases → Domain → Fields

2. Problem Definition
   Fields → Problem → Equations + BCs

3. Solving
   Problem → Solver → Time stepping loop

4. Output
   Fields → Handler → Files
```

### Transform Flow

```
Grid Space (physical)
    ↓ forward transform
Coefficient Space (spectral)
    ↓ apply operators
Coefficient Space
    ↓ inverse transform
Grid Space
```

## MPI Parallelism

### Pencil Decomposition

```
Global Array         Local Arrays (4 processes)
┌──────────┐        ┌────┬────┐
│          │        │ P0 │ P1 │
│          │   →    ├────┼────┤
│          │        │ P2 │ P3 │
└──────────┘        └────┴────┘
```

### Communication Patterns

- **All-to-all**: Layout transforms
- **Allreduce**: Global reductions
- **Gather**: Output to single file

## Transform Architecture

### Layout Management

```julia
function ensure_layout!(field, target_layout)
    if field.current_layout != target_layout
        transform!(field, target_layout)
    end
end
```

### Transform Types

- Forward: Grid → Spectral
- Inverse: Spectral → Grid

## Problem Architecture

### Problem Types

```julia
abstract type Problem end

struct IVP <: Problem
    fields::Vector{ScalarField}
    equations::Vector{Equation}
    boundary_conditions::Vector{BC}
    parameters::Dict{String, Any}
end
```

### Equation Parsing

```
String → Tokenize → Parse → AST → Evaluation
```

## Solver Architecture

### Time Steppers

```julia
abstract type TimeStepper end

struct RK222 <: TimeStepper end
struct SBDF2 <: TimeStepper end
```

### Solver State

```julia
struct InitialValueSolver
    problem::IVP
    timestepper::TimeStepper
    state::Vector
    sim_time::Float64
    iteration::Int
end
```

## Extension Points

### Custom Bases

Implement `SpectralBasis` interface:
- `size(basis)`
- `get_grid(basis)`
- `forward_transform(data, basis)`
- `inverse_transform(data, basis)`

### Custom Timesteppers

Implement `TimeStepper` interface:
- `step!(solver, dt)`
- `stages(stepper)`

## Design Principles

1. **Separation of concerns**: Coordinates, bases, fields, problems
2. **Lazy transforms**: Transform only when needed
3. **MPI transparency**: Users don't manage communication
4. **Extensibility**: Abstract types for customization
5. **Julia idioms**: Multiple dispatch, type stability

## See Also

- [Contributing](contributing.md): Development guidelines
- [Testing](testing.md): Test architecture
