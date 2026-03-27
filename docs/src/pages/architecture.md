# Architecture

Overview of Tarang.jl's internal architecture.

## Package Structure

```
Tarang.jl/
├── src/
│   ├── Tarang.jl          # Main module
│   ├── core/              # Core functionality
│   │   ├── coordinates.jl
│   │   ├── basis.jl
│   │   ├── domain.jl
│   │   ├── field.jl
│   │   ├── distributor.jl
│   │   ├── operators.jl
│   │   ├── transforms.jl
│   │   ├── problems.jl
│   │   ├── boundary_conditions.jl
│   │   ├── solvers.jl
│   │   └── evaluator.jl
│   ├── tools/             # Utilities and I/O
│   │   ├── array.jl         # Array manipulation utilities
│   │   ├── cache.jl         # Caching utilities
│   │   ├── config.jl        # Configuration management
│   │   ├── dispatch.jl      # Multiple dispatch helpers
│   │   ├── exceptions.jl    # Custom exception types
│   │   ├── general.jl       # General utilities
│   │   ├── logging.jl       # MPI-aware logging
│   │   ├── matsolvers.jl    # Matrix solvers
│   │   ├── netcdf_merge.jl  # NetCDF merging
│   │   ├── netcdf_output.jl # NetCDF output handlers
│   │   ├── parallel.jl      # Parallel utilities
│   │   ├── parsing.jl       # Expression parsing
│   │   ├── progress.jl      # Progress tracking
│   │   ├── random_arrays.jl # Random array generation
│   │   └── temporal_filters.jl # Time integration filters
│   └── extras/            # Convenience functions
│       ├── flow_tools.jl
│       └── quick_domains.jl
├── test/                  # Tests
└── docs/                  # Documentation
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
