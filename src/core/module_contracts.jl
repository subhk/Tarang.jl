# Abstract contracts used to break circular dependencies between implementation
# files. Concrete types live in their owning files; these shared names let early
# loaded code reference later subsystems without forcing a larger module split.
#
# Target submodule architecture:
#   Tarang.Core         - architecture, coordinates, bases, domains, fields
#   Tarang.Operators    - differential operators and evaluation
#   Tarang.TransformOps - transform operations; Tarang.Transforms is PencilFFTs
#   Tarang.Solvers      - problems, solvers, timesteppers, boundary conditions
#   Tarang.Output       - NetCDF, logging, progress, configuration
#   Tarang.Extras       - flow tools, plotting, quick domains, analysis tasks

abstract type AbstractNonlinearEvaluator end
abstract type AbstractEvaluator end
abstract type AbstractDistributedGPUConfig end
abstract type AbstractTransposeComms end
abstract type AbstractTransposeCounts end
abstract type TimeStepper end
abstract type AbstractTimestepperState end

# Custom PencilConfig struct for pencil array configuration.
struct PencilConfig{N, M}
    global_shape::NTuple{N, Int}
    mesh::NTuple{M, Int}
    comm::MPI.Comm
    decomp_dims::NTuple{M, Bool}
    dtype::Type

    function PencilConfig(global_shape::NTuple{N, Int}, mesh::NTuple{M, Int};
                         comm::MPI.Comm=MPI.COMM_WORLD,
                         decomp_dims::NTuple{M, Bool}=ntuple(i -> true, M),
                         dtype::Type=Float64) where {N, M}
        new{N, M}(global_shape, mesh, comm, decomp_dims, dtype)
    end
end
