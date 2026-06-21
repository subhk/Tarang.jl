# C4 guard: the forcing work/power diagnostics divide a per-rank LOCAL-slab
# partial sum by the GLOBAL domain area, so the partial must be combined across
# ranks first (MPI.Allreduce SUM) or each rank returns only its fraction. The fix
# is the `_forcing_reduce_partial` helper, applied in work_stratonovich/work_ito/
# instantaneous_power. This test exercises that helper directly (it does not go
# through `_matched_forcing_view`, which has a separate, pre-existing PencilArray
# permutation bug in MPI mode unrelated to C4).
using Tarang
using MPI
using PencilArrays
using Test

if !MPI.Initialized()
    MPI.Init()
end
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "C4 forcing-reduce test requires >= 2 ranks"
    MPI.Finalize(); exit(0)
end

# A distributed scalar field gives us a real PencilArray to dispatch on.
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π))
q = ScalarField(Domain(dist, (xb, yb)), "q")
ensure_layout!(q, :c)
sol = get_coeff_data(q)

@testset "C4 _forcing_reduce_partial combines slab partials globally (rank=$rank)" begin
    @test sol isa PencilArrays.PencilArray
    # Each rank contributes a distinct partial; the helper must return the GLOBAL
    # sum, identical on every rank (pre-fix the diagnostics used the bare partial).
    partial = Float64(rank + 1)
    g = Tarang._forcing_reduce_partial(sol, partial)
    @test g == sum(1:nprocs)
    # All ranks agree (byte-identical reduced scalar).
    @test g == MPI.Allreduce(partial, MPI.SUM, comm)
    # The plain-array overload is a serial no-op (serial path untouched).
    @test Tarang._forcing_reduce_partial(zeros(3), 7.0) == 7.0
end

MPI.Finalize()
