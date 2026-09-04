using Tarang
using MPI
using PencilArrays
using Test

if !MPI.Initialized()
    MPI.Init()
end
const comm   = MPI.COMM_WORLD
const rank   = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

if nprocs != 4
    rank == 0 && @warn "C1/C3 2D-mesh guard needs exactly 4 ranks; got $nprocs"
    MPI.Finalized() || MPI.Finalize()
    exit(0)
end

# Non-degenerate 2x2 mesh (exercises the coord-ordering bug C1 on off-diagonal
# ranks) with a NON-divisible decomposed axis N=5 (exercises remainder bug C3).
coords = CartesianCoordinates("x", "y", "z")
dist   = Distributor(coords; mesh=(2, 2), dtype=Float64, architecture=CPU())
gshape = (8, 5, 6)   # axis1 local; axes 2,3 decomposed over the 2x2 mesh

# Authoritative owned slab straight from PencilArrays (the SAME Pencil the field
# data uses for storage): decompose the LAST ndims_mesh dims, NoPermutation.
pa  = Tarang.create_pencil(dist, gshape, nothing)
pen = PencilArrays.pencil(pa)
rl  = range_local(pen, LogicalOrder())   # one UnitRange per global (logical) axis

# Tarang's notion of the owned ranges / shape, which MUST agree with `rl`.
li    = ntuple(ax -> Tarang.local_indices(dist, ax, gshape[ax]), length(gshape))
shape = Tarang.compute_local_shape(dist, gshape)
local_array_size = Tarang.get_local_array_size(dist, gshape)

ranges_match = all(ax -> li[ax] == rl[ax], 1:length(gshape))
shape_match  = shape == size(pa)
local_size_match = local_array_size == size(pa)
ok_local     = ranges_match && shape_match && local_size_match

for r in 0:nprocs-1
    if r == rank
        println("rank $rank coords_local=$(dist.mpi_topology.coords_local) ",
                "li=$(li) rl=$(Tuple(rl)) shape=$(shape) local_array_size=$(local_array_size) ",
                "size(pa)=$(size(pa)) ok=$ok_local")
    end
    MPI.Barrier(comm)
end

ok_global = MPI.Allreduce(ok_local ? 1 : 0, MPI.MIN, comm) == 1

@testset "C1/C3 Distributor matches PencilArrays slab (2x2 mesh, np=4, rank=$rank)" begin
    @test ranges_match
    @test shape_match
    @test local_size_match
    @test ok_global
end

# Coefficient storage uses PencilFFT's output pencil, whose orientation and
# first-RealFourier RFFT extent differ from the grid pencil above.  The public
# shape helper must report that authoritative local geometry.
bases = (
    RealFourier(coords["x"]; size=gshape[1], bounds=(0.0, 2π)),
    RealFourier(coords["y"]; size=gshape[2], bounds=(0.0, 2π)),
    RealFourier(coords["z"]; size=gshape[3], bounds=(0.0, 2π)),
)
domain = Domain(dist, bases)
field = ScalarField(domain, "coeff_shape")

reported_coeff_shape = get_local_coeff_shape(dist, domain)
canonical_coeff_shape = local_shape(domain, :c)
allocated_coeff_shape = size(get_coeff_data(field))

@testset "3D coefficient helper matches PencilFFT output (2x2 mesh, np=4, rank=$rank)" begin
    @test reported_coeff_shape == canonical_coeff_shape
    @test reported_coeff_shape == allocated_coeff_shape
end

# Exercise the non-cached metadata path too. Complex data over RealFourier uses
# a full first-axis spectrum, and the query must not build its collective plan.
complex_key = (objectid(domain), ComplexF64)
reported_complex_shape = get_local_coeff_shape(dist, domain; dtype=ComplexF64)
@testset "3D alternate-dtype coefficient metadata is non-collective (rank=$rank)" begin
    @test !haskey(dist.transform_plan_cache, complex_key)
end
complex_field = ScalarField(domain, "complex_coeff_shape"; dtype=ComplexF64)
@testset "3D alternate-dtype coefficient metadata matches storage (rank=$rank)" begin
    @test reported_complex_shape == size(get_coeff_data(complex_field))
end

# A 3D slab's final PencilFFT output is decomposed over logical axis 2 (not 1),
# so cover the geometry case that differs from both 2D slabs and 3D pencils.
slab_dist = Distributor(coords; mesh=(4,), dtype=Float64, architecture=CPU())
slab_domain = Domain(slab_dist, bases)
slab_complex_key = (objectid(slab_domain), ComplexF64)
reported_slab_complex_shape = get_local_coeff_shape(
    slab_dist, slab_domain; dtype=ComplexF64)
@testset "3D slab alternate-dtype metadata is non-collective (rank=$rank)" begin
    @test !haskey(slab_dist.transform_plan_cache, slab_complex_key)
end
slab_complex_field = ScalarField(
    slab_domain, "slab_complex_coeff_shape"; dtype=ComplexF64)
@testset "3D slab alternate-dtype metadata matches storage (rank=$rank)" begin
    @test reported_slab_complex_shape == size(get_coeff_data(slab_complex_field))
end

MPI.Finalized() || MPI.Finalize()
