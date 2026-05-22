using MPI, PencilArrays, Tarang
MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; mesh=(nprocs,), dtype=Float64, architecture=CPU())
N = 64
bx = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
by = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
domain = Domain(dist, (bx, by))

fa = ScalarField(domain, "fa"); fb = ScalarField(domain, "fb")
ensure_layout!(fa, :g); ensure_layout!(fb, :g)
da = get_grid_data(fa); db = get_grid_data(fb)
parent(da) .= rand(size(parent(da))...); parent(db) .= rand(size(parent(db))...)

ev = Tarang.NonlinearEvaluator(dist; dealiasing_factor=3/2)

# Real solver scenario: operands arrive in COEFF space.
ensure_layout!(fa, :c); ensure_layout!(fb, :c)

# warm up (compile) — separate fields so counts below are clean
fa2 = ScalarField(domain, "fa2"); fb2 = ScalarField(domain, "fb2")
ensure_layout!(fa2, :g); ensure_layout!(fb2, :g)
ensure_layout!(fa2, :c); ensure_layout!(fb2, :c)
Tarang.evaluate_transform_multiply(fa2, fb2, ev)

Tarang._XFORM_FWD[] = 0
Tarang._XFORM_BWD[] = 0
Tarang.evaluate_transform_multiply(fa, fb, ev)
fwd = Tarang._XFORM_FWD[]; bwd = Tarang._XFORM_BWD[]

tot_fwd = MPI.Allreduce(fwd, MPI.SUM, comm)
tot_bwd = MPI.Allreduce(bwd, MPI.SUM, comm)
rank == 0 && println("XFORM forward=$tot_fwd backward=$tot_bwd total=$(tot_fwd+tot_bwd) (summed over $nprocs ranks, 1 product, coeff inputs)")
