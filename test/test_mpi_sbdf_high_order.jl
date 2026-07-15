# Regression: subproblem-path SBDF3/SBDF4 startup must retain nominal order.
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NP = MPI.Comm_size(COMM)

if NP < 2
    RANK == 0 && @warn "MPI SBDF high-order test needs at least two ranks"
    MPI.Finalize()
    exit(0)
end

function _sbdf_diffusion_error(stepper, dt; tfinal=0.08, Nz=18, Nx=8, κ=0.1)
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    zb = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
    domain = Domain(dist, (zb, xb))
    b = ScalarField(domain, "b")
    τ1 = ScalarField(dist, "tau_b1", (), Float64)
    τ2 = ScalarField(dist, "tau_b2", (), Float64)
    _, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zb, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * τ_lift(τ1)
    problem = IVP([b, τ1, τ2])
    add_parameters!(problem, kappa=κ, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
    add_equation!(problem, "dt(b) - kappa*div(grad_b) + τ_lift(tau_b2) = 0")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, stepper; dt=dt)

    xs = [2π * (i - 1) / Nx for i in 1:Nx]
    zs = [0.5 * (1 - cos(π * (k - 1) / (Nz - 1))) for k in 1:Nz]
    initial = [sin(π*z) * (1 + 0.5cos(2x)) for z in zs, x in xs]
    ensure_layout!(b, :g)
    gd = get_grid_data(b)
    ax = PencilArrays.pencil(gd).axes_local
    parent(gd) .= initial[ax...]
    ensure_layout!(b, :c)

    for _ in 1:round(Int, tfinal / dt)
        step!(solver, dt)
    end

    exact = [sin(π*z) * (exp(-κ*π^2*tfinal) +
             0.5exp(-κ*(π^2 + 4)*tfinal)*cos(2x)) for z in zs, x in xs]
    ensure_layout!(b, :g)
    gd = get_grid_data(b)
    ax = PencilArrays.pencil(gd).axes_local
    local_error = maximum(abs.(parent(gd) .- exact[ax...]))
    return MPI.Allreduce(local_error, MPI.MAX, COMM)
end

@testset "MPI subproblem SBDF startup preserves formal order (rank=$RANK)" begin
    e3_coarse = _sbdf_diffusion_error(SBDF3(), 0.01)
    e3_fine = _sbdf_diffusion_error(SBDF3(), 0.005)
    rate3 = log2(e3_coarse / e3_fine)
    RANK == 0 && @info "SBDF3 subproblem convergence" e3_coarse e3_fine rate3
    @test rate3 > 2.5

    e4_coarse = _sbdf_diffusion_error(SBDF4(), 0.01)
    e4_fine = _sbdf_diffusion_error(SBDF4(), 0.005)
    rate4 = log2(e4_coarse / e4_fine)
    RANK == 0 && @info "SBDF4 subproblem convergence" e4_coarse e4_fine rate4
    @test rate4 > 3.2
end

MPI.Barrier(COMM)
MPI.Finalized() || MPI.Finalize()
