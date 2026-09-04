# Guard: distributed NONLINEAR Cheb-Fourier IVP solve == serial (np >= 2).
#
# The hardest MPI CPU integration path: a channel-like Burgers problem
#   ∂t(b) - κ·div(grad_b) + τ_lift(τ₂) = -b·∂x(b),  b(z=0)=b(z=1)=0
# exercises, all under MPI decomposition, the full stack together — Cheb-Fourier
# transforms, spectral ∂x derivative, the 3/2 padded dealiasing on the DECOMPOSED
# Fourier axis (the explicit -b·∂x(b) term), the per-Fourier-mode Chebyshev tau
# solve (coeff-space solve-transpose) with Dirichlet BCs, and the IMEX RK222 step.
# Reference = serial (np=1) result of this exact problem, including the RK
# final-state constraint projection.
using Tarang
using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "Distributed nonlinear Cheb-Fourier IVP test requires >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const SUMSQ_REF = 33.39084552380445
const BMAX_REF  = 1.4380128840994046

_loc(f) = get_grid_data(f) isa PencilArrays.PencilArray ? parent(get_grid_data(f)) : get_grid_data(f)
function _global_boundary_max(field, logical_dim, endpoints, comm)
    data = get_grid_data(field)
    if data isa PencilArrays.PencilArray
        raw = parent(data)
        nspatial = ndims(PencilArrays.pencil(data))
        # Parent storage follows memory order; translate the logical dimension.
        logical_dims_in_memory_order =
            PencilArrays.permutation(data) * ntuple(identity, nspatial)
        memory_dim = findfirst(==(logical_dim),
                               logical_dims_in_memory_order)::Int
        local_ranges = PencilArrays.range_local(data, PencilArrays.MemoryOrder())
    else
        raw = data
        memory_dim = logical_dim
        local_ranges = axes(raw)
    end
    local_max = zero(typeof(abs(zero(eltype(raw)))))
    for global_index in endpoints
        local_index = findfirst(==(global_index), local_ranges[memory_dim])
        local_index === nothing && continue
        local_max = max(local_max,
                        maximum(abs, selectdim(raw, memory_dim, local_index)))
    end
    return MPI.Allreduce(local_max, MPI.MAX, comm)
end

function _assign_local!(field, gdata)
    data = get_grid_data(field)
    if data isa PencilArrays.PencilArray
        ax = PencilArrays.pencil(data).axes_local
        parent(data) .= gdata[ax...]
    else
        data .= gdata
    end
end

@testset "Distributed NONLINEAR Cheb-Fourier IVP matches serial (rank=$rank)" begin
    kappa = 0.1; Lz = 1.0; dt = 1e-3; NSTEPS = 15; Nz = 12; Nx = 8
    coords = CartesianCoordinates("z", "x")
    dist = Distributor(coords; dtype=Float64, architecture=CPU())
    zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))
    xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π), dealias=3/2)
    domain = Domain(dist, (zbasis, xbasis))
    b = ScalarField(domain, "b")
    tau_b1 = ScalarField(dist, "tau_b1", (), Float64)
    tau_b2 = ScalarField(dist, "tau_b2", (), Float64)
    ex, ez = unit_vector_fields(coords, dist)
    lift_basis = derivative_basis(zbasis, 1)
    τ_lift(A) = lift(A, lift_basis, -1)
    grad_b = grad(b) + ez * τ_lift(tau_b1)
    problem = IVP([b, tau_b1, tau_b2])
    add_parameters!(problem, kappa=kappa, ez=ez, grad_b=grad_b, τ_lift=τ_lift)
    add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -b*∂x(b)")
    add_bc!(problem, "b(z=0) = 0")
    add_bc!(problem, "b(z=1) = 0")
    solver = InitialValueSolver(problem, RK222(); dt=dt)

    xfull = [2π*(i-1)/Nx for i in 1:Nx]
    zfull = [Lz/2*(1-cos(π*(k-1)/(Nz-1))) for k in 1:Nz]
    b0(z, x) = sin(π*z/Lz)*(1 + 0.5*cos(2*x))
    gdata = [b0(zfull[iz], xfull[ix]) for iz in 1:Nz, ix in 1:Nx]
    ensure_layout!(b, :g); _assign_local!(b, gdata); ensure_layout!(b, :c)

    for _ in 1:NSTEPS
        step!(solver, dt)
    end

    ensure_layout!(b, :g); lv = _loc(b)
    sumsq = MPI.Allreduce(sum(abs2, lv), MPI.SUM, comm)
    bmax = MPI.Allreduce(maximum(abs, lv), MPI.MAX, comm)
    bcmax = _global_boundary_max(b, 1, (1, Nz), comm)
    rank == 0 && println("  NL np=$nprocs sumsq=$sumsq bmax=$bmax bcmax=$bcmax " *
                         "(ref sumsq=$SUMSQ_REF)")
    @test isapprox(bmax, BMAX_REF; atol=1e-10)
    @test isapprox(sumsq, SUMSQ_REF; atol=1e-6)
    @test bcmax < 1e-12
end

MPI.Barrier(comm)
MPI.Finalized() || MPI.Finalize()
