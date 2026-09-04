using Tarang
using MPI
using Test
using PencilArrays
using Random

if !MPI.Initialized()
    MPI.Init()
end

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

# Assign a global, position-determined array into the field's local slab using the
# pencil's global index range (axes_local), exactly as the working MPI tests do.
function assign_global!(f, g)
    data = get_grid_data(f)
    if data isa PencilArrays.PencilArray
        ax = PencilArrays.pencil(data).axes_local
        parent(data) .= g[ax...]
    else
        data .= g
    end
end

local_grid_parent(f) = (d = get_grid_data(f); d isa PencilArrays.PencilArray ? parent(d) : d)

function roundtrip_and_energy(mesh, N, g)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; mesh=mesh, dtype=Float64, architecture=CPU())
    bx = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    by = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    domain = Domain(dist, (bx, by))
    f = ScalarField(domain, "f")
    ensure_layout!(f, :g)
    assign_global!(f, g)
    orig = copy(local_grid_parent(f))
    # forward (grid -> coeff): exercises the PencilFFT plan that crashes pre-fix
    ensure_layout!(f, :c)
    cdata = Tarang.get_coeff_data(f)
    cpar = cdata isa PencilArrays.PencilArray ? parent(cdata) : cdata
    local_energy = sum(abs2, cpar)
    # backward (coeff -> grid)
    ensure_layout!(f, :g)
    rt_err = maximum(abs.(local_grid_parent(f) .- orig))
    rt_err = MPI.Allreduce(rt_err, MPI.MAX, comm)
    energy = MPI.Allreduce(local_energy, MPI.SUM, comm)
    return rt_err, energy
end

@testset "unit-factor mesh (1,N) == slab mesh (N,) for 2D Fourier (rank=$rank)" begin
    if nprocs != 2
        rank == 0 && @warn "this test requires exactly 2 ranks"
    else
        N = 16
        g = randn(MersenneTwister(7), N, N)
        # Proven slab mesh (already in CI).
        err_slab, en_slab = roundtrip_and_energy((nprocs,), N, g)
        # Unit-factor mesh: PRE-FIX this throws AssertionError(allunique) during
        # plan construction; POST-FIX it is normalized to (nprocs,) and matches.
        err_unit, en_unit = roundtrip_and_energy((1, nprocs), N, g)
        @test err_slab < 1e-10
        @test err_unit < 1e-10
        @test isapprox(en_unit, en_slab; rtol=1e-10)
    end
end

@testset "invalid full/over-dimensional non-unit meshes fail fast (rank=$rank)" begin
    if nprocs != 4
        rank == 0 && @warn "over-dimensional mesh test requires exactly 4 ranks"
    else
        coords_1d = CartesianCoordinates("x")
        err_1d = try
            Distributor(coords_1d; mesh=(2, 2), dtype=Float64, architecture=CPU())
            nothing
        catch e
            e
        end
        @test err_1d isa ArgumentError
        @test occursin("exceeds domain dimensionality", sprint(showerror, err_1d))

        coords_2d = CartesianCoordinates("x", "y")
        dist_2d = try
            Distributor(coords_2d; mesh=(2, 2), dtype=Float64, architecture=CPU())
        catch e
            e
        end
        @test dist_2d isa Distributor

        if dist_2d isa Distributor
            # Full decomposition remains valid for data-only PencilArrays.
            storage = create_pencil(dist_2d, (8, 8), nothing; dtype=Float64)
            @test MPI.Allreduce(length(storage), MPI.SUM, comm) == 8 * 8

            bx = RealFourier(coords_2d["x"]; size=8, bounds=(0.0, 2π))
            by = RealFourier(coords_2d["y"]; size=8, bounds=(0.0, 2π))
            domain_err = try
                Domain(dist_2d, (bx, by))
                nothing
            catch e
                e
            end
            @test domain_err isa ArgumentError
            @test occursin("must be less than spectral domain dimensionality",
                           sprint(showerror, domain_err))
            @test dist_2d.pencil_fft_plan === nothing
        end
    end
end

if !MPI.Finalized()
    MPI.Finalize()
end
