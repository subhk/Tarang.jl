# Coefficient-level parity for the TransposableField distributed transform.
#
# The np>1 testsets in test_transposable_field.jl only ROUND-TRIP. A round trip
# is blind to any permutation that forward and backward both apply, which is
# exactly the failure mode a decomposition-convention refactor can introduce.
# These assertions compare distributed COEFFICIENTS against the serial reference
# sliced to each rank's own block.
using Test
using Tarang
using MPI

MPI.Initialized() || MPI.Init()
const COMM = MPI.COMM_WORLD
const NP = MPI.Comm_size(COMM)
const RANK = MPI.Comm_rank(COMM)

# Distinct per-axis structure so a swapped axis cannot coincidentally agree.
f2(i, j, Nx, Ny) = sin(2π * (i - 1) / Nx) * cos(4π * (j - 1) / Ny) +
                   0.3 * cos(6π * (i - 1) / Nx)
f3(i, j, k, Nx, Ny, Nz) = sin(2π * (i - 1) / Nx) * cos(4π * (j - 1) / Ny) +
                          0.3 * cos(2π * (k - 1) / Nz) * sin(2π * (j - 1) / Ny)

function serial_coeffs_2d(Nx, Ny)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; comm=MPI.COMM_SELF, mesh=(1,), dtype=ComplexF64,
                       architecture=CPU(), use_pencil_arrays=false)
    bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny))
    field = ScalarField(dist, "parity_ref_2d", bases)
    g = Tarang.get_grid_data(field)
    for j in 1:Ny, i in 1:Nx
        g[i, j] = complex(f2(i, j, Nx, Ny), 0.0)
    end
    forward_transform!(field)
    return copy(Tarang.get_coeff_data(field))
end

function serial_coeffs_3d(Nx, Ny, Nz)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; comm=MPI.COMM_SELF, mesh=(1,), dtype=ComplexF64,
                       architecture=CPU(), use_pencil_arrays=false)
    bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny),
             ComplexFourier(coords, "z", Nz))
    field = ScalarField(dist, "parity_ref_3d", bases)
    g = Tarang.get_grid_data(field)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        g[i, j, k] = complex(f3(i, j, k, Nx, Ny, Nz), 0.0)
    end
    forward_transform!(field)
    return copy(Tarang.get_coeff_data(field))
end

"Global x/y index ranges this rank owns under the TransposableField convention."
function block_ranges(mesh, Nx, Ny)
    Rx = mesh[1]
    Ry = length(mesh) >= 2 ? mesh[2] : 1
    rx = RANK % Rx
    ry = (RANK ÷ Rx) % Ry
    return (Tarang.local_range(Nx, Rx, rx), Tarang.local_range(Ny, Ry, ry))
end

@testset "TransposableField coefficient parity (np=$NP)" begin

    @testset "2D mesh=$mesh" for mesh in (NP == 4 ? ((4, 1), (1, 4), (2, 2)) : ((NP, 1),))
        Nx, Ny = 8, 6
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; comm=COMM, mesh=mesh, dtype=ComplexF64,
                           architecture=CPU(), use_pencil_arrays=false)
        bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny))
        field = ScalarField(dist, "parity_2d_$(mesh)", bases)

        ox, oy = block_ranges(mesh, Nx, Ny)
        g = Tarang.get_grid_data(field)
        @test size(g) == (length(ox), length(oy))
        for (jl, jg) in enumerate(oy), (il, ig) in enumerate(ox)
            g[il, jl] = complex(f2(ig, jg, Nx, Ny), 0.0)
        end
        original = copy(g)

        reference = serial_coeffs_2d(Nx, Ny)
        tf = TransposableField(field)
        distributed_forward_transform!(tf)

        c = Tarang.get_coeff_data(field)
        @test size(c) == (length(ox), length(oy))
        @test maximum(abs, c .- reference[ox, oy]; init=0.0) < 1e-10

        distributed_backward_transform!(tf)
        @test maximum(abs, Tarang.get_grid_data(field) .- original; init=0.0) < 1e-10
    end

    @testset "3D mesh=$mesh" for mesh in (NP == 4 ? ((2, 2), (4, 1), (1, 4)) : ((NP, 1),))
        Nx, Ny, Nz = 8, 6, 4
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; comm=COMM, mesh=mesh, dtype=ComplexF64,
                           architecture=CPU(), use_pencil_arrays=false)
        bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny),
                 ComplexFourier(coords, "z", Nz))
        field = ScalarField(dist, "parity_3d_$(mesh)", bases)

        ox, oy = block_ranges(mesh, Nx, Ny)
        g = Tarang.get_grid_data(field)
        @test size(g) == (length(ox), length(oy), Nz)
        for k in 1:Nz, (jl, jg) in enumerate(oy), (il, ig) in enumerate(ox)
            g[il, jl, k] = complex(f3(ig, jg, k, Nx, Ny, Nz), 0.0)
        end
        original = copy(g)

        reference = serial_coeffs_3d(Nx, Ny, Nz)
        tf = TransposableField(field)
        distributed_forward_transform!(tf)

        c = Tarang.get_coeff_data(field)
        @test maximum(abs, c .- reference[ox, oy, :]; init=0.0) < 1e-10

        distributed_backward_transform!(tf)
        @test maximum(abs, Tarang.get_grid_data(field) .- original; init=0.0) < 1e-10
    end

    # Over-decomposition: ranks that own an EMPTY block must not hang and must
    # not corrupt the ranks that own data.
    if NP == 4
        @testset "over-decomposed Nx=2 on mesh=(4,1)" begin
            Nx, Ny = 2, 8
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; comm=COMM, mesh=(4, 1), dtype=ComplexF64,
                               architecture=CPU(), use_pencil_arrays=false)
            bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny))
            field = ScalarField(dist, "parity_overdecomp", bases)
            g = Tarang.get_grid_data(field)
            g .= complex(1.0, 0.0)
            tf = TransposableField(field)
            distributed_forward_transform!(tf)
            c = Tarang.get_coeff_data(field)
            # u ≡ 1 has all energy in the DC mode, owned by rank 0.
            if RANK == 0 && !isempty(c)
                @test isapprox(c[1, 1], complex(Nx * Ny, 0.0); rtol=1e-10)
            end
            MPI.Barrier(COMM)
            @test true  # reaching the barrier on every rank is the deadlock assertion
        end
    end
end

MPI.Barrier(COMM)
