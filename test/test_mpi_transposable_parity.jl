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

            # u ≡ 1 has all energy in the DC mode. Derive who owns it from
            # block_ranges instead of hardcoding a rank: if a future refactor
            # moves ownership, a hardcoded RANK==0 guard would go false on
            # every rank and this testset would silently assert nothing.
            ox, _ = block_ranges((4, 1), Nx, Ny)
            owner = 1 in ox

            # Exactly one rank may own the DC mode. Reduced with the builtin
            # MPI.SUM op on a plain Int (not a Julia-function reduction over a
            # distributed array, which is unsupported here) so this check
            # itself cannot silently pass by going vacuous.
            owner_count = MPI.Allreduce(owner ? 1 : 0, MPI.SUM, COMM)
            @test owner_count == 1

            if owner
                @test isapprox(c[1, 1], complex(Nx * Ny, 0.0); rtol=1e-10)
            elseif !isempty(ox)
                # Non-empty block, not the owner: no energy should land here.
                @test maximum(abs, c; init=0.0) < 1e-10
            else
                # Over-decomposed: this rank's block is empty.
                @test isempty(c)
            end

            MPI.Barrier(COMM)
            @test true  # reaching the barrier on every rank is the deadlock assertion
        end
    end
end

MPI.Barrier(COMM)

# Task 13: prove that the is_transposable_storage dispatch branch added to the
# ordinary forward_transform!/backward_transform! in Task 12 actually EXECUTES,
# and produces correct values, without needing a GPU.
#
# The obvious approach — a JLArray field on a CPU Distributor — never reaches the
# branch at all: storage selection keys on is_gpu(dist.architecture), which is
# false for CPU() regardless of the array type, so a JLArray field there just
# gets ordinary SerialFieldStorage. And Tarang.GPU() cannot be constructed on
# this box (it raises CUDA installation guidance), so a genuinely GPU-
# architecture Distributor is unreachable here.
#
# Instead, build a field whose STORAGE is TransposableFieldStorage directly via
# ScalarField's explicit-storage inner constructor (dist, name, bases, dtype,
# storage) — bypassing the is_gpu(dist.architecture) && dist.size > 1 gate the
# ordinary constructor applies — while every underlying array stays a plain CPU
# Array and every FFT runs on FFTW. is_transposable_storage dispatches on the
# storage TYPE parameter, not on the `architecture` field carried inside it, so
# this legitimately takes the SAME branch a real GPU+MPI field would: through
# forward_transform!/backward_transform! -> transpose_workspace! ->
# distributed_forward_transform!/distributed_backward_transform!.
#
# Guarded to NP > 1 only: at one rank, distributed_forward_transform! delegates
# back to forward_transform!(tf.field) (see its `dist.size == 1` short-circuit).
# For an ordinary SerialFieldStorage field that delegation lands in a different
# branch, but tf.field here is itself transposable-storage, so that delegation
# would re-enter this same branch and recurse forever.
if NP > 1
    @testset "synthetic device-storage field dispatches through forward_transform!/backward_transform! (np=$NP)" begin
        Nx, Ny = 8, 6
        mesh = (NP, 1)
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; comm=COMM, mesh=mesh, dtype=ComplexF64,
                           architecture=CPU(), use_pencil_arrays=false)
        bases = (ComplexFourier(coords, "x", Nx), ComplexFourier(coords, "y", Ny))

        # An ordinary field on this Distributor gets SerialFieldStorage (architecture
        # is CPU, not GPU) with plain, per-rank block-shaped Arrays — exactly the grid
        # and coeff arrays a TransposableFieldStorage needs to wrap.
        host_field = ScalarField(dist, "device_dispatch_host", bases)
        ox, oy = block_ranges(mesh, Nx, Ny)
        g = Tarang.get_grid_data(host_field)
        @test size(g) == (length(ox), length(oy))
        for (jl, jg) in enumerate(oy), (il, ig) in enumerate(ox)
            g[il, jl] = complex(f2(ig, jg, Nx, Ny), 0.0)
        end
        original = copy(g)
        c = Tarang.get_coeff_data(host_field)

        # Wrap the SAME grid/coeff arrays in a TransposableFieldStorage. The
        # `architecture` recorded inside the storage is CPU() — it is the storage
        # TYPE (not this field, not this value) that routes
        # forward_transform!/backward_transform! to the distributed path.
        storage = Tarang.TransposableFieldStorage(CPU(), g, c)
        field = ScalarField(dist, "device_dispatch", bases, ComplexF64, storage)
        @test Tarang.is_transposable_storage(field) == true
        @test field.current_layout == :g

        forward_transform!(field)
        @test field.current_layout == :c
        reference = serial_coeffs_2d(Nx, Ny)
        @test maximum(abs, Tarang.get_coeff_data(field) .- reference[ox, oy]; init=0.0) < 1e-10

        backward_transform!(field)
        @test field.current_layout == :g
        @test maximum(abs, Tarang.get_grid_data(field) .- original; init=0.0) < 1e-10
    end
end

MPI.Barrier(COMM)
