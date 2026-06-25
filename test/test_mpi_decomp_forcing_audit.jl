# Regression guards for the MPI CPU correctness audit (2026-06-23).
#
# Bug #1/#4 — get_local_range used the legacy column-major + remainder-on-FIRST
#   formula, diverging from the actual PencilArrays slab (row-major MPI-Cart
#   coords + remainder-on-LAST). It now mirrors compute_local_shape/local_indices
#   via pencil_local_range. Consumed by _fill_random_reproducible! (seeded fields
#   were decomposition-dependent) and grid-layout metadata.
#
# Bug #2 — _matched_forcing_view(::PencilArray) indexed axes_local (logical order)
#   by the PERMUTED storage position, double-applying the pencil permutation, so
#   StochasticForcing was injected at the wrong wavenumbers under MPI (or crashed
#   with DimensionMismatch on non-square local blocks). It now slices the global
#   spectrum in logical order then permutes to storage order to match the
#   consumer's parent-order broadcast.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)

@testset "MPI decomposition + forcing audit (np=$nprocs)" begin

    # --- Bug #1/#4: get_local_range matches the actual PencilArrays slab ---
    @testset "get_local_range == local_indices on a non-divisible decomposed axis" begin
        # N=5 with a 2D Fourier domain → axis 2 is decomposed; non-divisible by 2.
        N = 5
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        # The decomposed (trailing) axis: get_local_range must equal local_indices.
        glr = Tarang.get_local_range(dist, N, 2)
        ref = Tarang.local_indices(dist, 2, N)
        @test glr == (first(ref), last(ref))

        # The owned local lengths must tile the global axis exactly (no gap/overlap).
        local_len = last(ref) - first(ref) + 1
        @test MPI.Allreduce(local_len, +, comm) == N
        # Leading (non-decomposed) axis is full on every rank.
        @test Tarang.get_local_range(dist, N, 1) == (1, N)
    end

    # --- Bug #2: StochasticForcing lands at the correct global wavenumbers ---
    @testset "_matched_forcing_view places forcing at correct wavenumbers" begin
        N = 5
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = ComplexFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        f = ScalarField(dist, "u", (xb, yb), ComplexF64)
        Tarang.ensure_layout!(f, :c)
        cd = Tarang.get_coeff_data(f)

        # Known global spectrum in LOGICAL order encoding each wavenumber's coords.
        G = ComplexF64[i + 1000im*j for i in 1:N, j in 1:N]
        forcing = Tarang.StochasticForcing(; field_size=(N, N), k_forcing=2.0, dk_forcing=1.0)
        forcing.cached_forcing = copy(G)

        # Real consumer op: coeff_data .+= F_view (parent/memory-order broadcast).
        fill!(cd, 0)
        F_view = Tarang._matched_forcing_view(forcing, cd)
        @test F_view !== nothing
        cd .+= F_view

        # Gather to global logical order and compare to the intended spectrum.
        g = isa(cd, PencilArrays.PencilArray) ? PencilArrays.gather(cd) : copy(cd)
        if rank == 0
            @test maximum(abs.(ComplexF64.(g) .- G)) < 1e-10
        end
    end

    # --- Bug #5: NetCDF gather start/count attrs match the actual slab ---
    # The metadata wrote a remainder-on-FIRST + column-major decomposition while
    # the data is the PencilArrays slab (remainder-on-LAST). On a non-divisible
    # decomposed axis the RECONSTRUCT merge size-guard then dropped every slab and
    # NaN-filled the whole field. Attrs must now equal the real slab and tile it.
    @testset "NetCDF gather attrs == PencilArrays slab (no NaN merge)" begin
        N = 5
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)

        base = joinpath(tempdir(), "tarang_nc_audit_np$(nprocs)_$(rank == 0 ? getpid() : 0)")
        handler = Tarang.NetCDFFileHandler(base, dist, Dict{String,Any}(); iter=1, parallel="gather")
        Tarang.add_task!(handler, u; name="u")
        Tarang.init_mpi!(handler)
        _, lstart, lshape = Tarang.get_data_distribution(handler, handler.tasks[1])

        actual = size(Tarang.get_cpu_data(get_grid_data(u)))
        @test Tuple(lshape) == actual                       # count attr == data slab
        @test MPI.Allreduce(prod(lshape), +, comm) == N * N  # slabs tile the global field
        # start (0-based) must be the slab's global offset on the decomposed axis.
        @test Int(lstart[2]) == first(Tarang.local_indices(dist, 2, N)) - 1
    end

    # --- Bug #3: distributed grid-space resample fails loud, never corrupts ---
    @testset "set_scales! grid-space resize is rejected under MPI" begin
        N = 8
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords)
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb, yb), Float64)
        ensure_layout!(u, :g)

        if nprocs > 1
            # A per-rank local FFT would silently corrupt the decomposed Fourier
            # axis; the resize must raise instead.
            @test_throws ErrorException set_scales!(u, 1.5)
        else
            # Serial: the local slab is the whole grid, so the resample is exact.
            set_scales!(u, 1.5)
            @test get_scaled_shape(u, u.scales) == (12, 12)
        end
    end
end
