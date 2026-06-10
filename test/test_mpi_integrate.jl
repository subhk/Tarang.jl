using Test
using MPI
MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)

# Fill a (possibly distributed) grid array from a function of GLOBAL indices.
function fill_by_global!(field, f, N)
    ensure_layout!(field, :g)
    data = get_grid_data(field)
    if isa(data, PencilArrays.PencilArray)
        gv = PencilArrays.global_view(data)   # global indices, permutation-aware
        for I in CartesianIndices(gv)
            gv[I] = f(I[1], I[2])
        end
    else
        for gj in 1:N, gi in 1:N
            data[gi, gj] = f(gi, gj)
        end
    end
    return field
end

@testset "MPI integrate/average (np=$nprocs)" begin
    N = 16
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), Float64)

    @testset "full integral of constant" begin
        fill_by_global!(u, (gi, gj) -> 1.0, N)
        val = Tarang.evaluate_integrate(Tarang.Integrate(u, (coords["x"], coords["y"])), :g)
        @test val isa Real
        @test isapprox(val, (2π)^2; rtol=1e-12)
    end

    @testset "full integral of non-constant field" begin
        # 1 + cos(x): uniform Fourier quadrature integrates cos(x) to exactly 0
        v = ScalarField(dist, "v", (xb, yb), Float64)
        fill_by_global!(v, (gi, gj) -> 1.0 + cos(2π * (gi - 1) / N), N)
        val = Tarang.evaluate_integrate(Tarang.Integrate(v, (coords["x"], coords["y"])), :g)
        @test isapprox(val, (2π)^2; rtol=1e-12)
    end

    @testset "partial integration (serial and MPI)" begin
        w = ScalarField(dist, "w", (xb, yb), Float64)
        fill_by_global!(w, (gi, gj) -> 1.0 + cos(2π * (gi - 1) / N), N)

        # ∫ (1 + cos x) dx = 2π, independent of y
        r = Tarang.evaluate_integrate(Tarang.Integrate(w, coords["x"]), :g)
        rdata = get_grid_data(r)
        @test length(rdata) == N
        @test all(isapprox.(rdata, 2π; rtol=1e-12))

        # avg over y of (1 + cos x) = 1 + cos x at each x
        a = Tarang.evaluate_average(Tarang.Average(w, coords["y"]), :g)
        adata = get_grid_data(a)
        expected_x = 1.0 .+ cos.(2π .* (0:N-1) ./ N)
        @test all(isapprox.(adata, expected_x; atol=1e-10))
    end
end

MPI.Finalize()
