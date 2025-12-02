using Test
using Tarang

@testset "Flow tools diagnostics" begin
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, device="cpu")
    basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))

    velocity = VectorField(dist, coords, "u", (basis,), Float64)
    component = velocity.components[1]
    Tarang.ensure_layout!(component, :g)
    fill!(component.data_g, 2.0)

    Re = Tarang.reynolds_number(velocity, 0.5, 1.0)
    @test isapprox(Re, 4.0; atol=1e-8)

    ke = Tarang.kinetic_energy(velocity, 2.0)
    Tarang.ensure_layout!(ke, :g)
    @test all(ke.data_g .≈ 4.0)

    reducer = Tarang.GlobalArrayReducer()
    @test Tarang.global_max(reducer, 3.5) ≈ 3.5
    @test Tarang.global_mean(reducer, ones(4)) ≈ 1.0
end
