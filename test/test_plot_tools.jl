using Test
using Tarang

@testset "Plot data extraction" begin
    coords1 = CartesianCoordinates("x")
    dist1 = Distributor(coords1; mesh=(1,), dtype=Float64)
    domain1 = Tarang.create_fourier_domain(dist1, 2π, 8)
    field1 = ScalarField(dist1, "phi", domain1.bases, Float64)
    Tarang.ensure_layout!(field1, :g)
    for (idx, _) in enumerate(Tarang.get_grid_data(field1))
        Tarang.get_grid_data(field1)[idx] = idx
    end
    plot1 = Tarang.extract_plot_data(field1)
    @test plot1.y === nothing
    @test plot1.title == "phi"

    coords2 = CartesianCoordinates("x", "y")
    dist2 = Distributor(coords2; mesh=(1, 1), dtype=Float64)
    domain2 = Tarang.create_2d_periodic_domain(dist2, 2π, 2π, 4, 4)
    field2 = ScalarField(dist2, "psi", domain2.bases, Float64)
    Tarang.ensure_layout!(field2, :g)
    for (idx, _) in enumerate(Tarang.get_grid_data(field2))
        Tarang.get_grid_data(field2)[idx] = idx
    end
    plot2 = Tarang.extract_plot_data(field2)
    @test plot2.y !== nothing
    @test plot2.labels["x"] == domain2.bases[1].meta.element_label

    vec_field = VectorField(dist2, coords2, "U", domain2.bases, Float64)
    for comp in vec_field.components
        Tarang.ensure_layout!(comp, :g)
        fill!(Tarang.get_grid_data(comp), 1.0)
    end
    mag_field = Tarang.vector_magnitude(vec_field)
    Tarang.ensure_layout!(mag_field, :g)
    @test all(abs.(Tarang.get_grid_data(mag_field) .- sqrt(2.0)) .< 1e-12)
end
