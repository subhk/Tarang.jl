using Test
using Tarang

@testset "Plot data extraction" begin
    domain1 = PeriodicDomain(8)
    field1 = ScalarField(domain1, "phi")
    Tarang.ensure_layout!(field1, :g)
    for (idx, _) in enumerate(Tarang.get_grid_data(field1))
        Tarang.get_grid_data(field1)[idx] = idx
    end
    plot1 = Tarang.extract_plot_data(field1)
    @test plot1.y === nothing
    @test plot1.title == "phi"

    domain2 = PeriodicDomain(4, 4)
    field2 = ScalarField(domain2, "psi")
    Tarang.ensure_layout!(field2, :g)
    for (idx, _) in enumerate(Tarang.get_grid_data(field2))
        Tarang.get_grid_data(field2)[idx] = idx
    end
    plot2 = Tarang.extract_plot_data(field2)
    @test plot2.y !== nothing
    @test plot2.labels["x"] == domain2.bases[1].meta.element_label

    vec_field = VectorField(domain2, "U")
    for comp in vec_field.components
        Tarang.ensure_layout!(comp, :g)
        fill!(Tarang.get_grid_data(comp), 1.0)
    end
    mag_field = Tarang.vector_magnitude(vec_field)
    Tarang.ensure_layout!(mag_field, :g)
    @test all(abs.(Tarang.get_grid_data(mag_field) .- sqrt(2.0)) .< 1e-12)
end
