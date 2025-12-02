using Test
using Tarang
using NetCDF

function simple_1d_setup()
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64, device="cpu")
    basis = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
    return coords, dist, basis
end

@testset "Problem construction and boundary conditions" begin
    coords, dist, basis = simple_1d_setup()
    u_lbvp = ScalarField(dist, "u", (basis,), Float64)

    lbvp = LBVP([u_lbvp])
    add_equation!(lbvp, "lap(u) = 0")
    add_dirichlet_bc!(lbvp, "u", "z", 0.0, 0.0)
    add_neumann_bc!(lbvp, "u", "z", 1.0, 1.0)
    add_robin_bc!(lbvp, "u", "z", 0.0, 1.0, 2.0, 0.5)
    add_stress_free_bc!(lbvp, "u", "z", 1.0)

    @test validate_problem(lbvp)
    @test length(lbvp.boundary_conditions) >= 4

    u_nlbvp = ScalarField(dist, "u_nl", (basis,), Float64)
    nlbvp = NLBVP([u_nlbvp])
    add_equation!(nlbvp, "u = 1 - u")
    add_dirichlet_bc!(nlbvp, "u_nl", "z", 0.0, 0.0)
    @test validate_problem(nlbvp)

    u_evp = ScalarField(dist, "u_evp", (basis,), Float64)
    evp = EVP([u_evp]; eigenvalue=:sigma)
    add_equation!(evp, "sigma*u_evp = lap(u_evp)")
    @test validate_problem(evp)
end

@testset "Time steppers construct" begin
    steppers = [
        RK111(), RK222(), RK443(),
        CNAB1(), CNAB2(),
        SBDF1(), SBDF2(), SBDF3(), SBDF4()
    ]
    @test all(ts -> ts isa Tarang.TimeStepper, steppers)
end

@testset "NetCDF metadata" begin
    tmp = mktempdir()
    cd(tmp) do
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64, device="cpu")
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2 * pi))
        u = ScalarField(dist, "u", (basis,), Float64)

        handler = add_file_handler(joinpath(tmp, "snapshots"), dist, Dict("u" => u);
                                   parallel="gather", max_writes=1)
        add_task(handler, u; name="u")

        ensure_layout!(u, :g)
        fill!(u.data_g, 2.5)

        process!(handler; iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.1)

        file = Tarang.current_file(handler)
        @test isfile(file)
        @test ncreadatt(file, "NC_GLOBAL", "title") == "Tarang.jl simulation output"
        @test ncreadatt(file, "NC_GLOBAL", "handler_name") == handler.name
        @test ncreadatt(file, "NC_GLOBAL", "tarang_version") == "0.1.0"
        @test ncreadatt(file, "NC_GLOBAL", "software") == "Tarang"

        data = ncread(file, "u")
        @test all(isapprox.(data[1, :], 2.5))
    end
end

@testset "NetCDF analysis helpers" begin
    tmp = mktempdir()
    cd(tmp) do
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64, device="cpu")
        basis = RealFourier(coords["x"]; size=4, bounds=(0.0, 1.0))
        u = ScalarField(dist, "u", (basis,), Float64)
        ensure_layout!(u, :g)
        u.data_g .= [1.0, 2.0, 3.0, 4.0]

        handler = add_file_handler(joinpath(tmp, "analysis"), dist, Dict("u" => u);
                                   parallel="gather", max_writes=1)
        add_task(handler, u; name="u_raw")
        add_mean_task!(handler, u; dims=1, name="u_mean")
        add_slice_task!(handler, u; dim=1, idx=2, name="u_slice")

        process!(handler; iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.1)

        file = Tarang.current_file(handler)
        raw = ncread(file, "u_raw")
        mean_val = ncread(file, "u_mean")
        slice_val = ncread(file, "u_slice")

        @test all(isapprox.(raw[1, :], u.data_g))
        @test isapprox(mean_val[1], 2.5; atol=1e-12)
        @test isapprox(slice_val[1], 2.0; atol=1e-12)
    end
end
