using Test
using Tarang
using NetCDF

function simple_1d_setup()
    coords = CartesianCoordinates("z")
    dist = Distributor(coords; mesh=(1,), dtype=Float64)
    basis = ChebyshevT(coords["z"]; size=8, bounds=(0.0, 1.0))
    return coords, dist, basis
end

@testset "Problem construction and boundary conditions" begin
    coords, dist, basis = simple_1d_setup()
    u_lbvp = ScalarField(dist, "u", (basis,), Float64)

    lbvp = Tarang.LBVP([u_lbvp])
    # PDE equation
    Tarang.add_equation!(lbvp, "Δ(u) = 0")
    # Boundary conditions (Dedalus-style - auto-detected by add_equation!)
    Tarang.add_equation!(lbvp, "u(z=0) = 0")                           # Dirichlet
    Tarang.add_equation!(lbvp, "∂z(u)(z=1) = 1")                       # Neumann
    Tarang.add_equation!(lbvp, "1.0*u(z=0) + 2.0*∂z(u)(z=0) = 0.5")   # Robin
    Tarang.add_equation!(lbvp, "u(z=1) = 0")                           # Another Dirichlet

    @test Tarang.validate_problem(lbvp)
    @test length(lbvp.equations) >= 5

    u_nlbvp = ScalarField(dist, "u_nl", (basis,), Float64)
    nlbvp = Tarang.NLBVP([u_nlbvp])
    Tarang.add_equation!(nlbvp, "u_nl = 1 - u_nl")  # Use correct variable name
    Tarang.add_equation!(nlbvp, "u_nl(z=0) = 0")    # Dirichlet BC
    @test Tarang.validate_problem(nlbvp)

    u_evp = ScalarField(dist, "u_evp", (basis,), Float64)
    evp = Tarang.EVP([u_evp]; eigenvalue=:sigma)
    Tarang.add_equation!(evp, "sigma*u_evp = Δ(u_evp)")
    @test Tarang.validate_problem(evp)
end

@testset "Time steppers construct" begin
    steppers = [
        Tarang.RK111(), Tarang.RK222(), Tarang.RK443(),
        Tarang.CNAB1(), Tarang.CNAB2(),
        Tarang.SBDF1(), Tarang.SBDF2(), Tarang.SBDF3(), Tarang.SBDF4()
    ]
    @test all(ts -> ts isa Tarang.TimeStepper, steppers)
end

@testset "NetCDF metadata" begin
    tmp = mktempdir()
    cd(tmp) do
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2 * pi))
        u = ScalarField(dist, "u", (basis,), Float64)

        handler = Tarang.add_file_handler(joinpath(tmp, "snapshots"), dist, Dict("u" => u);
                                   parallel="gather", max_writes=1)
        Tarang.add_task(handler, u; name="u")

        Tarang.ensure_layout!(u, :g)
        fill!(Tarang.get_grid_data(u), 2.5)

        Tarang.process!(handler; iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.1)

        file = Tarang.current_file(handler)
        @test isfile(file)
        @test NetCDF.ncgetatt(file, "NC_GLOBAL", "title") == "Tarang.jl simulation output"
        @test NetCDF.ncgetatt(file, "NC_GLOBAL", "handler_name") == handler.name
        @test NetCDF.ncgetatt(file, "NC_GLOBAL", "tarang_version") == Tarang.__version__
        @test NetCDF.ncgetatt(file, "NC_GLOBAL", "software") == "Tarang"

        data = NetCDF.ncread(file, "u")
        @test all(isapprox.(data[1, :], 2.5))
    end
end

@testset "NetCDF analysis helpers" begin
    tmp = mktempdir()
    cd(tmp) do
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=4, bounds=(0.0, 1.0))
        u = ScalarField(dist, "u", (basis,), Float64)
        Tarang.ensure_layout!(u, :g)
        Tarang.get_grid_data(u) .= [1.0, 2.0, 3.0, 4.0]

        handler = Tarang.add_file_handler(joinpath(tmp, "analysis"), dist, Dict("u" => u);
                                   parallel="gather", max_writes=1)
        Tarang.add_task(handler, u; name="u_raw")
        Tarang.add_mean_task!(handler, u; dims=1, name="u_mean")
        Tarang.add_slice_task!(handler, u; dim=1, idx=2, name="u_slice")

        Tarang.process!(handler; iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.1)

        file = Tarang.current_file(handler)
        raw = NetCDF.ncread(file, "u_raw")
        mean_val = NetCDF.ncread(file, "u_mean")
        slice_val = NetCDF.ncread(file, "u_slice")

        @test all(isapprox.(raw[1, :], Tarang.get_grid_data(u)))
        @test isapprox(mean_val[1], 2.5; atol=1e-12)
        @test isapprox(slice_val[1], 2.0; atol=1e-12)
    end
end
