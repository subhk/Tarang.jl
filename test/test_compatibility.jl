using Test
using Tarang
using NetCDF
using TOML

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

@testset "Exported transform setup functions are defined" begin
    @test isdefined(Tarang, :setup_pencil_fft_transforms_2d!)
    @test isdefined(Tarang, :setup_pencil_fft_transforms_3d!)
end

@testset "NetCDF metadata" begin
    tmp = mktempdir()
    cd(tmp) do
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2 * pi))
        domain = Domain(dist, (basis,))
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
        @test NetCDF.ncgetatt(file, "NC_GLOBAL", "tarang_version") == string(pkgversion(Tarang))
        @test NetCDF.ncgetatt(file, "NC_GLOBAL", "software") == "Tarang"

        data = Tarang.group_ncread(file, "vars", "u")
        @test all(isapprox.(data[1, :], 2.5))
        @test Tarang.group_ncread(file, "grids", "x") ≈ Tarang.get_grid_coordinates(domain; on_device=false)["x"]
        @test Tarang.group_ncread(file, "time", "sim_time") == [0.0]
        @test Tarang.group_ncread(file, "time", "timestep") == [0.1]

        folder_handler = Tarang.add_file_handler("sqg", dist, Dict("u" => u);
                                                 parallel="gather", max_writes=1)
        Tarang.add_task!(folder_handler, u; name="u")
        Tarang.process!(folder_handler; iteration=0, sim_time=0.0, timestep=0.1)

        folder_file = Tarang.current_file(folder_handler)
        @test folder_file == joinpath("sqg", "sqg_s1", "sqg_s1.nc")
        @test isfile(folder_file)
        @test Tarang.get_output_files(folder_handler) == [folder_file]
    end
end

@testset "NetCDF merge reconstructs distributed coordinates" begin
    tmp = mktempdir()
    cd(tmp) do
        setdir = joinpath("snapshots_s1")
        mkpath(setdir)

        for rank in 0:1
            file = joinpath(setdir, "snapshots_s1_p$(rank).nc")

            nccreate(file, "sim_time", "sim_time", 2, t=Float64)
            ncwrite([0.0, 1.0], file, "sim_time")

            nccreate(file, "q_dim1", "q_dim1", 4, t=Float64)
            ncwrite(collect(1.0:4.0), file, "q_dim1")

            nccreate(file, "q_dim2", "q_dim2", 2, t=Float64)
            ncwrite(collect(1.0:2.0) .+ 2rank, file, "q_dim2")

            nccreate(file, "q", "sim_time", 2, "q_dim1", 4, "q_dim2", 2;
                     t=Float64,
                     atts=Dict(
                         "global_shape" => [4, 4],
                         "start" => [0, 2rank],
                         "count" => [4, 2],
                         "grid_space" => 1,
                         "layout" => "g",
                     ))
            ncwrite(fill(float(rank), 2, 4, 2), file, "q")
        end

        @test Tarang.merge_netcdf_files("snapshots"; cleanup=false, verbose=false)

        merged = joinpath(setdir, "snapshots_s1.nc")
        @test isfile(merged)
        @test size(Tarang.group_ncread(merged, "vars", "q")) == (2, 4, 4)
        @test Tarang.group_ncread(merged, "grids", "q_dim2") == collect(1.0:4.0)

        legacy_setdir = joinpath("legacy_s1")
        mkpath(legacy_setdir)
        for rank in 0:1
            file = joinpath(legacy_setdir, "legacy_s1_p$(rank).nc")

            nccreate(file, "sim_time", "sim_time", 1, t=Float64)
            ncwrite([0.0], file, "sim_time")

            nccreate(file, "x", "x", 4, t=Float64)
            ncwrite(collect(1.0:4.0), file, "x")

            nccreate(file, "y", "y", 2, t=Float64)
            ncwrite(collect(1.0:2.0) .+ 2rank, file, "y")

            nccreate(file, "q", "sim_time", 1, "x", 4, "y", 2;
                     t=Float64,
                     atts=Dict(
                         "global_shape" => [4, 4],
                         "start" => [0, 2rank],
                         "count" => [4, 2],
                         "grid_space" => 1,
                         "layout" => "g",
                     ))
            ncwrite(fill(float(rank), 1, 4, 2), file, "q")
        end

        @test Tarang.merge_netcdf_files("legacy"; cleanup=false, verbose=false)
        legacy_merged = joinpath(legacy_setdir, "legacy_s1.nc")
        @test Tarang.group_ncread(legacy_merged, "grids", "x") == collect(1.0:4.0)
        @test Tarang.group_ncread(legacy_merged, "grids", "y") == collect(1.0:4.0)
        @test size(Tarang.group_ncread(legacy_merged, "vars", "q")) == (1, 4, 4)
        @test !Tarang.group_var_exists(legacy_merged, "vars", "x")
        @test !Tarang.group_var_exists(legacy_merged, "vars", "y")

        grouped_setdir = joinpath("grouped_s1")
        mkpath(grouped_setdir)
        for rank in 0:1
            file = joinpath(grouped_setdir, "grouped_s1_p$(rank).nc")
            Tarang.create_empty_netcdf4_file!(file)

            Tarang.group_nccreate(file, "time", "sim_time", "sim_time", 2; t=Float64)
            Tarang.group_ncwrite([0.0, 1.0], file, "time", "sim_time")

            Tarang.group_nccreate(file, "grids", "q_dim1", "q_dim1", 4; t=Float64)
            Tarang.group_ncwrite(collect(1.0:4.0), file, "grids", "q_dim1")
            Tarang.group_nccreate(file, "grids", "q_dim2", "q_dim2", 2; t=Float64)
            Tarang.group_ncwrite(collect(1.0:2.0) .+ 2rank, file, "grids", "q_dim2")

            Tarang.group_nccreate(file, "vars", "q", "sim_time", 2, "q_dim1", 4, "q_dim2", 2;
                                  t=Float64,
                                  atts=Dict(
                                      "global_shape" => [4, 4],
                                      "start" => [0, 2rank],
                                      "count" => [4, 2],
                                      "grid_space" => 1,
                                      "layout" => "g",
                                  ))
            Tarang.group_ncwrite(fill(float(rank), 2, 4, 2), file, "vars", "q")
        end

        @test Tarang.merge_netcdf_files("grouped"; cleanup=false, verbose=false)
        grouped_merged = joinpath(grouped_setdir, "grouped_s1.nc")
        @test size(Tarang.group_ncread(grouped_merged, "vars", "q")) == (2, 4, 4)
        @test Tarang.group_ncread(grouped_merged, "grids", "q_dim2") == collect(1.0:4.0)
    end
end

@testset "NetCDF direct process scheduling" begin
    tmp = mktempdir()
    cd(tmp) do
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=4, bounds=(0.0, 1.0))
        u = ScalarField(dist, "u", (basis,), Float64)
        fill!(Tarang.get_grid_data(u), 1.0)

        handler = Tarang.add_file_handler(joinpath(tmp, "scheduled"), dist, Dict("u" => u);
                                   parallel="gather", sim_dt=1.0, max_writes=50)
        Tarang.add_task(handler, u; name="u")

        @test Tarang.process!(handler; iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.1)
        @test !Tarang.process!(handler; iteration=1, wall_time=0.1, sim_time=0.1, timestep=0.1)
        @test !Tarang.process!(handler; iteration=2, wall_time=0.2, sim_time=0.2, timestep=0.1)
        @test Tarang.process!(handler; iteration=10, wall_time=1.0, sim_time=1.0, timestep=0.1)

        @test handler.total_write_num == 2
        @test handler.file_write_num == 2
        @test isdir(joinpath(tmp, "scheduled_s1"))
        @test !isdir(joinpath(tmp, "scheduled_s2"))
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
        raw = Tarang.group_ncread(file, "vars", "u_raw")
        mean_val = Tarang.group_ncread(file, "vars", "u_mean")
        slice_val = Tarang.group_ncread(file, "vars", "u_slice")

        @test all(isapprox.(raw[1, :], Tarang.get_grid_data(u)))
        @test isapprox(mean_val[1], 2.5; atol=1e-12)
        @test isapprox(slice_val[1], 2.0; atol=1e-12)
    end
end

@testset "Forced 2D turbulence example controls" begin
    example = read(joinpath(dirname(@__DIR__), "examples", "ivp", "forced_2d_turbulence.jl"), String)

    @test occursin("TARANG_FORCED_2D_NX", example)
    @test occursin("TARANG_FORCED_2D_STOP_TIME", example)
    @test occursin("TARANG_FORCED_2D_MAX_DT", example)
    @test occursin("TARANG_FORCED_2D_OUTPUT_DT", example)
    @test occursin("TARANG_FORCED_2D_INITIAL_NOISE", example)
    @test occursin("TARANG_FORCED_2D_LOG_INTERVAL", example)
    @test occursin(r"architecture\s*=\s*device", example)
    @test occursin("CUDA.allowscalar(false)", example)
    @test occursin("GPU(device_id=gpu_id)", example)
    @test occursin("∂t(ζ) = -u⋅∇(ζ) - drag*ζ - nu*Δ⁴(ζ)", example)
    @test !occursin("∂t(ζ) + drag*ζ", example)
    @test !occursin("refresh_streamfunction_from_vorticity!", example)
    @test occursin("add_file_handler(output_path, solver;", example)
    @test occursin("run!(solver", example)
    @test !occursin("process!(snapshots", example)
    @test !occursin("close!(snapshots", example)
end

@testset "NetCDF evaluates operator tasks" begin
    tmp = mktempdir()
    cd(tmp) do
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2pi))
        domain = Domain(dist, (basis,))
        u = ScalarField(domain, "u")
        x = Tarang.get_grid_coordinates(domain; on_device=false)["x"]
        Tarang.ensure_layout!(u, :g)
        Tarang.get_grid_data(u) .= sin.(x)

        handler = Tarang.add_file_handler(joinpath(tmp, "operators"), dist, Dict("u" => u);
                                          parallel="gather", max_writes=1)
        Tarang.add_task!(handler, Tarang.Δ(u); name="lap_u")
        Tarang.process!(handler; iteration=0, wall_time=0.0, sim_time=0.0, timestep=0.1)

        data = Tarang.group_ncread(Tarang.current_file(handler), "vars", "lap_u")
        @test data[1, :] ≈ -sin.(x) atol=1e-8
    end
end

@testset "NetCDF solver-backed handlers refresh diagnostics" begin
    tmp = mktempdir()
    cd(tmp) do
        Nx, Ny = 8, 8
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2pi), dealias=3/2)
        ybasis = RealFourier(coords["y"]; size=Ny, bounds=(0.0, 2pi), dealias=3/2)
        domain = Domain(dist, (xbasis, ybasis))

        ζ = ScalarField(domain, "ζ")
        ψ = ScalarField(domain, "ψ")
        u = VectorField(domain, "u")
        tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)

        problem = IVP([ζ, ψ, u, tau_ψ])
        add_parameters!(problem, nu=1e-6, drag=1e-3)
        add_equation!(problem, "∂t(ζ) + drag*ζ + nu*Δ⁴(ζ) = -u⋅∇(ζ)")
        add_equation!(problem, "Δ(ψ) + tau_ψ - ζ = 0")
        add_equation!(problem, "u - skew(grad(ψ)) = 0")
        add_bc!(problem, "integ(ψ) = 0")
        solver = InitialValueSolver(problem, RK222(); dt=1e-3)

        fill_random!(ζ, "g"; seed=42, distribution="normal", scale=1e-2)
        solver.iteration = 3
        solver.sim_time = 0.125
        solver.dt = 0.125

        handler = Tarang.add_file_handler(joinpath(tmp, "diagnostics"), solver,
                                          Dict("ζ" => ζ, "ψ" => ψ);
                                          parallel="gather", max_writes=1)
        Tarang.add_task!(handler, ψ; name="ψ")
        Tarang.process!(handler)

        file = Tarang.current_file(handler)
        ψ_data = Tarang.group_ncread(file, "vars", "ψ")
        @test maximum(abs, ψ_data[1, :, :]) > 0
        @test Tarang.group_ncread(file, "grids", "x") ≈ Tarang.get_grid_coordinates(domain; on_device=false)["x"]
        @test Tarang.group_ncread(file, "grids", "y") ≈ Tarang.get_grid_coordinates(domain; on_device=false)["y"]
        @test Tarang.group_ncread(file, "time", "iteration")[1] == solver.iteration
        @test Tarang.group_ncread(file, "time", "sim_time")[1] == solver.sim_time
        @test Tarang.group_ncread(file, "time", "timestep")[1] == solver.dt

        explicit_handler = Tarang.add_file_handler(joinpath(tmp, "explicit_diagnostics"), dist,
                                                   Dict("ζ" => ζ, "ψ" => ψ);
                                                   parallel="gather", max_writes=1)
        Tarang.add_task!(explicit_handler, ψ; name="ψ")
        Tarang.process!(explicit_handler, solver; wall_time=0.0)

        explicit_file = Tarang.current_file(explicit_handler)
        @test Tarang.group_ncread(explicit_file, "time", "iteration")[1] == solver.iteration
        @test Tarang.group_ncread(explicit_file, "time", "sim_time")[1] == solver.sim_time
        @test Tarang.group_ncread(explicit_file, "time", "timestep")[1] == solver.dt
    end
end

@testset "NetCDF writes do not mutate refreshed 3D constraints" begin
    tmp = mktempdir()
    cd(tmp) do
        N = 6
        dt = 2e-3
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; dtype=Float64, device=CPU())
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2pi), dealias=1.0)
        yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2pi), dealias=1.0)
        zb = RealFourier(coords["z"]; size=N, bounds=(0.0, 2pi), dealias=1.0)
        domain = Domain(dist, (xb, yb, zb))

        w = ntuple(i -> ScalarField(domain, "w$i"), 3)
        A = ntuple(i -> ScalarField(domain, "A$i"), 3)
        u = ntuple(i -> ScalarField(domain, "u$i"), 3)
        tau = ntuple(i -> ScalarField(dist, "tau_A$i", (), Float64), 3)

        problem = IVP([w..., A..., u..., tau...])
        add_parameters!(problem; nu=0.01)
        for i in 1:3
            add_equation!(problem, "∂t(w$i) - nu*Δ(w$i) = 0")
            add_equation!(problem, "Δ(A$i) + tau_A$i + w$i = 0")
        end
        add_equation!(problem, "u1 - (∂y(A3) - ∂z(A2)) = 0")
        add_equation!(problem, "u2 - (∂z(A1) - ∂x(A3)) = 0")
        add_equation!(problem, "u3 - (∂x(A2) - ∂y(A1)) = 0")
        for i in 1:3
            add_bc!(problem, "integ(A$i) = 0")
        end

        x, y, z = local_grids(dist, xb, yb, zb)
        Z = reshape(z, 1, 1, :)
        for field in w
            ensure_layout!(field, :g)
        end
        get_grid_data(w[1]) .= -cos.(x) .* sin.(y') .* sin.(Z)
        get_grid_data(w[2]) .= -sin.(x) .* cos.(y') .* sin.(Z)
        get_grid_data(w[3]) .= 2 .* sin.(x) .* sin.(y') .* cos.(Z)

        solver = InitialValueSolver(problem, RK222(); dt)
        step!(solver, dt)
        ensure_layout!(w[1], :g)
        written_field_before = copy(get_grid_data(w[1]))
        for field in u
            ensure_layout!(field, :g)
        end
        velocity_before = map(field -> copy(get_grid_data(field)), u)

        handler = add_file_handler(joinpath(tmp, "observational"), solver;
                                   iter=1, max_writes=1)
        add_task!(handler, w[1]; name="w1")
        process!(handler)

        written_field = dropdims(
            Tarang.group_ncread(Tarang.current_file(handler), "vars", "w1"); dims=1)
        @test written_field ≈ written_field_before rtol=1e-12 atol=1e-12

        for field in u
            ensure_layout!(field, :g)
        end
        for i in 1:3
            @test get_grid_data(u[i]) ≈ velocity_before[i] rtol=1e-12 atol=1e-12
        end
    end
end

@testset "Forced SQG turbulence example controls" begin
    example = read(joinpath(dirname(@__DIR__), "examples", "ivp", "forced_sqg_turbulence.jl"), String)

    @test !occursin("TARANG_FORCED_SQG", example)
    @test occursin("Nx, Ny   = 512, 512", example)
    @test occursin("output_dt = 1.0", example)
    @test occursin("initial_noise = 1e-3", example)
    @test occursin("log_interval = 100", example)
    @test occursin("output_path = \"sqg\"", example)
    @test occursin("ψ - invsqrtlap(θ) = 0", example)
    @test occursin("add_file_handler(output_path, solver;", example)
    @test occursin("add_task!(snapshots, θ; name=\"theta\")", example)
    @test occursin("run!(solver", example)
    @test !occursin("process!(snapshots", example)
    @test !occursin("close!(snapshots", example)
end

@testset "Pluto snapshot notebook spectra" begin
    notebook = read(joinpath(dirname(@__DIR__), "scripts", "visualize_snapshots_pluto.jl"), String)

    @test occursin("using Tarang", notebook)
    @test occursin("Tarang.group_ncread", notebook)
    @test occursin("x_name = \"x\"", notebook)
    @test occursin("y_name = \"y\"", notebook)
    @test occursin("using FFTW", notebook)
    @test occursin("streamfunction_var", notebook)
    @test occursin("angular_wavenumbers", notebook)
    @test occursin("kinetic_energy_spectrum_from_streamfunction", notebook)
    @test occursin("0.5 * (kx[i]^2 + ky[j]^2) * abs2(ψ̂[i, j])", notebook)
    @test occursin("xscale=log10", notebook)
    @test occursin("yscale=log10", notebook)
end

@testset "Script package boundaries" begin
    project = TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))
    deps = get(project, "deps", Dict{String, Any}())
    @test haskey(deps, "ArgParse")

    merge_script = read(joinpath(dirname(@__DIR__), "scripts", "merge_netcdf.jl"), String)
    @test occursin("Pkg.activate", merge_script)
    @test occursin(r"(?m)^\s*using\s+Tarang\b", merge_script)
    @test occursin("pkgversion(Tarang)", merge_script)
    @test !occursin("Tarang.__version__", merge_script)
    @test !occursin(r"include\(joinpath\(dirname\(@__DIR__\), \"src\", \"tools\", \"netcdf_merge\.jl\"\)\)", merge_script)

    @test occursin("discover_handler_sets", merge_script)
    @test occursin("for (set_number, workdir) in discover_handler_sets", merge_script)
    @test occursin("normalize_handler_arg", merge_script)
    @test occursin("normalize_handler_arg.(args[\"handlers\"])", merge_script)
    @test occursin("last(splitpath(normalized))", merge_script)
end
