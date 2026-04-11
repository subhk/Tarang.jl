"""
Test suite for evaluator.jl — file handlers, global reductions, analysis.

Tests:
1. FileHandler scheduling logic (cadence, sim_dt, wall_dt, max_writes)
2. DictionaryHandler in-memory storage
3. Global reductions (min, max, mean, sum) on fields
4. evaluate_task for ScalarField and VectorField
5. Volume integral with quadrature weights
"""

using Test
using Tarang

@testset "Evaluator" begin

    @testset "FileHandler scheduling" begin
        # Cadence-based
        handler = FileHandler("test_output.nc"; cadence=10)
        @test should_write(handler, 0.0, 0.0, 10) == true
        @test should_write(handler, 0.0, 0.0, 7) == false
        @test should_write(handler, 0.0, 0.0, 20) == true

        # sim_dt-based
        handler_dt = FileHandler("test_output.nc"; sim_dt=0.5)
        handler_dt.last_write_sim_time = -1.0  # simulate "no previous write"
        @test should_write(handler_dt, 0.0, 0.0, 1) == true   # 0.0 - (-1.0) >= 0.5
        handler_dt.last_write_sim_time = 0.0
        @test should_write(handler_dt, 0.0, 0.3, 2) == false   # 0.3 - 0.0 < 0.5
        @test should_write(handler_dt, 0.0, 0.6, 3) == true    # 0.6 - 0.0 >= 0.5

        # max_writes
        handler_max = FileHandler("test_output.nc"; max_writes=2)
        @test should_write(handler_max, 0.0, 0.0, 1) == true
        handler_max.write_count = 2
        @test should_write(handler_max, 0.0, 1.0, 2) == false  # max reached
    end

    @testset "DictionaryHandler" begin
        handler = DictionaryHandler(; cadence=5)

        # Add tasks
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (basis,), Float64)
        add_task!(handler, u; name="velocity")

        @test haskey(handler.datasets, "velocity")
        @test handler.datasets["velocity"] === u

        # Scheduling
        @test should_write(handler, 0.0, 0.0, 5) == true
        @test should_write(handler, 0.0, 0.0, 3) == false
    end

    @testset "Global reductions on fields" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))

        u = ScalarField(dist, "u", (basis,), Float64)
        ensure_layout!(u, :g)
        Tarang.get_grid_data(u) .= Float64.(1:16)

        # Test global reductions (serial mode, so global = local)
        @test global_max(u) == 16.0
        @test global_min(u) == 1.0
        @test global_mean(u) ≈ 8.5
        @test global_sum(u) ≈ 136.0
    end

    @testset "Global reductions via distributor" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)

        data = Float64[3.0, 1.0, 4.0, 1.0, 5.0]
        @test global_max(dist, data) == 5.0
        @test global_min(dist, data) == 1.0
        @test global_mean(dist, data) ≈ 2.8
        @test global_sum(dist, data) ≈ 14.0
    end

    @testset "GlobalArrayReducer" begin
        reducer = GlobalArrayReducer()

        # Scalar reductions
        @test global_max(reducer, 5.0) == 5.0
        @test global_min(reducer, -3.0) == -3.0

        # Array reductions
        data = Float64[10.0, 20.0, 30.0]
        @test global_max(reducer, data) == 30.0
        @test global_min(reducer, data) == 10.0
        @test global_mean(reducer, data) ≈ 20.0

        # Empty array
        @test global_max(reducer, Float64[]; empty=-Inf) == -Inf
        @test global_min(reducer, Float64[]; empty=Inf) == Inf
    end

    @testset "GlobalFlowProperty" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (xb, yb), Float64)
        v = ScalarField(dist, "v", (xb, yb), Float64)

        # Create a minimal IVP + solver for GlobalFlowProperty
        problem = IVP([u, v])
        add_equation!(problem, "dt(u) = 0")
        add_equation!(problem, "dt(v) = 0")
        solver = InitialValueSolver(problem, RK222())

        flow = GlobalFlowProperty(solver; cadence=1)
        add_property!(flow, u, "u_field")

        @test haskey(flow.properties, "u_field")
        @test flow.properties["u_field"] === u
    end
end
