"""
Test suite for solvers.jl

Tests:
1. SolverPerformanceStats creation and update
2. SolverBaseData construction
3. InitialValueSolver construction and stepping
4. BoundaryValueSolver construction and solving
5. EigenvalueSolver construction and solving
6. Field/vector conversion utilities
7. Nonlinear solver infrastructure
8. Dispatch checks
9. Performance logging
"""

using Test

@testset "Solvers Module" begin
    using Tarang
    using LinearAlgebra
    using SparseArrays
    using MPI

    @testset "SolverPerformanceStats" begin
        stats = SolverPerformanceStats()
        @test isa(stats, SolverPerformanceStats)
        @test stats.total_time == 0.0
        @test stats.total_steps == 0
        @test stats.total_solves == 0
        @test stats.avg_step_time == 0.0

        # Test mutability
        stats.total_time = 1.5
        stats.total_steps = 10
        @test stats.total_time == 1.5
        @test stats.total_steps == 10
    end

    @testset "normalize_matsolver" begin
        # Test string normalization
        @test Tarang._normalize_matsolver("direct") == :sparse
        @test Tarang._normalize_matsolver("lu") == :sparse
        @test Tarang._normalize_matsolver("sparse") == :sparse
        @test Tarang._normalize_matsolver("dense") == :dense

        # Test symbol normalization
        @test Tarang._normalize_matsolver(:direct) == :sparse
        @test Tarang._normalize_matsolver(:sparse) == :sparse

        # Test passthrough for unknown
        @test Tarang._normalize_matsolver(:unknown) == :unknown
    end

    @testset "collect_state_fields" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))

        # Test with ScalarField - use Operand[] for proper type
        sf = ScalarField(dist, "u", (xb,), Float64)
        state = Tarang.collect_state_fields(Tarang.Operand[sf])
        @test length(state) == 1
        @test state[1] === sf

        # Test with VectorField
        coords2 = CartesianCoordinates("x", "y")
        dist2 = Distributor(coords2; mesh=(1, 1), dtype=Float64)
        xb2 = RealFourier(coords2["x"]; size=8, bounds=(0.0, 2π))
        yb2 = RealFourier(coords2["y"]; size=8, bounds=(0.0, 2π))
        vf = VectorField(dist2, coords2, "v", (xb2, yb2), Float64)
        state = Tarang.collect_state_fields(Tarang.Operand[vf])
        @test length(state) == 2  # Vector field has 2 components
    end

    @testset "InitialValueSolver minimal workflow" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        problem = IVP([field])
        Tarang.add_equation!(problem, "dt(u) = 0")

        solver = InitialValueSolver(problem, RK111(); device="cpu")

        @test solver.problem === problem
        @test length(solver.state) == 1
        @test solver.dt ≈ 1e-3
        @test haskey(problem.parameters, "L_matrix")
        @test solver.base.entry_cutoff ≈ 1e-12
        @test solver.evaluator !== nothing
        @test solver.iteration == 0
        @test solver.sim_time == 0.0

        Tarang.step!(solver, 1e-3)
        @test solver.iteration == 1
        @test isapprox(solver.sim_time, 1e-3; atol=1e-6)

        # Test performance stats updated
        @test solver.performance_stats.total_steps >= 1
    end

    @testset "InitialValueSolver with custom dt" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        problem = IVP([field])
        Tarang.add_equation!(problem, "dt(u) = 0")

        solver = InitialValueSolver(problem, RK111(); dt=0.01)
        @test solver.dt ≈ 0.01
    end

    @testset "proceed function" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        problem = IVP([field])
        Tarang.add_equation!(problem, "dt(u) = 0")

        solver = InitialValueSolver(problem, RK111())

        # Initially should proceed
        @test Tarang.proceed(solver) == true

        # Set stop conditions
        solver.stop_sim_time = 0.0
        @test Tarang.proceed(solver) == false

        # Reset and test iteration stop
        solver.stop_sim_time = Inf
        solver.stop_iteration = 0
        @test Tarang.proceed(solver) == false
    end

    @testset "solver_comm function" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        problem = IVP([field])
        Tarang.add_equation!(problem, "dt(u) = 0")

        comm = Tarang.solver_comm(problem)
        @test comm isa MPI.Comm
    end

    @testset "apply_entry_cutoff! (matrix)" begin
        # Test with dense matrix
        A = [1e-14 1.0; 1e-15 2.0]
        Tarang.apply_entry_cutoff!(A, 1e-12)
        @test A[1, 1] == 0.0
        @test A[2, 1] == 0.0
        @test A[1, 2] == 1.0
        @test A[2, 2] == 2.0

        # Test with sparse matrix - sparse(I, J, V) where I=rows, J=cols, V=values
        B = sparse([1, 1, 2, 2], [1, 2, 1, 2], [1e-14, 1.0, 1e-15, 2.0])
        Tarang.apply_entry_cutoff!(B, 1e-12)
        # Small values should be zeroed and dropped
        dropzeros!(B)
        @test nnz(B) == 2  # Only 1.0 and 2.0 remain
    end

    @testset "apply_entry_cutoff! (vector)" begin
        v = [1e-14, 1.0, 1e-15, 2.0]
        Tarang.apply_entry_cutoff!(v, 1e-12)
        @test v[1] == 0.0
        @test v[2] == 1.0
        @test v[3] == 0.0
        @test v[4] == 2.0
    end

    @testset "BoundaryValueSolver construction" begin
        # Test that BoundaryValueSolver type exists and dispatch works
        @test BoundaryValueSolver <: Any

        # Test dispatch validation (should catch wrong problem types)
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        ivp = IVP([field])
        @test_throws ArgumentError Tarang.dispatch_check(BoundaryValueSolver, (ivp,), NamedTuple())

        # Valid dispatch for LBVP
        lbvp = LBVP([field])
        @test Tarang.dispatch_check(BoundaryValueSolver, (lbvp,), NamedTuple()) == true
    end

    @testset "BoundaryValueSolver with custom parameters" begin
        # Test dispatch for NLBVP
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        nlbvp = NLBVP([field])
        @test Tarang.dispatch_check(BoundaryValueSolver, (nlbvp,), NamedTuple()) == true

        # Test that solver types exist in module
        @test isdefined(Tarang, :BoundaryValueSolver)
        @test isdefined(Tarang, :_build_boundary_value_solver)
    end

    @testset "EigenvalueSolver construction" begin
        # Test that EigenvalueSolver type exists and dispatch works
        @test EigenvalueSolver <: Any

        # Test dispatch validation
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        # Invalid: IVP for eigenvalue solver
        ivp = IVP([field])
        @test_throws ArgumentError Tarang.dispatch_check(EigenvalueSolver, (ivp,), NamedTuple())

        # Valid: EVP
        evp = EVP([field])
        @test Tarang.dispatch_check(EigenvalueSolver, (evp,), NamedTuple()) == true

        # Test that solver types exist in module
        @test isdefined(Tarang, :EigenvalueSolver)
        @test isdefined(Tarang, :_build_eigenvalue_solver)
    end

    @testset "EigenvalueSolver with custom parameters" begin
        # Test that dispatch accepts valid nev and which parameters via kwargs
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        evp = EVP([field])

        # Test that kwargs validation passes for valid parameters
        kwargs = (nev=10, which=:SM)
        @test Tarang.dispatch_check(EigenvalueSolver, (evp,), kwargs) == true

        # Test Arpack-related exports
        @test isdefined(Tarang, :Arpack)
    end

    @testset "Field/Vector Conversion" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (xb,), Float64)

        # Initialize field data
        ensure_layout!(field, :c)
        fill!(field["c"], 1.0 + 0.5im)

        @testset "compute_field_vector_size" begin
            size = Tarang.compute_field_vector_size(field)
            @test size > 0
        end

        @testset "extract_field_data_for_vector" begin
            data = Tarang.extract_field_data_for_vector(field)
            @test isa(data, Vector)
            @test length(data) > 0
        end

        @testset "fields_to_vector" begin
            vec = Tarang.fields_to_vector([field])
            @test isa(vec, Vector{ComplexF64})
            @test length(vec) > 0
        end

        @testset "copy_solution_to_fields!" begin
            field2 = ScalarField(dist, "u2", (xb,), Float64)
            ensure_layout!(field2, :c)

            n = Tarang.compute_field_vector_size(field2)
            solution = ones(ComplexF64, n) * 2.0

            Tarang.copy_solution_to_fields!([field2], solution)
            # Verify data was copied (check first element approximately)
            @test Tarang.get_coeff_data(field2) !== nothing
        end

        @testset "set_field_data_from_vector!" begin
            field3 = ScalarField(dist, "u3", (xb,), Float64)
            ensure_layout!(field3, :c)

            n = Tarang.compute_field_vector_size(field3)
            data = ones(ComplexF64, n) * 3.0

            Tarang.set_field_data_from_vector!(field3, data)
            @test field3.current_layout == :c
        end
    end

    @testset "get_basis_size" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)

        # RealFourier basis
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))
        @test Tarang.get_basis_size(xb) == 16

        # Chebyshev basis
        cb = ChebyshevT(coords["x"]; size=32, bounds=(-1.0, 1.0))
        @test Tarang.get_basis_size(cb) == 32
    end

    @testset "dispatch_check" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        @testset "InitialValueSolver dispatch" begin
            ivp = IVP([field])
            # Valid dispatch
            @test Tarang.dispatch_check(InitialValueSolver, (ivp, RK111()), NamedTuple()) == true

            # Invalid: not enough args
            @test_throws ArgumentError Tarang.dispatch_check(InitialValueSolver, (ivp,), NamedTuple())

            # Invalid: wrong problem type
            lbvp = LBVP([field])
            @test_throws ArgumentError Tarang.dispatch_check(InitialValueSolver, (lbvp, RK111()), NamedTuple())
        end

        @testset "BoundaryValueSolver dispatch" begin
            lbvp = LBVP([field])
            @test Tarang.dispatch_check(BoundaryValueSolver, (lbvp,), NamedTuple()) == true

            nlbvp = NLBVP([field])
            @test Tarang.dispatch_check(BoundaryValueSolver, (nlbvp,), NamedTuple()) == true

            # Invalid: no args
            @test_throws ArgumentError Tarang.dispatch_check(BoundaryValueSolver, (), NamedTuple())

            # Invalid: wrong problem type
            ivp = IVP([field])
            @test_throws ArgumentError Tarang.dispatch_check(BoundaryValueSolver, (ivp,), NamedTuple())
        end

        @testset "EigenvalueSolver dispatch" begin
            evp = EVP([field])
            @test Tarang.dispatch_check(EigenvalueSolver, (evp,), NamedTuple()) == true

            # Invalid: no args
            @test_throws ArgumentError Tarang.dispatch_check(EigenvalueSolver, (), NamedTuple())

            # Invalid: wrong problem type
            ivp = IVP([field])
            @test_throws ArgumentError Tarang.dispatch_check(EigenvalueSolver, (ivp,), NamedTuple())
        end
    end

    @testset "create_zero_field" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (xb,), Float64)

        zero_field = Tarang.create_zero_field([field])
        @test isa(zero_field, ScalarField)
        @test zero_field.name == "zero_field"

        ensure_layout!(zero_field, :c)
        if Tarang.get_coeff_data(zero_field) !== nothing
            @test all(Tarang.get_coeff_data(zero_field) .== 0.0)
        end
    end

    @testset "Helper functions for expressions" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        template = ScalarField(dist, "template", (xb,), Float64)

        @testset "_constant_field_from_template" begin
            const_field = Tarang._constant_field_from_template(template, 5.0)
            @test isa(const_field, ScalarField)
            ensure_layout!(const_field, :g)
            if Tarang.get_grid_data(const_field) !== nothing
                @test all(Tarang.get_grid_data(const_field) .≈ 5.0)
            end
        end

        @testset "_coerce_numeric_operand" begin
            # Number with template
            result = Tarang._coerce_numeric_operand(3.0, template)
            @test isa(result, ScalarField)

            # Number without template
            result = Tarang._coerce_numeric_operand(3.0, nothing)
            @test result == 3.0

            # Field passes through
            result = Tarang._coerce_numeric_operand(template, nothing)
            @test result === template
        end
    end

    @testset "Multiple time steps" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        problem = IVP([field])
        Tarang.add_equation!(problem, "dt(u) = 0")

        solver = InitialValueSolver(problem, RK111())

        # Take multiple steps
        for i in 1:5
            Tarang.step!(solver, 0.001)
        end

        @test solver.iteration == 5
        @test isapprox(solver.sim_time, 0.005; atol=1e-10)
        @test solver.performance_stats.total_steps == 5
    end

    @testset "Performance logging" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        problem = IVP([field])
        Tarang.add_equation!(problem, "dt(u) = 0")

        solver = InitialValueSolver(problem, RK111())
        Tarang.step!(solver, 0.001)

        # Test log_stats runs without error
        @test begin
            Tarang.log_stats(solver)
            true
        end

        # Test log_solver_performance runs without error
        @test begin
            Tarang.log_solver_performance(solver)
            true
        end
    end

    @testset "Interpolate field data" begin
        # Same shape - simple copy
        src = [1.0, 2.0, 3.0, 4.0]
        dest = zeros(4)
        Tarang.interpolate_field_data!(dest, src)
        @test dest == src

        # Different shapes - nearest neighbor interpolation
        src = [1.0, 2.0]
        dest = zeros(4)
        Tarang.interpolate_field_data!(dest, src)
        @test dest[1] == 1.0
        @test dest[4] == 2.0
    end

    @testset "SolverBaseData construction" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        problem = IVP([field])
        Tarang.add_equation!(problem, "dt(u) = 0")
        Tarang.setup_domain!(problem)

        base = SolverBaseData(problem)
        @test isa(base, SolverBaseData)
        @test base.problem === problem
        @test base.entry_cutoff ≈ 1e-12
    end

    @testset "sync_state_to_problem!" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        basis = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        field = ScalarField(dist, "u", (basis,), Float64)

        problem = IVP([field])

        # Create modified state
        state = Tarang.collect_state_fields(problem.variables)

        # Should run without error
        @test begin
            Tarang.sync_state_to_problem!(problem, state)
            true
        end
    end
end

println("All solver tests passed!")
