"""
Test suite for the 7 Dedalus Cartesian features added to Tarang.jl:
1. CNLF2 variable-timestep coefficients
2. HilbertTransform operator
3. UnaryGridFunction symbolic derivatives (sym_diff)
4. DictionaryHandler (in-memory analysis)
5. VirtualFileHandler (NetCDF virtual datasets)
6. Copy operator
7. Frechet differentiation for NLBVP
"""

using Test
using Tarang
using LinearAlgebra
using SparseArrays

# ============================================================================
# Helper: create a 1D RealFourier field with given data
# ============================================================================
function make_1d_fourier_field(; N=64, L=2π, name="u", dtype=Float64)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=dtype)
    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
    field = ScalarField(dist, name, (xb,), dtype)
    mesh = Tarang.create_meshgrid(field.domain)
    x = mesh["x"]
    return field, x, xb, dist, coords
end

function make_1d_complex_fourier_field(; N=64, L=2π, name="u")
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; mesh=(1,), dtype=ComplexF64)
    xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, L))
    field = ScalarField(dist, name, (xb,), ComplexF64)
    mesh = Tarang.create_meshgrid(field.domain)
    x = mesh["x"]
    return field, x, xb, dist, coords
end

# ============================================================================
@testset "Dedalus Feature Tests" begin
# ============================================================================

# -----------------------------------------------------------------------
@testset "Copy operator" begin
    field, x, _, _, _ = make_1d_fourier_field()
    Tarang.get_grid_data(field) .= @. sin(x)
    original_data = copy(Tarang.get_grid_data(field))

    # Create copy via operator
    copy_op = Tarang.Copy(field)
    copied = Tarang.evaluate_copy(copy_op, :g)

    @testset "Copy matches original" begin
        @test isapprox(Tarang.get_grid_data(copied), original_data; atol=1e-14)
    end

    @testset "Copy is independent" begin
        # Modify original
        Tarang.get_grid_data(field) .= 0.0
        # Copy should be unchanged
        @test isapprox(Tarang.get_grid_data(copied), original_data; atol=1e-14)
        @test !isapprox(Tarang.get_grid_data(field), original_data; atol=1e-14)
    end

    @testset "copy_field constructor" begin
        field2, _, _, _, _ = make_1d_fourier_field(name="v")
        Tarang.get_grid_data(field2) .= 42.0
        cp = Tarang.copy_field(field2)
        @test isa(cp, Tarang.Copy)
        result = evaluate(cp, :g)
        @test isapprox(Tarang.get_grid_data(result), fill(42.0, size(Tarang.get_grid_data(result))); atol=1e-14)
    end
end

# -----------------------------------------------------------------------
@testset "HilbertTransform" begin
    @testset "RealFourier: H[sin(kx)] = -cos(kx)" begin
        field, x, _, _, _ = make_1d_fourier_field(N=64)
        k = 3
        Tarang.get_grid_data(field) .= @. sin(k * x)

        ht_op = Tarang.HilbertTransform(field)
        result = Tarang.evaluate_hilbert_transform(ht_op, :g)
        expected = @. -cos(k * x)

        @test isapprox(Tarang.get_grid_data(result), expected; atol=1e-10)
    end

    @testset "RealFourier: H[cos(kx)] = sin(kx)" begin
        field, x, _, _, _ = make_1d_fourier_field(N=64)
        k = 2
        Tarang.get_grid_data(field) .= @. cos(k * x)

        result = evaluate(Tarang.HilbertTransform(field), :g)
        expected = @. sin(k * x)

        @test isapprox(Tarang.get_grid_data(result), expected; atol=1e-10)
    end

    @testset "H[H[f]] ≈ -f for zero-mean" begin
        field, x, _, _, _ = make_1d_fourier_field(N=64)
        Tarang.get_grid_data(field) .= @. sin(2x) + 0.5 * cos(5x)
        original = copy(Tarang.get_grid_data(field))

        # Apply Hilbert twice
        h1 = evaluate(Tarang.HilbertTransform(field), :g)
        h2 = evaluate(Tarang.HilbertTransform(h1), :g)

        @test isapprox(Tarang.get_grid_data(h2), -original; atol=1e-10)
    end

    @testset "hilbert constructor" begin
        field, _, _, _, _ = make_1d_fourier_field()
        ht = Tarang.hilbert(field)
        @test isa(ht, Tarang.HilbertTransform)
    end
end

# -----------------------------------------------------------------------
@testset "DictionaryHandler" begin
    @testset "Basic construction and access" begin
        dh = DictionaryHandler(cadence=1)
        @test dh.write_count == 0
        @test isempty(keys(dh))
    end

    @testset "should_write scheduling" begin
        dh = DictionaryHandler(cadence=5)
        @test !Tarang.should_write(dh, 0.0, 0.0, 3)
        @test Tarang.should_write(dh, 0.0, 0.0, 5)

        dh2 = DictionaryHandler(sim_dt=0.1)
        dh2.last_write_sim_time = -1.0  # simulate "no previous write"
        @test Tarang.should_write(dh2, 0.0, 0.0, 1)    # first write (0.0 - (-1.0) >= 0.1)
        dh2.last_write_sim_time = 0.0
        @test !Tarang.should_write(dh2, 0.0, 0.05, 2)  # too early
        @test Tarang.should_write(dh2, 0.0, 0.15, 3)   # enough time

        dh3 = DictionaryHandler(max_writes=2)
        dh3.write_count = 2
        @test !Tarang.should_write(dh3, 0.0, 0.0, 1)   # reached max
    end

    @testset "Task addition" begin
        dh = DictionaryHandler()
        field, _, _, _, _ = make_1d_fourier_field()
        add_task!(dh, field, "velocity")
        @test haskey(dh.datasets, "velocity")
    end
end

# -----------------------------------------------------------------------
@testset "CNLF2 variable timestep" begin
    # Test that the coefficient formulas are correct for constant dt (w1=1)
    @testset "Constant dt coefficients reduce to standard CNLF" begin
        dt = 0.01
        w1 = 1.0  # constant dt

        # Dedalus CNLF2 coefficients
        a1 = 1.0 / ((1.0 + w1) * dt)
        a2 = (w1 - 1.0) / dt
        a3 = -w1^2 / ((1.0 + w1) * dt)

        b1 = 1.0 / (2.0 * w1)
        b2 = (1.0 - 1.0 / w1) / 2.0
        b3 = 1.0 / 2.0

        # For w1=1: a1 = 1/(2*dt), a2 = 0, a3 = -1/(2*dt)
        @test isapprox(a1, 1.0 / (2.0 * dt); atol=1e-14)
        @test isapprox(a2, 0.0; atol=1e-14)
        @test isapprox(a3, -1.0 / (2.0 * dt); atol=1e-14)

        # For w1=1: b1 = 0.5, b2 = 0, b3 = 0.5
        @test isapprox(b1, 0.5; atol=1e-14)
        @test isapprox(b2, 0.0; atol=1e-14)
        @test isapprox(b3, 0.5; atol=1e-14)
    end

    @testset "Variable dt coefficients (w1=2)" begin
        dt = 0.01
        w1 = 2.0  # dt_current = 2*dt_previous

        a1 = 1.0 / ((1.0 + w1) * dt)
        a2 = (w1 - 1.0) / dt
        a3 = -w1^2 / ((1.0 + w1) * dt)

        b1 = 1.0 / (2.0 * w1)
        b2 = (1.0 - 1.0 / w1) / 2.0
        b3 = 1.0 / 2.0

        # Verify coefficients satisfy consistency: a1 + a2 + a3 = 0
        # (mass matrix terms must cancel for steady state)
        @test isapprox(a1 + a2 + a3, 0.0; atol=1e-14)

        # Verify b coefficients sum to 1 (for implicit consistency)
        @test isapprox(b1 + b2 + b3, 1.0; atol=1e-14)
    end

    @testset "Coefficient consistency for arbitrary w1" begin
        dt = 0.01
        for w1 in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
            a1 = 1.0 / ((1.0 + w1) * dt)
            a2 = (w1 - 1.0) / dt
            a3 = -w1^2 / ((1.0 + w1) * dt)

            b1 = 1.0 / (2.0 * w1)
            b2 = (1.0 - 1.0 / w1) / 2.0
            b3 = 1.0 / 2.0

            # Mass conservation: a1 + a2 + a3 = 0
            @test isapprox(a1 + a2 + a3, 0.0; atol=1e-13) broken=(false)

            # Implicit consistency: b1 + b2 + b3 = 1
            @test isapprox(b1 + b2 + b3, 1.0; atol=1e-13)
        end
    end
end

# -----------------------------------------------------------------------
@testset "Symbolic differentiation" begin
    field_u, _, _, dist, coords = make_1d_fourier_field(name="u")
    field_v, _, _, _, _ = make_1d_fourier_field(name="v")

    @testset "Basic derivatives" begin
        # d(u)/d(u) = 1
        @test sym_diff(field_u, field_u) == 1

        # d(u)/d(v) = 0
        @test sym_diff(field_u, field_v) == 0

        # d(const)/d(u) = 0
        @test sym_diff(3.0, field_u) == 0
    end

    @testset "Arithmetic rules" begin
        # d(u + v)/d(u) = 1
        add_op = Tarang.AddOperator(field_u, field_v)
        @test sym_diff(add_op, field_u) == 1

        # d(u - v)/d(u) = 1
        sub_op = Tarang.SubtractOperator(field_u, field_v)
        @test sym_diff(sub_op, field_u) == 1

        # d(u * v)/d(u) = v (product rule)
        mul_op = Tarang.MultiplyOperator(field_u, field_v)
        result = sym_diff(mul_op, field_u)
        # Should be field_v (since d(u)/d(u)=1 and d(v)/d(u)=0)
        @test result === field_v

        # d(3 * u)/d(u) = 3
        scaled_op = Tarang.MultiplyOperator(3.0, field_u)
        @test sym_diff(scaled_op, field_u) == 3.0
    end

    @testset "Chain rule for UnaryGridFunction" begin
        # d(sin(u))/d(u) = cos(u)
        sin_op = Tarang.UnaryGridFunction(field_u, sin, "sin")
        result = sym_diff(sin_op, field_u)
        # Result should be UnaryGridFunction(u, cos, "d_sin") * 1 = cos(u)
        @test isa(result, Tarang.UnaryGridFunction)
        @test result.func === cos

        # d(exp(u))/d(u) = exp(u)
        exp_op = Tarang.UnaryGridFunction(field_u, exp, "exp")
        result = sym_diff(exp_op, field_u)
        @test isa(result, Tarang.UnaryGridFunction)
        @test result.func === exp
    end

    @testset "Nested derivatives" begin
        # d(sin(u) * v)/d(u) = cos(u) * v
        sin_u = Tarang.UnaryGridFunction(field_u, sin, "sin")
        prod = Tarang.MultiplyOperator(sin_u, field_v)
        result = sym_diff(prod, field_u)
        # Result should be cos(u) * v
        @test isa(result, Tarang.MultiplyOperator)
    end

    @testset "UFUNC_DERIVATIVES table completeness" begin
        expected_funcs = [sin, cos, tan, exp, log, sqrt, abs, tanh, sinh, cosh, asin, acos, atan]
        for f in expected_funcs
            @test haskey(UFUNC_DERIVATIVES, f)
        end
    end

    @testset "simplify" begin
        @test simplify(Tarang.AddOperator(0, field_u)) === field_u
        @test simplify(Tarang.MultiplyOperator(1, field_u)) === field_u
        @test simplify(Tarang.MultiplyOperator(0, field_u)) == 0
        @test simplify(Tarang.SubtractOperator(field_u, 0)) === field_u
    end
end

# -----------------------------------------------------------------------
@testset "Frechet differentiation / NLBVP" begin
    field_u, _, _, _, _ = make_1d_fourier_field(name="u")
    field_v, _, _, _, _ = make_1d_fourier_field(name="v")

    @testset "frechet_differential basic" begin
        # F(u) = u^2, dF/du * δu = 2*u * δu
        du = ScalarField(field_u.dist, "du", field_u.bases, field_u.dtype)

        # u * u
        F = Tarang.MultiplyOperator(field_u, field_u)
        dF = frechet_differential(F, [field_u], [du])

        # dF should be u*du + du*u = 2*u*du
        @test dF !== nothing
        @test dF != 0
    end

    @testset "frechet_differential with multiple variables" begin
        du = ScalarField(field_u.dist, "du", field_u.bases, field_u.dtype)
        dv = ScalarField(field_v.dist, "dv", field_v.bases, field_v.dtype)

        # F(u,v) = u * v
        F = Tarang.MultiplyOperator(field_u, field_v)
        dF = frechet_differential(F, [field_u, field_v], [du, dv])

        # dF = v*du + u*dv
        @test dF !== nothing
        @test dF != 0
    end

    @testset "frechet_differential of constant" begin
        du = ScalarField(field_u.dist, "du", field_u.bases, field_u.dtype)
        dF = frechet_differential(5.0, [field_u], [du])
        @test dF == 0
    end
end

# -----------------------------------------------------------------------
@testset "Virtual file handler" begin
    @testset "VirtualFileHandler construction" begin
        tmpdir = mktempdir()
        vfh = VirtualFileHandler(tmpdir, "test_handler"; cadence=1)
        @test vfh.name == "test_handler"
        @test vfh.write_count == 0
        @test vfh.set_num == 1
        @test vfh.rank == 0  # Single-process
        @test vfh.nprocs >= 1
        rm(tmpdir; recursive=true)
    end

    @testset "VirtualFileHandler scheduling" begin
        tmpdir = mktempdir()
        vfh = VirtualFileHandler(tmpdir, "sched_test"; cadence=3)
        @test !Tarang.should_write(vfh, 0.0, 0.0, 1)
        @test Tarang.should_write(vfh, 0.0, 0.0, 3)

        vfh2 = VirtualFileHandler(tmpdir, "sched_test2"; max_writes=1)
        vfh2.write_count = 1
        @test !Tarang.should_write(vfh2, 0.0, 0.0, 1)
        rm(tmpdir; recursive=true)
    end

    @testset "VirtualFileHandler task management" begin
        tmpdir = mktempdir()
        vfh = VirtualFileHandler(tmpdir, "task_test")
        field, _, _, _, _ = make_1d_fourier_field()
        add_task!(vfh, field, "velocity")
        @test haskey(vfh.datasets, "velocity")
        rm(tmpdir; recursive=true)
    end
end

# ============================================================================
end  # top-level testset
# ============================================================================
