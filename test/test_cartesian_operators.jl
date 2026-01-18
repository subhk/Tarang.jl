"""
Test Cartesian operators: skew, trace, transpose, curl, gradient, divergence, laplacian.

This test module provides comprehensive tests for:
- 2D operations with (Fourier,Fourier), (Fourier,Chebyshev), (Chebyshev,Chebyshev)
- 3D operations with (Fourier,Fourier,Fourier) and (Fourier,Fourier,Chebyshev)
- Explicit evaluation and result verification
- Implicit (matrix) evaluation for LBVP solvers
- Vector operations: skew, curl, trace, transpose
- Component extraction
"""

using Test
using LinearAlgebra
using Random: rand!
using Tarang

# Test parameters
const N_range = [16]
const dealias_range = [1.0]
const Lx = 1.3
const Ly = 2.4
const Lz = 1.9

# ============================================================================
# Helper functions for tests (defined first so they're available to all tests)
# ============================================================================

"""Fill field with random values."""
function fill_random!(f::Tarang.ScalarField)
    ensure_layout!(f, :g)
    rand!(Tarang.get_grid_data(f))
end

"""Get field data for specified layout."""
function get_field_data(f::Tarang.ScalarField, layout::Symbol)
    if layout == :g
        return Tarang.get_grid_data(f)
    else
        return Tarang.get_coeff_data(f)
    end
end

function fill_random!(f::Tarang.VectorField)
    for comp in f.components
        fill_random!(comp)
    end
end

function fill_random!(f::Tarang.TensorField)
    for comp in f.components  # Matrix iteration goes element-by-element
        fill_random!(comp)
    end
end

"""Ensure layout for VectorField (calls ensure_layout! on each component)"""
function ensure_layout_vec!(f::Tarang.VectorField, layout::Symbol)
    for comp in f.components
        ensure_layout!(comp, layout)
    end
end

"""Ensure layout for TensorField (calls ensure_layout! on each component)"""
function ensure_layout_tensor!(f::Tarang.TensorField, layout::Symbol)
    for row in f.components
        for comp in row
            ensure_layout!(comp, layout)
        end
    end
end

# ============================================================================
# Domain builders
# ============================================================================

"""Build 2D Fourier-Fourier domain."""
function build_FF(N::Int, dealias::Float64, dtype::Type=Float64)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=dtype)

    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, Lx), dealias=dealias)
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, Ly), dealias=dealias)

    bases = (xb, yb)

    # Get local grids
    x = local_grid(xb, dist, dealias)
    y = local_grid(yb, dist, dealias)

    return coords, dist, bases, (x, y)
end

"""Build 2D Fourier-Chebyshev domain."""
function build_FC(N::Int, dealias::Float64, dtype::Type=Float64)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=dtype)

    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, Lx), dealias=dealias)
    yb = ChebyshevT(coords["y"]; size=N, bounds=(0.0, Ly), dealias=dealias)

    bases = (xb, yb)

    x = local_grid(xb, dist, dealias)
    y = local_grid(yb, dist, dealias)

    return coords, dist, bases, (x, y)
end

"""Build 2D Chebyshev-Chebyshev domain."""
function build_CC(N::Int, dealias::Float64, dtype::Type=Float64)
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords; dtype=dtype)

    xb = ChebyshevT(coords["x"]; size=N, bounds=(0.0, Lx), dealias=dealias)
    yb = ChebyshevT(coords["y"]; size=N, bounds=(0.0, Ly), dealias=dealias)

    bases = (xb, yb)

    x = local_grid(xb, dist, dealias)
    y = local_grid(yb, dist, dealias)

    return coords, dist, bases, (x, y)
end

"""Build 3D Fourier-Fourier-Fourier domain."""
function build_FFF(N::Int, dealias::Float64, dtype::Type=Float64)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype=dtype)

    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, Lx), dealias=dealias)
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, Ly), dealias=dealias)
    zb = RealFourier(coords["z"]; size=N, bounds=(0.0, Lz), dealias=dealias)

    bases = (xb, yb, zb)

    x = local_grid(xb, dist, dealias)
    y = local_grid(yb, dist, dealias)
    z = local_grid(zb, dist, dealias)

    return coords, dist, bases, (x, y, z)
end

"""Build 3D Fourier-Fourier-Chebyshev domain."""
function build_FFC(N::Int, dealias::Float64, dtype::Type=Float64)
    coords = CartesianCoordinates("x", "y", "z")
    dist = Distributor(coords; dtype=dtype)

    xb = RealFourier(coords["x"]; size=N, bounds=(0.0, Lx), dealias=dealias)
    yb = RealFourier(coords["y"]; size=N, bounds=(0.0, Ly), dealias=dealias)
    zb = ChebyshevT(coords["z"]; size=N, bounds=(0.0, Lz), dealias=dealias)

    bases = (xb, yb, zb)

    x = local_grid(xb, dist, dealias)
    y = local_grid(yb, dist, dealias)
    z = local_grid(zb, dist, dealias)

    return coords, dist, bases, (x, y, z)
end

# ============================================================================
# Skew operator tests (2D only)
# Following test_cartesian_operators:101-131
# ============================================================================

@testset "Cartesian Skew Operator" begin
    @testset "Skew explicit evaluation - $basis_name" for (basis_name, basis_builder) in [
        ("FF", build_FF), ("FC", build_FC), ("CC", build_CC)
    ]
        for N in N_range
            for dealias in dealias_range
                for dtype in [Float64]
                    for layout in [:c, :g]
                        @testset "N=$N, dealias=$dealias, dtype=$dtype, layout=$layout" begin
                            coords, dist, bases, r = basis_builder(N, dealias, dtype)
                            x, y = r

                            # Create random vector field
                            f = VectorField(dist, coords, "f", bases, dtype)
                            fill_random!(f)

                            # Evaluate skew
                            ensure_layout!(f, layout)
                            g = evaluate(CartesianSkew(f))
                            ensure_layout!(g, layout)

                            # Check: skew(u_x, u_y) = (-u_y, u_x)
                            @test isapprox(get_field_data(g.components[1], layout), -get_field_data(f.components[2], layout), rtol=1e-10)
                            @test isapprox(get_field_data(g.components[2], layout), get_field_data(f.components[1], layout), rtol=1e-10)
                        end
                    end
                end
            end
        end
    end
end

# ============================================================================
# Trace operator tests
# Following test_cartesian_operators:133-187
# ============================================================================

@testset "Cartesian Trace Operator" begin
    @testset "Trace explicit evaluation 2D - $basis_name" for (basis_name, basis_builder) in [
        ("FF", build_FF), ("FC", build_FC), ("CC", build_CC)
    ]
        for N in N_range
            for dealias in dealias_range
                for dtype in [Float64]
                    for layout in [:c, :g]
                        @testset "N=$N, dealias=$dealias, dtype=$dtype, layout=$layout" begin
                            coords, dist, bases, r = basis_builder(N, dealias, dtype)

                            # Create random tensor field (use coords not (coords, coords))
                            f = TensorField(dist, coords, "f", bases, dtype)
                            fill_random!(f)

                            # Evaluate trace
                            ensure_layout!(f, layout)
                            g = evaluate(CartesianTrace(f))
                            ensure_layout!(g, layout)

                            # Check: trace(T) = T_xx + T_yy
                            expected = get_field_data(f.components[1, 1], layout) .+ get_field_data(f.components[2, 2], layout)
                            @test isapprox(get_field_data(g, layout), expected, rtol=1e-10)
                        end
                    end
                end
            end
        end
    end

    @testset "Trace explicit evaluation 3D - $basis_name" for (basis_name, basis_builder) in [
        ("FFF", build_FFF), ("FFC", build_FFC)
    ]
        for N in N_range
            for dealias in dealias_range
                for dtype in [Float64]
                    for layout in [:c, :g]
                        @testset "N=$N, dealias=$dealias, dtype=$dtype, layout=$layout" begin
                            coords, dist, bases, r = basis_builder(N, dealias, dtype)

                            # Create random tensor field
                            f = TensorField(dist, coords, "f", bases, dtype)
                            fill_random!(f)

                            # Evaluate trace
                            ensure_layout!(f, layout)
                            g = evaluate(CartesianTrace(f))
                            ensure_layout!(g, layout)

                            # Check: trace(T) = T_xx + T_yy + T_zz
                            expected = get_field_data(f.components[1, 1], layout) .+ get_field_data(f.components[2, 2], layout) .+ get_field_data(f.components[3, 3], layout)
                            @test isapprox(get_field_data(g, layout), expected, rtol=1e-10)
                        end
                    end
                end
            end
        end
    end
end

# ============================================================================
# Gradient operator tests
# ============================================================================

@testset "Cartesian Gradient Operator" begin
    # Test Fourier-Fourier with trigonometric function (exact for periodic)
    @testset "Gradient explicit evaluation 2D - FF" begin
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = build_FF(N, dealias, Float64)
                    x, y = r

                    # Create test scalar field: f = sin(2π*x/Lx) * cos(2π*y/Ly)
                    f = ScalarField(dist, "f", bases, Float64)
                    kx = 2π / Lx
                    ky = 2π / Ly

                    # Set field values
                    f_data = Tarang.get_grid_data(f)
                    for i in eachindex(x)
                        for j in eachindex(y)
                            f_data[i, j] = sin(kx * x[i]) * cos(ky * y[j])
                        end
                    end
                    f.current_layout = :g

                    # Evaluate gradient
                    grad_op = CartesianGradient(f, coords)
                    g = evaluate(grad_op, :g)

                    # Expected: ∂f/∂x = kx*cos(kx*x)*cos(ky*y)
                    #           ∂f/∂y = -ky*sin(kx*x)*sin(ky*y)
                    expected_gx = zeros(length(x), length(y))
                    expected_gy = zeros(length(x), length(y))

                    for i in eachindex(x)
                        for j in eachindex(y)
                            expected_gx[i, j] = kx * cos(kx * x[i]) * cos(ky * y[j])
                            expected_gy[i, j] = -ky * sin(kx * x[i]) * sin(ky * y[j])
                        end
                    end

                    @test isapprox(Tarang.get_grid_data(g.components[1]), expected_gx, rtol=1e-6)
                    @test isapprox(Tarang.get_grid_data(g.components[2]), expected_gy, rtol=1e-6)
                end
            end
        end
    end

    # Test Fourier-Chebyshev with mixed function:
    # - Fourier (x): use sin(kx*x) which Fourier can represent exactly
    # - Chebyshev (y): use polynomial y^2 which Chebyshev can represent exactly
    # f = sin(kx*x) * y^2, ∂f/∂x = kx*cos(kx*x)*y^2, ∂f/∂y = sin(kx*x)*2*y
    @testset "Gradient explicit evaluation 2D - FC" begin
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = build_FC(N, dealias, Float64)
                    x, y = r

                    f = ScalarField(dist, "f", bases, Float64)
                    kx = 2π / Lx  # Fourier direction

                    # Set field: f = sin(kx*x) * y^2
                    f_data = Tarang.get_grid_data(f)
                    for i in eachindex(x)
                        for j in eachindex(y)
                            f_data[i, j] = sin(kx * x[i]) * y[j]^2
                        end
                    end
                    f.current_layout = :g

                    # Evaluate gradient
                    grad_op = CartesianGradient(f, coords)
                    g = evaluate(grad_op, :g)

                    # Expected: ∂f/∂x = kx*cos(kx*x)*y^2, ∂f/∂y = sin(kx*x)*2*y
                    expected_gx = zeros(length(x), length(y))
                    expected_gy = zeros(length(x), length(y))

                    for i in eachindex(x)
                        for j in eachindex(y)
                            expected_gx[i, j] = kx * cos(kx * x[i]) * y[j]^2
                            expected_gy[i, j] = sin(kx * x[i]) * 2 * y[j]
                        end
                    end

                    @test isapprox(Tarang.get_grid_data(g.components[1]), expected_gx, rtol=1e-6)
                    @test isapprox(Tarang.get_grid_data(g.components[2]), expected_gy, rtol=1e-6)
                end
            end
        end
    end

    # Test Chebyshev-Chebyshev with polynomial function (exact for Chebyshev)
    # f = x^2 * y^2, ∂f/∂x = 2*x*y^2, ∂f/∂y = x^2*2*y
    @testset "Gradient explicit evaluation 2D - CC" begin
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = build_CC(N, dealias, Float64)
                    x, y = r

                    f = ScalarField(dist, "f", bases, Float64)

                    # Set field: f = x^2 * y^2 (polynomial - exact for Chebyshev)
                    f_data = Tarang.get_grid_data(f)
                    for i in eachindex(x)
                        for j in eachindex(y)
                            f_data[i, j] = x[i]^2 * y[j]^2
                        end
                    end
                    f.current_layout = :g

                    # Evaluate gradient
                    grad_op = CartesianGradient(f, coords)
                    g = evaluate(grad_op, :g)

                    # Expected: ∂f/∂x = 2*x*y^2, ∂f/∂y = x^2*2*y
                    expected_gx = zeros(length(x), length(y))
                    expected_gy = zeros(length(x), length(y))

                    for i in eachindex(x)
                        for j in eachindex(y)
                            expected_gx[i, j] = 2 * x[i] * y[j]^2
                            expected_gy[i, j] = x[i]^2 * 2 * y[j]
                        end
                    end

                    @test isapprox(Tarang.get_grid_data(g.components[1]), expected_gx, rtol=1e-6)
                    @test isapprox(Tarang.get_grid_data(g.components[2]), expected_gy, rtol=1e-6)
                end
            end
        end
    end
end

# ============================================================================
# Divergence operator tests
# ============================================================================

@testset "Cartesian Divergence Operator" begin
    @testset "Divergence explicit evaluation 2D - $basis_name" for (basis_name, basis_builder) in [
        ("FF", build_FF), ("FC", build_FC), ("CC", build_CC)
    ]
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = basis_builder(N, dealias, Float64)
                    x, y = r

                    # Create test vector field
                    u = VectorField(dist, coords, "u", bases, Float64)
                    kx = 2π / Lx
                    ky = 2π / Ly

                    # Set u_x = sin(kx*x), u_y = sin(ky*y)
                    for i in eachindex(x)
                        for j in eachindex(y)
                            Tarang.get_grid_data(u.components[1])[i, j] = sin(kx * x[i])
                            Tarang.get_grid_data(u.components[2])[i, j] = sin(ky * y[j])
                        end
                    end
                    u.components[1].current_layout = :g
                    u.components[2].current_layout = :g

                    # Evaluate divergence
                    div_op = CartesianDivergence(u)
                    d = evaluate(div_op, :g)

                    # Expected: ∇·u = kx*cos(kx*x) + ky*cos(ky*y)
                    expected = zeros(length(x), length(y))
                    for i in eachindex(x)
                        for j in eachindex(y)
                            expected[i, j] = kx * cos(kx * x[i]) + ky * cos(ky * y[j])
                        end
                    end

                    @test isapprox(Tarang.get_grid_data(d), expected, rtol=1e-6)
                end
            end
        end
    end
end

# ============================================================================
# Laplacian operator tests
# ============================================================================

@testset "Cartesian Laplacian Operator" begin
    @testset "Laplacian explicit evaluation 2D - $basis_name" for (basis_name, basis_builder) in [
        ("FF", build_FF), ("FC", build_FC), ("CC", build_CC)
    ]
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = basis_builder(N, dealias, Float64)
                    x, y = r

                    # Create test scalar field: f = sin(kx*x) * sin(ky*y)
                    f = ScalarField(dist, "f", bases, Float64)
                    kx = 2π / Lx
                    ky = 2π / Ly

                    for i in eachindex(x)
                        for j in eachindex(y)
                            Tarang.get_grid_data(f)[i, j] = sin(kx * x[i]) * sin(ky * y[j])
                        end
                    end
                    f.current_layout = :g

                    # Evaluate Laplacian
                    lap_op = CartesianLaplacian(f, coords)
                    lap_f = evaluate(lap_op, :g)

                    # Expected: ∇²f = -(kx² + ky²) * sin(kx*x) * sin(ky*y)
                    expected = zeros(length(x), length(y))
                    for i in eachindex(x)
                        for j in eachindex(y)
                            expected[i, j] = -(kx^2 + ky^2) * sin(kx * x[i]) * sin(ky * y[j])
                        end
                    end

                    @test isapprox(Tarang.get_grid_data(lap_f), expected, rtol=1e-5)
                end
            end
        end
    end
end

# ============================================================================
# Curl operator tests (3D only)
# ============================================================================

@testset "Cartesian Curl Operator (3D)" begin
    @testset "Curl explicit evaluation 3D - $basis_name" for (basis_name, basis_builder) in [
        ("FFF", build_FFF), ("FFC", build_FFC)
    ]
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = basis_builder(N, dealias, Float64)
                    x, y, z = r

                    # Create test vector field
                    u = VectorField(dist, coords, "u", bases, Float64)
                    kx = 2π / Lx
                    ky = 2π / Ly
                    kz = 2π / Lz

                    # Set u = (sin(ky*y), sin(kz*z), sin(kx*x))
                    for i in eachindex(x)
                        for j in eachindex(y)
                            for k in eachindex(z)
                                Tarang.get_grid_data(u.components[1])[i, j, k] = sin(ky * y[j])
                                Tarang.get_grid_data(u.components[2])[i, j, k] = sin(kz * z[k])
                                Tarang.get_grid_data(u.components[3])[i, j, k] = sin(kx * x[i])
                            end
                        end
                    end
                    for comp in u.components
                        comp.current_layout = :g
                    end

                    # Evaluate curl
                    curl_op = CartesianCurl(u)
                    curl_u = evaluate(curl_op, :g)

                    # Expected curl:
                    # curl_x = ∂u_z/∂y - ∂u_y/∂z = 0 - kz*cos(kz*z) = -kz*cos(kz*z)
                    # curl_y = ∂u_x/∂z - ∂u_z/∂x = 0 - kx*cos(kx*x) = -kx*cos(kx*x)
                    # curl_z = ∂u_y/∂x - ∂u_x/∂y = 0 - ky*cos(ky*y) = -ky*cos(ky*y)

                    expected_curl_x = zeros(length(x), length(y), length(z))
                    expected_curl_y = zeros(length(x), length(y), length(z))
                    expected_curl_z = zeros(length(x), length(y), length(z))

                    for i in eachindex(x)
                        for j in eachindex(y)
                            for k in eachindex(z)
                                expected_curl_x[i, j, k] = -kz * cos(kz * z[k])
                                expected_curl_y[i, j, k] = -kx * cos(kx * x[i])
                                expected_curl_z[i, j, k] = -ky * cos(ky * y[j])
                            end
                        end
                    end

                    @test isapprox(Tarang.get_grid_data(curl_u.components[1]), expected_curl_x, rtol=1e-5)
                    @test isapprox(Tarang.get_grid_data(curl_u.components[2]), expected_curl_y, rtol=1e-5)
                    @test isapprox(Tarang.get_grid_data(curl_u.components[3]), expected_curl_z, rtol=1e-5)
                end
            end
        end
    end
end

# ============================================================================
# CartesianComponent extraction tests
# ============================================================================

@testset "CartesianComponent Extraction" begin
    @testset "Component extraction 2D - $basis_name" for (basis_name, basis_builder) in [
        ("FF", build_FF), ("FC", build_FC), ("CC", build_CC)
    ]
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = basis_builder(N, dealias, Float64)
                    x, y = r

                    # Create test vector field
                    u = VectorField(dist, coords, "u", bases, Float64)

                    # Set different values for each component
                    for i in eachindex(x)
                        for j in eachindex(y)
                            Tarang.get_grid_data(u.components[1])[i, j] = sin(2π * x[i] / Lx)
                            Tarang.get_grid_data(u.components[2])[i, j] = cos(2π * y[j] / Ly)
                        end
                    end
                    u.components[1].current_layout = :g
                    u.components[2].current_layout = :g

                    # Extract x-component
                    comp_x_op = CartesianComponent(u; index=0, comp=coords["x"])
                    comp_x = evaluate(comp_x_op, :g)

                    # Extract y-component
                    comp_y_op = CartesianComponent(u; index=0, comp=coords["y"])
                    comp_y = evaluate(comp_y_op, :g)

                    # Verify
                    @test isapprox(Tarang.get_grid_data(comp_x), Tarang.get_grid_data(u.components[1]), rtol=1e-10)
                    @test isapprox(Tarang.get_grid_data(comp_y), Tarang.get_grid_data(u.components[2]), rtol=1e-10)
                end
            end
        end
    end
end

# ============================================================================
# TransposeComponents operator tests
# Following test_cartesian_operators transpose tests
# ============================================================================

@testset "Cartesian TransposeComponents Operator" begin
    @testset "Transpose explicit evaluation 2D - $basis_name" for (basis_name, basis_builder) in [
        ("FF", build_FF), ("FC", build_FC), ("CC", build_CC)
    ]
        for N in N_range
            for dealias in dealias_range
                for dtype in [Float64]
                    for layout in [:c, :g]
                        @testset "N=$N, dealias=$dealias, dtype=$dtype, layout=$layout" begin
                            coords, dist, bases, r = basis_builder(N, dealias, dtype)

                            # Create random tensor field
                            f = TensorField(dist, coords, "f", bases, dtype)
                            fill_random!(f)

                            # Evaluate transpose
                            ensure_layout!(f, layout)
                            g = evaluate(TransposeComponents(f))
                            ensure_layout!(g, layout)

                            # Check: transpose(T)_ij = T_ji
                            for i in 1:2
                                for j in 1:2
                                    @test isapprox(
                                        get_field_data(g.components[i, j], layout),
                                        get_field_data(f.components[j, i], layout),
                                        rtol=1e-10
                                    )
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "Transpose explicit evaluation 3D - $basis_name" for (basis_name, basis_builder) in [
        ("FFF", build_FFF), ("FFC", build_FFC)
    ]
        for N in N_range
            for dealias in dealias_range
                for dtype in [Float64]
                    for layout in [:c, :g]
                        @testset "N=$N, dealias=$dealias, dtype=$dtype, layout=$layout" begin
                            coords, dist, bases, r = basis_builder(N, dealias, dtype)

                            # Create random tensor field
                            f = TensorField(dist, coords, "f", bases, dtype)
                            fill_random!(f)

                            # Evaluate transpose
                            ensure_layout!(f, layout)
                            g = evaluate(TransposeComponents(f))
                            ensure_layout!(g, layout)

                            # Check: transpose(T)_ij = T_ji
                            dim = 3
                            for i in 1:dim
                                for j in 1:dim
                                    @test isapprox(
                                        get_field_data(g.components[i, j], layout),
                                        get_field_data(f.components[j, i], layout),
                                        rtol=1e-10
                                    )
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# ============================================================================
# Dot product tests
# ============================================================================

@testset "Dot Product Operator" begin
    @testset "Dot product 2D - $basis_name" for (basis_name, basis_builder) in [
        ("FF", build_FF), ("FC", build_FC), ("CC", build_CC)
    ]
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = basis_builder(N, dealias, Float64)
                    x, y = r

                    # Create two vector fields
                    u = VectorField(dist, coords, "u", bases, Float64)
                    v = VectorField(dist, coords, "v", bases, Float64)

                    # Set test values: u = (sin(x), cos(y)), v = (cos(x), sin(y))
                    kx = 2π / Lx
                    ky = 2π / Ly
                    for i in eachindex(x)
                        for j in eachindex(y)
                            Tarang.get_grid_data(u.components[1])[i, j] = sin(kx * x[i])
                            Tarang.get_grid_data(u.components[2])[i, j] = cos(ky * y[j])
                            Tarang.get_grid_data(v.components[1])[i, j] = cos(kx * x[i])
                            Tarang.get_grid_data(v.components[2])[i, j] = sin(ky * y[j])
                        end
                    end
                    for comp in u.components
                        comp.current_layout = :g
                    end
                    for comp in v.components
                        comp.current_layout = :g
                    end

                    # Evaluate dot product
                    dot_result = DotProduct(u, v)
                    result = evaluate(dot_result, :g)

                    # Expected: u·v = sin(x)*cos(x) + cos(y)*sin(y)
                    expected = zeros(length(x), length(y))
                    for i in eachindex(x)
                        for j in eachindex(y)
                            expected[i, j] = sin(kx * x[i]) * cos(kx * x[i]) + cos(ky * y[j]) * sin(ky * y[j])
                        end
                    end

                    @test isapprox(Tarang.get_grid_data(result), expected, rtol=1e-10)
                end
            end
        end
    end
end

# ============================================================================
# Cross product tests (3D)
# ============================================================================

@testset "Cross Product Operator" begin
    @testset "Cross product 3D - $basis_name" for (basis_name, basis_builder) in [
        ("FFF", build_FFF), ("FFC", build_FFC)
    ]
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = basis_builder(N, dealias, Float64)
                    x, y, z = r

                    # Create two vector fields
                    u = VectorField(dist, coords, "u", bases, Float64)
                    v = VectorField(dist, coords, "v", bases, Float64)

                    # Set test values: u = (1, 0, 0), v = (0, 1, 0)
                    # Expected: u × v = (0, 0, 1)
                    for i in eachindex(x)
                        for j in eachindex(y)
                            for k in eachindex(z)
                                Tarang.get_grid_data(u.components[1])[i, j, k] = 1.0
                                Tarang.get_grid_data(u.components[2])[i, j, k] = 0.0
                                Tarang.get_grid_data(u.components[3])[i, j, k] = 0.0
                                Tarang.get_grid_data(v.components[1])[i, j, k] = 0.0
                                Tarang.get_grid_data(v.components[2])[i, j, k] = 1.0
                                Tarang.get_grid_data(v.components[3])[i, j, k] = 0.0
                            end
                        end
                    end
                    for comp in u.components
                        comp.current_layout = :g
                    end
                    for comp in v.components
                        comp.current_layout = :g
                    end

                    # Evaluate cross product
                    cross_result = CrossProduct(u, v)
                    result = evaluate(cross_result, :g)

                    # Expected: u × v = (0, 0, 1)
                    @test isapprox(Tarang.get_grid_data(result.components[1]), zeros(length(x), length(y), length(z)), atol=1e-10)
                    @test isapprox(Tarang.get_grid_data(result.components[2]), zeros(length(x), length(y), length(z)), atol=1e-10)
                    @test isapprox(Tarang.get_grid_data(result.components[3]), ones(length(x), length(y), length(z)), atol=1e-10)
                end
            end
        end
    end
end

# ============================================================================
# 2D Curl operator tests (scalar result)
# ============================================================================

@testset "Cartesian Curl Operator (2D)" begin
    @testset "Curl explicit evaluation 2D - $basis_name" for (basis_name, basis_builder) in [
        ("FF", build_FF), ("FC", build_FC), ("CC", build_CC)
    ]
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = basis_builder(N, dealias, Float64)
                    x, y = r

                    # Create test vector field: u = (sin(ky*y), sin(kx*x))
                    u = VectorField(dist, coords, "u", bases, Float64)
                    kx = 2π / Lx
                    ky = 2π / Ly

                    for i in eachindex(x)
                        for j in eachindex(y)
                            Tarang.get_grid_data(u.components[1])[i, j] = sin(ky * y[j])
                            Tarang.get_grid_data(u.components[2])[i, j] = sin(kx * x[i])
                        end
                    end
                    u.components[1].current_layout = :g
                    u.components[2].current_layout = :g

                    # Evaluate 2D curl (using generic Curl, not CartesianCurl which is 3D only)
                    curl_u = evaluate(Curl(u, coords), :g)

                    # Expected: curl(u) = ∂u_y/∂x - ∂u_x/∂y = kx*cos(kx*x) - ky*cos(ky*y)
                    expected = zeros(length(x), length(y))
                    for i in eachindex(x)
                        for j in eachindex(y)
                            expected[i, j] = kx * cos(kx * x[i]) - ky * cos(ky * y[j])
                        end
                    end

                    @test isapprox(Tarang.get_grid_data(curl_u), expected, rtol=1e-5)
                end
            end
        end
    end
end

# ============================================================================
# div(skew(f)) = -curl(f) identity test (2D)
# Following Dedalus test pattern
# ============================================================================

@testset "div(skew(f)) = -curl(f) Identity (2D)" begin
    @testset "Identity test - $basis_name" for (basis_name, basis_builder) in [
        ("FF", build_FF), ("FC", build_FC), ("CC", build_CC)
    ]
        for N in N_range
            for dealias in dealias_range
                @testset "N=$N, dealias=$dealias" begin
                    coords, dist, bases, r = basis_builder(N, dealias, Float64)
                    x, y = r

                    # Create test vector field
                    u = VectorField(dist, coords, "u", bases, Float64)
                    kx = 2π / Lx
                    ky = 2π / Ly

                    for i in eachindex(x)
                        for j in eachindex(y)
                            Tarang.get_grid_data(u.components[1])[i, j] = sin(kx * x[i]) * cos(ky * y[j])
                            Tarang.get_grid_data(u.components[2])[i, j] = cos(kx * x[i]) * sin(ky * y[j])
                        end
                    end
                    u.components[1].current_layout = :g
                    u.components[2].current_layout = :g

                    # Compute div(skew(u))
                    skew_u = evaluate(CartesianSkew(u), :g)
                    div_skew_u = evaluate(CartesianDivergence(skew_u), :g)

                    # Compute -curl(u)
                    curl_u = evaluate(Curl(u, coords), :g)

                    # They should be equal (with sign convention)
                    # div(skew(u)) = -curl(u) in 2D
                    @test isapprox(Tarang.get_grid_data(div_skew_u), -Tarang.get_grid_data(curl_u), rtol=1e-5)
                end
            end
        end
    end
end

# ============================================================================
# Matrix operation tests
# ============================================================================

@testset "Operator Matrix Methods" begin
    @testset "matrix_dependence" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64)

        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, Lx))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, Ly))
        bases = (xb, yb)

        f = ScalarField(dist, "f", bases, Float64)
        g = ScalarField(dist, "g", bases, Float64)

        # Test gradient matrix dependence
        grad_op = CartesianGradient(f, coords)
        dep = matrix_dependence(grad_op, f, g)

        @test dep[1] == true   # Depends on f
        @test dep[2] == false  # Does not depend on g
    end

    @testset "check_conditions" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; dtype=Float64)

        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, Lx))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, Ly))
        bases = (xb, yb)

        u = VectorField(dist, coords, "u", bases, Float64)
        fill_random!(u)

        # Test condition checking
        skew_op = CartesianSkew(u)
        @test check_conditions(skew_op) == true

        div_op = CartesianDivergence(u)
        @test check_conditions(div_op) == true
    end
end

# Run tests if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Run all tests
    @testset "All Cartesian Operator Tests" begin
        include(@__FILE__)
    end
end
