"""
Test basic operator functionality for Tarang.jl.

Tests cover:
- grad on ScalarField (2D, 3D)
- div (divergence) on VectorField (2D)
- lap (Laplacian) on ScalarField (2D)
- curl on VectorField (3D)
- Unicode aliases: nabla, Delta, nabla-squared
- dot product (u cdot v)
- interpolate, integrate
- evaluate() returns correct type
"""

using Test
using Tarang

# ============================================================================
# Helper: fill a ScalarField with grid data from a function of mesh arrays
# ============================================================================

function set_grid!(field::ScalarField, data::AbstractArray)
    Tarang.get_grid_data(field) .= data
    field.current_layout = :g
end

function set_grid!(field::VectorField, data_per_comp::Vector{<:AbstractArray})
    for (i, arr) in enumerate(data_per_comp)
        Tarang.get_grid_data(field.components[i]) .= arr
        field.components[i].current_layout = :g
    end
end

# ============================================================================
# Gradient on ScalarField
# ============================================================================

@testset "grad on ScalarField (2D Fourier-Fourier)" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # f = sin(x) * cos(y)
    set_grid!(f, @. sin(x) * cos(y))

    grad_op = CartesianGradient(f, domain.dist.coordsys)
    g = evaluate(grad_op, :g)

    # df/dx = cos(x)*cos(y), df/dy = -sin(x)*sin(y)
    @test g isa VectorField
    @test isapprox(Tarang.get_grid_data(g.components[1]), @.(cos(x) * cos(y)); rtol=1e-6)
    @test isapprox(Tarang.get_grid_data(g.components[2]), @.(-sin(x) * sin(y)); rtol=1e-6)
end

@testset "grad on ScalarField (3D Fourier-Fourier-Fourier)" begin
    domain = PeriodicDomain(16, 16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y, z = mesh["x"], mesh["y"], mesh["z"]

    # f = sin(x)*cos(y)*sin(2z)
    set_grid!(f, @. sin(x) * cos(y) * sin(2z))

    grad_op = CartesianGradient(f, domain.dist.coordsys)
    g = evaluate(grad_op, :g)

    @test length(g.components) == 3
    @test isapprox(Tarang.get_grid_data(g.components[1]), @.(cos(x) * cos(y) * sin(2z)); rtol=1e-5)
    @test isapprox(Tarang.get_grid_data(g.components[2]), @.(-sin(x) * sin(y) * sin(2z)); rtol=1e-5)
    @test isapprox(Tarang.get_grid_data(g.components[3]), @.(2 * sin(x) * cos(y) * cos(2z)); rtol=1e-5)
end

# ============================================================================
# Divergence on VectorField
# ============================================================================

@testset "div on VectorField (2D Fourier-Fourier)" begin
    domain = PeriodicDomain(16, 16)
    u = VectorField(domain, "u")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # u = (sin(x), sin(y))  =>  div(u) = cos(x) + cos(y)
    set_grid!(u, [(@. sin(x) + 0.0 * y), (@. 0.0 * x + sin(y))])

    div_op = CartesianDivergence(u)
    d = evaluate(div_op, :g)

    @test d isa ScalarField
    expected = @. cos(x) + cos(y)
    @test isapprox(Tarang.get_grid_data(d), expected; rtol=1e-6)
end

# ============================================================================
# Laplacian on ScalarField
# ============================================================================

@testset "lap on ScalarField (2D Fourier-Fourier)" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # f = sin(x)*sin(y)  =>  lap(f) = -2*sin(x)*sin(y)
    set_grid!(f, @. sin(x) * sin(y))

    lap_op = CartesianLaplacian(f, domain.dist.coordsys)
    lap_f = evaluate(lap_op, :g)

    @test lap_f isa ScalarField
    expected = @. -2.0 * sin(x) * sin(y)
    @test isapprox(Tarang.get_grid_data(lap_f), expected; rtol=1e-5)
end

@testset "lap on ScalarField with higher wavenumber" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # f = cos(2x)*sin(3y)  =>  lap(f) = -(4+9)*cos(2x)*sin(3y)
    set_grid!(f, @. cos(2x) * sin(3y))

    lap_op = CartesianLaplacian(f, domain.dist.coordsys)
    lap_f = evaluate(lap_op, :g)

    expected = @. -13.0 * cos(2x) * sin(3y)
    @test isapprox(Tarang.get_grid_data(lap_f), expected; rtol=1e-5)
end

# ============================================================================
# Curl on VectorField (3D)
# ============================================================================

@testset "curl on VectorField (3D Fourier-Fourier-Fourier)" begin
    domain = PeriodicDomain(16, 16, 16)
    u = VectorField(domain, "u")
    mesh = Tarang.create_meshgrid(domain)
    x, y, z = mesh["x"], mesh["y"], mesh["z"]

    # u = (sin(y), sin(z), sin(x))
    ux = @. sin(y) + 0.0 * x + 0.0 * z
    uy = @. sin(z) + 0.0 * x + 0.0 * y
    uz = @. sin(x) + 0.0 * y + 0.0 * z
    set_grid!(u, [ux, uy, uz])

    curl_op = CartesianCurl(u)
    curl_u = evaluate(curl_op, :g)

    @test curl_u isa VectorField
    @test length(curl_u.components) == 3

    # curl = (duz/dy - duy/dz, dux/dz - duz/dx, duy/dx - dux/dy)
    #      = (0 - cos(z), 0 - cos(x), 0 - cos(y))
    expected_cx = @. 0.0 * x + 0.0 * y - cos(z)
    expected_cy = @. -cos(x) + 0.0 * y + 0.0 * z
    expected_cz = @. 0.0 * x - cos(y) + 0.0 * z

    @test isapprox(Tarang.get_grid_data(curl_u.components[1]), expected_cx; rtol=1e-5)
    @test isapprox(Tarang.get_grid_data(curl_u.components[2]), expected_cy; rtol=1e-5)
    @test isapprox(Tarang.get_grid_data(curl_u.components[3]), expected_cz; rtol=1e-5)
end

# ============================================================================
# Unicode aliases
# ============================================================================

@testset "Unicode aliases are defined" begin
    # Verify that the Unicode aliases are bound to the expected functions
    @test ∇ === grad
    @test Δ === lap
    @test ∇² === lap
end

@testset "Unicode alias grad (nabla) produces correct operator" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    set_grid!(f, @. sin(x) * cos(y))

    # Use the Unicode alias
    g_op = ∇(f)
    g = evaluate(g_op, :g)

    @test g isa VectorField
    @test isapprox(Tarang.get_grid_data(g.components[1]), @.(cos(x) * cos(y)); rtol=1e-6)
end

@testset "Unicode alias lap (Delta and nabla-squared)" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]
    set_grid!(f, @. sin(x) * sin(y))

    lap_1 = Δ(f)
    lap_2 = ∇²(f)

    r1 = evaluate(lap_1, :g)
    r2 = evaluate(lap_2, :g)

    expected = @. -2.0 * sin(x) * sin(y)
    @test isapprox(Tarang.get_grid_data(r1), expected; rtol=1e-5)
    @test isapprox(Tarang.get_grid_data(r2), expected; rtol=1e-5)
end

# ============================================================================
# Dot product (u cdot v)
# ============================================================================

@testset "dot product u cdot v (2D)" begin
    domain = PeriodicDomain(16, 16)
    u = VectorField(domain, "u")
    v = VectorField(domain, "v")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # u = (sin(x), cos(y)), v = (cos(x), sin(y))
    set_grid!(u, [(@. sin(x) + 0.0 * y), (@. 0.0 * x + cos(y))])
    set_grid!(v, [(@. cos(x) + 0.0 * y), (@. 0.0 * x + sin(y))])

    # Dot product: u.v = sin(x)*cos(x) + cos(y)*sin(y) = 0.5*sin(2x) + 0.5*sin(2y)
    dp = u ⋅ v
    @test dp isa DotProduct
end

# ============================================================================
# evaluate() returns correct types
# ============================================================================

@testset "evaluate returns correct types" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    u = VectorField(domain, "u")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    set_grid!(f, @. sin(x) * cos(y))
    set_grid!(u, [(@. sin(x) + 0.0 * y), (@. 0.0 * x + cos(y))])

    # grad of scalar => VectorField
    grad_result = evaluate(CartesianGradient(f, domain.dist.coordsys), :g)
    @test grad_result isa VectorField

    # div of vector => ScalarField
    div_result = evaluate(CartesianDivergence(u), :g)
    @test div_result isa ScalarField

    # lap of scalar => ScalarField
    lap_result = evaluate(CartesianLaplacian(f, domain.dist.coordsys), :g)
    @test lap_result isa ScalarField
end

# ============================================================================
# Interpolation
# ============================================================================

@testset "interpolate on ScalarField (Fourier)" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # f = sin(x) * cos(y)
    set_grid!(f, @. sin(x) * cos(y))

    coords = domain.dist.coordsys
    # Interpolate at x = pi/4 along x-coordinate
    interp_op = interpolate(f, coords["x"], pi / 4)
    result = evaluate(interp_op)

    # Result should be a field (reduced dimension)
    @test result isa ScalarField
end

# ============================================================================
# Integration
# ============================================================================

@testset "integrate on ScalarField (Fourier)" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # f = 1.0 (constant). Integral over [0,2pi]^2 = (2pi)^2
    set_grid!(f, @. 0.0 * x + 0.0 * y + 1.0)

    coords = domain.dist.coordsys
    int_op = integrate(f, coords["x"])
    result = evaluate(int_op)

    @test result isa ScalarField
end

# ============================================================================
# Fourier-Chebyshev mixed domain (ChannelDomain)
# ============================================================================

@testset "grad on ScalarField (Fourier-Chebyshev channel)" begin
    domain = ChannelDomain(16, 16; Lx=2pi, Lz=2.0)
    f = ScalarField(domain, "f")
    coords = domain.dist.coordsys
    dist = domain.dist

    xb, zb = domain.bases
    x_grid = Tarang.local_grid(xb, dist, 1.0)
    z_grid = Tarang.local_grid(zb, dist, 1.0)

    kx = 2pi / (2pi)  # = 1.0
    # f = sin(kx*x) * z^2  (trig in Fourier dir, polynomial in Chebyshev dir)
    f_data = Tarang.get_grid_data(f)
    for i in eachindex(x_grid)
        for j in eachindex(z_grid)
            f_data[i, j] = sin(kx * x_grid[i]) * z_grid[j]^2
        end
    end
    f.current_layout = :g

    grad_op = CartesianGradient(f, coords)
    g = evaluate(grad_op, :g)

    # df/dx = kx*cos(kx*x)*z^2, df/dz = sin(kx*x)*2*z
    expected_gx = zeros(length(x_grid), length(z_grid))
    expected_gz = zeros(length(x_grid), length(z_grid))
    for i in eachindex(x_grid)
        for j in eachindex(z_grid)
            expected_gx[i, j] = kx * cos(kx * x_grid[i]) * z_grid[j]^2
            expected_gz[i, j] = sin(kx * x_grid[i]) * 2 * z_grid[j]
        end
    end

    @test isapprox(Tarang.get_grid_data(g.components[1]), expected_gx; rtol=1e-6)
    @test isapprox(Tarang.get_grid_data(g.components[2]), expected_gz; rtol=1e-6)
end
