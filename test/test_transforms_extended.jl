"""
Extended transform tests for Tarang.jl.

Tests cover:
- forward_transform! on periodic fields
- backward_transform! roundtrip fidelity
- ensure_layout! switching between :g and :c
- Chebyshev (ChebyshevT) basis transforms
- Mixed Fourier-Chebyshev transforms (ChannelDomain)
- Roundtrip preservation of data to machine precision
"""

using Test
using Tarang

# ============================================================================
# Forward transform on periodic fields
# ============================================================================

@testset "forward_transform! on periodic 2D field" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    Tarang.get_grid_data(f) .= @. sin(x) * cos(y)
    f.current_layout = :g

    forward_transform!(f)

    # After forward transform, field should be in coefficient space
    @test f.current_layout == :c

    # Coefficient data should not be all zeros for a nonzero input
    coeff = Tarang.get_coeff_data(f)
    @test !all(iszero, coeff)
end

@testset "forward_transform! on periodic 1D field" begin
    domain = PeriodicDomain(16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x = mesh["x"]

    Tarang.get_grid_data(f) .= @. sin(3x) + 0.5
    f.current_layout = :g

    forward_transform!(f)
    @test f.current_layout == :c
    @test !all(iszero, Tarang.get_coeff_data(f))
end

# ============================================================================
# Backward transform roundtrip
# ============================================================================

@testset "backward_transform! roundtrip 2D Fourier" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    Tarang.get_grid_data(f) .= @. sin(2x) * cos(3y) + 0.5
    original = copy(Tarang.get_grid_data(f))
    f.current_layout = :g

    forward_transform!(f)
    backward_transform!(f)

    @test f.current_layout == :g
    @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-12)
end

@testset "backward_transform! roundtrip 1D Fourier" begin
    domain = PeriodicDomain(32)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x = mesh["x"]

    Tarang.get_grid_data(f) .= @. cos(4x) - 0.3 * sin(2x) + 1.0
    original = copy(Tarang.get_grid_data(f))
    f.current_layout = :g

    forward_transform!(f)
    backward_transform!(f)

    @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-12)
end

@testset "backward_transform! roundtrip 3D Fourier" begin
    domain = PeriodicDomain(16, 16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y, z = mesh["x"], mesh["y"], mesh["z"]

    Tarang.get_grid_data(f) .= @. sin(x) * cos(y) * sin(2z) + 0.1
    original = copy(Tarang.get_grid_data(f))
    f.current_layout = :g

    forward_transform!(f)
    backward_transform!(f)

    @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-12)
end

# ============================================================================
# ensure_layout! switching between :g and :c
# ============================================================================

@testset "ensure_layout! switches correctly" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    Tarang.get_grid_data(f) .= @. sin(x) * cos(y) + 1.0
    original = copy(Tarang.get_grid_data(f))
    f.current_layout = :g

    # Switch to coefficient space
    ensure_layout!(f, :c)
    @test f.current_layout == :c

    # Switch back to grid space
    ensure_layout!(f, :g)
    @test f.current_layout == :g
    @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-12)
end

@testset "ensure_layout! is idempotent" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    Tarang.get_grid_data(f) .= @. cos(2x) * sin(y)
    f.current_layout = :g

    ensure_layout!(f, :g)  # already in :g, should be no-op
    @test f.current_layout == :g

    ensure_layout!(f, :c)
    coeff_snapshot = copy(Tarang.get_coeff_data(f))
    ensure_layout!(f, :c)  # already in :c, should be no-op
    @test f.current_layout == :c
    @test Tarang.get_coeff_data(f) == coeff_snapshot
end

@testset "ensure_layout! on VectorField" begin
    domain = PeriodicDomain(16, 16)
    u = VectorField(domain, "u")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    Tarang.get_grid_data(u.components[1]) .= @. sin(x) + 0.0 * y
    Tarang.get_grid_data(u.components[2]) .= @. 0.0 * x + cos(y)
    for c in u.components
        c.current_layout = :g
    end

    orig_1 = copy(Tarang.get_grid_data(u.components[1]))
    orig_2 = copy(Tarang.get_grid_data(u.components[2]))

    ensure_layout!(u, :c)
    for c in u.components
        @test c.current_layout == :c
    end

    ensure_layout!(u, :g)
    for c in u.components
        @test c.current_layout == :g
    end

    @test isapprox(Tarang.get_grid_data(u.components[1]), orig_1; rtol=1e-10, atol=1e-12)
    @test isapprox(Tarang.get_grid_data(u.components[2]), orig_2; rtol=1e-10, atol=1e-12)
end

# ============================================================================
# ChebyshevT basis transforms
# ============================================================================

@testset "Chebyshev roundtrip (ChebyshevDomain 1D)" begin
    domain = ChebyshevDomain(16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x = mesh["x"]

    # Polynomial that Chebyshev can represent exactly
    Tarang.get_grid_data(f) .= @. x^2 - 0.5 * x + 0.3
    original = copy(Tarang.get_grid_data(f))
    f.current_layout = :g

    forward_transform!(f)
    @test f.current_layout == :c

    backward_transform!(f)
    @test f.current_layout == :g
    @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-10)
end

@testset "Chebyshev roundtrip (ChebyshevDomain 2D)" begin
    domain = ChebyshevDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    # Low-order polynomial for exact representation
    Tarang.get_grid_data(f) .= @. x^2 * y + 0.5 * y^2 - 0.3 * x + 0.1
    original = copy(Tarang.get_grid_data(f))
    f.current_layout = :g

    forward_transform!(f)
    backward_transform!(f)

    @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-10)
end

# ============================================================================
# Mixed Fourier-Chebyshev transforms (ChannelDomain)
# ============================================================================

@testset "ChannelDomain roundtrip (Fourier-Chebyshev)" begin
    domain = ChannelDomain(16, 16; Lx=2pi, Lz=2.0)
    f = ScalarField(domain, "f")

    dist = domain.dist
    xb, zb = domain.bases
    x_grid = Tarang.local_grid(xb, dist, 1.0)
    z_grid = Tarang.local_grid(zb, dist, 1.0)

    # f = sin(x) * z^2: trig for Fourier, polynomial for Chebyshev
    f_data = Tarang.get_grid_data(f)
    for i in eachindex(x_grid)
        for j in eachindex(z_grid)
            f_data[i, j] = sin(x_grid[i]) * z_grid[j]^2
        end
    end
    original = copy(f_data)
    f.current_layout = :g

    forward_transform!(f)
    @test f.current_layout == :c

    backward_transform!(f)
    @test f.current_layout == :g
    @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-10)
end

@testset "ChannelDomain ensure_layout! roundtrip" begin
    domain = ChannelDomain(16, 16; Lx=2pi, Lz=2.0)
    f = ScalarField(domain, "f")

    dist = domain.dist
    xb, zb = domain.bases
    x_grid = Tarang.local_grid(xb, dist, 1.0)
    z_grid = Tarang.local_grid(zb, dist, 1.0)

    f_data = Tarang.get_grid_data(f)
    for i in eachindex(x_grid)
        for j in eachindex(z_grid)
            f_data[i, j] = cos(x_grid[i]) * z_grid[j]
        end
    end
    original = copy(f_data)
    f.current_layout = :g

    ensure_layout!(f, :c)
    @test f.current_layout == :c

    ensure_layout!(f, :g)
    @test f.current_layout == :g
    @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-10, atol=1e-10)
end

# ============================================================================
# Transform preserves data: multiple roundtrips
# ============================================================================

@testset "Multiple roundtrips preserve data" begin
    domain = PeriodicDomain(16, 16)
    f = ScalarField(domain, "f")
    mesh = Tarang.create_meshgrid(domain)
    x, y = mesh["x"], mesh["y"]

    Tarang.get_grid_data(f) .= @. sin(x) * cos(2y) + 0.7
    original = copy(Tarang.get_grid_data(f))
    f.current_layout = :g

    # Perform three roundtrips
    for _ in 1:3
        forward_transform!(f)
        backward_transform!(f)
    end

    @test isapprox(Tarang.get_grid_data(f), original; rtol=1e-9, atol=1e-11)
end
