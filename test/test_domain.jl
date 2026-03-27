"""
Comprehensive test suite for domain.jl

Tests:
1. Domain construction with various basis types
2. Domain property accessors (bases_by_axis, full_bases, etc.)
3. Geometry functions (volume, shapes, spacing)
4. Grid coordinates and integration weights
5. Gauss-Legendre quadrature accuracy
6. Meshgrid generation
7. Domain queries and iteration
8. Performance statistics and caching
"""

using Test
using LinearAlgebra

@testset "Domain Module" begin
    using Tarang

    # ============================================================================
    # Domain Construction Tests
    # ============================================================================

    @testset "Domain Construction" begin
        @testset "1D Fourier Domain" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π), dealias=1.5)

            domain = Domain(dist, (xb,))

            @test domain.dim == 1
            @test length(domain.bases) == 1
            @test domain.bases[1] === xb
            @test architecture(domain) isa CPU
        end

        @testset "1D Chebyshev Domain" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = ChebyshevT(coords["x"]; size=16, bounds=(-1.0, 1.0), dealias=1.0)

            domain = Domain(dist, (xb,))

            @test domain.dim == 1
            @test domain.bases[1] isa ChebyshevT
        end

        @testset "2D Fourier-Fourier Domain" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π), dealias=1.5)
            yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 2π), dealias=1.5)

            domain = Domain(dist, (xb, yb))

            @test domain.dim == 2
            @test length(domain.bases) == 2
        end

        @testset "2D Fourier-Chebyshev Domain" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
            xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π), dealias=1.5)
            yb = ChebyshevT(coords["y"]; size=16, bounds=(-1.0, 1.0), dealias=1.0)

            domain = Domain(dist, (xb, yb))

            @test domain.dim == 2
            @test domain.bases[1] isa RealFourier
            @test domain.bases[2] isa ChebyshevT
        end

        @testset "3D Fourier-Fourier-Chebyshev Domain" begin
            coords = CartesianCoordinates("x", "y", "z")
            dist = Distributor(coords; mesh=(1, 1, 1), dtype=Float64)
            xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
            yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
            zb = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))

            domain = Domain(dist, (xb, yb, zb))

            @test domain.dim == 3
            @test length(domain.bases) == 3
        end

        @testset "Vararg Constructor" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
            xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
            yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))

            # Test vararg constructor
            domain = Domain(dist, xb, yb)

            @test domain.dim == 2
            @test length(domain.bases) == 2
        end

        @testset "Overlapping Bases Error" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb1 = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
            xb2 = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π))

            # Should throw error for overlapping bases
            @test_throws ArgumentError Domain(dist, (xb1, xb2))
        end
    end

    # ============================================================================
    # Domain Property Tests
    # ============================================================================

    @testset "Domain Properties" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π), dealias=1.5)
        yb = ChebyshevT(coords["y"]; size=6, bounds=(-1.0, 1.0), dealias=1.0)
        domain = Domain(dist, (xb, yb))

        @testset "bases_by_axis" begin
            axes_map = Tarang.bases_by_axis(domain)
            @test axes_map[0] === xb
            @test axes_map[1] === yb
        end

        @testset "full_bases" begin
            full = Tarang.full_bases(domain)
            @test length(full) == 2
            @test full[1] === xb
            @test full[2] === yb
        end

        @testset "bases_by_coord" begin
            coord_map = Tarang.bases_by_coord(domain)
            # The mapping maps coordinates to their associated basis
            # Note: bases_by_coord also maps the coordinate system itself
            @test haskey(coord_map, coords["x"])
            @test haskey(coord_map, coords["y"])
            # Both coordinates map to their respective bases
            @test coord_map[coords["x"]] !== nothing
            @test coord_map[coords["y"]] !== nothing
        end

        @testset "dealias" begin
            d = Tarang.dealias(domain)
            @test d == (1.5, 1.0)
        end

        @testset "constant and nonconstant" begin
            c = Tarang.constant(domain)
            nc = Tarang.nonconstant(domain)
            @test c == (false, false)
            @test nc == (true, true)
        end

        @testset "mode_dependence" begin
            md = Tarang.mode_dependence(domain)
            @test md == (true, true)
        end

        @testset "dim" begin
            @test Tarang.dim(domain) == 2
        end

        @testset "get_basis" begin
            @test Tarang.get_basis(domain, coords["x"]) === xb
            @test Tarang.get_basis(domain, coords["y"]) === yb
            @test Tarang.get_basis(domain, 0) === xb
            @test Tarang.get_basis(domain, 1) === yb
        end

        @testset "get_basis_subaxis" begin
            @test Tarang.get_basis_subaxis(domain, coords["x"]) == 0
            @test Tarang.get_basis_subaxis(domain, coords["y"]) == 0
        end

        @testset "get_coord" begin
            @test Tarang.get_coord(domain, "x") == coords["x"]
            @test Tarang.get_coord(domain, "y") == coords["y"]
            @test_throws ArgumentError Tarang.get_coord(domain, "z")
        end

        @testset "enumerate_unique_bases" begin
            pairs = Tarang.enumerate_unique_bases(domain)
            @test length(pairs) == 2
            @test pairs[1] == (0, xb)
            @test pairs[2] == (1, yb)
        end

        @testset "substitute_basis" begin
            new_yb = ChebyshevT(coords["y"]; size=10, bounds=(-1.0, 1.0))
            new_domain = Tarang.substitute_basis(domain, yb, new_yb)
            @test new_domain.bases[2].meta.size == 10
        end
    end

    # ============================================================================
    # Geometry Tests
    # ============================================================================

    @testset "Domain Geometry" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=16, bounds=(0.0, 2.0))
        yb = RealFourier(coords["y"]; size=16, bounds=(0.0, 3.0))
        domain = Domain(dist, (xb, yb))

        @testset "volume" begin
            vol = Tarang.volume(domain)
            @test vol ≈ 6.0  # 2.0 * 3.0
        end

        @testset "domain_volume" begin
            vol = Tarang.domain_volume(domain)
            @test vol ≈ 6.0
        end

        @testset "global_shape" begin
            gs_grid = Tarang.global_shape(domain, :g)
            @test gs_grid == (16, 16)

            gs_coef = Tarang.global_shape(domain, :c)
            # Both bases are RealFourier, so both get N/2+1 = 9
            @test gs_coef == (9, 9)
        end

        @testset "coefficient_shape" begin
            cs = Tarang.coefficient_shape(domain)
            # Both RealFourier bases: N/2+1 = 9 for each
            @test cs == (9, 9)
        end

        @testset "grid_spacing" begin
            spacings = Tarang.grid_spacing(domain)
            @test spacings[1] ≈ 2.0 / 16  # Lx / Nx
            @test spacings[2] ≈ 3.0 / 16  # Ly / Ny
        end
    end

    # ============================================================================
    # Grid Coordinates Tests
    # ============================================================================

    @testset "Grid Coordinates" begin
        @testset "Fourier Grid" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
            domain = Domain(dist, (xb,))

            grid_coords = Tarang.get_grid_coordinates(domain)
            @test haskey(grid_coords, "x")
            @test length(grid_coords["x"]) == 8
            @test grid_coords["x"][1] ≈ 0.0
            @test grid_coords["x"][end] ≈ 2π - 2π/8 atol=1e-10
        end

        @testset "Chebyshev Grid" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = ChebyshevT(coords["x"]; size=8, bounds=(-1.0, 1.0))
            domain = Domain(dist, (xb,))

            grid_coords = Tarang.get_grid_coordinates(domain)
            @test haskey(grid_coords, "x")
            @test length(grid_coords["x"]) == 8
            # Chebyshev-Gauss-Lobatto points should be in [-1, 1]
            @test all(-1.0 .<= grid_coords["x"] .<= 1.0)
        end

        @testset "Coordinate Caching" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
            domain = Domain(dist, (xb,))

            # First call
            grid1 = Tarang.get_grid_coordinates(domain)

            # Second call - should return same data from cache
            grid2 = Tarang.get_grid_coordinates(domain)

            # The cached data should be the same
            @test grid1["x"] == grid2["x"]

            # Verify cache is populated
            @test !isempty(domain.grid_coordinates)
        end
    end

    # ============================================================================
    # Integration Weights Tests
    # ============================================================================

    @testset "Integration Weights" begin
        @testset "Fourier Weights" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            L = 2π
            N = 16
            xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
            domain = Domain(dist, (xb,))

            weights = Tarang.integration_weights(domain)
            @test length(weights) == 1
            @test all(weights[1] .≈ L / N)

            # Sum of weights should equal domain length
            @test sum(weights[1]) ≈ L
        end

        @testset "Chebyshev Weights (Clenshaw-Curtis)" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = ChebyshevT(coords["x"]; size=8, bounds=(-1.0, 1.0))
            domain = Domain(dist, (xb,))

            weights = Tarang.integration_weights(domain)
            @test length(weights) == 1
            # Sum of weights should approximate interval length
            @test sum(weights[1]) ≈ 2.0 atol=0.1
        end

        @testset "Legendre Weights (Gauss-Legendre)" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = Legendre(coords["x"]; size=8, bounds=(-1.0, 1.0))
            domain = Domain(dist, (xb,))

            weights = Tarang.integration_weights(domain)
            @test length(weights) == 1
            # Gauss-Legendre weights sum to 2 on [-1, 1]
            @test sum(weights[1]) ≈ 2.0 atol=1e-10
        end
    end

    # ============================================================================
    # Gauss-Legendre Quadrature Tests
    # ============================================================================

    @testset "Gauss-Legendre Quadrature" begin
        @testset "Weights Sum" begin
            for N in [2, 4, 8, 16, 32]
                w = Tarang.gauss_legendre_weights(N)
                @test length(w) == N
                # Weights should sum to 2 (length of [-1,1])
                @test sum(w) ≈ 2.0 atol=1e-12
            end
        end

        @testset "Polynomial Integration Accuracy" begin
            # Gauss-Legendre with N points is exact for polynomials of degree 2N-1
            N = 5
            w = Tarang.gauss_legendre_weights(N)

            # Compute nodes (roots of Legendre polynomial)
            function gauss_legendre_nodes(N::Int)
                x = zeros(N)
                m = div(N + 1, 2)
                for i in 1:m
                    z = cos(π * (i - 0.25) / (N + 0.5))
                    for _ in 1:100
                        p1, p2 = 1.0, 0.0
                        for j in 1:N
                            p3, p2 = p2, p1
                            p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
                        end
                        pp = N * (z * p1 - p2) / (z^2 - 1)
                        z_old = z
                        z = z_old - p1 / pp
                        abs(z - z_old) < 1e-15 && break
                    end
                    x[i] = -z
                    x[N + 1 - i] = z
                end
                return x
            end

            nodes = gauss_legendre_nodes(N)

            # Test ∫₋₁¹ x² dx = 2/3
            f_vals = nodes .^ 2
            integral = dot(w, f_vals)
            @test integral ≈ 2/3 atol=1e-12

            # Test ∫₋₁¹ x⁴ dx = 2/5
            f_vals = nodes .^ 4
            integral = dot(w, f_vals)
            @test integral ≈ 2/5 atol=1e-12

            # Test ∫₋₁¹ x⁶ dx = 2/7 (still within exactness for N=5)
            f_vals = nodes .^ 6
            integral = dot(w, f_vals)
            @test integral ≈ 2/7 atol=1e-12
        end
    end

    # ============================================================================
    # Meshgrid Tests
    # ============================================================================

    @testset "Meshgrid Generation" begin
        @testset "1D Meshgrid" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 1.0))
            domain = Domain(dist, (xb,))

            mesh = Tarang.create_meshgrid(domain)
            @test haskey(mesh, "x")
            @test mesh["x"] isa Vector
        end

        @testset "2D Meshgrid" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
            xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 1.0))
            yb = RealFourier(coords["y"]; size=3, bounds=(0.0, 2.0))
            domain = Domain(dist, (xb, yb))

            mesh = Tarang.create_meshgrid(domain)
            @test haskey(mesh, "x")
            @test haskey(mesh, "y")
            # Both arrays should have shape (nx, ny) = (4, 3)
            @test size(mesh["x"]) == (4, 3)
            @test size(mesh["y"]) == (4, 3)
        end

        @testset "3D Meshgrid" begin
            coords = CartesianCoordinates("x", "y", "z")
            dist = Distributor(coords; mesh=(1, 1, 1), dtype=Float64)
            xb = RealFourier(coords["x"]; size=4, bounds=(0.0, 1.0))
            yb = RealFourier(coords["y"]; size=3, bounds=(0.0, 1.0))
            zb = RealFourier(coords["z"]; size=2, bounds=(0.0, 1.0))
            domain = Domain(dist, (xb, yb, zb))

            mesh = Tarang.create_meshgrid(domain)
            @test haskey(mesh, "x")
            @test haskey(mesh, "y")
            @test haskey(mesh, "z")
            @test size(mesh["x"]) == (4, 3, 2)
        end
    end

    # ============================================================================
    # Domain Query Tests
    # ============================================================================

    @testset "Domain Queries" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = ChebyshevT(coords["y"]; size=8, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, yb))

        @testset "is_compound" begin
            @test Tarang.is_compound(domain) == true

            # Single basis domain
            coords1 = CartesianCoordinates("x")
            dist1 = Distributor(coords1; mesh=(1,), dtype=Float64)
            domain1 = Domain(dist1, (xb,))
            @test Tarang.is_compound(domain1) == false
        end

        @testset "has_basis" begin
            @test Tarang.has_basis(domain, xb) == true
            @test Tarang.has_basis(domain, yb) == true

            other_basis = RealFourier(coords["x"]; size=16, bounds=(0.0, 4π))
            @test Tarang.has_basis(domain, other_basis) == false
        end

        @testset "basis_names" begin
            names = Tarang.basis_names(domain)
            @test "x" in names
            @test "y" in names
        end
    end

    # ============================================================================
    # Iteration Interface Tests
    # ============================================================================

    @testset "Iteration Interface" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; mesh=(1, 1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = RealFourier(coords["y"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, yb, zb))

        @testset "length" begin
            @test length(domain) == 3
        end

        @testset "getindex" begin
            @test domain[1] === xb
            @test domain[2] === yb
            @test domain[3] === zb
        end

        @testset "iterate" begin
            bases_collected = collect(domain)
            @test length(bases_collected) == 3
            @test bases_collected[1] === xb
            @test bases_collected[2] === yb
            @test bases_collected[3] === zb
        end

        @testset "for loop" begin
            count = 0
            for basis in domain
                count += 1
            end
            @test count == 3
        end
    end

    # ============================================================================
    # Cache Tests
    # ============================================================================

    @testset "Cache Operations" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = ChebyshevT(coords["y"]; size=8, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, yb))

        # Generate some cached data
        _ = Tarang.dealias(domain)
        _ = Tarang.constant(domain)
        _ = Tarang.get_grid_coordinates(domain)
        _ = Tarang.integration_weights(domain)

        @test !isempty(domain.attribute_cache)
        @test !isempty(domain.grid_coordinates)
        @test !isempty(domain.integration_weights_cache)

        # Clear cache
        Tarang.clear_domain_cache!(domain)

        @test isempty(domain.attribute_cache)
        @test isempty(domain.grid_coordinates)
        @test isempty(domain.integration_weights_cache)
    end

    # ============================================================================
    # Memory Info Tests
    # ============================================================================

    @testset "Memory Info" begin
        coords = CartesianCoordinates("x", "y")
        dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        yb = ChebyshevT(coords["y"]; size=8, bounds=(-1.0, 1.0))
        domain = Domain(dist, (xb, yb))

        # Populate caches
        _ = Tarang.get_grid_coordinates(domain)
        _ = Tarang.integration_weights(domain)

        mem_info = Tarang.get_domain_memory_info(domain)

        @test haskey(mem_info, :total_memory)
        @test haskey(mem_info, :available_memory)
        @test haskey(mem_info, :used_memory)
        @test haskey(mem_info, :domain_memory)
        @test haskey(mem_info, :memory_utilization)

        @test mem_info.domain_memory > 0
        @test 0.0 <= mem_info.memory_utilization <= 1.0
    end

    # ============================================================================
    # Performance Statistics Tests
    # ============================================================================

    @testset "Performance Statistics" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        domain = Domain(dist, (xb,))

        @test domain.performance_stats.coordinate_generations == 0
        @test domain.performance_stats.weight_computations == 0

        _ = Tarang.get_grid_coordinates(domain)
        @test domain.performance_stats.coordinate_generations >= 1

        _ = Tarang.integration_weights(domain)
        @test domain.performance_stats.weight_computations >= 1

        @test domain.performance_stats.total_time >= 0.0
    end
end

println("All domain tests passed!")
