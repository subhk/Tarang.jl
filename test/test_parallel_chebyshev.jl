"""
Test parallel Chebyshev-Chebyshev transforms.

Run with:
    mpiexec -n 4 julia --project test/test_parallel_chebyshev.jl
"""

using Test
using MPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# Only run parallel tests if we have multiple processes
if nprocs > 1
    using Tarang

    @testset "Parallel Chebyshev-Chebyshev" begin

        @testset "2D Domain Creation" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; comm=comm)

            Nx, Ny = 32, 32
            xb = ChebyshevT(coords["x"]; size=Nx, bounds=(0.0, 1.0))
            yb = ChebyshevT(coords["y"]; size=Ny, bounds=(0.0, 1.0))

            domain = Domain(dist, (xb, yb))

            @test length(domain.bases) == 2
            @test isa(domain.bases[1], ChebyshevT)
            @test isa(domain.bases[2], ChebyshevT)

            if rank == 0
                @info "Created 2D Chebyshev-Chebyshev domain"
                @info "  Global shape: ($Nx, $Ny)"
                @info "  Process mesh: $(dist.mesh)"
            end
        end

        @testset "Transform Setup" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; comm=comm)

            Nx, Ny = 32, 32
            xb = ChebyshevT(coords["x"]; size=Nx, bounds=(0.0, 1.0))
            yb = ChebyshevT(coords["y"]; size=Ny, bounds=(0.0, 1.0))

            domain = Domain(dist, (xb, yb))

            # Check that parallel Chebyshev transform is set up
            has_parallel_cheb = false
            for transform in dist.transforms
                if isa(transform, Tarang.ParallelChebyshevTransform)
                    has_parallel_cheb = true
                    @test transform.ndim == 2
                    @test length(transform.forward_plans) == 2
                    @test length(transform.backward_plans) == 2
                end
            end

            @test has_parallel_cheb

            if rank == 0
                @info "Parallel Chebyshev transform setup verified"
            end
        end

        @testset "Field Operations" begin
            coords = CartesianCoordinates("x", "y")
            dist = Distributor(coords; comm=comm)

            Nx, Ny = 32, 32
            xb = ChebyshevT(coords["x"]; size=Nx, bounds=(0.0, 1.0))
            yb = ChebyshevT(coords["y"]; size=Ny, bounds=(0.0, 1.0))

            T = ScalarField(dist, "T", (xb, yb))

            # Initialize with a simple function
            # T(x,y) = x^2 + y^2
            T_data = Tarang.get_grid_data(T)
            if T_data !== nothing
                x_local = xb.meta.grid_points
                y_local = yb.meta.grid_points  # This will be partial in parallel

                for j in 1:size(T_data, 2)
                    for i in 1:size(T_data, 1)
                        T_data[i, j] = x_local[i]^2
                    end
                end
            end

            if rank == 0
                @info "Field initialization complete"
            end
        end

        @testset "is_pencil_compatible" begin
            coords = CartesianCoordinates("x", "y")

            xb_cheb = ChebyshevT(coords["x"]; size=32, bounds=(0.0, 1.0))
            yb_cheb = ChebyshevT(coords["y"]; size=32, bounds=(0.0, 1.0))
            xb_four = RealFourier(coords["x"]; size=32, bounds=(0.0, 2π))

            # 2D Chebyshev-Chebyshev should be compatible
            @test Tarang.is_pencil_compatible((xb_cheb, yb_cheb)) == true

            # 2D Fourier-Chebyshev should be compatible
            @test Tarang.is_pencil_compatible((xb_four, yb_cheb)) == true

            # 1D should not be compatible
            @test Tarang.is_pencil_compatible((xb_cheb,)) == false

            if rank == 0
                @info "is_pencil_compatible tests passed"
            end
        end

    end

    if rank == 0
        println("\nAll parallel Chebyshev-Chebyshev tests passed!")
    end

else
    if rank == 0
        @warn "Parallel tests require multiple MPI processes. Run with: mpiexec -n 4 julia ..."
    end
end

MPI.Finalize()
