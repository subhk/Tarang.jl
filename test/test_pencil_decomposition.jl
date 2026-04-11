"""
PencilDecomposition Tests - Unit tests for pencil decomposition data structure

Run with: julia --project test/test_pencil_decomposition.jl

These tests verify the PencilDecomposition struct and helper functions
for 2D pencil decomposition used in distributed GPU transforms.

Note: PencilDecomposition is part of the CUDA extension but only requires MPI,
so we include it directly for testing when CUDA is not available.
"""

using Test
using MPI

# Initialize MPI for testing
if !MPI.Initialized()
    MPI.Init()
end

# Try to load CUDA extension for PencilDecomposition
const _HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

if _HAS_CUDA
    # Load via extension when CUDA is available
    using Tarang
    using TarangCUDAExt
    using TarangCUDAExt: PencilDecomposition, rank_to_grid, grid_to_rank
    using TarangCUDAExt: compute_pencil_shapes, current_orientation, set_orientation!, current_local_shape
else
    # Include directly for MPI-only testing
    include(joinpath(@__DIR__, "..", "ext", "cuda", "pencil.jl"))
end

@testset "PencilDecomposition" begin
    @testset "Shape calculations" begin
        # Test with 4 processes in 2x2 grid
        global_shape = (64, 64, 64)
        proc_grid = (2, 2)

        # Rank 0 should get correct local shapes
        pencil = PencilDecomposition(global_shape, proc_grid, 0, MPI.COMM_SELF)

        # X-pencil: full X, split Y by P1, split Z by P2
        @test pencil.x_pencil_shape == (64, 32, 32)

        # Y-pencil: split X by P1, full Y, split Z by P2
        @test pencil.y_pencil_shape == (32, 64, 32)

        # Z-pencil: split X by P1, split Y by P2, full Z
        @test pencil.z_pencil_shape == (32, 32, 64)
    end

    @testset "Rank to grid coordinates" begin
        proc_grid = (2, 2)

        # Rank 0 -> (0, 0)
        @test rank_to_grid(0, proc_grid) == (0, 0)
        # Rank 1 -> (0, 1)
        @test rank_to_grid(1, proc_grid) == (0, 1)
        # Rank 2 -> (1, 0)
        @test rank_to_grid(2, proc_grid) == (1, 0)
        # Rank 3 -> (1, 1)
        @test rank_to_grid(3, proc_grid) == (1, 1)
    end

    @testset "Grid to rank conversion" begin
        proc_grid = (2, 2)

        # (0, 0) -> Rank 0
        @test grid_to_rank(0, 0, proc_grid) == 0
        # (0, 1) -> Rank 1
        @test grid_to_rank(0, 1, proc_grid) == 1
        # (1, 0) -> Rank 2
        @test grid_to_rank(1, 0, proc_grid) == 2
        # (1, 1) -> Rank 3
        @test grid_to_rank(1, 1, proc_grid) == 3
    end

    @testset "Uneven division handling" begin
        # Test when N % P != 0
        global_shape = (65, 67, 70)
        proc_grid = (2, 3)

        # For rank 0 (grid coords 0,0)
        pencil = PencilDecomposition(global_shape, proc_grid, 0, MPI.COMM_SELF)

        # Check that shapes are computed correctly
        Nx, Ny, Nz = global_shape
        P1, P2 = proc_grid

        # For rank 0 (row=0, col=0):
        # X splits by P1=2: 65/2 = 32 remainder 1, so rank 0 gets 33
        # Y splits by P1=2 for x_pencil: 67/2 = 33 remainder 1, so rank 0 gets 34
        # Y splits by P2=3 for z_pencil: 67/3 = 22 remainder 1, so col 0 gets 23
        # Z splits by P2=3: 70/3 = 23 remainder 1, so col 0 gets 24

        @test pencil.x_pencil_shape[1] == 65  # Full X
        @test pencil.y_pencil_shape[2] == 67  # Full Y
        @test pencil.z_pencil_shape[3] == 70  # Full Z
    end

    @testset "Orientation accessor functions" begin
        global_shape = (64, 64, 64)
        proc_grid = (2, 2)
        pencil = PencilDecomposition(global_shape, proc_grid, 0, MPI.COMM_SELF)

        # Default orientation is z_pencil
        @test current_orientation(pencil) == :z_pencil
        @test current_local_shape(pencil) == pencil.z_pencil_shape

        # Change orientation
        set_orientation!(pencil, :y_pencil)
        @test current_orientation(pencil) == :y_pencil
        @test current_local_shape(pencil) == pencil.y_pencil_shape

        set_orientation!(pencil, :x_pencil)
        @test current_orientation(pencil) == :x_pencil
        @test current_local_shape(pencil) == pencil.x_pencil_shape
    end

    @testset "Struct field values" begin
        global_shape = (64, 64, 64)
        proc_grid = (2, 2)
        rank = 0

        pencil = PencilDecomposition(global_shape, proc_grid, rank, MPI.COMM_SELF)

        @test pencil.global_shape == global_shape
        @test pencil.proc_grid == proc_grid
        @test pencil.rank == rank
        @test pencil.grid_coords == (0, 0)
    end
end

println("PencilDecomposition tests completed!")
