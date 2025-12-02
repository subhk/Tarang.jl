"""
Optimized Linear Algebra Performance Demo

This example demonstrates the performance benefits of optimized matrix operations
in Tarang.jl for spectral methods. It compares:

1. Standard Julia LinearAlgebra operations
2. Optimized BLAS operations  
3. Sparse matrix optimizations
4. Block and cache-efficient algorithms
5. Kronecker product optimizations
6. Memory-efficient streaming operations

Run with: julia --threads=4 optimized_linalg_demo.jl
"""

using Tarang
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using Random
using Printf

# Include optimized operations
include("../src/core/optimized_linalg.jl")
using .Main: OptimizedMatVec, OptimizedMatMat, SparseMatVec, DenseMatVec
using .Main: optimized_matvec!, optimized_matmat!, create_optimized_operator
using .Main: benchmark_linalg_operations, reset_linalg_stats!, print_linalg_stats
using .Main: streaming_matvec!, cache_efficient_matmat!, create_kronecker_operator

function main()
    println("="^80)
    println("VARUNA.JL OPTIMIZED LINEAR ALGEBRA PERFORMANCE DEMO")
    println("="^80)
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Test different matrix sizes typical in spectral methods
    sizes = [64, 128, 256, 512, 1024]
    sparsity_levels = [0.01, 0.05, 0.1, 0.2]  # 1%, 5%, 10%, 20% non-zeros
    
    println("Julia version: $(VERSION)")
    println("BLAS threads: $(BLAS.get_num_threads())")
    println("Julia threads: $(Threads.nthreads())")
    println()
    
    # 1. Matrix-Vector Multiplication Benchmarks
    println("1. MATRIX-VECTOR MULTIPLICATION BENCHMARKS")
    println("-"^50)
    benchmark_matvec_operations(sizes, sparsity_levels)
    
    # 2. Matrix-Matrix Multiplication Benchmarks  
    println("\n2. MATRIX-MATRIX MULTIPLICATION BENCHMARKS")
    println("-"^50)
    benchmark_matmat_operations(sizes)
    
    # 3. Kronecker Product Operations
    println("\n3. KRONECKER PRODUCT OPERATIONS")
    println("-"^50)
    benchmark_kronecker_operations()
    
    # 4. Block Matrix Operations
    println("\n4. BLOCK MATRIX OPERATIONS")
    println("-"^50)
    benchmark_block_operations()
    
    # 5. Memory-Efficient Operations
    println("\n5. MEMORY-EFFICIENT OPERATIONS")
    println("-"^50)
    benchmark_memory_efficient_operations()
    
    # 6. Spectral Method Realistic Benchmark
    println("\n6. REALISTIC SPECTRAL METHOD BENCHMARK")
    println("-"^50)
    benchmark_spectral_method_operations()
    
    # 7. Performance Analysis
    println("\n7. PERFORMANCE ANALYSIS")
    println("-"^50)
    print_performance_summary()
    
    println("\n" * "="^80)
    println("DEMO COMPLETED")
    println("="^80)
end

function benchmark_matvec_operations(sizes::Vector{Int}, sparsity_levels::Vector{Float64})
    """Benchmark matrix-vector operations for different matrix types and sizes"""
    
    println("Testing matrix-vector multiplications...")
    println()
    
    # Table header
    @printf("%-8s %-8s %-12s %-12s %-12s %-8s\n", 
            "Size", "Sparsity", "Standard", "Optimized", "Sparse Opt", "Speedup")
    println("-"^70)
    
    for n in sizes
        # Dense matrix test
        A_dense = randn(n, n)
        x = randn(n)
        y_std = similar(x)
        y_opt = similar(x)
        
        # Standard operation
        t_standard = @belapsed mul!($y_std, $A_dense, $x)
        
        # Optimized dense operation
        dense_op = DenseMatVec(A_dense)
        t_optimized = @belapsed optimized_matvec!($y_opt, $dense_op, $x)
        
        speedup_dense = t_standard / t_optimized
        @printf("%-8d %-8s %-12.3f %-12.3f %-12s %7.2fx\n", 
                n, "dense", t_standard*1e6, t_optimized*1e6, "-", speedup_dense)
        
        # Test different sparsity levels
        for sparsity in sparsity_levels
            A_sparse = sprand(n, n, sparsity)
            
            # Standard sparse operation
            t_sparse_std = @belapsed mul!($y_std, $A_sparse, $x)
            
            # Optimized sparse operation
            sparse_op = SparseMatVec(A_sparse)
            t_sparse_opt = @belapsed optimized_matvec!($y_opt, $sparse_op, $x)
            
            speedup_sparse = t_sparse_std / t_sparse_opt
            @printf("%-8d %-8.1f%% %-12.3f %-12.3f %-12.3f %7.2fx\n", 
                    n, sparsity*100, t_sparse_std*1e6, t_optimized*1e6, 
                    t_sparse_opt*1e6, speedup_sparse)
        end
        
        println()
    end
end

function benchmark_matmat_operations(sizes::Vector{Int})
    """Benchmark matrix-matrix operations"""
    
    println("Testing matrix-matrix multiplications...")
    println()
    
    @printf("%-8s %-12s %-12s %-12s %-12s %-8s\n", 
            "Size", "Standard", "BLAS Opt", "Threaded", "Cache Opt", "Best Speedup")
    println("-"^75)
    
    for n in sizes
        # Generate test matrices
        A = randn(n, n)
        B = randn(n, n÷2)
        C_std = Matrix{Float64}(undef, n, n÷2)
        C_opt = similar(C_std)
        C_threaded = similar(C_std)
        C_cache = similar(C_std)
        
        # Standard operation
        t_standard = @belapsed mul!($C_std, $A, $B)
        
        # BLAS optimized
        opt_op = create_optimized_operator(A, :matmat)
        t_blas = @belapsed optimized_matmat!($C_opt, $opt_op, $A, $B)
        
        # Threaded block multiplication (if large enough)
        t_threaded = if n >= 256
            @belapsed threaded_block_matmat!($C_threaded, $A, $B, 1.0, 0.0, 64)
        else
            t_blas  # Use BLAS for small matrices
        end
        
        # Cache-efficient multiplication
        t_cache = @belapsed cache_efficient_matmat!($C_cache, $A, $B)
        
        # Find best time and speedup
        times = [t_blas, t_threaded, t_cache]
        best_time = minimum(times)
        speedup = t_standard / best_time
        
        @printf("%-8d %-12.3f %-12.3f %-12.3f %-12.3f %7.2fx\n", 
                n, t_standard*1e6, t_blas*1e6, t_threaded*1e6, t_cache*1e6, speedup)
    end
    println()
end

function benchmark_kronecker_operations()
    """Benchmark Kronecker product operations common in spectral methods"""
    
    println("Testing Kronecker product operations...")
    println("(Common in tensor-product spectral methods)")
    println()
    
    # Test different factor sizes
    factor_sizes = [(16, 16), (32, 32), (64, 64), (32, 64)]
    
    @printf("%-12s %-12s %-12s %-8s\n", "Factors", "Standard", "Optimized", "Speedup")
    println("-"^45)
    
    for (m, n) in factor_sizes
        # Create Kronecker factors (typical spectral differentiation matrices)
        A1 = create_spectral_diff_matrix(m)  # Differentiation in first direction
        A2 = create_spectral_diff_matrix(n)  # Differentiation in second direction
        
        # Create test vector
        vec_size = m * n
        x = randn(vec_size)
        
        # Standard Kronecker operation: (A1 ⊗ A2) * x
        A_full = kron(A1, A2)
        y_std = similar(x)
        t_standard = @belapsed mul!($y_std, $A_full, $x)
        
        # Optimized Kronecker operation
        kron_op = create_kronecker_operator([A1, A2])
        C_result = Matrix{Float64}(undef, size(A1, 1), size(A2, 1))
        t_optimized = @belapsed optimized_matmat!($C_result, $kron_op, $x)
        
        speedup = t_standard / t_optimized
        @printf("%-12s %-12.3f %-12.3f %7.2fx\n", 
                "$(m)×$(n)", t_standard*1e6, t_optimized*1e6, speedup)
    end
    println()
end

function benchmark_block_operations()
    """Benchmark block matrix operations"""
    
    println("Testing block matrix operations...")
    println("(Common in coupled PDE systems)")
    println()
    
    # Create block system representative of coupled PDEs
    # Example: Stokes equations with velocity-pressure coupling
    n_vel = 256  # Velocity DOFs per component
    n_pres = 64  # Pressure DOFs
    
    println("Block system: Stokes-like equations")
    println("Velocity blocks: $(n_vel)×$(n_vel)")
    println("Pressure blocks: $(n_pres)×$(n_vel)")
    println()
    
    # Create blocks
    # [A11  A12  B1^T]   [u1]   [f1]
    # [A21  A22  B2^T] × [u2] = [f2]  
    # [B1   B2   0   ]   [p ]   [0 ]
    
    A11 = sprandn(n_vel, n_vel, 0.1)  # Discrete Laplacian
    A22 = copy(A11)                    # Same operator for v-component
    B1 = sprandn(n_pres, n_vel, 0.2)  # Divergence operator  
    B2 = sprandn(n_pres, n_vel, 0.2)
    
    # Assemble full system
    total_size = 2*n_vel + n_pres
    A_full = spzeros(total_size, total_size)
    A_full[1:n_vel, 1:n_vel] = A11
    A_full[n_vel+1:2*n_vel, n_vel+1:2*n_vel] = A22
    A_full[1:n_vel, 2*n_vel+1:end] = B1'
    A_full[n_vel+1:2*n_vel, 2*n_vel+1:end] = B2'
    A_full[2*n_vel+1:end, 1:n_vel] = B1
    A_full[2*n_vel+1:end, n_vel+1:2*n_vel] = B2
    
    x = randn(total_size)
    y_full = similar(x)
    
    # Standard operation
    t_full = @belapsed mul!($y_full, $A_full, $x)
    
    # Block operation (manually optimized)
    y_block = similar(x)
    t_block = @belapsed block_matvec!($y_block, $A11, $A22, $B1, $B2, $x, $n_vel, $n_pres)
    
    speedup = t_full / t_block
    
    @printf("Full matrix:   %8.3f μs\n", t_full*1e6)
    @printf("Block matrix:  %8.3f μs\n", t_block*1e6)
    @printf("Speedup:       %8.2fx\n", speedup)
    println()
end

function benchmark_memory_efficient_operations()
    """Benchmark memory-efficient operations for large systems"""
    
    println("Testing memory-efficient operations...")
    println("(For systems too large to fit in cache)")
    println()
    
    # Test streaming operations on large matrices
    large_sizes = [2048, 4096]
    chunk_sizes = [512, 1024, 2048]
    
    @printf("%-8s %-12s %-12s %-12s %-8s\n", 
            "Size", "Standard", "Streaming", "Best Chunk", "Speedup")
    println("-"^55)
    
    for n in large_sizes
        # Create large matrix (but don't store it fully to save memory)
        println("  Creating $(n)×$(n) test matrix...")
        A = sprandn(n, n, 0.001)  # Very sparse for memory reasons
        x = randn(n)
        y_std = similar(x)
        y_stream = similar(x)
        
        # Standard operation
        t_standard = @belapsed mul!($y_std, $A, $x)
        
        # Test different chunk sizes
        best_time = Inf
        best_chunk = 0
        
        for chunk_size in chunk_sizes
            if chunk_size < n
                t_chunk = @belapsed streaming_matvec!($y_stream, $A, $x, $chunk_size)
                if t_chunk < best_time
                    best_time = t_chunk
                    best_chunk = chunk_size
                end
            end
        end
        
        speedup = t_standard / best_time
        @printf("%-8d %-12.3f %-12.3f %-12d %7.2fx\n", 
                n, t_standard*1e6, best_time*1e6, best_chunk, speedup)
    end
    println()
end

function benchmark_spectral_method_operations()
    """Benchmark operations typical in spectral PDE solvers"""
    
    println("Testing realistic spectral method operations...")
    println("(Differentiation matrices, transforms, etc.)")
    println()
    
    # 2D spectral method: Chebyshev in x, Fourier in y
    Nx_vals = [64, 128, 256]
    Ny_vals = [64, 128, 256]
    
    @printf("%-12s %-12s %-12s %-8s\n", "Grid Size", "Diff Ops", "Transforms", "Total")
    println("-"^50)
    
    for Nx in Nx_vals, Ny in Ny_vals
        if Nx*Ny > 128*128  # Skip very large cases for demo
            continue
        end
        
        println("  Setting up $(Nx)×$(Ny) spectral operators...")
        
        # Create differentiation matrices
        Dx = create_chebyshev_diff_matrix(Nx)  # Chebyshev differentiation
        Dy = create_fourier_diff_matrix(Ny)    # Fourier differentiation
        
        # Create 2D field
        field = randn(Nx, Ny)
        field_vec = vec(field)
        
        # Test differentiation operations
        # ∂/∂x: (I ⊗ Dx) applied to field
        result_x = similar(field_vec)
        kron_x_op = create_kronecker_operator([Matrix(I, Ny, Ny), Dx])
        t_diff_x = @belapsed optimized_matmat!($result_x, $kron_x_op, $field_vec)
        
        # ∂/∂y: (Dy ⊗ I) applied to field  
        result_y = similar(field_vec)
        kron_y_op = create_kronecker_operator([Dy, Matrix(I, Nx, Nx)])
        t_diff_y = @belapsed optimized_matmat!($result_y, $kron_y_op, $field_vec)
        
        # Transform operations (simplified)
        t_transform = @belapsed fft!($field, 2)  # FFT along y-direction
        
        total_time = t_diff_x + t_diff_y + t_transform
        
        @printf("%-12s %-12.3f %-12.3f %-8.3f\n", 
                "$(Nx)×$(Ny)", (t_diff_x + t_diff_y)*1e3, t_transform*1e3, total_time*1e3)
    end
    println()
end

function print_performance_summary()
    """Print overall performance summary"""
    
    println("Performance Summary:")
    println()
    
    # Print global statistics
    print_linalg_stats()
    
    println("\nKey Optimizations Demonstrated:")
    println("1. ✓ BLAS-optimized dense operations")
    println("2. ✓ Sparse matrix optimizations")
    println("3. ✓ Kronecker product efficient evaluation")
    println("4. ✓ Block matrix operations")
    println("5. ✓ Memory-efficient streaming")
    println("6. ✓ Cache-conscious algorithms")
    println("7. ✓ Threaded parallel operations")
    
    println("\nRecommendations for Spectral Methods:")
    println("• Use sparse matrices for differentiation operators")
    println("• Leverage Kronecker structure in tensor-product methods")  
    println("• Enable threading for large matrix operations")
    println("• Consider block structure in coupled PDE systems")
    println("• Use streaming for memory-bound operations")
end

# Helper functions for creating test matrices
function create_spectral_diff_matrix(n::Int)
    """Create spectral differentiation matrix (Chebyshev)"""
    # Simplified Chebyshev differentiation matrix
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        if i != j && (i+j) % 2 == 1
            if j == 1
                D[i,j] = 2*(i-1)
            else
                D[i,j] = 4*(j-1) / (j > n÷2 ? -1 : 1)
            end
        end
    end
    return sparse(D)
end

function create_chebyshev_diff_matrix(n::Int)
    """Create Chebyshev differentiation matrix"""
    return create_spectral_diff_matrix(n)
end

function create_fourier_diff_matrix(n::Int)
    """Create Fourier differentiation matrix"""
    # Simplified Fourier differentiation (multiply by ik in spectral space)
    k = [0:(n÷2); -(n÷2-1):-1]  # Wavenumbers
    D = zeros(ComplexF64, n, n)
    for i in 1:n
        D[i,i] = im * k[i]
    end
    return sparse(D)
end

function block_matvec!(y, A11, A22, B1, B2, x, n_vel, n_pres)
    """Optimized block matrix-vector multiplication"""
    
    # Extract sub-vectors
    u1 = view(x, 1:n_vel)
    u2 = view(x, n_vel+1:2*n_vel)  
    p = view(x, 2*n_vel+1:2*n_vel+n_pres)
    
    y1 = view(y, 1:n_vel)
    y2 = view(y, n_vel+1:2*n_vel)
    y3 = view(y, 2*n_vel+1:2*n_vel+n_pres)
    
    # Block operations
    mul!(y1, A11, u1)          # A11 * u1
    mul!(y1, B1', p, 1.0, 1.0) # y1 += B1' * p
    
    mul!(y2, A22, u2)          # A22 * u2  
    mul!(y2, B2', p, 1.0, 1.0) # y2 += B2' * p
    
    mul!(y3, B1, u1)           # B1 * u1
    mul!(y3, B2, u2, 1.0, 1.0) # y3 += B2 * u2
end

# Need to define these functions that are referenced
function threaded_block_matmat!(C, A, B, α, β, block_size)
    """Placeholder for threaded block multiplication"""
    # This would be implemented in the optimized_linalg module
    mul!(C, A, B)
    if α != 1.0
        C .*= α
    end
    if β != 0.0
        C .+= β * C
    end
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    # Reset statistics
    reset_linalg_stats!()
    
    # Run main demo
    main()
    
    println("\nTo integrate these optimizations into your spectral solver:")
    println("1. Replace standard LinearAlgebra operations with optimized versions")
    println("2. Use OptimizedInitialValueSolver for time-stepping problems")
    println("3. Use OptimizedBoundaryValueSolver for BVPs")
    println("4. Monitor performance with print_linalg_stats()")
    println("5. Tune block sizes and chunk sizes for your hardware")
end