#!/usr/bin/env julia
"""
Minimal test to verify CUDA.jl CUSPARSE API for ILU(0) preconditioner.

This tests ONLY the CUDA.jl API - no Tarang code involved.

Run with:
    julia test/verify_cuda_api.jl
"""

println("="^60)
println("Verifying CUDA.jl CUSPARSE API for ILU(0)")
println("="^60)

# Step 1: Check CUDA availability
println("\n[1] Checking CUDA availability...")
try
    using CUDA
    if CUDA.functional()
        println("    ✓ CUDA is functional")
        println("    Device: ", CUDA.name(CUDA.device()))
        println("    CUDA version: ", CUDA.version())
    else
        println("    ✗ CUDA not functional")
        exit(1)
    end
catch e
    println("    ✗ CUDA not available: $e")
    exit(1)
end

using SparseArrays
using LinearAlgebra

# Step 2: Check if CUSPARSE.ilu02 exists
println("\n[2] Checking if CUDA.CUSPARSE.ilu02 exists...")
try
    # Check if the function is defined
    if isdefined(CUDA.CUSPARSE, :ilu02)
        println("    ✓ CUDA.CUSPARSE.ilu02 is defined")
    else
        println("    ✗ CUDA.CUSPARSE.ilu02 is NOT defined")
        println("    Available CUSPARSE symbols containing 'ilu':")
        for name in names(CUDA.CUSPARSE, all=true)
            if occursin("ilu", lowercase(string(name)))
                println("      - $name")
            end
        end
        exit(1)
    end
catch e
    println("    ✗ Error checking ilu02: $e")
    exit(1)
end

# Step 3: Test ilu02 function
println("\n[3] Testing CUDA.CUSPARSE.ilu02 function...")
try
    n = 100
    A_cpu = sprand(n, n, 0.1) + 10I  # Ensure non-singular
    A_cpu = A_cpu + A_cpu'  # Make symmetric

    A_csr = CUDA.CUSPARSE.CuSparseMatrixCSR(A_cpu)
    println("    Created CuSparseMatrixCSR: size=$(size(A_csr)), nnz=$(nnz(A_csr))")

    LU = CUDA.CUSPARSE.ilu02(A_csr)
    println("    ✓ ilu02 returned: $(typeof(LU))")
    println("    LU size: $(size(LU)), nnz: $(nnz(LU))")
catch e
    println("    ✗ ilu02 failed: $e")
    println("    Stacktrace:")
    for (exc, bt) in current_exceptions()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

# Step 4: Test triangular wrappers
println("\n[4] Testing triangular wrappers with ldiv!...")
try
    n = 100
    A_cpu = sprand(n, n, 0.1) + 10I
    A_cpu = A_cpu + A_cpu'

    A_csr = CUDA.CUSPARSE.CuSparseMatrixCSR(A_cpu)
    LU = CUDA.CUSPARSE.ilu02(A_csr)

    r = CUDA.rand(Float64, n)
    z = CUDA.zeros(Float64, n)
    tmp = CUDA.zeros(Float64, n)

    # Test UnitLowerTriangular
    println("    Testing ldiv!(tmp, UnitLowerTriangular(LU), r)...")
    L = UnitLowerTriangular(LU)
    println("    UnitLowerTriangular wrapper created: $(typeof(L))")

    ldiv!(tmp, L, r)
    println("    ✓ Lower triangular solve succeeded")
    println("    tmp norm: $(norm(tmp))")

    # Test UpperTriangular
    println("    Testing ldiv!(z, UpperTriangular(LU), tmp)...")
    U = UpperTriangular(LU)
    println("    UpperTriangular wrapper created: $(typeof(U))")

    ldiv!(z, U, tmp)
    println("    ✓ Upper triangular solve succeeded")
    println("    z norm: $(norm(z))")

catch e
    println("    ✗ Triangular solve failed: $e")
    println("    Stacktrace:")
    for (exc, bt) in current_exceptions()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

# Step 5: Test full preconditioner application
println("\n[5] Testing full ILU(0) preconditioner application...")
try
    n = 100
    A_cpu = sprand(n, n, 0.1) + 10I
    A_cpu = A_cpu + A_cpu'

    A_csr = CUDA.CUSPARSE.CuSparseMatrixCSR(A_cpu)
    LU = CUDA.CUSPARSE.ilu02(A_csr)

    r = CUDA.rand(Float64, n)
    z = CUDA.zeros(Float64, n)
    tmp = CUDA.zeros(Float64, n)

    # Apply preconditioner: z = (LU)^{-1} * r
    ldiv!(tmp, UnitLowerTriangular(LU), r)  # tmp = L^{-1} * r
    ldiv!(z, UpperTriangular(LU), tmp)       # z = U^{-1} * tmp

    println("    ✓ Full preconditioner application succeeded")
    println("    Input norm:  $(norm(r))")
    println("    Output norm: $(norm(z))")

catch e
    println("    ✗ Full preconditioner failed: $e")
    println("    Stacktrace:")
    for (exc, bt) in current_exceptions()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end

println("\n" * "="^60)
println("ALL CUDA.jl API TESTS PASSED!")
println("The ILU(0) implementation should work correctly.")
println("="^60)
