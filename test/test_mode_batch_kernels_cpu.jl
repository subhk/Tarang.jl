"""
Value tests for the batched Fourier-mode kernels — on CPU arrays, through the
REAL kernel objects.

These are KernelAbstractions kernels, so the objects the CUDA path launches on a
`CUDABackend()` also run on `KernelAbstractions.CPU()` over plain `Array`s. That
matters more than usual here: the KA CPU backend miscompiles same-slot
read-modify-write around inner loops (found in `_cheb_coeff_to_deriv_kernel!`
during the PR #105 work), so running the real objects is the only way to catch
that class without hardware.

Each kernel is checked against the per-mode function it replaces. Everything is
bit-exact except `batched_assemble_lhs!`, which computes `M + c*L` and may be
FMA-contracted by the backend.
"""

using Test
using Tarang
using SparseArrays
using LinearAlgebra
using KernelAbstractions

@testset "batched mode kernels (CPU backend)" begin
    n, nmodes, nrows = 9, 5, 7
    rng_vals(k) = ComplexF64[(i * 0.37 + k) + (i * 0.11 - k) * im for i in 1:k]

    @testset "batched_gather! matches _gather_strided! bit-for-bit" begin
        # Emulate a coeff array with one Fourier axis and one coupled axis.
        cd = reshape(ComplexF64[(i + 0.25) + (i - 0.5) * im for i in 1:(nrows * nmodes)],
                     nrows, nmodes)
        step_ = stride(cd, 1)
        starts = [1 + (m - 1) * stride(cd, 2) for m in 1:nmodes]

        X = zeros(ComplexF64, nrows, nmodes)
        Tarang.batched_gather!(X, cd, starts, step_, nrows, 0)

        expected = zeros(ComplexF64, nrows, nmodes)
        for m in 1:nmodes
            buf = zeros(ComplexF64, nrows)
            Tarang._gather_strided!(buf, 0, cd, starts[m], step_, nrows)
            expected[:, m] .= buf
        end
        @test X == expected            # bit-exact, not approx
    end

    @testset "batched_scatter! matches _scatter_strided! bit-for-bit" begin
        cd_batched = zeros(ComplexF64, nrows, nmodes)
        cd_ref = zeros(ComplexF64, nrows, nmodes)
        step_ = stride(cd_batched, 1)
        starts = [1 + (m - 1) * stride(cd_batched, 2) for m in 1:nmodes]

        X = reshape(ComplexF64[(i * 0.5) + (i * 0.25) * im for i in 1:(nrows * nmodes)],
                    nrows, nmodes)

        Tarang.batched_scatter!(cd_batched, X, starts, step_, nrows, 0)
        for m in 1:nmodes
            Tarang._scatter_strided!(cd_ref, X[:, m], 0, starts[m], step_, nrows)
        end
        @test cd_batched == cd_ref
    end

    @testset "batched_spmv! matches per-mode mul! bit-for-bit" begin
        # NOTE: batched_spmv! iterates ROWS, so it takes the CSR pattern.
        # Passing A.colptr/A.rowval (CSC) would silently compute transpose(A)*x
        # and only agree when A is symmetric. Go through csr_pattern.
        A = sprand(ComplexF64, n, n, 0.4)
        rowptr, colval, perm = Tarang.csr_pattern(A)

        nzv_csc = zeros(ComplexF64, nnz(A), nmodes)
        X = zeros(ComplexF64, n, nmodes)
        for m in 1:nmodes
            nzv_csc[:, m] .= A.nzval .* (1 + 0.1m)
            X[:, m] .= rng_vals(n) .* (1 - 0.05m)
        end
        nzv_csr = nzv_csc[perm, :]

        Y = zeros(ComplexF64, n, nmodes)
        Tarang.batched_spmv!(Y, rowptr, colval, nzv_csr, X)

        for m in 1:nmodes
            Am = SparseMatrixCSC(n, n, copy(A.colptr), copy(A.rowval),
                                 nzv_csc[:, m])
            expected = zeros(ComplexF64, n)
            mul!(expected, Am, X[:, m])
            @test Y[:, m] == expected
        end
    end

    @testset "an asymmetric matrix distinguishes CSR from CSC" begin
        # Pins the bug the previous testset would otherwise hide: with a
        # symmetric A, feeding the CSC pattern to a row-iterating kernel gives
        # the right answer by accident.
        A = sparse([1, 2], [2, 1], ComplexF64[3.0, 0.0], 2, 2)
        rowptr, colval, perm = Tarang.csr_pattern(A)
        x = reshape(ComplexF64[1.0, 1.0], 2, 1)
        y = zeros(ComplexF64, 2, 1)
        Tarang.batched_spmv!(y, rowptr, colval, reshape(A.nzval[perm], :, 1), x)
        @test y[1, 1] == 3.0        # A[1,2]*x[2]
        @test y[2, 1] == 0.0
    end

    @testset "batched_bc_override! writes only bc rows" begin
        RHS = reshape(ComplexF64[i + 0.0im for i in 1:(n * nmodes)], n, nmodes)
        ALG = reshape(ComplexF64[100i + 0.0im for i in 1:(n * nmodes)], n, nmodes)
        bc = [2, 5]
        coeff = 0.375
        before = copy(RHS)

        Tarang.batched_bc_override!(RHS, ALG, bc, coeff)

        for m in 1:nmodes, r in 1:n
            if r in bc
                @test RHS[r, m] == coeff * ALG[r, m]
            else
                @test RHS[r, m] == before[r, m]   # untouched, bit-for-bit
            end
        end
    end

    @testset "batched_assemble_lhs! reproduces M + c*L densely" begin
        pattern = sprand(ComplexF64, n, n, 0.5)
        nnzp = nnz(pattern)
        Mv = zeros(ComplexF64, nnzp, nmodes)
        Lv = zeros(ComplexF64, nnzp, nmodes)
        for m in 1:nmodes
            Mv[:, m] .= pattern.nzval .* (0.5 + 0.1m)
            Lv[:, m] .= pattern.nzval .* (2.0 - 0.2m)
        end
        coeff = ComplexF64(0.25, 0.0)

        dense = zeros(ComplexF64, n, n, nmodes)
        Tarang.batched_assemble_lhs!(dense, pattern.colptr, pattern.rowval,
                                     Mv, Lv, coeff)

        for m in 1:nmodes
            expected = Matrix(SparseMatrixCSC(n, n, copy(pattern.colptr),
                                              copy(pattern.rowval),
                                              Mv[:, m] .+ coeff .* Lv[:, m]))
            # 1e-15, not bit-exact: the kernel may FMA-contract M + c*L.
            @test isapprox(dense[:, :, m], expected; rtol=1e-15, atol=1e-15)
        end
    end

    @testset "structural zeros are written, not left stale" begin
        # A dense workspace reused across dt changes must be fully overwritten;
        # a kernel that only touched stored nonzeros would leave the previous
        # factorization's values in the structural-zero slots.
        pattern = sparse([1, 3], [1, 2], ComplexF64[1.0, 2.0], 3, 3)
        dense = fill(ComplexF64(9999.0), 3, 3, 1)
        Mv = reshape(copy(pattern.nzval), :, 1)
        Lv = zeros(ComplexF64, nnz(pattern), 1)

        Tarang.batched_assemble_lhs!(dense, pattern.colptr, pattern.rowval,
                                     Mv, Lv, ComplexF64(1.0))

        @test dense[2, 1, 1] == 0        # structural zero, must be cleared
        @test dense[1, 3, 1] == 0
        @test dense[1, 1, 1] == 1.0
        @test dense[3, 2, 1] == 2.0
    end
end
