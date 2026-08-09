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
using Random

@testset "batched mode kernels (CPU backend)" begin
    n, nmodes, nrows = 9, 5, 7
    rng_vals(k) = ComplexF64[(i * 0.37 + k) + (i * 0.11 - k) * im for i in 1:k]
    # Explicit seed: two testsets below assert bit-exact `==` against a
    # `sprand`-generated matrix, so an unseeded RNG would make a failure
    # non-reproducible from one run to the next.
    rng = MersenneTwister(20260808)

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

    @testset "batched_gather!/batched_scatter! with step_ > 1 and row_offset > 0" begin
        # The two testsets above pin step_ == 1 (a dense (nrows, nmodes)
        # reshape has stride(cd,1) == 1) and row_offset == 0 — a kernel that
        # dropped `* step_` or `row_offset +` entirely would still pass both.
        # The real 2D layout supplies neither: a Fourier-first coefficient
        # array puts the coupled (z) axis SECOND, so step_ == nkx != 1, and
        # multi-variable stacking makes row_offset != 0. Reproduce that shape
        # here: mode is the FAST axis, so moving along the coupled axis
        # (length `len`) requires stride nmodes, not 1.
        len = 6
        cd = reshape(ComplexF64[(i + 0.4) + (i - 0.2) * im for i in 1:(nmodes * len)],
                     nmodes, len)
        step_ = stride(cd, 2)
        @test step_ == nmodes                 # sanity: genuinely non-degenerate
        starts = [m for m in 1:nmodes]        # mode m's run starts at position m

        row_offset = 3
        pad_below = 2
        sentinel = ComplexF64(-9999.0, -9999.0)
        X = fill(sentinel, row_offset + len + pad_below, nmodes)

        Tarang.batched_gather!(X, cd, starts, step_, len, row_offset)

        expected_block = zeros(ComplexF64, len, nmodes)
        for m in 1:nmodes
            buf = zeros(ComplexF64, len)
            Tarang._gather_strided!(buf, 0, cd, starts[m], step_, len)
            expected_block[:, m] .= buf
        end
        @test X[(row_offset + 1):(row_offset + len), :] == expected_block
        @test all(==(sentinel), X[1:row_offset, :])                # above block: untouched
        @test all(==(sentinel), X[(row_offset + len + 1):end, :])  # below block: untouched

        # Scatter's mirror: the SOURCE block sits at rows
        # row_offset2+1:row_offset2+len of a padded X; the padding rows carry
        # a different sentinel that must never be read. A dropped row_offset
        # would read the padding instead of the real block; a dropped step_
        # would place values contiguously in cd instead of striding by nmodes.
        row_offset2 = 2
        pad_below2 = 3
        real_block = reshape(ComplexF64[(i * 0.6) + (i * 0.15) * im for i in 1:(len * nmodes)],
                             len, nmodes)
        X2 = fill(ComplexF64(-1234.0, 4321.0), row_offset2 + len + pad_below2, nmodes)
        X2[(row_offset2 + 1):(row_offset2 + len), :] .= real_block

        cd_batched2 = zeros(ComplexF64, nmodes, len)
        cd_ref2 = zeros(ComplexF64, nmodes, len)
        starts2 = [m for m in 1:nmodes]
        step_2 = stride(cd_batched2, 2)
        @test step_2 == nmodes

        Tarang.batched_scatter!(cd_batched2, X2, starts2, step_2, len, row_offset2)
        for m in 1:nmodes
            Tarang._scatter_strided!(cd_ref2, real_block[:, m], 0, starts2[m], step_2, len)
        end
        @test cd_batched2 == cd_ref2
    end

    @testset "batched_spmv! matches per-mode mul! bit-for-bit" begin
        # NOTE: batched_spmv! iterates ROWS, so it takes the CSR pattern.
        # Passing A.colptr/A.rowval (CSC) would silently compute transpose(A)*x
        # and only agree when A is symmetric. Go through csr_pattern.
        A = sprand(rng, ComplexF64, n, n, 0.4)
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
        pattern = sprand(rng, ComplexF64, n, n, 0.5)
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
