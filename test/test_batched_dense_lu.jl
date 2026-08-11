"""
`BatchedDenseLU` — factor and solve every Fourier mode's stage matrix in one
call instead of one call per mode.

The singular-mode test is the important one. `getrf_batched` reports per-matrix
status in an `info` array; if that array goes unchecked, a singular mode returns
whatever happened to be in the buffer, which looks like a plausible answer and
propagates silently through the timestep. It must raise.
"""

using Test
using Tarang
using LinearAlgebra

@testset "BatchedDenseLU" begin
    n, nmodes = 6, 4

    function _well_conditioned_batch(n, nmodes)
        A = zeros(ComplexF64, n, n, nmodes)
        for m in 1:nmodes, j in 1:n, i in 1:n
            A[i, j, m] = (i == j) ? ComplexF64(n + m, 0.5) :
                                    ComplexF64(0.1 * (i - j), 0.05 * m)
        end
        return A
    end

    @testset "solve matches per-slice lu()" begin
        A = _well_conditioned_batch(n, nmodes)
        B = reshape(ComplexF64[(i * 0.3) + (i * 0.7) * im
                               for i in 1:(n * nmodes)], n, nmodes)

        expected = zeros(ComplexF64, n, nmodes)
        for m in 1:nmodes
            expected[:, m] .= lu(A[:, :, m]) \ B[:, m]
        end

        s = Tarang.BatchedDenseLU(copy(A))
        Tarang.batched_factor!(s)
        X = zeros(ComplexF64, n, nmodes)
        Tarang.batched_solve!(X, s, B)

        @test isapprox(X, expected; rtol=1e-12)
    end

    @testset "a singular mode raises, naming the mode" begin
        A = _well_conditioned_batch(n, nmodes)
        A[:, :, 3] .= 0            # mode 3 is exactly singular

        s = Tarang.BatchedDenseLU(A)
        err = try
            Tarang.batched_factor!(s)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("3", sprint(showerror, err))
    end

    @testset "solving before factoring raises" begin
        A = _well_conditioned_batch(n, nmodes)
        s = Tarang.BatchedDenseLU(A)
        X = zeros(ComplexF64, n, nmodes)
        B = ones(ComplexF64, n, nmodes)
        @test_throws Exception Tarang.batched_solve!(X, s, B)
    end

    # CPU-specific: this testset relies on `s.A` still holding the ORIGINAL
    # matrices after the first `batched_factor!`, so that `s.A .*= 2` scales
    # the operator and the second factor produces a genuinely new answer.
    # That survival is a CPU-only accident of calling `lu` (not `lu!`), which
    # copies before factoring — see the "differs by backend" note on
    # `BatchedDenseLU`'s docstring in src/tools/batched_matsolvers.jl. On GPU,
    # `getrf_strided_batched!` factors directly into `A`'s own buffers, so
    # after the first `batched_factor!`, `s.A` holds the LU factors, not the
    # operator; `s.A .*= 2` there would scale and re-factor the FACTORS, not
    # the operator, and this exact sequence would silently produce the wrong
    # answer with no error. Do not port this testset's `s.A .*=` pattern to a
    # GPU test — the supported lifecycle is assemble-then-factor (fully
    # overwrite `A`, then factor), never factor-mutate-refactor.
    @testset "refactoring after the matrix changes gives the new answer (CPU-only semantics)" begin
        A = _well_conditioned_batch(n, nmodes)
        s = Tarang.BatchedDenseLU(A)
        Tarang.batched_factor!(s)

        B = ones(ComplexF64, n, nmodes)
        X1 = zeros(ComplexF64, n, nmodes)
        Tarang.batched_solve!(X1, s, B)

        # Mutate in place, as batched_assemble_lhs! will, then refactor.
        # Valid here because the CPU path's `s.A` still holds the original
        # matrices at this point (see the CPU-specific note above) — this is
        # NOT a general guarantee `BatchedDenseLU` makes on either backend.
        s.A .*= 2
        Tarang.batched_factor!(s)
        X2 = zeros(ComplexF64, n, nmodes)
        Tarang.batched_solve!(X2, s, B)

        @test isapprox(X2, X1 ./ 2; rtol=1e-12)
    end
end
