"""
Tests for src/core/subsystems/subproblem_ncc.jl — the exported
`cartesian_mode_matrix` + `build_ncc_matrix` NCC (Non-Constant Coefficient)
helpers.

NOTE (corrected 2026-06-17): these are EXPORTED helpers that are NOT currently
invoked by the live solver matrix assembler. The live path is
`build_matrices!` → `expression_matrices` (matrices_subproblem_operators.jl),
which does not call `build_ncc_matrix`/`cartesian_mode_matrix` (verified: zero
`src/` callers). These tests pin the helpers' standalone behavior (they were at
0% coverage), they are NOT an end-to-end solver-NCC check.

IMPORTANT: `cartesian_mode_matrix` builds a per-axis CIRCULAR shift, which is the
correct convolution ONLY for periodic (ComplexFourier) bases. It is wrong for
RealFourier (half-spectrum) and for Chebyshev/Jacobi (whose products follow the
polynomial recurrence, not a shift). The basis-AWARE, correct builder is
`ncc_matrix`/`product_matrix` in basis_product_matrices.jl (test_ncc_product_matrices.jl).

Semantics being pinned (read off subproblem_ncc.jl):

* `cartesian_mode_matrix(sp_shape, arg_domain, out_domain, ncc_mode::Tuple)`
  builds a Kronecker product of 1D circulant shift matrices, one per axis.
  Per axis of length `n` with mode index `m` (a 1-based CartesianIndex
  component):
    - m == 0 or m == 1  -> n×n identity (no shift)
    - m >= 2            -> circulant shift: column j -> row mod1(j+m-1, n)
  `arg_domain`/`out_domain` are accepted for signature compatibility but are
  NOT consulted by this function — only `sp_shape` and `ncc_mode` matter.

* `build_ncc_matrix(ncc_data, sp, arg_domain, out_domain; ...)` forms
    M = Σ_k ν_k · S_k
  where ν_k is the k-th spectral coefficient of the NCC field and
  S_k = cartesian_mode_matrix(sp_shape, …, Tuple(k)). For a 1D periodic
  (Fourier) basis with coeffs stored as [ν_0, ν_1, …] (mode = index-1) this
  is exactly the circular-convolution (circulant) matrix whose first column
  is the coefficient vector. That gives a clean numeric oracle:
  `M * f̂ == ifft( fft-style circular conv of ν and f̂ )`, which we check as a
  plain circulant matrix-vector product.

Uniquely-prefixed names (spncc_*) — the full suite shares the Main namespace.
All Tarang internals are called fully-qualified (Tarang.foo).
"""

using Test
using Tarang
using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Mock solver/base mirroring test_subsystems.jl, so we can build a real
# Subproblem cheaply. `matrix_coupling = [true]` marks the single axis as
# coupled (group entry `nothing`) so coeff_shape == full coefficient shape.
# ---------------------------------------------------------------------------
struct SpnccBase
    matrix_coupling::Vector{Bool}
end
struct SpnccSolver
    problem::Tarang.Problem
    base::SpnccBase
end

"""Build a 1D coupled Subproblem over a length-N basis `bcons(coord; size, bounds)`."""
function spncc_make_sp(bcons, N::Int; dtype=ComplexF64)
    coords = CartesianCoordinates("x")
    dist = Distributor(coords; dtype=(dtype <: Real ? Float64 : Float64))
    xb = bcons(coords["x"]; size=N, bounds=(0.0, 2π))
    field = ScalarField(dist, "u", (xb,), dtype)
    problem = IVP([field])
    solver = SpnccSolver(problem, SpnccBase([true]))   # coupled axis
    subsys = Tarang.Subsystem(solver, (nothing,))
    sp = Tarang.Subproblem(solver, (subsys,), (nothing,))
    return sp, field
end

"""Reference circulant: first column = ν (length n). (C f)_p = Σ_q ν[mod1(p-q,n)+? ]...
We construct it directly from the documented shift rule so the oracle is
independent of any circulant library convention: index i of ν contributes a
shift-by-(i-1) matrix, where shift-by-s sends column j -> row mod1(j+s, n)."""
function spncc_reference_conv_matrix(nu::AbstractVector, n::Int)
    C = zeros(ComplexF64, n, n)
    for i in eachindex(nu)
        s = i - 1                       # shift amount for coeff index i
        νi = nu[i]
        νi == 0 && continue
        for j in 1:n
            target = mod1(j + s, n)
            C[target, j] += νi
        end
    end
    return C
end

@testset "subproblem_ncc" begin

    # ---------------------------------------------------------------------
    # 1. cartesian_mode_matrix — self-contained structural tests (1D)
    # ---------------------------------------------------------------------
    @testset "cartesian_mode_matrix: 1D shift structure" begin
        n = 5
        # mode index 0 and 1 are both the identity
        for m in (0, 1)
            M = Tarang.cartesian_mode_matrix((n,), nothing, nothing, (m,))
            @test size(M) == (n, n)
            @test Matrix(M) == Matrix(I, n, n)
        end

        # mode index m >= 2 -> circulant shift by (m-1):
        #   column j -> row mod1(j + (m-1), n)
        for m in 2:6              # include m-1 == n (full wrap) and m-1 > n
            M = Tarang.cartesian_mode_matrix((n,), nothing, nothing, (m,))
            @test size(M) == (n, n)
            @test nnz(M) == n               # permutation matrix: exactly one 1 per column
            Mfull = Matrix(M)
            s = m - 1
            for j in 1:n
                target = mod1(j + s, n)
                @test Mfull[target, j] == 1.0
                @test sum(Mfull[:, j]) == 1.0   # only that entry is set
            end
            # A permutation matrix is orthogonal: Mᵀ M == I
            @test Mfull' * Mfull == Matrix(I, n, n)
        end
    end

    @testset "cartesian_mode_matrix: arg/out_domain are ignored" begin
        # The function never consults arg_domain/out_domain; passing wildly
        # different junk must not change the result.
        A = Tarang.cartesian_mode_matrix((4,), nothing, nothing, (3,))
        B = Tarang.cartesian_mode_matrix((4,), "junk", 12345, (3,))
        @test Matrix(A) == Matrix(B)
    end

    # ---------------------------------------------------------------------
    # 2. cartesian_mode_matrix — multi-axis Kronecker structure
    # ---------------------------------------------------------------------
    @testset "cartesian_mode_matrix: 2D Kronecker product" begin
        sp_shape = (2, 3)
        # mode (m1, m2): result == kron(axis1, axis2)
        ax1 = Tarang.cartesian_mode_matrix((2,), nothing, nothing, (3,))   # shift by 2 on n=2
        ax2 = Tarang.cartesian_mode_matrix((3,), nothing, nothing, (2,))   # shift by 1 on n=3
        M = Tarang.cartesian_mode_matrix(sp_shape, nothing, nothing, (3, 2))
        @test size(M) == (prod(sp_shape), prod(sp_shape))
        @test Matrix(M) == kron(Matrix(ax1), Matrix(ax2))

        # mode (1,1) over a 2D shape is the full identity
        Mid = Tarang.cartesian_mode_matrix(sp_shape, nothing, nothing, (1, 1))
        @test Matrix(Mid) == Matrix(I, prod(sp_shape), prod(sp_shape))

        # Short ncc_mode tuple: missing axes default to mode 0 (identity).
        # (2,) means: axis1 -> mode 2 (shift 1 on n=2), axis2 -> mode 0 (identity)
        Mshort = Tarang.cartesian_mode_matrix(sp_shape, nothing, nothing, (2,))
        ax1b = Tarang.cartesian_mode_matrix((2,), nothing, nothing, (2,))
        ax2id = sparse(I, 3, 3)
        @test Matrix(Mshort) == kron(Matrix(ax1b), Matrix(ax2id))
    end

    # ---------------------------------------------------------------------
    # 3. build_ncc_matrix — empty / degenerate paths
    # ---------------------------------------------------------------------
    @testset "build_ncc_matrix: nil and all-zero coeffs" begin
        N = 6
        sp, field = spncc_make_sp(ComplexFourier, N)

        # No coeffs set -> returns nothing
        nc_nil = NCCData()
        @test Tarang.build_ncc_matrix(nc_nil, sp, field.domain, field.domain) === nothing

        # All-zero coeffs: the docstring promises "for coeff_max == 0
        # (all-zero NCC), treat as zero — build_ncc_matrix returns an empty
        # sparse matrix." That contract is currently BROKEN. With all-zero
        # coeffs `significant_modes` is empty, but the energy-cap branch
        # (taken when max_ncc_terms === nothing) forces `n_to_use = max(cap, 1)
        # = 1` (subproblem_ncc.jl:135), so the accumulation loop then indexes
        # `significant_modes[1]` out of bounds -> UndefRefError. See line 130-135.
        nc_zero = NCCData()
        nc_zero.coeffs = zeros(ComplexF64, N)
        @test_broken (Tarang.build_ncc_matrix(nc_zero, sp, field.domain, field.domain); true)

        # WORKAROUND path: passing max_ncc_terms avoids the max(cap,1) branch,
        # so the all-zero NCC correctly yields an empty matrix of the right shape.
        Mz = Tarang.build_ncc_matrix(nc_zero, sp, field.domain, field.domain; max_ncc_terms=4)
        @test Mz !== nothing
        @test size(Mz) == (N, N)
        @test nnz(Mz) == 0
    end

    # ---------------------------------------------------------------------
    # 4. build_ncc_matrix — single-mode NCC equals a pure shift matrix
    # ---------------------------------------------------------------------
    @testset "build_ncc_matrix: single mode -> shift matrix" begin
        N = 6
        sp, field = spncc_make_sp(ComplexFourier, N)

        # NCC = pure mode-1 (index 2) with unit coefficient -> shift-by-1 matrix.
        nc = NCCData()
        coeffs = zeros(ComplexF64, N); coeffs[2] = 1.0
        nc.coeffs = coeffs
        # Use a tiny cutoff so nothing is dropped.
        M = Tarang.build_ncc_matrix(nc, sp, field.domain, field.domain; ncc_cutoff=1e-12)
        Sexpected = Tarang.cartesian_mode_matrix((N,), nothing, nothing, (2,))
        @test size(M) == (N, N)
        @test Matrix(M) == Matrix(Sexpected)

        # And the DC mode (index 1) with coeff c -> c · I
        nc_dc = NCCData()
        cdc = zeros(ComplexF64, N); cdc[1] = 2.5
        nc_dc.coeffs = cdc
        Mdc = Tarang.build_ncc_matrix(nc_dc, sp, field.domain, field.domain; ncc_cutoff=1e-12)
        @test Matrix(Mdc) ≈ 2.5 * Matrix(I, N, N)
    end

    # ---------------------------------------------------------------------
    # 5. build_ncc_matrix — full convolution oracle (multi-mode NCC)
    # ---------------------------------------------------------------------
    @testset "build_ncc_matrix: circular-convolution oracle" begin
        N = 8
        sp, field = spncc_make_sp(ComplexFourier, N)

        # A few nonzero NCC modes with distinct complex weights.
        coeffs = zeros(ComplexF64, N)
        coeffs[1] = 1.0 + 0.0im     # DC  (shift 0)
        coeffs[2] = 0.5 - 0.25im    # mode 1 (shift 1)
        coeffs[4] = -0.3im          # mode 3 (shift 3)
        nc = NCCData()
        nc.coeffs = coeffs
        M = Tarang.build_ncc_matrix(nc, sp, field.domain, field.domain; ncc_cutoff=1e-12)

        Cref = spncc_reference_conv_matrix(coeffs, N)
        @test size(M) == (N, N)
        @test Matrix(M) ≈ Cref

        # Spot-check the action on a random coefficient vector against the
        # independent reference circulant: M·f̂ == Cref·f̂.
        f = randn(ComplexF64, N)
        @test Matrix(M) * f ≈ Cref * f
    end

    # ---------------------------------------------------------------------
    # 6. build_ncc_matrix — relative cutoff truncation behaviour
    # ---------------------------------------------------------------------
    @testset "build_ncc_matrix: relative cutoff drops tiny modes" begin
        N = 6
        sp, field = spncc_make_sp(ComplexFourier, N)

        # One dominant DC mode and one tiny mode-2 contribution. With a
        # generous relative cutoff the tiny mode (and its off-diagonal
        # entries) must be dropped, leaving only the diagonal DC block.
        coeffs = zeros(ComplexF64, N)
        coeffs[1] = 1.0          # dominant
        coeffs[3] = 1e-4         # 1e-4 relative -> below a 1e-2 cutoff
        nc = NCCData()
        nc.coeffs = coeffs

        # Tiny cutoff: keep both modes (DC diagonal + shift-2 off-diagonal).
        Mkeep = Tarang.build_ncc_matrix(nc, sp, field.domain, field.domain; ncc_cutoff=1e-12)
        @test nnz(Mkeep) == 2 * N      # identity (N) + shift-2 permutation (N)

        # Relative cutoff 1e-2 (> 1e-4): the small mode is dropped entirely.
        Mdrop = Tarang.build_ncc_matrix(nc, sp, field.domain, field.domain; ncc_cutoff=1e-2)
        @test Matrix(Mdrop) ≈ Matrix(I, N, N)
        @test nnz(Mdrop) == N
    end

    # ---------------------------------------------------------------------
    # 7. build_ncc_matrix — max_ncc_terms cap (energy-sorted retention)
    # ---------------------------------------------------------------------
    @testset "build_ncc_matrix: max_ncc_terms keeps dominant modes" begin
        N = 8
        sp, field = spncc_make_sp(ComplexFourier, N)

        coeffs = zeros(ComplexF64, N)
        coeffs[1] = 3.0     # largest energy
        coeffs[2] = 2.0     # second
        coeffs[5] = 1.0     # smallest — should be dropped when capped to 2
        nc = NCCData()
        nc.coeffs = coeffs

        # Cap to the 2 dominant (energy-sorted) modes: keep coeffs[1] & coeffs[2].
        Mcap = Tarang.build_ncc_matrix(nc, sp, field.domain, field.domain;
                                       ncc_cutoff=1e-12, max_ncc_terms=2)
        kept = copy(coeffs); kept[5] = 0.0
        Cref = spncc_reference_conv_matrix(kept, N)
        @test Matrix(Mcap) ≈ Cref

        # max_ncc_terms == 1 keeps only the single most-energetic mode (DC).
        Mone = Tarang.build_ncc_matrix(nc, sp, field.domain, field.domain;
                                       ncc_cutoff=1e-12, max_ncc_terms=1)
        @test Matrix(Mone) ≈ 3.0 * Matrix(I, N, N)
    end

    # ---------------------------------------------------------------------
    # 8. coeff_size legacy overloads (defined at bottom of subproblem_ncc.jl)
    # ---------------------------------------------------------------------
    @testset "coeff_size legacy field overloads" begin
        N = 6
        sp, field = spncc_make_sp(ComplexFourier, N)
        ss = sp.subsystems[1]
        # ScalarField overload -> field_size(subsystem, field)
        @test Tarang.coeff_size(ss, field) == Tarang.field_size(ss, field)
        @test Tarang.coeff_size(ss, field) isa Integer
    end
end
