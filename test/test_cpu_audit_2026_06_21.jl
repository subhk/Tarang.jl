# Guard tests for the CPU correctness fixes from the 2026-06-21 audit.
#
# Each testset pins behavior that was previously WRONG, exercising the real
# (serial-CPU) code path with an independent oracle. See
# memory/project_cpu_audit_2026_06_21.md for the full analysis.
#
# Bugs (all confirmed real, adversarially verified, then hand-checked):
#   1. QG RK4 surface dissipation applied with wrong sign (.+= vs .-=)
#   2. Scaled RealFourier backward never zero-pads on upsample (wrong shape/type)
#   3. Compound-constant coefficient silently dropped from L/M matrix block
#   4. Lazy-RHS Multiply branch discards imaginary part of complex scalar coeff
#   5. _matched_forcing_view given coeff array (not size tuple) drops non-Stochastic forcing

using Test
using Tarang

@testset "CPU audit 2026-06-21 fixes" begin

    # Bug #1 — qg_step_rk4!'s inner compute_rhs added κ(-Δ)^α θ with `.+=`, while
    # Euler/RK2 correctly subtract it. (-Δ)^α has POSITIVE eigenvalues, so the
    # default RK4 stepper was anti-diffusive (high-k blow-up).
    @testset "QG RK4 surface dissipation decays (not grows) high-k modes" begin
        # The buggy line lives inside qg_step_rk4!'s `compute_rhs` closure, which first
        # calls qg_invert! (a SEPARATE, pre-existing, universally-broken 3D Fourier-Cheb
        # LBVP solve that throws DimensionMismatch at every resolution). To reach the
        # dissipation line we neutralize ONLY that broken inversion: leave ψ = 0, so the
        # induced surface velocity is 0 and the u·∇θ advection term vanishes -> the surface
        # buoyancy evolves under dissipation alone. (Seeding a single x-dependent mode also
        # makes advection identically 0 even with nonzero ψ, so this isolation is clean.)
        @eval Tarang function qg_invert!(qg::QGSystem)
            get_grid_data(qg.ψ) .= 0.0
            return qg.ψ
        end

        qg = Tarang.qg_system_setup(Lx = 2π, Ly = 2π, H = 1.0,
                                    Nx = 8, Ny = 8, Nz = 8,
                                    f0 = 1.0, N = 1.0, κ = 0.5, α = 1.0)

        # Seed a high-wavenumber, x-only surface buoyancy mode: cos(3x). With ψ ≡ 0 the
        # advection is zero, so any amplitude change is purely the κ(-Δ)^α dissipation.
        X = Tarang.local_grids(qg.θ_bot.dist, qg.θ_bot.bases...)[1]
        get_grid_data(qg.θ_bot) .= cos.(3 .* X)
        get_grid_data(qg.θ_top) .= cos.(3 .* X)
        get_grid_data(qg.q)     .= 0.0

        amp0 = maximum(abs, get_grid_data(qg.θ_bot))

        # invokelatest so the just-@eval'd qg_invert! override (and, post-fix, qg_step_rk4!)
        # are dispatched despite the @testset's fixed world age.
        for _ in 1:5
            Base.invokelatest(Tarang.qg_step_rk4!, qg, 0.01)
        end

        amp1 = maximum(abs, get_grid_data(qg.θ_bot))

        # (-Δ)^α has POSITIVE eigenvalues |k|^{2α}; physical dissipation must DECAY the mode.
        # Buggy `.+=` -> anti-diffusion -> amp1/amp0 ≈ 1.25 (grows). Fixed `.-=` -> ≈ 0.80.
        @test amp1 < amp0
    end

    # Bug #2 — _backward_output_spec gated irfft on `axis_len == div(grid_n,2)+1`
    # with grid_n = SCALED M, which can never hold for an upsampled rfft axis
    # (stored coeffs are base div(N,2)+1), so it fell to a same-shape complex ifft.
    @testset "BUG2 RealFourier upsampled backward irfft zero-pad" begin
        coords = CartesianCoordinates("x")
        dist = Distributor(coords; mesh=(1,), dtype=Float64)
        N = 8
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2pi))
        f = ScalarField(Domain(dist, (xb,)), "f")

        nodes(M) = [2pi * (i - 1) / M for i in 1:M]
        # Band-limited: modes 2,3 < Nyquist 4, so exact on any grid M >= 8.
        sig(x) = sin(2x) + 0.5 * cos(3x)

        ensure_layout!(f, :g)
        get_grid_data(f) .= sig.(nodes(N))

        forward_transform!(f)            # -> :c  (half-spectrum length div(N,2)+1 = 5)
        @test f.current_layout == :c

        set_scales!(f, 2.0)              # upsample while in coefficient space
        @test f.scales == (2.0,)
        @test f.current_layout == :c

        backward_transform!(f)           # -> :g on the finer grid (M = 2N = 16)
        gd = vec(get_grid_data(f))
        M = 2N

        # On the buggy code the backward stage falls to the ifft fallback: it
        # returns a Complex vector of the BASE half-spectrum length (5), not a
        # real vector on the finer grid (16).
        @test eltype(gd) <: Real
        @test length(gd) == M
        @test maximum(abs.(gd .- sig.(nodes(M)))) < 1e-10
    end

    # Bug #3 — _is_const_or_param / _extract_scalar didn't recurse through
    # arithmetic nodes, so a compound-constant coefficient like (1/2) made the
    # MultiplyOperator branch fall to the else (_zero_block), dropping the term.
    @testset "build_expression_matrix_block: compound-constant coefficient not dropped" begin
        # A coefficient that is itself a compound-constant arithmetic node, e.g.
        # (1/2) = DivideOperator(ConstantOperator(1.0), ConstantOperator(2.0)),
        # multiplying a linear operand must scale that operand's matrix block,
        # NOT be silently dropped to a zero block.
        n = 3
        var = Tarang.UnknownOperator("u")              # acts as the problem variable
        coeff = Tarang.DivideOperator(Tarang.ConstantOperator(1.0),
                                      Tarang.ConstantOperator(2.0))   # == 0.5
        term = Tarang.MultiplyOperator(coeff, var)     # (1/2) * u

        block = Tarang.build_expression_matrix_block(term, var, n, n)
        dense = Matrix(block)

        # Pre-fix: _is_const_or_param(DivideOperator) == false → else branch →
        # _zero_block, so the term is silently dropped (all zeros).
        @test any(x -> abs(x) > 0, dense)

        # The block of `var` alone is the identity; (1/2)*u must give 0.5 on the diag.
        expected = zeros(ComplexF64, n, n)
        for i in 1:n
            expected[i, i] = 0.5
        end
        @test dense ≈ expected
    end

    # Bug #4 — the Multiply Future branch of translate_to_lazy folded only
    # Float64(real(a)), so `2im*u` compiled to LazyScale(u, 0.0), silently
    # zeroing the term instead of bailing to the (correct) interpreted RHS.
    @testset "lazy_rhs complex Multiply coeff bails (BUG #4)" begin
        # Minimal 1D periodic field + a trivial state list (mirrors make_periodic_field
        # in test/test_problems.jl).
        coords = CartesianCoordinates("x")
        dist   = Distributor(coords; mesh=(1,), dtype=Float64)
        basis  = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        u = ScalarField(dist, "u", (basis,), Float64)
        state = [u]

        # Sibling: a REAL scalar coefficient must still fold into a lazy op (not bail).
        real_expr = Tarang.Multiply(2.0, u)
        @test Tarang.translate_to_lazy(real_expr, state) !== nothing

        # Bug: a COMPLEX scalar coefficient `2im * u` must bail to the interpreted RHS.
        # Pre-fix the Multiply branch folds only real(2im)==0.0, returning a
        # LazyScale(..., 0.0) (non-nothing) that silently zeroes the term.
        cplx_expr = Tarang.Multiply(2im, u)
        @test Tarang.translate_to_lazy(cplx_expr, state) === nothing
    end

    # Bug #5 — generic _matched_forcing_view expected a size tuple but both call
    # sites pass the complex coeff ARRAY, so size(F)==array is false, then
    # min(Int, ComplexF64) throws → bare catch → nothing → forcing dropped.
    @testset "_matched_forcing_view accepts coeff ARRAY target (non-Stochastic forcing)" begin
        # Mirror the real call sites (state_utils.jl:133, lazy_rhs.jl:908) which pass the
        # complex coefficient ARRAY (not a size tuple) as the second argument.
        # DeterministicForcing has no typed overload, so it hits the generic
        # _matched_forcing_view(forcing, target_size) fallback in state_utils.jl.
        f = DeterministicForcing((args...) -> zeros(4, 4), (4, 4))
        f.cached_forcing .= 1.0                      # real-typed cache, size (4,4)

        coeff = zeros(ComplexF64, 4, 4)              # the matching complex coeff array

        F_view = Tarang._matched_forcing_view(f, coeff)

        # Pre-fix: size(F)==coeff is false, then min(Int, ComplexF64) throws ->
        # bare catch -> returns nothing (forcing silently dropped).
        @test F_view !== nothing
        @test size(F_view) == size(coeff)
        @test F_view == f.cached_forcing
    end

end
