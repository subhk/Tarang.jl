using Test
using Tarang
using Statistics
using Random

const _CUDA_AVAILABLE = try
    @eval begin
        using CUDA
        CUDA.functional()
    end
catch
    false
end

# PencilArrays/MPI are reached through Tarang's own namespace so this file adds no
# top-level `using` that could shadow test-local names.
const _PA = Tarang.PencilArrays
const _MPI = Tarang.MPI

# ============================================================================
# Independent oracles
# ============================================================================
#
# These are written from INDEX NOTATION with explicit loops over (i, j, k) and
# are deliberately NOT the expanded scalar expressions used by
# `src/core/les_models.jl`. The previous version of this file restated the
# implementation character-for-character, so any error in the implementation was
# copied into the oracle and the test passed regardless.
#
# Convention throughout: `G[i, k] = ∂uᵢ/∂xₖ` — velocity component i, derivative
# direction k. Gradient arrays are passed to the model in component-major order,
# i.e. `(∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)` in 2-D and
# `(∂u∂x, ∂u∂y, ∂u∂z, ∂v∂x, ∂v∂y, ∂v∂z, ∂w∂x, ∂w∂y, ∂w∂z)` in 3-D, which is the
# row-major flattening of G.
#
# Every oracle evaluates in Float64 even when the model runs in Float32, so the
# Float32 tests are checked against a genuinely more accurate reference.

"""Strain-rate tensor `S[i,j] = (G[i,j] + G[j,i]) / 2`."""
function _oracle_strain(G::AbstractMatrix{T}) where {T}
    N = size(G, 1)
    S = Matrix{T}(undef, N, N)
    for i in 1:N, j in 1:N
        S[i, j] = (G[i, j] + G[j, i]) / T(2)
    end
    return S
end

"""Strain magnitude `|S̄| = √(2 S̄ᵢⱼ S̄ᵢⱼ)` — the convention used by the models."""
function _oracle_strain_magnitude(G::AbstractMatrix{T}) where {T}
    S = _oracle_strain(G)
    acc = zero(T)
    for i in axes(S, 1), j in axes(S, 2)
        acc += S[i, j] * S[i, j]
    end
    return sqrt(T(2) * acc)
end

"""
AMD eddy-viscosity predictor (UNCLIPPED):

    num = Σₖ Δₖ² Σᵢⱼ G[i,k] G[j,k] S[i,j],   den = Σᵢₖ G[i,k]²,   ν† = -C num/den
"""
function _oracle_amd_nu(C::T, Δ::NTuple{N, T}, G::AbstractMatrix{T}) where {T, N}
    S = _oracle_strain(G)
    num = zero(T)
    for k in 1:N
        acc = zero(T)
        for i in 1:N, j in 1:N
            acc += G[i, k] * G[j, k] * S[i, j]
        end
        num += Δ[k]^2 * acc
    end
    den = zero(T)
    for i in 1:N, k in 1:N
        den += G[i, k]^2
    end
    isnan(den) && return T(NaN)
    den > zero(T) || return zero(T)
    return -C * num / den
end

"""
AMD eddy-diffusivity predictor (UNCLIPPED), Abkar–Bae–Moin (2016) eq. 2.7:

    num = Σₖ Σᵢ Δₖ² G[i,k] b[k] b[i],   den = Σₖ b[k]²,   κ† = -C num/den
"""
function _oracle_amd_kappa(C::T, Δ::NTuple{N, T}, G::AbstractMatrix{T},
                           b::AbstractVector{T}) where {T, N}
    num = zero(T)
    for k in 1:N, i in 1:N
        num += Δ[k]^2 * G[i, k] * b[k] * b[i]
    end
    den = zero(T)
    for k in 1:N
        den += b[k]^2
    end
    isnan(den) && return T(NaN)
    den > zero(T) || return zero(T)
    return -C * num / den
end

"""Extract the local gradient matrix `G[i,k]` at linear index `idx`."""
function _grad_matrix(grads, N::Int, idx)
    G = Matrix{Float64}(undef, N, N)
    for i in 1:N, k in 1:N
        G[i, k] = Float64(grads[(i - 1) * N + k][idx])
    end
    return G
end

_as64(grads) = map(g -> Float64.(g), grads)

"""Field of unclipped AMD eddy-viscosity predictors."""
function _oracle_nu_field(C::Float64, Δ::NTuple{N, Float64}, grads) where {N}
    out = Array{Float64}(undef, size(grads[1]))
    for idx in eachindex(out)
        out[idx] = _oracle_amd_nu(C, Δ, _grad_matrix(grads, N, idx))
    end
    return out
end

"""Field of unclipped AMD eddy-diffusivity predictors."""
function _oracle_kappa_field(C::Float64, Δ::NTuple{N, Float64}, grads, bgrads) where {N}
    out = Array{Float64}(undef, size(grads[1]))
    b = Vector{Float64}(undef, N)
    for idx in eachindex(out)
        for k in 1:N
            b[k] = Float64(bgrads[k][idx])
        end
        out[idx] = _oracle_amd_kappa(C, Δ, _grad_matrix(grads, N, idx), b)
    end
    return out
end

"""Field of Smagorinsky eddy viscosities `νₑ = (Cₛ Δ)² |S̄|`, Δ the geometric mean."""
function _oracle_smag_nu_field(C_s::Float64, Δ::NTuple{N, Float64}, grads) where {N}
    Δ_eff = prod(Δ)^(1 / N)
    out = Array{Float64}(undef, size(grads[1]))
    for idx in eachindex(out)
        out[idx] = (C_s * Δ_eff)^2 * _oracle_strain_magnitude(_grad_matrix(grads, N, idx))
    end
    return out
end

"""Field of Smagorinsky strain magnitudes."""
function _oracle_smag_strain_field(N::Int, grads)
    out = Array{Float64}(undef, size(grads[1]))
    for idx in eachindex(out)
        out[idx] = _oracle_strain_magnitude(_grad_matrix(grads, N, idx))
    end
    return out
end

# ============================================================================
# Comparison helper
# ============================================================================

"""
Elementwise relative comparison. A fixed `atol` is useless here: with filter
widths like Δz = 30 the eddy viscosity is O(10²), so absolute errors scale with
Δ². The companion `atol = rtol * maximum(abs, expected)` is likewise derived from
the field's own scale and only relaxes cells that sit near a cancellation zero.
"""
function _fields_match(actual, expected; rtol)
    length(expected) == 0 && return true
    scale = maximum(abs, expected)
    return all(isapprox.(actual, expected; rtol=rtol, atol=rtol * scale))
end

"""Worst elementwise relative deviation — reported when a comparison fails."""
function _max_rel_err(actual, expected)
    scale = maximum(abs, expected)
    scale == 0 && return maximum(abs, actual)
    return maximum(abs.(actual .- expected)) / scale
end

_randgrads(rng, T, n, field_size) = ntuple(_ -> randn(rng, T, field_size...), n)

# ============================================================================
# Parametrised AMD checks
# ============================================================================

"""
Check `compute_eddy_viscosity!` against the independent oracle for one
(dtype, Δ, clip) combination, using random fully-populated gradients.

Guards against the degenerate-assertion trap: the UNCLIPPED predictor must be
non-zero in every cell, and for `clip=true` a meaningful fraction of cells must
survive as strictly positive, so `≈ 0` can never be what makes the test pass.
"""
function _check_amd_viscosity(T::Type, Δ::NTuple{N, Float64}, field_size::NTuple{N, Int},
                              seed::Int, clip::Bool, rtol::Real) where {N}
    rng = MersenneTwister(seed)
    grads = _randgrads(rng, T, N * N, field_size)

    model = AMDModel(filter_width=Δ, field_size=field_size, clip_negative=clip, dtype=T)
    @test model.filter_width == T.(Δ)          # widths survive the constructor intact
    @test model.filter_width_sq == T.(Δ) .^ 2

    Δ64 = Float64.(T.(Δ))
    pred = _oracle_nu_field(Float64(model.C), Δ64, _as64(grads))

    @test count(!iszero, pred) == length(pred)            # nothing is trivially 0
    @test count(>(0.0), pred) >= div(length(pred), 5)     # positive cells present
    @test count(<(0.0), pred) >= div(length(pred), 5)     # negative cells present

    compute_eddy_viscosity!(model, grads...)
    actual = Float64.(Array(get_eddy_viscosity(model)))
    expected = clip ? max.(0.0, pred) : pred

    @test _fields_match(actual, expected; rtol=rtol) ||
          error("AMD νₑ mismatch T=$T Δ=$Δ clip=$clip rel_err=$(_max_rel_err(actual, expected))")

    if clip
        @test all(>=(0.0), actual)
        @test count(>(0.0), actual) >= div(length(actual), 5)
    else
        @test count(<(0.0), actual) >= div(length(actual), 5)   # negatives propagate
    end
    return nothing
end

"""
Check `compute_eddy_diffusivity!` against the independent oracle for one
(dtype, Δ, clip) combination. Same non-degeneracy guards as the viscosity check.
"""
function _check_amd_diffusivity(T::Type, Δ::NTuple{N, Float64}, field_size::NTuple{N, Int},
                                seed::Int, clip::Bool, rtol::Real) where {N}
    rng = MersenneTwister(seed)
    grads = _randgrads(rng, T, N * N, field_size)
    bgrads = _randgrads(rng, T, N, field_size)

    model = AMDModel(filter_width=Δ, field_size=field_size, clip_negative=clip, dtype=T)
    Δ64 = Float64.(T.(Δ))
    pred = _oracle_kappa_field(Float64(model.C), Δ64, _as64(grads), _as64(bgrads))

    @test count(!iszero, pred) == length(pred)
    @test count(>(0.0), pred) >= div(length(pred), 5)
    @test count(<(0.0), pred) >= div(length(pred), 5)

    compute_eddy_diffusivity!(model, grads..., bgrads...)
    actual = Float64.(Array(get_eddy_diffusivity(model)))
    expected = clip ? max.(0.0, pred) : pred

    @test _fields_match(actual, expected; rtol=rtol) ||
          error("AMD κₑ mismatch T=$T Δ=$Δ clip=$clip rel_err=$(_max_rel_err(actual, expected))")

    if clip
        @test all(>=(0.0), actual)
        @test count(>(0.0), actual) >= div(length(actual), 5)
    else
        @test count(<(0.0), actual) >= div(length(actual), 5)
    end
    return nothing
end

@testset "LES Models CPU" begin

    # ------------------------------------------------------------------
    # Smagorinsky
    # ------------------------------------------------------------------
    @testset "Smagorinsky νₑ = (Cₛ Δ)² |S̄| against closed form" begin
        # Uniform 2-D field, checked against the hand-evaluated closed form.
        field_size_2d = (4, 4)
        C_s = 0.2
        filter_width = (2.0, 1.0)
        u_x = fill(1.0, field_size_2d)
        u_y = fill(0.5, field_size_2d)
        v_x = fill(-0.25, field_size_2d)
        v_y = fill(-0.75, field_size_2d)
        model = SmagorinskyModel(C_s=C_s, filter_width=filter_width, field_size=field_size_2d)
        compute_eddy_viscosity!(model, u_x, u_y, v_x, v_y)

        Δ = prod(filter_width)^(1 / 2)
        S12 = 0.5 * (0.5 - 0.25)
        expected_strain = sqrt(2 * (1.0^2 + (-0.75)^2 + 2 * S12^2))
        expected = (C_s * Δ)^2 * expected_strain
        @test expected > 0
        @test all(isapprox.(get_eddy_viscosity(model), expected; rtol=1e-14))
        @test all(isapprox.(model.strain_magnitude, expected_strain; rtol=1e-14))

        # Random anisotropic 3-D field against the index-notation oracle.
        rng = MersenneTwister(1234)
        fs3 = (4, 4, 4)
        grads = _randgrads(rng, Float64, 9, fs3)
        Δ3 = (2.0, 0.25, 8.0)
        m3 = SmagorinskyModel(C_s=0.17, filter_width=Δ3, field_size=fs3)
        compute_eddy_viscosity!(m3, grads...)
        exp_nu = _oracle_smag_nu_field(0.17, Δ3, _as64(grads))
        exp_strain = _oracle_smag_strain_field(3, _as64(grads))
        @test all(>(0.0), exp_nu)
        @test _fields_match(get_eddy_viscosity(m3), exp_nu; rtol=1e-13)
        @test _fields_match(m3.strain_magnitude, exp_strain; rtol=1e-13)
    end

    @testset "compute_sgs_stress τᵢⱼ = -2 νₑ S̄ᵢⱼ" begin
        fs = (3, 3)
        model = SmagorinskyModel(C_s=0.2, filter_width=(2.0, 1.0), field_size=fs)
        rng = MersenneTwister(55)
        g = _randgrads(rng, Float64, 4, fs)
        compute_eddy_viscosity!(model, g...)
        νₑ = copy(get_eddy_viscosity(model))
        @test all(>(0.0), νₑ)

        S11 = g[1]
        S12 = (g[2] .+ g[3]) ./ 2
        S22 = g[4]
        τ11, τ12, τ22 = compute_sgs_stress(model, S11, S12, S22)
        @test all(isapprox.(τ11, -2 .* νₑ .* S11; rtol=1e-14))
        @test all(isapprox.(τ12, -2 .* νₑ .* S12; rtol=1e-14))
        @test all(isapprox.(τ22, -2 .* νₑ .* S22; rtol=1e-14))

        # 3-D returns six deviatoric components in (11,12,13,22,23,33) order.
        fs3 = (2, 2, 2)
        m3 = SmagorinskyModel(C_s=0.2, filter_width=(1.0, 2.0, 4.0), field_size=fs3)
        g3 = _randgrads(MersenneTwister(56), Float64, 9, fs3)
        compute_eddy_viscosity!(m3, g3...)
        ν3 = copy(get_eddy_viscosity(m3))
        S = ntuple(i -> fill(0.1 * i, fs3), 6)
        τ = compute_sgs_stress(m3, S...)
        @test length(τ) == 6
        for i in 1:6
            @test all(isapprox.(τ[i], -2 .* ν3 .* S[i]; rtol=1e-14))
        end
    end

    @testset "reset! clears every cached field" begin
        fs = (3, 3)
        smag = SmagorinskyModel(C_s=0.2, filter_width=(2.0, 1.0), field_size=fs)
        g = _randgrads(MersenneTwister(7), Float64, 4, fs)
        compute_eddy_viscosity!(smag, g...)
        @test any(!iszero, smag.eddy_viscosity)
        @test any(!iszero, smag.strain_magnitude)
        reset!(smag)
        @test all(iszero, smag.eddy_viscosity)
        # Regression: |S̄| feeds sgs_dissipation, so a reset that leaves it populated
        # lets a stale strain field flow into a diagnostic.
        @test all(iszero, smag.strain_magnitude)
        @test all(iszero, sgs_dissipation(smag, fill(1.0, fs)))

        amd = AMDModel(filter_width=(1.0, 0.01, 20.0), field_size=(3, 3, 3), clip_negative=false)
        g3 = _randgrads(MersenneTwister(8), Float64, 9, (3, 3, 3))
        b3 = _randgrads(MersenneTwister(9), Float64, 3, (3, 3, 3))
        compute_eddy_viscosity!(amd, g3...)
        compute_eddy_diffusivity!(amd, g3..., b3...)
        @test any(!iszero, amd.eddy_viscosity)
        @test any(!iszero, amd.eddy_diffusivity)
        reset!(amd)
        @test all(iszero, amd.eddy_viscosity)
        @test all(iszero, amd.eddy_diffusivity)
    end

    # ------------------------------------------------------------------
    # AMD eddy viscosity — random gradients, strongly anisotropic widths
    # ------------------------------------------------------------------
    @testset "AMD νₑ vs independent oracle (2-D, anisotropic)" begin
        for clip in (true, false)
            _check_amd_viscosity(Float64, (0.002, 25.0), (12, 12), 101, clip, 1e-12)
            _check_amd_viscosity(Float64, (7.0, 0.05), (12, 12), 102, clip, 1e-12)
        end
    end

    @testset "AMD νₑ vs independent oracle (3-D, anisotropic)" begin
        for clip in (true, false)
            # Extreme spread: Δ² spans 1e-6 … 9e2 across the three directions.
            _check_amd_viscosity(Float64, (1.0, 0.001, 30.0), (6, 6, 6), 201, clip, 1e-12)
            _check_amd_viscosity(Float64, (0.5, 12.0, 3.0), (6, 6, 6), 202, clip, 1e-12)
        end
    end

    @testset "AMD νₑ in Float32 (CPU)" begin
        for clip in (true, false)
            _check_amd_viscosity(Float32, (1.0, 0.001, 30.0), (6, 6, 6), 301, clip, 1e-5)
            _check_amd_viscosity(Float32, (0.002, 25.0), (10, 10), 302, clip, 1e-5)
        end
    end

    # ------------------------------------------------------------------
    # AMD eddy diffusivity — the same treatment
    # ------------------------------------------------------------------
    @testset "AMD κₑ vs independent oracle (2-D/3-D, anisotropic)" begin
        for clip in (true, false)
            _check_amd_diffusivity(Float64, (0.002, 25.0), (12, 12), 401, clip, 1e-12)
            _check_amd_diffusivity(Float64, (1.0, 0.001, 30.0), (6, 6, 6), 402, clip, 1e-12)
            _check_amd_diffusivity(Float64, (0.5, 12.0, 3.0), (6, 6, 6), 403, clip, 1e-12)
        end
    end

    @testset "AMD κₑ in Float32 (CPU)" begin
        for clip in (true, false)
            _check_amd_diffusivity(Float32, (1.0, 0.001, 30.0), (6, 6, 6), 501, clip, 1e-5)
            _check_amd_diffusivity(Float32, (0.002, 25.0), (10, 10), 502, clip, 1e-5)
        end
    end

    # ------------------------------------------------------------------
    # Anisotropy is actually exercised
    # ------------------------------------------------------------------
    @testset "every Δₖ changes the answer (anisotropy regression)" begin
        # The previous 3-D test fed eight of nine gradients as zero with isotropic
        # Δ, so Δy and Δz were provably never read: Δ=(1,1,1), (1,7,13) and (1,0,0)
        # all produced bit-identical output. Hold random gradients fixed and vary
        # ONE width at a time; every cell must move, and the moved field must still
        # match the oracle.
        rng = MersenneTwister(606)
        fs = (5, 5, 5)
        grads = _randgrads(rng, Float64, 9, fs)
        bgrads = _randgrads(rng, Float64, 3, fs)
        base = (1.0, 1.0, 1.0)

        function nu_for(Δ)
            m = AMDModel(filter_width=Δ, field_size=fs, clip_negative=false)
            return copy(compute_eddy_viscosity!(m, grads...)), Float64(m.C)
        end
        function kappa_for(Δ)
            m = AMDModel(filter_width=Δ, field_size=fs, clip_negative=false)
            return copy(compute_eddy_diffusivity!(m, grads..., bgrads...)), Float64(m.C)
        end

        ν_base, C = nu_for(base)
        κ_base, _ = kappa_for(base)
        @test count(!iszero, ν_base) == length(ν_base)
        @test count(!iszero, κ_base) == length(κ_base)
        @test _fields_match(ν_base, _oracle_nu_field(C, base, _as64(grads)); rtol=1e-12)
        @test _fields_match(κ_base, _oracle_kappa_field(C, base, _as64(grads), _as64(bgrads)); rtol=1e-12)

        for k in 1:3
            Δk = ntuple(j -> j == k ? 7.0 : 1.0, 3)
            νk, _ = nu_for(Δk)
            κk, _ = kappa_for(Δk)
            # Every single cell must respond to Δₖ, not merely "the field differs".
            @test count(νk .!= ν_base) == length(ν_base)
            @test count(κk .!= κ_base) == length(κ_base)
            @test _fields_match(νk, _oracle_nu_field(C, Δk, _as64(grads)); rtol=1e-12)
            @test _fields_match(κk, _oracle_kappa_field(C, Δk, _as64(grads), _as64(bgrads)); rtol=1e-12)
        end
    end

    # ------------------------------------------------------------------
    # Regression tests for the audited defects
    # ------------------------------------------------------------------
    @testset "gradient size validation fires unconditionally" begin
        # Deliberately NOT inside @boundscheck: it must survive --check-bounds=no,
        # because the kernels run @inbounds and would read past an undersized array.
        fs = (4, 4)
        good = fill(0.1, fs)
        small = fill(0.1, (3, 3))
        large = fill(0.1, (5, 5))

        amd = AMDModel(filter_width=(1.0, 3.0), field_size=fs)
        @test_throws DimensionMismatch compute_eddy_viscosity!(amd, good, good, good, small)
        @test_throws DimensionMismatch compute_eddy_viscosity!(amd, large, good, good, good)
        @test_throws DimensionMismatch compute_eddy_diffusivity!(amd, good, good, good, good, good, small)

        smag = SmagorinskyModel(filter_width=(1.0, 3.0), field_size=fs)
        @test_throws DimensionMismatch compute_eddy_viscosity!(smag, good, small, good, good)

        fs3 = (3, 3, 3)
        good3 = fill(0.1, fs3)
        small3 = fill(0.1, (2, 3, 3))
        amd3 = AMDModel(filter_width=(1.0, 0.01, 20.0), field_size=fs3)
        @test_throws DimensionMismatch compute_eddy_viscosity!(
            amd3, good3, good3, good3, good3, good3, good3, good3, good3, small3)
        smag3 = SmagorinskyModel(filter_width=(1.0, 0.01, 20.0), field_size=fs3)
        @test_throws DimensionMismatch compute_eddy_viscosity!(
            smag3, small3, good3, good3, good3, good3, good3, good3, good3, good3)

        # The matching-size call still works, so the check is not just "always throws".
        @test compute_eddy_viscosity!(amd, good, good, good, good) === amd.eddy_viscosity
    end

    @testset "κₑ is exactly invariant under b → αb" begin
        # Mathematical identity: numerator and denominator are both quadratic in b.
        # The old absolute `100*eps(T)` guard broke it — a Float64 scalar in
        # mixing-ratio units (α ≈ 1e-8) returned identically zero.
        rng = MersenneTwister(4242)
        fs = (5, 5, 5)
        Δ = (1.0, 0.001, 30.0)
        grads = _randgrads(rng, Float64, 9, fs)
        b = _randgrads(rng, Float64, 3, fs)
        model = AMDModel(filter_width=Δ, field_size=fs, clip_negative=false)

        κ_ref = copy(compute_eddy_diffusivity!(model, grads..., b...))
        @test count(!iszero, κ_ref) == length(κ_ref)
        @test _fields_match(κ_ref, _oracle_kappa_field(Float64(model.C), Δ, _as64(grads), _as64(b));
                            rtol=1e-12)

        for α in (1.0, 1e-2, 1e-4, 1e-6, 1e-8)
            bα = map(x -> α .* x, b)
            κα = copy(compute_eddy_diffusivity!(model, grads..., bα...))
            @test count(!iszero, κα) == length(κα)          # not silently zeroed
            @test _fields_match(κα, κ_ref; rtol=1e-12) ||
                  error("κₑ scale invariance broken at α=$α, rel_err=$(_max_rel_err(κα, κ_ref))")
        end

        # Float32 counterpart of the same guard bug: with 100*eps(Float32) ≈ 1.2e-5,
        # an ordinary weak-gradient field had |∇b|² below the threshold and was zeroed.
        fs2 = (8, 8)
        rng32 = MersenneTwister(4243)
        g32 = _randgrads(rng32, Float32, 4, fs2)
        b32 = map(x -> 1.0f-3 .* x, _randgrads(rng32, Float32, 2, fs2))
        m32 = AMDModel(filter_width=(1.0, 0.002), field_size=fs2, dtype=Float32, clip_negative=false)
        κ32 = compute_eddy_diffusivity!(m32, g32..., b32...)
        @test count(!iszero, κ32) == length(κ32)
        @test _fields_match(Float64.(κ32),
                            _oracle_kappa_field(Float64(m32.C), Float64.(m32.filter_width),
                                                _as64(g32), _as64(b32)); rtol=1e-5)
    end

    @testset "νₑ is homogeneous of degree 1 in the gradients" begin
        # ν(λG) = λ ν(G): numerator is cubic, denominator quadratic in G. The old
        # absolute epsilon guard zeroed the result for small but perfectly valid λ.
        rng = MersenneTwister(777)
        fs = (5, 5, 5)
        Δ = (1.0, 0.001, 30.0)
        grads = _randgrads(rng, Float64, 9, fs)
        model = AMDModel(filter_width=Δ, field_size=fs, clip_negative=false)

        ν1 = copy(compute_eddy_viscosity!(model, grads...))
        @test count(!iszero, ν1) == length(ν1)

        for λ in (1e-2, 1e-5, 1e-8)
            gλ = map(g -> λ .* g, grads)
            νλ = copy(compute_eddy_viscosity!(model, gλ...))
            @test count(!iszero, νλ) == length(νλ)          # not silently zeroed
            @test _fields_match(νλ, λ .* ν1; rtol=1e-12) ||
                  error("νₑ homogeneity broken at λ=$λ, rel_err=$(_max_rel_err(νλ, λ .* ν1))")
        end

        # Float32 weak-gradient field: |∇u|² ≈ 4e-6 < 100*eps(Float32) ≈ 1.2e-5, so the
        # old guard returned all zeros here.
        fs2 = (8, 8)
        g32 = map(x -> 1.0f-3 .* x, _randgrads(MersenneTwister(778), Float32, 4, fs2))
        m32 = AMDModel(filter_width=(1.0, 0.002), field_size=fs2, dtype=Float32, clip_negative=false)
        ν32 = compute_eddy_viscosity!(m32, g32...)
        @test count(!iszero, ν32) == length(ν32)
        @test _fields_match(Float64.(ν32),
                            _oracle_nu_field(Float64(m32.C), Float64.(m32.filter_width), _as64(g32));
                            rtol=1e-5)
    end

    @testset "NaN gradients propagate (blow-ups are not laundered into zero)" begin
        fs = (2, 2, 2)
        rng = MersenneTwister(31337)
        for clip in (true, false)
            grads = _randgrads(rng, Float64, 9, fs)
            poisoned = copy(grads[5])
            poisoned[1, 1, 1] = NaN
            gN = ntuple(i -> i == 5 ? poisoned : grads[i], 9)

            amd = AMDModel(filter_width=(1.0, 0.001, 30.0), field_size=fs, clip_negative=clip)
            ν = compute_eddy_viscosity!(amd, gN...)
            @test isnan(ν[1, 1, 1])
            @test count(isnan, ν) == 1
            @test all(isfinite, ν[2:end])

            # The scalar gradients are finite, so only the numerator carries the NaN.
            b = _randgrads(rng, Float64, 3, fs)
            κ = compute_eddy_diffusivity!(amd, gN..., b...)
            @test isnan(κ[1, 1, 1])
            @test count(isnan, κ) == 1
        end

        smag = SmagorinskyModel(filter_width=(1.0, 0.001, 30.0), field_size=fs)
        gs = _randgrads(rng, Float64, 9, fs)
        bad = copy(gs[1]); bad[2, 1, 1] = NaN
        νs = compute_eddy_viscosity!(smag, bad, gs[2:9]...)
        @test isnan(νs[2, 1, 1])
        @test count(isnan, νs) == 1
    end

    @testset "filter width mutation takes effect; set_filter_width! stays in sync" begin
        fs = (4, 4, 4)
        Δ = (1.0, 0.001, 30.0)
        grads = _randgrads(MersenneTwister(2468), Float64, 9, fs)
        model = AMDModel(filter_width=Δ, field_size=fs, clip_negative=false)
        ν1 = copy(compute_eddy_viscosity!(model, grads...))
        @test count(!iszero, ν1) == length(ν1)

        # Doubling every Δ multiplies the numerator by exactly 4 and leaves the
        # denominator alone — a power-of-two scaling, so this is bit-exact.
        set_filter_width!(model, (2.0 .* Δ))
        ν2 = copy(compute_eddy_viscosity!(model, grads...))
        @test ν2 == 4 .* ν1
        @test model.filter_width == 2.0 .* Δ
        @test model.filter_width_sq == (2.0 .* Δ) .^ 2      # cached square kept in sync

        # Direct field assignment also takes effect (kernels re-derive Δ² per call).
        model.filter_width = Δ
        ν3 = copy(compute_eddy_viscosity!(model, grads...))
        @test ν3 == ν1
        @test _fields_match(ν3, _oracle_nu_field(Float64(model.C), Δ, _as64(grads)); rtol=1e-12)

        # Diffusivity responds to the same mutation.
        b = _randgrads(MersenneTwister(2469), Float64, 3, fs)
        κ1 = copy(compute_eddy_diffusivity!(model, grads..., b...))
        set_filter_width!(model, (2.0 .* Δ))
        κ2 = copy(compute_eddy_diffusivity!(model, grads..., b...))
        @test count(!iszero, κ1) == length(κ1)
        @test κ2 == 4 .* κ1

        # Smagorinsky: effective_delta is the geometric mean and must follow.
        smag = SmagorinskyModel(C_s=0.2, filter_width=(1.0, 2.0, 4.0), field_size=fs)
        gs = _randgrads(MersenneTwister(2470), Float64, 9, fs)
        νa = copy(compute_eddy_viscosity!(smag, gs...))
        set_filter_width!(smag, (2.0, 4.0, 8.0))
        @test smag.filter_width == (2.0, 4.0, 8.0)
        @test smag.effective_delta == prod((2.0, 4.0, 8.0))^(1 / 3)
        νb = copy(compute_eddy_viscosity!(smag, gs...))
        @test all(>(0.0), νa)
        @test _fields_match(νb, 4 .* νa; rtol=1e-13)
        @test _fields_match(νb, _oracle_smag_nu_field(0.2, (2.0, 4.0, 8.0), _as64(gs)); rtol=1e-13)

        # set_filter_width! validates like the constructor.
        @test_throws ArgumentError set_filter_width!(smag, (-1.0, 1.0, 1.0))
        @test_throws ArgumentError set_filter_width!(model, (1.0, 0.0, 1.0))
    end

    @testset "constructor and setter validation" begin
        # Negative widths used to construct an AMD model silently (the sign vanishes
        # in the squaring) while raising DomainError from the Smagorinsky mean.
        @test_throws ArgumentError AMDModel(filter_width=(-1.0, -1.0, -1.0), field_size=(2, 2, 2))
        @test_throws ArgumentError AMDModel(filter_width=(1.0, -2.0), field_size=(2, 2))
        @test_throws ArgumentError SmagorinskyModel(filter_width=(-1.0, 1.0), field_size=(2, 2))

        # Zero widths.
        @test_throws ArgumentError AMDModel(filter_width=(0.0, 1.0), field_size=(2, 2))
        @test_throws ArgumentError SmagorinskyModel(filter_width=(1.0, 0.0, 1.0), field_size=(2, 2, 2))

        # Non-finite widths.
        @test_throws ArgumentError AMDModel(filter_width=(NaN, 1.0), field_size=(2, 2))
        @test_throws ArgumentError SmagorinskyModel(filter_width=(Inf, 1.0), field_size=(2, 2))

        # Negative / non-finite model constants.
        @test_throws ArgumentError AMDModel(C=-5.0, filter_width=(1.0, 1.0), field_size=(2, 2))
        @test_throws ArgumentError AMDModel(C=NaN, filter_width=(1.0, 1.0), field_size=(2, 2))
        @test_throws ArgumentError SmagorinskyModel(C_s=-0.1, filter_width=(1.0, 1.0), field_size=(2, 2))

        # Zero / negative grids.
        @test_throws ArgumentError AMDModel(filter_width=(1.0, 1.0), field_size=(0, 2))
        @test_throws ArgumentError AMDModel(filter_width=(1.0, 1.0), field_size=(2, -3))
        @test_throws ArgumentError SmagorinskyModel(filter_width=(1.0, 1.0), field_size=(2, 0))

        # Valid neighbours of every rejected case still construct.
        @test AMDModel(C=0.0, filter_width=(1.0, 1.0), field_size=(1, 1)) isa AMDModel
        @test SmagorinskyModel(C_s=0.0, filter_width=(1.0, 1.0), field_size=(1, 1)) isa SmagorinskyModel

        # set_constant! rejects NaN (and negatives) but accepts sane values.
        amd = AMDModel(filter_width=(1.0, 1.0), field_size=(2, 2))
        smag = SmagorinskyModel(filter_width=(1.0, 1.0), field_size=(2, 2))
        @test_throws ArgumentError set_constant!(amd, NaN)
        @test_throws ArgumentError set_constant!(smag, NaN)
        @test_throws ArgumentError set_constant!(amd, -1.0)
        @test_throws ArgumentError set_constant!(smag, -1.0)
        set_constant!(amd, 0.3)
        set_constant!(smag, 0.11)
        @test amd.C == 0.3
        @test smag.C_s == 0.11
    end

    @testset "get_eddy_diffusivity on a non-AMD model is an informative error" begin
        smag = SmagorinskyModel(filter_width=(1.0, 1.0), field_size=(2, 2))
        @test_throws ArgumentError get_eddy_diffusivity(smag)
        err = try
            get_eddy_diffusivity(smag)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Pr_t", err.msg)
        @test occursin("SmagorinskyModel", err.msg)
        # The AMD model of course still answers.
        amd = AMDModel(filter_width=(1.0, 1.0), field_size=(2, 2))
        @test get_eddy_diffusivity(amd) === amd.eddy_diffusivity
    end

    @testset "PencilArray gradients are rejected" begin
        # A PencilArray reports the same `size` but stores data in (possibly
        # permuted) parent order, so a size check alone lets it through and the
        # kernels then pair the wrong cells.
        _MPI.Initialized() || _MPI.Init()
        pencil_array = try
            topo = _PA.MPITopology(_MPI.COMM_WORLD, (1, 1))
            pen = _PA.Pencil(topo, (4, 4, 4), (2, 3))
            pa = _PA.PencilArray{Float64}(undef, pen)
            fill!(pa, 0.1)
            pa
        catch
            nothing
        end

        if pencil_array === nothing
            @test_skip "PencilArray could not be constructed in this environment"
        else
            fs = size(pencil_array)
            model = AMDModel(filter_width=(1.0, 0.01, 20.0), field_size=fs)
            dense = fill(0.1, fs)
            @test_throws ArgumentError compute_eddy_viscosity!(
                model, pencil_array, dense, dense, dense, dense, dense, dense, dense, dense)
            @test_throws ArgumentError compute_eddy_viscosity!(
                model, dense, dense, dense, dense, dense, dense, dense, dense, pencil_array)
            @test_throws ArgumentError compute_eddy_diffusivity!(
                model, dense, dense, dense, dense, dense, dense, dense, dense, dense,
                pencil_array, dense, dense)

            err = try
                compute_eddy_viscosity!(model, pencil_array, dense, dense, dense,
                                        dense, dense, dense, dense, dense)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("PencilArray", err.msg)

            smag = SmagorinskyModel(filter_width=(1.0, 0.01, 20.0), field_size=fs)
            @test_throws ArgumentError compute_eddy_viscosity!(
                smag, pencil_array, dense, dense, dense, dense, dense, dense, dense, dense)

            # …while the rank-local dense parent is accepted.
            raw = parent(pencil_array)
            @test size(raw) == fs
            @test compute_eddy_viscosity!(model, raw, dense, dense, dense,
                                          dense, dense, dense, dense, dense) ===
                  model.eddy_viscosity
        end
    end

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @testset "sgs_dissipation and mean_sgs_dissipation" begin
        smag = SmagorinskyModel(filter_width=(1.0, 1.0, 1.0), field_size=(2, 2, 2))
        fill!(smag.eddy_viscosity, 0.5)
        mags = fill(0.3, (2, 2, 2))
        diss = sgs_dissipation(smag, mags)
        # εₛₛ = νₑ|S̄|² with |S̄| = √(2 S̄ᵢⱼS̄ᵢⱼ); no extra factor of 2 (CPU audit 2026-06-20).
        @test all(diss .== 0.5 .* mags .^ 2)
        @test all(diss .!= 0)
        # An extra factor of 2 anywhere would show up here.
        @test diss[1] == 0.5 * 0.3^2
        avg = mean_sgs_dissipation(smag, mags)
        @test isapprox(avg, mean(diss); atol=1e-12)

        # Non-uniform νₑ so the mean is not trivially one value.
        smag.eddy_viscosity .= reshape(collect(1.0:8.0), (2, 2, 2))
        m2 = fill(2.0, (2, 2, 2))
        @test all(sgs_dissipation(smag, m2) .== 4 .* smag.eddy_viscosity)
        @test isapprox(mean_sgs_dissipation(smag, m2), mean(4 .* smag.eddy_viscosity); atol=1e-12)
        @test isapprox(mean_eddy_viscosity(smag), mean(1.0:8.0); atol=1e-12)
        @test max_eddy_viscosity(smag) == 8.0
        @test get_filter_width(smag) == (1.0, 1.0, 1.0)
    end
end

if _CUDA_AVAILABLE
    @testset "LES Models GPU" begin
        CUDA.allowscalar(false)
        # The implementation no longer keeps separate CPU/GPU code paths — both
        # broadcast the same scalar kernel — so these tests only check that the
        # device arrays and the host→device coercion behave, against the same
        # independent oracle used on the CPU.
        rng = MersenneTwister(31415)

        # Smagorinsky, anisotropic widths, random gradients.
        fs = (4, 4)
        Δ = (2.0, 0.5)
        g = _randgrads(rng, Float32, 4, fs)
        model = SmagorinskyModel(C_s=0.17f0, filter_width=Δ, field_size=fs,
                                 architecture=GPU(), dtype=Float32)
        compute_eddy_viscosity!(model, map(CuArray, g)...)
        expected = _oracle_smag_nu_field(Float64(model.C_s), Float64.(model.filter_width), _as64(g))
        @test all(>(0.0), expected)
        @test _fields_match(Float64.(Array(get_eddy_viscosity(model))), expected; rtol=1e-5)

        # CPU gradients are moved to the device automatically.
        reset!(model)
        @test all(iszero, Array(get_eddy_viscosity(model)))
        compute_eddy_viscosity!(model, g...)
        @test _fields_match(Float64.(Array(get_eddy_viscosity(model))), expected; rtol=1e-5)

        # AMD 3-D with strongly anisotropic widths, clipping on and off.
        fs3 = (4, 4, 4)
        Δ3 = (1.0, 0.001, 30.0)
        g3 = _randgrads(rng, Float32, 9, fs3)
        for clip in (true, false)
            amd = AMDModel(filter_width=Δ3, field_size=fs3, clip_negative=clip,
                           architecture=GPU(), dtype=Float32)
            pred = _oracle_nu_field(Float64(amd.C), Float64.(amd.filter_width), _as64(g3))
            @test count(!iszero, pred) == length(pred)
            @test count(>(0.0), pred) >= div(length(pred), 5)
            @test count(<(0.0), pred) >= div(length(pred), 5)
            compute_eddy_viscosity!(amd, map(CuArray, g3)...)
            actual = Float64.(Array(get_eddy_viscosity(amd)))
            @test _fields_match(actual, clip ? max.(0.0, pred) : pred; rtol=1e-5)
        end

        # Eddy diffusivity with CPU gradients (host→device conversion check).
        amd = AMDModel(filter_width=Δ3, field_size=fs3, clip_negative=false,
                       architecture=GPU(), dtype=Float32)
        b3 = _randgrads(rng, Float32, 3, fs3)
        compute_eddy_diffusivity!(amd, g3..., b3...)
        pred_k = _oracle_kappa_field(Float64(amd.C), Float64.(amd.filter_width),
                                     _as64(g3), _as64(b3))
        @test count(!iszero, pred_k) == length(pred_k)
        @test _fields_match(Float64.(Array(get_eddy_diffusivity(amd))), pred_k; rtol=1e-5)

        # Validation still fires on the device.
        @test_throws DimensionMismatch compute_eddy_viscosity!(
            amd, CuArray(zeros(Float32, 2, 2, 2)), map(CuArray, g3[2:9])...)
    end
else
    @testset "LES Models GPU" begin
        @test_skip "CUDA not available"
    end
end
