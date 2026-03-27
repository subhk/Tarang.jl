using Test
using Tarang
using Statistics

const _CUDA_AVAILABLE = try
    @eval begin
        using CUDA
        CUDA.functional()
    end
catch
    false
end

function _smag_expected(C_s, filter_width, S11, S22, S12)
    Δ = prod(filter_width)^(1 / length(filter_width))
    CΔ_sq = (C_s * Δ)^2
    S_mag = sqrt(2 * (S11^2 + S22^2 + 2 * S12^2))
    return CΔ_sq * S_mag, S_mag
end

function _amd_predictor_2d(C, Δx², Δy², u_x, u_y, v_x, v_y)
    half = 0.5
    eps_T = 100 * eps(Float64)
    S11 = u_x
    S22 = v_y
    S12 = half * (u_y + v_x)
    denom = u_x^2 + u_y^2 + v_x^2 + v_y^2
    numer_x = Δx² * (u_x^2 * S11 + 2u_x * v_x * S12 + v_x^2 * S22)
    numer_y = Δy² * (u_y^2 * S11 + 2u_y * v_y * S12 + v_y^2 * S22)
    numer = -(numer_x + numer_y)
    return denom > eps_T ? C * numer / denom : 0.0
end

function _amd_predictor_3d(C, Δx², Δy², Δz², u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z)
    half = 0.5
    eps_T = 100 * eps(Float64)
    S11 = u_x
    S22 = v_y
    S33 = w_z
    S12 = half * (u_y + v_x)
    S13 = half * (u_z + w_x)
    S23 = half * (v_z + w_y)
    denom = u_x^2 + u_y^2 + u_z^2 + v_x^2 + v_y^2 + v_z^2 + w_x^2 + w_y^2 + w_z^2
    numer_x = Δx² * (u_x^2 * S11 + v_x^2 * S22 + w_x^2 * S33 + 2 * (u_x * v_x * S12 + u_x * w_x * S13 + v_x * w_x * S23))
    numer_y = Δy² * (u_y^2 * S11 + v_y^2 * S22 + w_y^2 * S33 + 2 * (u_y * v_y * S12 + u_y * w_y * S13 + v_y * w_y * S23))
    numer_z = Δz² * (u_z^2 * S11 + v_z^2 * S22 + w_z^2 * S33 + 2 * (u_z * v_z * S12 + u_z * w_z * S13 + v_z * w_z * S23))
    numer = -(numer_x + numer_y + numer_z)
    return denom > eps_T ? C * numer / denom : 0.0
end

@testset "LES Models CPU" begin
    field_size_2d = (4, 4)
    C_s = 0.2
    filter_width = (2.0, 1.0)
    u_x = fill(1.0, field_size_2d)
    u_y = fill(0.5, field_size_2d)
    v_x = fill(-0.25, field_size_2d)
    v_y = fill(-0.75, field_size_2d)
    model = SmagorinskyModel(C_s=C_s, filter_width=filter_width, field_size=field_size_2d)
    compute_eddy_viscosity!(model, u_x, u_y, v_x, v_y)
    expected, expected_strain = _smag_expected(C_s, filter_width, 1.0, -0.75, 0.5 * (0.5 - 0.25))
    @test all(isapprox.(get_eddy_viscosity(model), expected; atol=1e-12))
    @test all(isapprox.(model.strain_magnitude, expected_strain; atol=1e-12))
    τ11, τ12, τ22 = compute_sgs_stress(model, u_x, (u_y .+ v_x) ./ 2, v_y)
    @test all(isapprox.(τ11, fill(-2 * expected * 1.0, field_size_2d); atol=1e-12))
    @test all(isapprox.(τ22, fill(-2 * expected * -0.75, field_size_2d); atol=1e-12))
    reset!(model)
    @test all(model.eddy_viscosity .== 0)

    # AMD 2D with clipping
    field_size = (3, 3)
    Δ = (1.0, 2.0)
    model_amd = AMDModel(filter_width=Δ, field_size=field_size)
    u_x = fill(0.5, field_size)
    u_y = fill(0.1, field_size)
    v_x = fill(-0.3, field_size)
    v_y = fill(0.4, field_size)
    compute_eddy_viscosity!(model_amd, u_x, u_y, v_x, v_y)
    predictor = _amd_predictor_2d(model_amd.C, Δ[1]^2, Δ[2]^2, 0.5, 0.1, -0.3, 0.4)
    @test all(isapprox.(get_eddy_viscosity(model_amd), max(0, predictor); atol=1e-12))

    # AMD 3D without clipping to check negative values propagate
    field_size_3d = (2, 2, 2)
    model_amd3 = AMDModel(filter_width=(1.0, 1.0, 1.0), field_size=field_size_3d, clip_negative=false)
    u_x3 = fill(0.25, field_size_3d)
    zeros_arr = fill(0.0, field_size_3d)
    compute_eddy_viscosity!(model_amd3, u_x3, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr, zeros_arr)
    predictor3 = _amd_predictor_3d(model_amd3.C, 1.0, 1.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    @test all(isapprox.(get_eddy_viscosity(model_amd3), predictor3; atol=1e-12))

    # Eddy diffusivity
    model_diff = AMDModel(filter_width=(1.0, 1.0), field_size=field_size)
    w_x = fill(0.2, field_size)
    w_y = fill(-0.1, field_size)
    b_x = fill(0.4, field_size)
    b_y = fill(-0.5, field_size)
    compute_eddy_diffusivity!(model_diff, w_x, w_y, b_x, b_y)
    denom = 0.4^2 + (-0.5)^2
    numer = -(1.0^2 * 0.2 * 0.4 + 1.0^2 * -0.1 * -0.5)
    expected_diff = model_diff.C * numer / denom
    @test all(isapprox.(get_eddy_diffusivity(model_diff), max(0, expected_diff); atol=1e-12))

    # Diagnostics
    smag = SmagorinskyModel(filter_width=(1.0, 1.0, 1.0), field_size=(2, 2, 2))
    fill!(smag.eddy_viscosity, 0.5)
    mags = fill(0.3, (2, 2, 2))
    diss = sgs_dissipation(smag, mags)
    @test all(diss .== 2 * 0.5 .* mags .^ 2)
    avg = mean_sgs_dissipation(smag, mags)
    @test isapprox(avg, mean(diss); atol=1e-12)
end

if _CUDA_AVAILABLE
    @testset "LES Models GPU" begin
        CUDA.allowscalar(false)
        field_size = (2, 2)
        u_x = fill(0.8f0, field_size)
        u_y = fill(0.3f0, field_size)
        v_x = fill(-0.2f0, field_size)
        v_y = fill(0.1f0, field_size)
        filter_width = (1.0f0, 1.0f0)
        model_gpu = SmagorinskyModel(C_s=0.17f0, filter_width=filter_width, field_size=field_size, architecture=GPU(), dtype=Float32)
        compute_eddy_viscosity!(model_gpu, CuArray(u_x), CuArray(u_y), CuArray(v_x), CuArray(v_y))
        expected, _ = _smag_expected(0.17, (1.0, 1.0), 0.8, 0.1, 0.5 * (0.3 - 0.2))
        vals_gpu = Array(get_eddy_viscosity(model_gpu))
        @test all(isapprox.(vals_gpu, expected; rtol=1e-6))

        # CPU gradients should be automatically moved to GPU
        reset!(model_gpu)
        compute_eddy_viscosity!(model_gpu, u_x, u_y, v_x, v_y)
        vals_cpu = Array(get_eddy_viscosity(model_gpu))
        @test all(isapprox.(vals_cpu, expected; rtol=1e-6))

        # AMD GPU test
        model_gpu_amd = AMDModel(filter_width=(1.0, 1.0, 1.0), field_size=(2, 2, 2), architecture=GPU(), dtype=Float32)
        ones_gpu = CuArray(fill(0.25f0, (2, 2, 2)))
        zeros_gpu = CuArray(zeros(Float32, (2, 2, 2)))
        compute_eddy_viscosity!(model_gpu_amd, ones_gpu, zeros_gpu, zeros_gpu, zeros_gpu, zeros_gpu, zeros_gpu, zeros_gpu, zeros_gpu, zeros_gpu)
        predictor = _amd_predictor_3d(model_gpu_amd.C, 1.0, 1.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        @test all(isapprox.(Array(get_eddy_viscosity(model_gpu_amd)), max(0, predictor); rtol=1e-6))

        # GPU eddy diffusivity with CPU gradients (conversion check)
        reset!(model_gpu_amd)
        w_x = fill(0.2f0, (2, 2, 2))
        w_y = fill(-0.1f0, (2, 2, 2))
        w_z = fill(0.05f0, (2, 2, 2))
        b_x = fill(0.4f0, (2, 2, 2))
        b_y = fill(-0.5f0, (2, 2, 2))
        b_z = fill(0.3f0, (2, 2, 2))
        compute_eddy_diffusivity!(model_gpu_amd, w_x, w_y, w_z, b_x, b_y, b_z)
        denom = 0.4^2 + (-0.5)^2 + 0.3^2
        numer = -(1.0 * 0.2 * 0.4 + 1.0 * -0.1 * -0.5 + 1.0 * 0.05 * 0.3)
        expected_diff = model_gpu_amd.C * numer / denom
        @test all(isapprox.(Array(get_eddy_diffusivity(model_gpu_amd)), max(0, expected_diff); rtol=1e-6))
    end
else
    @testset "LES Models GPU" begin
        @test_skip "CUDA not available"
    end
end
