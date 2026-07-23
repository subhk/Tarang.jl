"""
GPU-compatibility for the LES subgrid models WITHOUT a GPU.

The real GPU tests in test_les_models.jl need CUDA hardware and skip everywhere
else. This file exercises the same device code path on `JLArray` — GPUArrays'
CPU-backed reference GPU array — with `allowscalar(false)`, so any scalar
indexing throws exactly as it would on a CuArray. A model whose fields are
JLArrays runs `compute_eddy_viscosity!` etc. through the identical broadcast the
CUDA path uses; if it completes and matches the CPU result, the AMD/Smagorinsky
kernels are device-safe.

The models compute via one shared scalar kernel broadcast over the array type
(`eddy_visc .= _amd_nu.(...)`), so there is no separate GPU code to drift — this
guards that the shared kernel and its helpers (`_safe_quotient`, `_apply_clip`)
stay free of scalar indexing and host-only constructs.
"""

using Test
using Random
using Tarang

const _JL_OK = try
    @eval using JLArrays
    @eval using GPUArrays
    true
catch err
    @info "JLArrays/GPUArrays unavailable; skipping LES GPU-compat test" err
    false
end

if _JL_OK
    const _JL_LES_ARRAY = JLArrays.JLArray
    const _JL_LES_ARCH = Tarang.GPU(JLArrays.JLBackend())
    Tarang.is_gpu_array(::_JL_LES_ARRAY) = true
    Tarang.architecture(::_JL_LES_ARRAY) = _JL_LES_ARCH
    Tarang.on_architecture(::Tarang.GPU{JLArrays.JLBackend}, a::Array) = _JL_LES_ARRAY(a)
end

@testset "LES models GPU-compat (JLArray, no scalar indexing)" begin
    if !_JL_OK
        @test_skip "JLArrays not available"
    else
        GPUArrays.allowscalar(false)
        Random.seed!(3)

        # JLArray-backed twin of a CPU-built model: same scalars, device arrays.
        function jltwin(m::Tarang.AMDModel)
            E = eltype(m.eddy_viscosity); D = ndims(m.eddy_viscosity)
            Tarang.AMDModel{E, D, JLArray{E, D}, typeof(_JL_LES_ARCH)}(
                m.C, m.filter_width, m.filter_width_sq,
                JLArray(copy(m.eddy_viscosity)), JLArray(copy(m.eddy_diffusivity)),
                m.field_size, m.clip_negative, _JL_LES_ARCH)
        end
        function jltwin(m::Tarang.SmagorinskyModel)
            E = eltype(m.eddy_viscosity); D = ndims(m.eddy_viscosity)
            Tarang.SmagorinskyModel{E, D, JLArray{E, D}, typeof(_JL_LES_ARCH)}(
                m.C_s, m.filter_width, m.effective_delta,
                JLArray(copy(m.eddy_viscosity)), JLArray(copy(m.strain_magnitude)),
                m.field_size, _JL_LES_ARCH)
        end

        n = (6, 6, 6)
        G = [randn(n...) for _ in 1:9]     # 9 velocity gradients (component-major)
        B = [randn(n...) for _ in 1:3]     # 3 scalar gradients

        @testset "AMD compute_eddy_viscosity! / diffusivity on device" begin
            amd = AMDModel(C=1/12, filter_width=(0.1, 0.05, 0.2), field_size=n)
            compute_eddy_viscosity!(amd, G...)
            compute_eddy_diffusivity!(amd, G..., B...)

            jamd = jltwin(amd)
            compute_eddy_viscosity!(jamd, JLArray.(G)...)        # runs on device
            compute_eddy_diffusivity!(jamd, JLArray.(G)..., JLArray.(B)...)

            @test Array(get_eddy_viscosity(jamd)) ≈ get_eddy_viscosity(amd) rtol=1e-12
            @test Array(get_eddy_diffusivity(jamd)) ≈ get_eddy_diffusivity(amd) rtol=1e-12
        end

        @testset "AMD kernel edge cases on device" begin
            # 0/0 guard (zero gradients) and NaN propagation must hold in the
            # device broadcast, not just the CPU loop.
            amd = AMDModel(C=1/12, filter_width=(0.1, 0.1, 0.1), field_size=n)
            Gz = [zeros(n...) for _ in 1:9]
            jamd = jltwin(amd)
            compute_eddy_viscosity!(jamd, JLArray.(Gz)...)
            @test all(iszero, Array(get_eddy_viscosity(jamd)))    # 0/0 -> 0, not NaN

            Gn = deepcopy(G); Gn[1][1] = NaN
            jamd2 = jltwin(amd)
            compute_eddy_viscosity!(jamd2, JLArray.(Gn)...)
            @test isnan(Array(get_eddy_viscosity(jamd2))[1])      # blow-up not laundered
        end

        @testset "CPU models reject GPU gradients instead of downloading" begin
            cpu_model = SmagorinskyModel(
                C_s=0.17, filter_width=(0.1, 0.05, 0.2), field_size=n)
            @test_throws ErrorException compute_eddy_viscosity!(
                cpu_model, JLArray.(G)...)
        end

        @testset "Smagorinsky + stress + diagnostics on device" begin
            sm = SmagorinskyModel(C_s=0.17, filter_width=(0.1, 0.05, 0.2), field_size=n)
            compute_eddy_viscosity!(sm, G...)
            jsm = jltwin(sm)
            compute_eddy_viscosity!(jsm, JLArray.(G)...)
            @test Array(get_eddy_viscosity(jsm)) ≈ get_eddy_viscosity(sm) rtol=1e-12

            S = [randn(n...) for _ in 1:6]
            tau_c = compute_sgs_stress(sm, S...)
            tau_j = compute_sgs_stress(jsm, JLArray.(S)...)
            for (tj, tc) in zip(tau_j, tau_c)
                @test Array(tj) ≈ tc rtol=1e-12
            end

            @test mean_eddy_viscosity(jsm) ≈ mean_eddy_viscosity(sm) rtol=1e-12
            @test max_eddy_viscosity(jsm) ≈ max_eddy_viscosity(sm) rtol=1e-12

            reset!(jsm)
            @test all(iszero, Array(get_eddy_viscosity(jsm)))
            @test all(iszero, Array(jsm.strain_magnitude))
        end

        GPUArrays.allowscalar(true)   # restore for any later tests in the process
    end
end
