# ============================================================================
# GPU CI tests for the distributed RealFourier × Chebyshev (DCT-I) local
# primitives:  local_fft_along_dim!, gpu_dct1_along_dim! / local_dct1_along_dim!,
# and the dim-1 RealFourier truncate / Hermitian-expand pair.
#
# These tests require a CUDA GPU; they run ONLY on the GPU CI runner.
# NOTHING here has been executed on a real GPU — this is a parse-checked draft.
# ============================================================================

using Test
using Tarang
# These tests require a CUDA GPU; they run only on the GPU CI runner.
using CUDA
using MPI

# Buildkite launches one MPI process per GPU without setting
# CUDA_VISIBLE_DEVICES per rank. Select a distinct device before allocating any
# field data so NCCL does not see every rank on device 0.
const _DCT1_GPU_ARCH = if CUDA.functional()
    MPI.Initialized() || MPI.Init()
    gpu_id = MPI.Comm_rank(MPI.COMM_WORLD) % CUDA.ndevices()
    CUDA.device!(gpu_id)
    GPU(device_id=gpu_id)
else
    nothing
end

if !CUDA.functional()
    @info "test_gpu_distributed_dct1.jl skipped: no functional CUDA device"
else
    using LinearAlgebra, FFTW

    # The primitives live in the package extension (loaded once CUDA is). They
    # are not exported into `Tarang`, so reach them through the extension module
    # — the official `Base.get_extension` handle is robust across Julia versions.
    const _EXT = Base.get_extension(Tarang, :TarangCUDAExt)
    @assert _EXT !== nothing "TarangCUDAExt not loaded despite CUDA.functional()"

    const local_fft_along_dim!          = _EXT.local_fft_along_dim!
    const gpu_dct1_along_dim!           = _EXT.gpu_dct1_along_dim!
    const local_dct1_along_dim!         = _EXT.local_dct1_along_dim!
    const _realfourier_truncate         = _EXT._realfourier_truncate
    const _realfourier_hermitian_expand = _EXT._realfourier_hermitian_expand

    # ── CPU reference helpers ───────────────────────────────────────────────
    # Scale the k-th slice along `dim` by `f` (used for endpoint weight + flip).
    function _scale_slice!(a, dim::Int, k::Int, f)
        idx = ntuple(i -> i == dim ? (k:k) : Colon(), ndims(a))
        @views a[idx...] .*= f
        return a
    end

    # Forward DCT-I in the Tarang/Chebyshev convention, mirroring
    # `_chebyshev_forward` in src/core/transforms/transform_chebyshev.jl:
    #   REDFT00 → ×1/(N-1) → half-weight endpoints → flip odd-degree coeffs.
    function cpu_dct1_forward(x::AbstractArray{<:Real}, dim::Int)
        N = size(x, dim)
        c = FFTW.r2r(x, FFTW.REDFT00, (dim,)) .* (1.0 / (N - 1))
        _scale_slice!(c, dim, 1, 0.5)
        _scale_slice!(c, dim, N, 0.5)
        for k in 2:2:N
            _scale_slice!(c, dim, k, -1.0)
        end
        return c
    end

    @testset "local_fft_along_dim! round-trip" begin
        for dim in 1:3
            x = CUDA.rand(ComplexF64, 6, 5, 4); y = similar(x); z = similar(x)
            local_fft_along_dim!(y, x, dim, :forward)
            local_fft_along_dim!(z, y, dim, :backward)
            @test Array(z) ≈ Array(x)
            @test Array(y) ≈ mapslices(fft, Array(x); dims=dim)
        end
    end

    @testset "gpu_dct1_along_dim! (real) vs CPU REDFT00 reference + round-trip" begin
        for dim in 1:3
            xh = rand(Float64, 6, 5, 4)
            x  = CuArray(xh)
            c  = similar(x); r = similar(x)

            # Forward must match the framework DCT-I convention exactly.
            gpu_dct1_along_dim!(c, x, dim, :forward)
            @test Array(c) ≈ cpu_dct1_forward(xh, dim)

            # backward(forward(x)) == x  (DCT-I is its own inverse up to scaling).
            gpu_dct1_along_dim!(r, c, dim, :backward)
            @test Array(r) ≈ xh
        end
    end

    @testset "local_dct1_along_dim! (complex) round-trip + parts" begin
        for dim in 1:3
            xh = rand(ComplexF64, 6, 5, 4)
            x  = CuArray(xh)
            c  = similar(x); r = similar(x)

            local_dct1_along_dim!(c, x, dim, :forward)
            # Real and imaginary parts each transform as the real DCT-I.
            @test real.(Array(c)) ≈ cpu_dct1_forward(real.(xh), dim)
            @test imag.(Array(c)) ≈ cpu_dct1_forward(imag.(xh), dim)

            local_dct1_along_dim!(r, c, dim, :backward)
            @test Array(r) ≈ xh
        end
    end

    @testset "RealFourier dim-1 truncate / Hermitian expand round-trip" begin
        for N in (6, 8, 7)          # even and odd
            M  = div(N, 2) + 1
            xr = rand(Float64, N, 3, 2)            # real signal
            full_cpu = fft(xr, 1)                  # Hermitian-symmetric along dim 1
            full = CuArray(ComplexF64.(full_cpu))

            half = _realfourier_truncate(full)
            @test size(half, 1) == M
            @test Array(half) ≈ full_cpu[1:M, :, :]

            full2 = _realfourier_hermitian_expand(half, N)
            @test size(full2, 1) == N
            @test Array(full2) ≈ full_cpu       # symmetry rebuilds the full spectrum

            # Must agree element-for-element with the tested CPU reference.
            half_cpu = Array(half)
            for k in 1:size(half, 3), j in 1:size(half, 2)
                ref = Tarang._hermitian_full_from_half(half_cpu[:, j, k], N)
                @test Array(full2)[:, j, k] ≈ ref
            end
        end
    end
end

# ============================================================================
# Task 9 — end-to-end field-level coverage of the distributed multi-GPU DCT-I
# dispatch. The dispatch lives in ext/cuda/transforms.jl, gated by the runtime
# predicate `is_distributed_gpu(arch, nprocs) && needs_distributed_dct(field)`
# AND the layout predicate `Tarang.distributed_gpu_supported(field)`.
#
# WHAT THIS COVERS / WHERE IT RUNS:
#   * nprocs == 1 (this single-process GPU CI runner — GPU_TEST_FILES, included
#     WITHOUT mpiexec): `is_distributed_gpu` returns false (it requires nprocs>1
#     + NCCL), so a Chebyshev-containing field round-trips through the single-GPU
#     device transform. At 1 rank this is a *wiring + round-trip smoke test*, and (when
#     NCCL is present) the inner block drives distributed_forward_dct! /
#     distributed_backward_dct! directly on CuArrays at 1 rank.
#   * nprocs == 2 (JuliaGPU Buildkite, launched under mpiexec WITH NCCL): the
#     SAME forward_transform!/backward_transform! calls take the LIVE distributed
#     path (distributed_gpu_forward/backward_transform!). The .buildkite GPU
#     pipeline runs this file from DISTRIBUTED_GPU_TEST_FILES as well as the
#     single-process GPU list, so the Distributor sees both sizes 1 and 2.
#
# NOT GPU-VERIFIED: authored on a GPU-less machine — parse-checked only. The GPU
# assertions are first exercised on a CUDA node.
# ============================================================================
if CUDA.functional()
    @testset "distributed GPU DCT-I end-to-end (field-level)" begin
        # 3D RealFourier(x) × Chebyshev(y) × Chebyshev(z): RealFourier only on
        # dim 1 and no Fourier transverse axis → distributed_gpu_supported == true.
        # Real field (Float64): the distributed path truncates dim 1 to the
        # half-spectrum, which is only meaningful for a real-valued physical field.
        coords = CartesianCoordinates("x", "y", "z")
        dist   = Distributor(coords; dtype=Float64, device=_DCT1_GPU_ARCH)
        xb = RealFourier(coords["x"];    size=16, bounds=(0.0, 2π))
        yb = ChebyshevT(coords["y"];     size=8,  bounds=(-1.0, 1.0))
        zb = ChebyshevT(coords["z"];     size=12, bounds=(-1.0, 1.0))
        field  = ScalarField(dist, "u", (xb, yb, zb), Float64)
        nprocs = dist.size

        @test Tarang.distributed_gpu_supported(field) == true

        # Random real grid data (Z-pencil / local layout).
        ensure_layout!(field, :g)
        gshape = size(get_grid_data(field))
        data   = rand(Float64, gshape...)
        get_grid_data(field) .= CuArray(data)
        original = copy(Array(get_grid_data(field)))

        # Field-level round-trip: nprocs==1 uses the single-GPU device path;
        # nprocs>1 uses the live distributed GPU path.
        forward_transform!(field)
        backward_transform!(field)
        ensure_layout!(field, :g)
        @test isapprox(Array(get_grid_data(field)), original; rtol=1e-9, atol=1e-11)

        # Cross-check coeffs vs a single-process CPU reference. Only meaningful at
        # nprocs==1, where both fields are single-rank with identical coeff layouts.
        if nprocs == 1
            cdist = Distributor(coords; dtype=Float64, device=CPU())
            cxb = RealFourier(coords["x"];    size=16, bounds=(0.0, 2π))
            cyb = ChebyshevT(coords["y"];     size=8,  bounds=(-1.0, 1.0))
            czb = ChebyshevT(coords["z"];     size=12, bounds=(-1.0, 1.0))
            cpu = ScalarField(cdist, "u", (cxb, cyb, czb), Float64)
            ensure_layout!(cpu, :g)
            get_grid_data(cpu) .= data

            # Re-seed the GPU grid (overwritten by the round-trip above), forward both,
            # then compare coefficients.
            ensure_layout!(field, :g)
            get_grid_data(field) .= CuArray(data)
            forward_transform!(field); forward_transform!(cpu)
            ensure_layout!(field, :c); ensure_layout!(cpu, :c)
            @test isapprox(Array(get_coeff_data(field)), get_coeff_data(cpu);
                           rtol=1e-9, atol=1e-11)
        end

        # Direct distributed-driver smoke: exercises distributed_forward_dct! /
        # distributed_backward_dct! (the multi-GPU pipeline + transposes) even at
        # nprocs==1, but only when NCCL is available (the plan ctor requires it).
        if Tarang.nccl_available() || Tarang._try_load_nccl()
            plan   = _EXT.get_or_create_distributed_dct_plan(field)
            grid   = CUDA.rand(Float64, plan.pencil.z_pencil_shape...)
            orig   = copy(grid)
            coeffs = CUDA.zeros(ComplexF64, plan.coeff_pencil.z_pencil_shape...)
            _EXT.distributed_forward_dct!(coeffs, grid, plan)
            rec = CUDA.zeros(Float64, plan.pencil.z_pencil_shape...)
            _EXT.distributed_backward_dct!(rec, coeffs, plan)
            @test isapprox(Array(rec), Array(orig); rtol=1e-9, atol=1e-11)
        else
            @info "NCCL unavailable: direct distributed-driver smoke skipped " *
                  "(field-level round-trip still covered above)."
        end
    end

    @testset "unsupported distributed layout is rejected without CPU staging" begin
        # RealFourier(x) × ComplexFourier(y) × Chebyshev(z) needs transverse
        # conjugate partners that the distributed inverse cannot currently form.
        coords = CartesianCoordinates("x", "y", "z")
        dist   = Distributor(coords; dtype=Float64, device=_DCT1_GPU_ARCH)
        xb = RealFourier(coords["x"];    size=16, bounds=(0.0, 2π))
        yb = ComplexFourier(coords["y"]; size=8,  bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"];     size=12, bounds=(-1.0, 1.0))
        nprocs = dist.size

        bases = (xb, yb, zb)
        @test Tarang.distributed_gpu_supported(bases) == false

        # At one rank the general mixed transform is device-native and can round-trip.
        # At multiple ranks Domain validation rejects it before allocation, never stages.
        if nprocs == 1
            field = ScalarField(dist, "u", bases, Float64)
            ensure_layout!(field, :g)
            gshape = size(get_grid_data(field))
            data   = rand(Float64, gshape...)
            get_grid_data(field) .= CuArray(data)
            original = copy(Array(get_grid_data(field)))
            forward_transform!(field)
            backward_transform!(field)
            ensure_layout!(field, :g)
            @test isapprox(Array(get_grid_data(field)), original; rtol=1e-9, atol=1e-11)
        else
            @test_throws ErrorException ScalarField(dist, "u", bases, Float64)
        end
    end


    @testset "direct distributed backward transform refuses rank-local inverse" begin
        coords = CartesianCoordinates("x", "y", "z")
        dist = Distributor(coords; dtype=ComplexF64, device=_DCT1_GPU_ARCH)
        nprocs = dist.size
        if nprocs == 1
            @test_skip "requires a multi-rank GPU launch"
        else
            labels = ("x", "y", "z")
            bases = ntuple(i -> ComplexFourier(
                coords[labels[i]]; size=8, bounds=(0.0, 2π)), 3)
            field = ScalarField(dist, "u_coeff", bases, ComplexF64)
            coeff = get_coeff_data(field)
            coeff .= CUDA.rand(ComplexF64, size(coeff)...)
            field.current_layout = :c
            @test_throws ErrorException backward_transform!(field)
        end
    end
end
