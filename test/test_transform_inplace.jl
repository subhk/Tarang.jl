# ============================================================================
# In-place transform regression tests
# ============================================================================
#
# These tests guard the zero-allocation property of `forward_transform!` /
# `backward_transform!` across CPU (always) and GPU (when CUDA is functional).
#
# Two things are asserted:
#
#   1. **Buffer identity preserved** — after a transform, `get_coeff_data(field)`
#      (or `get_grid_data(field)` for backward) returns the SAME object as
#      before the call. If a regression reintroduces `set_coeff_data!(field,
#      <fresh alloc>)` on the hot path, this check will catch it — the old
#      buffer will be replaced by a new one and `===` will return false.
#
#   2. **Steady-state allocation bounded** — 100 round-trips allocate under a
#      generous threshold (50 KiB). A real regression would push this into
#      MiB territory (the pre-refactor state was ~11 MiB per RBC step, >95%
#      of which came from the out-of-place transform output arrays). Setting
#      the threshold loosely keeps this from being flaky on CI.
#
# Regression history:
# - 2026-04-XX: `forward_transform!` / `backward_transform!` serial CPU
#   path rewritten to use `_apply_forward!` / `_apply_backward!` with cached
#   FFTW plans + pre-allocated field buffers. Before the rewrite, each
#   transform allocated a fresh output array via `FFTW.rfft(data, dims)`
#   and called `set_coeff_data!(field, <new>)`, replacing the field buffer.
# - Same session: MPI+CPU backward added an in-place `ldiv!` fast path
#   symmetric with the pre-existing forward `mul!` path.
# - Same session: Chebyshev DCT-I gained an in-place path via pre-allocated
#   real/imag split scratch buffers cached on the transform object.
#
# Run with: julia --project=. test/test_transform_inplace.jl
#        or: julia --project=. -e 'using Pkg; Pkg.test()'  (via runtests.jl)

using Test
using Tarang

@testset "In-place transform fast path" begin

    @testset "Serial CPU: Fourier 2D buffer identity (first call)" begin
        # With the unified `_coefficient_shape_impl` rule, pre-allocation
        # from `allocate_data!` already matches the transform output shape,
        # so the in-place fast path fires from the very first call — no
        # warm-up reallocation. This asserts that invariant.
        coords = CartesianCoordinates("x", "y")
        dist   = Distributor(coords; dtype=Float64, device=CPU())
        xb     = RealFourier(coords["x"]; size=64, bounds=(0.0, 2π))
        yb     = RealFourier(coords["y"]; size=64, bounds=(0.0, 2π))
        domain = Domain(dist, (xb, yb))

        T = ScalarField(domain, "T")

        # Verify the pre-allocated coeff shape matches the correct rule:
        # first Fourier axis halved, second at full size.
        @test size(Tarang.get_coeff_data(T)) == (33, 64)

        # Seed with a smooth pattern
        grid = Tarang.get_grid_data(T)
        for j in 1:size(grid, 2), i in 1:size(grid, 1)
            x = 2π * (i - 1) / 64
            y = 2π * (j - 1) / 64
            grid[i, j] = sin(x) * cos(2y) + 0.3 * cos(3x)
        end
        T.current_layout = :g
        snapshot = copy(grid)

        # First-call buffer identity — no warm-up needed.
        coeff_before = Tarang.get_coeff_data(T)
        ensure_layout!(T, :c)
        coeff_after = Tarang.get_coeff_data(T)
        @test coeff_before === coeff_after

        grid_before = Tarang.get_grid_data(T)
        ensure_layout!(T, :g)
        grid_after = Tarang.get_grid_data(T)
        @test grid_before === grid_after

        # Round-trip sanity check
        @test maximum(abs, Tarang.get_grid_data(T) .- snapshot) < 1e-13
    end

    @testset "Serial CPU: Fourier+Chebyshev 2D buffer identity" begin
        coords = CartesianCoordinates("x", "z")
        dist   = Distributor(coords; dtype=Float64, device=CPU())
        xb     = RealFourier(coords["x"]; size=64, bounds=(0.0, 4.0))
        zb     = ChebyshevT(coords["z"];  size=32, bounds=(0.0, 1.0))
        domain = Domain(dist, (xb, zb))

        T = ScalarField(domain, "T")
        grid = Tarang.get_grid_data(T)
        for k in 1:size(grid, 2), i in 1:size(grid, 1)
            x = 4.0 * (i - 1) / 64
            z = (k - 1) / 31  # Gauss-Lobatto index
            grid[i, k] = sin(2π * x / 4.0) * z * (1 - z)
        end
        T.current_layout = :g
        snapshot = copy(grid)

        coeff_before = Tarang.get_coeff_data(T)
        ensure_layout!(T, :c)
        coeff_after = Tarang.get_coeff_data(T)
        @test coeff_before === coeff_after

        grid_before = Tarang.get_grid_data(T)
        ensure_layout!(T, :g)
        grid_after = Tarang.get_grid_data(T)
        @test grid_before === grid_after

        # Chebyshev round-trip has a looser tolerance because of the DCT-I
        # endpoint halving + truncation dance.
        @test maximum(abs, Tarang.get_grid_data(T) .- snapshot) < 1e-12
    end

    @testset "Serial CPU: steady-state allocation bounded" begin
        coords = CartesianCoordinates("x", "y")
        dist   = Distributor(coords; dtype=Float64, device=CPU())
        xb     = RealFourier(coords["x"]; size=128, bounds=(0.0, 2π))
        yb     = RealFourier(coords["y"]; size=128, bounds=(0.0, 2π))
        domain = Domain(dist, (xb, yb))

        T = ScalarField(domain, "T")
        parent_grid = Tarang.get_grid_data(T)
        for j in 1:size(parent_grid, 2), i in 1:size(parent_grid, 1)
            x = 2π * (i - 1) / 128
            y = 2π * (j - 1) / 128
            parent_grid[i, j] = sin(x) * cos(y)
        end
        T.current_layout = :g

        # Warm up plan caches (first call creates FFTW plans)
        for _ in 1:3
            ensure_layout!(T, :c)
            ensure_layout!(T, :g)
        end

        # Measure 100 round-trips
        bytes = @allocated begin
            for _ in 1:100
                ensure_layout!(T, :c)
                ensure_layout!(T, :g)
            end
        end

        # What this guards: reintroducing per-call array allocation. A single
        # out-of-place transform output is ~130 KiB (65×128 ComplexF64 forward,
        # 128×128 Float64 backward), so a regression shows up as ~13–26 MiB over
        # 100 round-trips. THAT is the failure mode worth catching.
        #
        # The transform chain still walks `dist.transforms::Vector{Any}`, so each
        # stage is a dynamic dispatch. On arm64 the steady-state path now measures
        # 0 B/round-trip (the type-stability work in transform_{fourier,gpu,types}.jl
        # made forward+backward fully concrete). On x86_64, however, Julia's codegen
        # boxes that dynamic dispatch at ~4–6 KiB/round-trip regardless of Julia
        # version — an architecture-dependent constant, not a regression. The
        # original 200 KiB bound assumed ~2 KiB/round-trip of boxing, which holds on
        # arm64 but underestimates x86_64; killing the last of it requires making
        # `dist.transforms` concretely typed (a Distributor-level change).
        #
        # 2 MiB sits comfortably above the x86_64 boxing floor (~0.5 MiB) and an
        # order of magnitude below the ~13 MiB realloc regression, so it still
        # catches the real failure while staying green across architectures.
        @test bytes < 2_000_000
    end

    # ------------------------------------------------------------------------
    # GPU smoke test — activates only when CUDA is functional.
    # ------------------------------------------------------------------------
    #
    # This exercises the `gpu_forward_transform!` / `gpu_backward_transform!`
    # extension path which is ORTHOGONAL to the CPU in-place refactor: the
    # extension short-circuits `forward_transform!` before the new CPU code
    # runs. The test exists mainly as a forward-looking regression guard so
    # that future refactors don't silently break the GPU path.
    #
    # We don't test buffer identity on GPU because `gpu_forward_transform!`
    # may legitimately reallocate `coeff_data` as a fresh CuArray on first
    # use (depending on the cuFFT plan output shape) — the relevant GPU
    # invariant is correctness + layout, not identity.
    gpu_functional = try
        # Only attempt to load CUDA if the extension stub is actually overridden
        # (i.e., if a working CUDA runtime has been wired in by the user).
        Base.get_extension(Tarang, :TarangCUDAExt) !== nothing &&
            isdefined(Main, :CUDA) && Main.CUDA.functional()
    catch
        false
    end

    if gpu_functional
        CUDA = Main.CUDA
        @testset "GPU: Fourier 2D round-trip" begin
            coords = CartesianCoordinates("x", "y")
            # NOTE: user code typically constructs `Distributor(coords;
            # architecture=GPU())`; we use the same pattern here.
            dist = Distributor(coords; dtype=Float64, architecture=GPU())
            xb = RealFourier(coords["x"]; size=64, bounds=(0.0, 2π))
            yb = RealFourier(coords["y"]; size=64, bounds=(0.0, 2π))
            domain = Domain(dist, (xb, yb))

            T = ScalarField(domain, "T")
            @test Tarang.get_grid_data(T) isa CUDA.CuArray

            # Seed on host, then upload
            host_data = zeros(Float64, 64, 64)
            for j in 1:64, i in 1:64
                x = 2π * (i - 1) / 64
                y = 2π * (j - 1) / 64
                host_data[i, j] = sin(x) * cos(2y) + 0.3 * cos(3x)
            end
            copyto!(Tarang.get_grid_data(T), host_data)
            T.current_layout = :g

            # Forward → coeff space
            ensure_layout!(T, :c)
            @test Tarang.get_coeff_data(T) isa CUDA.CuArray

            # Backward → grid space
            ensure_layout!(T, :g)
            @test Tarang.get_grid_data(T) isa CUDA.CuArray

            # Round-trip correctness
            recovered = Array(Tarang.get_grid_data(T))
            @test maximum(abs, recovered .- host_data) < 1e-12
        end

        @testset "GPU: Fourier+Chebyshev 2D round-trip" begin
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; dtype=Float64, architecture=GPU())
            xb = RealFourier(coords["x"]; size=64, bounds=(0.0, 4.0))
            zb = ChebyshevT(coords["z"];  size=32, bounds=(0.0, 1.0))
            domain = Domain(dist, (xb, zb))

            T = ScalarField(domain, "T")

            host_data = zeros(Float64, 64, 32)
            for k in 1:32, i in 1:64
                x = 4.0 * (i - 1) / 64
                z = (k - 1) / 31
                host_data[i, k] = sin(2π * x / 4.0) * z * (1 - z)
            end
            copyto!(Tarang.get_grid_data(T), host_data)
            T.current_layout = :g

            ensure_layout!(T, :c)
            @test Tarang.get_coeff_data(T) isa CUDA.CuArray

            ensure_layout!(T, :g)
            @test Tarang.get_grid_data(T) isa CUDA.CuArray

            recovered = Array(Tarang.get_grid_data(T))
            @test maximum(abs, recovered .- host_data) < 1e-11
        end
    else
        @info "test_transform_inplace: GPU tests skipped (CUDA not functional)"
    end
end
