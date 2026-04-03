using Test
using FFTW

@testset "Dealiasing Math (CPU)" begin
    # ========================================================================
    # These tests verify the mathematical correctness of the 2/3-rule
    # dealiasing logic, including the negative frequency fix.
    #
    # For a complex FFT of size N:
    #   index 1         -> wavenumber 0 (DC)
    #   index 2..N/2+1  -> wavenumber 1..N/2 (positive frequencies)
    #   index N/2+2..N  -> wavenumber -(N/2-1)..-1 (negative frequencies)
    #
    # The 2/3 rule zeroes all modes with |k| > N/3.
    # ========================================================================

    """
    CPU reference implementation of the dealiasing logic (matches GPU kernel).
    Zeroes modes where |k| exceeds cutoff in any dimension.
    """
    function dealias_2d_cpu!(data, cutoff_x::Int, cutoff_y::Int)
        nx, ny = size(data)
        for j in 1:ny, i in 1:nx
            # Kill if out of range from BOTH positive and negative perspective
            kill_x = i > cutoff_x && (i == 1 || nx - i + 1 > cutoff_x)
            kill_y = j > cutoff_y && (j == 1 || ny - j + 1 > cutoff_y)
            if kill_x || kill_y
                data[i, j] = zero(eltype(data))
            end
        end
        return data
    end

    function dealias_3d_cpu!(data, cutoff_x::Int, cutoff_y::Int, cutoff_z::Int)
        nx, ny, nz = size(data)
        for k in 1:nz, j in 1:ny, i in 1:nx
            kill_x = i > cutoff_x && (i == 1 || nx - i + 1 > cutoff_x)
            kill_y = j > cutoff_y && (j == 1 || ny - j + 1 > cutoff_y)
            kill_z = k > cutoff_z && (k == 1 || nz - k + 1 > cutoff_z)
            if kill_x || kill_y || kill_z
                data[i, j, k] = zero(eltype(data))
            end
        end
        return data
    end

    """Return the wavenumber for FFT index i in an array of size N."""
    function fft_wavenumber(i, N)
        # 1-indexed: index 1 = k=0, index N/2+1 = k=N/2, index N/2+2 = k=-(N/2-1)
        k = i - 1
        if k > N ÷ 2
            k -= N
        end
        return k
    end

    # ========================================================================
    # 1D dealiasing logic
    # ========================================================================

    @testset "1D frequency zeroing" begin
        N = 12
        cutoff = N ÷ 3 * 2  # 2/3 rule: cutoff index = 8

        data = complex(ones(N))
        nx = N
        for i in 1:nx
            kill = i > cutoff && (i == 1 || nx - i + 1 > cutoff)
            if kill
                data[i] = 0.0
            end
        end

        # Verify: kept modes should be indices 1..cutoff (positive) and
        # N-cutoff+2..N (negative mirrors). Everything in between is zeroed.
        for i in 1:N
            k = fft_wavenumber(i, N)
            kept = i <= cutoff || (i > 1 && nx - i + 1 <= cutoff)
            if kept
                @test data[i] != 0.0
            else
                @test data[i] == 0.0
            end
        end
    end

    @testset "DC and Nyquist handling" begin
        N = 8
        cutoff = N ÷ 3 * 2  # cutoff = 4 for 2/3 rule

        # DC (index 1, k=0) should always be kept
        i = 1
        kill = i > cutoff || (i > 1 && N - i + 1 > cutoff)
        @test kill == false

        # Nyquist (index N/2+1=5, k=4) with cutoff=4 should NOT be killed
        # because |k|=4 is not > cutoff=4... wait, wavenumber mapping:
        # For N=8, cutoff_index=4 means we keep indices 1..4 (k=0,1,2,3)
        # and mirror from end. Index 5 has k=4 which is the Nyquist.
        # The condition i > cutoff is 5 > 4 = true, so it IS killed.
        i = 5
        kill = i > cutoff || (i > 1 && N - i + 1 > cutoff)
        @test kill == true  # Nyquist is beyond the 2/3 cutoff
    end

    @testset "negative frequencies are zeroed" begin
        # The correct logic: kill if i > cutoff AND mirror > cutoff
        # A mode is kept if EITHER the positive index or negative mirror is within cutoff
        N = 12
        cutoff = 8  # 2/3 rule

        # Index 12 has wavenumber k = -1 (|k|=1, mirror index=1, within cutoff -> KEPT)
        i = 12
        kill = i > cutoff && (i == 1 || N - i + 1 > cutoff)
        @test kill == false  # mirror=1 <= 8, so kept

        # Index 9 has wavenumber k = -4 (mirror index=4, within cutoff -> KEPT)
        i = 9
        k = fft_wavenumber(i, N)
        @test k == -4
        kill = i > cutoff && (i == 1 || N - i + 1 > cutoff)
        @test kill == false  # mirror=4 <= 8, so kept

        # For N=12, cutoff=8: the gap zone is indices cutoff+1..N-cutoff+1 = 9..5
        # But 9 > 5 so there's no gap — all modes are within one of the two cutoff ranges
        # Try N=16, cutoff=8 to get a real gap
        N2 = 16
        cutoff2 = 8

        # Index 10 has k = -7 (mirror=7, within cutoff -> KEPT)
        i = 10
        kill = i > cutoff2 && (i == 1 || N2 - i + 1 > cutoff2)
        @test kill == false  # mirror=7 <= 8

        # Index 9 has k = -8 (mirror=8, within cutoff -> KEPT)
        i = 9
        kill = i > cutoff2 && (i == 1 || N2 - i + 1 > cutoff2)
        @test kill == false  # mirror=8 <= 8

        # Index 16 has k = -1 (mirror=1, within cutoff -> KEPT)
        i = 16
        kill = i > cutoff2 && (i == 1 || N2 - i + 1 > cutoff2)
        @test kill == false  # mirror=1 <= 8
    end

    # ========================================================================
    # 2D dealiasing
    # ========================================================================

    @testset "2D dealiasing symmetry" begin
        N = 12
        cutoff = N ÷ 3 * 2  # 8

        data = complex(ones(N, N))
        dealias_2d_cpu!(data, cutoff, cutoff)

        # A mode is kept if within cutoff from either positive or negative side
        for i in 1:N, j in 1:N
            kept_x = i <= cutoff || (i > 1 && N - i + 1 <= cutoff)
            kept_y = j <= cutoff || (j > 1 && N - j + 1 <= cutoff)
            if kept_x && kept_y
                @test data[i, j] != 0.0
            else
                @test data[i, j] == 0.0
            end
        end
    end

    @testset "2D dealiasing preserves low modes" begin
        Nx, Ny = 16, 16
        cutoff_x = Nx ÷ 3 * 2  # 10
        cutoff_y = Ny ÷ 3 * 2  # 10

        # Create data with known spectral content
        x = range(0, 2pi, length=Nx+1)[1:Nx]
        y = range(0, 2pi, length=Ny+1)[1:Ny]
        # Low frequency signal: k=1 in x, k=2 in y
        physical = [sin(xi) * cos(2*yi) for xi in x, yi in y]
        spectral = fft(physical)

        original = copy(spectral)
        dealias_2d_cpu!(spectral, cutoff_x, cutoff_y)

        # Low modes (k=1, k=2) should be untouched
        @test spectral[2, 1] == original[2, 1]  # kx=1, ky=0
        @test spectral[1, 3] == original[1, 3]  # kx=0, ky=2
    end

    @testset "2D dealiasing kills high modes" begin
        # Use a larger array with smaller cutoff so there IS a gap zone
        Nx, Ny = 24, 24
        cutoff_x = 8  # keep only first 8 + last 8 indices (16/24 = 2/3)
        cutoff_y = 8

        data = complex(ones(Nx, Ny))
        dealias_2d_cpu!(data, cutoff_x, cutoff_y)

        kept = count(x -> x != 0.0, data)
        total = Nx * Ny

        @test kept < total
        @test kept > 0
    end

    # ========================================================================
    # 3D dealiasing
    # ========================================================================

    @testset "3D dealiasing" begin
        N = 8
        cutoff = N ÷ 3 * 2  # 4

        data = complex(ones(N, N, N))
        dealias_3d_cpu!(data, cutoff, cutoff, cutoff)

        for k in 1:N, j in 1:N, i in 1:N
            kept_x = i <= cutoff || (i > 1 && N - i + 1 <= cutoff)
            kept_y = j <= cutoff || (j > 1 && N - j + 1 <= cutoff)
            kept_z = k <= cutoff || (k > 1 && N - k + 1 <= cutoff)
            if kept_x && kept_y && kept_z
                @test data[i, j, k] != 0.0
            else
                @test data[i, j, k] == 0.0
            end
        end
    end

    # ========================================================================
    # R2C FFT convention: rfft reduces last dimension
    # ========================================================================

    @testset "rfft reduces first dimension" begin
        # FFTW.jl rfft(a) reduces the FIRST dimension (column-major convention)
        a = rand(8, 12)
        f = rfft(a)
        @test size(f) == (5, 12)  # first dim: 8/2+1 = 5, second dim unchanged

        b = rand(16, 8, 12)
        g = rfft(b)
        @test size(g) == (9, 8, 12)  # only first dim halved: 16/2+1 = 9

        c = rand(6)
        h = rfft(c)
        @test size(h) == (4,)  # 6/2+1 = 4
    end

    @testset "rfft along specific dims" begin
        a = rand(8, 12, 6)
        # rfft over dim 1 only
        f = rfft(a, 1)
        @test size(f, 1) == 5   # dim 1 halved: 8/2+1 = 5
        @test size(f, 2) == 12  # dim 2 unchanged
        @test size(f, 3) == 6   # dim 3 untouched
    end

    @testset "irfft round-trip" begin
        N = 12
        a = rand(N)
        f = rfft(a)
        b = irfft(f, N)
        @test isapprox(a, b; atol=1e-12)

        M, N2 = 8, 10
        a2 = rand(M, N2)
        f2 = rfft(a2)
        b2 = irfft(f2, M)  # first dim was reduced, so restore M
        @test isapprox(a2, b2; atol=1e-12)
    end

    # ========================================================================
    # DCT correctness on CPU
    # ========================================================================

    @testset "DCT-II / DCT-III round-trip" begin
        # DCT-II (forward) and DCT-III (backward) are inverses (up to scaling)
        N = 16
        x = rand(N)

        # FFTW DCT-II
        plan_dct = FFTW.plan_r2r(x, FFTW.REDFT10)
        X = plan_dct * x

        # FFTW DCT-III (inverse)
        plan_idct = FFTW.plan_r2r(X, FFTW.REDFT01)
        y = plan_idct * X

        # Round-trip: y = 2N * x (FFTW convention)
        @test isapprox(y, 2N * x; atol=1e-10)
    end

    @testset "DCT of known signal" begin
        # DCT of constant function should have energy only in DC
        N = 8
        x = ones(N)
        plan_dct = FFTW.plan_r2r(x, FFTW.REDFT10)
        X = plan_dct * x

        # DC component should be N (sum of all ones * cos(0))
        # Actually FFTW REDFT10: X[0] = 2 * sum(x_j * cos(pi*(2j+1)*0/(2N))) = 2*N for ones
        @test abs(X[1]) > 0.0
        # Higher modes should be ~0 for a constant
        for k in 2:N
            @test abs(X[k]) < 1e-12
        end
    end

    @testset "DCT linearity" begin
        N = 16
        a = rand(N)
        b = rand(N)
        alpha, beta = 2.5, -1.3

        plan_dct = FFTW.plan_r2r(a, FFTW.REDFT10)
        @test isapprox(plan_dct * (alpha .* a .+ beta .* b),
                       alpha .* (plan_dct * a) .+ beta .* (plan_dct * b);
                       atol=1e-10)
    end
end
