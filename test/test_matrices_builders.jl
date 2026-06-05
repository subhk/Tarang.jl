# Tests for src/core/operators/matrices/matrices_builders.jl
#
# Covers the low-level operator-matrix builders:
#   - fourier_differentiation_matrix(::ComplexFourier, order)
#   - fourier_differentiation_matrix(::RealFourier, order)
#   - build_operator_differentiation_matrix(var, coord, order)
#   - build_lift_matrix(var, basis, n)
#   - get_coord_for_basis(basis)
#   - field_dofs(field)
#
# All functions are internal to Tarang, so they are called fully qualified
# as Tarang.foo(...) to avoid Main namespace collisions in the shared suite.

using Test
using Tarang
using LinearAlgebra
using SparseArrays

@testset "matrices_builders.jl" begin

    # ------------------------------------------------------------------
    # ComplexFourier differentiation matrix
    #
    # The function builds a diagonal matrix with entries (i*k)^order where
    # k = k0 * k_native, k0 = 2pi/L, and the native wavenumbers are
    #   k_native[j] = (j-1 <= N/2) ? (j-1) : (j-1-N)   for j = 1..N.
    # ------------------------------------------------------------------
    @testset "fourier_differentiation_matrix ComplexFourier" begin
        coords = CartesianCoordinates("x")
        N = 8
        L = 2π
        xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, L))
        k0 = 2π / L

        # Expected native wavenumber ordering used by the builder.
        k_native = [j <= N ÷ 2 ? j : j - N for j in 0:N-1]
        k_phys = k0 .* k_native

        @testset "order 1 is diagonal i*k" begin
            D = Tarang.fourier_differentiation_matrix(xb, 1)
            @test size(D) == (N, N)
            Df = Matrix(D)
            expected = Diagonal(im .* k_phys)
            @test Df ≈ Matrix(expected)
            # Off-diagonals must be exactly zero (pure diagonal operator).
            for i in 1:N, j in 1:N
                if i != j
                    @test Df[i, j] == 0
                end
            end
        end

        @testset "order 2 is diagonal (i*k)^2 = -k^2" begin
            D2 = Tarang.fourier_differentiation_matrix(xb, 2)
            D2f = Matrix(D2)
            expected2 = Diagonal((im .* k_phys) .^ 2)
            @test D2f ≈ Matrix(expected2)
            # (i k)^2 is real and non-positive.
            @test all(imag.(diag(D2f)) .== 0)
            @test all(real.(diag(D2f)) .<= 1e-12)
        end

        @testset "D^2 == D*D (operator composition)" begin
            D = Matrix(Tarang.fourier_differentiation_matrix(xb, 1))
            D2 = Matrix(Tarang.fourier_differentiation_matrix(xb, 2))
            @test D2 ≈ D * D
        end

        @testset "oracle: differentiate exp(i k x) coefficients" begin
            # In the ComplexFourier coeff convention here, a single Fourier
            # mode m corresponds to the coefficient vector e_{m+1} (native
            # index). Differentiating must scale that coefficient by i*k_m.
            D = Matrix(Tarang.fourier_differentiation_matrix(xb, 1))
            for native_idx in 0:N-1
                coeff = zeros(ComplexF64, N)
                coeff[native_idx + 1] = 1.0
                out = D * coeff
                km = k_phys[native_idx + 1]
                expected = zeros(ComplexF64, N)
                expected[native_idx + 1] = im * km
                @test out ≈ expected
            end
        end

        @testset "non-2pi bounds scale k0" begin
            Lb = 4π
            xbL = ComplexFourier(coords["x"]; size=N, bounds=(0.0, Lb))
            DL = Matrix(Tarang.fourier_differentiation_matrix(xbL, 1))
            k0L = 2π / Lb
            k_physL = k0L .* k_native
            @test DL ≈ Matrix(Diagonal(im .* k_physL))
        end
    end

    # ------------------------------------------------------------------
    # RealFourier differentiation matrix
    #
    # Real spectral layout (1-indexed):
    #   mode 1     : cos(0 x)              (DC, derivative 0)
    #   mode 2k    : cos(k x)
    #   mode 2k+1  : -sin(k x)  (msin convention)
    #
    # The 2x2 derivative block for the (cos, msin) pair at wavenumber k is
    #   D^1 = [ 0  -k ;  k  0 ] * (k_phys factors via phase rotation).
    # Specifically the builder uses, per order n,
    #   block = k_phys^n * [ cos(phase)  -sin(phase) ;
    #                        sin(phase)   cos(phase) ],  phase = n*pi/2.
    # ------------------------------------------------------------------
    @testset "fourier_differentiation_matrix RealFourier" begin
        coords = CartesianCoordinates("x")
        N = 9          # odd so k_max = (N-1)/2 = 4, all cos/sin pairs present
        L = 2π
        xb = RealFourier(coords["x"]; size=N, bounds=(0.0, L))
        k0 = 2π / L
        k_max = (N - 1) ÷ 2

        @testset "order 1 structure" begin
            D = Tarang.fourier_differentiation_matrix(xb, 1)
            @test size(D) == (N, N)
            Df = Matrix(D)
            # DC mode column/row are zero.
            @test all(Df[1, :] .== 0)
            @test all(Df[:, 1] .== 0)
            for k in 1:k_max
                c_idx = 2k
                s_idx = 2k + 1
                kp = k0 * k
                # order=1: phase = pi/2 -> c=0, s=1.
                # cos_out = -s*k*msin_in => entry (cos, msin) = -k
                # msin_out = s*k*cos_in => entry (msin, cos) = +k
                @test Df[c_idx, s_idx] ≈ -kp
                @test Df[s_idx, c_idx] ≈ kp
                # diagonal of block is zero at order 1 (c = cos(pi/2) = 0)
                @test abs(Df[c_idx, c_idx]) < 1e-12
                @test abs(Df[s_idx, s_idx]) < 1e-12
            end
        end

        @testset "oracle: derivative of cos(k x) and -sin(k x)" begin
            # d/dx cos(kx) = -k sin(kx) = k * (-sin(kx)) = k * msin  -> coeff k in msin slot
            # d/dx (-sin(kx)) = -k cos(kx)                            -> coeff -k in cos slot
            D = Matrix(Tarang.fourier_differentiation_matrix(xb, 1))
            for k in 1:k_max
                c_idx = 2k
                s_idx = 2k + 1
                kp = k0 * k

                # input: pure cos(kx) coefficient
                cin = zeros(Float64, N); cin[c_idx] = 1.0
                cout = D * cin
                expected = zeros(Float64, N); expected[s_idx] = kp   # k * msin
                @test cout ≈ expected

                # input: pure msin = -sin(kx) coefficient
                sin_in = zeros(Float64, N); sin_in[s_idx] = 1.0
                sout = D * sin_in
                expected_s = zeros(Float64, N); expected_s[c_idx] = -kp  # -k * cos
                @test sout ≈ expected_s
            end
        end

        @testset "order 2 == D*D and diagonal -k^2 blocks" begin
            D = Matrix(Tarang.fourier_differentiation_matrix(xb, 1))
            D2 = Matrix(Tarang.fourier_differentiation_matrix(xb, 2))
            @test D2 ≈ D * D
            for k in 1:k_max
                c_idx = 2k
                s_idx = 2k + 1
                kp = k0 * k
                # order=2: phase=pi -> c=-1, s=0 => block = -k^2 * I_2
                @test D2[c_idx, c_idx] ≈ -kp^2
                @test D2[s_idx, s_idx] ≈ -kp^2
                @test abs(D2[c_idx, s_idx]) < 1e-12
                @test abs(D2[s_idx, c_idx]) < 1e-12
            end
        end
    end

    # ------------------------------------------------------------------
    # build_operator_differentiation_matrix
    # ------------------------------------------------------------------
    @testset "build_operator_differentiation_matrix" begin
        @testset "1D RealFourier returns the 1D matrix" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            N = 9
            xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            coord = coords["x"]

            D = Tarang.build_operator_differentiation_matrix(u, coord, 1)
            D1d = Tarang.fourier_differentiation_matrix(xb, 1)
            @test D !== nothing
            @test size(D) == (N, N)
            @test Matrix(D) ≈ Matrix(D1d)
        end

        @testset "1D ComplexFourier returns the 1D matrix" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=ComplexF64)
            N = 8
            xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), ComplexF64)
            coord = coords["x"]

            D = Tarang.build_operator_differentiation_matrix(u, coord, 1)
            D1d = Tarang.fourier_differentiation_matrix(xb, 1)
            @test Matrix(D) ≈ Matrix(D1d)
        end

        @testset "2D Kronecker structure (derivative in x, identity in z)" begin
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
            Nx, Nz = 5, 4
            xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
            zb = RealFourier(coords["z"]; size=Nz, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb, zb), Float64)

            Dx = Tarang.build_operator_differentiation_matrix(u, coords["x"], 1)
            @test Dx !== nothing
            @test size(Dx) == (Nx * Nz, Nx * Nz)

            # Builder convention: result = kron over reversed basis order so that
            # axis 1 (x) varies fastest. For derivative along x with identity in z,
            # that means kron(I_z, Dx_1d) (axis-1-fastest column-major layout).
            Dx1d = Matrix(Tarang.fourier_differentiation_matrix(xb, 1))
            Iz = Matrix(I, Nz, Nz)
            expected = kron(Iz, Dx1d)
            @test Matrix(Dx) ≈ expected

            # Derivative along z: identity in x.
            Dz = Tarang.build_operator_differentiation_matrix(u, coords["z"], 1)
            Dz1d = Matrix(Tarang.fourier_differentiation_matrix(zb, 1))
            Ix = Matrix(I, Nx, Nx)
            expected_z = kron(Dz1d, Ix)
            @test Matrix(Dz) ≈ expected_z
        end

        @testset "unmatched coordinate returns nothing" begin
            coords = CartesianCoordinates("x")
            other = CartesianCoordinates("y")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            # coord whose name ("y") does not match any basis label
            @test Tarang.build_operator_differentiation_matrix(u, other["y"], 1) === nothing
        end

        @testset "var without :bases field returns nothing" begin
            coords = CartesianCoordinates("x")
            @test Tarang.build_operator_differentiation_matrix(42, coords["x"], 1) === nothing
        end
    end

    # ------------------------------------------------------------------
    # build_lift_matrix
    # ------------------------------------------------------------------
    @testset "build_lift_matrix" begin
        @testset "1D lift selects the requested mode" begin
            coords = CartesianCoordinates("z")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            N = 10
            zb = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
            # var has NO basis matching the lift coordinate (empty bases)
            empty_field = ScalarField(dist, "tau", (), Float64)

            # n = 0 -> mode 1 (Julia 1-indexed)
            L0 = Tarang.build_lift_matrix(empty_field, zb, 0)
            @test size(L0) == (N, 1)
            @test Matrix(L0)[1, 1] == 1.0
            @test sum(Matrix(L0)) == 1.0

            # n = -1 -> last mode N
            Lm1 = Tarang.build_lift_matrix(empty_field, zb, -1)
            @test Matrix(Lm1)[N, 1] == 1.0
            @test sum(Matrix(Lm1)) == 1.0

            # n = -2 -> second-to-last mode N-1
            Lm2 = Tarang.build_lift_matrix(empty_field, zb, -2)
            @test Matrix(Lm2)[N - 1, 1] == 1.0

            # n = 3 -> mode 4
            L3 = Tarang.build_lift_matrix(empty_field, zb, 3)
            @test Matrix(L3)[4, 1] == 1.0
        end

        @testset "out-of-range mode warns and returns zeros of right shape" begin
            coords = CartesianCoordinates("z")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            N = 6
            zb = ChebyshevT(coords["z"]; size=N, bounds=(-1.0, 1.0))
            empty_field = ScalarField(dist, "tau", (), Float64)
            # n = N (=6) resolves to lift_mode = 7 > N -> out of range
            local L
            @test_logs (:warn,) (L = Tarang.build_lift_matrix(empty_field, zb, N))
            tau_dofs = max(1, Tarang.field_dofs(empty_field))
            @test size(L) == (N * tau_dofs, tau_dofs)
            @test iszero(Matrix(L))
        end

        @testset "tau method: lift basis tangential to a Fourier x-direction" begin
            # var lives on the Fourier x-basis only; lift basis is the
            # Chebyshev z-direction (not present in var). Standard tau case:
            # result = kron over coordinate order placing e_lift in z slot.
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
            Nx, Nz = 4, 8
            xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
            zb = ChebyshevT(coords["z"]; size=Nz, bounds=(-1.0, 1.0))
            # tau variable only has the x basis
            tau = ScalarField(dist, "tau", (xb,), Float64)

            Llift = Tarang.build_lift_matrix(tau, zb, -1)  # last z mode
            @test Llift !== nothing
            Lm = Matrix(Llift)
            # The matrix has Nx * Nz rows (full field) and Nx columns (tau dofs).
            @test size(Lm) == (Nx * Nz, Nx)
            # Each column has exactly one nonzero (= 1.0): the selected z mode.
            @test count(!iszero, Lm) == Nx
            @test all(Lm[Lm .!= 0] .== 1.0)
        end
    end

    # ------------------------------------------------------------------
    # get_coord_for_basis
    # ------------------------------------------------------------------
    @testset "get_coord_for_basis" begin
        coords = CartesianCoordinates("x", "z")
        xb = RealFourier(coords["x"]; size=8, bounds=(0.0, 2π))
        zb = ChebyshevT(coords["z"]; size=8, bounds=(-1.0, 1.0))

        cx = Tarang.get_coord_for_basis(xb)
        cz = Tarang.get_coord_for_basis(zb)
        @test cx !== nothing
        @test cz !== nothing
        @test cx.name == "x"
        @test cz.name == "z"
        # Returned coord is the one from the basis's own coordinate system.
        @test cx === coords["x"]
        @test cz === coords["z"]
    end

    # ------------------------------------------------------------------
    # field_dofs
    # ------------------------------------------------------------------
    @testset "field_dofs" begin
        # ScalarField exposes a :buffers property (backward-compat alias for
        # :storage), so field_dofs hits the coeff-data branch and returns the
        # length of the spectral coefficient storage (the coeff DOFs), which
        # equals the length returned by get_coeff_data.
        @testset "1D scalar field == coeff-data length (RealFourier half-spectrum)" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            N = 12
            xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), Float64)
            dofs = Tarang.field_dofs(u)
            @test dofs == length(Tarang.get_coeff_data(u))
            # RealFourier stores an unnormalized complex half-spectrum: N÷2 + 1.
            @test dofs == N ÷ 2 + 1
        end

        @testset "1D ComplexFourier coeff-data length == N" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=ComplexF64)
            N = 8
            xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, 2π))
            u = ScalarField(dist, "u", (xb,), ComplexF64)
            dofs = Tarang.field_dofs(u)
            @test dofs == length(Tarang.get_coeff_data(u))
            @test dofs == N
        end

        @testset "2D scalar field == coeff-data length" begin
            coords = CartesianCoordinates("x", "z")
            dist = Distributor(coords; mesh=(1, 1), dtype=Float64)
            Nx, Nz = 6, 5
            xb = RealFourier(coords["x"]; size=Nx, bounds=(0.0, 2π))
            zb = ChebyshevT(coords["z"]; size=Nz, bounds=(-1.0, 1.0))
            u = ScalarField(dist, "u", (xb, zb), Float64)
            dofs = Tarang.field_dofs(u)
            @test dofs == length(Tarang.get_coeff_data(u))
            # Half-spectrum along the Fourier x-axis, full Chebyshev z-axis.
            @test dofs == (Nx ÷ 2 + 1) * Nz
        end

        @testset "empty-bases field returns 1 (product over zero bases)" begin
            coords = CartesianCoordinates("x")
            dist = Distributor(coords; mesh=(1,), dtype=Float64)
            ue = ScalarField(dist, "tau", (), Float64)
            @test Tarang.field_dofs(ue) == 1
        end

        @testset "object without recognized fields returns 0" begin
            @test Tarang.field_dofs(3.14) == 0
        end
    end

end
