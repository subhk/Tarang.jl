# Guard: distributed stochastic-forcing work/power diagnostics match serial (np>=2).
#
# work_stratonovich / work_ito / instantaneous_power pair the cached forcing
# spectrum against the solution and reduce. Under MPI the solution is a
# PencilArray: its CARTESIAN getindex pa[I] is LOGICAL, but its LINEAR index is
# parent/storage order. The diagnostics now pair logical Cartesian indices with
# the cached forcing's global indices, and store_prevsol! copies
# the slab with an explicit dest[I]=sol[I] loop (a CartesianIndices comprehension
# collects in linear/storage order, transposing a permuted pencil). Round-5 audit.
using Test
using MPI
MPI.Initialized() || MPI.Init()
using Tarang
using PencilArrays

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
if nprocs < 2
    rank == 0 && @warn "MPI forcing-work test needs >= 2 ranks; got $nprocs"
    MPI.Finalize(); exit(0)
end

const N = 5
# Deterministic global arrays (identical across rank counts).
const G    = ComplexF64[ (0.3*i - 0.2*j) + (0.1*i*j)im for i in 1:N, j in 1:N]
const solG = ComplexF64[ (0.5*i + 0.7*j) - (0.05*i - 0.03*j)im for i in 1:N, j in 1:N]
const prvG = ComplexF64[ (0.2*j - 0.4*i) + (0.06*i + 0.02*j)im for i in 1:N, j in 1:N]
# Serial reference (computed at np=1 from the same code path).
const W_STRAT_REF = 0.001596
const W_ITO_REF   = 0.099056
const P_REF       = 0.04136

@testset "Distributed forcing work/power == serial (np=$nprocs)" begin
    coords = CartesianCoordinates("x", "y")
    dist = Distributor(coords)
    xb = ComplexFourier(coords["x"]; size=N, bounds=(0.0, 2π))
    yb = ComplexFourier(coords["y"]; size=N, bounds=(0.0, 2π))
    u = ScalarField(dist, "u", (xb, yb), ComplexF64); ensure_layout!(u, :c)
    setc!(Gg) = (cd = get_coeff_data(u); gv = PencilArrays.global_view(cd);
                 for I in CartesianIndices(gv); gv[I] = Gg[I]; end)

    f = Tarang.StochasticForcing(; field_size=(N, N), k_forcing=2.0, dk_forcing=1.0,
                                 dt=0.1, energy_injection_rate=1.0)
    f.cached_forcing = copy(G)
    f.domain_size = (2π, 2π)

    setc!(prvG); wi = Tarang.work_ito(f, get_coeff_data(u))
    setc!(prvG); Tarang.store_prevsol!(f, get_coeff_data(u))
    setc!(solG)
    ws = Tarang.work_stratonovich(f, get_coeff_data(u))
    p  = Tarang.instantaneous_power(f, get_coeff_data(u))

    if rank == 0
        @test isapprox(ws, W_STRAT_REF; atol=1e-7)
        @test isapprox(wi, W_ITO_REF;   atol=1e-7)
        @test isapprox(p,  P_REF;       atol=1e-7)
    end
end

@testset "Distributed half-spectrum vorticity diagnostics (np=$nprocs)" begin
    n = 8
    half_shape = (n ÷ 2 + 1, n)
    dt = 0.05
    coords_half = CartesianCoordinates("x", "y")
    dist_half = Distributor(coords_half)
    xb = RealFourier(coords_half["x"]; size=n, bounds=(0.0, 2π))
    yb = RealFourier(coords_half["y"]; size=n, bounds=(0.0, 2π))
    u = ScalarField(dist_half, "u_half", (xb, yb)); ensure_layout!(u, :c)

    forcing_full = ComplexF64[
        (0.11i - 0.04j) + (0.02i * j)im for i in 1:n, j in 1:n
    ]
    sol_prev = ComplexF64[
        (0.09i + 0.03j) - (0.015i * j)im
        for i in 1:half_shape[1], j in 1:half_shape[2]
    ]
    forcing_half = @view forcing_full[1:half_shape[1], :]
    sol_next = sol_prev .+ dt .* forcing_half

    set_half!(global_data) = begin
        coeffs = get_coeff_data(u)
        global_coeffs = PencilArrays.global_view(coeffs)
        for I in CartesianIndices(global_coeffs)
            global_coeffs[I] = global_data[I]
        end
    end

    forcing = StochasticForcing(
        field_size=(n, n), energy_injection_rate=0.2,
        injection_metric=:vorticity_kinetic, k_forcing=2.0,
        dk_forcing=0.1, spectrum_type=:band, dt=dt,
    )
    forcing.cached_forcing = copy(forcing_full)

    multiplicity = reshape([1.0, 2.0, 2.0, 2.0, 1.0], :, 1)
    kx = Float64[0, 1, 2, 3, 4]
    ky = Float64[0, 1, 2, 3, 4, -3, -2, -1]
    metric_weight = [
        iszero(x^2 + y^2) ? 0.0 : inv(x^2 + y^2) for x in kx, y in ky
    ]
    weights = multiplicity .* metric_weight ./ n^4
    pairing(a) = sum(weights .* real.(a .* conj.(forcing_half)))
    ws_ref = pairing((sol_prev .+ sol_next) ./ 2) * dt
    wi_ref = pairing(sol_prev) * dt + forcing.energy_injection_rate * dt
    p_ref = pairing(sol_next)

    set_half!(sol_prev)
    wi = work_ito(forcing, get_coeff_data(u))
    store_prevsol!(forcing, get_coeff_data(u))
    set_half!(sol_next)
    ws = work_stratonovich(forcing, get_coeff_data(u))
    p = instantaneous_power(forcing, get_coeff_data(u))

    @test ws ≈ ws_ref atol=1e-12 rtol=1e-12
    @test wi ≈ wi_ref atol=1e-12 rtol=1e-12
    @test p ≈ p_ref atol=1e-12 rtol=1e-12
end

@testset "Three-dimensional Pencil snapshots preserve logical shape (np=$nprocs)" begin
    n = 4
    coords_3d = CartesianCoordinates("x", "y", "z")
    dist_3d = Distributor(coords_3d)
    bases = ntuple(3) do d
        ComplexFourier(coords_3d[("x", "y", "z")[d]]; size=n, bounds=(0.0, 2π))
    end
    u = ScalarField(dist_3d, "u_3d", bases, ComplexF64); ensure_layout!(u, :c)
    coeffs = get_coeff_data(u)
    @inbounds for I in CartesianIndices(coeffs)
        coeffs[I] = ComplexF64(sum(Tuple(I)), I[1] - I[3])
    end

    forcing = StochasticForcing(
        field_size=(n, n, n), forcing_rate=0.0,
        k_forcing=1.0, dk_forcing=0.1, spectrum_type=:band,
    )
    store_prevsol!(forcing, coeffs)

    @test ndims(forcing.prevsol) == 3
    @test size(forcing.prevsol) == size(coeffs)
    @test all(I -> forcing.prevsol[I] == coeffs[I], CartesianIndices(coeffs))
end
