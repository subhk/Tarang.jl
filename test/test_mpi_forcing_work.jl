# Guard: distributed stochastic-forcing work/power diagnostics match serial (np>=2).
#
# work_stratonovich / work_ito / instantaneous_power slice the cached forcing
# spectrum against the solution and reduce. Under MPI the solution is a
# PencilArray: its CARTESIAN getindex pa[I] is LOGICAL, but its LINEAR index is
# parent/storage order. The diagnostics now use a LOGICAL forcing view
# (_forcing_view_logical) + CartesianIndices iteration, and store_prevsol! copies
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
const W_STRAT_REF = 0.02526697
const W_ITO_REF   = 0.08505513
const P_REF       = 0.65478815

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
