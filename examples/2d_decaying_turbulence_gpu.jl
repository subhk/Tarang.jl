"""
2D decaying turbulence on a GPU (doubly-periodic, vorticity–streamfunction form).

This is the GPU's strongest path: a pure Fourier×Fourier domain → all transforms
are CUFFT (no DCT, no MPI). Set `USE_GPU = true` to run on the GPU; the ONLY
difference from the CPU version is `device = GPU()`.

Incompressible 2D Navier–Stokes in vorticity (q = ∇²ψ) / streamfunction (ψ) form:

    ∂t(q) - ν Δq = -u·∇q          (vorticity transport; viscous term implicit)
    Δψ + τ_ψ - q = 0              (streamfunction Poisson; τ_ψ fixes the gauge)
    u - skew(∇ψ) = 0              (velocity u = ∇⊥ψ, divergence-free by construction)
    ∫ψ = 0                        (gauge: zero-mean streamfunction)

Starting from a band-limited random vorticity field, the flow self-organizes into
coherent vortices. The hallmark of *2D* (vs 3D) turbulence: enstrophy Z = ½⟨q²⟩
cascades to small scales and is strongly dissipated by viscosity (decays), while
energy E = ½⟨|u|²⟩ is nearly conserved (inverse cascade keeps it at large scales,
so it decays only slowly). Both are tracked below.

Run:  julia --project=. examples/2d_decaying_turbulence_gpu.jl
"""

using Tarang
using Random
using Statistics: mean

const USE_GPU = true          # ← false to run the identical problem on CPU
const N       = 256           # grid points per side (use 64–128 on CPU)
const ν       = 1.0e-4        # kinematic viscosity (small → long-lived turbulence)
const dt      = 2.0e-3
const NSTEPS  = 4000
const REPORT  = 200

# ── Architecture ────────────────────────────────────────────────────────────
device = USE_GPU ? GPU() : CPU()      # GPU() needs CUDA.jl loaded + a device
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=device)

# 3/2 dealiasing is required for the quadratic nonlinearity u·∇q.
xb = RealFourier(coords["x"]; size=N, bounds=(0.0, 2π), dealias=3/2)
yb = RealFourier(coords["y"]; size=N, bounds=(0.0, 2π), dealias=3/2)
domain = Domain(dist, (xb, yb))

# ── Fields ──────────────────────────────────────────────────────────────────
q       = ScalarField(domain, "q")              # vorticity
psi     = ScalarField(domain, "psi")            # streamfunction
u       = VectorField(domain, "u")              # velocity (2 components)
tau_psi = ScalarField(dist, "tau_psi", (), Float64)   # gauge multiplier

# ── Problem ─────────────────────────────────────────────────────────────────
problem = IVP([q, psi, u, tau_psi])
add_parameters!(problem; nu=ν)
add_equation!(problem, "∂t(q) - nu*Δ(q) = -u⋅∇(q)")   # viscous implicit, advection explicit
add_equation!(problem, "Δ(psi) + tau_psi - q = 0")    # Δψ = q
add_equation!(problem, "u - skew(grad(psi)) = 0")     # u = ∇⊥ψ
add_bc!(problem, "integ(psi) = 0")                    # zero-mean gauge

# DiagonalIMEX_SBDF2, not SBDF2: on a pure-Fourier GPU IVP the solver builds no
# global matrix, so the implicit viscous term `nu*Δ(q)` has no per-mode solve on a
# standard IMEX/multistep scheme and would be silently dropped — the flow would run
# INVISCID (no enstrophy dissipation, the whole point of this example). The
# diagonal-IMEX schemes solve the diagonal Fourier operator (−nu k²) per mode
# on-device, which is exactly right here. On CPU it gives identical results to
# SBDF2, so this one line works on both devices.
solver = InitialValueSolver(problem, DiagonalIMEX_SBDF2(); dt=dt)

# ── Initial condition: band-limited random vorticity (energy in k ∈ [4, 8]) ──
# Built on the host, then assigned to the field (uploaded to device automatically).
Random.seed!(42)
xc = Tarang.get_grid_coordinates(domain; on_device=false)["x"]   # (N,)
yc = Tarang.get_grid_coordinates(domain; on_device=false)["y"]   # (N,)
q0 = zeros(N, N)
for kx in 1:10, ky in 1:10
    k = hypot(kx, ky)
    (4 <= k <= 8) || continue
    a, b = randn(), 2π * rand()
    @. q0 += a * sin(kx * xc + b) * cos(ky * yc' + b)
end
q0 .*= 10.0 / maximum(abs, q0)        # set a representative amplitude
q["g"] = q0

# ── Diagnostics ───────────────────────────────────────────────────────────--
function energy_enstrophy(q, u)
    ensure_layout!(q, :g)
    ensure_layout!(u.components[1], :g); ensure_layout!(u.components[2], :g)
    ux = get_grid_data(u.components[1]); uy = get_grid_data(u.components[2])
    qg = get_grid_data(q)
    E = 0.5 * mean(@. ux^2 + uy^2)    # reductions run on-device for CuArray
    Z = 0.5 * mean(qg .^ 2)
    return E, Z
end

# Populate u/ψ from the initial q (one RHS eval refreshes the algebraic state).
Tarang.evaluate_rhs(solver, solver.state, 0.0)
E0, Z0 = energy_enstrophy(q, u)
@info "2D decaying turbulence" device=typeof(device) N ν dt NSTEPS E0 Z0

# ── Time integration ─────────────────────────────────────────────────────────
# run! drives the loop with the solver's fixed dt (no CFL controller here). The
# REPORT callback fires every REPORT iterations and prints the same diagnostics
# the manual loop did; s.sim_time == step*dt at that point.
function report(s)
    E, Z = energy_enstrophy(q, u)
    @info "t=$(round(s.sim_time; digits=3))" E=round(E; sigdigits=4) Z=round(Z; sigdigits=4) E_frac=round(E/E0; digits=3) Z_frac=round(Z/Z0; digits=3)
end

run!(solver; stop_iteration=NSTEPS, callbacks=[REPORT => report])

Ef, Zf = energy_enstrophy(q, u)
@info "done" E_final=Ef Z_final=Zf energy_decayed=(Ef < E0) enstrophy_decayed=(Zf < Z0)
