# Notebook: Taylor-Green Vortex

This notebook demonstrates the Taylor-Green vortex, a classic benchmark for spectral codes.

## Overview

The two-dimensional Taylor-Green vortex is an *exact* solution of the incompressible
Navier-Stokes equations: the nonlinear advection term is balanced identically by the pressure
gradient, so the initial vortex array simply decays under viscosity without changing shape.
That makes it the cleanest available check on a spectral code — the numerical answer can be
compared against a closed-form solution at every instant.

We solve it in vorticity-streamfunction form, which is how Tarang handles incompressible flow
on a periodic (all-Fourier) domain:

```math
\begin{aligned}
\partial_t \zeta - \nu \Delta \zeta &= -\mathbf{u} \cdot \nabla \zeta \\
\Delta \psi &= \zeta \\
\mathbf{u} &= (-\partial_y \psi,\; \partial_x \psi)
\end{aligned}
```

The streamfunction formulation enforces incompressibility by construction, so no pressure
variable is needed.

## Setup

```julia
using Tarang
using Statistics
using Printf
using Plots
```

Tarang initializes MPI itself when it loads, so there is no `MPI.Init()`/`MPI.Finalize()` to
call. The script below is unchanged whether you run it serially or under `mpiexec`.

## Parameters

```julia
Re = 100.0       # Reynolds number
nu = 1.0 / Re    # Kinematic viscosity
L = 2π           # Domain size
N = 64           # Resolution
```

## Domain (Doubly Periodic)

```julia
coords = CartesianCoordinates("x", "y")
dist = Distributor(coords; dtype=Float64, device=CPU())

# Fourier bases for periodic boundaries
xbasis = RealFourier(coords["x"]; size=N, bounds=(0.0, L), dealias=3/2)
ybasis = RealFourier(coords["y"]; size=N, bounds=(0.0, L), dealias=3/2)

domain = Domain(dist, (xbasis, ybasis))
```

`dealias=3/2` is the padding factor for the 3/2 rule, and it is already the default for
Fourier bases. (Note that `dealias=2/3` would *disable* dealiasing — the factor is a padding
ratio, not a truncation fraction.)

## Fields

```julia
ζ     = ScalarField(domain, "ζ")      # Vorticity
ψ     = ScalarField(domain, "ψ")      # Streamfunction
u     = VectorField(domain, "u")      # Velocity
tau_ψ = ScalarField(dist, "tau_ψ", (), Float64)   # Gauge for the Poisson solve
```

`tau_ψ` pins the constant mode of `ψ`, which the Poisson equation leaves undetermined.

## Problem Definition

```julia
problem = IVP([ζ, ψ, u, tau_ψ])
add_parameters!(problem, nu=nu)

add_equation!(problem, "∂t(ζ) - nu*lap(ζ) = -u⋅∇(ζ)")   # Vorticity transport
add_equation!(problem, "lap(ψ) + tau_ψ - ζ = 0")        # Poisson: Δψ = ζ
add_equation!(problem, "u - skew(grad(ψ)) = 0")         # u = (-∂y ψ, ∂x ψ)

add_bc!(problem, "integ(ψ) = 0")                        # Fix the constant in ψ
```

Parameters referenced from an equation string (`nu` here) must be registered with
`add_parameters!` — a plain Julia global is not visible to the parser.

## Analytical Solution

The Taylor-Green initial condition and its exact time evolution:

```math
\begin{aligned}
\psi(x,y,t)   &= \phantom{-}\cos(x)\cos(y)\, e^{-2\nu t} \\
\zeta(x,y,t)  &= -2\cos(x)\cos(y)\, e^{-2\nu t} \\
u(x,y,t)      &= \phantom{-}\cos(x)\sin(y)\, e^{-2\nu t} \\
v(x,y,t)      &= -\sin(x)\cos(y)\, e^{-2\nu t}
\end{aligned}
```

Because ``\zeta = -2\psi``, the advection term ``\mathbf{u}\cdot\nabla\zeta`` vanishes
identically and the vortex decays at the pure viscous rate ``e^{-2\nu t}``. Energy and
enstrophy, being quadratic, decay twice as fast:

```math
E(t) = \tfrac{1}{4} L^2 e^{-4\nu t}, \qquad
Z(t) = \tfrac{1}{2} L^2 e^{-4\nu t}
```

(These are domain *integrals*, matching what `total_kinetic_energy` and `total_enstrophy`
return.)

## Initial Conditions

`local_grids` returns each rank's slice of the grid, so this works serially and distributed.
(`set!(field, ::Function)` builds a global mesh and is serial-only — do not use it here.)

```julia
x, y = local_grids(dist, xbasis, ybasis)

ensure_layout!(ζ, :g)
get_grid_data(ζ) .= -2 .* cos.(x) .* cos.(y')
ensure_layout!(ζ, :c)
```

Only `ζ` needs initializing: `ψ` and `u` are diagnosed from it by the constraint equations.

## Diagnostic Functions

Tarang ships the diagnostics, so define only the analytical reference:

```julia
area = L^2
KE_exact(t)  = 0.25 * area * exp(-4 * nu * t)
ENS_exact(t) = 0.5  * area * exp(-4 * nu * t)
```

`total_kinetic_energy(u)` and `total_enstrophy(u)` take a `VectorField` and return a `Float64`
(the domain integral). The un-prefixed `kinetic_energy(u)` and `enstrophy(u)` return the
corresponding `ScalarField`s if you want the spatial distribution.

## Solver

```julia
solver = InitialValueSolver(problem, RK443(); dt=0.01)
@show solver.rhs_plan.is_compiled   # true — the fast compiled RHS is in use

cfl = CFL(solver; initial_dt=0.01, cadence=10, safety=0.5, max_dt=0.05)
add_velocity!(cfl, u)
```

`CFL` takes the **solver**, not the problem, and `add_velocity!` takes the whole
**`VectorField`** — there is no per-component form.

## Simulation

`run!` drives the loop: it applies the CFL controller, fires callbacks, and stops at
`stop_time`.

```julia
t_end = 10.0

times        = Float64[]
KE_numerical = Float64[]
KE_reference = Float64[]

function record!(s)
    t = s.sim_time
    push!(times, t)
    push!(KE_numerical, total_kinetic_energy(u))
    push!(KE_reference, KE_exact(t))
    @printf("t = %6.3f   KE = %.8f   exact = %.8f\n", t, KE_numerical[end], KE_reference[end])
end

run!(solver; stop_time=t_end, cfl=cfl, callbacks=[10 => record!], progress=false)
```

Output (abridged):

```
t =  0.139   KE = 9.81481264   exact = 9.81481264
t =  5.057   KE = 8.06222401   exact = 8.06222401
t =  9.974   KE = 6.62258755   exact = 6.62258755
```

## Results

### Energy Decay

```julia
plot(times, KE_numerical, label="Numerical", marker=:circle)
plot!(times, KE_reference, label="Analytical", linestyle=:dash)
xlabel!("Time")
ylabel!("Kinetic Energy")
title!("Taylor-Green Vortex Energy Decay (Re=$Re)")
```

### Error

```julia
relative_error = abs.(KE_numerical .- KE_reference) ./ KE_reference
@printf("max relative KE error = %.3e\n", maximum(relative_error))   # 9.734e-12

plot(times, relative_error,
    xlabel="Time",
    ylabel="Relative Error",
    title="Energy Error",
    yscale=:log10,
    legend=false
)
```

The remaining error is time-integration error from RK443, not spatial error — the Fourier
representation of the initial condition is exact. Checking the velocity field directly at
``t = 10``:

```julia
ensure_layout!(u, :g)
ux, uy = u.components
uxe =   cos.(x) .* sin.(y') .* exp(-2 * nu * solver.sim_time)
uye = .-sin.(x) .* cos.(y') .* exp(-2 * nu * solver.sim_time)

@printf("max|ux - exact| = %.3e\n", maximum(abs, get_grid_data(ux) .- uxe))   # 4.003e-12
@printf("max|uy - exact| = %.3e\n", maximum(abs, get_grid_data(uy) .- uye))   # 4.003e-12
@printf("enstrophy = %.6f  exact = %.6f\n",
        total_enstrophy(u), ENS_exact(solver.sim_time))   # 13.219147  13.219147
```

### Vorticity Visualization

```julia
ensure_layout!(ζ, :g)
heatmap(x, y, get_grid_data(ζ)',
    xlabel="x", ylabel="y",
    title="Vorticity at t = $(round(solver.sim_time, digits=2))",
    aspect_ratio=:equal, color=:balance
)
```

### Energy Spectrum

All axes are Fourier, so `power_spectrum` applies:

```julia
ps = power_spectrum(ζ)
@printf("peak bin: k = %.2f\n", ps.k[argmax(ps.power)])   # 1.50 — the |k| = √2 mode
```

## Running in Parallel

The script needs no changes to run distributed — the initial condition already uses
`local_grids`, and `total_kinetic_energy` reduces across ranks:

```bash
mpiexec -n 4 julia --project=. taylor_green.jl
```

Verified at 1, 2 and 4 ranks: the kinetic energy agrees with the analytical value to
roundoff (relative error ~5e-15 after 100 steps).

## The Three-Dimensional Taylor-Green Vortex

The Re = 1600 three-dimensional Taylor-Green vortex is the classic transition-to-turbulence
benchmark, and it is *not* what this notebook computes. Two things are worth being clear about:

1. **It has no closed-form solution.** The exact decay above is a two-dimensional result. In
   3D the initial vortex is not a solution of Navier-Stokes; it stretches, breaks down, and
   the enstrophy peaks around ``t \approx 9`` before decaying. There is nothing to validate
   against except reference data.

2. **Tarang cannot currently run it.** A 3D incompressible flow needs a velocity-pressure
   formulation (`div(u) = 0` plus `∇(p)` in the momentum equation), and on a triply periodic
   all-Fourier domain Tarang builds no per-mode implicit solve for that constraint. The
   assembled global matrix is singular, the timestepper warns once
   (`IMEX RK: system matrix is singular, falling back to explicit`) and then drops *every*
   implicit term — pressure and viscosity alike — so the run proceeds and silently produces a
   wrong answer (the energy does not decay at all).

   Per-mode tau subproblems are only built when a coupled (Chebyshev/Jacobi) direction is
   present. So a 3D incompressible run needs at least one wall-bounded direction; see
   `examples/ivp/rotating_rayleigh_benard_3d.jl` for a working velocity-pressure setup.

## Exercises

1. **Convergence study**: N = 16, 32, 64 — the spatial error is already at roundoff, so what
   you are really measuring is the RK443 temporal error. Vary `max_dt` instead.
2. **Reynolds number scan**: Re = 100, 400, 1600. The decay rate should track ``e^{-4\nu t}``
   exactly at every Re, since the solution is exact for all ``\nu``.
3. **Perturbed vortex**: add noise with `fill_random!(ζ, "g"; seed=42, scale=1e-2)` on top of
   the Taylor-Green field and watch the exact-solution agreement break down.
4. **Enstrophy budget**: compare `total_enstrophy(u)` against ``\tfrac{1}{2}L^2 e^{-4\nu t}``.

## References

- Taylor, G.I. & Green, A.E. (1937). Mechanism of the production of small eddies from large ones
- Brachet, M.E. et al. (1983). Small-scale structure of the Taylor-Green vortex
