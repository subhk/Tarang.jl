# Tutorial: 2D Rayleigh-Bénard Convection

This tutorial walks through the canonical Tarang example: 2D Rayleigh-Bénard convection between two horizontal plates. The code shown here **mirrors `examples/ivp/rayleigh_benard_2d.jl` in the repository** — so you can copy-paste either and they produce the same simulation.

## Physical problem

Rayleigh-Bénard convection is driven by heating a fluid layer from below and cooling it from above. The setup:

- Horizontally periodic layer of fluid between two rigid plates
- Bottom plate held at temperature `T = 1` (hot)
- Top plate held at temperature `T = 0` (cold)
- Gravity acts downward
- No-slip velocity on both plates

Above a critical Rayleigh number (`Ra_c ≈ 1708`), conduction becomes unstable and convection cells form.

### Governing equations (thermal-diffusive non-dimensionalization)

We non-dimensionalize using the box height `H` and the thermal diffusion time `τ_κ = H² / κ`:

- length scale: `H`
- time scale: `τ_κ = H² / κ`
- velocity scale: `κ / H`
- pressure scale: `ρ (κ/H)²`
- temperature scale: `ΔT = T_bottom − T_top`

Define `T̃ = (T − T_top) / ΔT ∈ [0, 1]`. The dimensionless equations become:

```math
\begin{aligned}
\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u}
  &= -\nabla p + \mathrm{Pr}\, \nabla^2 \mathbf{u} + \mathrm{Ra}\cdot\mathrm{Pr}\, T\, \hat{\mathbf{z}} \\
\partial_t T + \mathbf{u} \cdot \nabla T &= \nabla^2 T \\
\nabla \cdot \mathbf{u} &= 0
\end{aligned}
```

Dimensionless parameters:
- **`Pr = ν/κ`** — Prandtl number (the viscous coefficient in the momentum equation)
- **`Ra = g α ΔT H³ / (ν κ)`** — Rayleigh number (buoyancy amplitude is `Ra·Pr`)

The defining feature of the thermal-diffusive time scale is that the temperature equation has **unit diffusivity** — one time unit equals one thermal diffusion time, so the onset of convection and Nusselt scaling read off naturally.

Boundary conditions:
- `T(z=0) = 1`, `T(z=Lz) = 0` (fixed hot/cold)
- `u(z=0) = u(z=Lz) = 0` (no-slip)

This problem runs **in serial only** — the explicit advection terms need a Chebyshev
derivative, which Tarang cannot evaluate on a distributed field. See
[Running the simulation](#Running-the-simulation) below.

### First-order reformulation

For better spectral conditioning — especially at high resolution — we rewrite the second-order operators in **first-order form** using auxiliary gradient variables:

```math
\begin{aligned}
\nabla_T &= \nabla T + \hat{\mathbf{z}}\, \mathrm{Lift}(\tau_{T_1}) \\
\nabla_u &= \nabla \mathbf{u} + \hat{\mathbf{z}}\, \mathrm{Lift}(\tau_{u_1})
\end{aligned}
```

and replace `∇²T` with `∇·∇_T` and `∇²u` with `∇·∇_u`. The equations then become:

```math
\begin{aligned}
\mathrm{tr}(\nabla_u) + \tau_p &= 0 \\
\partial_t T - \nabla \cdot \nabla_T + \mathrm{Lift}(\tau_{T_2}) &= -\mathbf{u} \cdot \nabla T \\
\partial_t \mathbf{u} - \mathrm{Pr}\, \nabla \cdot \nabla_u + \nabla p - \mathrm{Ra}\cdot\mathrm{Pr}\, T\, \hat{\mathbf{z}} + \mathrm{Lift}(\tau_{u_2}) &= -\mathbf{u} \cdot \nabla \mathbf{u}
\end{aligned}
```

with five tau fields (`τ_p`, `τ_{T_1}`, `τ_{T_2}`, `τ_{u_1}`, `τ_{u_2}`) supplying the extra degrees of freedom needed to enforce the five algebraic constraints (2 T BCs, 2 u BCs, 1 pressure gauge). See the [Tau method](../pages/tau_method.md) page for why the first-order formulation is recommended.

## Complete implementation

### Setup and parameters

```julia
using Tarang
using Printf

# ─── Parameters ───────────────────────────────────────────────
Lx, Lz      = 4.0, 1.0
Nx, Nz      = 256, 64
Rayleigh    = 2e6
Prandtl     = 1.0
dealias     = 3/2
stop_time   = 25.0    # thermal diffusion times
max_dt      = 0.001   # small to resolve viscous/thermal transport

# Coefficients in the diffusive-time formulation:
#   ∂t(T) - ∇²T = -u⋅∇T              (unit diffusivity)
#   ∂t(u) - Pr ∇²u + ∇p - Ra·Pr T ẑ = -u⋅∇u
nu   = Prandtl                  # viscous coefficient (= Pr in diffusive units)
buoy = Rayleigh * Prandtl       # buoyancy forcing Ra·Pr

@root_only begin
    println("2D Rayleigh-Benard Convection (thermal-diffusive scaling)")
    @printf("  Ra=%.2e, Pr=%.1f\n", Rayleigh, Prandtl)
    @printf("  Coefficients: Pr=%.4f, Ra·Pr=%.2e\n", Prandtl, buoy)
    @printf("  Domain: %.1f × %.1f, Resolution: %d × %d\n", Lx, Lz, Nx, Nz)
    @printf("  Stop time: %.1f thermal diffusion times\n\n", stop_time)
end
```

### Bases and domain

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())

xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=dealias)
zbasis = ChebyshevT(coords["z"];  size=Nz, bounds=(0.0, Lz), dealias=dealias)
```

### Fields

```julia
domain = Domain(dist, (xbasis, zbasis))

p = ScalarField(domain, "p")
T = ScalarField(domain, "T")        # Dimensionless temperature in [0, 1]
u = VectorField(domain, "u")
```

### Tau fields

Five tau fields for five algebraic constraints. Each "drops the coupled direction" — they live on `(xbasis,)` only, not on both axes, because the lift injects them at a single Chebyshev mode.

```julia
tau_p  = ScalarField(dist, "tau_p",  (),         Float64)
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,),  Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,),  Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)
```

- **`tau_p`** is 0-D (no bases) — a gauge-fixing scalar at the DC Fourier mode only. Valid-mode filtering drops it from non-DC subproblems automatically.
- **`tau_T1`, `tau_T2`** are scalars on `(xbasis,)` — 1 DOF per Fourier mode.
- **`tau_u1`, `tau_u2`** are vectors on `(xbasis,)` — 1 DOF per component per Fourier mode (= 2 DOFs per mode in 2D).

### First-order substitutions

```julia
ex, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A) = lift(A, lift_basis, -1)

grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)
```

The `lift_basis = derivative_basis(zbasis)` is the idiomatic convention (see the [Tau method](../pages/tau_method.md) page for the reason). The closure `τ_lift(A)` injects the tau field at the last Chebyshev coefficient (`-1` = last mode, wraparound-indexed).

`grad_u` and `grad_T` are the augmented gradients — they carry the tau corrections that will enforce the bottom-wall BCs on the gradient itself.

### Problem and equations

```julia
problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])

add_parameters!(problem,
    nu=nu, buoy=buoy, ez=ez,
    grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

# Continuity (first-order form)
add_equation!(problem, "trace(grad_u) + tau_p = 0")

# Temperature: ∂t(T) - ∇²T = -u⋅∇T
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")

# Momentum: ∂t(u) - Pr∇²u + ∇p - Ra·Pr T ẑ = -u⋅∇u
add_equation!(problem,
    "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")
```

Key substitutions:
- `trace(grad_u)` replaces `div(u)` in the continuity equation and implicitly carries the `τ_{u_1}` lift.
- `div(grad_T)` replaces `∇²T`; `div(grad_u)` replaces `∇²u`.
- `τ_lift(tau_T2)` and `τ_lift(tau_u2)` are the tau corrections for the evolution-equation side (top-wall BCs).

### Boundary conditions

```julia
add_bc!(problem, "T(z=0) = 1")        # hot bottom
add_bc!(problem, "T(z=$Lz) = 0")      # cold top
add_bc!(problem, "u(z=0) = 0")        # no-slip bottom
add_bc!(problem, "u(z=$Lz) = 0")      # no-slip top
add_bc!(problem, "integ(p) = 0")      # pressure gauge
```

!!! warning "BC strings are parsed by Tarang, not by Julia"
    The string handed to `add_bc!` is parsed by Tarang's own expression parser, which only
    knows the problem's variables and its `add_parameters!` names — **it cannot see Julia
    globals**. Writing `"T(z=Lz) = 0"` does not fail loudly: the parser warns
    `Unknown variable: Lz`, builds the equation with zero output DOFs, and the run then dies
    with a `DimensionMismatch` from the (now non-square) subproblem solve. Either interpolate
    the Julia value into the string (`"T(z=$Lz) = 0"`, which is what is done above and expands
    to `T(z=1.0)`), or write a literal (`"T(z=1) = 0"`).

Boundary conditions must always go through `add_bc!` — declaring one with `add_equation!`
compiles, but the constraint is silently not enforced.

The tau DOFs and the BC rows balance: per Fourier mode there are six tau unknowns
(`tau_T1`, `tau_T2`, and two components each of `tau_u1`, `tau_u2`) against six BC rows
(`T` and `u` at both walls, counting velocity components), plus the global gauge pair
`tau_p` ↔ `integ(p)`. That balance is what makes the filtered subproblem matrix square.

### Solver

```julia
solver = InitialValueSolver(problem, RK222(); dt=max_dt)

@root_only diagnose(solver)
```

`diagnose(solver)` prints a tree-style summary of the solver — domain, bases, state fields, equations, BC count, lazy RHS plan status. Useful for debugging setup problems.

### Output handler

```julia
snapshots = add_file_handler("snapshots", solver; sim_dt=0.1, max_writes=50)
add_task!(snapshots, T;               name="temperature")
add_task!(snapshots, u.components[1]; name="ux")
add_task!(snapshots, u.components[2]; name="uz")
```

Pass the **solver** — that form registers the handler with the solver's evaluator, so `run!`
processes it every step with no further wiring. The other method,
`add_file_handler(path, dist, vars)`, does *not* auto-register; a handler built that way is
never written unless you also pass it as `run!(solver; outputs=[snapshots], …)`.

The written file is NetCDF-4 with groups, so read tasks back out of the `vars` group rather
than from the file root:

```julia
f = "snapshots/snapshots_s1/snapshots_s1.nc"
Tarang.group_variable_names(f, "vars")        # ["temperature", "ux", "uz"]
Tarang.group_ncread(f, "vars", "temperature") # (write, x, z)
Tarang.group_ncread(f, "time", "sim_time")    # (write,)
```

### Initial conditions

Start from the conduction profile `T(z) = 1 − z/Lz` with a small random perturbation to trigger convection:

```julia
x, z = local_grids(dist, xbasis, zbasis)
fill_random!(T, "g"; seed=42, distribution="normal", scale=1e-3)
get_grid_data(T) .*= z' .* (Lz .- z')        # damp noise at walls
get_grid_data(T) .+= 1.0 .- z' ./ Lz          # add linear conduction profile
ensure_layout!(T, :c)                          # pre-compute coefficients for timestepper
```

### CFL-adaptive time stepping

```julia
cfl = CFL(solver; initial_dt=max_dt, cadence=10, safety=0.5,
          threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
add_velocity!(cfl, u)
```

`CFL` is constructed from the **solver** (not the problem), and `add_velocity!` takes a
`VectorField`. Building the controller is not enough — you must hand it to `run!` with
`cfl=cfl` (below), otherwise `solver.dt` stays pinned at its constructor value for the whole
run.

### Diagnostic callback

```julia
ensure_layout!(T, :g)
T_data = get_grid_data(T)
@root_only @printf("Initial T: max=%.4f, min=%.4f, mean=%.4f\n",
                    maximum(T_data), minimum(T_data), sum(T_data)/length(T_data))
@root_only @printf("  z at max|T|: %.4f\n", z[argmax(T_data)[2]])
ensure_layout!(T, :c)
```

### Main loop

```julia
@root_only println("Starting main loop")

run!(solver;
     stop_time=stop_time,
     cfl=cfl,
     log_interval=100,
     callbacks=[
         on_interval(1) do s
             if s.iteration <= 5 || s.iteration % 10 == 0
                 ensure_layout!(T, :g)
                 max_T = global_max(dist, abs.(get_grid_data(T)))
                 ensure_layout!(u.components[2], :g)
                 max_uz = global_max(dist, abs.(get_grid_data(u.components[2])))
                 @root_only @printf("  iter=%d, t=%.4e, dt=%.4e, max|T|=%.4f, max|uz|=%.4e\n",
                                     s.iteration, s.sim_time, s.dt, max_T, max_uz)
             end
         end
     ])

@root_only println("Done!")
```

## Running the simulation

```bash
julia --project=. examples/ivp/rayleigh_benard_2d.jl
```

### Why this problem is serial-only

Do not launch it under `mpiexec`. Two independent limits of the distributed Chebyshev support
block it, and both are loud rather than silently wrong:

1. **The Chebyshev axis cannot be decomposed.** PencilArrays splits the *trailing* dimensions,
   and a Chebyshev DCT needs the whole axis local on each rank. `Domain(dist, (xbasis, zbasis))`
   — the Fourier-first order this tutorial uses — therefore errors at construction under
   `nprocs > 1`:

   > `MPI mixed Fourier-Chebyshev: the decomposed (trailing) axis/axes [2] are non-Fourier … Reorder your bases so the Chebyshev axis comes BEFORE the Fourier axes`

2. **A Chebyshev derivative cannot appear on the explicit side.** Reordering to
   `(zbasis, xbasis)` fixes the decomposition, and the solver still builds — but the advection
   terms `-u⋅∇(T)` and `-u⋅∇(u)` expand to a `∂z` on the RHS, and the first step dies:

   > `Lazy RHS: cannot differentiate along the non-Fourier axis 1 (ChebyshevT) of a DISTRIBUTED field — this rank owns only … of the … coefficients on that axis, so the global differentiation matrix does not apply to its local slab … Move the term to the implicit (L) side of the equation, or run in serial.`

   Reordering also forces the tau fields onto no bases (`()` instead of `(xbasis,)`), because
   per-Fourier-mode taus require the Fourier axis to come first.

What *does* run distributed on a Chebyshev–Fourier domain is a problem whose explicit RHS
differentiates only along **Fourier** axes (all Chebyshev derivatives on the implicit `L` side)
— see `test/test_mpi_cheb_fourier_ivp_nonlinear.jl`. Fully periodic problems parallelize
without restriction; see the [3D Turbulence](ivp_3d_turbulence.md) tutorial.

Expected early output:

```
2D Rayleigh-Benard Convection (thermal-diffusive scaling)
  Ra=2.00e+06, Pr=1.0
  Coefficients: Pr=1.0000, Ra·Pr=2.00e+06
  Domain: 4.0 × 1.0, Resolution: 256 × 64

Initial T: max=1.0000, min=0.0000, mean=0.5000
  z at max|T|: 0.0000
Starting main loop
  iter=1, t=1.0000e-03, dt=1.0000e-03, max|T|=1.0000, ...
  iter=2, t=2.0000e-03, dt=1.0000e-03, max|T|=1.0000, ...
  ...
```

`max|T|` should stay at `≈ 1.0` throughout (pinned by the bottom-wall BC). If you see it decay to zero or stick at `2 + √2 ≈ 3.414`, that's a BC-enforcement bug — see the [Tau method](../pages/tau_method.md) troubleshooting section.

## Physical interpretation

### Flow regimes

| Ra | Regime | Characteristics |
|---|---|---|
| < 1708 | Conduction | No flow, pure linear conduction profile |
| 10³–10⁴ | Steady rolls | Stationary convection cells |
| 10⁵–10⁶ | Transitional | Time-dependent, irregular rolls |
| > 10⁶ | Turbulent | Chaotic small-scale structures |

### Nusselt number

The Nusselt number measures the ratio of total heat flux to the conductive heat flux:

```math
\mathrm{Nu} = \frac{\langle u_z T \rangle_{xz} + \langle -\partial_z T \rangle_{xz}}{\langle -\partial_z T_\mathrm{cond} \rangle_{xz}}
```

- `Nu = 1` → pure conduction (no convection)
- `Nu > 1` → enhanced heat transfer from convection
- Asymptotically, `Nu ~ Ra^(1/3)` in the turbulent regime

You can compute it online via a callback that integrates over the full domain.

## Visualization

Plots.jl is not a Tarang dependency — add it to your own environment first.

```julia
using Plots

ensure_layout!(T, :g)
T_grid = Array(get_grid_data(T))       # (Nx, Nz)
heatmap(T_grid', xlabel="x", ylabel="z", aspect_ratio=:equal,
        title="Temperature (Ra = $(Rayleigh))")
savefig("temperature.png")
```

## Parameter studies

### Exploring Ra

```julia
Ra_values = [1e4, 1e5, 1e6, 2e6]

for Ra in Ra_values
    nu   = Prandtl
    buoy = Ra * Prandtl
    # ... rebuild problem with new `buoy` parameter and run
end
```

### Aspect ratio

```julia
aspect_ratios = [2.0, 4.0, 8.0]

for ar in aspect_ratios
    Lx = ar * Lz
    xbasis = RealFourier(coords["x"]; size=Int(64*ar), bounds=(0.0, Lx), dealias=3/2)
    # ... rebuild problem and run
end
```

## Performance notes

### Resolution

For `Ra = 2×10⁶` in 2D:
- Minimum: `128 × 32` (under-resolved but stable)
- Recommended: `256 × 64`
- High-fidelity: `512 × 128`

Scaling: roughly `Nx, Nz × 1.5` for each `Ra × 10`.

### Timestep control

```julia
# For high Ra, reduce the safety factor to stay stable
cfl = CFL(solver; initial_dt=max_dt, safety=0.3, cadence=10, max_dt=max_dt)

# Smooth dt evolution to avoid thrashing the LHS cache
cfl = CFL(solver; initial_dt=max_dt, max_change=1.2, min_change=0.5, max_dt=max_dt)

add_velocity!(cfl, u)
```

The LHS solver cache in `step_subproblem_rk!` (`sp.LHS_solvers`) is keyed by the stage
coefficient `a_ii`, and every entry is marked dirty whenever `dt` changes — so if `dt` changes
every step, the sparse LU is refactored every step, which is wasteful. Keeping `max_change`
modest, and `cadence` above 1, lets CFL ride out several steps at a fixed `dt` between
re-factorizations.

## Troubleshooting

### `max|T|` decays to 0

Symptom: `T(z=0) = 1` BC isn't being enforced; temperature drifts toward zero.

Most common cause: a missing or mis-declared tau field. Check that you have exactly one tau DOF per BC (counting vector components and gauge constraints). See the [Tau method → Common Pitfalls](../pages/tau_method.md) section.

### `max|T|` sticks at `2 + √2 ≈ 3.414`

Symptom: RK222 is scaling the BC by `1/γ = 1/(1 − 1/√2) = 2 + √2`.

Cause: the `apply_bc_override!` DAE path in `step_subproblem_rk!` isn't firing for this BC. Make sure the BC equation is classified into `sp.bc_rows` (it should be, for any `eq_size < Nz` algebraic equation). If you see this with a custom equation type, file an issue.

### NaN in the solution

Causes and fixes:
- Timestep too large → reduce `cfl.safety` to 0.2–0.3 (`CFL` is mutable, so you can set the field directly)
- Insufficient resolution → bump `Nz` up
- Missing dealiasing → `dealias` is a **padding factor**, so `3/2` is the 3/2 rule and any value `≤ 1` switches dealiasing off. It only acts on Fourier axes; the factor passed to `zbasis` is ignored.

### Simulation not converging to a steady state

- Run longer — thermal diffusion time is `τ_κ = H² / κ`, and steady state takes many diffusion times
- Check that initial perturbation is large enough to grow above roundoff
- Verify `Ra > 1708` (the critical Rayleigh number for onset)

### Memory issues

- Reduce `Nx`, `Nz`
- Switch to `CNAB2` or `SBDF2` (smaller LU cache than multi-stage RK443)

More MPI ranks will not help here — this problem is serial-only, see
[Running the simulation](#Running-the-simulation).

## Complete script

The full example is at `examples/ivp/rayleigh_benard_2d.jl` in the repository. Copy-paste either from this tutorial or from that file — they're kept in sync.

## Next steps

- **[3D Turbulence](ivp_3d_turbulence.md)** — extend to 3D with two Fourier axes
- **[Boundary Conditions](boundary_conditions.md)** — more BC patterns, including time/space-dependent
- **[Eigenvalue Problems](eigenvalue_problems.md)** — linear stability around the conduction profile
- **[Tau method](../pages/tau_method.md)** — the full story on how BCs are enforced

## References

1. Chandrasekhar, S. (1961). *Hydrodynamic and Hydromagnetic Stability*. Oxford.
2. Tritton, D. J. (1988). *Physical Fluid Dynamics*. Oxford.
3. Burns, K. J., Vasil, G. M., Oishi, J. S., Lecoanet, D., & Brown, B. P. (2020). "Dedalus: A flexible framework for numerical simulations with spectral methods." *Physical Review Research*, 2, 023068.
