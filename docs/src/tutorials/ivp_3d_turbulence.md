# Tutorial: 3D Turbulence Simulation

This tutorial demonstrates setting up and running a 3D turbulent flow simulation using Tarang.jl.

## Physical Problem

We simulate decaying turbulence in a triply-periodic box, governed by the incompressible Navier-Stokes equations:

```math
\begin{aligned}
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} &= -\nabla p + \nu \nabla^2 \mathbf{u} \\
\nabla \cdot \mathbf{u} &= 0
\end{aligned}
```

where $\mathbf{u} = (u, v, w)$ is the velocity, $p$ the pressure (divided by density), and $\nu$ the
kinematic viscosity.

### Formulation: vorticity and vector potential

Tarang's pure-Fourier time-stepping path integrates each field with an implicit linear operator that
is *diagonal* in Fourier space (a Laplacian, a hyper/fractional Laplacian, a constant damping, or a
derivative of the field itself). The pressure-velocity saddle-point system is not of that form: an
implicit $\nabla p$ couples a *different* field into the momentum equation, and the solver rejects it
(under MPI it raises an error; in serial the coupled system is singular and the implicit solve is
silently dropped, leaving $p = 0$ and a velocity that is no longer divergence-free).

In a triply-periodic box the pressure can be removed analytically, which is what we do here. Taking
the curl of the momentum equation gives the vorticity equation, and the velocity is recovered from a
vector potential $\mathbf{A}$ in the Coulomb gauge:

```math
\begin{aligned}
\frac{\partial \boldsymbol{\omega}}{\partial t} - \nu \nabla^2 \boldsymbol{\omega}
   &= -(\mathbf{u}\cdot\nabla)\boldsymbol{\omega} + (\boldsymbol{\omega}\cdot\nabla)\mathbf{u} \\
\nabla^2 \mathbf{A} &= -\boldsymbol{\omega}, \qquad \mathbf{u} = \nabla \times \mathbf{A}
\end{aligned}
```

Every implicit term is now a Laplacian (diagonal), every nonlinear term is explicit and differentiated
along Fourier axes only, and the two constraint equations are exactly the Poisson / algebraic-substitution
pattern the solver refreshes each step.

A velocity built as $\mathbf{u} = \nabla\times\mathbf{A}$ is divergence-free *whatever* $\mathbf{A}$ is, so
$\max|\nabla\cdot\mathbf{u}| \approx 8\times10^{-15}$ (measured on the $32^3$ run below) confirms nothing on
its own. The check that actually tests the Poisson solve is the reconstructed velocity itself: after one
step `max|u1| = 0.999940`, matching the analytic viscous decay of the Taylor-Green amplitude,
$e^{-3\nu\,\Delta t} = 0.999940$, to seven digits.

## Domain Setup

### Coordinates and Distribution

```julia
using Tarang
using MPI
using Printf

MPI.Init()

coords = CartesianCoordinates("x", "y", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
```

Leave `mesh` unset unless you have a reason: the distributor picks a valid process mesh for you
(`(1, 2)` on 2 ranks, `(2, 2)` on 4 ranks).

!!! warning "An N-dimensional domain takes an (N−1)-dimensional process mesh"
    PencilFFTs must keep at least one axis process-local to transform it. A 3-element mesh over a 3D
    domain decomposes all three axes and the run dies at plan creation with
    `PencilFFT plan creation failed with 8 MPI processes`. For a 3D domain the mesh is a **2D pencil
    grid**: `mesh=(2, 4)` on 8 ranks, `mesh=(8, 8)` on 64, `mesh=(16, 32)` on 512.

### Spectral Bases

All directions are periodic, so we use Fourier bases:

```julia
L  = 2π       # domain size
N  = 32       # N^3 grid points
Re = 100.0
nu = 1.0 / Re

x_basis = RealFourier(coords["x"]; size=N, bounds=(0.0, L), dealias=3/2)
y_basis = RealFourier(coords["y"]; size=N, bounds=(0.0, L), dealias=3/2)
z_basis = RealFourier(coords["z"]; size=N, bounds=(0.0, L), dealias=3/2)

domain = Domain(dist, (x_basis, y_basis, z_basis))
```

!!! note "Dealiasing"
    `dealias` is a **padding factor**, not a cutoff fraction. `dealias=3/2` pads the transform by 3/2
    and truncates the nonlinear product back to $|k| \le N/3$ — the 3/2 rule (equivalently, the 2/3
    rule). Any value **≤ 1 disables dealiasing entirely**, so `dealias=2/3` leaves the product fully
    aliased. `3/2` is also the default for `RealFourier` and `ComplexFourier`.

!!! info "No Tau Fields Needed for Boundaries"
    Since all directions use periodic Fourier bases, **no boundary conditions are required**. The tau
    variables below are gauge constants for the Poisson equations (they fix the $k=0$ mode of the
    vector potential), not boundary-condition unknowns. Tau fields with the `lift()` operator are
    needed for non-periodic directions — see the [Boundary Conditions Tutorial](boundary_conditions.md).

## Fields and Problem Definition

The unknowns are the three vorticity components (time-stepped), the three vector-potential components
and the three velocity components (both solved from constraints each step), plus one gauge constant per
Poisson equation:

```julia
w1 = ScalarField(domain, "w1"); w2 = ScalarField(domain, "w2"); w3 = ScalarField(domain, "w3")
A1 = ScalarField(domain, "A1"); A2 = ScalarField(domain, "A2"); A3 = ScalarField(domain, "A3")
u1 = ScalarField(domain, "u1"); u2 = ScalarField(domain, "u2"); u3 = ScalarField(domain, "u3")

tau_A1 = ScalarField(dist, "tau_A1", (), Float64)
tau_A2 = ScalarField(dist, "tau_A2", (), Float64)
tau_A3 = ScalarField(dist, "tau_A3", (), Float64)

problem = IVP([w1, w2, w3, A1, A2, A3, u1, u2, u3, tau_A1, tau_A2, tau_A3])
add_parameters!(problem, nu=nu)

# Vorticity transport: advection + vortex stretching on the explicit side,
# viscous diffusion on the implicit side.
for (w, u) in (("w1", "u1"), ("w2", "u2"), ("w3", "u3"))
    add_equation!(problem,
        "∂t($w) - nu*Δ($w) = -(u1*∂x($w) + u2*∂y($w) + u3*∂z($w))" *
        " + (w1*∂x($u) + w2*∂y($u) + w3*∂z($u))")
end

# Vector potential: Δ(A) = -ω, one Poisson equation per component.
add_equation!(problem, "Δ(A1) + tau_A1 + w1 = 0")
add_equation!(problem, "Δ(A2) + tau_A2 + w2 = 0")
add_equation!(problem, "Δ(A3) + tau_A3 + w3 = 0")

# Velocity: u = ∇ × A.
add_equation!(problem, "u1 - (∂y(A3) - ∂z(A2)) = 0")
add_equation!(problem, "u2 - (∂z(A1) - ∂x(A3)) = 0")
add_equation!(problem, "u3 - (∂x(A2) - ∂y(A1)) = 0")

# Gauge: fix the mean of each vector-potential component.
add_bc!(problem, "integ(A1) = 0")
add_bc!(problem, "integ(A2) = 0")
add_bc!(problem, "integ(A3) = 0")
```

!!! note "Equation Syntax"
    - `∂t(w1)` — time derivative
    - `Δ(w1)` — Laplacian
    - `∂x(w1)`, `∂y(w1)`, `∂z(w1)` — partial derivatives (`dx(...)` is **not** valid syntax)
    - `integ(A1)` — domain integral, used here as the gauge condition
    - The problem is validated with one equation per variable (counting `add_bc!` conditions), which is
      why each of the three tau constants carries its own gauge condition.

## Solver Setup

```julia
solver = InitialValueSolver(problem, RK443(); dt=2e-3)

# The RHS compiles to a type-specialized plan (no interpreted fallback):
@assert solver.rhs_plan.is_compiled
```

`RK443` is an IMEX Runge-Kutta scheme; `RK222`, `SBDF2` and `SBDF3` are also exported and work with
this problem. Under MPI the pure-Fourier diagonal-IMEX path takes over automatically — the viscous
term stays implicit, the advection and vortex-stretching terms explicit.

!!! note "Three `Laplacian ... approximated as IDENTITY-on-operand` warnings are expected"
    Building the solver prints, once per Poisson equation:

    ```
    ┌ Warning: Implicit global matrix: operator Laplacian{...} has no spectral matrix builder and is
    │ approximated as IDENTITY-on-operand; the implicit solve will be wrong for this term.
    ```

    They come from the *global* matrix assembly, which this problem does not use — the actual stepping
    goes through the per-subproblem solve path, and the result is correct (the `Δ(A) = -ω` solve
    reproduces the exact Taylor-Green velocity, as above). The warnings are noise here; they are not a
    sign that the run is wrong.

!!! note "Adaptive timestep"
    The `CFL` controller takes a **`VectorField`** velocity (`add_velocity!(cfl, u)`). This formulation
    carries the velocity as three constraint-solved `ScalarField`s, so it uses a fixed `dt` chosen from
    the initial velocity: `dt ≲ safety · Δx / max|u|` (here `Δx = 2π/32 ≈ 0.2` and `max|u| = 1`).

## Initial Conditions

### Taylor-Green Vortex

The classic test case. The velocity

```math
\mathbf{u} = (\sin x \cos y \cos z,\; -\cos x \sin y \cos z,\; 0)
```

has the vorticity $\boldsymbol{\omega} = \nabla\times\mathbf{u}$ we initialize below (the solver
reconstructs $\mathbf{A}$ and $\mathbf{u}$ from it on the first step):

```julia
x, y, z = local_grids(dist, x_basis, y_basis, z_basis)
Z = reshape(z, 1, 1, :)          # broadcast the third axis

for w in (w1, w2, w3)
    ensure_layout!(w, :g)
end
get_grid_data(w1) .= -cos.(x) .* sin.(y') .* sin.(Z)
get_grid_data(w2) .= -sin.(x) .* cos.(y') .* sin.(Z)
get_grid_data(w3) .=  2 .* sin.(x) .* sin.(y') .* cos.(Z)
for w in (w1, w2, w3)
    ensure_layout!(w, :c)
end
```

`local_grids` returns each rank's **local** slice of the coordinate axes, so the same code is correct
in serial and under MPI. (`set!(field, ::Function)` builds the *global* meshgrid and therefore throws a
`DimensionMismatch` when the field is distributed.)

### Random Initial Conditions

For developed turbulence, seed the vorticity with noise instead — `fill_random!` is distributed-safe
and reproducible — and **band-limit it**:

```julia
for w in (w1, w2, w3)
    fill_random!(w, "g"; seed=42, distribution="normal", scale=1e-1)
    low_pass_filter!(w; scales=(0.25, 0.25, 0.25))     # keep |k| ≤ N/8
end
```

The low-pass filter matters. Grid noise puts energy right up to the Nyquist mode, where the spectral
derivative is not invertible, and the reconstructed velocity comes out only approximately
divergence-free (`max|∇·u| ≈ 5e-2` after 20 steps at `16^3`). With the noise band-limited, the same run
gives `max|∇·u| ≈ 3e-13`.

## Analysis and Output

### Diagnostics

Kinetic energy, enstrophy and dissipation are box averages. `global_mean` reduces the local slab and
performs the MPI reduction itself — never wrap it in another `MPI.Allreduce`:

```julia
function kinetic_energy()
    for f in (u1, u2, u3)
        ensure_layout!(f, :g)
    end
    0.5 * global_mean(dist, get_grid_data(u1).^2 .+ get_grid_data(u2).^2 .+ get_grid_data(u3).^2)
end

function enstrophy()
    for f in (w1, w2, w3)
        ensure_layout!(f, :g)
    end
    0.5 * global_mean(dist, get_grid_data(w1).^2 .+ get_grid_data(w2).^2 .+ get_grid_data(w3).^2)
end

dissipation() = 2 * nu * enstrophy()     # ε = 2νΩ
```

For the Taylor-Green initial condition these give `Ω = 0.375` and `ε = 7.5e-3` at `Re = 100` as soon as the
vorticity is set, and `KE = 0.125` **once the solver has taken a step**.

!!! note "`kinetic_energy()` reads 0.0 before the first step"
    Only the vorticity is initialized; `u1, u2, u3` are constraint variables that the solver fills in
    during the first step. Call `kinetic_energy()` straight after the initial condition and you get
    exactly `0.0`, not `0.125`. After one step it is `0.124985`.

!!! warning "Reading the velocity back"
    The velocity is a *substitution constraint* (`u = ∇ × A`), and there are two situations in which the
    copy you read out of `u1, u2, u3` is **corrupted**:

    - **In serial, on any iteration in which an output handler writes.** Right after a write,
      `max|u1| = 0.664` where the true value is `0.996`, `max|u2| = 0.332` (true `0.997`) and
      `max|u3| = 0.333` (true `0.013`). This is *not* a uniform rescale — the components come out
      differently wrong, so do not try to correct it by multiplying by `1.5`. The **next step restores
      them** (`max|u1|` is back to `0.996` one iteration later), so the damage is confined to the
      writing iteration. It bites in two places: a callback that fires on the same iteration as a
      write, and any inspection *after* `run!` returns — the final iteration is normally a write, so
      the velocity you find sitting in `u1, u2, u3` at the end of a run is the corrupted copy.
    - **Under MPI, on every step**, handler or no handler (measured identically at np = 2, 4 and 8:
      `max|u1| = 0.6658` against a true `0.9987`).

    What is *not* affected is the physics. The time-stepped vorticity is bit-identical at np = 1, 2 and 4
    (`max|w1| = 0.998933308710438` at all three), and the enstrophy after 20 steps agrees to ~15 digits
    (`0.374138356478886`; the MPI reduction can differ in the last ulp because it sums in a different
    order) whether a handler writes every iteration or not — the solver's internal RHS uses a correct
    velocity. So `enstrophy()` and `dissipation()` are trustworthy everywhere. Take `kinetic_energy()` and
    the velocity spectrum below from a serial run on a non-writing iteration; otherwise diagnose with the
    vorticity, or reconstruct the velocity from saved vorticity snapshots in post-processing.

### Energy Spectrum

`power_spectrum` requires every axis to be Fourier (true here) and returns a NamedTuple
`(k, power, bin_counts, bin_edges)` with shell-averaged power at the bin centres `k`:

```julia
function velocity_spectrum()
    s1 = power_spectrum(u1)
    s2 = power_spectrum(u2)
    s3 = power_spectrum(u3)
    return (k = s1.k, E = 0.5 .* (s1.power .+ s2.power .+ s3.power))
end
```

`enstrophy_spectrum` and `energy_spectrum` are also available, but they take a `VectorField`.

## Time Integration

`run!` drives the loop: it steps the solver, writes any handler created with the solver, fires
callbacks at their intervals, and closes the handlers at the end. An `Int` callback interval fires every
N iterations; a `Float64` interval fires every Δt of simulation time.

```julia
function report(s)
    @root_only @printf("t = %.3f  KE = %.6e  Ω = %.6e  ε = %.6e\n",
                       s.sim_time, kinetic_energy(), enstrophy(), dissipation())
end

@root_only @printf("3D Taylor-Green: Re = %.0f, N = %d^3\n", Re, N)

run!(solver; stop_iteration=100, callbacks=[20 => report], progress=false)

MPI.Finalize()
```

Output of the run above (serial, `32^3`, `RK443`, `dt = 2e-3`):

```
3D Taylor-Green: Re = 100, N = 32^3
t = 0.040  KE = 1.247024e-01  Ω = 3.741384e-01  ε = 7.482767e-03
t = 0.080  KE = 1.244096e-01  Ω = 3.733526e-01  ε = 7.467051e-03
t = 0.120  KE = 1.241216e-01  Ω = 3.726414e-01  ε = 7.452828e-03
t = 0.160  KE = 1.238381e-01  Ω = 3.720037e-01  ε = 7.440074e-03
t = 0.200  KE = 1.235593e-01  Ω = 3.714385e-01  ε = 7.428770e-03
```

The energy decays at the viscous rate while the vortex tubes are still smooth; run longer (and at
higher `N`) to see the enstrophy rise as the Taylor-Green vortex breaks down.

## Saving Snapshots

`add_file_handler(path, solver; ...)` registers the handler **with the solver**, so `run!` writes it
automatically — declare it *before* `run!`, in place of the bare `run!` above.

!!! warning "Pass field objects to `add_task!`, not strings"
    `add_task!(handler, w1; name="w1")` — a field object — is the only form that reliably works. Strings
    behave three different ways: a bare field name (`"w1"`) is resolved and written correctly; a compound
    expression (`"w1*w1"`, `"w1 + w1"`) is **silently written as a zero scalar** — not even the right
    shape, you get a `(n_writes,)` array of `0.0` instead of a field; and a derivative expression
    (`"∂x(w1)"`) throws a `MethodError` outright.

```julia
snapshots = add_file_handler("turb3d/turb3d", solver; sim_dt=0.1, max_writes=10)
add_task!(snapshots, w1; name="w1")
add_task!(snapshots, w2; name="w2")
add_task!(snapshots, w3; name="w3")

run!(solver; stop_iteration=100, callbacks=[20 => report], progress=false)
```

The output is NetCDF-4 with the data in groups (`vars`, `time`, `grids`), one file per rank under MPI:

```julia
f = "turb3d/turb3d_s1/turb3d_s1.nc"
Tarang.group_variable_names(f, "vars")       # ["w1", "w2", "w3"]
Tarang.group_ncread(f, "vars", "w3")         # (write, x, y, z)
Tarang.group_ncread(f, "time", "sim_time")   # (write,)
```

!!! warning "Snapshot the vorticity, not the velocity"
    `add_task!(snapshots, u1)` writes the corrupted velocity described under
    [Diagnostics](#Diagnostics) — the constraint-solved velocity is wrong precisely on the iterations a
    handler writes, which is exactly when it gets sampled. (Merely *attaching* the handler is enough:
    the corruption happens on write even when the only task is `w1`.) The time-stepped vorticity is
    written correctly — `vars/w3` peaks at the exact `2.0`. Snapshot `w1, w2, w3` and reconstruct the
    velocity in post-processing.

## Performance Considerations

### Resolution Requirements

| Reynolds Number | Minimum Grid | Recommended Grid |
|----------------|--------------|------------------|
| Re ~ 100       | 32³          | 64³              |
| Re ~ 1000      | 64³          | 128³             |
| Re ~ 10000     | 128³         | 256³             |
| Re ~ 100000    | 256³         | 512³+            |

### MPI Scaling

A 3D domain is decomposed with a 2D process mesh (see the warning above):

| Processes | Mesh       | Notes |
|-----------|------------|-------|
| 8         | (2, 4)     | Basic |
| 64        | (8, 8)     | Good scaling |
| 512       | (16, 32)   | Large scale |

Omitting `mesh=` lets the distributor choose one of these shapes for you.

### Memory Usage

Estimate: ~100 bytes per grid point (double precision, velocity + vorticity + vector potential + work
arrays)

```julia
N_est = 256
bytes_per_point = 100
total_memory = N_est^3 * bytes_per_point / 1e9  # GB
println("Estimated memory: $(total_memory) GB")   # Estimated memory: 1.6777216 GB
```

## Running the Simulation

```bash
# Set environment
export OMP_NUM_THREADS=1

# Run with MPI (8 ranks: a 2x4 pencil mesh)
mpiexec -n 8 julia --project turbulence_3d.jl
```

The script above runs unchanged in serial and under MPI. The time-stepped vorticity — and therefore the
enstrophy and dissipation — agrees to roundoff across rank counts: after 20 steps `max|w1|` is
`0.998933308710438` at 1, 2 and 4 ranks and `0.998933308710437` at 8, and the enstrophy agrees to ~15
digits. Do not expect *bit*-identical results at every rank count — the distributed transforms and
reductions reassociate the arithmetic, so the last ulp can move. (Reading the velocity back distributed
hits the corruption noted under [Diagnostics](#Diagnostics).)

## Visualization

There is no VTK/ParaView writer in Tarang. Read a slice back out of the NetCDF snapshots and hand it to
the plotting package of your choice (Plots.jl, Makie.jl — neither is a dependency of Tarang):

```julia
f  = "turb3d/turb3d_s1/turb3d_s1.nc"
w3 = Tarang.group_ncread(f, "vars", "w3")    # (write, x, y, z)
xg = Tarang.group_ncread(f, "grids", "x")
yg = Tarang.group_ncread(f, "grids", "y")

slice = w3[end, :, :, 1]                     # last write, z = z[1] plane
# heatmap(xg, yg, slice')
```

Under MPI each rank writes its own file (`turb3d_s1_p0.nc`, `turb3d_s1_p1.nc`, …) containing its slab.

## References

1. Taylor, G. I., & Green, A. E. (1937). Mechanism of the production of small eddies from large ones.
2. Pope, S. B. (2000). Turbulent Flows. Cambridge University Press.
3. Canuto, C., et al. (2007). Spectral Methods: Fundamentals in Single Domains.
