# The Tau Method for Boundary Conditions

Tarang enforces boundary conditions using the **tau method**, a spectral technique that lets you solve PDEs with non-periodic boundary conditions without modifying the spectral basis. This page explains what the tau method is, how Tarang's implementation works, and how to write code that uses it correctly.

If you have used [Dedalus](https://dedalus-project.readthedocs.io/), the approach here is almost identical: tau fields are added to the state vector, and `lift()` operators inject them into the equations as extra degrees of freedom to match boundary conditions.

## Why Do We Need the Tau Method?

Consider the steady diffusion problem:

```math
\frac{d^2 u}{dz^2} = f(z), \quad z \in [-1, 1], \quad u(-1) = u(+1) = 0.
```

In a Chebyshev spectral method, `u` is expanded as

```math
u(z) = \sum_{n=0}^{N-1} a_n\, T_n(z),
```

and the PDE is projected onto the basis to give an `N × N` linear system in the coefficients ``a_n``. The problem: **this system has no leftover degrees of freedom to enforce the boundary conditions**. All ``N`` unknowns are already determined by the interior equations, but we also need two boundary constraints.

The tau method solves this by **adding extra unknowns** (the *tau fields*) that act as corrections designed to make the boundary conditions hold. The classical formulation, due to Lanczos (1938), replaces the highest-order rows of the spectral system with BC rows — but that makes the interior equation different at those rows, which is awkward for nonlinear and time-dependent problems.

Modern spectral codes (Dedalus, Tarang) use a cleaner variant: instead of *replacing* rows, they *add* a tau term to the equation itself:

```math
\mathcal{L}[u] \;+\; \tau_1\,\phi_1(z) \;+\; \tau_2\,\phi_2(z) \;=\; f(z),
```

where ``\phi_1, \phi_2`` are chosen basis functions that live in a "lift basis". The tau scalars ``\tau_1, \tau_2`` become new unknowns in the system, and the boundary conditions become new equation rows that link them to the state. The interior of the PDE is untouched; the tau corrections only perturb the equation at a small number of high spectral modes, and for smooth solutions the required ``\tau_k`` are spectrally small.

That is exactly how Tarang implements BCs.

## Quick-Start Example

Here's the smallest complete example — a 1D steady heat equation with homogeneous Dirichlet BCs:

```julia
using Tarang

coords = CartesianCoordinates("z")
dist   = Distributor(coords; dtype=Float64)

zbasis = ChebyshevT(coords["z"]; size=64, bounds=(-1.0, 1.0))
lift_basis = derivative_basis(zbasis)   # ChebyshevU of size 64

# State field
u = ScalarField(dist, "u", (zbasis,))

# Tau fields — one per boundary. They have NO bases here (the problem is
# 1D and the coupled direction is z, so the tau drops z and has nothing left).
tau_u1 = ScalarField(dist, "tau_u1", ())
tau_u2 = ScalarField(dist, "tau_u2", ())

problem = LBVP([u, tau_u1, tau_u2])

# The equation carries its own tau corrections via lift():
add_equation!(problem,
    "∂²(u)/∂z² + lift(tau_u1, -1) + lift(tau_u2, -2) = f")

add_bc!(problem, "u(z=-1) = 0")
add_bc!(problem, "u(z=1)  = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
```

Notice that:

1. Tau fields (`tau_u1`, `tau_u2`) are **added to the state vector**, not computed from the state afterwards.
2. The equation itself carries `lift(tau, -1)` and `lift(tau, -2)` terms.
3. The BCs are plain-looking equations (`u(z=-1) = 0`) — Tarang converts each into an algebraic row that constrains the tau fields.

The solver determines `u`, `tau_u1`, and `tau_u2` **simultaneously** in a single linear solve.

## The `lift` Operator

`lift` is how a tau field enters an equation:

```julia
lift(tau, basis, n)   # full form:  explicit lift basis
lift(tau, n)          # short form: auto-detects the lift basis
```

- **`tau`** is the tau field (a `ScalarField` or `VectorField` whose bases are a strict subset of the state field's bases — it's missing the coupled direction).
- **`basis`** is the *lift basis*. For Chebyshev-T state fields, this is almost always `derivative_basis(zbasis)`, which returns ChebyshevU of the same size. See [Why the Derivative Basis?](#why-the-derivative-basis) below.
- **`n`** is an integer mode index. `n = -1` is the last mode of `basis`, `n = -2` is the second-to-last, and so on. For ChebyshevU of size `N`, `-1` picks `U_{N-1}` and `-2` picks `U_{N-2}`.

The short form `lift(tau, n)` inspects `tau.dist.layouts` to find a non-periodic basis that the tau field is missing compared to the full state, and uses that as the lift basis. This works in most situations, but being explicit is always safe.

### What `lift(tau, basis, n)` actually computes (solver view)

At the solver level, `lift(tau, basis, n)` resolves to a **single-column sparse matrix** with a `1` at row `lift_mode` (where `lift_mode = n` is wraparound-resolved: `-1` → last, `-2` → second-to-last, etc.). This column is a discrete delta in the coupled direction's coefficient space.

```math
\text{lift}(\tau, \cdot, n) \;\longrightarrow\; \tau \cdot e_{\text{lift\_mode}}
```

where ``e_{\text{lift\_mode}}`` is the unit vector with a `1` at the `lift_mode`-th Chebyshev-coefficient slot. When this is added to an equation's LHS, it contributes the unknown ``\tau`` to exactly one coefficient row, leaving every other interior row untouched. The linear solver then chooses ``\tau`` so that the BC rows hold — and because the perturbation is confined to one high-order coefficient, the interior PDE residual stays spectrally small for smooth solutions.

> **Implementation detail worth knowing**: In Tarang's current solver, `subproblem_matrix(op::Lift, sp)` reads the dimension `N` from `_subproblem_cheb_basis(sp)` — the problem's own Chebyshev basis — and does **not** use `op.basis` to compute the matrix. That means `lift(tau, zbasis, -1)` and `lift(tau, derivative_basis(zbasis), -1)` produce **identical** delta columns at row `N-1`. The `basis` argument is a semantic hint inherited from Dedalus's type system; it's retained so that future refinements (e.g. different lift-basis behaviour for non-Jacobi bases, or explicit basis tracking through expression trees) can hook in without breaking user code.

### Why still pass `derivative_basis(zbasis)`?

Even though the two choices produce identical matrices in the current solver, **writing `lift(tau, derivative_basis(zbasis), -1)` is the recommended idiom**, for three reasons:

1. **Forward-compatibility**: if Tarang later adopts Dedalus's fully basis-tracked lift semantics (where `op.basis` *does* determine the output coefficient space), your code will already be correct. Passing `zbasis` would silently start producing different matrices.
2. **Semantic clarity**: in first-order formulations like `grad_u = grad(u) + ez * lift(tau_u1, -1)`, the lift lives alongside `grad(u)`, which naturally maps a ChebyshevT field into the derivative (ChebyshevU) space. Passing `derivative_basis(zbasis)` tells the reader what the lift is conceptually contributing, even if the linear algebra doesn't care.
3. **Consistency with examples and tutorials**: all of Tarang's shipped examples use `derivative_basis(zbasis)`, so sticking with it makes your code pattern-match familiar code.

**Rule of thumb**: always write `lift_basis = derivative_basis(zbasis)` at the top of your setup block and use `lift(tau, lift_basis, -1)` (or the auto-form `lift(tau, -1)`) throughout. Don't invent your own lift basis.

## Boundary Conditions as Algebraic Constraint Rows

When you write

```julia
add_bc!(problem, "u(z=-1) = 0")
```

Tarang produces a new equation row that looks like

```math
\sum_{n=0}^{N-1} a_n\, T_n(-1) \;=\; 0,
```

i.e. a linear combination of `u`'s coefficients that equals the boundary value. **This row is added to the system, not substituted in.** It's an *algebraic* equation — there is no time derivative, so it contributes nothing to the `M` matrix in the `M·dX/dt + L·X = F` formulation. In the linear-algebra picture:

- `u` contributes `Nz` rows (PDE interior) + BC rows
- `tau_u1`, `tau_u2` contribute columns (one each, at the lift modes)
- BC rows are zero in `M` (no time derivative) → they're pure algebraic constraints
- The combined LHS matrix is square because `(PDE rows + BC rows) = (state cols + tau cols)`

This square-system condition is enforced automatically when you declare the right number of tau fields to match the number of BCs. If you miss a BC you'll get a singular / non-square system at solver-build time.

### DAE-style handling in IVP steppers

For **initial-value problems**, BC rows have `M_row = 0`, which makes the full system a **differential-algebraic equation** (DAE) rather than a pure ODE. Tarang's subproblem stepper handles this correctly via a **per-stage row override** on what it classifies as "BC rows".

A note on how this classification works: `sp.bc_rows` is not built by scanning `M` for zero rows directly. Instead, `build_matrices!` labels rows by **equation size** — any equation whose per-subproblem row count is smaller than the coupled-direction basis size `Nz` is classified as a BC row (pressure gauge `integ(p) = 0` → 1 row; wall BC `T(z=0) = 1` → 1 row; vector wall BC → 2 rows; etc.). Rows from equations with `eq_size ≥ Nz` — the PDE interior and any *Nz-sized algebraic* equation like the continuity equation — are classified as "bulk".

At each IMEX-RK stage (or multistep update), the solver assembles the stage RHS from the accumulated `dt·Σ(AᴱF − Aⁱ·L·X)` formula for **every** row, then calls `apply_bc_override!` to **replace** the `sp.bc_rows` entries with `dt·a_ii·F_BC`. After the LHS solve, those rows yield exactly `L_row·X = F_BC`, which is the correct BC enforcement.

**Why a size-based classifier is good enough**: the only algebraic rows that don't get overridden are Nz-sized, F-zero equations like continuity. For them, the accumulated-RHS formula naturally produces zero on the right-hand side (since their `F` is zero and `L*X = 0` is the target), so no override is needed. The override only matters for *small* algebraic rows whose `F` can be nonzero — i.e., BCs and gauge constraints — and those are exactly what `sp.bc_rows` captures.

**Why the override is necessary**: without it, the raw accumulated-RHS formula produces a wrong `1/γ` scaling factor for inhomogeneous algebraic rows. For RK222 (γ = 1 - 1/√2, 1/γ = 2 + √2 ≈ 3.414), a BC like `T(z=0) = 1` would be enforced as `T(z=0) = 2+√2` instead of `1`. The override is built into `step_subproblem_rk!` and `step_subproblem_multistep!`, so you don't have to do anything to enable it — but it's worth knowing the mechanism exists if you're debugging a "BC value is off by a constant factor" symptom.

## First-Order Formulation (Recommended)

For time-dependent problems and anything involving the Navier–Stokes equations, the **first-order formulation** is the right default. Instead of writing a 2nd-order operator like `Δ(u)` directly, introduce an auxiliary gradient field with a tau correction:

```julia
grad_u = grad(u) + ez * lift(tau_u1, -1)
```

Now `grad_u` is an "augmented gradient" that lives in the derivative (ChebyshevU) space and carries its own tau. Second derivatives become `div(grad_u)` rather than `Δ(u)`:

```julia
add_equation!(problem, "∂t(u) - ν*div(grad_u) + ∇(p) + lift(tau_u2, -1) = -u⋅∇(u)")
```

Why this is better than the 2nd-order form:

| Aspect | 2nd-order (`Δ(u)`) | First-order (`div(grad_u)`) |
|---|---|---|
| Tau fields | 1 per equation | 2 per equation (one in grad, one in ∂t) |
| Differentiation matrices | Powers of `D` (ill-conditioned) | Products of first-order `D` (better) |
| Boundary enforcement | Via a single lift mode | Split: one tau in grad, one in the evolution |
| High-resolution behavior | Conditioning degrades as `N²` | Conditioning degrades as `N` |
| Recommended | Prototyping / low `N` | Production / any `N > 64` |

The convention we use throughout Tarang's examples is:

- `tau_u1` — correction added inside `grad_u`, one-dimensional (xbasis only)
- `tau_u2` — correction added to the evolution equation directly, one-dimensional (xbasis only)

For a vector field `u`, both `tau_u1` and `tau_u2` are VectorFields too (one component per velocity component).

## Worked Example: 2D Rayleigh–Bénard Convection

Here is the full first-order RBC setup, which is the `examples/ivp/rayleigh_benard_2d.jl` example in the repository. This is the **canonical pattern** to copy for any 2D channel-flow problem with non-periodic BCs.

```julia
using Tarang

# Domain
Lx, Lz = 4.0, 1.0
Nx, Nz = 256, 64
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=Nx, bounds=(0.0, Lx), dealias=3/2)
zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz), dealias=3/2)

domain = Domain(dist, (xbasis, zbasis))

# State variables: pressure, temperature, velocity
p = ScalarField(domain, "p")
T = ScalarField(domain, "T")
u = VectorField(domain, "u")

# Tau fields. Each lives on `(xbasis,)` only — the coupled direction (z)
# is dropped because the tau correction lives at a single z mode.
#
# tau_p:  scalar, no bases — gauge for the pressure constraint
# tau_T1: gradient-substitution correction for T
# tau_T2: evolution-equation correction for T
# tau_u1, tau_u2: ditto for each velocity component (so they are VectorFields)
tau_p  = ScalarField(dist, "tau_p",  (),         Float64)
tau_T1 = ScalarField(dist, "tau_T1", (xbasis,),  Float64)
tau_T2 = ScalarField(dist, "tau_T2", (xbasis,),  Float64)
tau_u1 = VectorField(dist, coords, "tau_u1", (xbasis,), Float64)
tau_u2 = VectorField(dist, coords, "tau_u2", (xbasis,), Float64)

# First-order substitutions
ex, ez = unit_vector_fields(coords, dist)
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A) = lift(A, lift_basis, -1)

grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_T = grad(T) + ez * τ_lift(tau_T1)

# Problem declaration includes ALL tau fields as state
problem = IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2])

add_parameters!(problem,
    nu=Pr, buoy=Ra*Pr, ez=ez,
    grad_u=grad_u, grad_T=grad_T, τ_lift=τ_lift)

# Equations
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_equation!(problem, "∂t(T) - div(grad_T) + τ_lift(tau_T2) = -u⋅∇(T)")
add_equation!(problem,
    "∂t(u) - nu*div(grad_u) + ∇(p) - buoy*T*ez + τ_lift(tau_u2) = -u⋅∇(u)")

# Boundary conditions (one per algebraic constraint row)
add_bc!(problem, "T(z=0)  = 1")       # hot bottom wall
add_bc!(problem, "T(z=Lz) = 0")       # cold top wall
add_bc!(problem, "u(z=0)  = 0")       # no-slip
add_bc!(problem, "u(z=Lz) = 0")       # no-slip
add_bc!(problem, "integ(p) = 0")      # pressure gauge

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
```

**Pattern summary**:

- **5 BCs** (2 for `T`, 2 for `u`, 1 gauge for `p`) → **5 tau scalars per Fourier mode** (`tau_T1`, `tau_T2`, two components each of `tau_u1` and `tau_u2`, and `tau_p`).
- **Each tau field drops the coupled direction** (`xbasis` only, not `zbasis`).
- **`tau_p` is a 0-D scalar** (no bases) — it's a single number per Fourier mode, used as a gauge constraint, not a per-x correction.
- **The pressure gauge `integ(p) = 0`** is an algebraic constraint on the mean; it lives alongside the other BCs.

## Pressure Gauge and Valid-Mode Filtering

In incompressible flow the pressure is only defined up to a constant, so continuity `trace(grad_u) = 0` is **rank-deficient** by one at every Fourier mode. Tarang handles this with two different mechanisms that are sometimes confused:

**1. Pressure gauge (a user-visible tau)**. You add `tau_p` to continuity and provide a gauge-fixing BC like `integ(p) = 0`:

```julia
add_equation!(problem, "trace(grad_u) + tau_p = 0")
add_bc!(problem, "integ(p) = 0")
```

`tau_p` is a single scalar per Fourier mode (or a single scalar total, depending on how you declare it). It absorbs the pressure-gauge degree of freedom, and the `integ(p) = 0` row fixes its value.

**2. Valid-mode filtering (invisible, done by the solver)**. At the `(kx=0, ky=0, ...)` DC Fourier mode, the integral constraint `integ(p) = 0` produces a **zero row** in the raw LHS matrix (because `integ` of a non-DC Fourier mode is zero, and the DC mode's integ is well-defined). Similarly, `trace(grad_u) = 0` at DC is automatically satisfied by the no-slip BCs, producing another zero row.

The solver detects these zero rows during matrix assembly (`subsystems.jl`'s `build_matrices!`) and pairs each with the least-used 0-D tau column, dropping both the row and the column. The result is a smaller square system that the sparse LU can factorize cleanly.

**You do not have to do anything to enable this.** Just declare `tau_p` and the `integ(p) = 0` BC, and the valid-mode filter takes care of the rest. The take-away is: if you ever see a `"129 singular pencils"` or `"non-square filtered system"` warning, it usually means a BC is missing, a tau field is missing, or the eq_size and state_size don't match.

## Number of Tau Terms

The number of tau terms must match the total number of boundary conditions (including gauge conditions):

| PDE order in coupled direction | BCs needed | Tau terms per equation |
|---|---|---|
| 1st (∂u/∂z) | 1 | 1 |
| 2nd (∂²u/∂z², via first-order form) | 2 | 2 (one in grad, one in evolution) |
| 4th (biharmonic) | 4 | 4 |

**In a 2D problem (x-Fourier, z-coupled)** each equation carries tau corrections only in the `z` direction — **not "2 per direction"**. The Fourier `x` direction is periodic and needs no tau terms. Only the coupled `z` direction contributes.

**Vector equations need vector tau fields.** In RBC, `u` is a 2-component velocity, so `tau_u1` and `tau_u2` are `VectorField`s (one scalar per component per Fourier mode).

## Time- and Space-Dependent BCs

Tarang supports boundary conditions whose value varies in time, space, or both. They are refreshed by the stepper:

- **Time-dependent BCs** (e.g. `T(z=0) = sin(t)`): the RK stepper re-evaluates the BC value at each stage time `t + c[i]*dt`, so multi-stage methods retain their full formal order of accuracy for rapidly-varying BCs.
- **Space-dependent BCs** (e.g. `T(z=0) = sin(2*pi*x/Lx)`): at solver-build time the BC expression is evaluated on the **global** coordinate grid, and the resulting array is projected onto the Fourier modes via an unnormalized `FFTW.rfft`. Each subproblem picks its own mode from the cached coefficient array.
- **Space+time BCs** (e.g. `T(z=0) = sin(2*pi*x/Lx) * cos(2*pi*t)`): combined — re-projected on every stage.

You don't need to register coordinate fields manually. The solver auto-registers global grid arrays for every Fourier/Chebyshev axis under its element label (`"x"`, `"y"`, `"z"`, ...), so BC string expressions can reference those coordinates directly:

```julia
add_bc!(problem, "T(z=0) = 1 + 0.1*sin(2*pi*x/Lx)")
```

Under MPI, every rank evaluates the BC expression on the full (global) grid and computes a local FFT — no inter-rank communication is needed because all ranks produce identical coefficient arrays.

## Inspecting BC Satisfaction

After a solve, you can verify that the BCs are actually being enforced:

```julia
# After solve!()
ensure_layout!(u, :g)
u_data = get_grid_data(u)

# For Chebyshev-T, the Gauss-Lobatto grid includes the endpoints.
# u_data[:, 1]    is the solution at z = z_min (first Cheb grid point)
# u_data[:, end]  is the solution at z = z_max (last Cheb grid point)
@assert maximum(abs.(u_data[:, 1]))   < 1e-10  "Bottom BC not satisfied"
@assert maximum(abs.(u_data[:, end])) < 1e-10  "Top BC not satisfied"

# The tau field values tell you how large the correction was. For smooth
# solutions these should be "spectrally small" — decaying with N.
tau_data = get_coeff_data(tau_u2)
@info "tau_u2 magnitude: $(maximum(abs.(tau_data)))"
```

For time-dependent problems, check BC satisfaction inside a callback:

```julia
run!(solver;
     stop_time=25.0,
     callbacks=[on_interval(10) do s
         ensure_layout!(T, :g)
         T_data = get_grid_data(T)
         @info "T(z=0) residual: $(maximum(abs.(T_data[:, 1] .- 1.0)))"
     end])
```

## Common Pitfalls

### 1. Wrong number of tau fields

If you have two wall BCs on `u` but only declare one tau field in the evolution equation, the matrix system is under-determined and you'll see `singular pencils` or `non-square filtered system` warnings at solver build time.

**Fix**: count the BCs (including gauge conditions like `integ(p) = 0`), and declare one tau field (or tau component for vector fields) per BC.

### 2. Missing `lift()` term in the equation

A tau field declared in `problem.variables` but never referenced in any equation contributes a zero column to the LHS matrix, making the system rank-deficient.

**Fix**: every tau field must appear inside exactly one `lift()` (or `τ_lift()` substitution) somewhere in the equation RHS.

### 3. Wrong lift basis

Using `lift(tau, zbasis, -1)` instead of `lift(tau, derivative_basis(zbasis), -1)` works mathematically but conditions the system poorly at high resolution. Symptoms: solutions look correct at `Nz=32` but diverge or develop noise at `Nz=128`.

**Fix**: always use `derivative_basis(zbasis)` (or the short form `lift(tau, -1)` which auto-selects it).

### 4. Tau field bases don't match

A tau field must live on the *complement* of the state field's bases — the state's bases **minus the coupled direction**. For a 2D state on `(xbasis, zbasis)`, the tau lives on `(xbasis,)`; for a 0-D gauge like `tau_p` it's `()`.

If you accidentally declare `tau_u1 = ScalarField(dist, "tau_u1", (xbasis, zbasis))` (same bases as `u`), `lift(tau_u1, -1)` will have nothing to lift into and will throw a construction error.

**Fix**: drop the Chebyshev axis when declaring tau fields.

### 5. Non-square system at DC mode

Most non-square warnings at the DC Fourier mode come from a missing gauge BC like `integ(p) = 0`. The valid-mode filter can only drop a zero row if there's a paired 0-D tau column (like `tau_p`) to drop along with it.

**Fix**: make sure every PDE with a gauge ambiguity has both a `tau_*` field and a corresponding `integ()` BC.

### 6. BC F value not reaching the stepper

For inhomogeneous BCs like `T(z=0) = 1`, the stepper has to carry the value `1` through the stage RHS assembly. This happens automatically through `gather_alg_F!` + the `apply_bc_override!` path in `step_subproblem_rk!`, but only if the BC equation has a non-zero `F` expression in `equation_data[eq_idx]["F"]`.

**Symptom of the bug**: `max|T|` decays to zero over time even though `T(z=0) = 1` is declared, OR `max|T|` sticks at `1/γ = 2+√2 ≈ 3.414` (the classical 1/γ scaling factor from RK222's implicit coefficient).

**If you see this**: confirm you're on a recent version — this was a regression fixed after the subproblem-architecture rewrite. The `apply_bc_override!` override is what enforces `L_row·X = F_BC` at every stage regardless of accumulation history.

## Historical Note

The tau method was introduced by **Cornelius Lanczos** in 1938 as an approximation technique: rather than solving a PDE exactly, he sought polynomial approximations that satisfied the PDE with a small residual (the "tau error"). It was refined for spectral methods by **Steven Orszag**, **David Gottlieb**, and others in the 1970s–80s, and adapted to modern lift-based formulations by **Keaton Burns et al.** in the Dedalus project in the 2010s.

The name "tau" (τ) comes from Lanczos's notation for the residual/correction terms introduced when truncating the polynomial expansion and enforcing boundary conditions.

## References

### Textbooks

1. **Canuto, C., Hussaini, M. Y., Quarteroni, A., & Zang, T. A.** (2006). *Spectral Methods: Fundamentals in Single Domains*. Springer. — Rigorous treatment of tau and Galerkin methods.

2. **Boyd, J. P.** (2001). *Chebyshev and Fourier Spectral Methods* (2nd ed.). Dover. — Very readable; freely available online.

3. **Trefethen, L. N.** (2000). *Spectral Methods in MATLAB*. SIAM. — Practical, code-oriented.

4. **Peyret, R.** (2002). *Spectral Methods for Incompressible Viscous Flow*. Springer. — Detailed treatment of Navier–Stokes with spectral methods.

### Key papers

5. **Lanczos, C.** (1938). "Trigonometric interpolation of empirical and analytical functions." *Journal of Mathematics and Physics*, 17(1–4), 123–199. — Original tau method paper.

6. **Burns, K. J., Vasil, G. M., Oishi, J. S., Lecoanet, D., & Brown, B. P.** (2020). "Dedalus: A flexible framework for numerical simulations with spectral methods." *Physical Review Research*, 2, 023068. — Modern lift-based tau method as implemented in Tarang and Dedalus.

7. **Orszag, S. A.** (1971). "Accurate solution of the Orr-Sommerfeld stability equation." *Journal of Fluid Mechanics*, 50(4), 689–703. — Classic application to hydrodynamic stability.

## See Also

- [Boundary Conditions Tutorial](../tutorials/boundary_conditions.md): step-by-step BC examples
- [Bases](bases.md): spectral bases (Chebyshev, Fourier, Legendre, Jacobi)
- [Solvers](solvers.md): using IVP / LBVP / NLBVP solvers
- [API: Problems](../api/problems.md): programmatic API for adding equations and BCs
- [2D RBC Tutorial](../tutorials/ivp_2d_rbc.md): complete Rayleigh–Bénard convection walkthrough
