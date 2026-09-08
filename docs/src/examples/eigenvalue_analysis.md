# Eigenvalue Analysis Examples

Collection of linear stability and eigenvalue problem examples with Tarang.jl.

## Linear Stability Overview

Eigenvalue problems determine the stability of base states by solving:

```math
A \mathbf{x} = \sigma B \mathbf{x}
```

where σ is the eigenvalue (growth rate + frequency).

In Tarang the eigenvalue **replaces the time derivative**: you write the linearised
equation exactly as you would for an InitialValueProblem, keeping the `dt(·)` term, and the solver
assembles `σ M x + L x = 0`. You never multiply the eigenvalue symbol into the equation
yourself. Every boundary condition is declared with `add_bc!` and paired with a `tau`
variable lifted into the bulk equation.

!!! warning "Fold constant coefficients into a single parameter"
    A coefficient built from *several* constants multiplied together inside the equation
    string — `Pr*Ra*k2*T`, or `2*k2*∂z(∂z(w))` — loses its value when the implicit matrix
    is assembled: the term enters `L` with coefficient **1**. Compute the product in Julia
    and register it as **one** parameter (`buoy = Pr*Ra*k^2`, `twok2 = 2k^2`), then write
    `buoy*T` and `twok2*∂z(∂z(w))`. A single scalar multiplying an expression — `Pr*(…)`,
    `invRe*(…)` — is fine.

## Rayleigh-Bénard Stability

Critical Rayleigh number for convection onset. Eliminating the pressure and the horizontal
velocity leaves a fourth-order system for the vertical velocity `w` and the temperature
perturbation `T` at a single horizontal wavenumber `k`:

```math
\sigma (\partial_z^2 - k^2) w = Pr\left[(\partial_z^2 - k^2)^2 w - Ra\, k^2 T\right],
\qquad \sigma T = w + (\partial_z^2 - k^2) T
```

`w` is fourth order, so it needs four boundary conditions and four `tau` variables; `T` is
second order and needs two. With stress-free walls (`w = ∂z²w = 0`) the onset is known
analytically — `Ra_c(k) = (π² + k²)³ / k²`, minimised at `Ra_c = 657.51` for `k = 2.2214` —
which makes it a good check that the problem is assembled correctly.

```julia
using Tarang
using Printf

function rbc_evp(Ra, k; Pr=1.0, Nz=32, slip=:free)
    coords = CartesianCoordinates("z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))
    domain = Domain(dist, (zbasis,))
    lift_basis = derivative_basis(zbasis, 2)

    w = ScalarField(domain, "w")
    T = ScalarField(domain, "T")
    # tau variables: four for the fourth-order w equation, two for T
    tw = [ScalarField(dist, "tw$i", (), Float64) for i in 1:4]
    tT = [ScalarField(dist, "tT$i", (), Float64) for i in 1:2]

    evp = EigenvalueProblem([w, T, tw..., tT...]; eigenvalue=:σ)
    add_parameters!(evp; Pr=Pr, k2=k^2, k4=k^4, twok2=2k^2, buoy=Pr*Ra*k^2,
                    lw1=lift(tw[1], lift_basis, -1), lw2=lift(tw[2], lift_basis, -2),
                    lw3=lift(tw[3], lift_basis, -3), lw4=lift(tw[4], lift_basis, -4),
                    lT1=lift(tT[1], lift_basis, -1), lT2=lift(tT[2], lift_basis, -2))

    # KEEP dt(·) — it builds the mass matrix M. Never multiply σ into the equation.
    add_equation!(evp, "dt(∂z(∂z(w)) - k2*w) " *
                       "- Pr*(∂z(∂z(∂z(∂z(w)))) - twok2*∂z(∂z(w)) + k4*w) " *
                       "+ buoy*T - lw1 - lw2 - lw3 - lw4 = 0")
    add_equation!(evp, "dt(T) - (∂z(∂z(T)) - k2*T) - w - lT1 - lT2 = 0")

    # boundary conditions — always add_bc!, never add_equation!
    add_bc!(evp, "w(z=0) = 0")
    add_bc!(evp, "w(z=1) = 0")
    if slip == :free                       # stress-free
        add_bc!(evp, "∂z(∂z(w))(z=0) = 0")
        add_bc!(evp, "∂z(∂z(w))(z=1) = 0")
    else                                   # no-slip
        add_bc!(evp, "∂z(w)(z=0) = 0")
        add_bc!(evp, "∂z(w)(z=1) = 0")
    end
    add_bc!(evp, "T(z=0) = 0")
    add_bc!(evp, "T(z=1) = 0")
    return evp
end

function growth_rate(Ra, k; Pr=1.0, Nz=32)
    solver = EigenvalueSolver(rbc_evp(Ra, k; Pr=Pr, Nz=Nz); nev=4, which=:LR)
    λ, _ = solve!(solver)
    return maximum(real, λ)
end

for Ra in (500.0, 800.0)
    @printf("Ra = %5.0f   σ_max = %+.6f\n", Ra, growth_rate(Ra, 2.2214))
end
```

```
Ra =   500   σ_max = -1.894439
Ra =   800   σ_max = +1.525506
```

The sign of `σ_max` brackets the onset, so a bisection gives the critical Rayleigh number:

```julia
function critical_Ra(k; Pr=1.0, Nz=32, Ra_lo=100.0, Ra_hi=5000.0, tol=1e-4)
    while (Ra_hi - Ra_lo) / Ra_hi > tol
        Ra_mid = (Ra_lo + Ra_hi) / 2
        if growth_rate(Ra_mid, k; Pr=Pr, Nz=Nz) > 0
            Ra_hi = Ra_mid
        else
            Ra_lo = Ra_mid
        end
    end
    return (Ra_lo + Ra_hi) / 2
end

@printf("Ra_c = %.2f   (exact: %.2f)\n", critical_Ra(2.2214), (π^2 + 2.2214^2)^3 / 2.2214^2)
```

```
Ra_c = 657.53   (exact: 657.51)
```

The no-slip case (`slip=:noslip`) has no closed form, and its tau system throws off large
spurious eigenvalues that `which=:LR` picks up first — see
[Spurious Modes](#Spurious-Modes) for the filter that makes it usable.

## Orr-Sommerfeld (Plane Poiseuille)

Stability of channel flow with base profile `U(z) = 1 - z²` on `z ∈ [-1, 1]`. Written for
the growth rate `σ` (the complex wave speed is then `c = iσ/k`):

```math
\sigma (\partial_z^2 - k^2)\psi = -ikU(\partial_z^2 - k^2)\psi + ikU''\psi
    + \tfrac{1}{Re}(\partial_z^2 - k^2)^2\psi
```

!!! warning "The base-flow field must be real"
    `U(z)` enters as a spatially varying (non-constant) coefficient, and only a **real**
    coefficient field is representable in the implicit operator. Build the fields with
    `dtype=Float64` and keep the complex factor `-ik` outside, as a plain scalar parameter
    (`mik = -1im*k`). If you fold `-ik` into the field instead — making it complex, or
    declaring the fields `ComplexF64` — the coefficient is silently replaced by 1 and the
    solver returns a converged-looking but wrong spectrum. The assembled matrices are
    complex regardless of the field dtype, so nothing is lost.

```julia
using Tarang
using Printf

# A fourth-order tau system produces spurious eigenvalues. A spurious eigenvalue moves
# when the resolution changes; a physical one does not. See Tips → Spurious Modes.
function converged_eigenvalues(build, N1, N2; nev=25, which=:LR, rtol=1e-5)
    λ1, _ = solve!(EigenvalueSolver(build(N1); nev=nev, which=which))
    λ2, _ = solve!(EigenvalueSolver(build(N2); nev=nev, which=which))
    return [λ for λ in λ1 if minimum(abs.(λ .- λ2)) <= rtol * max(1.0, abs(λ))]
end

function os_evp(Re, k, Nz)
    coords = CartesianCoordinates("z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    zbasis = ChebyshevT(coords["z"]; size=Nz, bounds=(-1.0, 1.0))
    domain = Domain(dist, (zbasis,))
    lift_basis = derivative_basis(zbasis, 2)

    psi = ScalarField(domain, "psi")
    tp  = [ScalarField(dist, "tp$i", (), Float64) for i in 1:4]   # fourth order → 4 taus

    U = ScalarField(domain, "U")            # REAL base flow, U(z) = 1 - z²
    z = local_grid(zbasis, dist, 1)
    ensure_layout!(U, :g)
    get_grid_data(U) .= 1 .- z .^ 2
    ensure_layout!(U, :c)

    evp = EigenvalueProblem([psi, tp...]; eigenvalue=:σ)
    add_parameters!(evp; U=U, k2=k^2, k4=k^4, twok2=2k^2,
                    mik=-1im*k,           # -ik
                    Upp=1im*k*(-2.0),     #  ik U''   (U'' = -2)
                    invRe=1.0/Re,
                    l1=lift(tp[1], lift_basis, -1), l2=lift(tp[2], lift_basis, -2),
                    l3=lift(tp[3], lift_basis, -3), l4=lift(tp[4], lift_basis, -4))

    add_equation!(evp, "dt(∂z(∂z(psi)) - k2*psi) " *
                       "- mik*(U*(∂z(∂z(psi)) - k2*psi)) - Upp*psi " *
                       "- invRe*(∂z(∂z(∂z(∂z(psi)))) - twok2*∂z(∂z(psi)) + k4*psi) " *
                       "- l1 - l2 - l3 - l4 = 0")

    # No-slip: ψ = ∂ψ/∂z = 0 at both walls
    add_bc!(evp, "psi(z=-1) = 0")
    add_bc!(evp, "psi(z=1) = 0")
    add_bc!(evp, "∂z(psi)(z=-1) = 0")
    add_bc!(evp, "∂z(psi)(z=1) = 0")
    return evp
end

Re, k = 10000.0, 1.0
λ = converged_eigenvalues(N -> os_evp(Re, k, N), 64, 80; nev=30)
σ = λ[argmax(real.(λ))]                 # most unstable growth rate
@printf("c = %s\n", string(round(1im * σ / k, digits=8)))
```

```
c = 0.23752649 + 0.00373967im
```

This is Orszag's (1971) benchmark value for `Re = 10000`, `k = 1`, to all eight digits;
`Im(c) > 0`, so the mode grows.

## Neutral Curves

The stability boundary in parameter space is one bisection per wavenumber, reusing
`critical_Ra` from the Rayleigh-Bénard example:

```julia
ks   = [1.5, 2.0, 2.2214, 2.5, 3.0]
Ra_n = [critical_Ra(k) for k in ks]

for (k, Ra) in zip(ks, Ra_n)
    @printf("k = %.4f   Ra_c = %8.2f   (exact %8.2f)\n", k, Ra, (π^2 + k^2)^3 / k^2)
end

i = argmin(Ra_n)
@printf("critical point: k = %.4f, Ra_c = %.2f\n", ks[i], Ra_n[i])
```

```
k = 1.5000   Ra_c =   791.19   (exact   791.19)
k = 2.0000   Ra_c =   667.02   (exact   667.01)
k = 2.2214   Ra_c =   657.53   (exact   657.51)
k = 2.5000   Ra_c =   670.16   (exact   670.17)
k = 3.0000   Ra_c =   746.54   (exact   746.53)
critical point: k = 2.2214, Ra_c = 657.53
```

Refine the `k` grid around the minimum to locate the critical point more precisely.

## Eigenmode Structure

`solve!` returns `(eigenvalues, eigenvectors)`. The eigenvector columns live in the
subproblem's coefficient space — every variable and tau concatenated, not a per-field
dictionary. Write a column back onto the problem's fields with `scatter_inputs`, then
transform to the grid:

```julia
Nz  = 32
evp = rbc_evp(2000.0, 2.2214; Nz=Nz)
solver = EigenvalueSolver(evp; nev=4, which=:LR)
λ, v = solve!(solver)

# most unstable mode → scatter it back onto the problem's variables
mode = v[:, argmax(real.(λ))]
Tarang.scatter_inputs(solver.subproblems[1], mode, evp.variables)

w = evp.variables[1]
ensure_layout!(w, :g)
w_profile = real.(copy(get_grid_data(w)))
w_profile ./= maximum(abs, w_profile)          # normalise

coords = CartesianCoordinates("z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
z      = local_grid(ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0)), dist, 1)

@printf("σ = %.6f\n", maximum(real, λ))
@printf("max | |w| - |sin(πz)| | = %.2e\n",
        maximum(abs, abs.(w_profile) .- abs.(sin.(π .* z))))
```

```
σ = 11.015345
max | |w| - |sin(πz)| | = 3.16e-03
```

The stress-free eigenfunction is `sin(πz)`, recovered here to three digits. `z` and
`w_profile` are plain vectors — plot them with the package of your choice.

## Solver Options

### Finding Specific Eigenvalues

Building a solver merges the boundary conditions into the equation set, which **mutates
the problem**: a given `EigenvalueProblem` can be handed to `EigenvalueSolver` only once. Reusing it
raises `Number of equations (14) does not match number of variables (8)`. Build a fresh
problem for each solver — which is why every example here wraps the problem in a function.

```julia
# Most unstable (largest real part)
solver = EigenvalueSolver(rbc_evp(2000.0, 2.2214); nev=10, which=:LR)

# Largest magnitude
solver = EigenvalueSolver(rbc_evp(2000.0, 2.2214); nev=10, which=:LM)

# Near a target value
solver = EigenvalueSolver(rbc_evp(2000.0, 2.2214); nev=5, target=0.1+1.5im)

# Most oscillatory (largest imaginary)
solver = EigenvalueSolver(rbc_evp(2000.0, 2.2214); nev=10, which=:LI)
```

`which` accepts the symbols `:LM` `:SM` `:LR` `:SR` `:LI` `:SI`, or pass
`target=…` to order by proximity to a shift. The `EigenvalueSolver` constructor
takes only `nev`, `which`, `target`, and `matsolver` keywords.

### Selecting eigenvalues by magnitude

```julia
# Smallest-magnitude eigenvalues (e.g. least-damped modes)
solver = EigenvalueSolver(rbc_evp(2000.0, 2.2214); nev=20, which=:SM)
λ, _ = solve!(solver)
@printf("least-damped: σ = %+.6f\n", λ[1])
```

```
least-damped: σ = +11.015345
```

## Tips

### Resolution

- Start coarse (N=32), refine until eigenvalues converge
- Chebyshev provides exponential convergence
- Boundary layer modes need more resolution

### Spurious Modes

The tau rows carry zero mass, so a tau system throws off spurious eigenvalues — and they
are not small. The no-slip Rayleigh-Bénard problem above returns modes near `+8e5` at
`Nz = 24`, `+3e6` at `Nz = 32` and `+1.7e7` at `Nz = 48`: large enough to clear the
solver's internal `|λ| < 1e10` cut, and picked first by `which=:LR`, which would report a
comfortably stable state as violently unstable.

The tell is in those numbers — a spurious eigenvalue **moves with the resolution**, a
physical one does not. Solve at two resolutions and keep the modes that agree (this is the
`converged_eigenvalues` used in the Orr-Sommerfeld example):

```julia
function critical_Ra_filtered(k; N1=24, N2=32, Ra_lo=500.0, Ra_hi=4000.0, tol=1e-4)
    while (Ra_hi - Ra_lo) / Ra_hi > tol
        Ra_mid = (Ra_lo + Ra_hi) / 2
        λ = converged_eigenvalues(N -> rbc_evp(Ra_mid, k; Nz=N, slip=:noslip),
                                  N1, N2; nev=10)
        maximum(real, λ) > 0 ? (Ra_hi = Ra_mid) : (Ra_lo = Ra_mid)
    end
    return (Ra_lo + Ra_hi) / 2
end

@printf("no-slip Ra_c = %.2f   (Chandrasekhar: 1707.762)\n", critical_Ra_filtered(3.117))
```

```
no-slip Ra_c = 1707.77   (Chandrasekhar: 1707.762)
```

Also check the mode structure — spurious eigenfunctions oscillate wildly at the grid scale
— and confirm that the answer is insensitive to `nev`.

### Physical Validation

- Compare with known analytical results: stress-free Rayleigh-Bénard has
  `Ra_c(k) = (π² + k²)³/k²`; no-slip has `Ra_c = 1707.762` at `k = 3.117`; the
  Orr-Sommerfeld benchmark is `c = 0.23752649 + 0.00373967i` at `Re = 10⁴`, `k = 1`
- Check limiting cases
- Verify neutral curve shape

## See Also

- [Eigenvalue Problems Tutorial](../tutorials/eigenvalue_problems.md)
- [API: Solvers](../api/solvers.md)
- [Example Gallery](gallery.md)
