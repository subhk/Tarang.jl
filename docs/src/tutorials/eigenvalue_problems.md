# Tutorial: Eigenvalue Problems

This tutorial demonstrates solving eigenvalue problems (EVP) with Tarang.jl for linear stability analysis.

## Overview

Eigenvalue problems arise in linear stability analysis where we seek modes of the form:

```math
\mathbf{u}(x,z,t) = \hat{\mathbf{u}}(z) e^{i k x + \sigma t}
```

where $\sigma$ is the eigenvalue (growth rate + frequency) and $\hat{\mathbf{u}}$ is the eigenfunction.

## Eigenvalue Convention

Tarang solves the generalized problem $L\,\mathbf{x} = \sigma M\,\mathbf{x}$, where
the mass matrix $M$ is built from the **time-derivative terms** `dt(·)`. The
eigenvalue $\sigma$ replaces the time derivative ($\partial_t \to \sigma$), which
matches the normal-mode ansatz $\mathbf{u}\sim e^{\sigma t}$. So you write the
equation with `dt(u)` kept in place — **not** by multiplying the eigenvalue symbol
into the equation. (`sigma*u = …` produces an empty $M$ and returns no
eigenvalues.)

`dt(·)` must wrap a **variable**, not an expression: the mass matrix contributed by
`dt(X)` is the identity on `X`. A problem in which the eigenvalue multiplies an
*operator* — the classic Orr–Sommerfeld form $c\,(D^2-k^2)\hat\psi = \dots$ — must
first be reduced to that shape by introducing an auxiliary variable (see
[Orr-Sommerfeld Equation](#Orr-Sommerfeld-Equation) below).

Bounded directions use the **tau method**, identical to the BVP: add one `tau`
variable per boundary condition, lift it into the bulk equation with
`lift(tau, derivative_basis(basis, 2), -k)`, and declare BCs with `add_bc!`.

### Coefficients on the implicit side

Two rules that the solver does not enforce for you, and that silently change the
physics if you break them:

1. **Fold constant products into a single parameter.** A coefficient built from
   two parameters — `Ra*Pr*T_hat`, `Pr*k2*u_hat` — is assembled with a coefficient
   of **1**, not `Ra*Pr`. Register the product itself:
   `add_parameters!(evp; RaPr=Ra*Pr)` and write `RaPr*T_hat`. (A single parameter,
   a literal, or a right-nested `Pr*(k2*u_hat)` are all handled correctly.)
2. **A spatially varying coefficient field must be real.** A field coefficient
   `U(z)*q` on the implicit side is supported when `U` varies along a single
   Chebyshev/Jacobi axis, but only if `U` is declared `Float64`. A `ComplexF64`
   coefficient field is dropped (coefficient 1). Declare base-flow profiles as
   real fields — the eigenvalues and eigenvectors are complex regardless of the
   field element type.

## Basic EVP Setup

A complete example — the 1D diffusion eigenproblem $\sigma u = \Delta u$
with Dirichlet walls. Its spectrum is the Dirichlet Laplacian,
$\sigma_n = -(n\pi/L_z)^2$.

```julia
using Tarang

coords = CartesianCoordinates("z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))
dom    = Domain(dist, (z_basis,))

# Eigenmode field + one tau per boundary condition
u_hat = ScalarField(dom, "u_hat")
tau1  = ScalarField(dist, "tau1", (), Float64)
tau2  = ScalarField(dist, "tau2", (), Float64)
lb2   = derivative_basis(z_basis, 2)

evp = Tarang.EVP([u_hat, tau1, tau2]; eigenvalue=:σ)
add_parameters!(evp; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))

# σ u = Δu  ⇒  keep dt(u) (the solver maps dt → σ), move linear terms to the LHS
Tarang.add_equation!(evp, "dt(u_hat) - Δ(u_hat) - l1 - l2 = 0")
Tarang.add_bc!(evp, "u_hat(z=0)   = 0")
Tarang.add_bc!(evp, "u_hat(z=1.0) = 0")

solver = Tarang.EigenvalueSolver(evp; nev=5, which=:SM)   # 5 smallest |σ|
eigenvalues, eigenvectors = Tarang.solve!(solver)

for (i, σ) in enumerate(eigenvalues)
    println("Mode $i: σ = $(real(σ)) + $(imag(σ))im")   # |σ_n| ≈ (nπ)²
end
```

prints $\sigma_n = -9.8696,\ -39.478,\ -88.826,\ -157.91,\ -246.74$ — that is $-(n\pi)^2$,
matched to about ten significant digits.

### What `solve!` returns

```julia
eigenvalues     # Vector{ComplexF64},  length nev, ordered by `which`
eigenvectors    # Matrix{ComplexF64},  column i = mode i
```

`eigenvectors[:, i]` is the *i*-th mode as a single stacked coefficient vector in
the subproblem's variable ordering (all state fields, then the taus) — not a
dictionary keyed by field name. [Extracting Eigenmodes](#Extracting-Eigenmodes)
below shows how to unpack it back into fields.

Eigenvectors are only returned when the problem has a **single** subproblem, which
is the case here (no Fourier axis, serial run). When the problem decomposes into
several per-Fourier-mode subproblems, or runs distributed, `eigenvectors` comes
back as an empty `0×0` matrix and only the eigenvalues are meaningful.

## Rayleigh-Bénard Stability

Classical linear stability problem for thermal convection.

### Governing Equations

Linearized Boussinesq equations for perturbations $\propto e^{ikx+\sigma t}$:

```math
\begin{aligned}
\sigma \hat{u} &= -i k \hat{p} + \text{Pr} \nabla^2 \hat{u} \\
\sigma \hat{w} &= -D \hat{p} + \text{Pr} \nabla^2 \hat{w} + \text{Ra} \cdot \text{Pr} \cdot \hat{T} \\
i k \hat{u} + D \hat{w} &= 0 \\
\sigma \hat{T} &= \hat{w} + \nabla^2 \hat{T}
\end{aligned}
```

where $D = d/dz$ and $\nabla^2 = D^2 - k^2$.

### Implementation

The horizontal wavenumber $k$ is a *parameter* here, so the domain is the single
bounded direction $z$. Each second-order equation carries two tau variables; the
continuity equation carries a scalar gauge tau, fixed by `integ(p_hat) = 0`.

Because parameter values are baked into the equations when `add_equation!` parses
them, a parameter sweep must rebuild the problem — so wrap the whole setup in a
function.

```julia
using Tarang

function rbc_evp(Ra, k; Pr=1.0, Nz=24, nev=4, which=:LR)
    coords = CartesianCoordinates("z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    zb     = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, 1.0))

    # complex amplitudes û(z), ŵ(z), p̂(z), T̂(z)
    u = ScalarField(dist, "u_hat", (zb,), ComplexF64)
    w = ScalarField(dist, "w_hat", (zb,), ComplexF64)
    p = ScalarField(dist, "p_hat", (zb,), ComplexF64)
    T = ScalarField(dist, "T_hat", (zb,), ComplexF64)
    # two taus per second-order equation + one gauge tau for the pressure
    tau = [ScalarField(dist, "tau$i", (), ComplexF64) for i in 1:7]

    lb  = derivative_basis(zb, 2)
    evp = Tarang.EVP([u, w, p, T, tau...]; eigenvalue=:σ)

    # every implicit coefficient is ONE parameter: Ra*Pr and Pr*k² are folded here
    add_parameters!(evp; Pr=Pr, RaPr=Ra*Pr, Prk2=Pr*k^2, ik=1im*k, k2=k^2,
                    lu1=lift(tau[1], lb, -1), lu2=lift(tau[2], lb, -2),
                    lw1=lift(tau[3], lb, -1), lw2=lift(tau[4], lb, -2),
                    lT1=lift(tau[5], lb, -1), lT2=lift(tau[6], lb, -2),
                    taup=tau[7])

    # Momentum: keep dt(·) — the eigenvalue σ replaces ∂t and builds the mass matrix M.
    Tarang.add_equation!(evp,
        "dt(u_hat) + ik*p_hat - Pr*∂z(∂z(u_hat)) + Prk2*u_hat + lu1 + lu2 = 0")
    Tarang.add_equation!(evp,
        "dt(w_hat) + ∂z(p_hat) - Pr*∂z(∂z(w_hat)) + Prk2*w_hat - RaPr*T_hat + lw1 + lw2 = 0")
    # Continuity: algebraic constraint — no dt term, so its M rows are zero
    Tarang.add_equation!(evp, "ik*u_hat + ∂z(w_hat) + taup = 0")
    # Temperature
    Tarang.add_equation!(evp,
        "dt(T_hat) - w_hat - ∂z(∂z(T_hat)) + k2*T_hat + lT1 + lT2 = 0")

    # No-slip, fixed temperature, and a pressure gauge
    for f in ("u_hat", "w_hat", "T_hat"), z in (0, 1)
        Tarang.add_bc!(evp, "$f(z=$z) = 0")
    end
    Tarang.add_bc!(evp, "integ(p_hat) = 0")

    return Tarang.solve!(Tarang.EigenvalueSolver(evp; nev=nev, which=which))
end

growth(Ra, k; kwargs...) = maximum(real.(first(rbc_evp(Ra, k; kwargs...))))

for Ra in (1000.0, 1707.76, 2500.0)
    println("Ra = $Ra, k = 3.117:  max Re σ = ", round(growth(Ra, 3.117), digits=6))
end
```

```
Ra = 1000.0, k = 3.117:  max Re σ = -6.002789
Ra = 1707.76, k = 3.117:  max Re σ = -1.4e-5
Ra = 2500.0, k = 3.117:  max Re σ = 5.514215
```

At the textbook critical point ($\text{Ra}_c = 1707.76$, $k_c = 3.117$) the leading
growth rate is zero to five digits, with 24 Chebyshev modes.

## Orr-Sommerfeld Equation

Stability of parallel shear flows.

### Equation

```math
\left[ (U - c)(D^2 - k^2) - U'' + \frac{i}{\text{Re} \cdot k}(D^2 - k^2)^2 \right] \hat{\psi} = 0
```

where $c = \sigma/(-ik)$ is the complex wave speed. As written, the eigenvalue
multiplies the operator $(D^2-k^2)$, which Tarang cannot assemble (the mass matrix
from `dt(X)` is the identity on `X`). Introduce the auxiliary variable
$\hat q = (D^2-k^2)\hat\psi$ and the problem becomes a standard generalized
eigenproblem in $(\hat\psi, \hat q)$:

```math
\begin{aligned}
\sigma \hat q &= -ik\left(U \hat q - U'' \hat\psi\right) + \frac{1}{\text{Re}}(D^2-k^2)\hat q \\
\hat q &= (D^2 - k^2)\hat\psi
\end{aligned}
```

The second equation has no `dt` term, so it contributes zero rows to $M$ — exactly
like the continuity constraint above.

### Implementation

Plane Poiseuille flow, $U = 1 - z^2$ on $z\in[-1,1]$, so $U'' = -2$. The base-flow
profile is a **real** field (see the coefficient rules above); the four no-slip
conditions $\hat\psi = \hat\psi' = 0$ at both walls give four taus.

```julia
using Tarang

function orr_sommerfeld(Re, k; Nz=64, nev=4)
    coords = CartesianCoordinates("z")
    dist   = Distributor(coords; dtype=Float64, device=CPU())
    zb     = ChebyshevT(coords["z"]; size=Nz, bounds=(-1.0, 1.0))

    psi = ScalarField(dist, "psi", (zb,), Float64)   # ψ̂
    q   = ScalarField(dist, "q",   (zb,), Float64)   # q̂ = (D² − k²)ψ̂
    U   = ScalarField(dist, "U",   (zb,), Float64)   # base flow: REAL field
    set!(U, z -> 1.0 - z^2)
    ensure_layout!(U, :c)

    tau = [ScalarField(dist, "t$i", (), Float64) for i in 1:4]
    lb  = derivative_basis(zb, 2)
    evp = Tarang.EVP([psi, q, tau...]; eigenvalue=:σ)
    add_parameters!(evp; U=U, ik=1im*k, k2=k^2, invRe=1/Re, invRe_k2=k^2/Re,
                    ikUpp=1im*k*(-2.0),                 # U'' = -2 for U = 1 − z²
                    l1=lift(tau[1], lb, -1), l2=lift(tau[2], lb, -2),
                    l3=lift(tau[3], lb, -1), l4=lift(tau[4], lb, -2))

    Tarang.add_equation!(evp,
        "dt(q) + ik*(U*q) - ikUpp*psi - invRe*∂z(∂z(q)) + invRe_k2*q + l1 + l2 = 0")
    Tarang.add_equation!(evp, "q - ∂z(∂z(psi)) + k2*psi + l3 + l4 = 0")

    Tarang.add_bc!(evp, "psi(z=-1) = 0")
    Tarang.add_bc!(evp, "psi(z=1) = 0")
    Tarang.add_bc!(evp, "∂z(psi)(z=-1) = 0")
    Tarang.add_bc!(evp, "∂z(psi)(z=1) = 0")

    λ, _ = Tarang.solve!(Tarang.EigenvalueSolver(evp; nev=nev, which=:LR))
    return λ
end

k = 1.02056
σ = orr_sommerfeld(5772.22, k)[1]     # least-stable mode at the critical point
c = 1im * σ / k                       # wave speed:  σ = -i k c
println("σ = ", round(σ, digits=6))
println("c = ", round(c, digits=6))
```

```
σ = -0.0 - 0.26943im
c = 0.264002 - 0.0im
```

which is the textbook neutral point of plane Poiseuille flow:
$\text{Re}_c = 5772.22$, $k_c = 1.02056$, $c_r = 0.264$, $c_i = 0$. The growth rate
is zero only to roundoff — unrounded, $\text{Re}\,\sigma \approx -3\times10^{-9}$,
whose trailing digits shift from run to run, which is why the printout above is
rounded. Lowering the Reynolds number damps the mode and raising it destabilises —
the same call at `Re = 4000` gives $c_i = -0.0045$, and at `Re = 10000` gives
$c_i = +0.0032$.

## Parameter Studies

Parameter values are substituted into the equations at `add_equation!` time, so a
sweep **rebuilds the problem** for each value — that is what the `rbc_evp` /
`growth` functions above are for. (Assigning to `evp.parameters[...]` after the
equations are parsed changes nothing; and a `Problem` can only be handed to one
solver, because building a solver merges the boundary conditions into its equation
list.)

### Scanning Rayleigh Number

```julia
Ra_values   = range(1000.0, 2500.0; length=6)
growth_rates = [growth(Ra, 3.117) for Ra in Ra_values]

# critical Ra: linear interpolation of the zero crossing
i = findfirst(>(0), growth_rates)
Ra_lo, Ra_hi = Ra_values[i-1], Ra_values[i]
g_lo,  g_hi  = growth_rates[i-1], growth_rates[i]
Ra_crit = Ra_lo - g_lo * (Ra_hi - Ra_lo) / (g_hi - g_lo)

println("growth rates: ", round.(growth_rates, digits=4))
println("Critical Rayleigh number: ", round(Ra_crit, digits=1))
```

```
growth rates: [-6.0028, -3.2885, -0.8321, 1.4287, 3.5347, 5.5142]
Critical Rayleigh number: 1710.4
```

(1710.4 rather than 1707.8 because the zero crossing is interpolated linearly from
a coarse six-point scan; the bisection below converges to the exact value.)

### Scanning Wavenumber

```julia
k_values = range(2.0, 4.5; length=6)
gr = [growth(2500.0, k) for k in k_values]

println("growth rates: ", round.(gr, digits=4))
println("Most unstable wavenumber at Ra = 2500: ", k_values[argmax(gr)])
```

```
growth rates: [1.4701, 3.9323, 5.3431, 5.6553, 4.9042, 3.159]
Most unstable wavenumber at Ra = 2500: 3.5
```

## Neutral Curves

Computing the stability boundary in parameter space — bisect on `Ra` at fixed `k`:

```julia
function neutral_Ra(k; Ra_lo=500.0, Ra_hi=5000.0, tol=1.0)
    while Ra_hi - Ra_lo > tol
        Ra_mid = (Ra_lo + Ra_hi) / 2
        growth(Ra_mid, k) > 0 ? (Ra_hi = Ra_mid) : (Ra_lo = Ra_mid)
    end
    return (Ra_lo + Ra_hi) / 2
end

for k in (2.5, 3.117, 4.0)
    println("k = $k   neutral Ra ≈ ", round(neutral_Ra(k), digits=1))
end
```

```
k = 2.5   neutral Ra ≈ 1822.5
k = 3.117   neutral Ra ≈ 1707.7
k = 4.0   neutral Ra ≈ 1879.1
```

The curve has its minimum at $k \approx 3.117$ with $\text{Ra} \approx 1708$ — the
critical point of Rayleigh-Bénard convection between rigid plates.

## Extracting Eigenmodes

`eigenvectors[:, i]` is a stacked coefficient vector, not per-field arrays. Scatter
it back into the problem's fields to get a grid-space profile — the fields are
overwritten in place, so this reuses the `solver` from the Basic EVP Setup:

```julia
sp   = solver.subproblems[1]              # single subproblem: no Fourier axis
mode = eigenvectors[:, 1]                 # column 1 = the mode you want
Tarang.scatter_inputs(sp, mode, [u_hat, tau1, tau2])   # same order as EVP(...)

ensure_layout!(u_hat, :g)
z       = local_grids(dist, z_basis)[1]
profile = Array(get_grid_data(u_hat))
profile ./= profile[argmax(abs.(profile))]            # normalise: peak = +1

reference = sin.(π .* z)
reference ./= reference[argmax(abs.(reference))]
println("max |mode − sin(πz)| = ",
        round(maximum(abs, profile .- reference), sigdigits=2))
```

```
max |mode − sin(πz)| = 1.9e-12
```

`profile` and `z` are plain vectors, ready for whichever plotting package you use.
Note that `u_hat` here is a `Float64` field, so only the real part of the mode is
kept; declare the fields `ComplexF64` (as in `rbc_evp`) when the eigenfunction is
genuinely complex.

## Solver Options

### Which Eigenvalues

`nev`, `which` and `target` can be given to the constructor, or overridden per call
on an existing solver — the latter avoids rebuilding the matrices, and is the only
way to re-select on a problem you have already solved (a `Problem` cannot be handed
to a second solver):

```julia
Tarang.solve!(solver; nev=3, which=:LM)   # largest magnitude
Tarang.solve!(solver; nev=3, which=:LR)   # largest real part (most unstable)
Tarang.solve!(solver; nev=3, which=:SM)   # smallest magnitude
Tarang.solve!(solver; nev=3, which=:SR)   # smallest real part
```

### Shift-invert near a target

```julia
# order eigenvalues by proximity to a target (useful for interior spectra)
λ, _ = Tarang.solve!(solver; nev=3, target = -90.0 + 0.0im)
println(round.(real.(λ), digits=4))     # [-88.8264, -39.4784, -157.9137]
```

`EigenvalueSolver` accepts `nev`, `which` (`:LM`/`:SM`/`:LR`/`:SR`/`:LI`/`:SI`),
`target`, and `matsolver`. It solves `eigen(L, M)` per Fourier mode on the square
tau matrices and discards spurious eigenvalues from the singular mass matrix.

## See Also

- [Problems API](../api/problems.md): EVP problem definition
- [Solvers API](../api/solvers.md): Eigenvalue solver details
- [Rayleigh-Bénard Tutorial](ivp_2d_rbc.md): Time-dependent version
