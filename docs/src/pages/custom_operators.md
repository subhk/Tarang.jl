# Custom Operators

Creating custom differential and integral operators.

## Overview

Beyond built-in operators (grad, div, curl, lap), you can define custom operators for specialized
physics. There is no plugin API to implement — a custom operator is just a Julia function that
*composes* built-in operators into an expression. Such an expression is lazy: nothing is computed
until you either

- pass it to the solver as part of an equation, or
- call `evaluate(expr, :g)` to get a field back.

For diagnostics you can also skip the operator layer entirely and work directly on grid data with
`get_grid_data`.

All examples below share this setup:

```julia
using Tarang

coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xbasis = RealFourier(coords["x"]; size=16, bounds=(0.0, 2π), dealias=3/2)
zbasis = RealFourier(coords["z"]; size=16, bounds=(0.0, 2π), dealias=3/2)
domain = Domain(dist, (xbasis, zbasis))

T = ScalarField(domain, "T")
u = VectorField(domain, "u")

x, z = local_grids(dist, xbasis, zbasis)

ensure_layout!(T, :g)
get_grid_data(T) .= sin.(x) .* cos.(z')      # T = sin(x) cos(z)
ensure_layout!(T, :c)

ensure_layout!(u, :g)
get_grid_data(u.components[1]) .= ones(length(x)) .* sin.(z)'   # u = (sin z, 0): a shear flow
get_grid_data(u.components[2]) .= 0
ensure_layout!(u, :c)
```

## Helper Functions

### Simple Custom Operators

An operator is a function returning an operator expression. Build the expression from the
built-ins and let `evaluate` do the transforms:

```julia
# Advection operator: u·∇f
advect(u, f) = u ⋅ grad(f)

adv    = advect(u, T)          # DotProduct — nothing computed yet
result = evaluate(adv, :g)     # ScalarField, in grid layout

maximum(abs, get_grid_data(result))   # 0.5, matching sin(z)·cos(x)cos(z)
```

Note that `u ⋅ grad(f)` is `dot(u, grad(f))`: `⋅` and `×` between two vector-valued operands build
`DotProduct` / `CrossProduct` nodes, which `evaluate` understands.

### What `evaluate` accepts

`evaluate(expr, layout)` handles operator nodes (`Differentiate`, `Gradient`, `Divergence`, `Curl`,
`Laplacian`, `Integrate`, …) and `⋅` / `×` products of vector fields:

```julia
evaluate(d(T, coords["x"]), :g)   # ScalarField
evaluate(lap(T), :g)              # ScalarField
evaluate(grad(u), :g)             # TensorField
```

Plain arithmetic around an operator (`2.0 * lap(T)`, `a + b`) builds a deferred `Multiply` / `Add`
node instead. Those nodes are meant for *equations*; `evaluate` does not compute them and throws a
`MethodError`. Scale the evaluated field instead:

```julia
lapT = evaluate(lap(T), :g)
get_grid_data(lapT) .*= 2.0
```

### Using in Equations

A custom operator becomes usable inside an equation string once you register it as a parameter.
Equation strings resolve names against the problem's variables and its registered parameters only —
they cannot see Julia globals — so `add_parameters!` is what makes the name visible:

```julia
hyperdiff(f) = lap(lap(f))        # custom operator: ∇⁴

problem = IVP([T])
add_parameters!(problem, nu4=0.01, hyperdiff=hyperdiff)
add_equation!(problem, "∂t(T) + nu4*hyperdiff(T) = 0")

solver = InitialValueSolver(problem, RK222(); dt=1e-3)
run!(solver; stop_iteration=100, progress=false)

# T = sin(x)cos(z) is a ∇⁴ eigenfunction with eigenvalue (kx²+kz²)² = 4
ensure_layout!(T, :g)
maximum(abs, get_grid_data(T))    # 0.99600798934 ≈ exp(-nu4 * 4 * 0.1)
```

The same trick works for operator *values*, not just functions: `add_parameters!(problem, gradT=grad(T))`
registers the expression under the name `gradT`.

!!! warning "An operator used in an equation must return an operator expression"
    The parser calls your function with the parsed operands and splices the result into the
    equation tree. If the function computes eagerly and returns a raw array — anything built with
    `get_grid_data` — the term is **silently dropped**: no error, no warning, and the equation
    integrates as if the term were absent. Keep equation-side operators symbolic (`lap`, `grad`,
    `d`, `lift`, …). Grid-space kernels are for diagnostics and post-processing only.

## Using Operators in Equations

The equation parser recognizes all built-in operators. Use them directly:

```julia
# Differential:  grad (∇), div, curl, lap (Δ, laplacian), d (differentiate), dt (∂t)
#                hyperlap, fraclap, sqrtlap, invsqrtlap, hilbert
# Reductions:    integrate (integ), average (avg, ave), interpolate
# Tensor:        trace, skew, transpose_components (trans), outer, component (comp)
# Basis:         convert, lift, grid, coeff, copy
# Ufuncs:        sin, cos, tan, tanh, exp, log, sqrt, abs

T2 = ScalarField(domain, "T2")
u2 = VectorField(domain, "u2")

problem2 = IVP([T2, u2])
add_parameters!(problem2, kappa=0.05, nu=0.05)
add_equation!(problem2, "∂t(T2) - kappa*Δ(T2) = -u2⋅∇(T2)")
add_equation!(problem2, "∂t(u2) - nu*lap(u2) = -u2⋅∇(u2)")

solver2 = InitialValueSolver(problem2, RK222(); dt=1e-3)
solver2.rhs_plan.is_compiled     # true — the RHS was compiled, not interpreted
```

`-u⋅∇(u)` (vector advection) is likewise valid *inside an equation*. It has no direct `evaluate`
form: `evaluate(u ⋅ grad(u), :g)` throws, because `grad(u)` is a tensor and `DotProduct` requires
two vector fields. Compute it componentwise if you need it outside a solve (see
[Convective Derivative](#Convective-Derivative) below).

## Spectral Differentiation

### Fourier Derivative

Coefficient arrays store the first axis in RFFT layout (`N÷2+1` complex values, non-negative
wavenumbers) and every trailing axis in FFT layout (signed wavenumbers). Use the matching
wavenumber vector:

```julia
function fourier_dx!(f, basis)
    ensure_layout!(f, :c)
    k = Tarang.wavenumbers_rfft(basis)    # first axis: [0, k₀, 2k₀, …, (N/2)k₀]
    get_coeff_data(f) .*= im .* k         # multiply by ik
    return f
end

g = ScalarField(domain, "g")
ensure_layout!(g, :g)
get_grid_data(g) .= sin.(x) .* cos.(z')
ensure_layout!(g, :c)

fourier_dx!(g, xbasis)
ensure_layout!(g, :g)
maximum(abs, get_grid_data(g) .- cos.(x) .* cos.(z'))   # 2.0e-15
```

For a trailing axis the layout is the standard FFT ordering, so the wavenumbers are
`wavenumbers_fft` and they broadcast along that axis:

```julia
function fourier_dz!(f, basis)
    ensure_layout!(f, :c)
    k = Tarang.wavenumbers_fft(basis)     # [0, …, N/2-1, -N/2, …, -1] · k₀
    get_coeff_data(f) .*= im .* k'
    return f
end
```

(`wavenumbers(basis)` returns the *native* RealFourier cos/sin ordering of length `N`, which does
**not** match the coefficient array. `Tarang.wavenumbers_rfft` / `Tarang.wavenumbers_fft` are the
ones that do.)

### Chebyshev Derivative

There is no hand-rolled path worth writing here: `d(f, coord)` already applies the correct
recurrence, in the correct basis, for Chebyshev and every other Jacobi basis.

```julia
zcoords = CartesianCoordinates("x", "z")
zdist   = Distributor(zcoords; dtype=Float64, device=CPU())
xb = RealFourier(zcoords["x"]; size=8,  bounds=(0.0, 2π))
zb = ChebyshevT(zcoords["z"];  size=16, bounds=(0.0, 1.0))
cdomain = Domain(zdist, (xb, zb))

q = ScalarField(cdomain, "q")
xq, zq = local_grids(zdist, xb, zb)
ensure_layout!(q, :g)
get_grid_data(q) .= ones(length(xq)) .* (zq .^ 3)'
ensure_layout!(q, :c)

dq = evaluate(d(q, zcoords["z"]), :g)
maximum(abs, get_grid_data(dq) .- (3 .* zq .^ 2)')     # 1.9e-14
```

If you need the underlying coefficient-space matrix — to assemble your own operator, say — it is
`differentiation_matrix(basis, order)` (a sparse `N×N` matrix, cached on the basis):

```julia
D = differentiation_matrix(zb, 1)      # 16×16 SparseMatrixCSC
```

## Integral Operators

### Spatial Integration

Integration is a built-in operator, and it carries the right quadrature weights for every basis
(uniform on Fourier, Chebyshev–Gauss on Chebyshev), so do not roll your own:

```julia
integrate(T)                                     # Float64: integral over the whole domain
Tx = evaluate(integrate(T, coords["x"]), :g)     # ScalarField on z: integral along x only
Tz = evaluate(average(T, coords["z"]), :g)       # ScalarField on x: mean along z

integrate(q)                                     # ∫∫ z³ dz dx over [0,2π]×[0,1] = 1.5707963268
```

`integrate(field)` is also the MPI-safe form: it reduces across ranks for you.

## Vector Calculus

### Strain Rate Tensor

```julia
function strain_rate(u)
    cs = u.coordsys
    S  = TensorField(u.dist, cs, "S", u.bases, u.dtype)

    for i in 1:cs.dim, j in 1:cs.dim   # S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)
        dij = evaluate(d(u.components[i], cs.coords[j]), :g)
        dji = evaluate(d(u.components[j], cs.coords[i]), :g)
        Sij = S.components[i, j]
        ensure_layout!(Sij, :g)
        get_grid_data(Sij) .= 0.5 .* (get_grid_data(dij) .+ get_grid_data(dji))
    end
    return S
end

S = strain_rate(u)
maximum(abs, get_grid_data(S.components[1,2]))   # 0.5, matching ½cos(z) for u = (sin z, 0)
```

`TensorField` components live in a `dim × dim` matrix — `S.components[i,j]` — and are `ScalarField`s
like any other. There is no `S[i,j] = …` setindex and no `symmetric=true` option.

The velocity gradient itself is a single call, with the convention `(∇u)_ij = ∂u_j/∂x_i`:

```julia
gu   = evaluate(grad(u), :g)                     # TensorField
gu_T = evaluate(transpose_components(gu), :g)    # its transpose
```

`transpose_components` needs a concrete `TensorField`, so evaluate `grad(u)` first.

### Vorticity (2D)

In two dimensions `curl` returns a `ScalarField`:

```julia
omega = evaluate(curl(u), :g)                    # ω = ∂x u_z − ∂z u_x
maximum(abs, get_grid_data(omega))               # 1.0, matching −cos(z)
```

which is the same thing as writing it out by hand:

```julia
omega2 = evaluate(d(u.components[2], coords["x"]) - d(u.components[1], coords["z"]), :g)
```

### Helicity (3D)

```julia
coords3 = CartesianCoordinates("x", "y", "z")
dist3   = Distributor(coords3; dtype=Float64, device=CPU())
bx = RealFourier(coords3["x"]; size=8, bounds=(0.0, 2π))
by = RealFourier(coords3["y"]; size=8, bounds=(0.0, 2π))
bz = RealFourier(coords3["z"]; size=8, bounds=(0.0, 2π))
domain3 = Domain(dist3, (bx, by, bz))

v = VectorField(domain3, "v")
x3, y3, z3 = local_grids(dist3, bx, by, bz)
ensure_layout!(v, :g)
vx, vy, vz = (get_grid_data(c) for c in v.components)
for k in eachindex(z3), j in eachindex(y3), i in eachindex(x3)
    vx[i,j,k] = sin(z3[k]) + cos(y3[j])          # ABC flow with A=B=C=1: ∇×v = v
    vy[i,j,k] = sin(x3[i]) + cos(z3[k])
    vz[i,j,k] = sin(y3[j]) + cos(x3[i])
end
ensure_layout!(v, :c)

H = evaluate(v ⋅ curl(v), :g)      # H = u·ω, a ScalarField
integrate(H)                       # 744.15 — total helicity (= ∫|v|² for this flow)
```

## Nonlinear Operators

### Convective Derivative

`u ⋅ grad(f)` is the one-liner (see above). When you want the loop — to reuse buffers, or to get at
the individual terms — write it in grid space:

```julia
function convective_derivative(u, f)
    ensure_layout!(f, :g)
    result = zeros(size(get_grid_data(f)))
    cs = u.coordsys

    for (i, comp) in enumerate(u.components)
        ensure_layout!(comp, :g)
        df = evaluate(d(f, cs.coords[i]), :g)
        result .+= get_grid_data(comp) .* get_grid_data(df)
    end
    return result
end

convective_derivative(u, T)   # matches evaluate(u ⋅ grad(T), :g) to roundoff (2.8e-16)
```

`cs.coords[i]` is the `Coordinate` for axis `i`; `d(f, coord)` differentiates along it.

### Nonlinear Term (u·∇u)

Inside an equation, write it directly: `add_equation!(problem, "∂t(u) - nu*lap(u) = -u⋅∇(u)")`.
Outside a solve, apply `convective_derivative` to each velocity component:

```julia
function nonlinear_advection(u)
    result = VectorField(u.dist, u.coordsys, "adv", u.bases, u.dtype)

    for (j, uj) in enumerate(u.components)
        rj = result.components[j]
        ensure_layout!(rj, :g)
        get_grid_data(rj) .= convective_derivative(u, uj)
    end
    return result
end
```

## Physics-Specific Operators

### Coriolis Force

For rotation about the z-axis, `-2Ω ẑ × u = (2Ω u_y, -2Ω u_x, 0)` has no z-component and needs no
derivatives, so build it straight in grid space:

```julia
function coriolis(u, Omega)
    cs = u.coordsys
    f  = VectorField(u.dist, cs, "coriolis", u.bases, u.dtype)
    ux, uy = u.components[1], u.components[2]
    ensure_layout!(ux, :g); ensure_layout!(uy, :g)
    for comp in f.components
        ensure_layout!(comp, :g)
    end

    get_grid_data(f.components[1]) .=  2 .* Omega .* get_grid_data(uy)   # +2Ω v
    get_grid_data(f.components[2]) .= -2 .* Omega .* get_grid_data(ux)   # -2Ω u
    get_grid_data(f.components[3]) .= 0
    return f
end

fc = coriolis(v, 0.5)      # VectorField on the same bases as v
```

Note that the `ez` from `unit_vector_fields` is a *basis-less* `VectorField`, so `cross(ez, u)`
raises `ArgumentError: Cannot compute cross product of VectorFields with different bases`. Unit
vectors are for equation strings (`buoy*T*ez`), not for `evaluate`.

### Lorentz Force (MHD)

`cross` (`×`) works directly on two vector fields that share a domain:

```julia
J = VectorField(domain3, "J")
B = VectorField(domain3, "B")
ensure_layout!(J, :g); ensure_layout!(B, :g)
get_grid_data(J.components[1]) .= 1.0        # J = x̂
get_grid_data(B.components[2]) .= 1.0        # B = ŷ
ensure_layout!(J, :c); ensure_layout!(B, :c)

F = evaluate(J × B, :g)                      # x̂ × ŷ = ẑ
get_grid_data(F.components[3])[1,1,1]        # 1.0
```

## Tips

### Performance

- Keep operations in same space (grid or spectral)
- Minimize transforms
- Pre-compute constant factors

### Validation

- Test against analytical solutions
- Check symmetries
- Verify conservation properties

## See Also

- [Operators](operators.md): Built-in operators
- [Fields](fields.md): Field types
- [API: Operators](../api/operators.md): Reference
