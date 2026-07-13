# Operators

Operators perform mathematical operations on fields, including differentiation and vector calculus.

## Differential Operators

### First Derivatives

`∂x`, `∂y`, `∂z` are **equation-string syntax**: the parser reads `∂<name>` as a derivative
along the coordinate named `<name>`. They are not Julia functions — a bare `∂x(field)` in
Julia code raises `UndefVarError`.

```julia
# Syntax in equations: ∂<coordinate name>
add_equation!(problem, "∂t(T) = -ux*∂x(T) - uz*∂z(T)")
```

In Julia code, use the exported `d` constructor, which takes a `Coordinate`:

```julia
d(T, coords["x"])      # ∂T/∂x
d(T, coords["x"], 2)   # ∂²T/∂x²
```

### Second Derivatives

```julia
# Composed derivatives (equation-string syntax)
add_equation!(problem, "∂t(T) - kappa*(∂x(∂x(T)) + ∂z(∂z(T))) = 0")

# ∂x(∂x(f))   ∂²/∂x²
# ∂z(∂z(f))   ∂²/∂z²
# ∂x(∂z(f))   ∂²/∂x∂z
```

### Laplacian

`Δ` (alias `lap`, `∇²`) works both in equation strings and in Julia code.

```julia
# Δ(f) = ∇²f
Δ(field)

# Example: Diffusion equation
add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")
```

### Fractional Laplacian

The fractional Laplacian `(-Δ)^α` generalizes the Laplacian to non-integer powers:

```julia
# fraclap(f, α) = (-Δ)^α f  — Julia code
fraclap(field, 0.5)    # Square root: (-Δ)^(1/2)
fraclap(field, -0.5)   # Inverse square root: (-Δ)^(-1/2)
fraclap(field, 1.0)    # Standard Laplacian: (-Δ)^1 = -Δ
fraclap(field, 2.0)    # Biharmonic: (-Δ)^2 = Δ²

# Convenience functions (Julia code only — see the note below)
sqrtlap(field)         # Same as fraclap(f, 0.5)
invsqrtlap(field)      # Same as fraclap(f, -0.5)
```

**In spectral space**: Multiplication by `|k|^(2α)` where `k` is the wavenumber.

**Applications**:
- **SQG dynamics**: `ψ = (-Δ)^(-1/2) θ` for streamfunction from buoyancy
- **Fractional diffusion**: `∂θ/∂t = -κ(-Δ)^α θ` for anomalous diffusion
- **Hyperviscosity**: `(-Δ)^n` for numerical dissipation at small scales

Inside an equation string, write `fraclap(f, α)` (or the Unicode alias `Δᵅ(f, α)`).
Put the term on the **implicit (LHS)** side: the compiled RHS declines a fractional
Laplacian and falls back to the slower interpreted path if it appears on the right.

```julia
# Buoyancy with fractional dissipation, treated implicitly
add_equation!(problem, "∂t(θ) + κ*fraclap(θ, 0.5) = -u⋅∇(θ)")
```

!!! warning "`sqrtlap` does not parse inside an equation string"
    `sqrtlap(f)` in an `add_equation!` string hits a parser bug, and the whole term is
    **silently dropped** from the equation (you get a one-time warning and an
    `UnknownOperator` placeholder). Write `fraclap(f, 0.5)` instead. `invsqrtlap(f)` and
    `fraclap(f, -0.5)` both parse correctly.

### Hyperviscosity (Higher-Order Laplacian)

For turbulence simulations, hyperviscosity provides selective dissipation at small scales while preserving large-scale dynamics:

```julia
# General form (Julia code): hyperlap(f, n) = (-Δ)^n = |k|^(2n) in Fourier space
hyperlap(field, 2)   # Biharmonic: (-Δ)² = |k|⁴
hyperlap(field, 4)   # 8th-order: (-Δ)⁴ = |k|⁸
hyperlap(field, 8)   # 16th-order: (-Δ)⁸ = |k|¹⁶

# Unicode shortcuts — these are the forms to use inside equation strings
Δ²(field)   # Biharmonic (4th-order derivative)
Δ⁴(field)   # 8th-order derivative
Δ⁶(field)   # 12th-order derivative
Δ⁸(field)   # 16th-order derivative
```

**In spectral space**: Multiplication by `|k|^(2n)` - very efficient for Fourier bases.

**Usage in equations**:

```julia
# 2D turbulence with biharmonic hyperviscosity
add_equation!(problem, "∂t(ω) + ν₄*Δ²(ω) = -u⋅∇(ω)")

# Incompressible turbulence with 8th-order hyperviscosity
add_equation!(problem, "∂t(u) + ∇(p) + ν₈*Δ⁴(u) = -u⋅∇(u)")
add_equation!(problem, "div(u) = 0")
```

!!! warning "`hyperlap(f, n)` does not parse inside an equation string"
    `hyperlap` requires an **`Integer`** order, but the equation parser turns every numeric
    literal into a `Float64`. `hyperlap(u, 4)` inside an `add_equation!` string therefore
    fails to build and the hyperviscosity term is **silently dropped** from the equation.
    Use `Δ²`/`Δ⁴`/`Δ⁶`/`Δ⁸` in equation strings, and reserve `hyperlap(f, n)` for direct
    Julia calls.

**Why use hyperviscosity?**

| Order | Operator | Spectral | Use Case |
|-------|----------|----------|----------|
| 2 | `Δ` | `-k²` | Standard viscosity |
| 4 | `Δ²` | `k⁴` | Mild scale separation |
| 8 | `Δ⁴` | `k⁸` | Strong scale separation |
| 16 | `Δ⁸` | `k¹⁶` | Extreme Reynolds numbers |

Higher orders concentrate dissipation at the smallest resolved scales, extending the inertial range.

**Typing Unicode**:

| Symbol | Type |
|--------|------|
| `Δ²` | `\Delta` + Tab, `\^2` + Tab |
| `Δ⁴` | `\Delta` + Tab, `\^4` + Tab |
| `Δ⁶` | `\Delta` + Tab, `\^6` + Tab |
| `Δ⁸` | `\Delta` + Tab, `\^8` + Tab |

## Tau Method Operators

These operators support the first-order formulation (tau method) for problems with Chebyshev bases, where explicit tau lifting is required for correct boundary condition enforcement.

### `trace(expr)`

Algebraic trace reduction: converts a first-order gradient variable (a rank-2 tensor or matrix expression) into its trace, i.e., the sum of diagonal components. Used in the first-order formulation to replace `div(u) = 0` with the trace of the velocity gradient:

```julia
# In the first-order formulation, use trace instead of div on the gradient variable
add_equation!(problem, "trace(grad_u) + tau_p = 0")
```

### `derivative_basis(basis, order)`

Returns the `order`-th derivative basis associated with `basis`. For Chebyshev polynomials, differentiating once maps ChebyshevT to ChebyshevU:

```julia
# Get the first-derivative basis for the vertical Chebyshev basis
lift_basis = derivative_basis(zbasis, 1)
```

This basis is then passed to `lift` to construct the tau lifting operator.

### `lift(operand, basis, n)`

The tau method lifting operator. Embeds `operand` (a tau variable) into the full spectral space using `basis`, selecting the `n`-th mode from the end (`n = -1` selects the last mode):

```julia
lift_basis = derivative_basis(zbasis, 1)
τ_lift(A) = lift(A, lift_basis, -1)

# Apply lifting to tau variables in first-order gradient definitions
grad_u = grad(u) + ez * τ_lift(tau_u1)
grad_b = grad(b) + ez * τ_lift(tau_b1)

# Tau variables also appear in the equations themselves
add_equation!(problem, "∂t(b) - kappa*div(grad_b) + τ_lift(tau_b2) = -u⋅∇(b)")
```

## Vector Calculus

### Gradient

Converts scalar to vector field.

```julia
# grad(p) = ∇p
add_equation!(problem, "∂t(u) + grad(p) = 0")

# Components:
# ∂p/∂x, ∂p/∂y, ∂p/∂z
```

### Divergence

Converts vector to scalar field.

```julia
# div(u) = ∇·u
add_equation!(problem, "div(u) = 0")

# Expands to:
# ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z = 0
```

!!! note "`div` vs `divergence` in Julia code"
    `div` is a parser alias, valid inside equation strings. In Julia code the name resolves
    to `Base.div` (integer division), so `div(u)` on a field is a `MethodError`. Call
    **`divergence(u)`** instead:

    ```julia
    divu = divergence(u)   # Divergence operator
    ```

### Curl

`curl(u)` takes a `VectorField` (not a `Gradient` or other operator node). In 3D it returns
a vector; in 2D it evaluates to the scalar vorticity `ω = ∂v/∂x - ∂u/∂y`.

```julia
# curl(u) = ∇×u
omega = curl(u)
```

### Perpendicular Gradient

For 2D flows, the perpendicular gradient creates a divergence-free velocity from a streamfunction.
Unlike the operators above, `perp_grad` is **eager**: it takes a 2-D `ScalarField` and returns a
`VectorField` computed immediately, rather than a symbolic operator node.

```julia
# perp_grad(ψ) = ∇⊥ψ = (-∂ψ/∂y, ∂ψ/∂x)
u = perp_grad(psi)

# This gives: u_x = -∂ψ/∂y, u_y = ∂ψ/∂x
# Automatically satisfies: ∇·u = 0
```

**Applications**:
- **2D incompressible flow**: `u = ∇⊥ψ` from streamfunction
- **SQG velocity**: `u = ∇⊥((-Δ)^(-1/2) θ)`
- **QG geostrophic flow**: `u = ∇⊥ψ`

## Time Derivatives

```julia
# ∂t(field) for IVP equations
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")

# Only valid in Initial Value Problems
```

## Unicode Operators

Tarang supports Unicode mathematical notation for cleaner, more readable code:

| Unicode | ASCII Equivalent | Description |
|---------|------------------|-------------|
| `∇` | `grad` | Gradient |
| `Δ` | `lap` | Laplacian |
| `∇²` | `lap` | Laplacian (alternative) |
| `∂t` | `dt` | Time derivative |
| `⋅` | `dot` | Dot product |
| `×` | `cross` | Cross product |
| `∇⊥` | `perp_grad` | Perpendicular gradient |
| `Δᵅ` | `fraclap` | Fractional Laplacian |

### In Equations

```julia
# Unicode syntax
add_equation!(problem, "∂t(u) + ∇(p) - nu*Δ(u) = -u⋅∇(u)")
```

### In Code

```julia
# ASCII
pressure_grad = grad(pressure)   # ScalarField -> Gradient
vorticity     = curl(u)          # VectorField -> Curl

# Unicode (equivalent)
pressure_grad = ∇(pressure)
vorticity     = curl(u)

# Mixed - use what's clearest
velocity    = ∇⊥(ψ)   # Perpendicular gradient (2D ScalarField -> VectorField)
dissipation = Δ(T)    # Laplacian
```

`curl` needs a `VectorField`; passing it an operator node such as `grad(p)` throws.

### Typing Unicode in Julia

In Julia REPL or editors with Julia support:

| Symbol | Type |
|--------|------|
| `∇` | `\nabla` + Tab |
| `Δ` | `\Delta` + Tab |
| `∂` | `\partial` + Tab |
| `⋅` | `\cdot` + Tab |
| `×` | `\times` + Tab |
| `⊥` | `\perp` + Tab |
| `α` | `\alpha` + Tab |
| `²` `⁴` `⁶` `⁸` | `\^2` + Tab, `\^4` + Tab, … |

## Using Operators in Equations

### Symbolic Syntax

```julia
# Equations are strings parsed symbolically
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -ux*∂x(T)")

# Supports:
# - Addition/subtraction: +, -
# - Multiplication: *
# - Division: /
# - Parentheses: (, )
# - Function calls: sin, cos, exp, etc.
```

### Common Patterns

```julia
# Advection (ux, uz are ScalarFields)
"ux*∂x(f) + uz*∂z(f)"

# Diffusion
"nu*Δ(u)"

# Pressure gradient
"∂x(p)"  # or "∇(p)" for vector

# Navier-Stokes viscous term
"nu*(∂x(∂x(u)) + ∂z(∂z(u)))"
# or simply:
"nu*Δ(u)"
```

### Advection Operator: `u⋅∇(f)`

The `u⋅∇(f)` notation is automatically expanded component-wise by the parser:

```
u⋅∇(f)  →  u₁∂₁f + u₂∂₂f + ... + uₙ∂ₙf
```

This works for both scalar fields `f` and vector fields. No manual expansion is needed:

```julia
# Scalar advection — automatically expands to Σᵢ uᵢ ∂ᵢf
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -u⋅∇(T)")

# Vorticity advection
add_equation!(problem, "∂t(ω) + ν₄*Δ²(ω) = -u⋅∇(ω)")
```

### VectorField Differentiation

Applying `Differentiate` to a `VectorField` operates component-wise — each component is differentiated with respect to the given coordinate independently:

```julia
# Differentiate(VectorField, coord) works component-wise
# e.g. ∂z(u) differentiates each component of u with respect to z
add_bc!(problem, "∂z(u)(z=0) = 0")   # stress-free BC for a vector field
add_bc!(problem, "∂z(u)(z=1) = 0")
```

This means you can apply `∂z`, `∂x`, etc. directly to vector fields in equations and boundary conditions without manually indexing components. Note that boundary conditions must be declared with `add_bc!`, not `add_equation!` — see [Boundary Conditions](../tutorials/boundary_conditions.md).

## Implementation Details

### Fourier Differentiation

For Fourier bases, differentiation is multiplication by wavenumber:

```math
\frac{\partial}{\partial x} \sum_k \hat{f}_k e^{ikx} = \sum_k (ik) \hat{f}_k e^{ikx}
```

### Chebyshev Differentiation

Uses recurrence relations:

```math
\frac{dT_n}{dx} = n U_{n-1}(x)
```

Implemented as sparse matrix operations.

### Computational Cost

| Operation | Cost |
|-----------|------|
| Fourier derivative | O(N) |
| Chebyshev derivative | O(N²) sparse |
| Laplacian | Same as 2× first derivative |

## Custom Operators

### Helper Functions

Build operator trees in Julia with `d(field, coord, order)` — the coordinates come from the
field's distributor, so the same helper works in any dimension. (Remember that `∂x`/`∂z` are
equation-string syntax, not Julia functions.)

```julia
function advection(u::VectorField, field)
    # u·∇f
    cs = u.dist.coordsys
    result = u.components[1] * d(field, cs[1], 1)
    for i in 2:length(u.components)
        result = result + u.components[i] * d(field, cs[i], 1)
    end
    return result
end
```

The helper returns a deferred expression tree, not a field. Give it to the problem with
`add_parameters!` and refer to it by name in the equation — the same mechanism the tau method
uses for `grad_u`:

```julia
add_parameters!(problem, kappa=0.05, adv=advection(u, T))
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -adv")
```

This produces bit-identical results to writing `-u⋅∇(T)` directly, which is what you would
normally do — the parser performs exactly this expansion for you.

### Using Built-in Operators in Equations

Operators the equation parser recognises out of the box:

```julia
# Differential:  ∂t (or dt), ∂x/∂y/∂z, d(f, x, n), grad (∇), div, curl,
#                lap (Δ, ∇²), trace, lift
# Fractional:    fraclap(f, α) (Δᵅ), invsqrtlap, Δ², Δ⁴, Δ⁶, Δ⁸
# Reductions:    integ(f) / integrate(f)
# Interpolation: f(z=0)      # boundary/point evaluation
# Products:      a⋅b (dot), a×b (cross), component(u, i)
# Elementwise:   sin, cos, tan, exp, log, sqrt, abs, tanh

# Use operators directly
add_equation!(problem, "∂t(T) - kappa*Δ(T) = -ux*∂x(T) - uz*∂z(T)")
```

`average(f, x)`, `integrate(f, x)` and `interpolate(f, x, pos)` take a `Coordinate` argument,
and coordinate *names* are not bound to `Coordinate` objects in the default namespace. Register
the coordinate first if you need them:

```julia
add_parameters!(problem, x=coords["x"])
add_equation!(problem, "∂t(T) - kappa*Δ(T) + average(T, x) = 0")
```

## Performance Tips

1. **Minimize transforms**: Group operations in same space
2. **Use Laplacian**: `Δ(f)` is optimized vs `∂x(∂x(f)) + ∂z(∂z(f))`
3. **Spectral derivatives**: Free in Fourier, cheap in Chebyshev
4. **Nonlinear terms**: Require transform to grid space

## See Also

- [Fields](fields.md): What operators act on
- [Problems](problems.md): Using operators in PDEs
- [Surface Dynamics](../tutorials/surface_dynamics.md): SQG, QG, and boundary advection-diffusion
- [API: Operators](../api/operators.md): Complete reference
