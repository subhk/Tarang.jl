# Tau Method

The tau method is a spectral technique for enforcing boundary conditions.

## Overview

In spectral methods, boundary conditions cannot be applied directly at grid points (except for Gauss-Lobatto points at boundaries). The tau method modifies the highest-order spectral coefficients to satisfy boundary constraints.

## How It Works

### Standard Spectral Expansion

For a Chebyshev expansion:

```math
f(x) = \sum_{n=0}^{N-1} a_n T_n(x)
```

Without tau method, all coefficients are determined by the PDE.

### With Tau Method

Replace highest-order equations with boundary conditions:

1. PDE determines coefficients a₀, a₁, ..., a_{N-k-1}
2. Boundary conditions determine a_{N-k}, ..., a_{N-1}

where k is the number of boundary conditions.

## Implementation in Tarang.jl

### Automatic Handling

Tarang automatically applies the tau method when you add boundary conditions:

```julia
# Creates tau terms internally
Tarang.add_dirichlet_bc!(problem, "u(z=0) = 0")
Tarang.add_dirichlet_bc!(problem, "u(z=1) = 0")
```

### Number of Tau Terms

| Operator Order | Tau Terms Needed |
|---------------|------------------|
| 1st (∂/∂z) | 1 |
| 2nd (∂²/∂z²) | 2 |
| 4th (biharmonic) | 4 |

## Example: Diffusion Equation

### Continuous Problem

```math
\frac{\partial u}{\partial t} = \kappa \frac{\partial^2 u}{\partial z^2}
```

with u(0) = 0, u(1) = 0.

### Spectral Form

In Chebyshev space:

```math
\frac{d\hat{u}_n}{dt} = \kappa \sum_m D^{(2)}_{nm} \hat{u}_m
```

where D² is the Chebyshev differentiation matrix.

### With Tau Method

Last two equations replaced:

```math
\begin{aligned}
\text{Equation N-2:} \quad & T_{N-2}(0) \hat{u}_{N-2} + T_{N-1}(0) \hat{u}_{N-1} + ... = 0 \\
\text{Equation N-1:} \quad & T_{N-2}(1) \hat{u}_{N-2} + T_{N-1}(1) \hat{u}_{N-1} + ... = 0
\end{aligned}
```

## Advantages

1. **Spectral accuracy**: Exponential convergence maintained
2. **Flexibility**: Works with various BC types
3. **Clean implementation**: BCs separated from PDE

## Disadvantages

1. **Matrix conditioning**: Large matrices can be ill-conditioned
2. **Order reduction**: Effective resolution slightly reduced
3. **Complexity**: More complex than collocation methods

## Tips

### Conditioning

For high-order problems, use appropriate scaling:

```julia
# Tarang handles this automatically
# But for custom implementations, scale tau rows
```

### Debugging

Check tau field residuals:

```julia
# After solving, verify BCs are satisfied
residual = evaluate_bc(field, "z", 0.0) - bc_value
@assert abs(residual) < 1e-10
```

## Alternative: Galerkin Method

Some problems use Galerkin bases that satisfy BCs by construction:

```julia
# Sine basis for Dirichlet BCs
# Automatically satisfies u(0) = u(L) = 0
```

Tarang primarily uses tau method for flexibility.

## See Also

- [Boundary Conditions](../tutorials/boundary_conditions.md): BC types
- [Bases](bases.md): Spectral bases
- [API: Problems](../api/problems.md): Adding BCs
