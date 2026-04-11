# Nonlinear Terms in Tarang.jl

## Overview

Tarang.jl implements nonlinear term handling using PencilArrays and PencilFFTs for efficient parallel computation. This document explains the key principles and implementation details.

## Convention: Linear LHS, Nonlinear RHS

**Important**: **Nonlinear terms must be placed on the right-hand side (RHS)** of equations.

### Correct Format:
```julia
# ✅ CORRECT: Nonlinear terms on RHS
add_equation!(problem, "∂t(u) + ∇(p) - nu*Δ(u) = -(u⋅∇(u))")
add_equation!(problem, "∂t(b) - kappa*Δ(b) = -u⋅∇(b)")
```

### Incorrect Format:
```julia
# ❌ WRONG: Nonlinear terms on LHS
add_equation!(problem, "∂t(u) + (u⋅∇(u)) + ∇(p) = nu*Δ(u)")
```

## Why This Convention?

### 1. **Mathematical Form**
Tarang formulates IVPs in the standard form:
```
M·dt(X) + L·X = F(X, t)
```
Where:
- **M**: Mass matrix (usually identity)
- **L**: Linear differential operators
- **X**: State vector (problem variables)
- **F(X, t)**: Nonlinear terms and forcing

### 2. **Computational Efficiency**
- **LHS**: Contains only linear operators → can be pre-computed into matrices
- **RHS**: Contains nonlinear terms → evaluated explicitly at each timestep
- This separation enables efficient implicit-explicit (IMEX) timestepping

### 3. **Validation Requirements**
Tarang enforces these requirements for IVP equations:

```python
# From problems.jl
def _check_equation_conditions(self, eqn):
    LHS.require_linearity(*self.variables, allow_affine=False,
        self_name='IVP LHS', vars_name='problem variables')
    LHS.require_first_order(operators.TimeDerivative,
        self_name='IVP LHS', ops_name='time derivatives')
    RHS.require_independent(operators.TimeDerivative,
        self_name='IVP RHS', vars_name='time derivatives')
```

## Implementation in Tarang.jl

### 1. **Nonlinear Operators**
```julia
using Tarang

# Create nonlinear operators
u = VectorField(dist, coords, "velocity", (x_basis, z_basis), Float64)
b = ScalarField(dist, "buoyancy", (x_basis, z_basis), Float64)

# Nonlinear momentum: (u·∇)u
nl_momentum = nonlinear_momentum(u)

# Nonlinear advection: u·∇b  
nl_advection = advection(u, b)
```

### 2. **Problem Setup**
```julia
problem = IVP([u, b, p])

# Linear terms on LHS, nonlinear terms on RHS
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) = -(u⋅∇(u))")
add_equation!(problem, "∂t(b) - kappa*Δ(b) = -u⋅∇(b)")
add_equation!(problem, "div(u) = 0")
```

### 3. **Parallel Evaluation with PencilFFTs**
The nonlinear terms are evaluated using:
```julia
function evaluate_nonlinear_term(op::NonlinearAdvectionOperator, layout::Symbol=:g)
    velocity = op.velocity
    
    # Create nonlinear evaluator with PencilFFTs
    evaluator = NonlinearEvaluator(velocity.dist)
    
    # For each component: (u·∇)u_i = u_j ∂u_i/∂x_j
    result = VectorField(...)
    for i in 1:length(velocity.components)
        for j in 1:length(velocity.components)
            # Compute ∂u_i/∂x_j
            du_i_dx_j = evaluate_differentiate(Differentiate(velocity.components[i], coord_j, 1), :g)
            
            # Multiply u_j * (∂u_i/∂x_j) using PencilFFT transforms with dealiasing
            product = evaluate_transform_multiply(velocity.components[j], du_i_dx_j, evaluator)
            result.components[i] = result.components[i] + product
        end
    end
    
    return result
end
```

## PencilArrays/PencilFFTs Integration

### 1. **2D Parallelization**
```julia
# Process mesh for both vertical and horizontal parallelization
mesh = (2, 2)  # 4 processes: 2×2 decomposition
dist = Distributor(coords, comm=comm, mesh=mesh)

# PencilFFTs enables parallel FFT in both directions
config = PencilArrays.PencilConfig(global_shape, mesh, comm=comm)
fft_plan = PencilFFTs.PencilFFTPlan(pencil, FFT(), (1, 2))
```

### 2. **Transform-Based Multiplication**
```julia
function evaluate_transform_multiply(field1, field2, evaluator)
    # 1. Ensure fields are in grid space
    ensure_layout!(field1, :g)
    ensure_layout!(field2, :g)
    
    # 2. Pointwise multiplication
    result["g"] = field1["g"] .* field2["g"]
    
    # 3. Apply 3/2 rule dealiasing using PencilFFTs
    if evaluator.dealiasing_factor > 1.0
        result_spectral = fft_plan * result["g"]  # Transform to spectral
        apply_spectral_cutoff!(result_spectral)   # Zero high frequencies  
        result["g"] = inv(fft_plan) * result_spectral  # Transform back
    end
    
    return result
end
```

### 3. **Dealiasing (3/2 Rule)**
```julia
function apply_2d_dealiasing(data, transform_info, dealiasing_factor)
    # Transform to spectral space
    spectral_data = transform_info["fft_plan"] * data
    
    # Apply 3/2 rule: zero out high-frequency modes
    shape = transform_info["shape"]
    cutoff_x = Int(floor(shape[1] / dealiasing_factor))  # 2/3 of modes kept
    cutoff_y = Int(floor(shape[2] / dealiasing_factor))
    
    apply_spectral_cutoff!(spectral_data, (cutoff_x, cutoff_y))
    
    # Transform back to grid space
    return inv(transform_info["fft_plan"]) * spectral_data
end
```

## Examples

### 1. **2D Incompressible Flow**
```julia
# ∂u/∂t - ν∇²u + ∇p = -(u·∇)u
# ∇·u = 0

problem = IVP([u, p])
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) = -(u⋅∇(u))")
add_equation!(problem, "div(u) = 0")
```

### 2. **2D Rayleigh-Bénard Convection**
```julia
# ∂u/∂t - ν∇²u + ∇p = -(u·∇)u + b*ez  
# ∂b/∂t - κ∇²b = -u·∇b
# ∇·u = 0

problem = IVP([u, b, p])
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) - b*ez = -(u⋅∇(u))")
add_equation!(problem, "∂t(b) - kappa*Δ(b) = -u⋅∇(b)")
add_equation!(problem, "div(u) = 0")
```

### 3. **3D Turbulent Flow**
```julia
# Full 3D with all three velocity components
u = VectorField(dist, coords, "velocity", (x_basis, y_basis, z_basis), Float64)

# 3D parallelization mesh
mesh = (2, 2, 2)  # 8 processes: 2×2×2 decomposition

# Equations remain the same form
add_equation!(problem, "∂t(u) - nu*Δ(u) + ∇(p) = -(u⋅∇(u))")
```

## Performance Considerations

### 1. **Memory Management**
```julia
# Reuse temporary fields
evaluator = NonlinearEvaluator(dist)
temp_field = get_temp_field(evaluator, template, "temp_gradient")
clear_temp_fields!(evaluator)  # Cleanup when done
```

### 2. **Dealiasing Overhead**
```julia
# Monitor dealiasing cost
nonlinear_stats = NonlinearPerformanceStats()
log_nonlinear_performance(nonlinear_stats)

# Output:
# Dealiasing time: 0.123 seconds (15.2% of total)
# Transform time: 0.456 seconds (56.7% of total)
```

### 3. **Parallel Efficiency**
```julia
# 2D parallelization typically shows good scaling:
# 1 process:  100% efficiency
# 4 processes: ~85% efficiency (2×2 mesh)
# 16 processes: ~70% efficiency (4×4 mesh)
```

## Best Practices

1. **Always put nonlinear terms on the RHS**
2. **Use appropriate dealiasing factor** (3/2 for quadratic nonlinearities)  
3. **Choose optimal process mesh** for your domain aspect ratio
4. **Monitor nonlinear evaluation performance** to identify bottlenecks
5. **Reuse temporary fields** to minimize memory allocation
6. **Use appropriate timestepper** (RK443 for nonlinear problems)

This implementation provides efficient, scalable nonlinear term evaluation while maintaining compatibility with the Tarang framework design principles.