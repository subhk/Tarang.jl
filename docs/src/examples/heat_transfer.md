# Heat Transfer Examples

Collection of heat transfer simulations with Tarang.jl.

## Pure Diffusion

### 1D Heat Equation

```julia
using Tarang, MPI
MPI.Init()

coords = CartesianCoordinates("x")
dist = Distributor(coords; mesh=(1,), dtype=Float64)
basis = ChebyshevT(coords["x"]; size=64, bounds=(0.0, 1.0))

T = ScalarField(dist, "T", (basis,), Float64)

problem = IVP([T])
problem.parameters["kappa"] = 0.01

Tarang.add_equation!(problem, "∂t(T) = kappa*Δ(T)")
Tarang.add_equation!(problem, "T(x=0) = 1")  # Hot
Tarang.add_equation!(problem, "T(x=1) = 0")  # Cold

solver = InitialValueSolver(problem, CNAB2(); dt=0.01)

# Initial condition: step function
Tarang.ensure_layout!(T, :g)
x = get_grid(basis)
T.data_g .= x .< 0.5

# Solve to steady state
while solver.sim_time < 10.0
    step!(solver)
end
# Steady state: linear profile

MPI.Finalize()
```

### 2D Diffusion

```julia
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(2, 2), dtype=Float64)

x_basis = ChebyshevT(coords["x"]; size=64, bounds=(0.0, 1.0))
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))

T = ScalarField(dist, "T", (x_basis, z_basis), Float64)

problem = IVP([T])
problem.parameters["kappa"] = 0.1

Tarang.add_equation!(problem, "∂t(T) = kappa*Δ(T)")

# Dirichlet on all boundaries
Tarang.add_equation!(problem, "T(x=0) = 0")
Tarang.add_equation!(problem, "T(x=1) = 0")
Tarang.add_equation!(problem, "T(z=0) = 1")
Tarang.add_equation!(problem, "T(z=1) = 0")
```

## Convection-Diffusion

### Advection by Known Velocity

```julia
# Given velocity field U (not solved)
U = 1.0  # Constant advection velocity

problem = IVP([T])
problem.parameters["U"] = U
problem.parameters["kappa"] = 0.01

Tarang.add_equation!(problem, "∂t(T) + U*∂x(T) = kappa*Δ(T)")
```

### Natural Convection

Buoyancy-driven flow with temperature coupling.

```julia
problem = IVP([ux, uz, p, T])
problem.parameters["Ra"] = 1e5
problem.parameters["Pr"] = 0.7

Tarang.add_equation!(problem, "∂t(ux) + ... + ∂x(p) = Pr*Δ(ux)")
Tarang.add_equation!(problem, "∂t(uz) + ... + ∂z(p) = Pr*Δ(uz) + Ra*Pr*T")
Tarang.add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")
Tarang.add_equation!(problem, "∂t(T) + ux*∂x(T) + uz*∂z(T) = Δ(T)")
```

## Boundary Conditions

### Convective (Robin) BC

Heat transfer to environment.

```julia
# h*T + k*∂T/∂n = h*T_ambient
h = 10.0   # Heat transfer coefficient
k = 1.0    # Thermal conductivity
T_amb = 0.0

Tarang.add_equation!(problem, "$(h)*T(z=1) + $(k)*∂z(T)(z=1) = $(h*T_amb)")
```

### Insulated (Neumann) BC

Zero heat flux.

```julia
Tarang.add_equation!(problem, "∂z(T)(z=1) = 0")  # ∂T/∂z = 0
```

### Time-Varying BC

```julia
# Oscillating temperature boundary
function update_bc!(problem, t)
    T_bc = sin(2π * t)
    # Update boundary condition value
end
```

## Steady-State Problems

### Laplace Equation

Steady heat conduction.

```julia
problem = LBVP([T])
Tarang.add_equation!(problem, "Δ(T) = 0")

# Boundary conditions define the solution
Tarang.add_equation!(problem, "T(x=0) = 0")
Tarang.add_equation!(problem, "T(x=1) = 1")
Tarang.add_equation!(problem, "∂z(T)(z=0) = 0")
Tarang.add_equation!(problem, "∂z(T)(z=1) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
```

### Poisson Equation

With heat source.

```julia
# Heat source (e.g., volumetric heating)
q = ScalarField(dist, "q", bases, Float64)
Tarang.ensure_layout!(q, :g)
# Define heat source distribution
q.data_g .= sin.(π .* x) .* sin.(π .* z)

problem = LBVP([T])
problem.fields["q"] = q
Tarang.add_equation!(problem, "Δ(T) = -q")
```

## Heat Transfer Analysis

### Nusselt Number

```julia
function compute_nusselt(T, direction, H, kappa)
    Tarang.ensure_layout!(T, :g)

    # Heat flux at boundary
    dT_dz = compute_gradient(T, direction)
    q_wall = -kappa * mean(dT_dz[boundary])

    # Conductive reference
    Delta_T = 1.0  # Temperature difference
    q_cond = kappa * Delta_T / H

    return q_wall / q_cond
end
```

### Thermal Boundary Layer

```julia
function boundary_layer_thickness(T, z_grid, T_wall, T_bulk)
    # Find z where T = 0.99 * (T_bulk - T_wall) + T_wall
    profile = mean(T, dims=1)  # x-averaged
    # ... interpolation to find δ_T
end
```

## Multi-Physics

### Conjugate Heat Transfer

Heat transfer across solid-fluid interface.

```julia
# Solid domain
T_solid = ScalarField(dist_solid, "T_s", bases_solid, Float64)

# Fluid domain
T_fluid = ScalarField(dist_fluid, "T_f", bases_fluid, Float64)

# Coupling at interface:
# T_s = T_f (continuity)
# k_s * ∂T_s/∂n = k_f * ∂T_f/∂n (flux balance)
```

### Melting/Solidification

Phase change with enthalpy method.

```julia
# Enthalpy formulation
# ∂H/∂t = ∇·(k∇T)
# H = c_p * T + L * f_l (liquid fraction)
```

## Tips

### Numerical Stability

- Diffusion is always stable with implicit methods
- Advection-diffusion: use IMEX methods
- High Peclet number: may need upwinding or higher resolution

### Convergence

- Exponential convergence for smooth solutions
- Check spectral coefficient decay
- Refine near boundaries for boundary layers

## See Also

- [Rayleigh-Bénard Tutorial](../tutorials/ivp_2d_rbc.md)
- [Fluid Dynamics Examples](fluid_dynamics.md)
- [Example Gallery](gallery.md)
