# Heat Transfer Examples

Collection of heat transfer simulations with Tarang.jl.

## Pure Diffusion

### 1D Heat Equation

```julia
using Tarang, MPI
MPI.Init()

coords = CartesianCoordinates("x")
dist = Distributor(coords; mesh=(1,), dtype=Float64, device=CPU())
basis = ChebyshevT(coords["x"]; size=64, bounds=(0.0, 1.0))

T = ScalarField(dist, "T", (basis,), Float64)

problem = IVP([T])
problem.namespace["kappa"] = 0.01

Tarang.add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")
Tarang.add_equation!(problem, "T(x=0) = 1")  # Hot
Tarang.add_equation!(problem, "T(x=1) = 0")  # Cold

solver = InitialValueSolver(problem, CNAB2(); dt=0.01)

# Initial condition: step function
Tarang.ensure_layout!(T, :g)
x = get_grid(basis)
get_grid_data(T) .= x .< 0.5

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
dist = Distributor(coords; mesh=(2, 2), dtype=Float64, device=CPU())

x_basis = ChebyshevT(coords["x"]; size=64, bounds=(0.0, 1.0))
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))

T = ScalarField(dist, "T", (x_basis, z_basis), Float64)

problem = IVP([T])
problem.namespace["kappa"] = 0.1

Tarang.add_equation!(problem, "∂t(T) - kappa*Δ(T) = 0")

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
problem.namespace["U"] = U
problem.namespace["kappa"] = 0.01

Tarang.add_equation!(problem, "∂t(T) + U*∂x(T) - kappa*Δ(T) = 0")
```

### Natural Convection

Buoyancy-driven flow with temperature coupling.

```julia
problem = IVP([ux, uz, p, T])
problem.namespace["Ra"] = 1e5
problem.namespace["Pr"] = 0.7

Tarang.add_equation!(problem, "∂t(ux) + ... + ∂x(p) - Pr*Δ(ux) = 0")
Tarang.add_equation!(problem, "∂t(uz) + ... + ∂z(p) - Pr*Δ(uz) - Ra*Pr*T = 0")
Tarang.add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")
Tarang.add_equation!(problem, "∂t(T) - Δ(T) = -ux*∂x(T) - uz*∂z(T)")
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

Steady heat conduction. This is a linear boundary value problem (LBVP): boundary
conditions are enforced with the **tau method** (one `tau` variable per BC, lifted
into the bulk equation and declared with `add_bc!`). The bounded direction here is
`z` (Chebyshev); the periodic `x` direction is separable.

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 1.0))   # periodic (separable)
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))   # bounded  (coupled)
dom = Domain(dist, (xb, zb))

T    = ScalarField(dom, "T")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)            # one tau per z-BC
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

problem = LBVP([T, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2))
Tarang.add_equation!(problem, "Δ(T) + l1 + l2 = 0")

# Boundary conditions on the bounded (z) direction, via the tau method
Tarang.add_bc!(problem, "T(z=0)   = 1")
Tarang.add_bc!(problem, "T(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
Tarang.ensure_layout!(T, :g)            # scatter writes coefficients; switch to grid
```

!!! note "1D pure-Chebyshev BVP"
    The example keeps a periodic `x` direction, but a pure single-axis Chebyshev
    BVP (no Fourier) also works — drop the `x` axis and put the `tau` variables on
    `()`. The solver builds one coupled tau subproblem over the Chebyshev spectrum.

### Poisson Equation

With heat source. Like the Laplace example this is a steady LBVP, so the wall
boundary conditions use the **tau method** (`tau` variables + `lift` + `add_bc!`).

```julia
coords = CartesianCoordinates("x", "z")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb = RealFourier(coords["x"]; size=4,  bounds=(0.0, 1.0))   # periodic (separable)
zb = ChebyshevT(coords["z"];  size=16, bounds=(0.0, 1.0))   # bounded  (coupled)
dom = Domain(dist, (xb, zb))

T    = ScalarField(dom, "T")
tau1 = ScalarField(dist, "tau1", (xb,), Float64)            # one tau per z-BC
tau2 = ScalarField(dist, "tau2", (xb,), Float64)
lb2  = derivative_basis(zb, 2)

# Heat source (e.g., volumetric heating)
q  = ScalarField(dom, "q")
Tarang.ensure_layout!(q, :g)
zg = create_meshgrid(dom; on_device=false)["z"]
get_grid_data(q) .= sin.(π .* zg)                           # source distribution

problem = LBVP([T, tau1, tau2])
add_parameters!(problem; l1=lift(tau1, lb2, -1), l2=lift(tau2, lb2, -2), q=q)
Tarang.add_equation!(problem, "Δ(T) + l1 + l2 = -q")
Tarang.add_bc!(problem, "T(z=0)   = 0")
Tarang.add_bc!(problem, "T(z=1.0) = 0")

solver = BoundaryValueSolver(problem)
solve!(solver)
Tarang.ensure_layout!(T, :g)
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
