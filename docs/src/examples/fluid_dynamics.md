# Fluid Dynamics Examples

Collection of fluid dynamics simulations with Tarang.jl.

## Incompressible Navier-Stokes

### 2D Lid-Driven Cavity

Classic benchmark for viscous flow.

```julia
using Tarang, MPI
MPI.Init()

Re = 1000
Lx, Lz = 1.0, 1.0
Nx, Nz = 128, 128

coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(1,), dtype=Float64)

x_basis = ChebyshevT(coords["x"]; size=Nx, bounds=(0.0, Lx))
z_basis = ChebyshevT(coords["z"]; size=Nz, bounds=(0.0, Lz))

ux = ScalarField(dist, "ux", (x_basis, z_basis), Float64)
uz = ScalarField(dist, "uz", (x_basis, z_basis), Float64)
p = ScalarField(dist, "p", (x_basis, z_basis), Float64)

problem = IVP([ux, uz, p])
problem.parameters["nu"] = 1.0/Re

Tarang.add_equation!(problem, "∂t(ux) + ux*∂x(ux) + uz*∂z(ux) + ∂x(p) = nu*Δ(ux)")
Tarang.add_equation!(problem, "∂t(uz) + ux*∂x(uz) + uz*∂z(uz) + ∂z(p) = nu*Δ(uz)")
Tarang.add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")

# Bottom, left, right walls: no-slip
Tarang.add_equation!(problem, "ux(z=0) = 0")
Tarang.add_equation!(problem, "ux(x=0) = 0")
Tarang.add_equation!(problem, "ux(x=$Lx) = 0")
# Top wall: moving lid
Tarang.add_equation!(problem, "ux(z=$Lz) = 1")
# All walls: no penetration
Tarang.add_equation!(problem, "uz(z=0) = 0")
Tarang.add_equation!(problem, "uz(z=$Lz) = 0")
Tarang.add_equation!(problem, "uz(x=0) = 0")
Tarang.add_equation!(problem, "uz(x=$Lx) = 0")

solver = InitialValueSolver(problem, SBDF2(); dt=1e-3)
# ... time integration ...

MPI.Finalize()
```

### Kelvin-Helmholtz Instability

Shear layer instability in periodic domain.

```julia
# Setup
coords = CartesianCoordinates("x", "z")
dist = Distributor(coords; mesh=(2, 2), dtype=Float64)

x_basis = RealFourier(coords["x"]; size=256, bounds=(0.0, 1.0))
z_basis = RealFourier(coords["z"]; size=256, bounds=(0.0, 1.0))

# Initial shear layer with perturbation
function init_kh!(ux, uz, δ, A)
    x = get_grid(ux.bases[1])
    z = get_grid(ux.bases[2])

    Tarang.ensure_layout!(ux, :g)
    Tarang.ensure_layout!(uz, :g)

    for i in eachindex(x), j in eachindex(z)
        ux.data_g[i,j] = tanh((z[j] - 0.5) / δ)
        uz.data_g[i,j] = A * sin(2π * x[i]) * exp(-((z[j]-0.5)/δ)^2)
    end
end

init_kh!(ux, uz, 0.05, 0.01)
```

## Thermal Convection

### Double-Diffusive Convection

Convection with both temperature and salinity.

```julia
problem = IVP([ux, uz, p, T, S])
problem.parameters["Pr"] = 7.0    # Prandtl
problem.parameters["tau"] = 0.01  # Diffusivity ratio
problem.parameters["Ra_T"] = 1e6  # Thermal Rayleigh
problem.parameters["Ra_S"] = 1e5  # Solutal Rayleigh

Tarang.add_equation!(problem, "∂t(ux) + ... = Pr*Δ(ux)")
Tarang.add_equation!(problem, "∂t(uz) + ... = Pr*Δ(uz) + Ra_T*Pr*T - Ra_S*Pr*S")
Tarang.add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")
Tarang.add_equation!(problem, "∂t(T) + ... = Δ(T)")
Tarang.add_equation!(problem, "∂t(S) + ... = tau*Δ(S)")
```

### Rotating Convection

With Coriolis force.

```julia
problem.parameters["Ek"] = 1e-4   # Ekman number
problem.parameters["Ra"] = 1e7

# Include Coriolis term: 2Ω × u
Tarang.add_equation!(problem,
    "∂t(ux) + ... = Pr*Δ(ux) - (2/Ek)*uy")
Tarang.add_equation!(problem,
    "∂t(uy) + ... = Pr*Δ(uy) + (2/Ek)*ux")
```

## Stratified Flows

### Internal Gravity Waves

```julia
problem = IVP([ux, uz, p, b])  # b = buoyancy
problem.parameters["N2"] = 1.0  # Brunt-Väisälä frequency squared

Tarang.add_equation!(problem, "∂t(ux) + ... + ∂x(p) = nu*Δ(ux)")
Tarang.add_equation!(problem, "∂t(uz) + ... + ∂z(p) = nu*Δ(uz) + b")
Tarang.add_equation!(problem, "∂x(ux) + ∂z(uz) = 0")
Tarang.add_equation!(problem, "∂t(b) + N2*uz + ... = kappa*Δ(b)")
```

## Turbulence

### Homogeneous Isotropic Turbulence

Forced turbulence in triply-periodic box.

```julia
# Add forcing to low wavenumbers
function add_forcing!(problem)
    # Forcing term F added to momentum equations
    # Target specific wavenumber shells
end

# Monitor energy spectrum
function energy_spectrum(u)
    # Shell-averaged E(k)
end

# Monitor dissipation
function dissipation_rate(u, nu)
    # ε = 2ν⟨S_ij S_ij⟩
end
```

### Decaying Turbulence

Study energy cascade without forcing.

```julia
# Initialize with random velocity field
# Filter to specific wavenumber range
# Watch energy decay and cascade
```

## Tips

### Resolution Requirements

| Problem | Typical Resolution |
|---------|-------------------|
| Laminar | 64-128 |
| Transitional | 128-256 |
| Turbulent (moderate Re) | 256-512 |
| Turbulent (high Re) | 512-2048+ |

### Timestepping

- Use `RK443` for advection-dominated flows
- Use `SBDF2/3` for diffusion-dominated or stiff problems
- Always use CFL-based adaptive stepping

### Diagnostics

Always monitor:
- Kinetic energy
- Maximum velocity (CFL check)
- Divergence (should be ~machine precision)
- Physical observables (Nu, drag, etc.)

## See Also

- [Rayleigh-Bénard Tutorial](../tutorials/ivp_2d_rbc.md)
- [Heat Transfer Examples](heat_transfer.md)
- [Example Gallery](gallery.md)
