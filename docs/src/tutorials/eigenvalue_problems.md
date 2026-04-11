# Tutorial: Eigenvalue Problems

This tutorial demonstrates solving eigenvalue problems (EVP) with Tarang.jl for linear stability analysis.

## Overview

Eigenvalue problems arise in linear stability analysis where we seek modes of the form:

```math
\mathbf{u}(x,z,t) = \hat{\mathbf{u}}(z) e^{i k x + \sigma t}
```

where $\sigma$ is the eigenvalue (growth rate + frequency) and $\hat{\mathbf{u}}$ is the eigenfunction.

## Basic EVP Setup

### Problem Definition

```julia
using Tarang
using MPI

MPI.Init()

# Setup domain
coords = CartesianCoordinates("z")
dist = Distributor(coords; mesh=(1,), dtype=Float64)
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))

# Create fields for eigenmodes
u_hat = ScalarField(dist, "u_hat", (z_basis,), Float64)

# Create eigenvalue problem
evp = Tarang.EVP([u_hat]; eigenvalue=:sigma)
```

### Adding Equations

The EVP format is: $A \mathbf{x} = \sigma B \mathbf{x}$

```julia
# Example: diffusion eigenvalue problem
# σu = d²u/dz² (eigenvalues of Laplacian)
Tarang.add_equation!(evp, "sigma*u_hat = ∂z(∂z(u_hat))")

# Or equivalently
Tarang.add_equation!(evp, "sigma*u_hat = Δ(u_hat)")
```

### Boundary Conditions

```julia
# Dirichlet BCs: u = 0 at boundaries
Tarang.add_equation!(evp, "u_hat(z=0) = 0")
Tarang.add_equation!(evp, "u_hat(z=1) = 0")
```

### Solving

```julia
# Validate problem setup
@assert Tarang.validate_problem(evp)

# Create solver and solve
solver = Tarang.EigenvalueSolver(evp; nev=10, which="LR")
eigenvalues, eigenvectors = Tarang.solve!(solver)

# Print results
for (i, σ) in enumerate(eigenvalues)
    println("Mode $i: σ = $(real(σ)) + $(imag(σ))im")
end
```

## Rayleigh-Bénard Stability

Classical linear stability problem for thermal convection.

### Governing Equations

Linearized Boussinesq equations for perturbations:

```math
\begin{aligned}
\sigma \hat{u} &= -i k \hat{p} + \text{Pr} \nabla^2 \hat{u} \\
\sigma \hat{w} &= -D \hat{p} + \text{Pr} \nabla^2 \hat{w} + \text{Ra} \cdot \text{Pr} \cdot \hat{T} \\
i k \hat{u} + D \hat{w} &= 0 \\
\sigma \hat{T} &= -\hat{w} + \nabla^2 \hat{T}
\end{aligned}
```

where $D = d/dz$ and $\nabla^2 = D^2 - k^2$.

### Implementation

```julia
using Tarang
using MPI

MPI.Init()

# Parameters
Ra = 1708.0  # Critical Rayleigh number
Pr = 1.0     # Prandtl number
k = 3.117    # Critical wavenumber

# Domain
coords = CartesianCoordinates("z")
dist = Distributor(coords; mesh=(1,), dtype=Float64)
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))

# Fields (complex amplitudes)
u_hat = ScalarField(dist, "u_hat", (z_basis,), ComplexF64)
w_hat = ScalarField(dist, "w_hat", (z_basis,), ComplexF64)
p_hat = ScalarField(dist, "p_hat", (z_basis,), ComplexF64)
T_hat = ScalarField(dist, "T_hat", (z_basis,), ComplexF64)

# EVP setup
evp = Tarang.EVP([u_hat, w_hat, p_hat, T_hat]; eigenvalue=:sigma)

# Add parameters
evp.parameters["Ra"] = Ra
evp.parameters["Pr"] = Pr
evp.parameters["k"] = k
evp.parameters["k2"] = k^2

# Momentum equations (modified Laplacian: D² - k²)
Tarang.add_equation!(evp, "sigma*u_hat + 1im*k*p_hat = Pr*(∂z(∂z(u_hat)) - k2*u_hat)")
Tarang.add_equation!(evp, "sigma*w_hat + ∂z(p_hat) = Pr*(∂z(∂z(w_hat)) - k2*w_hat) + Ra*Pr*T_hat")

# Continuity
Tarang.add_equation!(evp, "1im*k*u_hat + ∂z(w_hat) = 0")

# Temperature
Tarang.add_equation!(evp, "sigma*T_hat + w_hat = ∂z(∂z(T_hat)) - k2*T_hat")

# Boundary conditions (no-slip, fixed temperature)
Tarang.add_equation!(evp, "u_hat(z=0) = 0")
Tarang.add_equation!(evp, "u_hat(z=1) = 0")
Tarang.add_equation!(evp, "w_hat(z=0) = 0")
Tarang.add_equation!(evp, "w_hat(z=1) = 0")
Tarang.add_equation!(evp, "T_hat(z=0) = 0")
Tarang.add_equation!(evp, "T_hat(z=1) = 0")

# Solve
solver = Tarang.EigenvalueSolver(evp; nev=10, which="LR")
eigenvalues, eigenvectors = Tarang.solve!(solver)

# Find critical mode
max_idx = argmax(real.(eigenvalues))
sigma_crit = eigenvalues[max_idx]

println("Critical eigenvalue: σ = $(sigma_crit)")
println("Growth rate: $(real(sigma_crit))")
println("Frequency: $(imag(sigma_crit))")

MPI.Finalize()
```

## Orr-Sommerfeld Equation

Stability of parallel shear flows.

### Equation

```math
\left[ (U - c)(D^2 - k^2) - U'' + \frac{i}{\text{Re} \cdot k}(D^2 - k^2)^2 \right] \hat{\psi} = 0
```

where $c = \sigma / (ik)$ is the complex wave speed.

### Implementation

```julia
# Channel flow with parabolic profile
U(z) = 1.0 - (2z - 1)^2
U_pp(z) = -8.0  # U''

# Setup
evp = Tarang.EVP([psi_hat]; eigenvalue=:c)

# Add equation (expanded form)
Tarang.add_equation!(evp, """
    c*(∂z(∂z(psi_hat)) - k2*psi_hat) =
    U*(∂z(∂z(psi_hat)) - k2*psi_hat) - U_pp*psi_hat +
    (1im/(Re*k))*(∂z(∂z(∂z(∂z(psi_hat)))) - 2*k2*∂z(∂z(psi_hat)) + k4*psi_hat)
""")

# No-slip: ψ = ∂ψ/∂z = 0 at walls
Tarang.add_equation!(evp, "psi_hat(z=0) = 0")
Tarang.add_equation!(evp, "psi_hat(z=1) = 0")
Tarang.add_equation!(evp, "∂z(psi_hat)(z=0) = 0")
Tarang.add_equation!(evp, "∂z(psi_hat)(z=1) = 0")
```

## Parameter Studies

### Scanning Rayleigh Number

```julia
Ra_values = 10 .^ range(3, 5, length=20)
growth_rates = Float64[]

for Ra in Ra_values
    evp.parameters["Ra"] = Ra

    solver = Tarang.EigenvalueSolver(evp; nev=5, which="LR")
    eigenvalues, _ = Tarang.solve!(solver)

    push!(growth_rates, maximum(real.(eigenvalues)))
end

# Find critical Ra (where growth rate crosses zero)
using Interpolations
itp = LinearInterpolation(growth_rates, Ra_values)
Ra_crit = itp(0.0)
println("Critical Rayleigh number: $Ra_crit")
```

### Scanning Wavenumber

```julia
k_values = range(1.0, 5.0, length=50)
growth_rates = zeros(length(k_values))

for (i, k) in enumerate(k_values)
    evp.parameters["k"] = k
    evp.parameters["k2"] = k^2

    solver = Tarang.EigenvalueSolver(evp; nev=3, which="LR")
    eigenvalues, _ = Tarang.solve!(solver)

    growth_rates[i] = maximum(real.(eigenvalues))
end

# Find most unstable wavenumber
max_idx = argmax(growth_rates)
k_max = k_values[max_idx]
println("Most unstable wavenumber: $k_max")
```

## Neutral Curves

Computing the stability boundary in parameter space:

```julia
function find_neutral_curve(Ra_range, k_range)
    Ra_neutral = Float64[]
    k_neutral = Float64[]

    for k in k_range
        evp.parameters["k"] = k
        evp.parameters["k2"] = k^2

        # Binary search for neutral Ra
        Ra_lo, Ra_hi = Ra_range

        while Ra_hi - Ra_lo > 1.0
            Ra_mid = (Ra_lo + Ra_hi) / 2
            evp.parameters["Ra"] = Ra_mid

            solver = Tarang.EigenvalueSolver(evp; nev=3, which="LR")
            eigenvalues, _ = Tarang.solve!(solver)
            max_growth = maximum(real.(eigenvalues))

            if max_growth > 0
                Ra_hi = Ra_mid
            else
                Ra_lo = Ra_mid
            end
        end

        push!(k_neutral, k)
        push!(Ra_neutral, (Ra_lo + Ra_hi) / 2)
    end

    return k_neutral, Ra_neutral
end
```

## Visualizing Eigenmodes

```julia
using Plots

function plot_eigenmode(eigenvector, title="Eigenmode")
    z = get_grid(z_basis)

    # Extract components
    u_mode = real.(eigenvector["u_hat"])
    w_mode = real.(eigenvector["w_hat"])
    T_mode = real.(eigenvector["T_hat"])

    # Normalize
    max_val = maximum(abs.([u_mode; w_mode; T_mode]))
    u_mode ./= max_val
    w_mode ./= max_val
    T_mode ./= max_val

    # Plot
    p = plot(layout=(1,3), size=(900,300))
    plot!(p[1], u_mode, z, xlabel="û", ylabel="z", title="Velocity")
    plot!(p[2], w_mode, z, xlabel="ŵ", title="Vertical velocity")
    plot!(p[3], T_mode, z, xlabel="T̂", title="Temperature")

    return p
end

# Plot critical mode
plot_eigenmode(eigenvectors[max_idx], "Critical Mode")
savefig("critical_mode.png")
```

## Solver Options

### Which Eigenvalues

```julia
# Largest magnitude (default for Arnoldi)
solver = Tarang.EigenvalueSolver(evp; nev=10, which="LM")

# Largest real part (most unstable)
solver = Tarang.EigenvalueSolver(evp; nev=10, which="LR")

# Smallest magnitude
solver = Tarang.EigenvalueSolver(evp; nev=10, which="SM")

# Near a target value (shift-invert)
solver = Tarang.EigenvalueSolver(evp; nev=10, target=0.1+1.5im)
```

### Convergence

```julia
# Increase iterations for difficult problems
solver = Tarang.EigenvalueSolver(evp;
    nev=20,
    which="LR",
    tolerance=1e-10,
    max_iterations=1000
)
```

## See Also

- [Problems API](../api/problems.md): EVP problem definition
- [Solvers API](../api/solvers.md): Eigenvalue solver details
- [Rayleigh-Bénard Tutorial](ivp_2d_rbc.md): Time-dependent version
