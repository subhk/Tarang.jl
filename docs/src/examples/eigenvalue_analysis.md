# Eigenvalue Analysis Examples

Collection of linear stability and eigenvalue problem examples with Tarang.jl.

## Linear Stability Overview

Eigenvalue problems determine the stability of base states by solving:

```math
A \mathbf{x} = \sigma B \mathbf{x}
```

where σ is the eigenvalue (growth rate + frequency).

## Rayleigh-Bénard Stability

Critical Rayleigh number for convection onset.

```julia
using Tarang, MPI
MPI.Init()

# Parameters
Pr = 1.0
k = 3.117  # Critical wavenumber

# Domain
coords = CartesianCoordinates("z")
dist = Distributor(coords; mesh=(1,), dtype=Float64)
z_basis = ChebyshevT(coords["z"]; size=64, bounds=(0.0, 1.0))

# Fields (complex amplitudes)
u_hat = ScalarField(dist, "u_hat", (z_basis,), ComplexF64)
w_hat = ScalarField(dist, "w_hat", (z_basis,), ComplexF64)
p_hat = ScalarField(dist, "p_hat", (z_basis,), ComplexF64)
T_hat = ScalarField(dist, "T_hat", (z_basis,), ComplexF64)

# EVP
evp = Tarang.EVP([u_hat, w_hat, p_hat, T_hat]; eigenvalue=:sigma)
evp.parameters["Pr"] = Pr
evp.parameters["k2"] = k^2

# Scan Ra to find critical value
function find_critical_Ra(evp, Ra_range)
    for Ra in Ra_range
        evp.parameters["Ra"] = Ra
        solver = Tarang.EigenvalueSolver(evp; nev=5, which="LR")
        eigenvalues, _ = Tarang.solve!(solver)
        max_growth = maximum(real.(eigenvalues))

        println("Ra = $Ra, σ_max = $max_growth")

        if max_growth > 0
            return Ra  # Found critical Ra
        end
    end
end

Ra_crit = find_critical_Ra(evp, 1000:100:2000)
println("Critical Ra ≈ $Ra_crit")

MPI.Finalize()
```

## Orr-Sommerfeld (Plane Poiseuille)

Stability of channel flow.

```julia
# Base flow
U(z) = 1 - (2z - 1)^2  # Parabolic profile
U_pp(z) = -8.0          # U''

# Parameters
Re = 5000
k = 1.0
k2 = k^2

# EVP for complex wave speed c
evp = Tarang.EVP([psi_hat]; eigenvalue=:c)

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

## Neutral Curves

Compute stability boundary in parameter space.

```julia
function compute_neutral_curve(evp, k_range, Ra_range)
    k_neutral = Float64[]
    Ra_neutral = Float64[]

    for k in k_range
        evp.parameters["k"] = k
        evp.parameters["k2"] = k^2

        # Binary search for neutral Ra
        Ra_lo, Ra_hi = Ra_range
        while Ra_hi - Ra_lo > 10
            Ra_mid = (Ra_lo + Ra_hi) / 2
            evp.parameters["Ra"] = Ra_mid

            solver = Tarang.EigenvalueSolver(evp; nev=3, which="LR")
            eigenvalues, _ = Tarang.solve!(solver)
            growth = maximum(real.(eigenvalues))

            if growth > 0
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

k_range = range(2.0, 4.5, length=20)
k_n, Ra_n = compute_neutral_curve(evp, k_range, (1000, 3000))

# Find minimum (critical point)
idx = argmin(Ra_n)
k_crit, Ra_crit = k_n[idx], Ra_n[idx]
```

## Taylor-Couette Stability

Rotating cylinder flow stability.

```julia
# Parameters
eta = 0.5  # Radius ratio
Omega_i = 1.0  # Inner cylinder rotation
Omega_o = 0.0  # Outer cylinder rotation
Re = 100

# Cylindrical coordinates (simplified to 1D)
evp = Tarang.EVP([u_r, u_theta, u_z, p]; eigenvalue=:sigma)

# Base flow: Couette profile
# Add perturbation equations with centrifugal effects
```

## Thermal Instabilities

### Marangoni Convection

Surface tension driven flow.

```julia
evp.parameters["Ma"] = 1000  # Marangoni number
evp.parameters["Bi"] = 1.0   # Biot number

# Free surface boundary conditions
# Include surface tension gradient terms
```

### Double-Diffusive Convection

Two buoyancy sources (temperature and salinity).

```julia
evp.parameters["Ra_T"] = 1e6   # Thermal Rayleigh
evp.parameters["Ra_S"] = 5e5   # Solutal Rayleigh
evp.parameters["tau"] = 0.01   # Diffusivity ratio

# Additional equation for salinity perturbation
```

## Visualizing Eigenmodes

```julia
function plot_eigenmode(eigenvector, z_grid)
    using Plots

    u = real.(eigenvector["u_hat"])
    w = real.(eigenvector["w_hat"])
    T = real.(eigenvector["T_hat"])

    # Normalize
    max_val = maximum(abs.([u; w; T]))
    u ./= max_val
    w ./= max_val
    T ./= max_val

    p = plot(layout=(1,3), size=(900,300))
    plot!(p[1], u, z_grid, xlabel="û", ylabel="z")
    plot!(p[2], w, z_grid, xlabel="ŵ")
    plot!(p[3], T, z_grid, xlabel="T̂")

    return p
end
```

## Solver Options

### Finding Specific Eigenvalues

```julia
# Most unstable (largest real part)
solver = Tarang.EigenvalueSolver(evp; nev=10, which="LR")

# Largest magnitude
solver = Tarang.EigenvalueSolver(evp; nev=10, which="LM")

# Near a target value
solver = Tarang.EigenvalueSolver(evp; nev=5, target=0.1+1.5im)

# Most oscillatory (largest imaginary)
solver = Tarang.EigenvalueSolver(evp; nev=10, which="LI")
```

### Convergence

```julia
solver = Tarang.EigenvalueSolver(evp;
    nev=20,
    which="LR",
    tolerance=1e-12,
    max_iterations=2000
)
```

## Tips

### Resolution

- Start coarse (N=32), refine until eigenvalues converge
- Chebyshev provides exponential convergence
- Boundary layer modes need more resolution

### Spurious Modes

- Check mode structure (spurious modes often oscillate wildly)
- Verify with different N
- Check energy balance

### Physical Validation

- Compare with known analytical results
- Check limiting cases
- Verify neutral curve shape

## See Also

- [Eigenvalue Problems Tutorial](../tutorials/eigenvalue_problems.md)
- [API: Solvers](../api/solvers.md)
- [Example Gallery](gallery.md)
