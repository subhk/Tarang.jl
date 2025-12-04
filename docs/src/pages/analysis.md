# Analysis

Analysis tools for computing diagnostics, statistics, and derived quantities.

## CFL Condition

Compute stable timesteps based on flow velocity.

```julia
using Tarang

# Create CFL calculator
cfl = CFL(problem;
    safety=0.5,      # Safety factor (0.3-0.5 typical)
    max_change=1.5,  # Max dt increase per step
    min_change=0.5   # Max dt decrease per step
)

# Register velocity field
add_velocity!(cfl, u)

# Optional limits
cfl.max_dt = 0.01
cfl.min_dt = 1e-8

# Compute timestep
dt = compute_timestep(cfl)
```

## Global Reductions

Compute global statistics across MPI processes.

```julia
using MPI

# Create reducer
reducer = GlobalArrayReducer(MPI.COMM_WORLD)

# Maximum value
global_max = reduce_scalar(reducer, local_max, MPI.MAX)

# Sum
global_sum = reduce_scalar(reducer, local_sum, MPI.SUM)

# Mean (requires division by total elements)
global_mean = global_sum / total_elements
```

## Flow Statistics

### Kinetic Energy

```julia
function compute_kinetic_energy(u, reducer)
    local_energy = 0.0

    for component in u.components
        Tarang.ensure_layout!(component, :g)
        local_energy += sum(component.data_g.^2) / 2
    end

    return reduce_scalar(reducer, local_energy, MPI.SUM)
end
```

### Enstrophy

```julia
function compute_enstrophy(u, reducer)
    # For 2D: ω = ∂v/∂x - ∂u/∂y
    # Enstrophy = ∫ ω² dV

    ux, uy = u.components[1], u.components[2]

    # Compute vorticity (simplified)
    # ... derivative calculation ...

    return reduce_scalar(reducer, local_enstrophy, MPI.SUM)
end
```

### Reynolds Number

```julia
function compute_reynolds_number(u, nu, L, reducer)
    Tarang.ensure_layout!(u.components[1], :g)

    # RMS velocity
    local_u2 = sum(u.components[1].data_g.^2)
    global_u2 = reduce_scalar(reducer, local_u2, MPI.SUM)
    u_rms = sqrt(global_u2 / total_points)

    return u_rms * L / nu
end
```

## Heat Transfer

### Nusselt Number

```julia
function compute_nusselt(T, w, L, kappa, reducer)
    Tarang.ensure_layout!(T, :g)
    Tarang.ensure_layout!(w, :g)

    # Convective heat flux
    local_flux = sum(T.data_g .* w.data_g)
    global_flux = reduce_scalar(reducer, local_flux, MPI.SUM)

    # Normalize
    flux_mean = global_flux / total_points

    # Nusselt = 1 + convective/conductive
    Nu = 1.0 + flux_mean * L / kappa

    return Nu
end
```

## Spectral Analysis

### Energy Spectrum

```julia
function compute_spectrum(field, kmax)
    Tarang.ensure_layout!(field, :c)

    # Initialize spectrum bins
    E_k = zeros(kmax)

    # Get wavenumbers
    k = get_wavenumbers(field.bases[1])

    # Bin energy by wavenumber
    for (i, ki) in enumerate(k)
        k_bin = round(Int, abs(ki))
        if 1 <= k_bin <= kmax
            E_k[k_bin] += abs2(field.data_c[i])
        end
    end

    return E_k
end
```

### Shell-Averaged 3D Spectrum

```julia
function compute_3d_spectrum(u, kmax)
    E_k = zeros(kmax)

    for component in u.components
        Tarang.ensure_layout!(component, :c)

        kx = get_wavenumbers(component.bases[1])
        ky = get_wavenumbers(component.bases[2])
        kz = get_wavenumbers(component.bases[3])

        for i in eachindex(kx), j in eachindex(ky), k in eachindex(kz)
            k_mag = sqrt(kx[i]^2 + ky[j]^2 + kz[k]^2)
            k_bin = round(Int, k_mag)

            if 1 <= k_bin <= kmax
                E_k[k_bin] += abs2(component.data_c[i,j,k])
            end
        end
    end

    return E_k
end
```

## Time Series

### Recording Diagnostics

```julia
# Storage
times = Float64[]
energies = Float64[]
nusselts = Float64[]

# During simulation
while solver.sim_time < t_end
    step!(solver, dt)

    # Record
    push!(times, solver.sim_time)
    push!(energies, compute_kinetic_energy(u, reducer))
    push!(nusselts, compute_nusselt(T, w, L, kappa, reducer))
end
```

### Saving Time Series

```julia
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    using JLD2
    @save "diagnostics.jld2" times energies nusselts
end
```

## Spatial Averages

### Horizontal Average

```julia
function horizontal_average(field)
    Tarang.ensure_layout!(field, :g)

    # Average over x (first axis)
    mean(field.data_g, dims=1)
end
```

### Volume Average

```julia
function volume_average(field, reducer)
    Tarang.ensure_layout!(field, :g)

    local_sum = sum(field.data_g)
    global_sum = reduce_scalar(reducer, local_sum, MPI.SUM)

    return global_sum / total_points
end
```

## See Also

- [Output](../tutorials/analysis_and_output.md): File output
- [Solvers](solvers.md): Time integration
- [API: Analysis](../api/analysis.md): Complete reference
