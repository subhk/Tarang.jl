# Tutorial: Analysis and Output

This tutorial covers saving simulation data and computing diagnostics in Tarang.jl.

## File Output

Tarang.jl supports multiple output formats for saving simulation data.

### NetCDF Output

NetCDF is the recommended format for parallel simulations.

```julia
using Tarang

# Create output handler
handler = add_file_handler(
    "simulation_output",   # Base filename
    dist,                  # Distributor
    Dict("u" => u, "T" => T);  # Fields to track
    parallel="gather",     # Parallel I/O mode
    max_writes=100         # Files per handler
)

# Add field outputs
add_task(handler, u; name="velocity")
add_task(handler, T; name="temperature")

# Process output at each timestep
process!(handler;
    iteration=solver.iteration,
    wall_time=solver.wall_time,
    sim_time=solver.sim_time,
    timestep=solver.dt
)
```

### Output Modes

```julia
# Gather mode: All data collected to root process
handler = add_file_handler(path, dist, fields; parallel="gather")

# Virtual mode: Each process writes its own file
handler = add_file_handler(path, dist, fields; parallel="virtual")
```

### Output Frequency

```julia
# Write every N iterations
if solver.iteration % output_cadence == 0
    process!(handler; iteration=solver.iteration, ...)
end

# Write at specific simulation times
if solver.sim_time >= next_output_time
    process!(handler; sim_time=solver.sim_time, ...)
    next_output_time += output_interval
end
```

## Analysis Tasks

### Computing Means

```julia
# Horizontal average
add_mean_task!(handler, T; dims=1, name="T_mean_x")

# Vertical average
add_mean_task!(handler, T; dims=2, name="T_mean_z")

# Full spatial average (scalar)
add_mean_task!(handler, T; dims=(1,2), name="T_avg")
```

### Extracting Slices

```julia
# Slice at specific index
add_slice_task!(handler, T; dim=1, idx=64, name="T_slice_x")

# Multiple slices
add_slice_task!(handler, T; dim=2, idx=1, name="T_bottom")
add_slice_task!(handler, T; dim=2, idx=Nz, name="T_top")
```

### Custom Analysis

```julia
# Define custom postprocessing function
function kinetic_energy(u_data)
    return 0.5 * sum(u_data.^2)
end

# Add custom task
add_task(handler, u;
    name="kinetic_energy",
    postprocess=kinetic_energy
)
```

## Global Diagnostics

### CFL Condition

```julia
using Tarang

# Create CFL calculator
cfl = CFL(problem; safety=0.5, max_change=1.5, min_change=0.5)

# Register velocity field
add_velocity!(cfl, u)

# Compute adaptive timestep
dt = compute_timestep(cfl)
```

### Flow Statistics

```julia
using Tarang

# Create global reducer for MPI operations
reducer = GlobalArrayReducer(dist.comm)

# Compute global max
u_max = reduce_scalar(reducer, maximum(abs.(u.data_g)), MPI.MAX)

# Compute global mean
u_mean = reduce_scalar(reducer, mean(u.data_g), MPI.SUM) / dist.size

# Compute global energy
function global_energy(u, reducer)
    local_energy = sum(u.data_g.^2) / 2
    return reduce_scalar(reducer, local_energy, MPI.SUM)
end
```

### Reynolds Number

```julia
function compute_reynolds_number(u, nu, L)
    ensure_layout!(u, :g)

    # RMS velocity
    u_rms = sqrt(mean(u.data_g.^2))

    # Reynolds number
    Re = u_rms * L / nu

    return Re
end
```

### Nusselt Number

For thermal convection:

```julia
function compute_nusselt(T, uz, L, kappa)
    ensure_layout!(T, :g)
    ensure_layout!(uz, :g)

    # Convective heat flux
    flux_conv = mean(T.data_g .* uz.data_g)

    # Conductive flux (from temperature gradient)
    dT = 1.0  # Temperature difference

    # Nusselt = total flux / conductive flux
    Nu = 1.0 + flux_conv * L / (kappa * dT)

    return Nu
end
```

## Energy Spectra

### 1D Spectrum

```julia
function compute_1d_spectrum(u, axis)
    ensure_layout!(u, :c)  # Spectral space

    data = u.data_c
    N = size(data, axis)

    # Sum over other dimensions
    spectrum = zeros(N÷2)
    for k in 1:N÷2
        spectrum[k] = sum(abs2.(selectdim(data, axis, k)))
    end

    return spectrum
end
```

### Shell-Averaged Spectrum

```julia
function compute_shell_spectrum(u, kmax)
    ensure_layout!(u, :c)

    # Initialize spectrum bins
    E_k = zeros(kmax)
    counts = zeros(Int, kmax)

    # Get wavenumbers
    kx = get_wavenumbers(u.bases[1])
    ky = get_wavenumbers(u.bases[2])
    kz = get_wavenumbers(u.bases[3])

    # Bin energy by wavenumber magnitude
    for i in eachindex(kx), j in eachindex(ky), k in eachindex(kz)
        k_mag = sqrt(kx[i]^2 + ky[j]^2 + kz[k]^2)
        k_bin = round(Int, k_mag)

        if 1 <= k_bin <= kmax
            E_k[k_bin] += abs2(u.data_c[i,j,k])
            counts[k_bin] += 1
        end
    end

    return E_k
end
```

## Time Series

### Storing Scalar Time Series

```julia
# Initialize storage
times = Float64[]
energies = Float64[]
nusselts = Float64[]

# During simulation
while solver.sim_time < t_end
    step!(solver, dt)

    # Record diagnostics
    push!(times, solver.sim_time)
    push!(energies, compute_kinetic_energy(u))
    push!(nusselts, compute_nusselt(T, uz, L, kappa))
end

# Save to file
if rank == 0
    using JLD2
    @save "diagnostics.jld2" times energies nusselts
end
```

## Visualization Integration

### Plots.jl

```julia
using Plots

function plot_field(field, title="")
    ensure_layout!(field, :g)
    data = field.data_g

    heatmap(data',
        xlabel="x", ylabel="z",
        title=title,
        colorbar=true
    )
end

# Save figure
plot_field(T, "Temperature")
savefig("temperature.png")
```

### Makie.jl

```julia
using CairoMakie

function plot_field_makie(field)
    ensure_layout!(field, :g)
    data = field.data_g

    fig = Figure()
    ax = Axis(fig[1,1], xlabel="x", ylabel="z")
    hm = heatmap!(ax, data')
    Colorbar(fig[1,2], hm)

    return fig
end
```

## Checkpointing

### Saving State

```julia
function save_checkpoint(solver, filename)
    state = Dict(
        "sim_time" => solver.sim_time,
        "iteration" => solver.iteration,
        "dt" => solver.dt,
        "fields" => Dict()
    )

    for (name, field) in solver.problem.fields
        ensure_layout!(field, :c)
        state["fields"][name] = copy(field.data_c)
    end

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        using JLD2
        @save filename state
    end
end
```

### Loading State

```julia
function load_checkpoint!(solver, filename)
    using JLD2
    @load filename state

    solver.sim_time = state["sim_time"]
    solver.iteration = state["iteration"]
    solver.dt = state["dt"]

    for (name, data) in state["fields"]
        field = solver.problem.fields[name]
        field.data_c .= data
        field.current_layout = :c
    end
end
```

## Complete Example

```julia
using Tarang
using MPI
using Printf

MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# Setup (abbreviated)
# ... create domain, fields, problem ...

# Output handler
handler = add_file_handler("output", dist, Dict("T" => T);
    parallel="gather", max_writes=100)
add_task(handler, T; name="temperature")
add_mean_task!(handler, T; dims=1, name="T_mean")

# Create solver
solver = InitialValueSolver(problem, RK222(), dt=1e-3)

# CFL
cfl = CFL(problem, safety=0.5)
add_velocity!(cfl, u)

# Diagnostics storage
times, energies = Float64[], Float64[]

# Main loop
output_interval = 0.1
next_output = output_interval

while solver.sim_time < 10.0
    dt = compute_timestep(cfl)
    step!(solver, dt)

    # Store diagnostics
    push!(times, solver.sim_time)
    push!(energies, compute_kinetic_energy(u))

    # Periodic output
    if solver.sim_time >= next_output
        process!(handler;
            iteration=solver.iteration,
            sim_time=solver.sim_time,
            wall_time=0.0,
            timestep=dt
        )

        if rank == 0
            @printf("t = %.3f, E = %.6e\n", solver.sim_time, energies[end])
        end

        next_output += output_interval
    end
end

MPI.Finalize()
```

## See Also

- [Analysis API](../api/analysis.md): Full API reference
- [I/O API](../api/io.md): NetCDF output documentation
- [Parallelism](../pages/parallelism.md): Parallel I/O configuration
