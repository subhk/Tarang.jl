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
add_task!(handler, u; name="velocity")
add_task!(handler, T; name="temperature")

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
add_task!(handler, u;
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
u_max = reduce_scalar(reducer, maximum(abs.(get_grid_data(u))), MPI.MAX)

# Compute global mean
u_mean = reduce_scalar(reducer, mean(get_grid_data(u)), MPI.SUM) / dist.size

# Compute global energy
function global_energy(u, reducer)
    local_energy = sum(get_grid_data(u).^2) / 2
    return reduce_scalar(reducer, local_energy, MPI.SUM)
end
```

### Reynolds Number

```julia
function compute_reynolds_number(u, nu, L)
    ensure_layout!(u, :g)

    # RMS velocity
    u_rms = sqrt(mean(get_grid_data(u).^2))

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
    flux_conv = mean(get_grid_data(T) .* get_grid_data(uz))

    # Conductive flux (from temperature gradient)
    dT = 1.0  # Temperature difference

    # Nusselt = total flux / conductive flux
    Nu = 1.0 + flux_conv * L / (kappa * dT)

    return Nu
end
```

## Energy Spectra

Use the built-in `power_spectrum` (radially-binned shell spectrum of a scalar
field with at least one Fourier basis) — no need to hand-roll wavenumber binning.
It returns a NamedTuple `(k, power, bin_edges)`.

```julia
ps = power_spectrum(u)                  # u::ScalarField with a Fourier axis

using Plots
plot(ps.k, ps.power; xscale=:log10, yscale=:log10, xlabel="k", ylabel="E(k)")

# For a velocity VectorField, the enstrophy (vorticity) spectrum:
es = enstrophy_spectrum(velocity)       # → (k, power, bin_edges)
```

See the [Analysis API](../api/analysis.md#Spectral-Analysis) for options
(`max_wavenumber`, `radial_average`, binning).

## Time Series

### Storing Scalar Time Series

```julia
# Initialize storage
times = Float64[]
energies = Float64[]
nusselts = Float64[]

# Record diagnostics each step via a run! callback (Int interval 1 = every step;
# use a larger Int for every-N-steps, or a Float for every-Δt sim-time).
# global_energy / compute_nusselt are the helpers defined under Global Diagnostics.
run!(solver; stop_time=t_end,
     callbacks=[1 => function (s)
         push!(times, s.sim_time)
         push!(energies, global_energy(u, reducer))
         push!(nusselts, compute_nusselt(T, uz, L, kappa))
     end])

# Save to file (plain vectors → any format; here a CSV via the stdlib)
if rank == 0
    using DelimitedFiles
    writedlm("diagnostics.csv", [times energies nusselts], ',')
end
```

## Visualization Integration

### Plots.jl

```julia
using Plots

function plot_field(field, title="")
    ensure_layout!(field, :g)
    data = get_grid_data(field)

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
    data = get_grid_data(field)

    fig = Figure()
    ax = Axis(fig[1,1], xlabel="x", ylabel="z")
    hm = heatmap!(ax, data')
    Colorbar(fig[1,2], hm)

    return fig
end
```

## Checkpointing

Tarang has no built-in checkpoint type — write a small helper over the evolving
state fields (`solver.state`, each a `ScalarField` with a `.name`) and plain
NetCDF. Grid space is exact and real-valued, so it round-trips losslessly.

### Saving State

```julia
using NetCDF

function save_checkpoint(solver, path)
    isfile(path) && rm(path)
    for f in solver.state
        ensure_layout!(f, :g)
        g = get_grid_data(f)
        dimspec = collect(Iterators.flatten(
            ("$(f.name)_d$i" => s for (i, s) in enumerate(size(g)))))
        nccreate(path, f.name, dimspec...; t=NC_DOUBLE)   # NC_DOUBLE: keep Float64
        ncwrite(g, path, f.name)
    end
    ncputatt(path, "Global", Dict("sim_time" => solver.sim_time,
                                  "iteration" => solver.iteration, "dt" => solver.dt))
    return path
end
```

### Loading State

```julia
function load_checkpoint!(solver, path)
    for f in solver.state
        ensure_layout!(f, :g)                 # ensure the grid buffer exists / layout is :g
        get_grid_data(f) .= ncread(path, f.name)
    end
    solver.sim_time  = ncgetatt(path, "Global", "sim_time")
    solver.iteration = Int(ncgetatt(path, "Global", "iteration"))
    solver.dt        = ncgetatt(path, "Global", "dt")
    return solver
end
```

Use `solver.state` (the integrator's live fields), **not** the problem-variable
handles. Restart then continues from `run!` as usual:

```julia
save_checkpoint(solver, "chk.nc")
# … later, in a fresh session with the same problem/solver built …
load_checkpoint!(solver, "chk.nc")
run!(solver; stop_time=20.0, cfl=cfl)
```

!!! note "MPI"
    `get_grid_data` is the rank-local slab, so this helper is per-rank/serial.
    Under MPI either write one file per rank (include the rank in `path`), or
    gather to rank 0 with `gather_array(f.dist, get_grid_data(f))` before writing
    and scatter on load.

## Complete Example

```julia
using Tarang
using MPI
using Printf

MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# Setup (abbreviated)
# ... create domain, fields, problem ...

# Create solver
solver = InitialValueSolver(problem, RK222(), dt=1e-3)

# CFL
cfl = CFL(problem, safety=0.5)
add_velocity!(cfl, u)

# Output handler — pass the SOLVER so the handler auto-registers and `run!`
# processes it at its `sim_dt` cadence (no manual `process!` in the loop). No
# `vars` Dict needed: T is a problem variable, already in the solver namespace.
handler = add_file_handler("output", solver; parallel="gather", sim_dt=0.1, max_writes=100)
add_task!(handler, T; name="temperature")           # field snapshot
add_mean_task!(handler, T; dims=1, name="T_mean")    # x-averaged profile

# Diagnostics storage (time series accumulated in a callback)
times, max_T = Float64[], Float64[]

# run! drives the whole loop: CFL-adaptive dt, auto-writes the registered handler,
# runs callbacks, and closes the handler at the end. A Float callback interval
# (0.1) fires every 0.1 sim-time units; an Int interval fires every N iterations.
# global_max is MPI-reduced, so the value is identical on every rank.
run!(solver; stop_time=10.0, cfl=cfl,
     callbacks=[0.1 => function (s)
         ensure_layout!(T, :g)
         push!(times, s.sim_time)
         push!(max_T, global_max(dist, abs.(get_grid_data(T))))
         rank == 0 && @printf("t = %.3f, max|T| = %.6e\n", s.sim_time, max_T[end])
     end])

MPI.Finalize()
```

## See Also

- [Analysis API](../api/analysis.md): Full API reference
- [I/O API](../api/io.md): NetCDF output documentation
- [Parallelism](../pages/parallelism.md): Parallel I/O configuration
