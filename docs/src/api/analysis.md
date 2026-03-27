# Analysis API

Analysis tools for computing diagnostics, managing output, and monitoring simulations in real-time.

## CFL Conditions

### CFL

Computes adaptive timesteps based on the Courant-Friedrichs-Lewy (CFL) stability criterion.

**Constructor**:
```julia
CFL(
    problem::IVP;
    safety::Float64=0.5,
    max_change::Float64=2.0,
    min_change::Float64=0.5,
    max_dt::Float64=Inf,
    min_dt::Float64=0.0
)
```

**Arguments**:
- `problem`: IVP problem
- `safety`: Safety factor (0 < safety < 1)
- `max_change`: Maximum timestep increase factor per step
- `min_change`: Maximum timestep decrease factor per step
- `max_dt`: Maximum allowed timestep
- `min_dt`: Minimum allowed timestep

**Examples**:

```julia
# Standard CFL
cfl = CFL(problem, safety=0.5)

# Conservative settings
cfl = CFL(problem, safety=0.3, max_change=1.2, min_change=0.8)

# With timestep bounds
cfl = CFL(problem, safety=0.5, max_dt=0.01, min_dt=1e-6)
```

**Methods**:

#### add_velocity!

Add velocity field for CFL calculation.

```julia
add_velocity!(cfl, velocity_field)
```

**Example**:

```julia
cfl = CFL(problem)
add_velocity!(cfl, u)  # u is a VectorField

# For multiple velocity scales
add_velocity!(cfl, u)  # Advection velocity
add_velocity!(cfl, c)  # Sound speed (for compressible flow)
```

#### compute_timestep

Calculate adaptive timestep.

```julia
dt = compute_timestep(cfl)
```

**Returns**: Adaptive timestep satisfying CFL condition

**Example**:

```julia
while solver.sim_time < t_end
    dt = compute_timestep(cfl)
    step!(solver, dt)
end
```

**CFL Condition**:

For explicit methods:
```math
\Delta t \leq C \frac{\Delta x}{|u|_{max}}
```

where C is the safety factor.

**Properties**:
```julia
cfl.safety          # Safety factor
cfl.max_dt          # Maximum timestep
cfl.min_dt          # Minimum timestep
cfl.current_dt      # Last computed timestep
```

---

## Output Handlers

### NetCDF Output

#### add_netcdf_handler

Create NetCDF output handler for saving fields to files.

**Syntax**:
```julia
add_netcdf_handler(
    solver::InitialValueSolver,
    base_path::String;
    fields::Vector{<:AbstractField},
    write_interval::Float64=1.0,
    mode::String="overwrite"
)
```

**Arguments**:
- `solver`: IVP solver
- `base_path`: Base path for output files (without extension)
- `fields`: Fields to save
- `write_interval`: Time between writes
- `mode`: File mode ("overwrite", "append")

**Examples**:

```julia
# Basic output
handler = add_netcdf_handler(
    solver,
    "output/snapshots",
    fields=[u, p, T],
    write_interval=0.1
)

# Multiple handlers for different cadences
snapshots = add_netcdf_handler(solver, "snapshots", fields=[u, p, T], write_interval=0.1)
checkpoints = add_netcdf_handler(solver, "checkpoints", fields=[u, p, T], write_interval=1.0)
```

**Output files**:
- Creates files like: `snapshots_proc0.nc`, `snapshots_proc1.nc`, etc.
- One file per MPI process
- Use `merge_processor_files` to combine

---

#### NetCDFFileHandler

Direct handler class for more control.

**Constructor**:
```julia
NetCDFFileHandler(
    filename::String,
    fields::Vector{<:AbstractField};
    write_mode::String="overwrite",
    compression::Int=4
)
```

**Methods**:
```julia
# Write current state
write_fields!(handler, sim_time)

# Close file
close!(handler)
```

**Example**:

```julia
handler = NetCDFFileHandler("output.nc", [u, p, T], compression=6)

while solver.sim_time < t_end
    step!(solver)

    if solver.iteration % 100 == 0
        write_fields!(handler, solver.sim_time)
    end
end

close!(handler)
```

---

### HDF5 Output

#### add_hdf5_handler

Similar to NetCDF but uses HDF5 format.

**Syntax**:
```julia
add_hdf5_handler(
    solver::InitialValueSolver,
    base_path::String;
    fields::Vector{<:AbstractField},
    write_interval::Float64=1.0
)
```

**Example**:

```julia
handler = add_hdf5_handler(
    solver,
    "output/data",
    fields=[u, p, T],
    write_interval=0.1
)
```

---

## Analysis Evaluators

### Scalar Evaluators

Compute scalar diagnostics during simulation.

#### add_scalar_evaluator

```julia
add_scalar_evaluator(
    solver::InitialValueSolver,
    name::String,
    expression::String
)
```

**Example**:

```julia
# Kinetic energy
add_scalar_evaluator(solver, "KE", "0.5*integrate(u*u + v*v + w*w)")

# Enstrophy
add_scalar_evaluator(solver, "enstrophy", "0.5*integrate(omega*omega)")

# Nusselt number
add_scalar_evaluator(solver, "Nu", "1 + mean(w*T)")

# Access values
KE = get_scalar(solver, "KE")
```

---

### Field Evaluators

Compute derived fields during simulation.

#### add_field_evaluator

```julia
add_field_evaluator(
    solver::InitialValueSolver,
    name::String,
    expression::String
)
```

**Example**:

```julia
# Vorticity
add_field_evaluator(solver, "omega", "∂x(v) - ∂z(u)")

# Q-criterion
add_field_evaluator(solver, "Q", "0.5*(Omega*Omega - S*S)")

# Temperature gradient magnitude
add_field_evaluator(solver, "grad_T_mag", "sqrt(∂x(T)^2 + ∂z(T)^2)")

# Access computed field
omega = get_field(solver, "omega")
```

---

## Analysis Tasks

### Unified Evaluator

Combine multiple analysis operations.

```julia
evaluator = UnifiedEvaluator(solver)

# Add various tasks
add_task!(evaluator, "scalar", "KE", "0.5*integrate(u*u)")
add_task!(evaluator, "field", "omega", "curl(u)")
add_task!(evaluator, "profile", "T_mean", "mean(T, dim=1)")  # Mean over x

# Evaluate all tasks
evaluate!(evaluator)

# Access results
KE = evaluator.results["KE"]
omega = evaluator.results["omega"]
T_profile = evaluator.results["T_mean"]
```

---

## Statistical Analysis

### Time Averaging

```julia
# Create time averager
avg = TimeAverager(fields=[u, T], averaging_interval=1.0)

# During simulation
while solver.sim_time < t_end
    step!(solver)
    accumulate!(avg, solver.sim_time)
end

# Get averaged fields
u_mean = get_average(avg, "u")
T_mean = get_average(avg, "T")
```

### Spatial Averaging

```julia
# Mean over dimension
T_mean_x = mean(T, dim=1)  # Average over x
u_mean_profile = mean(u, dim=1)  # Velocity profile

# Volume average
T_vol_avg = mean(T)  # Average over entire domain
```

### Reynolds Decomposition

```julia
# Compute fluctuations
function reynolds_decomposition(field, mean_field)
    fluctuation = field - mean_field
    return fluctuation
end

# Example
u_mean = mean(u, dim=1)
u_prime = reynolds_decomposition(u, u_mean)

# Reynolds stresses
uu_mean = mean(u_prime * u_prime, dim=1)
uv_mean = mean(u_prime * v_prime, dim=1)
```

---

## Spectral Analysis

### Energy Spectra

```julia
# 1D energy spectrum
function compute_energy_spectrum_1d(u, direction)
    # Transform to spectral space
    to_spectral!(u)
    u_hat = get_spectral_data(u)

    # Compute energy
    E = abs2.(u_hat)

    # Bin by wavenumber magnitude
    k, E_k = bin_spectrum(E, direction)

    return k, E_k
end

# Usage
k, E_k = compute_energy_spectrum_1d(u, 1)  # Spectrum in x-direction

# Plot
using Plots
plot(k, E_k, xscale=:log10, yscale=:log10,
     xlabel="k", ylabel="E(k)")
```

### 3D Isotropic Spectrum

```julia
function compute_isotropic_spectrum(u)
    # Compute 3D energy distribution
    to_spectral!(u)

    E_hat = sum(abs2.(component) for component in u.components)

    # Bin by |k|
    k_mag = compute_k_magnitude(domain)
    k, E_k = spherical_average(E_hat, k_mag)

    return k, E_k
end

# Kolmogorov scaling check
k, E_k = compute_isotropic_spectrum(u)
plot(k, E_k, xscale=:log10, yscale=:log10)
plot!(k, k.^(-5/3), linestyle=:dash, label="k^(-5/3)")
```

---

## Flow Property Analysis

### Nusselt Number

Heat transfer efficiency in convection:

```julia
function compute_nusselt(T, w, domain)
    # Nu = 1 + <w*T> (horizontally averaged)
    to_grid!(T)
    to_grid!(w)

    wT = w.data .* T.data
    wT_mean = mean(wT, dims=1)  # Average over x

    Nu = 1.0 .+ wT_mean
    return mean(Nu)  # Domain average
end
```

### Reynolds Number

From velocity statistics:

```julia
function compute_reynolds_number(u, nu, L)
    # Re = U*L/nu
    to_grid!(u)
    U = sqrt(mean(u.data .^ 2))  # RMS velocity
    Re = U * L / nu
    return Re
end
```

### Richardson Number

Stratification vs. shear:

```julia
function compute_richardson_number(u, T, dz, g, T0)
    # Ri = (g/T0) * (∂T/∂z) / (∂u/∂z)^2
    dTdz = ∂z(T)
    dudz = ∂z(u)

    to_grid!(dTdz)
    to_grid!(dudz)

    Ri = (g / T0) .* dTdz.data ./ (dudz.data .^ 2)
    return mean(Ri)
end
```

---

## Probe Points

Sample fields at specific locations.

```julia
# Define probe locations
probes = ProbeSet([
    ("probe1", x=1.0, z=0.5),
    ("probe2", x=2.0, z=0.5),
    ("probe3", x=3.0, z=0.5)
])

# Sample during simulation
while solver.sim_time < t_end
    step!(solver)

    if solver.iteration % 10 == 0
        sample_probes!(probes, [u, T], solver.sim_time)
    end
end

# Get time series
u_probe1 = get_probe_data(probes, "probe1", "u")
t_series = get_probe_times(probes)

# Plot
plot(t_series, u_probe1, xlabel="t", ylabel="u", label="Probe 1")
```

---

## Monitoring and Diagnostics

### Simulation Monitoring

```julia
# Create monitor
monitor = SimulationMonitor(
    print_interval=100,
    print_fields=true,
    print_diagnostics=true
)

# Add to solver
attach_monitor!(solver, monitor)

# Automatically prints during simulation:
# Iteration: 100, t = 0.100, dt = 0.001, KE = 1.234, max(u) = 2.345
```

### Custom Diagnostics

```julia
function my_diagnostics(solver, fields)
    u, T = fields

    # Compute custom quantities
    KE = 0.5 * mean(u.data .^ 2)
    max_T = maximum(T.data)
    min_T = minimum(T.data)

    return Dict(
        "KE" => KE,
        "T_max" => max_T,
        "T_min" => min_T
    )
end

# Register diagnostic
add_diagnostic!(solver, my_diagnostics, interval=10)
```

---

## Data Post-Processing

### Merging Parallel Output

After simulation with MPI:

```julia
# Merge NetCDF files from all ranks
using Tarang.IO

merge_netcdf_files(
    "output/snapshots",
    output_file="output/snapshots_merged.nc",
    cleanup=true  # Remove individual rank files
)
```

### Loading Data

```julia
using NCDatasets

# Load merged data
ds = Dataset("output/snapshots_merged.nc")

# Access fields
u_data = ds["u"][:]
T_data = ds["T"][:]
time = ds["time"][:]

close(ds)

# Analyze or visualize
using Plots
heatmap(T_data[:, :, end], title="Temperature at t=$(time[end])")
```

---

## Complete Example

```julia
using Tarang, MPI

MPI.Init()

# ... setup problem and solver ...

# CFL condition
cfl = CFL(problem, safety=0.5, max_dt=0.01)
add_velocity!(cfl, u)

# Output handlers
snapshots = add_netcdf_handler(solver, "snapshots",
                                fields=[u, p, T], write_interval=0.1)

# Scalar diagnostics
add_scalar_evaluator(solver, "KE", "0.5*integrate(u*u + w*w)")
add_scalar_evaluator(solver, "Nu", "1 + mean(w*T)")

# Monitoring
monitor = SimulationMonitor(print_interval=100)
attach_monitor!(solver, monitor)

# Time integration
while solver.sim_time < 10.0
    dt = compute_timestep(cfl)
    step!(solver, dt)

    # Custom analysis every 1000 steps
    if solver.iteration % 1000 == 0
        KE = get_scalar(solver, "KE")
        Nu = get_scalar(solver, "Nu")

        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            println("t=$(solver.sim_time): KE=$KE, Nu=$Nu")
        end
    end
end

# Merge output files
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    merge_netcdf_files("snapshots", output_file="snapshots.nc", cleanup=true)
end

MPI.Finalize()
```

---

## See Also

- [Solvers](solvers.md): Integration methods
- [Fields](fields.md): Field operations for analysis
- [Tutorial: Analysis](../tutorials/analysis_and_output.md): Detailed examples
