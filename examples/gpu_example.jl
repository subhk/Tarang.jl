"""
    GPU Example for Tarang.jl

This example demonstrates how to use GPU acceleration in Tarang.jl.
GPU support requires CUDA.jl to be installed and a compatible NVIDIA GPU.

The architecture abstraction allows switching between CPU and GPU
with minimal code changes.

## GPU Parallelization Modes

1. **Single GPU (Serial)**: Full GPU acceleration with CUFFT
   - All data stays on GPU
   - No MPI communication

2. **Multi-GPU (Distributed)**: GPU + MPI
   - Each MPI rank owns one GPU
   - Local FFTs use CUFFT
   - Communication via MPI (with CPU staging or CUDA-aware MPI)
   - Uses slab decomposition instead of PencilArrays

## Important Notes

- PencilArrays/PencilFFTs are CPU-only libraries
- For GPU+MPI, Tarang uses custom distributed GPU implementation
- Recommended: one MPI rank per GPU

## Running Multi-GPU

```bash
# Single GPU
julia gpu_example.jl

# Multi-GPU with MPI (4 GPUs)
mpiexec -n 4 julia gpu_example.jl
```
"""

# First, load CUDA if available
using CUDA
using Tarang

# ============================================================================
# Basic GPU Setup
# ============================================================================

println("=" ^ 60)
println("Tarang.jl GPU Example")
println("=" ^ 60)

# Check if CUDA is available
if CUDA.functional()
    println("CUDA is available!")
    println("  Device: $(CUDA.name(CUDA.device()))")
    println("  Memory: $(round(CUDA.totalmem(CUDA.device()) / 1e9, digits=2)) GB")
    println()

    # Create GPU architecture
    arch = GPU()  # Uses default device (device 0)
    # For multi-GPU systems, specify device: GPU(device_id=1)

else
    println("CUDA not available, using CPU")
    arch = CPU()
end

# ============================================================================
# Creating a Domain with GPU Architecture
# ============================================================================

println("\n--- Creating Domain with Architecture ---")

# Define coordinate system
coordsys = CartesianCoordinates(2, "xy")

# Create distributor with GPU architecture
dist = Distributor(coordsys, architecture=arch)
println("Distributor architecture: $(typeof(dist.architecture))")

# Define bases
Nx, Ny = 128, 128
x_basis = RealFourier(coordsys["x"], Nx)
y_basis = RealFourier(coordsys["y"], Ny)

# Create domain
domain = Domain(dist, (x_basis, y_basis))
println("Domain created with global shape: $(global_shape(domain))")

# ============================================================================
# Creating Fields on GPU
# ============================================================================

println("\n--- Creating Fields ---")

# Create scalar field (automatically allocated on GPU when using GPU architecture)
u = ScalarField(dist, "u", domain.bases, Float64)
println("Field 'u' created")
println("  Grid data type: $(typeof(u.data_g))")
println("  Coefficient data type: $(typeof(u.data_c))")

# Create vector field
vel = VectorField(dist, coordsys, "velocity", domain.bases, Float64)
println("Vector field 'velocity' created")

# ============================================================================
# Setting Initial Conditions
# ============================================================================

println("\n--- Setting Initial Conditions ---")

# For GPU arrays, we need to set data appropriately
# The field["g"] accessor returns the grid-space data (CuArray on GPU)

ensure_layout!(u, :g)
x = get_grid(domain, 1)  # x coordinates
y = get_grid(domain, 2)  # y coordinates

# Initialize with a simple pattern
# Using broadcasting works for both CPU and GPU arrays
u["g"] .= sin.(2π .* x) .* cos.(2π .* y)
println("Initial condition set")

# ============================================================================
# Transforms (Automatic GPU FFT when on GPU)
# ============================================================================

println("\n--- Forward/Backward Transforms ---")

# Forward transform to spectral space
# On GPU, this automatically uses CUFFT
forward_transform!(u)
println("Forward transform complete")
println("  Current layout: $(u.current_layout)")

# Backward transform to grid space
backward_transform!(u)
println("Backward transform complete")
println("  Current layout: $(u.current_layout)")

# ============================================================================
# Computing Derivatives
# ============================================================================

println("\n--- Computing Derivatives ---")

# Compute gradient (automatically uses GPU if field is on GPU)
grad_u = Gradient(u, coordsys)
grad_u_result = evaluate(grad_u, :g)
println("Gradient computed")

# Compute Laplacian
lap_u = Laplacian(u)
lap_u_result = evaluate(lap_u, :g)
println("Laplacian computed")

# ============================================================================
# Moving Data Between Architectures
# ============================================================================

println("\n--- Data Movement ---")

# Move data to CPU for analysis/plotting
if is_gpu(arch)
    # Get data from GPU to CPU
    u_cpu = on_architecture(CPU(), u["g"])
    println("Data moved to CPU")
    println("  Type on CPU: $(typeof(u_cpu))")
    println("  Max value: $(maximum(u_cpu))")

    # You can also move back to GPU
    u_gpu = on_architecture(GPU(), u_cpu)
    println("Data moved back to GPU")
    println("  Type on GPU: $(typeof(u_gpu))")
else
    println("Already on CPU, no data movement needed")
    println("  Max value: $(maximum(u["g"]))")
end

# ============================================================================
# Synchronization (Important for timing)
# ============================================================================

println("\n--- Synchronization ---")

if is_gpu(arch)
    # Ensure all GPU operations are complete before timing/saving
    synchronize(arch)
    println("GPU synchronized")
else
    println("CPU (no synchronization needed)")
end

# ============================================================================
# File I/O with GPU Arrays
# ============================================================================

println("\n--- File I/O (GPU-aware) ---")

# When saving to NetCDF or HDF5, data is automatically transferred from GPU to CPU
# The get_cpu_data() and get_cpu_local_data() functions handle this transparently

# Extract data for analysis (automatically moves to CPU if on GPU)
u_data = get_cpu_local_data(u, :g)
println("Data extracted for file I/O")
println("  Extracted type: $(typeof(u_data)) (always CPU Array)")
println("  Shape: $(size(u_data))")

# Check if field is on GPU
if is_gpu_field(u)
    println("  Field 'u' is on GPU - data was automatically transferred")
else
    println("  Field 'u' is on CPU - no transfer needed")
end

# Note: When using NetCDFFileHandler or HDF5 output, the GPU→CPU transfer
# happens automatically. You don't need to manually call get_cpu_data().
# Example:
#   handler = NetCDFFileHandler("output", dist, [u])
#   add_task!(handler, u, "velocity")
#   process!(handler)  # Automatically handles GPU→CPU transfer

# ============================================================================
# Memory Management
# ============================================================================

println("\n--- Memory Information ---")

if is_gpu(arch)
    # Check GPU memory usage
    free, total = CUDA.Mem.info()
    used = total - free
    println("GPU Memory:")
    println("  Used: $(round(used / 1e9, digits=2)) GB")
    println("  Free: $(round(free / 1e9, digits=2)) GB")
    println("  Total: $(round(total / 1e9, digits=2)) GB")
else
    println("CPU memory managed by Julia GC")
end

println("\n" * "=" ^ 60)
println("GPU Example Complete!")
println("=" ^ 60)
