# Multi-GPU Cluster Support in Tarang.jl

## Overview

Tarang.jl provides comprehensive support for multi-node GPU clusters, enabling efficient distributed spectral computations across multiple GPU boxes. This document details the multi-GPU capabilities and usage patterns.

## Multi-GPU Architecture

### Supported Configurations

✅ **Single Node, Multiple GPUs**
- Multiple CUDA/AMD/Metal GPUs per node
- Automatic GPU-process mapping
- NUMA-aware GPU allocation
- Peer-to-peer GPU transfers

✅ **Multiple Nodes, Multiple GPUs**  
- GPU clusters across multiple nodes
- MPI + GPU communication
- Inter-node GPU-aware data transfers
- Heterogeneous GPU support

✅ **Hybrid CPU-GPU Clusters**
- Mixed CPU and GPU nodes
- Automatic workload distribution
- Graceful degradation when GPUs unavailable

### Key Features

🎯 **Automatic GPU Detection**: Discovers all available GPUs across cluster  
🔄 **Smart GPU Assignment**: Maps MPI processes to GPUs optimally  
⚡ **Efficient Data Transfers**: Minimizes CPU↔GPU and inter-node transfers  
📊 **Performance Monitoring**: Tracks GPU utilization and transfer efficiency  
🛠️ **Heterogeneous Support**: Works with mixed GPU types and capabilities  

## Usage Examples

### Basic Multi-GPU Setup

```julia
using Tarang
using MPI

# Initialize MPI (automatic in multi-GPU manager)
# MPI.Init() - handled automatically

# Create multi-GPU optimized distributor
dist = create_multi_gpu_distributor(
    CartesianCoordinates("x", "y"), 
    (512, 512),  # Global shape
    auto_mesh=true  # Automatic GPU-optimized mesh
)

# The system automatically:
# 1. Detects GPUs on each node
# 2. Assigns GPUs to MPI processes  
# 3. Creates optimal process mesh
# 4. Sets up GPU-aware communication

# Check cluster configuration
check_multi_gpu_health(dist.multi_gpu_config)
```

### Advanced Configuration

```julia
# Manual GPU assignment for fine control
gpu_config = MultiGPUConfig()

# Create distributor with specific settings
coordsys = CartesianCoordinates("x", "y")
dist = Distributor(coordsys, 
                  comm=MPI.COMM_WORLD,
                  mesh=(4, 8),  # Specific 2D mesh
                  device="cuda")

# Enable multi-GPU features
dist.multi_gpu_config = gpu_config

# Setup distributed computing with GPU awareness
setup_pencil_arrays(dist, (1024, 1024))
```

### Field Operations on Multi-GPU

```julia
# Create fields that automatically use assigned GPUs
u = Field(dist, name="velocity", bases=(xbasis, ybasis))
v = Field(dist, name="vorticity", bases=(xbasis, ybasis)) 

# Operations automatically use local GPU
u['g'] = sin.(x) .* cos.(y)  # Computed on local GPU

# Global operations use GPU-aware MPI
global_energy = allreduce_array(dist, u['g'] .* u['g'], MPI.SUM)
```

## Multi-GPU Process Mesh Optimization

### 2D Problem Optimization (Project Requirement)

For 2D simulations with both horizontal and vertical parallelization:

```julia
# Automatic optimization for multi-node GPU clusters
global_shape = (1024, 1024)
gpu_config = MultiGPUConfig()

# System creates optimal 2D mesh considering:
# - GPU distribution across nodes
# - Inter-node communication costs  
# - Memory per GPU
# - Load balancing

mesh = create_gpu_optimized_mesh(gpu_config, global_shape)
# Example result: (8, 4) for 32 processes across 4 nodes with 8 GPUs each
```

### Node-Aware Mesh Creation

```julia
# For 4 nodes with 8 GPUs each (32 total processes)
nodes = 4
gpus_per_node = 8
total_procs = 32

# Optimal mesh minimizes inter-node communication:
# Option 1: (4, 8) - 4 process groups per node
# Option 2: (8, 4) - 8 process groups across nodes  

# System automatically chooses based on problem geometry
```

## GPU Memory Management

### Multi-GPU Memory Strategy

```julia
# Automatic memory distribution across GPUs
function optimize_gpu_memory_layout(dist::Distributor, global_shape)
    config = dist.multi_gpu_config
    
    # Calculate memory per GPU
    local_elements = prod(global_shape) ÷ config.mpi_size
    memory_per_gpu = local_elements * 16  # Complex Float64
    
    # Check against available GPU memory
    for (i, cap) in enumerate(config.gpu_capabilities)
        if cap.device_type != CPU_DEVICE
            available = cap.available_memory
            utilization = memory_per_gpu / available
            
            if utilization > 0.8
                @warn "GPU $i memory usage: $(round(utilization*100))%"
            end
        end
    end
end
```

### Smart Data Transfers

```julia
# Automatic optimization of data movement
function efficient_gpu_transfer!(src_field, dest_field, dist)
    src_config = get_field_gpu_config(src_field)  
    dest_config = get_field_gpu_config(dest_field)
    
    if same_node(src_config, dest_config)
        # Use fast intra-node GPU transfers
        gpu_to_gpu_copy!(src_field.data, dest_field.data)
    else  
        # Use MPI for inter-node transfers
        mpi_gpu_transfer!(src_field.data, dest_field.data, dist.comm)
    end
end
```

## Performance Optimization

### Load Balancing Across GPU Nodes

```julia
# Balance workload based on GPU capabilities
function balance_multi_gpu_workload(dist::Distributor)
    config = dist.multi_gpu_config
    
    # Get GPU capabilities across cluster
    gpu_capabilities = config.gpu_capabilities
    total_memory = sum(cap.total_memory for cap in gpu_capabilities 
                      if cap.device_type != CPU_DEVICE)
    
    # Adjust workload per process based on GPU memory
    my_capability = gpu_capabilities[config.mpi_rank + 1]
    my_fraction = my_capability.total_memory / total_memory
    
    @info "Process $(config.mpi_rank): GPU capability fraction = $(round(my_fraction*100, digits=1))%"
    
    return my_fraction
end
```

### Communication Pattern Optimization

```julia
# Minimize inter-node communication for spectral methods
function optimize_spectral_communication(dist::Distributor)
    config = dist.multi_gpu_config
    
    # Group processes by node for efficient transposes  
    node_groups = group_processes_by_node(config)
    
    # Prefer intra-node communication for FFT transposes
    # Use inter-node communication only when necessary
    
    return optimized_transpose_plan
end
```

## Monitoring and Debugging

### Multi-GPU Health Checks

```julia
# Comprehensive cluster health monitoring
function monitor_multi_gpu_cluster(dist::Distributor)
    config = dist.multi_gpu_config
    
    # Check GPU health across cluster
    check_multi_gpu_health(config)
    
    # Monitor performance metrics
    log_multi_gpu_performance(config)
    
    # Check for common issues
    detect_gpu_imbalance(config)
    detect_memory_pressure(config)
    detect_communication_bottlenecks(config)
end
```

### Performance Profiling

```julia
# Profile multi-GPU operations
function profile_multi_gpu_performance(dist::Distributor)
    stats = dist.multi_gpu_config.performance_stats
    
    println("Multi-GPU Performance Profile:")
    println("  Inter-node transfers: $(stats.inter_node_transfers)")  
    println("  Intra-node transfers: $(stats.intra_node_transfers)")
    println("  GPU memory moves: $(stats.gpu_memory_moves)")
    println("  MPI-GPU operations: $(stats.mpi_gpu_operations)")
    
    # Calculate efficiency metrics
    total_transfers = stats.inter_node_transfers + stats.intra_node_transfers
    if total_transfers > 0
        efficiency = stats.intra_node_transfers / total_transfers * 100
        println("  Intra-node efficiency: $(round(efficiency, digits=1))%")
    end
end
```

## Best Practices

### 1. GPU Cluster Setup

```bash
# Launch multi-GPU job with proper binding
mpirun -np 32 \
       --bind-to hwthread \
       --map-by node:PE=2 \
       --mca btl ^openib \
       julia --project=. simulation.jl
```

### 2. Memory Management

```julia
# Pre-allocate GPU memory pools
function setup_gpu_memory_pools(dist::Distributor)
    config = dist.multi_gpu_config
    
    # Allocate memory pools on each GPU
    if config.assigned_gpu.device_type != CPU_DEVICE
        pool_size = estimate_memory_requirements(dist)
        preallocate_gpu_memory(config.assigned_gpu, pool_size)
    end
end
```

### 3. Communication Optimization

```julia
# Use node-aware communication patterns
function optimize_mpi_gpu_communication(dist::Distributor)
    config = dist.multi_gpu_config
    
    # Create node-local subcommunicators
    node_comm = create_node_communicator(config)
    
    # Use hierarchical communication:
    # 1. Reduce within nodes (fast GPU-GPU)
    # 2. Reduce across nodes (slower but necessary)
    
    return hierarchical_communication_plan
end
```

## Troubleshooting

### Common Issues

1. **GPU Detection Failures**
   ```julia
   # Check GPU availability
   if has_cuda
       @info "CUDA devices: $(CUDA.ndevices())"
   else
       @warn "CUDA not available"
   end
   ```

2. **Memory Pressure**
   ```julia
   # Monitor GPU memory usage
   memory_info = get_gpu_memory_info(device_config)
   if memory_info.available < memory_info.total * 0.2
       @warn "GPU memory pressure detected"
   end
   ```

3. **Communication Bottlenecks**
   ```julia
   # Profile MPI-GPU transfers
   @timed multi_gpu_transfer!(src, dest, config1, config2)
   ```

## System Requirements

- **MPI**: OpenMPI 4.0+ or Intel MPI
- **CUDA**: CUDA Toolkit 11.0+ (for NVIDIA GPUs)
- **ROCm**: ROCm 4.0+ (for AMD GPUs)  
- **Julia**: Julia 1.8+ with GPU packages
- **Network**: InfiniBand or high-speed Ethernet for multi-node

The multi-GPU support in Tarang.jl provides a complete solution for GPU clusters, automatically handling the complexity of multi-node GPU computing while providing optimal performance for distributed spectral methods.