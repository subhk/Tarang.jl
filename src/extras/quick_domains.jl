"""
Quick domain creation utilities

Convenient functions for creating common domains
"""

using LinearAlgebra

function _require_coords(dist::Distributor, needed::Int)
    coords = dist.coords
    if length(coords) < needed
        throw(ArgumentError("Distributor provides $(length(coords)) coordinates, but $needed are required for this domain."))
    end
    return ntuple(i -> coords[i], needed)
end

# 1D domains
function create_fourier_domain(dist::Distributor, L::Float64, N::Int; dtype::Type=Float64, device::String="cpu")
    """Create 1D periodic Fourier domain with GPU support"""
    
    x_coord = _require_coords(dist, 1)[1]
    x_basis = RealFourier(x_coord, size=N, bounds=(0.0, L), dtype=dtype)
    
    return Domain(dist, (x_basis,); device=device)
end

function create_chebyshev_domain(dist::Distributor, L::Float64, N::Int; dtype::Type=Float64, device::String="cpu")
    """Create 1D Chebyshev domain with GPU support"""
    
    x_coord = _require_coords(dist, 1)[1]
    x_basis = ChebyshevT(x_coord, size=N, bounds=(0.0, L), dtype=dtype)
    
    return Domain(dist, (x_basis,); device=device)
end

function create_legendre_domain(dist::Distributor, L::Float64, N::Int; dtype::Type=Float64, device::String="cpu")
    """Create 1D Legendre domain with GPU support"""
    
    x_coord = _require_coords(dist, 1)[1]
    x_basis = Legendre(x_coord, size=N, bounds=(0.0, L), dtype=dtype)
    
    return Domain(dist, (x_basis,); device=device)
end

# 2D domains
function create_2d_periodic_domain(dist::Distributor, Lx::Float64, Ly::Float64, Nx::Int, Ny::Int; 
                                  dtype::Type=Float64, dealias::Float64=3.0/2.0, device::String="cpu")
    """Create 2D doubly-periodic domain with Fourier bases in both directions and GPU support"""
    
    x_coord, y_coord = _require_coords(dist, 2)
    
    x_basis = RealFourier(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    y_basis = RealFourier(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (x_basis, y_basis); device=device)
end

function create_channel_domain(dist::Distributor, Lx::Float64, Ly::Float64, Nx::Int, Ny::Int;
                              dtype::Type=Float64, dealias::Float64=3.0/2.0, device::String="cpu")
    """Create 2D channel domain: periodic in x, Chebyshev in y with GPU support"""
    
    x_coord, y_coord = _require_coords(dist, 2)
    
    x_basis = RealFourier(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    y_basis = ChebyshevT(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (x_basis, y_basis); device=device)
end

function create_rectangular_domain(dist::Distributor, Lx::Float64, Ly::Float64, Nx::Int, Ny::Int;
                                  x_basis_type::Type=ChebyshevT, y_basis_type::Type=ChebyshevT,
                                  dtype::Type=Float64, dealias::Float64=1.0)
    """Create 2D rectangular domain with specified basis types"""
    
    x_coord, y_coord = _require_coords(dist, 2)
    
    if x_basis_type == RealFourier
        x_basis = RealFourier(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    elseif x_basis_type == ChebyshevT
        x_basis = ChebyshevT(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    elseif x_basis_type == Legendre
        x_basis = Legendre(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    else
        throw(ArgumentError("Unsupported x basis type: $x_basis_type"))
    end
    
    if y_basis_type == RealFourier
        y_basis = RealFourier(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    elseif y_basis_type == ChebyshevT
        y_basis = ChebyshevT(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    elseif y_basis_type == Legendre
        y_basis = Legendre(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    else
        throw(ArgumentError("Unsupported y basis type: $y_basis_type"))
    end
    
    return Domain(dist, (x_basis, y_basis))
end

# 3D domains
function create_3d_periodic_domain(dist::Distributor, Lx::Float64, Ly::Float64, Lz::Float64, 
                                  Nx::Int, Ny::Int, Nz::Int; dtype::Type=Float64, dealias::Float64=3.0/2.0, device::String="cpu")
    """Create 3D triply-periodic domain with GPU support"""
    
    x_coord, y_coord, z_coord = _require_coords(dist, 3)
    
    x_basis = RealFourier(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    y_basis = RealFourier(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    z_basis = RealFourier(z_coord, size=Nz, bounds=(0.0, Lz), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (x_basis, y_basis, z_basis); device=device)
end

function create_box_domain(dist::Distributor, Lx::Float64, Ly::Float64, Lz::Float64,
                          Nx::Int, Ny::Int, Nz::Int; dtype::Type=Float64, dealias::Float64=1.0)
    """Create 3D box domain with Chebyshev bases"""
    
    x_coord, y_coord, z_coord = _require_coords(dist, 3)
    
    x_basis = ChebyshevT(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    y_basis = ChebyshevT(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    z_basis = ChebyshevT(z_coord, size=Nz, bounds=(0.0, Lz), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (x_basis, y_basis, z_basis))
end

# Specialized domains for common problems
function rayleigh_benard_domain(dist::Distributor, aspect_ratio::Float64=4.0, Nx::Int=256, Nz::Int=64;
                               dtype::Type=Float64, dealias::Float64=3.0/2.0)
    """Create domain for Rayleigh-Bénard convection"""
    
    Lx = aspect_ratio  # Horizontal extent
    Lz = 1.0          # Vertical extent
    
    x_coord, z_coord = _require_coords(dist, 2)
    
    # Periodic in x (horizontal), Chebyshev in z (vertical)
    x_basis = RealFourier(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    z_basis = ChebyshevT(z_coord, size=Nz, bounds=(0.0, Lz), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (x_basis, z_basis))
end

function taylor_couette_domain(dist::Distributor, inner_radius::Float64=0.5, outer_radius::Float64=1.0,
                              Nr::Int=64, Ntheta::Int=128; dtype::Type=ComplexF64, dealias::Float64=3.0/2.0)
    """Create domain for Taylor-Couette flow (cylindrical coordinates)"""
    
    r_coord, theta_coord = _require_coords(dist, 2)
    
    # Chebyshev in r, Fourier in θ
    r_basis = ChebyshevT(r_coord, size=Nr, bounds=(inner_radius, outer_radius), dealias=dealias, dtype=Float64)
    theta_basis = ComplexFourier(theta_coord, size=Ntheta, bounds=(0.0, 2π), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (r_basis, theta_basis))
end

function disk_domain(dist::Distributor, radius::Float64=1.0, N::Int=64; dtype::Type=ComplexF64, dealias::Float64=1.0)
    """Create disk domain using disk basis"""
    
    disk_coord = _require_coords(dist, 1)[1]
    
    disk_basis = DiskBasis(disk_coord, radius=radius, size=N, dealias=dealias, dtype=dtype)
    
    return Domain(dist, (disk_basis,))
end

function annulus_domain(dist::Distributor, inner_radius::Float64=0.5, outer_radius::Float64=1.0,
                       N::Int=64; dtype::Type=ComplexF64, dealias::Float64=1.0)
    """Create annulus domain using annulus basis"""
    
    annulus_coord = _require_coords(dist, 1)[1]
    
    annulus_basis = AnnulusBasis(annulus_coord, radii=(inner_radius, outer_radius), size=N, dealias=dealias, dtype=dtype)
    
    return Domain(dist, (annulus_basis,))
end

# Quick field creation with GPU support
function create_fields(domain::Domain, field_names::Vector{String}, field_types::Vector{String}=String[])
    """Create multiple fields on domain with GPU support"""
    
    if isempty(field_types)
        field_types = fill("scalar", length(field_names))
    end
    
    if length(field_names) != length(field_types)
        throw(ArgumentError("Number of field names must match number of field types"))
    end
    
    fields = Dict{String, Any}()
    
    for (name, ftype) in zip(field_names, field_types)
        if ftype == "scalar"
            field = ScalarField(domain.dist, name, domain.bases, domain.dist.dtype)
            # Ensure field data is on same device as domain
            field.data_g = ensure_device!(field.data_g, domain.device_config)
            field.data_c = ensure_device!(field.data_c, domain.device_config)
            fields[name] = field
        elseif ftype == "vector"
            field = VectorField(domain.dist, domain.dist.coordsys, name, domain.bases, domain.dist.dtype)
            # Ensure vector components are on same device as domain
            for comp in field.components
                comp.data_g = ensure_device!(comp.data_g, domain.device_config)
                comp.data_c = ensure_device!(comp.data_c, domain.device_config)
            end
            fields[name] = field
        elseif ftype == "tensor"
            field = TensorField(domain.dist, domain.dist.coordsys, name, domain.bases, domain.dist.dtype)
            # Ensure tensor components are on same device as domain
            for i in 1:size(field.components, 1), j in 1:size(field.components, 2)
                comp = field.components[i, j]
                comp.data_g = ensure_device!(comp.data_g, domain.device_config)
                comp.data_c = ensure_device!(comp.data_c, domain.device_config)
            end
            fields[name] = field
        else
            throw(ArgumentError("Unknown field type: $ftype"))
        end
    end
    
    return fields
end

function create_scalar_fields(domain::Domain, names::String...)
    """Create multiple scalar fields on domain"""
    return create_fields(domain, collect(names), fill("scalar", length(names)))
end

function create_vector_fields(domain::Domain, names::String...)
    """Create multiple vector fields on domain"""
    return create_fields(domain, collect(names), fill("vector", length(names)))
end

# Domain information utilities
function domain_info(domain::Domain)
    """Print information about domain"""
    
    @info "Domain Information:"
    @info "  Dimension: $(domain.dim)"
    @info "  Number of bases: $(length(domain.bases))"
    @info "  Global shape: $(global_shape(domain))"
    @info "  Local shape: $(local_shape(domain))"
    @info "  Volume: $(volume(domain))"
    
    for (i, basis) in enumerate(domain.bases)
        coord_name = basis.meta.element_label
        basis_type = typeof(basis)
        size = basis.meta.size
        bounds = basis.meta.bounds
        dealias = basis.meta.dealias
        
        @info "  Basis $i ($coord_name):"
        @info "    Type: $basis_type"
        @info "    Size: $size"
        @info "    Bounds: $bounds"
        @info "    Dealias factor: $dealias"
    end
    
    # MPI information
    @info "  MPI processes: $(domain.dist.size)"
    @info "  Process mesh: $(domain.dist.mesh)"
    
    # PencilArrays compatibility
    pencil_compatible = is_pencil_compatible(domain.bases)
    @info "  PencilFFTs compatible: $pencil_compatible"
end

# Advanced 3D domains for specific applications
function taylor_green_vortex_domain(dist::Distributor, N::Int=128; dtype::Type=Float64, dealias::Float64=3.0/2.0)
    """Create domain for 3D Taylor-Green vortex simulation"""
    
    coords = CartesianCoordinates("x", "y", "z")
    x_coord = coords["x"]
    y_coord = coords["y"]
    z_coord = coords["z"]
    
    # Triple periodic domain [0, 2π]³
    x_basis = RealFourier(x_coord, size=N, bounds=(0.0, 2π), dealias=dealias, dtype=dtype)
    y_basis = RealFourier(y_coord, size=N, bounds=(0.0, 2π), dealias=dealias, dtype=dtype)
    z_basis = RealFourier(z_coord, size=N, bounds=(0.0, 2π), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (x_basis, y_basis, z_basis))
end

function channel_flow_3d_domain(dist::Distributor, Lx::Float64=4π, Ly::Float64=2π, Lz::Float64=2.0,
                               Nx::Int=128, Ny::Int=128, Nz::Int=64; dtype::Type=Float64, dealias::Float64=3.0/2.0)
    """Create domain for 3D turbulent channel flow"""
    
    coords = CartesianCoordinates("x", "y", "z")
    x_coord = coords["x"]
    y_coord = coords["y"] 
    z_coord = coords["z"]
    
    # Periodic in x and y, Chebyshev in z (wall-normal)
    x_basis = RealFourier(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    y_basis = RealFourier(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    z_basis = ChebyshevT(z_coord, size=Nz, bounds=(-1.0, 1.0), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (x_basis, y_basis, z_basis))
end

function mixing_layer_3d_domain(dist::Distributor, Lx::Float64=20.0, Ly::Float64=10.0, Lz::Float64=10.0,
                               Nx::Int=256, Ny::Int=128, Nz::Int=128; dtype::Type=Float64, dealias::Float64=3.0/2.0)
    """Create domain for 3D mixing layer simulation"""
    
    coords = CartesianCoordinates("x", "y", "z")
    x_coord = coords["x"]
    y_coord = coords["y"]
    z_coord = coords["z"]
    
    # Periodic in y and z, Fourier in x (streamwise)
    x_basis = RealFourier(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    y_basis = RealFourier(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    z_basis = RealFourier(z_coord, size=Nz, bounds=(0.0, Lz), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (x_basis, y_basis, z_basis))
end

function turbulent_convection_3d_domain(dist::Distributor, aspect_ratio::Float64=2.0, 
                                       Nx::Int=128, Ny::Int=128, Nz::Int=64; 
                                       dtype::Type=Float64, dealias::Float64=3.0/2.0)
    """Create domain for 3D Rayleigh-Bénard convection"""
    
    Lx = Ly = aspect_ratio  # Horizontal extent
    Lz = 1.0               # Vertical extent
    
    coords = CartesianCoordinates("x", "y", "z")
    x_coord = coords["x"]
    y_coord = coords["y"]
    z_coord = coords["z"]
    
    # Periodic in x and y (horizontal), Chebyshev in z (vertical)
    x_basis = RealFourier(x_coord, size=Nx, bounds=(0.0, Lx), dealias=dealias, dtype=dtype)
    y_basis = RealFourier(y_coord, size=Ny, bounds=(0.0, Ly), dealias=dealias, dtype=dtype)
    z_basis = ChebyshevT(z_coord, size=Nz, bounds=(0.0, Lz), dealias=dealias, dtype=dtype)
    
    return Domain(dist, (x_basis, y_basis, z_basis))
end

# 3D field creation utilities
function create_navier_stokes_3d_fields(domain::Domain)
    """Create complete set of fields for 3D Navier-Stokes equations with GPU support"""
    
    dist = domain.dist
    coordsys = dist.coordsys
    bases = domain.bases
    dtype = dist.dtype
    
    # Primary fields
    u = VectorField(dist, coordsys, "velocity", bases, dtype)      # Velocity vector
    p = ScalarField(dist, "pressure", bases, dtype)               # Pressure
    
    # For incompressible flow with tau method (simplified)
    tau_u = VectorField(dist, coordsys, "tau_u", bases, dtype)    # Velocity tau terms
    tau_p = ScalarField(dist, "tau_p", (), dtype)                 # Pressure gauge
    
    # Ensure all fields are on same device as domain
    for comp in u.components
        comp.data_g = ensure_device!(comp.data_g, domain.device_config)
        comp.data_c = ensure_device!(comp.data_c, domain.device_config)
    end
    
    p.data_g = ensure_device!(p.data_g, domain.device_config)
    p.data_c = ensure_device!(p.data_c, domain.device_config)
    
    for comp in tau_u.components
        comp.data_g = ensure_device!(comp.data_g, domain.device_config)
        comp.data_c = ensure_device!(comp.data_c, domain.device_config)
    end
    
    tau_p.data_g = ensure_device!(tau_p.data_g, domain.device_config)
    tau_p.data_c = ensure_device!(tau_p.data_c, domain.device_config)
    
    return Dict(
        "velocity" => u,
        "pressure" => p,
        "tau_velocity" => tau_u,
        "tau_pressure" => tau_p
    )
end

function create_thermal_convection_3d_fields(domain::Domain)
    """Create fields for 3D thermal convection (Navier-Stokes + temperature)"""
    
    dist = domain.dist
    coordsys = dist.coordsys
    bases = domain.bases
    dtype = dist.dtype
    
    # Get base Navier-Stokes fields
    fields = create_navier_stokes_3d_fields(domain)
    
    # Add temperature field
    fields["temperature"] = ScalarField(dist, "temperature", bases, dtype)
    fields["tau_temperature"] = ScalarField(dist, "tau_temperature", bases, dtype)
    
    return fields
end

function create_mhd_3d_fields(domain::Domain)
    """Create fields for 3D magnetohydrodynamics"""
    
    dist = domain.dist
    coordsys = dist.coordsys
    bases = domain.bases
    dtype = dist.dtype
    
    # Get base flow fields
    fields = create_navier_stokes_3d_fields(domain)
    
    # Add magnetic field
    fields["magnetic"] = VectorField(dist, coordsys, "magnetic", bases, dtype)
    fields["tau_magnetic"] = VectorField(dist, coordsys, "tau_magnetic", bases, dtype)
    
    return fields
end

# 3D performance analysis
function analyze_3d_performance(domain::Domain, n_fields::Int=4)
    """Analyze performance characteristics for 3D domain"""
    
    global_shape = global_shape(domain)
    local_shape = local_shape(domain)
    
    @info "3D Performance Analysis:"
    @info "  Global shape: $global_shape"
    @info "  Local shape: $local_shape"
    @info "  Process mesh: $(domain.dist.mesh)"
    
    # Memory analysis
    global_size = prod(global_shape)
    local_size = prod(local_shape)
    
    bytes_per_element = sizeof(domain.dist.dtype)
    
    # Memory per field (grid + coefficient layouts)
    memory_per_field_local = 2 * local_size * bytes_per_element
    memory_per_field_global = 2 * global_size * bytes_per_element
    
    total_local = n_fields * memory_per_field_local
    total_global = n_fields * memory_per_field_global
    
    @info "  Memory per process: $(round(total_local / 1024^3, digits=2)) GB"
    @info "  Total memory: $(round(total_global / 1024^3, digits=2)) GB"
    
    # Communication analysis
    if length(domain.dist.mesh) == 3
        nx_proc, ny_proc, nz_proc = domain.dist.mesh
        nx_local, ny_local, nz_local = local_shape
        
        # Estimate communication volume for FFTs
        if any(isa(basis, Union{RealFourier, ComplexFourier}) for basis in domain.bases)
            # PencilFFTs communication patterns
            comm_volume_xy = nx_local * ny_local * nz_proc  # xy-plane transposes
            comm_volume_xz = nx_local * nz_local * ny_proc  # xz-plane transposes  
            comm_volume_yz = ny_local * nz_local * nx_proc  # yz-plane transposes
            
            total_comm = (comm_volume_xy + comm_volume_xz + comm_volume_yz) * bytes_per_element
            
            @info "  Est. communication per FFT: $(round(total_comm / 1024^2, digits=2)) MB"
        end
        
        # Load balance analysis
        ideal_load = global_size / domain.dist.size
        actual_load = local_size
        efficiency = min(ideal_load / actual_load, actual_load / ideal_load)
        
        @info "  Load balance efficiency: $(round(efficiency * 100, digits=1))%"
    end
    
    # Computational complexity
    if any(isa(basis, Union{RealFourier, ComplexFourier}) for basis in domain.bases)
        fourier_count = count(b -> isa(b, Union{RealFourier, ComplexFourier}), domain.bases)
        @info "  FFT dimensions: $fourier_count/3"
        @info "  3D FFT complexity: O(N³ log N) per field"
    end
    
    # Scalability prediction
    max_efficient_procs = div(minimum(global_shape), 4)  # Rule of thumb: at least 4 points per process per dimension
    @info "  Est. max efficient processes: $(max_efficient_procs^3)"
end

function estimate_memory_usage(domain::Domain, n_fields::Int=1; dtype::Type=Float64)
    """Estimate memory usage for domain and fields with GPU considerations"""
    
    global_size = prod(global_shape(domain))
    local_size = prod(local_shape(domain))
    
    bytes_per_element = sizeof(dtype)
    
    # Memory per field (grid + coefficient layouts)
    memory_per_field_local = 2 * local_size * bytes_per_element
    memory_per_field_global = 2 * global_size * bytes_per_element
    
    total_local = n_fields * memory_per_field_local
    total_global = n_fields * memory_per_field_global
    
    @info "Memory Usage Estimate ($(domain.device_config.device_type)):"
    @info "  Element type: $dtype ($(bytes_per_element) bytes per element)"
    @info "  Global elements per field: $global_size"
    @info "  Local elements per field: $local_size"
    @info "  Fields: $n_fields"
    @info "  Memory per process: $(round(total_local / 1024^2, digits=2)) MB"
    @info "  Total memory: $(round(total_global / 1024^2, digits=2)) MB"
    
    # GPU-specific memory considerations
    if domain.device_config.device_type != CPU_DEVICE
        gpu_memory_info = get_domain_memory_info(domain)
        @info "  GPU Memory Information:"
        @info "    Available: $(round(gpu_memory_info.available_memory / 1024^3, digits=2)) GB"
        @info "    Required for fields: $(round(total_local / 1024^3, digits=3)) GB"
        @info "    Domain cache usage: $(round(gpu_memory_info.domain_memory / 1024^2, digits=2)) MB"
        
        # Check if fields will fit in GPU memory
        field_memory_gb = total_local / 1024^3
        available_gb = gpu_memory_info.available_memory / 1024^3
        
        if field_memory_gb > available_gb * 0.8  # Leave 20% safety margin
            @warn "  Fields may not fit in GPU memory! Consider reducing field count or resolution."
        else
            utilization = field_memory_gb / available_gb * 100
            @info "    Expected GPU utilization: $(round(utilization, digits=1))%"
        end
    end
    
    if domain.dist.size > 1
        avg_memory_per_process = total_global / domain.dist.size / 1024^2
        @info "  Average per process: $(round(avg_memory_per_process, digits=2)) MB"
    end
end

# GPU-specific domain utilities
function benchmark_domain_operations(domain::Domain; n_iterations::Int=100)
    """Benchmark domain operations on GPU vs CPU"""
    
    @info "Benchmarking domain operations ($(domain.device_config.device_type)):"
    
    # Benchmark coordinate generation
    @info "  Testing coordinate generation..."
    coord_time = @elapsed for i in 1:n_iterations
        coords = get_grid_coordinates_gpu(domain)
    end
    @info "    Average time: $(round(coord_time / n_iterations * 1000, digits=2)) ms"
    
    # Benchmark integration weights
    @info "  Testing integration weights..."
    weight_time = @elapsed for i in 1:n_iterations
        weights = integration_weights(domain)
    end
    @info "    Average time: $(round(weight_time / n_iterations * 1000, digits=2)) ms"
    
    # Benchmark meshgrid creation (for multi-dimensional domains)
    if length(domain.bases) > 1
        @info "  Testing meshgrid creation..."
        meshgrid_time = @elapsed for i in 1:n_iterations
            meshgrid = create_meshgrid_gpu(domain)
        end
        @info "    Average time: $(round(meshgrid_time / n_iterations * 1000, digits=2)) ms"
    end
    
    # Memory transfer benchmarks (GPU only)
    if domain.device_config.device_type != CPU_DEVICE
        @info "  Testing GPU memory transfers..."
        coords = get_grid_coordinates_gpu(domain)
        coord_name = domain.bases[1].meta.element_label
        test_array = coords[coord_name]
        
        # GPU to CPU transfer
        gpu_to_cpu_time = @elapsed for i in 1:n_iterations
            Array(test_array)
        end
        
        # CPU to GPU transfer  
        cpu_array = Array(test_array)
        cpu_to_gpu_time = @elapsed for i in 1:n_iterations
            device_array(cpu_array, domain.device_config)
        end
        
        array_size_mb = sizeof(test_array) / 1024^2
        @info "    GPU→CPU: $(round(gpu_to_cpu_time / n_iterations * 1000, digits=2)) ms ($(round(array_size_mb, digits=2)) MB)"
        @info "    CPU→GPU: $(round(cpu_to_gpu_time / n_iterations * 1000, digits=2)) ms ($(round(array_size_mb, digits=2)) MB)"
        
        gpu_to_cpu_bandwidth = array_size_mb * n_iterations / gpu_to_cpu_time * 1000  # MB/s
        cpu_to_gpu_bandwidth = array_size_mb * n_iterations / cpu_to_gpu_time * 1000  # MB/s
        
        @info "    Transfer bandwidth: GPU→CPU $(round(gpu_to_cpu_bandwidth, digits=1)) MB/s, CPU→GPU $(round(cpu_to_gpu_bandwidth, digits=1)) MB/s"
    end
end

function create_gpu_optimized_domain(dist::Distributor, domain_type::Symbol, args...; kwargs...)
    """Create GPU-optimized domain for common use cases"""
    
    # Select appropriate device
    device = get(kwargs, :device, "gpu")  # Default to GPU
    
    domain = if domain_type == :rayleigh_benard_2d
        aspect_ratio = get(args, 1, 4.0)
        Nx = get(args, 2, 256)
        Nz = get(args, 3, 64)
        rayleigh_benard_domain(dist, aspect_ratio, Nx, Nz; device=device, kwargs...)
        
    elseif domain_type == :channel_3d
        Lx = get(args, 1, 4π)
        Ly = get(args, 2, 2π)
        Lz = get(args, 3, 2.0)
        Nx = get(args, 4, 128)
        Ny = get(args, 5, 128)
        Nz = get(args, 6, 64)
        channel_flow_3d_domain(dist, Lx, Ly, Lz, Nx, Ny, Nz; device=device, kwargs...)
        
    elseif domain_type == :taylor_green_3d
        N = get(args, 1, 128)
        taylor_green_vortex_domain(dist, N; device=device, kwargs...)
        
    elseif domain_type == :periodic_2d
        Lx = get(args, 1, 2π)
        Ly = get(args, 2, 2π)
        Nx = get(args, 3, 128)
        Ny = get(args, 4, 128)
        create_2d_periodic_domain(dist, Lx, Ly, Nx, Ny; device=device, kwargs...)
        
    elseif domain_type == :periodic_3d
        Lx = get(args, 1, 2π)
        Ly = get(args, 2, 2π)
        Lz = get(args, 3, 2π)
        Nx = get(args, 4, 64)
        Ny = get(args, 5, 64)
        Nz = get(args, 6, 64)
        create_3d_periodic_domain(dist, Lx, Ly, Lz, Nx, Ny, Nz; device=device, kwargs...)
        
    else
        throw(ArgumentError("Unknown domain type: $domain_type"))
    end
    
    @info "Created GPU-optimized $domain_type domain on $(domain.device_config.device_type)"
    
    return domain
end

function optimize_domain_for_gpu!(domain::Domain)
    """Optimize domain settings for GPU performance"""
    
    if domain.device_config.device_type == CPU_DEVICE
        @info "Domain is on CPU, no GPU optimizations applied"
        return domain
    end
    
    @info "Optimizing domain for GPU performance..."
    
    # Pre-generate and cache commonly used arrays
    @info "  Pre-generating grid coordinates..."
    coords = get_grid_coordinates_gpu(domain)
    
    @info "  Pre-computing integration weights..."
    weights = integration_weights(domain)
    
    # Create meshgrid for multi-dimensional domains
    if length(domain.bases) > 1
        @info "  Creating meshgrid..."
        meshgrid = create_meshgrid_gpu(domain)
    end
    
    # Report memory usage
    mem_info = get_domain_memory_info(domain)
    @info "  GPU memory usage: $(round(mem_info.domain_memory / 1024^2, digits=2)) MB ($(round(mem_info.memory_utilization, digits=1))%)"
    
    @info "Domain optimization complete!"
    
    return domain
end

function compare_gpu_cpu_performance(domain_spec::Tuple)
    """Compare GPU vs CPU performance for domain operations"""
    
    dist, args = domain_spec[1], domain_spec[2:end]
    
    @info "Comparing GPU vs CPU performance for domain operations..."
    
    # Create CPU domain
    @info "Creating CPU domain..."
    cpu_domain = create_2d_periodic_domain(dist, args...; device="cpu")
    
    # Create GPU domain
    @info "Creating GPU domain..."
    gpu_domain = create_2d_periodic_domain(dist, args...; device="gpu")
    
    # Benchmark both
    @info "\nCPU Performance:"
    benchmark_domain_operations(cpu_domain; n_iterations=50)
    
    @info "\nGPU Performance:"
    benchmark_domain_operations(gpu_domain; n_iterations=50)
    
    # Memory comparison
    @info "\nMemory Usage Comparison:"
    @info "CPU Domain:"
    estimate_memory_usage(cpu_domain, 4)
    @info "\nGPU Domain:"
    estimate_memory_usage(gpu_domain, 4)
end
