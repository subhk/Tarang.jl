"""
GPU-Compatible Exponential Timestepping Demo
===========================================

This example demonstrates how to use the GPU-compatible exponential timestepping
methods (ETDRK2, ETDAB2, ETDBDF2) with CUDA, AMD ROCm, Metal, or CPU.

Implemented Methods:
--------------------
1. ETDRK2 (ETD-RK222): 2nd-order exponential Runge-Kutta
   - Predictor-corrector structure with proper weighting
   - Formula: u_{n+1} = exp(hL)u_n + h*φ₁(hL)*[2*N(c) - N(u_n)]
   - Reference: Cox & Matthews (2002)

2. ETDAB2 (ETD-CNAB2): 2nd-order exponential Adams-Bashforth
   - Exponential propagation + Adams-Bashforth extrapolation
   - Formula: u_{n+1} = exp(hL)u_n + h*φ₁(hL)*[(1+w/2)*N(u_n) - (w/2)*N(u_{n-1})]
   - Reference: Hochbruck & Ostermann (2010)

3. ETDBDF2 (ETD-SBDF2): 2nd-order exponential BDF-style method
   - BDF2-like stability with exponential propagation
   - Formula: u_{n+1} = a₀*exp(hL)u_n + a₁*exp(hL)u_{n-1} + h*φ₁(hL)*N_BDF2
   - Provides enhanced stability for stiff problems

All methods use φ₁(z) = (exp(z)-1)/z for proper exponential integration.

Corrections Applied (2024):
---------------------------
- Fixed ETDRK2 to use proper second-order weighting (2*N(c) - N(u_n))
- Corrected ETDAB2 documentation (was mislabeled as CNAB)
- Reimplemented ETDBDF2 with mathematically sound formulation

Author: Claude (Anthropic AI Assistant)
Date: 2024
"""

using Tarang
using LinearAlgebra
using Printf

# Optional: GPU packages (loaded automatically by Tarang if available)
# using CUDA        # For NVIDIA GPUs
# using AMDGPU      # For AMD GPUs  
# using Metal       # For Apple Silicon GPUs

function demo_exponential_timesteppers()
    println("=== GPU-Compatible Exponential Timestepping Demo ===\n")
    
    # Example: Simple linear system du/dt = Lu + N(u)
    # where L is a linear operator and N(u) represents nonlinear terms
    n = 64  # Problem size
    
    # Create a simple diffusion-like linear operator: L = -k*∇²
    # For demonstration, use a discrete Laplacian
    k = 0.1  # Diffusion coefficient
    L_matrix = create_discrete_laplacian(n, k)
    M_matrix = Matrix{Float64}(I, n, n)  # Identity mass matrix
    
    println("Problem setup:")
    println("  - System size: $n × $n")
    println("  - Linear operator: L = -k*∇² (diffusion)")
    println("  - Diffusion coefficient: k = $k")
    println()
    
    # Initial condition: Gaussian profile
    x = range(-1.0, 1.0, length=n)
    u0 = exp.(-20.0 * x.^2)
    
    # Available devices
    available_devices = String[]
    push!(available_devices, "cpu")
    
    if Tarang.has_cuda
        push!(available_devices, "cuda")
        println("✓ CUDA GPU available")
    else
        println("✗ CUDA GPU not available")
    end
    
    if Tarang.has_amdgpu
        push!(available_devices, "amdgpu") 
        println("✓ AMD GPU available")
    else
        println("✗ AMD GPU not available")
    end
    
    if Tarang.has_metal
        push!(available_devices, "metal")
        println("✓ Metal GPU available")
    else
        println("✗ Metal GPU not available")
    end
    println()
    
    # Test each available device
    for device in available_devices
        println("Testing device: $device")
        println("-" ^ 40)
        
        test_exponential_methods(L_matrix, M_matrix, u0, device)
        println()
    end
end

function test_exponential_methods(L_matrix, M_matrix, u0, device::String)
    """Test all exponential methods on specified device"""
    
    n = length(u0)
    dt = 0.01
    T_final = 0.1
    n_steps = Int(T_final / dt)
    
    # Create fields (simplified - normally would use ScalarField from Tarang)
    initial_state = [SimpleField(copy(u0))]
    
    # Test each exponential method
    methods = [
        ("ETDRK2 (ETD-RK222)", ETD_RK222()),
        ("ETDAB2 (ETD-CNAB2)", ETD_CNAB2()),
        ("ETDBDF2 (ETD-SBDF2)", ETD_SBDF2())
    ]
    
    for (method_name, timestepper) in methods
        try
            println("  $method_name:")
            
            # Create solver (simplified)
            solver = create_dummy_solver(L_matrix, M_matrix)
            
            # Create timestepper state with device specification
            state = TimestepperState(timestepper, dt, initial_state, device)
            
            # Time the integration
            start_time = time()
            
            for step in 1:n_steps
                step!(state, solver)
                solver.sim_time += dt
            end
            
            # Synchronize GPU operations
            gpu_synchronize(state.device_config)
            elapsed_time = time() - start_time
            
            # Get final solution
            final_solution = fields_to_vector(state.history[end])
            solution_norm = norm(final_solution)
            
            @printf("    Time: %.4f s, ||u||: %.6f, Device: %s\n", 
                    elapsed_time, solution_norm, state.device_config.device_type)
                    
        catch e
            println("    ✗ Failed: $e")
        end
    end
end

# Utility functions for demo
function create_discrete_laplacian(n::Int, k::Float64)
    """Create discrete 1D Laplacian matrix"""
    h = 2.0 / (n - 1)  # Grid spacing
    
    # Second-order finite differences: -k * (u[i-1] - 2*u[i] + u[i+1]) / h²
    L = zeros(Float64, n, n)
    
    # Interior points
    for i in 2:n-1
        L[i, i-1] = k / h^2
        L[i, i]   = -2k / h^2
        L[i, i+1] = k / h^2
    end
    
    # Boundary conditions (homogeneous Dirichlet)
    L[1, 1] = -k / h^2
    L[1, 2] = k / h^2
    L[n, n-1] = k / h^2  
    L[n, n] = -k / h^2
    
    return L
end

struct SimpleField
    """Simplified field for demo purposes"""
    data::Vector{Float64}
    
    SimpleField(data) = new(copy(data))
end

# Adapt ScalarField interface for demo
Base.copy(f::SimpleField) = SimpleField(copy(f.data))

function fields_to_vector(fields::Vector{SimpleField})
    """Convert SimpleField array to vector"""
    if length(fields) == 1
        return fields[1].data
    else
        return vcat([f.data for f in fields]...)
    end
end

function copy_solution_to_fields!(fields::Vector{SimpleField}, solution::Vector{Float64})
    """Copy solution back to SimpleField"""
    if length(fields) == 1
        fields[1].data .= solution
    else
        # Handle multiple fields (not needed for this demo)
        offset = 1
        for field in fields
            len = length(field.data)
            field.data .= solution[offset:offset+len-1]
            offset += len
        end
    end
end

function create_dummy_solver(L_matrix, M_matrix)
    """Create simplified solver for demo"""
    
    # Create dummy solver structure
    solver = (
        problem = (
            parameters = Dict(
                "L_matrix" => L_matrix,
                "M_matrix" => M_matrix
            ),
        ),
        sim_time = 0.0
    )
    
    return solver
end

function evaluate_rhs(solver, state::Vector{SimpleField}, time::Float64)
    """Simplified RHS evaluation for demo"""
    
    # For this demo, just return linear term: L*u
    L_matrix = solver.problem.parameters["L_matrix"]
    u = fields_to_vector(state)
    
    # Linear evolution: du/dt = L*u
    rhs_data = L_matrix * u
    
    return [SimpleField(rhs_data)]
end

# User interface function
function run_exponential_demo(device::String="auto", device_id::Int=0)
    """
    Run exponential timestepping demo on specified device.
    
    Parameters:
    -----------
    device : String
        Device to use: "cpu", "gpu", "cuda", "amdgpu", "metal", or "auto"
    device_id : Int
        Device ID for multi-GPU systems (default: 0)
        
    Examples:
    ---------
    # Run on CPU
    run_exponential_demo("cpu")
    
    # Run on any available GPU
    run_exponential_demo("gpu")
    
    # Run on specific CUDA device
    run_exponential_demo("cuda", 1)
    
    # Auto-select best device
    run_exponential_demo("auto")
    """
    
    if device == "auto"
        # Auto-select best available device
        if Tarang.has_cuda
            device = "cuda"
        elseif Tarang.has_amdgpu
            device = "amdgpu"
        elseif Tarang.has_metal
            device = "metal"
        else
            device = "cpu"
        end
        println("Auto-selected device: $device")
    end
    
    println("\n=== Running Exponential Timestepper Demo ===")
    println("Device: $device")
    println("Device ID: $device_id")
    println()
    
    # Run the demo
    demo_exponential_timesteppers()
    
    println("\n=== Demo completed ===")
    println("\nUsage tips:")
    println("- ETD methods are most effective for stiff PDEs with linear + nonlinear terms")
    println("- ETDRK2: Best for general-purpose stiff problems (2 RHS evaluations/step)")
    println("- ETDAB2: More efficient multistep method (1 RHS evaluation/step after startup)")
    println("- ETDBDF2: Enhanced stability for very stiff problems, BDF-like damping")
    println("- GPU acceleration is most beneficial for large systems (n > 1000)")
    println("- Memory transfer overhead may dominate for small problems")
    println("- Use 'cpu' device for development and debugging")
    println("\nMethod Characteristics:")
    println("- All methods are 2nd-order accurate in time")
    println("- ETDRK2: One-step, self-starting")
    println("- ETDAB2, ETDBDF2: Two-step, require startup with ETDRK2")
    println("- φ₁ function handles stiffness analytically via matrix exponential")
end

# Run demo if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_exponential_demo()
end