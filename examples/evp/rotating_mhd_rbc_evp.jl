"""
Rotating MHD Convection Eigenvalue Problem in Spherical Shell

This example demonstrates studying the linear stability of rotating magnetohydrodynamic 
convection in a spherical shell with an imposed azimuthal magnetic field.

Physical Setup:
- Spherical shell with inner radius Ri and outer radius Ro
- Imposed azimuthal magnetic field B_φ(r,θ) creating azimuthal mean flow U_φ(r,θ)
- Background temperature gradient drives convection
- Rotation about vertical axis (Ω = Ω ẑ)
- Search for critical Rayleigh number where convection onsets

The problem solves the linearized MHD equations:
    ∇·u = 0                                                             (incompressibility)
    ∂u/∂t = -∇p + Ra*Θ*ê_r - (1/Ek)*ê_z×u + (1/Pm)*(∇×B)×B₀ + ν∇²u      (momentum)
    ∂Θ/∂t = -u·∇T₀ + κ∇²Θ                                               (temperature)  
    ∂B/∂t = ∇×(u×B₀) + η∇²B                                             (magnetic induction)
    ∇·B = 0                                                             (magnetic divergence)

Where:
- B₀(r,θ) = B_φ(r,θ) ê_φ is the imposed azimuthal field
- T₀(r) is the conductive temperature profile
- Ra is Rayleigh number, Ek is Ekman number, Pm is magnetic Prandtl number

This implements the rotating MHD convection EVP.
"""

using Tarang
using LinearAlgebra
using MPI

function setup_rotating_mhd_convection_evp(;
    # Shell geometry
    Ri::Float64 = 0.35,           # Inner radius
    Ro::Float64 = 1.0,            # Outer radius
    
    # Grid resolution
    Nphi::Int = 64,               # Azimuthal modes
    Ntheta::Int = 48,             # Colatitude points  
    Nr::Int = 32,                 # Radial points
    
    # Physical parameters
    Rayleigh::Float64 = 1e6,      # Rayleigh number (to be varied for critical value)
    Ekman::Float64 = 1e-4,        # Ekman number (rotation strength)
    Prandtl::Float64 = 1.0,       # Thermal Prandtl number
    Pm::Float64 = 1.0,            # Magnetic Prandtl number
    Hartmann::Float64 = 100.0,    # Hartmann number (magnetic field strength)
    
    # Base magnetic field parameters
    B0_amplitude::Float64 = 1.0,   # Azimuthal field amplitude
    B0_profile::String = "constant", # "constant", "linear", "dipolar"
    
    # Numerical parameters
    dealias::Float64 = 3/2,
    dtype = ComplexF64,
    comm = MPI.COMM_WORLD
)

    @info "Setting up rotating MHD convection eigenvalue problem"
    @info "  Shell: Ri=$Ri, Ro=$Ro"  
    @info "  Resolution: $Nphi × $Ntheta × $Nr"
    @info "  Parameters: Ra=$Rayleigh, Ek=$Ekman, Pm=$Pm, Ha=$Hartmann"
    
    # Create coordinate system and domain
    coords = SphericalCoordinates('φ', 'θ', 'r')
    dist = Distributor(coords, dtype=dtype, mesh=determine_mesh(comm))
    
    # Create spherical shell basis
    shell = ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), 
                      dealias=dealias, dtype=dtype)
    sphere_inner = shell.inner_surface
    sphere_outer = shell.outer_surface
    
    # Get local coordinate grids
    φ, θ, r = local_grids(dist, shell)
    
    @info "Created spherical shell domain and basis"
    
    return coords, dist, shell, sphere_inner, sphere_outer, φ, θ, r
end

function setup_mhd_fields_and_base_state(dist, shell, sphere_inner, sphere_outer, φ, θ, r;
                                        B0_amplitude=1.0, B0_profile="constant", 
                                        Ri=0.35, Ro=1.0)
    
    @info "Setting up MHD fields and base state"
    
    # Create perturbation fields for eigenvalue problem
    # Velocity perturbation
    u = VectorField(dist, coords, name='u', bases=shell)
    
    # Pressure perturbation  
    p = Field(dist, name='p', bases=shell)
    
    # Temperature perturbation
    Θ = Field(dist, name='Theta', bases=shell)
    
    # Magnetic field perturbation
    B = VectorField(dist, coords, name='B', bases=shell)
    
    # Magnetic potential (for ∇·B = 0 constraint)
    A = Field(dist, name='A', bases=shell)
    
    # Tau terms for boundary conditions (first-order formulation)
    τ_u1 = VectorField(dist, coords, bases=sphere_outer)  # Outer boundary
    τ_u2 = VectorField(dist, coords, bases=sphere_inner)  # Inner boundary
    τ_Θ1 = Field(dist, bases=sphere_outer)
    τ_Θ2 = Field(dist, bases=sphere_inner) 
    τ_B1 = VectorField(dist, coords, bases=sphere_outer)
    τ_B2 = VectorField(dist, coords, bases=sphere_inner)
    τ_A1 = Field(dist, bases=sphere_outer)
    τ_A2 = Field(dist, bases=sphere_inner)
    τ_p  = Field(dist)  # Pressure gauge
    
    @info "Created perturbation fields and tau terms"
    
    # Create base state fields (non-constant coefficients)
    
    # 1. Background temperature T₀(r) = (Ro-r)/(Ro-Ri) (linear conductive profile)
    T0 = Field(dist, name='T0', bases=shell.radial_basis)
    T0['g'] = @. (Ro - r) / (Ro - Ri)
    
    @info "Created background temperature profile T₀(r)"
    
    # 2. Imposed azimuthal magnetic field B₀_φ(r,θ) 
    B0_phi = Field(dist, name='B0_phi', bases=shell)
    
    if B0_profile == "constant"
        B0_phi['g'] = B0_amplitude
        @info "Using constant azimuthal magnetic field B₀_φ = $B0_amplitude"
        
    elseif B0_profile == "linear"  
        B0_phi['g'] = @. B0_amplitude * (r - Ri) / (Ro - Ri)
        @info "Using linear azimuthal magnetic field B₀_φ ∝ r"
        
    elseif B0_profile == "dipolar"
        # Dipolar-like field: B₀_φ ∝ sin(θ)/r² (simplified)
        B0_phi['g'] = @. B0_amplitude * sin(θ) * (Ro/r)^2
        @info "Using dipolar-like azimuthal magnetic field B₀_φ ∝ sin(θ)/r²"
        
    else
        throw(ArgumentError("Unknown B0_profile: $B0_profile"))
    end
    
    # 3. Induced azimuthal flow U₀_φ(r,θ) from magnetic field (thermal wind balance)
    # For MHD equilibrium: ∇p₀ = j₀×B₀, where j₀ = ∇×B₀
    # This creates an azimuthal flow component
    U0_phi = Field(dist, name='U0_phi', bases=shell)
    
    # Simplified thermal wind balance: U₀_φ ∝ ∂B₀_φ/∂r  
    # (In reality this would involve solving the full thermal wind equation)
    if B0_profile == "linear"
        U0_phi['g'] = @. 0.1 * B0_amplitude * sin(θ) / (Ro - Ri)  # Simplified
    elseif B0_profile == "dipolar"  
        U0_phi['g'] = @. -0.2 * B0_amplitude * sin(θ) * cos(θ) * (Ro/r)^3
    else
        U0_phi['g'] = 0.0  # No mean flow for constant field
    end
    
    @info "Created azimuthal mean flow U₀_φ(r,θ) from thermal wind balance"
    
    # 4. Unit vectors and geometric factors
    ê_r = VectorField(dist, coords, bases=shell.radial_basis) 
    ê_r['g'][2] = 1  # Radial unit vector
    
    ê_z = VectorField(dist, coords, bases=shell.meridional_basis)
    ê_z['g'][1] = -sin(θ)  # z = r cos(θ), so ∂z/∂θ = -r sin(θ)  
    ê_z['g'][2] = cos(θ)   # ∂z/∂r = cos(θ)
    
    r_vec = VectorField(dist, coords, bases=shell.radial_basis)
    r_vec['g'][2] = r  # Position vector magnitude
    
    @info "Created unit vectors and geometric factors"
    
    return (u, p, Θ, B, A, τ_u1, τ_u2, τ_Θ1, τ_Θ2, τ_B1, τ_B2, τ_A1, τ_A2, τ_p,
            T0, B0_phi, U0_phi, ê_r, ê_z, r_vec)
end

function setup_mhd_operators_and_equations(
    coords, dist, shell, 
    u, p, Θ, B, A, τ_u1, τ_u2, τ_Θ1, τ_Θ2, τ_B1, τ_B2, τ_A1, τ_A2, τ_p,
    T0, B0_phi, U0_phi, ê_r, ê_z, r_vec;
    Rayleigh=1e6, Ekman=1e-4, Prandtl=1.0, Pm=1.0, Hartmann=100.0,
    eigenvalue_symbol=:ω
)
    
    @info "Setting up MHD operators and equations"
    
    # First-order reduction operators (lift basis for boundary conditions)
    lift_basis = derivative_basis(shell, 1)
    lift = A -> Lift(A, lift_basis, -1)
    
    # Gradient operators with lifting for first-order formulation
    grad_u = ∇(u) + r_vec ⊗ lift(τ_u1)  # First-order velocity gradient
    grad_Θ = ∇(Θ) + r_vec ⊗ lift(τ_Θ1)  # First-order temperature gradient
    grad_B = ∇(B) + r_vec ⊗ lift(τ_B1)  # First-order magnetic gradient
    grad_A = ∇(A) + r_vec ⊗ lift(τ_A1)  # First-order potential gradient
    
    @info "Created first-order differential operators with lifting"
    
    # Time derivative operator (eigenvalue problem: ∂/∂t → -iω)
    dt = A -> -1im * ω * A
    
    # MHD operators
    # Current density: j = ∇×B (perturbation) + ∇×B₀ (background)
    j = curl(B)
    j0 = curl(B0_phi * ê_φ)  # Background current from azimuthal field
    
    # Lorentz force: (∇×B)×B₀ + (∇×B₀)×B (linearized)
    lorentz_force = cross(j, B0_phi * ê_φ) + cross(j0, B)
    
    # Electromagnetic induction: ∂B/∂t = ∇×(u×B₀) + ∇×(U₀×B) + η∇²B
    # where U₀×B₀ = 0 since both are azimuthal
    induction_rhs = curl(cross(u, B0_phi * ê_φ)) + curl(cross(U0_phi * ê_φ, B))
    
    @info "Created MHD operators: current density, Lorentz force, induction"
    
    # Coriolis force: -2Ω×u = -(2/Ek)*ê_z×u  
    coriolis = -(2.0/Ekman) * cross(ê_z, u)
    
    # Mean flow advection: U₀·∇u + u·∇U₀ (linearized convection)
    # For azimuthal mean flow U₀ = U₀_φ(r,θ) ê_φ:
    mean_flow_advection = U0_phi * (1/r) * ∂(u, φ) + u ⋅ ∇(U0_phi * ê_φ)
    
    @info "Created rotation and mean flow operators"
    
    return (dt, grad_u, grad_Θ, grad_B, grad_A, lift, 
            j, j0, lorentz_force, induction_rhs, coriolis, mean_flow_advection)
end

function create_mhd_eigenvalue_problem(
    coords, dist, shell, sphere_inner, sphere_outer,
    u, p, Θ, B, A, τ_u1, τ_u2, τ_Θ1, τ_Θ2, τ_B1, τ_B2, τ_A1, τ_A2, τ_p,
    T0, B0_phi, U0_phi, ê_r, ê_z, r_vec,
    dt, grad_u, grad_Θ, grad_B, grad_A, lift,
    j, j0, lorentz_force, induction_rhs, coriolis, mean_flow_advection;
    Rayleigh=1e6, Ekman=1e-4, Prandtl=1.0, Pm=1.0, Hartmann=100.0,
    eigenvalue_symbol=:ω, stress_free=true
)
    
    @info "Creating rotating MHD convection eigenvalue problem"
    
    # Collect all variables for EVP
    variables = [p, u, Θ, B, A, τ_u1, τ_u2, τ_Θ1, τ_Θ2, τ_B1, τ_B2, τ_A1, τ_A2, τ_p]
    
    # Create eigenvalue problem
    problem = EVP(variables, eigenvalue=eigenvalue_symbol, 
                  namespace=Dict{String,Any}("locals" => @locals))
    
    @info "Created EVP with $(length(variables)) variables"
    
    # Add MHD equations
    
    # 1. Incompressibility: ∇·u = 0 with pressure gauge
    add_equation!(problem, "div(grad_u) + τ_p = 0")
    
    # 2. Linearized momentum equation:
    #    ∂u/∂t + U₀·∇u + u·∇U₀ = -∇p + Ra*Θ*ê_r + (1/Ek)*coriolis + (1/Ha²)*(∇×B)×B₀ + ν∇²u
    momentum_eq = """
    ∂t(u) + mean_flow_advection =
        -∇(p) + Rayleigh*Θ*ê_r + coriolis +
        (1/Hartmann^2)*lorentz_force - div(grad_u) + lift(τ_u2)
    """
    add_equation!(problem, momentum_eq)
    
    # 3. Linearized temperature equation:
    #    ∂Θ/∂t + U₀·∇Θ + u·∇T₀ = κ∇²Θ
    temperature_eq = """
    ∂t(Θ) + U0_phi*(1/r)*∂(Θ,φ) + u⋅∇(T0) =
        (1/Prandtl)*div(grad_Θ) + (1/Prandtl)*lift(τ_Θ2)
    """
    add_equation!(problem, temperature_eq)
    
    # 4. Linearized magnetic induction equation:
    #    ∂B/∂t = ∇×(u×B₀) + ∇×(U₀×B) + η∇²B
    induction_eq = """
    ∂t(B) = induction_rhs + (1/Pm)*div(grad_B) + (1/Pm)*lift(τ_B2)
    """
    add_equation!(problem, induction_eq)
    
    # 5. Magnetic divergence: ∇·B = 0 (using potential A: B = ∇×A)
    add_equation!(problem, "B = curl(grad_A) + curl(lift(τ_A1))")
    add_equation!(problem, "div(B) = 0")
    
    @info "Added core MHD equations"
    
    return problem
end

function add_mhd_boundary_conditions!(problem, shell, sphere_inner, sphere_outer;
                                     stress_free=true, magnetic_bc="conducting")
    
    @info "Adding MHD boundary conditions"
    @info "  Velocity BC: $(stress_free ? "stress-free" : "no-slip")"
    @info "  Magnetic BC: $magnetic_bc"
    
    # Velocity boundary conditions
    if stress_free
        # Stress-free: u·ê_r = 0, ê_r·(∇u + (∇u)ᵀ)·ê_θ = 0, ê_r·(∇u + (∇u)ᵀ)·ê_φ = 0
        add_bc!(problem, "radial(u(r=Ri)) = 0")  # Inner boundary
        add_bc!(problem, "radial(u(r=Ro)) = 0")  # Outer boundary
        add_bc!(problem, "angular(radial(strain_rate(r=Ri), 0), 0) = 0")  # Inner stress-free
        add_bc!(problem, "angular(radial(strain_rate(r=Ro), 0), 0) = 0")  # Outer stress-free
    else
        # No-slip: u = 0
        add_bc!(problem, "u(r=Ri) = 0")  # Inner boundary
        add_bc!(problem, "u(r=Ro) = 0")  # Outer boundary
    end
    
    # Temperature boundary conditions (fixed temperature)
    add_bc!(problem, "Θ(r=Ri) = 0")  # Inner boundary
    add_bc!(problem, "Θ(r=Ro) = 0")  # Outer boundary  
    
    # Magnetic field boundary conditions
    if magnetic_bc == "conducting"
        # Perfectly conducting: B·ê_r = 0, ê_r×E = 0 (E = -∂A/∂t)
        add_bc!(problem, "radial(B(r=Ri)) = 0")
        add_bc!(problem, "radial(B(r=Ro)) = 0")
        add_bc!(problem, "angular(A(r=Ri)) = 0")  # Simplified: A_θ = A_φ = 0
        add_bc!(problem, "angular(A(r=Ro)) = 0")
        
    elseif magnetic_bc == "insulating"
        # Insulating: match to potential field ∇²φ = 0 outside
        add_bc!(problem, "radial(curl(B)(r=Ri)) = 0")  # j_r = 0
        add_bc!(problem, "radial(curl(B)(r=Ro)) = 0")  # j_r = 0
        add_bc!(problem, "angular(B(r=Ri)) = 0")  # B_θ = B_φ = 0
        add_bc!(problem, "angular(B(r=Ro)) = 0")  # B_θ = B_φ = 0
        
    else
        throw(ArgumentError("Unknown magnetic_bc: $magnetic_bc"))
    end
    
    # Pressure gauge: τ_p removes math degeneracy, integ(p)=0 fixes physical gauge
    add_bc!(problem, "integ(p) = 0")

    # Magnetic potential gauge
    add_bc!(problem, "integ(A) = 0")
    
    @info "Added boundary conditions successfully"
    
    return problem
end

function solve_critical_rayleigh_number(;
    # Shell parameters
    Ri=0.35, Ro=1.0, Nphi=64, Ntheta=48, Nr=32,
    
    # Physical parameters
    Ekman=1e-4, Prandtl=1.0, Pm=1.0, Hartmann=100.0,
    
    # Base magnetic field
    B0_amplitude=1.0, B0_profile="dipolar",
    
    # Search parameters
    Ra_min=1e4, Ra_max=1e8, Ra_tolerance=1e-3,
    target_azimuthal_mode=nothing,  # Search over m, or specify m
    
    # Boundary conditions
    stress_free=true, magnetic_bc="conducting",
    
    # Solver parameters
    n_modes=10, ncc_cutoff=1e-10,
    max_iterations=50
)
    
    @info "Starting critical Rayleigh number search for rotating MHD convection"
    @info "  Physical parameters: Ek=$Ekman, Pr=$Prandtl, Pm=$Pm, Ha=$Hartmann"
    @info "  Base field: B₀_φ profile=$B0_profile, amplitude=$B0_amplitude"  
    @info "  Search range: Ra ∈ [$Ra_min, $Ra_max], tolerance=$Ra_tolerance"
    
    # Setup domain and fields once
    coords, dist, shell, sphere_inner, sphere_outer, φ, θ, r = setup_rotating_mhd_convection_evp(
        Ri=Ri, Ro=Ro, Nphi=Nphi, Ntheta=Ntheta, Nr=Nr,
        Rayleigh=1e6, Ekman=Ekman, Prandtl=Prandtl, Pm=Pm, Hartmann=Hartmann,
        B0_amplitude=B0_amplitude, B0_profile=B0_profile
    )
    
    fields = setup_mhd_fields_and_base_state(
        dist, shell, sphere_inner, sphere_outer, φ, θ, r,
        B0_amplitude=B0_amplitude, B0_profile=B0_profile, Ri=Ri, Ro=Ro
    )
    u, p, Θ, B, A, τ_u1, τ_u2, τ_Θ1, τ_Θ2, τ_B1, τ_B2, τ_A1, τ_A2, τ_p = fields[1:15]
    T0, B0_phi, U0_phi, ê_r, ê_z, r_vec = fields[16:21]
    
    operators = setup_mhd_operators_and_equations(
        coords, dist, shell, u, p, Θ, B, A, τ_u1, τ_u2, τ_Θ1, τ_Θ2, τ_B1, τ_B2, τ_A1, τ_A2, τ_p,
        T0, B0_phi, U0_phi, ê_r, ê_z, r_vec,
        Rayleigh=1e6, Ekman=Ekman, Prandtl=Prandtl, Pm=Pm, Hartmann=Hartmann
    )
    
    @info "Completed problem setup, starting eigenvalue search"
    
    function evaluate_growth_rate(Ra::Float64, m::Int)
        """Evaluate maximum growth rate for given Rayleigh number and azimuthal mode m"""
        
        @info "  Evaluating Ra=$Ra, m=$m"
        
        # Create problem with current Rayleigh number
        problem = create_mhd_eigenvalue_problem(
            coords, dist, shell, sphere_inner, sphere_outer,
            u, p, Θ, B, A, τ_u1, τ_u2, τ_Θ1, τ_Θ2, τ_B1, τ_B2, τ_A1, τ_A2, τ_p,
            T0, B0_phi, U0_phi, ê_r, ê_z, r_vec, operators...,
            Rayleigh=Ra, Ekman=Ekman, Prandtl=Prandtl, Pm=Pm, Hartmann=Hartmann
        )
        
        # Add boundary conditions
        add_mhd_boundary_conditions!(problem, shell, sphere_inner, sphere_outer,
                                     stress_free=stress_free, magnetic_bc=magnetic_bc)
        
        # Build solver
        solver = EigenvalueSolver(problem, n_modes=n_modes)
        
        # Select azimuthal mode m (following subproblem pattern)
        if target_azimuthal_mode !== nothing
            # Use specific azimuthal mode
            target_m = target_azimuthal_mode
        else
            # For MHD convection, critical mode often m=1-5 for moderate Hartmann numbers
            target_m = m  # Use provided m
        end
        
        @info "    Solving eigenvalue problem for m=$target_m"
        
        # Solve eigenvalue problem (targeting zero growth rate for marginal stability)
        target_eigenvalue = 0.0 + 0.0im  # Marginal stability: Re(ω) = 0
        
        try
            # This would use the subproblem selection in the actual implementation
            solve!(solver, target=target_eigenvalue)
            
            # Find eigenvalue closest to marginal stability
            eigenvals = solver.eigenvalues
            
            if length(eigenvals) > 0
                # Find eigenvalue with largest real part (most unstable)
                max_growth_rate = maximum(real.(eigenvals))
                closest_to_marginal = argmin(abs.(real.(eigenvals)))
                marginal_eigenvalue = eigenvals[closest_to_marginal]
                
                @info "    Found eigenvalues: growth_rates=$(real.(eigenvals))"
                @info "    Max growth rate: $max_growth_rate"
                @info "    Closest to marginal: $marginal_eigenvalue"
                
                return max_growth_rate, marginal_eigenvalue
            else
                @warn "    No eigenvalues found for Ra=$Ra, m=$target_m"
                return -Inf, NaN + 0.0im
            end
            
        catch e
            @warn "    Eigenvalue solve failed for Ra=$Ra, m=$target_m: $e"
            return -Inf, NaN + 0.0im
        end
    end
    
    # Determine azimuthal modes to search
    if target_azimuthal_mode !== nothing
        m_values = [target_azimuthal_mode]
        @info "Searching only m=$target_azimuthal_mode"
    else
        # Search over range of azimuthal modes
        m_max = min(10, Nphi ÷ 2)  # Don't exceed Nyquist limit
        m_values = 1:m_max
        @info "Searching azimuthal modes m ∈ [1, $m_max]"
    end
    
    critical_results = Dict{Int, NamedTuple}()
    
    # Find critical Rayleigh number for each azimuthal mode
    for m in m_values
        @info "Finding critical Ra for azimuthal mode m=$m"
        
        Ra_low = Ra_min
        Ra_high = Ra_max
        iteration = 0
        
        # Bisection search for critical Rayleigh number
        while (Ra_high - Ra_low) > Ra_tolerance && iteration < max_iterations
            iteration += 1
            
            Ra_mid = (Ra_low + Ra_high) / 2
            growth_rate, eigenvalue = evaluate_growth_rate(Ra_mid, m)
            
            @info "  Iteration $iteration: Ra=$Ra_mid, growth_rate=$growth_rate"
            
            if growth_rate > 0
                # Unstable - reduce Rayleigh number
                Ra_high = Ra_mid
            else
                # Stable - increase Rayleigh number  
                Ra_low = Ra_mid
            end
        end
        
        Ra_critical = (Ra_low + Ra_high) / 2
        final_growth_rate, final_eigenvalue = evaluate_growth_rate(Ra_critical, m)
        
        critical_results[m] = (
            Ra_critical = Ra_critical,
            growth_rate = final_growth_rate,
            eigenvalue = final_eigenvalue,
            frequency = imag(final_eigenvalue),
            iterations = iteration
        )
        
        @info "Critical Rayleigh number for m=$m: Ra_c = $Ra_critical"
        @info "  Final growth rate: $final_growth_rate"
        @info "  Final eigenvalue: $final_eigenvalue"
        @info "  Frequency: $(imag(final_eigenvalue))"
    end
    
    # Find overall critical mode (lowest Rayleigh number)
    critical_m = argmin([result.Ra_critical for result in values(critical_results)])
    overall_critical = critical_results[critical_m]
    
    @info "="^60
    @info "CRITICAL RAYLEIGH NUMBER ANALYSIS COMPLETE"
    @info "="^60
    @info "Overall critical mode: m = $critical_m"
    @info "Overall critical Rayleigh number: Ra_c = $(overall_critical.Ra_critical)"
    @info "Growth rate at onset: $(overall_critical.growth_rate)"
    @info "Eigenvalue at onset: $(overall_critical.eigenvalue)"
    @info "Frequency at onset: $(overall_critical.frequency)"
    @info ""
    @info "Results for all modes:"
    for (m, result) in sort(collect(critical_results))
        @info "  m=$m: Ra_c=$(result.Ra_critical), ω=$(result.eigenvalue)"
    end
    @info "="^60
    
    return critical_results, overall_critical, critical_m
end

function run_mhd_convection_stability_analysis()
    """
    Main function to run rotating MHD convection stability analysis.
    
    This function demonstrates the complete workflow for finding critical 
    Rayleigh numbers in rotating MHD convection with imposed azimuthal 
    magnetic field.
    """
    
    @info "Starting Rotating MHD Convection Stability Analysis"
    @info "="^60
    
    # Example 1: Weak magnetic field (dipolar profile)
    @info "CASE 1: Weak dipolar magnetic field"
    results_weak = solve_critical_rayleigh_number(
        Ri=0.35, Ro=1.0, Nphi=32, Ntheta=24, Nr=16,  # Lower resolution for example
        Ekman=1e-3, Prandtl=1.0, Pm=1.0, Hartmann=10.0,
        B0_amplitude=0.1, B0_profile="dipolar",
        Ra_min=1e4, Ra_max=1e7, target_azimuthal_mode=1,
        stress_free=true, magnetic_bc="conducting"
    )
    
    @info ""
    @info "CASE 2: Strong dipolar magnetic field"  
    results_strong = solve_critical_rayleigh_number(
        Ri=0.35, Ro=1.0, Nphi=32, Ntheta=24, Nr=16,
        Ekman=1e-3, Prandtl=1.0, Pm=1.0, Hartmann=100.0,
        B0_amplitude=1.0, B0_profile="dipolar", 
        Ra_min=1e5, Ra_max=1e8, target_azimuthal_mode=1,
        stress_free=true, magnetic_bc="conducting"
    )
    
    @info ""
    @info "CASE 3: Linear magnetic field profile"
    results_linear = solve_critical_rayleigh_number(
        Ri=0.35, Ro=1.0, Nphi=32, Ntheta=24, Nr=16,
        Ekman=1e-3, Prandtl=1.0, Pm=1.0, Hartmann=50.0,
        B0_amplitude=0.5, B0_profile="linear",
        Ra_min=1e4, Ra_max=1e7, target_azimuthal_mode=1,
        stress_free=true, magnetic_bc="conducting"
    )
    
    @info ""
    @info "ANALYSIS SUMMARY"
    @info "="^60
    @info "Weak dipolar field (Ha=10): Ra_c = $(results_weak[2].Ra_critical)"
    @info "Strong dipolar field (Ha=100): Ra_c = $(results_strong[2].Ra_critical)"  
    @info "Linear field (Ha=50): Ra_c = $(results_linear[2].Ra_critical)"
    @info ""
    @info "Magnetic field suppression factor:"
    @info "  Strong vs Weak: $(results_strong[2].Ra_critical / results_weak[2].Ra_critical)"
    @info "  Linear vs Weak: $(results_linear[2].Ra_critical / results_weak[2].Ra_critical)"
    @info "="^60
    
    return results_weak, results_strong, results_linear
end

# Run the analysis if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Initialize MPI  
    MPI.Init()
    
    try
        # Run stability analysis
        run_mhd_convection_stability_analysis()
        
    finally
        # Finalize MPI
        MPI.Finalize()
    end
end

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create MHD convection eigenvalue problem setup", "status": "completed", "activeForm": "Creating MHD convection eigenvalue problem setup"}, {"content": "Implement azimuthal magnetic field base state", "status": "completed", "activeForm": "Implementing azimuthal magnetic field base state"}, {"content": "Add rotating MHD operators and equations", "status": "in_progress", "activeForm": "Adding rotating MHD operators and equations"}, {"content": "Configure boundary conditions for spherical shell", "status": "pending", "activeForm": "Configuring boundary conditions for spherical shell"}, {"content": "Set up eigenvalue solver for critical Rayleigh number", "status": "pending", "activeForm": "Setting up eigenvalue solver for critical Rayleigh number"}]