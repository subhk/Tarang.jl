# ============================================================================
# Pretty Printing for Tarang.jl 
# ============================================================================

using Printf

# Box drawing characters
const BOX_TL = "┌"
const BOX_TR = "┐"
const BOX_BL = "└"
const BOX_BR = "┘"
const BOX_H  = "─"
const BOX_V  = "│"
const BOX_LT = "├"
const BOX_RT = "┤"
const BOX_TT = "┬"
const BOX_BT = "┴"
const BOX_X  = "┼"

# Default box width
const BOX_WIDTH = 60

function _box_line(left, fill, right, width=BOX_WIDTH)
    return left * repeat(fill, width) * right
end

function _box_text(text, width=BOX_WIDTH)
    # Use textwidth for proper Unicode character width handling
    text_width = textwidth(text)
    pad_needed = max(0, width - 2 - text_width)
    return BOX_V * " " * text * repeat(" ", pad_needed) * " " * BOX_V
end

function _box_text_centered(text, width=BOX_WIDTH)
    # Use textwidth for proper Unicode character width handling
    # Must match _box_text which adds 1 space on each side
    text_width = textwidth(text)
    pad_total = max(0, width - text_width)
    pad_left = div(pad_total, 2)
    pad_right = pad_total - pad_left
    return BOX_V * repeat(" ", pad_left) * text * repeat(" ", pad_right) * BOX_V
end

# ============================================================================
# Basis pretty printing
# ============================================================================

function _basis_type_name(basis::Basis)
    T = typeof(basis)
    if T <: RealFourier
        return "RealFourier"
    elseif T <: ComplexFourier
        return "ComplexFourier"
    elseif T <: ChebyshevT
        return "ChebyshevT"
    elseif T <: ChebyshevU
        return "ChebyshevU"
    elseif T <: ChebyshevV
        return "ChebyshevV"
    elseif T <: Legendre
        return "Legendre"
    elseif T <: Jacobi
        return "Jacobi"
    elseif T <: Ultraspherical
        return "Ultraspherical"
    else
        return string(T)
    end
end

function Base.show(io::IO, basis::Basis)
    name = _basis_type_name(basis)
    coord = basis.meta.element_label
    N = basis.meta.size
    bounds = basis.meta.bounds
    print(io, "$name($coord, N=$N, bounds=$bounds)")
end

function Base.show(io::IO, ::MIME"text/plain", basis::Basis)
    name = _basis_type_name(basis)
    coord = basis.meta.element_label
    N = basis.meta.size
    bounds = basis.meta.bounds
    dealias = basis.meta.dealias

    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("$name Basis"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    println(io, _box_text("Coordinate:    $coord"))
    println(io, _box_text("Grid points:   $N"))
    println(io, _box_text(@sprintf("Bounds:        [%.4g, %.4g]", bounds[1], bounds[2])))
    println(io, _box_text(@sprintf("Dealias:       %.2f", dealias)))
    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# ScalarField pretty printing
# ============================================================================

function Base.show(io::IO, field::ScalarField)
    name = field.name
    ndims = length(field.bases)
    if ndims == 0
        print(io, "ScalarField($name, 0D)")
    else
        sizes = [b.meta.size for b in field.bases]
        print(io, "ScalarField($name, $(join(sizes, "×")))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", field::ScalarField)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("ScalarField: $(field.name)"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    if isempty(field.bases)
        println(io, _box_text("Dimensions:    0D (tau variable)"))
    else
        sizes = [b.meta.size for b in field.bases]
        total = prod(sizes)
        println(io, _box_text("Grid size:     $(join(sizes, " × "))"))
        println(io, _box_text(@sprintf("Total points:  %d (%.2fM)", total, total/1e6)))
        println(io, _box_text("Layout:        $(field.current_layout == :g ? "grid" : "spectral")"))
        println(io, _box_text("Data type:     $(field.dtype)"))
        println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
        println(io, _box_text("Bases:"))
        for (i, b) in enumerate(field.bases)
            bname = _basis_type_name(b)
            coord = b.meta.element_label
            N = b.meta.size
            println(io, _box_text("  [$i] $bname($coord, N=$N)"))
        end
    end
    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# VectorField pretty printing
# ============================================================================

function Base.show(io::IO, field::VectorField)
    name = field.name
    ncomp = length(field.components)
    if isempty(field.bases)
        print(io, "VectorField($name, $(ncomp)D, 0D)")
    else
        sizes = [b.meta.size for b in field.bases]
        print(io, "VectorField($name, $(ncomp)D, $(join(sizes, "×")))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", field::VectorField)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("VectorField: $(field.name)"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    ncomp = length(field.components)
    comp_names = [c.name for c in field.components]
    println(io, _box_text("Components:    $ncomp ($(join(comp_names, ", ")))"))

    if !isempty(field.bases)
        sizes = [b.meta.size for b in field.bases]
        total = prod(sizes) * ncomp
        println(io, _box_text("Grid size:     $(join(sizes, " × ")) × $ncomp"))
        println(io, _box_text(@sprintf("Total points:  %d (%.2fM)", total, total/1e6)))
        println(io, _box_text("Data type:     $(field.dtype)"))
    end
    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# Problem pretty printing
# ============================================================================

function _problem_type_name(prob::Problem)
    T = typeof(prob)
    if T == IVP
        return "Initial Value Problem (IVP)"
    elseif T == LBVP
        return "Linear Boundary Value Problem (LBVP)"
    elseif T == NLBVP
        return "Nonlinear Boundary Value Problem (NLBVP)"
    elseif T == EVP
        return "Eigenvalue Problem (EVP)"
    else
        return string(T)
    end
end

function Base.show(io::IO, prob::Problem)
    name = _problem_type_name(prob)
    nvars = length(prob.variables)
    neqs = length(prob.equations)
    print(io, "$name($nvars variables, $neqs equations)")
end

function Base.show(io::IO, ::MIME"text/plain", prob::Problem)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered(_problem_type_name(prob)))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    # Variables
    nvars = length(prob.variables)
    println(io, _box_text("Variables ($nvars):"))
    for var in prob.variables
        if isa(var, ScalarField)
            if isempty(var.bases)
                println(io, _box_text("  • $(var.name) [ScalarField, 0D]"))
            else
                sizes = [b.meta.size for b in var.bases]
                println(io, _box_text("  • $(var.name) [ScalarField, $(join(sizes, "×"))]"))
            end
        elseif isa(var, VectorField)
            ncomp = length(var.components)
            if isempty(var.bases)
                println(io, _box_text("  • $(var.name) [VectorField, $(ncomp)D, 0D]"))
            else
                sizes = [b.meta.size for b in var.bases]
                println(io, _box_text("  • $(var.name) [VectorField, $(ncomp)D, $(join(sizes, "×"))]"))
            end
        else
            println(io, _box_text("  • $(typeof(var))"))
        end
    end

    # Equations
    neqs = length(prob.equations)
    if neqs > 0
        println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
        println(io, _box_text("Equations ($neqs):"))
        for (i, eq) in enumerate(prob.equations)
            # Truncate long equations
            eq_short = length(eq) > 50 ? first(eq, 47) * "..." : eq
            println(io, _box_text("  [$i] $eq_short"))
        end
    end

    # Boundary conditions
    nbcs = length(prob.boundary_conditions)
    if nbcs > 0
        println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
        println(io, _box_text("Boundary Conditions ($nbcs):"))
        for (i, bc) in enumerate(prob.boundary_conditions)
            bc_short = length(bc) > 50 ? first(bc, 47) * "..." : bc
            println(io, _box_text("  [$i] $bc_short"))
        end
    end

    # Parameters
    nparams = length(prob.parameters)
    if nparams > 0 && nparams <= 10
        println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
        println(io, _box_text("Parameters ($nparams):"))
        for (k, v) in prob.parameters
            if isa(v, Number)
                println(io, _box_text(@sprintf("  • %s = %.4g", k, v)))
            end
        end
    end

    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# Solver pretty printing
# ============================================================================

function Base.show(io::IO, solver::InitialValueSolver)
    nvars = length(solver.state)
    t = solver.sim_time
    print(io, "InitialValueSolver($nvars fields, t=$t)")
end

function Base.show(io::IO, ::MIME"text/plain", solver::InitialValueSolver)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("Tarang.jl InitialValueSolver"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    # Timestepper
    ts_name = string(typeof(solver.timestepper))
    println(io, _box_text("Timestepper:   $ts_name"))
    println(io, _box_text(@sprintf("Time step:     %.2e", solver.dt)))
    println(io, _box_text(@sprintf("Current time:  %.4g", solver.sim_time)))
    println(io, _box_text(@sprintf("Stop time:     %.4g", solver.stop_sim_time)))
    println(io, _box_text("Iteration:     $(solver.iteration)"))

    # State fields
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    nfields = length(solver.state)
    println(io, _box_text("State Fields ($nfields):"))
    for field in solver.state
        if isempty(field.bases)
            println(io, _box_text("  • $(field.name) [0D]"))
        else
            sizes = [b.meta.size for b in field.bases]
            println(io, _box_text("  • $(field.name) [$(join(sizes, "×"))]"))
        end
    end

    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

function Base.show(io::IO, solver::BoundaryValueSolver)
    nvars = length(solver.state)
    print(io, "BoundaryValueSolver($nvars fields)")
end

function Base.show(io::IO, ::MIME"text/plain", solver::BoundaryValueSolver)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("Tarang.jl BoundaryValueSolver"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    nfields = length(solver.state)
    println(io, _box_text("State Fields ($nfields):"))
    for field in solver.state
        if isempty(field.bases)
            println(io, _box_text("  • $(field.name) [0D]"))
        else
            sizes = [b.meta.size for b in field.bases]
            println(io, _box_text("  • $(field.name) [$(join(sizes, "×"))]"))
        end
    end

    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# Distributor pretty printing
# ============================================================================

function Base.show(io::IO, dist::Distributor)
    ndims = dist.dim
    arch = dist.architecture
    arch_name = isa(arch, CPU) ? "CPU" : "GPU"
    if dist.size > 1
        print(io, "Distributor($(ndims)D, $arch_name, MPI)")
    else
        print(io, "Distributor($(ndims)D, $arch_name)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", dist::Distributor)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("Tarang.jl Distributor"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    ndims = dist.dim
    arch = dist.architecture
    arch_name = isa(arch, CPU) ? "CPU" : "GPU"

    println(io, _box_text("Dimensions:    $ndims"))
    println(io, _box_text("Architecture:  $arch_name"))
    println(io, _box_text("Data type:     $(dist.dtype)"))
    _use_mpi = dist.size > 1
    println(io, _box_text("MPI enabled:   $(_use_mpi)"))

    if _use_mpi
        println(io, _box_text("MPI ranks:     $(dist.size)"))
        println(io, _box_text("This rank:     $(dist.rank)"))
    end

    # Coordinate system
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    coord_names = dist.coordsys.names
    println(io, _box_text("Coordinates:   $(join(coord_names, ", "))"))

    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# Domain pretty printing
# ============================================================================

function Base.show(io::IO, domain::Domain)
    ndims = domain.dim
    sizes = [b.meta.size for b in domain.bases]
    print(io, "Domain($(ndims)D, $(join(sizes, "×")))")
end

function Base.show(io::IO, ::MIME"text/plain", domain::Domain)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("Tarang.jl Domain"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    println(io, _box_text("Dimensions:    $(domain.dim)"))

    # Grid shape
    sizes = [b.meta.size for b in domain.bases]
    total = prod(sizes)
    println(io, _box_text("Grid size:     $(join(sizes, " × "))"))
    println(io, _box_text(@sprintf("Total points:  %d (%.2fM)", total, total/1e6)))

    # Architecture
    arch = domain.dist.architecture
    arch_name = isa(arch, CPU) ? "CPU" : "GPU"
    println(io, _box_text("Architecture:  $arch_name"))
    println(io, _box_text("Data type:     $(domain.dist.dtype)"))

    # Bases
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
    println(io, _box_text("Bases:"))
    for (i, b) in enumerate(domain.bases)
        bname = _basis_type_name(b)
        coord = b.meta.element_label
        N = b.meta.size
        bounds = b.meta.bounds
        dealias = b.meta.dealias
        println(io, _box_text(@sprintf("  [%s] %s(N=%d, [%.4g, %.4g], dealias=%.1f)", coord, bname, N, bounds[1], bounds[2], dealias)))
    end

    # Volume
    vol = volume(domain)
    if isfinite(vol)
        println(io, _box_line(BOX_LT, BOX_H, BOX_RT))
        println(io, _box_text(@sprintf("Volume:        %.4g", vol)))
    end

    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# TensorField pretty printing
# ============================================================================

function Base.show(io::IO, field::TensorField)
    name = field.name
    ncomp = size(field.components)
    if isempty(field.bases)
        print(io, "TensorField($name, $(ncomp[1])×$(ncomp[2]), 0D)")
    else
        sizes = [b.meta.size for b in field.bases]
        print(io, "TensorField($name, $(ncomp[1])×$(ncomp[2]), $(join(sizes, "×")))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", field::TensorField)
    println(io, _box_line(BOX_TL, BOX_H, BOX_TR))
    println(io, _box_text_centered("TensorField: $(field.name)"))
    println(io, _box_line(BOX_LT, BOX_H, BOX_RT))

    nrows, ncols = size(field.components)
    println(io, _box_text("Components:    $(nrows) × $(ncols)"))

    if !isempty(field.bases)
        sizes = [b.meta.size for b in field.bases]
        total = prod(sizes) * nrows * ncols
        println(io, _box_text("Grid size:     $(join(sizes, " × ")) × $(nrows)×$(ncols)"))
        println(io, _box_text(@sprintf("Total points:  %d (%.2fM)", total, total/1e6)))
        println(io, _box_text("Data type:     $(field.dtype)"))
    end
    print(io, _box_line(BOX_BL, BOX_H, BOX_BR))
end

# ============================================================================
# Exports
# ============================================================================

# No explicit exports needed - Base.show methods are automatically used
