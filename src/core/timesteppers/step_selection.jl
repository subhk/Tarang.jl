# ============================================================================
# Timestepper Runtime Path Selection
#
# Keep environment/capability decisions here so timestepper implementation files
# can focus on their numerical algorithms.
# ============================================================================

const GLOBAL_MATRIX_IMPLICIT_DOF_LIMIT = 1_000_000

"""
    _timestepper_subproblems(solver) -> Tuple or nothing

Return the solver's assembled subproblems when the subproblem runtime path is
available. Non-tuple values are treated as absent to preserve older parameter
payloads without making every timestepper inspect `problem.parameters`.
"""
function _timestepper_subproblems(solver::InitialValueSolver)
    sps = get(solver.problem.parameters, "subproblems", nothing)
    return sps isa Tuple ? sps : nothing
end

@inline function _mpi_pencil_distribution_active(dist::Distributor)
    return dist.use_pencil_arrays && dist.size > 1
end

function _distributed_field_path_reason(fields::Vector{<:ScalarField})
    isempty(fields) && return nothing

    first_field = fields[1]
    is_gpu(field_architecture(first_field)) && return :gpu
    _mpi_pencil_distribution_active(first_field.dist) && return :mpi_pencil
    return nothing
end

"""
    _distributed_field_path_required(fields) -> Bool

Return true when a timestepper must operate on field objects instead of a
single CPU coefficient vector. This is required for GPU-resident fields and
for MPI PencilArrays distributions.
"""
_distributed_field_path_required(fields::Vector{<:ScalarField}) =
    _distributed_field_path_reason(fields) !== nothing

function _linear_operator_is_zero(L_matrix::AbstractMatrix)
    return (L_matrix isa SparseMatrixCSC && nnz(L_matrix) == 0) ||
           (norm(L_matrix, Inf) < 1e-14)
end

"""
    _imex_rk_explicit_fallback_reason(state, solver, current_state, L_matrix)

Return a symbolic reason for using the explicit RK fallback, or `nothing` when
the global-matrix IMEX RK path can run.
"""
function _imex_rk_explicit_fallback_reason(state,
                                           solver::InitialValueSolver,
                                           current_state::Vector{<:ScalarField},
                                           L_matrix)
    distributed_reason = _distributed_field_path_reason(current_state)
    if distributed_reason === :gpu
        return :gpu_without_subproblems
    elseif distributed_reason === :mpi_pencil
        return :mpi_without_subproblems
    end

    L_matrix === nothing && return :missing_linear_operator
    _linear_operator_is_zero(L_matrix) && return :zero_linear_operator
    return nothing
end

function _log_imex_rk_explicit_fallback(reason::Symbol)
    if reason === :gpu_without_subproblems
        @debug "IMEX RK: GPU detected without subproblems, falling back to explicit treatment"
    elseif reason === :mpi_without_subproblems
        @debug "IMEX RK: MPI PencilArrays detected without subproblems, falling back to explicit treatment"
    elseif reason === :missing_linear_operator
        @debug "IMEX RK: No L_matrix found, treating all terms explicitly"
    elseif reason === :zero_linear_operator
        @debug "IMEX RK: L_matrix is zero, falling back to explicit treatment"
    else
        @debug "IMEX RK: falling back to explicit treatment" reason
    end
end

"""
    _global_matrix_implicit_total_dofs(solver) -> Int

Count local coefficient degrees of freedom for legacy global-matrix implicit
methods. This centralizes the size heuristic used by MPI compatibility checks.
"""
function _global_matrix_implicit_total_dofs(solver::InitialValueSolver)
    total = 0
    for field in solver.state
        ensure_layout!(field, :c)
        coeffs = get_coeff_data(field)
        coeffs === nothing && continue
        total += length(coeffs)
    end
    return total
end

"""
    _global_matrix_implicit_matrices(solver) -> (L_matrix, M_matrix)

Return the matrix pair used by legacy global-matrix implicit methods.
Keeping this lookup centralized prevents every timestepper from knowing the
problem-parameter keys directly.
"""
function _global_matrix_implicit_matrices(solver::InitialValueSolver)
    L_matrix = _get_problem_matrix(solver.problem, "L_matrix")
    M_matrix = _get_problem_matrix(solver.problem, "M_matrix")
    return L_matrix, M_matrix
end

function _global_matrix_implicit_missing_matrix_reason(L_matrix, M_matrix)
    L_matrix === nothing && return :missing_linear_operator
    M_matrix === nothing && return :missing_mass_operator
    return nothing
end

function _global_matrix_implicit_distributed_fallback_reason(fields::Vector{<:ScalarField})
    isempty(fields) && return nothing
    if _mpi_pencil_distribution_active(fields[1].dist)
        return :mpi_pencil_without_subproblems
    end
    return nothing
end

function _log_global_matrix_implicit_distributed_fallback(method_name::String,
                                                         reason::Symbol,
                                                         fallback_name::String)
    if reason === :mpi_pencil_without_subproblems
        @debug "$method_name: MPI PencilArrays detected without subproblems, falling back to $fallback_name"
    else
        @debug "$method_name: global-matrix path unavailable, falling back to $fallback_name" reason
    end
end

function _log_global_matrix_implicit_matrix_fallback(method_name::String,
                                                    reason::Symbol,
                                                    fallback_name::String)
    if reason === :missing_linear_operator
        @warn "$method_name requires L_matrix and M_matrix, falling back to $fallback_name" maxlog=1
    elseif reason === :missing_mass_operator
        @warn "$method_name requires L_matrix and M_matrix, falling back to $fallback_name" maxlog=1
    else
        @warn "$method_name global-matrix path unavailable, falling back to $fallback_name" maxlog=1
    end
end

"""
    _check_mpi_implicit_compat!(solver, method_name)

Check if a global-matrix implicit timestepper can run with MPI fields.
These methods (SBDF3/4, MCNAB2, CNLF2) use global matrix solves that require
all field data on a single rank, a limitation of the legacy stepper path that
has not been ported to the subproblem architecture.
"""
function _check_mpi_implicit_compat!(solver::InitialValueSolver, method_name::String)
    dist = solver.state[1].dist
    if dist.size <= 1
        return nothing
    end

    total_dof = _global_matrix_implicit_total_dofs(solver)
    if total_dof > GLOBAL_MATRIX_IMPLICIT_DOF_LIMIT
        throw(ArgumentError(
            "$method_name with MPI requires global matrix solve (gather/scatter). " *
            "Total DOF=$total_dof exceeds $(GLOBAL_MATRIX_IMPLICIT_DOF_LIMIT) limit for gather/scatter approach. " *
            "Use a subproblem-compatible method instead: RK111, RK222, RK443, " *
            "CNAB1, CNAB2, SBDF1, SBDF2, or DiagonalIMEX_RK222/RK443 " *
            "(for purely Fourier domains)."))
    end

    @debug "$method_name: using global gather/scatter for MPI with $total_dof DOF"
    return nothing
end
