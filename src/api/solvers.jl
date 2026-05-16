"""
    Tarang.Solvers

Facade for solver types, diagnostics, reducers, and linear algebra backends.
"""
module Solvers
import ..Tarang:
    InitialValueSolver, EigenvalueSolver, BoundaryValueSolver,
    diagnose, MatSolvers,
    GlobalFlowProperty, GlobalArrayReducer, CFL,
    global_max, global_min, global_mean, global_sum, reduce_scalar

export
    InitialValueSolver, EigenvalueSolver, BoundaryValueSolver,
    diagnose, MatSolvers,
    GlobalFlowProperty, GlobalArrayReducer, CFL,
    global_max, global_min, global_mean, global_sum, reduce_scalar
end
