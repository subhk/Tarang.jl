"""
    Tarang.Problems

Facade for problem definitions, equations, and boundary conditions.
"""
module Problems
import ..Tarang:
    IVP, EVP, LBVP, NLBVP,
    add_parameters!, add_equation!, add_bc!,
    no_slip!, fixed_value!, free_slip!, insulating!,
    BoundaryConditionManager,
    DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
    set_time_variable!, add_coordinate_field!,
    update_time_dependent_bcs!, has_time_dependent_bcs,
    has_space_dependent_bcs, requires_bc_update

export
    IVP, EVP, LBVP, NLBVP,
    add_parameters!, add_equation!, add_bc!,
    no_slip!, fixed_value!, free_slip!, insulating!,
    BoundaryConditionManager,
    DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
    set_time_variable!, add_coordinate_field!,
    update_time_dependent_bcs!, has_time_dependent_bcs,
    has_space_dependent_bcs, requires_bc_update
end
