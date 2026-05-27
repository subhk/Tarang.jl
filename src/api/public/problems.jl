# Public exports: boundary conditions and time/space-dependent BC values.
export
    BoundaryConditionManager,
    DirichletBC, NeumannBC, RobinBC, PeriodicBC, StressFreeBC, CustomBC,
    dirichlet_bc, neumann_bc, robin_bc, periodic_bc, stress_free_bc, custom_bc,
    TimeDependentValue, SpaceDependentValue, TimeSpaceDependentValue, FieldReference,
    set_time_variable!, add_coordinate_field!, update_time_dependent_bcs!,
    has_time_dependent_bcs, has_space_dependent_bcs, requires_bc_update
