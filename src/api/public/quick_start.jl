export
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    ScalarField, VectorField, TensorField,
    IVP, EVP, LBVP, NLBVP,
    InitialValueSolver, EigenvalueSolver, BoundaryValueSolver,
    diagnose,
    add_parameters!, add_equation!, add_bc!,
    no_slip!, fixed_value!, free_slip!, insulating!,
    grid_data, coeff_data, set!,
    on_interval, on_sim_time,
    @root_only
