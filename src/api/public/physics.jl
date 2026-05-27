# Public exports: physics setups — SQG, QG, boundary advection-diffusion.
export
    perp_grad, ∇⊥,
    sqg_streamfunction, sqg_velocity, sqg_problem_setup,
    QGSystem, qg_system_setup, qg_invert!, qg_step!,
    qg_surface_velocity!, qg_advection_rhs, qg_energy, extract_surface,
    BoundaryAdvectionDiffusion, BoundarySpec, DiffusionSpec,
    VelocitySource, PrescribedVelocity, InteriorDerivedVelocity, SelfDerivedVelocity,
    boundary_advection_diffusion_setup,
    bad_step!, bad_compute_velocity!, bad_compute_rhs!, bad_add_source!,
    bad_energy, bad_enstrophy, bad_max_velocity, bad_cfl_dt
