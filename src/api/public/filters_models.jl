# Public exports: temporal filters and subgrid-scale (LES) models.
export
    TemporalFilter, ExponentialMean, ButterworthFilter, LagrangianFilter,
    get_mean, get_auxiliary, set_α!,
    update_displacement!, lagrangian_mean!, get_mean_velocity, get_displacement,
    filter_response, effective_averaging_time,
    add_temporal_filter!, has_temporal_filters, get_temporal_filter, get_all_temporal_filters,
    SGSModel, EddyViscosityModel,
    SmagorinskyModel, AMDModel,
    compute_eddy_viscosity!, compute_eddy_diffusivity!,
    compute_sgs_stress,
    get_eddy_viscosity, get_eddy_diffusivity, get_filter_width,
    mean_eddy_viscosity, max_eddy_viscosity,
    sgs_dissipation, mean_sgs_dissipation,
    set_constant!
