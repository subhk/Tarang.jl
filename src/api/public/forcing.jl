# Public exports: stochastic and deterministic forcing.
export
    Forcing, StochasticForcingType, DeterministicForcingType,
    StochasticForcing, DeterministicForcing,
    generate_forcing!, reset_forcing!, set_dt!,
    apply_forcing!, get_forcing_real,
    energy_injection_rate, instantaneous_power,
    compute_forcing_spectrum,
    add_stochastic_forcing!, has_stochastic_forcing, get_stochastic_forcing
