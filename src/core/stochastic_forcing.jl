"""
Stochastic and deterministic forcing.

Split into `core/forcing/` — this file only sets the load order, which matters:
the types must exist before the generation/application methods that dispatch on
them, and the exports come last.
"""

include("forcing/stochastic_forcing_types.jl")
include("forcing/stochastic_forcing_generation.jl")
include("forcing/stochastic_forcing_application.jl")
include("forcing/stochastic_forcing_diagnostics.jl")
include("forcing/stochastic_forcing_deterministic.jl")
