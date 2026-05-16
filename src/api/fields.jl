"""
    Tarang.Fields

Facade for field, domain, and data-layout operations.

This module intentionally re-exports existing Tarang bindings instead of moving
implementations. It gives users and internal code a stable architecture boundary
while preserving the long-standing `Tarang.X` API.
"""
module Fields
import ..Tarang:
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    Domain, Distributor, Field,
    ScalarField, VectorField, TensorField,
    grid_data, coeff_data, set!,
    ensure_layout!, forward_transform!, backward_transform!,
    get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!,
    field_architecture, synchronize_field_architecture!,
    get_cpu_data, get_cpu_local_data, get_local_data,
    local_grid, local_grids

export
    PeriodicDomain, ChebyshevDomain, ChannelDomain, ChannelDomain3D,
    Domain, Distributor, Field,
    ScalarField, VectorField, TensorField,
    grid_data, coeff_data, set!,
    ensure_layout!, forward_transform!, backward_transform!,
    get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!,
    field_architecture, synchronize_field_architecture!,
    get_cpu_data, get_cpu_local_data, get_local_data,
    local_grid, local_grids
end
