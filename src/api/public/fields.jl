# Public exports: coordinates, bases, domains, fields, and layout/data access.
@public_api(
    CartesianCoordinates,
    coords, unit_vector_fields,
    RealFourier, ComplexFourier, Fourier,
    ChebyshevT, ChebyshevU, ChebyshevV, Legendre, Jacobi,
    derivative_basis, product_matrix, ncc_matrix,
    Domain, Distributor, Field,
    field_architecture, synchronize_field_architecture!,
    gpu_fft_mode, set_gpu_fft_mode!,
    get_grid_data, get_coeff_data, set_grid_data!, set_coeff_data!,
    # Layout-enforcing reads: transform if needed, then return that layout's buffer.
    # Prefer these to the `ensure_layout!` + `get_*_data` pair.
    grid_data!, coeff_data!,
    FieldStorageMode, SerialStorage, PencilStorage,
    storage_mode, is_pencil_storage, is_serial_storage,
    stack_components, unstack_components!,
    ensure_layout!, forward_transform!, backward_transform!,
    setup_pencil_fft_transforms_2d!, setup_pencil_fft_transforms_3d!,
    get_cpu_data, get_cpu_local_data, get_local_data, is_gpu_field,
    local_grid, local_grids)
