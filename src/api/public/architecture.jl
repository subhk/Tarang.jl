export
    AbstractArchitecture, AbstractSerialArchitecture,
    CPU, GPU,
    device, array_type, architecture,
    on_architecture, is_gpu, has_cuda,
    synchronize, unsafe_free!,
    launch_config, workgroup_size, launch!, KernelOperation,
    create_array, move_to_architecture,
    set_gpu_fft_min_elements!, gpu_fft_min_elements, should_use_gpu_fft,
    allocate_like, similar_zeros, copy_to_device, is_gpu_array,
    _gpu_chebyshev_deriv!
