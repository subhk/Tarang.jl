# ---------------------------------------------------------------------------
# Merge-time layout and spectral reconstruction
#
# Deciding what SPACE a stored field is in (grid, coefficient, or mixed per axis),
# converting between them, and rebuilding a global spectral field from per-rank
# pieces. Split out of `netcdf_merge.jl`, which had grown past 2000 lines mixing
# this with the merge orchestration — opening files, walking processor sets,
# concatenating along time, and writing the merged output.
#
# The two concerns fail differently and are worth reading apart. Orchestration
# fails loudly: a missing file or an absent variable raises. This layer fails
# QUIETLY, because every question it answers is an inference — is this axis
# spectral? is this array grid-space? which layout should the merged field use? —
# and a wrong inference produces a plausible array rather than an error. The
# functions here are the ones a value-level audit needs to reach.
#
# Everything is a plain function over data plus a `merger` passed as an argument;
# nothing here constructs a `NetCDFMerger` or owns file lifetime.
# ---------------------------------------------------------------------------

"""Determine field layout type from grid_space flags array."""
function determine_layout_type(grid_space_flags)
    if isa(grid_space_flags, AbstractArray) && length(grid_space_flags) > 0
        flags = length(grid_space_flags) > 1 ? grid_space_flags[2:end] : grid_space_flags
        all_grid = all(flags)
        all_coeff = all(.!flags)
        
        if all_grid
            return :grid_space
        elseif all_coeff
            return :coeff_space
        else
            return :mixed_layout
        end
    else
        return :unknown
    end
end

function _bool_from_value(value)
    if value isa Bool
        return value
    elseif value isa Number
        return value != 0
    elseif value isa AbstractString
        lower = lowercase(strip(value))
        return lower in ("true", "t", "1", "yes", "y", "grid", "g")
    else
        return false
    end
end

"""
Normalize grid_space flags to a Bool vector matching data dimensions.
Assumes the first dimension is time and should not be transformed.
"""
function normalize_grid_space_flags(flags, ndims_data::Int)
    if flags === nothing
        return nothing
    end

    if flags isa AbstractArray
        bools = Bool[_bool_from_value(f) for f in flags]
    else
        bools = Bool[_bool_from_value(flags)]
    end

    if ndims_data <= 0
        return bools
    end

    if isempty(bools)
        return fill(true, ndims_data)
    end

    if length(bools) == ndims_data - 1
        return vcat(true, bools)
    elseif length(bools) == ndims_data
        return bools
    else
        # Fallback: assume all spatial dims are grid space
        return vcat(true, fill(bools[1], ndims_data - 1))
    end
end

"""
Merge field in grid space (physical space).
Grid space fields are typically distributed spatially across processors.
"""
function merge_grid_space_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    merger.verbose && println("        Merging grid space field")
    var_attrs["merge_strategy"] = "grid_space_spatial_reconstruction"
    
    # Use spatial reconstruction (like merge_variable_reconstruct!)
    data_type = eltype(processor_data[1]["data"])
    global_shape = nothing
    
    # Try to determine global shape from layout info
    for proc_info in processor_data
        layout_info = proc_info["layout_info"]
        if layout_info["global_shape"] !== nothing
            global_shape = layout_info["global_shape"]
            break
        end
    end
    
    # If no global shape, estimate from domain decomposition
    if global_shape === nothing
        inferred_shape = infer_slab_decomposition!(processor_data, merger)
        if inferred_shape !== nothing
            global_shape = inferred_shape
        else
            global_shape = estimate_global_shape_from_decomposition(processor_data)
        end
    end
    
    # Infer missing start/count metadata if needed
    has_domain_info = any(p["layout_info"]["start"] !== nothing && p["layout_info"]["count"] !== nothing for p in processor_data)
    if !has_domain_info
        inferred_shape = infer_slab_decomposition!(processor_data, merger)
        if inferred_shape !== nothing
            merger.verbose && println("        Inferred slab decomposition for grid space field")
            if global_shape === nothing
                global_shape = inferred_shape
            end
        end
    end

    # Initialize global field
    reconstructed_data = zeros(data_type, global_shape)
    coverage_mask = falses(global_shape)
    
    # Reconstruct using spatial domain information
    for proc_info in processor_data
        data = proc_info["data"]
        layout_info = proc_info["layout_info"]
        
        start_indices = layout_info["start"]
        count_indices = layout_info["count"]
        
        if start_indices !== nothing && count_indices !== nothing
            # Create spatial slices (skip time dimension)
            spatial_slices = Any[Colon()]  # Time dimension
            for (s, c) in zip(start_indices, count_indices)
                push!(spatial_slices, (s+1):(s+c))  # Convert to 1-based
            end
            slices = tuple(spatial_slices...)
            size(data) == size(reconstructed_data[slices...]) || error(
                "Slab shape mismatch for '$var_name' in " *
                "$(basename(proc_info["file"]))")
            any(coverage_mask[slices...]) && error(
                "Overlapping slabs while reconstructing '$var_name': " *
                "$(basename(proc_info["file"])) overlaps prior coverage at $slices")
            reconstructed_data[slices...] = data
            coverage_mask[slices...] .= true
        end
    end

    uncovered_count = count(!, coverage_mask)
    uncovered_count == 0 || error(
        "Incomplete reconstruction of '$var_name': $uncovered_count of " *
        "$(length(coverage_mask)) points are uncovered")
    
    # Write reconstructed field
    write_reconstructed_field(reconstructed_data, var_attrs, output_file, var_name, dim_names, data_type)
    merger.verbose && println("        Grid space field merged")
end

"""
Merge field in coefficient space (spectral space).
Coefficient space fields are typically distributed across spectral modes.
"""
function merge_coeff_space_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    merger.verbose && println("        Merging coefficient space field")  
    var_attrs["merge_strategy"] = "coeff_space_spectral_reconstruction"
    
    # Coefficient space often requires different handling than grid space
    # Modes may be distributed differently than spatial points
    
    data_type = eltype(processor_data[1]["data"])
    
    # Try to use mode-based reconstruction
    reconstructed_data = reconstruct_spectral_modes(processor_data, data_type, merger)
    reconstructed_data === nothing && error(
        "Could not verify coefficient-space reconstruction of '$var_name'")
    write_reconstructed_field(reconstructed_data, var_attrs, output_file, var_name, dim_names, data_type)
    merger.verbose && println("        Coefficient space field merged")
end

"""
Merge field with mixed layout (some dimensions in grid space, others in coefficient space).
Following Tarang patterns, mixed layouts are transformed to pure layouts before merging.

Based on Tarang post and field - mixed layout fields cannot be directly merged
and must be transformed to either pure grid space or pure coefficient space first.
"""
function merge_mixed_layout_field!(processor_data, var_attrs, output_file, var_name, dim_names, merger)
    merger.verbose && println("        Merging mixed layout field")
    var_attrs["merge_strategy"] = "mixed_layout_transform_and_reconstruct"
    
    # Analyze grid_space flags to understand layout pattern
    grid_space_flags = nothing
    sample_layout_info = processor_data[1]["layout_info"]
    
    if sample_layout_info["grid_space"] !== nothing
        grid_space_flags = sample_layout_info["grid_space"]
        merger.verbose && println("          Grid space pattern: $grid_space_flags")
    else
        error("Cannot merge mixed-layout '$var_name' without grid_space metadata")
    end
    
    # Determine target layout based on predominant layout and data characteristics
    target_layout = determine_optimal_target_layout(grid_space_flags, processor_data, merger)
    merger.verbose && println("          Target layout: $target_layout")
    
    var_attrs["original_layout"] = string(grid_space_flags)
    var_attrs["target_layout"] = string(target_layout)
    
    # Transform processor data to target layout
    transformed_data = transform_to_target_layout!(processor_data, grid_space_flags, target_layout, merger)
    
    if transformed_data !== nothing
        # Update layout info to reflect pure target layout
        for proc_info in transformed_data
            if target_layout == :grid_space
                proc_info["layout_info"]["grid_space"] = trues(length(grid_space_flags))
            else  # :coeff_space
                proc_info["layout_info"]["grid_space"] = falses(length(grid_space_flags))
            end
        end
        
        # Apply appropriate pure layout merging strategy
        if target_layout == :grid_space
            merger.verbose && println("          Applying grid space merge to transformed data")
            merge_grid_space_field!(transformed_data, var_attrs, output_file, var_name, dim_names, merger)
        else  # :coeff_space
            merger.verbose && println("          Applying coefficient space merge to transformed data")
            merge_coeff_space_field!(transformed_data, var_attrs, output_file, var_name, dim_names, merger)
        end
        
        var_attrs["layout_transformation"] = "mixed_to_$(target_layout)"
        merger.verbose && println("        Mixed layout field transformed and merged")
    else
        error("Could not transform mixed-layout '$var_name' to $target_layout; " *
              "refusing to relabel the untransformed data")
    end
end

"""
Determine optimal target layout for mixed layout transformation.
Following Tarang patterns - prefer grid space for most cases unless
coefficient space is clearly more appropriate.
"""
function determine_optimal_target_layout(grid_space_flags, processor_data, merger)
    if !isa(grid_space_flags, AbstractArray)
        return :grid_space
    end
    
    flags = length(grid_space_flags) > 1 ? grid_space_flags[2:end] : grid_space_flags
    grid_dims = count(flags)
    coeff_dims = count(!, flags)
    
    merger.verbose && println("            Layout analysis: $grid_dims grid dims, $coeff_dims coeff dims")
    
    # Decision logic based on Tarang patterns:
    # 1. If majority are in grid space, transform to grid space
    # 2. If field is primarily spectral (more coeff dims), prefer coefficient space
    # 3. For tie cases, prefer grid space (Tarang default for output)
    
    if grid_dims >= coeff_dims
        return :grid_space
    else
        # More coefficient dimensions - check if this looks like a spectral field
        sample_data = processor_data[1]["data"]
        
        # Heuristic: if data is complex or has spectral characteristics, prefer coefficient
        if eltype(sample_data) <: Complex
            merger.verbose && println("            Complex data detected, preferring coefficient space")
            return :coeff_space
        else
            # For real data, still prefer grid space for easier interpretation
            return :grid_space
        end
    end
end

"""
Transform mixed layout processor data to target pure layout.
This implements the equivalent of Tarang layout transformation operations.
"""
function transform_to_target_layout!(processor_data, grid_space_flags, target_layout, merger)
    merger.verbose && println("            Transforming $(length(processor_data)) processor datasets")

    transformed_data = Any[]
    
    for proc_info in processor_data
        try
            original_data = proc_info["data"]
            layout_info = proc_info["layout_info"]
            
            # Apply layout transformation 
            transformed_field = apply_layout_transformation(original_data, grid_space_flags, target_layout, merger)
            
            if transformed_field !== nothing
                # Create new processor info with transformed data
                new_proc_info = Dict(
                    "data" => transformed_field,
                    "layout_info" => deepcopy(layout_info),
                    "file" => proc_info["file"]
                )
                push!(transformed_data, new_proc_info)
            else
                merger.verbose && println("              Failed to transform data from $(basename(proc_info["file"]))")
                return nothing
            end
            
        catch e
            merger.verbose && println("              Error transforming $(basename(proc_info["file"])): $e")
            return nothing
        end
    end
    
    merger.verbose && println("            Successfully transformed $(length(transformed_data)) datasets")
    return transformed_data
end

"""
Apply layout transformation to individual field data.

Transforms data between grid space and coefficient space layouts using FFT/IFFT.

Arguments:
- field_data: Array of field values (can be real or complex)
- grid_space_flags: Tuple/Vector of booleans indicating which dimensions are in grid space
                   (true = grid space, false = coefficient space)
- target_layout: :grid_space or :coeff_space
- merger: NetCDFMerger instance for configuration

Returns:
- Transformed array, or nothing if transformation fails
"""
function apply_layout_transformation(field_data, grid_space_flags, target_layout, merger)
    try
        # Validate inputs
        if field_data === nothing || isempty(field_data)
            return nothing
        end

        ndims_data = ndims(field_data)

        grid_space_flags = normalize_grid_space_flags(grid_space_flags, ndims_data)
        if grid_space_flags === nothing
            merger.verbose && println("              Warning: grid_space_flags missing, using default")
            grid_space_flags = fill(true, ndims_data)
        end

        # Check if transformation is needed
        flags = length(grid_space_flags) > 1 ? grid_space_flags[2:end] : grid_space_flags
        all_grid = all(flags)
        all_coeff = all(.!flags)

        if target_layout == :grid_space && all_grid
            # Already in grid space
            merger.verbose && println("              Data already in grid space")
            return field_data
        elseif target_layout == :coeff_space && all_coeff
            # Already in coefficient space
            merger.verbose && println("              Data already in coefficient space")
            return field_data
        end

        # Perform transformation
        if target_layout == :grid_space
            return transform_to_grid_space(field_data, grid_space_flags, merger)
        else  # target_layout == :coeff_space
            return transform_to_coeff_space(field_data, grid_space_flags, merger)
        end

    catch e
        merger.verbose && println("              Layout transformation failed: $e")
        @debug "Layout transformation error" exception=(e, catch_backtrace())
        return nothing
    end
end

"""Transform field data to grid space by applying inverse FFT to coefficient dimensions."""
function transform_to_grid_space(field_data, grid_space_flags, merger)
    result = copy(field_data)
    input_real = eltype(field_data) <: Real
    ndims_data = ndims(result)

    # Convert to complex if needed for FFT operations
    if eltype(result) <: Real
        result = complex(result)
    end

    transforms_applied = 0

    # Apply inverse FFT to each dimension that is in coefficient space
    for dim in 1:ndims_data
        if dim == 1
            continue  # Skip time dimension
        end
        if !grid_space_flags[dim]
            # This dimension is in coefficient space - apply inverse FFT
            try
                # Use FFTW for the inverse transform along this dimension
                result = apply_ifft_along_dim(result, dim)
                transforms_applied += 1
            catch e
                merger.verbose && println("              IFFT failed for dimension $dim: $e")
                # Continue with other dimensions
            end
        end
    end

    merger.verbose && println("              Applied $transforms_applied inverse transforms to grid space")

    # Return real part if the result should be real-valued
    # (for physical fields, the imaginary part should be negligible)
    if input_real && !isempty(result)
        # Check if imaginary part is negligible
        max_imag = maximum(abs.(imag(result)))
        max_real = maximum(abs.(real(result)))
        if max_real > 0 && max_imag / max_real < 1e-10
            return real(result)
        end
    end

    return result
end

"""Transform field data to coefficient space by applying forward FFT to grid dimensions."""
function transform_to_coeff_space(field_data, grid_space_flags, merger)
    result = copy(field_data)
    ndims_data = ndims(result)

    # Convert to complex for FFT operations
    if eltype(result) <: Real
        result = complex(result)
    end

    transforms_applied = 0

    # Apply forward FFT to each dimension that is in grid space
    for dim in 1:ndims_data
        if dim == 1
            continue  # Skip time dimension
        end
        if grid_space_flags[dim]
            # This dimension is in grid space - apply forward FFT
            try
                result = apply_fft_along_dim(result, dim)
                transforms_applied += 1
            catch e
                merger.verbose && println("              FFT failed for dimension $dim: $e")
                # Continue with other dimensions
            end
        end
    end

    merger.verbose && println("              Applied $transforms_applied forward transforms to coefficient space")

    return result
end

"""
Apply forward FFT along a specific dimension.
Uses normalized FFT (1/N factor applied).
"""
function apply_fft_along_dim(data::AbstractArray{T}, dim::Int) where T <: Complex
    n = size(data, dim)

    # Create FFT plan for this dimension
    # We use fft with the dims keyword to transform along a specific axis
    result = fft(data, dim)

    # Normalize by 1/N for proper spectral coefficients
    result ./= n

    return result
end

"""Apply FFT to real data along a specific dimension."""
function apply_fft_along_dim(data::AbstractArray{T}, dim::Int) where T <: Real
    return apply_fft_along_dim(complex(data), dim)
end

"""
Apply inverse FFT along a specific dimension.
Uses unnormalized IFFT (multiply by N to invert the forward normalization).
"""
function apply_ifft_along_dim(data::AbstractArray{T}, dim::Int) where T <: Complex
    n = size(data, dim)

    # Apply inverse FFT along specified dimension
    result = ifft(data, dim)

    # IFFT in Julia is already normalized by 1/N, but we used 1/N in forward FFT
    # So we need to multiply by N to get back the original values
    result .*= n

    return result
end

"""Apply IFFT to real data along a specific dimension."""
function apply_ifft_along_dim(data::AbstractArray{T}, dim::Int) where T <: Real
    return apply_ifft_along_dim(complex(data), dim)
end

"""
Attempt to detect whether data is in grid space or coefficient space
based on data characteristics.

Heuristics:
1. Complex data with significant imaginary parts likely in coefficient space
2. Data with values concentrated near zero indices likely in coefficient space
3. Smooth real data likely in grid space

Returns a tuple of booleans (grid_space_flags) for each dimension.
"""
function detect_layout_from_data(field_data, field_name::String="")
    ndims_data = ndims(field_data)

    # Default: assume grid space
    grid_space_flags = fill(true, ndims_data)

    # Check if data is complex
    if eltype(field_data) <: Complex && !isempty(field_data)
        # Check imaginary content
        total_mag = sum(abs.(field_data))
        imag_mag = sum(abs.(imag(field_data)))

        if total_mag > 0 && imag_mag / total_mag > 0.01
            # Significant imaginary content - likely coefficient space
            # For Fourier dimensions, mark as coefficient space
            for dim in 1:ndims_data
                # Check if energy is concentrated at low wavenumbers
                if is_spectral_dimension(field_data, dim)
                    grid_space_flags[dim] = false
                end
            end
        end
    end

    return tuple(grid_space_flags...)
end

"""
Check if a dimension appears to be in spectral (coefficient) space
by examining the energy distribution.

In coefficient space, energy is typically concentrated at low wavenumbers.
"""
function is_spectral_dimension(data::AbstractArray, dim::Int)
    n = size(data, dim)
    if n < 4
        return false  # Too small to determine
    end

    # Sum absolute values along this dimension
    # Move the target dimension to first position for easier slicing
    perm = collect(1:ndims(data))
    perm[1], perm[dim] = perm[dim], perm[1]
    permuted = permutedims(data, perm)

    # Compute energy in low vs high wavenumber regions
    quarter_n = max(1, n ÷ 4)

    # Low wavenumber region (first and last quarter for symmetric spectra)
    low_k_energy = sum(abs.(selectdim(permuted, 1, 1:quarter_n))) +
                   sum(abs.(selectdim(permuted, 1, (n - quarter_n + 1):n)))

    # High wavenumber region (middle half)
    mid_start = quarter_n + 1
    mid_end = n - quarter_n
    if mid_end >= mid_start
        high_k_energy = sum(abs.(selectdim(permuted, 1, mid_start:mid_end)))
    else
        high_k_energy = 0.0
    end

    total_energy = low_k_energy + high_k_energy

    if total_energy == 0
        return false
    end

    # If more than 80% of energy is in low wavenumbers, likely spectral
    return low_k_energy / total_energy > 0.8
end

"""Convert grid_space_flags to a readable string."""
function get_layout_string(grid_space_flags)
    parts = [flag ? "G" : "C" for flag in grid_space_flags]
    return join(parts, "-")
end

"""Parse layout string like 'G-C-G' to grid_space_flags tuple."""
function parse_layout_string(layout_str::String)
    parts = split(layout_str, "-")
    return tuple([uppercase(strip(p)) == "G" for p in parts]...)
end

"""
Infer start/count indices by assuming a slab decomposition along the last spatial dimension.
Returns inferred global shape, or nothing if inference is not possible.
"""
function infer_slab_decomposition!(processor_data, merger)
    if isempty(processor_data)
        return nothing
    end

    sample = processor_data[1]["data"]
    ndims_data = ndims(sample)
    if ndims_data < 2
        return nothing
    end

    spatial_ndims = ndims_data - 1
    shapes = [size(proc["data"])[2:end] for proc in processor_data]
    base = shapes[1]

    if spatial_ndims > 1
        for dim in 1:(spatial_ndims - 1)
            if any(shape[dim] != base[dim] for shape in shapes)
                merger.verbose && println("        Cannot infer slab decomposition: spatial dims vary beyond last axis")
                return nothing
            end
        end
    end

    offset = 0
    for (proc, shape) in zip(processor_data, shapes)
        start = zeros(Int, spatial_ndims)
        count = collect(shape)
        start[end] = offset
        proc["start"] = tuple(start...)
        proc["count"] = tuple(count...)
        if haskey(proc, "layout_info")
            proc["layout_info"]["start"] = tuple(start...)
            proc["layout_info"]["count"] = tuple(count...)
        end
        offset += shape[end]
    end

    prefix = spatial_ndims > 1 ? base[1:end-1] : ()
    return (size(sample, 1), prefix..., offset)
end

"""Estimate global shape from domain decomposition info."""
function estimate_global_shape_from_decomposition(processor_data)
    sample_data = processor_data[1]["data"]
    sample_shape = collect(size(sample_data))
    
    # Try to find maximum extents from start+count info
    for proc_info in processor_data
        layout_info = proc_info["layout_info"]
        start_indices = layout_info["start"]
        count_indices = layout_info["count"]
        
        if start_indices !== nothing && count_indices !== nothing
            for (i, (s, c)) in enumerate(zip(start_indices, count_indices))
                dim_index = i + 1  # Skip time dimension
                if dim_index <= length(sample_shape)
                    sample_shape[dim_index] = max(sample_shape[dim_index], s + c)
                end
            end
        end
    end
    
    return tuple(sample_shape...)
end

"""
Reconstruct spectral coefficient field from distributed modes.

Based on Tarang distributor Layout class and post merge_data function.
Handles block distribution of spectral coefficients across processors.
"""
function reconstruct_spectral_modes(processor_data, data_type, merger)
    merger.verbose && println("          Reconstructing spectral coefficient field")

    if isempty(processor_data)
        merger.verbose && println("            No processor data available")
        return nothing
    end
    
    # Extract metadata from first processor to determine global structure
    first_proc_info = processor_data[1]
    first_layout_info = first_proc_info["layout_info"]
    
    # Check if this is actually a coefficient space field
    grid_space_flags = first_layout_info["grid_space"]
    if grid_space_flags === nothing
        merger.verbose && println("            No grid_space flags found")
        return nothing
    end
    
    # For pure coefficient space, all grid_space flags should be false
    if isa(grid_space_flags, AbstractArray)
        flags = length(grid_space_flags) > 1 ? grid_space_flags[2:end] : grid_space_flags
        if any(flags)
            merger.verbose && println("            Mixed or grid space field, not pure coefficient")
            return nothing
        end
    elseif grid_space_flags != false
        merger.verbose && println("            Not coefficient space field")
        return nothing
    end
    
    # Get global shape from processor metadata
    global_shape = first_layout_info["global_shape"]
    if global_shape === nothing
        merger.verbose && println("            No global shape information available")
        return nothing
    end
    
    merger.verbose && println("            Global coefficient array shape: $global_shape")
    
    # Initialize global coefficient array
    global_coeffs = zeros(data_type, global_shape...)
    filled_mask = zeros(Bool, global_shape...)
    
    merger.verbose && println("            Processing $(length(processor_data)) processor datasets")
    
    # Reconstruct by placing each processor's data at correct indices
    processors_placed = 0
    for proc_info in processor_data
        layout_info = proc_info["layout_info"]
        start_indices = layout_info["start"]
        count_sizes = layout_info["count"]
        local_data = proc_info["data"]
        proc_file = basename(proc_info["file"])
        
        if start_indices === nothing || count_sizes === nothing
            error("Missing start/count metadata for coefficient slab '$proc_file'")
        end
        
        try
            # Convert to Julia 1-based indexing and create slices
            # Skip time dimension (index 1), work on spatial/spectral dimensions
            spatial_slices = Any[Colon()]  # Time dimension - take all
            
            for (start_idx, count_size) in zip(start_indices, count_sizes)
                # Convert from 0-based to 1-based indexing
                julia_start = start_idx + 1
                julia_end = start_idx + count_size
                push!(spatial_slices, julia_start:julia_end)
            end
            
            slices = tuple(spatial_slices...)
            
            # Verify dimensions match
            expected_size = size(global_coeffs[slices...])
            actual_size = size(local_data)
            
            if expected_size != actual_size
                error("Coefficient slab '$proc_file' has shape $actual_size; " *
                      "expected $expected_size")
            end
            
            # Check for overlap (should not happen with proper distribution)
            if any(filled_mask[slices...])
                error("Overlapping coefficient slab '$proc_file' at $slices")
            end
            
            # Place data and mark as filled
            global_coeffs[slices...] = local_data
            filled_mask[slices...] .= true
            processors_placed += 1
            
            merger.verbose && println("              Placed coefficients from $proc_file at $slices")
            
        catch e
            error("Could not place coefficient slab '$proc_file': " *
                  sprint(showerror, e))
        end
    end
    
    # Verify complete reconstruction
    uncovered_points = count(!, filled_mask)
    total_points = length(filled_mask)
    coverage_fraction = count(filled_mask) / total_points
    
    merger.verbose && println("            Reconstruction coverage: $(processors_placed)/$(length(processor_data)) processors")
    merger.verbose && println("            Coverage fraction: $(round(coverage_fraction * 100, digits=1))% ($uncovered_points uncovered points)")
    
    uncovered_points == 0 || error(
        "Incomplete coefficient reconstruction: $uncovered_points of $total_points " *
        "points are uncovered")
    
    # Verify spectral field characteristics
    if validate_spectral_coefficients(global_coeffs, merger)
        merger.verbose && println("            Successfully reconstructed spectral coefficient field")
        return global_coeffs
    else
        merger.verbose && println("            Reconstructed data failed spectral validation")
        return nothing
    end
end

"""
Validate reconstructed spectral coefficients for basic sanity checks.
Based on typical spectral field characteristics.
"""
function validate_spectral_coefficients(coeffs, merger)
    try
        # Basic checks for spectral coefficient arrays
        
        # 1. Check for reasonable coefficient magnitudes
        max_coeff = maximum(abs.(coeffs))
        if max_coeff == 0.0
            # All-zero is valid (e.g., initial conditions, perturbation fields)
            merger.verbose && println("              Note: All coefficients are zero (valid)")
            return true
        end
        
        # 2. Check for numerical stability (no infinities or NaNs)
        if !all(isfinite.(coeffs))
            merger.verbose && println("              Error: Non-finite coefficients detected")
            return false
        end
        
        # 3. For complex coefficients, check Hermitian symmetry where applicable
        if eltype(coeffs) <: Complex
            merger.verbose && println("              Complex spectral coefficients detected")
            # Could add Hermitian symmetry checks for Fourier coefficients here
        end
        
        # 4. Check for reasonable dynamic range
        nonzero_coeffs = coeffs[abs.(coeffs) .> 1e-12 * max_coeff]
        if length(nonzero_coeffs) == 0
            merger.verbose && println("              Warning: No significant coefficients found")
            return false
        end
        
        dynamic_range = log10(max_coeff / minimum(abs.(nonzero_coeffs)))
        if dynamic_range > 15  # More than 15 orders of magnitude might indicate numerical issues
            merger.verbose && println("              Warning: Very large dynamic range ($dynamic_range orders)")
        end
        
        merger.verbose && println("              Spectral validation passed ($(length(nonzero_coeffs)) significant modes)")
        return true
        
    catch e
        merger.verbose && println("              Spectral validation failed: $e")
        return false
    end
end

"""Write reconstructed field data to NetCDF file."""
function write_reconstructed_field(data, var_attrs, output_file, var_name, dim_names, data_type)
    dim_sizes = [size(data)...]

    # Build alternating dim_name, dim_size pairs for nccreate
    dim_args = Any[]
    for (dn, ds) in zip(dim_names, dim_sizes)
        push!(dim_args, dn)
        push!(dim_args, ds)
    end
    group_nccreate(output_file, NETCDF_VARS_GROUP, var_name, dim_args..., t=data_type, atts=var_attrs)

    group_ncwrite(data, output_file, NETCDF_VARS_GROUP, var_name)
end
