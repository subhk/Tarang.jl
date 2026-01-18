"""
Plotting and visualization tools

Basic plotting utilities for Tarang fields
"""

# Basic structure for plotting - actual implementation would use Plots.jl or similar

struct PlotData
    x::Vector{Float64}
    y::Union{Vector{Float64}, Nothing}
    z::Vector{Float64}
    data::Array{Float64}
    labels::Dict{String, String}
    title::String
end

function extract_plot_data(field::ScalarField; layout::Symbol=:g)
    """Extract data for plotting from scalar field"""
    
    ensure_layout!(field, layout)
    
    if field.domain === nothing
        throw(ArgumentError("Field must have a domain for plotting"))
    end
    
    # Get coordinate grids (CPU arrays for plotting)
    grids = local_grids(field.dist, field.bases...; move_to_arch=false)

    if length(grids) == 1
        # 1D plot
        x = collect(grids[1])
        data = Array(field[string(layout)])
        
        return PlotData(x, nothing, Float64[], data, 
                       Dict("x" => field.bases[1].meta.element_label),
                       field.name)
        
    elseif length(grids) == 2
        # 2D plot
        x = collect(grids[1])
        y = collect(grids[2])
        data = Array(field[string(layout)])

        return PlotData(x, y, Float64[], data,
                       Dict("x" => field.bases[1].meta.element_label,
                            "y" => field.bases[2].meta.element_label),
                       field.name)
    elseif length(grids) == 3
        # 3D field - extract middle z-slice for 2D visualization
        x = collect(grids[1])
        y = collect(grids[2])
        z = collect(grids[3])
        full_data = Array(field[string(layout)])

        # Take middle slice along z-axis
        z_mid = div(length(z), 2) + 1
        data = full_data[:, :, z_mid]

        return PlotData(x, y, z, data,
                       Dict("x" => field.bases[1].meta.element_label,
                            "y" => field.bases[2].meta.element_label,
                            "z" => field.bases[3].meta.element_label,
                            "slice" => "z = $(z[z_mid])"),
                       "$(field.name) (z-slice)")
    else
        throw(ArgumentError("Plotting supports 1D, 2D, and 3D fields only"))
    end
end

function extract_plot_data(field::VectorField; component::Int=1, layout::Symbol=:g)
    """Extract data for plotting from vector field component"""
    return extract_plot_data(field.components[component], layout=layout)
end

# Plotting functions - return PlotData for integration with plotting backends (Plots.jl, Makie.jl, etc.)

function plot_1d(field::ScalarField; layout::Symbol=:g, kwargs...)
    """Create 1D line plot"""
    
    plot_data = extract_plot_data(field, layout=layout)
    
    if plot_data.y !== nothing
        throw(ArgumentError("Use plot_2d for 2D fields"))
    end
    
    @info "Plotting 1D field '$(plot_data.title)'"
    @info "  X range: $(minimum(plot_data.x)) to $(maximum(plot_data.x))"
    @info "  Data range: $(minimum(plot_data.data)) to $(maximum(plot_data.data))"
    
    # In actual implementation, would create plot using plotting backend
    return plot_data
end

function plot_2d(field::ScalarField; layout::Symbol=:g, contour::Bool=false, kwargs...)
    """Create 2D contour or heatmap plot"""
    
    plot_data = extract_plot_data(field, layout=layout)
    
    if plot_data.y === nothing
        throw(ArgumentError("Use plot_1d for 1D fields"))
    end
    
    @info "Plotting 2D field '$(plot_data.title)'"
    @info "  X range: $(minimum(plot_data.x)) to $(maximum(plot_data.x))"
    @info "  Y range: $(minimum(plot_data.y)) to $(maximum(plot_data.y))"
    @info "  Data range: $(minimum(plot_data.data)) to $(maximum(plot_data.data))"
    
    # In actual implementation, would create contour or heatmap
    return plot_data
end

function plot_vector_field(field::VectorField; layout::Symbol=:g, subsample::Int=1, kwargs...)
    """Create vector field plot (quiver plot)"""
    
    if field.coordsys.dim != 2
        throw(ArgumentError("Vector field plotting only implemented for 2D"))
    end
    
    # Extract data for both components
    u_data = extract_plot_data(field.components[1], layout=layout)
    v_data = extract_plot_data(field.components[2], layout=layout)
    
    @info "Plotting vector field '$(field.name)'"
    @info "  Grid size: $(size(u_data.data))"
    @info "  U range: $(minimum(u_data.data)) to $(maximum(u_data.data))"
    @info "  V range: $(minimum(v_data.data)) to $(maximum(v_data.data))"
    
    # In actual implementation, would create quiver plot
    return (u=u_data, v=v_data)
end

function plot_streamlines(field::VectorField; layout::Symbol=:g, n_lines::Int=10, kwargs...)
    """Create streamline plot for 2D vector field"""
    
    if field.coordsys.dim != 2
        throw(ArgumentError("Streamlines only implemented for 2D vector fields"))
    end
    
    # Calculate streamfunction
    psi = streamfunction(field)
    
    plot_data = extract_plot_data(psi, layout=layout)
    
    @info "Plotting streamlines for '$(field.name)'"
    @info "  Number of streamlines: $n_lines"
    
    # In actual implementation, would create streamline plot
    return plot_data
end

# Animation utilities
mutable struct Animation
    frames::Vector{PlotData}
    filename::String
    fps::Float64
    
    function Animation(filename::String="animation"; fps::Float64=10.0)
        new(PlotData[], filename, fps)
    end
end

function add_frame!(anim::Animation, field::Union{ScalarField, VectorField}; kwargs...)
    """Add frame to animation"""
    
    if isa(field, ScalarField)
        plot_data = extract_plot_data(field; kwargs...)
    else
        # For vector fields, plot magnitude
        mag_field = vector_magnitude(field)
        plot_data = extract_plot_data(mag_field; kwargs...)
    end
    
    push!(anim.frames, plot_data)
    
    @info "Added frame $(length(anim.frames)) to animation"
end

function save_animation(anim::Animation)
    """Save animation to file"""
    
    if isempty(anim.frames)
        throw(ArgumentError("No frames in animation"))
    end
    
    @info "Saving animation with $(length(anim.frames)) frames to '$(anim.filename)'"
    @info "  Frame rate: $(anim.fps) fps"
    @info "  Duration: $(length(anim.frames) / anim.fps) seconds"
    
    # In actual implementation, would save animation file
    return anim.filename
end

# Utility functions
function vector_magnitude(field::VectorField; layout::Symbol=:g)
    """Calculate magnitude of vector field"""
    
    mag_field = ScalarField(field.dist, "$(field.name)_magnitude", field.bases, field.dtype)
    ensure_layout!(mag_field, layout)
    
    # Calculate |u|² 
    magnitude_squared = zeros(size(field.components[1][string(layout)]))
    
    for component in field.components
        ensure_layout!(component, layout)
        magnitude_squared .+= abs2.(component[string(layout)])
    end
    
    mag_field[string(layout)] .= sqrt.(magnitude_squared)
    
    return mag_field
end

function save_field_data(field::Union{ScalarField, VectorField}, filename::String; format::String="h5")
    """Save field data to file for external plotting"""
    
    if format == "h5"
        save_field_hdf5(field, filename)
    elseif format == "csv"
        save_field_csv(field, filename)
    else
        throw(ArgumentError("Unsupported format: $format"))
    end
end

function save_field_hdf5(field::Union{ScalarField, VectorField}, filename::String)
    """Save field to HDF5 format"""
    
    if isa(field, ScalarField)
        save_field(field, filename)
    else
        # Save vector components
        for (i, component) in enumerate(field.components)
            component_name = "component_$i"
            save_field(component, filename, component_name)
        end
    end
    
    @info "Saved field data to HDF5 file: $filename"
end

function save_field_csv(field::ScalarField, filename::String)
    """Save scalar field to CSV format"""
    
    ensure_layout!(field, :g)
    
    # Get coordinate grids
    if field.domain === nothing
        throw(ArgumentError("Field must have domain for CSV export"))
    end
    
    grids = local_grids(field.dist, field.bases...; move_to_arch=false)  # CPU arrays for file I/O
    data = Array(field["g"])
    
    # Write CSV file
    open(filename, "w") do io
        if length(grids) == 1
            # 1D data
            println(io, "x,$(field.name)")
            for (i, x) in enumerate(grids[1])
                println(io, "$x,$(data[i])")
            end
        elseif length(grids) == 2
            # 2D data
            println(io, "x,y,$(field.name)")
            for (j, y) in enumerate(grids[2]), (i, x) in enumerate(grids[1])
                println(io, "$x,$y,$(data[i,j])")
            end
        end
    end
    
    @info "Saved field data to CSV file: $filename"
end

# Plot styling and configuration
struct PlotStyle
    colormap::String
    line_width::Float64
    font_size::Int
    figure_size::Tuple{Int, Int}
    dpi::Int
    
    function PlotStyle(; colormap::String="viridis", line_width::Float64=1.0,
                      font_size::Int=12, figure_size::Tuple{Int,Int}=(800,600), dpi::Int=100)
        new(colormap, line_width, font_size, figure_size, dpi)
    end
end

# Global plot style (mutable Ref to allow updates)
const DEFAULT_STYLE = Ref{PlotStyle}(PlotStyle())

function get_plot_style()
    """Get current global plotting style"""
    return DEFAULT_STYLE[]
end

function set_plot_style!(style::PlotStyle)
    """Set global plotting style"""
    DEFAULT_STYLE[] = style
    @info "Updated plotting style"
end

function create_subplot_layout(fields::Vector{Union{ScalarField, VectorField}}, 
                              rows::Int, cols::Int)
    """Create subplot layout for multiple fields"""
    
    if length(fields) > rows * cols
        throw(ArgumentError("Too many fields for $(rows)×$(cols) subplot layout"))
    end
    
    subplot_data = []
    
    for (i, field) in enumerate(fields)
        if isa(field, ScalarField)
            data = extract_plot_data(field)
        else
            # For vector fields, plot magnitude
            mag_field = vector_magnitude(field)
            data = extract_plot_data(mag_field)
        end
        push!(subplot_data, data)
    end
    
    @info "Created $(rows)×$(cols) subplot layout for $(length(fields)) fields"

    return subplot_data
end

# ============================================================================
# Exports
# ============================================================================

# Data structures
export PlotData, PlotStyle, Animation

# Plot data extraction
export extract_plot_data

# Plotting functions
export plot_1d, plot_2d, plot_vector_field, plot_streamlines

# Animation functions
export add_frame!, save_animation

# Utility functions
export vector_magnitude, save_field_data
export save_field_hdf5, save_field_csv

# Style configuration
export DEFAULT_STYLE, get_plot_style, set_plot_style!

# Subplot utilities
export create_subplot_layout