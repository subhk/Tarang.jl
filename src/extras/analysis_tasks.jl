"""
Analysis task helpers for Tarang.

These helpers wrap common reductions/slices used in analysis tasks so they
can be scheduled via NetCDF handlers.
"""

"""
Add a spatial mean task. `dims` follows Julia `mean` semantics.
"""
function add_mean_task!(handler, task; dims=:, name=nothing, layout="g", scales=nothing)
    post = data -> dropdims(mean(data; dims=dims), dims=dims)
    return add_task!(handler, task; name=name, layout=layout, scales=scales, postprocess=post)
end

"""
Add a slice task along dimension `dim` at index `idx`.
"""
function add_slice_task!(handler, task; dim::Int=1, idx::Int=1, name=nothing, layout="g", scales=nothing)
    post = data -> data |> x -> @views x[ntuple(i -> i == dim ? idx : :, ndims(x))...]
    return add_task!(handler, task; name=name, layout=layout, scales=scales, postprocess=post)
end

"""
Add a profile task: mean over all dims except `dim`, retaining that dimension.
"""
function add_profile_task!(handler, task; dim::Int=1, name=nothing, layout="g", scales=nothing)
    post = data -> begin
        dims_to_mean = setdiff(collect(1:ndims(data)), [dim])
        dropdims(mean(data; dims=dims_to_mean), dims=dims_to_mean)
    end
    return add_task!(handler, task; name=name, layout=layout, scales=scales, postprocess=post)
end
