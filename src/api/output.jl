"""
    Tarang.Output

Facade for output handlers and NetCDF merge utilities.
"""
module Output
import ..Tarang:
    NetCDFFileHandler, NetCDFEvaluator, UnifiedEvaluator, add_netcdf_handler,
    DictionaryHandler, VirtualFileHandler,
    add_dictionary_handler, add_virtual_file_handler, merge_virtual!,
    NetCDFMerger, merge_netcdf_files, batch_merge_netcdf, find_mergeable_handlers,
    MergeMode, SIMPLE_CONCAT, RECONSTRUCT, DOMAIN_DECOMP

export
    NetCDFFileHandler, NetCDFEvaluator, UnifiedEvaluator, add_netcdf_handler,
    DictionaryHandler, VirtualFileHandler,
    add_dictionary_handler, add_virtual_file_handler, merge_virtual!,
    NetCDFMerger, merge_netcdf_files, batch_merge_netcdf, find_mergeable_handlers,
    MergeMode, SIMPLE_CONCAT, RECONSTRUCT, DOMAIN_DECOMP
end
