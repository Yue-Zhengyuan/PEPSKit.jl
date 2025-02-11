function show_memory_info(; show_cache::Bool=true, run_gc::Bool=true)
    @info "memory usage (GB)" => Sys.maxrss() / 1024^3
    if show_cache
        @info "treepermutercache" => cache_info(TensorKit.treepermutercache)
        @info "treetransposercache" => cache_info(TensorKit.treetransposercache)
        @info "treebraidercache" => cache_info(TensorKit.treebraidercache)
        @info "GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE" =>
            cache_info(TensorKit.GLOBAL_FUSIONBLOCKSTRUCTURE_CACHE)
    end
    run_gc && GC.gc()
    return nothing
end
