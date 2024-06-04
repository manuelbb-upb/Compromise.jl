function optimize_with_algo(
    MOP::AbstractMOP, outer_opts::ThreadedOuterAlgorithmOptions, ξ0::RVecOrMat; 
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
)
    _ξ0 = ξ0 isa RVec ? reshape(ξ0, :, 1) : ξ0

    algo_opts = outer_opts.inner_opts
    log_level = algo_opts.log_level

    ret_lock = ReentrantLock()
    ret = Any[]

    @logmsg log_level "Starting threaded evaluation of columns. (No Logging)."
    ncols = size(_ξ0, 2)
    Threads.@threads for i=1:ncols
        ξ0_i = lock(ret_lock) do
            @logmsg log_level "\tInspecting column $(i)."
            _ξ0[:, i]
        end
        r = Logging.with_logger(Logging.NullLogger()) do
            optimize_with_algo(MOP, algo_opts, ξ0_i; user_callback)
        end
        lock(ret_lock) do 
            push!(ret, r)
        end
    end

    return ret
end

function optimize_with_algo(
    MOP::AbstractMOP, outer_opts::SequentialOuterAlgorithmOptions, ξ0::RVecOrMat; 
    user_callback :: AbstractStoppingCriterion = NoUserCallback(),
)
    _ξ0 = ξ0 isa RVec ? reshape(ξ0, :, 1) : ξ0

    algo_opts = outer_opts.inner_opts
    log_level = outer_opts.log_level

    num_sites = size(_ξ0, 2)
    all_optimizer_caches = Any[]

    @logmsg log_level "Initializing all $num_sites cache objects."
    algo_opts = outer_opts.inner_opts
    for ξ0 in eachcol(_ξ0)
        optimizer_caches = initialize_structs(MOP, ξ0, algo_opts, user_callback)
        push!(all_optimizer_caches, optimizer_caches)
    end
        
    @logmsg log_level "Starting sequential optimization of columns."
    flags_dominated = zeros(Bool, length(all_optimizer_caches))
    if outer_opts.initial_nondominance_testing
        @logmsg log_level "NONDOMINANCE TESTING"
        flags_dominated = _flags_of_dominated_caches(flags_dominated, all_optimizer_caches)
    end
    is_running = .!(flags_dominated)

    all_return_objects = Vector{Any}(undef, num_sites)

    for (i, ocache) in enumerate(all_optimizer_caches)
        flags_dominated[i] && continue
        if ocache isa AbstractStoppingCriterion
            is_running[i] = false
            ret = ReturnObject(copy(_ξ0[:, i]), nothing, ocache)
            all_return_objects[i] = ret
        end
    end

    delta_func = opt_cache -> opt_cache.iteration_scalars.delta
    crit_func = opt_cache -> opt_cache.step_vals.crit_ref[]
    I = collect(1:length(all_optimizer_caches))
    
    outer_it_index = 0
    nd_offset = outer_opts.nondominance_testing_offset
    is_nd_tested = true
    while any(is_running)
        outer_it_index += 1
        is_nd_tested = false
        
        if outer_opts.sort_by_delta
            sortperm!(I, all_optimizer_caches; by = delta_func, rev = true)
        end
        
        Δ_largest = mapreduce(delta_func, max, @view(all_optimizer_caches[is_running]); init=0)
        crit_largest = -Inf
        for i in I
            !is_running[i] && continue
            opt_cache_i = all_optimizer_caches[i]
            opt_cache_i.iteration_scalars.it_index < 1 && continue
            χ = crit_func(opt_cache_i)
            if χ > crit_largest
                crit_largest = χ
            end
        end
        
        counter = 0
        for i in I
            if flags_dominated[i]
                is_running[i] = false
            end
        
            !is_running[i] && continue
            opt_cache_i = all_optimizer_caches[i]
            if (
                #counter > dim_objectives(MOP) && 
                delta_func(opt_cache_i) <= Δ_largest * outer_opts.delta_factor
                #!isinf(crit_largest) &&
                #false
                #crit_func(opt_cache_i) < 0.05 * crit_largest
            )
                continue
            end
            counter += 1
            ξ0_i = @view(_ξ0[:, i])
            @logmsg log_level """\n
            #!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
            # COLUMN $(i).
            #!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!"""
            stop_code = do_inner_iteration!(opt_cache_i, algo_opts)
            if stop_code isa AbstractStoppingCriterion
                is_running[i] = false

                @logmsg log_level """\n
                    #!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
                    # FINISHED COLUMN $(i):
                    # $(stop_message(stop_code))
                    #!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!"""
            
                ret = ReturnObject(ξ0_i, all_optimizer_caches[i], stop_code)
                all_return_objects[i] = ret
            end 
        end
        if outer_it_index % nd_offset == 0
            @logmsg log_level "NONDOMINANCE TESTING"
            flags_dominated = _flags_of_dominated_caches(flags_dominated, all_optimizer_caches)
            for i = eachindex(all_optimizer_caches)
                if flags_dominated[i]
                    is_running[i] = false
                end
            end
                is_nd_tested = true
        end
    end
    if outer_opts.final_nondominance_testing && !is_nd_tested
        @logmsg log_level "NONDOMINANCE TESTING"
        flags_dominated = _flags_of_dominated_caches(flags_dominated, all_optimizer_caches)        
        for i = eachindex(all_optimizer_caches)
            if flags_dominated[i]
                is_running[i] = false
            end
        end
    end

    deleteat!(all_return_objects, flags_dominated)

    return all_return_objects
end

function _flags_of_dominated_caches(caches)
    flags_dominated = zeros(Bool, length(caches))
    return _flags_of_dominated_caches(flags_dominated, caches)
end

function _flags_of_dominated_caches(flags_dominated, caches)
    for (i, opt_cache_i) in enumerate(caches)
        flags_dominated[i] && continue
        # `opt_cache_i` is currently not flagged as dominated
        vals_i = opt_cache_i.vals
        fi = cached_fx(vals_i)
        # check if it is dominated by any other cache
        for (j, opt_cache_j) in enumerate(caches)
            flags_dominated[i] && break
            i == j && continue
            flags_dominated[j] && continue  # if j is dominated by l, and i is dominated by j, then i is dominated by l as well
                                            # l ≤ j & j ≤ i ⟹ l ≤ i
            vals_j = opt_cache_j.vals
            fj = cached_fx(vals_j)
            
            flags_dominated[i] = is_dominated_by(fi, fj)
        end
    end
    return flags_dominated
end

"""
   is_dominated_by(a, b)
Return `true`, if `b .<= a` and `b[k] < a[k]` for some index `k`.
"""
function is_dominated_by(a, b)
    is_dom = false 
    for (ai, bi) in zip(a, b)
        if bi > ai
            is_dom = false
            break
        end
        if bi < ai
            is_dom = true
        end
    end
    return is_dom
end

function do_inner_iteration!(optimizer_caches, algo_opts)
    @unpack (
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
    ) = optimizer_caches
    return do_iteration!(
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    )        
end