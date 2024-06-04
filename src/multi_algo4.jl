function optimize_many(
    X::RMat, 
    mop;
    algo_opts = AlgorithmOptions(), 
    user_callback = NoUserCallback(),
    filter_gamma = 1e-6,
)
    @unpack log_level, max_iter = algo_opts

    optimizer_caches = initialize_structs(mop, @view(X[:, 1]), algo_opts, user_callback; log_time=false)

    if optimizer_caches isa AbstractStoppingCriterion
        return optimizer_caches
    end
    
    @unpack (
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars,
    ) = optimizer_caches

    @unpack stop_delta_min = algo_opts
    stop_crits = MinimumRadiusStopping(; delta_min = stop_delta_min)
    @reset optimizer_caches.stop_crits = stop_crits

    F = float_type(mop)
    prepopulation = initialize_prepopulation(F, X, mop, scaler, lin_cons, vals; log_level)   
    sol, population = initialize_population(
        prepopulation, mod_vals, step_vals, step_cache, iteration_scalars, mop, mod, stop_crits
    )

    ndset = initialize_filter(F, sol, filter_gamma)

    restoration_seeds = AugmentedVectorFilter(;
        elems = empty(population.elems),
        gamma = population.gamma,
        tmp = nothing #deepcopy(population.tmp)
    )

    optimize_population!(population, ndset, restoration_seeds, optimizer_caches, algo_opts)
    return population
end

function reset_restoration_seeds!(restoration_seeds)
    empty!(restoration_seeds.elems)
    empty!(restoration_seeds.is_stale)
    restoration_seeds.min_scalar_val[] = Inf
    empty!(restoration_seeds.min_vector_vals)
    restoration_seeds.counter[] = 0
    nothing
end

function optimize_population!(
    population, ndset, restoration_seeds, optimizer_caches, algo_opts;
    check_filter=true
)
    @unpack max_iter = algo_opts
    @unpack log_level = algo_opts
    
    final_stop_crit = MaxIterStopping(max_iter)
    
    restoration_required = stop_optimization = false
    for it_id = 1 : max_iter
        remove_stale_elements!(population)  # optional, we should do some benchmarks ...
        remove_stale_elements!(ndset)
        stop_optimization && break

        @logmsg log_level "🕐 ITERATION $(it_id), len pop = $(length(population.elems))"

        if restoration_required
            sol, status = restore!(
                population, restoration_seeds, ndset, optimizer_caches, algo_opts; 
                gen_id=it_id,
                augment_filter=true
            )
            if status isa AbstractStoppingCriterion
                log_stop_code(status, log_level)
                mark_stale!(population, sol)
                sol.status_ref[] = status
                stop_optimization = true
            end
            restoration_required = false
            continue
        end

        #remove_stale_elements!(restoration_seeds)
        reset_restoration_seeds!(restoration_seeds)
        sorted_sol_oids = sortperm(population.elems; by=solution_sorting_vector, rev=true)

        for sol_objectid in sorted_sol_oids
            sol = population.elems[sol_objectid]
            is_stale(population, sol) && continue
            is_converged(sol) && continue
            sol.gen_id_ref[] >= it_id && continue
            
            if check_filter
                if !is_filter_acceptable(ndset, sol)
                    @logmsg log_level "💀 🦜 marking $(get_identifier(sol)) as stale."
                    mark_stale!(population, sol)
                    continue
                end
            end

            n_is_compatible = step_normally!(sol, ndset, optimizer_caches, algo_opts)
            if n_is_compatible isa AbstractStoppingCriterion
                sol.status_ref[] = n_is_compatible
                if is_ultimate_stop_crit(n_is_compatible) 
                    final_stop_crit = n_is_compatible
                    stop_optimization = true
                end
                continue
            end

            if !n_is_compatible
                #@logmsg log_level "… normal step not compatible. 1/2) Augmenting filter."
                #add_to_filter_and_mark_population!(ndset, population, sol; log_level)
                mark_stale!(population, sol)
                @logmsg log_level "Storing seed."
                add_to_set!(
                    restoration_seeds, 
                    copy_solution_structs(sol);     # I think copying is very important, but I can't remember why
                    log_level, 
                    elem_identifier=get_identifier(sol) # use old id for consistent logging
                )
                continue
            end
            
            status = step_tangentially!(sol, ndset, optimizer_caches, algo_opts)
            if status isa AbstractStoppingCriterion
                sol.status_ref[] = status
                if is_ultimate_stop_crit(status) 
                    final_stop_crit = status
                    stop_optimization = true
                end
                continue
            end
            status = test_trial_point!(
                sol, ndset, population, optimizer_caches, algo_opts, Val(:version2);
                gen_id = it_id,
            )
            if status isa AbstractStoppingCriterion
                sol.status_ref[] = status
                if is_ultimate_stop_crit(status) 
                    final_stop_crit = status
                    stop_optimization = true
                end
                continue
            end
        end

        stop_optimization && continue
        
        stop_optimization = restoration_required = true
        for sol in population.elems
            is_stale(population, sol) && continue
            restoration_required = false
            is_converged(sol) && continue
            stop_optimization = false
            break
        end
        
    end
    @logmsg log_level "🏁 🏁 🏁 🏁 🏁 FINISHED 🏁 🏁 🏁 🏁 🏁 ."

    for sol in population.elems
        if !is_converged(sol)
            sol.status_ref[] = final_stop_crit
        end
    end

    return population
end

function solution_sorting_vector(sol)
    @unpack vals, iteration_scalars = sol
    return vcat(cached_theta(vals), iteration_scalars.delta, -sol.gen_id_ref[])
end


function _test_trial_point!(
    ::Val{:version2}, sol, ndset, 
    population, mop, scaler, trial_caches, vals_tmp , algo_opts;
    indent=0,
)
    @unpack vals, step_vals, mod_vals, mod, iteration_scalars = sol
    @unpack psi_theta, kappa_theta, nu_accept, nu_success, gamma_shrink, gamma_shrink_much, 
        gamma_grow, delta_max = algo_opts
    @unpack log_level = algo_opts

    @unpack diff_x, diff_fx, diff_fx_mod = trial_caches
    
    delta = delta_new = iteration_scalars.delta
    is_good_trial_point = false
    augment_filter = false
    
    tol = 2*mapreduce(eps, min, step_vals.s)
    if _delta_too_small(step_vals.s, tol)
        @logmsg log_level "Zero step. No trial point."
        
        universal_copy!(vals_tmp, vals)
        diff_x .= 0
        diff_fx .= 0
        diff_fx_mod .= 0

        delta_new = gamma_shrink_much * delta
        is_good_trial_point = false
        augment_filter = false
    else

        ## evaluate mop at trial point and store results in vals_tmp
        copyto!(cached_x(vals_tmp), step_vals.xs)
        xs = cached_x(vals_tmp)
        @ignoraise eval_mop!(vals_tmp, mop, scaler)
        
        ## make sure that model is also evaluated at trial point
        mxs = step_vals.fxs
        @ignoraise objectives!(mxs, mod, xs)
        x = cached_x(vals)
        fx = cached_fx(vals)
        fxs = cached_fx(vals_tmp)
        mx = cached_fx(mod_vals)
        θxs = cached_theta(vals_tmp)
      
        @. diff_x = x - xs
        @. diff_fx = fx - fxs
        @. diff_fx_mod = mx - mxs


        @logmsg log_level """
            \n$(indent_str(indent))- x - xs = $(pretty_row_vec(diff_x)), 
            $(indent_str(indent))- fx - fxs = $(pretty_row_vec(diff_fx)), 
            $(indent_str(indent))- mx - mxs = $(pretty_row_vec(diff_fx_mod))"""

        θx = cached_theta(vals)
    
        is_pos = zeros(Bool, length(diff_fx_mod))
        Δmx = Inf
        for (l, del_mx) in enumerate(diff_fx_mod)
            del_mx <= 0 && continue
            is_pos[l] = true
            if del_mx < Δmx 
                Δmx = del_mx
            end
        end

        if !(is_filter_acceptable(ndset, vals_tmp, vals)) || isinf(Δmx)
            ## trial point rejected
            ## radius is reduced
            is_good_trial_point = false
            delta_new = gamma_shrink_much * delta
        else
            ## if m(x) ⪨ m(x+s) + κ⋅θ(x)^ψ
            if vector_dominates(mx, mxs .+ (kappa_theta * θx^psi_theta))
                ## (theta step)
                ## trial point is accepted    
                is_good_trial_point = true
                delta_new = delta
                ## sol is added to filter (sync population!)
                ## (adding sol to filter makes it incompatible with filter, 
                ## it is removed from population
                augment_filter = true
            else
                offset = Δmx * nu_accept
                if is_nondominated_with_offset(population, θxs, fxs, offset)
                    is_good_trial_point = true

                    delta_new = gamma_shrink * delta
                    if minimum( diff_fx[is_pos] ./ diff_fx_mod[is_pos] ) >= nu_success
                        if delta < delta_max 
                            delta = min(delta_max, gamma_grow * delta)
                        end
                    end
                else
                    ## (innacceptable)
                    ## trial point is rejected
                    is_good_trial_point = false
                    ## radius is reduced much
                    delta_new = gamma_shrink_much * delta
                end
            end        
        end
    end
    iteration_scalars.delta = delta_new
    @logmsg log_level "$(indent_str(indent)) Δ_new = $(delta_new) (was Δ = $(delta))"
    return delta_new, is_good_trial_point, augment_filter
end

function vector_dominates(lhs, rhs)
    return all( lhs .<= rhs ) && any( lhs .<= rhs )
end

function cosine_similarity(p1, p2)
    return LA.dot(p1, p2) / (LA.norm(p1) * LA.norm(p2))
end