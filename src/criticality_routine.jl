function criticality_routine!(optimizer_caches, algo_opts; indent=0)
   
    @unpack (
        mod, step_vals, mod_vals, step_cache, crit_cache,
        ## not modified
        mop, scaler, lin_cons, scaled_cons, vals, vals_tmp, stop_crits,
        iteration_scalars
    ) = optimizer_caches

    indent += 1
    pad_str = indent_str(indent)

    @unpack log_level, eps_theta, eps_crit, crit_M, crit_B, crit_alpha, 
        backtrack_in_crit_routine = algo_opts
    @unpack it_index = iteration_scalars

    χ = step_vals.crit_ref[]
    θ = cached_theta(vals)
    stop_code = nothing
    if θ < eps_theta && (χ < eps_crit && Δ > crit_M * χ )
        @logmsg log_level "$(pad_str)ITERATION $(it_index): CRITICALITY ROUTINE."
        ## init
        Δ_init = iteration_scalars.delta
        j=0
        Δj = Δ
        
        stop_code=nothing
        while Δ > crit_M * χ
            # sync `crit_cache` with local vars for stopping criteria
            crit_cache.num_crit_loops = j
            crit_cache.delta = Δj
            # at this point, `step_vals` have a compatible normal step and descent step
            # we sync this in case we cannot find a normal step for smaller radius `Δj`
            universal_copy!(crit_cache.step_vals, step_vals)

            @ignorebreak stop_code =  check_stopping_criterion(
                stop_crits, CheckPreCritLoop(), optimizer_caches, algo_opts
            ) indent
            
            @logmsg log_level "$(pad_str) CRITICALITY LOOP $(j+1), Δ=$Δj > Mχ=$(crit_M * χ)."
            
            Δj *= crit_alpha 
            
            if depends_on_radius(mod)
                @ignorebreak stop_code = update_models!(mod, Δj, scaler, vals, scaled_cons; log_level, indent) indent
                @ignorebreak stop_code = eval_and_diff_mod!(mod_vals, mod, vals.x) indent
                @ignorebreak stop_code = do_normal_step!(
                    step_cache, step_vals, Δj, mop, mod, scaler, lin_cons, scaled_cons, 
                    vals, mod_vals; 
                    log_level, indent
                ) indent
            end
            if !compatibility_test(step_vals.n, algo_opts, Δj)
                break
            end
            @ignorebreak stop_code = do_descent_step!(
                step_cache, step_vals, Δj, mop, mod, scaler, lin_cons, scaled_cons, vals,
                mod_vals; log_level, finalize=backtrack_in_crit_routine
            ) indent
            
            χ = step_vals.crit_ref[]
            Δ = Δj
           
            @ignorebreak stop_code =  check_stopping_criterion(
                stop_crits, CheckPreCritLoop(), optimizer_caches, algo_opts
            ) indent

            j+=1
        end       
        _Δ = Δ  # storred for logging only
        Δ = min(max(_Δ, algo_opts.crit_B*χ), Δ_init)

        # make sure we have compatible steps
        universal_copy!(step_vals, crit_cache.step_vals)

        if !backtrack_in_crit_routine
            @ignoraise finalize_step_vals!(
                step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals; log_level) indent
        end
        @logmsg log_level """
            $(pad_str) Finished after $j criticality loop(s), 
            $(pad_str) Δ=$_Δ > Mχ=$(crit_M * χ), now Δ=$Δ."""
    end
    if isa(stop_code, AbstractStoppingCriterion)
        return stop_code 
    else
        return Δ
    end
end