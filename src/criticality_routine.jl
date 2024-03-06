function criticality_routine(
    ## possibly modified
    Δ, 
    mod, step_vals, mod_vals, step_cache, crit_cache,
    ## not modified
    mop, scaler, lin_cons, scaled_cons, vals, vals_tmp, stop_crits, algo_opts,
    user_callback;
    it_index
)
    @unpack log_level, eps_theta, eps_crit, crit_M, crit_B, crit_alpha = algo_opts
    χ = step_vals.crit_ref[]
    θ = vals.theta_ref[]
    Δ_init = Δ
    stop_code = nothing
    if θ < eps_theta && (χ < eps_crit && Δ > crit_M * χ )
        @logmsg log_level "ITERATION $(it_index): CRITICALITY ROUTINE."
        ## init
        j=0
        Δj = Δ
        modj = crit_cache.mod
        step_cachej = crit_cache.step_cache
        step_valsj = crit_cache.step_vals
        mod_valsj = crit_cache.mod_vals
        universal_copy!(step_cachej, step_cache)
        universal_copy!(step_valsj, step_vals)
        universal_copy!(mod_valsj, mod_vals)
        universal_copy_model!(modj, mod)

        while Δ > crit_M * χ
            stop_code=nothing

            @ignorebreak stop_code = check_stopping_criteria(
                stop_crits, CheckPreCritLoop(),
                mop, mod, scaler, lin_cons, scaled_cons,
                vals, mod_vals, step_vals, filter, algo_opts;
                it_index, delta=Δ, num_crit_loops=j
            )
            @ignorebreak stop_code = check_stopping_criterion(
                user_callback, CheckPreCritLoop(),
                mop, mod, scaler, lin_cons, scaled_cons,
                vals, mod_vals, step_vals, filter, algo_opts;
                it_index, delta=Δ, num_crit_loops=j
            )
            
            @logmsg log_level "\tCRITICALITY LOOP $(j+1), Δ=$Δj > Mχ=$(crit_M * χ)."
            
            Δj *= crit_alpha 
            
            if depends_on_radius(modj)
                @ignorebreak stop_code = update_models!(modj, Δj, mop, scaler, vals, scaled_cons, algo_opts)
                @ignorebreak stop_code = eval_and_diff_mod!(mod_valsj, modj, vals.x)
                @ignorebreak do_normal_step!(
                    step_cachej, step_valsj, Δj, mop, modj, scaler, lin_cons, scaled_cons, 
                    vals, mod_valsj; log_level, it_index
                )
            end
            if !compatibility_test(step_valsj.n, algo_opts, Δj)
                break
            end
            @ignorebreak stop_code = do_descent_step!(
                step_cachej, step_valsj, Δj, mop, modj, scaler, lin_cons, scaled_cons, vals,
                mod_valsj; log_level
            )
            
            χ = step_valsj.crit_ref[]
            Δ = Δj
            universal_copy!(step_cache, step_cachej)
            universal_copy!(step_vals, step_valsj)
            universal_copy!(mod_vals, mod_valsj)
            universal_copy_model!(mod, modj)
            
            @ignorebreak stop_code = check_stopping_criteria(
                stop_crits, CheckPostCritLoop(),
                mop, mod, scaler, lin_cons, scaled_cons,
                vals, mod_vals, step_vals, filter, algo_opts;
                it_index, delta=Δ, num_crit_loops=j
            )
            @ignorebreak stop_code = check_stopping_criterion(
                user_callback, CheckPostCritLoop(),
                mop, mod, scaler, lin_cons, scaled_cons,
                vals, mod_vals, step_vals, filter, algo_opts;
                it_index, delta=Δ, num_crit_loops=j
            )

            j+=1
        end
       
        _Δ = Δ
        Δ = min(max(Δ, algo_opts.crit_B*χ), Δ_init)
        @logmsg log_level """
            \tFinished after $j criticality loop(s), 
            \t Δ=$_Δ > Mχ=$(crit_M * χ), now Δ=$Δ."""
    end
    if isa(stop_code, AbstractStoppingCriterion)
        return stop_code 
    else
        return Δ
    end
end