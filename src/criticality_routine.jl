function criticality_routine!(
    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
    iteration_status, iteration_scalars, stop_crits,
    algo_opts;
    indent=0
)
   
    indent += 1
    pad_str = indent_str(indent)

    @unpack log_level, eps_theta, eps_crit, crit_M, crit_B, crit_alpha, 
        backtrack_in_crit_routine = algo_opts
    @unpack it_index = iteration_scalars
    Δ = iteration_scalars.delta

    χ = step_vals.crit_ref[]
    θ = cached_theta(vals)
    stop_code = nothing
    break_because_incompatible = false
    if θ < eps_theta && (χ < eps_crit && Δ > crit_M * χ )
        @logmsg log_level "$(pad_str)ITERATION $(it_index): CRITICALITY ROUTINE (θ = $θ)."
        ## init
        Δj = Δ_init = Δ
        j=0
        
        stop_code=nothing
        while Δ > crit_M * χ
            # sync `crit_cache` with local vars for stopping criteria
            crit_cache.num_crit_loops = j
            crit_cache.delta = Δj
            # at this point, `step_vals` have a compatible normal step and descent step
            # we sync this in case we cannot find a normal step for smaller radius `Δj`
            universal_copy!(crit_cache.step_vals, step_vals)
            @ignorebreak stop_code =  check_stopping_criterion(
                stop_crits, CheckPreCritLoop(), 
                mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                iteration_status, iteration_scalars, stop_crits,
                algo_opts
            ) indent
            
            @logmsg log_level "$(pad_str) CRITICALITY LOOP $(j+1), Δ=$Δj > Mχ=$(crit_M * χ)."
            
            Δj *= crit_alpha 
            
            if depends_on_radius(mod)
                @ignorebreak stop_code = update_models!(mod, Δj, scaler, vals, scaled_cons; log_level, indent) indent
                @ignorebreak stop_code = eval_and_diff_mod!(mod_vals, mod, cached_x(vals)) indent
                @ignorebreak stop_code = do_normal_step!(
                    step_cache, step_vals, Δj, mop, mod, scaler, lin_cons, scaled_cons, 
                    vals, mod_vals; 
                    log_level, indent
                ) indent
            end
            
            if !compatibility_test(step_vals.n, algo_opts, Δj)
                # make sure we have compatible steps after exiting the loop
                universal_copy!(step_vals, crit_cache.step_vals)
                break_because_incompatible = true
                # do not exit just yet;
                # before, we recompute the descent direction in case we have to do 
                # backtracking after exiting the while loop
            else
                Δ = Δj
            end
            
            @ignorebreak stop_code = do_descent_step!(
                step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, scaled_cons, vals,
                mod_vals; log_level, finalize=backtrack_in_crit_routine
            ) indent

            break_because_incompatible && break

            ## read criticality value to local var
            χ = step_vals.crit_ref[]
           
            @ignorebreak stop_code = check_stopping_criterion(
                stop_crits, CheckPostCritLoop(), 
                mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
                mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
                iteration_status, iteration_scalars, stop_crits,
                algo_opts
            ) indent

            j+=1
        end       
        _Δ = Δ  # `_Δ` storred for logging only
        Δ = min(max(_Δ, crit_B*χ), Δ_init)
        
        if !backtrack_in_crit_routine           
            @ignoraise finalize_step_vals!(
                step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals; log_level) indent
        end
        @logmsg log_level """\n
            $(pad_str) Finished after $j criticality loop(s),\
            $(if break_because_incompatible
                "\n$(pad_str) normal step would be incompatible,"
            else
                ""
            end)
            $(pad_str) Values: 
            $(pad_str) _Δ    = $_Δ
            $(pad_str) Mξ    = $(crit_M * χ)
            $(pad_str)   χ   = $(step_vals.crit_ref[]), 
            $(pad_str)  ‖d‖₂ = $(LA.norm(step_vals.d)), 
            $(pad_str)  ‖s‖₂ = $(LA.norm(step_vals.s)),
            $(pad_str)  Δ    = $(Δ)""" 
    end
    if isa(stop_code, AbstractStoppingCriterion)
        return stop_code 
    else
        return Δ
    end
end