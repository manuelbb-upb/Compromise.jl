function criticality_routine(
    ## possibly modified
    iter_meta,  # field `crit_val` and `num_crit_loops`
    mod, step_vals, mod_vals, step_cache, crit_cache,
    ## not modified
    mop, scaler, lin_cons, scaled_cons, vals, vals_tmp, stop_crits, algo_opts,
    user_callback
)
    @unpack log_level, eps_theta, eps_crit, crit_M, crit_B, crit_alpha = algo_opts
    @unpack it_index = iter_meta
    χ = iter_meta.crit_val
    θ = vals.θ[]
    Δ = iter_meta.Δ_pre
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
        copyto!(step_cachej, step_cache)
        copyto!(step_valsj, step_vals)
        copyto!(mod_valsj, mod_vals)
        _copyto_model!(modj, mod)
        while Δj > crit_M * χ
            iter_meta.num_crit_loops = j
            stop_code=nothing
            for stopping_criterion in stop_crits
                if check_pre_crit_loop(stopping_criterion)
                    stop_code = _evaluate_stopping_criterion(
                        stopping_criterion, Δj, mop, modj, scaler, lin_cons, scaled_cons, 
                        vals, vals_tmp, step_valsj, mod_valsj, filter, iter_meta, step_cachej, algo_opts
                    )
                    !isnothing(stop_code) && break
                end
            end
            !isnothing(stop_code) && break
            if check_pre_crit_loop(user_callback)
                stop_code = evaluate_stopping_criterion(
                    user_callback, Δj, mop, modj, scaler, lin_cons, scaled_cons, 
                    vals, vals_tmp, step_valsj, mod_valsj, filter, iter_meta, step_cachej, algo_opts
                )
                !isnothing(stop_code) && break
            end
            !isnothing(stop_code) && break
            @logmsg log_level "\tCRITICALITY LOOP $(j+1), Δ=$Δj, Mχ=$(crit_M * χ)."
            Δj *= crit_alpha 
            if depends_on_radius(modj)
                _stop_code = update_models!(modj, Δj, mop, scaler, vals, scaled_cons, algo_opts)
                if !isnothing(_stop_code)
                    stop_code = GenericStopping(_stop_code, log_level)
                    break
                end
                _stop_code = eval_and_diff_mod!(mod_valsj, modj, vals.x)
                if !isnothing(_stop_code)
                    stop_code = GenericStopping(_stop_code, log_level)
                    break
                end
                compute_normal_step!(
                    step_cachej, step_valsj.n, step_valsj.xn, Δj, vals.θ[], 
                    vals.ξ, vals.x, vals.fx, vals.hx, vals.gx, 
                    mod_valsj.fx, mod_valsj.hx, mod_valsj.gx, mod_valsj.Dfx, mod_valsj.Dhx, mod_valsj.Dgx,
                    vals.Eres, vals.Ex, vals.Ares, vals.Ax, scaled_cons.lb, scaled_cons.ub, 
                    scaled_cons.E_c, scaled_cons.A_b, modj
                )
            end
            if !compatibility_test(step_valsj.n, algo_opts, Δj)
                break
            end
            _stop_code = compute_descent_step!(
                step_cachej, step_valsj.d, step_valsj.s, step_valsj.xs, step_valsj.fxs, 
                step_valsj.crit_ref,
                Δj, θ, vals.ξ, vals.x, step_valsj.n, step_valsj.xn, vals.fx, vals.hx, vals.gx, 
                mod_valsj.fx, mod_valsj.hx, mod_valsj.gx, mod_valsj.Dfx, mod_valsj.Dhx, mod_valsj.Dgx,
                vals.Eres, vals.Ex, vals.Ares, vals.Ax, scaled_cons.lb, scaled_cons.ub, 
                scaled_cons.E_c, scaled_cons.A_b, modj, mop, scaler
            )
            χ = step_valsj.crit_ref[]
            Δ = Δj
            copyto!(step_cache, step_cachej)
            copyto!(step_vals, step_valsj)
            copyto!(mod_vals, mod_valsj)
            _copyto_model!(mod, modj)        # check if this is non-allocating
            if !isnothing(_stop_code)
                stop_code = GenericStopping(_stop_code, log_level)
                break
            end
            for stopping_criterion in stop_crits
                if check_post_crit_loop(stopping_criterion)
                    stop_code = _evaluate_stopping_criterion(
                        stopping_criterion, Δ, mop, mod, scaler, lin_cons, scaled_cons, 
                        vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts
                    )
                    !isnothing(stop_code) && break
                end
            end
            !isnothing(stop_code) && break
            if check_post_crit_loop(user_callback)
                stop_code = evaluate_stopping_criterion(
                    user_callback, Δ, mop, mod, scaler, lin_cons, scaled_cons, 
                    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts
                )
                !isnothing(stop_code) && break
            end
            !isnothing(stop_code) && break
            j+=1
        end
        #=
        if j == algo_opts.stop_max_crit_loops
            @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT -- MAXIMUM NUMBER OF CRITICALITY LOOPS."
            return CRITICAL_LOOP, CRITICAL, false, Δ
        end
        if Δ <= algo_opts.stop_delta_min
            @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT -- SMALL TRUST REGION RADIUS."
            return CRITICAL_LOOP, TOLERANCE_Δ, false, Δ
        end
        =#
        @logmsg log_level "\tFinished after $j criticality loop(s)."
        Δ = min(max(Δ, algo_opts.crit_B*χ), iter_meta.Δ_pre)
        iter_meta.crit_val = χ
    end
    return Δ, stop_code
end