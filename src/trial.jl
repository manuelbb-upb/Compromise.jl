function test_trial_point!(
    mop, scaler, iter_meta, filter, vals, mod_vals, vals_tmp, step_vals, algo_opts
)
    # During step computation, `step_vals` was set to contain
    # the step vectors `n`, `xn`, `d` and `xs`.
    # We also have the model objective value vector `fxs`.
    xs = step_vals.xs
    mod_fxs = step_vals.fxs
    
    # Use `vals_tmp` to hold the true values at `xs`:
    copyto!(vals_tmp.x, xs)
    @ignoraise eval_mop!(vals_tmp, mop, scaler)

    # To test the trial point against the filter, extract
    # current values and trial point values
    θx, Φx = vals.θ[], vals.Φ[]
    θxs, Φxs = vals_tmp.θ[], vals_tmp.Φ[]

    fits_filter = is_acceptable(filter, θxs, Φxs, θx, Φx)
    
    ## compute the full difference vectors for stopping criteria
    @. iter_meta.vals_diff_vec = vals.fx - vals_tmp.fx
    @. iter_meta.mod_vals_diff_vec = mod_vals.fx .- mod_fxs

    if fits_filter
        @unpack strict_acceptance_test, kappa_theta, psi_theta, nu_accept = algo_opts
        objf_decrease, model_decrease = if strict_acceptance_test
            iter_meta.vals_diff_vec, iter_meta.mod_vals_diff_vec
        else
            maximum(vals.fx) - maximum(vals_tmp.fx),
                maximum(mod_vals.fx) - maximum(mod_fxs) 
        end
        rho = minimum( objf_decrease ./ model_decrease )            
        model_decrease_condition = all(model_decrease .>= kappa_theta * θx^psi_theta)
        sufficient_decrease_condition = rho >= nu_accept
    end

    @unpack it_index = iter_meta
    iter_meta.it_stat_post = if !fits_filter
        @logmsg algo_opts.log_level "ITERATION $(it_index): Trial point does not fit filter."
        FILTER_FAIL
    else
        @logmsg algo_opts.log_level "ITERATION $(it_index): Trial point is filter-acceptable."
        if model_decrease_condition
            if !sufficient_decrease_condition
                @logmsg algo_opts.log_level "ITERATION $(it_index): Step INACCEPTABLE."
                INACCEPTABLE
            else
                @unpack nu_success = algo_opts
                if rho < nu_success
                    @logmsg algo_opts.log_level "ITERATION $(it_index): Step ACCEPTABLE."
                    ACCEPTABLE
                else
                    @logmsg algo_opts.log_level "ITERATION $(it_index): Step SUCCESSFUL."
                    SUCCESSFUL
                end
            end
        else
            if !sufficient_decrease_condition
                @logmsg algo_opts.log_level "ITERATION $(it_index): Model decrease insufficient and bad step, FILTER_ADD_SHRINK."
                FILTER_ADD_SHRINK
            else
                @logmsg algo_opts.log_level "ITERATION $(it_index): Model decrease insufficient, okay step, FILTER_ADD."
                FILTER_ADD
            end
        end
    end
    
    iter_meta.args_diff_len = LA.norm(step_vals.s)
    iter_meta.vals_diff_len = LA.norm(iter_meta.vals_diff_vec)
    iter_meta.mod_vals_diff_len = LA.norm(iter_meta.mod_vals_diff_vec)

    return nothing
end