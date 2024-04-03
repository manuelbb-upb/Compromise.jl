function do_restoration(
    mop, Δ, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, update_results, 
    stop_crits, 
    user_callback, 
    algo_opts;
    it_index, indent
)
    indent += 1
    pad_str = lpad("", indent)
    @logmsg algo_opts.log_level "$(pad_str)Starting restoration."
    update_results.it_stat = RESTORATION

    ret = :TODO
    if !any(isnan.(step_vals.n))
        ret = :NOTNAN
        @. vals_tmp.x = step_vals.xn
        @ignoraise eval_mop!(vals_tmp, mop, scaler)
        if iszero(vals_tmp.theta_ref[])
            @logmsg algo_opts.log_level "$(pad_str) Using `xn` as next iterate."
            xr_opt = copy(step_vals.xn)
            ret = :SUCCESS
        end
    end

    if ret != :SUCCESS
        if (
            (dim_nl_eq_constraints(mop) > 0 || dim_nl_ineq_constraints(mop) > 0) ||
            ret == :TODO
        )
            @logmsg algo_opts.log_level "$(pad_str) Trying to solve nonlinear subproblem"
            x0 = ret == :NONAN ? step_vals.xn : vals.x
	        (θ_opt, xr_opt, ret) = solve_restoration_problem(
                mop, vals_tmp, scaler, scaled_cons, x0, algo_opts.nl_opt
            )
        else
            xr_opt = copy(step_vals.xn)
            ret = :SUCCESS
        end
    end
    
    if ret in (
        :SUCCESS, 
        :STOPVAL_REACHED, 
        :FTOL_REACHED, 
        :XTOL_REACHED, 
        :MAXEVAL_REACHED,
		:MAXTIME_REACHED
    )
        @ignoraise postproccess_restoration(
            xr_opt, Δ, update_results, mop, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
            it_index, indent
        )
    else
        return InfeasibleStopping(indent)
    end
    @ignoraise finish_iteration(
        stop_crits, user_callback,
        update_results, mop, mod, scaler, lin_cons, scaled_cons,
        mod_vals, vals, vals_tmp, step_vals, filter, algo_opts;
        it_index, indent
    )
    
    accept_trial_point!(vals, vals_tmp)
    return nothing
end

function restoration_objective(mop, vals_tmp, scaler, scaled_cons)
    @unpack hx, gx, Ex, Ax, Ex_min_c, Ax_min_b, ξ = vals_tmp
    @unpack A, b, E, c = scaled_cons
    function objf(xr::Vector, grad::Vector)
        if !isempty(grad)
            @error "Restoration only supports derivative-free NLopt algorithms."
        end

        copyto!(ξ, xr)
        apply_scaling!(ξ, scaler, InverseScaling())
        @ignoraise nl_eq_constraints!(hx, mop, ξ)
        @ignoraise nl_ineq_constraints!(gx, mop, ξ)
        lin_cons!(Ex_min_c, Ex, E, c, xr)
        lin_cons!(Ax_min_b, Ax, A, b, xr)
        
        theta = constraint_violation(hx, gx, Ex_min_c, Ax_min_b)
        return theta
    end
    return objf
end

function solve_restoration_problem(mop, vals_tmp, scaler, scaled_cons, x, nl_opt)
    n_vars = length(x)

	opt = NLopt.Opt(nl_opt, n_vars)

	opt.min_objective = restoration_objective(mop, vals_tmp, scaler, scaled_cons)

	if !isnothing(scaled_cons.lb)
        opt.lower_bounds = scaled_cons.lb
    end
    if !isnothing(scaled_cons.ub)
        opt.upper_bounds = scaled_cons.ub
    end
	
    #src # TODO make some of these settings configurable
    opt.xtol_rel = sqrt(eps(eltype(x)))
    opt.xtol_abs = eps(eltype(x))
	opt.maxeval = 50 * n_vars^2
	opt.stopval = 0

    xr0 = deepcopy(x)   # this allocation should not matter too much
	return NLopt.optimize(opt, xr0)
end

function postproccess_restoration(
    xr_opt, Δ, update_results, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
    it_index, indent
)
    @unpack log_level = algo_opts
    ## the nonlinear problem was solved successfully.
    ## pretend, trial point was next iterate and set values
    copyto!(vals_tmp.x, xr_opt)
    @ignoraise eval_mop!(vals_tmp, mop, scaler)

    @. update_results.diff_x = vals.x - vals_tmp.x
    @. update_results.diff_fx = vals.fx - vals_tmp.fx
    @. update_results.diff_fx_mod = mod_vals.fx

    ## the next point should be acceptable for the filter:
    if is_acceptable(filter, vals_tmp.theta_ref[], vals_tmp.phi_ref[])
        ## make models valid at `x + r` and set model values
        ## also compute normal step based on models
        @ignoraise update_models!(mod, Δ, scaler, vals_tmp, scaled_cons; log_level, indent)
        @ignoraise eval_and_diff_mod!(mod_vals, mod, vals_tmp.x)

        @. update_results.diff_fx_mod -= mod_vals.fx

        @ignoraise do_normal_step!(
            step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, scaled_cons, vals_tmp, mod_vals;
            it_index, log_level, indent
        )

        @unpack c_delta, c_mu, mu, delta_max = algo_opts
        n = step_vals.n
        norm_n = LA.norm(n, Inf)
        n_is_compatible = norm_n <= compatibility_test_rhs(c_delta, c_mu, mu, Δ)
        
        ## if the normal step is compatible, we are done and can use vals_tmp as the next
        ## iterate with next radius `_Δ=Δ`.
        _Δ = Δ

        ## if it is not compatible, we might try to increase the radius:
        if !n_is_compatible
            ## a) ‖n‖ ≤ c Δ         ⇔ ‖n‖/c ≤ Δ 
            ## b) ‖n‖ ≤ c κ Δ^{1+μ} ⇔ (‖n‖/(cκ))^{1/(1+μ)} ≤ Δ
            ## ⇒ Δ ≥ max(‖n‖/c, (‖n‖\(cκ))^{1/(1+μ)})).
            _Δ1 = norm_n/c_delta
            _Δ2 = (_Δ1/c_mu)^(1/1+mu)
            _Δ = max(_Δ1, _Δ2)
            if _Δ <= delta_max
                n_is_compatible = true
            end                
        end
        if n_is_compatible
            finalize_update_results!(update_results, _Δ, RESTORATION, true)
            
            @. step_vals.n = vals_tmp.x - vals.x
            @. step_vals.d = 0
            @. step_vals.s = step_vals.n

            return nothing
        end
    end
    return InfeasibleStopping(indent)
end