function do_restoration(
    mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
    mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
    iteration_status, iteration_scalars, stop_crits,
    algo_opts
)
    indent = 1
    Δ = iteration_scalars.delta
    
    indent += 1
    pad_str = lpad("", indent)
    @logmsg algo_opts.log_level "$(pad_str)Starting restoration."
    
    iteration_status.iteration_type = RESTORATION

    ret = :TODO
    if !any(isnan.(step_vals.n))
        ret = :NOTNAN
        copyto!(cached_x(vals_tmp), step_vals.xn)
        @ignoraise eval_mop!(vals_tmp, mop, scaler)
        if iszero(cached_theta(vals_tmp))
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
            x0 = ret == :NONAN ? step_vals.xn : cached_x(vals)
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
            xr_opt, Δ, trial_caches, mop, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
            indent
        ) indent
    else
        return InfeasibleStopping(indent)
    end
    
    # If we are here, then restoration was successfull and `trial_caches` are set
    iteration_scalars.delta = Δ
    @ignoraise check_stopping_criterion(
        stop_crits, CheckPostIteration(), 
        mop, mod, scaler, lin_cons, scaled_cons, vals, vals_tmp,
        mod_vals, filter, step_vals, step_cache, crit_cache, trial_caches, 
        iteration_status, iteration_scalars, stop_crits,
        algo_opts
    ) indent
    accept_trial_point!(vals, vals_tmp)
    return nothing
end

function restoration_objective(mop, vals_tmp, scaler, scaled_cons)
    @unpack A, b, E, c = scaled_cons
    function objf(xr::Vector, grad::Vector)
        if !isempty(grad)
            error("Restoration only supports derivative-free NLopt algorithms.")
        end

        ξ = cached_ξ(vals_tmp)
        unscale!(ξ, scaler, xr)
        @ignoraise nl_eq_constraints!(cached_hx(vals_tmp), mop, ξ)
        @ignoraise nl_ineq_constraints!(cached_gx(vals_tmp), mop, ξ)
        lin_cons!(cached_Ex_min_c(vals_tmp), cached_Ex(vals_tmp), E, c, xr)
        lin_cons!(cached_Ax_min_b(vals_tmp), cached_Ax(vals_tmp), A, b, xr)
        
        theta = constraint_violation(
            cached_hx(vals_tmp), 
            cached_gx(vals_tmp), 
            cached_Ex_min_c(vals_tmp), 
            cached_Ax_min_b(vals_tmp)
        )
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
    xr_opt, Δ, trial_caches, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
    indent
)
    @unpack log_level = algo_opts
    ## the nonlinear problem was solved successfully.
    ## pretend, trial point was next iterate and set values
    copyto!(cached_x(vals_tmp), xr_opt)
    @ignoraise eval_mop!(vals_tmp, mop, scaler)

    trial_caches.diff_x .= cached_x(vals) .- cached_x(vals_tmp)
    trial_caches.diff_fx .= cached_fx(vals) .- cached_fx(vals_tmp)
    trial_caches.diff_fx_mod .= cached_fx(mod_vals)

    ## the next point should be acceptable for the filter:
    if is_acceptable(filter, cached_theta(vals_tmp), cached_Phi(vals_tmp))
        ## make models valid at `x + r` and set model values
        ## also compute normal step based on models
        @ignoraise update_models!(mod, Δ, scaler, vals_tmp, scaled_cons; log_level, indent)
        @ignoraise eval_and_diff_mod!(mod_vals, mod, cached_x(vals_tmp))

        trial_caches.diff_fx_mod .-= cached_fx(mod_vals)

        @ignoraise do_normal_step!(
            step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, scaled_cons, vals_tmp, mod_vals;
            log_level, indent
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
            step_vals.n .= cached_x(vals_tmp) .- cached_x(vals)
            @. step_vals.d = 0
            @. step_vals.s = step_vals.n

            return nothing
        end
    end
    # if we have not returned here, then no compatible normal step was found :(
    return InfeasibleStopping()
end