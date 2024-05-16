change_float_type(cfg::AbstractStepConfig, ::Type{new_float_type}) where new_float_type = cfg

function init_step_cache(
    cfg::AbstractStepConfig, vals, mod_vals
)
    return nothing
end

function compute_normal_step!(
    step_cache::AbstractStepCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)
    ## set `step_vals.n` to hold normal step
    nothing
end

function compute_descent_step!(
    step_cache::AbstractStepCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)
    ## set `step_vals.d` to hold descent step
    ## set `step_vals.crit_ref` to hold criticality measure
    return nothing
end

function do_normal_step!(
    step_cache::AbstractStepCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level, indent::Int=0
)
    if cached_theta(vals) > 0
        pad_str = indent_str(indent)
        @logmsg log_level "$(pad_str)* Computing a normal step."   

        @ignoraise compute_normal_step!(
            step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, 
            scaled_cons, vals, mod_vals; log_level
        )
        step_vals.xn .= cached_x(vals) .+ step_vals.n
        @logmsg log_level """
            $(pad_str) Found normal step $(pretty_row_vec(step_vals.n; cutoff=60)). 
            $(pad_str) \t Hence xn=$(pretty_row_vec(step_vals.xn; cutoff=60)).""" 

    else
        step_vals.n .= 0
        step_vals.xn .= cached_x(vals) .+ step_vals.n
    end
    nothing
end

function finalize_step_vals!(
    step_cache::AbstractStepCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)
    ## find stepsize and scale `step_vals.d`
    ## set `step_vals.s`, `step_vals.xs`, `step_vals.fxs`
    @. step_vals.s = step_vals.n + step_vals.d
    step_vals.xs .= cached_x(vals) .+ step_vals.s
    @ignoraise objectives!(step_vals.fxs, mod, step_vals.xs)
    return nothing
end

function do_descent_step!(
    step_cache::AbstractStepCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level, finalize::Bool=true
)
    @ignoraise compute_descent_step!(
        step_cache, step_vals, Δ, mop, mod, scaler, lin_cons,
        scaled_cons, vals, mod_vals; log_level)
    
    if finalize
        @ignoraise finalize_step_vals!(
            step_cache, step_vals, Δ, mop, mod, scaler, lin_cons,
            scaled_cons, vals, mod_vals; log_level
        )
    end
    return nothing
end