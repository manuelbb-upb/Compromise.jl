abstract type AbstractStoppingCriterion end

mutable struct WrappedStoppingCriterion{F} <: AbstractStoppingCriterion
    crit :: F
    source :: LineNumberNode
    has_logged :: Bool
end
struct NoUserCallback <: AbstractStoppingCriterion end

stop_message(::AbstractStoppingCriterion)=nothing

check_pre_iteration(crit::AbstractStoppingCriterion)=false
#src check_post_normal_step(crit::AbstractStoppingCriterion)=false
check_post_descent_step(crit::AbstractStoppingCriterion)=false
check_post_iteration(crit::AbstractStoppingCriterion)=false
check_pre_crit_loop(crit::AbstractStoppingCriterion)=false
#src check_post_crit_loop_normal_step(crit::AbstractStoppingCriterion)=false
check_post_crit_loop(crit::AbstractStoppingCriterion)=false

function evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts,
)
    return nothing
end

function _evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion,
    args...
)
    # TODO safeguards and sanity checks...
    return evaluate_stopping_criterion(crit, args...)
end

@with_kw struct MaxIterStopping <: AbstractStoppingCriterion
    num_max_iter :: Int = 500
end

check_pre_iteration(crit::MaxIterStopping)=true

function evaluate_stopping_criterion(
    crit::MaxIterStopping,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts;
)
    @unpack it_index = iter_meta
    if it_index <= crit.num_max_iter
        return nothing
    else
        @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT, reached maximum number of iterations ($(crit.num_max_iter))."
        return crit
    end
end

@with_kw struct MinimumRadiusStopping{F} <: AbstractStoppingCriterion
    delta_min :: F = eps(Float64)
end

check_pre_iteration(crit::MinimumRadiusStopping)=true
check_post_crit_loop(crit::MinimumRadiusStopping)=true

function evaluate_stopping_criterion(
    crit::MinimumRadiusStopping,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts;
)
    @unpack it_index = iter_meta
    if Δ >= crit.delta_min
        return nothing
    else
        @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT, trust region radius $(Δ) reduced to below `delta_min` ($(crit.delta_min))."
        return crit
    end
end

@with_kw struct ArgsRelTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
end

check_post_iteration(crit::ArgsRelTolStopping)=true
function evaluate_stopping_criterion(
    crit::ArgsRelTolStopping,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts;
)
    @unpack it_index, point_has_changed = iter_meta
    if !crit.only_if_point_changed || point_has_changed
        if iter_meta.args_diff_len <= crit.tol * LA.norm(vals.x)
            @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT, relative parameter tolerance criterion."
            return crit
        end
    end
    return nothing
end

@with_kw struct ArgsAbsTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
end

check_post_iteration(crit::ArgsAbsTolStopping)=true
function evaluate_stopping_criterion(
    crit::ArgsAbsTolStopping,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts;
)
    @unpack it_index, point_has_changed = iter_meta
    if !crit.only_if_point_changed || point_has_changed
        if iter_meta.args_diff_len <= crit.tol
            @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT, absolute parameter tolerance criterion."
            return crit
        end
    end
    return nothing
end

@with_kw struct ValsRelTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
end

check_post_iteration(crit::ValsRelTolStopping)=true
function evaluate_stopping_criterion(
    crit::ValsRelTolStopping,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts;
)
    @unpack it_index, point_has_changed = iter_meta
    if !crit.only_if_point_changed || point_has_changed
        if iter_meta.vals_diff_len <= crit.tol * LA.norm(vals.fx)
            @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT, relative value tolerance criterion."
            return crit
        end
    end
    return nothing
end

@with_kw struct ValsAbsTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
end

check_post_iteration(crit::ValsAbsTolStopping)=true
function evaluate_stopping_criterion(
    crit::ValsAbsTolStopping,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts;
)
    @unpack it_index, point_has_changed = iter_meta
    if !crit.only_if_point_changed || point_has_changed
        if iter_meta.vals_diff_len <= crit.tol
            @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT, absolute value tolerance criterion."
            return crit
        end
    end
    return nothing
end

@with_kw struct CritAbsTolStopping{F} <: AbstractStoppingCriterion
    crit_tol :: F
    theta_tol :: F
end

check_post_descent_step(crit::CritAbsTolStopping)=true
check_post_crit_loop(crit::CritAbsTolStopping)=true

function evaluate_stopping_criterion(
    crit::CritAbsTolStopping,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts;
)
    @unpack it_index = iter_meta
    if abs(iter_meta.crit_val) <= crit.crit_tol && abs(vals.θ[]) <= crit.theta_tol
        @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT, absolute criticality tolerance criterion."
        return crit
    end
    return nothing
end

struct MaxCritLoopsStopping <: AbstractStoppingCriterion
    num :: Int
end

check_pre_crit_loop(crit::MaxCritLoopsStopping)=true

function evaluate_stopping_criterion(
    crit::MaxCritLoopsStopping,
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts;
)
    @unpack it_index = iter_meta
    if iter_meta.num_crit_loops >= crit.num
        @logmsg algo_opts.log_level "ITERATION $(it_index): EXIT, maximum number of criticality loops."
        return crit
    end
    return nothing
end

struct InfeasibleStopping <: AbstractStoppingCriterion end