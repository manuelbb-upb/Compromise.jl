abstract type AbstractStoppingCriterion end

mutable struct WrappedStoppingCriterion{F} <: AbstractStoppingCriterion
    crit :: F
    source :: LineNumberNode
    has_logged :: Bool
end
struct NoUserCallback <: AbstractStoppingCriterion end

stop_message(::AbstractStoppingCriterion)=nothing

abstract type AbstractStopPoint end

struct CheckPreIteration <: AbstractStopPoint end
struct CheckPostIteration <: AbstractStopPoint end
struct CheckPostDescentStep <: AbstractStopPoint end
struct CheckPreCritLoop <: AbstractStopPoint end
struct CheckPostCritLoop <: AbstractStopPoint end

function evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion, ::CheckPreIteration,
    mop, scaler, lin_cons, scaled_cons,
    vals, filter, algo_opts;
    it_index::Int, delta::Real
)
    return nothing
end

function evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion, ::CheckPostDescentStep,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real
)
    return nothing
end

function evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion, ::Union{CheckPreCritLoop, CheckPostCritLoop},
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real, num_crit_loops::Int
)
    return nothing
end

function evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion, ::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    it_index
)
    return nothing
end

function check_stopping_criterion(
    crit::AbstractStoppingCriterion,
    stop_point::AbstractStopPoint,
    args...;
    kwargs...
)
    return evaluate_stopping_criterion(crit, stop_point, args...; kwargs...)
end

function check_stopping_criteria(crits, stop_point::AbstractStopPoint, args...; kwargs...)
    for crit in crits
        @ignoraise check_stopping_criterion(crit, stop_point, args...;kwargs...)
    end
    return nothing
end

@with_kw struct MaxIterStopping <: AbstractStoppingCriterion
    num_max_iter :: Int = 500
end

stop_message(crit::MaxIterStopping)="EXIT, reached maximum number of iterations."

function evaluate_stopping_criterion(
    crit::MaxIterStopping, ::CheckPreIteration,
    mop, scaler, lin_cons, scaled_cons,
    vals, filter, algo_opts;
    it_index::Int, delta::Real
)
    if it_index > crit.num_max_iter
        return crit
    end
    return nothing
end

@with_kw struct MinimumRadiusStopping{F} <: AbstractStoppingCriterion
    delta_min :: F = eps(Float64)
end
function stop_message(crit::MinimumRadiusStopping)
    "EXIT, trust region radius reduced to below `delta_min` ($(crit.delta_min))."
end

function evaluate_stopping_criterion(
    crit::MinimumRadiusStopping, ::CheckPreIteration,
    mop, scaler, lin_cons, scaled_cons,
    vals, filter, algo_opts;
    it_index::Int, delta::Real
)
    if delta < crit.delta_min
        return crit
    end
    return nothing
end

function evaluate_stopping_criterion(
    crit::MinimumRadiusStopping, ::CheckPreCritLoop,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real, num_crit_loops::Int
)
    if delta < crit.delta_min
        return crit
    end
    return nothing
end

@with_kw struct ArgsRelTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
end

function evaluate_stopping_criterion(
    crit::ArgsRelTolStopping,::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    it_index
)
    @unpack x = vals
    @unpack norm2_x, point_has_changed = update_results
    if !crit.only_if_point_changed || point_has_changed
        if norm2_x <= crit.tol * LA.norm(x)
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

function evaluate_stopping_criterion(
    crit::ArgsAbsTolStopping, ::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    it_index
)
    @unpack point_has_changed, norm2_x = update_results
    if !crit.only_if_point_changed || point_has_changed
        if norm2_x <= crit.tol
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

function evaluate_stopping_criterion(
    crit::ValsRelTolStopping,::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    it_index
)
    @unpack fx = vals
    @unpack norm2_fx, point_has_changed = update_results
    if !crit.only_if_point_changed || point_has_changed
        if norm2_fx <= crit.tol * LA.norm(fx)
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

function evaluate_stopping_criterion(
    crit::ValsAbsTolStopping,::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    it_index
)
    @unpack norm2_fx, point_has_changed = update_results
    if !crit.only_if_point_changed || point_has_changed
        if norm2_fx <= crit.tol
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

function evaluate_stopping_criterion(
    crit::CritAbsTolStopping,::CheckPostDescentStep,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real
)
    return crit_abs_tol_stopping(crit, step_vals, vals, it_index, algo_opts)
end

function evaluate_stopping_criterion(
    crit::CritAbsTolStopping,::CheckPostCritLoop,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real, num_crit_loops::Int
)

    return crit_abs_tol_stopping(crit, step_vals, vals, it_index, algo_opts)
end
function crit_abs_tol_stopping(crit, step_vals, vals, it_index, algo_opts)
    χ = abs(step_vals.crit_ref[])
    θ = abs(vals.theta_ref[])
    @unpack log_level = algo_opts
    @unpack crit_tol, theta_tol = crit
    if crit_abs_tol_stopping(χ, θ, crit_tol, theta_tol, log_level, it_index)
        return crit
    end
    return nothing
end
function crit_abs_tol_stopping(χ, θ, crit_tol, theta_tol, log_level, it_index)
    if χ <= crit_tol && θ <= theta_tol
        @logmsg log_level "ITERATION $(it_index): EXIT, absolute criticality tolerance criterion."
        return true
    end
    return false
end

struct MaxCritLoopsStopping <: AbstractStoppingCriterion
    num :: Int
end

check_pre_crit_loop(crit::MaxCritLoopsStopping)=true

function evaluate_stopping_criterion(
    crit::MaxCritLoopsStopping,::CheckPreCritLoop,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real, num_crit_loops::Int
)

    @unpack log_level = algo_opts
    if num_crit_loops >= crit.num
        @logmsg log_level "ITERATION $(it_index): EXIT, maximum number of criticality loops."
        return crit
    end
    return nothing
end

struct InfeasibleStopping <: AbstractStoppingCriterion end