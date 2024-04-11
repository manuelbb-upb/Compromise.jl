abstract type AbstractStoppingCriterion end

mutable struct WrappedStoppingCriterion{F} <: AbstractStoppingCriterion
    crit :: F
    source :: LineNumberNode
    indent :: Int
end
function stop_message(wcrit)
    "$(indent_str(writ.indent[]))"*stop_message(wcrit.crit)
end

function wrap_stop_crit(ret_val, lnn, indent=0)
	return ret_val
end
function wrap_stop_crit(ret_val::WrappedStoppingCriterion, lnn, indent=0)
	return ret_val
end
function wrap_stop_crit(ret_val::AbstractStoppingCriterion, lnn, indent=0)
	return WrappedStoppingCriterion(ret_val, lnn, indent)
end

struct NoUserCallback <: AbstractStoppingCriterion end

stop_message(::AbstractStoppingCriterion)=nothing

abstract type AbstractStopPoint end

struct CheckPreIteration <: AbstractStopPoint end
struct CheckPostIteration <: AbstractStopPoint end
struct CheckPostDescentStep <: AbstractStopPoint end
struct CheckPreCritLoop <: AbstractStopPoint end
struct CheckPostCritLoop <: AbstractStopPoint end
#struct CheckPostRestoration <: AbstractStopPoint end

function check_stopping_criterion(
    crit::AbstractStoppingCriterion,
    stop_point::AbstractStopPoint,
    optimizer_caches, algo_opts;
)
    return nothing
end

Base.@kwdef struct MaxIterStopping <: AbstractStoppingCriterion
    num_max_iter :: Int = 500    
end
function Base.show(io::IO, crit::MaxIterStopping)
    print(io, "MaxIterStopping($(crit.num_max_iter))")
end
function stop_message(crit::MaxIterStopping)
    "Maximum number of iterations reached."
end
function check_stopping_criterion(
    crit::MaxIterStopping, ::CheckPreIteration,
    optimizer_caches, algo_opts
)
    it_index = optimizer_caches.iteration_scalars.it_index
    return check_max_iter_stopping(crit, it_index)
end
function check_max_iter_stopping(crit, it_index)
    if it_index > crit.num_max_iter
        return crit
    end
    return nothing
end

Base.@kwdef struct MinimumRadiusStopping{F} <: AbstractStoppingCriterion
    delta_min :: F = eps(Float64)
end
function Base.show(io::IO, crit::MinimumRadiusStopping)
    print(io, "MinimumRadiusStopping($(crit.delta_min))")
end
function stop_message(crit::MinimumRadiusStopping)
    "Trust region radius reduced to below `delta_min` ($(crit.delta_min))."
end
function check_stopping_criterion(
    crit::MinimumRadiusStopping, ::CheckPreIteration,
    optimizer_caches, algo_opts
)
    delta = optimizer_caches.iteration_scalars.delta
    return check_minimum_radius_stopping(crit, delta)
end
function check_stopping_criterion(
    crit::MinimumRadiusStopping, ::CheckPreCritLoop,
    optimizer_caches, algo_opts
)
    delta = optimizer_caches.crit_cache.delta
    return check_minimum_radius_stopping(crit, delta)
end
function check_minimum_radius_stopping(crit, delta)
    if delta < crit.delta_min
        return crit
    end
    return nothing
end

Base.@kwdef struct ArgsRelTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
end
function Base.show(io::IO, crit::ArgsRelTolStopping)
    print(io, "ArgsRelTolStopping(tol=$(crit.tol))")
end 
function stop_message(crit::ArgsRelTolStopping) 
    "Relative parameter tolerance reached."
end

function check_stopping_criterion(
    crit::ArgsRelTolStopping,::CheckPostIteration,
    optimizer_caches, algo_opts
)
    point_has_changed = _trial_point_accepted(optimizer_caches.iteration_type)
    @unpack diff_x_norm2, x_norm2 = optimizer_caches.stop_crits
    return check_args_rel_tol_stopping(crit, point_has_changed, diff_x_norm2, x_norm2)
end
function check_args_rel_tol_stopping(crit, point_has_changed, diff_x_norm2, x_norm2)
    if !crit.only_if_point_changed || point_has_changed
        if diff_x_norm2 <= crit.tol * x_norm2
            return crit
        end
    end
    return nothing
end

Base.@kwdef struct ArgsAbsTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
end
function Base.show(io::IO, crit::ArgsAbsTolStopping)
    print(io, "ArgsAbsTolStopping(tol=$(crit.tol))")
end 
function stop_message(crit::ArgsAbsTolStopping) 
    "Absolute parameter tolerance reached."
end
function check_stopping_criterion(
    crit::ArgsAbsTolStopping,::CheckPostIteration,
    optimizer_caches, algo_opts
)
    point_has_changed = _trial_point_accepted(optimizer_caches.iteration_type)
    @unpack diff_x_norm2 = optimizer_caches.stop_crits
    return check_args_abs_tol_stopping(crit, point_has_changed, diff_x_norm2)
end
function check_args_abs_tol_stopping(crit, point_has_changed, diff_x_norm2)
    if !crit.only_if_point_changed || point_has_changed
        if diff_x_norm2 <= crit.tol
            return crit
        end
    end
    return nothing
end

Base.@kwdef struct ValsRelTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
end
function Base.show(io::IO, crit::ValsRelTolStopping)
    print(io, "ValsRelTolStopping(tol=$(crit.tol))")
end 
function stop_message(crit::ValsRelTolStopping) 
    "Relative value tolerance reached."
end

function check_stopping_criterion(
    crit::ValsRelTolStopping,::CheckPostIteration,
    optimizer_caches, algo_opts
)
    point_has_changed = _trial_point_accepted(optimizer_caches.iteration_type)
    @unpack diff_fx_norm2, fx_norm2 = optimizer_caches.stop_crits
    return check_vals_rel_tol_stopping(crit, point_has_changed, diff_fx_norm2, fx_norm2)
end
function check_vals_rel_tol_stopping(crit, point_has_changed, diff_fx_norm2, fx_norm2)
    if !crit.only_if_point_changed || point_has_changed
        if diff_fx_norm2 <= crit.tol * fx_norm2
            return crit
        end
    end
    return nothing
end

Base.@kwdef struct ValsAbsTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
end
function Base.show(io::IO, crit::ValsAbsTolStopping)
    print(io, "ValsAbsTolStopping(tol=$(crit.tol))")
end 
function stop_message(crit::ValsAbsTolStopping) 
    "Absolute value tolerance reached."
end
function check_stopping_criterion(
    crit::ValsAbsTolStopping,::CheckPostIteration,
    optimizer_caches, algo_opts
)
    point_has_changed = _trial_point_accepted(optimizer_caches.iteration_type)
    @unpack diff_fx_norm2 = optimizer_caches.stop_crits
    return check_vals_abs_tol_stopping(crit, point_has_changed, diff_fx_norm2)
end
function check_vals_abs_tol_stopping(crit, point_has_changed, diff_fx_norm2)
    if !crit.only_if_point_changed || point_has_changed
        if diff_fx_norm2 <= crit.tol
            return crit
        end
    end
    return nothing
end

Base.@kwdef struct CritAbsTolStopping{F} <: AbstractStoppingCriterion
    crit_tol :: F
    theta_tol :: F
end

function Base.show(io::IO, crit::CritAbsTolStopping)
    print(io, "CritAbsTolStopping(tol=$(crit.crit_tol))")
end 
function stop_message(crit::CritAbsTolStopping) 
    "Absolute inexact criticality reached."
end
function check_stopping_criterion(
    crit::CritAbsTolStopping,
    ::Union{CheckPostDescentStep, CheckPostCritLoop},
    optimizer_caches, algo_opts
)
    @unpack step_vals, vals = optimizer_caches
    return check_crit_abs_tol_stopping(
        crit, step_vals, vals, algo_opts
    )
end
function check_crit_abs_tol_stopping(crit, step_vals, vals, algo_opts)
    χ = abs(step_vals.crit_ref[])
    θ = abs(cached_theta(vals))
    @unpack log_level = algo_opts
    @unpack crit_tol, theta_tol = crit
    return check_crit_abs_tol_stopping(crit, χ, θ, crit_tol, theta_tol)
end
function check_crit_abs_tol_stopping(crit, χ, θ, crit_tol, theta_tol)
    if χ <= crit_tol && θ <= theta_tol
        return crit
    end
    return nothing
end

Base.@kwdef struct MaxCritLoopsStopping <: AbstractStoppingCriterion
    num :: Int
end
function Base.show(io::IO, crit::MaxCritLoopsStopping)
    print(io, "MaxCritLoopsStopping($(crit.num))")
end
function stop_message(crit::MaxCritLoopsStopping) 
    "Maximum number of criticality loops reached."
end
function check_stopping_criterion(
    crit::MaxCritLoopsStopping,::CheckPreCritLoop,
    optimizer_caches, algo_opts
)
    @unpack num_crit_loops = optimizer_caches.crit_cache
    if num_crit_loops >= crit.num
        return crit
    end
    return nothing
end

struct InfeasibleStopping <: AbstractStoppingCriterion end
Base.show(io::IO, ::InfeasibleStopping)=print(io, "InfeasibleStopping()")
function stop_message(crit::InfeasibleStopping)
    return "INFEASIBLE: Cannot find a feasible point."
end

mutable struct DefaultStoppingCriteriaContainer{F, UC, DC} <: AbstractStoppingCriterion
    x_norm2 :: F
    fx_norm2 :: F
    diff_x_norm2 :: F
    diff_fx_norm2 :: F
    user_callback :: UC
    default_crits :: DC
end

function _prepare_criteria_container(crit_container, optimizer_caches, p::AbstractStopPoint)
    nothing
end

function _prepare_criteria_container(crit_container, optimizer_caches, p::CheckPostIteration)
    @unpack vals, trial_caches = optimizer_caches
    x = cached_x(vals)
    fx = cached_fx(vals)
    diff_x = trial_caches.diff_x
    diff_fx = trial_caches.diff_fx
    crit_container.x_norm2 = LA.norm(x)    
    crit_container.fx_norm2 = LA.norm(fx)    
    crit_container.diff_x_norm2 = LA.norm(diff_x)
    crit_container.diff_fx_norm2 = LA.norm(diff_fx)
    return nothing
end

function check_stopping_criterion(
    crit_container::DefaultStoppingCriteriaContainer, p::AbstractStopPoint,
    optimizer_caches, algo_opts
)
    _prepare_criteria_container(crit_container, optimizer_caches, p)
    for crit in crit_container.default_crits
        @ignoraise check_stopping_criterion(crit, p, optimizer_caches, algo_opts)
    end
    @ignoraise check_usercallback(crit_container, p, optimizer_caches, algo_opts)
    return nothing
end

function check_usercallback(crit_container::DefaultStoppingCriteriaContainer, p, optimizer_caches, algo_opts)
    return check_usercallback(crit_container.user_callback, p, optimizer_caches, algo_opts)
end
function check_usercallback(@nospecialize(user_callback), p, optimizer_caches, algo_opts)
    return check_stopping_criterion(user_callback, p, optimizer_caches, algo_opts)
end

function stopping_criteria(algo_opts, @nospecialize(user_callback))
    F = float_type(algo_opts)
    NaNF = F(NaN)
    default_crits = (
        MaxIterStopping(;num_max_iter=algo_opts.max_iter),
        MinimumRadiusStopping(;delta_min=algo_opts.stop_delta_min),
        ArgsRelTolStopping(;tol=algo_opts.stop_xtol_rel),
        ArgsAbsTolStopping(;tol=algo_opts.stop_xtol_abs),
        ValsRelTolStopping(;tol=algo_opts.stop_ftol_rel),
        ValsAbsTolStopping(;tol=algo_opts.stop_ftol_abs),
        CritAbsTolStopping(;
            crit_tol=algo_opts.stop_crit_tol_abs,
            theta_tol=algo_opts.stop_theta_tol_abs
        ),
        MaxCritLoopsStopping(;num=algo_opts.stop_max_crit_loops)
    )
    return DefaultStoppingCriteriaContainer(NaNF, NaNF, NaNF, NaNF, user_callback, default_crits)
end