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
    indent::Int, it_index::Int, delta::Real
)
    return nothing
end

function evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion, ::CheckPostDescentStep,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    indent::Int, it_index::Int, delta::Real
)
    return nothing
end

function evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion, ::Union{CheckPreCritLoop, CheckPostCritLoop},
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    indent::Int, it_index::Int, delta::Real, num_crit_loops::Int
)
    return nothing
end

function evaluate_stopping_criterion(
    crit::AbstractStoppingCriterion, ::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    indent::Int, it_index
)
    return nothing
end

function check_stopping_criterion(
    crit::AbstractStoppingCriterion,
    stop_point::AbstractStopPoint,
    args...;
    indent::Int=0,
    kwargs...
)
    return evaluate_stopping_criterion(crit, stop_point, args...; indent, kwargs...)
end

function check_stopping_criteria(
    crits, stop_point::AbstractStopPoint, args...; indent::Int=0, kwargs...
)
    for crit in crits
        @ignoraise check_stopping_criterion(crit, stop_point, args...; indent, kwargs...)
    end
    return nothing
end

Base.@kwdef struct MaxIterStopping <: AbstractStoppingCriterion
    num_max_iter :: Int = 500
    
    indent :: Base.RefValue{Int} = Ref(0)
end
function Base.show(io::IO, crit::MaxIterStopping)
    print(io, "MaxIterStopping($(crit.num_max_iter))")
end

function stop_message(crit::MaxIterStopping)
    pad_str = lpad("", crit.indent[])
    "$(pad_str)EXIT, reached maximum number of iterations."
end

function evaluate_stopping_criterion(
    crit::MaxIterStopping, ::CheckPreIteration,
    mop, scaler, lin_cons, scaled_cons,
    vals, filter, algo_opts;
    it_index::Int, delta::Real, indent::Int
)
    if it_index > crit.num_max_iter
        crit.indent[] = indent
        return crit
    end
    return nothing
end

Base.@kwdef struct MinimumRadiusStopping{F} <: AbstractStoppingCriterion
    delta_min :: F = eps(Float64)
    
    indent :: Base.RefValue{Int} = Ref(0)
end

function stop_message(crit::MinimumRadiusStopping)
    pad_str = lpad("", crit.indent[])
    "$(pad_str)EXIT, trust region radius reduced to below `delta_min` ($(crit.delta_min))."
end
function Base.show(io::IO, crit::MinimumRadiusStopping)
    print(io, "MinimumRadiusStopping($(crit.delta_min))")
end

function evaluate_stopping_criterion(
    crit::MinimumRadiusStopping, ::CheckPreIteration,
    mop, scaler, lin_cons, scaled_cons,
    vals, filter, algo_opts;
    it_index::Int, delta::Real, indent::Int
)
    if delta < crit.delta_min
        crit.indent[] = indent
        return crit
    end
    return nothing
end

function evaluate_stopping_criterion(
    crit::MinimumRadiusStopping, ::CheckPreCritLoop,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real, num_crit_loops::Int, indent::Int
)
    if delta < crit.delta_min
        crit.indent[] = indent
        return crit
    end
    return nothing
end

Base.@kwdef struct ArgsRelTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true

    indent :: Base.RefValue{Int} = Ref(0)
end

function Base.show(io::IO, crit::ArgsRelTolStopping)
    print(io, "ArgsRelTolStopping(tol=$(crit.tol))")
end 

function stop_message(crit::ArgsRelTolStopping) 
    pad_str = lpad("", crit.indent[])
    "$(pad_str)EXIT, relative parameter tolerance criterion."
end

function evaluate_stopping_criterion(
    crit::ArgsRelTolStopping,::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    it_index, indent::Int
)
    x = cached_x(vals)
    @unpack norm2_x, point_has_changed = update_results
    if !crit.only_if_point_changed || point_has_changed
        if norm2_x <= crit.tol * LA.norm(x)
            crit.indent[] = indent
            return crit
        end
    end
    return nothing
end

Base.@kwdef struct ArgsAbsTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
    indent :: Base.RefValue{Int} = Ref(0)
end

function Base.show(io::IO, crit::ArgsAbsTolStopping)
    print(io, "ArgsAbsTolStopping(tol=$(crit.tol))")
end 

function stop_message(crit::ArgsAbsTolStopping) 
    pad_str = lpad("", crit.indent[])
    "$(pad_str)EXIT, absolute parameter tolerance criterion."
end

function evaluate_stopping_criterion(
    crit::ArgsAbsTolStopping, ::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    it_index, indent::Int
)
    @unpack point_has_changed, norm2_x = update_results
    if !crit.only_if_point_changed || point_has_changed
        if norm2_x <= crit.tol
            crit.indent[] = indent
            return crit
        end
    end
    return nothing
end

Base.@kwdef struct ValsRelTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true

    indent :: Base.RefValue{Int} =Ref(0)
end

function Base.show(io::IO, crit::ValsRelTolStopping)
    print(io, "ValsRelTolStopping(tol=$(crit.tol))")
end 

function stop_message(crit::ValsRelTolStopping) 
    pad_str = lpad("", crit.indent[])
    "$(pad_str)EXIT, relative value tolerance criterion."
end

function evaluate_stopping_criterion(
    crit::ValsRelTolStopping,::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    it_index, indent::Int
)
    fx = cached_fx(vals)
    @unpack norm2_fx, point_has_changed = update_results
    if !crit.only_if_point_changed || point_has_changed
        if norm2_fx <= crit.tol * LA.norm(fx)
            crit.indent[] = indent
            return crit
        end
    end
    return nothing
end

Base.@kwdef struct ValsAbsTolStopping{F} <: AbstractStoppingCriterion
    tol :: F = -Inf
    only_if_point_changed :: Bool = true
    indent :: Base.RefValue{Int} =Ref(0)
end

function Base.show(io::IO, crit::ValsAbsTolStopping)
    print(io, "ValsAbsTolStopping(tol=$(crit.tol))")
end 

function stop_message(crit::ValsAbsTolStopping) 
    pad_str = lpad("", crit.indent[])
    "$(pad_str)EXIT, absolute value tolerance criterion."
end

function evaluate_stopping_criterion(
    crit::ValsAbsTolStopping,::CheckPostIteration,
    update_results,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, vals_tmp, step_vals, filter, algo_opts;
    it_index, indent::Int
)
    @unpack norm2_fx, point_has_changed = update_results
    if !crit.only_if_point_changed || point_has_changed
        if norm2_fx <= crit.tol
            crit.indent[] = indent
            return crit
        end
    end
    return nothing
end

Base.@kwdef struct CritAbsTolStopping{F} <: AbstractStoppingCriterion
    crit_tol :: F
    theta_tol :: F
    indent :: Base.RefValue{Int} =Ref(0)
end

function Base.show(io::IO, crit::CritAbsTolStopping)
    print(io, "CritAbsTolStopping(tol=$(crit.crit_tol))")
end 

function stop_message(crit::CritAbsTolStopping) 
    pad_str = lpad("", crit.indent[])
    "$(pad_str)EXIT, absolute inexact criticality criterion."
end

function evaluate_stopping_criterion(
    crit::CritAbsTolStopping,::CheckPostDescentStep,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real, indent::Int
)
    return crit_abs_tol_stopping(crit, step_vals, vals, it_index, algo_opts; indent)
end

function evaluate_stopping_criterion(
    crit::CritAbsTolStopping,::CheckPostCritLoop,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real, num_crit_loops::Int, indent::Int
)

    return crit_abs_tol_stopping(crit, step_vals, vals, it_index, algo_opts; indent)
end
function crit_abs_tol_stopping(crit, step_vals, vals, it_index, algo_opts; indent)
    χ = abs(step_vals.crit_ref[])
    θ = abs(cached_theta(vals))
    @unpack log_level = algo_opts
    @unpack crit_tol, theta_tol = crit
    if crit_abs_tol_stopping(χ, θ, crit_tol, theta_tol, log_level, it_index)
        crit.indent[] = indent
        return crit
    end
    return nothing
end

function crit_abs_tol_stopping(χ, θ, crit_tol, theta_tol, log_level, it_index)
    if χ <= crit_tol && θ <= theta_tol
        return true
    end
    return false
end

Base.@kwdef struct MaxCritLoopsStopping <: AbstractStoppingCriterion
    num :: Int
    indent :: Base.RefValue{Int} = Ref(0)
end
function Base.show(io::IO, crit::MaxCritLoopsStopping)
    print(io, "MaxCritLoopsStopping($(crit.num))")
end

function stop_message(crit::MaxCritLoopsStopping) 
    pad_str = lpad("", crit.indent[])
    "$(pad_str)EXIT, maximum number of criticality loops."
end
 
function evaluate_stopping_criterion(
    crit::MaxCritLoopsStopping,::CheckPreCritLoop,
    mop, mod, scaler, lin_cons, scaled_cons,
    vals, mod_vals, step_vals, filter, algo_opts;
    it_index::Int, delta::Real, num_crit_loops::Int, indent::Int
)

    @unpack log_level = algo_opts
    if num_crit_loops >= crit.num
        crit.indent[] = indent
        return crit
    end
    return nothing
end

Base.@kwdef struct InfeasibleStopping <: AbstractStoppingCriterion
    indent :: Int = 0 
end
Base.show(io::IO, ::InfeasibleStopping)=print(io, "InfeasibleStopping()")
function stop_message(crit::InfeasibleStopping)
    pad_str = lpad("", crit.indent)
    return "$(pad_str)INFEASIBLE: Cannot find a feasible point."
end