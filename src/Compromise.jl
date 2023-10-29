module Compromise

# ## Imports

# ### External Dependencies
# We use the macros in `Parameters` quite often, because of their convenience:
import Parameters: @with_kw, @unpack
# Everything in this module needs at least some linear algebra:
import LinearAlgebra as LA

import Logging: @logmsg, LogLevel

# Re-export symbols from important sub-modules
import Reexport: @reexport

# With the external dependencies available, we can include global type definitions and constants:
include("types.jl")
include("utils.jl")

# #### Optimization Packages
# At some point, the choice of solver is meant to be configurable, with different
# extensions to choose from:
import JuMP     # LP and QP modelling for descent and normal steps
import COSMO    # actual QP solver
const DEFAULT_QP_OPTIMIZER=COSMO.Optimizer
# For restoration we currently use `NLopt`. This is also meant to become 
# configurable...
import NLopt

# ### Interfaces and Algorithm Types
# Abstract types and interfaces to handle multi-objective optimization problems...
include("mop.jl")
# ... and how to model them:
include("models.jl")

# Tools to scale and unscale variables:
include("affine_scalers.jl")
# Implementations of Filter(s):
include("filter.jl")
# Types and methods to compute inexact normal steps and descent steps:
include("descent.jl")
# The restoration utilities:
include("restoration.jl")
# Trial point testing:
include("trial.jl")
# Criticality Routine has its own file too:
include("criticality_routine.jl")
# Stopping criteria:
include("stopping.jl")

# ### Internal Dependencies or Extensions
# Import operator types and interface definitions:
include("CompromiseEvaluators.jl")
using .CompromiseEvaluators
const CE = CompromiseEvaluators

# Import wrapper types to make user-provided functions conform to the operator interface:
include("evaluators/NonlinearFunctions.jl")
using .NonlinearFunctions

# Import the optional extension `ForwardDiffBackendExt`, if `ForwardDiff` is available:
if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/ForwardDiffBackendExt/ForwardDiffBackendExt.jl")
            import .ForwardDiffBackendExt
        end
    end
end

# As of now, it is not easy to export stuff from extensions.
# We have this Getter instead:
function ForwardDiffBackend()
    if !isdefined(Base, :get_extension)
        if isdefined(@__MODULE__, :ForwardDiffBackendExt)
            return ForwardDiffBackendExt.ForwardDiffBackend()
        end
            return nothing
    else
        m = Base.get_extension(@__MODULE__, :ForwardDiffBackendExt)
        return isnothing(m) ? m : m.ForwardDiffBackend()
    end
end
export ForwardDiffBackend

# Import Radial Basis Function surrogates:
include("evaluators/RBFModels.jl")
@reexport using .RBFModels

# Taylor Polynomial surrogates:
include("evaluators/TaylorPolynomialModels.jl")
@reexport using .TaylorPolynomialModels

# Exact “Surrogates”:
include("evaluators/ExactModels.jl")
@reexport using .ExactModels

# The helpers in `simple_mop.jl` depend on those model types:
include("simple_mop.jl")
export MutableMOP, add_objectives!, add_nl_ineq_constraints!, add_nl_eq_constraints!

# ## The Algorithm
# (This still neads some re-factoring...)
var_bounds_valid(lb, ub)=true
var_bounds_valid(lb::Nothing, ub::RVec)=!(any(isequal(-Inf), ub))
var_bounds_valid(lb::RVec, ub::Nothing)=!(any(isequal(Inf), lb))
var_bounds_valid(lb::RVec, ub::RVec)=all(lb .<= ub)

function init_scaler(scaler_cfg::Symbol, mod_type, lin_cons, dim)
    return init_scaler(Val(scaler_cfg), mod_type, lin_cons, dim)
end

function init_scaler(::Val{:box}, mod_type, lin_cons, dim)
    if supports_scaling(mod_type) isa AbstractAffineScalingIndicator
        @unpack lb, ub = lin_cons
        return init_box_scaler(lb, ub, dim)
    end
    @warn "Problem structure does not support scaling according to `scaler_cfg=:box`. Proceeding without."
    return IdentityScaler(dim)    
end
init_scaler(::Val{:none}, mod_type, lin_cons, dim) = IdentityScaler(dim)

function init_lin_cons(mop)
    lb = lower_var_bounds(mop)
    ub = upper_var_bounds(mop)

    if !var_bounds_valid(lb, ub)
        error("Variable bounds unvalid.")
    end

    A_b = lin_eq_constraints(mop)
    E_c = lin_ineq_constraints(mop)

    return LinearConstraints(lb, ub, A_b, E_c)
end

function scale_lin_cons!(scaled_cons, scaler, lin_cons)
    scale!(scaled_cons.lb, scaler, lin_cons.lb)
    scale!(scaled_cons.ub, scaler, lin_cons.ub)
    scale_eq!(scaled_cons.A_b, scaler, lin_cons.A_b)
    scale_eq!(scaled_cons.E_c, scaler, lin_cons.E_c)
    return nothing
end

reset_lin_cons!(_b::Nothing, b::Nothing)=nothing 
function reset_lin_cons!(_b, b)
    _b .= b
    return nothing
end
function reset_lin_cons!((_A, _b)::Tuple, (A, b)::Tuple)
    _A .= A
    _b .= b
    return nothing
end
function reset_lin_cons!(scaled_cons::LinearConstraints, lin_cons::LinearConstraints)
    for fn in (:lb, :ub, :A_b, :E_c)
        reset_lin_cons!(getfield(scaled_cons, fn), getfield(lin_cons, fn))
    end
    return nothing
end
function update_lin_cons!(scaled_cons, scaler, lin_cons)
    reset_lin_cons!(scaled_cons, lin_cons)
    scale_lin_cons!(scaled_cons, scaler, lin_cons)
    return nothing
end

constraint_violation(::Nothing, type_val)=0
constraint_violation(ex::RVec, ::Val{:eq})=maximum(abs.(ex))
constraint_violation(ix::RVec, ::Val{:ineq})=maximum(ix)
function constraint_violation(hx, gx, Eres, Ares)
    return max(
        constraint_violation(hx, Val(:eq)),
        constraint_violation(Eres, Val(:eq)),
        constraint_violation(gx, Val(:ineq)),
        constraint_violation(Ares, Val(:ineq)),
        0
    )
end

function init_vals(mop, scaler, ξ0)
    T = precision(mop)
    
    ## make unscaled variables have correct type
    ξ = T.(ξ0)
    
    ## initialize scaled variables
    x = similar(ξ)
    scale!(x, scaler, ξ)

    ## pre-allocate value arrays
    fx = prealloc_objectives_vector(mop)
    hx = prealloc_nl_eq_constraints_vector(mop)
    gx = prealloc_nl_ineq_constraints_vector(mop)
    Ex = prealloc_lin_eq_constraints_vector(mop)
    Ax = prealloc_lin_ineq_constraints_vector(mop)
    Eres = deepcopy(Ex)
    Ares = deepcopy(Ax)

    ## constraint violation and filter value
    θ = Ref(zero(T))
    Φ = Ref(zero(T))

    ## set values by evaluating all functions
    mod_code = eval_mop!(fx, hx, gx, Eres, Ex, Ares, Ax, θ, Φ, mop, ξ)

    return ValueArrays(ξ, x, fx, hx, gx, Eres, Ex, Ares, Ax, θ, Φ), mod_code
end

function init_step_vals(vals)
    return StepValueArrays(vals.x, vals.fx)
end 

function init_model_vals(mod, n_vars)
    ## pre-allocate value arrays
    fx = prealloc_objectives_vector(mod)
    hx = prealloc_nl_eq_constraints_vector(mod)
    gx = prealloc_nl_ineq_constraints_vector(mod)
    Dfx = prealloc_objectives_grads(mod, n_vars)
    Dhx = prealloc_nl_eq_constraints_grads(mod, n_vars)
    Dgx = prealloc_nl_ineq_constraints_grads(mod, n_vars)
    return SurrogateValueArrays(fx, hx, gx, Dfx, Dhx, Dgx)
end

function eval_mop!(fx, hx, gx, Eres, Ex, Ares, Ax, θref, Φref, mop, ξ)
    @serve objectives!(fx, mop, ξ)
    @serve nl_eq_constraints!(hx, mop, ξ)
    @serve nl_ineq_constraints!(gx, mop, ξ)
    lin_eq_constraints!(Eres, Ex, mop, ξ)
    lin_ineq_constraints!(Ares, Ax, mop, ξ)

    θref[] = constraint_violation(hx, gx, Eres, Ares)
    Φref[] = maximum(fx) # TODO this depends on the type of filter, but atm there is only WeakFilter

    return nothing
end

function eval_mop!(vals, mop, scaler)
    ## evaluate problem at unscaled site
    @unpack ξ, x, fx, hx, gx, Eres, Ex, Ax, Ares, θ, Φ = vals
    
    unscale!(ξ, scaler, x)
    return eval_mop!(fx, hx, gx, Eres, Ex, Ares, Ax, θ, Φ, mop, ξ)
end

function compatibility_test_rhs(c_delta, c_mu, mu, Δ)
    return c_delta * min(Δ, c_mu + Δ^(1+mu))
end

function compatibility_test(n, c_delta, c_mu, mu, Δ)
    return LA.norm(n, Inf) <= compatibility_test_rhs(c_delta, c_mu, mu, Δ)
end

function compatibility_test(n, algo_opts, Δ)
    @unpack c_delta, c_mu, mu = algo_opts
    return compatibility_test(n, c_delta, c_mu, mu, Δ)
end

function accept_trial_point!(vals, vals_tmp)
    vals.θ[] = vals_tmp.θ[]
    vals.Φ[] = vals_tmp.Φ[]
    for fn in (:x, :ξ, :fx, :hx, :gx, :Ex, :Ax)
        src = getfield(vals_tmp, fn)
        isnothing(src) && continue
        copyto!(getfield(vals, fn), src)
    end
    return nothing
end

function stopping_criteria(algo_opts)
    return (
        MaxIterStopping(algo_opts.max_iter),
        MinimumRadiusStopping(algo_opts.stop_delta_min),
        ArgsRelTolStopping(;tol=algo_opts.stop_xtol_rel),
        ArgsAbsTolStopping(;tol=algo_opts.stop_xtol_abs),
        ValsRelTolStopping(;tol=algo_opts.stop_ftol_rel),
        ValsAbsTolStopping(;tol=algo_opts.stop_ftol_abs),
        CritAbsTolStopping(;
            crit_tol=algo_opts.stop_crit_tol_abs,
            theta_tol=algo_opts.stop_theta_tol_abs
        ),
        MaxCritLoopsStopping(algo_opts.stop_max_crit_loops)
    )
end

function optimize(
    MOP::AbstractMOP, ξ0::RVec;
    algo_opts :: AlgorithmOptions = AlgorithmOptions(),
    user_callback = NoUserCallback(),
)
    @assert !isempty(ξ0) "Starting point array `x0` is empty."
    @assert dim_objectives(MOP) > 0 "Objective Vector dimension of problem is zero."
    
    ## INITIALIZATION (Iteration 0)
    mop = initialize(MOP, ξ0)
    T = precision(mop)
    n_vars = length(ξ0)

    ## struct holding constant linear constraint informaton (lb, ub, A_b, E_c)
    ## set these first, because they are needed to initialize a scaler 
    ## if `algo_opts.scaler_cfg==:box`:
    lin_cons = init_lin_cons(mop)

    ## initialize a scaler according to configuration
    mod_type = model_type(mop)
    scaler = init_scaler(algo_opts.scaler_cfg, mod_type, lin_cons, n_vars)
    ## whenever the scaler changes, we have to re-scale the linear constraints
    scaled_cons = deepcopy(lin_cons)
    update_lin_cons!(scaled_cons, scaler, lin_cons)

    ## caches for working arrays x, fx, hx, gx, Ex, Ax, …
    ## (perform 1 evaluation to set values already)
    vals, vals_code = init_vals(mop, scaler, ξ0)
    !isnothing(vals_code) && return vals, GenericStopping(vals_code, algo_opts.log_level)

    vals_tmp = deepcopy(vals)
 
    ## pre-allocate surrogates `mod`
    ## (they are not trained yet)
    mod = init_models(mop, n_vars, scaler)
    
    ## chaches for surrogate value vectors fx, hx, gx, Dfx, Dhx, Dgx
    ## (values not set yet, only after training)
    mod_vals = init_model_vals(mod, n_vars)
 
    ## pre-allocate working arrays for normal and descent step calculation:
    step_vals = init_step_vals(vals)
    step_cache = init_step_cache(algo_opts.step_config, vals, mod_vals)

    crit_cache = CriticalityRoutineCache(
        deepcopy(mod_vals),
        deepcopy(step_vals),
        deepcopy(step_cache),
        _copy_model(mod)
    )

    ## initialize empty filter
    filter = WeakFilter{T}()

    ## finally, compose information about the 0-th iteration for next_iterations:
    iter_meta = init_iter_meta(T, dim_objectives(mop), algo_opts)
    
    ## prepare stopping
    stop_crits = stopping_criteria(algo_opts)
    stop_code = nothing
    while true
        !isnothing(stop_code) && break
        stop_code = do_iteration!(      # assume `copyto_model!`, otherwise, `mod, stop_code = do_iteration!...`
            iter_meta, mop, mod, scaler, lin_cons, scaled_cons, 
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, crit_cache, stop_crits, algo_opts,
            user_callback
        )
    end
    return vals, stop_code
end

function do_iteration!(
    iter_meta, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, crit_cache, stop_crits, algo_opts,
    user_callback
)
    ## assumptions at the start of an iteration:
    ## * `vals` holds valid values for the stored argument vector `x`.
    ## * If the models `mod` are valid for `vals`, then `mod_vals` are valid, too.
    Δ = iter_meta.Δ_post
    radius_has_changed = Δ != iter_meta.Δ_pre
    iter_meta.Δ_pre = Δ
    it_stat = iter_meta.it_stat_pre = iter_meta.it_stat_post
    iter_meta.it_index += 1
    @unpack it_index = iter_meta

    for stopping_criterion in stop_crits
        if check_pre_iteration(stopping_criterion)
            @serve _evaluate_stopping_criterion(
                stopping_criterion, Δ, mop, mod, scaler, lin_cons, scaled_cons, 
                vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts
            )
        end
    end
    if check_pre_iteration(user_callback)
        @serve evaluate_stopping_criterion(
            user_callback, Δ, mop, mod, scaler, lin_cons, scaled_cons, 
            vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts
        )
    end

    @logmsg algo_opts.log_level """\n
    ###########################
    #  ITERATION $(it_index).
    ###########################
    Δ = $(Δ)
    θ = $(vals.θ[])
    x = $(vec2str(vals.ξ))
    fx = $(vec2str(vals.fx))
    """

    ## The models are valid
    ## - the last iteration was a successfull restoration iteration.
    ## - if the point has not changed and the models do not depend on the radius or  
    ##   radius has not changed neither.
    models_valid = (
        it_stat == RESTORATION || 
        !iter_meta.point_has_changed && !(depends_on_radius(mod) && radius_has_changed)
    )

    if !models_valid
        @logmsg algo_opts.log_level "ITERATION $(it_index): Updating Surrogates."
        update_code = update_models!(mod, Δ, mop, scaler, vals, scaled_cons, algo_opts)
        !isnothing(update_code) && return GenericStopping(update_code, algo_opts.log_level)
        mod_code = eval_and_diff_mod!(mod_vals, mod, vals.x)
        !isnothing(mod_code) && return GenericStopping(mod_code, algo_opts.log_level)

        if vals.θ[] > 0
            @logmsg algo_opts.log_level "ITERATION $(it_index): Computing a normal step."
        end 

        compute_normal_step!(
            step_cache, step_vals.n, step_vals.xn, Δ, vals.θ[], 
            vals.ξ, vals.x, vals.fx, vals.hx, vals.gx, 
            mod_vals.fx, mod_vals.hx, mod_vals.gx, mod_vals.Dfx, mod_vals.Dhx, mod_vals.Dgx,
            vals.Eres, vals.Ex, vals.Ares, vals.Ax, scaled_cons.lb, scaled_cons.ub, 
            scaled_cons.E_c, scaled_cons.A_b, mod
        )
        
        if vals.θ[] > 0
            @logmsg algo_opts.log_level "ITERATION $(it_index): Found normal step $(vec2str(step_vals.n)). Hence xn=$(vec2str(step_vals.xn))."
        end
        
    end

    n = step_vals.n
    n_is_compatible = compatibility_test(n, algo_opts, Δ)

    if !n_is_compatible
        ## Try to do a restoration
        @logmsg algo_opts.log_level "ITERATION $(it_index): Normal step incompatible. Trying restoration."
        add_to_filter!(filter, vals.θ[], vals.Φ[])
        return do_restoration(
            mop, mod, scaler, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, iter_meta, algo_opts;
        )
    end

    @logmsg algo_opts.log_level "ITERATION $(it_index): Computing a descent step."
    cd_code = compute_descent_step!(
        step_cache, step_vals.d, step_vals.s, step_vals.xs, step_vals.fxs, step_vals.crit_ref,
        Δ, vals.θ[], vals.ξ, vals.x, step_vals.n, step_vals.xn, vals.fx, vals.hx, vals.gx, 
        mod_vals.fx, mod_vals.hx, mod_vals.gx, mod_vals.Dfx, mod_vals.Dhx, mod_vals.Dgx,
        vals.Eres, vals.Ex, vals.Ares, vals.Ax, scaled_cons.lb, scaled_cons.ub, 
        scaled_cons.E_c, scaled_cons.A_b, mod, mop, scaler
    )
    χ = step_vals.crit_ref[]
    iter_meta.crit_val = χ
    
    @logmsg algo_opts.log_level "\t Criticality χ=$(χ), ‖d‖₂=$(LA.norm(step_vals.d)), ‖s‖₂=$(LA.norm(step_vals.s))."
    !isnothing(cd_code) && return GenericStopping(cd_code, algo_opts.log_level)

    for stopping_criterion in stop_crits
        if check_post_descent_step(stopping_criterion)
            @serve _evaluate_stopping_criterion(
                stopping_criterion, Δ, mop, mod, scaler, lin_cons, scaled_cons, 
                vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts
            )
        end
    end
    if check_post_descent_step(user_callback)
        @serve evaluate_stopping_criterion(
            user_callback, Δ, mop, mod, scaler, lin_cons, scaled_cons, 
            vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts
        )
    end
    ## `χ` is the inexact criticality.
    ## For the convergence analysis to work, we also have to have the Criticality Routine:
    Δ, stop_code = criticality_routine(
        iter_meta, mod, step_vals, mod_vals, step_cache, crit_cache,
        mop, scaler, lin_cons, scaled_cons, vals, vals_tmp, stop_crits, algo_opts, 
        user_callback
    )
    !isnothing(stop_code) && return stop_code
    
    ## test if trial point is acceptable for filter and set missing meta data in `iter_meta`:
    trial_code = test_trial_point!(
        mop, scaler, iter_meta, filter, vals, mod_vals, vals_tmp, step_vals, algo_opts)
    !isnothing(trial_code) && return GenericStopping(trial_code, algo_opts.log_level)
    this_it_stat = iter_meta.it_stat_post
    
    ## process trial point test results, order of operations IMPORTANT!
    ## First, finalize `iter_meta`
    iter_meta.point_has_changed = Int(this_it_stat) > 0
    ## to this end, update trust region radius
    @unpack gamma_grow, gamma_shrink, gamma_shrink_much, delta_max = algo_opts
    _Δ = if this_it_stat == FILTER_ADD_SHRINK || this_it_stat == INACCEPTABLE
        gamma_shrink_much * Δ
    elseif this_it_stat == ACCEPTABLE || this_it_stat == FILTER_FAIL
        gamma_shrink * Δ
    elseif this_it_stat == SUCCESSFUL
        min(gamma_grow * Δ, delta_max)
    else
        Δ
    end
    iter_meta.Δ_post = _Δ

    # Now, check stopping criteria.
    # It has to happen here, so that the criteria have access to `vals` and `vals_tmp`.
    for stopping_criterion in stop_crits
        if check_post_iteration(stopping_criterion)
            @serve _evaluate_stopping_criterion(
                stopping_criterion, _Δ, mop, mod, scaler, lin_cons, scaled_cons, 
                vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts
            )
        end
    end
    if check_post_iteration(user_callback)
        @serve evaluate_stopping_criterion(
            user_callback, _Δ, mop, mod, scaler, lin_cons, scaled_cons, 
            vals, vals_tmp, step_vals, mod_vals, filter, iter_meta, step_cache, algo_opts
        )
    end
    # Finally, change filter and values
    if this_it_stat == FILTER_ADD || this_it_stat == FILTER_ADD_SHRINK
        add_to_filter!(filter, vals_tmp.θ[], vals_tmp.Φ[])
    end

    if iter_meta.point_has_changed
        ## accept trial point, mathematically ``xₖ₊₁ ← xₖ + sₖ``, pseudo-code `copyto!(vals, vals_tmp)`: 
        accept_trial_point!(vals, vals_tmp)
    end

    return nothing
end

export optimize, AlgorithmOptions

end