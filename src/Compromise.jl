module Compromise

import Parameters: @with_kw, @unpack
import JuMP
import COSMO

import NLopt 

const DEFAULT_QP_OPTIMIZER=COSMO.Optimizer
const DEFAULT_PRECISION=Float32

include("CompromiseEvaluators/CompromiseEvaluators.jl")
import .CompromiseEvaluators as CE

import LinearAlgebra as LA
 
# However, files are allowed to depend on other files for function/method definitions.
# For those, order is not relevant, but a note in the respective file would be nice.
include("types.jl")
include("utils.jl")

include("mop.jl")

include("models.jl")

include("affine_scalers.jl")

include("filter.jl")

include("descent.jl")

include("restoration.jl")

include("simple_mop.jl")

var_bounds_valid(lb, ub)=true
var_bounds_valid(lb::Nothing, ub::RVec)=!(any(isequal(-Inf), ub))
var_bounds_valid(lb::RVec, ub::Nothing)=!(any(isequal(Inf), lb))
var_bounds_valid(lb::RVec, ub::RVec)=all(lb .<= ub)

function init_scaler(scaler_cfg::Symbol, mod_type, lin_cons)
    return init_scaler(Val(scaler_cfg), mod_type, lin_cons)
end

function init_scaler(::Val{:box}, mod_type, lin_cons)
    if supports_scaling(mod_type) isa AbstractAffineScalingIndicator
        @unpack lb, ub = lin_cons
        return init_box_scaler(lb, ub)
    end
    @warn "Problem structure does not support scaling according to `scaler_cfg=:box`. Proceeding without."
    return IdentityScaler()    
end
init_scaler(::Val{:none}, mod_type, lin_cons) = IdentityScaler()

function init_lin_cons(mop)
    lb = lower_var_bounds(mop)
    ub = upper_var_bounds(mop)

    if !var_bounds_valid(lb, ub)
        error("Variable bounds unvalid.")
    end

    Ab = lin_eq_constraints(mop)
    Ec = lin_ineq_constraints(mop)

    return LinearConstraints(lb, ub, Ab, Ec)
end

function scale_lin_cons!(scaled_cons, scaler, lin_cons)
    scale!(scaled_cons.lb, scaler, lin_cons.lb)
    scale!(scaled_cons.ub, scaler, lin_cons.ub)
    scale_eq!(scaled_cons.Ab, scaler, lin_cons.Ab)
    scale_eq!(scaled_cons.Ec, scaler, lin_cons.Ec)
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
    for fn in (:lb, :ub, :Ab, :Ec)
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

    ## set values by evaluating all functions
    θ, Φ = eval_mop!(fx, hx, gx, Eres, Ex, Ares, Ax, mop, ξ)

    ## constraint violation and filter value
    θ = Ref(T(θ))
    Φ = Ref(T(Φ))

    return ValueArrays(ξ, x, fx, hx, gx, Eres, Ex, Ares, Ax, θ, Φ)
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

function eval_mop!(fx, hx, gx, Eres, Ex, Ares, Ax, mop, ξ)
    objectives!(fx, mop, ξ)
    nl_eq_constraints!(hx, mop, ξ)
    nl_ineq_constraints!(gx, mop, ξ)
    lin_eq_constraints!(Eres, Ex, mop, ξ)
    lin_ineq_constraints!(Ares, Ax, mop, ξ)

    θ = constraint_violation(hx, gx, Eres, Ares)
    Φ = maximum(fx) # TODO this depends on the type of filter, but atm there is only WeakFilter

    return θ, Φ
end

function eval_mop!(vals, mop, scaler)
    ## evaluate problem at unscaled site
    @unpack ξ, x, fx, hx, gx, Eres, Ex, Ax, Ares = vals
    
    unscale!(ξ, scaler, x)
    θ, Φ = eval_mop!(fx, hx, gx, Eres, Ex, Ares, Ax, mop, ξ)

    vals.θ[] = θ
    vals.Φ[] = Φ
    return nothing
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
    for fn in (:x, :ξ, :fx, :hx, :gx, :Ex, :Ax)
        src = getfield(vals_tmp, fn)
        isnothing(src) && continue
        copyto!(getfield(vals, fn), src)
    end
    return nothing
end

function optimize(
    MOP::AbstractMOP, ξ0::RVec;
    algo_opts :: AlgorithmOptions = AlgorithmOptions()
)
    @assert !isempty(ξ0) "Starting point array `x0` is empty."
    @assert dim_objectives(MOP) > 0 "Objective Vector dimension of problem is zero."
    
    ## INITIALIZATION (Iteration 0)
    mop = initialize(MOP, ξ0)
    T = precision(mop)
    n_vars = length(ξ0)

    ## struct holding constant linear constraint informaton (lb, ub, Ab, Ec)
    ## set these first, because they are needed to initialize a scaler 
    ## if `algo_opts.scaler_cfg==:box`:
    lin_cons = init_lin_cons(mop)

    ## initialize a scaler according to configuration
    mod_type = model_type(mop)
    scaler = init_scaler(algo_opts.scaler_cfg, mod_type, lin_cons)
    ## whenever the scaler changes, we have to re-scale the linear constraints
    scaled_cons = deepcopy(lin_cons)
    update_lin_cons!(scaled_cons, scaler, lin_cons)

    ## caches for working arrays x, fx, hx, gx, Ex, Ax, …
    ## (perform 1 evaluation to set values already)
    vals = init_vals(mop, scaler, ξ0)
    vals_tmp = deepcopy(vals)
 
    ## pre-allocate surrogates `mod`
    ## (they are not trained yet)
    mod = init_models(mop, n_vars, scaler)
    
    ## chaches for surrogate value vectors fx, hx, gx, Dfx, Dhx, Dgx
    ## (values not set yet, only after training)
    mod_vals = init_model_vals(mod, n_vars)
 
    ## pre-allocate working arrays for normal and descent step calculation:
    step_vals = init_step_vals(vals)
    step_cache = init_step_cache(SteepestDescentConfig(), vals, mod_vals)

    ## initialize empty filter
    filter = WeakFilter{T}()

    ## finally, compose information about the 0-th iteration for next_iterations:
    Δ = T(algo_opts.delta_init)
    it_index = 0
    it_stat = INITIALIZATION
    ret_code = CONTINUE
    point_has_changed = true
    radius_has_changed = true
    for outer it_index=1:algo_opts.max_iter
        ret_code != CONTINUE && break
        it_stat, ret_code, point_has_changed, Δ_new = do_iteration(
            it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
            point_has_changed, radius_has_changed
        )
        radius_has_changed = Δ_new != Δ
        Δ = Δ_new
    end
    if ret_code == CONTINUE
        ret_code = BUDGET
    end
    return vals, it_index, ret_code
end

function do_iteration(
    it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
    point_has_changed, radius_has_changed
)
    ## assumptions at the start of an iteration:
    ## * `it_res` classifies the last iteration
    ## * `vals` holds valid values for the stored argument vector `x`.
    ## * If the models `mod` are valid for `vals`, then `mod_vals` are valid, too.
    
    @info """
    ITERATION $(it_index).
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
        !point_has_changed && !(depends_on_trust_region(mod) && radius_has_changed)
    )

    if !models_valid
        update_models!(mod, Δ, mop, scaler, vals, scaled_cons, algo_opts; point_has_changed)
        eval_and_diff_mod!(mod_vals, mod, vals.x)

        compute_normal_step(
            it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts
        )
        @info "For θ=$(vals.θ[]), computed normal step of length $(LA.norm(step_vals.n))."
    end

    n = step_vals.n
    n_is_compatible = compatibility_test(n, algo_opts, Δ)

    if !n_is_compatible
        ## Try to do a restoration
        add_to_filter!(filter, vals.θ[], vals.Φ[])
        return do_restoration(
            it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
            vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts;
        )
    end

    χ = compute_descent_step(
        it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
        vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts
    )
    
    @unpack xs = step_vals
    _fxs = step_vals.fxs        # surrogate objective values at `xs`
    #src # TODO stopping based on χ
    copyto!(vals_tmp.x, xs)
    eval_mop!(vals_tmp, mop, scaler)

    trial_point_fits_filter = is_acceptable(
        filter, vals_tmp.θ[], vals_tmp.Φ[], vals.θ[], vals.Φ[])
    
    if trial_point_fits_filter
        @unpack strict_acceptance_test, kappa_theta, psi_theta, nu_accept = algo_opts
        objf_decrease, model_decrease = if strict_acceptance_test
            vals.fx .- vals_tmp.fx, mod_vals.fx .- _fxs     # TODO pre-allocate caches
        else
            maximum(vals.fx) - maximum(vals_tmp.fx), maximum(mod_vals.fx) - maximum(_fxs)
        end
        rho = minimum( objf_decrease ./ model_decrease )
        model_decrease_condition = all(model_decrease .>= kappa_theta * vals.θ[]^psi_theta)
        succifient_decrease_condition = rho >= nu_accept
    end

    this_it_stat = if !trial_point_fits_filter
        FILTER_FAIL
    else
        if model_decrease_condition
            if !succifient_decrease_condition
                INACCEPTABLE
            else
                @unpack nu_success = algo_opts
                if rho < nu_success
                    ACCEPTABLE
                else
                    SUCCESSFUL
                end
            end
        else
            if !succifient_decrease_condition
                FILTER_ADD_SHRINK
            else
                FILTER_ADD
            end
        end
    end

    if this_it_stat == FILTER_ADD || this_it_stat == FILTER_ADD_SHRINK
        add_to_filter!(filter, vals_tmp.θ[], vals_tmp.Φ[])
    end

    point_has_changed = Int(this_it_stat) > 0
    if point_has_changed
        ## accept trial point
        accept_trial_point!(vals, vals_tmp)
    end

    @unpack gamma_grow, gamma_shrink, gamma_shrink_much, delta_max = algo_opts
    _Δ = if this_it_stat == FILTER_ADD_SHRINK || this_it_stat == INACCEPTABLE
        gamma_shrink_much * Δ
    elseif this_it_stat == ACCEPTABLE
        gamma_shrink * Δ
    elseif this_it_stat == SUCCESSFUL
        min(gamma_grow * Δ, delta_max)
    else
        Δ
    end

    rcode = CONTINUE
    if χ <= algo_opts.stop_crit_tol_abs && vals.θ[] <= algo_opts.stop_theta_tol_abs
        rcode = CRITICAL
    end
    return this_it_stat, rcode, point_has_changed, _Δ
end

end