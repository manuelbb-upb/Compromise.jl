init_jump_model(qp_cfg::Type{<:MOI.AbstractOptimizer}, vals, delta) = JuMP.Model(qp_cfg)
function init_jump_model(qp_cfg::JuMP.Model, vals, delta)
    empty!(qp_cfg)
    return qp_cfg
end

abstract type AbstractJuMPModelConfig end
init_jump_model(::AbstractJuMPModelConfig, vals, delta)=nothing

function OSQPOptimizerConfig(args...; kwargs...)
    highs_ext = Base.get_extension(@__MODULE__, :OSQPStepsExt)
    if isnothing(highs_ext)
        @warn "Could not load OSQP extension."
        return nothing
    end
    return highs_ext.OSQPOptimizerConfig(args...; kwargs...)
end

Base.@kwdef struct HiGHSOptimizerConfig <: AbstractJuMPModelConfig
    silent :: Bool = true
    ## restricting the number of threads might be necessary for thread-safety in case
    ## of parallel solver runs
    num_threads :: Union{Nothing, Int} = 1
end
@batteries HiGHSOptimizerConfig selfconstructor=false

function default_qp_normal_cfg()
    return HiGHSOptimizerConfig()
end
function default_qp_descent_cfg()
    return HiGHSOptimizerConfig()
end

function init_jump_model(highs_jump_cfg::HiGHSOptimizerConfig, vals, delta)
    opt = JuMP.Model(HiGHS.Optimizer)

    if highs_jump_cfg.silent
        JuMP.set_silent(opt)
    end
    if !isnothing(highs_jump_cfg.num_threads) && highs_jump_cfg.num_threads > 0
        JuMP.set_attribute(opt, MOI.NumberOfThreads(), highs_jump_cfg.num_threads)
    end

    num_vars = dim_vars(vals)
    JuMP.set_attribute(opt, "time_limit", max(10, Float64(2*num_vars)))
    
    return opt
end

Base.@kwdef struct SteepestDescentConfig{
    F,
    qp_normal_cfgType,
    qp_descent_cfgType,
} <: AbstractStepConfig
    float_type :: Type{F} = DEFAULT_FLOAT_TYPE
    backtracking_factor :: F = 0.5
    rhs_factor :: F = 1e-3
    
    normalize_gradients :: Bool = false
    backtracking_mode :: Union{Val{:all}, Val{:any}, Val{:max}} = Val(:max)
    descent_step_norm :: F = 1
    normal_step_norm :: F = 1

    guard_backtracking :: Bool = false

    qp_normal_cfg :: qp_normal_cfgType = default_qp_normal_cfg()
    qp_descent_cfg :: qp_descent_cfgType = default_qp_descent_cfg()
end
@batteries SteepestDescentConfig selfconstructor=false

function SteepestDescentConfig(
    ::Type{float_type}, 
    backtracking_factor,
    rhs_factor,
    normalize_gradients,
    backtracking_mode,
    descent_step_norm,
    normal_step_norm,
    guard_backtracking,
    qp_normal_cfg::qp_normal_cfgType,
    qp_descent_cfg::qp_descent_cfgType,
) where {float_type, qp_normal_cfgType, qp_descent_cfgType}
    @assert 0 < backtracking_factor < 1 "`backtracking_factor` must be in (0,1)."
    @assert 0 < rhs_factor < 1 "`rhs_factor` must be in (0,1)."

    @assert normal_step_norm == 1 || normal_step_norm == 2 || normal_step_norm == Inf "`normal_step_norm` must be 1, 2 or Inf."
    @assert descent_step_norm == 1 || descent_step_norm == 2 || descent_step_norm == Inf "`descent_step_norm` must be 1, 2 or Inf."
    
    if backtracking_mode isa Symbol
        backtracking_mode = Val(backtracking_mode)
    end
    if !isa(backtracking_mode, Union{Val{:all},Val{:any},Val{:max}})
        backtracking_mode = Val(:max)
    end
    return SteepestDescentConfig{float_type, qp_normal_cfgType, qp_normal_cfgType}(
        float_type, 
        backtracking_factor,
        rhs_factor,
        normalize_gradients,
        backtracking_mode,
        descent_step_norm,
        normal_step_norm,
        guard_backtracking,
        qp_normal_cfg,
        qp_descent_cfg
    )
end

change_float_type(cfg::SteepestDescentConfig{F}, ::Type{F}) where{F} = cfg
function change_float_type(cfg::SteepestDescentConfig{F}, ::Type{new_float_type}) where {F, new_float_type}
    new_cfg = @set cfg.float_type = new_float_type
    return new_cfg
end

Base.@kwdef struct SteepestDescentCache{
    F<:AbstractFloat, 
    qp_normal_cfgType,
    qp_descent_cfgType,
} <: AbstractStepCache
    ## static information
    backtracking_factor :: F
    rhs_factor :: F
    normalize_gradients :: Bool
    backtracking_mode :: Union{Val{:all},Val{:any},Val{:max}}
    descent_step_norm :: F
    normal_step_norm :: F

    guard_backtracking :: Bool
    
    ## caches for intermediate values
    fxn :: Vector{F}
    fx_tmp :: Vector{F}

    lb_tr :: Vector{F}
    ub_tr :: Vector{F}

    ## caches for product of constraint jacobians and normal step
    Axn :: Vector{F}    # Aξ + _A*n
    Dgx_n :: Vector{F}  # g(x) + ∇g(x) * n

    ## caches for normal step backtracking
    xn :: Union{Nothing, Vector{F}}     # TODO remove because unused
    gxn :: Union{Nothing, Vector{F}}
    hxn :: Union{Nothing, Vector{F}}
    
    qp_normal_cfg :: qp_normal_cfgType = HiGHSOptimizerConfig()
    qp_descent_cfg :: qp_descent_cfgType = qp_normal_cfg
end

function init_step_cache(
    cfg::SteepestDescentConfig, vals, mod_vals
)
    fxn = copy(cached_fx(vals))
    fx_tmp = copy(fxn)
    F = eltype(fxn)
    
    backtracking_factor = F(cfg.backtracking_factor)
    rhs_factor = F(cfg.rhs_factor)

    @unpack normalize_gradients, backtracking_mode, descent_step_norm, 
        normal_step_norm, qp_normal_cfg, qp_descent_cfg, guard_backtracking = cfg

    lb_tr = array(F, dim_vars(vals))
    ub_tr = similar(lb_tr)

    Axn = array(F, dim_lin_ineq_constraints(vals))
    Dgx_n = array(F, dim_nl_ineq_constraints(vals))

    if guard_backtracking
        xn = array(F, dim_vars(vals))
        gxn = array(F, dim_nl_ineq_constraints(vals))
        hxn = array(F, dim_nl_eq_constraints(vals))
    else
        xn = gxn = hxn = nothing
    end
    return SteepestDescentCache(;
        backtracking_factor, rhs_factor, normalize_gradients, 
        backtracking_mode, descent_step_norm, normal_step_norm, 
        fxn, fx_tmp, lb_tr, ub_tr, Axn, Dgx_n, 
        qp_normal_cfg, qp_descent_cfg, 
        xn, gxn, hxn, guard_backtracking
    )
end

function compute_normal_step!(
    step_cache::SteepestDescentCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)
    x = cached_x(vals)
    Ex_min_c = cached_Ex_min_c(vals)
    Ax_min_b = cached_Ax_min_b(vals)
    @unpack A, E, lb, ub = scaled_cons
    gx = cached_gx(mod_vals)
    Dgx = cached_Dgx(mod_vals)
    hx = cached_hx(mod_vals)
    Dhx = cached_Dhx(mod_vals)

    qp_cfg = step_cache.qp_normal_cfg
    opt = init_jump_model(qp_cfg, vals, Δ)
    try
        n = solve_normal_step_problem(
            x, lb, ub, Ex_min_c, E, Ax_min_b, A, hx, Dhx, gx, Dgx, opt;
            step_norm = step_cache.normal_step_norm
        )
        Base.copyto!(step_vals.n, n)
    catch err
        @warn "Error in normal step computation." exception=err #(err, catch_backtrace())
        step_vals.n .= NaN
    end

    nothing 
end

function compute_descent_step!(
    step_cache::SteepestDescentCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)

    return compute_steepest_descent_step!(
        step_cache, step_vals, mod, scaled_cons, vals, mod_vals, Δ)
end

function compute_steepest_descent_step!(
    step_cache, step_vals, mod, scaled_cons, vals, mod_vals, Δ
)
    x = cached_x(vals)
    Dfx = cached_Dfx(mod_vals)
    @unpack n = step_vals
    if !iszero(n)
        @ignoraise diff_objectives!(Dfx, mod, x)
    end
    gx = cached_gx(mod_vals)
    Dgx = cached_Dgx(mod_vals)
    @unpack Axn, Dgx_n = step_cache
    Ax_min_b = cached_Ax_min_b(vals)
    @unpack A, E, lb, ub = scaled_cons

    postprocess_normal_step_results!(Axn, Dgx_n, n, Ax_min_b, A, Dgx, gx)

    @unpack xn = step_vals
    Dhx = cached_Dhx(mod_vals)
    @unpack normalize_gradients, qp_descent_cfg = step_cache
    opt = init_jump_model(qp_descent_cfg, vals, Δ)
    χ, _d = solve_steepest_descent_problem(
        xn, Dfx, lb, ub, E, Axn, A, Dhx, Dgx_n, Dgx, opt;
        normalize_gradients, ball_norm = step_cache.descent_step_norm
    )
    
    step_vals.d .= _d
    step_vals.crit_ref[] = χ
    return nothing
end

function finalize_step_vals!(
    step_cache::SteepestDescentCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)
    x = cached_x(vals)
    @unpack xn, fxs, d = step_vals
    @unpack A, E, lb, ub = scaled_cons
    gx = cached_gx(mod_vals)
    Dgx = cached_Dgx(mod_vals)
    @unpack lb_tr, ub_tr, Axn, Dgx_n = step_cache
    trust_region_bounds!(lb_tr, ub_tr, x, Δ, lb, ub)
    _, σ = initial_steplength(xn, d, lb_tr, ub_tr, Axn, A, gx, Dgx_n, Dgx)
    if isnan(σ) || σ <= 0
        #src #del _χ = 0
        _χ = step_vals.crit_ref[]
        d .= 0
    else
        @unpack fxn, fx_tmp, rhs_factor, backtracking_mode, backtracking_factor, gxn, hxn = step_cache
        @unpack xs = step_vals
        χ = step_vals.crit_ref[]
        @ignoraise _χ = backtrack!(
            d, xs, fxn, fxs,
            mod, xn, lb_tr, ub_tr, χ, backtracking_factor, rhs_factor, backtracking_mode,
            gxn, hxn;
            fx_tmp, σ_init=σ
        )
    end
    step_vals.crit_ref[] = _χ
  
    return nothing
end

function solve_normal_step_problem(
    x::RVec, 
    lb::Union{RVec, Nothing}, ub::Union{RVec, Nothing},
    Ex_min_c::RVec, E::Union{RMat, Nothing}, 
    Ax_min_b::RVec, A::Union{RMat, Nothing},
    hx::RVec, Dhx::RMat, gx::RVec, Dgx::RMat,
    opt;
    step_norm::Real=2
)

    n_vars = length(x)
    JuMP.@variable(opt, n[1:n_vars])

    normal_step_objective!(opt, n, step_norm)

    xn = x .+ n
    set_lower_bounds!(opt, xn, lb)
    set_upper_bounds!(opt, xn, ub)

    if !isnothing(A)
        # A(x + n) ≤ b ⇔ A(x + n) - b ≤ 0 ⇔ Ax - b + An ≤ 0
        JuMP.@constraint(opt, Ax_min_b .+ A * n .<= 0)
    end

    if !isnothing(E)
        JuMP.@constraint(opt, Ex_min_c .+ E * n .== 0)
    end

    JuMP.@constraint(opt, hx .+ Dhx'n .== 0)
    JuMP.@constraint(opt, gx .+ Dgx'n .<= 0)

    JuMP.optimize!(opt)

    if MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
        return fill(eltype(x)(NaN), length(n))
    end
   
    _n = JuMP.value.(n)
    return _n
end

function normal_step_objective!(opt, n, p)
    if p == 2
        return normal_step_objective_2!(opt, n)
    elseif p==1
        return normal_step_objective_1!(opt, n)
    elseif isinf(p)
        return normal_step_objective_inf!(opt, n)
    end
    error("`normal_step_objective!` not defined for p-norm $(p).")
end

function normal_step_objective_2!(opt, n)
    JuMP.@objective(opt, Min, sum( n.^2 ))
end

function normal_step_objective_inf!(opt, n)
    JuMP.@variable(opt, norm_n)
    JuMP.@objective(opt, Min, norm_n)
    JuMP.@constraint(opt, -norm_n .<= n)
    JuMP.@constraint(opt, n .<= norm_n)
end

function normal_step_objective_1!(opt, n)
    JuMP.@variable(opt, norm_n)
    JuMP.@objective(opt, Min, norm_n)
    JuMP.@constraint(opt, [norm_n; n] ∈ MOI.NormOneCone(length(n)+1))
end

function postprocess_normal_step_results!(
    Axn, Dgx_n,
    # not modified:
    n::RVec,
    Ax_min_b::RVec, 
    A::Union{RMat, Nothing},
    Dgx::RMat, gx::RVec
)
    Axn .= Ax_min_b
    if !isnothing(A)
        ## Axn .+= _A * n
        LA.mul!(Axn, A, n, 1, 1)
    end
    Dgx_n .= gx
    Dgx_n .+= Dgx'n
    nothing
end

function solve_steepest_descent_problem(
    xn::RVec, Dfx::RMat,
    lb::Union{RVec, Nothing}, ub::Union{RVec, Nothing},
    E::Union{RMat, Nothing},
    Axn::RVec, A::Union{RMat, Nothing},
    Dhx::RMat, Dgx_n::RVec, Dgx::RMat,
    opt, #::Type{<:MOI.AbstractOptimizer};
    ;
    ball_norm::Real,
    normalize_gradients::Bool
)
    χ = 0
    dir = zero(xn)
    try
        β, d = setup_steepest_descent_problem!(
            opt, xn, Dfx, lb, ub, E, Axn, A, Dhx, Dgx_n, Dgx; 
            ball_norm, normalize_gradients
        )
        JuMP.optimize!(opt)

        if MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
            @warn "Steepest descent problem infeasible."
            return 0, zero(xn)
        end
        #χ = abs(JuMP.value(β))
        dir .= JuMP.value.(d)
        χ = -min(0, maximum(dir'Dfx))

    catch err
        @warn "Exception in Descent Step Computation." exception=err
        χ = 0
        dir = zero(xn)
    end
    return χ, dir
end

function setup_steepest_descent_problem!(
    opt, xn, Dfx, lb, ub, E, Axn, A, Dhx, Dgx_n, Dgx; 
    ball_norm, normalize_gradients
)
    n_vars = length(xn)

    JuMP.@variable(opt, β <= 0)
    JuMP.@variable(opt, d[1:n_vars])

    JuMP.@objective(opt, Min, β)
    descent_step_unit_constraint!(opt, d, ball_norm)
    
    ## descent constraints, ∇f(x)*d ≤ ν * β
    ν = 1
    for df in eachcol(Dfx)
        if normalize_gradients
            ν = LA.norm(df)
            if iszero(ν)
                return ν, df
            end
        end
        JuMP.@constraint(opt, df'd <= ν * β)
    end

    xs = xn .+ d
    ## lb .<= x + s
    set_lower_bounds!(opt, xs, lb)
    ## x + s .<= ub
    set_upper_bounds!(opt, xs, ub)

    if !isnothing(A)
        ## Axn = Ax - b + An
        ## Axn + Ad ≤ 0 ⇔ A(x + n + d) ≤ b
        JuMP.@constraint(opt, Axn .+ A * d .<= 0)
    end
    if !isnothing(E)
        JuMP.@constraint(opt, E * d .== 0)
    end

    JuMP.@constraint(opt, Dhx'd .== 0)
    JuMP.@constraint(opt, Dgx_n .+ Dgx'd .<= 0)
    return β, d
end

@enum ARMIJO_RESULT :: UInt8 begin
    ARMIJO_TEST_PASSED
    ARMIJO_TEST_FAILED
    ARMIJO_ABBORTED
end

function backtrack!(
    d, xs, fxn, fxs, 
    mod, xn, lb_tr, ub_tr, 
    χ, backtracking_factor, rhs_factor,
    mode :: Union{Val{:all}, Val{:any}, Val{:max}},
    gxs, hxs;
    fx_tmp = copy(fxn),
    σ_init = one(eltype(xs))
)
    if χ <= 0
        d .= 0
        return 0
    end
    ## initialize stepsize `σ=1`
    σ = σ_init
    x_rel_tol = mapreduce(eps, min, xn)   # TODO make configurable ?

    d .*= σ_init
    set_to_zero = _delta_too_small(d, x_rel_tol)

    if set_to_zero
        d .= 0
        return χ
    end
    
    ## pre-compute RHS for Armijo test
    rhs = χ * rhs_factor * σ

    ## evaluate objectives at `xn` and trial point `xs`
    #src project_into_box!(xs, lb_tr, ub_tr)
    #src d .= xs .- xn
    @ignoraise objectives!(fxn, mod, xn)
    xs .= xn .+ d
    @ignoraise objectives!(fxs, mod, xs)

    fx_tol_rel = mapreduce(eps, min, fxn) 

    ## avoid re-computation of maximum for non-strict test:
    phi_xn = _armijo_Phi_xn(mode, fxn)

    ## shrink stepsize until armijo condition is fullfilled
    θ0 = _model_theta(gxs, hxs, mod, xn)
    while true 
        if σ <= 0
            set_to_zero = true
            break
        end
        θs = _model_theta(gxs, hxs, mod, xs)
        armijo_result = _armijo_condition(mode, fx_tmp, phi_xn, fxs, rhs, fx_tol_rel)
        if armijo_result == ARMIJO_ABBORTED
            set_to_zero = true
            break
        elseif armijo_result == ARMIJO_TEST_PASSED
            if θ0 > 0 || θ0 <= 0 && θs <= 0
                break
            end
        end
        
        ## shrink `σ`, `rhs` and `d` by multiplication with `backtracking_factor`
        σ *= backtracking_factor
        rhs *= backtracking_factor
        d .*= backtracking_factor
        
        set_to_zero = _delta_too_small(d, x_rel_tol)
        if set_to_zero
            break
        end

        ## reset trial point and compute objectives
        xs .= xn .+ d
        @ignoraise objectives!(fxs, mod, xs)
    end

    if set_to_zero
        d .= 0
    end
   
    return χ
end

_model_theta(gx::Nothing, hx::Nothing, mod, x)=0
function _model_theta(gx::AbstractVector, hx::AbstractVector, mod, x)
    nl_ineq_constraints!(gx, mod, x)
    nl_eq_constraints!(hx, mod, x)
    return max( maximum(gx; init=0), maximum(hx; init=0) )
end

_armijo_Phi_xn(::Val{:all}, fxn)=fxn
_armijo_Phi_xn(::Val{:any}, fxn)=fxn
_armijo_Phi_xn(::Val{:max}, fxn)=maximum(fxn)

function _armijo_condition(mode::Val{:max}, _tmp, Φxn, fxs, rhs, tol)
    tmp = Φxn - maximum(fxs)
    if _delta_too_small(tmp, tol)
        return ARMIJO_ABBORTED
    end
    if tmp >= rhs
        return ARMIJO_TEST_PASSED
    end
    return ARMIJO_TEST_FAILED
end
function _armijo_condition(mode::Union{Val{:any},Val{:all}}, tmp, Φxn, fxs, rhs, tol)
    @. tmp = Φxn - fxs
    if _delta_too_small(tmp, tol)
        return ARMIJO_ABBORTED
    end
    res = reduce(_armijo_selector(mode), tmp .>= rhs)
    if res
        return ARMIJO_TEST_PASSED
    end
    return ARMIJO_TEST_FAILED
end
_armijo_selector(::Val{:any})=|
_armijo_selector(::Val{:all})=&

_delta_too_small(xdel::Real, tol) = (abs(xdel) <= tol)
function _delta_too_small(xdel, tol)
    return all( _d -> abs(_d) <= tol, xdel)
end

function descent_step_unit_constraint!(opt, d, p)
    if p == 2
        return descent_step_unit_constraint_2!(opt, d)
    elseif p == 1
        return descent_step_unit_constraint_1!(opt, d)
    elseif isinf(p)
        return descent_step_unit_constraint_inf!(opt, d)
    end
    error("Unit Ball norm $p not supported.")
end

function descent_step_unit_constraint_inf!(opt, d)
    JuMP.@constraint(opt, -1 .<= d)
    JuMP.@constraint(opt, d .<= 1)
end

function descent_step_unit_constraint_2!(opt, d)
    JuMP.@constraint(opt, sum( d.^2 ) <= 1)
end

function descent_step_unit_constraint_1!(opt, d)
    JuMP.@variable(opt, norm_d)
    JuMP.@constraint(opt, [norm_d; d] ∈ MOI.NormOneCone(length(d) + 1))
end

# # Helpers

# Helpers to set linear constraints for a `JuMP` model.
# We need them to conveniently handle `nothing` constraints.
set_lower_bounds!(opt, x, lb::Nothing)=nothing
set_upper_bounds!(opt, x, ub::Nothing)=nothing
function set_lower_bounds!(opt, x, lb)
    for (i, li) in enumerate(lb)
        if !isinf(li)
            JuMP.@constraint(opt, li <= x[i])
        end
    end
end
function set_upper_bounds!(opt, x, ub)
    for (i, ui) in enumerate(ub)
        if !isinf(ui)
            JuMP.@constraint(opt, x[i] <= ui)
        end
    end
end

function initial_steplength(
    xn::RVec, d::RVec,
    lb_tr::RVec, ub_tr::RVec,
    Axn::RVec, A::Union{RMat, Nothing},
    gx::RVec, Dgx_n::RVec, Dgx::RMat
)

    ## By construction, `d` is in the kernel of the linear equality constraint matrix
    ## `E`, so any steplength `σ ∈ ℝ` will be compatible with these constraints.
    
    ## Same holds for the linearized nonlinear equality constraints:
    ## `Dhx'd .== 0`.
    
    T = eltype(xn)
    σ_min, σ_max = intersect_box(xn, d, lb_tr, ub_tr)
    if !isnothing(A)
        for (i, ai) = enumerate(eachrow(A))
            # Axn + σ * A*d .<= b
            σl, σr = intersect_bound(Axn[i], LA.dot(ai, d), 0, T)
            σ_min, σ_max = intersect_intervals(σ_min, σ_max, σl, σr, T)
        end
    end
    for (i, dgi) = enumerate(eachcol(Dgx))
        # gx + Dgx_n + σ * Dgx * d .<= 0
        #src #del σl, σr = intersect_bound(@show(gx[i] + Dgx_n[i]), LA.dot(dgi, d), 0, T)
        σl, σr = intersect_bound(Dgx_n[i], LA.dot(dgi, d), 0, T)
        σ_min, σ_max = intersect_intervals(σ_min, σ_max, σl, σr, T)
    end
    return σ_min, σ_max
end

"""
    trust_region_bounds!(lb, ub, x, Δ)

Make `lb` the lower left corner of a trust region hypercube with 
radius `Δ` and make `ub` the upper right corner."""
function trust_region_bounds!(lb, ub, x, Δ)
    lb .= x .- Δ 
    ub .= x .+ Δ
    return nothing
end
"""
    trust_region_bounds!(lb, ub, x, Δ, global_lb, global_ub)

Make `lb` the lower left corner of a trust region hypercube with 
radius `Δ` and make `ub` the upper right corner.
`global_lb` and `global_ub` are the global bound vectors or `nothing`."""
function trust_region_bounds!(lb, ub, x, Δ, global_lb, global_ub)
    trust_region_bounds!(lb, ub, x, Δ)
    intersect_lower_bounds!(lb, global_lb)
    intersect_upper_bounds!(ub, global_ub)
    return nothing
end    
## helpers to deal with `nothing` bounds
intersect_lower_bounds!(lb, _lb)=map!(max, lb, lb, _lb)
intersect_lower_bounds!(lb, ::Nothing)=nothing
intersect_upper_bounds!(ub, _ub)=map!(min, ub, ub, _ub)
intersect_upper_bounds!(ub, ::Nothing)=nothing

# The functions below are used to find a suitable scaling factor 
# for new directions by intersecting the ray `x + σ * z`
# with local boundary vectors.

"""
    intersect_box(x, z, lb, ub)

Given vectors `x`, `z`, `lb` and `ub`, compute and return the largest 
interval `I` (a tuple with 2 elements) such that 
`lb .<= x .+ σ .* z .<= ub` is true for all `σ` in `I`. 
If the constraints are not feasible, `(NaN, NaN)` is returned.
If the direction `z` is zero, the interval could contain infinite elements.
"""
function intersect_box(x::X, z::Z, lb::L, ub::U) where {X, Z, L, U}
    _T = Base.promote_eltype(X, Z, L, U)
    T = _T <: AbstractFloat ? _T : DEFAULT_FLOAT_TYPE

    σ_min, σ_max = T(-Inf), T(Inf)
    for (xi, zi, lbi, ubi) = zip(x, z, lb, ub)    
        ## x + σ * z <= ub
        σl, σr = intersect_bound(xi, zi, ubi, T)
        σ_min, σ_max = intersect_intervals(σ_min, σ_max, σl, σr)
        ## lb <= x + σ * z ⇔ -x + σ * (-z) <= -lb
        σl, σr = intersect_bound(-xi, -zi, -lbi, T)
        σ_min, σ_max = intersect_intervals(σ_min, σ_max, σl, σr)
    end
    σ_min, σ_max
end

"""
    intersect_bound(xi, zi, bi)

Given number `xi`, `zi` and `bi`, compute and return an interval 
`I` (a tuple with 2 elements) such that `xi + σ * zi <= bi` is true
for all `σ` in `I`. 
If the constraint is feasible, at least one of the interval elements is infinite.
If it is infeasible, `(NaN, NaN)` is returned.
"""
function intersect_bound(
    xi::X, zi::Z, bi::B, T=Base.promote_type(X, Z, B)
) where {X<:Number, Z<:Number, B<:Number}
    ## xi + σ * zi <= bi ⇔ σ * zi <= bi - xi == ri
    ri = bi - xi
    if iszero(zi)
        if ri < 0
            return T(NaN), T(NaN)
        else
            return T(-Inf), T(Inf)
        end
    elseif zi < 0
        return ri/zi, T(Inf)
    else
        return T(-Inf), ri/zi
    end
end

"Helper to intersect to intervals."
function intersect_intervals(l1, r1, l2, r2, T=typeof(l1))
    (isnan(l1) || isnan(r1)) && return l1, r1
    (isnan(l2) || isnan(r2)) && return l2, r2
    l = max(l1, l2)
    r = min(r1, r2)
    if l > r
        return T(NaN), T(NaN)
    end
    return l, r
end