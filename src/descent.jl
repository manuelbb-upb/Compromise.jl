const MOI = JuMP.MOI

function compute_normal_step!(
    ## modify these: 
    step_cache::SC, n, xn,
    ## assume `Δ` to be the current trust-region radius, θ to be constraint violation
    Δ, θ,
    ## use these arrays to set up linear or quadratic sub-problems:
    ξ, x,                   ## scaled and unscaled iteration site
    fx, hx, gx,             ## true objective, nonlinear equality and inequality constraint vectors (or nothing)
    mod_fx, mod_hx, mod_gx, ## surrogate objective and constraint values
    mod_Dfx, mod_Dhx, mod_Dgx, ## surrogate objective and constraint Jacobians
    Eres, Ex, Ares, Ax,     ## scaled linear constraint residuals and lhs products
    lb, ub, E_c, A_b,          ## scaled linear constraints
    ## use `mod` to evaluate the surrogates in the scaled domain,
    mod,
    ) where SC<:AbstractStepCache
    error("`compute_normal_step!` not defined for step cache of type $(SC).")
end

function compute_descent_step!(
    ## modify these:
    step_cache::SC, 
    d, s, xs, mod_fxs,   ## descent direction, step vector, trial point, trial point surrogate values
    crit_ref,               ## reference to criticality value
    ## assume `Δ` to be the current trust-region radius, θ to be constraint violation
    Δ, θ,
    ## use these arrays to set up linear or quadratic sub-problems:
    ξ, x,                   ## scaled and unscaled iteration site
    n, xn,                  ## normal step and temporary trial point
    fx, hx, gx,             ## true objective, nonlinear equality and inequality constraint vectors (or nothing)
    mod_fx, mod_hx, mod_gx, ## surrogate objective and constraint values
    mod_Dfx, mod_Dhx, mod_Dgx, ## surrogate objective and constraint Jacobians
    Eres, Ex, Ares, Ax,     ## scaled linear constraint residuals and lhs products
    lb, ub, E_c, A_b,          ## scaled linear constraints
    ## use `mod` to evaluate the surrogates in the scaled domain,
    mod,
    ## use `mop` and `scaler` to evaluate the problem in the unscaled domain
    mop, scaler
) where SC<:AbstractStepCache
    error("`compute_descent_step!` not defined for step cache of type $(SC).")
end
#=
"""
    compute_normal_step(
        it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
        vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts
    )

* Modify `step_vals.n` to hold the normal step at `vals.x`.
* Modify `step_vals.xn` to hold `vals.x + step_vals.n`.
* The normal step is computed according to `step_cache` and this cache can also be 
  modified.
* Usually, the normal step is computed using the surrogate values stored in `mod_vals`.  
  These should not be modified!
"""
function compute_normal_step(
    Δ, # radius to compute the normal step for, might be different from what is in `iter_meta`
    # objects to evaluate objectives, models and constraints
    mop, mod, scaler, lin_cons, scaled_cons,
    # caches for necessary arrays
    vals, vals_tmp, step_vals, mod_vals, 
    # other important building blocks
    filter,     # the filter used to drive feasibility
    iter_meta,  # iteration information
    step_cache::SC, # an object defining step calculation and holding caches
    algo_opts;  # general algorithmic settings
) where SC <: AbstractStepCache
    return error("`compute_normal_step` not defined for step cache of type $(SC).")
end
=#
#=
"""
    compute_descent_step(
        it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
        vals, vals_tmp, step_vals, mod_vals, filter, step_cache, algo_opts
    ) :: Real

* Return a criticalty value `χ`.
* Modify `step_vals.d` to hold the scaled descent step at `step_vals.xn`.
* Modify `step_vals.xs` to hold `vals.x + step_vals.n + step_vals.d`, 
  or equivalently, `step_vals.xn + step_vals.d`.
* Modify `step_vals.fxs` to hold the surrogate objective values at `step_vals.xs`.
* The descent step is computed according to `step_cache` and this cache can also be 
  modified.
* Usually, the descent step is computed using the surrogate values stored in `mod_vals`.  
  These should not be modified!
"""
function compute_descent_step(
    Δ, # radius to compute the normal step for, might be different from what is in `iter_meta`
    # objects to evaluate objectives, models and constraints
    mop, mod, scaler, lin_cons, scaled_cons,
    # caches for necessary arrays
    vals, vals_tmp, step_vals, mod_vals, 
    # other important building blocks
    filter,     # the filter used to drive feasibility
    iter_meta,  # iteration information
    step_cache::SC, # an object defining step calculation and holding caches
    algo_opts;  # general algorithmic settings
) where SC <: AbstractStepCache
    return error("`compute_descent_step` not defined for step cache of type $(SC).")
end
=#
@with_kw struct SteepestDescentConfig{
    BT<:Real, RT<:Real, DN<:Real, NN<:Real, QPOPT
} <: AbstractStepConfig
    
    backtracking_factor :: BT = 1//2
    rhs_factor :: RT = DEFAULT_PRECISION(0.001)
    normalize_gradients :: Bool = false
    strict_backtracking :: Bool = true    
    
    descent_step_norm :: DN = Inf
    normal_step_norm :: NN = 2

    qp_opt :: QPOPT = DEFAULT_QP_OPTIMIZER

    @assert 0 < backtracking_factor < 1 "`backtracking_factor` must be in (0,1)."
    @assert 0 < rhs_factor < 1 "`rhs_factor` must be in (0,1)."
    @assert descent_step_norm == Inf    # TODO enable other norms
    @assert normal_step_norm == 2 || normal_step_norm == Inf      # TODO enable other norms
end

@batteries SteepestDescentConfig selfconstructor=false

Base.@kwdef struct SteepestDescentCache{
    T, DN, NN, EN, AN, HN, GN, QPOPT} <: AbstractStepCache
    ## static information
    backtracking_factor :: T
    rhs_factor :: T
    normalize_gradients :: Bool
    strict_backtracking :: Bool
    descent_step_norm :: DN
    normal_step_norm :: NN
    
    ## caches for intermediate values
    fxn :: Vector{T}

    ## caches for product of constraint jacobians and normal step
    En :: EN        # E*n
    An :: AN        # A*n
    Hn :: HN        # ∇h(x)*n
    Gn :: GN        # ∇g(x)*n

    qp_opt :: QPOPT
end

Base.copyto!(sc_trgt::AbstractStepCache, sc_src::AbstractStepCache)=error("Cannot copy descent caches.")
function Base.copyto!(sc_trgt::SteepestDescentCache, sc_src::SteepestDescentCache)
    for fn in (:backtracking_factor, :rhs_factor, :normalize_gradients, :strict_backtracking,
        :descent_step_norm, :normal_step_norm, :qp_opt)
        @assert getfield(sc_trgt, fn) == getfield(sc_src, fn)
    end
    for fn in (:fxn, :En, :An, :Hn, :Gn)
        trgt_fn = getfield(sc_trgt, fn)
        if !isnothing(trgt_fn)
            copyto!(trgt_fn, getfield(sc_src, fn))
        end
    end
    return nothing
end

undef_or_nothing(::Nothing, T=nothing)=nothing
function undef_or_nothing(arr::AbstractVector{X}, T=X) where X
    return Vector{T}(undef, length(arr))
end
function undef_or_nothing(arr::AbstractMatrix{X}, T=X) where X
    return Matrix{T}(undef, size(arr))
end
similar_or_nothing(::Nothing)=nothing
similar_or_nothing(arr)=similar(arr)
function init_step_cache(
    cfg::SteepestDescentConfig, vals, mod_vals
)
    T = eltype(vals)

    backtracking_factor = T(cfg.backtracking_factor)
    rhs_factor = T(cfg.rhs_factor)

    @unpack normalize_gradients, strict_backtracking, descent_step_norm, 
        normal_step_norm = cfg

    fxn = undef_or_nothing(vals.fx, T)
    En = undef_or_nothing(vals.Ex, T)
    An = undef_or_nothing(vals.Ax, T)
    Hn = undef_or_nothing(mod_vals.hx, T)    
    Gn = undef_or_nothing(mod_vals.gx, T)

    return SteepestDescentCache(;
        backtracking_factor, rhs_factor, normalize_gradients, 
        strict_backtracking, descent_step_norm, normal_step_norm, 
        fxn, En, An, Hn, Gn, cfg.qp_opt
    )
end

# Helpers to set linear constraints for a `JuMP` model.
# We need them to easily handle `nothing` constraints.
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

"""
    set_linear_constraints!(opt, affine_vec, var_vec, mat, rhs, ctype)

Add linear (in-)equality constraints to JuMP model `opt`.
* If `ctype` is `:eq`, then the constraints read
  `affine_vec + mat * var_vec .== rhs`.
* Otherwise, they read
  `affine_vec + mat * var_vec .<= rhs`.

This helper function returns a JuMP expression for the 
matrix-vector-product `mat*var_vec`.
"""
function set_linear_constraints!(opt, affine_vec, var_vec, mat, rhs, ctype::Symbol)
    lin_expr = JuMP.@expression(opt, vec(var_vec'mat))
    if ctype == :eq 
        JuMP.@constraint(opt, affine_vec .+ lin_expr .== rhs)
    else
        JuMP.@constraint(opt, affine_vec .+ lin_expr .<= rhs)
    end
    return lin_expr
end

# Helper for `LinearConstraints` if there aren't actually any linear constraints:
function set_linear_constraints!(opt, affine_vec, var_vec, cons::Nothing, ctype::Symbol)
    return nothing
end
# Helper for `LinearConstraints` if there *are* linear constraints:
function set_linear_constraints!(opt, affine_vec, var_vec, (mat, rhs)::Tuple, ctype::Symbol)
    return set_linear_constraints!(opt, affine_vec, var_vec, mat, rhs, ctype)
end
# Helper for surrogate models if there are no constraints stored
# (if `mat` is nothing, then `affine_vec` is likely `nothing` as well)
# (`rhs` might be set to something that is not `nothing` by hand):
function set_linear_constraints!(opt, affine_vec, var_vec, mat::Nothing, rhs, ctype::Symbol)
    return nothing
end

read_linear_constraint_expression!(trgt_arr::Nothing, src_ex::Nothing)=nothing
function read_linear_constraint_expression!(trgt_arr, src_ex)
    trgt_arr .= JuMP.value.(src_ex)
    return nothing
end

raw"""
Using `qp_opt`, solve
```math
    \min_{n ∈ ℝⁿ} ‖n‖ₚ
```
subject to the following constraints:
* if `lb` is not `nothing`, then ``lb ≤ x + n``,
* if `ub` is not `nothing`, then ``x + n ≤ ub``,
* if `E_c` is not `nothing`, but a matrix-vector-tuple, 
  and `Ex` is not nothing, but a vector, ``Ex + E*n = c``,
* if `A_b` is not `nothing`, but a matrix-vector-tuple, 
  and `Ax` is not nothing, but a vector, ``Ax + A*n ≤ b``,
* if `Dhx` is not `nothing`, but a matrix, ``h(x) + ∇h(x)*n = 0``,
* if `Dgx` is not `nothing`, but a matrix, ``g(x) + ∇g(x)*n ≤ 0``.
```
"""
function solve_normal_step_problem!(
    En, An, Hn, Gn,
    qp_opt, 
    x, lb, ub, 
    Ex, E_c, Ax, A_b,
    hx, Dhx, gx, Dgx;
    step_norm = 2
)
    n_vars = length(x)
        
    opt = JuMP.Model(qp_opt)
    JuMP.set_silent(opt)
    #src JuMP.set_optimizer_attribute(itrn, "polish", true)

    JuMP.@variable(opt, n[1:n_vars])
    if step_norm == 2
        JuMP.@objective(opt, Min, sum(n.^2))
    elseif step_norm == Inf
        JuMP.@variable(opt, norm_n)
        JuMP.@objective(opt, Min, norm_n)
        JuMP.@constraint(opt, -norm_n .<= n)
        JuMP.@constraint(opt, n .<= norm_n)
    end

    xn = x .+ n

    ## lb .<= x + n .<= ub
    set_lower_bounds!(opt, xn, lb)  
    set_upper_bounds!(opt, xn, ub)

    ## E * (x + n) .== c ⇔ Ex + A*n .== c
    En_ex = set_linear_constraints!(opt, Ex, n, E_c, :eq)
    ## A * (x + n) .<= b
    An_ex = set_linear_constraints!(opt, Ax, n, A_b, :ineq)

    ## hx + ∇h(x) * n .== 0
    Hn_ex = set_linear_constraints!(opt, hx, n, Dhx, 0, :eq)

    ## gx + ∇g(x) * n .<= 0
    Gn_ex = set_linear_constraints!(opt, gx, n, Dgx, 0, :ineq)

    JuMP.optimize!(opt)

    if MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
        return fill(eltype(x)(NaN), length(n))
    end

    ## read values from inner problem expressions
    read_linear_constraint_expression!(En, En_ex)
    read_linear_constraint_expression!(An, An_ex)
    read_linear_constraint_expression!(Hn, Hn_ex)
    read_linear_constraint_expression!(Gn, Gn_ex)

    return JuMP.value.(n)
end

# In `step_cache`, 
# * modify `En` and `An` to hold the vectors `E*n` and `A*n`, for the scaled linear 
#   constraint matrices,
# * modify `Hn` and `Gn` to hold the vectors `∇h(x)*n` and `∇g(x)*n` for the 
#   hessians of the modelled constraint functions
function compute_normal_step!(
    step_cache::SteepestDescentCache, n, xn, Δ, θ, ξ, x, fx, hx, gx, 
    mod_fx, mod_hx, mod_gx, mod_Dfx, mod_Dhx, mod_Dgx, 
    Eres, Ex, Ares, Ax, lb, ub, E_c, A_b, mod
)
    ## initialize assuming a zero step
    n .= 0
    copyto!(xn, x)
    if θ > 0
        n .= solve_normal_step_problem!(
            step_cache.En, step_cache.An, step_cache.Hn, step_cache.Gn, step_cache.qp_opt,
            x, lb, ub, Ex, E_c, Ax, A_b, mod_hx, mod_Dhx, mod_gx, mod_Dgx;
            step_norm=step_cache.normal_step_norm
        )
        xn .+= n 
    end
end
#=
function compute_normal_step(
    Δ, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, iter_meta,
    step_cache::SteepestDescentCache, algo_opts
)
    @unpack x = vals
    @unpack xn, n = step_vals
    copyto!(xn, x)
    if vals.θ[] > 0
        @logmsg algo_opts.log_level "ITERATION $(iter_meta.it_index): θ=$(vals.θ[]), computing normal step."

        @unpack Ax, Ex = vals
        @unpack lb, ub, A_b, E_c = scaled_cons
        @unpack hx, gx, Dhx, Dgx = mod_vals
        @unpack En, An, Hn, Gn = step_cache
        n .= solve_normal_step_problem(
            x, algo_opts.qp_opt, lb, ub, 
            Ex, A_b, En,
            Ax, E_c, An, 
            hx, Dhx, Hn, gx, Dgx, Gn
        )
        @views xn .+= n

        @logmsg algo_opts.log_level "\tComputed normal step of length $(LA.norm(n))."
    else
        fill!(n, 0)
    end 
    copyto!(step_vals.s, n)

    return nothing
end
=#

function solve_steepest_descent_problem(
    Δ, xn, Dfx, qp_opt, lb, ub,
    Exn, E_c, Axn, A_b, Hxn, Dhx, Gxn, Dgx;
    descent_step_norm, normalize_gradients
)
    n_vars = length(xn)

    opt = JuMP.Model(qp_opt)
    JuMP.set_silent(opt)

    JuMP.@variable(opt, β)
    JuMP.@variable(opt, d[1:n_vars])

    JuMP.@objective(opt, Min, β)
    if descent_step_norm == Inf
        JuMP.@constraint(opt, -Δ .<= d)
        JuMP.@constraint(opt, d .<= Δ)
    end

    ## descent constraints, ∇f(x)*d ≤ β
    for df in eachcol(Dfx)
        ndf = normalize_gradients ? LA.norm(df) : 1
        JuMP.@constraint(opt, df'd <= ndf * β)
    end

    xs = xn .+ d
    ## lb .<= x + s
    set_lower_bounds!(opt, xs, lb)
    ## x + s .<= ub
    set_upper_bounds!(opt, xs, ub)

    ## A(x + n + d) ≤ b ⇔ A(x+n) + A*d <= b
    ## ⇒ c = A(x+n)
    set_linear_constraints!(opt, Exn, d, E_c, :eq)
    set_linear_constraints!(opt, Axn, d, A_b, :ineq)

    ## hx + H(n + d) = 0 ⇔ (hx + Hn) + Hd = 0
    set_linear_constraints!(opt, Hxn, d, Dhx, 0, :eq)
    set_linear_constraints!(opt, Gxn, d, Dgx, 0, :ineq)

    JuMP.optimize!(opt)
    _d = JuMP.value.(d)  # this allocation should be negligible
    #src @show d_norm = LA.norm(_d, descent_step_norm)
    #srcχ = iszero(d_norm) ? d_norm : @show(-JuMP.value(β))/d_norm
    χ = abs(JuMP.value(β)) / Δ
    return χ, _d
end

vec_sum!(::Nothing, ::Nothing)=nothing
function vec_sum!(a, b)
    @views a .+= b
    return nothing
end

function compute_descent_step!(
    step_cache::SteepestDescentCache, d, s, xs, mod_fxs, crit_ref,
    Δ, θ, ξ, x, n, xn, fx, hx, gx, mod_fx, mod_hx, mod_gx, mod_Dfx, mod_Dhx, mod_Dgx,
    Eres, Ex, Ares, Ax, lb, ub, E_c, A_b, mod, mop, scaler
)
    @unpack En, An, Hn, Gn, normalize_gradients, descent_step_norm = step_cache;

    ## for constraints like `A*x + A*n * A*d`, make `An` hold `A*x+A*n`.
    vec_sum!(En, Ex)
    vec_sum!(An, Ax)
    ## for constraints like `h(x) + ∇h(x)*(n + d)`, make `Hn` hold `h(x)+∇h(x)*n`.
    vec_sum!(Hn, mod_hx)
    vec_sum!(Gn, mod_gx)
    χ, _d = solve_steepest_descent_problem(
        Δ, xn, mod_Dfx, step_cache.qp_opt, lb, ub, 
        En, E_c, An, A_b, Hn, mod_Dhx, Gn, mod_Dgx;
        descent_step_norm, normalize_gradients
    )
    crit_ref[] = χ
    copyto!(s, n)
    ## set `d`, scale it in place, and set `xs` and `mod_fxs` in backtracking
    copyto!(d, _d)
    @unpack fxn, backtracking_factor, rhs_factor, strict_backtracking = step_cache;
    @ignoraise χ = backtrack!(d, mod, xn, xs, fxn, mod_fxs, lb, ub, χ, Δ, backtracking_factor, rhs_factor, Val(strict_backtracking))
    crit_ref[] = χ
    @. s += d
    return nothing
end

function backtrack!(
    d, mod, xn, xs, fxn, fxs, lb, ub, χ, Δ, backtracking_factor, rhs_factor, strict_backtracking_val :: Val{strict_backtracking}
) where strict_backtracking
    if χ <= 0
        d .*= 0
        return 0
    end
    ## initialize stepsize `σ=1`
    T = eltype(d)
    σ = one(T)
    σ_min = nextfloat(zero(T), 2)   # TODO make configurable

    ## pre-compute RHS for Armijo test
    rhs = χ * rhs_factor * Δ

    ## evaluate objectives at `xn` and trial point `xs`
    xs .= xn .+ d
    project_into_box!(xs, lb, ub)
    d .= xs .- xn
    @ignoraise objectives!(fxn, mod, xn)
    @ignoraise objectives!(fxs, mod, xs)

    ## avoid re-computation of maximum for non-strict test:
    _fxn = strict_backtracking ? fxn : maximum(fxn)

    ## Until the `armijo_condition` is fullfilled
    while σ > σ_min && !armijo_condition(strict_backtracking_val, _fxn, fxs, rhs)
        ## shrink `σ`, `rhs` and `d` by multiplication with `backtracking_factor`
        σ *= backtracking_factor
        rhs *= backtracking_factor
        d .*= backtracking_factor

        ## reset trial point and compute objectives
        xs .= xn .+ d       # d = xs - xn
        project_into_box!(xs, lb, ub)
        d .= xs .- xn
        @ignoraise objectives!(fxs, mod, xs)
    end
    _χ = σ < σ_min ? 0 : χ
    return _χ
end

armijo_condition(strict::Val{true}, fxn, fxs, rhs) = all(fxn .-  fxs .>= rhs)
armijo_condition(strict::Val{false}, fxn, fxs, rhs) = fxn - maximum(fxs) >= rhs
