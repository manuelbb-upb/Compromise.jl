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
    # iteration information
    it_index, Δ, it_stat,
    # objects to evaluate objectives, models and constraints
    mop, mod, scaler, lin_cons, scaled_cons,
    # caches for necessary arrays
    vals, vals_tmp, step_vals, mod_vals, 
    # other important building blocks
    filter,     # the filter used to drive feasibility
    step_cache::SC, # an object defining step calculation and holding caches
    algo_opts;  # general algorithmic settings
) where SC <: AbstractStepCache
    return error("`compute_normal_step` not defined for step cache of type $(SC).")
end

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
    # iteration information
    it_index, Δ, it_stat,
    # objects to evaluate objectives, models and constraints
    mop, mod, scaler, lin_cons, scaled_cons,
    # caches for necessary arrays
    vals, vals_tmp, step_vals, mod_vals, 
    # other important building blocks
    filter,     # the filter used to drive feasibility
    step_cache::SC, # an object defining step calculation and holding caches
    algo_opts;  # general algorithmic settings
) where SC <: AbstractStepCache
    return error("`compute_descent_step` not defined for step cache of type $(SC).")
end

@with_kw struct SteepestDescentConfig{
    BT<:Real, RT<:Real, DN<:Real, NN<:Real
} <: AbstractStepConfig
    
    backtracking_factor :: BT = 1//2
    rhs_factor :: RT = DEFAULT_PRECISION(0.001)
    normalize_gradients :: Bool = false
    strict_backtracking :: Bool = true    
    
    descent_step_norm :: DN = Inf
    normal_step_norm :: NN = 2

    @assert 0 < backtracking_factor < 1 "`backtracking_factor` must be in (0,1)."
    @assert 0 < rhs_factor < 1 "`rhs_factor` must be in (0,1)."
    @assert descent_step_norm == Inf    # TODO enable other norms
    @assert normal_step_norm == 2 || normal_step_norm == Inf      # TODO enable other norms
end

Base.@kwdef struct SteepestDescentCache{T, DN, NN, EN, AN, HN, GN} <: AbstractStepCache
    ## static information
    backtracking_factor :: T
    rhs_factor :: T
    normalize_gradients :: Bool
    strict_backtracking :: Bool
    descent_step_norm :: DN
    normal_step_norm :: NN
    
    ## caches for intermediate values
    fxn :: Vector{T}

    ## caches for constraint jacobians times normal step
    En :: EN        # E*n
    An :: AN        # A*n
    Hn :: HN        # ∇h(x)*n
    Gn :: GN        # ∇g(x)*n
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
        fxn, En, An, Hn, Gn
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

"Add the constraint `c + A * x .?= b` to opt and return a JuMP expression for `A*x`."
function set_linear_constraints!(opt, c, x, A, b, ctype::Symbol)
    Ax_expr = JuMP.@expression(opt, vec(x'A))
    if ctype == :eq 
        JuMP.@constraint(opt, c .+ Ax_expr .== b)
    else
        JuMP.@constraint(opt, c .+ Ax_expr .<= b)
    end
    return Ax_expr
end

# Helper for `LinearConstraints` if there are no linear constraints stored:
function set_linear_constraints!(opt, c, x, Ab::Nothing, ctype::Symbol)
    return nothing
end
# Helper for `LinearConstraints` if there *are* linear constraints:
function set_linear_constraints!(opt, c, x, (A, b)::Tuple, ctype::Symbol)
    return set_linear_constraints!(opt, c, x, A, b, ctype)
end
# Helper for surrogate models if there are no constraints stored
# (`b` might be set to something that is not `nothing` by hand):
function set_linear_constraints!(opt, c, x, A::Nothing, b, ctype::Symbol)
    return nothing
end

read_linear_constraint_expression!(trgt_arr::Nothing, src_ex::Nothing)=nothing
function read_linear_constraint_expression!(trgt_arr, src_ex)
    trgt_arr .= JuMP.value.(src_ex)
    return nothing
end

function solve_normal_step_problem(
    x, qp_opt, lb, ub, 
    Ex, Ax, Ab, 
    Ec, En, An, 
    hx, Dhx, Hn, 
    gx, Dgx, Gn
)
    n_vars = length(x)
        
    opt = JuMP.Model( qp_opt )
    JuMP.set_silent(opt)
    #src JuMP.set_optimizer_attribute(itrn, "polish", true)

    JuMP.@variable(opt, n[1:n_vars])
    if normal_cache.normal_step_norm == 2
        JuMP.@objective(opt, Min, sum(n.^2))
    elseif normal_cache.normal_step_norm == Inf
        JuMP.@variable(opt, norm_n)
        JuMP.@objective(opt, Min, norm_n)
        JuMP.@constraint(opt, -norm_n .<= n)
        JuMP.@constraint(opt, n .<= norm_n)
    end

    xn = x .+ n

    ## lb .<= x + n .<= ub
    set_lower_bounds!(opt, xn, lb)  
    set_upper_bounds!(opt, xn, ub)

    ## A * (x + n) .== b ⇔ Ax + A*n .== b
    En_ex = set_linear_constraints!(opt, Ex, n, Ab, :eq)
    ## A * (x + n) .<= b
    An_ex = set_linear_constraints!(opt, Ax, n, Ec, :ineq)

    ## hx + ∇h(x) * n .== 0
    Hn_ex = set_linear_constraints!(opt, hx, n, Dhx, 0, :eq)

    ## gx + ∇g(x) * n .<= 0
    Gn_ex = set_linear_constraints!(opt, gx, n, Dgx, 0, :ineq)

    JuMP.optimize!(opt)

    ## read values from inner problem expressions
    read_linear_constraint_expression!(En, En_ex)
    read_linear_constraint_expression!(An, An_ex)
    read_linear_constraint_expression!(Hn, Hn_ex)
    read_linear_constraint_expression!(Gn, Gn_ex)

    return JuMP.value.(n)
end

# Set `step_vals.n` and `step_vals.xn`.
# In `step_cache`, 
# * modify `En` and `An` to hold the vectors `E*n` and `A*n`, for the scaled linear 
#   constraint matrices,
# * modify `Hn` and `Gn` to hold the vectors `∇h(x)*n` and `∇g(x)*n` for the 
#   hessians of the modelled constraint functions
function compute_normal_step(
    it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache::SteepestDescentCache, algo_opts
)
    @unpack x = vals
    @unpack xn, n = step_vals
    copyto!(xn, x)
    if vals.θ[] > 0
        @unpack Ax, Ex = vals
        @unpack lb, ub, Ab, Ec = scaled_cons
        @unpack hx, gx, Dhx, Dgx = mod_vals
        @unpack En, An, Hn, Gn = step_cache
        n .= solve_normal_step_problem(
            x, algo_opts.qp_opt, lb, ub, 
            Ex, Ab, En,
            Ax, Ec, An, 
            hx, Dhx, Hn, gx, Dgx, Gn
        )
        @views xn .+= n
    else
        fill!(n, 0)
    end 

    return nothing
end

function solve_steepest_descent_problem(
    Δ, xn, Dfx, qp_opt, lb, ub,
    Exn, Ab, Axn, Ec, Hxn, Dhx, Gxn, Dgx;
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
    set_linear_constraints!(opt, Exn, d, Ab, :eq)
    set_linear_constraints!(opt, Axn, d, Ec, :ineq)

    ## hx + H(n + d) = 0 ⇔ (hx + Hn) + Hd = 0
    set_linear_constraints!(opt, Hxn, d, Dhx, 0, :eq)
    set_linear_constraints!(opt, Gxn, d, Dgx, 0, :ineq)

    JuMP.optimize!(opt)
    _d = JuMP.value.(d)  # this allocation should be negligible
    #src @show d_norm = LA.norm(_d, descent_step_norm)
    # srcχ = iszero(d_norm) ? d_norm : @show(-JuMP.value(β))/d_norm
    χ = abs(JuMP.value(β) * LA.norm(_d, descent_step_norm))
    return χ, _d
end

vec_sum!(::Nothing, ::Nothing)=nothing
function vec_sum!(a, b)
    @views a .+= b
    return nothing
end

## set `step_vals.d`, `step_vals.xs` and `step_vals.fxs`
## modify `step_cache` as needed.
function compute_descent_step(
    it_index, Δ, it_stat, mop, mod, scaler, lin_cons, scaled_cons,
    vals, vals_tmp, step_vals, mod_vals, filter, step_cache::SteepestDescentCache, algo_opts
) 
    @unpack xn, d = step_vals
    @unpack lb, ub, Ab, Ec = scaled_cons
    @unpack Ex, Ax = vals
    @unpack fx, hx, gx, Dfx, Dhx, Dgx = mod_vals
    @unpack En, An, Hn, Gn, normalize_gradients, descent_step_norm = step_cache;

    ## for constraints like `A*x + A*n * A*d`, make `An` hold `A*x+A*n`.
    vec_sum!(En, Ex)
    vec_sum!(An, Ax)
    ## for constraints like `h(x) + ∇h(x)*(n + d)`, make `Hn` hold `h(x)+∇h(x)*n`.
    vec_sum!(Hn, hx)
    vec_sum!(Gn, gx)
    χ, _d = solve_steepest_descent_problem(
        Δ, xn, Dfx, algo_opts.qp_opt, lb, ub, 
        En, Ab, An, Ec, Hn, Dhx, Gn, Dgx;
        descent_step_norm, normalize_gradients
    )
    
    ## set `step_vals.d`, scale it in place, and set `xs` and `fxs` in backtracking
    copyto!(d, _d)
    @unpack xs, fxs = step_vals
    @unpack fxn, backtracking_factor, rhs_factor, strict_backtracking = step_cache;
    backtrack!(d, mod, xn, xs, fxn, fxs, χ, backtracking_factor, rhs_factor, Val(strict_backtracking))

    return χ
end

function backtrack!(
    d, mod, xn, xs, fxn, fxs, χ, backtracking_factor, rhs_factor, strict_backtracking_val :: Val{strict_backtracking}
) where strict_backtracking
    ## initialize stepsize `σ=1`
    T = eltype(d)
    σ = one(T)
    σ_min = nextfloat(zero(T), 2)   # TODO make configurable

    ## pre-compute RHS for Armijo test
    rhs = χ * rhs_factor

    ## evaluate objectives at `xn` and trial point `xs`
    objectives!(fxn, mod, xn)
    xs .= xn .+ d
    objectives!(fxs, mod, xs)

    ## avoid re-computation of maximum for non-strict test:
    _fxn = strict_backtracking ? fxn : maximum(fxn)

    ## Until the `armijo_condition` is fullfilled
    while σ > σ_min && !armijo_condition(strict_backtracking_val, _fxn, fxs, rhs)
        ## shrink `σ`, `rhs` and `d` by multiplication with `backtracking_factor`
        σ *= backtracking_factor
        rhs *= backtracking_factor
        d .*= backtracking_factor

        ## reset trial point and compute objectives
        xs .= xn .+ d
        objectives!(fxs, mod, xs)
    end
    return nothing
end

armijo_condition(strict::Val{true}, fxn, fxs, rhs) = all(fxn .-  fxs .>= rhs)
armijo_condition(strict::Val{false}, fxn, fxs, rhs) = fxn - maximum(fxs) >= rhs
