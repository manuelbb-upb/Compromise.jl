const MOI = JuMP.MOI

function universal_copy!(::AbstractStepCache, ::AbstractStepCache)
    error("`universal_copy!` not defined.")
end

function compute_normal_step!(
    step_cache::AbstractStepCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)
    ## set `step_vals.n` to hold normal step
    nothing
end

function do_normal_step!(
    step_cache::AbstractStepCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    it_index, log_level, indent::Int=0
)
    if vals.theta_ref[] > 0
        pad_str = lpad("", indent)
        @logmsg log_level "$(pad_str)* Computing a normal step."   

        @ignoraise compute_normal_step!(
            step_cache, step_vals, Δ, mop, mod, scaler, lin_cons, 
            scaled_cons, vals, mod_vals; log_level
        )
        @. step_vals.xn = vals.x + step_vals.n
        @logmsg log_level """
            $(pad_str) Found normal step $(pretty_row_vec(step_vals.n; cutoff=60)). 
            $(pad_str) \t Hence xn=$(pretty_row_vec(step_vals.xn; cutoff=60)).""" 

    else
        step_vals.n .= 0
    end
    step_vals.xn .= vals.x .+ step_vals.n
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

function finalize_step_vals!(
    step_cache::AbstractStepCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)
    ## find stepsize and scale `step_vals.d`
    ## set `step_vals.s`, `step_vals.xs`, `step_vals.fxs`
    @. step_vals.s = step_vals.n + step_vals.d
    @. step_vals.xs = vals.x + step_vals.s
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

@with_kw mutable struct SteepestDescentConfig{
    BT<:Real, RT<:Real, DN<:Real, NN<:Real
} <: AbstractStepConfig
    
    backtracking_factor :: BT = 1//2
    rhs_factor :: RT = DEFAULT_FLOAT_TYPE(0.001)
    normalize_gradients :: Bool = false
    strict_backtracking :: Bool = true    
    
    descent_step_norm :: DN = Inf
    normal_step_norm :: NN = 2

    qp_opt :: Any = DEFAULT_QP_OPTIMIZER

    @assert 0 < backtracking_factor < 1 "`backtracking_factor` must be in (0,1)."
    @assert 0 < rhs_factor < 1 "`rhs_factor` must be in (0,1)."
    @assert descent_step_norm == Inf    # TODO enable other norms
    @assert normal_step_norm == 2 || isinf(normal_step_norm) # TODO enable other norms
end

@batteries SteepestDescentConfig selfconstructor=false

Base.@kwdef struct SteepestDescentCache{
    F<:AbstractFloat, 
    DN<:Real, 
    NN<:Real,
    QPOPT
} <: AbstractStepCache
    ## static information
    backtracking_factor :: F
    rhs_factor :: F
    normalize_gradients :: Bool
    strict_backtracking :: Bool
    descent_step_norm :: DN
    normal_step_norm :: NN
    
    ## caches for intermediate values
    fxn :: Vector{F}
    fx_tmp :: Vector{F}

    lb_tr :: Vector{F}
    ub_tr :: Vector{F}

    ## caches for product of constraint jacobians and normal step
    Axn :: Vector{F}    # Aξ + _A*n
    Dgx_n :: Vector{F}  # g(x) + ∇g(x) * n

    qp_opt :: QPOPT
end

function init_step_cache(
    cfg::SteepestDescentConfig, vals, mod_vals
)
    fxn = copy(vals.fx)
    fx_tmp = copy(fxn)
    F = eltype(fxn)
    
    backtracking_factor = F(cfg.backtracking_factor)
    rhs_factor = F(cfg.rhs_factor)

    @unpack normalize_gradients, strict_backtracking, descent_step_norm, 
        normal_step_norm, qp_opt = cfg

    lb_tr = Vector{F}(undef, vals.n_vars)
    ub_tr = similar(lb_tr)

    Axn = Vector{F}(undef, vals.dim_lin_ineq_constraints)
    Dgx_n = Vector{F}(undef, vals.dim_nl_ineq_constraints)

    return SteepestDescentCache(;
        backtracking_factor, rhs_factor, normalize_gradients, 
        strict_backtracking, descent_step_norm, normal_step_norm, 
        fxn, fx_tmp, lb_tr, ub_tr, Axn, Dgx_n, qp_opt 
    )
end

function universal_copy!(trgt::SteepestDescentCache, src::SteepestDescentCache)
    for fn in (:fxn, :lb_tr, :ub_tr, :Axn, :Dgx_n)
        universal_copy!(
            getfield(trgt, fn),
            getfield(src, fn)
        )
    end
end

function compute_normal_step!(
    step_cache::SteepestDescentCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)
    @unpack x = vals
    Eξ = vals.Ex
    Aξ = vals.Ax
    @unpack lb, ub = scaled_cons
    @unpack b, c = lin_cons
    _A = scaled_cons.A
    _E = scaled_cons.E
    @unpack gx, hx, Dgx, Dhx = mod_vals
    @unpack qp_opt = step_cache
    n = solve_normal_step_problem(
        x, lb, ub, Eξ, _E, c, Aξ, _A, b, hx, Dhx, gx, Dgx;
        qp_opt, step_norm = step_cache.normal_step_norm
    )
    Base.copyto!(step_vals.n, n)
    nothing 
end

function compute_descent_step!(
    step_cache::SteepestDescentCache, step_vals,
    Δ, mop, mod, scaler, lin_cons, scaled_cons, vals, mod_vals;
    log_level
)
    @unpack x = vals
    @unpack Dfx = mod_vals
    @unpack n = step_vals
    if !iszero(n)
        @ignoraise diff_objectives!(Dfx, mod, x)
    end    
    
    @unpack gx, Dgx = mod_vals
    @unpack Axn, Dgx_n = step_cache
    Aξ = vals.Ax
    _A = scaled_cons.A

    postprocess_normal_step_results!(Axn, Dgx_n, n, Aξ, _A, Dgx, gx)

    @unpack xn = step_vals
    @unpack lb, ub = scaled_cons
    _E = scaled_cons.E
    @unpack b, c = lin_cons
    @unpack Dhx = mod_vals
    @unpack normalize_gradients, qp_opt = step_cache
    χ, _d = solve_steepest_descent_problem(
        xn, Dfx, lb, ub, _E, Axn, _A, b, Dhx, Dgx_n, Dgx; 
        normalize_gradients, qp_opt, ball_norm = step_cache.descent_step_norm
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
    @unpack x = vals 
    @unpack xn, fxs, d = step_vals
    @unpack lb, ub = scaled_cons
    _A = scaled_cons.A
    @unpack b = lin_cons
    @unpack gx, Dgx = mod_vals
    @unpack lb_tr, ub_tr, Axn, Dgx_n = step_cache
    trust_region_bounds!(lb_tr, ub_tr, x, Δ, lb, ub)
    _, σ = initial_steplength(xn, d, lb_tr, ub_tr, Axn, _A, b, gx, Dgx_n, Dgx)
    
    if isnan(σ)
        step_vals.crit_ref[] = 0
        d .= 0
        step_vals.s .= step_vals.n
        step_vals.xs .= step_vals.xn
        @ignoraise objectives!(step_vals.fxs, mod, step_vals.xn)
        return nothing
    end
    
    @unpack fxn, fx_tmp, rhs_factor, strict_backtracking, backtracking_factor = step_cache
    @unpack xs = step_vals
    χ = step_vals.crit_ref[]
    @ignoraise _χ = backtrack!(
        d, xs, fxn, fxs,
        mod, xn, lb_tr, ub_tr, χ, backtracking_factor, rhs_factor, strict_backtracking;
        fx_tmp, σ_init=σ
    )
    step_vals.crit_ref[] = _χ
    @. step_vals.s = step_vals.n + d
    return nothing
end

function solve_normal_step_problem(
    x::RVec, 
    lb::Union{RVec, Nothing}, ub::Union{RVec, Nothing},
    Eξ::RVec, _E::Union{RMat, Nothing}, c::Union{RVec, Nothing},
    Aξ::RVec, _A::Union{RMat, Nothing}, b::Union{RVec, Nothing},
    hx::RVec, Dhx::RMat, gx::RVec, Dgx::RMat,
    ;
    qp_opt::Type{<:MOI.AbstractOptimizer},
    step_norm::Real=2
)
    n_vars = length(x)

    opt = JuMP.Model(qp_opt)
    JuMP.set_silent(opt)

    JuMP.@variable(opt, n[1:n_vars])

    normal_step_objective!(opt, n, step_norm)

    xn = x .+ n
    set_lower_bounds!(opt, xn, lb)
    set_upper_bounds!(opt, xn, ub)

    if !(isnothing(b) || isnothing(_A))
        ## Suppose `_A` and `_b` are applicable in the scaled domain, `_A * x ≤ _b`.
        ## In the unscaled domain, `A * ξ ≤ b`.
        ## `Aξ` actually holds `A*ξ = A*(S*x + s) = _A * x + A * s` ⇒ `_A * x = Aξ - A*s`.
        ## Additionally, `_b = b - A * s`.
        ## Thus,  `_A * x + _A * n .<= _b` is equivalent to 
        ## `Aξ - A*s + _A*n .<= b - A*s` ⇔ `Aξ + _A*n .<= b`
        JuMP.@constraint(opt, Aξ .+ _A * n .<= b)
    end

    if !(isnothing(_E) || isnothing(c))
        JuMP.@constraint(opt, Eξ .+ _E * n .== c)
    end

    JuMP.@constraint(opt, hx .+ Dhx'n .== 0)
    JuMP.@constraint(opt, gx .+ Dgx'n .<= 0)

    JuMP.optimize!(opt)

    if MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
        return fill(eltype(x)(NaN), length(n))
    end
    return JuMP.value.(n)
end

function normal_step_objective!(opt, n, p)
    if p == 2
        return normal_step_objective_2!(opt, n)
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

function postprocess_normal_step_results!(
    Axn, Dgx_n,
    # not modified:
    n::RVec,
    Aξ::RVec, 
    _A::Union{RMat, Nothing},
    Dgx::RMat, gx::RVec
)
    Axn .= Aξ
    if !isnothing(_A)
        Axn .+= _A * n
    end
    Dgx_n .= gx
    Dgx_n .+= Dgx'n
    nothing
end

function solve_steepest_descent_problem(
    xn::RVec, Dfx::RMat,
    lb::Union{RVec, Nothing}, ub::Union{RVec, Nothing},
    _E::Union{RMat, Nothing},
    Axn::RVec, _A::Union{RMat, Nothing}, b::Union{RVec, Nothing},
    Dhx::RMat, Dgx_n::RVec, Dgx::RMat,
    ;
    qp_opt::Type{<:MOI.AbstractOptimizer},
    ball_norm::Real,
    normalize_gradients::Bool
)
    n_vars = length(xn)

    opt = JuMP.Model(qp_opt)
    JuMP.set_silent(opt)

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

    if !isnothing(_E)
        JuMP.@constraint(opt, _E * d .== 0)
    end
    if !(isnothing(b) || isnothing(_A))
        JuMP.@constraint(opt, Axn .+ _A * d .<= b)
    end

    JuMP.@constraint(opt, Dhx'd .== 0)
    JuMP.@constraint(opt, Dgx_n .+ Dgx'd .<= 0)

    JuMP.optimize!(opt)

    if MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
        return 0, zero(xn)
    end
    χ = abs(JuMP.value(β))
    return χ, JuMP.value.(d)
end

function backtrack!(
    d, xs, fxn, fxs, 
    mod, xn, lb_tr, ub_tr, 
    χ, backtracking_factor, rhs_factor,
    strict :: Bool;
    fx_tmp = copy(fxn),
    σ_init = one(eltype(xs))
)
    if χ <= 0
        d .= 0
        return 0
    end
    ## initialize stepsize `σ=1`
    σ = σ_init
    x_delta_min = mapreduce(eps, min, xn)   # TODO make configurable ?

    x_delta_too_small = xdel -> all( _d -> abs(_d) <= x_delta_min, xdel )

    d .*= σ_init
    set_to_zero = x_delta_too_small(d)

    if set_to_zero
        d .= 0
        xs .= xn
        @ignoraise objectives!(fxs, mod, xs)
    end

    if !set_to_zero
        ## pre-compute RHS for Armijo test
        rhs = χ * rhs_factor * σ

        ## evaluate objectives at `xn` and trial point `xs`
        #src project_into_box!(xs, lb_tr, ub_tr)
        #src d .= xs .- xn
        @ignoraise objectives!(fxn, mod, xn)
        xs .= xn .+ d
        @ignoraise objectives!(fxs, mod, xs)
    
        fx_delta_min = mapreduce(eps, min, fxn)
        fx_delta_too_small = if strict
            fd -> all( _fd -> abs(_fd) <= fx_delta_min, fd )
        else
            fd -> abs(fd[1] <= fx_delta_min)
        end

        ## avoid re-computation of maximum for non-strict test:
        phi_xn = strict ? fxn : maximum(fxn)

        ## shrink stepsize until armijo condition is fullfilled
        while true 
            if σ <= 0
                set_to_zero = true
                break
            end
            if strict
                @. fx_tmp = phi_xn - fxs
                if fx_delta_too_small(fx_tmp)
                    set_to_zero = true
                    break
                end
                if all( fx_tmp .>= rhs )
                    break
                end
            else
                maximum!(fx_tmp, fxs)
                fx_tmp[1] -= phi_xn
                if fx_delta_too_small(fx_tmp)
                    set_to_zero = true
                    break
                end
                fx_tmp[1] *= -1
                if fx_tmp[1] >= rhs
                    break
                end
            end
            
            ## shrink `σ`, `rhs` and `d` by multiplication with `backtracking_factor`
            σ *= backtracking_factor
            rhs *= backtracking_factor
            d .*= backtracking_factor
            
            set_to_zero = x_delta_too_small(d)
            if set_to_zero
                d .= 0
            end

            ## reset trial point and compute objectives
            xs .= xn .+ d
            @ignoraise objectives!(fxs, mod, xs)
            
            set_to_zero && break
        end
    end
    _χ = set_to_zero ? 0 : χ
    return _χ
end

function descent_step_unit_constraint!(opt, d, p)
    if p == 2
        return descent_step_unit_constraint_2!(opt, d)
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

# # Helpers

similar_or_nothing(::Nothing)=nothing
similar_or_nothing(arr)=similar(arr)

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
    Axn::RVec, _A::Union{RMat, Nothing}, b::Union{RVec, Nothing},
    gx::RVec, Dgx_n::RVec, Dgx::RMat
)

    ## By construction, `d` is in the kernel of the linear equality constraint matrix
    ## `_E`, so any steplength `σ ∈ ℝ` will be compatible with these constraints.
    
    ## Same holds for the linearized nonlinear equality constraints:
    ## `Dhx'd .== 0`.
    
    T = eltype(xn)
    σ_min, σ_max = intersect_box(xn, d, lb_tr, ub_tr)
        
    if !(isnothing(_A) || isnothing(b))
        for (i, ai) = enumerate(eachrow(_A))
            # Axn + σ * A*d .<= b
            σl, σr = intersect_bound(Axn[i], LA.dot(ai, d), b[i], T)
            σ_min, σ_max = intersect_intervals(σ_min, σ_max, σl, σr, T)
        end
    end
    for (i, dgi) = enumerate(eachcol(Dgx))
        # gx + Dgx_n + σ * Dgx * d .<= 0
        σl, σr = intersect_bound(gx[i] + Dgx_n[i], LA.dot(dgi, d), 0, T)
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