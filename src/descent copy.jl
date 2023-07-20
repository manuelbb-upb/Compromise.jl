# This file is archived and a TODO.
# In "descent.jl" I have removed the polytope intersection and include the trust region
# constraint in the linear/quadratic program.
# Conceptually, I am sure that should work.
# Mandatory Methods:
descent_step(::AbstractStepCache)::RVec=nothing
normal_step(::AbstractStepCache)::RVec=nothing

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
    ## working arrays
    descent_step :: Vector{T}
    normal_step :: Vector{T}
    criticality :: Base.RefValue{T}

    xn :: Vector{T}

    ## caches for constraint jacobians times normal step
    En :: En        # E*n
    An :: AN        # A*n
    Hn :: HN        # ∇h(x)*n
    Gn :: GN        # ∇g(x)*n

    ## cache for stepsize computation
    Ed :: EN
    Ad :: AN
    Hd :: HN
    Hd :: GN
end

normal_step(step_cache::NormalStepCache)::RVec=step_cache.normal_step
descent_step(step_cache::SteepestDescentCache)=step_cache.descent_step

function init_step_cache(
    cfg::SteepestDescentConfig, vals, mod_vals
)
    T = eltype(vals.x)
    n_vars = length(vals.x)
    
    backtracking_factor = T(cfg.backtracking_factor)
    rhs_factor = T(cfg.rhs_factor)

    @unpack normalize_gradients, strict_backtracking, descent_step_norm, 
        normal_step_norm = cfg

    descent_step = zeros(T, n)
    normal_step = zeros(T, n)
    criticality = Ref(zero(T))

    xn = deepcopy(vals.x)

    En = deepcopy(vals.Ex)
    Ed = deepcopy(En)

    An = deepcopy(vals.Ax)
    Ad = deepcopy(An)

    Hn = deepcopy(mod_vals.hx)
    Hd = deepcopy(Hn)
    
    Gn = deepcopy(mod_vals.gx)
    Gd = deepcopy(Gn)

    return SteepestDescentCache(;
        backtracking_factor, rhs_factor, normalize_gradients, strict_backtracking,
        descent_step_norm, normal_step_norm, descent_step, normal_step, criticality,
        xn, En, An, Hn, Gn, Ed, Ad, Hd, Gd
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
        JuMP.@constraint!(opt, c .+ Ax_expr .== b)
    else
        JuMP.@constraint!(opt, c .+ Ax_expr .<= b)
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
# Helper for surrogate models if there are no constraints stored:
function set_linear_constraints!(opt, c, x, A::Nothing, b::Nothing, ctype::Symbol)
    return nothing
end

read_linear_constraint_expression!(trgt_arr::Nothing, src_ex::Nothing)=nothing
function read_linear_constraint_expression!(trgt_arr, src_ex)
    trgt_arr .= value.(src_ex)
    return nothing
end

function solve_normal_step_problem(
    x, qp_opt, lb, ub, 
    Ex, Ax, Ab, 
    Ec, En, An, 
    hx, H, Hn, 
    gx, G, Gn
)
    n_vars = length(x)
        
    opt = JuMP.Model( qp_opt )
    JuMP.set_silent(opt)
    #src JuMP.set_optimizer_attribute(itrn, "polish", true)

    JuMP.@variable(opt, n[1:n_vars])
    if normal_cache.normal_step_norm == 2
        JuMP.@objective(opt, Min, sum(n.^2))
    elseif normal_cache.normal_step_norm == Inf
        @variable(opt, norm_n)
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

    ## hx + Hx * n .== 0
    Hn_ex = set_linear_constraints!(opt, hx, n, H, 0, :eq)

    ## gx + Gx * n .<= 0
    Gn_ex = set_linear_constraints!(opt, gx, n, G, 0, :ineq)

    JuMP.optimize!(opt)

    ## read values from inner problem expressions
    read_linear_constraint_expression!(En, En_ex)
    read_linear_constraint_expression!(An, An_ex)
    read_linear_constraint_expression!(Hn, Hn_ex)
    read_linear_constraint_expression!(Gn, Gn_ex)

    return value.(n)
end

function compute_normal_step!(
    step_cache, scaled_cons, vals, mod_vals, algo_opts
)
    xn = step_cache.xn
    copyto!(xn, vals.x)
    n = step_cache.normal_step
    if vals.θ[] > 0
        @unpack Ax, Ex = vals
        @unpack lb, ub, Ab, Ec = scaled_cons
        @unpack hx, gx, Dhx, Dgx = mod_vals
        @unpack En, An, Hn, Gn = step_cache
        x = vals.x
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
    xn, Dfx, qp_opt, lb, ub, 
    Ed, Exn, Ab, Ad, Axn, Ec, Hd, Hxn, Dhx, Gd, Gxn, Dgx;
    descent_step_norm, normalize_gradients
)
    n_vars = length(xn)

    opt = JuMP.Model(qp_opt)
    JuMP.set_silent(opt)

    JuMP.@variable(opt, β)
    JuMP.@variable(opt, d[1:n_vars])

    JuMP.@objective(opt, Min, β)
    if descent_step_norm == Inf
        JuMP.@constraint(opt, -1 .<= d)
        JuMP.@constraint(opt, d .<= 1)
    end

    ## descent constraints, ∇f(x)*d ≤ β
    for df in eachcol(Dfx)
        ndf = normalize_gradients ? LA.norm(df) : 1
        JuMP.@constraint(opt, df'd <= ndf * β)
    end

    xs = xn .+ s
    ## lb .<= x + s
    set_lower_bounds!(opt, xs, lb)
    ## x + s .<= ub
    set_upper_bounds!(opt, xs, ub)

    ## A(x + n + d) ≤ b ⇔ A(x+n) + A*d <= b
    ## ⇒ c = A(x+n)
    Ed_ex = set_linear_constraints!(opt, Exn, d, Ab, :eq)
    Ad_ex = set_linear_constraints!(opt, Axn, d, Ec, :ineq)

    ## hx + H(n + d) = 0 ⇔ (hx + Hn) + Hd = 0
    Hd_ex = set_linear_constraints!(opt, Hxn, d, Dhx, 0, :eq)
    Gd_ex = set_linear_constraints!(opt, Gxn, d, Dgx, 0, :ineq)

    JuMP.optimize!(opt)

    ## read values from inner problem expressions
    read_linear_constraint_expression!(Ed, Ed_ex)
    read_linear_constraint_expression!(Ad, Ad_ex)
    read_linear_constraint_expression!(Hd, Hd_ex)
    read_linear_constraint_expression!(Gd, Gd_ex)

    return -value(β), value.(d)
end

mat_sum(::Nothing, ::Nothing)=nothing
mat_sum(A, B) = A .+ B

function compute_descent_step!(
    step_cache::SteepestDescentCache, scaled_cons, vals, mod_vals, qp_opt
)
    @unpack lb, ub, Ab, Ec = scaled_cons
    @unpack Ex, Ax = vals
    @unpack hx, gx, Dfx, Dhx, Dgx = mod_vals
    @unpack xn, Ed, En, Ad, An, Hd, Hn, Gd, Gn, 
        criticality, descent_step, normalize_gradients, descent_step_norm = step_cache
    Exn = mat_sum(Ex, En)
    Axn = mat_sum(Ax, An)
    Hxn = mat_sum(hx, Hn)
    Gxn = mat_sum(gx, Gn)
    χ, d = solve_steepest_descent_problem(
        xn, Dfx, qp_opt, lb, ub, 
        Ed, Exn, Ab, 
        Ad, Axn, Ec, 
        Hd, Hxn, Dhx, 
        Gd, Gxn, Dgx;
        descent_step_norm, normalize_gradients
    )
    descent_step .= d
    criticality[] = χ

end

#=
###### Polytope Intersection
To start backtracking, we sometimes need to solve problems of the form
```math
\operatorname{arg\,max}_{σ\in ℝ} σ \quad\text{  s.t.  }\quad
x + σ ⋅ d ∈ B ∩ L,
```
where ``x\in ℝ^N, d \in ℝ^N,`` and ``B\cap L`` is a polytope.
The problem can be solved using any LP algorithm.

Below, we use an iterative approach with the following logic: \
For each dimension, we determine the interval of allowed step sizes and then intersect these intervals.

For scalars ``x_i, σ, d_i, b_i``, consider the inequality
```math 
x_i + σ ⋅ d_i ≤ b_i.
```
What are possible values for ``σ``?
* If ``d_i=0``:
  - If ``x_i ≤ b_i``, then ``σ ∈ [-∞, ∞]``.
  - If ``x_i > b_i``, then ``σ ∈ ∅`` (does not exist!!!).
* Else:
  - If ``x_i = b_i``:
    * If ``d_i < 0``, then ``σ ∈ [0, ∞]``.
    * If ``d_i > 0``, then ``σ ∈ [0,0]``.
  - If ``x_i < b_i``:
    * If ``d_i < 0``, then ``σ ∈ [\underbrace{(b_i-x_i)/d_i}_{<0}, ∞]``.
    * If ``d_i > 0``, then ``σ ∈ [-∞, (b_i-x_i)/d_i].``
  - If ``x_i > b_i`` (``x`` infeasible):
    * If ``d_i < 0``, then ``σ ∈ [\underbrace{(b_i-x_i)/d_i}_{>0}, ∞]``.
    * If ``d_i > 0``, then ``σ ∈ [-∞, (b_i-x_i)/d_i]``.

This decision tree is implemented in `stepsize_interval`:
=#
function stepsize_interval(
    x::X, d::D, b::B, T = Base.promote_type(DEFAULT_PRECISION, X, D, B)
) where {X<:Real, D<:Real, B<:Real}
	T_Inf = T(Inf)
	T_NaN = T(NaN)
	T_Zero = zero(T)
	if iszero(d)
		if x <= b
			return (-T_Inf, T_Inf)
		else
			return (T_NaN, T_NaN)
		end
	else
		if x == b
			if d < 0
				return (T_Zero, T_Inf)
			else
				return (T_Zero, T_Zero)
			end
		else
			r = T((b-x) / d)
			if d < 0
				return (r, T_Inf)
			else
				return (-T_Inf, r)
			end
		end
	end
end

# In case of vector constraints, we simply intersect the stepsize intervals:
function stepsize_interval(
    x::AbstractVector{X}, d::AbstractVector{D}, b, 
    T=Base.promote_type(DEFAULT_PRECISION, X, D)
) where {X<:Real, D<:Real}
	T_Inf = T(Inf)
    ## start with (-∞, ∞)
	l = -T_Inf
	r = T_Inf
	for (xi, di, bi) = zip(x, d, b)
		li, ri = stepsize_interval(xi, di, bi, T)
		l, r = intersect_interval((l,r), (li, ri))
        isnan(l) && break
	end
	return (l, r)
end

function stepsize_interval(
    x::AstractVector{X}, d::AbstractVector{D}, b::Real, 
    T=Base.promote_type(DEFAULT_PRECISION, X, D)
)
    return stepsize_interval(x, d, Iterators.repeated(b), T)
end

function intersect_interval(int1, int2, T=Base.promote_type(eltype(int1), eltype(int2)))
    l1, r1 = int1
    isnan(l1) || isnan(r1) && return l1, r1
    l2, r2 = int2
    isnan(l2) || isnan(r2) && return l2, r2
    L = max(l1, l2)
	R = min(r1, r2)
	if L > R
		return (T(NaN), T(NaN))
	end
	return (L, R)
end

# We can now successively compute the stepsize intervals for inequality constraints
# and intersect the them, starting with `(l, r)`:

upper_bounds_interval(x, d, ub::Nothing, l, r)=(l, r)
lower_bounds_interval(x, d, lb::Nothing, l, r)=(l, r)
function upper_bounds_interval(x, d, b, l, r)
    ## ``x + σd ≤ b``
    _l, _r = stepsize_interval(x, d, b)
    return intersect_interval((l, r), (_l, _r))
end
function lower_bounds_interval(x, d, b, l, r)
    ## ``b ≤ x + σd`` ⇔ ``-x - σd ≤ -b``
    _l, _r = stepsize_interval(-x, -d, -b)
    return intersect_interval((l, r), (_l, _r))
end

# The case `A (x + n + σd) ≤ b` can be transformed to work with `stepsize_interval`,
# ``A(x+n+σd) ≤ b`` ⇔ ``A(x+n) + σ (Ad) ≤ b``.
# Moreover, if there are constraints, then the values ``A(x+n)`` and ``Ad`` have already
# been computed.

lin_cons_interval(Ab::Nothing, Axn, Ad, l, r)=(l,r)
lin_cons_interval(A::Nothing, b::Nothing, Axn, Ad, l, r)=(l,r)
lin_cons_interval((A,b)::Tuple, Axn, Ad, l, r)=lin_cons_interval(A, b, Axn, Ad, l, r)
function lin_cons_interval(A, b, Axn, Ad, l, r)
    isnan(l) && return l, r
    _l, _r = stepsize_interval(Axn, Ad, b)
    return intersect_interval((l, r), (_l, _r))
end

"""
Return a tuple ``(σ_-,σ_+)`` of minimum and maximum stepsize ``σ ∈ ℝ`` 
such that ``xn + σd`` conforms to the linear constraints ``lb ≤ xn + σd ≤ ub`` 
and ``A(xn+σd) - b ≦ 0`` along all dimensions.
If there is no such stepsize then `(NaN, NaN)` is returned.
"""
function intersect_linear_ineq_constraints( 
	xn::RVec, d::RVec, lb, ub, 
    Ab, Axn, Ad,
    G, Gxn, Gd
)
    T = Base.promote_type(eltype(xn), eltype(d))

	T_Inf = T(Inf)
	
    l, r = (-T_Inf, T_Inf)

	# if `d` is zero vector, we can move infinitely in all dimensions and directions
	if iszero(d)
        return (l, r)
    end

    l, r = lower_bounds_interval(xn, d, lb, l, r)
    l, r = upper_bounds_interval(xn, d, ub, l, r)
    
    l, r = lin_cons_interval(Ab, Axn, Ad, l, r)
    l, r = lin_cons_interval(G, zero(T), Gxn, Gd, l, r)
	
    return l, r
end
#=
		# there are equality constraints
		# they have to be all fullfilled and we loop through them one by one (rows of A_eq)
		N = size(A_eq, 1)
		_b_eq = isempty(b_eq) ? zeros(T, N) : b_eq

		σ = T_NaN
		for i = 1 : N
            # a'(x+ σd) - b = 0 ⇔ σ a'd = -(a'x - b) ⇔ σ = -(a'x -b)/a'd 
            ad = A_eq[i,:]'d
			if !iszero(ad)
				σ_i = - (A_eq[i, :]'x - _b_eq[i]) / ad
			else
                # check for primal feasibility of `x`:
				if !iszero( A_eq[i,:]'x .- _b_veq[i] )
					return interval_nan
				end
			end
			
			if isnan(σ)
				σ = σ_i
			else
				if !(σ_i ≈ σ)
					return interval_nan
				end
			end
		end
		
		if isnan(σ)
			# only way this could happen:
			# ad == 0 for all i && x feasible w.r.t. eq const
			return intersect_linear_constraints(x, d, lb, ub, A, b_ineq )
		end
			
		# check if x + σd is compatible with the other constraints
		x_trial = x + σ * d
		
		(!isempty(lb) && any(x_trial .< lb )) && return interval_nan
		(!isempty(ub) && any(x_trial .> ub )) && return interval_nan
		(!isempty(A) && any( A * x_trial > _b_ineq )) && return interval_nan
		return (σ, σ)
	end
end

=#