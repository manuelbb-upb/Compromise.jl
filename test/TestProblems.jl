module TestProblems

function iip_ineq_func(val::Val)
    return function (g, x)
        ineq_constraint!(val, g, x)
    end
end

function oop_ineq_func(val::Val, n_vars::Int, F=Float64)
    return function (x)
        g = vec_for_ineq(val, n_vars, F)
        ineq_constraint!(val, g, x)
        return g 
    end
end
function vec_for_ineq(val, n_vars::Int, F=Float64)
    dim_g = dim_ineq_constraints(val, n_vars)
    return zeros(F, dim_g)
end

function iip_ineq_grads(val::Val)
    return function (Dg, x)
        grads_ineq_constraint!(val, Dg, x)
    end
end

function oop_ineq_grads(val::Val, n_vars::Int, F=Float64)
    return function (x)
        Dg = mat_for_ineq(val, n_vars, F)
        grads_ineq_constraint!(val, Dg, x)
        return Dg        
    end
end

function mat_for_ineq(val, n_vars::Int, F=Float64)
    dim_g = dim_ineq_constraints(val, n_vars)
    return zeros(F, n_vars, dim_g)
end

"""
For ``x ∈ ℝ^n``, compute in-place
```math
	g_j = (3-2x_{j+1})x_{j+1} - x_j - 2x_{j+2} + 1,
```
for all ``j=1, …, m`` with ``m=n-2``.
"""
@views function ineq_constraint!(::Val{1}, g, x)
	@. g = (3 - 2*x[2:end-1])*x[2:end-1] - x[1:end-2] - 2*x[3:end] + 1
	return nothing
end

function grads_ineq_constraint!(::Val{1}, Dg, x)
    for (j, dg) = enumerate(eachcol(Dg))
        dg .= 0
        dg[j] = -1
        dg[j+1] = 3 - 2 * x[j+1]
        dg[j+2] = -2
    end
    return nothing
end
dim_ineq_constraints(::Val{1}, n_vars) = n_vars - 2


"""
For ``x ∈ ℝ^n``, compute in-place
```math
	g_j = (3-2x_{j+1})x_{j+1} - x_j - 2x_{j+2} + 2.5,
```
for all ``j=1, …, m`` with ``m=n-2``.
"""
@views function ineq_constraint!(::Val{2}, g, x)
	@. g = (3 - 2*x[2:end-1])*x[2:end-1] - x[1:end-2] - 2*x[3:end] + 2.5
	return nothing
end
function grads_ineq_constraint!(::Val{2}, Dg, x)
    grads_ineq_constraint!(Val(1), Dg, x)
end
dim_ineq_constraints(::Val{2}, n_vars) = n_vars - 2

"""
For ``x ∈ ℝ^n``, compute in-place
```math
	g_j = x_j^2 + x_{j+1}^2 + x_jx_{j+1} - 2x_j - 2 x_{j+1} + 1
```
for all ``j=1, …, m`` with ``m=n-1``.
"""
@views function ineq_constraint!(::Val{3}, g , x)
	#src @. ciq = x[1:end-1]*(x[1:end-1] - 2 + x[2:end]) + x[2:end]*(x[2:end] - 2) + 1
    @. g = x[1:end-1]^2 + x[2:end]^2 + x[1:end-1]*x[2:end] - 2*x[1:end-1] - 2*x[2:end] + 1
	return nothing
end
function grads_ineq_constraint!(::Val{3}, Dg, x)
    for (j, dg) = enumerate(eachcol(Dg))
        dg .= 0
        dg[j] = 2 * x[j] + x[j+1] - 2
        dg[j+1] = 2 * x[j+1] + x[j] - 2
    end
end
dim_ineq_constraints(::Val{3}, n_vars) = n_vars - 1

"""
For ``x ∈ ℝ^n``, compute in-place
```math
	g_j = x_j^2 + x_{j+1}^2 + x_jx_{j+1} - 1
```
for all ``j=1, …, m`` with ``m=n-1``.
"""
@views function ineq_constraint!(::Val{4}, g, x)
	@. g = x[1:end-1]^2 + x[2:end]^2 + x[1:end-1]*x[2:end] - 1
	return nothing
end
function grads_ineq_constraint!(::Val{4}, g, x)
    for (j, dg) in enumerate(eachcol(Dg))
        dg .= 0
        dg[j] = 2 * x[j] + x[j+1]
        dg[j+1] = 2 * x[j+1] + x[j]
    end
end
dim_ineq_constraints(::Val{4}, n_vars) = n_vars - 1

"""
For ``x ∈ ℝ^n``, compute in-place
```math
	g_j = (3 - 0.5x_{j+1})x_{j+1} - x_j - 2x_{j+2} + 1
```
for all ``j=1, …, m`` with ``m=n-2``.
"""
@views function ineq_constraint!(::Val{5}, g, x)
	@. g = (3 - 0.5*x[2:end-1])*x[2:end-1] - x[1:end-2] - 2*x[3:end] + 1
	return nothing
end
function grads_ineq_constraint!(::Val{5}, g, x)
    for (j, dg) in enumerate(eachcol(Dg))
        dg .= 0
        dg[j] = -1
        dg[j+1] = 3 - x[j+1]
        dg[j+2] = -2
    end
end
dim_ineq_constraints(::Val{5}, n_vars) = n_vars - 2

"""
For ``x ∈ ℝ^n``, compute in-place
```math
	g_1 = \\sum_{j=1}^{n-2}((3 - 0.5x_{j+1})x_{j+1} - x_j - 2x_{j+2} + 1)
```
"""
@views function ineq_constraint!(::Val{6}, g, x)
	g[1] = 0
	for j=1:length(x)-2
		g[1] += (3 - 0.5*x[j+1])*x[j+1] - x[j] - 2*x[j+2] + 1
	end
	return nothing
end
dim_ineq_constraints(::Val{6}, n_vars) = 1
function grads_ineq_constraint!(::Val{6}, Dg, x)
    dg = @view Dg[:, 1]
    dg .= 0
    for j=1:length(x)-2
        dg[j] -= 1
        dg[j+1] += 3 - x[j+1]
        dg[j+2] -= 2
	end
    return nothing
end

struct SimplexParaboloidsInPlaceFunction <: Function
    counter :: Base.RefValue{Int}
end
SimplexParaboloidsInPlaceFunction() = SimplexParaboloidsInPlaceFunction(Ref(0))

function (f::SimplexParaboloidsInPlaceFunction)(y, x)
    f.counter[] += 1
    return simplex_paraboloids!(y, x)
end

function simplex_paraboloids!(y, x)
    y .= 0
    for i = eachindex(y)
        y[i] = sum( (x[1:i-1]).^2 ) + sum( (x[i+1:end]).^2 ) + (x[i] - 1)^2
    end
    return nothing
end

function grads_simplex_paraboloids!(Dy, x)
    for (i, dy) = enumerate(eachcol(Dy))
        dy .= 2 .* x
        dy[i] -= 2
    end
end

function sphere_constraint!(y, x)
    y[1] = sum(x.^2) - 1
end
eqconstraint_by_index(::Val{1})=sphere_constraint!
num_vars_to_num_eqconstraints(::Val{1}, n_vars)=1

function simplex_arrays(n_vars)
    A = ones(1, n_vars)
    b = [1,]
    return (A, b)
end

Base.@kwdef mutable struct SimplexParaboloids{F<:AbstractFloat}
    n_vars :: Int = 2
    objective_function :: SimplexParaboloidsInPlaceFunction = SimplexParaboloidsInPlaceFunction()
    x0 :: Vector{F} = rand(F, n_vars)
    eq_constraint_index :: Union{Nothing, Int} = nothing
    ineq_constraint_index :: Union{Nothing, Int} = nothing
    lb :: Union{Nothing, Vector{F}} = nothing
    ub :: Union{Nothing, Vector{F}} = nothing
    E :: Union{Nothing, Matrix{F}} = nothing
    c :: Union{Nothing, Vector{F}} = nothing
    A :: Union{Nothing, Matrix{F}} = nothing
    b :: Union{Nothing, Vector{F}} = nothing 
end

function _test_problem(::Val{1}, n_vars::Int=2, F=Float64)
    mop = SimplexParaboloids{F}(; n_vars)
    mop.x0 = zeros(F, n_vars)
    return mop
end

function _test_problem(::Val{2}, n_vars::Int=2, F=Float64)
    mop = SimplexParaboloids{F}(; n_vars)
    mop.lb = fill(F(-0.5), n_vars)
    mop.ub = fill(F(1.5), n_vars)
    mop.x0 = zeros(F, n_vars)
    return mop
end

function _test_problem(::Val{3}, n_vars::Int=2, F=Float64)
    mop = _test_problem(Val(2), n_vars, F)
    A, b = simplex_arrays(n_vars)
    mop.A = A
    mop.b = b
    return mop
end

function _test_problem(::Val{4}, n_vars::Int=2, F=Float64)
    mop = _test_problem(Val(3), n_vars, F)
    mop.x0 = ones(n_vars) # infeasible for simplex constraints
    return mop
end

function _test_problem(::Val{5}, n_vars::Int=2, F=Float64)
    mop = _test_problem(Val(4), n_vars, F)
    mop.ineq_constraint_index = 3
    # x0 infeasible for both constraints, but problem has non-empty feasible region
    return mop
end

function _test_problem(::Val{6}, n_vars::Int=2, F=Float64)
    mop = _test_problem(Val(5), n_vars, F)
    mop.b[1] = .1
    # problem infeasible ?
    return mop
end

using Compromise
function to_mutable_mop(mop::SimplexParaboloids; mcfg=:rbf, max_func_calls=typemax(Int))
    p = MutableMOP(;
        num_vars = mop.n_vars,
        lb = mop.lb,
        ub = mop.ub,
        E = mop.E,
        c = mop.c,
        A = mop.A,
        b = mop.b
    )
    add_objectives!(
        p, mop.objective_function, :rbf; 
        dim_out=mop.n_vars, func_iip=true,
        max_func_calls
    )
    if !isnothing(mop.ineq_constraint_index)
        v = Val(mop.ineq_constraint_index)
        if mcfg == :rbf || mcfg isa RBFConfig
            add_nl_ineq_constraints!(
                p, iip_ineq_func(v), mcfg;
                    func_iip=true, dim_out=dim_ineq_constraints(v, mop.n_vars),
                    max_func_calls
            )
        end
    end
    return p
end

end#module