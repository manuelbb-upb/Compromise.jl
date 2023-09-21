
# # AbstractMOP Interface
# An object subtyping `AbstractMOP` is a glorified wrapper around vector-vector functions.
# The methods below were originally meant to be used to implement our algorithm similar to 
# how it has been stated in the article, rather “mathematically“.
# That is, we do not care for how the problem has been modelled and set up.
# We only need function handles and some meta-data concerning dimensions and data types.

#=
Formally, our problem reads
```math
\begin{aligned}
&\min_{x = [x₁, …, x_N]} 
    \begin{bmatrix}
        f₁(x)   \\
        ⋮       \\
        f_K(x)
    \end{bmatrix}
        &\text{subject to}\\
    & g(x) = [g₁(x), …, g_P(x)] ≤ 0, \\
    & h(x) = [h₁(x), …, h_M(x)] = 0, \\
    & lb ≤ x ≤ ub,  A ⋅ x ≤ b, E ⋅ x = c.
\end{aligned}
```
In the code, we often follow this notation: 
* `f` indicates an objective function.
* `g` a nonlinear inequality constraint.
* `h` a nonlinear equality constriant.
* `A` is the matrix of linear inequality constraints, `b` the right hand side vector. 
* `E` is the matrix of linear equality constraints, `c` the right hand side vector.
* `lb` and `ub` define box constraints.
=#

# At the beginning of an optimization routine, initialization based on the initial site 
# can be performed:
initialize(mop::AbstractMOP, ξ0::RVec)=mop

# ## Meta-Data
# The optional function `precision` returns the type of result and derivative vectors:
precision(::AbstractMOP)::Type{<:AbstractFloat}=DEFAULT_PRECISION

# We would also like to deterministically query the expected surrogate model types:
model_type(::AbstractMOP)::Type{<:AbstractMOPSurrogate}=AbstractMOPSurrogate

# Below functions are used to query dimension information. 
dim_objectives(::AbstractMOP)::Int=0            # mandatory
dim_nl_eq_constraints(::AbstractMOP)::Int=0     # optional
dim_nl_ineq_constraints(::AbstractMOP)::Int=0   # optional

# ## Linear Constraints
# An `AbstractMOP` can have constrained variables.
# The corresponding functions should return full bound vectors or `nothing`.
# For lower bounds, `nothing` corresponds to `-Inf`, but we do not necessarily use such 
# vectors in the inner solver. Upper bounds would be `Inf` in case of `nothing`.
lower_var_bounds(::AbstractMOP)::Union{Nothing, Vec}=nothing
upper_var_bounds(::AbstractMOP)::Union{Nothing, Vec}=nothing

# Moreover, problems can have linear equality constraints and linear inequality constraints
# ```math
#   E x = c
#   \quad
#   A x ≤ b 
# ```
lin_eq_constraints(::AbstractMOP)::Union{Nothing, Tuple{RMat,RVecOrMat}}=nothing
lin_ineq_constraints(::AbstractMOP)::Union{Nothing, Tuple{RMat,RVecOrMat}}=nothing

# From that we can derive dimension getters as well:
## helper
dim_lin_constraints(dat::Nothing)=0
function dim_lin_constraints((A,b)::Tuple{RMat, RVecOrMat})
    dim = length(b)
    @assert size(A, 2) == dim "Dimension mismatch in linear constraints."
    return dim
end
## actual functions
dim_lin_eq_constraints(mop::AbstractMOP)=dim_lin_constraints(lin_eq_constraints(mop))
dim_lin_ineq_constraints(mop::AbstractMOP)=dim_lin_constraints(lin_ineq_constraints(mop))

# ## Evaluation
# Evaluation of nonlinear objective functions requires the following method:
function eval_objectives!(y::RVec, mop::M, x::RVec) where {M<:AbstractMOP}
    error("`eval_objectives!(y, mop, x) not implemented for mop of type $(M).")
end
# If there are constraints, these have to be defined as well:
function eval_nl_eq_constraints!(y::RVec, mop::M, x::RVec) where {M<:AbstractMOP}
    error("`eval_nl_eq_constraints!(y, mop, x) not implemented for mop of type $(M).")
end
function eval_nl_ineq_constraints!(y::RVec, mop::M, x::RVec) where {M<:AbstractMOP}
    error("`eval_nl_ineq_constraints!(y, mop, x) not implemented for mop of type $(M).")
end

# To ensure, they only get called if needed, we wrap them and assign shorter names:
objectives!(y::RVec, mop::AbstractMOP, x::RVec)=eval_objectives!(y, mop, x)
nl_eq_constraints!(y::Nothing, mop::AbstractMOP, x::RVec)=nothing
nl_ineq_constraints!(y::Nothing, mop::AbstractMOP, x::RVec)=nothing
nl_eq_constraints!(y::RVec, mop::AbstractMOP, x::RVec)=eval_nl_eq_constraints!(y, mop, x)
nl_ineq_constraints!(y::RVec, mop::AbstractMOP, x::RVec)=eval_nl_ineq_constraints!(y, mop, x)

# Similar methods can be defined for linear constraints.
"""
    lin_cons!(residual_vector, prod_cache, constraint_data, x)

Given a linear constraint `A*x .<= b` or `A*x .== b`, compute the product `A*x` and store 
the result in `prod_cache`, and also compute `A*x .- b` and store the result in 
`residual_vector`.
`constraint_data` should either be the tuple `(A,b)::Tuple{RMat,RVec}` or `nothing`.
"""
lin_cons!(residual_vector, prod_cache, constraint_data, x)=nothing
lin_cons!(res::Nothing, mat_vec::Nothing, cons::Nothing, x::RVec) = nothing
function lin_cons!(res::RVec, mat_vec::RVec, (A, b)::Tuple, x::RVec)
    LA.mul!(mat_vec, A, x)
    @. res = mat_vec - b
    return nothing
end
# More specific methods with descriptive names applicable to an `AbstractMOP`:
lin_eq_constraints!(res::Nothing, mat_vec::Nothing, mop::AbstractMOP, x::RVec)=nothing
lin_ineq_constraints!(res::Nothing, mat_vec::Nothing, mop::AbstractMOP, x::RVec)=nothing
lin_eq_constraints!(res::RVec, mat_vec::Nothing, mop::AbstractMOP, x::RVec)=lin_cons!(res, mat_vec, lin_eq_constraints(mop), x)
lin_ineq_constraints!(res::RVec, mat_vec::Nothing, mop::AbstractMOP, x::RVec)=lin_cons!(res, mat_vec, lin_ineq_constraints(mop), x)

# ## Pre-Allocation
# Why do we also allow `nothing` as the target for constraints?
# Because that is the default cache returned if there are none:
function prealloc_objectives_vector(mop::AbstractMOP)
    T = precision(mop)
    return Vector{T}(undef, dim_objectives(mop))
end
## hx = nonlinear equality constraints at x
## gx = nonlinear inequality constraints at x
## Ex = linear equality constraints at x
## Ax = linear inequality constraints at x
## These are defined below (and I put un-specific definitions here for the Linter)
function prealloc_nl_eq_constraints_vector(mop) end
function prealloc_nl_ineq_constraints_vector(mop) end
function prealloc_lin_eq_constraints_vector(mop) end
function prealloc_lin_ineq_constraints_vector(mop) end
for (dim_func, prealloc_func) in (
    (:dim_nl_eq_constraints, :prealloc_nl_eq_constraints_vector),
    (:dim_nl_ineq_constraints, :prealloc_nl_ineq_constraints_vector),
    (:dim_lin_eq_constraints, :prealloc_lin_eq_constraints_vector),
    (:dim_lin_ineq_constraints, :prealloc_lin_ineq_constraints_vector),
)
    @eval function $(prealloc_func)(mop::AbstractMOP)
        dim = $(dim_func)(mop) 
        if dim > 0
            T = precision(mop)
            return Vector{T}(undef, dim)
        else
            return nothing
        end
    end
end