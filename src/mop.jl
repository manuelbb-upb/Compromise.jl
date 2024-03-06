
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
# The optional function `float_type` returns the type of result and derivative vectors:
float_type(::AbstractMOP)::Type{<:AbstractFloat}=DEFAULT_FLOAT_TYPE

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
lin_eq_constraints_matrix(::AbstractMOP)::Union{Nothing, RMat}=nothing
lin_ineq_constraints_matrix(::AbstractMOP)::Union{Nothing, RMat}=nothing
lin_eq_constraints_vector(::AbstractMOP)::Union{Nothing, RVec}=nothing
lin_ineq_constraints_vector(::AbstractMOP)::Union{Nothing, RVec}=nothing

# From that we can derive dimension getters as well:
dim_lin_cons(A::Nothing, b)=0
function dim_lin_cons(A, b)
    dimA = size(A, 1)
    @assert length(b) == dimA "Dimension mismatch in linear constraints."
    return dimA
end
    
function dim_lin_eq_constraints(mop::AbstractMOP)
    A = lin_eq_constraints_matrix(mop)
    b = lin_eq_constraints_vector(mop)
    return dim_lin_cons(A, b)
end
function dim_lin_ineq_constraints(mop::AbstractMOP)
    A = lin_ineq_constraints_matrix(mop)
    b = lin_ineq_constraints_vector(mop)
    return dim_lin_cons(A, b)
end

# ## Evaluation

# !!! note
#     All evaluation and differentiation methods that you see below should always 
#     return `nothing`, **unless** you want to stop early.
#     Then return something else, for example a string.

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

# To ensure they only get called if needed, we wrap them and assign shorter names:
function objectives!(y::RVec, mop::AbstractMOP, x::RVec)
    eval_objectives!(y, mop, x)
end
function nl_eq_constraints!(y::RVec, mop::AbstractMOP, x::RVec)
    dim_nl_eq_constraints(mop) <= 0 && return nothing
    eval_nl_eq_constraints!(y, mop, x)
end
function nl_ineq_constraints!(y::RVec, mop::AbstractMOP, x::RVec)
    dim_nl_ineq_constraints(mop) <= 0 && return nothing
    eval_nl_ineq_constraints!(y, mop, x)
end

function lin_cons!(LHSx_min_rhs, LHSx, LHS, rhs, x)
    isnothing(LHS) && return nothing
    isnothing(rhs) && return nothing
    LA.mul!(LHSx, LHS, x)
    @. LHSx_min_rhs = LHSx - rhs
    return nothing
end

function lin_eq_constraints!(Ex_min_e, Ex, mop::AbstractMOP, x::RVec)
    E = lin_eq_constraints_matrix(mop)
    c = lin_eq_constraints_vector(mop)
    return lin_cons!(Ex_min_e, Ex, E, c, x)
end

function lin_ineq_constraints!(Ax_min_b, Ax, mop::AbstractMOP, x::RVec)
    A = lin_ineq_constraints_matrix(mop)
    b = lin_ineq_constraints_vector(mop)
    return lin_cons!(Ax_min_b, Ax, A, b, x)
end

# ## Pre-Allocation

# The vectors to hold objective values, constraint values, etc., 
# are allocated using these functions:
## hx = nonlinear equality constraints at x
## gx = nonlinear inequality constraints at x
function prealloc_objectives_vector(mop) end
function prealloc_lin_eq_constraints_vector(mop) end
function prealloc_lin_ineq_constraints_vector(mop) end
function prealloc_nl_eq_constraints_vector(mop) end
function prealloc_nl_ineq_constraints_vector(mop) end
# I have implemented some defaults to just allocate a `Vector` of correct
# length.
# The `yoink_` functions are called internally, do not overwrite them, 
# unless you know what you are doing.
for (dim_func, func_suffix) in (
    (:dim_objectives, :objectives_vector),
    (:dim_nl_eq_constraints, :nl_eq_constraints_vector),
    (:dim_nl_ineq_constraints, :nl_ineq_constraints_vector),
    (:dim_lin_eq_constraints, :lin_eq_constraints_vector),  # don't modify this…
    (:dim_lin_ineq_constraints, :lin_ineq_constraints_vector),  # and this, else type mismatch in value structs
)
    func_name = Symbol("prealloc_", func_suffix)
    yoink_name = Symbol("yoink_", func_suffix)
    @eval begin 
        function $(func_name)(mop::AbstractMOP)
            dim = $(dim_func)(mop) 
            T = float_type(mop)
            return Vector{T}(undef, dim)
        end

        function $(yoink_name)(mop::AbstractMOP)
            y = $(func_name)(mop) :: RVec
            @assert eltype(y) == float_type(mop) "Vector eltype does not match problem float type."
            return y
        end
    end
end