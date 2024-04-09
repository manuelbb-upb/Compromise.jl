# AbstractMOP Interface

An object subtyping `AbstractMOP` is a glorified wrapper around vector-vector functions.
The methods below were originally meant to be used to implement our algorithm similar to
how it has been stated in the article, rather “mathematically“.
That is, we do not care for how the problem has been modelled and set up.
We only need function handles and some meta-data concerning dimensions and data types.

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

At the beginning of an optimization routine, initialization
can be performed:

````julia
initialize(mop::AbstractMOP)=mop
````

## Meta-Data
The optional function `float_type` returns the type of result and derivative vectors:

````julia
float_type(::AbstractMOP)::Type{<:AbstractFloat}=DEFAULT_FLOAT_TYPE
````

We would also like to deterministically query the expected surrogate model types:

````julia
model_type(::AbstractMOP)::Type{<:AbstractMOPSurrogate}=AbstractMOPSurrogate
````

Below functions are used to query dimension information.

````julia
dim_vars(::AbstractMOP)::Int=0
dim_objectives(::AbstractMOP)::Int=0            # mandatory
dim_nl_eq_constraints(::AbstractMOP)::Int=0     # optional
dim_nl_ineq_constraints(::AbstractMOP)::Int=0   # optional

initial_vars(::AbstractMOP)::Union{RVec, RMat, Nothing}=nothing
````

## Linear Constraints
An `AbstractMOP` can have constrained variables.
The corresponding functions should return full bound vectors or `nothing`.
For lower bounds, `nothing` corresponds to `-Inf`, but we do not necessarily use such
vectors in the inner solver. Upper bounds would be `Inf` in case of `nothing`.

````julia
lower_var_bounds(::AbstractMOP)::Union{Nothing, Vec}=nothing
upper_var_bounds(::AbstractMOP)::Union{Nothing, Vec}=nothing
````

Moreover, problems can have linear equality constraints and linear inequality constraints
```math
  E x = c
  \quad
  A x ≤ b
```

````julia
lin_eq_constraints_matrix(::AbstractMOP)::Union{Nothing, RMat}=nothing
lin_ineq_constraints_matrix(::AbstractMOP)::Union{Nothing, RMat}=nothing
lin_eq_constraints_vector(::AbstractMOP)::Union{Nothing, RVec}=nothing
lin_ineq_constraints_vector(::AbstractMOP)::Union{Nothing, RVec}=nothing
````

From that we can derive dimension getters as well:

````julia
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
````

## Evaluation

!!! note
    All evaluation and differentiation methods that you see below should always
    return `nothing`, **unless** you want to stop early.
    Then return something else, for example a string.

Evaluation of nonlinear objective functions requires the following method:

````julia
function eval_objectives!(y::RVec, mop::M, x::RVec) where {M<:AbstractMOP}
    error("`eval_objectives!(y, mop, x) not implemented for mop of type $(M).")
end
````

If there are constraints, these have to be defined as well:

````julia
function eval_nl_eq_constraints!(y::RVec, mop::M, x::RVec) where {M<:AbstractMOP}
    error("`eval_nl_eq_constraints!(y, mop, x) not implemented for mop of type $(M).")
end
function eval_nl_ineq_constraints!(y::RVec, mop::M, x::RVec) where {M<:AbstractMOP}
    error("`eval_nl_ineq_constraints!(y, mop, x) not implemented for mop of type $(M).")
end
````

To ensure they only get called if needed, we wrap them and assign shorter names:

````julia
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
````

## Pre-Allocation

Every `mop` of type `AbstractMOP` has to implement `init_value_caches(mop)`.
It should return an object of type `AbstractValueCache`.
This cache is queried for evaluation data by methods such as
`cached_fx(mop_cache)` to retrieve the objective values for example.
Note, that the getter calls should return arrays, and we want to modify
these array.
When scalar values are expected (`cached_theta`, `cached_Phi`),
then the cache should implement setters (`cached_theta!`, `cached_Phi!`).

````julia
function init_value_caches(::AbstractMOP)::AbstractMOPCache
    return nothing
end
````

The function `init_value_caches` replaces previous pre-allocation methods,
i.e.,
`prealloc_objectives_vector(mop)`,
`prealloc_lin_eq_constraints_vector(mop)`,
`prealloc_lin_ineq_constraints_vector(mop)`,
`prealloc_nl_eq_constraints_vector(mop)`,
`prealloc_nl_ineq_constraints_vector(mop)`.
Internally, the safe-guarded `yoink_` methods are no longer needed, neither.

From the above definitions we can derive a cached evaluation function:

````julia
function eval_mop!(mop_cache, mop, scaler)
    unscale!(cached_ξ(mop_cache), scaler, cached_x(mop_cache))
    return eval_mop!(mop_cache, mop)
end

function eval_mop!(mop_cache, mop)
    # evaluate problem at unscaled site
    ξ = cached_ξ(mop_cache)
    @ignoraise θ, Φ = eval_mop!(
        cached_fx(mop_cache),
        cached_hx(mop_cache),
        cached_gx(mop_cache),
        cached_Ex_min_c(mop_cache),
        cached_Ex(mop_cache),
        cached_Ax_min_b(mop_cache),
        cached_Ax(mop_cache),
        mop, ξ
    )
    cached_theta!(mop_cache, θ)
    cached_Phi!(mop_cache, Φ)
    return nothing
end

"Evaluate `mop` at unscaled site `ξ` and modify result arrays in place."
function eval_mop!(
    fx, hx, gx, Ex_min_c, Ex, Ax_min_b, Ax, mop, ξ
)
    @ignoraise objectives!(fx, mop, ξ)
    @ignoraise nl_eq_constraints!(hx, mop, ξ)
    @ignoraise nl_ineq_constraints!(gx, mop, ξ)
    lin_eq_constraints!(Ex_min_c, Ex, mop, ξ)
    lin_ineq_constraints!(Ax_min_b, Ax, mop, ξ)
    θ = constraint_violation(hx, gx, Ex_min_c, Ax_min_b)
    Φ = maximum(fx)
    return (θ, Φ)
end

function constraint_violation(hx, gx, Ex_min_c, Ax_min_b)
    return max(
        constraint_violation(hx, Val(:eq)),
        constraint_violation(gx, Val(:ineq)),
        constraint_violation(Ex_min_c, Val(:eq)),
        constraint_violation(Ax_min_b, Val(:ineq)),
        0
    )
end

constraint_violation(::Nothing, type_val::Val)=0
constraint_violation(ex::RVec, ::Val{:eq})=maximum(abs.(ex); init=0)
constraint_violation(ix::RVec, ::Val{:ineq})=max(maximum(ix; init=0), 0)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

