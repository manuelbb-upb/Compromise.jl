# AbstractMOPSurrogate Interface
The speciality of our algorithm is its use of local surrogate models.
These should subtype and implement `AbstractMOPSurrogate`.
Every nonlinear function can be modelled, but we leave it to the implementation of
an `AbstractMOP`, how exactly that is done.

## Meta-Data
For convenience, we'd like to have the same meta information available
as for the original MOP:

````julia
float_type(::AbstractMOPSurrogate)::Type{<:AbstractFloat}=DEFAULT_FLOAT_TYPE
stop_type(::AbstractMOPSurrogate) = Any
dim_vars(::AbstractMOPSurrogate)::Int=-1
dim_objectives(::AbstractMOPSurrogate)::Int=-1            # mandatory
dim_nl_eq_constraints(::AbstractMOPSurrogate)::Int=0     # optional
dim_nl_ineq_constraints(::AbstractMOPSurrogate)::Int=0   # optional
````

Additionally, we require information on the model variability
and if we can build models for the scaled domain:

````julia
depends_on_radius(::AbstractMOPSurrogate)::Bool=true
````

## Construction
Define a function to return a model for some MOP.
The model does not yet have to be trained.

````julia
init_models(
    mop::AbstractMOP, scaler;
    delta_max::Union{Number,AbstractVector{<:Number}},
    require_fully_linear::Bool=true,
)::AbstractMOPSurrogate=nothing
````

It is trained with the update method `update_models!`.

!!! note
    This method should always return `nothing`, **unless** you want to stop the algorithm.

````julia
function update_models!(
    mod::AbstractMOPSurrogate, Δ, scaler, vals, scaled_cons;
    log_level::LogLevel, indent::Int
)
    return nothing
end

function process_trial_point!(mod::AbstractMOPSurrogate, vals_trial, was_accepted)
    nothing
end
````

## Evaluation

!!! note
    Methods are in-place, return values are ignored, except for `AbstractStoppingCriterion`.

Evaluation of nonlinear objective models requires the following method.
`x` will be from the scaled domain.

````julia
function eval_objectives!(y::RVec, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`eval_objectives!(y, mod, x) not implemented for mod of type $(M).")
end
````

If there are constraints, these have to be defined as well:

````julia
function eval_nl_eq_constraints!(y::RVec, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`eval_nl_eq_constraints!(y, mod, x) not implemented for mod of type $(M).")
end
function eval_nl_ineq_constraints!(y::RVec, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`eval_nl_ineq_constraints!(y, mod, x) not implemented for mod of type $(M).")
end
````

As before, we use shorter function names in the algorithm.

````julia
function objectives!(y::RVecOrNothing, mop::AbstractMOPSurrogate, x::RVec)
    eval_objectives!(y, mop, x)
end
function nl_eq_constraints!(y::RVecOrNothing, mop::AbstractMOPSurrogate, x::RVec)
    dim_nl_eq_constraints(mop) <= 0 && return nothing
    eval_nl_eq_constraints!(y, mop, x)
end
function nl_ineq_constraints!(y::RVecOrNothing, mop::AbstractMOPSurrogate, x::RVec)
    dim_nl_ineq_constraints(mop) <= 0 && return nothing
    eval_nl_ineq_constraints!(y, mop, x)
end
````

## Differentiation
The surrogate models are also used to query approximate derivative information.
We hence need the following functions to make `Dy` transposed model Jacobians:

````julia
function grads_objectives!(Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`grads_objectives!(Dy, mod, x) not implemented for mod of type $(M).")
end
````

If there are constraints, these have to be defined as well:

````julia
function grads_nl_eq_constraints!(Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`grads_nl_eq_constraints!(Dy, mod, x) not implemented for mod of type $(M).")
end
function grads_nl_ineq_constraints!(Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`grads_nl_ineq_constraints!(Dy, mod, x) not implemented for mod of type $(M).")
end
````

Here, the names of the wrapper functions start with “diff“.

````julia
function diff_objectives!(Dy::RMatOrNothing, mod::AbstractMOPSurrogate, x::RVec)
    return grads_objectives!(Dy, mod, x)
end
function diff_nl_eq_constraints!(Dy::RMatOrNothing, mod::AbstractMOPSurrogate, x::RVec)
    dim_nl_eq_constraints(mod) <= 0 && return nothing
    return grads_nl_eq_constraints!(Dy, mod, x)
end
function diff_nl_ineq_constraints!(Dy::RMatOrNothing, mod::AbstractMOPSurrogate, x::RVec)
    dim_nl_ineq_constraints(mod) <= 0 && return nothing
    return grads_nl_ineq_constraints!(Dy, mod, x)
end
````

Optionally, we can have evaluation and differentiation in one go:

````julia
function eval_and_grads_objectives!(y::RVec, Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    @ignoraise eval_objectives!(y, mod, x)
    @ignoraise grads_objectives!(Dy, mod, x)
    return nothing
end
function eval_grads_nl_eq_constraints!(y::RVec, Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    @ignoraise eval_nl_eq_constraints!(y, mod, x)
    @ignoraise grads_nl_eq_constraints!(Dy, mod, x)
    return nothing
end
function eval_grads_nl_ineq_constraints!(y::RVec, Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    @ignoraise eval_nl_ineq_constraints!(y, mod, x)
    @ignoraise grads_nl_ineq_constraints!(Dy, mod, x)
    return nothing
end
````

Wrappers for use in the algorithm:

````julia
function vals_diff_objectives!(y::RVecOrNothing, Dy::RMatOrNothing, mod::AbstractMOPSurrogate, x::RVec)
    eval_and_grads_objectives!(y, Dy, mod, x)
end
function vals_diff_nl_eq_constraints!(y::RVecOrNothing, Dy::RMatOrNothing, mod::AbstractMOPSurrogate, x::RVec)
    dim_nl_eq_constraints(mod) <= 0 && return nothing
    eval_and_grads_nl_eq_constraints!(y, Dy, mod, x)
end
function vals_diff_nl_ineq_constraints!(y::RVecOrNothing, Dy::RMatOrNothing, mod::AbstractMOPSurrogate, x::RVec)
    dim_nl_ineq_constraints(mod) <= 0 && return nothing
    eval_and_grads_nl_ineq_constraints!(y, Dy, mod, x)
end
````

### Pre-Allocation

Just like with an `AbstractMOP`, we want to pre-allocate and modify
evaluation arrays.
To this end, `init_value_caches(mod)` has to be implemented.
The returned object should be of type `AbstractMOPCache` and
adhere to the caching interface.
Whenever arrays are returned by a getter, expect them to be modified in place.

````julia
function init_value_caches(::AbstractMOPSurrogate)::AbstractMOPSurrogateCache
    return nothing
end
````

!!! note
    For nonlinear subproblem solvers it might be desirable to have partial evaluation
    and differentiation functions. Also, out-of-place functions could be useful for
    external nonlinear tools, but I don't need them yet.
    Defining the latter methods would simply call `prealloc_XXX` first and then use some
    in-place-functions.

Here is what is called later on:

````julia
"""
    eval_mod!(mod_cache::AbstractMOPSurrogateCache, mod::AbstractMOPSurrogate, x)

Evaluate `mod` at `x` and update cache `mod_cache`.
"""
function eval_mod!(mod_vals::AbstractMOPSurrogateCache, mod::AbstractMOPSurrogate, x)
    Base.copyto!(cached_x(mod_vals), x)
    @ignoraise objectives!(cached_fx(mod_vals), mod, x)
    @ignoraise nl_eq_constraints!(cached_hx(mod_vals), mod, x)
    @ignoraise nl_ineq_constraints!(cached_gx(mod_vals), mod, x)
    return nothing
end

"Evaluate the model gradients of `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`."
function diff_mod!(mod_vals::AbstractMOPSurrogateCache, mod::AbstractMOPSurrogate, x)
    Base.copyto!(cached_Dx(mod_vals), x)
    @ignoraise diff_objectives!(cached_Dfx(mod_vals), mod, x)
    @ignoraise diff_nl_eq_constraints!(cached_Dhx(mod_vals), mod, x)
    @ignoraise diff_nl_ineq_constraints!(cached_Dhx(mod_vals), mod, x)
    return nothing
end

"Evaluate and differentiate `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`."
function eval_and_diff_mod!(mod_vals::AbstractMOPSurrogateCache, mod, x)
    Base.copyto!(cached_x(mod_vals), x)
    Base.copyto!(cached_Dx(mod_vals), x)
    @ignoraise vals_diff_objectives!(cached_fx(mod_vals), cached_Dfx(mod_vals), mod, x)
    @ignoraise vals_diff_nl_eq_constraints!(cached_hx(mod_vals), cached_Dhx(mod_vals), mod, x)
    @ignoraise vals_diff_nl_ineq_constraints!(cached_gx(mod_vals), cached_Dgx(mod_vals), mod, x)
    return nothing
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

