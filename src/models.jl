
# # AbstractMOPSurrogate Interface
# The speciality of our algorithm is its use of local surrogate models.
# These should subtype and implement `AbstractMOPSurrogate`.
# Every nonlinear function can be modelled, but we leave it to the implementation of 
# an `AbstractMOP`, how exactly that is done.

# ## Meta-Data
# For convenience, we'd like to have the same meta information available 
# as for the original MOP:
precision(::AbstractMOPSurrogate)::Type{<:AbstractFloat}=DEFAULT_PRECISION
dim_objectives(::AbstractMOPSurrogate)::Int=0            # mandatory
dim_nl_eq_constraints(::AbstractMOPSurrogate)::Int=0     # optional
dim_nl_ineq_constraints(::AbstractMOPSurrogate)::Int=0   # optional

# Additionally, we require information on the model variability
# and if we can build models for the scaled domain:
depends_on_radius(::AbstractMOPSurrogate)::Bool=true
supports_scaling(T::Type{<:AbstractMOPSurrogate})=NoScaling()

# ## Construction
# Define a function to return a model for some MOP.
# The model does not yet have to be trained.
init_models(mop::AbstractMOP, n_vars, scaler)::AbstractMOPSurrogate=nothing
# It is trained with the update method.
function update_models!(
    mod::AbstractMOPSurrogate, Δ, mop, scaler, vals, scaled_cons, algo_opts
)
    return nothing
end

# If a model is radius-dependent, 
# we also need a function to copy the parameters from a source model to a target model:
copy_model(mod::AbstractMOPSurrogate)=deepcopy(mod)
copyto_model!(mod_trgt::AbstractMOPSurrogate, mod_src::AbstractMOPSurrogate)=mod_trgt
_copy_model(mod::AbstractMOPSurrogate)=depends_on_radius(mod) ? copy_model(mod) : mod
_copyto_model!(mod_trgt::AbstractMOPSurrogate, mod_src::AbstractMOPSurrogate)=depends_on_radius(mod_trgt) ? copyto_model!(mod_trgt, mod_src) : mod_trgt

# ## Evaluation
# Evaluation of nonlinear objective models requires the following method.
# `x` will be from the scaled domain, but if a model does not support scaling, 
# then internally the `IdentityScaler()` is used:
function eval_objectives!(y::RVec, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`eval_objectives!(y, mod, x) not implemented for mod of type $(M).")
end
# If there are constraints, these have to be defined as well:
function eval_nl_eq_constraints!(y::RVec, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`eval_nl_eq_constraints!(y, mod, x) not implemented for mod of type $(M).")
end
function eval_nl_ineq_constraints!(y::RVec, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`eval_nl_ineq_constraints!(y, mod, x) not implemented for mod of type $(M).")
end

# As before, we use shorter function names in the algorithm.
objectives!(y::RVec, mod::AbstractMOPSurrogate, x::RVec)=eval_objectives!(y, mod, x)
nl_eq_constraints!(y::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing
nl_ineq_constraints!(y::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing
nl_eq_constraints!(y::RVec, mod::AbstractMOPSurrogate, x::RVec)=eval_nl_eq_constraints!(y, mod, x)
nl_ineq_constraints!(y::RVec, mod::AbstractMOPSurrogate, x::RVec)=eval_nl_ineq_constraints!(y, mod, x)

# ## Pre-Allocation
# The preallocation functions look the same as for `AbstractMOP`:
for (dim_func, prealloc_func) in (
    (:dim_objectives, :prealloc_objectives_vector),
    (:dim_nl_eq_constraints, :prealloc_nl_eq_constraints_vector),
    (:dim_nl_ineq_constraints, :prealloc_nl_ineq_constraints_vector),
)
    @eval function $(prealloc_func)(mod::AbstractMOPSurrogate)
        dim = $(dim_func)(mod) 
        if dim > 0
            T = precision(mod)
            return Vector{T}(undef, dim)
        else
            return nothing
        end
    end
end

# ## Differentiation
# The surrogate models are also used to query approximate derivative information.
# We hence need the following functions to make `Dy` transposed model Jacobians:
function grads_objectives!(Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`grads_objectives!(Dy, mod, x) not implemented for mod of type $(M).")
end
# If there are constraints, these have to be defined as well:
function grads_nl_eq_constraints!(Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`grads_nl_eq_constraints!(Dy, mod, x) not implemented for mod of type $(M).")
end
function grads_nl_ineq_constraints!(Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    error("`grads_nl_ineq_constraints!(Dy, mod, x) not implemented for mod of type $(M).")
end

# Here, the names of the wrapper functions start with “diff“.
diff_objectives!(Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=grads_objectives!(Dy, mod, x)
diff_nl_eq_constraints!(Dy::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing
diff_nl_ineq_constraints!(Dy::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing
diff_nl_eq_constraints!(Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=grads_nl_eq_constraints!(Dy, mod, x)
diff_nl_ineq_constraints!(Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=grads_nl_ineq_constraints!(Dy, mod, x)

# Optionally, we can have evaluation and differentiation in one go:
function eval_and_grads_objectives!(y::RVec, Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    eval_objectives!(y, mop, x)
    grads_objectives!(Dy, mod, x)
    return nothing
end
function eval_grads_nl_eq_constraints!(y::RVec, Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    eval_nl_eq_constraints!(y, mop, x)
    grads_nl_eq_constraints!(Dy, mod, x)
    return nothing
end
function eval_grads_nl_ineq_constraints!(y::RVec, Dy::RMat, mod::M, x::RVec) where {M<:AbstractMOPSurrogate}
    eval_nl_ineq_constraints!(y, mop, x)
    grads_nl_ineq_constraints!(Dy, mod, x)
    return nothing
end
# Wrappers for use in the algorithm:
vals_diff_objectives!(y::RVec, Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=eval_and_grads_objectives!(y, Dy, mod, x)
vals_diff_nl_eq_constraints!(y::Nothing, Dy::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing
vals_diff_nl_ineq_constraints!(y::Nothing, Dy::Nothing, mod::AbstractMOPSurrogate, x::RVec)=nothing
vals_diff_nl_eq_constraints!(y::RVec, Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=eval_and_grads_nl_eq_constraints!(y, Dy, mod, x)
vals_diff_nl_ineq_constraints!(y::RVec, Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)=eval_and_grads_nl_ineq_constraints!(y, Dy, mod, x)

# Here is what is called later on:
"Evaluate the models `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`."
function eval_mod!(mod_vals, mod, x)
    @unpack fx, hx, gx = mod_vals
    objectives!(fx, mod, x)
    nl_eq_constraints!(hx, mod, x)
    nl_ineq_constraints!(hx, mod, x)
    return nothing
end

"Evaluate the model gradients of `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`."
function diff_mod!(mod_vals, mod, x)
    @unpack Dfx, Dhx, Dgx = mod_vals
    diff_objectives!(Dfx, mod, x)
    diff_nl_eq_constraints!(Dhx, mod, x)
    diff_nl_ineq_constraints!(hx, mod, x)
    return nothing
end

"Evaluate and differentiate `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`."
function eval_and_diff_mod!(mod_vals, mod, x)
    @unpack fx, hx, gx, Dfx, Dhx, Dgx = mod_vals
    vals_diff_objectives!(fx, Dfx, mod, x)
    vals_diff_nl_eq_constraints!(hx, Dhx, mod, x)
    vals_diff_nl_ineq_constraints!(gx, Dgx, mod, x)
    return nothing
end

# ### Gradient Pre-Allocation
# We also would like to have pre-allocated gradient arrays ready:
function prealloc_objectives_grads(mod::AbstractMOPSurrogate, n_vars)
    T = precision(mod)
    return Matrix{T}(undef, n_vars, dim_objectives(mod))
end
## These are defined below (and I put un-specific definitions here for the Linter)
function prealloc_nl_eq_constraints_grads(mod, n_vars) end
function prealloc_nl_ineq_constraints_grads(mod, n_vars) end
for (dim_func, prealloc_func) in (
    (:dim_nl_eq_constraints, :prealloc_nl_eq_constraints_grads),
    (:dim_nl_ineq_constraints, :prealloc_nl_ineq_constraints_grads),
)
    @eval function $(prealloc_func)(mod::AbstractMOPSurrogate, n_vars)
        n_out = $(dim_func)(mod)
        if n_out > 0
            T = precision(mod)
            return Matrix{T}(undef, n_vars, n_out)
        else
            return nothing
        end
    end
end

# !!! note 
#     For nonlinear subproblem solvers it might be desirable to have partial evaluation
#     and differentiation functions. Also, out-of-place functions could be useful for 
#     external nonlinear tools, but I don't need them yet.
#     Defining the latter methods would simply call `prealloc_XXX` first and then use some
#     in-place-functions. 
