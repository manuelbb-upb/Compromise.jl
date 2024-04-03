
# # AbstractMOPSurrogate Interface
# The speciality of our algorithm is its use of local surrogate models.
# These should subtype and implement `AbstractMOPSurrogate`.
# Every nonlinear function can be modelled, but we leave it to the implementation of 
# an `AbstractMOP`, how exactly that is done.

# ## Meta-Data
# For convenience, we'd like to have the same meta information available 
# as for the original MOP:
float_type(::AbstractMOPSurrogate)::Type{<:AbstractFloat}=DEFAULT_FLOAT_TYPE
dim_vars(::AbstractMOPSurrogate)::Int=-1
dim_objectives(::AbstractMOPSurrogate)::Int=-1            # mandatory
dim_nl_eq_constraints(::AbstractMOPSurrogate)::Int=0     # optional
dim_nl_ineq_constraints(::AbstractMOPSurrogate)::Int=0   # optional

# Additionally, we require information on the model variability
# and if we can build models for the scaled domain:
depends_on_radius(::AbstractMOPSurrogate)::Bool=true

# ## Construction
# Define a function to return a model for some MOP.
# The model does not yet have to be trained.
init_models(
    mop::AbstractMOP, scaler; 
    delta_max::Union{Number,AbstractVector{<:Number}},
    require_fully_linear::Bool=true,
)::AbstractMOPSurrogate=nothing
# It is trained with the update method `update_models!`.
#
# !!! note
#     This method should always return `nothing`, **unless** you want to stop the algorithm.

function update_models!(
    mod::AbstractMOPSurrogate, Δ, scaler, vals, scaled_cons;
    log_level::LogLevel, indent::Int
)
    return nothing
end

function process_trial_point!(mod::AbstractMOPSurrogate, vals_trial, update_results)
    nothing
end

# If a model is radius-dependent, 
# we also need a function to copy the parameters from a source model to a target model:
universal_copy(mod::AbstractMOPSurrogate)=mod
universal_copy!(mod_trgt::AbstractMOPSurrogate, mod_src::AbstractMOPSurrogate)=mod_trgt
# These internal helpers are derived:
function universal_copy_model(mod::AbstractMOPSurrogate)
    depends_on_radius(mod) ? universal_copy(mod) : mod
end
function universal_copy_model!(mod_trgt::AbstractMOPSurrogate, mod_src::AbstractMOPSurrogate)
    depends_on_radius(mod_trgt) ? universal_copy!(mod_trgt, mod_src) : mod_trgt
end

# ## Evaluation

# !!! note
#     Methods are in-place, return values are ignored, except for `AbstractStoppingCriterion`.

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
function objectives!(y::RVec, mop::AbstractMOPSurrogate, x::RVec)
    eval_objectives!(y, mop, x)
end
function nl_eq_constraints!(y::RVec, mop::AbstractMOPSurrogate, x::RVec)
    dim_nl_eq_constraints(mop) <= 0 && return nothing
    eval_nl_eq_constraints!(y, mop, x) 
end
function nl_ineq_constraints!(y::RVec, mop::AbstractMOPSurrogate, x::RVec)
    dim_nl_ineq_constraints(mop) <= 0 && return nothing
    eval_nl_ineq_constraints!(y, mop, x)
end

# ## Pre-Allocation
# The preallocation functions look the same as for `AbstractMOP`:
for (dim_func, func_suffix) in (
    (:dim_objectives, :objectives_vector),
    (:dim_nl_eq_constraints, :nl_eq_constraints_vector),
    (:dim_nl_ineq_constraints, :nl_ineq_constraints_vector),
)
    func_name = Symbol("prealloc_", func_suffix)
    yoink_name = Symbol("yoink_", func_suffix)
    @eval begin 
        function $(func_name)(mop::AbstractMOPSurrogate)
            dim = $(dim_func)(mop) 
            T = float_type(mop)
            return Vector{T}(undef, dim)
        end

        function $(yoink_name)(mop::AbstractMOPSurrogate)
            y = $(func_name)(mop) :: RVec
            @assert eltype(y) == float_type(mop) "Vector eltype does not match problem float type."
            return y
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
function diff_nl_eq_constraints!(Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)
    dim_nl_eq_constraints(mod) <= 0 && return nothing
    return grads_nl_eq_constraints!(Dy, mod, x)
end
function diff_nl_ineq_constraints!(Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)
    dim_nl_ineq_constraints(mod) <= 0 && return nothing
    return grads_nl_ineq_constraints!(Dy, mod, x)
end

# Optionally, we can have evaluation and differentiation in one go:
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
# Wrappers for use in the algorithm:
function vals_diff_objectives!(y::RVec, Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)
    eval_and_grads_objectives!(y, Dy, mod, x)
end
function vals_diff_nl_eq_constraints!(y::RVec, Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)
    dim_nl_eq_constraints(mod) <= 0 && return nothing
    eval_and_grads_nl_eq_constraints!(y, Dy, mod, x)
end
function vals_diff_nl_ineq_constraints!(y::RVec, Dy::RMat, mod::AbstractMOPSurrogate, x::RVec)
    dim_nl_ineq_constraints(mod) <= 0 && return nothing
    eval_and_grads_nl_ineq_constraints!(y, Dy, mod, x)
end

# Here is what is called later on:
"Evaluate the models `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`."
function eval_mod!(mod_vals, mod, x)
    @unpack fx, hx, gx = mod_vals
    @ignoraise objectives!(fx, mod, x)
    @ignoraise nl_eq_constraints!(hx, mod, x)
    @ignoraise nl_ineq_constraints!(hx, mod, x)
    return nothing
end

"Evaluate the model gradients of `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`."
function diff_mod!(mod_vals, mod, x)
    @unpack Dfx, Dhx, Dgx = mod_vals
    @ignoraise diff_objectives!(Dfx, mod, x)
    @ignoraise diff_nl_eq_constraints!(Dhx, mod, x)
    @ignoraise diff_nl_ineq_constraints!(hx, mod, x)
    return nothing
end

"Evaluate and differentiate `mod` at `x` and store results in `mod_vals::SurrogateValueArrays`."
function eval_and_diff_mod!(mod_vals, mod, x)
    @unpack fx, hx, gx, Dfx, Dhx, Dgx = mod_vals
    @ignoraise vals_diff_objectives!(fx, Dfx, mod, x)
    @ignoraise vals_diff_nl_eq_constraints!(hx, Dhx, mod, x)
    @ignoraise vals_diff_nl_ineq_constraints!(gx, Dgx, mod, x)
    return nothing
end

# ### Gradient Pre-Allocation
# We also would like to have pre-allocated gradient arrays ready:
## These are defined below (and I put un-specific definitions here for the Linter)
function prealloc_objectives_grads(mod) end
function prealloc_nl_eq_constraints_grads(mod) end
function prealloc_nl_ineq_constraints_grads(mod) end
for (dim_func, func_suffix) in (
    (:dim_objectives, :objectives_grads),
    (:dim_nl_eq_constraints, :nl_eq_constraints_grads),
    (:dim_nl_ineq_constraints, :nl_ineq_constraints_grads),
)   
    func_name = Symbol("prealloc_", func_suffix)
    yoink_name = Symbol("yoink_", func_suffix)
    @eval begin
        function $(func_name)(mod::AbstractMOPSurrogate)
            n_out = $(dim_func)(mod)
            n_in = dim_vars(mod)
            T = float_type(mod)
            return Matrix{T}(undef, n_in, n_out)
        end

        function $(yoink_name)(mod::AbstractMOPSurrogate)
            y = $(func_name)(mod) :: RMat
            @assert eltype(y) == float_type(mod) "Vector eltype does not match problem float type."
            return y
        end
    end
end

# !!! note 
#     For nonlinear subproblem solvers it might be desirable to have partial evaluation
#     and differentiation functions. Also, out-of-place functions could be useful for 
#     external nonlinear tools, but I don't need them yet.
#     Defining the latter methods would simply call `prealloc_XXX` first and then use some
#     in-place-functions. 
