#src This file is meant to be parsed by Literate.jl
# # Module `CompromiseEvaluators`
# This file provides a submodule defining abstract types and interfaces for evaluation
# of vector-vector-functions and surrogate models.

module CompromiseEvaluators #src

import Logging: Debug

# ## `AbstractNonlinearOperator` Interface
#=
An object subtyping `AbstractNonlinearOperator` represents a function mapping real-valued
vectors to real-valued vectors.
The interface defines methods to evaluate such a function.
These methods are used internally by Compromise, and we made the decision to assume
in-place functions.
If the user has out-of-place functions, they have to transform them accordingly.
Alternatively, this functionality can be provided by utility types implementing the interface.
=#

abstract type AbstractNonlinearOperator end

# A function can have parameters that are constant in a single optimization run, and 
# the type structure reflects this distinction:
abstract type AbstractNonlinearOperatorWithParams <: AbstractNonlinearOperator end
abstract type AbstractNonlinearOperatorNoParams <: AbstractNonlinearOperator end

const Vec = AbstractVector
const Mat = AbstractMatrix

# ### Common Methods
# Both, `AbstractNonlinearOperatorWithParams` and `AbstractNonlinearOperatorNoParams`
# have methods like `eval_op!`.
# The signatures look different, though.
# That is why there is a separate section for both types.
# The below methods have the same signature for both operator supertypes:
#
# Evaluation of derivatives is optional if evaluation-based models are used.
# We have functions to indicate if an operator implements `eval_grads!`, `eval_hessians!`:
provides_grads(op::AbstractNonlinearOperator)=false
provides_hessians(op::AbstractNonlinearOperator)=false

# In certain situations (nonlinear subproblems relying on minimization of scalar-valued 
# objective or constraint compoments) it might be beneficial if only certain outputs
# of a vector-function could be evaluated.
# The method `supports_partial_evaluation` signals this feature.
# If it returns `true`, the feature is assumed to be available for derivatives as well.
# In this situation, the type should implment methods starting with `partial_`, see below
# for details.
supports_partial_evaluation(op::AbstractNonlinearOperator) = false

# Stopping based on the number of evaluations is surprisingly hard if we don't
# want to give up most of the flexibility and composability of Operators and(/in) Problems
# and models.
# To implement such a stopping mechanism, we would like the operator to 
# count the number of evaluations.
is_counted(op::AbstractNonlinearOperator)=false
# If `is_counted` returns true, we assume that we can safely call
# `num_calls` and get a 3-tuple of values:
# the number of function evaluations, gradient evaluations and Hessian evaluations:
num_calls(op::AbstractNonlinearOperator)::NTuple{3, <:Integer}=nothing
# In addition, there should be a method to (re-)set the counters:
set_num_calls!(op::AbstractNonlinearOperator,vals::Tuple{Int,Int,Int})=nothing
# Stopping based on the number of evaluations is so fundamental, I make it 
# part of the interface:
const MCALLS = typemax(Int)
max_calls(op::AbstractNonlinearOperator)::NTuple{3, <:Integer}=(MCALLS, MCALLS, MCALLS)

# !!! note
#     Whether or not `max_calls` is respected depends on the implementation of 
#     `AbstractMOPSurrogate` or the implementation of `update!` for `AbstractSurrogateModel`...


# ### Methods for `AbstractNonlinearOperatorWithParams`
# !!! note 
#     The evaluation methods should respect `is_counted` and internally increase any counters.

# The methods below should be implemented to evaluate parameter dependent operators: 
"""
    eval_op!(y, op, x, p)

Evaluate the operator `op` at variable vector `x` with parameters `p`
and mutate the target vector `y` to contain the result.
"""
function eval_op!(y::Vec, op::AbstractNonlinearOperatorWithParams, x::Vec, p)
    return error("No implementation of `eval_op!` for operator $op.")
end

# Optional Parallel Evaluation:
function eval_op!(Y::AbstractMatrix, op::AbstractNonlinearOperatorWithParams, X::AbstractMatrix, p)
    for (x, y) = zip(eachcol(X), eachcol(Y))
        c = eval_op!(y, op, x)
        !isnothing(c) && return c
    end
    return nothing 
end

"""
    eval_grads!(Dy, op, x, p)

Compute the gradients of the operator `op` at variable vector `x` with parameters `p` 
and mutate the target matrix `Dy` to contain the gradients w.r.t. `x` in its columns. 
That is, `Dy` is the transposed Jacobian at `x`.
"""
function eval_grads!(Dy, op::AbstractNonlinearOperatorWithParams, x, p)
    return error("No implementation of `eval_grads!` for operator $op.")
end

# The combined forward-function `eval_op_and_grads!` is derived from `eval_op!` and 
# `eval_grads!`, but can be customized easily:
## helper macro `@ignoraise` instead of `return`
import ..Compromise: @ignoraise

function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x, p)
    @ignoraise eval_op!(y, val, x, p)
    @ignoraise eval_grads!(Dy, val, x, p)
    return nothing
end

# If Hessian matrices are needed, implement `eval_hessians!(H, op, x, p)`.
# Assume `H` to be a 3D array, where the last index iterates over the function outputs.
# That is, `H[:,:,1]` is a matrix containing second order partial derivatives of the first 
# output, `H[:,:,2]` has second order derivatives for the second output, and so forth...
# Moreover, in unlike with `eval_grads!`, the Hessian information should be stored in 
# correct order - `H[i,j,k]` should correspond to `∂yₖ/∂xᵢ∂xⱼ`.
# After `eval_grads!(D, op, x, p)`, the column `D[:,k]` contains partial derivatives 
# `∂₁yₖ, ∂₂yₖ, …, ∂ₘyₖ`, the gradient of `y[k]`.
# After `eval_hessians!(H, op, x, p)`, the “column” `H[:, j, k]` contains
# `∂₁(∂ⱼyₖ), ∂₂(∂ⱼyₖ), …, ∂ₘ(∂ⱼyₖ)`, the gradient of `Dy[j, k]`.

"""
    eval_hessians!(H, op, x, p)

Compute the Hessians of the operator `op` at variable vector `x` with parameters `p` 
and mutate the target array `H` to contain the Hessians along its last index.
That is, `H[:,:,i]` is the Hessian at `x` and `p` w.r.t. `x` of output `i`.
"""
function eval_hessians!(H, op::AbstractNonlinearOperatorWithParams, x, p)
    return error("No implementation of `eval_hessians!` for operator $op.")
end

# The combined forward-function `eval_op_and_grads_and_hessians!` 
# is derived from `eval_op_and_grads!` and `eval_hessians!`,
# but can be customized easily:
function eval_op_and_grads!(y, Dy, H, op::AbstractNonlinearOperatorWithParams, x, p)
    @ignoraise eval_op_and_grads!(Dy, y, op, x, p)
    @ignoraise eval_hessians!(H, val, x, p)
    return nothing
end

# Some operators might support partial evaluation. 
# They should implement these methods, if `supports_partial_evaluation` returns `true`.
# The argument `outputs` is an iterable of output indices, assuming `1` to be the first output.
# `y` is the full length vector, and `partial_op!` should set `y[outputs]`.
function partial_op!(y, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    ## length(y)==length(outputs)
    return error("Partial evaluation not implemented.")
end
function partial_grads!(Dy, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    ## size(Dy)==(length(x), length(outputs))
    return error("Partial Jacobian not implemented.")
end
function partial_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    @ignoraise partial_op!(y, op, x, p, outputs)
    @ignoraise partial_grads!(Dy, op, x, p, outputs)
    return nothing
end
function partial_hessians!(H, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    return error("Partial Hessians not implemented.")
end
function partial_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    @ignoraise partial_op_and_grads!(y, Dy, op, x, p, outputs)
    @ignoraise partial_hessians!(H, op, x, p, outputs)
    return nothing
end

# From the above, we derive safe-guarded functions, that can be used to pass `outputs`
# whenever convenient.
# Note, that these are defined for `AbstractNonlinearOperator`.
# By implementing the parametric-interface for `AbstractNonlinearOperatorNoParams`, 
# they work out-of-the box for non-paremetric operators, too:
import ..Compromise: AbstractStoppingCriterion, stop_message
struct BudgetExhausted <: AbstractStoppingCriterion
    ni :: Int
    mi :: Int
    order :: Int
end
function BudgetExhausted(op, order)
    mcalls = max_calls(op)
    ncalls = num_calls(op)
    i = order + 1
    return BudgetExhausted(ncalls[i], mcalls[i], order)
end
function stop_message(crit::BudgetExhausted)
    return "Maximum evaluation count reached, order=$(crit.order), is=$(crit.ni), max=$(crit.mi)."
end

function check_num_calls(op, order::Integer)::Union{Nothing, BudgetExhausted}
    order < 0 && return nothing
    order > 2 && return nothing
    !is_counted(op) && return nothing
    call_budget = budget_num_calls(op, order)
    isnothing(call_budget) || call_budget > 0 && return nothing
    return BudgetExhausted(op, order)
end

function budget_num_calls(op, order::Integer, do_checks::Val{true})
    order < 0 && return nothing
    order > 2 && return nothing
    !is_counted(op) && return nothing
    return budget_num_calls(op, order)
end

function budget_num_calls(op, order::Integer)
    mcalls = max_calls(op)
    ncalls = num_calls(op)
    if isnothing(ncalls)
        @warn "`is_counted(op), but `num_calls(op)` is nothing."
        return nothing
    end
    i = order + 1
    return mcalls[i] - ncalls[i] 
end

function func_vals!(
    y::AbstractVector, op::AbstractNonlinearOperator, x::AbstractVector, p; 
    outputs=nothing
)
    @ignoraise check_num_calls(op, 0)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        return partial_op!(y, op, x, p, outputs)
    end
    return eval_op!(y, op, x, p)
end

function func_vals!(
    y::AbstractMatrix, op::AbstractNonlinearOperator, x::AbstractMatrix, p; 
    outputs=nothing
)
    n_rem = budget_num_calls(op, 0, Val(true))
    if n_rem <= 0
        return BudgetExhausted(op, 0)
    end
    n_x = size(x, 2)
    Y, X = if n_rem < n_x
        @view(y[:, 1:n_rem]), @view(x[:, 1:n_rem])
    else
        y, x
    end
    return eval_op!(Y, op, X, p)
end

function func_grads!(
    Dy, op::AbstractNonlinearOperator, x, p; 
    outputs=nothing
)
    @ignoraise check_num_calls(op, 0)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        return partial_grads!(Dy, op, x, p, outputs)
    end
    return eval_grads!(Dy, op, x, p)
end
function func_vals_and_grads!(
    y, Dy, op::AbstractNonlinearOperator, x, p; 
    outputs=nothing
)
    @ignoraise check_num_calls(op, 0)
    @ignoraise check_num_calls(op, 1)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        return partial_op_and_grads!(y, Dy, op, x, p, outputs)
    end
    return eval_op_and_grads!(y, Dy, op, x, p)
end
function func_hessians!(
    H, op::AbstractNonlinearOperator, x, p; 
    outputs=nothing
)
    @ignoraise check_num_calls(op, 2)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        return partial_hessians!(H, op, x, p, outputs)
    end
    return eval_hessians!(H, op, x, p)
end
function func_vals_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperator, x, p; 
    outputs=nothing
)
    @ignoraise check_num_calls(op, 0)
    @ignoraise check_num_calls(op, 1)
    @ignoraise check_num_calls(op, 2)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        return partial_op_and_grads_and_hessians!(y, Dy, H, op, x, p, outputs)
    end
    return eval_op_and_grads_and_hessians!(y, Dy, H, op, x, p)
end

# ### Methods `AbstractNonlinearOperatorNoParams`
# 
# The interface for operators without parameters is very similar to what's above.
# In fact, we could always treat it as a special case of `AbstractNonlinearOperatorWithParams`
# and simply ignore the parameter vector.
# However, in some situation it might be desirable to have the methods without `p`
# readily at hand.
# This also makes writing extensions a tiny bit easier.

#src The below methods have been written by ChatGPT according to what is above:
function eval_op!(y::Vec, op::AbstractNonlinearOperatorNoParams, x::Vec)
    return error("No implementation of `eval_op!` for operator $op.")
end
function eval_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x)
    return error("No implementation of `eval_grads!` for operator $op.")
end
# Optional Parallel Evaluation:
function eval_op!(Y::Mat, op::AbstractNonlinearOperatorWithParams, X::Mat)
    for (x, y) = zip(eachcol(X), eachcol(Y))
        c = eval_op!(y, op, x)
        !isnothing(c) && return c
    end
    return nothing 
end
# Optional, derived method for values and gradients:
function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x)
    @ignoraise eval_op!(y, op, x)
    @ignoraise eval_grads!(Dy, op, x)
    return nothing
end
# Same for Hessians:
function eval_hessians!(H, op::AbstractNonlinearOperatorNoParams, x)
    return error("No implementation of `eval_hessians!` for operator $op.")
end
function eval_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorNoParams, x)
    @ignoraise eval_op_and_grads!(y, Dy, op, x)
    @ignoraise eval_hessians!(H, op, x)
    return nothing
end
# Some operators might support partial evaluation. 
# They should implement these methods, if `supports_partial_evaluation` returns `true`:
function partial_op!(y, op::AbstractNonlinearOperatorNoParams, x, outputs)
    return error("Partial evaluation not implemented.")
end
function partial_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x, outputs)
    return error("Partial Jacobian not implemented.")
end
function partial_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x, outputs)
    @ignoraise partial_op!(y, op, x, outputs)
    @ignoraise partial_grads!(Dy, op, x, outputs)
    return nothing
end
function partial_hessians!(H, op::AbstractNonlinearOperatorNoParams, x, outputs)
    return error("Partial Hessians not implemented.")
end
function partial_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorNoParams, x, outputs)
    @ignoraise partial_op_and_grads!(y, Dy, op, x, outputs)
    @ignoraise partial_hessians!(H, op, x, outputs)
    return nothing
end

# #### Parameter-Methods for Non-Parametric Operators
# To also be able to use non-parametric operators in the more general setting, 
# implement the parametric-interface:

function eval_op!(y::Vec, op::AbstractNonlinearOperatorNoParams, x::Vec, p)
    return eval_op!(y, op, x)
end
function eval_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x, p)
    return eval_grads!(Dy, op, x)
end
function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x, p)
    return eval_op_and_grads!(y, Dy, op, x)
end
function eval_hessians!(H, op::AbstractNonlinearOperatorNoParams, x, p)
    return eval_hessians!(H, op, x)
end
function eval_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorNoParams, x, p)
    return eval_op_and_grads_and_hessians!(y, Dy, H, op, x)
end

# Partial evaluation or differentiation:
function partial_op!(y, op::AbstractNonlinearOperatorNoParams, x, p, outputs)
    return partial_op!(y, op, x, outputs)
end
function partial_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x, p, outputs)
    return partial_grads!(Dy, op, x, outputs)
end
function partial_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x, p, outputs)
    return partial_op_and_grads!(y, Dy, op, x, outputs)
end
function partial_hessians!(H, op::AbstractNonlinearOperatorNoParams, x, p, outputs)
    return partial_hessians!(H, op, x, outputs)
end
function partial_op_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperatorNoParams, x, p, outputs
)
    return partial_op_and_grads_and_hessians!(y, Dy, H, op, x, outputs)
end

# The safe-guarded methods can now simply forward to the parametric versions 
# and pass `nothing` parameters:
function func_vals!(
    y, op::AbstractNonlinearOperator, x; 
    outputs=nothing
)
    return func_vals!(y, op, x, nothing; outputs)
end
function func_grads!(
    Dy, op::AbstractNonlinearOperator, x; 
    outputs=nothing
)
    return func_grads!(Dy, op, x, nothing; outputs)
end
function func_vals_and_grads!(
    y, Dy, op::AbstractNonlinearOperator, x; 
    outputs=nothing
)
    return func_vals_and_grads!(y, Dy, op, x, nothing; outputs)
end
function func_hessians!(
    H, op::AbstractNonlinearOperator, x; 
    outputs=nothing
)
    return func_hessians!(H, op, x, nothing; outputs)
end
function func_vals_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperator, x; 
    outputs=nothing
)
    return func_vals_and_grads_and_hessians!(y, Dy, H, op, x, nothing; outputs)
end
# #### Non-Parametric Methods for Parametric Operators
# These should only be used when you know what you are doing, as we set the parameters
# to `nothing`.
# This is safe only if we now that an underlying function is not-parametric, but somehow
# wrapped in something implementing `AbstractNonlinearOperatorWithParams`.
function eval_op!(y, op::AbstractNonlinearOperatorWithParams, x)
    return eval_op!(y, op, x, nothing)
end 
function eval_grads!(Dy, op::AbstractNonlinearOperatorWithParams, x)
    return eval_grads!(Dy, op, x, nothing)
end

function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x)
    return eval_op_and_grads!(y, Dy, op, x, nothing)
end

function eval_hessians!(H, op::AbstractNonlinearOperatorWithParams, x)
    return eval_hessians!(H, op, x, nothing)
end

function eval_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorWithParams, x)
    return eval_op_and_grads_and_hessians!(y, Dy, H, op, x, nothing)
end

function partial_op!(y, op::AbstractNonlinearOperatorWithParams, x, outputs)
    return partial_op!(y, op, x, nothing, outputs)
end
function partial_grads!(Dy, op::AbstractNonlinearOperatorWithParams, x, outputs)
    return partial_grads!(Dy, op, x, nothing, outputs)
end
function partial_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x, outputs)
    return partial_op_and_grads!(y, Dy, op, x, nothing, outputs)
end
function partial_hessians!(H, op::AbstractNonlinearOperatorWithParams, x, outputs)
    return partial_hessians!(H, op, x, nothing, outputs)
end
function partial_op_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperatorWithParams, x, outputs
)
    return partial_op_and_grads_and_hessians!(y, Dy, H, op, x, nothing, outputs)
end

# ## Types and Methods for Surrogate Models

# An `AbstractSurrogateModel` is similar to `AbstractNonlinearOperator`.
# Such a surrogate model is always non-parametric, as parameters of operators are assumed 
# to be fix in between runs.
abstract type AbstractSurrogateModel end

# We also want to be able to define the behavior of models with light-weight objects:
abstract type AbstractSurrogateModelConfig end

# ### Indicator Methods
# Functions to indicate the order of the surrogate model:
requires_grads(::AbstractSurrogateModelConfig)=false
requires_hessians(::AbstractSurrogateModelConfig)=false

# A function to indicate that a model should be updated when the trust region has changed:
depends_on_radius(::AbstractSurrogateModel)=true

# ### Initialization and Modification

# The choice to don't separate between a model and its parameters (like `Lux.jl` does) 
# is historic.
# There are pros and cons to both approaches.
# The most obvious point in favor of how it is now are the unified evaluation interfaces.
# However, for the Criticality Routine we might need to copy models and retrain them for 
# smaller trust-region radii.
# That is why we require implementation of `copy_model(source_model)` and 
# `copyto_model!(trgt_model, src_model)`.
# A modeller should take care to really only copy parameter arrays and pass other large
# objects, such as databases, by reference so as to avoid a large memory-overhead.
# Moreover, we only need copies for radius-dependent models!
# You can ignore those methods otherwise.

# A surrogate is initialized from its configuration and the operator it is meant to model:
"""
    init_surrogate(
        model_config, nonlin_op, dim_in, dim_out, params, T
    )

Return a model subtyping `AbstractSurrogateModel`, as defined by 
`model_config::AbstractSurrogateModelConfig`, for the nonlinear operator `nonlin_op`.
The operator (and model) has input dimension `dim_in` and output dimension `dim_out`.
`params` is the current parameter object for `nonlin_op` and is cached.
`T` is a subtype of `AbstractFloat` to indicate precision of cache arrays.
"""
function init_surrogate(
    ::AbstractSurrogateModelConfig, op, dim_in, dim_out, params, T;
    require_fully_linear::Bool=true, 
    delta_max::Union{Number, AbstractVector{<:Number}}=Inf,
)::AbstractSurrogateModel
    return nothing
end

# A function to return a copy of a model. Should be implemented if 
# `depends_on_radius` returns `true`.
# Note, that the returned object does not have to be an “independent” copy, we allow 
# for shared objects (like mutable database arrays or something of that sort)...
copy_model(mod_src)=deepcopy(mod_src)

# A function to copy parameters between source and target models, like `Base.copy!` or 
# `Base.copyto!`. Relevant mostly for trainable parameters.
copyto_model!(mod_trgt::AbstractSurrogateModel, mod_src::AbstractSurrogateModel)=mod_trgt

function _copy_model(mod)
    depends_on_radius(mod) && return copy_model(mod)
    return mod
end

function _copyto_model!(mod_trgt, mod_src)
    depends_on_radius(mod_trgt) && return copyto_model!(mod_trgt, mod_src)
    return mod_trgt
end

# Because parameters are implicit, updates are in-place operations:
"""
    update!(surrogate_model, nonlinear_operator, Δ, x, fx, lb, ub)

Update the model on a trust region of size `Δ` in a box with lower left corner `lb`
and upper right corner `ub` (in the scaled variable domain)
`x` is a sub-vector of the current iterate conforming to the inputs of `nonlinear_operator`
in the scaled domain. `fx` are the outputs of `nonlinear_operator` at `x`.
"""
function update!(
    surr::AbstractSurrogateModel, op, Δ, x, fx, lb, ub; log_level, kwargs...
)
    return nothing    
end

# ### Evaluation
# In place evaluation and differentiation, similar to `AbstractNonlinearOperatorNoParams`.
# Mandatory:
function model_op!(y::Vec, surr::AbstractSurrogateModel, x::Vec)
    return nothing
end
# Mandatory:
function model_grads!(Dy, surr::AbstractSurrogateModel, x)
    return nothing
end
# Optional:
function model_op_and_grads!(y, Dy, surr::AbstractSurrogateModel, x)
    model_op!(y, surr, x )
    model_grads!(Dy, surr, x)
    return nothing
end
# #### Optional Partial Evaluation
supports_partial_evaluation(::AbstractSurrogateModel)=false
function model_op!(y, surr::AbstractSurrogateModel, x, outputs)
    return error("Surrogate model does not support partial evaluation.")
end
function model_grads!(y, surr::AbstractSurrogateModel, x, outputs)
    return error("Surrogate model does not support partial Jacobian.")
end
function model_op_and_grads!(y, Dy, surr::AbstractSurrogateModel, x, outputs)
    model_op!(y, surr, x, outputs)
    model_grads!(y, surr, x, outputs)
    return nothing
end

# #### Optional Parallel Evaluation
function model_op!(Y::Mat, surr::AbstractSurrogateModel, X::Mat)
    for (x, y) = zip(eachcol(X), eachcol(Y))
        c = model_op!(y, surr, x)
        if !isnothing(c)
            return c
        end
    end
    return nothing
end

# #### Safe-guarded, internal Methods
# The methods below are used in the algorithm and have the same signature as 
# the corresponding methods for `AbstractNonlinearOperator`.
# Thus, we do not have to distinguish types in practice.
function func_vals!(y, surr::AbstractSurrogateModel, x, p=nothing;
    outputs=nothing
)
    if !isnothing(outputs)
        if supports_partial_evaluation(surr)
            return model_op!(y, surr, x, outputs)
        ## else
        ##     @warn "Partial evaluation not supported by surrogate model."
        end
    end
    return model_op!(y, surr, x)
end
function func_grads!(
    Dy, surr::AbstractSurrogateModel, x, p=nothing,
    outputs=nothing
)
    if !isnothing(outputs)
        if supports_partial_evaluation(surr)
            return model_grads!(Dy, surr, x, outputs)
        ## else
        ##     @warn "Partial evaluation not supported by surrogate model."
        end
    end
    return model_grads!(Dy, surr, x)
end
function func_vals_and_grads!(
    y, Dy, surr::AbstractSurrogateModel, x, p=nothing;
    outputs=nothing
)
    if !isnothing(outputs)
        if supports_partial_evaluation(surr)
            return model_op_and_grads!(y, Dy, surr, x, outputs)
        ## else
        ##     @warn "Partial evaluation not supported by surrogate model."
        end
    end
    return model_op_and_grads!(y, Dy, surr, x)
end

# ## Module Exports
export AbstractNonlinearOperator, AbstractNonlinearOperatorNoParams, AbstractNonlinearOperatorWithParams
export AbstractSurrogateModel, AbstractSurrogateModelConfig
export supports_partial_evaluation, provides_grads, provides_hessians, requires_grads, requires_hessians
export func_vals!, func_grads!, func_hessians!, func_vals_and_grads!, func_vals_and_grads_and_hessians!
export eval_op!, eval_grads!, eval_hessians!, eval_op_and_grads!, eval_op_and_grads_and_hessians!
export func_vals!, func_grads!, func_vals_and_grads!, func_vals_and_grads_and_hessians!
export model_op!, model_grads!, model_op_and_grads!
export init_surrogate, update!

end#module