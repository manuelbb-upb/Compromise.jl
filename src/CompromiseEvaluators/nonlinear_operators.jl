abstract type AbstractNonlinearOperator end

abstract type AbstractNonlinearOperatorWithParams <: AbstractNonlinearOperator end
abstract type AbstractNonlinearOperatorNoParams <: AbstractNonlinearOperator end

#=
# # `AbstractNonlinearOperator` Interface
The `AbstractNonlinearOperator` interface is designed to best suit the needs of the 
evaluation tree structure.
For a user, this might make its implementation a bit awkward or counter-intuitive,
because we want evaluation and differentiation calls to be mutating.
For example, evaluation of a function requires a function with signature similar to 
`func!(y, x)`, where the target array is a view into some pre-allocated memory.
We also assume, that it is preferrable to do evaluation and differentiation in one pass, 
whenever possible, so it is advised to overwrite `eval_op_and_grads!` and similar 
methods.
=#

# ## Common Methods
# Functions to indicate if an operator implements `eval_grads!`, `eval_hessians!`, etc ...
provides_grads(op::AbstractNonlinearOperator)=false
provides_hessians(op::AbstractNonlinearOperator)=false

# Indicate, if individual outputs of a vector-valued function can be queried seperately:
supports_partial_evaluation(op::AbstractNonlinearOperator) = false

# ## `AbstractNonlinearOperatorWithParams`
# The methods below should be implemented to evaluate parameter dependent operators: 
"""
    eval_op!(y, op, x, p)

Evaluate the operator `op` at variable vector `x` with parameters `p`
and mutate the target vector `y` to contain the result.
"""
function eval_op!(y, op::AbstractNonlinearOperatorWithParams, x, p)
    return error("No implementation of `eval_op!` for operator $op.")
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
function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x, p)
    eval_op!(y, val, x, p)
    eval_grads!(Dy, val, x, p)
    return nothing
end

# If Hessian matrices are needed, implement `eval_hessians!(H, op, x, p)`.
# Assume `H` to be a vector of matrices, each matrix corresponding to the hessian of 
# some operator output.
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
    eval_op_and_grads!(Dy, y, op, x, p)
    eval_hessians!(H, val, x, p)
    return nothing
end

# Some operators might support partial evaluation. 
# They should implement these methods, if `supports_partial_evaluation` returns `true`:
function partial_op!(y, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    ## length(y)==length(outputs)
    return error("Partial evaluation not implemented.")
end
function partial_grads!(Dy, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    ## size(Dy)==(length(x), length(outputs))
    return error("Partial Jacobian not implemented.")
end
function partial_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    partial_op!(y, op, x, p, outputs)
    partial_grads!(Dy, op, x, p, outputs)
    return nothing
end
function partial_hessians!(H, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    return error("Partial Hessians not implemented.")
end
function partial_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorWithParams, x, p, outputs)
    partial_op_and_grads!(y, Dy, op, x, p, outputs)
    partial_hessians!(H, op, x, p, outputs)
    return nothing
end

# From the above, we derive safe-guarded functions, that can be used to pass `outputs`
# whenever convenient.
# Note, that these are defined for `AbstractNonlinearOperator`.
# By implementing the parametric-interface for `AbstractNonlinearOperatorNoParams`, 
# they work out-of-the box for non-paremetric operators, too:
function func_vals!(y, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_op!(y, op, x, p, outputs)
    end
    return eval_op!(y, op, x, p)
end
function func_grads!(Dy, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_grads!(Dy, op, x, p, outputs)
    end
    return eval_grads!(Dy, op, x, p)
end
function func_vals_and_grads!(y, Dy, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_op_and_grads!(y, Dy, op, x, p, outputs)
    end
    return eval_op_and_grads!(y, Dy, op, x, p)
end
function func_hessians!(H, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_hessians!(H, op, x, p, outputs)
    end
    return eval_hessians!(H, op, x, p)
end
function func_vals_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        partial_op_and_grads_and_hessians!(y, Dy, H, op, x, p, outputs)
    end
    eval_op_and_grads_and_hessians!(y, Dy, H, op, x, p)
end

# ## `AbstractNonlinearOperatorNoParams`
# 
# The interface for operators without parameters is very similar to what's above.
# In fact, we could always treat it as a special case of `AbstractNonlinearOperatorWithParams`
# and simply ignore the parameter vector.
# However, in some situation it might be desirable to have the methods without `p`
# readily at hand.

#src The below methods have been written by ChatGPT according to what is above:
function eval_op!(y, op::AbstractNonlinearOperatorNoParams, x)
    return error("No implementation of `eval_op!` for operator $op.")
end
function eval_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x)
    return error("No implementation of `eval_grads!` for operator $op.")
end
# Optional, derived method for values and gradients:
function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x)
    eval_op!(y, op, x)
    eval_grads!(Dy, op, x)
    return nothing
end
# Same for Hessians:
function eval_hessians!(H, op::AbstractNonlinearOperatorNoParams, x)
    return error("No implementation of `eval_hessians!` for operator $op.")
end
function eval_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorNoParams, x)
    eval_op_and_grads!(y, Dy, op, x)
    eval_hessians!(H, op, x)
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
    partial_op!(y, op, x, outputs)
    partial_grads!(Dy, op, x, outputs)
    return nothing
end
function partial_hessians!(H, op::AbstractNonlinearOperatorNoParams, x, outputs)
    return error("Partial Hessians not implemented.")
end
function partial_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperatorNoParams, x, outputs)
    partial_op_and_grads!(y, Dy, op, x, outputs)
    partial_hessians!(H, op, x, outputs)
    return nothing
end

# The safe-guarded methods simply forward to the parametric versions and pass `nothing`
# parameters:
function func_vals!(y, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing)
    return func_vals!(y, op, x, nothing, outputs)
end
function func_grads!(Dy, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing)
    return func_grads!(Dy, op, x, nothing, outputs)
end
function func_vals_and_grads!(y, Dy, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing)
    return func_vals_and_grads!(y, Dy, op, x, nothing, outputs)
end
function func_hessians!(H, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing)
    return func_hessians!(H, op, x, nothing, outputs)
end
function func_vals_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperatorNoParams, x, outputs=nothing
)
    return func_vals_and_grads_and_hessians!(y, Dy, H, op, x, nothing, outputs)
end

# ### Parameter-Methods for Non-Parametric Operators
# To also be able to use non-parametric operators in the more general setting, 
# implement the parametric-interface:

function eval_op!(y, op::AbstractNonlinearOperatorNoParams, x, p)
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

# ### Non-Parametric Methods for Parametric Operators
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

include("autodiff_backends.jl")
include("wrapped_function.jl")
include("nonlinear_function.jl")