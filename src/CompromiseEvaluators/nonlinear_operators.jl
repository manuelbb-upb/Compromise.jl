abstract type AbstractNonlinearOperator end

#=
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

"""
    eval_op!(y, op, x, p)

Evaluate the operator `op` at variable vector `x` with parameters `p`
and mutate the target vector `y` to contain the result.
"""
function eval_op!(y, op::AbstractNonlinearOperator, x, p)
    return error("No implementation of `eval_op!` for operator $op.")
end

# Functions to indicate if an operator implements `eval_grads!`, `eval_hessians!`, etc ...
provides_grads(op::AbstractNonlinearOperator)=false
provides_hessians(op::AbstractNonlinearOperator)=false

"""
    eval_grads!(Dy, op, x, p)

Compute the gradients of the operator `op` at variable vector `x` with parameters `p` 
and mutate the target matrix `Dy` to contain the gradients w.r.t. `x` in its columns. 
That is, `Dy` is the transposed Jacobian at `x`.
"""
function eval_grads!(Dy, op::AbstractNonlinearOperator, x, p)
    return error("No implementation of `eval_grads!` for operator $op.")
end

# The combined forward-function `eval_op_and_grads!` is derived from `eval_op!` and 
# `eval_grads!`, but can be customized easily:
function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperator, x, p)
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
function eval_hessians!(H, op::AbstractNonlinearOperator, x, p)
    return error("No implementation of `eval_hessians!` for operator $op.")
end

# The combined forward-function `eval_op_and_grads_and_hessians!` 
# is derived from `eval_op_and_grads!` and `eval_hessians!`,
# but can be customized easily:
function eval_op_and_grads!(y, Dy, H, op::AbstractNonlinearOperator, x, p)
    eval_op_and_grads!(Dy, y, op, x, p)
    eval_hessians!(H, val, x, p)
    return nothing
end

# Some operators might support partial evaluation. 
# They should implement these methods:
supports_partial_evaluation(op::AbstractNonlinearOperator) = false
function eval_op!(y, op::AbstractNonlinearOperator, x, p, outputs)
    ## length(y)==length(outputs)
    return error("Partial evaluation not implemented.")
end
function eval_grads!(Dy, op::AbstractNonlinearOperator, x, p, outputs)
    ## size(Dy)==(length(x), length(outputs))
    return error("Partial Jacobian not implemented.")
end
function eval_op_and_grads!(y, Dy, op::AbstractNonlinearOperator, x, p, outputs)
    eval_op!(y, op, x, p, outputs)
    eval_grads!(Dy, op, x, p, outputs)
    return nothing
end
function eval_hessians!(H, op::AbstractNonlinearOperator, x, p, outputs)
    return error("Partial Hessians not implemented.")
end
function eval_op_and_grads_and_hessians!(y, Dy, H, op::AbstractNonlinearOperator, x, p, outputs)
    eval_op_and_grads!(y, Dy, op, x, p, outputs)
    eval_hessians!(H, op, x, p, outputs)
    return nothing
end
function func_vals!(y, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        eval_op!(y, op, x, p, outputs)
    end
    return eval_op!(y, op, x, p)
end
function func_grads!(Dy, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        eval_grads!(Dy, op, x, p, outputs)
    end
    return eval_grads!(Dy, op, x, p)
end
function func_vals_and_grads!(y, Dy, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        eval_op_and_grads!(y, Dy, op, x, p, outputs)
    end
    return eval_op_and_grads!(y, Dy, op, x, p)
end
function func_hessians!(H, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        eval_hessians!(H, op, x, p, outputs)
    end
    return eval_hessians!(H, op, x, p)
end
function func_vals_and_grads_and_hessians!(
    y, Dy, H, op::AbstractNonlinearOperator, x, p, outputs=nothing)
    if !isnothing(outputs) && supports_partial_evaluation(op)
        eval_op_and_grads_and_hessians!(y, Dy, H, op, x, p, outputs)
    end
    eval_op_and_grads_and_hessians!(y, Dy, H, op, x, p)
end

include("autodiff_backends.jl")
include("wrapped_function.jl")