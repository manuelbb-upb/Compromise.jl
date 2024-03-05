module NonlinearFunctions

include("abstract_autodiff.jl")

import ..Compromise: RVec, RVecOrMat
import ..Compromise.CompromiseEvaluators as CE

using Parameters: @with_kw

# ## Wrapper for Parametric Functions

@doc """
    NonlinearParametricFunction(; 
        func, grads=nothing, hessians=nothing, 
        func_and_grads=nothing, func_and_grads_and_hessians=nothing,
        backend=nothing,
        func_iip=true, grads_iip=true, hessians_iip=true, 
        func_and_grads_iip=true, func_and_grads_and_hessians_iip=true)

A flexible function wrapper to conveniently query evaluations and derivatives of user
provided functions.

If the user provided function is used in a derivative-free alogrithm, only `func` has 
to be provided. The flag `func_iip` indicates its signature:
If `func_iip==true`, the function should mutate the target array and 
have signature `func!(y, x, p)`. Otherwise, `y = func(x, p)`.

Should gradients be needed, function handles can be provided, and the respective 
flags indicate the following signatures:
* `grads_iip == true` implies `grads!(Dy, x, p)`, otherwise `Dy = grads(x, p)`.
* `hessians_iip == true` implies `hessians!(H, x, p)`, otherwise `H = hessians(x, p)`.
* `func_and_grads_iip == true` implies `func_and_grads!(y, Dy, x, p)`,  
   otherwise `y, Dy = func_and_grads(x, p)`.
* `func_and_grads_and_hessians_iip == true` implies  
  `func_and_grads_and_hessians!(y, Dy, H, x, p)`,  
   else `y, Dy, H = func_and_grads_and_hessians(x, p)`.

Alternatively (or additionally), an `AbstractAutoDiffBackend` can be passed to 
compute the derivatives if the relevant field `isnothing`.
"""
@with_kw struct NonlinearParametricFunction{
    F, G, H, FG, FGH, B
} <: CE.AbstractNonlinearOperator
    func :: F
    grads :: G = nothing
    hessians :: H = nothing
    func_and_grads :: FG = nothing
    func_and_grads_and_hessians :: FGH = nothing
    backend :: B = NoBackend()

    func_iip :: Bool = false # true -> func!(y, x, p), false -> y = func(x, p)
    grads_iip :: Bool = false # true -> grads!(Dy, x, p), false -> Dy = grads(x, p)
    hessians_iip :: Bool = false # true -> hessians!(H, x, p), false -> H = hessians(x, p)
    func_and_grads_iip :: Bool = grads_iip  # true -> func_and_grads!(y, Dy, x, p), false -> y, Dy = func_and_grads(x, p)
    func_and_grads_and_hessians_iip :: Bool = hessians_iip # true -> func_and_grads_and_hessians!(y, Dy, H, x, p), false -> y, Dy, H = func_and_grads_and_hessians(x, p)

    num_func_calls :: Base.RefValue{Int} = Ref(0)
    num_grad_calls :: Base.RefValue{Int} = Ref(0)
    num_hess_calls :: Base.RefValue{Int} = Ref(0)
    
    max_func_calls :: Int = typemax(Int)
    max_grad_calls :: Int = typemax(Int)
    max_hess_calls :: Int = typemax(Int)
    
    # Hessians should imply gradients
    @assert (
        isnothing(backend) && isnothing(hessians) && isnothing(func_and_grads_and_hessians)
    ) || (
        !isnothing(backend) || !isnothing(grads) || !isnothing(func_and_grads)
    ) "If Hessians can be computed, then gradients should be computed too."
end

function CE.optrait_params(op::NonlinearParametricFunction)
    CE.IsParametricOperator()
end
CE.optrait_partial(::NonlinearParametricFunction) = CE.OperatorOnlyFullEvaluation()
CE.optrait_multi(::NonlinearParametricFunction) = CE.OperatorSequential()

function CE.provides_grads(op::NonlinearParametricFunction)
    !isnothing(op.backend) && return true
    !isnothing(op.grads) && return true
    !isnothing(op.func_and_grads) && return true
    return false
end
function CE.provides_hessians(op::NonlinearParametricFunction)
    !isnothing(op.backend) && return true
    !isnothing(op.hessians) && return true
    !isnothing(op.func_and_grads_and_hessians) && return true
    return false
end

import ..Compromise: AbstractStoppingCriterion, stop_message, @ignoraise

struct BudgetExhausted <: AbstractStoppingCriterion
    ni :: Int
    mi :: Int
    order :: Int
end

function stop_message(crit::BudgetExhausted)
    return "Maximum evaluation count reached, order=$(crit.order), is=$(crit.ni), max=$(crit.mi)."
end

function check_num_calls(op::NonlinearParametricFunction, i=0)
    ni, mi = if i== 1
        op.num_grad_calls[], op.max_grad_calls
    elseif i==2
        op.num_hess_calls[], op.max_hess_calls
    else
        op.num_func_calls[], op.max_func_calls
    end
    if ni >= mi
        return BudgetExhausted(ni, mi, i)
    end
    return nothing
end

function CE.eval_op!(y::RVec, op::NonlinearParametricFunction, x::RVec, p)
    @ignoraise check_num_calls(op, 0)
    
    if op.func_iip 
        op.func(y, x, p)
    else
        y .= op.func(x, p)
    end
    op.num_func_calls[] += 1
    
    return nothing
end

function CE.eval_grads!(Dy, op::NonlinearParametricFunction, x, p)
    @ignoraise check_num_calls(op, 1)
     
    if !isnothing(op.grads)
        if op.grads_iip
            op.grads(Dy, op.grads!, x, p)
        else
            Dy .= op.grads(x, p)
        end
    elseif !isnothing(op.func_and_grads)
        @debug "Allocating temporary output array for `eval_grads!`."
        if op.func_and_grads_iip
            y = similar(Dy, size(Dy, 1))
            op.func_and_grads(y, Dy, x, p)
        else
            _, _Dy = op.func_and_grads(x, p)
            Dy .= _Dy
        end
    else
        ad_grads!(Dy, op.backend, op.func, x, p, Val(op.func_iip))
    end
    op.num_grad_calls[] += 1
    return nothing
end

function CE.eval_hessians!(H, op::NonlinearParametricFunction, x, p)
    @ignoraise check_num_calls(op, 2)
    
    if !isnothing(op.hessians)
        if op.hessians_iip
            op.hessians(H, x, p)
        else
            _H = op.hessians(x, p)
            H .= _H
        end
    elseif !isnothing(op.func_and_grads_and_hessians)
        @debug "Allocating temporary output arrays for `eval_hessians!`."
        if op.func_and_grads_and_hessians_iip
            y = similar(H, size(H, 3))
            Dy = similar(H, size(H, 1), length(y))
            op.func_and_grads_and_hessians(y, Dy, H, x, p)
        else
            _, _, _H = op.func_and_grads_and_hessians(x, p)
            H .= _H
        end
    else
        ad_hessians!(H, op.backend, op.func, x, p, Val(op.func_iip))
    end
    op.num_hess_calls[] += 1
    return nothing
end

function CE.eval_op_and_grads!(y, Dy, op::NonlinearParametricFunction, x, p)
    @ignoraise check_num_calls(op, 0)
    @ignoraise check_num_calls(op, 1)

    inc_counter = false
    if !isnothing(op.func_and_grads)
        if op.func_and_grads_iip
            op.func_and_grads(y, Dy, x, p)
        else
            @debug "Allocating temporary output arrays for `eval_op_and_grads!`."
            _y, _Dy = op.func_and_grads(x, p)
            y .= _y
            Dy .= _Dy
        end
        inc_counter = true
    elseif !isnothing(op.grads)
        CE.eval_op!(y, op, x, p)
        CE.eval_grads!(Dy, op, x, p)
    else
        ad_op_and_grads!(y, Dy, op.backend, op.func, x, p, Val(op.func_iip))
        inc_counter = true
    end
    
    if inc_counter
        op.num_func_calls[] += 1
        op.num_grad_calls[] += 1
    end
    return nothing
end

function CE.eval_op_and_grads_and_hessians!(y, Dy, H, op::NonlinearParametricFunction, x, p)
    @ignoraise check_num_calls(op, 0)
    @ignoraise check_num_calls(op, 1)
    @ignoraise check_num_calls(op, 2)
    
    inc_couter = false
    if !isnothing(op.func_and_grads_and_hessians)
        if op.func_and_grads_and_hessians_iip
            op.func_and_grads_and_hessians(y, Dy, H, x, p)
        else
            @debug "Allocating temporary output arrays for `eval_op_and_grads_and_hessians!`."
            _y, _Dy, _H = op.func_and_grads_and_hessians(x, p)
            y .= _y
            Dy .= _Dy
            H .= _H
        end
        inc_couter = true
    elseif !isnothing(op.hessians)
        CE.eval_op_and_grads!(y, Dy, op, x, p)
        CE.eval_hessians!(H, op, x, p)
    else
        ad_op_and_grads_and_hessians!(y, Dy, H, op.backend, op.func, x, p, Val(op.func_iip))
        inc_couter = true
    end

    if inc_couter
        op.num_func_calls[] += 1
        op.num_grad_calls[] += 1
        op.num_hess_calls[] += 1
    end
    
    return nothing
end

# ## Wrapper for Non-Parametric Functions
# Historically, `NonlinearParametricFunction` was implemented first: 
# An `AbstractNonlinearOperator` with `IsParametricOperator()` trait to 
# automatically make user defined functions compatible with the 
# `AbstractNonlinearOperator` interface.
# The user functions were also expected to take a parameter vector `p` last.
# Sometimes, we do not need that, and to make life easier for the user, we
# provide `NonlinearFunction`.
# 1) Its keyword-argument constructor intercepts all functions and wraps them 
#    in `MakeParametric`, to make them conform to the implementation of the
#    evaluation methods for `NonlinearParametricFunction`.
# 2) It then automatically supports the interface for `AbstractNonlinearOperator`
#    with `IsNonparametricOperator()` through the wrapped field.
#
# This whole affair is a bit convoluted, and tidying things up is a TODO.

## helper to intercept and modify functions without parameters and make them accept some
struct MakeParametric{F}
    func :: F
end

# Re-define evaluation to take parameters.
# For the general case, with arbitrarily many arguments,
# a `@generated` function appears to work reasonably well...
(f::MakeParametric)(x, p) = f.func(x)
(f::MakeParametric)(y, x, p) = f.func(y, x)
(f::MakeParametric)(y, Dy, x, p) = f.func(y, Dy, x)
(f::MakeParametric)(y, Dy, Hy, x, p) = f.func(y, Dy, Hy, x)

# `NonlinearFunction` simply wraps a `NonlinearParametricFunction`.
struct NonlinearFunction{
    WF<:CE.AbstractNonlinearOperator} <: CE.AbstractNonlinearOperator
    wrapped_function :: WF
end

# The magic happens with the keyword-argument constructor.
# We intercept user functions without parameters and wrap them first.
# How we parse the `kwargs` is not necessarily performant, so this should not be 
# done in hot-loops:
function NonlinearFunction(; kwargs...)
    new_kwargs = Dict{Any, Any}(kwargs...)
    for fn in (
        :func, :grads, :hessians, :func_and_grads, :func_and_grads_and_hessians
    )
        if haskey(kwargs, fn)
            new_kwargs[fn] = MakeParametric(kwargs[fn])
        end
    end
    return NonlinearFunction(NonlinearParametricFunction(;new_kwargs...))
end

CE.optrait_params(op::NonlinearFunction)=CE.IsNonparametricOperator()
CE.optrait_partial(op::NonlinearFunction)=CE.optrait_partial(op.wrapped_function)
CE.optrait_multi(op::NonlinearFunction)=CE.optrait_multi(op.wrapped_function)
CE.provides_grads(op::NonlinearFunction)=CE.provides_grads(op.wrapped_function)
CE.provides_hessians(op::NonlinearFunction)=CE.provides_hessians(op.wrapped_function)

function CE.eval_op!(y::RVec, op::NonlinearFunction, x::RVec)
    CE.eval_op!(y, op.wrapped_function, x, nothing)
end
function CE.eval_grads!(Dy, op::NonlinearFunction, x)
    CE.eval_grads!(Dy, op.wrapped_function, x, nothing)
end
function CE.eval_hessians!(H, op::NonlinearFunction, x)
    CE.eval_hessians!(H, op.wrapped_function, x, nothing)
end
function CE.eval_op_and_grads!(y, Dy, op::NonlinearFunction, x)
    CE.eval_op_and_grads!(y, Dy, op.wrapped_function, x, nothing)
end
function CE.eval_op_and_grads_and_hessians!(y, Dy, H, op::NonlinearFunction, x)
    CE.eval_op_and_grads_and_hessians!(y, Dy, H, op.wrapped_function, x, nothing)
end

export AbstractAutoDiffBackend, NoBackend
export ad_grads!, ad_hessians!, ad_op_and_grads!, ad_op_and_grads_and_hessians!
export NonlinearFunction, NonlinearParametricFunction
end#module