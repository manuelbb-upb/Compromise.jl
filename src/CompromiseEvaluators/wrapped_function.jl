# requires "autodiff_backends.jl" to be included

"""
    WrappedFunction(; 
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

@with_kw struct WrappedFunction{F, G, H, FG, FGH, B} <: AbstractNonlinearOperatorWithParams
    func :: F
    grads :: G = nothing
    hessians :: H = nothing
    func_and_grads :: FG = nothing
    func_and_grads_and_hessians :: FGH = nothing
    backend :: B = NoBackend()

    func_iip :: Bool = true # true -> func!(y, x, p), false -> y = func(x, p)
    grads_iip :: Bool = true # true -> grads!(Dy, x, p), false -> Dy = grads(x, p)
    hessians_iip :: Bool = true # true -> hessians!(H, x, p), false -> H = hessians(x, p)
    func_and_grads_iip :: Bool = grads_iip  # true -> func_and_grads!(y, Dy, x, p), false -> y, Dy = func_and_grads(x, p)
    func_and_grads_and_hessians_iip :: Bool = hessians_iip # true -> func_and_grads_and_hessians!(y, Dy, H, x, p), false -> y, Dy, H = func_and_grads_and_hessians(x, p)
end

function provides_grads(op::WrappedFunction)
    return !isnothing(op.grads) || !isnothing(op.func_and_grads) || !(op.backend isa NoBackend)
end
function provides_hessians(op::WrappedFunction)
    return !isnothing(op.hessians) || !isnothing(op.func_and_grads_and_hessians) || 
        !(op.backend isa NoBackend)
end

function eval_op!(y, op::WrappedFunction, x, p)
    if op.func_iip 
        op.func(y, x, p)
    else
        y .= op.func(x, p)
    end
    return nothing
end

function eval_grads!(Dy, op::WrappedFunction, x, p)
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
        eval_grads!(Dy, op.backend, op.func, x, p, Val(op.func_iip))
    end
    return nothing
end

function eval_hessians!(H, op::WrappedFunction, x, p)
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
        eval_hessians!(H, op.backend, op.func, x, p, Val(op.func_iip))
    end
end

function eval_op_and_grads!(y, Dy, op::WrappedFunction, x, p)
    if !isnothing(op.func_and_grads)
        if op.func_and_grads_iip
            op.func_and_grads(y, Dy, x, p)
        else
            @debug "Allocating temporary output arrays for `eval_op_and_grads!`."
            _y, _Dy = op.func_and_grads(x, p)
            y .= _y
            Dy .= _Dy
        end
    elseif !isnothing(op.grads)
        eval_op!(y, op, x, p)
        eval_grads!(Dy, op, x, p)
    else
        eval_op_and_grads!(y, Dy, op.backend, op.func, x, p, Val(op.func_iip))
    end
    return nothing
end


function eval_op_and_grads_and_hessians!(y, Dy, H, op::WrappedFunction, x, p)
    if !isnothing(op.func_and_grads_and_hessians)
        if op.func_and_grads_and_hessians_iip
            op.func_and_grads_and_hessians(y, Dy, H, x, p)
        else
            @debug "Allocating temporary output arrays for `eval_op_and_grads_and_hessians!`."
            _y, _Dy, _H = op.func_and_grads(x, p)
            y .= _y
            Dy .= _Dy
            H .= _H
        end
    elseif !isnothing(op.hessians)
        eval_op_and_grads!(y, Dy, op, x, p)
        eval_hessians!(H, op, x, p)
    else
        eval_op_and_grads_and_hessians!(y, Dy, H, op.backend, op.func, x, p, Val(op.func_iip))
    end
    return nothing
end