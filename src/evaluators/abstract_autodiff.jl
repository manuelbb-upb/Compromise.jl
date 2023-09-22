abstract type AbstractAutoDiffBackend end

function ad_grads!(Dy, backend::AbstractAutoDiffBackend, func!, x, p, f_is_in_place::Val{true})
    return error("`eval_grads!` not implemented for backend of type $(typeof(backend)) and funtions that are in-place.")
end

function ad_grads!(Dy, backend::AbstractAutoDiffBackend, func, x, p, f_is_in_place::Val{false})
    return error("`eval_grads!` not implemented for backend of type $(typeof(backend)) and funtions that are out-of-place.")
end

function ad_hessians!(H, backend::AbstractAutoDiffBackend, func, x, p, f_is_in_place::Union{Val{true}, Val{false}})
    return error("`eval_hessians!` not implemented for backend of type $(typeof(backend)).")
end

function ad_op_and_grads!(y, Dy, backend::AbstractAutoDiffBackend, func, x, p, f_is_in_place::Val{true})
    @warn ("`eval_op_and_grads!` not implemented for backend of type $(typeof(backend)).")
    func(y, x, p)
    return eval_grads!(Dy, backend, func, x, p, f_is_in_place) 
end

function ad_op_and_grads!(y, Dy, backend::AbstractAutoDiffBackend, func, x, p, f_is_in_place::Val{false})
    @warn ("`eval_op_and_grads!` not implemented for backend of type $(typeof(backend)).")
    y .= func(x, p)
    return eval_grads!(Dy, backend, func, x, p, f_is_in_place) 
end

function ad_op_and_grads_and_hessians!(
    y, Dy, H, backend::AbstractAutoDiffBackend, func, x, p, f_is_in_place::Union{Val{true}, Val{false}}
)
    @warn ("`eval_op_and_grads_and_hessians!` not implemented for backend of type $(typeof(backend)).")
    eval_op_and_grads!(y, Dy, backend, func, x, p, f_is_in_place)
    return eval_hessians!(H, backend, func, x, p, f_is_in_place)    
end

struct NoBackend <: AbstractAutoDiffBackend end