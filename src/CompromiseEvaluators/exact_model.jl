Base.@kwdef struct ExactModel{O,P} <: AbstractSurrogateModel#{O<:AbstractNonlinearOperator, P}
    op :: O
    params :: P = nothing
end

depends_on_trust_region(::ExactModel)=false
requires_grads(::Type{<:ExactModel})=true
requires_hessians(::Type{<:ExactModel})=false

function init_surrogate(::Type{<:ExactModel}, op, dim_in, dim_out, params, T)
    return ExactModel(op, params)
end

function model_op!(y, surr::ExactModel, x)
    eval_op!(y, surr.op, x, surr.params)
end
function model_grads!(Dy, surr::ExactModel, x)
    return eval_grads!(Dy, surr.op, x, surr.params)
end
function model_op_and_grads!(y, Dy, surr::ExactModel, x)
    return eval_op_and_grads!(y, Dy, surr.op, x, surr.params)
end
supports_partial_evaluation(::ExactModel)=false