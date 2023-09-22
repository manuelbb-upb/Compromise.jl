module ExactModels

using ..Compromise.CompromiseEvaluators
const CompromiseEvaluators = CE

Base.@kwdef struct ExactModel{O,P} <: AbstractSurrogateModel#{O<:AbstractNonlinearOperator, P}
    op :: O
    params :: P = nothing
end

struct ExactModelConfig <: AbstractSurrogateModelConfig end

CE.depends_on_trust_region(::ExactModel)=false
CE.requires_grads(::Type{<:ExactModel})=true
CE.requires_hessians(::Type{<:ExactModel})=false

function CE.init_surrogate(::ExactModelConfig, op, dim_in, dim_out, params, T)
    return ExactModel(op, params)
end

function CE.model_op!(y, surr::ExactModel, x)
    eval_op!(y, surr.op, x, surr.params)
end
function CE.model_grads!(Dy, surr::ExactModel, x)
    return eval_grads!(Dy, surr.op, x, surr.params)
end
function CE.model_op_and_grads!(y, Dy, surr::ExactModel, x)
    return eval_op_and_grads!(y, Dy, surr.op, x, surr.params)
end
CE.supports_partial_evaluation(::ExactModel)=false

end#module