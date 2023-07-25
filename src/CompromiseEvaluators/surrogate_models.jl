abstract type AbstractSurrogateModel end
abstract type AbstractSurrogateModelConfig end

depends_on_trust_region(::AbstractSurrogateModel)=true

requires_grads(::AbstractSurrogateModelConfig)=false
requires_hessians(::AbstractSurrogateModelConfig)=false

init_surrogate(::AbstractSurrogateModelConfig, op, dim_in, dim_out, params, T)::AbstractSurrogateModel=nothing

function update!(
    surr::AbstractSurrogateModel, op, Δ, x, fx, lb, ub; kwargs...
)
    return nothing    
end

function model_op!(y, surr::AbstractSurrogateModel, x)
    return nothing
end

function model_grads!(Dy, surr::AbstractSurrogateModel, x)
    return nothing
end

function model_op_and_grads!(y, Dy, surr::AbstractSurrogateModel, x)
    model_op!(y, surr, x )
    model_grads!(Dy, surr, x)
    return nothing
end

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

function func_vals!(y, surr::AbstractSurrogateModel, x, p, outputs=nothing)
    if !isnothing(outputs)
        if supports_partial_evaluation(surr)
            return model_op!(y, surr, x, outputs)
        ## else
        ##     @warn "Partial evaluation not supported by surrogate model."
        end
    end
    return model_op!(y, surr, x)
end
function func_grads!(Dy, surr::AbstractSurrogateModel, x, p, outputs=nothing)
    if !isnothing(outputs)
        if supports_partial_evaluation(surr)
            return model_grads!(Dy, surr, x, outputs)
        ## else
        ##     @warn "Partial evaluation not supported by surrogate model."
        end
    end
    return model_grads!(Dy, surr, x)
end
function func_vals_and_grads!(y, Dy, surr::AbstractSurrogateModel, x, p, outputs=nothing)
    if !isnothing(outputs)
        if supports_partial_evaluation(surr)
            return model_op_and_grads!(y, Dy, surr, x, outputs)
        ## else
        ##     @warn "Partial evaluation not supported by surrogate model."
        end
    end
    return model_op_and_grads!(y, Dy, surr, x)
end

include("taylor_polynomials.jl")
include("rbf_models.jl")
include("exact_model.jl")