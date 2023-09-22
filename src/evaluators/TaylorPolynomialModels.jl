module TalyorPolynomialModels

import LinearAlgebra as LA
using ..Compromise.CompromiseEvaluators
const CompromiseEvaluators = CE

struct TaylorPolynomial1{
    X <: AbstractVector{<:Real},
    F <: AbstractVector{<:Real},
    D <: AbstractMatrix{<:Real},
#src    H <: Union{Nothing, AbstractArray{<:Real, 3}}
} <: AbstractSurrogateModel
    x0 :: X
    Δx :: X
    fx :: F
    Dfx :: D 
end

struct TaylorPolynomial2{X, TP1<:TaylorPolynomial1{X}, H} <: AbstractSurrogateModel
    tp :: TP1
    Hfx :: H
    xtmp :: X
end

@with_kw struct TaylorPolynomialConfig <: AbstractSurrogateModelConfig
    degree :: Int = 1
    @assert 1 <= degree <= 2 "Taylor polynomial must have degree 1 or 2."
end    


function TaylorPolynomial1(dim_in, dim_out, T)
    x0 = Vector{T}(undef, dim_in)
    Δx = similar(x0)
    fx = Vector{T}(undef, dim_out)
    Dfx = Matrix{T}(undef, dim_in, dim_out)
    return TaylorPolynomial1(x0, Δx, fx, Dfx)
end

function TaylorPolynomial2(dim_in, dim_out, T)
    tp1 = TaylorPolynomial1(dim_in, dim_out, T)
    Hfx = Array{T, 3}(undef, dim_in, dim_in, dim_out)
    xtmp = similar(tp1.x0)
    return TaylorPolynomial2(tp1, Hfx, xtmp)
end

function CE.init_surrogate(tp_cfg::TaylorPolynomialConfig, op, dim_in, dim_out, params, T)
    if tp_cfg.degree == 1
        return TaylorPolynomial2(dim_in, dim_out, T)
    else
        return TaylorPolynomial2(dim_in, dim_out, T)
    end
end

const TaylorPoly = Union{TaylorPolynomial1, TaylorPolynomial2}
CE.depends_on_trust_region(::TaylorPoly)=false
CE.requires_grads(::Type{<:TaylorPoly})=true
CE.requires_hessians(tp::Type{TaylorPolynomial2})=true

function CE.model_op!(y, tp::TaylorPolynomial1, x)
    Δx = tp.Δx
    Δx .= x .- tp.x0

    ## `y = fx + Δx' Dfx`
    y .= tp.fx
    LA.mul!(y, tp.Dfx', Δx, 1, 1)
    return nothing
end

function CE.model_op!(y, tp::TaylorPolynomial2, x)
    model_op!(y, tp.tp, x)
    Δx = tp.tp.Δx
    H = tp.Hfx
    @views for i = axes(H, 3)
        y[i] += 0.5 * only(Δx' * H[:, :, i] * Δx)
    end
    return nothing
end

function CE.model_grads!(Dy, tp::TaylorPolynomial1, x)
    Dy .= tp.Dfx
    return nothing
end

function CE.model_grads!(Dy, tp::TaylorPolynomial2, x)
    tp1 = tp.tp
    model_grads!(Dy, tp1, x)
    Δx = tp1.Δx
    Δx .= x .- tp1.x0
    H = tp.Hfx
    @views for i = axes(H, 3)
        ## (assuming symmetric Hessians here)
        Hi = H[:, :, i]
        LA.mul!(Dy[:, i], Hi, Δx, 2, 1)   
    end
    return nothing
end

function CE.model_op_and_grads!(y, Dy, tp::TaylorPolynomial1, x)
    Δx = tp.Δx
    Δx .= x .- tp.x0

    y .= tp.fx
    LA.mul!(y, tp.Dfx', Δx, 1, 1)
    Dy .= tp.Dfx
    return nothing
end

function CE.model_op_and_grads!(y, Dy, tp::TaylorPolynomial2, x)
    tp1 = tp.tp
    model_op_and_grads!(y, Dy, tp1, x)
    Δx = tp1.Δx
    H = tp.Hfx
    HΔx = tp.xtmp
    @views for i = axes(H, 3)
        Hi = H[:, :, i]
        LA.mul!(HΔx, Hi, Δx, 0.5, 0)

        ## 1) add Hessian term to value `y[i]`
        y[i] += Δx'HΔx

        ## 2) add Hessian terms to gradients `Dy[:, i]`
        ## (assuming symmetric Hessians here)
        HΔx .*= 4
        Dy[:, i] .+= HΔx
    end

    return nothing
end

function CE.update!(tp::TaylorPolynomial1, op, Δ, x, fx, lb, ub; kwargs...)
    copyto!(tp.x0, x)
    eval_op_and_grads!(tp.fx, tp.Dfx, op, x)
end

function CE.update!(tp::TaylorPolynomial2, op, Δ, x, fx, lb, ub; kwargs...)
    tp1 = tp.tp
    copyto!(tp1.x0, x)
    eval_op_and_grads_and_hessians!(tp1.fx, tp1.Dfx, tp.Hfx, op, x)
    return nothing
end

export TaylorPoly, TaylorPolynomial1, TaylorPolynomial2, TaylorPolynomialConfig

end#module