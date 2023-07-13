import LinearAlgebra as LA

struct TaylorPolynomial{
    X <: AbstractVector{<:Real},
    F <: AbstractVector{<:Real},
    D <: AbstractMatrix{<:Real},
    H <: Union{Nothing, AbstractArray{<:Real, 3}}
} <: AbstractSurrogateModel
    x0 :: X
    Δx :: X
    fx :: F
    Dfx :: D
    Hfx :: H

    xtmp :: X
    #=
    cache_results :: Bool
    xhash :: Base.RefValue{UInt64}
    y :: Union{F, Nothing}
    Dy :: Union{D, Nothing}
    =#
end

function TaylorPolynomial(n_vars::Int, n_out::Int, T=Float64; degree=2)
    x0 = Vector{T}(undef, n_vars)
    Δx = similar(x0)
    xtmp = similar(x0)
    fx = Vector{T}(undef, n_out)
    Dfx = Matrix{T}(undef, n_vars, n_out)
    Hfx = degree <= 1 ? nothing : Array{T, 3}(undef, n_vars, n_vars, n_out)

    return TaylorPolynomial(x0, Δx, fx, Dfx, Hfx, xtmp)
    #=
    xhash = Ref(hash(x0))
    if cache_results
        y = copy(fx)
        Dy = copy(Dfx)
    else
        y = nothing
        Dy = nothing
    end
    return TaylorPolynomial(x0, Δx, fx, Dfx, Hfx, cache_results, xhash, y, Dy)
    =#
end

requires_grads(::TaylorPolynomial)=true
requires_hessians(tp::TaylorPolynomial)=!isnothing(tp.Hfx)

function model_op!(y, tp::TaylorPolynomial, x)
    Δx = tp.Δx
    Δx .= x
    Δx .-= tp.x0

    ## `y = fx + Δx' Dfx`
    y .= tp.fx
    LA.mul!(y, tp.Dfx', Δx, 1, 1)
    if !isnothing(tp.Hfx)
        H = tp.Hfx
        @views for i = axes(H, 3)
            y[i] += 0.5 * only(Δx' * H[:, :, i] * Δx)
        end
    end
    return nothing
end

function model_grads!(Dy, tp::TaylorPolynomial, x)
    Dy .= tp.Dfx
    if !isnothing(tp.Hfx)
        Δx = tp.Δx
        Δx .= x
        Δx .-= tp.x0
        
        H = tp.Hfx
        @views for i = axes(H, 3)
            ## (assuming symmetric Hessians here)
            Hi = H[:, :, i]
            LA.mul!(Dy[:, i], Hi, Δx, 2, 1)   
        end
    end
    return nothing
end

function model_op_and_grads!(y, Dy, tp::TaylorPolynomial, x)
    Δx = tp.Δx
    Δx .= x
    Δx .-= tp.x0

    ## 1a) Model values `y` without Hessians
    y .= tp.fx
    LA.mul!(y, tp.Dfx', Δx, 1, 1)
    ## 2a) Model gradients `Dy` without Hessians
    Dy .= tp.Dfx

    if !isnothing(tp.Hfx)
        H = tp.Hfx
        HΔx = tp.xtmp
        @views for i = axes(H, 3)
            Hi = H[:, :, i]
            LA.mul!(HΔx, Hi, Δx, 0.5, 0)

            ## 1b) add Hessian term to value `y[i]`
            y[i] += Δx'HΔx

            ## 2b) add Hessian terms to gradients `Dy[:, i]`
            ## (assuming symmetric Hessians here)
            HΔx .*= 4
            Dy[:, i] .+= HΔx
        end
    end

    return nothing
end