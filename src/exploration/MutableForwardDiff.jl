module MutableForwardDiff
#%%
#=
import ForwardDiff: Dual, Tag, checktag, seed!, value, extract_value!, chunksize, Chunk
import ForwardDiff.DiffResults
import ForwardDiff.DiffResults: DiffResult, MutableDiffResult

import ForwardDiff: derivative!, DerivativeConfig, extract_derivative!

function DerivativeConfig(
    f::F, result::DiffResult, x::X, tag::T = Tag(f, X)
) where {F,X<:Real,T}
    return DerivativeConfig(f, DiffResults.value(result), x, tag)
end

function derivative!(
    result::MutableDiffResult, f::F, x::Real, 
    cfg::DerivativeConfig{T}=DerivativeConfig(f, result, x),
    ::Val{CHK}=Val{true}();
    f_is_in_place::Bool=false
) where {F,T,CHK}
    CHK && checktag(T, f, x)
    xdual = Dual{T}(x, one(x))
    y = DiffResults.value(result)
    ydual = cfg.duals
    if f_is_in_place
        seed!(ydual, y)
        f(ydual, xdual)
    else
        ydual .= f(xdual)
    end
    extract_value!(T, result, ydual)
    extract_derivative!(T, result, ydual)
    return nothing
end

import ForwardDiff: jacobian!, JacobianConfig, extract_jacobian!, vector_mode_jacobian!, 
    chunk_mode_jacobian!, vector_mode_dual_eval!, jacobian_chunk_mode_expr

function JacobianConfig(f::F,
                        result::DiffResult,
                        x::AbstractArray{X},
                        chunk ::Chunk{N} = Chunk(x),
                        tag ::T = Tag(f, X)) where {F,X,N,T}
    return JacobianConfig(f, DiffResults.value(result), x, chunk, tag)
end

function vector_mode_dual_eval!(f::F, cfg::JacobianConfig, y, x, ::Val{f_is_in_place}) where {F, f_is_in_place}
    ydual, xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    if f_is_in_place
        seed!(ydual, y)
        f(ydual, xdual)
    else
        ydual .= f(xdual)
    end
    return ydual
end

function vector_mode_jacobian!(result, f!::F, y, x, cfg::JacobianConfig{T}, f_is_in_place_val::Val{f_is_in_place}) where {F,T,f_is_in_place}
    N = chunksize(cfg)
    ydual = vector_mode_dual_eval!(f!, cfg, y, x, f_is_in_place_val)
    map!(d -> value(T,d), y, ydual)
    extract_jacobian!(T, result, ydual, N)
    extract_value!(T, result, y, ydual)
    # if `y==DiffResults.value(result)`, `extract_value!` is unnecessary
    # because it becomes `DiffResults.value!(result, DiffResults.value(result))``
    # however, Julia seems to have that optimized away in `copyto!`, 
    # so we can leave it out of an abundance of caution
    return nothing
end

import ForwardDiff: reshape_jacobian, extract_jacobian_chunk!
@eval function chunk_mode_jacobian!(result, f::F, y, x, cfg::JacobianConfig{T,V,N}, ::Val{f_is_in_place}) where {F,T,V,N,f_is_in_place}
    $(jacobian_chunk_mode_expr(quote
                                   ydual, xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               quote 
                                    if f_is_in_place 
                                        f(seed!(ydual, y), xdual)
                                    else
                                        ydual .= f(xdual)
                                    end
                               end,
                               :(),
                               :(extract_value!(T, result, y, ydual))))
end

function jacobian!(
    result::MutableDiffResult, f::F, x::AbstractArray, 
    cfg::JacobianConfig{T} = JacobianConfig(f, result, x), ::Val{CHK}=Val{true}();
    f_is_in_place::Bool=false
) where {F,T,CHK}
    CHK && checktag(T, f, x)
    y = DiffResults.value(result)
    if chunksize(cfg) == length(x)
        vector_mode_jacobian!(result, f, y, x, cfg, Val(f_is_in_place))
    else
        chunk_mode_jacobian!(result, f, y, x, cfg, Val(f_is_in_place))
    end
    return nothing
end
=#

import ForwardDiff.DiffResults
import ForwardDiff: jacobian, jacobian!, JacobianConfig, DiffResult, MutableDiffResult,
    Dual, Chunk, Tag, value, checktag

struct VectorHessianConfig{T, V, N, DI, DO}
    inner_config::JacobianConfig{T, Dual{T, V, N}, N, DI}
    outer_config::JacobianConfig{T, V, N, DO}
    ydual::Vector{Dual{T,V,N}}
end

function VectorHessianConfig(
    f::F, result::DiffResult, x::AbstractArray{V}, chunk::Chunk = Chunk(x), tag = Tag(f, V)
) where {F, V}
    Dy = DiffResults.jacobian(result)
    nout = size(Dy, 1)

    outer_config = JacobianConfig((f, jacobian), Dy, x, chunk, tag)
    xdual = outer_config.duals[2]
    ydual = zeros(eltype(xdual), nout)
    inner_config = JacobianConfig(f, ydual, xdual, chunk, tag)
    return VectorHessianConfig(inner_config, outer_config, ydual)
end

mutable struct InnerJacobianForHess!{R,C,F}
    result::R
    cfg::C
    f::F
    f_is_in_place::Bool
end

function (g::InnerJacobianForHess!)(Dydual, xdual)
    inner_result = DiffResult(g.cfg.ydual, Dydual)
    
    if g.f_is_in_place
        jacobian!(inner_result, g.f, g.cfg.ydual, xdual, g.cfg.inner_config, Val(false))
    else
        jacobian!(inner_result, g.f, xdual, g.cfg.inner_config, Val(false))
    end
    # jacobian!(inner_result, g.f, xdual, g.cfg.inner_config, Val{false}(); f_is_in_place=g.f_is_in_place)
    g.result = DiffResults.value!(g.result, value.(DiffResults.value(inner_result)))
    return nothing
end

function vector_hessian!(
    result::MutableDiffResult{O}, 
    f::F, 
    x::AbstractArray, 
    cfg::VectorHessianConfig{T} = VectorHessianConfig(f, result, x), ::Val{CHK}=Val{true}();
    f_is_in_place::Bool=false,
) where {O,F,T,CHK}
    @assert O >= 2 "Target `result` must contain arrays for second-order derivatives."
    CHK && checktag(T, f, x)
    ∇f! = InnerJacobianForHess!(result, cfg, f, f_is_in_place)

    Dy = DiffResults.jacobian(result)
    nout, nvars = size(Dy)
    H = DiffResults.hessian(result)
    h = if ndims(H) == 3
        #=
        Assume, we want to store the Hessians in a 3D array `H`, such that `H[:,:,k]` 
        is the `k`-th Hessian matrix, i.e., `H[i,j,k]` is ∂/∂ᵢ∂ⱼ Hₖ.
        Given the Jacobian 
        ```
        J = [
            ∂₁f₁ ∂₂f₁  …  ∂ₙf₁ ;
            ∂₁f₂ ∂₂f₂  …  ∂ₙf₂ ;
             ⋮   ⋮    ⋱  ⋮  
            ∂₁fₘ ∂₂fₘ …  ∂ₙfₘ
        ]
        ```
        the function `ForwardDiff.jacobian` interprets the colums as seperate outputs
        and stacks their respective Jacobians vertically.
        Consider `J[:,i] = [∂ᵢf₁, …, ∂ᵢfₘ]`. 
        Its Jacobian is
        ```
        ∇(J[:,i]) = [
            (∇∂ᵢf₁)ᵀ;
            (∇∂ᵢf₂)ᵀ;
                ⋮
            (∇∂ᵢfₘ)ᵀ;
        ] = [
            ∂₁∂ᵢf₁ ∂₂∂ᵢf₁  …  ∂ₙ∂ᵢf₁ ;
            ∂₁∂ᵢf₂ ∂₂∂ᵢf₂  …  ∂ₙ∂ᵢf₂ ;
              ⋮     ⋮     ⋱    ⋮  
            ∂₁∂ᵢfₘ ∂₂∂ᵢfₘ …  ∂ₙ∂ᵢfₘ ;
        ] 
        ```
        Thus, the `i`-th block of the stacked Hessian contains as its rows the 
        `i`-th columns of all function Hessians.
        To get `H` from `∇J`, we can do 
        `H = permutedims(reshape(transpose(reshape(∇J, m, n*n)), n, n, m), (2,1,3))`
        But if `H` is provided as a pre-allocated buffer, we need the inverse operation:
        `∇J = reshape(transpose(reshape(permutedims(H, (2,1,3)), n*n, m)), m*n, n)`
        In case of symmetric Hessians, we could forgo the `permutedims` operation...
        =#
        reshape(transpose(reshape(PermutedDimsArray(H, (2,1,3)), nvars*nvars, nout)), nout*nvars, nvars)
    else
        H
    end
    jacobian!(h, ∇f!, Dy, x, cfg.outer_config, Val{false}())
    return nothing
end
#%%
end