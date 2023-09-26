module ForwardDiffBackendExt

export ForwardDiffBackend

import Compromise as C
import Compromise: ad_grads!, ad_op_and_grads!, ad_hessians!, 
    ad_op_and_grads_and_hessians!

if isdefined(Base, :get_extension)
    import ForwardDiff as FD
    import ForwardDiff.DiffResults
    import ForwardDiff: jacobian, jacobian!, JacobianConfig, DiffResult, MutableDiffResult,
        Dual, Chunk, Tag, value, checktag
else
    import ..ForwardDiff as FD
    import ..ForwardDiff.DiffResults
    import ..ForwardDiff: jacobian, jacobian!, JacobianConfig, DiffResult, MutableDiffResult,
        Dual, Chunk, Tag, value, checktag
end

struct ForwardDiffBackend <: C.AbstractAutoDiffBackend end

#src given f(x, p), return x -> f(x, p)
#src const FixParams = Base.Fix2

struct FixParams{F, P}
    func :: F
    params :: P
    FixParams(f::F, p) where {F} = new{F, Base._stable_typeof(p)}(f, p)
    FixParams(f::Type{F}, p) where {F} = new{Type{F}, Base._stable_typeof(p)}(f, p)
end
ensure_vec(x)=x
ensure_vec(x::Number)=[x,]
(fp::FixParams)(x) = ensure_vec(fp.func(x, fp.params))

#src given f!(y, x, p), return (y, x) -> f!(y, x)
struct FixParams!{F, P}
    func! :: F
    params :: P

    FixParams!(f::F, p) where {F} = new{F, Base._stable_typeof(p)}(f, p)
    FixParams!(f::Type{F}, p) where {F} = new{Type{F}, Base._stable_typeof(p)}(f, p)
end
(fp::FixParams!)(y, x) = fp.func!(y, x, fp.params)

#================================ Vector Hessians ========================================#

struct VectorHessianConfig{T, V, Y, N, DI, DO}
    inner_config::JacobianConfig{T, Dual{T, V, N}, N, DI}
    outer_config::JacobianConfig{T, V, N, DO}
    ydual::Vector{Dual{T,Y,N}}
end

function VectorHessianConfig(
    f::F, result::DiffResult, x::AbstractArray{V}, 
    f_is_in_place::Bool=false, chunk::Chunk{N}=Chunk(x), tag::T=Tag(f, V)
) where {F, V, T, N}
    Dy = DiffResults.jacobian(result)
    outer_config = JacobianConfig((f, jacobian), Dy, x, chunk, tag)
    xdual = outer_config.duals[2]
    nout = size(Dy, 1)
    ydual = zeros(Dual{T,eltype(Dy),N}, nout)
    inner_config = if f_is_in_place
        JacobianConfig(f, ydual, xdual, chunk, tag)
    else
        JacobianConfig(f, xdual, chunk, tag)
    end
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
    f_is_in_place::Bool=false,
    cfg::VectorHessianConfig{T}=VectorHessianConfig(f, result, x, f_is_in_place), 
    ::Val{CHK}=Val{true}();
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
#================================ Vector Hessians ========================================#

function ad_grads!(Dy, backend::ForwardDiffBackend, func!, x, p, f_is_in_place::Val{true})
    @debug "Allocating temporary output array for `eval_grads!`."
    y = similar(Dy, size(Dy, 1))
    FD.jacobian!(transpose(Dy), FixParams!(func!, p), y, x)
    return nothing
end

function ad_grads!(Dy, backend::ForwardDiffBackend, func, x, p, f_is_in_place::Val{false})
    FD.jacobian!(transpose(Dy), FixParams(func, p), x)
    return nothing
end
#=
function ad_op_and_grads!(y, Dy, backend::ForwardDiffBackend, func!, x, p, f_is_in_place::Val{true})
    FD.jacobian!(transpose(Dy), FixParams!(func!, p), y, x)
    return nothing
end

function ad_op_and_grads!(y, Dy, backend::ForwardDiffBackend, func, x, p, f_is_in_place::Val{false})
    res = FD.DiffResults.DiffResult(y, transpose(Dy))
    FD.jacobian!(res, FixParams(func, p), x)
    return nothing
end
=#

function ad_hessians!(
    H, backend::ForwardDiffBackend, func!, x, p, f_is_in_place::Val{true}
)
    @debug "Allocating temporary output arrays for `eval_hessians!`."
    nvars, _, nout = size(H)
    y = similar(H, nout)
    Dy = similar(H, nout, nvars)
    res = DiffResult(y, Dy, H)
    vector_hessian!(res, FixParams!(func!, p), x, true)
    return nothing
end

function ad_hessians!(
    H, backend::ForwardDiffBackend, func, x, p, f_is_in_place::Val{false}
)
    @debug "Allocating temporary output arrays for `eval_hessians!`."
    nvars, _, nout = size(H)
    y = similar(H, nout)
    Dy = similar(H, nout, nvars)
    res = DiffResult(y, Dy, H)
    vector_hessian!(res, FixParams(func, p), x, false)
    return nothing
end

function ad_op_and_grads_and_hessians!(
    y, Dy, H, backend::ForwardDiffBackend, func, x, p, f_is_in_place::Val{true}
)
    F = FixParams!(func, p)
    res = DiffResult(y, transpose(Dy), H)
    vector_hessian!(res, F, x, true)
    return nothing
end

function ad_op_and_grads_and_hessians!(
    y, Dy, H, backend::ForwardDiffBackend, func, x, p, f_is_in_place::Val{false}
)
    F = FixParams(func, p)
    res = DiffResult(y, transpose(Dy), H)
    vector_hessian!(res, F, x, false,)
    return nothing
end

function ad_op_and_grads!(
    y, Dy, backend::ForwardDiffBackend, func, x, p, f_is_in_place::Val{true}
)
    res = FD.DiffResults.DiffResult(y, transpose(Dy))
    FD.jacobian!(res, FixParams!(func, p), y, x)
    return nothing
end

function ad_op_and_grads!(
    y, Dy, backend::ForwardDiffBackend, func, x, p, f_is_in_place::Val{false}
)
    res = FD.DiffResults.DiffResult(y, transpose(Dy))
    FD.jacobian!(res, FixParams(func, p), x)   
    return nothing
end
end#module