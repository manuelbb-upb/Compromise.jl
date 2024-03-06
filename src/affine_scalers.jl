# In case of badly scaled problems, we offer tools to try to mitigate bad conditioning (a bit).
# The algorithm always assumes the variable `x` to live in the scaled domain.
# By `ξ` we denote the unscaled variables (the original problem variables).

update_scaler!(AbstractDynamicAffineScaler, args...;kwargs...)=nothing # TODO
#src When implementing it, also build a safe-guarding wrapper to re-scale linear constraints.

scaling_matrix(scaler::AbstractAffineScaler)=1
unscaling_matrix(scaler::AbstractAffineScaler)=LA.inv(scaling_matrix(scaler))

scaling_offset(scaler::AbstractAffineScaler)=0
unscaling_offset(scaler::AbstractAffineScaler)=-unscaling_matrix(scaler)*scaling_offset(scaler)

"Scale `ξ` and set `x` according to `x = T*ξ + t`."
function scale!(x, scaler::AbstractAffineScaler, ξ)
    T = scaling_matrix(scaler)
    t = scaling_offset(scaler)
    LA.mul!(x, T, ξ)
    @views x .+= t
    return nothing
end
"Unscale `x` and set `ξ` according to `ξ = inv(T)*x - inv(T)*t`."
function unscale!(ξ, scaler::AbstractAffineScaler, x)
    Tinv = unscaling_matrix(scaler)
    tinv = unscaling_offset(scaler)
    LA.mul!(ξ, Tinv, x)
    @views ξ .+= tinv
    return nothing
end

scale!(x::Nothing, scaler::AbstractAffineScaler, ξ::Nothing) = nothing
unscale!(ξ::Nothing, scaler::AbstractAffineScaler, x::Nothing) = nothing

"Make `Aξ + b ? 0` applicable in scaled domain via `A(inv(T)*x - inv(T)*t) + b ? 0`."
@views function scale_eq!(
    _A::AbstractMatrix, _b::AbstractVector, 
    scaler::AbstractAffineScaler, A::AbstractMatrix, b::AbstractVector
)
    Tinv = unscaling_matrix(scaler)
    tinv = unscaling_offset(scaler)
    _b .= b
    _b .+= tinv
    LA.mul!(_A, A, Tinv)
    return nothing
end
scale_eq!(_A, _b, scaler::AbstractAffineScaler, A, b)=nothing

# Consider ``f: ℝ^n → ℝ^m`` and the unscaling map ``u: ℝ^n → ℝ^n``.
# By the chain rule we have ``∇(f∘u)(x) = ∇f(ξ)*T``, where ``T`` is 
# the Jacobian of ``u``.
# As we are actually working with transposed Jacobians `Dy`, we compute
# `T'Dy`:
function scale_grads!(Dy, scaler::AbstractAffineScaler)
    _Dy = copy(Dy)
    LA.mul!(Dy, transpose(unscaling_matrix(scaler)), _Dy)
    return nothing
end

# Usually, the chain rule for second-order derivatives is more complicated:
# ```math
#   ∇²(f∘u)(x) = Tᵀ∇²f(ξ)T + ∑ₖ ∇fₖ(ξ) ∇²u
# ```
# But ``u`` is affine, so the second term vanishes and we are left with 
# the matrix-matrix-matrix product:
function scale_hessians!(H, scaler)
    T = unscaling_matrix(scaler)
    _H = copy(H)
    for i = axes(H, 3)
        LA.mul!(_H[:, :, i], transpose(T), H[:, :, i])
        LA.mul!(H[:, :, i], _H[:, :, i], T)
    end
    return nothing
end

struct IdentityScaler <: AbstractConstantAffineScaler 
    dim :: Int
end
@batteries IdentityScaler

scale!(x::RVec, scaler::IdentityScaler, ξ::RVec)=copyto!(x, ξ)
unscale!(ξ::RVec, scaler::IdentityScaler, x::RVec)=copyto!(ξ, x)
scaling_matrix(scaler::IdentityScaler) = LA.I(scaler.dim)
unscaling_matrix(scaler::IdentityScaler) = LA.I(scaler.dim)

## ξ = Tx + b
## x = T⁻¹(ξ - b) = T⁻¹ξ - T⁻¹b
struct DiagonalVarScaler{F<:AbstractFloat} <: AbstractConstantAffineScaler
    T :: LA.Diagonal{F, Vector{F}}
    Tinv :: LA.Diagonal{F, Vector{F}}
    b :: Vector{F}
    binv :: Vector{F}
end
@batteries DiagonalVarScaler

scaling_matrix(scaler::DiagonalVarScaler)=scaler.T
unscaling_matrix(scaler::DiagonalVarScaler)=scaler.Tinv
scaling_offset(scaler::DiagonalVarScaler)=scaler.b
unscaling_offset(scaler::DiagonalVarScaler)=scaler.binv

init_box_scaler(lb, ub, dim)=IdentityScaler(dim)
function init_box_scaler(lb::RVec, ub::RVec, dim)
    if any(isinf.(lb)) || any(isinf.(ub))
        return init_box_scaler(nothing, nothing, dim)
    end

    ## set up a min-max scaler
    ## xᵢ is scaled to be contained in [0,1] via xᵢ = (ξᵢ - lᵢ)/(uᵢ - lᵢ)
    ## We setup `T` to contain the divisors `w` and `b` to have the offset `-lb ./ w`.
    w = ub .- lb
    T = LA.Diagonal(1 ./ w)
    b = - lb ./ w

    ## the unscaling is `ξ = (x + lb ./ w) .* w` = x .* w + lb
    Tinv = LA.Diagonal(w)
    binv = lb

    return DiagonalVarScaler(T, Tinv, b, binv)
end

function scale_grads!(Dy, scaler::IdentityScaler)
    return nothing
end
function scale_grads!(Dy, scaler::DiagonalVarScaler)
    LA.lmul!(transpose(unscaling_matrix(scaler)), Dy)
    return nothing
end
function scale_hessians!(H, scaler::IdentityScaler)
    return nothing
end
function scale_hessians!(H, scaler::DiagonalVarScaler)
    T = unscaling_matrix(scaler)
    for i = axes(H, 3)
        LA.lmul!(transpose(T), H[:, :, i])
        LA.rmul!(H[:, :, i], T)
    end
    return nothing
end