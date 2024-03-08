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
    return affine_map!(x, T, t, ξ)
end

"Unscale `x` and set `ξ` according to `ξ = S*x + s`."
function unscale!(ξ, scaler::AbstractAffineScaler, x)
    ## Suppose `T = scaling_matrix(scaler)` and `t = unscaling_offset(scaler)`.
    ## If `x = T*ξ + t`, then `ξ = T⁻¹(x - t)`, 
    ## so usually `S = inv(T)` and `s = -inv(T) * t`.
    S = unscaling_matrix(scaler)
    s = unscaling_offset(scaler)
    return affine_map!(ξ, S, s, x)
end

function affine_map!(out, factor, offset, in)
    LA.mul!(out, factor, in)
    out .+= offset
    return nothing
end

scale!(x::Nothing, scaler::AbstractAffineScaler, ξ::Nothing) = nothing
unscale!(ξ::Nothing, scaler::AbstractAffineScaler, x::Nothing) = nothing

"Make `Aξ ? b` applicable in scaled domain."
@views function scale_eq!(
    _A::AbstractMatrix, _b::AbstractVector, 
    scaler::AbstractAffineScaler, A::AbstractMatrix, b::AbstractVector
)
    # ξ = S * x + s ⇒ A*ξ = A*S*x + A*s ? b ⇒ (A * S) * x ? (b - (A * s))
    # Hence, `_A = A*S`, `_b = b - A*s`
    S = unscaling_matrix(scaler)
    s = unscaling_offset(scaler)
    _b .= b
    LA.mul!(_b, A, s, -1, 1)
    LA.mul!(_A, A, S)
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
scaling_offset(scaler::IdentityScaler) = 0
unscaling_offset(scaler::IdentityScaler) = 0

## ξ = Tx + b
## x = T⁻¹(ξ - b) = T⁻¹ξ - T⁻¹b
struct DiagonalVarScaler{F<:AbstractFloat} <: AbstractConstantAffineScaler
    T :: LA.Diagonal{F, Vector{F}}
    S :: LA.Diagonal{F, Vector{F}}
    t :: Vector{F}
    s :: Vector{F}
end
@batteries DiagonalVarScaler

scaling_matrix(scaler::DiagonalVarScaler)=scaler.T
unscaling_matrix(scaler::DiagonalVarScaler)=scaler.S
scaling_offset(scaler::DiagonalVarScaler)=scaler.t
unscaling_offset(scaler::DiagonalVarScaler)=scaler.s

init_box_scaler(lb, ub, dim)=IdentityScaler(dim)
function init_box_scaler(lb::RVec, ub::RVec, dim)
    if any(isinf.(lb)) || any(isinf.(ub))
        return init_box_scaler(nothing, nothing, dim)
    end

    ## set up a min-max scaler
    ## xᵢ is scaled to be contained in [0,1] via xᵢ = (ξᵢ - lᵢ)/(uᵢ - lᵢ).
    ## ⇒ x = ξ ./ w - lb ./ w
    ## We setup `T` to contain the divisors `w` and `t` to have the offset `-lb ./ w`.
    w = ub .- lb
    T = LA.Diagonal(1 ./ w)
    t = -lb ./ w

    ## the unscaling is `ξ = x .* w + lb
    S = LA.Diagonal(w)
    s = lb

    return DiagonalVarScaler(T, S, t, s)
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