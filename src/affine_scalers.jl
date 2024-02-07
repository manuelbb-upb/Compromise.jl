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
@views function scale_eq!((_A, _b), scaler::AbstractAffineScaler, (A,b))
    Tinv = unscaling_matrix(scaler)
    tinv = unscaling_offset(scaler)
    _b .= b
    _b .+= tinv
    LA.mul!(_A, A, Tinv)
    return nothing
end
scale_eq!(::Nothing, scaler::AbstractAffineScaler, ::Nothing)=nothing

struct IdentityScaler <: AbstractConstantAffineScaler 
    dim :: Int
end

scale!(x::RVec, scaler::IdentityScaler, ξ::RVec)=copyto!(x, ξ)
unscale!(ξ::RVec, scaler::IdentityScaler, x::RVec)=copyto!(ξ, x)
scaling_matrix(scaler::IdentityScaler) = LA.I(scaler.dim)
unscaling_matrix(scaler::IdentityScaler) = LA.I(scaler.dim)

## ξ = Tx + b
## x = T⁻¹(ξ - b) = T⁻¹ξ - T⁻¹b
struct AffineVarScaler{
    TType1<:RMat, bType1<:RVec, TType2<:RMat, bType2<:RVec
} <: AbstractConstantAffineScaler
    T :: TType1
    Tinv :: TType2 
    b :: bType1
    binv :: bType2
end

scaling_matrix(scaler::AffineVarScaler)=scaler.T
unscaling_matrix(scaler::AffineVarScaler)=scaler.Tinv
scaling_offset(scaler::AffineVarScaler)=scaler.b
unscaling_offset(scaler::AffineVarScaler)=scaler.binv

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

    return AffineVarScaler(T, Tinv, b, binv)
end