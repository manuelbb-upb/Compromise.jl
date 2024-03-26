abstract type AbstractDiagonalScaler{F} <: AbstractAffineScaler end
float_type(::AbstractDiagonalScaler{F}) where F=F

abstract type AbstractScalingDirection end
struct ForwardScaling <: AbstractScalingDirection end
struct InverseScaling <: AbstractScalingDirection end

toggle_scaling_dir(::ForwardScaling)=InverseScaling()
toggle_scaling_dir(::InverseScaling)=ForwardScaling()

supports_scaling_dir(::AbstractDiagonalScaler, ::AbstractScalingDirection)=Val(false)

abstract type AbstractAutomaticScalingArray end
struct IsAutomaticArray <: AbstractAutomaticScalingArray end
struct IsDefinedArray <: AbstractAutomaticScalingArray end

diag_matrix(::AbstractDiagonalScaler, ::AbstractScalingDirection)=error("define `diag_matrix`.")
scaler_offset(::AbstractDiagonalScaler, ::AbstractScalingDirection)=error("define `diag_matrix`.")

function smatrix(scaler::AbstractDiagonalScaler, sdir::AbstractScalingDirection)
    return smatrix(scaler, sdir, supports_scaling_dir(scaler, sdir))
end

function smatrix(scaler::AbstractDiagonalScaler, sdir::AbstractScalingDirection, ::Val{true})
    return diag_matrix(scaler, sdir)
end

function smatrix(scaler::AbstractDiagonalScaler, sdir::AbstractScalingDirection, ::Val{false})
    S = smatrix(scaler, toggle_scaling_dir(sdir), Val(true))
    return LA.inv(S)
end

function soffset(scaler::AbstractDiagonalScaler, sdir::AbstractScalingDirection)
    return soffset(scaler, sdir, supports_scaling_dir(scaler, sdir))
end

function soffset(scaler::AbstractDiagonalScaler, sdir::AbstractScalingDirection, ::Val{true})
    return scaler_offset(scaler, sdir)
end

function soffset(scaler::AbstractDiagonalScaler, sdir::AbstractScalingDirection, ::Val{false})
    # ξ = S * x + s ⇔ x = S⁻¹(ξ-s) = S⁻¹ξ - S⁻¹s
    idir = toggle_scaling_dir(sdir)
    s = soffset(scaler, idir, Val(true))
    Sinv = smatrix(scaler, idir)
    return - Sinv * s
end

abstract type AbstractScalerOffsetTrait end
struct HasOffset <: AbstractScalerOffsetTrait end
struct NoOffset <: AbstractScalerOffsetTrait end

offset_trait(::AbstractDiagonalScaler) = NoOffset()

function apply_scaling!(x::Nothing, scaler::AbstractDiagonalScaler, sdir::AbstractScalingDirection)
    return nothing
end

function apply_scaling!(x, scaler::AbstractDiagonalScaler, sdir::AbstractScalingDirection)
    return apply_scaling!(x, scaler, sdir, offset_trait(scaler))
end

function apply_scaling!(
    x, scaler::AbstractDiagonalScaler, 
    sdir::AbstractScalingDirection, has_offset::NoOffset
)
    S = smatrix(scaler, sdir)
    LA.lmul!(S, x)
    return x
end

function apply_scaling!(
    x, scaler::AbstractDiagonalScaler, 
    sdir::AbstractScalingDirection, has_offset::HasOffset
)
    apply_scaling!(x, scaler, sdir, NoOffset())
    s = soffset(scaler, sdir)
    x .+= s
    return x
end

struct IdentityScaler{N, F} <: AbstractDiagonalScaler{F} end

supports_scaling_dir(::IdentityScaler, ::AbstractScalingDirection)=Val(true)
diag_matrix(::IdentityScaler{N}, ::Union{ForwardScaling, InverseScaling}) where N = LA.I(N)

struct DiagonalScaler{F} <: AbstractDiagonalScaler{F}
    fmat :: LA.Diagonal{F, Vector{F}}
    imat :: LA.Diagonal{F, Vector{F}}
end

supports_scaling_dir(::DiagonalScaler, ::AbstractScalingDirection)=Val(true)
diag_matrix(scaler::DiagonalScaler, ::ForwardScaling)=scaler.fmat
diag_matrix(scaler::DiagonalScaler, ::InverseScaling)=scaler.imat

struct DiagonalScalerWithOffset{F, W<:AbstractDiagonalScaler{F}} <: AbstractDiagonalScaler{F}
    wrapped :: W
    foff :: Vector{F}
    ioff :: Vector{F}
end
@forward DiagonalScalerWithOffset.wrapped supports_scaling_dir(scaler::DiagonalScalerWithOffset, ::AbstractScalingDirection)
@forward DiagonalScalerWithOffset.wrapped diag_matrix(scaler::DiagonalScalerWithOffset, ::ForwardScaling)
@forward DiagonalScalerWithOffset.wrapped diag_matrix(scaler::DiagonalScalerWithOffset, ::InverseScaling)

offset_trait(::DiagonalScalerWithOffset) = HasOffset()
scaler_offset(scaler::DiagonalScalerWithOffset, ::ForwardScaling)=scaler.foff
scaler_offset(scaler::DiagonalScalerWithOffset, ::InverseScaling)=scaler.ioff

# ============================================================================= #
function scale_lin_cons!(trgt_cons, scaler, lin_cons)
    universal_copy!(trgt_cons, lin_cons)
    @unpack A, b, E, c, lb, ub = trgt_cons
    scaling_sense = ForwardScaling()

    apply_scaling!(lb, scaler, scaling_sense)
    apply_scaling!(ub, scaler, scaling_sense)
    
    scale_eq!(A, b, scaler)
    scale_eq!(E, c, scaler)
    return nothing
end

"Make `Aξ ? b` applicable in scaled domain."
@views function scale_eq!(
    A::AbstractMatrix, b::AbstractVector, 
    scaler
)
    scaling_sense = InverseScaling()
    # ξ = S * x + s ⇒ A*ξ = A*S*x + A*s ? b ⇒ (A * S) * x ? (b - (A * s))
    # Hence, `_A = A*S`, `_b = b - A*s`
    S = smatrix(scaler, scaling_sense)
    s = soffset(scaler, scaling_sense)

    LA.mul!(b, A, s, -1, 1)     # b ← b - A * s
    LA.rmul!(A, S)              # A ← A * S

    return nothing
end
scale_eq!(A, b, scaler) = nothing

# Consider ``f: ℝ^n → ℝ^m`` and the unscaling map ``u: ℝ^n → ℝ^n``.
# By the chain rule we have ``∇(f∘u)(x) = ∇f(ξ)*T``, where ``T`` is 
# the Jacobian of ``u``.
# As we are actually working with transposed Jacobians `Dy`, we compute
# `T'Dy`:
function scale_grads!(Dy, scaler::AbstractAffineScaler)
    T = transpose(smatrix(scaler, InverseScaling()))
    LA.lmul!(T, Dy)
    return nothing
end

# Usually, the chain rule for second-order derivatives is more complicated:
# ```math
#   ∇²(f∘u)(x) = Tᵀ∇²f(ξ)T + ∑ₖ ∇fₖ(ξ) ∇²u
# ```
# But ``u`` is affine, so the second term vanishes and we are left with 
# the matrix-matrix-matrix product:
function scale_hessians!(H, scaler)
    T = smatrix(scaler, InverseScaling())
    T = unscaling_matrix(scaler)
    for i = axes(H, 3)
        Hi = @view(H[:, :, i])
        LA.lmul!(T', Hi)
        LA.rmul!(Hi, T)
    end
    return nothing
end

# ============================================================================= #
function scale_to_unit_length(
    lb::AbstractVector{T}, ub::AbstractVector{S};
    allow_inf::Bool=true, throw_error::Bool=false
) where {T, S}
    return first(_scale_to_unit_length(lb, ub; allow_inf, throw_error))
end

function _scale_to_unit_length(
    lb::AbstractVector{T}, ub::AbstractVector{S};
    allow_inf::Bool=true, throw_error::Bool=false
) where {T, S}
    n = length(lb)
    @assert n == length(ub) "`lb` and `ub` have to be of same length."
    F = Base.promote_op(/, Int, promote_type(T, S))
    
    w = Vector{F}(undef, length(lb))
    is_inf = zeros(Bool, n)
    for i=1:n
        l = lb[i]
        if isinf(l)
            is_inf[i] = true
            if !allow_inf
                break
            end
        end
        u = ub[i]
        if isinf(u)
            is_inf[i] = true
            if !allow_inf
                break
            end
        end
        w[i] = u - l
    end

    if !allow_inf && any(is_inf)
        if throw_error
            error("Infinite boundary values not allowd.")
        end
        return IdentityScaler{n, F}(), is_inf
    end

    if all(is_inf)
        return IdentityScaler{n, F}(), is_inf
    end

    w[is_inf] .= 1

    imat = LA.Diagonal(w)
    fmat = LA.inv(imat)

    return DiagonalScaler(fmat, imat), is_inf
end

function scale_to_zero_one(
    lb::AbstractVector, ub::AbstractVector;
    allow_inf::Bool=true, throw_error::Bool=false
)
    wrapped, is_inf = _scale_to_unit_length(lb, ub; allow_inf, throw_error)
    F = float_type(wrapped)
    foff = - wrapped.fmat * lb
    ioff = F.(lb)
    
    foff[is_inf] .= 0
    ioff[is_inf] .= 0

    return DiagonalScalerWithOffset(wrapped, foff, ioff)     
end

init_box_scaler(::Nothing, ub, N, F) = IdentityScaler{N,F}()
init_box_scaler(lb, ::Nothing, N, F) = IdentityScaler{N,F}()
init_box_scaler(::Nothing, ::Nothing, N, F) = IdentityScaler{N,F}()
function init_box_scaler(lb, ub, N, F)
    return scale_to_zero_one(lb, ub)
end