include("CheetahViews/CheetahViews.jl")

import PaddedViews: PaddedView
import LazyArrays: BroadcastArray

import .CheetahViews: cheetah_vcat, cheetah_blockcat

Base.@kwdef struct RestorationMOP{
    F, M, LB<:AbstractVector{F}, UB<:AbstractVector{F}, AType, BType, OS, IS
} <: AbstractMOP

    mop :: M
    dim_vars :: Int
    dim_nl_ineq_constraints :: Int
    lb :: LB
    ub :: UB

    A :: AType
    b :: BType

    outer_scaler :: OS
    inner_scaler :: IS
end

float_type(::RestorationMOP{F}) where{F} = F
dim_vars(rmop::RestorationMOP) = rmop.dim_vars
dim_objectives(rmop::RestorationMOP) = 1
dim_nl_ineq_constraints(rmop::RestorationMOP) = rmop.dim_nl_ineq_constraints
lower_var_bounds(rmop::RestorationMOP) = rmop.lb
upper_var_bounds(rmop::RestorationMOP) = rmop.ub

lin_ineq_constraints_matrix(rmop::RestorationMOP) = rmop.A
lin_ineq_constraints_vector(rmop::RestorationMOP) = rmop.b

function RestorationMOP(mop::M, theta_k, outer_scaler) where M<:AbstractMOP
    lb = _restoration_lower_var_bounds(mop)    
    ub = _restoration_upper_var_bounds(mop, theta_k)
    A, b = _restoration_constraint_arrays(mop)
    _dim_vars = dim_vars(mop) + 1
    _dim_nl_ineq_constraints = dim_nl_ineq_constraints(mop) + 2 * dim_nl_eq_constraints(mop)
    inner_scaler = _restoration_scaler(theta_k, outer_scaler)
    return RestorationMOP(;mop, lb, ub, A, b, 
        dim_vars=_dim_vars, dim_nl_ineq_constraints=_dim_nl_ineq_constraints, 
        outer_scaler, inner_scaler
    )
end

function _restoration_scaler(theta_k, scaler)
    f = theta_k
    i = 1/theta_k
    fmat = cat(f, smatrix(scaler, ForwardScaling()); dims=(1,2))
    imat = cat(i, smatrix(scaler, InverseScaling()); dims=(1,2))
    foff = vcat(0, soffset(scaler, ForwardScaling()))
    ioff = vcat(0, soffset(scaler, InverseScaling()))
    return DiagonalScalerWithOffset(
        DiagonalScaler(fmat, imat),
        foff, 
        ioff
    )
end

function _restoration_lower_var_bounds(mop::AbstractMOP)
    return _restoration_lower_var_bounds(
        mop, lower_var_bounds(mop)
    )
end

function _restoration_lower_var_bounds(mop, ::Nothing)
    mop_lb = _inner_lb(mop)
    return _restoration_lower_var_bounds(mop, mop_lb)
end

function _restoration_lower_var_bounds(mop, mop_lb)
    return _restoration_padded_bounds(mop_lb, 0)
end

function _restoration_upper_var_bounds(mop::AbstractMOP, theta_k)
    return _restoration_upper_var_bounds(
        mop, upper_var_bounds(mop), theta_k
    )
end
function _restoration_upper_var_bounds(mop, ::Nothing, theta_k)
    mop_ub = _inner_ub(mop)
    return _restoration_upper_var_bounds(mop, mop_ub, theta_k)
end

function _restoration_upper_var_bounds(mop, mop_ub, theta_k)
    return _restoration_padded_bounds(mop_ub, theta_k)
end

function _restoration_padded_bounds(bb, pv)
    N = length(bb) + 1
    return PaddedView(pv, bb, (N,), (2,))
end

_inner_lb(mop) = _inner_bounds(mop, minus_inf(mop))
_inner_ub(mop) = _inner_bounds(mop, plus_inf(mop))

minus_inf(mop) = convert(float_type(mop), -Inf)
plus_inf(mop) = convert(float_type(mop), Inf)

function _inner_bounds(mop, finf::F) where F
    return PaddedView(finf, Vector{F}(undef, 0), (dim_vars(mop),))
end

function _restoration_constraint_arrays(mop)
    ## |Ex - c| ≤ t
    ## (i)   -t + Ex ≤ c    ⇔    Ex - c ≤ t
    ## (ii)  -t - Ex ≤ -c   ⇔   -t ≤ Ex - c
    ## and 
    ## Ax - b ≤ t
    E = lin_eq_constraints_matrix(mop)
    c = lin_eq_constraints_vector(mop)
    A = lin_ineq_constraints_matrix(mop)
    b = lin_ineq_constraints_vector(mop)
   
    g = _combine_constraint_arrays(c, b)

    _G = _combine_constraint_arrays(E, A)
    G = if isnothing(_G)
        nothing
    else
        m, n = size(_G)
        PaddedView(-1, _G, (m, n+1), (1,2))
    end
    return G, g
end

_combine_constraint_arrays(::Nothing, ::Nothing)=nothing
_combine_constraint_arrays(E::Nothing, A)=A
function _combine_constraint_arrays(E, A::Nothing)
    _E = BroadcastArray(-, E)
    return cheetah_vcat(E, _E)
end
function _combine_constraint_arrays(E, A)
    _E = BroadcastArray(-, E)
    return cheetah_vcat(E, _E, A)
end

function eval_objectives!(y::RVec, rmop::RestorationMOP, x::RVec)
    y[1] = x[1]
    return nothing
end

function eval_nl_ineq_constraints!(y::RVec, rmop::RestorationMOP, tx::RVec)
    t = tx[1]
    x = @view(tx[2:end])
    mop = rmop.mop

    i = 1
    
    ## |h(x)| ≤ t ⇔ -t ≤ h(x) ≤ t ⇔ 
    ## (i)   h(x)-t ≤ 0
    ## (ii) -h(x)-t ≤ 0
    _j = dim_nl_eq_constraints(mop)
    if _j > 0
        j = i + _j - 1
        yh = @view(y[i:j])
        @ignoraise nl_eq_constraints!(yh, mop, x)
        
        i = j + 1
        j = i + _j - 1
        _yh = @view(y[i:j])
        _yh .= -yh
        
        yh .-= t
        _yh .-= t

        i = j + 1
    end

    # g(x) ≤ t
    _j = dim_nl_ineq_constraints(mop) 
    if _j > 0
        j = i + _j - 1
        yg = @view(y[i:j])
        @ignoraise nl_ineq_constraints!(yg, mop, x)
        yg .-= t
        i = j + 1
    end

    return nothing
end