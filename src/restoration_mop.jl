# This file describes type to treat the restoration problem of an outer run 
# as a MOP, which is to be solved in an inner run.
# Mathematically, the restoration problem of the outer run is 
#=
```math
\min_{x} 
\max \{0, g(x), |h(x)|, Ax-b, |Ex-c|, lb - x, x - ub\}
```
Let's assume that ``x`` is feasible for the box constraints and that we can maintain 
feasibility for box-constraints.
We can rewrite as
```math
\begin{aligned}
&\min_{x, t} t &\text{subject to} \\
&t ≥ 0, \\
g(x) ≤ t, \\
|h(x)| ≤ t, \\
Ax - b ≤ t, \\
|Ex - c| ≤ t.
\end{aligned}
```
Further, the absolute value of equality constraints can be eliminated:
```math
|h(x)| ≤ t ⇔ -t ≤ h(x) ≤ t ⇔ (-t - h(x) ≤ 0) ∧ (-t + h(x) ≤ 0) 
```
In standard form, the new problem reads:
```math
\begin{aligned}
&
    \min{t, x} t 
    &
        \text{subject to} 
    \\
&
    t ∈ [0, ∞), \; lb ≤ x ≤ ub,
    \\
&
    -t +g(x) ≤ 0,
    \\
&
    -t -h(x) ≤ 0,
    \\
&
    -t +h(x) ≤ 0,
    \\
&
    -t + Ax - b ≤ 0,
    \\
&
    -t - Ex + c ≤ 0,
    \\
&
    -t + Ex - c ≤ 0.
\end{aligned}
```
The new problem has 
* ``n+1`` variables, ``[t, x_1, …, x_n]``,
* one objective,
* and only inequality constraints (nonlinear and linear).

We want to re-use as much stuff as possible from the outer optimization loop.
=#

include("CheetahViews/CheetahViews.jl")

import PaddedViews: PaddedView
import LazyArrays: BroadcastArray

import .CheetahViews: cheetah_vcat, cheetah_blockcat

Base.@kwdef function RestorationValueCache{F, wrappedType} <: AbstractValueCache{F}
    wrapped :: wrappedType
    t :: Base.RefValue{F} = Ref(float_type(wrapped)(NaN))
    const_empty_vec :: Vector{F} = array(float_type(wrapped), 0)
end

Base.@kwdef struct RestorationMOP{
    F, MOPType, MODType, LB<:AbstractVector{F}, UB<:AbstractVector{F},
    AType, BType, ValsType, SType
} <: AbstractMOP

    mop :: MOPType
    mod :: MODType
    
    dim_vars :: Int
    dim_nl_ineq_constraints :: Int

    lb :: LB
    ub :: UB

    A :: AType
    b :: BType

    vals :: ValsType
    scaler :: SType
end

float_type(::RestorationMOP{F}) where{F} = F
dim_vars(rmop::RestorationMOP) = rmop.dim_vars
dim_objectives(rmop::RestorationMOP) = 1
dim_nl_ineq_constraints(rmop::RestorationMOP) = rmop.dim_nl_ineq_constraints
lower_var_bounds(rmop::RestorationMOP) = rmop.lb
upper_var_bounds(rmop::RestorationMOP) = rmop.ub

lin_ineq_constraints_matrix(rmop::RestorationMOP) = rmop.A
lin_ineq_constraints_vector(rmop::RestorationMOP) = rmop.b

function RestorationMOP(mop, mod, theta_k, scaler, vals, scaled_lin_cons)
    lb = _restoration_lower_var_bounds(mop, scaled_lin_cons)    
    ub = _restoration_upper_var_bounds(mop, scaled_lin_cons, theta_k)
    #ξ = array(float_type(mop), dim_vars(mop))
    A, b = _restoration_constraint_arrays(scaled_lin_cons)
    _dim_vars = dim_vars(mop) + 1
    _dim_nl_ineq_constraints = dim_nl_ineq_constraints(mop) + 2 * dim_nl_eq_constraints(mop)
    return RestorationMOP(;
        mop, mod, lb, ub, A, b, 
        vals,
        dim_vars=_dim_vars, dim_nl_ineq_constraints=_dim_nl_ineq_constraints, 
        scaler
    )
end
#=

struct RestorationScaler{F, W<:AbstractDiagonalScaler} <: AbstractDiagonalScaler
    f :: F
    i :: F
    wrapped :: W
end

_restoration_scaler_constant(rscaler, ::ForwardScaling)=rscaler.f
_restoration_scaler_constant(rscaler, ::InverseScaling)=rscaler.i

function _restoration_scaler(theta_k, scaler)
    f = theta_k
    i = 1/theta_k
    return RestorationScaler(f, i, scaler)
end

function supports_scaling_dir(::RestorationScaler, ::AbstractScalingDirection)
    return Val(true)
end

function offset_trait(rscaler::RestorationScaler)
    return offset_trait(rscaler.wrapped)
end

function diag_matrix(rscaler::RestorationScaler, sense::AbstractScalingDirection)
    # this is allocating, but should not be called anywhere, as we are also specializing
    # `apply_scaling`...
    return LA.Diagonal(
        cat(
            _restoration_scaler_constant(rscaler, sense), 
            smatrix(rscaler.wrapped, sense);
            dims=(1,2)
        )
    )
end

function scaler_offset(rscaler::RestorationScaler, sense::AbstractScalingDirection)
    return vcat(0, soffset(rscaler.wrapped, sense))
end

function apply_scaling!(x, rscaler::RestorationScaler, sense::AbstractScalingDirection)
    c = _restoration_scaler_constant(rscaler, sense)
    x[1] *= c
    apply_scaling!(@view(x[2:end]), rscaler.wrapped, sense)
    return x
end
=#

function _restoration_lower_var_bounds(mop, scaled_lin_cons)
    return _restoration_lower_var_bounds(
        mop, scaled_lin_cons.lb
    )
end

function _restoration_lower_var_bounds(mop, ::Nothing)
    mop_lb = _inner_lb(mop)
    return _restoration_lower_var_bounds(mop, mop_lb)
end

function _restoration_lower_var_bounds(mop, mop_lb)
    return _restoration_padded_bounds(mop_lb, 0)
end

function _restoration_upper_var_bounds(mop, scaled_lin_cons, theta_k)
    return _restoration_upper_var_bounds(
        mop, scaled_lin_cons.ub, theta_k
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

function _restoration_constraint_arrays(scaled_lin_cons)
    @unpack E, c, A, b = scaled_lin_cons
    ## |Ex - c| ≤ t
    ## (i)   -t + Ex ≤ c    ⇔    Ex - c ≤ t
    ## (ii)  -t - Ex ≤ -c   ⇔   -t ≤ Ex - c
    ## and 
    ## Ax - b ≤ t
   
    ## build RHS vector `g=[c; -c; b]`
    g = _combine_constraint_arrays(c, b)

    ## first stack `_G=[E; -E; A]`
    _G = _combine_constraint_arrays(E, A)
    ## then augment with column containing `-1` (factor for `t`).
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
    @unpack ξ, scaler, mop = rmop
    t = tx[1]
    @views ξ .= tx[2:end]
    apply_scaling!(ξ, scaler, InverseScaling())

    i = 1
    
    ## |h(x)| ≤ t ⇔ -t ≤ h(x) ≤ t ⇔ 
    ## (i)   h(x)-t ≤ 0
    ## (ii) -h(x)-t ≤ 0
    _j = dim_nl_eq_constraints(mop)
    if _j > 0
        j = i + _j - 1
        yh = @view(y[i:j])
        @ignoraise nl_eq_constraints!(yh, mop,ξx)
        
        i = j + 1
        j = i + _j - 1
        _yh = @view(y[i:j])
        _yh .= -yh
        
        yh .-= t
        _yh .-= t

        i = j + 1
    end

    # g(x) - t ≤ 0
    _j = dim_nl_ineq_constraints(mop) 
    if _j > 0
        j = i + _j - 1
        yg = @view(y[i:j])
        @ignoraise nl_ineq_constraints!(yg, mop, ξ)
        yg .-= t
        i = j + 1
    end

    return nothing
end

struct RestorationSurrogate{M} <: AbstractMOPSurrogate 
    mop :: M
end

function init_models(
    mop::RestorationMOP, scaler; 
    kwargs...
)
    return RestorationSurrogate(mop)
end

@forward RestorationSurrogate.mop float_type(mod::RestorationSurrogate)
@forward RestorationSurrogate.mop dim_vars(mod::RestorationSurrogate)
@forward RestorationSurrogate.mop dim_objectives(mod::RestorationSurrogate)
@forward RestorationSurrogate.mop dim_nl_ineq_constraints(mod::RestorationSurrogate)
# TODO prealloc with views

function update_models!(
    mod::RestorationSurrogate, Δ, scaler, vals, scaled_cons;
    log_level, indent::Int
)
    @unpack mop = mod
    @unpack x = mop.vals
    @unpack scaler = mop
    #apply_scaling
end
    