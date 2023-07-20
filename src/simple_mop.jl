import .CompromiseEvaluators: NonlinearFunction, AbstractSurrogateModel, 
    AbstractNonlinearOperator, AbstractNonlinearOperatorNoParams, ExactModel
import .CompromiseEvaluators: requires_grads, provides_grads, requires_hessians, provides_hessians
import .CompromiseEvaluators: eval_op!, eval_grads!, eval_op_and_grads!
import .CompromiseEvaluators: model_op!, model_grads!, model_op_and_grads!
import .CompromiseEvaluators: NoBackend
import .CompromiseEvaluators: TaylorPolynomial1, TaylorPolynomial2
import .CompromiseEvaluators: init_surrogate, update!

struct ScaledOperator{O, S, XI, D, H} <: AbstractNonlinearOperatorNoParams
    op :: O
    scaler :: S
    ## cache for unscaled site
    ξ :: XI
    ## cache for gradients of `op` at `ξ`
    Dy :: D
    ## cache for Hessians of `op` at `ξ`, multiplied by unscaling matrix
    Hy :: H
end

function scale_grads!(Dy, sop)
    ## unscaling: u(x) = ξ = Tx + b => jacobian is T
    ## composition op(u(x)) => jacobian is Dy'T => grads are T'Dy
    LA.mul!(Dy, transpose(unscaling_matrix(sop.scaler)), sop.Dy)
    return nothing
end
function scale_hessians!(H, sop)
    ## H(op ∘ u) = T' H(op) * T + ∑_k (∇op)_k * Hu
    ## but u is affine, the last term vanishes
    T = unscaling_matrix(sop.scaler)
    for i = axes(H, 3)
        LA.mul!(sop.Hy[:, :, i], transpose(T), H[:, :, i])
        LA.mul!(H[:, :, i], sop.Hy[:, :, i], T)
    end
    return nothing
end

function eval_op!(y, sop::ScaledOperator, x)
    unscale!(sop.ξ, sop.scaler, x)
    return eval_op!(y, sop.op, sop.ξ)
end

function eval_grads!(Dy, sop::ScaledOperator, x)
    unscale!(sop.ξ, sop.scaler, x)
    eval_grads!(sop.Dy, sop.op, sop.ξ)
    scale_grads!(Dy, sop)
    return nothing
end

function eval_op_and_grads!(y, Dy, sop::ScaledOperator, x)
    unscale!(sop.ξ, sop.scaler, x)
    eval_op_and_grads!(y, sop.Dy, sop.op, sop.ξ)
    scale_grads!(Dy, so)
    return nothing
end

function eval_hessians!(H, sop::ScaledOperator, x)
    unscale!(sop.ξ, sop.scaler, x)
    eval_hessians!(H, sop.op, sop.ξ)
    scale_hessians!(H, sop)
    return nothing
end

function eval_op_and_grads_and_hessians!(y, Dy, H, sop, x)
    unscale!(sop.ξ, sop.scaler, x)
    eval_op_and_grads_and_hessians!(y, sop.Dy, H, sop.op, sop.ξ)
    scale_grads!(Dy, sop)
    scale_hessians!(H, sop)
    return nothing
end

abstract type SimpleMOP <: AbstractMOP end

@with_kw mutable struct MutableMOP <: SimpleMOP
    objectives :: Union{Nothing, NonlinearFunction} = nothing
    nl_eq_constraints :: Union{Nothing, NonlinearFunction} = nothing
    nl_ineq_constraints :: Union{Nothing, NonlinearFunction} = nothing

    num_vars :: Int

    dim_objectives :: Int = 0
    dim_nl_eq_constraints :: Int = 0
    dim_nl_ineq_constraints :: Int = 0

    mtype_objectives :: Type = ExactModel
    mtype_nl_eq_constraints :: Type = ExactModel
    mtype_nl_ineq_constraints :: Type = ExactModel

    lb :: Union{Nothing, Vector{Float64}} = nothing
    ub :: Union{Nothing, Vector{Float64}} = nothing

    E :: Union{Nothing, Matrix{Float64}} = nothing
    c :: Union{Nothing, Vector{Float64}} = nothing
    A :: Union{Nothing, Matrix{Float64}} = nothing
    b :: Union{Nothing, Vector{Float64}} = nothing
end

struct TypedMOP{
    O, NLEC,NLIC,
    MTO, MTNLEC, MTNLIC,
    LB, UB, EC, AB
} <: SimpleMOP
    objectives :: O
    nl_eq_constraints :: NLEC
    nl_ineq_constraints :: NLIC

    num_vars :: Int
    
    dim_objectives :: Int
    dim_nl_eq_constraints :: Int
    dim_nl_ineq_constraints :: Int

    mtype_objectives :: MTO
    mtype_nl_eq_constraints :: MTNLEC
    mtype_nl_ineq_constraints :: MTNLIC 

    lb :: LB
    ub :: UB

    Ec :: EC
    Ab :: AB
end

function initialize(mop::MutableMOP, ξ0::RVec)
    Ec = lin_eq_constraints(mop)
    Ab = lin_ineq_constraints(mop)
    return TypedMOP(
        mop.objectives,
        mop.nl_eq_constraints,
        mop.nl_ineq_constraints,
        mop.num_vars,
        mop.dim_objectives,
        mop.dim_nl_eq_constraints,
        mop.dim_nl_ineq_constraints,
        mop.mtype_objectives,
        mop.mtype_nl_eq_constraints,
        mop.mtype_nl_ineq_constraints,
        mop.lb,
        mop.ub,
        Ec,
        Ab
   )
end

MutableMOP(num_vars::Int)=MutableMOP(;num_vars)

parse_mtype(::Nothing)=parse_mtype(:exact)
parse_mtype(mtype::Type{<:AbstractSurrogateModel})=mtype
parse_mtype(mtype_symb::Symbol)=parse_mtype(Val(mtype_symb))
parse_mtype(::Val{:exact})=ExactModel
parse_mtype(::Val{:taylor1})=TaylorPolynomial1
parse_mtype(::Val{:taylor2})=TaylorPolynomial2

function add_function!(
    func_field::Symbol, mop::MutableMOP, op::AbstractNonlinearOperator, 
    model::Union{Type{<:AbstractSurrogateModel}, Nothing, Symbol}=nothing;
    dim_out::Int, backend=NoBackend()
)

    setfield!(mop, func_field, op)
    setfield!(mop, Symbol("dim_", func_field), dim_out)

    mod = parse_mtype(model)
    setfield!(mop, Symbol("mtype_", func_field), mod)
    return nothing
end
function add_function!(
    func_field::Symbol, mop::MutableMOP, func::Function,
    model::Union{Type{<:AbstractSurrogateModel}, Nothing, Symbol}=nothing; 
    dim_out::Int, backend=NoBackend(), func_iip::Bool=false
)   
    op = NonlinearFunction(; func, func_iip, backend)
    return add_function!(func_field, mop, op, model; dim_out)
end

function add_function!(
    func_field::Symbol, mop::MutableMOP, func::Function, grads::Function,
    model::Union{Type{<:AbstractSurrogateModel}, Nothing, Symbol}=nothing;
    dim_out::Int, backend=NoBackend(), func_iip=false, grads_iip=false
)
    op = NonlinearFunction(; func, grads, func_iip, grads_iip, backend)
    return add_function!(func_field, mop, op, model; dim_out) 
end

function add_function!(
    func_field::Symbol, mop::MutableMOP, func::Function, grads::Function,
    func_and_grads::Function,
    model::Union{Type{<:AbstractSurrogateModel}, Nothing, Symbol}=nothing;
    dim_out::Int, backend=NoBackend(), func_iip=false, grads_iip=false, func_and_grads_iip=false
)
    op = NonlinearFunction(; 
        func, grads, func_and_grads, func_iip, grads_iip, func_and_grads_iip, backend)
    return add_function!(func_field, mop, op, model; dim_out) 
end

for fntype in (:objectives, :nl_eq_constraints, :nl_ineq_constraints)
    add_fn = Symbol("add_", fntype, "!")
    @eval function $(add_fn)(args...; kwargs...)
        return add_function!($(Meta.quot(fntype)), args...; kwargs...)
    end
end

precision(::SimpleMOP) = Float64
model_type(::SimpleMOP) = SimpleMOPSurrogate

for dim_func in (:dim_objectives, :dim_nl_eq_constraints, :dim_nl_ineq_constraints)
    @eval function $(dim_func)(mop::SimpleMOP)
        return getfield(mop, $(Meta.quot(dim_func)))
    end
end

lower_var_bounds(mop::SimpleMOP)=mop.lb
upper_var_bounds(mop::SimpleMOP)=mop.ub

function lin_eq_constraints(mop::MutableMOP)
    isnothing(mop.E) && return nothing
    if isnothing(mop.c)
        mop.c = zeros(Float64, size(E, 1))
    end
    return mop.E, mop.C
end

function lin_ineq_constraints(mop::MutableMOP)
    isnothing(mop.A) && return nothing
    if isnothing(mop.b)
        mop.b = zeros(Float64, size(A, 1))
    end
    return mop.A, mop.b
end

lin_eq_constraints(mop::TypedMOP) = mop.Ec
lin_ineq_constraints(mop::TypedMOP) = mop.Ab

eval_objectives!(y::RVec, mop::SimpleMOP, x::RVec)=eval_op!(y, mop.objectives, x)
eval_nl_eq_constraints!(y::RVec, mop::SimpleMOP, x::RVec)=eval_op!(y, mop.nl_eq_constraints, x)
eval_nl_ineq_constraints!(y::RVec, mop::SimpleMOP, x::RVec)=eval_op!(y, mop.nl_ineq_constraints, x)

struct SimpleMOPSurrogate{O, NLEC, NLIC, MO, MNLEC, MNLIC} <: AbstractMOPSurrogate
    objectives :: O
    nl_eq_constraints :: NLEC
    nl_ineq_constraints :: NLIC
    
    num_vars :: Int
    dim_objectives :: Int
    dim_nl_eq_constraints :: Int
    dim_nl_ineq_constraints :: Int

    mod_objectives :: MO
    mod_nl_eq_constraints :: MNLEC
    mod_nl_ineq_constraints :: MNLIC 
end

precision(mod::SimpleMOPSurrogate)=Float64

for dim_func in (:dim_objectives, :dim_nl_eq_constraints, :dim_nl_ineq_constraints)
    @eval function $(dim_func)(mod::SimpleMOPSurrogate)
        return getfield(mod, $(Meta.quot(dim_func)))
    end
end

function depends_on_trust_region(mod::SimpleMOPSurrogate)
    return CE.depends_on_trust_region(mod.mod_objectives) ||
        CE.depends_on_trust_region(mod.mod_nl_eq_constraints) ||
        CE.depends_on_trust_region(mod.mod_nl_ineq_constraints)
end

supports_scaling(::Type{<:SimpleMOPSurrogate})=ConstantAffineScaling()

function eval_objectives!(y::RVec, mod::SimpleMOPSurrogate, x::RVec)
    return model_op!(y, mod.mod_objectives, x)
end
function eval_nl_eq_constraints!(y::RVec, mod::SimpleMOPSurrogate, x::RVec)
    return model_op!(y, mod.mod_nl_eq_constraints, x)
end
function eval_nl_ineq_constraints!(y::RVec, mod::SimpleMOPSurrogate, x::RVec)
    return model_op!(y, mod.mod_nl_ineq_constraints, x)
end
function grads_objectives!(Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return model_grads!(Dy, mod.mod_objectives, x)
end
function grads_nl_eq_constraints!(Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return model_grads!(Dy, mod.mod_nl_eq_constraints, x)
end
function grads_nl_ineq_constraints!(Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return model_grads!(Dy, mod.mod_nl_ineq_constraints, x)
end

function eval_and_grads_objectives!(y::RVec, Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return model_op_and_grads!(y, Dy, mod.mod_objectives, x)
end
function eval_and_grads_nl_eq_constraints!(y::RVec, Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return model_op_and_grads!(y, Dy, mod.mod_nl_eq_constraints, x)
end
function eval_and_grads_nl_ineq_constraints!(y::RVec, Dy::RMat, mod::SimpleMOPSurrogate, x::RVec)
    return model_op_and_grads!(y, Dy, mod.mod_nl_ineq_constraints, x)
end

scale_wrap_op(scaler::IdentityScaler, op, mtype, dim_in, dim_out, T)=op
function scale_wrap_op(scaler, op, mtype, dim_in, dim_out, T)
    ξ = zeros(T, dim_in)
    Dy = if requires_grads(mtype)
        zeros(T, dim_in, dim_out)
    else
        nothing
    end
    Hy = if requires_hessians(mtyp)
        zeros(T, dim_in, dim_in, dim_out)
    end
    return ScaledOperatorOrModel(op, scaler, ξ, Dy, Hy)
end

function init_models(mop::SimpleMOP, n_vars, scaler)
    d_objf = dim_objectives(mop)
    d_nl_eq = dim_nl_eq_constraints(mop)
    d_nl_ineq = dim_nl_ineq_constraints(mop)
    d_in = mop.num_vars
    objf = scale_wrap_op(
        scaler, mop.objectives, mop.mtype_objectives, d_in, d_objf, Float64)
    nl_eq = scale_wrap_op(
        scaler, mop.nl_eq_constraints, mop.mtype_nl_eq_constraints, d_in, d_nl_eq, Float64)
    nl_ineq = scale_wrap_op(
        scaler, mop.nl_ineq_constraints, mop.mtype_nl_ineq_constraints, d_in, d_nl_ineq, Float64)
    mobjf = init_surrogate(
        mop.mtype_objectives, objf, d_in, d_objf, nothing, Float64)
    mnl_eq = init_surrogate(
        mop.mtype_nl_eq_constraints, nl_eq, d_in, d_nl_eq, nothing, Float64)
    mnl_ineq = init_surrogate(
        mop.mtype_nl_ineq_constraints, nl_ineq, d_in, d_nl_ineq, nothing, Float64)

    return SimpleMOPSurrogate(
        objf, nl_eq, nl_ineq, d_in, d_objf, d_nl_eq, d_nl_ineq, mobjf, mnl_eq, mnl_ineq)
end

function update_models!(mod::SimpleMOPSurrogate, mop, scaler, vals)
    @unpack x, fx, gx, hx = vals

    update!(mod.mod_objectives, mod.objectives, x, fx)
    update!(mod.mod_nl_eq_constraints, mod.nl_eq_constraints, x, hx)
    update!(mod.mod_nl_ineq_constraints, mod.nl_ineq_constraints, x, gx)
    return nothing
end