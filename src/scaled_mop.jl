struct ScaledMOP{M, S, L, X} <: AbstractMOP
    mop :: M
    scaler :: S
    scaled_lin_cons :: L
    x0 :: X
    ξ :: X
end

scale_mop(mop::AbstractMOP, ::Val{:none})=mop

function scale_mop(mop::AbstractMOP, scaling_type::Val)
    lin_cons = init_lin_cons(mop)
    @unpack lb, ub = lin_cons
    scaler = init_scaler(scaling_type, lin_cons)
    new_cons = deepcopy(lin_cons)
    scale_lin_cons!(new_cons, scaler, lin_cons)

    F = float_type(x)
    x = initial_vars(mop)
    if isnothing(x)
        x0 = x
    else
        x0 = F.(x)
        scale!(x0, scaler, x)
    end
    ξ = array(F, dim_vars(mop))
        
    return ScaledMOP(mop, scaler, new_cons, x0, ξ)
end

function init_scaler(::Val{:box}, lin_cons)
    @unpack n_vars, lb, ub = lin_cons
    return init_box_scaler(lb, ub, n_vars)
end
init_scaler(::Union{Val{:force_none}, Val{:none}}, lin_cons) = IdentityScaler(lin_cons.n_vars)

function init_lin_cons(mop)
    lb = lower_var_bounds(mop)
    ub = upper_var_bounds(mop)

    if !var_bounds_valid(lb, ub)
        error("Variable bounds inconsistent.")
    end

    A = lin_ineq_constraints_matrix(mop)
    b = lin_ineq_constraints_vector(mop)
    E = lin_eq_constraints_matrix(mop)
    c = lin_eq_constraints_vector(mop)

    return LinearConstraints(dim_vars(mop), lb, ub, A, b, E, c)
end

function scale_lin_cons!(trgt, scaler, lin_cons)
    @unpack A, b, E, c, lb, ub = lin_cons
    
    scale!(trgt.lb, scaler, lb)
    scale!(trgt.ub, scaler, ub)
    scale_eq!(trgt.A, trgt.b, scaler, A, b)
    scale_eq!(trgt.E, trgt.c, scaler, E, c)
    return nothing
end

lower_var_bounds(smop::ScaledMOP)=smop.scaled_lin_cons.lb
upper_var_bounds(smop::ScaledMOP)=smop.scaled_lin_cons.ub
lin_eq_constraints_matrix(smop::ScaledMOP)=smop.scaled_lin_cons.E
lin_eq_constraints_vector(smop::ScaledMOP)=smop.scaled_lin_cons.c
lin_ineq_constraints_matrix(smop::ScaledMOP)=smop.scaled_lin_cons.A
lin_ineq_constraints_vector(smop::ScaledMOP)=smop.scaled_lin_cons.b
initial_vars(smop::ScaledMOP)=smop.x0
@forward ScaledMOP.mop float_type(smop::ScaledMOP)
@forward ScaledMOP.mop dim_vars(smop::ScaledMOP)
@forward ScaledMOP.mop dim_objectives(smop::ScaledMOP)
@forward ScaledMOP.mop dim_nl_eq_constraints(smop::ScaledMOP)
@forward ScaledMOP.mop dim_nl_ineq_constraints(smop::ScaledMOP)
@forward ScaledMOP.mop prealloc_objectives_vector(smop::ScaledMOP)
@forward ScaledMOP.mop prealloc_lin_eq_constraints_vector(smop::ScaledMOP)
@forward ScaledMOP.mop prealloc_lin_ineq_constraints_vector(smop::ScaledMOP)
@forward ScaledMOP.mop prealloc_nl_eq_constraints_vector(smop::ScaledMOP)
@forward ScaledMOP.mop prealloc_nl_ineq_constraints_vector(smop::ScaledMOP)

for eval_func in (
    :eval_objectives!,
    :eval_nl_eq_constraints!,
    :eval_nl_ineq_constraints!
)
    @eval function $(eval_func)(y::RVec, smop::ScaledMOP, x::RVec)
        @unpack mop, scaler, ξ = smop
        unscale!(ξ, scaler, x)
        return $(eval_func)(y, mop, ξ)
    end
end

struct ScaledMOPSurrogate{M, S, X} <: AbstractMOPSurrogate
    mod :: M
    scaler :: S
    ξ :: X
    Δ :: X
end

@forward ScaledMOPSurrogate.mod float_type(smod::ScaledMOPSurrogate)
@forward ScaledMOPSurrogate.mod dim_vars(smod::ScaledMOPSurrogate)
@forward ScaledMOPSurrogate.mod dim_objectives(smod::ScaledMOPSurrogate)
@forward ScaledMOPSurrogate.mod dim_nl_eq_constraints(smod::ScaledMOPSurrogate)
@forward ScaledMOPSurrogate.mod dim_nl_ineq_constraints(smod::ScaledMOPSurrogate)
@forward ScaledMOPSurrogate.mod depends_on_radius(smod::ScaledMOPSurrogate)

function init_models(
    smop::ScaledMOP;
    kwargs...
)
    @unpack mop, scaler, ξ = smop
    mod = init_models(smop; kwargs...)
    return ScaledMOPSurrogate(mod, scaler, similar(ξ), similar(ξ))
end

function unscale_radius!(Δ, scaler, _Δ)
end