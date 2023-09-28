# # Basic Proof-Of-Concept Implementations of Multi-Objective Problems

# ## Scaled Evaluation
# The optimizer might internally apply a scaling to the variables.
# The user-provided multi-objective problem does not know about this scaling
# and we have to consider it when evaluating and differntiating the problem
# functions.
#
# Currently we restrict ourselves to affine-linear transformations.
# Given `scaler::AbstractAffineScaler`,
# consider the unscaled variable vector ``ξ``.
# The scaled variable is ``x = Tξ + t``, where ``T`` is the `scaling_matrix(scaler)`
# and ``t`` is the `scaling_offset(scaler)`.
# Unscaling happens via ``ξ = Sx + s``, where ``S=T^{-1}`` is 
# `unscaling_matrix(scaler)` and ``s=-T^{-1}t`` is 
# `unscaling_offset(scaler)`.
#
# A `ScaledOperator` wraps some other `AbstractNonlinearOperator` and enables caching
# of (un-)scaling results as well taking care of the chain rule for derivatives:

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

# Consider ``f: ℝ^n → ℝ^m`` and the unscaling map ``u: ℝ^n → ℝ^n``.
# By the chain rule we have ``∇(f∘u)(x) = ∇f(ξ)*T``, where ``T`` is 
# the Jacobian of ``u``.
# As we are actually working with transposed Jacobians `Dy`, we compute
# `T'Dy`:
function scale_grads!(Dy, sop)
    LA.mul!(Dy, transpose(unscaling_matrix(sop.scaler)), sop.Dy)
    return nothing
end

# Usually, the chain rule for second-order derivatives is more complicated:
# ```math
#   ∇²(f∘u)(x) = Tᵀ∇²f(ξ)T + ∑ₖ ∇fₖ(ξ) ∇²u
# ```
# But ``u`` is affine, so the second term vanishes and we are left with 
# the matrix-matrix-matrix product:
function scale_hessians!(H, sop)
    T = unscaling_matrix(sop.scaler)
    for i = axes(H, 3)
        LA.mul!(sop.Hy[:, :, i], transpose(T), H[:, :, i])
        LA.mul!(H[:, :, i], sop.Hy[:, :, i], T)
    end
    return nothing
end

# ### Operator Interface

# Evaluation of a `ScaledOperator` is straight-forward:
function CE.eval_op!(y, sop::ScaledOperator, x)
    unscale!(sop.ξ, sop.scaler, x)
    return eval_op!(y, sop.op, sop.ξ)
end

# With the above chain-rule functions, it is now easy to implement the 
# complete operator interface for `ScaledOperator`.
# Taking gradients requires unscaling of the variables, differntiation at 
# the unscaled site and a re-scaling of the gradients:
function CE.eval_grads!(Dy, sop::ScaledOperator, x)
    unscale!(sop.ξ, sop.scaler, x)
    eval_grads!(sop.Dy, sop.op, sop.ξ)
    scale_grads!(Dy, sop)
    return nothing
end
# We don't have to unscale twice when evaluating and differntiating:
function CE.eval_op_and_grads!(y, Dy, sop::ScaledOperator, x)
    unscale!(sop.ξ, sop.scaler, x)
    eval_op_and_grads!(y, sop.Dy, sop.op, sop.ξ)
    scale_grads!(Dy, so)
    return nothing
end
# The procedure is similar for Hessians:
function CE.eval_hessians!(H, sop::ScaledOperator, x)
    unscale!(sop.ξ, sop.scaler, x)
    eval_hessians!(H, sop.op, sop.ξ)
    scale_hessians!(H, sop)
    return nothing
end
function CE.eval_op_and_grads_and_hessians!(y, Dy, H, sop, x)
    unscale!(sop.ξ, sop.scaler, x)
    eval_op_and_grads_and_hessians!(y, sop.Dy, H, sop.op, sop.ξ)
    scale_grads!(Dy, sop)
    scale_hessians!(H, sop)
    return nothing
end

# ## `SimpleMOP`
# In this section, we will define two kinds of simple problem structures:
abstract type SimpleMOP <: AbstractMOP end

# A `MutableMOP` is meant to be initialized empty and built up iteratively.
"""
    MutableMOP(; num_vars, kwargs...)

Initialize a multi-objective problem with `num_vars` variables.

# Functions
There can be exactly one (possibly vector-valued) objective function,
one nonlinear equality constraint function, and one nonlinear inequality constraint function.
For now, they have to be of type `NonlinearFunction`.
You could provide these functions with the keyword-arguments `objectives`,
`nl_eq_constraints` or `nl_ineq_constraints` or set the fields of the same name.
To conveniently add user-provided functions, there are helper functions, 
like `add_objectives!`.

## LinearConstraints
Box constraints are defined by the vectors `lb` and `ub`.
Linear equality constraints ``Ex=c`` are defined by the matrix `E` and the vector `c`.
Inequality constraints read ``Ax≤b`` and use `A` and `b`.

# Surrogate Configuration
Use the keyword-arguments `mcfg_objectives` to provide an `AbstractSurrogateModelConfig`
to define how the objectives should be modelled.
By default, we assume `ExactModelConfig()`, which requires differentiable objectives.
"""
@with_kw mutable struct MutableMOP <: SimpleMOP
    objectives :: Union{Nothing, NonlinearFunction} = nothing
    nl_eq_constraints :: Union{Nothing, NonlinearFunction} = nothing
    nl_ineq_constraints :: Union{Nothing, NonlinearFunction} = nothing

    num_vars :: Int

    dim_objectives :: Int = 0
    dim_nl_eq_constraints :: Int = 0
    dim_nl_ineq_constraints :: Int = 0

    mcfg_objectives :: AbstractSurrogateModelConfig = ExactModelConfig()
    mcfg_nl_eq_constraints :: AbstractSurrogateModelConfig = ExactModelConfig()
    mcfg_nl_ineq_constraints :: AbstractSurrogateModelConfig = ExactModelConfig()

    lb :: Union{Nothing, Vector{Float64}} = nothing
    ub :: Union{Nothing, Vector{Float64}} = nothing

    E :: Union{Nothing, Matrix{Float64}} = nothing
    c :: Union{Nothing, Vector{Float64}} = nothing
    A :: Union{Nothing, Matrix{Float64}} = nothing
    b :: Union{Nothing, Vector{Float64}} = nothing
end
MutableMOP(num_vars::Int)=MutableMOP(;num_vars)

# The `TypedMOP` looks nearly the same, but is strongly typed and immutable.
# We initialize a `TypedMOP` from a `MutableMOP` before optimization for 
# performance reasons:
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

    mcfg_objectives :: MTO
    mcfg_nl_eq_constraints :: MTNLEC
    mcfg_nl_ineq_constraints :: MTNLIC 

    lb :: LB
    ub :: UB

    E_c :: EC
    A_b :: AB
end

# This initialization really is just a forwarding of all fields:
function initialize(mop::MutableMOP, ξ0::RVec)
    E_c = lin_eq_constraints(mop)
    A_b = lin_ineq_constraints(mop)
    return TypedMOP(
        mop.objectives,
        mop.nl_eq_constraints,
        mop.nl_ineq_constraints,
        mop.num_vars,
        mop.dim_objectives,
        mop.dim_nl_eq_constraints,
        mop.dim_nl_ineq_constraints,
        mop.mcfg_objectives,
        mop.mcfg_nl_eq_constraints,
        mop.mcfg_nl_ineq_constraints,
        mop.lb,
        mop.ub,
        E_c,
        A_b
   )
end

# ### Iterative Problem Setup

# First, let's define some helpers to parse model configuration specifications:
# If a configuration is given already, do nothing:
parse_mcfg(mcfg::AbstractSurrogateModelConfig)=mcfg
# We can also allow Symbols:
parse_mcfg(mcfg_symb::Symbol)=parse_mcfg(Val(mcfg_symb))
parse_mcfg(::Val{:exact})=ExactModelConfig()
parse_mcfg(::Val{:rbf})=RBFConfig()
parse_mcfg(::Val{:taylor1})=TaylorPolynomialConfig(;degree=1)
parse_mcfg(::Val{:taylor2})=TaylorPolynomialConfig(;degree=2)
# The default value `nothing` redirects to an `ExactModelConfig`:
parse_mcfg(::Nothing)=parse_mcfg(:exact)

# Add function is the backend for the helper functions `add_objectives!` etc.
"""
    add_function!(func_field, mop, op, model; dim_out, backend=NoBackend())

Add the operator `op` to `mop` at `func_field` and use model configuration `model`.
Keyword argument `dim_out::Int` is mandatory.
E.g., `add_function!(:objectives, mop, op, :rbf; dim_out=2)` adds `op`
as the bi-valued objective to `mop`.
"""
function add_function!(
    func_field::Symbol, mop::MutableMOP, op::AbstractNonlinearOperator, 
    model::Union{AbstractSurrogateModelConfig, Nothing, Symbol}=nothing;
    dim_out::Int, backend=NoBackend()
)

    setfield!(mop, func_field, op)
    setfield!(mop, Symbol("dim_", func_field), dim_out)

    mod = parse_mcfg(model)
    setfield!(mop, Symbol("mcfg_", func_field), mod)
    return nothing
end

# Allow for adding basic `Function`s instead of operators:
function add_function!(
    func_field::Symbol, mop::MutableMOP, func::Function,
    model::Union{AbstractSurrogateModelConfig, Nothing, Symbol}=nothing; 
    dim_out::Int, backend=NoBackend(), func_iip::Bool=false
)   
    op = NonlinearFunction(; func, func_iip, backend)
    return add_function!(func_field, mop, op, model; dim_out)
end

# Define methods to allow hand-crafted gradient functions:
function add_function!(
    func_field::Symbol, mop::MutableMOP, func::Function, grads::Function,
    model::Union{AbstractSurrogateModelConfig, Nothing, Symbol}=nothing;
    dim_out::Int, backend=NoBackend(), func_iip=false, grads_iip=false
)
    op = NonlinearFunction(; func, grads, func_iip, grads_iip, backend)
    return add_function!(func_field, mop, op, model; dim_out) 
end
function add_function!(
    func_field::Symbol, mop::MutableMOP, func::Function, grads::Function,
    func_and_grads::Function,
    model::Union{AbstractSurrogateModelConfig, Nothing, Symbol}=nothing;
    dim_out::Int, backend=NoBackend(), func_iip=false, grads_iip=false, func_and_grads_iip=false
)
    op = NonlinearFunction(; 
        func, grads, func_and_grads, func_iip, grads_iip, func_and_grads_iip, backend)
    return add_function!(func_field, mop, op, model; dim_out) 
end

#src #TODO allow for passing Hessians...
# We now generate the derived helper functions `add_objectives!`,
# `add_nl_ineq_constraints!` and `add_nl_eq_constraints!`:
for fntype in (:objectives, :nl_eq_constraints, :nl_ineq_constraints)
    add_fn = Symbol("add_", fntype, "!")
    @eval function $(add_fn)(args...; kwargs...)
        return add_function!($(Meta.quot(fntype)), args...; kwargs...)
    end
end

# ### Interface Implementation
# Define the most basic Getters:
precision(::SimpleMOP) = Float64
model_type(::SimpleMOP) = SimpleMOPSurrogate

# Box constraints are easy:
lower_var_bounds(mop::SimpleMOP)=mop.lb
upper_var_bounds(mop::SimpleMOP)=mop.ub

# To access functions, retrieve the corresponding fields of the MOP:
for dim_func in (:dim_objectives, :dim_nl_eq_constraints, :dim_nl_ineq_constraints)
    @eval function $(dim_func)(mop::SimpleMOP)
        return getfield(mop, $(Meta.quot(dim_func)))
    end
end

# Linear constraints are returned only if the matrix is not nothing.
# If the offset vector is nothing, we return zeros:
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

# The `TypedMOP` stores tuples (of a matrix and a vector instead):
lin_eq_constraints(mop::TypedMOP) = mop.E_c
lin_ineq_constraints(mop::TypedMOP) = mop.A_b

# Because we store `AbstractNonlinearOperator`s, evaluation can simply be redirected:
function eval_objectives!(y::RVec, mop::SimpleMOP, x::RVec)
    return eval_op!(y, mop.objectives, x)
end
function eval_nl_eq_constraints!(y::RVec, mop::SimpleMOP, x::RVec)
    return eval_op!(y, mop.nl_eq_constraints, x)
end
function eval_nl_ineq_constraints!(y::RVec, mop::SimpleMOP, x::RVec)
    return eval_op!(y, mop.nl_ineq_constraints, x)
end

# ## Surrogate Modelling of `SimpleMOP`

# For `SimpleMOP`, the surrogates are stored in the type-stable structure
# `SimpleMOPSurrogate`, with exactly one surrogate for each kind of problem 
# function.
# We additionally store the operators, because we might wrap them as
# `ScaledOperator` and don't want to modify the original problem.
# (Also, I am not sure if the scaler is available when the problem is initialized).
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

# The precision is again fixed:
precision(mod::SimpleMOPSurrogate)=Float64

# Getters are generated automatically:
for dim_func in (:dim_objectives, :dim_nl_eq_constraints, :dim_nl_ineq_constraints)
    @eval function $(dim_func)(mod::SimpleMOPSurrogate)
        return getfield(mod, $(Meta.quot(dim_func)))
    end
end

# The model container is dependent on the trust-region if any of the models is:
function depends_on_radius(mod::SimpleMOPSurrogate)
    return CE.depends_on_radius(mod.mod_objectives) ||
        CE.depends_on_radius(mod.mod_nl_eq_constraints) ||
        CE.depends_on_radius(mod.mod_nl_ineq_constraints)
end

# At the moment, I have not yet thought about what we would need to change for dynamic scaling:
supports_scaling(::Type{<:SimpleMOPSurrogate})=ConstantAffineScaling()

# Evaluation redirects to evaluation of the surrogates:
function eval_objectives!(y::RVec, mod::SimpleMOPSurrogate, x::RVec)
    return model_op!(y, mod.mod_objectives, x)
end
function eval_nl_eq_constraints!(y::RVec, mod::SimpleMOPSurrogate, x::RVec)
    return model_op!(y, mod.mod_nl_eq_constraints, x)
end
function eval_nl_ineq_constraints!(y::RVec, mod::SimpleMOPSurrogate, x::RVec)
    return model_op!(y, mod.mod_nl_ineq_constraints, x)
end
# Gradients:
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
# ### Initialization and Training
# These helpers make an operator respect scaling by returning `ScaledOperator`:
scale_wrap_op(scaler::IdentityScaler, op, mcfg, dim_in, dim_out, T)=op
function scale_wrap_op(scaler, op, mcfg, dim_in, dim_out, T)
    ξ = zeros(T, dim_in)
    Dy = if requires_grads(mcfg)
        zeros(T, dim_in, dim_out)
    else
        nothing
    end
    Hy = if requires_hessians(mcfg)
        zeros(T, dim_in, dim_in, dim_out)
    else
        nothing
    end
    return ScaledOperator(op, scaler, ξ, Dy, Hy)
end

# We use the `ScaledOperator` in the `SimpleMOPSurrogate` to make writing models
# as easy as possible. 
# Modellers should always assume working in the scaled domain and not be bothered
# with transformations...
function init_models(mop::SimpleMOP, n_vars, scaler)
    d_objf = dim_objectives(mop)
    d_nl_eq = dim_nl_eq_constraints(mop)
    d_nl_ineq = dim_nl_ineq_constraints(mop)
    d_in = mop.num_vars
    objf = scale_wrap_op(
        scaler, mop.objectives, mop.mcfg_objectives, d_in, d_objf, Float64)
    nl_eq = scale_wrap_op(
        scaler, mop.nl_eq_constraints, mop.mcfg_nl_eq_constraints, d_in, d_nl_eq, Float64)
    nl_ineq = scale_wrap_op(
        scaler, mop.nl_ineq_constraints, mop.mcfg_nl_ineq_constraints, d_in, d_nl_ineq, Float64)
    mobjf = init_surrogate(
        mop.mcfg_objectives, objf, d_in, d_objf, nothing, Float64)
    mnl_eq = init_surrogate(
        mop.mcfg_nl_eq_constraints, nl_eq, d_in, d_nl_eq, nothing, Float64)
    mnl_ineq = init_surrogate(
        mop.mcfg_nl_ineq_constraints, nl_ineq, d_in, d_nl_ineq, nothing, Float64)

    return SimpleMOPSurrogate(
        objf, nl_eq, nl_ineq, d_in, d_objf, d_nl_eq, d_nl_ineq, mobjf, mnl_eq, mnl_ineq)
end

# The sub-models are trained separately:
function update_models!(
    mod::SimpleMOPSurrogate, Δ, mop, scaler, vals, scaled_cons, algo_opts;
)
    @unpack x, fx, gx, hx = vals
    @unpack lb, ub = scaled_cons
    Δ_max = algo_opts.delta_max
    update!(mod.mod_objectives, mod.objectives, Δ, x, fx, lb, ub; Δ_max)
    update!(mod.mod_nl_eq_constraints, mod.nl_eq_constraints, Δ, x, hx, lb, ub; Δ_max)
    update!(mod.mod_nl_ineq_constraints, mod.nl_ineq_constraints, Δ, x, gx, lb, ub; Δ_max)
    return nothing
end

function copy_model(mod::SimpleMOPSurrogate)
    return SimpleMOPSurrogate(
        mod.objectives,
        mod.nl_eq_constraints,
        mod.nl_ineq_constraints,
        mod.num_vars,
        mod.dim_objectives,
        mod.dim_nl_eq_constraints,
        mod.dim_nl_ineq_constraints,
        CE._copy_model(mod.mod_objectives),
        CE._copy_model(mod.mod_nl_eq_constraints),
        CE._copy_model(mod.mod_nl_ineq_constraints),
    )
end

function copyto_model!(mod_trgt::SimpleMOPSurrogate, mod_src::SimpleMOPSurrogate)
    CE._copyto_model!(mod_trgt.mod_objectives, mod_src.mod_objectives)
    CE._copyto_model!(mod_trgt.mod_nl_eq_constraints, mod_src.mod_nl_eq_constraints)
    CE._copyto_model!(mod_trgt.mod_nl_ineq_constraints, mod_src.mod_nl_ineq_constraints)
    return mod_trgt
end