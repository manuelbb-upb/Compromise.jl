module RBFModels

using ..Compromise.CompromiseEvaluators
const CE = CompromiseEvaluators
import ..Compromise: @serve, DEFAULT_PRECISION, project_into_box!
import ..Compromise: subscript, supscript, pretty_row_vec
import Printf: @sprintf

using ElasticArrays
using ElasticArrays: resize! # explicit import to avoid false linter hints
import LinearAlgebra as LA
using Parameters: @with_kw, @unpack
using StructHelpers: @batteries

import Logging: @logmsg, Info

const NumOrVec = Union{Number, AbstractVector}
const VecOrMat = Union{AbstractVector, AbstractMatrix}
const MatOrNothing = Union{AbstractMatrix, Nothing}

## `MutableNumber` instead of `Base.RefValue` because of 
## `(Ref(1) == Ref(1)) == false`
mutable struct MutableNumber{T}
  val :: T
end
@batteries MutableNumber

val(m::MutableNumber) = m.val
val!(m::MutableNumber, v) = (m.val = v)

include("cached_qr.jl") # must be first becaues of QRWYWs

include("utils.jl")
include("backend.jl")
include("database.jl")
include("frontend_types.jl")
include("qr_sampling.jl")
include("update.jl")
include("cholesky_sampling.jl")

CE.depends_on_radius(::RBFModel)=true
CE.requires_grads(::RBFConfig)=false
CE.requires_hessians(::RBFConfig)=false

function prepare_eval_buffers(rbf, x)
    @unpack dim_x, dim_y, dim_π, poly_deg, kernel, params, buffers = rbf
    @unpack coeff_φ, coeff_π, x0, X = params
    n_X = val(params.n_X_ref)
    ε = val(params.shape_parameter_ref)

    cφ = coeff_φ[1:n_X, 1:dim_y]
    cπ = coeff_π[1:dim_π, 1:dim_y]

    Φ = buffers.Φ[1, 1:n_X]
    Π = buffers.Π[1, 1:dim_π]

    x_in = buffers.xZ
    x_in .= x .- x0

    centers = X[1:dim_x, 1:n_X]

    return vec2row(Φ), vec2row(Π), vec2col(x_in), centers, cφ, cπ, ε, n_X
end

@views function CE.model_op!(y::AbstractVector, rbf::RBFModel, x::AbstractVector)
    Φ, Π, x_in, centers, cφ, cπ, ε, n_X = prepare_eval_buffers(rbf, x)

    return _rbf_eval!(
        vec2col(y), Φ, Π, x_in, centers, cφ, cπ, ε;
        DIM_x = rbf.dim_x,
        DIM_y = rbf.dim_y,
        KERNEL = rbf.kernel,
        POLY_DEG = rbf.poly_deg,
        DIM_π = rbf.dim_π,
        DIM_φ = n_X
    )
end

function CE.model_grads!(Dy, rbf::RBFModel, x)
    _, _, x_in, centers, cφ, cπ, ε, _ = prepare_eval_buffers(rbf, x)
    Δx = @view rbf.buffers.v1[1:rbf.dim_x]
    return _rbf_diff!(
        Dy, Δx, x_in, centers, cφ, cπ, ε;
        KERNEL = rbf.kernel,
        POLY_DEG = rbf.poly_deg,
        DIM_π = rbf.dim_π,
    )
end

function CE.init_surrogate(cfg::RBFConfig, op, dim_in, dim_out, params, T)
    @unpack (
        kernel, search_factor, max_search_factor, th_qr, th_cholesky, max_points, 
        database_size, database_chunk_size, enforce_fully_linear, poly_deg, 
        shape_parameter_function
    ) = cfg
    return rbf_init_model(
        dim_in, dim_out, poly_deg, kernel, shape_parameter_function, 
        database_size, database_chunk_size, max_points, enforce_fully_linear, 
        search_factor, max_search_factor, th_qr, th_cholesky, T
    )
end

#=
function CE.copy_model(mod::RBFModel)
    return error("TODO")
end

function CE.copyto_model!(mod_trgt::RBFModel, mod_src::RBFModel)
    return error("TODO")
end

function _is_fully_linear(rbf::RBFModel)
    return rbf.n_X1 == rbf.surrogate.dim_x
end
=#

#=

# ## Interface Implementation

function CE.init_surrogate(cfg::RBFConfig, op, dim_in, dim_out, params, T)
    @unpack (kernel, search_factor, max_search_factor, th_qr, th_cholesky, max_points, database_size, 
        database_chunk_size, enforce_fully_linear, poly_deg, shape_parameter_function) = cfg
    return init_rbf_model(
        dim_in, dim_out, poly_deg, kernel, shape_parameter_function, 
        database_size, database_chunk_size, max_points, 
        enforce_fully_linear, search_factor, max_search_factor,
        th_qr, th_cholesky, T
    )
end


function CE.update!(
    rbf::RBFModel, op, Δ, x, fx, lb, ub; 
    Δ_max=Δ, log_level, kwargs...
)
    update_rbf_model!(rbf, op, Δ, x, fx, lb, ub; Δ_max, log_level, norm_p=Inf)
end

#src function model_op_and_grads! end # TODO
# TODO partial evaluation
=#
export RBFModel, RBFConfig
export CubicKernel, GaussianKernel, InverseMultiQuadricKernel

end#module