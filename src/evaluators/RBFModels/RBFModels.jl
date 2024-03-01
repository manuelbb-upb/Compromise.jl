module RBFModels

using ..Compromise.CompromiseEvaluators
const CE = CompromiseEvaluators
import ..Compromise: @ignoraise, DEFAULT_PRECISION, project_into_box!
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

@views function prepare_eval_buffers(rbf, x)
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

function CE.model_op!(y::AbstractVector, rbf::RBFModel, x::AbstractVector)
    Φ, Π, x_in, centers, cφ, cπ, ε, n_X = prepare_eval_buffers(rbf, x)

    _rbf_eval!(
        vec2col(y), Φ, Π, x_in, centers, cφ, cπ, ε;
        DIM_x = rbf.dim_x,
        DIM_y = rbf.dim_y,
        KERNEL = rbf.kernel,
        POLY_DEG = rbf.poly_deg,
        DIM_π = rbf.dim_π,
        DIM_φ = n_X
    )
    return nothing
end

function CE.model_grads!(Dy, rbf::RBFModel, x)
    _, _, x_in, centers, cφ, cπ, ε, _ = prepare_eval_buffers(rbf, x)
    Δx = @view rbf.buffers.v1[1:rbf.dim_x]
    _rbf_diff!(
        Dy, Δx, x_in, centers, cφ, cπ, ε;
        KERNEL = rbf.kernel,
        POLY_DEG = rbf.poly_deg,
        DIM_π = rbf.dim_π,
    )
    return nothing
end

function CE.init_surrogate(
    cfg::RBFConfig, op, dim_in, dim_out, params, T;
    require_fully_linear::Bool=true,
    delta_max::Number=T(Inf),
)
    @unpack (
        kernel, search_factor, max_search_factor, th_qr, th_cholesky, max_points, 
        database_size, database_chunk_size, enforce_fully_linear, poly_deg, 
        shape_parameter_function, sampling_factor, max_sampling_factor,
    ) = cfg
    if require_fully_linear
        enforce_fully_linear = require_fully_linear
    end
    return rbf_init_model(
        dim_in, dim_out, poly_deg, delta_max, kernel, shape_parameter_function, 
        database_size, database_chunk_size, max_points, enforce_fully_linear, 
        search_factor, max_search_factor, sampling_factor, max_sampling_factor, 
        th_qr, th_cholesky, T
    )
end

function CE.update!(
    rbf::RBFModel, op, Δ, x, fx, lb, ub; 
    log_level=Info, kwargs...
)
    update_rbf_model!(rbf, op, Δ, x, fx, lb, ub; log_level, norm_p=Inf)
end

function CE.copy_model(rbf::RBFModel)
    @unpack dim_x, dim_y, dim_π, min_points, max_points, delta_max, poly_deg, kernel = rbf
    @unpack shape_parameter_function, enforce_fully_linear, search_factor = rbf
    @unpack max_search_factor, sampling_factor, max_sampling_factor, th_qr, th_cholesky = rbf
    return RBFModel(;
        dim_x, dim_y, dim_π, min_points, max_points, delta_max, poly_deg, kernel,
        shape_parameter_function, enforce_fully_linear, search_factor, max_search_factor,
        th_qr, th_cholesky,
        database = rbf.database,        # pass by reference, this is okay, as we have decoupled it from evaluation
        params = deepcopy(rbf.params),  # these must!! be copied
        buffers = rbf.buffers,          # these can be passed by reference, don't really influence training
        sampling_factor, max_sampling_factor
    )
end

function CE.copyto_model!(mod_trgt::RBFModel, mod_src::RBFModel)
    copyto!(mod_trgt.params, mod_src.params)
end

# TODO partial evaluation

export RBFModel, RBFConfig
export CubicKernel, GaussianKernel, InverseMultiQuadricKernel

end#module