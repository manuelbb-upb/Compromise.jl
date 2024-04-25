"""
    RBFConfig(; <keyword arguments>)

# Keyword Arguments
* `kernel::AbstractRBFKernel=CubicKernel()`
* `poly_deg::Union{Int, Nothing}=1`
* `shape_parameter_function::Any=nothing`: Either `nothing` to use the shape parameter set 
  in kernel, or a real number to use the shape parameter, or a function mapping the trust 
  region radius to a real number.
* `max_points::Union{Int, Nothing}=nothing`
* `database_size::Union{Int, Nothing}=nothing`
* `database_chunk_size::Union{Int, Nothing}=nothing`
* `enforce_fully_linear::Bool=true`: (Dis-)allow interpolation models that are not fully linear by 
   sampling looking in a larger box.
* `search_factor::Real=2`: Enlargement factor for trust region to look for affinely 
  independent points in.
* `max_search_factor::Real=2`: Enlargement factor for maximum trust region to look for
  affinely independent points in.
* `th_qr::Real=1/(2*search_factor)`: Pivoting threshold to determine a poised interpolation 
  set.
* `th_cholesky::Real=1e-7`: Threshold for accepting additional points based on the Cholesky 
  factors.
"""
@with_kw struct RBFConfig{
  T<:AbstractFloat,
  kernelType <: AbstractRBFKernel,
  databaseType <: Union{Nothing, RBFDatabase},
  shape_parameter_functionType,
  database_rwlockType <: Union{Nothing, AbstractReadWriteLock},
} <: AbstractSurrogateModelConfig
    kernel :: kernelType = CubicKernel()
    poly_deg :: Union{Int, Nothing}=1
    shape_parameter_function :: shape_parameter_functionType = nothing
    max_points :: Union{Int, Nothing} = nothing

    database :: databaseType = nothing
    database_rwlock :: database_rwlockType =nothing
    database_size :: Union{Int, Nothing} = nothing
    database_chunk_size :: Union{Int, Nothing} = nothing

    "(Dis-)allow interpolation models that are not fully linear by sampling looking in a larger box."
    enforce_fully_linear :: Bool = true

    "Enlargement factor for trust region to look for affinely independent points in."
    search_factor :: T = 2.0
    sampling_factor :: T = 1.0
    
    "Enlargement factor for maximum trust region to look for affinely independent points in."
    max_search_factor :: T = search_factor
    max_sampling_factor :: T = sampling_factor

    "Pivoting threshold to determine a poised interpolation set."
    th_qr :: T = 1/(2*search_factor)

    "Threshold for accepting additional points based on the Cholesky factors."
    th_cholesky :: T = 1e-7

    ## TODO `max_evals` (soft limit on maximum number of evaluations)
    @assert isnothing(poly_deg) || poly_deg in (0,1)
end

@batteries RBFConfig selfconstructor=false

Base.@kwdef struct RBFParameters{T<:Real}
    ## meta data for `Base.show`
    dim_x :: Int
    dim_y :: Int
    dim_π :: Int
    min_points :: Int
    max_points :: Int

    n_X_ref :: MutableNumber{Int}

    "`dim_x` × `max_points` matrix of interpolation sites."
    X :: Matrix{T}

    "`max_points` × `dim_y` RBF coefficient matrix."
    coeff_φ :: Matrix{T}
    "`dim_π` × `dim_y` polynomial coefficient matrix."
    coeff_π :: Matrix{T}

    is_fully_linear_ref :: MutableNumber{Bool}
    has_z_new_ref :: MutableNumber{Bool}

    "`dim_X` improvement direction if a model is not fully linear and we want to do few evals."
    z_new :: Vector{T}

    "`dim_x` vector of current trust region center."
    x0 :: Vector{T}
    xtrial :: Vector{T}

    "Trust region radius."
    delta_ref :: Union{MutableNumber{T}, Vector{T}}
    "Reference to the current shape parameter, which might depend on Δ."
    shape_parameter_ref :: MutableNumber{T}

    database_state_ref :: MutableNumber{UInt64}
end

@batteries RBFParameters selfconstructor=false

function param_top_str(params)
  repr_str = "( ℝ$(supscript(params.dim_x)) → ℝ$(supscript(params.dim_y))"
  repr_str *= ", dim(Π)=$(params.dim_π), dim(Φ)=$(val(params.n_X_ref))∈[$(params.min_points), $(params.max_points)] )"
  return repr_str
end
function param_str(params; iscompact=false)
  repr_str = param_top_str(params)
  if !iscompact
    repr_str *= "\n  `x0`    : $(pretty_row_vec(params.x0))"
    repr_str *= "\n  `(Δ, ε)`: $(@sprintf("(%.2e, %.2e)", val(params.delta_ref), val(params.shape_parameter_ref)))"
    repr_str *= "\n  fully linear: $(val(params.is_fully_linear_ref))"
    repr_str *= "\n  has `z_new` : $(val(params.has_z_new_ref))"
    repr_str *= "\n  SIZE = $(Base.format_bytes(Base.summarysize(params)))"
  end
  return repr_str
end
function Base.show(io::IO, params::RBFParameters{T}) where T
    iscompact = get(io, :compact, false)
    repr_str = "RBFParamaters{$T}"
    repr_str *= param_str(params; iscompact)
    print(io, repr_str)
end

function Base.copyto!(dst :: RBFParameters, src :: RBFParameters)
  for fn in (
      :n_X_ref, :is_fully_linear_ref, :has_z_new_ref, :delta_ref, 
      :shape_parameter_ref, :database_state_ref
  )
      val!(
          getfield(dst, fn),
          val(
              getfield(src, fn)
          )
      )
  end
  for fn in (:X, :coeff_φ, :coeff_π, :z_new, :x0)
      copyto!(getfield(dst, fn), getfield(src, fn))
  end

end

Base.@kwdef struct RBFTrainingBuffers{T<:Real}
    ## meta data for `Base.show`
    dim_x :: Int
    dim_y :: Int
    dim_π :: Int
    min_points :: Int
    max_points :: Int

    x0_db_index_ref :: MutableNumber{Int}

    "FastLapackInterface Workspace for repeated QR decomposition of a `dim_x` × `dim_x` matrix."
    qr_ws_dim_x :: Union{Nothing, QRWYWs{T, Matrix{T}}}
    
    "`dim_x` cache for trust region bounds."
    lb :: Vector{T}
    "`dim_x` cache for trust region bounds."
    ub :: Vector{T}

    "`dim_y` × `max_points` matrix of interpolation values."
    FX :: Matrix{T}

    "`dim_x` vector, temporary buffer, e.g. for testing points."
    xZ :: Vector{T}
    "`dim_y` vector, temporary buffer."
    fxZ :: Vector{T}

    "`dim_x + 1` vector to mark database points."
    db_index :: Vector{Int}
    sorting_flags :: Vector{Bool}  # flag vector for batch evaluation

    filter_flags :: Vector{Bool}

    "`max_points` × `max_points` buffer for RBF basis matrix."
    Φ :: Matrix{T}
    #src "`dim_x + 1` × `dim_π` buffer for (transpose) Polynomial basis matrix."
    #src Π :: Matrix{T}
    
    "Workspace for repeated QR decomposition."
    qr_ws_min_points :: Union{Nothing, QRWYWs{T, Matrix{T}}}
    "`max_points` × `max_points` buffer for full Q factor."
    Q :: Matrix{T}
    "`dim_π` × `dim_π` buffer for truncated R factor."
    R :: Matrix{T}
    
    "`max_points` × `dim_π + 1` buffer for new Q factor update."
    Qj :: Matrix{T}
    "`dim_π+1` × `dim_π` buffer for new R factor update."
    Rj :: Matrix{T}
    "`max_points - dim_π` × `max_points` buffer for matrix-matrix-product."
    NΦ :: Matrix{T}
    "`max_points - dim_π` × `max_points - dim_π` buffer for coefficient LES."
    NΦN :: Matrix{T}

    "`max_points - dim_π` × `max_points - dim_π` buffer for lower cholesky factor."
    L :: Matrix{T}
    "`max_points - dim_π` × `max_points - dim_π` buffer for inverse cholesky factor."
    Linv :: Matrix{T}

    "`max_points` buffer vector to augment cholesky factors."
    v1 :: Vector{T}
    "`max_points - dim_π` buffer vector to augment cholesky factors."
    v2 :: Vector{T}
end

function Base.copyto!(dst::RBFTrainingBuffers, src::RBFTrainingBuffers)
  for fn in (
    :lb, :ub, :FX, :xZ, :fxZ, :db_index, :sorting_flags, :Φ, :Q, :R, :Qj, :Rj, #src :Π,
    :NΦ, :NΦN, :L, :Linv, :v1, :v2
  )
    copyto!(getfield(dst, fn), getfield(src, fn))
  end
  if length(dst.filter_flags) < length(src.filter_flags)
    resize!(dst.filter_flags, length(src.filter_flags))
  end
  copyto!(dst.filter_flags, src.filter_flags)

  val!(dst.x0_db_index_ref, val(src.x0_db_index_ref))
  if !isnothing(dst.qr_ws_dim_x) && !isnothing(src.qr_ws_dim_x)
    copyto!(dst.qr_ws_dim_x, src.qr_ws_dim_x)
  end
  if !isnothing(dst.qr_ws_min_points) && !isnothing(src.qr_ws_min_points)
    copyto!(dst.qr_ws_min_points, src.qr_ws_min_points)
  end
end

function Base.show(io::IO, buffers::RBFTrainingBuffers{T}) where T
    iscompact = get(io, :compact, false)
    repr_str = "RBFTrainingBuffers{$T}("
    repr_str *= "ℝ$(supscript(buffers.dim_x)) → ℝ$(supscript(buffers.dim_y))"
    repr_str *= ", dim(Π)=$(buffers.dim_π), dim(Φ)∈[$(buffers.min_points), $(buffers.max_points)])"
  
    if !iscompact
      repr_str *= "\n  SIZE = $(Base.format_bytes(Base.summarysize(buffers)))"
    end
    print(io, repr_str)
end

Base.@kwdef struct RBFModel{
    T<:Number,
    F<:Union{Nothing, Number, Function}, 
    K<:AbstractRBFKernel,
    D<:RBFDatabase
} <: AbstractSurrogateModel
    ## „static” fields: define the surrogate type and how to evaluate/differentiate/train etc.
    
    ##meta data for `Base.show`
    dim_x :: Int
    dim_y :: Int
    dim_π :: Int
    min_points :: Int
    max_points :: Int
    delta_max :: T

    poly_deg :: Int
    kernel :: K

    shape_parameter_function :: F
    
    "Flag to allow surrogates that are not fully linear."
    enforce_fully_linear :: Bool

    "Enlargement factor for trust region that is queried for samples 
    to make the model fully linear."
    search_factor :: T
    "Enlargement factor for trust region that is queried for additional samples."
    max_search_factor :: T

    sampling_factor :: T
    max_sampling_factor :: T

    "Training parameter: Threshold for sample acceptance."
    th_qr :: T
    "Training parameter: Threshold for sample acceptance to capture nonlinear behavior."
    th_cholesky :: T
    
    ## „non-static” fields
    database :: D

    params :: RBFParameters{T}
   
    buffers :: RBFTrainingBuffers{T}
end

function Base.show(io::IO, rbf::RBFModel{T}) where T
    iscompact = get(io, :compact, false)
    repr_str = "RBFModel{$T}"
    repr_str *= param_top_str(rbf.params)
    if !iscompact
      repr_str *= "\n  kernel         : $(repr(rbf.kernel))"
      repr_str *= "\n  shape parameter: "
      if rbf.shape_parameter_function isa Function
        repr_str *= "dynamic, currently `$(val(rbf.params.shape_parameter_ref))`"
      elseif rbf.shape_parameter_function isa Number
        repr_str *= "constant, set to `$(rbf.shape_parameter_function)`"
      else
        repr_str *= "constant by kernel as `$(_shape_parameter(rbf.kernel))`"
      end
      repr_str *= "\n  SIZE = $(Base.format_bytes(Base.summarysize(rbf)))"
    end
    print(io, repr_str)
end

function array(T::Type, size...)
  return Array{T}(undef, size...)
end

function rbf_params_and_buffers(
  dim_x, dim_y, dim_π, max_points, T=DEFAULT_FLOAT_TYPE
)
  
  min_points = dim_x + 1
  max_points = max(min_points, max_points)
  max_dim_N = max_points - dim_π
  min_dim_N = min_points - dim_π

  n_X_ref = MutableNumber(0)
  
  coeff_φ = array(T, max_points, dim_y)
  coeff_π = array(T, dim_π, dim_y)
  z_new = array(T, dim_x)
  x0 = array(T, dim_x)
  xtrial = similar(x0)
  delta_ref = MutableNumber{T}(1)
  shape_parameter_ref = MutableNumber{T}(1)
  is_fully_linear_ref = MutableNumber(false)
  has_z_new_ref = MutableNumber(false)
  database_state_ref = MutableNumber(rand(UInt64))

  X = array(T, dim_x, max_points)
  params = RBFParameters(;
    X,
    dim_x, dim_y, dim_π, max_points, min_points,
    n_X_ref, coeff_φ, coeff_π, z_new, x0, xtrial, delta_ref, shape_parameter_ref,
    is_fully_linear_ref, has_z_new_ref, database_state_ref
  )

  FX = array(T, dim_y, max_points)

  qr_ws_dim_x = if T isa BlasFloat 
    QRWYWs(@view(X[:, 1:dim_x]))
  else
    nothing
  end
  qr_ws_min_points = if T isa BlasFloat
    QRWYWs(@view(X[:, 1:min_points]))
  else
    nothing
  end
  
  lb = array(T, dim_x)
  ub = array(T, dim_x)

  xZ = array(T, dim_x)
  fxZ = array(T, dim_y)
  db_index = fill(-1, min_points)
  sorting_flags = ones(Bool, min_points) 

  Φ = array(T, max_points, max_points)
  #src Π = array(T, min_points, dim_π)

  Q = array(T, max_points, max_points)
  R = array(T, dim_π, dim_π)

  Qj = array(T, max_points, dim_π + 1)
  Rj = array(T, dim_π + 1, dim_π)

  NΦ = array(T, min_dim_N, min_points)
  NΦN = array(T, min_dim_N, min_dim_N)
  
  L = array(T, max_dim_N, max_dim_N)
  Linv = array(T, 
    max(dim_π, max_dim_N),
    max(max_dim_N, dim_y)
  )

  v1 = array(T, max_points)
  v2 = array(T, max_dim_N)

  x0_db_index_ref = MutableNumber(-1)

  filter_flags = zeros(Bool, 0)

  buffers = RBFTrainingBuffers(;
    dim_x, dim_y, dim_π, max_points, min_points,
    qr_ws_dim_x, lb, ub, FX, xZ, fxZ, db_index, sorting_flags, Φ, qr_ws_min_points,
    Q, R, Qj, Rj, NΦ, NΦN, L, Linv, v1, v2, x0_db_index_ref, filter_flags
  )

  return params, buffers
end

function rbf_init_model(
    dim_x :: Integer, dim_y :: Integer, 
    poly_deg :: Union{Nothing, Integer}, 
    delta_max :: Number,
    kernel :: AbstractRBFKernel,
    shape_parameter_function :: Union{Nothing, Number, Function},
    database :: Union{Nothing, RBFDatabase},
    database_rwlock :: Union{Nothing, AbstractReadWriteLock}, 
    database_size :: Union{Nothing, Integer}, 
    database_chunk_size :: Union{Nothing, Integer},
    max_points :: Union{Nothing, Integer}, 
    enforce_fully_linear :: Bool, 
    search_factor :: Real, max_search_factor :: Real,
    sampling_factor :: Real, max_sampling_factor :: Real,
    th_qr :: Real, th_cholesky :: Real,
    T :: Type{<:Number} = DEFAULT_FLOAT_TYPE;
)
    @assert th_qr >=  0
    @assert th_cholesky >= 0
    @assert search_factor >= 1
    @assert max_search_factor >= 1
    min_points = dim_x + 1
    if isnothing(max_points)
        max_points = 2 * min_points
    end

    if max_points < min_points
        @warn "`max_points` must be at least $(min_points)."
        max_points = min_points
    end
    surrogate = RBFSurrogate(; dim_x, dim_y, kernel, poly_deg, dim_φ=-1)

    if isnothing(database) || database.dim_x != dim_x || database.dim_y != dim_y
      database = init_rbf_database(
        dim_x, dim_y, database_size, database_chunk_size, T, database_rwlock)
    end

    @unpack poly_deg, dim_π = surrogate
    params, buffers = rbf_params_and_buffers(dim_x, dim_y, dim_π, max_points, T)

    return RBFModel(;
      dim_x,
      dim_y,
      dim_π,
      min_points,
      max_points,
      delta_max = T(delta_max),
      poly_deg,
      kernel,
      shape_parameter_function, 
      enforce_fully_linear,
      search_factor = T(search_factor),
      max_search_factor = T(max_search_factor),
      sampling_factor = T(sampling_factor),
      max_sampling_factor = T(max_sampling_factor),
      th_qr = T(th_qr),
      th_cholesky = T(th_cholesky), 
      database,
      params,
      buffers
    )
end