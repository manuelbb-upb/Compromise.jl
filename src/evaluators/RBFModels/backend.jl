
abstract type AbstractRBFKernel end
Base.broadcastable(φ::AbstractRBFKernel) = Ref(φ)

function _apply_kernel(φ::AbstractRBFKernel, r, ε)
    error("`_apply_kernel` not implemented for kernel of type $(typeof(φ)).")
end
function _apply_kernel_derivative(φ::AbstractRBFKernel, r, ε)
    error("`_apply_kernel_derivative` not implemented for kernel of type $(typeof(φ)).")
end
function _shape_parameter(φ::AbstractRBFKernel)
    error("`_shape_parameter` not implemented for kernel of type $(typeof(φ)).")
end

function apply_kernel(φ::AbstractRBFKernel, r, _ε::Nothing)
    ε = _shape_parameter(φ)
    return _apply_kernel(φ, r, ε)
end
apply_kernel(φ::AbstractRBFKernel, r, ε)=_apply_kernel(φ, r, ε)

function apply_kernel_derivative(φ::AbstractRBFKernel, r, _ε::Nothing)
    ε = _shape_parameter(φ)
    return _apply_kernel_derivative(φ, r, ε)
end
apply_kernel_derivative(φ::AbstractRBFKernel, r, ε)=_apply_kernel_derivative(φ, r, ε)

# Concerning the shape parameter ``ε``, we follow the definitions in 
# [Wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function)
# and use it in such a way that a larger shape parameter implies a pointier kernel.

"""
    CubicKernel(; eps=1)

Kernel for ``φ(r) = (εr)^3``."""
Base.@kwdef struct CubicKernel{F} <: AbstractRBFKernel
    eps :: F = 1
end
_apply_kernel(kernel::CubicKernel, r, ε) = (r*ε)^3
_apply_kernel_derivative(kernel::CubicKernel, r, ε) = (3 * ε^3)*r^2
_shape_parameter(kernel::CubicKernel)=kernel.eps

"""
    GaussianKernel(; eps=1)

Kernel for ``φ_ε(r) = \\exp(-(εr)^2)``.
"""
Base.@kwdef struct GaussianKernel{F} <: AbstractRBFKernel
    eps :: F = 1
end
_apply_kernel(kernel::GaussianKernel, r, ε)=exp( -(ε*r)^2 )
_apply_kernel_derivative(kernel::GaussianKernel, r, ε) = let εsq = ε^2;
    -2 * εsq * r * exp(-εsq*r^2) 
end
_shape_parameter(kernel::GaussianKernel)=kernel.eps

"""
    InverseMultiQuadricKernel(; eps=1)

Kernel for ``φ_ε(r) = 1 / \\sqrt{1 + (εr)^2}``.
"""
Base.@kwdef struct InverseMultiQuadricKernel{F} <: AbstractRBFKernel
    eps :: F = 1
end
_apply_kernel(kernel::InverseMultiQuadricKernel, r, ε)=1/sqrt(1+(ε*r)^2)
_apply_kernel_derivative(kernel::InverseMultiQuadricKernel, r, ε) = let esq=ε^2;
    -esq*r/(esq*r^2 + 1)^(3//2) 
end
_shape_parameter(kernel::InverseMultiQuadricKernel)=kernel.eps

function Base.show(io::IO, kernel::K) where K<:AbstractRBFKernel
    tname = Base.typename(K).name
    print(io, "$(tname)( ε = $(_shape_parameter(kernel)) )")
end

# A type used for evaluation of an RBF surrogate 
# ```
# \mathbf{x} ↦ \sum_{j=1}^{D} v_j π_j(\mathbf{x}) + 
#   \sum_{i=1}^{N_{\text{centers}}} w_i φ_ε( ‖\mathbf{x} - \mathbf{ξ}_i‖ ),
# ```
# where ``D`` is the dimension of the multi-variate polynomial space with basis ``π_j``.

_poly_dim(dim_x, ::Nothing) = 0
_poly_dim(dim_x, poly_deg) = binomial(dim_x + poly_deg, dim_x)
@with_kw struct RBFSurrogate{K<:AbstractRBFKernel}
    dim_x :: Int
    dim_y :: Int

    kernel :: K = GaussianKernel()

    poly_deg :: Union{Nothing, Int} = 1
    dim_φ :: Int
    dim_π :: Int = _poly_dim(dim_x, poly_deg)

    @assert isnothing(poly_deg) || (poly_deg) in (0, 1)
    @assert dim_π == _poly_dim(dim_x, poly_deg)
    @assert dim_x >= 1
    @assert dim_y >= 1
end

function _rbf_params(rbf::RBFSurrogate, T = DEFAULT_PRECISION; kwargs...)
    @unpack dim_x, dim_y, dim_φ, dim_π, kernel = rbf
    return _rbf_params(T, dim_x, dim_y, dim_φ, dim_π, kernel)
end

function _rbf_params(
    T, dim_x, dim_y, dim_φ, dim_π, kernel;
    centers=nothing,
    ε=nothing
)
    if isnothing(centers)
        centers = rand(T, dim_x, dim_φ)
    end
    @assert size(centers) == (dim_x, dim_φ)

    if isnothing(ε)
        ε = T(_shape_parameter(kernel))
    end

    return (
        centers = centers,
        coeff_φ = rand(T, dim_φ, dim_y),
        coeff_π = rand(T, dim_π, dim_y),
        ε = ε 
    )
end

function _rbf_eval_caches(rbf::RBFSurrogate, n_x, n_y, T = DEFAULT_PRECISION)
    @unpack dim_x, dim_y, dim_φ, dim_π = rbf
    return _rbf_eval_caches(T, n_x, n_y, dim_x, dim_y, dim_φ, dim_π, )
end

function _rbf_eval_caches(T, n_x, n_y, dim_x, dim_y, dim_φ, dim_π)
    #LHS = zeros(T, n_x, dim_φ + dim_π)
    Φ = zeros(T, n_x, dim_φ)
    Π = zeros(T, n_x, dim_π)
    return (
        Φ = Φ,
        Π = Π 
    )
end

function _rbf_diff_caches(rbf::RBFSurrogate, T)
    return (
        Δx = zeros(T, rbf.dim_x),
    )
end

vec2col(::Nothing)=nothing
vec2col(x::AbstractMatrix) = x
vec2row(x::AbstractMatrix) = x
vec2col(x::AbstractVector) = reshape(x, :, 1)
vec2row(x::AbstractVector) = transpose(x)
num2vec(x::AbstractVector)=x
num2vec(x::Number)=[x,]

function _rbf_eval(rbf::RBFSurrogate, x, params)
    x = vec2col(x)
    dim_x, n_x = size(x)
    T = Base.promote_eltype(DEFAULT_PRECISION, eltype(x))
    y = zeros(T, rbf.dim_y, n_x)
    _rbf_eval!(y, rbf, x, params)
    return y
end

"""
    _rbf_eval!(y, rbf::RBFSurrogate, x, params::NamedTuple)

Evaluate `rbf` at `x` and store result in `y`.
Input `x` can be a vector or a matrix with several inputs stored column-wise.
Output `y` can be a vector or a matrix with several output slots column-wise.
"""
function _rbf_eval!(
    y,
    rbf::RBFSurrogate,
    x,
    params::NamedTuple{
        (:centers, :coeff_φ, :coeff_π, :ε),
        <:NTuple{4,Any}
    };
    kwargs...
)   
    @unpack centers, coeff_φ, coeff_π, ε = params
    return _rbf_eval!(
        y, rbf, x, centers, coeff_φ, coeff_π, ε; kwargs...
    )
end

function _rbf_eval!(
    y, rbf::RBFSurrogate, x, centers, coeff_φ, coeff_π, ε; kwargs...
)
    return _rbf_eval!(
        vec2col(y),
        rbf,
        vec2col(x),
        vec2col(centers),
        vec2col(coeff_φ),
        vec2col(coeff_π), 
        ε;
        kwargs...
    )
end

function _rbf_eval!(
    y::AbstractMatrix, rbf::RBFSurrogate,
    x::AbstractMatrix, centers::AbstractMatrix, coeff_φ::AbstractMatrix, 
    coeff_π::AbstractMatrix, ε::Number;
    kwargs...
)  
    n_x = size(x, 2)
    n_y = size(y, 2)
    T = eltype(y)
    caches = _rbf_eval_caches(rbf, n_x, n_y, T)
    @unpack Π, Φ = caches
    return _rbf_eval!(
        y, Φ, Π, x, centers, coeff_φ, coeff_π, ε;
        DIM_x = rbf.dim_x,
        DIM_y = rbf.dim_y,
        POLY_DEG = rbf.poly_deg,
        DIM_φ = rbf.dim_φ,
        DIM_π = rbf.dim_π,
        KERNEL = rbf.kernel,
        kwargs...
    )
end

function _rbf_eval!(
    y::AbstractMatrix, Φ::AbstractMatrix, Π::AbstractMatrix,
    x::AbstractMatrix, centers::AbstractMatrix, coeff_φ::AbstractMatrix, 
    coeff_π::AbstractMatrix, ε::Number;
    check_sizes::Bool=true,
    check_dim_fields::Bool=false,
    DIM_x :: Int = size(x, 1),
    DIM_y :: Int = size(y, 1),
    KERNEL :: AbstractRBFKernel = GaussianKernel(),
    POLY_DEG :: Union{Nothing, Int} = 1,
    DIM_φ :: Int = size(centers, 2),
    DIM_π :: Int = _poly_dim(DIM_x, POLY_DEG)
)   
    y .= 0
    _rbf_eval_add_poly!(
        y, Π, POLY_DEG, coeff_π, x; check_sizes, check_dim_fields, DIM_π, DIM_x
    )
    _rbf_eval_add_kernel!(
        y, Φ, KERNEL, coeff_φ, centers, x, ε; 
        check_sizes, check_dim_fields, DIM_x, DIM_y, DIM_φ
    )
    return y
end

function _rbf_diff(rbf::RBFSurrogate, x::NumOrVec, params)
    x = num2vec(x)
    dim_x = length(x)
    @assert dim_x == rbf.dim_x
    T = Base.promote_eltype(DEFAULT_PRECISION, eltype(x))
    Dy = zeros(T, dim_x, rbf.dim_y)
    _rbf_diff!(Dy, rbf, x, params)
    return Dy
end

function _rbf_diff!(
    Dy, 
    rbf::RBFSurrogate,
    x::NumOrVec,
    params::NamedTuple{
        (:centers, :coeff_φ, :coeff_π, :ε),
        <:NTuple{4,Any}
    }
)
    x = num2vec(x)
    @unpack centers, coeff_φ, coeff_π, ε = params
    return _rbf_diff!(Dy, rbf, x, centers, coeff_φ, coeff_π, ε)
end

function _rbf_diff!(
    Dy, rbf::RBFSurrogate, x::AbstractVector, centers, coeff_φ, coeff_π, ε
)
    Dy = vec2col(Dy)
    coeff_φ = vec2col(coeff_φ)
    coeff_π = vec2col(coeff_π)
    centers = vec2col(centers)
    x = vec2col(x)    

    caches = _rbf_diff_caches(rbf, eltype(Dy))
    @unpack Δx = caches
    return _rbf_diff!(
        Dy, Δx, x, centers, coeff_φ, coeff_π, ε;
        KERNEL = rbf.kernel,
        DIM_π = rbf.dim_π,
        POLY_DEG = rbf.poly_deg
    )
end

function _rbf_diff!(
    Dy::AbstractMatrix, Δx::AbstractVector, x, centers, coeff_φ, coeff_π, ε;
    DIM_π = size(coeff_π, 1),
    KERNEL = GaussianKernel(),
    POLY_DEG = 1
)
    Dy .= 0
     _rbf_diff_add_poly!(Dy, POLY_DEG, coeff_π; DIM_π)
    _rbf_diff_add_kernel!(Dy, Δx, KERNEL, coeff_φ, centers, x, ε)
end

function _rbf_eval_add_poly!(
    y::AbstractMatrix, Π::AbstractMatrix, 
    poly_deg::Union{Nothing, Integer}, coeff::MatOrNothing, x::AbstractMatrix;
    check_sizes::Bool=true,
    check_dim_fields::Bool=false, 
    DIM_π = size(Π, 2),
    DIM_x = size(x, 2),
    kwargs...
)
    if isnothing(poly_deg)
        return nothing
    end

    dim_x, n_x = size(x)
    dim_π, _dim_y = size(coeff)
    if check_sizes
        dim_y, n_y = size(y)
        @assert dim_y == _dim_y "Output matrix has $(dim_y) rows, but polynomial coefficients give $(_dim_y) outpus."
        @assert n_x == n_y "There are $(n_x) input samples, but $(n_y) outputs."
        if check_dim_fields
            @assert dim_π == DIM_π
            @assert dim_x == DIM_x
        end

        _n_x, _dim_π = size(Π)
        @assert _n_x == n_x
        @assert _dim_π == dim_π 
    end

    _rbf_poly_mat!(Π, poly_deg, x; check_sizes)
    ## `Π` is `n_y` × `dim_π`, `coeff` is `dim_π` × `dim_y`
    ## product has dimensions `n_y` × `dim_y`, so we have to transpose target:
    LA.mul!(transpose(y), Π, coeff, 1, 1)
    return nothing
end

"""
    _rbf_poly_mat!(Π, rbf::RBFSurrogate, _features)

Fill columns of matrix `Π` with values of polynomial basis function evaluated at all
columns in `_features`."""
function _rbf_poly_mat!(
    Π::AbstractMatrix, deg, features::AbstractMatrix;
    check_sizes::Bool=true,
    check_dim_fields::Bool=false

)
    if check_sizes
        _n_x, dim_π = size(Π)
        dim_x, n_x = size(features)
        @assert _n_x == n_x
    end
    return _rbf_poly_mat!(Val(deg), Π, features)
end
function _rbf_poly_mat!(::Val{D}, Π, features) where D
    error("Unsupported polynomial degree $D.")
end
function _rbf_poly_mat!(::Val{0}, Π, features)
    fill!(Π, 1)
    return nothing
end
function _rbf_poly_mat!(::Val{1}, Π, features)
    Π[:, 1:end-1] .= features'
    Π[:, end] .= 1
    return nothing
end

function _rbf_eval_add_kernel!(
    y::AbstractMatrix, Φ::AbstractMatrix, kernel::AbstractRBFKernel, 
    coeff::AbstractMatrix, centers::AbstractMatrix, x::AbstractMatrix, ε::Number=1;
    check_sizes::Bool=true,
    check_dim_fields::Bool=false,
    DIM_φ = size(coeff, 1),
    DIM_x = size(x, 1),
    DIM_y = size(y, 1),
    kwargs...
)

    if check_sizes
        dim_y, n_y = size(y)
        dim_φ, _dim_y = size(coeff)
        _dim_x, _dim_φ = size(centers)
        @assert dim_φ == _dim_φ
        @assert dim_y == _dim_y "Output matrix has $(dim_y) rows, but RBF coefficients give $(_dim_y) outpus."
        dim_x, n_x = size(x)
        @assert n_x == n_y "There are $(n_x) input samples, but $(n_y) outputs."
        @assert dim_x == _dim_x "Inputs have dimension $(dim_x) and centers have $(_dim_x)."
        
        _n_x, __dim_φ = size(Φ)
        @assert n_x == _n_x
        @assert dim_φ == __dim_φ
        if check_dim_fields
            @assert dim_φ == DIM_φ
            @assert dim_y == DIM_y
            @assert dim_x == DIM_x
        end
    end

    @assert ε > 0 "Shape parameter must be positive."

    _rbf_kernel_mat!(Φ, kernel, x, centers, ε; check_sizes, check_dim_fields)
    LA.mul!(transpose(y), Φ, coeff, 1, 1)
    return nothing
end

function dist(x1, x2)
    d = 0
    for i=eachindex(x1)
        d += (x1[i] - x2[i])^2
    end
    return sqrt(d)
end

function _rbf_kernel_mat!(
    Φ::AbstractMatrix, kernel, features::AbstractMatrix, centers::AbstractMatrix, ε::Number; 
    check_sizes::Bool=true,
    check_dim_fields::Bool=false,
    centers_eq_features::Bool=false
)    
    if check_sizes
        dim_x, n_x = size(features)
        _dim_x, _dim_φ = size(centers)
        _n_x, dim_φ = size(Φ)
        @assert dim_x == _dim_x
        @assert n_x == _n_x
        @assert dim_φ == _dim_φ       
    end
    @assert ε > 0

    φ(r) = apply_kernel(kernel, r, ε)
    if !centers_eq_features
        for (j, x_center) = enumerate(eachcol(centers))
            for (i, x_feature) = enumerate(eachcol(features))
                Φ[i, j] = φ(dist(x_feature, x_center))
            end
            #=
            map!( 
                x_feature -> LA.norm( x_feature .- x_center ),
                @view(Φ[:, j]),
                eachcol(features)
            )
            =#
        end
    else
        φ0 = φ(0)
        for (j, x_center) = enumerate(eachcol(centers))
            for (i, x_feature) = enumerate(eachcol(features))
                i < j && continue
                if i==j
                    Φ[i, i] = φ0
                    continue
                end
                Φ[i, j] = φ(dist(x_feature, x_center))
            end
            #=map!( 
                x_feature -> φ(LA.norm( x_feature .- x_center )),
                @view(Φ[j+1:end, j]),
                #Iterators.drop(eachcol(features), j)
                eachcol(@view(features[:, j+1:end]))
            )=#
        end
        ## this is a symmetric **view**:
        Φsym = LA.Symmetric(Φ, :L)
        copyto!(Φ, Φsym)
    end
    #map!(r -> apply_kernel(kernel, r, ε), Φ, Φ)    # used many allocs
    return nothing
end

function _rbf_diff_add_poly!(Dy, poly_deg, coeff; DIM_π = size(coeff, 1))
    if isnothing(poly_deg) || poly_deg == 0
        ## no polynomial tail or constant, derivative of constant is zero
        return nothing
    end
    @assert poly_deg == 1
    ## `coeff` stores the coefficients of the constant term in the last column,
    ## we assume monomial basis `x[1], …, x[n], 1`.
    dim_π, _dim_y = size(coeff)
    @assert dim_π == DIM_π
    @assert size(Dy) == (dim_π - 1, _dim_y)    # `Dy` is transposed Jacobian

    ## `Dy[i,j] += coeff[i, j]` => output `j` differentiated with respect to 
    ## variable `i` is coefficient of monomial `x[i]`
    Dy .+= @view(coeff[1:end-1, :])
    return nothing
end

function _rbf_diff_add_kernel!(
    Dy, Δx::AbstractVector, kernel , coeff, centers, x, ε
)
    dim_φ, _dim_y = size(coeff)
    _dim_x, dim_y = size(Dy)
    dim_x, n_x = size(x) 
    __dim_x, _dim_φ = size(centers)
    @assert dim_φ == _dim_φ
    @assert _dim_y == dim_y
    @assert dim_x == _dim_x == __dim_x
    @assert length(Δx) == dim_x
   
    φ = kernel
    ## iterate centers
    for (j, ξ) = enumerate(eachcol(centers))
        ## assume kernel ``φ`` has derivative zero for `r=0`.
        ## the map ``x ↦ φ(‖x - ξ‖)`` has gradient ``∇φ = ( φ′(r) / r ) * (x - ξ)``,
        ## where ``r`` is the distance
        Δx .= x .- ξ
        r = LA.norm(Δx)
        if r == 0
            continue
        end
        dφ = apply_kernel_derivative(φ, r, ε) 
        ## there are `dim_y` outputs and we modify all columns of `Dy`:
        for l=axes(Dy, 2)
            @views Dy[:, l] .+= (coeff[j, l] * dφ  / r ).* Δx
        end
    end
    return nothing
end

function _rbf_fit!(params, rbf::RBFSurrogate, features, targets)
    dim_x, n_x = size(features)
    dim_y, n_y = size(targets)
    @assert rbf.dim_x == dim_x
    @assert n_x == n_y "Number of input and output training samples does not match."

    @unpack centers, coeff_φ, coeff_π, ε = params
    _dim_x, _dim_φ = size(centers)
    @assert _dim_x == rbf.dim_x
    dim_φ, _dim_y = size(coeff_φ)
    @assert dim_φ == _dim_φ
    @assert dim_y == _dim_y
    dim_π, _dim_y = size(coeff_φ)
    @assert dim_y == _dim_y
    @assert dim_π == rbf.dim_π
    @assert ε > 0

    n_coeff = dim_φ + rbf.dim_π
    T = eltype(coeff_φ)
    LHS = zeros(T, n_x, n_coeff)
    Φ = @view(LHS[:, 1:dim_φ])
    Π = @view(LHS[:, dim_φ+1:end])
    
    _rbf_kernel_mat!(Φ, rbf.kernel, features, centers, ε)
    _rbf_poly_mat!(Π, rbf.poly_deg, features)

    RHS = targets'
    _rbf_solve_normal_eqs!(coeff_φ, coeff_π, LHS, RHS) 
end

function _rbf_fit(
    rbf::RBFSurrogate, features, targets, centers=nothing, ε=nothing
)
    T = Base.promote_type(DEFAULT_PRECISION, eltype(features), eltype(targets))
    params = _rbf_params(rbf, T; centers, ε)
    @unpack coeff_φ, coeff_π, centers, ε = params

    _dim_x, _dim_φ = size(centers)
    @assert _dim_x == rbf.dim_x
    @assert ε > 0
    
    n_coeff = n_c + rbf.dim_π
    LHS = zeros(T, n_f, n_coeff)
    RHS = copy(targets')

    ## each column of Φ evaluates a RBF on all features, 
    ## `Φ[i,j]` is value of basis function `j` at features `i`
    Φ = @view(LHS[:, 1:n_c])
    _rbf_kernel_mat!(Φ, rbf.kernel, features, centers, ε)
    ## same for Π, columns store values of polynomial basis functions on all features
    Π = @view(LHS[:, n_c+1:end])
    _rbf_poly_mat!(Π, rbf.poly_deg, features)

    _rbf_solve_normal_eqs!(coeff_φ, coeff_π, LHS, RHS)

    return (
        centers = centers,
        coeff_φ = coeff_φ,
        coeff_π = coeff_π,
        ε = ε
    )
end

function _rbf_solve_normal_eqs!(
    coeff_φ, coeff_π, LHS, RHS;
)
    dim_φ = size(coeff_φ, 1)
    dim_π = size(coeff_π, 1)

    coeff = LHS \ RHS 

    coeff_φ .= @view(coeff[1:dim_φ, :])
    coeff_π .= @view(coeff[dim_φ+1:end, :])
    return nothing
end