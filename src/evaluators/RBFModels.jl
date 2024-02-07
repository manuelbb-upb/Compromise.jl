module RBFModels

using ..Compromise.CompromiseEvaluators
const CE = CompromiseEvaluators
import ..Compromise: @serve, DEFAULT_PRECISION, project_into_box!

using ElasticArrays
import LinearAlgebra as LA
using Parameters: @with_kw, @unpack

import Logging: @logmsg, Info

abstract type AbstractRBFKernel end

# Concerning the shape parameter ``ε``, we follow the definitions in 
# [Wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function)
# and use it in such a way that a larger shape parameter implies a pointier kernel.

"Kernel for ``φ(r) = r^3``."
struct CubicKernel <: AbstractRBFKernel end
apply_kernel(kernel::CubicKernel, r, ε) = r^3
apply_kernel_derivative(kernel::CubicKernel, r, ε) = 3*r^2
shape_parameter(::CubicKernel, Δ)=1

"""
Kernel for ``φ_ε(r) = \\exp(-(εr)^2)``.
If `eps` is a Function, it should take a single, real argument.
It is used to compute ``ε`` from the trust region radius ``Δₖ``.
"""
@with_kw struct GaussianKernel{F} <: AbstractRBFKernel
    eps :: F = 1
end

"""
Kernel for ``φ_ε(r) = 1 / \\sqrt{1 + (εr)^2}``.
If `eps` is a Function, it should take a single, real argument.
It is used to compute ``ε`` from the trust region radius ``Δₖ``.
"""
@with_kw struct InverseMultiQuadricKernel{F} <: AbstractRBFKernel
    eps :: F = 1
end

shape_paramater(φ::Union{InverseMultiQuadricKernel, GaussianKernel}, Δ)=φ.eps
function shape_paramater(
    φ::Union{GaussianKernel{F}, InverseMultiQuadricKernel{F}}, Δ
) where F<:Function
    return φ.eps(Δ)
end

apply_kernel(kernel::GaussianKernel, r, ε)=exp( -(ε*r)^2 )
apply_kernel(kernel::InverseMultiQuadricKernel, r, ε)=1/sqrt(1+(ε*r)^2)
apply_kernel_derivative(kernel::GaussianKernel, r, ε) = let εsq = ε^2; -2*εsq*r*exp(-esq*r^2) end
apply_kernel_derivative(kernel::InverseMultiQuadricKernel, r, ε) = let esq=ε^2; -esq*r/(esq*r^2 + 1)^(3//2) end

@with_kw mutable struct RBFConfig <: AbstractSurrogateModelConfig
    kernel :: AbstractRBFKernel = CubicKernel()
    max_points :: Union{Int, Nothing} = nothing
    database_size :: Union{Int, Nothing} = nothing
    database_chunk_size :: Union{Int, Nothing} = nothing

    "Allow interpolation models that are not fully linear by sampling looking in a larger box."
    allow_nonlinear :: Bool = false

    "Enlargement factor for trust region to look for affinely independent points in."
    search_factor :: Real = 2
    "Enlargement factor for maximum trust region to look for affinely independent points in."
    max_search_factor :: Real = 2

    "Pivoting threshold to determine a poised interpolation set."
    th_qr :: Real = 1/(2*search_factor)

    "Threshold for accepting additional points based on the Cholesky factors."
    th_cholesky :: Real = 1e-7

    ## TODO `max_evals` (soft limit on maximum number of evaluations)
end

struct RBFDatabase{T}
    dim_x :: Int
    dim_y :: Int

    ## `max_database_size` is a soft limit. in the criticality routine we don't want to delete
    ## any points belonging to a prior model, so lock the corresponding entries and safe them
    ## from deletion. We still allow up to `num_vars` additional entries...
    max_database_size :: Int
    chunk_size :: Int
    database_x :: ElasticArray{T, 2}
    database_y :: ElasticArray{T, 2}
    database_flags_x :: Vector{Bool}
    database_flags_y :: Vector{Bool}
    point_flags :: Vector{Bool}
    round1_flags :: Union{Nothing, Vector{Bool}}

    x_index :: Base.RefValue{Int}

    locked_flags :: Vector{Bool}
end

## helpers for `round1_flags`, which might be `nothing`
setflag!(flag_vec::Nothing, i, v)=nothing
function setflag!(flag_vec, i, v)
    flag_vec[i] = v
end

istrue(flag_vec::Nothing, i)=false
istrue(flag_vec, i)=flag_vec[i]

append_zeros!(flag_vec::Nothing, chunk_size)=nothing
function append_zeros!(flag_vec, chunk_size)
    append!(flag_vec, zeros(Bool, chunk_size))
end

struct RBFModel{K, T} <: AbstractSurrogateModel
    kernel :: K

    shape_param :: Base.RefValue{T}
    
    database :: RBFDatabase{T}
    point_indices :: Vector{Int}

    allow_nonlinear :: Bool
    max_points :: Int

    search_factor :: T
    max_search_factor :: T

    th_qr :: T
    th_cholesky :: T

    ## central interpolation point
    x0 :: Vector{T}
    s :: Vector{T}

    ## RBF basis matrix
    Φ :: Matrix{T}

    ## caches
    lb :: Vector{T}
    ub :: Vector{T}
    lb_max :: Vector{T}
    ub_max :: Vector{T}
    Y :: Matrix{T}
    dists :: Matrix{T}
    Z :: Matrix{T}
    Pr :: Matrix{T}     # orthogonal projection matrix
    Pr_xi :: Vector{T}  # projected vector `Pr*xi`

    LHS :: Matrix{T}
    RHS :: Matrix{T}
    coeff :: Matrix{T}
end

function CE.copy_model(mod::RBFModel)
    return RBFModel(
        mod.kernel, 
        mod.shape_param,
        mod.database,   # should be passed by reference, so that modifications to either model affect the other
        copy(mod.point_indices),
        mod.allow_nonlinear,
        mod.max_points,
        mod.search_factor,
        mod.max_search_factor,
        mod.th_qr,
        mod.th_cholesky,
        copy(mod.x0),
        copy(mod.s),
        copy(mod.Φ),
        copy(mod.lb),
        copy(mod.ub),
        copy(mod.lb_max),
        copy(mod.ub_max),
        copy(mod.Y),
        copy(mod.dists),
        copy(mod.Z),
        copy(mod.Pr),
        copy(mod.Pr_xi),
        copy(mod.LHS),
        copy(mod.RHS),
        copy(mod.coeff)
    )
end

function CE.copyto_model!(mod_trgt::RBFModel, mod_src::RBFModel)
    for fn in (
        :allow_nonlinear, :max_points, :search_factor, 
        :max_search_factor, :th_qr, :th_cholesky)
        @assert getfield(mod_trgt, fn) == getfield(mod_src, fn)
    end
    for fn in (
        #:point_indices, 
        :x0, :s, :Φ, :lb, :ub, :lb_max, :ub_max, :Y, :dists, :Z, :Pr, :Pr_xi, :LHS, 
        :RHS, :coeff
    )
        copyto!(getfield(mod_trgt, fn), getfield(mod_src, fn))
    end 
    empty!(mod_trgt.point_indices)
    append!(mod_trgt.point_indices, mod_src.point_indices)
    return nothing
end

function RBFDatabase(cfg::RBFConfig, dim_in, dim_out, T)
    min_points = dim_in + 1

    ## some heuristics to initialize database of points
    dat_size = if isnothing(cfg.database_size)
        ## 1 GB = 1 billion bytes = 10^9 bytes
        ## byte-size of one colum: `sizeof(T)*dim_in`
        ## 1 GB database => 10^9/(sizeof(T)*dim_in)
        ## When is `(dim_in + 1) * N >= 10^9/(sizeof(T)*dim_in)`?
        ## If `dim_in^2 + dim_in >= 10^9/(sizeof(T)*N)`...
        ## That is, there are many variables...
        max(min_points, min(min_points*100, round(Int, 10^9/(sizeof(T)*dim_in))))
    else
        if cfg.database_size < min_points
            @warn "There is not enough storage in the database, so we are using $(min_points) columns."
            min_points
        else
            cfg.database_size
        end
    end

    chunk_size = if isnothing(cfg.database_chunk_size)
        min(2*min_points, dat_size)
    else
        max(1, min(cfg.database_chunk_size, dat_size))
    end

    database_x = ElasticArray{T, 2}(undef, dim_in, chunk_size)
    database_y = ElasticArray{T, 2}(undef, dim_out, chunk_size)

    database_flags_x = zeros(Bool, chunk_size)
    database_flags_y = zeros(Bool, chunk_size)
    point_flags = zeros(Bool, chunk_size)
    locked_flags = copy(point_flags)
    round1_flags = cfg.allow_nonlinear ? zeros(Bool, chunk_size) : nothing

    x_index = Ref(0)

    return RBFDatabase(
        dim_in, dim_out, dat_size, chunk_size, database_x, database_y, database_flags_x, 
        database_flags_y, point_flags, round1_flags, x_index, locked_flags)
end

function trust_region_bounds!(lb, ub, x, Δ)
    lb .= x .- Δ 
    ub .= x .+ Δ
    return nothing
end
intersect_lower_bounds!(lb, _lb)=map!(max, lb, lb, _lb)
intersect_lower_bounds!(lb, ::Nothing)=nothing
intersect_upper_bounds!(ub, _ub)=map!(min, ub, ub, _ub)
intersect_upper_bounds!(ub, ::Nothing)=nothing
function trust_region_bounds!(lb, ub, x, Δ, global_lb, global_ub)
    trust_region_bounds!(lb, ub, x, Δ)
    intersect_lower_bounds!(lb, global_lb)
    intersect_upper_bounds!(ub, global_ub)
    return nothing
end    

function fill_diag!(Z, v)
    m, n = size(Z)
    k = min(m, n)
    for i=1:k
        Z[i, i] = v
    end
    nothing
end

pow2(x)=x^2
norm_squared(v) = sum(pow2, v; init=0)

function intersect_bound(xi::X, zi::Z, bi::B, T=Base.promote_type(X, Z, B)) where {X, Z, B}
    ## xi + σ * zi <= bi ⇔ σ * zi <= bi - xi == ri
    ri = bi - xi
    if iszero(zi)
        if ri < 0
            return T(NaN), T(NaN)
        else
            return T(-Inf), T(Inf)
        end
    elseif zi < 0
        return ri/zi, T(Inf)
    else
        return T(-Inf), ri/zi
    end
end

function intersect_intervals(l1, r1, l2, r2, T=typeof(l1))
    isnan(l1) && return l1, r1
    isnan(l2) && return l2, r2
    l = max(l1, l2)
    r = min(r1, r2)
    if l > r
        return T(NaN), T(NaN)
    end
    return l, r
end

function intersect_box(x::X, z::Z, lb::L, ub::U) where {X, Z, L, U}
    _T = Base.promote_eltype(X, Z, L, U)
    T = _T <: AbstractFloat ? _T : DEFAULT_PRECISION

    σ_min, σ_max = T(-Inf), T(Inf)
    for (xi, zi, lbi, ubi) = zip(x, z, lb, ub)    
        ## x + σ * z <= ub
        σl, σr = intersect_bound(xi, zi, ubi, T)
        σ_min, σ_max = intersect_intervals(σ_min, σ_max, σl, σr)
        ## lb <= x + σ * z ⇔ -x + σ * (-z) <= -lb
        σl, σr = intersect_bound(-xi, -zi, -lbi, T)
        σ_min, σ_max = intersect_intervals(σ_min, σ_max, σl, σr)
    end
    if abs(σ_min) >= abs(σ_max)
        return σ_min
    else
        return σ_max
    end
end

function grow_database!(rbf_database, force_chunk=0)
    @unpack database_x, locked_flags, database_y, database_flags_x, database_flags_y, point_flags, round1_flags = rbf_database
    @unpack chunk_size, max_database_size = rbf_database
   
    dim_in, current_size = size(database_x)
    dim_out, sz2 = size(database_y)
    @assert current_size == sz2 == length(database_flags_x) == length(database_flags_y) == length(point_flags)
    free_slots = max_database_size - current_size
    if free_slots > 0 || force_chunk > 0
        new_chunk = force_chunk > 0 ? force_chunk : min(chunk_size, free_slots, 1)
        new_size = current_size + new_chunk 
        resize!(database_x, dim_in, new_size)
        resize!(database_y, dim_out, new_size)
        append_zeros!(database_flags_x, new_chunk)
        append_zeros!(database_flags_y, new_chunk)
        append_zeros!(point_flags, new_chunk)
        append_zeros!(locked_flags, new_chunk)
        append_zeros!(round1_flags, new_chunk)
        return current_size + 1
    end
    return 0
end

function add_to_database!(rbf_database::RBFDatabase, x)
    @unpack locked_flags, database_x, database_flags_x, point_flags = rbf_database
    
    x_index = 0
    for (i, isused) = enumerate(database_flags_x)
        isused && continue
        locked_flags[i] && continue
        x_index = i
        break 
    end

    ## If we did not find a free slot, try to grow database
    if x_index == 0
        x_index = grow_database!(rbf_database)
    
        if x_index == 0
            ## growing the database dit not work. overwrite existing result
            for (i, ismodpoint) = enumerate(point_flags)
                ismodpoint && continue  # avoid throing out interpolation points
                locked_flags[i] && continue # and points that are locked
                x_index = i
                break
            end
        end
    end
    if x_index == 0
        @warn "At this point, the index of the new point should be positive. Growing the database over its specified limits."
        x_index = grow_database!(rbf_database, 1)
    end

    database_x[:, x_index] .= x
    database_flags_x[x_index] = true
    return x_index
end

function add_to_database!(rbf_database::RBFDatabase, x, fx)
    @unpack database_y, database_flags_y = rbf_database
    x_index = add_to_database!(rbf_database, x)
    database_y[:, x_index] .= fx
    database_flags_y[x_index] = true
    return x_index
end

function evaluate!(rbf_database, op)
    @unpack database_x, database_y, database_flags_x, database_flags_y = rbf_database
    budget = typemax(Int)
    if CE.is_counted(op)
        t = CE.max_calls(op)
        if t isa Tuple
            _t = first(t)
            if !isnothing(_t)
                budget = _t - first(CE.num_calls(op))
            end
        end
    end

    ## find those entries, where `x` is set, but `y` is missing
    ## TODO in case of parallelization, get rid of the for loop(s)
    for (i, xset) = enumerate(database_flags_x)
        if xset
            if !database_flags_y[i]
                if budget <= 0
                    return "RBF Training: No sampling budget. Aborting."
                end
                x = @view(database_x[:, i])
                y = @view(database_y[:, i])
                #src #delete eval_op!(y, op, x)
                @serve func_vals!(y, op, x)
                database_flags_y[i] = true
                budget -= 1
            end
        end
    end
    return nothing
end

function find_poised_points!(
    ## mutated:
    Y, Z, database_x, database_flags_x, point_flags, point_indices, round1_flags, Pr, Pr_xi, lb, ub,
    ## not mutated
    x, 
    Δ,      # actual trust-region radius, used to compute the pivot value
    Δ_box,  # unscaled radius of the sampling box; is scaled by search_factor to define the sampling region
    global_lb, global_ub, search_factor, th_qr, j0=0; 
    norm_p = Inf, log_level = Info
)
    dim_in = length(x)
    
    ## compute trust region box corners with enlarged radius
    trust_region_bounds!(lb, ub, x, search_factor*Δ_box, global_lb, global_ub)

    ## reset Z to identity matrix:
    if iszero(j0)
        fill!(Z, 0)
        fill_diag!(Z, 1)
    end

    proj_th = (th_qr * Δ)^2
    j = j0
    _Z = @view(Z[:,:])
    
    @logmsg log_level """
    \tRBF Construction: 
    \t                 Would like $(dim_in - j) points in box of radius θ*Δ=$(search_factor*Δ_box). 
    \t                 Pivot value is $(proj_th)."""
    @views for i = reverse(axes(database_x, 2))
        !database_flags_x[i] && continue  # not a point in the database
        point_flags[i] && continue        # point already considered an interpolation point
        istrue(round1_flags, i) && continue # point has already been looked at before
        xi = database_x[:, i]
        if all( lb .<= xi .<= ub )
            ## compute orthogonal projector `Pr` (onto complement of `Y[:, 1:j]`)
            LA.mul!(Pr, _Z, transpose(_Z))
            ## and project `xi`
            ξi = Y[:, j+1]
            ξi .= xi .- x
            LA.mul!(Pr_xi, Pr, ξi)
            ## threshold test passed => accept xi and update `Y` and `Z`
            if norm_squared(Pr_xi) >= proj_th
                point_flags[i] = true
                setflag!(round1_flags, i, true)
                push!(point_indices, i)
                j += 1
                #Y[:, j] .= xi .- x
                j == dim_in && break
                ## normalize basis `_Z` of orthogonal complement via QR decomposition
                Q, _ = LA.qr(Y[:, 1:j])    # NOTE I did not find a way to pre-allocate QR results yet
                _Z = Z[:, j+1:end]
                _Q = Q[:, j+1:end]
                _Z .= _Q ./ transpose(LA.norm.(eachcol(_Q), norm_p))
            end                
        end
    end
    return j
end

reset_flags!(flag_vec::Nothing)=nothing
reset_flags!(flag_vec)=fill!(flag_vec, false)
function update_rbf_model!(
    rbf, op, Δ, x, fx, global_lb=nothing, global_ub=nothing; 
    Δ_max=Δ, norm_p=Inf, log_level=Info
)
    dim_in = length(x)

    ## set shape parameter for current model
    rbf.shape_param[] = shape_parameter(rbf.kernel, Δ)

    ## reset indicators of interpolation points
    @unpack database, point_indices = rbf
    @unpack locked_flags, point_flags, round1_flags = database
    ## before, make sure that points of a prior model are not deleted from the database (in criticality routine)
    copy!(locked_flags, point_flags)
    ## now reset:
    reset_flags!(point_flags)
    reset_flags!(round1_flags)
    empty!(point_indices)

    ## make sure to include `x` as an interpolation point and put it into the database
    if rbf.x0 != x
        rbf.database.x_index[] = add_to_database!(database, x, fx)
    end
    copyto!(rbf.x0, x)
    x_index = rbf.database.x_index[]
    point_flags[x_index] = true
    push!(point_indices, x_index)
    rbf.Y[1:end-1, 1] .= 0
   
    ## look for a poised set in the enlarged trust region
    @unpack lb, ub, Z, Pr, Pr_xi, search_factor, th_qr = rbf
    _Y = rbf.Y
    Y = @view(_Y[1:end-1, 2:dim_in+1])
    @unpack database_x, database_flags_x = database
    j = find_poised_points!(Y, Z, database_x, database_flags_x, point_flags, point_indices, round1_flags,
        Pr, Pr_xi, lb, ub, x, Δ, Δ, global_lb, global_ub, search_factor, th_qr; norm_p, log_level)

    ## if points are missing, look in maximum trust region
    @unpack max_search_factor, lb_max, ub_max = rbf
    n_missing = dim_in - j
    fully_linear = n_missing == 0
    if !fully_linear && rbf.allow_nonlinear
        j = find_poised_points!(Y, Z, database_x, database_flags_x, point_flags, point_indices, round1_flags,
            Pr, Pr_xi, lb_max, ub_max, x, Δ, Δ_max, global_lb, global_ub, max_search_factor, th_qr, j; norm_p, log_level)
        n_missing = dim_in - j
        can_be_linear = false
    else
        can_be_linear = true
        trust_region_bounds!(lb_max, ub_max, x, max_search_factor*Δ_max, global_lb, global_ub)
    end

    ## if points are missing, sample along columns of `Z` 
    ## (in enlarged trust region, not maximum trust region)
    if n_missing > 0
        for i = j+1:dim_in
            z = @view(Z[:, i])
            σ = intersect_box(x, z, lb, ub)
            if abs(σ) < Δ * th_qr
                _σ = sign(σ) * Δ * th_qr
                @warn "Incompatible Box constraints for RBF models. Enlarging σ from $σ to $_σ."
                σ = _σ
            end
            z .*= σ
            Pr_xi .= x .+ z     # ignoring the name, use `Pr_xi` as a temporary cache
            project_into_box!(Pr_xi, global_lb, global_ub)
            xi_index = add_to_database!(database, Pr_xi)
            point_flags[xi_index] = true
            push!(point_indices, xi_index)

            j += 1
            Y[:, j] .= z
        end
    end
    @serve evaluate!(database, op)
    fully_linear = can_be_linear
    
    @assert j == dim_in    ## just to be sure
    
    ## add constant polynomial basis function to last row 
    rbf.Y[end, :] .= 1
    
    #@logmsg log_level "\tRBF Construction: Using samples $(point_indices). Looking for more."
    @logmsg log_level "\tRBF Construction: Using samples $(point_indices)."
    find_more_points!(rbf, x, log_level)
    compute_coefficients!(rbf)
end

function compute_coefficients!(rbf)
    ## we could re-use cholesky factors, but I want to avoid rounding errors
    ## (and I am lazy...)
    @unpack LHS, RHS, coeff, Φ, point_indices, database = rbf
    _Y = rbf.Y
    @unpack database_y = database
    npoints = length(point_indices)
    Y = @view(_Y[:, 1:npoints])

    dim_p = size(Y, 1)
    LHS[1:npoints, 1:npoints] .= @view(Φ[1:npoints, 1:npoints])
    LHS[1:npoints, npoints+1:npoints+dim_p] .= Y'
    LHS[npoints+1:npoints+dim_p, 1:npoints] .= Y
    LHS[npoints+1:npoints+dim_p, npoints+1:npoints+dim_p] .= 0
    A = @view(LHS[1:npoints+dim_p, 1:npoints+dim_p])
    B = @view(RHS[1:npoints+dim_p, :])
    B .= 0
    B[1:npoints, :] .= transpose(@view(database_y[:, point_indices]))
    try
        ## until https://github.com/JuliaLang/julia/pull/52957 is main, we have to try-catch 
        coeff[1:npoints+dim_p, :] .= A\B
    catch
        _A = LA.qr(A, LA.ColumnNorm())
        coeff[1:npoints+dim_p, :] .= _A\B
    end

    return nothing
end

function find_more_points!(rbf, x, log_level)
    φ = kernel_func(rbf)
    @unpack database = rbf
    @unpack database_x, point_flags, database_flags_x = database
    @unpack Y, Φ, Pr_xi, dists, lb_max, ub_max, max_points, th_cholesky, point_indices = rbf
    npoints = dim_p = length(point_indices)
    return find_more_points!(Y, Φ, Pr_xi, dists, point_flags, point_indices, database_flags_x, 
        x, lb_max, ub_max, φ, database_x, npoints, dim_p, max_points, th_cholesky, log_level)
end

function find_more_points!(
    Y, _Φ, Pr_xi, dists, point_flags, point_indices, database_flags_x,
    x, lb, ub, φ, database_x, npoints, dim_p, max_points, th_chol,
    log_level
)
    T = eltype(Y)
    @assert npoints == dim_p "This version does not yet support `npoints < `dim_p`."
    if npoints < max_points
        @logmsg log_level """
        \tRBF Construction: 
        \t                 Looking for additional points."""
        ## Finally, look for additional points in `lb_max`, `ub_max`...
        φ = build_rbf_matrix!(_Φ, dists, φ, database_x, point_indices)
        Φ = @view _Φ[1:npoints, 1:npoints]

        Π = @view(Y[:, 1:npoints])
        
        ## the poly matrix Π has size `(dim_p, npoints)`
        ## new (shifted) points are be added as columns
        Πt = transpose(Π)
        Q, _R = LA.qr(Πt)
        R = Matrix{T}(_R)
        ## Usually, `LA.qr` returns the thin factor `R`. At this point, it's equal to the 
        ## full factor, because `dim_p == npoints`. Otherwise, we would have to append 0s.
        ## `size(Πt) == (npoints, dim_p)` 
        ## ⇒ `size(Q) == (npoints, npoints)`, `size(_R) == (dim_p, dim_p)`
        ## Partition `Q = [Q1 Z]`, 
        ## ⇒ `size(Q1) == (npoints, dim_p)`, size(Z) == (npoints, npoints-dim_p)`
        ## `Z = Q[:, dim_p + 1:end]` is empty right now
        ## so would be `ZΦZ=Z'_Φ*Z` and `L = LA.cholesky(ZΦZ)`:

        #src _Z_ = Q[:, dim_p+1:end]
        #src _ZΦZ_ = _Z_'*Φ*_Z_
        #src _L_ = LA.cholesky(_ZΦZ_) |> Matrix
        #src _Linv_ = LA.inv(_L_)
        Z = Matrix{T}(undef, npoints, 0)
        Linv = Matrix{T}(undef, 0, 0)
        φ0 = Φ[1,1]

        for i in reverse(axes(database_x, 2))
            point_flags[i] && continue
            !database_flags_x[i] && continue
            xi = database_x[:, i]
            if all( lb .<= xi .<= ub)   ## TODO could store this info, too
                j = npoints + 1
                
                ## shifted site
                Pr_xi .= xi .- x
                
                ## rbf vector
                dist_xi(x) = LA.norm(x .- xi)
                φ_xi = map(φ, map(dist_xi, eachcol(@view(database_x[:, point_indices]))))

                ## augment factors (implicit copies, to keep `Q` and `R` from before)
                R_xi = [ R; Pr_xi' 1]
                Q_xi = cat(Q, 1, dims=(1,2))

                #src _Πtxi_ = [Πt; Pr_xi' 1]
                #src _Q_, _R_ = LA.qr(_Πtxi_)
                #src _Zxi_ = _Q_[:, dim_p+1:end]

                ## Suppose `Πt == Q*R`. Apply `dim_p` Givens rotations simultaneously:
                for l=1:dim_p
                    # in column `l`, use diagonal element to turn last element to zero:
                    g = first(LA.givens(R_xi[l, l], R_xi[j, l], l, j))
                    LA.lmul!(g, R_xi)
                    LA.rmul!(Q_xi, g')
                end
                ## new basis vector is contained in last column of Q
                Qg = @view(Q_xi[1:end-1, end])
                g = Q_xi[end, end]

                #src _Qg_ = @view(_Q_[1:end-1, end])
                #src _g_ = _Q_[end, end]

                v_xi = Z' * ( Φ * Qg + g .* φ_xi )
                σ_xi = Qg' * Φ * Qg + 2 .* g .* Qg'φ_xi + g^2 .* φ0
                #src _v_xi_ = _Z_' * ( Φ * _Qg_ + _g_ .* φ_xi )
                #src _σ_xi_ = _Qg_' * Φ * _Qg_ + 2 .* _g_ .* _Qg_'φ_xi + _g_^2 .* φ0
                σ_xi <= 0 && continue # the augmented matrix Φ is no longer p.d., xi is likely contained already
                #src _Φxi_ = [Φ φ_xi; φ_xi' φ0]
                τ_xi = sqrt(abs(σ_xi - norm_squared(Linv * v_xi)))
                #src _ZΦZxi_ = [_ZΦZ_ v_xi; v_xi' σ_xi]
                #src __ZΦZxi__ = _Zxi_' * _Φxi_ * _Zxi_
                #src _Lxi_ = [_L_ zeros(size(_L_,1), 1); (Linv*v_xi)' τ_xi]
                #src @show _Lxi_*_Lxi_' .- _ZΦZxi_

                if τ_xi >= th_chol
                    @logmsg log_level "\tRBF Construction: Adding point $i from database."
                    point_flags[i] = true
                    push!(point_indices, i)
                    Y[1:end-1, j] .= Pr_xi
                    Y[end, j] = 1
                   
                    _Φ[1:npoints, j] .= φ_xi
                    _Φ[j, 1:npoints] .= φ_xi
                    _Φ[j, j] = φ0
                    Φ = @view _Φ[1:j, 1:j]

                    npoints = j
                    npoints == max_points && break

                    Q = Q_xi
                    R = R_xi

                    Linv = [
					    Linv                        zeros(T, size(Linv,1));
					    -(v_xi'Linv'Linv)./τ_xi     1/τ_xi
				    ]

                    Z = @view(Q_xi[:, dim_p+1:end]) 
                end# if τ_x >= th_chol
            end
        end
    end
    return nothing
end

function kernel_func(rbf)
    _φ = rbf.kernel
    sp = rbf.shape_param[]
    return function(r) apply_kernel(_φ, r, sp) end
end

function kernel_diff(rbf)
    _φ = rbf.kernel
    sp = rbf.shape_param[]
    return function(r) apply_kernel_derivative(_φ, r, sp) end
end
   
function build_rbf_matrix!(rbf)
    @unpack database, Φ, dists = rbf
    @unpack database_x, point_indices, dim_in = database
    φ = kernel_func(rbf)
    return build_rbf_matrix(Φ, dists, φ, database_x, point_indices)
end

@views function build_rbf_matrix!(
    ## modified
    _Φ, dists, 
    ## not-modified
    φ, database_x, point_indices
)
    
    ## Each point `xi` defines a basis function ``ψᵢ(x) = φ(‖x-xᵢ‖)`` with kernel `φ`
    ## `Φ` is meant to store in `Φ[i,j]` the value ``ψᵢ(xⱼ)``, but its symmetric. 
    ## We first make dists hold ``‖xᵢ-xⱼ‖``, and then compute Ψ values.
    ## To safe some effort, we are going to exploit symmetry and build a lower triangular matrix
    X = database_x[:, point_indices]
    npoints = size(X, 2)

    Φ = _Φ[1:npoints, 1:npoints]
    for i=axes(X, 2)
        xi = X[:, i]
        R = dists[i:end, i]
        ## map!(x -> LA.norm(x .- xi), R, eachcol(X[:, i:end])) ## originally I used `i+1:end`, but I am too lazy to care for the more complicated indexing
        for (_j,j)=enumerate(i:npoints)
            xj = X[:, j]
            R[_j]= LA.norm(xi .- xj)
        end
        map!(φ, Φ[i:end, i], R)
    end

    ## this is a symmetric **view**:
    Φsym = LA.Symmetric(Φ, :L)
    copyto!(Φ, Φsym)
    return φ
end

# ## Interface Implementation
CE.depends_on_radius(::RBFModel)=true
CE.requires_grads(::RBFConfig)=false
CE.requires_hessians(::RBFConfig)=false

function CE.init_surrogate(cfg::RBFConfig, op, dim_in, dim_out, params, T)
    @unpack kernel, search_factor, max_search_factor, th_qr, th_cholesky = cfg
    
    shape_param = Ref(zero(T))

    database = RBFDatabase(cfg, dim_in, dim_out, T)
    point_indices = Int[]
    
    min_points = dim_in + 1
    max_points = if isnothing(cfg.max_points)
        min(database.max_database_size, min_points*10, Int(min_points*(dim_in+2)/2))
    else
        max(min_points, cfg.max_points)
    end

    x0 = fill(T(NaN), dim_in)
    s = zeros(T, min_points)

    ## RBF Basis Matrix
    Φ = Matrix{T}(undef, max_points, max_points)

    ## Polynomial matrix, columns correspond to shifted (and possibly scaled) sites
    Y = zeros(T, min_points, max_points)
    dists = zeros(T, min_points, min_points)
    Z = Matrix{T}(LA.I(dim_in))
    Pr = copy(Z)
    Pr_xi = zeros(T, dim_in)

    lb = zeros(T, dim_in)
    ub = zeros(T, dim_in)
    lb_max = zeros(T, dim_in)
    ub_max = zeros(T, dim_in)

    LHS = zeros(T, max_points + min_points, max_points + min_points)
    RHS = zeros(T, max_points + min_points, dim_out)
    coeff = copy(RHS)

    return RBFModel(
        kernel, shape_param, database, point_indices, 
        cfg.allow_nonlinear, max_points, 
        T(search_factor), T(max_search_factor), T(th_qr), T(th_cholesky), 
        x0, s, Φ, lb, ub, lb_max, ub_max, Y, dists, Z, Pr, Pr_xi, LHS, RHS, coeff)
end

@views function CE.model_op!(y, rbf::RBFModel, x)
    @unpack s, x0, point_indices, database, coeff, Φ, Y = rbf
    @unpack database_x, dim_x = database
    
    dim_p = dim_x + 1
    npoints = length(point_indices)

    Φ_coeff = transpose(coeff[1:npoints, :])                # transpose `coeff` before
    Π_coeff = transpose(coeff[npoints+1:npoints+dim_p, :])
    s[1:end-1] .= x .- x0
    s[end] = 1

    LA.mul!(y, Π_coeff, s)

    φ = kernel_func(rbf)
    _Φ = Φ[1:npoints, 1]
    for (_j,j) = enumerate(point_indices)
        #src map!(dist, _Φ, eachcol(database_x[:, point_indices]))
        _Φ[_j] = LA.norm(x .- database_x[:, j])
    end
    map!(φ, _Φ, _Φ)

    LA.mul!(y, Φ_coeff, _Φ, 1, 1)
    return nothing
end

function CE.model_grads!(Dy, rbf::RBFModel, x)
    ## size(Dy) = dim_x * dim_y
    @unpack s, x0, point_indices, database, coeff, Φ, Y, Pr_xi = rbf
    @unpack database_x, dim_x = database
    
    npoints = length(point_indices)
    dφ = kernel_diff(rbf)        

    Dy .= coeff[npoints+1:npoints+dim_x, :] # no coefficients for constant term

    for (_i, i) = enumerate(point_indices)
        ## rbf gradients
        ## φ(‖x‖), φ'(0) = 0 ⇒ ∇φ(x) = ∑ᵢ λᵢ φ'(rᵢ)/rᵢ (x - xᵢ)
        Pr_xi .= x .- database_x[:, i]
        r = LA.norm(Pr_xi)
        if iszero(r)
            continue
        end
        φdr = dφ(r)
        for j=axes(Dy, 2)
            c = coeff[_i, j] * φdr / r
            @views Dy[:, j] .+= c .* Pr_xi
        end 
    end
    return nothing
end

function CE.update!(
    rbf::RBFModel, op, Δ, x, fx, lb, ub; 
    Δ_max=Δ, log_level, kwargs...
)
    update_rbf_model!(rbf, op, Δ, x, fx, lb, ub; Δ_max, log_level, norm_p=Inf)
end

#src function model_op_and_grads! end # TODO
# TODO partial evaluation

export RBFModel, RBFConfig
export CubicKernel, GaussianKernel, InverseMultiQuadricKernel

end#module