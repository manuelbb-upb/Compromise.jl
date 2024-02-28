# Trust region boundaries are computed for the unshifted
# coordinate system, i.e., they can be used to query 
# the database.
# On the other hand, if we inspect “directions” as new 
# points, we might have to add the trust region center 
# `x` to shift them into these boxes.
 
"""
    trust_region_bounds!(lb, ub, x, Δ)

Make `lb` the lower left corner of a trust region hypercube with 
radius `Δ` and make `ub` the upper right corner."""
function trust_region_bounds!(lb, ub, x, Δ)
    lb .= x .- Δ 
    ub .= x .+ Δ
    return nothing
end
"""
    trust_region_bounds!(lb, ub, x, Δ, global_lb, global_ub)

Make `lb` the lower left corner of a trust region hypercube with 
radius `Δ` and make `ub` the upper right corner.
`global_lb` and `global_ub` are the global bound vectors or `nothing`."""
function trust_region_bounds!(lb, ub, x, Δ, global_lb, global_ub)
    trust_region_bounds!(lb, ub, x, Δ)
    intersect_lower_bounds!(lb, global_lb)
    intersect_upper_bounds!(ub, global_ub)
    return nothing
end    
## helpers to deal with `nothing` bounds
intersect_lower_bounds!(lb, _lb)=map!(max, lb, lb, _lb)
intersect_lower_bounds!(lb, ::Nothing)=nothing
intersect_upper_bounds!(ub, _ub)=map!(min, ub, ub, _ub)
intersect_upper_bounds!(ub, ::Nothing)=nothing

# The functions below are used to find a suitable scaling factor 
# for new directions by intersecting the ray `x + σ * z`
# with local boundary vectors.

# !!! note
#     These methods can also be used to intersect rays with 
#     polyhedral inequality constraints.
#     In the past, I have done so to get an initial steplength 
#     for backtracking along descent directions, but now we 
#     use a modified step-problem and that should work too (I think).

"""
    intersect_box(x, z, lb, ub)

Given vectors `x`, `z`, `lb` and `ub`, compute and return the largest 
interval `I` (a tuple with 2 elements) such that 
`lb .<= x .+ σ .* z .<= ub` is true for all `σ` in `I`. 
If the constraints are not feasible, `(NaN, NaN)` is returned.
If the direction `z` is zero, the interval could contain infinite elements.
"""
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

"""
    intersect_bound(xi, zi, bi)

Given number `xi`, `zi` and `bi`, compute and return an interval 
`I` (a tuple with 2 elements) such that `xi + σ * zi <= bi` is true
for all `σ` in `I`. 
If the constraint is feasible, at least one of the interval elements is infinite.
If it is infeasible, `(NaN, NaN)` is returned.
"""
function intersect_bound(
    xi::X, zi::Z, bi::B, T=Base.promote_type(X, Z, B)
) where {X<:Number, Z<:Number, B<:Number}
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

"Helper to intersect to intervals."
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

function do_qr!(::Any, A, _A=nothing)
    return LA.qr(A)
end

function do_qr!(ws::QRWYWs, A, _A=similar(A))
    m, n = size(A)
    __A = @view(_A[1:m, 1:n])
    copyto!(__A, A)
    return LA.QRCompactWY(geqrt!(ws, __A)...)
end

function _q_factor(F)
    return LA.QRCompactWYQ(getfield(F, :factors), F.T)
end
function _r_factor(F)    
    m, n = size(F)
    return LA.triu!(getfield(F, :factors)[1:min(m,n), 1:n])
end

function qr!(Q, R, A, ws::QRWYWs, _A=similar(A))
    qr = do_qr!(ws, A, _A)
    copyto!(Q, _q_factor(qr))
    copyto!(R, _r_factor(qr))
    nothing
end

function qr!(Q, R, A, ws::Nothing, _A=nothing)
    q, r = LA.qr(A)
    copyto!(Q, q)
    copyto!(R, r)        
end