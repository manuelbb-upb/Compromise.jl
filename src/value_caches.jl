float_type(::AbstractValueCache{F}) where{F}=F

## suffix `_x` or `_ξ` must be fully mutable arrays 
## other suffixes:
## these must be mutable in the sense that …
## * `objectives!` must be able to mutate `cached_fx(cache)`,
## * `nl_ineq_constraints!` must be able to mutate `cached_gx(cache)`, 
## etc.

cached_x(::AbstractValueCache)::RVec=nothing
cached_fx(::AbstractValueCache)::RVec=nothing
cached_hx(::AbstractValueCache)::RVec=nothing
cached_gx(::AbstractValueCache)::RVec=nothing

cached_ξ(::AbstractMOPCache)::RVec=nothing
cached_Ax(::AbstractMOPCache)::RVec=nothing
cached_Ex(::AbstractMOPCache)::RVec=nothing
cached_Ax_min_b(::AbstractMOPCache)::RVec=nothing
cached_Ex_min_c(::AbstractMOPCache)::RVec=nothing

cached_Dx(::AbstractMOPSurrogateCache)::RVec=nothing
cached_Dfx(::AbstractMOPSurrogateCache)::RMat=nothing
cached_Dhx(::AbstractMOPSurrogateCache)::RMat=nothing
cached_Dgx(::AbstractMOPSurrogateCache)::RMat=nothing

function universal_copy!(
    trgt::AbstractValueCache, src::AbstractValueCache)
    error("`universal_copy!` not implemented.")
end

dim_vars(c::AbstractValueCache)=length(cached_x(c))
dim_objectives(c::AbstractValueCache)=length(cached_fx(c))
dim_nl_eq_constraints(c::AbstractValueCache)=length(cached_hx(c))
dim_lin_eq_constraints(c::AbstractValueCache)=length(cached_Ex(c))
dim_nl_ineq_constraints(c::AbstractValueCache)=length(cached_gx(c))
dim_lin_ineq_constraints(c::AbstractValueCache)=length(cached_Ax(c))

struct WrappedMOPCache{F, cacheType<:AbstractMOPCache{F}}<:AbstractMOPCache{F}
	cache :: cacheType
	hash_theta :: Base.RefValue{UInt64}
	hash_Phi :: Base.RefValue{UInt64}
	theta_ref :: Base.RefValue{F}
	Phi_ref :: Base.RefValue{F}
end

function WrappedMOPCache(cache::AbstractMOPCache{F}) where F
    NaNF = F(NaN)
    return WrappedMOPCache(
        cache,
        Ref(hash(NaNF)),
        Ref(hash(NaNF)),
        Ref(NaNF),
        Ref(NaNF),
    )
end

@forward WrappedMOPCache.cache float_type(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache cached_x(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache cached_fx(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache cached_hx(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache cached_gx(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache cached_ξ(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache cached_Ax(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache cached_Ex(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache cached_Ax_min_b(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache cached_Ex_min_c(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache dim_vars(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache dim_objectives(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache dim_nl_eq_constraints(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache dim_nl_ineq_constraints(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache dim_lin_eq_constraints(wcache::WrappedMOPCache)
@forward WrappedMOPCache.cache dim_lin_ineq_constraints(wcache::WrappedMOPCache)

function cached_theta(wcache::WrappedMOPCache)
    cache = wcache.cache
    x = cached_x(cache)
    hash_old = wcache.hash_theta[]
    hash_x = hash(x)
    if hash_old != hash_x
        theta = constraint_violation(
            cached_hx(cache),
            cached_gx(cache),
            cached_Ex_min_c(cache),
            cached_Ax_min_b(cache)
        )
        wcache.theta_ref[] = theta
        wcache.hash_theta[] = hash_x
    end
    return wcache.theta_ref[]
end

function cached_Phi(wcache::WrappedMOPCache)
    cache = wcache.cache
    x = cached_x(cache)
    hash_old = wcache.hash_Phi[]
    hash_x = hash(x)
    if hash_old != hash_x
        Phi = maximum(cached_fx(cache))
        wcache.Phi_ref[] = Phi
        wcache.hash_Phi[] = hash_x
    end
    return wcache.Phi_ref[]
end
