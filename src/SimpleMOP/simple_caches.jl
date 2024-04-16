"A struct holding values computed for or derived from an `AbstractMOP`."
Base.@kwdef struct SimpleValueCache{
	F<:AbstractFloat,
} <: AbstractMOPCache{F}
	"Unscaled variable vector used for evaluation."
    ξ :: Vector{F}
	"Internal (scaled) variable vector."
    x :: Vector{F}
	"Objective value vector."
    fx :: Vector{F}
	"Nonlinear equality constraints value vector."
    hx :: Vector{F}
	"Nonlinear inequality constraints value vector."
    gx :: Vector{F}
    
	Ex :: Vector{F}
    Ax :: Vector{F}

	Ax_min_b :: Vector{F}
	Ex_min_c :: Vector{F}
	
	"Reference to maximum constraint violation."
    theta_ref :: Base.RefValue{F}
	"Reference to maximum function value."
    phi_ref :: Base.RefValue{F}
end

struct SimpleSurrogateValueCache{
	F <: AbstractFloat,
} <: AbstractMOPSurrogateCache{F}
    Dx :: Vector{F}
    x :: Vector{F}
    fx :: Vector{F} 
    hx :: Vector{F}
    gx :: Vector{F}
    Dfx :: Matrix{F}
    Dhx :: Matrix{F}
    Dgx :: Matrix{F}
end

const SIMPLE_CACHE = Union{SimpleValueCache, SimpleSurrogateValueCache}

cached_x(simple_cache::SIMPLE_CACHE)=simple_cache.x
cached_fx(simple_cache::SIMPLE_CACHE)=simple_cache.fx
cached_hx(simple_cache::SIMPLE_CACHE)=simple_cache.hx
cached_gx(simple_cache::SIMPLE_CACHE)=simple_cache.gx

cached_ξ(simple_cache::SimpleValueCache)=simple_cache.ξ
cached_Ex(simple_cache::SimpleValueCache)=simple_cache.Ex
cached_Ax(simple_cache::SimpleValueCache)=simple_cache.Ax
cached_Ex_min_c(simple_cache::SimpleValueCache)=simple_cache.Ex_min_c
cached_Ax_min_b(simple_cache::SimpleValueCache)=simple_cache.Ax_min_b

cached_Dx(simple_cache::SimpleSurrogateValueCache)=simple_cache.Dx
cached_Dfx(simple_cache::SimpleSurrogateValueCache)=simple_cache.Dfx
cached_Dhx(simple_cache::SimpleSurrogateValueCache)=simple_cache.Dhx
cached_Dgx(simple_cache::SimpleSurrogateValueCache)=simple_cache.Dgx

function init_value_caches(mop::SimpleMOP{T}) where T
    
    ## initialize unscaled and scaled variables
    nx = dim_vars(mop) :: Int
    nfx = dim_objectives(mop) :: Int
    nhx = dim_nl_eq_constraints(mop) :: Int
    ngx = dim_nl_ineq_constraints(mop) :: Int
    nE = dim_lin_eq_constraints(mop) :: Int
    nA = dim_lin_ineq_constraints(mop) :: Int

    return init_simple_value_caches_for_mop(T, nx, nfx, nhx, ngx, nE, nA)
end
    
function init_simple_value_caches_for_mop(T, nx, nfx, nhx, ngx, nE, nA)
    ξ = array(T, nx)
    x = similar(ξ)

    ## pre-allocate value arrays
    fx = array(T, nfx)
    hx = array(T, nhx)
    gx = array(T, ngx)
    
    Ex = array(T, nE)
    Ax = array(T, nA)
    Ax_min_b = similar(Ax)
    Ex_min_c = similar(Ex)

    ## constraint violation and filter value
    theta_ref = Ref(zero(T))
    phi_ref = Ref(zero(T))

    return SimpleValueCache(; 
        ξ, x, fx, hx, gx, 
        Ax, Ex, Ax_min_b, Ex_min_c, 
        theta_ref, phi_ref
    )
end

function init_value_caches(mod::SimpleMOPSurrogate)
    T = float_type(mod)
    
    nx = dim_vars(mod) :: Int
    nfx = dim_objectives(mod) :: Int
    nhx = dim_nl_eq_constraints(mod) :: Int
    ngx = dim_nl_ineq_constraints(mod) :: Int
    return init_simple_value_caches_for_mod(T, nx, nfx, nhx, ngx)
end

function init_simple_value_caches_for_mod(T, nx, nfx, nhx, ngx)
    x = array(T, nx)
    Dx = similar(x)

    ## pre-allocate value arrays
    fx = array(T, nfx)
    hx = array(T, nhx)
    gx = array(T, ngx)

    Dfx = array(T, nx, nfx)
    Dhx = array(T, nx, nhx)
    Dgx = array(T, nx, ngx)

   return SimpleSurrogateValueCache(Dx, x, fx, hx, gx, Dfx, Dhx, Dgx)
end

function universal_copy!(
	mod_vals_trgt::SIMPLE_CACHE, 
	mod_vals_src::SIMPLE_CACHE
)
	for fn in fieldnames(typeof(mod_vals_trgt))
		trgt_fn = getfield(mod_vals_trgt, fn)
		custom_copy!(trgt_fn, getfield(mod_vals_src, fn))
	end
	return nothing
end