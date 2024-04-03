abstract type AbstractValueCache{F<:AbstractFloat} end
abstract type AbstractMOPCache{F} <: AbstractValueCache{F} end
abstract type AbstractMOPSurrogateCache{F} <: AbstractValueCache{F} end

float_type(::AbstractValueCache{F}) where{F}=F

cached_x(::AbstractValueCache)::RVec=nothing
cached_fx(::AbstractValueCache)::RVec=nothing
cached_hx(::AbstractValueCache)::RVec=nothing
cached_gx(::AbstractValueCache)::RVec=nothing

cached_ξ(::AbstractMOPCache)::RVec=nothing
cached_Ax(::AbstractMOPCache)::RVec=nothing
cached_Ex(::AbstractMOPCache)::RVec=nothing
cached_Ax_min_b(::AbstractMOPCache)::RVec=nothing
cached_Ex_min_c(::AbstractMOPCache)::RVec=nothing

cached_theta(::AbstractMOPCache)::Real=nothing
cached_Phi(::AbstractMOPCache)::Real=nothing
cached_theta!(::AbstractMOPCache,val)::Bool=nothing
cached_Phi!(::AbstractMOPCache,val)::Bool=nothing

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