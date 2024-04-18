module CheetahViews

import OffsetArrays

const SlicesType = Union{Tuple, AbstractArray{<:AbstractArray}}

include("cheetah_cats.jl")

struct CheetahView{
    T,  # eltype
    N,  # view dimension
    SN, # slices dimension
    D,  # cat dimension
    A,  # type of slices
} <: AbstractArray{T, N}
    
    slices :: A

    sz :: Int
    idx :: Union{Nothing, Vector{Tuple{Int, Int}}}

    function CheetahView{T, N, D}(slices::A) where {T, N, D, A}
        SN = _slices_dimension(slices)
        @assert all(S -> ndims(S) == SN, slices) "Arrays must have equal dimensions."

        S1 = first(slices)
        @assert all(
            i -> all(S -> axes(S, i) == axes(S1, i), slices),
            Iterators.filter(!isequal(D), 1:SN)
        )

        if N > SN
            sz = length(slices)
            idx = nothing
        else
            sz = 0
            idx = Vector{NTuple{2,Int}}()
            for (l,S) in enumerate(slices)
                sz_S = size(S, D)
                for j = 1:sz_S
                    push!(idx, (l, j))
                end
                sz += sz_S
            end
        end
        return new{T, N, SN, D, A}(slices, sz, idx)
    end
    CheetahView{T, N, 0}(::A) where {T, N, A} = throw(ArgumentError("`dims=Val(0)` is not supported, do you mean `dims=Val(1)`?"))
end

_slices_dimension(arr::AbstractArray) = ndims(arr)
_slices_dimension(slices) = _slices_dimension(first(slices))

function CheetahView(slices::AbstractArray...; dims=Val(_default_dims(slices)))
    return CheetahView(slices, dims)
end
function CheetahView{T}(slices::AbstractArray...; dims=Val(_default_dims(slices))) where T
    return CheetahView{T}(slices, dims)
end
function CheetahView(slices, dims::Int)
    return CheetahView(slices, Val(dims))
end # type-unstable

function CheetahView(slices::SlicesType, dims::Val = Val(_default_dims(slices)))
   return CheetahView{_default_eltype(slices)}(slices, dims)
end

function CheetahView{T}(slices::SlicesType, dims::Val = Val(_default_dims(slices))) where T
    N = _max(dims, Val(_slices_dimension(slices)))
    # unify all the axes to 1-based ranges
    slices = map(OffsetArrays.no_offset_view, slices)
    return CheetahView{T, _value(N), _value(dims)}(slices)
end

@inline _default_dims(slices) = ndims(first(slices)) + 1
@inline function _default_eltype(slices)
    T = mapreduce(eltype, promote_type, slices)
    _isconcretetype(T) || throw(ArgumentError("Input arrays should be homogenous."))
    return T
end
function _isconcretetype(T)
    # We relax the restriction and allow `Union`
    # This is particularily useful for arrays with `missing` and `nothing`
    isconcretetype(T) && return true
    isa(T, Union) && return isconcretetype(T.a) && _isconcretetype(T.b) # NOTE recurse down? `_isconcretetype` instead of `isconcretetype`
    return false
end

function Base.size(A::CheetahView{T,N,SN,D}) where {T,N,SN,D}
    
    frame_size = size(first(A.slices))
    prev, post = Base.IteratorsMD.split(frame_size, Val(D-1))
    pre = _append_tuple(prev, Val(D-1))
    ## If N == SN ⇔ D <= N, then `prev` is a tuple of legnth `D-1`,
    ## and `_append_tuple` is a no-op.
    ## Otherwise, `pre` is right-padded with `1` until it has length `D-1`.
    succ = _tail(post, Val(N), Val(SN))
    
    return (pre..., A.sz, succ...)
end

function Base.axes(A::CheetahView{T,N,SN,D}) where {T,N,SN,D}
    frame_axes = axes(first(A.slices))
    prev, post = Base.IteratorsMD.split(frame_axes, Val(D-1))

    # use homogenous range to make _append_tuple happy
    fill_range = convert(typeof(first(frame_axes)), Base.OneTo(1))
    pre = _append_tuple(prev, Val(D-1), fill_range)
    succ = _tail(post, Val(N), Val(SN))

    return (
        pre...,
        Base.OneTo(A.sz),
        succ...
    )
end
@inline function Base.getindex(A::CheetahView{T,N,SN,D}, inds::Vararg{Int,N}) where {T,N,SN,D}
    @boundscheck checkbounds(A, inds...)
    prev, post = Base.IteratorsMD.split(inds, Val(D-1))
    idx, post = first(post), Base.tail(post)
    return _getindex(A, A.idx, idx, prev, post)
end

@inline function _getindex(A, ::Nothing, idx, prev, post)
    return @inbounds A.slices[idx][prev..., post...]
end
@inline function _getindex(A, idx_vec, idx, prev, post)
    l, j = @inbounds idx_vec[idx]
    pre = (prev..., j)
    return _getindex(A, nothing, l, pre, post)
end

@inline function Base.setindex!(A::CheetahView{T,N,SN,D}, x, inds::Vararg{Int, N}) where {T,N,SN,D}
    @boundscheck checkbounds(A, inds...)
    prev, post = Base.IteratorsMD.split(inds, Val(D-1))
    idx, post = first(post), Base.tail(post)
    return _setindex!(A, A.idx, idx, prev, post, x)
end
@inline function _setindex!(A, ::Nothing, idx, prev, post, x)
    @inbounds A.slices[idx][prev..., post...] = x
end
@inline function _setindex!(A, idx_vec, idx, prev, post, x)
    l, j = @inbounds idx_vec[idx]
    pre = (prev..., j)
    return _setindex!(A, nothing, l, pre, post, x)
end

# utils

# For type stability
@inline _max(::Val{x}, ::Val{y}) where {x, y} = Val(max(x, y))
@inline _value(::Val{N}) where N = N
@inline _append_tuple(t::NTuple{N1}, ::Val{N1}, x=1) where N1 = t
@inline _append_tuple(t::NTuple{N1}, ::Val{N2}, x=1) where {N1, N2} = _append_tuple((t..., x), Val(N2), x)
@inline _tail(t::Tuple, ::Val{N}, ::Val{N}) where N = Base.tail(t)
@inline _tail(t::Tuple, ::Val{N}, ::Val{SN}) where {N, SN} = t

export CheetahView
end
