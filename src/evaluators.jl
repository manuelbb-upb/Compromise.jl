# depends on 
# * `indices.jl` (`ScalarIndex`, etc.)

"""
	@forward_pipe T, get_inner, method1, method2, ...

For each method `m` in the list of methods, define a method `m`
that exepts an object `x` of type `T` as its first argument and
passes all other `args...` and `kwargs...` done to 
`m( get_inner(x), args...; kwargs...)`.
"""
macro forward_pipe(typename_ex, extractor_method_ex, fs)
  T = esc(typename_ex)
  getter = esc(extractor_method_ex)
  fs = isexpr(fs, :tuple) ? map(esc, fs.args) : [esc(fs)]
  :($([:($f(x::$T, args...; kwargs...) = (Base.@_inline_meta; $f( $(getter)(x), args...; kwargs...)))
       for f in fs]...);
    nothing)
end

"""
	@forward_pipes T, get_inner, method1, method2, ...

For each method `m` in the list of methods, define a method `m`
that exepts an object `x` of type `T` as its first argument, 
some index `l` as its second argument and
passes all other `args...` and `kwargs...` done to 
`m( get_inner(x, l), args...; kwargs...)`.
"""
macro forward_pipes(typename_ex, extractor_method_ex, fs)
  T = esc(typename_ex)
  getter = esc(extractor_method_ex)
  fs = isexpr(fs, :tuple) ? map(esc, fs.args) : [esc(fs)]
  :($([:($f(x::$T, sub_index, args...; kwargs...) = (Base.@_inline_meta; $f( $(getter)(x, sub_index), args...; kwargs...)))
       for f in fs]...);
    nothing)
end

#=====================================================================
AbstractInnerEvaluator
=====================================================================#

# `AbstractInnerEvaluator` interface 
# This interface meant to be implemented by surrogate models and 
# user provided functions alike. The latter will likely be wrapped in other 
# types to make the below methods availabe.
# An `AbstractInnerEvaluator` is a “mathematical” object: 
# It takes real-valued vectors as input and returns real-valued vectors for output.
# The mapping of variable dictionaries to vectors happens elsewhere.
Base.broadcastable( ev :: AbstractInnerEvaluator ) = Ref( ev )

#_num_inputs( :: AbstractInnerEvaluator ) :: Int = 0
# (mandatory)
num_outputs( :: AbstractInnerEvaluator ) :: Int = 0

# ##################### counter helpers #########################

# (optional) used for stopping when user provided functions reach a limit
struct EmptyRefCounter end 
const EMPTY_COUNTER = EmptyRefCounter()

num_eval_counter( :: AbstractInnerEvaluator ) :: Union{EmptyRefCounter, Base.RefValue{Int}} = EMPTY_COUNTER

Base.getindex( :: EmptyRefCounter ) = 0 
Base.setindex!( :: EmptyRefCounter, x ) = nothing

function increase_counter!( c :: Union{EmptyRefCounter, Base.RefValue{Int}}, N = 1 )
	c[] += N
	return nothing 
end

function set_counter!( c :: Union{EmptyRefCounter, Base.RefValue{Int}}, N = 0 )
	c[] = N
	return nothing 
end

function increase_counter!( ov :: AbstractInnerEvaluator, N = 1 )
	return increase_counter!(num_eval_counter( ov ), N)
end

function set_counter!( ov :: AbstractInnerEvaluator, N = 1 )
	return set_counter!(num_eval_counter( ov ), N)
end

function num_evals( ov :: AbstractInnerEvaluator ) 
	return num_eval_counter( ov )[]
end
# #################################################################

# (mandatory)
_eval_at_vec( :: AbstractInnerEvaluator, :: Vec ) :: Vec = nothing

# derived defaults:
function _eval_at_vec( ev :: AbstractInnerEvaluator, x :: Vec, output_number )
	return _eval_at_vec(ev, x)[ output_number ]
end

function _eval_at_vecs( ev :: AbstractInnerEvaluator, X :: VecVec )
	return _eval_at_vec.(ev, X)
end

function _eval_at_vecs( ev :: AbstractInnerEvaluator, X :: VecVec, output_number )
	return [v[output_number] for v in _eval_at_vecs(ev, X, output_number)]
end

function eval_at_vec( ev :: AbstractInnerEvaluator, args... ) 
	increase_counter!(ev)
	return _eval_at_vec( ev, args... )
end

function eval_at_vecs( ev :: AbstractInnerEvaluator, X :: VecVec, args... ) 
	increase_counter!(ev, length(X))
	return _eval_at_vecs( ev, X, args... )
end

_provides_gradients( :: AbstractInnerEvaluator ) :: Bool = false
_provides_jacobian( :: AbstractInnerEvaluator ) :: Bool = false
_provides_hessians( :: AbstractInnerEvaluator ) :: Bool = false

_gradient( :: AbstractInnerEvaluator, :: Vec; output_number ) = nothing
_jacobian( :: AbstractInnerEvaluator, :: Vec ) = nothing 
_partial_jacobian( :: AbstractInnerEvaluator, :: Vec ; output_numbers ) = nothing 
_hessian( :: AbstractInnerEvaluator, :: Vec; output_number ) = nothing

# derived methods that are used in other places
function provides_gradients( ev :: AbstractInnerEvaluator ) 
	if _provides_gradients( ev )
		return true 
	else
		return _provides_jacobian( ev )
	end
end

function provides_jacobian( ev :: AbstractInnerEvaluator ) 
	if _provides_jacobian( ev )
		return true 
	else
		return _provides_gradients( ev )
	end
end

provides_hessians( ev :: AbstractInnerEvaluator ) = _provides_hessians( ev )

function partial_jacobian( ev :: AbstractInnerEvaluator, x :: Vec; output_numbers )
	J = _partial_jacobian( ev, x; output_numbers )
	if isnothing(J)
		return _jacobian(ev, x)[ output_numbers, : ]
	else
		return J
	end
end

# helper
function _jacobian_from_grads( ev, x, :: Nothing )
	return transpose( 
		hcat( (gradient(ev, x; output_number) for output_number=1:num_outputs(ev) )... )
	)
end

#helper 
function _jacobian_from_grads( ev, x, output_numbers )
	return transpose(
		hcat( (gradient(ev, x; output_number) for output_number in output_numbers)... )
	)
end

function jacobian( ev :: T, x :: Vec; output_numbers = nothing ) where T <: AbstractInnerEvaluator
	if _provides_jacobian( ev )
		if isnothing( output_numbers )
			return _jacobian( ev, x, )
		else
			return partial_jacobian( ev, x; output_numbers )
		end
	elseif _provides_gradients( ev )
		return _jacobian_from_grads( ev, x, output_numbers)
	else
		error("`jacobian` not availabe for evaluator of type $(T).")
	end
end

function gradient( ev :: T, x :: Vec; output_number ) where T <: AbstractInnerEvaluator
	if _provides_gradients( ev )
		return _gradient( ev, x; output_number )
	elseif _provides_jacobian( ev )
		return vec( jacobian( ev, x; output_numbers = [output_number,] ) )
	else
		error("`gradient` not availabe for evaluator of type $(T).")
	end
end

function hessian( ev :: T, x :: Vec; output_number ) where T <: AbstractInnerEvaluator
	if _provides_hessians( ev )
		return _hessian( ev, x; output_number )
	else
		error("`hessian` not availabe for evaluator of type $(T).")
	end
end


#=====================================================================
AbstractOuterEvaluator
=====================================================================#
# An `ov::AbstractOuterEvaluator` has one set or several sets of 
# `input_indices(ov)`, referencing either variables or 
# dependent variables (output indices of other evaluators).
# `ov` also has a **single** set of `output_indices(ov)` and 
# it somehow transforms values associated with the concatenated 
# input indices to result values for the output indices. 
# More precisely, an `inner_transformer(ov)::AbstractInnerEvaluator` is specified,
# and  this inner evaluator is used to transfrom the value vectors.
# Alternatively, the method `_transform_input_dict` can be overwritten, 
# but then custom derivatives have to be specified too.
# Each `ov::AbstractOuterEvaluator` is a node in an evaluator tree: 
# It either is a root node (`is_atomic(ov) == true`) or the function 
# `input_provider(ov,l)` gives some other `av::AbstractOuterEvaluator`
# such that `input_indices(ov,l) == output_indices(av)`.

# The AbstractOuterEvaluator interface also "forwards" the AbstractInnerEvaluator methods.
Base.broadcastable( ov :: AbstractOuterEvaluator ) = Ref(ov)

"Return `true` if the outer evaluator takes only variables as its input 
and evaluates a single `AbstractInnerEvaluator`."
is_atomic( :: AbstractOuterEvaluator{F} ) where F = F

input_indices( :: AbstractOuterEvaluator{true} ) = Indices(VariableIndex[])
output_indices( :: AbstractOuterEvaluator ) = Indices(DependentIndex[])

num_input_sets( :: AbstractOuterEvaluator{false} ) = 1 

input_indices( :: AbstractOuterEvaluator{false}, i ) = Indices(DependentIndex[])

function input_indices( ov :: AbstractOuterEvaluator{false} )
	return reduce( union, input_indices( ov, l ) for l = 1 : num_input_sets(ov) )
end

# ( mandatory )
inner_transformer( :: AbstractOuterEvaluator ) = nothing
input_provider(:: AbstractOuterEvaluator{false}, l) :: AbstractOuterEvaluator = nothing

num_inputs( ov :: AbstractOuterEvaluator{true} ) = length(input_indices(ov))
num_outputs( ov :: AbstractOuterEvaluator ) = length(output_indices(ov))
num_inputs( ov :: AbstractOuterEvaluator{false}, i ) = length(input_indices(ov, i))
function num_inputs( ov :: AbstractOuterEvaluator{false} )
	return sum( num_inputs(ov, i) for i=1:num_input_sets(ov) )
end

model_cfg( :: AbstractOuterEvaluator ) :: AbstractSurrogateConfig = DUMMY_CONFIG
max_evals( ov :: AbstractOuterEvaluator ) = max_evals( model_cfg(ov) )

@forward_pipe(
	AbstractOuterEvaluator{true},
	inner_transformer,
	( 
		provides_gradients, 
		provides_jacobian,
		provides_hessians,
		gradient,
		jacobian,
		hessian,
	)
)

# (helper)
function _collect_input_vector(ov, xd)
	return collect( getindices(xd, input_indices(ov)) )
end

# (helper)
function _transform_input_vector(ov, x)
	return Dictionary( 
		output_indices( ov ),
		eval_at_vec( inner_transformer(ov), x )
	)
end

function _transform_input_dict(
	ov :: AbstractOuterEvaluator, 
	xd # :: AbstractDictionary{<:ScalarIndex, <:Real} 
)
	x = _collect_input_vector(ov, xd)
	return _transform_input_vector(ov, x)
end

function eval_at_dict( 
	ov :: AbstractOuterEvaluator{true}, 
	xd # :: AbstractDictionary{<:ScalarIndex,<:Real}
)
	_transform_input_dict(ov, xd)
end

# (helper)
function overwrite( _out :: AbstractDictionary{K,T}, _src :: AbstractDictionary{K,V} ) where{K,V,T}
	X = Base.promote_type(V,T)
	out = map( X, _out )
	src = map( X, _src )
	return overwrite(out, src)
end

function overwrite( out :: AbstractDictionary{K,V}, src :: AbstractDictionary{K,V} ) where{K,V}
	overwrite!(out, src)
	return out
end

function overwrite!( out :: AbstractDictionary{K,V}, src :: AbstractDictionary{K,V} ) where{K,V}
	for (k,v) = pairs(src)
		setindex!(out, v, k)
	end
	return nothing
end

function overwrite!( out :: AbstractDictionary{K,T}, src :: AbstractDictionary{K,V} ) where{K,T,V}
	for (k,v) = pairs(src)
		setindex!(out, convert(T,v), k)
	end
	return nothing
end

function _recursive_input_dict(ov, xd :: AbstractDictionary{K,T}) where {K,T}
	N = num_input_sets( ov )
	out = similar( input_indices(ov), T )
	for l = 1 : N
		iv = input_provider(ov, l)
		vals_dict = eval_at_dict( iv, xd )
		out = overwrite(out, vals_dict)
	end
	return out
end

function _recursive_input_dict!(ov, xd :: AbstractDictionary )
	N = num_input_sets( ov )
	for l = 1 : N
		iv = input_provider(ov, l)
		eval_at_dict!( iv, xd )
	end
	return xd
end

function eval_at_dict!( 
	ov :: AbstractOuterEvaluator{false}, 
	xd :: AbstractDictionary{>:ScalarIndex,<:Real}
)
	oind = output_indices(ov)
	if !(haskey( xd, oind[end]))
		ξd = _recursive_input_dict!(ov, xd)
		setindices!(xd, oind, _transform_input_dict(ov, ξd))
	end
	return xd
end

function eval_at_dict!( 
	ov :: AbstractOuterEvaluator{true}, 
	xd :: AbstractDictionary{>:ScalarIndex,<:Real}
)
	oind = output_indices(ov)
	if !(haskey( xd, oind[end]))
		setindices!(xd, oind, eval_at_dict(ov, xd))
	end
	return xd
end


# (helper)
function _full_make_cotangent_from_input_vector(ov, x)
	J = jacobian( inner_transformer(ov), x )
	T = eltype(J)
	m = size(J,1)

	in_ind = input_indices( ov )
	x_ind = keys(x)
	out_ind = output_indices( ov )
	return Dictionary( 
		x_ind,
		[ let i = findfirst( in_ind, x_i );	
			Dictionary(
				out_ind,
				isnothing(i) ? zeros(T, m) : J[:,i]
			) 
			end	for x_i in x_inds
		]
	)
end

function _make_cotangent_from_input_vector(ov, x)
	J = jacobian( inner_transformer(ov), x )
	in_ind = input_indices( ov )
	out_ind = output_indices( ov )
	return Dictionary( 
		in_ind,
		[
			Dictionary(
				out_ind,
				Jcol	
			) 
			for Jcol = eachcol(J)
		]
	)
end

function _full_make_jacobi_from_input_vector(ov, x)
	J = jacobian( inner_transformer(ov), x )
	T = eltype(J)
	m = size(J,2)

	in_ind = input_indices( ov )
	x_ind = keys(x)
	out_ind = output_indices( ov )
	return Dictionary( 
		out_ind,
		[ let i = findfirst( in_ind, x_i );	
			Dictionary(
				x_ind,
				isnothing(i) ? zeros(T, m) : J[i,:]
			) 
			end	for x_i in x_inds
		]
	)
end

function _make_jacobi_from_input_vector(ov, x)
	J = jacobian( inner_transformer(ov), x )
	in_ind = input_indices( ov )
	out_ind = output_indices( ov )
	return Dictionary( 
		out_ind,
		[ 
			Dictionary(
				in_ind,
				Jcol 
			) 
			for Jcol = eachrow(J)	# ! not copying, each Jcol is a `SubArray`
		]
	)
end

function jacobi_and_primal( ov :: AbstractOuterEvaluator{true}, xd )
	x = _collect_input_vector(ov, xd)
	cotangent_dict = _make_jacobi_from_input_vector(ov, x)
	primal_dict = _transform_input_vector(ov, x)
	return cotangent_dict, primal_dict
end

function jacobi( ov :: AbstractOuterEvaluator{true}, xd )
	x = _collect_input_vector(ov, xd)
	cotangent_dict = _make_jacobi_from_input_vector(ov, x)
	return cotangent_dict
end

function cotangent_and_primal( ov :: AbstractOuterEvaluator{true}, xd )
	x = _collect_input_vector(ov, xd)
	cotangent_dict = _make_cotangent_from_input_vector(ov, x)
	primal_dict = _transform_input_vector(ov, x)
	return cotangent_dict, primal_dict
end

function cotangent( ov :: AbstractOuterEvaluator{true}, xd )
	x = _collect_input_vector(ov, xd)
	cotangent_dict = _make_cotangent_from_input_vector(ov, x)
	return cotangent_dict
end

# helper
function _concat_sub_dicts( 
	_out :: AbstractDictionary{K,<:AbstractDictionary{I,V}}, 
	src :: AbstractDictionary{K,<:AbstractDictionary{I,W}},
	args...
) where{K,I,V,W}
	X = Base.promote_type( V, W )
	out = map( Dictionary{K,X}, _out )
	return concat_sub_dicts( out, src, args... )
end

# helper
function _concat_sub_dicts( 
	out :: AbstractDictionary{K,<:AbstractDictionary{I,V}}, 
	src :: AbstractDictionary{K,<:AbstractDictionary{I,V}},
	full_sub_indices
) where{K,I,V}
	for (k, sub_dict) = pairs(src)
		if haskey( out, k )
			set!( out, k, overwrite( getindex(out, k) , sub_dict ) )
		else
			insert!( out, k, overwrite( 
				zeros(V, full_sub_indices),
				sub_dict 
			))
		end
	end
	return out
end

# helper
function _concat_sub_dicts( 
	out :: AbstractDictionary{K,<:AbstractDictionary{I,V}}, 
	src :: AbstractDictionary{K,<:AbstractDictionary{I,V}},
) where{K,I,V}
	for (k, sub_dict) = pairs(src)
		if haskey( out, k )
			set!( out, k, merge( getindex(out, k) , sub_dict ) )
		else
			insert!( out, k, sub_dict )
		end
	end
	return out
end

function _recursive_cotangent_and_primal_dicts( ov, xd :: AbstractDictionary{K,V} ) where {K,V}
	in_ind = input_indices(ov)
	II = eltype(in_ind)
	
	_cot = Dictionary{K, Dictionary{II, V}}()
	_prim = zeros(V, in_ind)
	for l = 1:num_input_sets(ov)
		iv = input_provider(ov, l)
		__cot, __prim = cotangent_and_primal(iv, xd)
		_prim = overwrite(_prim, __prim)

		# `__cot` is a Dictionary of Dictionaries.
		# its keys are input indices from the leaves in the evaluation tree
		# i.e, they must be of type VariableIndex
		# each sub-dict has a subset of the current input_indices as its 
		# keys (i.e., intermediant variables) and stores the accumulated 
		# derivative values
		_cot = _concat_sub_dicts( _cot, __cot, in_ind )
	end
	
	return _cot, _prim
end

function jacobi_and_primal_vector( ov ::  AbstractOuterEvaluator{false}, xd )
	RHS, _prim = _recursive_cotangent_and_primal_dicts(ov, xd)
	x = _collect_input_vector(ov, _prim)
	
	LHS = _make_jacobi_from_input_vector(ov, x)
	ret_dict = dictionary(
		[
			lk => dictionary(
				[ rk => sum( rhs .* lhs ) for (rk, rhs) = pairs(RHS) ]
			)
			for (lk, lhs) = pairs(LHS)
		]
	)
	return ret_dict, x
end

function cotangent_and_primal_vector( ov ::  AbstractOuterEvaluator{false}, xd )
	LHS, _prim = _recursive_cotangent_and_primal_dicts(ov, xd)
	x = _collect_input_vector(ov, _prim)
	
	RHS = _make_jacobi_from_input_vector(ov, x)
	ret_dict = dictionary(
		[
			lk => dictionary(
				[ rk => sum( rhs .* lhs ) for (rk, rhs) = pairs(RHS) ]
			)
			for (lk, lhs) = pairs(LHS)
		]
	)
	return ret_dict, x
end

function jacobi( ov ::  AbstractOuterEvaluator{false}, xd )
	return jacobi_and_primal_vector(ov, xd)[1]
end

function jacobi_and_primal( ov ::  AbstractOuterEvaluator{false}, xd )
	Jd, x = jacobi_and_primal_vector(ov, xd)
	return Jd, _transform_input_vector(ov, x)
end 

function cotangent( ov ::  AbstractOuterEvaluator{false}, xd )
	return cotangent_and_primal_vector(ov, xd)[1]
end

function cotangent_and_primal( ov ::  AbstractOuterEvaluator{false}, xd )
	Jd, x = cotangent_and_primal_vector(ov, xd)
	return Jd, _transform_input_vector(ov, x)
end 

#%%
include("_differentiation.jl")

#=====================================================================
WrappedUserFunc <: AbstractInnerEvaluator 
=====================================================================#
struct WrappedUserFunc{
#	F <: Function, 
	B, 
	D <: Union{Nothing, FuncContainerBackend}
} <: AbstractInnerEvaluator
	#func :: F
	func :: Function
	num_outputs :: Int
	differentiator :: D
	counter :: Base.RefValue{Int}
end

function WrappedUserFunc( func :: F;
 	num_outputs :: Int, 
	differentiator :: D = nothing,
	can_batch :: Bool = false,
) where {
	F <:Function, 
	D <: Union{Nothing, FuncContainerBackend},
}
	#return WrappedUserFunc{F, can_batch, D}(
	return WrappedUserFunc{can_batch, D}(
		func,
		num_outputs,
		differentiator,
		Ref(0)
	)
end

num_outputs( wuf :: WrappedUserFunc ) = wuf.num_outputs
num_eval_counter( wuf :: WrappedUserFunc ) = wuf.counter

_eval_at_vec( wuf :: WrappedUserFunc, x :: Vec ) = wuf.func(x)

function _eval_at_vecs( wuf :: WrappedUserFunc{true, <:Any}, X :: VecVec )
	return wuf.func( X )
end

_provides_gradients( wuf :: WrappedUserFunc ) = true
_provides_jacobian( wuf :: WrappedUserFunc ) = true
_provides_hessians( wuf :: WrappedUserFunc ) = true 

_provides_gradients( wuf :: WrappedUserFunc{<:Any, Nothing} ) = false
_provides_jacobian( wuf :: WrappedUserFunc{<:Any, Nothing} ) = false
_provides_hessians( wuf :: WrappedUserFunc{<:Any, Nothing} ) = false

function _gradient( wuf :: WrappedUserFunc, x :: Vec; output_number )
	return gradient( wuf.differentiator, wuf, x; output_number )
end

function _jacobian( wuf :: WrappedUserFunc, x :: Vec )
	return jacobian( wuf.differentiator, wuf, x )
end

function _partial_jacobian( wuf :: WrappedUserFunc, x :: Vec; output_numbers)
	return partial_jacobian( wuf.differentiator, wuf, x; output_numbers )
end

function _hessian( wuf :: WrappedUserFunc, x :: Vec; output_number )
	return hessian( wuf.differentiator, wuf, x; output_number )
end

#=====================================================================
InnerIdentity <: AbstractInnerEvaluator 
=====================================================================#
struct InnerIdentity <: AbstractInnerEvaluator 
	n_out :: Int 
end 

function _eval_at_vec( ii :: InnerIdentity, x :: Vec )
	@assert length(x) == ii.n_out 
	return x 
end

num_outputs( ii :: InnerIdentity ) = ii.n_out

_provides_gradients( :: InnerIdentity ) = true
_provides_jacobian( :: InnerIdentity ) = true
_provides_hessians( :: InnerIdentity ) = true 

function _gradient( :: InnerIdentity,  x :: Vec, output_number )
	return sparsevec( [output_number,], true, length(x) )
end

function _jacobian( ii :: InnerIdentity, x :: Vec )
	#return LinearAlgebra.I( num_outputs(ii) )
	return LinearAlgebra.I( length(x) )
end

function _partial_jacobian(  :: InnerIdentity, x :: Vec, output_numbers)
	m = length(output_numbers)
	n = length(x)
	return sparse( 1:m, output_numbers, ones(Bool, m), m, n)
end

function _hessian( ii :: InnerIdentity, args... )
	n = num_outputs( ii )
	return spzeros(Bool, n, n)
end

#=====================================================================
OuterIdentity <: AbstractOuterEvaluator{true}
=====================================================================#
# ("atomic" building block)
@with_kw struct OuterIdentity{
	II <: AbstractIndices{VariableIndex},
} <: AbstractOuterEvaluator{true}
	input_indices :: II 
	num_inputs :: Int = length(input_indices)
	inner :: InnerIdentity = InnerIdentity( num_inputs )
	output_indices = Indices([DependentIndex() for i=1:num_inputs])
	
	function OuterIdentity{II}(
		input_indices :: II, num_inputs :: Int, inner :: InnerIdentity, output_indices
	) where II <: AbstractIndices{VariableIndex}
		return new{II}(input_indices, num_inputs, inner, output_indices)
	end
	function OuterIdentity(
		_input_indices :: _II, num_inputs :: Int, inner :: InnerIdentity, output_indices
	) where {_II <: AbstractVector{VariableIndex}}
		input_indices = Indices( _input_indices )
		return new{typeof(input_indices)}(input_indices, num_inputs, inner, output_indices)
	end
end

inner_transformer( oi :: OuterIdentity ) = oi.inner
input_indices( oi :: OuterIdentity ) = oi.input_indices
output_indices( oi :: OuterIdentity ) = oi.output_indices
num_inputs(oi :: OuterIdentity) = oi.num_inputs
num_outputs(oi :: OuterIdentity) = num_inputs(oi)
model_cfg( :: OuterIdentity ) = DUMMY_CONFIG

#=====================================================================
VecFun <: AbstractOuterEvaluator{true}
=====================================================================#
@with_kw struct VecFunc{
	I <: AbstractInnerEvaluator,
	C <: AbstractSurrogateConfig,
	II <: AbstractIndices{VariableIndex},
#OI <: AbstractVector{<:DependentIndex}
} <: AbstractOuterEvaluator{true}
	transformer :: I
	model_cfg :: C 
	num_outputs :: Int 
	
	input_indices :: II 
	output_indices = Indices([DependentIndex() for i=1:num_outputs])

	num_inputs :: Int = length(input_indices)
	function VecFunc{I,C,II}(
		transformer::I, model_cfg::C, num_outputs::Int, input_indices::II, output_indices, num_inputs
	) where {I <: AbstractInnerEvaluator,C <: AbstractSurrogateConfig,II <: AbstractIndices{VariableIndex}}
		return new{I,C,II}(transformer, model_cfg, num_outputs, input_indices, output_indices, num_inputs)
	end
	function VecFunc(
		transformer::I, model_cfg::C, num_outputs::Int, _input_indices::_II, output_indices, num_inputs
	) where {I <: AbstractInnerEvaluator,C <: AbstractSurrogateConfig, _II <: AbstractVector{VariableIndex}}
		input_indices = Indices(_input_indices) 
		return new{I,C,typeof(input_indices)}(transformer, model_cfg, num_outputs, input_indices, output_indices, num_inputs)
	end
end

inner_transformer( vfun :: VecFunc ) = vfun.transformer

num_inputs( vfun :: VecFunc ) = vfun.num_inputs 
num_outputs( vfun :: VecFunc ) = vfun.num_outputs
input_indices( vfun :: VecFunc ) = vfun.input_indices
output_indices( vfun :: VecFunc ) = vfun.output_indices
model_cfg(vfun :: VecFunc) = vfun.model_cfg

#=====================================================================
ForwardingOuterEvaluator <: AbstractOuterEvaluator
=====================================================================#
struct ForwardingOuterEvaluator{A,VF<:AbstractOuterEvaluator{A}} <: AbstractOuterEvaluator{A}
	inner :: VF 
end

@forward(
	ForwardingOuterEvaluator.inner, 
	(
		is_atomic,
		input_indices,
		output_indices,
		num_input_sets,
		inner_transformer,
		input_provider,
		num_inputs,
		num_outputs,
		model_cfg,
		#_transform_input_dict,
		eval_at_dict
	)
)

#=====================================================================
ProductOuterEvaluator <: AbstractOuterEvaluator{false}
=====================================================================#
@with_kw struct ProductOuterEvaluator{
	ILeft <: AbstractOuterEvaluator,
	IRight <: AbstractOuterEvaluator,
} <: AbstractOuterEvaluator{false}
	inner_left :: ILeft 
	inner_right :: IRight 

	input_indices_left :: Indices{DependentIndex} = output_indices(inner_left)
	input_indices_right :: Indices{DependentIndex} = output_indices(inner_right)

	input_indices :: Indices{DependentIndex} = union(input_indices_left, input_indices_right)

	num_inputs_left :: Int = length(input_indices_left)
	num_inputs_right :: Int = length(input_indices_right)

	num_outputs :: Int = num_inputs_left + num_inputs_right

	output_indices :: Indices{DependentIndex} = Indices([DependentIndex() for i=1:num_outputs])

	transformer :: InnerIdentity = InnerIdentity( num_outputs )
end

inner_transformer( vfun :: ProductOuterEvaluator ) = vfun.transformer

num_outputs( vfun :: ProductOuterEvaluator ) = vfun.num_outputs
num_inputs( vfun :: ProductOuterEvaluator ) = vfun.num_outputs
num_input_sets( vfun :: ProductOuterEvaluator ) = 2
num_inputs( vfun :: ProductOuterEvaluator, :: Val{1} ) = vfun.num_inputs_left
num_inputs( vfun :: ProductOuterEvaluator, :: Val{2} ) = vfun.num_inputs_right
num_inputs( vfun :: ProductOuterEvaluator, l ) = num_inputs( vfun, Val(l) )
output_indices( vfun :: ProductOuterEvaluator ) = vfun.output_indices
input_indices( vfun :: ProductOuterEvaluator, :: Val{1} ) = vfun.input_indices_left
input_indices( vfun :: ProductOuterEvaluator, :: Val{2} ) = vfun.input_indices_right
input_indices( vfun :: ProductOuterEvaluator, l ) = input_indices(vfun, Val(l))
input_indices( vfun :: ProductOuterEvaluator ) = vfun.input_indices

input_provider( vfun :: ProductOuterEvaluator, :: Val{1} ) = vfun.inner_left
input_provider( vfun :: ProductOuterEvaluator, :: Val{2} ) = vfun.inner_right
input_provider( vfun :: ProductOuterEvaluator, l ) = input_provider(vfun, Val(l))

model_cfg( :: ProductOuterEvaluator) = DUMMY_CONFIG

#=====================================================================
CompositeEvaluator <: AbstractOuterEvaluator{false}
=====================================================================#
@with_kw struct CompositeEvaluator{
	IE <: AbstractInnerEvaluator,
	IP <: AbstractOuterEvaluator,
	MC <: AbstractSurrogateConfig
} <: AbstractOuterEvaluator{false}
	
	outer :: IE
	
	input_provider :: IP
	input_indices :: Indices{DependentIndex} = output_indices(input_provider)

	num_outputs :: Int 
	num_inputs = length(input_indices)
	output_indices = Indices([ DependentIndex() for i = 1 : num_outputs ])
	
	model_cfg :: MC = DUMMY_CONFIG
end

num_input_sets(::CompositeEvaluator) = 1
num_inputs(ce::CompositeEvaluator, args...) = ce.num_inputs
num_outputs(ce::CompositeEvaluator) = ce.num_outputs
inner_transformer(ce::CompositeEvaluator) = ce.outer
input_indices(ce::CompositeEvaluator, args...) = ce.input_indices
input_provider(ce::CompositeEvaluator, args...) = ce.input_provider
output_indices(ce::CompositeEvaluator) = ce.output_indices
model_cfg(ce::CompositeEvaluator) = ce.model_cfg